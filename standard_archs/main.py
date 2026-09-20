import json
import argparse
import os
import numpy as np
import math

import logging
logging.getLogger("transformers.utils.args_doc").setLevel(logging.CRITICAL)

from model_utils import get_model
from data_utils import get_train_dataset, get_tokenizer
from train_utils import train, save_model, make_dir, get_ident_name, get_data_ident_name, get_task_dir_name, get_lrs, add_lr
from test_utils import evaluation
from generate import force_args, task_choices


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_number', default=-1, type=int, help="The current run number. Will not save if the run has already been saved")

    # Task parameters
    parser.add_argument('--train_task', choices=task_choices,
                        required=True, help="Task to train the model")
    parser.add_argument('--eval_task', choices=task_choices,
                        required=True, help="tasks to evaluate the model")
    
    parser.add_argument('--num_vocab', default=26, type=int, help="vocabulary size in the strings. maximum is 26.")
    parser.add_argument('--num_numbers', default=5, type=int, help="vocabulary (number) size in the strings. maximum is 9.")
    parser.add_argument('--min_number', default=0, type=int, help="The smallest number token")

    parser.add_argument('--ood_eval', default=False, type=bool, help="If true, perform out-of-distribution evaluation.")
    parser.add_argument('--p', default=0.2, type=float, help="proportion, depends on task")    
    parser.add_argument('--eval_p', default=None, type=float, help="proportion, depends on task")    

    parser.add_argument('--mixed', default=False, type=bool, help="If true, use mixed distribution when generating data.")

    # Model
    parser.add_argument('--nope', default=False, type=bool, help="If true, use the no positional encoding version of the hybrid model")

    parser.add_argument('--model', type=str, default=None, help='The model architecture.')
    parser.add_argument('--num_layers', type=int, default=None, help="Number of layers in the model. Cannot specify layers.")

    parser.add_argument('--balance_mem', default=-1, type=int, help="Amount of memory to attempt to match.")
    parser.add_argument('--hidden_size', default=8, type=int, help="Hidden size of the models")
    parser.add_argument('--heads', default=1, type=int, help="Number of heads in the transformer models.")
    parser.add_argument('--num_masked_heads', default=1, type=int, help='''Only when model = ''T_hard_alibi''. 
            Number of heads where we apply hard alibi. The remaining heads are set to nope.''')
    parser.add_argument('--state_dim', default=1, type=int, help='''Only when model = ''mamba'' or ''hybrid''. 
            Sets the state dimension of the model.''')

    # Optimization
    parser.add_argument('--lr', default=1e-3, type=float, help="choice of learning rate")
    parser.add_argument('--auto_lr', default=False, type=bool, help="If true, find the best lr with some training")
    parser.add_argument('--force_do_lr', default=False, type=bool, help="If true, learn a new lr")

    parser.add_argument('--epochs', default=4, type=int, help="number of epochs")
    parser.add_argument('--num_examples', default=1000, type=int, help="number of samples for each epoch")
    parser.add_argument('--num_eval_examples', default=100, type=int, help="number of evaluation examples per length")
    parser.add_argument('--window', default=20, type=int, help="width of the sliding window attention")

    parser.add_argument('--train_batch_size', default=8, type=int, help="training batch size")
    parser.add_argument('--eval_batch_size', default=8, type=int, help="evaluation batch size")
    parser.add_argument('--eval_num_batches', default=1, type=int, help='''number of batches to use for evaluation.
            useful to have a mean + std over results.''')
    
    parser.add_argument('--pack_examples', default=False, type=bool, help='If true, fill context with multiple examples, deliniated')
    parser.add_argument('--min_train_length', default=97, type=int, help="minimum length of a training example")
    parser.add_argument('--max_train_length', default=98, type=int, help="maximum length of a training example")
    parser.add_argument('--min_eval_length', default=97, type=int, help="minimum length of an evaluation example")
    parser.add_argument('--max_eval_length', default=98, type=int, help="maximum length of an evaluation example")

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help="number of gradient accumulation steps")

    # Context length
    parser.add_argument('--sequence_length', default=100, type=int, help="context length during training")
    parser.add_argument('--eval_sequence_length', default=100, type=int, help="context length at evaluation time")

    # Saving parameters
    parser.add_argument('--save_model', default=False, type=bool, help="If true, save the model after training")
    parser.add_argument('--save_results', default=False, type=bool, help="If true, save the results after training")
    parser.add_argument('--run_anyways', default=False, type=bool, help="If true, run even if the results have already been saved")
    
    # Visual parameters
    parser.add_argument('--print', default=False, type=bool, help="If true, show helpful print statements")
    parser.add_argument('--progress_bar', default=False, type=bool, help="If true, show the process of each epoch")
    parser.add_argument('--num_log_steps', default=50, type=int, help="number of steps between each log when training")

    parser.add_argument("--test_generate", default=False, type=bool, help="If true, test the synthetic tasks generation")
    
    return parser.parse_args()

args = parse_args()


# Set the layers based on either the model or the specified layers
args.layers = []
args.memory = 0

memory = 0
if args.balance_mem > 0:
    target_mem = args.balance_mem
    for c in args.model:
        if c == "T":
            memory += args.window * args.hidden_size
        elif c == "S":
            memory += args.state_dim * args.hidden_size
        elif c == "D":
            #memory += args.hidden_size * args.hidden_size
            target_mem -= args.hidden_size * args.hidden_size
            if target_mem < 0:
                assert False, "Error: GDN state is too large for set memory requirements"
        else:
            assert False, "Bad Letter in args.model"
    
    # Scale the window the SSM state, _not_ the hidden dimension. Does not work for
    # GDN

    if memory > 0:
        factor = target_mem / memory

        args.state_dim = int(math.ceil(factor * args.state_dim))   
        args.window = int(math.ceil(factor * args.window))   
    

for c in args.model:
    if c == "T":
        args.layers.append("TF")
        args.memory += args.window * args.hidden_size
    elif c == "S":
        args.layers.append("SSM")
        args.memory += args.state_dim * args.hidden_size
    elif c == "D":
        args.layers.append("GDN")
        args.memory += args.hidden_size * args.hidden_size
    else:
        assert False, "Bad Letter in args.model"

# Set the eval dataset to be the same as the train dataset if not specified
if args.eval_p is None:
    args.eval_p = args.p

# Force task specific arguments
force_args(args)

args.data_name = get_data_ident_name(args)

if not args.auto_lr and args.save_results and args.run_number >= 0 and not args.run_anyways:
    result_filename = 'results/' + get_task_dir_name(args) + '/%d.json' % args.run_number
    if os.path.exists(result_filename):
        exit(0)

if args.print:
    print(args)


## Get train dataset & tokenizer
tokenizer = get_tokenizer(args)
train_dataset = get_train_dataset(args, tokenizer) 

batch = next(iter(train_dataset))

if args.print:
    print("v"*100)
    print("EXAMPLE:", batch['input'][0])
    # print("STRUNG:", tokenizer.to_string(batch['input_ids'][0]))
    print("-"*100)
    print("TOKENIZED:", batch['input_ids'][0][batch['mask'][0]==1])
    print("^"*100)

if args.test_generate:
    i = batch['input'][0].index("#0")
    print(batch['input'][0][i])
    print(batch['input'][0][i+1])
    print(batch['output'][0][-1])
    exit(0)


## Find the best LR
if args.auto_lr:
    lrs = get_lrs(args)
    key = get_data_ident_name(args) + "_" + get_ident_name(args)
    if args.print:
        print(key)

    if key not in lrs.keys() or (args.force_do_lr and args.run_number == 0):
        losses = []
        for itr in range(2):
        # for itr in range(3):
            for lr in np.geomspace(1e-4, 1e-3, num=9):
                model = get_model(args, tokenizer)
                args.lr = lr
                if args.print:
                    print("Testing LR:", lr)
                # _, final_loss = train(args, model, tokenizer, train_dataset)
                _, final_loss = train(args, model, tokenizer, train_dataset, one_epoch=True, do_print=args.print)
                if args.print:
                    print("Final loss:", final_loss)

                losses.append((final_loss, lr))
        losses.sort()
        best_lr = losses[0][1]
        add_lr(args, best_lr)

        args.lr = best_lr
    else:
        args.lr = lrs[key]

if args.save_results and args.run_number >= 0 and not args.run_anyways:
    result_filename = 'results/' + get_task_dir_name(args) + '/%d.json' % args.run_number
    if os.path.exists(result_filename):
        exit(0)

## Get model
model = get_model(args, tokenizer)
args.param_count = count_parameters(model)

if args.print:
    print()
    print("v"*100)
    print(model)
    print(f"Number of parameters of the model: {count_parameters(model)}")
    print("^"*100)
    print()


## train the model
accs, final_loss = train(args, model, tokenizer, train_dataset, do_print=args.print)

## save model
if args.save_model:
    save_model(args, model)

## evaluation of the model
if args.print:
    print("###EVALUATION")

model.eval()

str_acc_mean_list, str_acc_std_list, char_accuracy_list = evaluation(args, model, tokenizer)

if args.print:
    print(args)

    print("DONE")

    print("String")
    print(str_acc_mean_list)
    print("Char")
    print(char_accuracy_list)


if args.save_results and args.run_number >= 0:
    # assert False, "Decide what we want to actually save"
    results = {
        "train_accs": accs,
        "final_acc": char_accuracy_list,
        "final_loss": final_loss,
        "params": count_parameters(model),
        "args": vars(args)
    }

    print_results = json.dumps(results)
    print(print_results)

    make_dir(args)

    save_path = 'results/' + get_task_dir_name(args)
    with open(save_path + '/%d.json' % args.run_number, 'w') as f:
        json.dump(results, f)
