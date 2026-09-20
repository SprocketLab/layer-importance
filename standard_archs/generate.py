# Sequence Generation
import numpy as np
import math
from collections import defaultdict


task_choices = ["var-copy", "var-copy-rep", "decode-recall", "decode-recall-last", "assoc-recall", "assoc-recall-mk", \
                "needle", "double-recall", "single-recall", "exact-copy", "segmented-retrieval", "segmented-recall", \
                "matrix-lookup", "tst-task"]


def force_args(args):
    if args.train_task in ["var-copy", "var-copy-rep"]:
        pass

    if args.train_task in ["decode-recall", "decode-recall-last"]:
        args.num_numbers = 2
        args.num_vocab = int(2 ** math.floor(math.log(args.num_vocab) / math.log(2)))

    if args.train_task == "assoc-recall":
        args.num_numbers = 0

    if args.train_task == "assoc-recall-mk":
        args.num_numbers = 0
        # args.num_vocab = 1 + int(args.num_vocab ** (1./size_key))

    if args.train_task == "needle":
        args.num_numbers = 2

    if args.train_task in ["single-recall", "double-recall"]:
        args.num_vocab = 0

    if args.train_task == "exact-copy":
        args.num_numbers = 2

    if args.train_task == "segmented-recall":
        args.num_numbers = 0

    if args.train_task == "matrix-lookup":
        # TODO: Change this later
        args.num_vocab = int(np.sqrt((args.sequence_length - 2) / 2))
        args.num_numbers = 0

    if args.train_task == "tst-task":
        # TODO: Change this later
        #args.num_vocab = int(np.sqrt((args.sequence_length - 2) / 2))
        args.num_vocab = int(np.sqrt((args.sequence_length - 2) / 3))
        args.num_numbers = args.num_vocab


def generate_seq(tokenizer, length, task, p=0.2, mixed=False):
    num_vocab = tokenizer.num_vocab
    num_numbers = tokenizer.num_numbers
    
    if task == "var-copy":
        # Start with num_numbers vocab tokens
        if mixed:
            if np.random.rand() < 0.5:
                # Hard for the TF
                input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.03) 
            else:
                # Hard for the SSM
                input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.2) 
                input_seq[-1] = np.random.choice(tokenizer.number_tokens)
        else:
            input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers) 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "var-copy-rep":
        # Start with num_numbers vocab tokens
        # input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        input_seq = rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=p, special_type="repetitive_vocab") 
        
        nums = [(i, int(c[1:])) for (i, c) in enumerate(input_seq) if c in tokenizer.number_tokens]
        
        # The real task, if not degenerate
        if len(nums) > 0:
            output_seq = ["<null>"] * nums[0][0]

            for i in range(len(nums)-1):
                if nums[i][0]-nums[i][1] < 0:
                    if nums[i+1][0]-nums[i][1] < 0:
                        output_seq += ["<null>"] * (nums[i+1][0]-nums[i][0])
                    else:
                        output_seq += ["<null>"] * (nums[i][1]-nums[i][0])
                        output_seq += input_seq[:nums[i+1][0]-nums[i][1]]
                else:
                    output_seq += input_seq[nums[i][0]-nums[i][1]:nums[i+1][0]-nums[i][1]]

            if nums[-1][0]-nums[-1][1] < 0:
                output_seq += ["<null>"] * (nums[-1][1]-nums[-1][0])
                output_seq += input_seq[:-nums[-1][1]]
            else:
                output_seq += input_seq[nums[-1][0]-nums[-1][1]:-nums[-1][1]]
        else:
            output_seq = ["<null>"] * length

        input_seq = ["<bos>"] + input_seq + ["<eos>"]
        output_seq = ["<bos>"] + output_seq + ["<eos>"]
        # output_seq = ["<bos>"] + output_seq[:length] + ["<eos>"]

    elif task == "decode-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=p) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}
        s = 0
        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]
            
            if input_seq[i][0] == '#':
                # s = (2 * s + int(input_seq[i][1:])) % num_numbers
                s = (2 * s + int(input_seq[i][1:])) % num_vocab

            # if i-s < 0:
            #     output_seq[i] = "<null>"
            # else:
            #     output_seq[i] = input_seq[i-s]

            output_seq[i] = assoc["V%d" % s]

    elif task == "decode-recall-last":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        n_bits = int(math.log(num_vocab)/math.log(2))

        target = np.random.randint(0, num_vocab)
        temp = target
        for i in range(length-1, length-1-n_bits, -1):
            input_seq[i] = "#%d" % (temp % 2)
            temp = temp // 2

        try:
            i = length-2-n_bits - input_seq[-2-n_bits::-1].index("V%d" % target)
            output_seq[-1] = input_seq[i+1]
        except ValueError:
            pass

    elif task == "assoc-recall":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.2) 
        output_seq = [None for _ in range(len(input_seq))]

        assoc = {v: "<null>" for v in tokenizer.vocab}

        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]

            output_seq[i] = assoc[input_seq[i]]

    elif task == "assoc-recall-mk":
        size_key = 1
        # size_key = 2
        
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0.0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        assoc = defaultdict(lambda: "<null>")

        for i in range(len(output_seq)):
            if i > size_key:
                key = tuple(input_seq[i-size_key:i])
                assoc[key] = input_seq[i]

            if i+1 > size_key:
                key = tuple(input_seq[i-size_key+1:i+1])
                output_seq[i] = assoc[key]

    elif task == "needle":
        needle_length = 1
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        input_seq[-needle_length:] = ["#1" for _ in range(needle_length)]

        # Needle position
        needle_pos = np.random.randint(0, length // 2)
        input_seq[needle_pos] = "#0"
        needle = input_seq[needle_pos + 1:needle_pos + needle_length + 1]

        # Ask for the needle at the end
        output_seq = ["<null>" for _ in range(len(input_seq))]

        assert needle_length == 1
        output_seq[needle_pos:] = [needle[-1] for i in range(len(input_seq) - needle_pos)]
        # output_seq[needle_pos:] = needle
        # output_seq[-needle_length:] = needle

    # See if there is an exact copy of part of the input sequence in the input sequence

    # UPDATE: Now see if there is a sequence of 4 consecutive subseqneces in the input
    elif task == "exact-copy":
        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        if np.random.random() < 0.5:
            # output_seq[-1] = "#0"
            output_seq[:] = ["#0"] * len(input_seq)
        else:
            # a = np.random.randint(0, 3*length//4)
            # a = np.random.randint(length//2, 3*length//4)
            a = np.random.randint(0, length//2)
            # print("a:", a)
            # a = length + 10
            # l = np.random.randint(5, 10)
            l = 10
            reps = 2
            for start in range(a+l, a+reps*l, l):
                input_seq[start:start+l] = input_seq[a:a+l]


            # output_seq[-1] = "#1"
            #output_seq = ["#0"] * int(a+reps*l) + ["#1"] * int(len(output_seq) - (a+reps*l))
            output_seq = ["#0"] * int(a+reps*l) + [input_seq[a]] * int(len(output_seq) - (a+reps*l))

    # This is for segmented retrieval
    elif task == "segmented-retrieval":
        retrive_distance = 8
        num_indices = 10

        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.5) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        # Select a random set of indices
        indices = np.random.choice(length, size=num_indices, replace=False)
        indices.sort()

        last_index = 0
        last_update = -1
        last_value = "<null>"
        for i in range(num_indices):
            input_seq[indices[i]] = "<sep>"  # Use the separator token as the segment marker

            if indices[i] - last_index > retrive_distance and input_seq[indices[i] - retrive_distance][0] == '#':
                if last_update != -1:
                    output_seq[last_update:indices[i]] = [last_value] * (indices[i] - last_update)

                last_update = indices[i]
                last_value = input_seq[indices[i] - retrive_distance]

            last_index = indices[i]

        if last_update != -1:
            output_seq[last_update:] = [last_value] * (len(output_seq) - last_update)

        if len(input_seq) != len(output_seq):
            print("Input sequence:", input_seq)
            print("Output sequence:", output_seq)
            assert False, "Input and output sequences must be of the same length"



    elif task == "segmented-recall":
        num_indices = 10

        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=0.4) 
        interm_seq = ["<null>" for _ in range(len(input_seq))]
        output_seq = ["<null>" for _ in range(len(input_seq))]

        # Select a random set of indices
        indices = 1 + np.random.choice(length-1, size=num_indices, replace=False)
        indices.sort()

        assoc = {v: "<null>" for v in tokenizer.vocab}
        for i in range(len(output_seq)):
            if i != 0:
                assoc[input_seq[i-1]] = input_seq[i]

            interm_seq[i] = assoc[input_seq[i]]

        for ind, next_ind in zip(indices[:-1], indices[1:]):
            output_seq[ind:next_ind] = [interm_seq[ind-1]] * (next_ind - ind)
        output_seq[indices[-1]:] = [interm_seq[indices[-1]-1]] * (length - indices[-1])



    elif task.endswith("-recall"):
        if task.startswith("single"):
            k = 1
        elif task.startswith("double"):
            k = 2
        else:
            assert False, "Not implemented"

        input_seq = rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=1) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        # target = input_seq[-1]
        # for i in range(k):
        #     target_value = int(target[1:])
        #     target = input_seq[-1-target_value]

        target_ind = len(input_seq)-1
        for i in range(k):
            target_value = int(input_seq[target_ind][1:])
            target_ind -= target_value

            # Just in case
            if target_ind < 0:
                target_ind = 0
                break

        target = input_seq[target_ind]

        output_seq[-1] = target

    elif task == "tst-task":
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0) 
        input_seq[len(tokenizer.number_tokens)] = np.random.choice(tokenizer.number_tokens)
        
        # Do an array lookup with this special number token
        ind1 = int(input_seq[int(input_seq[len(tokenizer.number_tokens)][1:])][1:])

        output_seq = ["<null>" for _ in range(len(input_seq))]

        # We are going to treat the last-2 num_vocab*num_vocab as a matrix
        mat_size = num_vocab**2

        # To help with learning, put the correct lookup position at ind1
        output_seq[len(tokenizer.number_tokens):mat_size] = ["V%d" % ind1] * (mat_size - len(tokenizer.number_tokens))
        for i in range(mat_size, len(input_seq)-1):
            #ind1 = int(input_seq[0][1:])
            #ind2 = int(input_seq[i+1][1:])
            ind2 = int(input_seq[i][1:])
            mat_ind = ind1*num_vocab + ind2
            #output_seq[i+1] = input_seq[i-mat_size+mat_ind]
            output_seq[i] = input_seq[i-mat_size+mat_ind]

    elif task == "matrix-lookup":
        input_seq = rand_seq(tokenizer, length, num_vocab, 0, p_numbers=0) 
        output_seq = ["<null>" for _ in range(len(input_seq))]

        # We are going to treat the last-2 num_vocab*num_vocab as a matrix
        mat_size = num_vocab**2
        for i in range(mat_size, len(input_seq)-1):
            ind1 = int(input_seq[i][1:])
            ind2 = int(input_seq[i+1][1:])
            mat_ind = ind1*num_vocab + ind2
            output_seq[i+1] = input_seq[i-mat_size+mat_ind]


    else:
        print("Task name:", task)
        assert False # Not implemented

    return input_seq, output_seq

################################################################################################

# SEQUENCE GENERATION HELPERS

def rand_seq(tokenizer, length, num_vocab, num_numbers, p_numbers=-1):
    if p_numbers == -1:
        p_numbers = num_numbers / (num_vocab + num_numbers)

    if num_numbers != 0:
        if num_vocab != 0:
            props = {"V": (1-p_numbers)/num_vocab, "#": p_numbers/num_numbers, "<": 0}  
        else:
            props = {"V": 0, "#": p_numbers/num_numbers, "<": 0}
    else:
        props = {"V": 1/num_vocab, "#": 0, "<": 0}  
    props = np.array([props[i[0]] for i in tokenizer.vocab])

    return np.random.choice(tokenizer.vocab, size=length, p=props).tolist()


# For other special generations
def rand_seq_special(tokenizer, length, num_vocab, num_numbers, p_numbers=-1, special_type=None):
    if special_type == "repetitive_vocab":
        if p_numbers == -1:
            p_numbers = num_numbers / (num_vocab + num_numbers)
    
        if num_numbers != 0:
            props = {"V": 0, "#": p_numbers/num_numbers, "<": 0} 
        else:
            props = {"V": 0, "#": 0, "<": 0}  
        if num_numbers != 0:
            props_V0 = (1-p_numbers)
        else:
            props_V0 = 1
        props = np.array([props[i[0]] if i != "V0" else props_V0 for i in tokenizer.vocab])
        
        tile_length = 3

        props_tile = {"V": 1./num_vocab, "#": 0, "<": 0} 
        props_tile = np.array([props_tile[i[0]] for i in tokenizer.vocab])

        ret_seq = np.random.choice(tokenizer.vocab, size=length, p=props)
        ret_seq2 = np.tile(np.random.choice(tokenizer.vocab, size=tile_length, p=props_tile), (length // tile_length + 1))[:length]

        return np.where(ret_seq == "V0", ret_seq2, ret_seq).tolist()

    else:
        assert False, "Not implemented"
