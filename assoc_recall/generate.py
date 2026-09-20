# Sequence Generation
import numpy as np
import math
from collections import defaultdict


task_choices = ["var-copy", "var-copy-rep", "decode-recall", "decode-recall-last", "assoc-recall", "assoc-recall-mk", "needle", "mqar", "mkar", "fuzzy", "flip", "noisy_icr"]

# MAD-style associative-recall variants (see the "mqar"/"mkar"/"fuzzy" branch below).
MQAR_VALUE_LEN = 2   # mqar: 1-token key, multi-token VALUE
MKAR_KEY_LEN = 2     # mkar: multi-token KEY, 1-token value
FUZZY_KEY_LEN = 2    # fuzzy: multi-token KEY *and* multi-token value (John 2026-06-13)
FUZZY_VALUE_LEN = 2

# "flip" counting task (John 2026-07-02, from the "Knee Deep in C-RASP" paper).
# FLIP_K = how many flips must occur before the label turns 1; the difficulty knob
# (higher k = count deeper). Set by run_flip.py per sweep point.
FLIP_K = 6

# "noisy_icr" = noisy in-context recall (MAD synthetic-skills task; John 2026-06-07
# and 2026-09-03 post-train task list). Same explicit item format as mqar/mkar with a
# 1-token key and a 1-token value, but runs of NOISE tokens from a disjoint vocabulary
# are interleaved between items. Noise never carries a target and must be ignored.
NOISY_N_NOISE = 16   # size of the noise vocabulary (number tokens #1..#16; #0 = sep)
NOISY_P = 0.5        # probability of emitting a noise run instead of an item
NOISY_MAX_RUN = 4    # noise run length ~ U{1..NOISY_MAX_RUN}


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

    if args.train_task in ["mqar", "mkar", "fuzzy"]:
        args.num_numbers = 1  # one number token (#0) used as the key/value separator

    if args.train_task == "flip":
        args.num_numbers = 2  # labels #0 / #1 (vocab stays 2, NOT rounded)

    if args.train_task == "noisy_icr":
        args.num_numbers = 1 + NOISY_N_NOISE  # #0 = separator, #1..#N = noise tokens


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
        size_key = 2
        
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
        output_seq[needle_pos:] = [needle[-1] for i in range(len(input_seq) - needle_pos)]
        # output_seq[-needle_length:] = needle

    elif task in ("mqar", "mkar", "fuzzy"):
        # MAD-style associative recall in the explicit item format John specified:
        #   <bos> <key...> #0(sep) <value...> <eos>
        #   - mqar:  key_len=1, value_len>1  (multi-token VALUE)
        #   - mkar:  key_len>1, value_len=1  (multi-token KEY)
        #   - fuzzy: key_len>1, value_len>1  (both multi-token; John 2026-06-13,
        #            here 2+2 to separate the top hybrids vs TF-at-end models)
        # The FIRST occurrence of a key is a DEFINITION: its value tokens are shown
        # in the input and the output is <null> across the whole item. A LATER
        # occurrence of the same key is a QUERY: the value span in the INPUT is
        # <null> placeholders while the OUTPUT carries the stored value, so the
        # model must recall it. ce_loss is unshifted (same-position), so targets
        # sit directly on the value-span positions; the placeholder input there
        # keeps the query from being a trivial copy. char-acc is therefore scored
        # only on query value spans. Keys are unique (never redefined) to avoid
        # ambiguity. With window<<sequence_length a far-back definition can't be
        # reached by a single attention layer, so recall needs SSM memory / layer
        # composition -- the same expressivity regime as the decode-recall sweep.
        if task == "mqar":
            key_len, val_len = 1, MQAR_VALUE_LEN
        elif task == "mkar":
            key_len, val_len = MKAR_KEY_LEN, 1
        else:  # fuzzy
            key_len, val_len = FUZZY_KEY_LEN, FUZZY_VALUE_LEN
        item_len = 1 + key_len + 1 + val_len + 1  # bos, key, sep, value, eos
        sep = "#0"

        store = {}  # key tuple -> list of value tokens

        def rand_key():
            for _ in range(50):
                k = tuple("V%d" % np.random.randint(num_vocab) for _ in range(key_len))
                if k not in store:
                    return k
            return None

        input_seq, output_seq = [], []
        while len(input_seq) + item_len <= length:
            if store and np.random.rand() < 0.5:                 # QUERY
                key = list(store.keys())[np.random.randint(len(store))]
                in_vals, out_vals = ["<null>"] * val_len, list(store[key])
            else:                                                # DEFINITION
                key = rand_key()
                if key is None:
                    break
                value = ["V%d" % np.random.randint(num_vocab) for _ in range(val_len)]
                store[key] = value
                in_vals, out_vals = list(value), ["<null>"] * val_len
            input_seq  += ["<bos>"] + list(key) + [sep] + in_vals + ["<eos>"]
            output_seq += ["<null>"] * (2 + key_len) + out_vals + ["<null>"]

        pad = length - len(input_seq)
        input_seq  += ["<null>"] * pad
        output_seq += ["<null>"] * pad

    elif task == "noisy_icr":
        # Noisy in-context recall (see NOISY_* constants at the top of this file).
        # Items are <bos> key #0 value <eos> exactly as in mqar with key_len=val_len=1;
        # first occurrence of a key = DEFINITION (value shown, target <null>), later
        # occurrence = QUERY (input <null>, target = stored value). Between items the
        # generator emits, with prob NOISY_P, a run of 1..NOISY_MAX_RUN noise tokens
        # drawn from the disjoint noise vocabulary #1..#NOISY_N_NOISE. Noise positions
        # have <null> targets, so char-acc is scored only on query value positions.
        key_len, val_len = 1, 1
        item_len = 1 + key_len + 1 + val_len + 1
        sep = "#0"
        noise_vocab = ["#%d" % (1 + i) for i in range(num_numbers - 1)]
        assert len(noise_vocab) == NOISY_N_NOISE, "force_args must set num_numbers"

        store = {}

        def rand_key():
            for _ in range(50):
                k = tuple("V%d" % np.random.randint(num_vocab) for _ in range(key_len))
                if k not in store:
                    return k
            return None

        input_seq, output_seq = [], []
        while len(input_seq) + item_len <= length:
            if np.random.rand() < NOISY_P:                        # NOISE run
                run = int(np.random.randint(1, NOISY_MAX_RUN + 1))
                run = min(run, length - len(input_seq))
                input_seq  += [noise_vocab[np.random.randint(len(noise_vocab))] for _ in range(run)]
                output_seq += ["<null>"] * run
                continue
            if store and np.random.rand() < 0.5:                 # QUERY
                key = list(store.keys())[np.random.randint(len(store))]
                in_vals, out_vals = ["<null>"] * val_len, list(store[key])
            else:                                                # DEFINITION
                key = rand_key()
                if key is None:
                    break
                value = ["V%d" % np.random.randint(num_vocab) for _ in range(val_len)]
                store[key] = value
                in_vals, out_vals = list(value), ["<null>"] * val_len
            input_seq  += ["<bos>"] + list(key) + [sep] + in_vals + ["<eos>"]
            output_seq += ["<null>"] * (2 + key_len) + out_vals + ["<null>"]

        pad = length - len(input_seq)
        input_seq  += ["<null>"] * pad
        output_seq += ["<null>"] * pad

    elif task == "flip":
        # C-RASP "flip" counting task (John 2026-07-02, from "Knee Deep in C-RASP").
        # Vocab = 2 symbols (V0/V1). A FLIP = two adjacent tokens differ. The label
        # is 0 until the k-th flip has occurred, then 1 from that position onward
        # (k = FLIP_K). Example, k=6:
        #   in : V0 V1 V1 V0 V1 V0 V0 V1 V0 V1
        #   flp:     1     2  3  4     5     6   (running count of adjacent changes)
        #   out:  0  0  0  0  0  0  0  0  0  1   (fires once the 6th flip lands)
        # Difficulty knob = FLIP_K: the model must hold a running count up to k (the
        # C-RASP nested-count analogue). Labels are the number tokens #0/#1 and are
        # scored at EVERY position (nothing is <null>), so char-acc is over the whole
        # sequence. NOTE the base rate is k-dependent -- larger k pushes the 0->1
        # transition later so most labels are 0; read char-acc against the majority-
        # class baseline (drawn by plot_flip.py), not against 0.5.
        input_seq = ["V%d" % np.random.randint(num_vocab) for _ in range(length)]
        output_seq = []
        flips = 0
        for i in range(length):
            if i > 0 and input_seq[i] != input_seq[i - 1]:
                flips += 1
            output_seq.append("#1" if flips >= FLIP_K else "#0")

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
        props = {"V": (1-p_numbers)/num_vocab, "#": p_numbers/num_numbers, "<": 0}  
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