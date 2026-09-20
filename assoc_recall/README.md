# Associative recall: 8-layer hybrid sweeps

Companion code for the associative-recall experiments in **Expressivity-Efficiency Tradeoffs for
Hybrid Sequence Models**. The question here is the same as in `exact_copy/` but asked over the
whole 8-layer design space: with the layer count fixed, which SSM:TF ratio and which layer
*ordering* does each recall task prefer?

Every architecture is an 8-layer stack of Mamba blocks (`S`) and RoPE transformer blocks (`T`)
with hidden size 384, one attention head, and a **sliding attention window of 20** on a
sequence of 100 tokens. The small window is deliberate: a single attention layer cannot reach a
definition 25-40 positions back, so recall has to come from SSM state or from how the layers
compose. Learning rate is chosen per architecture by a short search, so the tuning budget is
matched too.

## Contents

| File | What it is |
| --- | --- |
| `decode_recall.py` | the architecture table (`CONFIGS`, 36 layouts), the auto-LR search, the training / evaluation loop, and the decode-recall sweep CLI |
| `mqar.py`, `mkar.py`, `fuzzy.py` | the same sweep on the three MAD-style recall tasks (each is a 40-line wrapper that sets the task and the config subset) |
| `generate.py` | sequence generators for every task |
| `models/` | `HybridForCausalLM` (`hybrid.py`) built from a list of layer tokens, the Mamba block, RoPE / NoPE / hard-ALiBi transformer blocks, and an optional DeltaProduct block |
| `data_utils.py`, `train_utils.py`, `test_utils.py`, `model_utils.py` | tokenizer, dataset, training loop, windowed evaluation, model factory |
| `plot_architecture_sweep.py` | accuracy vs #SSM for the decode-recall sweep, coloured by last-layer type |

## The tasks

All four are next-token tasks scored by per-position accuracy on the target positions only
(`<null>` targets are ignored). Vocabulary: 32 value tokens `V0..V31` plus a few number tokens
`#k` used as bits or separators.

**decode-recall.** A random stream of value tokens with bit tokens `#0` / `#1` mixed in
(probability 0.2). The bits accumulate a running index `s = (2s + bit) mod 32`, and the target at
every position is the token that most recently *followed* `V_s` in the stream. The model has to
keep a full "what came after each token" table up to date and decode a changing pointer into it.
This is the hardest of the four; pure windowed attention plateaus well below the hybrids.

**MQAR / MKAR / fuzzy** use one explicit item format, `<bos> key #0 value <eos>`, packed back to
back. The first occurrence of a key *defines* it (the value is shown, targets are `<null>`); a
later occurrence of the same key is a *query*: the value slots in the input are `<null>` and the
target carries the stored value. Keys are never redefined. They differ only in lengths:

| Task | key | value | what is hard |
| --- | --- | --- | --- |
| `mqar` | 1 token | 2 tokens | emitting a multi-token value |
| `mkar` | 2 tokens | 1 token | matching a multi-token key |
| `fuzzy` | 2 tokens | 2 tokens | both; used to separate hybrids that tie on MQAR |

## Architectures

`decode_recall.CONFIGS` maps a name to an 8-character layout, e.g.

```
pure_tf         TTTTTTTT        4s4t_alternate  STSTSTST        6s2t_start  TTSSSSSS
pure_ssm        SSSSSSSS        4s4t_block      SSTTSSTT        6s2t_end    SSSSSSTT
7s1t_start      TSSSSSSS        5s3t_end        SSSSSTTT        5s3t_even   STSSTSST
```

The full table covers every ratio from 7:1 to 1:7 with TF-at-start, TF-at-end, alternating and
even orderings, and all eight positions of a single TF layer. `D` layouts (`pure_dp`,
`4d4t_alternate`, `6d2t_start`) swap the Mamba block for a DeltaProduct block and need
`flash-linear-attention`.

## Protocol

| | |
| --- | --- |
| hidden size / heads / SSM state | 384 / 1 / 1 |
| attention window | 20 (sequence length 100) |
| training | 1000 sequences per epoch, batch 8, 48 epochs, AdamW |
| learning rate | per config: lowest loss over a 7-point grid `1e-4 .. 1e-3` (log-spaced), 2 passes of 2 epochs each, cached in `results/<cond>/auto_lr.json` |
| seeds | 3 to screen, 5 for finalists (decode-recall); 5 (MQAR); 7 (MKAR, fuzzy) |
| metric | per-position accuracy on target positions, best over epochs |

## Installation

```bash
pip install -r requirements.txt
```

`mamba_ssm` and `causal_conv1d` need the prebuilt wheel matching your torch / CUDA build (see
their release pages); the pins in `requirements.txt` are the ones we ran with. A GPU is required
for the Mamba kernels. Set `WANDB_MODE=disabled` unless you want logging.

## Usage

Run everything from this directory.

```bash
# decode-recall: screen all configs at 3 seeds, then 5 seeds on the finalists
python decode_recall.py --round1
python decode_recall.py --round2
python decode_recall.py --rank                   # print the ranking from the result files

# one architecture, all seeds (auto-LR searched and cached on first use)
python decode_recall.py --config 4s4t_alternate
python decode_recall.py --config 4s4t_alternate --lr 3e-4 --seed 0

# the three MAD-style tasks (config subsets are set at the top of each file)
python mqar.py
python mkar.py
python fuzzy.py

python plot_architecture_sweep.py               # decode-recall figure
```

Every run writes `results/<cond>/<task-dir>/<config>/<config>__lr<lr>__seed<k>.json` with the
per-epoch accuracy and loss curves; existing files are skipped, so every sweep resumes.
`--dry-run` prints the plan without training.

## Results

Mean best accuracy over seeds, d=384, window 20, 48 epochs, auto-LR per config.

| | decode-recall (5 seeds) | MQAR (5) | MKAR (7) | fuzzy (7) |
| --- | --- | --- | --- | --- |
| `pure_tf` (TTTTTTTT) | 0.74 | 0.47 | 0.31 | - |
| `pure_ssm` (SSSSSSSS) | 0.49 | 0.59 | 0.33 | - |
| `4s4t_alternate` (STSTSTST) | **0.81** | 0.92 | 0.50 | 0.57 |
| `5s3t_end` (SSSSSTTT) | 0.77 | **0.94** | 0.63 | 0.57 |
| `6s2t_start` (TTSSSSSS) | 0.55 | 0.70 | **0.67** | **0.77** |
| `7s1t_start` (TSSSSSSS) | 0.79 | 0.64 | 0.66 | - |

What to read off it:

- **Ratio barely matters, ordering does.** On decode-recall the best ordering at every ratio
  from 2:6 to 7:1 lands within 0.75-0.81; the only cliff is `pure_ssm`. Within a ratio, every
  ordering whose *last* layer is TF beats every ordering whose last layer is SSM, and putting
  the TF layers first (`TTSSSSSS`, 0.55) is worse than pure TF.
- **The best ordering is task-dependent.** Alternating (`STSTSTST`) wins decode-recall and is
  within noise of the TF-at-end winner on MQAR, but the multi-token-key tasks (MKAR, fuzzy)
  prefer the SSM-heavy TF-first stack that loses on decode-recall.
- **Pure SSM beats pure TF on MQAR** (0.59 vs 0.47), the reverse of decode-recall: unbounded
  recurrence reaches definitions a width-20 window cannot.
