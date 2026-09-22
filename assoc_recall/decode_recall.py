#!/usr/bin/env python3
"""
8-layer hybrid sweeps on decode-recall (and, through mqar.py / mkar.py / fuzzy.py, on the
MAD-style associative-recall tasks).

Goal: find the best layer ratio (#SSM : #TF) and layer ordering for 8-layer hybrids, with the
learning rate chosen per architecture by a small auto-LR search.

The models are the HybridForCausalLM of models/hybrid.py, whose stack is built straight from
config.layers, so an arbitrary 8-element list of "SSM" / "TF" (/ "DP") layers just works; this
file drives get_model / train / evaluation in-process and names a result file per
(config, lr, seed).

  --round1   all curated architectures, 3 screening seeds, auto-LR each   (resume-safe)
  --round2   5 seeds on the round-1 finalists (top 5 + bottom 3)
  --config   run a single named config (all seeds, auto-LR unless --lr)
  --rank     print the round-1 ranking from the result files
  --dry-run  print the plan without training

Results -> results/<cond>/<config>/<config>__lr<lr>__seed<k>.json  (cond = d384 by default)
"""
import os
import sys
import gc
import json
import time
import math
import random
import argparse
import traceback
from argparse import Namespace

import numpy as np
import torch

# experiments modules (run from the experiments directory, or we add it to the path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model_utils import get_model
from data_utils import get_tokenizer, get_train_dataset
from train_utils import train
from test_utils import evaluation
from generate import force_args

# ───────────────────────── experiment definition ──────────────────────────────

# Results land here (a NEW top-level dir, separate from experiments/results).
RESULTS_DIR = os.path.join(_HERE, "results")

TASK = "decode-recall"

# Corrected decode-recall defaults — empirically validated, NOT the prompt's
# d=24/window=20/lr=1e-3 (which leave the task in an unlearnable dead zone; see
# notes below).  decode-recall only trains when ALL THREE hold together:
#   * window = 100  (full attention; the bits + binding to recall sit 25-40
#                    positions back, so a width-20 window can't see them and the
#                    model can't even fit the training set — flat loss at ln 32)
#   * large d       (paper: d=384 -> ~0.85, d=768 -> ~0.98; d<=100 is ~noise)
#   * small lr      (paper auto-LR chose 1e-4..3e-4 at d=768; 3e-3 is too high)
# Validated: SSM-TF d=768 w=100 lr=3e-4 reaches 0.40 and rising at 4 epochs.
DEFAULTS = dict(
    hidden_size=384,     # learnable regime; d-smoke showed configs still separate
                         #   (4s4t 0.76 / pure_tf 0.59 / pure_ssm 0.28 at E=16) AND the
                         #   hybrid-over-pure_tf edge is LARGER here than at d=768, while
                         #   wall time is ~15% lower. d=768 is used only for the d-stability
                         #   recheck. (Paper main is d=16/24 dead zone; this is a scaling pt.)
    heads=1,             # nh=1 in the paper's decode-recall sweep
    state_dim=1,         # sd=1 in the paper's decode-recall sweep
    window=20,           # small window: the expressivity regime — TF can't brute-force
                         #   the global recall, so SSM memory / layer composition matters
                         #   (w=100 full attention instead favours TF-heavy configs)
    num_vocab=33,        # force_args rounds down to 32 = 2**5 (bit length 5)
    sequence_length=100,
    epochs=48,           # convergence probe: TF-rich configs plateau by ~24, but
                         #   SSM-heavy (pure_ssm/7s1t) are still rising at 48. 48 is the
                         #   fair common budget where the fast half has PROVABLY plateaued;
                         #   SSM-heavy gets a separate E=72 robustness check (does the
                         #   ranking flip with more budget?). 12 badly under-converges SSM.
    num_examples=1000,
    train_batch_size=8,
    eval_batch_size=8,
)

# Learning rate is chosen PER CONFIG by auto-LR (the paper's method), NOT fixed:
# find_best_lr() picks the lr with the lowest loss over LR_GRID x LR_ITERS
# and caches it. LR sweep results showed optimum clusters around 5e-4, so the grid
# spans 1e-4..1e-3. Uses 2-epoch passes for more stable selection (1-epoch was
# systematically undershooting by ~5x).
LR_GRID = list(np.geomspace(1e-4, 1e-3, 7))
LR_ITERS = 2             # 2 noisy passes per lr; take the global-min loss
LR_SEARCH_EPOCHS = 2     # epochs per LR search pass (was 1)
DEFAULT_LR = 3e-4        # fallback for --config / dry-run when no auto-LR is cached

# 3 seeds to SCREEN all 18 configs (round1), then 5 seeds on the FINALISTS (round2).
ROUND1_SEEDS = [0, 1, 2]
ROUND2_SEEDS = [0, 1, 2, 3, 4]
SEEDS = ROUND2_SEEDS     # back-compat for analyze_decode_recall's import
ROUND2_LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]  # only for the optional --lr_sweep

# 18 curated configs: test ratio AND ordering jointly. S=SSM, T=TF.
CONFIGS = {
    # pure baselines
    "pure_ssm":       "SSSSSSSS",
    "pure_tf":        "TTTTTTTT",
    # 7:1 — one TF layer, all 8 positions
    "7s1t_start":     "TSSSSSSS",
    "7s1t_pos2":      "STSSSSSS",
    "7s1t_pos3":      "SSTSSSSS",
    "7s1t_mid":       "SSSTSSSS",
    "7s1t_pos5":      "SSSSTSSS",
    "7s1t_pos6":      "SSSSSTSS",
    "7s1t_pos7":      "SSSSSSTS",
    "7s1t_end":       "SSSSSSST",
    # 6:2
    "6s2t_end":       "SSSSSSTT",
    "6s2t_start":     "TTSSSSSS",
    "6s2t_even":      "SSSTSSST",
    # 5:3
    "5s3t_end":       "SSSSSTTT",
    "5s3t_start":     "TTTSSSSS",
    "5s3t_even":      "STSSTSST",
    # 4:4
    "4s4t_ssm_first": "SSSSTTTT",
    "4s4t_tf_first":  "TTTTSSSS",
    "4s4t_alternate": "STSTSTST",
    "4s4t_block":     "SSTTSSTT",
    # TF-heavy (expected worse), SSM layers at the front
    "3s5t":           "SSSTTTTT",
    "2s6t":           "SSTTTTTT",
    "1s7t":           "STTTTTTT",
    # ordering-principle screening (added after round1): test cross-ratio whether
    # interleave > ssm-first > tf-first holds beyond the ratios where round1
    # already had 3+ orderings. Run at 3 seeds via --round1 (resume-skips the 18).
    "5s3t_alt":       "STSTSTSS",  # interleave at 5:3 (alt-win generalises SSM-heavy?)
    "3s5t_alt":       "TSTSTSTT",  # interleave at 3:5 (alt-win generalises TF-heavy?)
    "3s5t_tf_first":  "TTTTTSSS",  # tf-first anchor at 3:5
    "2s6t_tf_first":  "TTTTTTSS",  # tf-first anchor at 2:6
    "1s7t_tf_first":  "TTTTTTTS",  # tf-first anchor at 1:7
    "6s2t_alt":       "STSTSSSS",  # alternate at 6:2
    "2s6t_alt":       "TSTSTTTT",  # alternate at 2:6
    "3s5t_even":      "TSTTSTTS",  # mirror of 5s3t_even
    "2s6t_even":      "TTTSTTTS",  # mirror of 6s2t_even
    "1s7t_even":      "TTTSTTTT",  # mirror of 7s1t_mid
    # DeltaProduct (D) configs
    "pure_dp":        "DDDDDDDD",
    "4d4t_alternate": "DTDTDTDT",
    "6d2t_start":     "TTDDDDDD",
}

# Configs added for the ordering study; NOT part of the round-2 finalist pool
# (they are 3-seed screening, not 5-seed finals).
ORDERING_EXTRA = ["5s3t_alt", "3s5t_alt", "3s5t_tf_first", "2s6t_tf_first",
                   "1s7t_tf_first", "6s2t_alt", "2s6t_alt",
                   "3s5t_even", "2s6t_even", "1s7t_even"]


def layers_list(config_name):
    """'SSSSTTTT' -> ['SSM','SSM','SSM','SSM','TF','TF','TF','TF']."""
    m = {"S": "SSM", "T": "TF", "D": "DP"}
    return [m[c] for c in CONFIGS[config_name]]


def ssm_count(config_name):
    return CONFIGS[config_name].count("S")


def lr_tag(lr):
    """1e-3 -> 'lr1e-03', 3e-4 -> 'lr3e-04'  (stable, reversible-ish filename tag)."""
    return "lr" + f"{lr:.0e}"


def cond_tag():
    """Run-condition stamp. Only encodes parameters that vary across experiments
    (d is always present; ep/nv appended only when non-default)."""
    tag = f"d{DEFAULTS['hidden_size']}"
    if DEFAULTS['epochs'] != 48:
        tag += f"_ep{DEFAULTS['epochs']}"
    return tag


def result_path(config_name, lr, seed):
    fname = f"{config_name}__{lr_tag(lr)}__seed{seed}.json"
    return os.path.join(RESULTS_DIR, cond_tag(), config_name, fname)


# ───────────────────────────── single run ─────────────────────────────────────

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_args(config_name, lr):
    """Construct the Namespace the experiments pipeline expects for one run."""
    a = Namespace(
        train_task=TASK, eval_task=TASK,
        num_vocab=DEFAULTS["num_vocab"], num_numbers=2, min_number=0,
        p=0.2, eval_p=0.2, mixed=False, ood_eval=False,
        nope=False,
        model=None, num_layers=None,
        layer1=None, layer2=None, layer3=None,
        layers=layers_list(config_name),
        hidden_size=DEFAULTS["hidden_size"], heads=DEFAULTS["heads"],
        num_masked_heads=1, state_dim=DEFAULTS["state_dim"],
        lr=lr, auto_lr=False, force_do_lr=False,
        epochs=DEFAULTS["epochs"], num_examples=DEFAULTS["num_examples"],
        num_eval_examples=100, window=DEFAULTS["window"],
        train_batch_size=DEFAULTS["train_batch_size"],
        eval_batch_size=DEFAULTS["eval_batch_size"], eval_num_batches=1,
        pack_examples=False,
        min_train_length=DEFAULTS.get("min_train_length", 97),
        max_train_length=DEFAULTS.get("max_train_length", 98),
        min_eval_length=DEFAULTS.get("min_eval_length", 97),
        max_eval_length=DEFAULTS.get("max_eval_length", 98),
        gradient_accumulation_steps=1,
        sequence_length=DEFAULTS["sequence_length"],
        eval_sequence_length=DEFAULTS["sequence_length"],
        save_model=False, save_results=False, run_anyways=False,
        run_number=-1,
        print=False, progress_bar=False, num_log_steps=50, test_generate=False,
    )
    # force_args sets num_numbers=2 and rounds num_vocab down to a power of two.
    force_args(a)
    return a


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────── auto-LR ──────────────────────────────────────

def auto_lr_path():
    """Per-condition cache of each config's chosen lr (keyed by config name)."""
    return os.path.join(RESULTS_DIR, cond_tag(), "auto_lr.json")


def load_auto_lr():
    p = auto_lr_path()
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"WARN: could not read {p}")
    return {}


def save_auto_lr(cache):
    p = auto_lr_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(cache, f, indent=2)


def find_best_lr(config_name, force=False, verbose=True):
    """Auto-LR: lowest training loss over LR_GRID x LR_ITERS x LR_SEARCH_EPOCHS.
    Cached per (cond_tag, config)."""
    cache = load_auto_lr()
    if config_name in cache and not force:
        return cache[config_name]
    losses = []
    for it in range(LR_ITERS):
        for lr in LR_GRID:
            args = build_args(config_name, lr)
            set_all_seeds(it)
            tok = get_tokenizer(args)
            ds = get_train_dataset(args, tok)
            model = get_model(args, tok)
            _, final_loss = train(args, model, tok, ds, max_epochs=LR_SEARCH_EPOCHS)
            losses.append((float(final_loss), float(lr)))
            del model, ds, tok
            gc.collect()
            torch.cuda.empty_cache()
    losses.sort(key=lambda t: t[0])
    best = losses[0][1]
    cache[config_name] = best
    save_auto_lr(cache)
    if verbose:
        print(f"  auto-LR {config_name:16s} ({CONFIGS[config_name]}): best={best:.2e} "
              f"(min 1-ep loss {losses[0][0]:.4f})", flush=True)
    return best


def resolve_lrs(dry_run=False):
    """Map config -> lr. Searches (and caches) unless dry-run, which reads the cache
    and falls back to DEFAULT_LR for configs not yet searched."""
    if dry_run:
        cache = load_auto_lr()
        return {c: cache.get(c, DEFAULT_LR) for c in CONFIGS}
    return {c: find_best_lr(c) for c in CONFIGS}


def run_single(config_name, lr, seed, round_tag, do_print=False):
    """Train+eval one (config, lr, seed). Returns the result dict (also saved)."""
    args = build_args(config_name, lr)
    set_all_seeds(seed)

    tokenizer = get_tokenizer(args)
    train_dataset = get_train_dataset(args, tokenizer)

    model = get_model(args, tokenizer)
    n_params = count_parameters(model)

    t0 = time.time()
    accs, final_loss = train(args, model, tokenizer, train_dataset)
    model.eval()
    str_mean, str_std, char_acc = evaluation(args, model, tokenizer, do_print=do_print)
    wall = time.time() - t0

    result = {
        "config": config_name,
        "architecture": CONFIGS[config_name],
        "layers": args.layers,
        "ssm_count": ssm_count(config_name),
        "lr": lr,
        "seed": seed,
        "round": round_tag,
        "train_accs": accs,                 # char acc per epoch (at eval length)
        "final_acc": char_acc,              # char acc per eval length
        "mean_acc": float(np.mean(char_acc)),
        "str_acc": str_mean,
        "final_loss": final_loss,
        "params": n_params,
        "wall_time_s": wall,
        "hyperparams": {k: DEFAULTS[k] for k in DEFAULTS},
    }

    path = result_path(config_name, lr, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    # GPU cleanup so a long sweep doesn't accumulate memory.
    del model, train_dataset, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


# ───────────────────────────── orchestration ──────────────────────────────────

def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3


def fmt_eta(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def already_done(config_name, lr, seed):
    return os.path.exists(result_path(config_name, lr, seed))


def run_plan(plan, round_tag):
    """plan: list of (config, lr, seed). Resume-aware, with ETA + error handling."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, f"{round_tag}_run.log")

    todo = [(c, lr, s) for (c, lr, s) in plan if not already_done(c, lr, s)]
    skipped = len(plan) - len(todo)
    print(f"[{round_tag}] {len(plan)} runs total | {skipped} already done | {len(todo)} to run")
    if not todo:
        print(f"[{round_tag}] nothing to do — all results present.")
        return

    durations = []
    failures = []
    with open(log_path, "a") as logf:
        logf.write(f"\n===== {round_tag}: starting {len(todo)} runs =====\n")
        for i, (config, lr, seed) in enumerate(todo, 1):
            eta = ""
            if durations:
                mean_d = sum(durations) / len(durations)
                eta = f" | ETA {fmt_eta(mean_d * (len(todo) - i + 1))}"
            print(f"[{round_tag} {i}/{len(todo)}] {config} ({CONFIGS[config]}) "
                  f"lr={lr:g} seed={seed}{eta}", flush=True)
            t0 = time.time()
            try:
                res = run_single(config, lr, seed, round_tag)
                dt = time.time() - t0
                durations.append(dt)
                msg = (f"  -> acc={res['mean_acc']:.4f} loss={res['final_loss']:.4f} "
                       f"params={res['params']} {fmt_eta(dt)} vram={vram_gb():.2f}GB")
                print(msg, flush=True)
                logf.write(f"OK   {config} {lr_tag(lr)} seed{seed} "
                           f"acc={res['mean_acc']:.4f} {fmt_eta(dt)}\n")
            except Exception as e:  # OOM or anything else: log, clean up, continue
                dt = time.time() - t0
                failures.append((config, lr, seed, repr(e)))
                print(f"  -> FAILED ({repr(e)})", flush=True)
                logf.write(f"FAIL {config} {lr_tag(lr)} seed{seed} {repr(e)}\n")
                logf.write(traceback.format_exc() + "\n")
                gc.collect()
                torch.cuda.empty_cache()
            logf.flush()

    print(f"\n[{round_tag}] done. {len(todo) - len(failures)}/{len(todo)} succeeded, "
          f"{len(failures)} failed.")
    for c, lr, s, e in failures:
        print(f"  FAIL {c} lr={lr:g} seed={s}: {e}")


def rank_round1():
    """Rank configs by mean acc across the round-1 seeds, each at its auto-LR."""
    lrs = load_auto_lr()
    rows = []
    for config in CONFIGS:
        if config not in lrs:
            continue
        lr = lrs[config]
        accs = []
        for seed in ROUND1_SEEDS:
            p = result_path(config, lr, seed)
            if os.path.exists(p):
                with open(p) as f:
                    accs.append(json.load(f)["mean_acc"])
        if accs:
            rows.append((config, float(np.mean(accs)), float(np.std(accs)), len(accs)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def select_round2_configs():
    # ordering-study configs are 3-seed screening only; keep them out of the
    # finalist ranking so they can't displace a real finalist.
    rows = [r for r in rank_round1() if r[0] not in ORDERING_EXTRA]
    have = {r[0] for r in rows}
    pool = [c for c in CONFIGS if c not in ORDERING_EXTRA]
    missing = [c for c in pool if c not in have]
    if len(rows) < len(pool):
        print(f"WARNING: round1 incomplete — {len(rows)}/{len(pool)} configs have "
              f"results. Missing: {missing}")
    top5 = [r[0] for r in rows[:5]]
    bottom3 = [r[0] for r in rows[-3:]]
    selected = list(dict.fromkeys(top5 + bottom3))  # dedupe, preserve order
    return selected, top5, bottom3


# ───────────────────────────────── main ───────────────────────────────────────

def make_plan_round1(dry_run=False):
    """18 configs x 3 screening seeds, each at its auto-LR (searched here if needed)."""
    lrs = resolve_lrs(dry_run=dry_run)
    return [(c, lrs[c], s) for c in CONFIGS for s in ROUND1_SEEDS]


def make_plan_round2(selected, dry_run=False):
    """Finalists at their auto-LR, all 5 seeds (round-1 seeds 0/1/2 resume-skip)."""
    lrs = resolve_lrs(dry_run=dry_run)
    return [(c, lrs[c], s) for c in selected for s in ROUND2_SEEDS]


def print_plan(plan, round_tag):
    print(f"=== DRY RUN: {round_tag} — {len(plan)} runs "
          f"({sum(not already_done(*p) for p in plan)} not yet done) ===")
    by_cfg = {}
    for c, lr, s in plan:
        by_cfg.setdefault(c, []).append((lr, s))
    for c in by_cfg:
        lrs = sorted({lr for lr, _ in by_cfg[c]})
        seeds = sorted({s for _, s in by_cfg[c]})
        print(f"  {c:16s} {CONFIGS[c]}  lrs={[f'{l:g}' for l in lrs]} "
              f"seeds={seeds}  ({len(by_cfg[c])} runs)")


def main():
    ap = argparse.ArgumentParser(description="8-layer hybrid decode-recall sweep")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--round1", action="store_true",
                   help="18-config architecture sweep at default lr, 5 seeds (90 runs)")
    g.add_argument("--round2", action="store_true",
                   help="lr sweep on top-5 + bottom-3 configs from round1")
    g.add_argument("--config", type=str, choices=list(CONFIGS),
                   help="run a single named config (all seeds, default lr)")
    g.add_argument("--rank", action="store_true",
                   help="just print the round1 ranking and the round2 selection")
    g.add_argument("--autolr", action="store_true",
                   help="run the per-config auto-LR search for all 18 configs and cache it")
    ap.add_argument("--seed", type=int, default=None,
                    help="restrict --config to a single seed (for smoke tests)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the lr for --config (else use the config's auto-LR)")
    ap.add_argument("--force", action="store_true",
                    help="with --autolr, recompute lrs even if already cached")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    ap.add_argument("--finalists", type=str, default=None,
                    help="with --round2, comma-separated config names to use as "
                         "finalists, overriding the auto top5+bottom3 selection")
    args = ap.parse_args()

    if args.autolr:
        print(f"Auto-LR search [{cond_tag()}]: {LR_ITERS}x{len(LR_GRID)} {LR_SEARCH_EPOCHS}-epoch runs "
              f"per config over {LR_GRID[0]:.0e}..{LR_GRID[-1]:.0e}", flush=True)
        for c in CONFIGS:
            find_best_lr(c, force=args.force)
        cache = load_auto_lr()
        print("\nCached auto-LR per config:")
        for c in CONFIGS:
            if c in cache:
                print(f"  {c:16s} {CONFIGS[c]}  lr={cache[c]:.2e}")
        return

    if args.rank:
        rows = rank_round1()
        print("Round1 ranking (mean acc across seeds):")
        for i, (c, m, sd, n) in enumerate(rows, 1):
            print(f"  {i:2d}. {c:16s} {CONFIGS[c]}  acc={m:.4f} ± {sd:.4f}  (n={n})")
        sel, top5, bottom3 = select_round2_configs()
        print(f"\nRound2 selection (top5 + bottom3): {sel}")
        return

    if args.round1:
        plan, tag = make_plan_round1(dry_run=args.dry_run), "round1"
    elif args.round2:
        if args.finalists:
            selected = [c.strip() for c in args.finalists.split(",") if c.strip()]
            bad = [c for c in selected if c not in CONFIGS]
            if bad:
                ap.error(f"--finalists has unknown configs: {bad}")
            print(f"Round2 configs (explicit --finalists): {selected}")
        else:
            selected, top5, bottom3 = select_round2_configs()
            print(f"Round2 configs — top5={top5} bottom3={bottom3}")
        plan, tag = make_plan_round2(selected, dry_run=args.dry_run), "round2"
    else:  # --config
        if args.lr is not None:
            lr = args.lr
        elif args.dry_run:
            lr = load_auto_lr().get(args.config, DEFAULT_LR)
        else:
            lr = find_best_lr(args.config)
        seeds = [args.seed] if args.seed is not None else ROUND1_SEEDS
        plan, tag = [(args.config, lr, s) for s in seeds], f"config_{args.config}"

    if args.dry_run:
        print_plan(plan, tag)
        return

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training/eval are hardcoded to cuda.")
    run_plan(plan, tag)


if __name__ == "__main__":
    main()
