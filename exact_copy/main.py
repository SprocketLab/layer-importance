from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task import *
from src.model import *

# Default task setting. l = 5 is wider than a single generic Mamba conv
# (kernel 4) so no layer can fake the comparison locally, and c = 3 asks for
# r = (c-1) l = 10 consecutive equalities.
DEFAULT_L = 100
DEFAULT_VOCAB = 20
DEFAULT_BLOCK = 5
DEFAULT_REPS = 3

# Both arms are permutations of the same blocks, so one width list serves both.
DEFAULT_WIDTHS = [16, 24, 32, 48]

# The stack the construction is written into. "The construction" always means
# this one -- --verify, self-test [11], the star on the sweep figures -- and its
# depth is the one the layer-count ablations vary away from.
CONSTRUCTION_LAYERS = ["TF", "SSM", "SSM"]
CONSTRUCTION_DEPTH = len(CONSTRUCTION_LAYERS)

# The pair of arms every parameter-matched figure starts from: the same three
# blocks, in the two orders the theory separates.
DEFAULT_CONFIGS = ["construction:TSS", "construction:SST"]

# The four orders of the equal-memory figure: the two hybrids and the two pure
# stacks. Plain "SSM" makes the first SSM of a stack two-gated, which is the block
# the construction latches with, so TF->SSM here is the depth-2 stack that has a
# chance of expressing the task rather than a deliberately weakened one. Pass the
# SSM1 spellings instead for the single-gated family.
MEMORY_ARMS = ["construction:TS", "construction:ST",
               "construction:TT", "construction:SS"]


def CONTROL_WINDOWS(args):
    """Attention windows for the control arm, as multiples of the copy length.

    The lower bound is driven by h = min(l, (L-W)_+), so it is at full strength
    for every W <= L-l and says nothing at all once W = L. These three windows
    all sit in the regime where the bound bites; what separates them is the
    reach of a depth-n attention stack, n(W-1)+1, which is what decides whether
    a pure TF can carry "a copy occurred" forward to position L.
    """
    l = args.block
    return tuple(w for w in (l + 1, 2 * l + 1, 4 * l + 1) if w <= args.length)

# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #

_EVAL_CACHE: dict = {}


def fixed_eval_set(args, n: int):
    """One held-out set, identical for every run, so curves differ only by model."""
    key = (n, args.length, args.vocab, args.block, args.reps, args.hard_neg_frac,
           args.eval_seed)
    if key not in _EVAL_CACHE:
        rng = np.random.default_rng(args.eval_seed)
        x, y = sample_batch(rng, n, args.length, args.vocab, args.block, args.reps,
                            hard_neg_frac=args.hard_neg_frac)
        _EVAL_CACHE[key] = (torch.as_tensor(x), torch.as_tensor(y))
    return _EVAL_CACHE[key]


@torch.no_grad()
def evaluate(model, x, y, batch):
    """Accuracy of y_L (the task metric, chance 0.5) and over all positions."""
    model.eval()
    final_hits = dense_hits = 0
    for s in range(0, len(x), batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        pred = model(xb).argmax(-1)
        final_hits += (pred[:, -1] == yb[:, -1]).sum().item()
        dense_hits += (pred == yb).sum().item()
    return final_hits / len(x), dense_hits / y.numel()


def train_one(args, d, variant, layers, seed, lr, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)                            # same data for both arms
    x_eval, y_eval = fixed_eval_set(args, args.eval_examples)
    x_eval, y_eval = x_eval.to(device), y_eval.to(device)
    probe = min(1024, args.eval_examples)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(args, d, variant, layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup = max(1, args.steps // 10)

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    best_final, t0 = 0.0, time.time()
    history, solve_step = [], None
    for step in range(1, args.steps + 1):
        model.train()
        x, y = sample_batch(rng, args.batch, args.length, args.vocab, args.block,
                            args.reps, hard_neg_frac=args.hard_neg_frac)
        x = torch.as_tensor(x, device=device)
        y = torch.as_tensor(y, device=device)

        logits = model(x)
        if args.supervision == "last":
            loss = F.cross_entropy(logits[:, -1], y[:, -1])
        else:
            loss = F.cross_entropy(logits.reshape(-1, 2), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            final_acc, dense_acc = evaluate(model, x_eval[:probe], y_eval[:probe], args.batch)
            best_final = max(best_final, final_acc)
            history.append([step, round(final_acc, 4), round(dense_acc, 4)])
            if solve_step is None and final_acc >= args.solve_threshold:
                solve_step = step
            if args.verbose:
                print(f"    step {step:5d}  loss {loss.item():.4f}  "
                      f"final_acc {final_acc:.3f}  dense_acc {dense_acc:.3f}", flush=True)

    final_acc, dense_acc = evaluate(model, x_eval, y_eval, args.batch)
    peak = (torch.cuda.max_memory_reserved(device) / 1024 ** 3
            if device.type == "cuda" else 0.0)
    return {
        "variant": variant,
        "layers": ",".join(layers),
        "d": d,
        "seed": seed,
        "lr": lr,
        "params": count_params(model),
        "window": resolve_window(args),
        # the other budget: what this run carries along the sequence at inference,
        # which is what the equal-memory figure matches instead of the parameters
        "state_dim": args.state_dim,
        "memory": stack_memory(layers, d, resolve_window(args), args.state_dim,
                               args.length),
        "final_acc": final_acc,
        "best_final_acc": max(best_final, final_acc),
        "dense_acc": dense_acc,
        "solved": bool(final_acc >= args.solve_threshold),
        "solve_step": solve_step,
        "history": history,
        "peak_gpu_gib": peak,
        "minutes": (time.time() - t0) / 60,
    }


def widths_for(args, variant, layers):
    if args.widths:
        return sorted(set(args.widths))
    if args.match_params:
        return widths_for_targets(args, variant, layers, args.targets)
    return list(DEFAULT_WIDTHS)


def widths_for_targets(args, variant, layers, targets):
    """Pick, per arm, the width landing closest to each parameter budget.

    Needed once arms stop being permutations of one another: a TF block costs
    almost twice an SSM block, so equal width would mean very unequal capacity.
    Widths are searched one at a time (odd values included) because the budgets
    have to be hit as tightly as the quadratic-in-d grid allows.
    """
    widths = []
    for target in targets:
        best = None
        for d in range(8, 400):
            params = count_params(build_model(args, d, variant, layers))
            if best is None or abs(params - target) < abs(best[1] - target):
                best = (d, params)
            if params > 1.6 * target:
                break
        widths.append(best[0])
    return sorted(set(widths))


def run_config(args, variant, layers, device, on_run=None):
    runs = []
    for d in widths_for(args, variant, layers):
        for lr in args.lrs:
            # for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            seed = args.seed
            res = train_one(args, d, variant, layers, seed, lr, device)
            runs.append(res)
            jump = "" if res["solve_step"] is None else f" solved@{res['solve_step']}"
            if args.verbose:
                print(f"  {variant} {','.join(layers)} d={d:>3} params={res['params']:>6} "
                    f"lr={lr:g} seed={seed} final_acc={res['final_acc']:.3f} "
                    f"best={res['best_final_acc']:.3f} dense={res['dense_acc']:.3f}{jump} "
                    f"({res['minutes']:.1f} min, peak {res['peak_gpu_gib']:.2f} GiB)",
                    flush=True)
            if on_run is not None:
                on_run(runs)
    return runs


def wilson_interval(hits: int, total: int, z: float = 1.96):
    """95% interval for a success rate. The right error bar for "k of n runs solved"."""
    if total == 0:
        return 0.0, 0.0
    p = hits / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def best_lr_curve(runs, partial_threshold=0.55):
    """Per width, keep the lr that solves most often (matched tuning budget).

    The outcome distribution is bimodal -- a run either stays at chance or jumps
    to near-perfect -- so the headline number is the solve rate, not the mean
    accuracy: averaging a 0.99 with two 0.49 produces a 0.66 that no run ever
    achieved. The mean is still reported, as a secondary statistic.
    """
    by_d = {}
    for run in runs:
        by_d.setdefault(run["d"], {}).setdefault(run["lr"], []).append(run)
    points = []
    for d in sorted(by_d):
        def score(rs):
            return (np.mean([r["solved"] for r in rs]), np.mean([r["final_acc"] for r in rs]))
        best = max(by_d[d].values(), key=score)
        accs = [r["final_acc"] for r in best]
        hits = int(sum(r["solved"] for r in best))
        lo, hi = wilson_interval(hits, len(best))
        steps = [r["solve_step"] for r in best if r["solve_step"] is not None]
        points.append({
            "d": d, "params": best[0]["params"], "lr": best[0]["lr"],
            "n": len(best), "solved": hits, "solve_rate": hits / len(best),
            "ci_lo": lo, "ci_hi": hi,
            "partial": int(sum(a >= partial_threshold for a in accs)),
            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "max": float(np.max(accs)),
            "median_solve_step": float(np.median(steps)) if steps else None,
        })
    return points


# --------------------------------------------------------------------------- #

layer_assoc = {'S': 'SSM', 'T': 'TF'}
def parse_configs(specs):
    configs = []
    for spec in specs:
        variant, _, layers = spec.partition(":")
        assert variant in ("construction", "generic"), f"bad variant in {spec!r}"
        #layers = layers.split(",")
        layers = [layer_assoc[c] for c in layers]
        bad = [k for k in layers if k not in LAYER_KINDS]
        assert not bad, f"unknown layer(s) {bad} in {spec!r}; expected {LAYER_KINDS}"
        configs.append((variant, layers))
    return configs



# --------------------------------------------------------------------------- #
# Accuracy against token dimension at one inference-memory budget
# --------------------------------------------------------------------------- #

def memory_points(runs, partial_threshold=0.55):
    """Per token dimension: the best-lr mean/sd, and the (W, N, memory) behind it.

    `best_lr_curve` already groups by width and keeps the learning rate that solves
    most often, which is exactly one point per token dimension here, since the
    equal-memory sweep runs one width per setting.
    """
    points = best_lr_curve(runs, partial_threshold)
    settings = {}
    for r in runs:
        settings.setdefault(r["d"], set()).add(
            (r["window"], r.get("state_dim"), r.get("memory")))
    for p in points:
        found = settings[p["d"]]
        assert len(found) == 1, \
            f"d={p['d']} was trained at more than one (W, N, memory): {found}"
        (p["window"], p["state_dim"], p["memory"]), = found
    return sorted(points, key=lambda p: p["d"])


def memory_sweep(args, device):
    """Train every arm at every token dimension under one inference-memory budget.

    The four arms are the four ways to order two kinds of block, which is what the
    figure compares; `--token-dims` moves width against window and state, since at
    a fixed budget a wider model can only afford a shorter window and a smaller
    state. For exact copy that is a real cliff rather than a gentle trade: the
    construction needs W >= l+1 to read offset l, so beyond d = budget/(2(l+1)depth)
    no width can express the task at all.
    """
    #os.makedirs(args.out_dir, exist_ok=True)
    specs = args.configs if args.configs != DEFAULT_CONFIGS else MEMORY_ARMS
    if specs is MEMORY_ARMS:
        print(f"--configs left at its default, using the four orders: "
              f"{' '.join(MEMORY_ARMS)}")
    configs = parse_configs(specs)
    depths = {len(layers) for _, layers in configs}
    layers = [layers for _, layers in configs]
    assert len(depths) == 1, f"the arms must share a depth, got {sorted(depths)}"
    depth = depths.pop()
    layers = layers.pop()

    plan = [(d,) + equal_memory_setting(d, layers, args.memory_budget, args.length, args.w_to_s)
            for d in sorted(args.token_dims)]

    # Fail before spending any compute: if a token dimension does not divide the
    # per-layer share, the arms end up carrying slightly different state and the
    # figure's one claim stops being true.
    """
    uneven = []
    for d, w, n in plan:
        spent = {stack_memory(l, d, w, n, args.length) for _, l in configs}
        if len(spent) > 1 or spent.pop() != args.memory_budget:
            uneven.append(d)
    assert not uneven, (
        f"token dims {uneven} do not split {args.memory_budget} floats evenly at "
        f"depth {depth}: a layer's share of {args.memory_budget // depth} must be a "
        f"multiple of both 2d and d. Dims that work here: "
        f"{memory_grid_dims(depth, args.memory_budget, hi=max(args.token_dims))}")
    """

    if args.verbose:
        print(f"equal-memory sweep, depth {depth}, budget {args.memory_budget} floats, "
            f"{len(configs)} arms x {len(plan)} token dims x {len(args.lrs)} lrs x "
            f"{args.seeds} seeds = {len(configs) * len(plan) * len(args.lrs) * args.seeds} runs")
    for d, w, n in plan:
        mems = {"-".join(l): stack_memory(l, d, w, n, args.length) for _, l in configs}
        flag = "" if w >= args.block + 1 else f"   W < l+1={args.block + 1}"
    if args.verbose:
        print(f"  d={d:>4}  W={w:>4}  N={n:>4}  memory={sorted(set(mems.values()))}"
              f"  params={sorted({count_params(build_model_at(args, d, v, l, w, n)) for v, l in configs})}"
              f"{flag}")

    saved = (args.window, args.state_dim, args.widths)
    curves = {}
    try:
        for d, w, n in plan:
            args.window, args.state_dim, args.widths = w, n, [d]
            for variant, layers in configs:
                name = f"{variant}: {'->'.join(layers)}"
                #print(f"=== {name}  d={d} W={w} N={n} ===", flush=True)
                curves.setdefault(name, []).extend(
                    run_config(args, variant, layers, device))
                # A long sweep should survive being killed. The args written out
                # are the sweep's, not the point currently being trained, since
                # (W, N, d) vary within one file and live on the runs themselves.
                blob = dict(vars(args))
                blob.update(window=saved[0], state_dim=saved[1], widths=saved[2])
                #with open(os.path.join(args.out_dir, "runs.json"), "w") as f:
                #    json.dump({"args": blob, "curves": curves}, f)
                print({'args':blob, 'curbes': curves})
    finally:
        args.window, args.state_dim, args.widths = saved
    return curves


def build_model_at(args, d, variant, layers, window, state_dim):
    """A model at an explicit (W, N), for counting parameters without mutating args."""
    return Hybrid(vocab=args.vocab, d=d, layers=layers, variant=variant,
                  heads=args.heads, window=window, state_dim=state_dim,
                  ssm_conv=args.ssm_conv, expand=args.expand,
                  conv_kernel=args.conv_kernel)



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true", help="run correctness checks and exit")
    p.add_argument("--verify", action="store_true",
                   help="hand-build the construction and check it is exact")
    p.add_argument("--params-table", action="store_true", help="print parameter counts and exit")
    p.add_argument("--compare", action="store_true", help="run every --configs entry, one figure")
    p.add_argument("--sweep", action="store_true", help="run a single layer order")
    p.add_argument("--replot", type=str, nargs="+", default=None,
                   help="merge the runs.json of these dirs into one figure and exit")
    p.add_argument("--acc-figures", type=str, nargs="+", default=None,
                   help="from these dirs, draw accuracy vs parameters alone, one "
                        "figure per gating family, and exit (no training)")
    p.add_argument("--memory-sweep", action="store_true",
                   help="sweep --token-dims at a fixed inference-memory budget, "
                        "deriving the window and state dim from --memory-budget, and "
                        "draw accuracy against token dimension for the four orders")
    p.add_argument("--memory-figure", type=str, nargs="+", default=None,
                   help="redraw the equal-memory figure from these dirs and exit")

    p.add_argument("--length", type=int, default=DEFAULT_L)
    p.add_argument("--vocab", type=int, default=DEFAULT_VOCAB, help="|M|, content tokens")
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="block length l")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS, help="repetition count c")
    p.add_argument("--hard-neg-frac", type=float, default=1.0,
                   help="fraction of negatives built by planting then breaking a copy; "
                        "1.0 closes the token-frequency shortcut, 0.0 is plain noise")

    p.add_argument("--configs", type=str, nargs="+", default=DEFAULT_CONFIGS,
                   help="arms to run; the default pair differs only in layer order. "
                        "Stacks of any depth are accepted, e.g. construction:TF,SSM "
                        "and construction:SSM,TF for the depth-2 ablation")
    p.add_argument("--variant", choices=["construction", "generic"], default="construction")
    p.add_argument("--layers", type=str, default="TSS")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--state-dim", type=int, default=4)
    p.add_argument("--window", type=str, default="full",
                   help="shared attention window: an int, or 'full' for the whole prefix")
    p.add_argument("--ssm-conv", type=int, default=0,
                   help="depthwise causal conv width inside every SSM block (0 = none)")
    p.add_argument("--expand", type=int, default=2, help="generic Mamba expansion factor")
    p.add_argument("--conv-kernel", type=int, default=4, help="generic Mamba conv width")

    p.add_argument("--widths", type=int, nargs="+", default=None,
                   help=f"model widths to run (default {DEFAULT_WIDTHS})")
    p.add_argument("--targets", type=int, nargs="+",
                   default=[5000, 8000, 12000, 16000, 20000, 28000, 40000],
                   help="parameter budgets, only used with --match-params")
    p.add_argument("--match-params", action="store_true",
                   help="solve --targets per arm; only needed if arms use different blocks")
    p.add_argument("--token-dims", type=int, nargs="+", default=DEFAULT_TOKEN_DIMS,
                   help=f"token dimensions of the equal-memory sweep "
                        f"(default {DEFAULT_TOKEN_DIMS})")
    p.add_argument("--memory-budget", type=int, default=DEFAULT_MEMORY_BUDGET,
                   help="floats of inference state every arm is allowed to carry: a TF "
                        "layer spends its share on a window (Wd), an SSM layer on a "
                        f"state (Nd). Default {DEFAULT_MEMORY_BUDGET}")
    p.add_argument("--w-to-s", type=float, default=1)
    p.add_argument("--verify-width", type=int, default=None,
                   help="width of the hand-built construction (default: the minimum)")
    p.add_argument("--plot-construction", action="store_true", default=True)
    p.add_argument("--no-plot-construction", dest="plot_construction", action="store_false")

    p.add_argument("--supervision", choices=["latched", "last"], default="latched",
                   help="'latched' supervises y_t at every position, as the construction emits")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2],
                   help="learning-rate grid, shared by all arms; the best one is plotted")
    p.add_argument("--seed", type=int, default=1)
    # p.add_argument("--seed-offset", type=int, default=0,
    #                help="first seed; --seeds counts up from here. A seed fixes both "
    #                     "the init and the training batches, so one sweep can be split "
    #                     "across processes (or scheduler jobs) by seed and merged later "
    #                     "into exactly the runs a single process would have produced")
    p.add_argument("--solve-threshold", type=float, default=0.9,
                   help="y_L accuracy at which a run counts as having solved the task")
    p.add_argument("--partial-threshold", type=float, default=0.55,
                   help="secondary threshold for 'learned something above chance'")
    p.add_argument("--scan", choices=["parallel", "sequential"], default="parallel",
                   help="'parallel' is the log(L) associative scan, ~6x faster and "
                        "numerically identical in the construction's 0/1 regime")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-examples", type=int, default=4096,
                   help="size of the fixed held-out set")
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--max-gpu-gb", type=float, default=18.0,
                   help="hard per-process GPU memory cap in GiB (0 disables); this job "
                        "needs well under 1 GiB, the cap is there to protect other users")
    p.add_argument("--out-dir", type=str, default="results")
    #p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def cap_gpu_memory(device, max_gb: float) -> None:
    """Refuse to allocate past `max_gb`, so a shared GPU cannot be starved by us.

    The cap is enforced by the caching allocator inside this process: crossing it
    raises OOM here instead of taking memory other jobs are relying on.
    """
    if device.type != "cuda" or not max_gb:
        return
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, max_gb * (1024 ** 3) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    print(f"gpu cap: {max_gb:.1f} GiB of {total / 1024 ** 3:.1f} GiB "
          f"(fraction {fraction:.3f}) on {torch.cuda.get_device_name(index)}")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    # Check to see if this setting makes sense
    depth_tf = sum([1 if i == 'T' else 0 for i in args.layers])
    depth_ssm = sum([1 if i == 'S' else 0 for i in args.layers])
    minimum_budget = args.token_dims[0] * (depth_tf * args.w_to_s + depth_ssm)
    if minimum_budget <= args.memory_budget:
        memory_sweep(args, device)
    else:
        pass

