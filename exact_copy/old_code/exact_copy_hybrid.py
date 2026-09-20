"""Exact Copy: TF-first vs SSM-first hybrids, compared under one architecture family.

The question is about *layer order*: does putting the attention layer before the
two SSM layers matter for a task whose only hard step is a fixed-offset
comparison?  So the two arms of the experiment are

    TF -> SSM -> SSM        and        SSM -> SSM -> TF

built from the *same three blocks*, only permuted:

    TF   sliding-window attention with a learnable bias per offset, plus an MLP.
         Window and bias are what the construction uses to fetch x_{t-l}; the
         hard bias B_{j-l} is generalised to a learnable per-offset bias.
    SSM  (first one in the stack) a two-gated selective SSM with independent
         Delta_A, Delta_B, which is what the construction's run-length counter
         H_t = e_t (H_{t-1} + 1) needs to exceed 1.
    SSM  (second one) a single-gate selective SSM, which is all the latch
         y_t = (1 - z_t) y_{t-1} + z_t needs.

Because the two arms are permutations of one multiset of blocks, they have
*exactly* the same parameter count at every width, so the parameter axis is
matched by construction rather than by hand-tuned widths. Everything else is
shared too: the same attention window, the same optional depthwise conv, the
same training batches (same numpy seed), one fixed held-out evaluation set for
every run, and the same learning-rate grid with the best lr picked per arm.
`--self-test` asserts the parameter-equality invariant.

On top of the two learned curves, `--verify` sets the weights of the TF-first
model *by hand* to the construction and checks it classifies every sequence
correctly without any training. That point is drawn on the figure as the
construction's ceiling: it says the architecture expresses the task at that
parameter count, so whatever a trained model fails to reach is an optimisation
gap, not an expressivity gap.

The task distribution uses hard negatives by default: a negative is built by
planting a copy and then swapping two tokens inside one of the repeated blocks.
That leaves the token multiset untouched, so no frequency statistic can separate
the classes and the model has to verify the l(c-1) equalities.

Usage
    python exact_copy_hybrid.py --self-test      # correctness checks, no training
    python exact_copy_hybrid.py --verify         # hand-built construction -> 100%
    python exact_copy_hybrid.py --params-table   # parameter counts per width
    python exact_copy_hybrid.py --compare        # both orders, one figure
    python exact_copy_hybrid.py --sweep          # a single order

Notes on the knobs that decide whether the comparison is fair:
    --window      shared by both arms. `full` (default) gives both arms causal
                  attention over the whole prefix, so the SSM-first arm is not
                  handicapped by locality. `--window 6` (= l+1) is the setting
                  the construction actually needs.
    --ssm-conv    a depthwise causal conv inside every SSM block, off by
                  default. Turning it on hands both arms local access to
                  x_{t-k+1..t}; note that n stacked SSM layers reach offset
                  n(k-1), so keep l > n(k-1) (e.g. --block 8 --ssm-conv 4) if
                  you do not want the conv to be able to do the comparison.
    --lrs         one grid for both arms; the plotted point is the best lr.
"""

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

# Default task setting. l = 5 is wider than a single generic Mamba conv
# (kernel 4) so no layer can fake the comparison locally, and c = 3 asks for
# r = (c-1) l = 10 consecutive equalities.
DEFAULT_L = 100
DEFAULT_VOCAB = 20
DEFAULT_BLOCK = 5
DEFAULT_REPS = 3

# Both arms are permutations of the same blocks, so one width list serves both.
DEFAULT_WIDTHS = [16, 24, 32, 48]


def CONTROL_WINDOWS(args):
    """Attention windows for the control arm, as multiples of the copy length.

    The lower bound is driven by h = min(l, (L-W)_+), so it is at full strength
    for every W <= L-l and says nothing at all once W = L. These three windows
    all sit in the regime where the bound bites; what separates them is the
    reach of a depth-3 attention stack, 3(W-1)+1, which is what decides whether
    a pure TF can carry "a copy occurred" forward to position L.
    """
    l = args.block
    return tuple(w for w in (l + 1, 2 * l + 1, 4 * l + 1) if w <= args.length)


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #

def latched_targets(x: np.ndarray, l: int, c: int) -> np.ndarray:
    """Running indicator y_t = 1{x_{1:t} contains a completed exact copy}.

    Uses the run-length characterisation: with e_t = 1{t > l and x_t == x_{t-l}},
    a copy completes at t iff the r = (c-1) l equalities ending at t are all one.
    `contains_exact_copy` checks the definition directly instead.
    """
    r = (c - 1) * l
    n, L = x.shape
    run = np.zeros(n, dtype=np.int64)
    y = np.zeros((n, L), dtype=np.int64)
    latched = np.zeros(n, dtype=bool)
    for t in range(L):
        if t >= l:
            run = np.where(x[:, t] == x[:, t - l], run + 1, 0)
        else:
            run[:] = 0
        latched |= run >= r
        y[:, t] = latched
    return y


def contains_exact_copy(seq, l: int, c: int) -> int:
    """First position at which a copy completes, or -1. Brute force over the definition."""
    span = c * l
    for a in range(0, len(seq) - span + 1):
        if all(
            seq[a + j + k * l] == seq[a + j + (k + 1) * l]
            for j in range(l)
            for k in range(c - 1)
        ):
            return a + span - 1
    return -1


def _sample_block(rng, vocab: int, l: int) -> np.ndarray:
    """A block with at least two distinct values, so that a swap can break it."""
    while True:
        block = rng.integers(0, vocab, size=l)
        if l < 2 or len(np.unique(block)) > 1:
            return block


def _break_copy(rng, row: np.ndarray, a: int, l: int, c: int, vocab: int) -> None:
    """Destroy a planted copy in place, keeping the token multiset unchanged.

    Swapping two unequal tokens inside one of the c blocks leaves every count-based
    statistic of the sequence identical to the positive it was built from, while
    the longest run of equalities drops below r. Constant blocks (or l = 1) have no
    swap available, so one token is replaced instead.
    """
    k = int(rng.integers(0, c))
    base = a + k * l
    block = row[base:base + l]
    pairs = [(p, q) for p in range(l) for q in range(p + 1, l) if block[p] != block[q]]
    if pairs:
        p, q = pairs[int(rng.integers(0, len(pairs)))]
        row[base + p], row[base + q] = row[base + q], row[base + p]
    else:
        p = int(rng.integers(0, l))
        new = int(rng.integers(0, vocab - 1))
        row[base + p] = new + (new >= row[base + p])


def sample_batch(rng, n, L, vocab, l, c, p_pos=0.5, hard_neg_frac=1.0):
    """Balanced batch of sequences with latched targets.

    A `p_pos` fraction of rows get a planted copy. Of the remaining rows, a
    `hard_neg_frac` fraction also get a planted copy that is then broken by
    `_break_copy` (a hard negative); the rest are left as uniform noise.
    Labels always come from detection on the final sequence, so a row that
    accidentally contains a copy is still labelled correctly.
    """
    x = rng.integers(0, vocab, size=(n, L))
    span = c * l
    positive = rng.random(n) < p_pos
    hard = (~positive) & (rng.random(n) < hard_neg_frac)
    for i in np.flatnonzero(positive | hard):
        a = int(rng.integers(0, L - span + 1))
        block = _sample_block(rng, vocab, l)
        for k in range(c):
            x[i, a + k * l:a + (k + 1) * l] = block
        if hard[i]:
            _break_copy(rng, x[i], a, l, c, vocab)
    return x, latched_targets(x, l, c)


def frequency_shortcut_accuracy(x: np.ndarray, y_final: np.ndarray, l: int, c: int) -> float:
    """Best accuracy reachable by thresholding a pure token-frequency feature.

    The feature is the largest number of excess repeats inside any window of
    length c*l, which carries no positional or periodic information at all. With
    easy negatives it separates the classes almost perfectly; with hard negatives
    it should sit at chance, which is the point of the hard negatives.
    """
    span = c * l
    feats = np.array([
        max(span - len(np.unique(row[s:s + span])) for s in range(len(row) - span + 1))
        for row in x
    ])
    best = 0.0
    for threshold in np.unique(feats):
        best = max(best, ((feats >= threshold) == y_final).mean())
        best = max(best, ((feats < threshold) == y_final).mean())
    return float(best)


# --------------------------------------------------------------------------- #
# Recurrence
# --------------------------------------------------------------------------- #

SCAN_MODE = "parallel"


def linear_scan_sequential(decay, inp):
    """Reference implementation of S_t = decay_t * S_{t-1} + inp_t, one step at a time."""
    state = torch.zeros_like(inp[:, 0])
    out = []
    for t in range(inp.shape[1]):
        state = decay[:, t] * state + inp[:, t]
        out.append(state)
    return torch.stack(out, dim=1)


def linear_scan_parallel(decay, inp):
    """Same recurrence in log2(L) rounds instead of L, via the associative form.

    Writing the step at t as the pair (a_t, b_t) acting by S -> a_t S + b_t,
    composing "first (a1,b1) then (a2,b2)" gives (a2 a1, a2 b1 + b2), which is
    associative, so a Hillis-Steele prefix scan computes every S_t at once. The
    sequential version needs L rounds of tiny kernels and is launch-bound on a
    GPU; this one needs 7 rounds at L = 100 and is several times faster.
    """
    a, b = decay, inp
    step = 1
    while step < inp.shape[1]:
        a_prev = F.pad(a[:, :-step], (0, 0, 0, 0, step, 0), value=1.0)
        b_prev = F.pad(b[:, :-step], (0, 0, 0, 0, step, 0), value=0.0)
        b = a * b_prev + b
        a = a * a_prev
        step *= 2
    return b


def linear_scan(decay, inp, C):
    """S_t = decay_t * S_{t-1} + inp_t,  out_t = <C_t, S_t>.

    decay/inp: (n, L, d, s); C: (n, L, s); returns (n, L, d).
    """
    states = (linear_scan_parallel(decay, inp) if SCAN_MODE == "parallel"
              else linear_scan_sequential(decay, inp))
    return (states * C.unsqueeze(2)).sum(-1)


def selective_scan(u, dt_a, dt_b, B, C, A):
    """The construction's recurrence: S_t = (1 - dt_a_t A) S_{t-1} + (dt_b_t u_t) B_t.

    With dt in (0, 1) and A in (0, 1) the decay stays in (0, 1), so this is
    stable. Passing dt_a == dt_b recovers a single shared gate.
    """
    decay = 1.0 - dt_a.unsqueeze(-1) * A
    inp = (dt_b * u).unsqueeze(-1) * B.unsqueeze(2)
    return linear_scan(decay, inp, C)


# --------------------------------------------------------------------------- #
# Blocks (one family, used by both arms)
# --------------------------------------------------------------------------- #

class OffsetAttention(nn.Module):
    """Causal attention over the last `window` positions with a learnable bias per offset.

    The construction picks out exactly offset l by setting W_q = W_k = 0 and using
    a hard positional bias, so the learnable analogue is one bias per offset added
    to the logits. Positions before the start of the sequence behave like zero
    keys and values rather than being dropped, which is what makes the
    construction's boundary rule (attention output vanishes for t < l) exact.
    Those slots are folded into a single extra logit per query ("sink"), since a
    zero key gives logit 0 + bias and a zero value contributes nothing.
    """

    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        assert d % heads == 0
        self.heads, self.head_dim, self.window = heads, d // heads, window
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.rel_bias = nn.Parameter(torch.zeros(heads, window))

    def forward(self, h):
        n, L, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (n, L, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        idx = torch.arange(L, device=h.device)
        offset = idx[:, None] - idx[None, :]                      # (L, L)
        allowed = (offset >= 0) & (offset < self.window)

        bias = self.rel_bias[:, offset.clamp(0, self.window - 1)]  # (H, L, L)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim) + bias
        logits = logits.masked_fill(~allowed, float("-inf"))

        # Slots before the start of the sequence: a zero key makes their logit the
        # bias alone and a zero value contributes nothing, so they only enter the
        # softmax denominator. All of them collapse into one extra column.
        offsets = torch.arange(self.window, device=h.device)
        out_of_range = (idx[:, None] - offsets[None, :]) < 0        # (L, window)
        pad_bias = self.rel_bias[:, None, :].expand(self.heads, L, self.window)
        sink = torch.where(out_of_range[None], pad_bias,
                           torch.full_like(pad_bias, float("-inf")))
        sink = torch.logsumexp(sink, dim=-1)                       # (H, L)

        full = torch.cat([logits, sink[None, :, :, None].expand(n, -1, -1, 1)], dim=-1)
        out = full.softmax(-1)[..., :L] @ v
        return self.proj(out.transpose(1, 2).reshape(n, L, -1))


class TFBlock(nn.Module):
    """Attention that can fetch x_{t-l}, then an MLP that can form the equality bit.

    With +-1 codes the MLP realises the comparison as a Hamming distance,
    e_t = 1 iff sum_j |psi_j(x_t) - psi_j(x_{t-l})| = 0, using 2 log|M| ReLU
    units; the threshold itself is applied by the sigmoid inside the next
    layer's gate.
    """

    def __init__(self, d: int, heads: int, window: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = OffsetAttention(d, heads, window)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, h):
        h = h + self.attn(self.norm1(h))
        return h + self.fc2(F.relu(self.fc1(self.norm2(h))))


class SSMBlock(nn.Module):
    """Selective SSM layer. `two_gated=True` gives independent Delta_A, Delta_B.

    The projections carry biases because the construction needs exact constants
    on the value path (u = 1), on B and C (= 1) and on the output gate; without a
    bias those constants would have to be read off a LayerNormed channel whose
    scale moves with H_t. The optional depthwise causal conv on the normalised
    input is identity-initialisable, so it never breaks the construction.
    """

    def __init__(self, d: int, state_dim: int, two_gated: bool, conv_kernel: int = 0):
        super().__init__()
        self.two_gated = two_gated
        self.norm = nn.LayerNorm(d)
        self.conv = None
        if conv_kernel and conv_kernel > 1:
            self.conv = nn.Conv1d(d, d, conv_kernel, groups=d, padding=conv_kernel - 1)
        self.in_proj = nn.Linear(d, 2 * d)                      # value path + gate
        self.bc_proj = nn.Linear(d, 2 * state_dim)
        self.dt_a_proj = nn.Linear(d, d)
        self.dt_b_proj = nn.Linear(d, d) if two_gated else None
        self.A_logit = nn.Parameter(torch.zeros(d, state_dim))
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, h):
        z = self.norm(h)
        if self.conv is not None:
            L = z.shape[1]
            z = self.conv(z.transpose(1, 2))[..., :L].transpose(1, 2)
        u, gate = self.in_proj(z).chunk(2, dim=-1)
        B, C = self.bc_proj(z).chunk(2, dim=-1)
        dt_a = torch.sigmoid(self.dt_a_proj(z))
        dt_b = torch.sigmoid(self.dt_b_proj(z)) if self.two_gated else dt_a
        y = selective_scan(u, dt_a, dt_b, B, C, torch.sigmoid(self.A_logit))
        return h + self.out_proj(y * F.silu(gate))


# --------------------------------------------------------------------------- #
# Off-the-shelf blocks, kept as a robustness check on the ordering result
# --------------------------------------------------------------------------- #

def rope_tables(L: int, dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(torch.arange(L, device=device).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class GenericTFBlock(nn.Module):
    """Standard RoPE transformer block with a plain causal sliding window."""

    def __init__(self, d: int, heads: int, window: int, mlp_ratio: int = 2):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0, "RoPE needs an even head dim"
        self.heads, self.head_dim, self.window = heads, d // heads, window
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, h):
        n, L, _ = h.shape
        z = self.norm1(h)
        q, k, v = self.qkv(z).chunk(3, dim=-1)
        shape = (n, L, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        cos, sin = rope_tables(L, self.head_dim, h.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        idx = torch.arange(L, device=h.device)
        offset = idx[:, None] - idx[None, :]
        allowed = (offset >= 0) & (offset < self.window)

        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~allowed, float("-inf"))
        out = logits.softmax(-1) @ v
        h = h + self.proj(out.transpose(1, 2).reshape(n, L, -1))
        return h + self.fc2(F.gelu(self.fc1(self.norm2(h))))


class GenericMambaBlock(nn.Module):
    """Standard Mamba block: single shared Delta, depthwise causal conv, D skip."""

    def __init__(self, d: int, state_dim: int, expand: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.d_inner = expand * d
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.dt_rank = max(1, math.ceil(d / 16))

        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, conv_kernel,
                              groups=self.d_inner, padding=conv_kernel - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float())
                                  .repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d, bias=False)

    def forward(self, h):
        L = h.shape[1]
        u, gate = self.in_proj(self.norm(h)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))

        dt, B, C = self.x_proj(u).split([self.dt_rank, self.state_dim, self.state_dim], -1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        decay = torch.exp(dt.unsqueeze(-1) * A)
        inp = (dt * u).unsqueeze(-1) * B.unsqueeze(2)
        y = linear_scan(decay, inp, C) + self.D * u
        return h + self.out_proj(y * F.silu(gate))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

LAYER_KINDS = ("TF", "SSM", "SSM1", "SSM2")


class Hybrid(nn.Module):
    """A stack of TF and SSM layers ending in a binary head, one output per position.

    Layer tokens: 'TF'; 'SSM2' two-gated; 'SSM1' single-gated; and plain 'SSM',
    which means "two-gated if this is the first SSM in the stack, single-gated
    otherwise" -- the construction's rule, where the first SSM has to count and
    the second only has to latch. That rule ignores where the TF layer sits, so
    any permutation of the same multiset of tokens has the same parameter count.
    Arms built from *different* multisets (a pure stack, or SSM1 where another
    arm has SSM2) do not, and have to be matched with --match-params instead.
    """

    def __init__(self, vocab, d, layers, variant, heads, window, state_dim,
                 ssm_conv=0, expand=2, conv_kernel=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        blocks, seen_ssm = [], 0
        for kind in layers:
            if kind == "TF":
                blocks.append(
                    TFBlock(d, heads, window) if variant == "construction"
                    else GenericTFBlock(d, heads, window)
                )
            elif kind in ("SSM", "SSM1", "SSM2"):
                if variant == "construction":
                    two = kind == "SSM2" or (kind == "SSM" and seen_ssm == 0)
                    blocks.append(SSMBlock(d, state_dim, two_gated=two,
                                           conv_kernel=ssm_conv))
                else:
                    blocks.append(GenericMambaBlock(d, state_dim, expand, conv_kernel))
                seen_ssm += 1
            else:
                raise ValueError(f"unknown layer {kind!r}")
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_window(args) -> int:
    if str(args.window).lower() in ("full", "all", "0", "none"):
        return args.length
    return int(args.window)


def build_model(args, d: int, variant: str, layers) -> Hybrid:
    return Hybrid(
        vocab=args.vocab, d=d, layers=layers, variant=variant, heads=args.heads,
        window=resolve_window(args), state_dim=args.state_dim, ssm_conv=args.ssm_conv,
        expand=args.expand, conv_kernel=args.conv_kernel,
    )


# --------------------------------------------------------------------------- #
# The construction: exact weights, no training
# --------------------------------------------------------------------------- #

E_LOGIT = 8.0     # magnitude of the equality logit the TF block writes
GAIN = 30.0       # sigmoid gain used to turn a sign into a hard 0/1 gate
NEG = -1e9        # stand-in for -infinity in the attention bias


class Layout:
    """Channel layout of the residual stream used by the hand-built construction.

    Token codes are *balanced*, psi'(v) followed by -psi'(v), so that every
    LayerNorm in the stack sees mean 0 and a norm that does not depend on which
    token it is looking at. The constant channel is the construction's row of
    ones; the read-out head needs it as a reference against which to compare y_t
    after the final LayerNorm has divided by a scale that moves with H_t.
    """

    def __init__(self, vocab: int, d: int | None = None):
        m = max(1, math.ceil(math.log2(vocab)))
        self.m = m
        self.code = slice(0, 2 * m)
        self.prev = slice(2 * m, 4 * m)
        self.e, self.H, self.y, self.one = 4 * m, 4 * m + 1, 4 * m + 2, 4 * m + 3
        self.width = 4 * m + 4
        if d is not None:
            assert d >= self.width, f"construction needs d >= {self.width}, got d = {d}"


def code_table(vocab: int, m: int) -> torch.Tensor:
    table = torch.zeros(vocab, 2 * m)
    for v in range(vocab):
        psi = torch.tensor([1.0 if (v >> j) & 1 else -1.0 for j in range(m)])
        table[v, :m], table[v, m:] = psi, -psi
    return table


def _ln_stats(vec: torch.Tensor, eps: float = 1e-5):
    """(mean, sigma) of LayerNorm for one residual vector."""
    mu = vec.mean(-1)
    sigma = (vec.var(-1, unbiased=False) + eps).sqrt()
    return mu, sigma


def _silu_inverse_one() -> float:
    """g with g * sigmoid(g) = 1, so that the output gate multiplies by exactly 1."""
    g = 1.3
    for _ in range(100):
        s = 1.0 / (1.0 + math.exp(-g))
        f = g * s - 1.0
        g -= f / (s + g * s * (1 - s))
    return g


def set_construction_weights(model: Hybrid, args, verbose: bool = False) -> dict:
    """Write the paper construction into a TF -> SSM -> SSM model. No training.

    Layer 1 fetches psi'(x_{t-l}) with a hard offset bias and turns the pair into
    a large-margin equality logit. Layer 2 is the two-gated counter
    H_t = e_t (H_{t-1} + 1). Layer 3 is the single-gate latch
    y_t = (1 - z_t) y_{t-1} + z_t with z_t = 1{H_t >= r}. Every threshold is
    computed from the exact LayerNorm statistics of the finitely many residual
    vectors the stack can produce, which is why the model is exact rather than
    approximate.
    """
    l, c, L, vocab = args.block, args.reps, args.length, args.vocab
    r = (c - 1) * l
    d = model.embed.weight.shape[1]
    lay = Layout(vocab, d)
    m, s_dim = lay.m, model.blocks[1].A_logit.shape[1]
    assert args.heads == 1, "the hand-built construction assumes one head"
    assert resolve_window(args) > l, "the attention window must contain offset l"
    tf, ssm1, ssm2 = model.blocks[0], model.blocks[1], model.blocks[2]
    assert isinstance(tf, TFBlock) and isinstance(ssm1, SSMBlock) and ssm1.two_gated
    assert isinstance(ssm2, SSMBlock) and not ssm2.two_gated

    def stream(code=None, prev=None, e=None, H=None, y=None):
        v = torch.zeros(d)
        if code is not None:
            v[lay.code] = code
        if prev is not None:
            v[lay.prev] = prev
        for idx, val in ((lay.e, e), (lay.H, H), (lay.y, y)):
            if val is not None:
                v[idx] = float(val)
        v[lay.one] = 1.0
        return v

    for p in model.parameters():
        with torch.no_grad():
            p.zero_()
    for norm in (tf.norm1, tf.norm2, ssm1.norm, ssm2.norm, model.norm):
        with torch.no_grad():
            norm.weight.fill_(1.0)
    for ssm in (ssm1, ssm2):
        if ssm.conv is not None:
            with torch.no_grad():
                ssm.conv.weight[:, 0, -1] = 1.0                  # identity conv

    codes = code_table(vocab, m)
    with torch.no_grad():
        model.embed.weight[:, lay.code] = codes
        model.embed.weight[:, lay.one] = 1.0

    # ---- layer 1a: attention copies psi'(x_{t-l}) into the `prev` slot ------
    mu1, s1 = _ln_stats(stream(code=codes[0]))
    for v in range(vocab):
        assert torch.allclose(_ln_stats(stream(code=codes[v]))[1], s1, atol=1e-6)
    with torch.no_grad():
        # q = k = 0, so the logits are the offset bias alone; v undoes LayerNorm
        tf.attn.qkv.weight[2 * d:, :].zero_()
        for j in range(2 * m):
            tf.attn.qkv.weight[2 * d + j, j] = s1
            tf.attn.qkv.bias[2 * d + j] = mu1
        tf.attn.rel_bias.fill_(NEG)
        tf.attn.rel_bias[0, l] = 0.0
        for j in range(2 * m):
            tf.attn.proj.weight[2 * m + j, j] = 1.0

    # ---- layer 1b: MLP writes a large-margin equality logit ------------------
    _, s2_pair = _ln_stats(stream(code=codes[0], prev=codes[1]))
    _, s2_solo = _ln_stats(stream(code=codes[0]))
    dist_min = min(2.0 / s2_pair.item(), m / s2_solo.item())      # smallest "unequal"
    theta = 0.5 * dist_min
    lam = E_LOGIT / theta
    with torch.no_grad():
        for j in range(m):
            tf.fc1.weight[2 * j, j] = 1.0
            tf.fc1.weight[2 * j, 2 * m + j] = -1.0
            tf.fc1.weight[2 * j + 1, j] = -1.0
            tf.fc1.weight[2 * j + 1, 2 * m + j] = 1.0
            tf.fc2.weight[lay.e, 2 * j] = -lam
            tf.fc2.weight[lay.e, 2 * j + 1] = -lam
        tf.fc2.bias[lay.e] = lam * theta

    e_pos = E_LOGIT
    e_negs = [lam * (theta - 2.0 * h / s2_pair.item()) for h in range(1, m + 1)]
    e_negs.append(lam * (theta - m / s2_solo.item()))

    # ---- layer 2: two-gated counter H_t = e_t (H_{t-1} + 1) -----------------
    def ln_channel(vec, channel):
        mu, sigma = _ln_stats(vec)
        return ((vec[channel] - mu) / sigma).item()

    pos_e = ln_channel(stream(code=codes[0], prev=codes[0], e=e_pos), lay.e)
    neg_e = [ln_channel(stream(code=codes[0], prev=codes[1], e=v), lay.e) for v in e_negs[:-1]]
    neg_e.append(ln_channel(stream(code=codes[0], e=e_negs[-1]), lay.e))
    assert pos_e > 0 > max(neg_e), "the equality logit must survive LayerNorm"
    gamma = GAIN / min(pos_e, -max(neg_e))
    g_star = _silu_inverse_one()
    with torch.no_grad():
        for ssm, target in ((ssm1, lay.H), (ssm2, lay.y)):
            ssm.in_proj.bias[0] = 1.0                            # u = 1
            ssm.in_proj.bias[d] = g_star                         # silu(gate) = 1
            ssm.bc_proj.bias[0] = 1.0                            # B = e_0
            ssm.bc_proj.bias[s_dim] = 1.0                        # C = e_0
            ssm.A_logit[0, 0] = GAIN                             # A = 1
            ssm.out_proj.weight[target, 0] = 1.0
        ssm1.dt_b_proj.weight[0, lay.e] = gamma                  # Delta_B = e_t
        ssm1.dt_a_proj.weight[0, lay.e] = -gamma                 # Delta_A = 1 - e_t

    # ---- layer 3: single-gate latch on 1{H_t >= r} --------------------------
    ln_H = {}
    for H in range(0, L + 1):
        if H >= 1:                                               # H >= 1 implies e_t = 1
            ln_H[H] = [ln_channel(stream(code=codes[0], prev=codes[0], e=e_pos, H=H), lay.H)]
        else:
            ln_H[0] = [ln_channel(stream(code=codes[0], prev=codes[1], e=v, H=0), lay.H)
                       for v in e_negs[:-1]]
            ln_H[0].append(ln_channel(stream(code=codes[0], e=e_negs[-1], H=0), lay.H))
    lo = max(v for H in range(0, r) for v in ln_H[H])
    hi = min(v for H in range(r, L + 1) for v in ln_H[H])
    assert hi > lo, f"H_t is not separable at r after LayerNorm ({lo:.4f} vs {hi:.4f})"
    mid, gain2 = 0.5 * (lo + hi), GAIN / (0.5 * (hi - lo))
    with torch.no_grad():
        ssm2.dt_a_proj.weight[0, lay.H] = gain2
        ssm2.dt_a_proj.bias[0] = -gain2 * mid

    # ---- read-out: compare y_t against the constant channel -----------------
    sig_y0 = []
    for H in range(0, r):
        es = [e_pos] if H >= 1 else e_negs
        for v in es:
            sig_y0.append(_ln_stats(stream(code=codes[0], prev=codes[0], e=v, H=H, y=0))[1].item())
    gain3 = 20.0 * max(sig_y0)
    with torch.no_grad():
        model.head.weight[1, lay.y] = 0.5 * gain3
        model.head.weight[1, lay.one] = -0.5 * gain3
        model.head.weight[0] = -model.head.weight[1]
        model.head.bias[1] = 0.25 * gain3 / max(sig_y0)
        model.head.bias[0] = -model.head.bias[1]

    consts = {"layout_width": lay.width, "sigma_attn": s1.item(), "lambda": lam,
              "theta": theta, "gamma": gamma, "H_margin": hi - lo, "gain_H": gain2}
    if verbose:
        print("  construction constants: " + "  ".join(f"{k}={v:.4g}" for k, v in consts.items()))
    return consts


def construction_point(args, device, verbose: bool = False, n_examples: int = 4096):
    """Build the construction by hand and measure it. Returns (params, final, dense)."""
    d = args.verify_width or Layout(args.vocab).width
    model = build_model(args, d, "construction", ["TF", "SSM", "SSM"]).to(device)
    set_construction_weights(model, args, verbose=verbose)
    model.eval()
    x, y = fixed_eval_set(args, n_examples)
    final, dense = evaluate(model, x.to(device), y.to(device), args.batch)
    return count_params(model), final, dense


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
            for seed in range(args.seeds):
                res = train_one(args, d, variant, layers, seed, lr, device)
                runs.append(res)
                jump = "" if res["solve_step"] is None else f" solved@{res['solve_step']}"
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
# Checks
# --------------------------------------------------------------------------- #

def self_test(args, device):
    l, c, r = args.block, args.reps, (args.reps - 1) * args.block
    rng = np.random.default_rng(0)
    print(f"task: L={args.length} |M|={args.vocab} l={l} c={c} r=(c-1)l={r} "
          f"hard_neg_frac={args.hard_neg_frac} window={resolve_window(args)}")

    # 1. labels agree with the definition, checked by brute force
    x, y = sample_batch(rng, 3000, args.length, args.vocab, l, c,
                        hard_neg_frac=args.hard_neg_frac)
    bad = pos = 0
    for i in range(len(x)):
        end = contains_exact_copy(x[i], l, c)
        want = np.zeros(args.length, dtype=np.int64)
        if end >= 0:
            want[end:] = 1
            pos += 1
        bad += int(not np.array_equal(want, y[i]))
    print(f"[1] labels vs brute-force definition: {bad} mismatches in {len(x)} seqs "
          f"(positive rate {pos / len(x):.3f}, all-zero dense acc {1 - y.mean():.3f})")
    assert bad == 0

    # 2. hard negatives keep the token multiset of the positive they came from,
    #    so their longest run of equalities is nonzero but short of r
    hard_rng = np.random.default_rng(1)
    runs_seen, multiset_ok = [], 0
    for _ in range(300):
        a = int(hard_rng.integers(0, args.length - c * l + 1))
        block = _sample_block(hard_rng, args.vocab, l)
        row = hard_rng.integers(0, args.vocab, size=args.length)
        for k in range(c):
            row[a + k * l:a + (k + 1) * l] = block
        before = np.bincount(row, minlength=args.vocab)
        _break_copy(hard_rng, row, a, l, c, args.vocab)
        multiset_ok += int(np.array_equal(before, np.bincount(row, minlength=args.vocab)))
        e = (row[l:] == row[:-l]).astype(int)
        longest = best = 0
        for bit in e:
            best = best + 1 if bit else 0
            longest = max(longest, best)
        runs_seen.append(longest)
        assert contains_exact_copy(row, l, c) == -1
    print(f"[2] hard negatives: multiset preserved {multiset_ok}/300, all labelled 0, "
          f"longest equality run {min(runs_seen)}-{max(runs_seen)} (< r={r})")
    assert multiset_ok == 300 and max(runs_seen) < r

    # 3. the frequency shortcut is closed by hard negatives
    xe, ye = sample_batch(np.random.default_rng(2), 1500, args.length, args.vocab, l, c,
                          hard_neg_frac=0.0)
    xh, yh = sample_batch(np.random.default_rng(2), 1500, args.length, args.vocab, l, c,
                          hard_neg_frac=1.0)
    acc_easy = frequency_shortcut_accuracy(xe, ye[:, -1], l, c)
    acc_hard = frequency_shortcut_accuracy(xh, yh[:, -1], l, c)
    print(f"[3] token-frequency shortcut: {acc_easy:.3f} accuracy with easy negatives, "
          f"{acc_hard:.3f} with hard negatives (chance 0.5)")
    assert acc_hard < 0.6 <= acc_easy

    # 4. the fast scan computes exactly the same recurrence as the reference loop
    global SCAN_MODE
    saved_mode = SCAN_MODE
    g = torch.Generator().manual_seed(7)
    rand_decay = torch.rand(8, args.length, 6, 4, generator=g)
    rand_inp = torch.randn(8, args.length, 6, 4, generator=g)
    rand_C = torch.randn(8, args.length, 4, generator=g)
    SCAN_MODE = "sequential"
    ref = linear_scan(rand_decay, rand_inp, rand_C)
    SCAN_MODE = "parallel"
    fast = linear_scan(rand_decay, rand_inp, rand_C)
    gate = (torch.rand(8, args.length, 6, 4, generator=g) > 0.4).float()
    ones_C = torch.ones(8, args.length, 4)
    exact_fast = linear_scan(gate, gate, ones_C)
    SCAN_MODE = "sequential"
    exact_ref = linear_scan(gate, gate, ones_C)
    SCAN_MODE = saved_mode
    print(f"[4] parallel scan vs sequential scan: max abs err "
          f"{(ref - fast).abs().max():.2e} on random inputs, "
          f"{(exact_ref - exact_fast).abs().max():.2e} in the 0/1 counter regime")
    assert torch.allclose(ref, fast, atol=1e-4)
    assert torch.equal(exact_ref, exact_fast)

    # 5. the two-gated recurrence realises the counter H_t = e_t (H_{t-1} + 1)
    seq = x[0]
    e = torch.zeros(1, args.length, 1)
    for t in range(l, args.length):
        e[0, t, 0] = float(seq[t] == seq[t - l])
    ones = torch.ones(1, args.length, 1)
    H = selective_scan(ones, 1 - e, e, ones, ones, torch.ones(1, 1))
    want_H, run = [], 0.0
    for t in range(args.length):
        run = (run + 1) * e[0, t, 0].item()
        want_H.append(run)
    print(f"[5] two-gated scan vs counter H_t: max abs err "
          f"{(H[0, :, 0] - torch.tensor(want_H)).abs().max():.2e}")
    assert torch.allclose(H[0, :, 0], torch.tensor(want_H), atol=1e-5)

    # 6. the single-gate recurrence realises the latch y_t = (1-z) y_{t-1} + z
    z = (H >= r - 0.5).float()
    y_hat = selective_scan(ones, z, z, ones, ones, torch.ones(1, 1))
    print(f"[6] single-gate scan vs latch y_t: max abs err "
          f"{(y_hat[0, :, 0] - torch.as_tensor(y[0]).float()).abs().max():.2e}")
    assert torch.allclose(y_hat[0, :, 0], torch.as_tensor(y[0]).float(), atol=1e-5)

    # 7. what the second gate actually buys. With the construction's constant
    #    value path a shared gate is a convex combination towards 1, so it cannot
    #    reach r. It can if the value path is allowed to scale like 1/eps, at the
    #    cost of large weights and an O(eps) error per step -- so the second gate
    #    buys exactness at O(1) weights, it is not an expressivity requirement.
    worst = 0.0
    for _ in range(200):
        dt = torch.rand(1, args.length, 1)
        worst = max(worst, selective_scan(ones, dt, dt, ones, ones, torch.ones(1, 1)).abs().max().item())
    errs = []
    for eps in (1e-2, 1e-4):
        dt = torch.where(e > 0.5, torch.full_like(e, eps), torch.ones_like(e))
        H_shared = selective_scan(e / dt, dt, dt, ones, ones, torch.ones(1, 1))
        errs.append((eps, (H_shared[0, :, 0] - torch.tensor(want_H)).abs().max().item()))
    print(f"[7] shared gate, constant value path: max state {worst:.3f} <= 1 (need r={r}); "
          "value path free: " + ", ".join(f"eps={s:g} -> err {v:.1e}" for s, v in errs))
    assert worst <= 1.0 + 1e-5 and errs[-1][1] < 0.05

    # 8. the fairness invariant: permuting the layers cannot change the parameter
    #    count, so the two arms sit at exactly the same x-coordinates
    orders = [["TF", "SSM", "SSM"], ["SSM", "SSM", "TF"], ["SSM", "TF", "SSM"]]
    for variant in ("construction", "generic"):
        counts = {}
        for d in (16, 24, 32):
            got = [count_params(build_model(args, d, variant, o)) for o in orders]
            assert len(set(got)) == 1, f"{variant} d={d} params differ across orders: {got}"
            counts[d] = got[0]
        print(f"[8] {variant:13s} params identical across layer orders: {counts}")

    # 8b. explicit gating tokens: SSM1/SSM2 name what plain SSM decides by position,
    #     and arms built from different multisets are matched by budget instead
    for d in (16, 24, 32):
        legacy = build_model(args, d, "construction", ["TF", "SSM", "SSM"])
        spelled = build_model(args, d, "construction", ["TF", "SSM2", "SSM1"])
        assert count_params(legacy) == count_params(spelled)
        assert [b.two_gated for b in legacy.blocks if isinstance(b, SSMBlock)] == [True, False]
    gates = {a: [b.two_gated for b in build_model(args, 32, "construction", a.split(","))
                 .blocks if isinstance(b, SSMBlock)]
             for a in ("SSM1,SSM1,SSM1", "TF,SSM1,SSM1", "SSM1,SSM1,TF",
                       "SSM1,TF,SSM1", "TF,TF,TF")}
    assert gates["SSM1,SSM1,SSM1"] == [False, False, False] and gates["TF,TF,TF"] == []
    assert (gates["TF,SSM1,SSM1"] == [False, False] == gates["SSM1,SSM1,TF"]
            == gates["SSM1,TF,SSM1"])
    xb8 = torch.as_tensor(x[:8], device=device)
    for a in gates:
        assert build_model(args, 32, "construction", a.split(",")).to(device)(xb8).shape \
            == (8, args.length, 2)
    targets = [count_params(build_model(args, d, "construction", ["TF", "SSM", "SSM"]))
               for d in (16, 24, 32, 48)]
    worst, got = 0.0, {}
    for a in ("TF,SSM,SSM", "SSM,SSM,TF", "SSM,TF,SSM", "TF,SSM1,SSM1",
              "SSM1,SSM1,TF", "SSM1,TF,SSM1", "SSM1,SSM1,SSM1", "TF,TF,TF"):
        ws = widths_for_targets(args, "construction", a.split(","), targets)
        ps = [count_params(build_model(args, w, "construction", a.split(","))) for w in ws]
        assert len(ws) == len(targets), f"{a}: {len(ws)} widths for {len(targets)} budgets"
        worst = max(worst, max(abs(p - t) / t for p, t in zip(ps, targets)))
        got[a] = ws
    print(f"[8b] gating tokens ok; --match-params hits every budget within "
          f"{100 * worst:.1f}% for all {len(got)} arms: {got}")
    assert worst < 0.06

    # 9. the evaluation set is the same object for every run
    a = fixed_eval_set(args, 256)
    b = fixed_eval_set(args, 256)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    print(f"[9] held-out eval set is fixed (seed {args.eval_seed}), "
          f"P(y_L=1)={a[1][:, -1].float().mean():.3f}")

    # 10. both orders run forward in both families
    xb = torch.as_tensor(x[:8], device=device)
    for variant in ("construction", "generic"):
        for layers in (["TF", "SSM", "SSM"], ["SSM", "SSM", "TF"]):
            model = build_model(args, 32, variant, layers).to(device)
            assert model(xb).shape == (8, args.length, 2)
    print("[10] forward pass ok for both orders in both families")

    # 11. the hand-built construction is exact
    params, final, dense = construction_point(args, device, verbose=args.verbose, n_examples=1024)
    print(f"[11] hand-built construction: params {params}, "
          f"final_acc {final:.4f}, dense_acc {dense:.4f}")
    assert final == 1.0 and dense == 1.0

    # 12. the reach of the control arm, measured rather than assumed. A depth-3
    #     windowed attention stack cannot see further back than 3(W-1)+1, so at
    #     position L it is blind to a copy that ended earlier than that -- this
    #     is the whole reason a bounded window is expected to break a pure TF
    #     while leaving TF->SSM->SSM (which only ever reads offset l) intact.
    saved_window = args.window
    xe, ye = fixed_eval_set(args, 1024)
    # the copy ends at the first position where the latched label turns on
    end, pos = ye.argmax(dim=1), ye[:, -1] == 1
    for w in CONTROL_WINDOWS(args):
        args.window = w
        reach = 3 * (w - 1) + 1

        # (a) blindness beyond the reach, measured on a real TF stack
        torch.manual_seed(0)
        tf = build_model(args, 32, "generic", ["TF", "TF", "TF"]).to(device).eval()
        base = xe[:64].to(device)
        moved = []
        with torch.no_grad():
            ref = tf(base)[:, -1]
            for back in (reach - 1, reach, reach + 8):
                pert = base.clone()
                t = args.length - 1 - back
                pert[:, t] = (pert[:, t] + 1) % args.vocab
                moved.append((tf(pert)[:, -1] - ref).abs().max().item())
        inside, at, beyond = moved
        assert inside > 0, "pure TF should depend on inputs inside its reach"
        assert at == 0.0 and beyond == 0.0, "pure TF must be blind beyond its reach"

        # (b) a model blind beyond the reach can do no better than answering 0
        #     on every copy that ended earlier than that, which caps y_L
        #     accuracy at 1 - P(copy ended out of reach)
        far = pos & ((args.length - 1 - end) > reach)
        cap = 1.0 - far.float().mean().item()

        # (c) the hand-built hybrid answers those same sequences exactly, because
        #     its TF only ever reads offset l and the SSM carries the rest
        d12 = args.verify_width or Layout(args.vocab).width
        hyb = build_model(args, d12, "construction", ["TF", "SSM", "SSM"]).to(device)
        set_construction_weights(hyb, args)
        hyb.eval()
        with torch.no_grad():
            pred = hyb(xe.to(device))[:, -1].argmax(-1).cpu()
        acc_far = (pred[far] == 1).float().mean().item()
        print(f"[12] W={w:<3} depth-3 TF reach 3(W-1)+1={reach:<3} "
              f"| y_L unchanged by a token {reach} back ({at:.0e}) but moves at "
              f"{reach - 1} back ({inside:.1e}) | {far.sum().item()}/{pos.sum().item()} "
              f"positives ended out of reach, so a windowed TF caps at {cap:.3f}; "
              f"hand-built TF->SSM->SSM gets {acc_far:.3f} of them")
        assert acc_far == 1.0, "the construction must not depend on the window"
    args.window = saved_window
    print("all checks passed")


def verify(args, device):
    print(f"task: L={args.length} |M|={args.vocab} l={args.block} c={args.reps} "
          f"r={(args.reps - 1) * args.block}")
    saved = args.window
    for window in ("full",) + CONTROL_WINDOWS(args):
        args.window = window
        params, final, dense = construction_point(args, device, verbose=True,
                                                  n_examples=args.eval_examples)
        w = resolve_window(args)
        print(f"  window={w:<4} params={params:<7} y_L accuracy {final:.4f}  "
              f"all-position accuracy {dense:.4f}  (no training)")
        assert final == 1.0 and dense == 1.0
    args.window = saved
    print(f"construction is exact on {args.eval_examples} held-out sequences "
          f"({args.eval_examples // 2} of them hard negatives)")
    print("the TF->SSM->SSM construction only ever reads offset l, so it survives "
          f"every window down to the bound W=l+1={args.block + 1} of the theory")


def params_table(args, device):
    configs = parse_configs(args.configs)
    print(f"{'d':>5}" + "".join(f"{v + ' ' + ','.join(l):>28}" for v, l in configs))
    for d in (args.widths or DEFAULT_WIDTHS):
        row = "".join(f"{count_params(build_model(args, d, v, l)):>28}" for v, l in configs)
        print(f"{d:>5}{row}")
    lay = Layout(args.vocab)
    d_min = args.verify_width or lay.width
    print(f"\nhand-built construction needs d >= {lay.width} "
          f"(4*ceil(log2|M|)+4); at d={d_min} it has "
          f"{count_params(build_model(args, d_min, 'construction', ['TF', 'SSM', 'SSM']))} params")
    for variant, layers in configs:
        widths = widths_for(args, variant, layers)
        got = [count_params(build_model(args, d, variant, layers)) for d in widths]
        print(f"  {variant} {','.join(layers)}: d={widths} params={got}")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #

def parse_configs(specs):
    configs = []
    for spec in specs:
        variant, _, layers = spec.partition(":")
        assert variant in ("construction", "generic"), f"bad variant in {spec!r}"
        layers = layers.split(",")
        bad = [k for k in layers if k not in LAYER_KINDS]
        assert not bad, f"unknown layer(s) {bad} in {spec!r}; expected {LAYER_KINDS}"
        configs.append((variant, layers))
    return configs


def report_fairness(args, configs):
    print(f"window={resolve_window(args)}  ssm_conv={args.ssm_conv}  "
          f"state_dim={args.state_dim}  heads={args.heads}  steps={args.steps}  "
          f"batch={args.batch}  lrs={args.lrs}  seeds={args.seeds}")
    grid = {f"{variant}:{','.join(layers)}":
            [(d, count_params(build_model(args, d, variant, layers)))
             for d in widths_for(args, variant, layers)]
            for variant, layers in configs}

    if args.match_params:
        # arms use different blocks, so they meet at budgets rather than at widths
        print(f"  matched by parameter budget (targets {args.targets}):")
        worst = 0.0
        for k, target in enumerate(args.targets):
            cells = []
            for name, row in grid.items():
                if k >= len(row):
                    continue
                d, p = row[k]
                worst = max(worst, abs(p - target) / target)
                cells.append(f"{name.split(':')[1]} d={d}({p}, {100 * (p - target) / target:+.1f}%)")
            print(f"    target {target:>7}: " + "  ".join(cells))
        print(f"  worst deviation from a budget: {100 * worst:.1f}%")
        return

    per_d = {}
    for name, row in grid.items():
        for d, p in row:
            per_d.setdefault(d, {})[name] = p
    for d, row in sorted(per_d.items()):
        same = "matched" if len(set(row.values())) == 1 else "MISMATCH"
        print(f"  d={d:>3} params={row} -> {same}")
    if any(len(set(row.values())) != 1 for row in per_d.values()):
        print("  note: arms differ in parameter count at the same width; use "
              "--match-params so each arm gets its own widths")


def sweep(args, device):
    os.makedirs(args.out_dir, exist_ok=True)
    layers = args.layers.split(",")
    report_fairness(args, [(args.variant, layers)])
    runs = run_config(args, args.variant, layers, device)
    save_and_plot({f"{args.variant}: {'->'.join(layers)}": runs}, args, device)


def compare(args, device):
    os.makedirs(args.out_dir, exist_ok=True)
    configs = parse_configs(args.configs)
    report_fairness(args, configs)
    curves = {}
    for variant, layers in configs:
        name = f"{variant}: {'->'.join(layers)}"
        print(f"=== {name} ===", flush=True)

        def checkpoint(runs, name=name):
            """A long sweep should survive being killed: keep runs.json current."""
            curves[name] = runs
            with open(os.path.join(args.out_dir, "runs.json"), "w") as f:
                json.dump({"args": vars(args), "curves": curves,
                           "summary": {k: best_lr_curve(v, args.partial_threshold)
                                       for k, v in curves.items()}}, f)

        curves[name] = run_config(args, variant, layers, device, on_run=checkpoint)
        save_and_plot(curves, args, device)
    return curves


def load_runs(paths, solve_threshold):
    """Merge the runs.json of several output dirs, keyed by arm name.

    Runs are independent (each is seeded from its own seed and evaluated on the
    same fixed held-out set), so splitting a sweep across processes and merging
    here gives exactly the curves a single --compare would have produced.
    """
    curves: dict = {}
    for path in paths:
        with open(os.path.join(path, "runs.json")) as f:
            blob = json.load(f)
        window = blob["args"].get("window", "full")
        window = blob["args"]["length"] if str(window).lower() in ("full", "all", "0", "none") \
            else int(window)
        for name, runs in blob["curves"].items():
            for r in runs:
                # results written before the solve-rate metric existed carry only
                # an accuracy, so re-derive the verdict at the current threshold
                r.setdefault("history", [])
                r.setdefault("solve_step", None)
                r.setdefault("window", window)
                r["solved"] = bool(r["final_acc"] >= solve_threshold)
            curves.setdefault(name, []).extend(runs)
    return curves


def replot(args, device):
    """Merge several output dirs into the full three-panel figure."""
    os.makedirs(args.out_dir, exist_ok=True)
    curves = load_runs(args.replot, args.solve_threshold)
    for name, runs in curves.items():
        print(f"  {name}: {len(runs)} runs, widths "
              f"{sorted({r['d'] for r in runs})}, lrs {sorted({r['lr'] for r in runs})}")
    save_and_plot(curves, args, device)


# --------------------------------------------------------------------------- #
# Accuracy vs parameters on its own, one figure per gating family
# --------------------------------------------------------------------------- #

ARM_COLORS = {"TF-TF-TF": "#1BAF7A", "TF-SSM-SSM": "#2A78D6",
              "SSM-SSM-TF": "#C44E3F", "SSM-TF-SSM": "#7B4FA8",
              "SSM-SSM-SSM": "#EDA100"}

# Alternating dashes matter here: the two TF-first arms both sit at 1.000 and
# the two arms that put the TF later land in the same band, so a solid line
# would simply hide whichever is drawn first.
ARM_STYLES = {"TF-TF-TF": ("o", "-"), "TF-SSM-SSM": ("s", "--"),
              "SSM-SSM-TF": ("^", "-"), "SSM-TF-SSM": ("v", "--"),
              "SSM-SSM-SSM": ("D", "--")}


def arm_role(layers):
    """The shape of the stack with the gating spelling erased: 'TF-SSM-SSM' etc.

    This is what the colour keys index, because a hybrid keeps its identity in
    the figure whether its first SSM is one- or two-gated.
    """
    return "-".join("TF" if k == "TF" else "SSM" for k in layers)


def arm_gating(layers):
    """'two' if any SSM block in the stack is two-gated, else 'single'.

    None means the stack holds no SSM at all, so it belongs to both families.
    """
    seen_ssm, two = 0, False
    for k in layers:
        if k in ("SSM", "SSM1", "SSM2"):
            two |= k == "SSM2" or (k == "SSM" and seen_ssm == 0)
            seen_ssm += 1
    return None if seen_ssm == 0 else ("two" if two else "single")


def windowed_tf_ceiling(args, window, depth=3):
    """Best y_L accuracy available to a depth-`depth` windowed attention stack.

    Its output at position L cannot depend on anything earlier than
    depth*(W-1)+1 positions back (measured in check [12]), so a copy that ended
    before that is indistinguishable from a negative and answering 0 there is
    the best it can do. Returns None when the window reaches the whole sequence.
    """
    if window >= args.length:
        return None
    _, y = fixed_eval_set(args, 4096)
    end, pos = y.argmax(dim=1), y[:, -1] == 1
    reach = depth * (window - 1) + 1
    out_of_reach = pos & ((args.length - 1 - end) > reach)
    return 1.0 - out_of_reach.float().mean().item()


def acc_figures(args, device):
    """Redraw accuracy vs parameters alone, per attention window and gating family.

    An arm with no TF block never reads the window, so its curve is the same
    reference in every window's figure and its runs are reused rather than
    demanded again for each setting.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    curves = load_runs(args.acc_figures, args.solve_threshold)

    available, windows = {}, set()
    for name, runs in curves.items():
        layers = name.split(": ")[-1].split("->")
        key = (arm_role(layers), arm_gating(layers))
        if "TF" not in layers:
            available[key + (None,)] = (name, runs)
            continue
        for w in sorted({r["window"] for r in runs}):
            windows.add(w)
            available[key + (w,)] = (name, [r for r in runs if r["window"] == w])
    print("arms found:")
    for (role, gating, w), (name, runs) in sorted(available.items(), key=lambda kv: str(kv[0])):
        print(f"  {name:<34} role={role:<14} gating={str(gating):<7} "
              f"window={'any' if w is None else w:<5} runs={len(runs)}")
    if not windows:
        print("no arm with a TF block found")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = (f"Exact copy ($\\ell={args.block}$, $c={args.reps}$, $L={args.length}$, "
            f"$|\\mathcal{{M}}|={args.vocab}$, hard negatives)")
    written = []
    for w in sorted(windows):
        w_text = ("unwindowed attention ($W=L=" + str(w) + "$)" if w >= args.length
                  else f"windowed attention ($W={w}$, $h=\\min(\\ell, L-W)="
                       f"{min(args.block, max(0, args.length - w))}$)")
        # the single-gated figure carries no gating line: which blocks are gated
        # how belongs in the caption, and both arms there use the same kind
        for family, family_text in (("single", ""),
                                    ("two", "two-gated first SSM block")):
            # Without an arm that really carries this gating at this window
            # there is nothing for the figure to be about, and the reused
            # window-independent arms alone would make it look populated.
            if not any((r, family, w) in available for r in ARM_COLORS):
                continue
            other = "two" if family == "single" else "single"
            fig, ax = plt.subplots(figsize=(6.8, 5.3))
            seeds, drawn = set(), 0
            for role, color in ARM_COLORS.items():
                # A stack with no SSM has no gating to key on. A stack with no
                # TF never reads the window, so its one finished set of runs is
                # keyed ww=None and is the right reference in either family --
                # that is the only place a gating may be substituted. An arm
                # keyed to a window is genuinely absent if it is missing, and
                # borrowing the other gating there would mislabel the figure.
                pick = next((available[(role, g, ww)] for g, ww in (
                    (family, w), (None, w), (family, None), (None, None),
                    (other, None)) if (role, g, ww) in available), None)
                if pick is None:
                    print(f"  [W={w} {family}] no data for {role}, skipped")
                    continue
                name, runs = pick
                points = best_lr_curve(runs, args.partial_threshold)
                gating = arm_gating(name.split(": ")[-1].split("->"))
                params = np.array([p["params"] for p in points], dtype=float)
                mean = np.array([p["mean"] for p in points])
                sd = np.array([p["std"] for p in points])
                order = np.argsort(params)
                label = role if gating in (family, None) else f"{role} ({gating}-gated)"
                marker, dashes = ARM_STYLES[role]
                ax.plot(params[order], mean[order], marker=marker, ls=dashes, color=color,
                        lw=1.9, ms=5.5, label=label, zorder=3)
                ax.fill_between(params[order], (mean - sd)[order], (mean + sd)[order],
                                color=color, alpha=0.18, lw=0, zorder=2)
                seeds.update(p["n"] for p in points)
                drawn += 1
            if not drawn:
                plt.close(fig)
                continue

            ax.axhline(0.5, ls="--", c="grey", lw=1, zorder=1)
            ax.axhline(args.solve_threshold, ls=":", c="black", lw=1, zorder=1)
            cap = windowed_tf_ceiling(args, w)
            if cap is not None:
                ax.axhline(cap, ls="-.", lw=1.2, color=ARM_COLORS["TF-TF-TF"],
                           alpha=0.85, zorder=1)
                ax.annotate(f"reach limit of a depth-3 windowed TF ({cap:.2f})",
                            xy=(0.015, cap), xycoords=("axes fraction", "data"),
                            va="bottom", ha="left", fontsize=8,
                            color=ARM_COLORS["TF-TF-TF"])
            n_text = f"{seeds.pop()} seeds" if len(seeds) == 1 else "seeds"
            ax.set_xlabel("number of parameters (matched budgets)")
            ax.set_ylabel(f"accuracy of $y_L$ (mean $\\pm$ s.d., {n_text})")
            head = ", ".join(t for t in (family_text, w_text) if t)
            ax.set_title(f"{head}\n{task}", fontsize=10)
            ax.set_ylim(0.45, 1.03)
            ax.grid(alpha=0.3)
            # under the axes: the arms that plateau leave no gap wide enough for
            # a legend box, and covering them would hide what is being compared
            ax.legend(fontsize=9, ncol=2, frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, -0.13))
            fig.tight_layout()
            tag = "full" if w >= args.length else f"{w}"
            path = os.path.join(args.out_dir,
                                f"acc_vs_params_w{tag}_{family}_gated.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path)

    for path in written:
        print("wrote", path)


def save_and_plot(curves, args, device):
    ceiling = None
    if args.plot_construction:
        try:
            params, final, _ = construction_point(args, device, n_examples=1024)
            ceiling = {"params": params, "acc": final}
        except AssertionError as exc:
            print("construction point unavailable:", exc)

    summary = {name: best_lr_curve(runs, args.partial_threshold)
               for name, runs in curves.items()}
    with open(os.path.join(args.out_dir, "runs.json"), "w") as f:
        json.dump({"args": vars(args), "curves": curves, "summary": summary,
                   "construction": ceiling}, f, indent=2)

    print_summary(summary, args)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
              "#9467bd", "#8c564b", "#17becf", "#e377c2"]
    title = (f"Exact copy ($\\ell={args.block}$, $c={args.reps}$, $L={args.length}$, "
             f"$|\\mathcal{{M}}|={args.vocab}$, hard negatives)")

    # every arm shares the same variant in practice, so drop it from the legend
    names = list(summary)
    strip = all(": " in n for n in names) and len({n.split(": ", 1)[0] for n in names}) == 1
    label_of = {n: (n.split(": ", 1)[1] if strip else n) for n in names}

    # -- figure 1: accuracy, solve rate, and every individual run ------------
    n_labels = len(summary) + 3
    legend_cols = 3 if n_labels > 6 else n_labels
    legend_rows = math.ceil(n_labels / legend_cols)
    fig, (ax_acc, ax_rate, ax_runs) = plt.subplots(
        1, 3, figsize=(17.5, 4.6 + 0.26 * (legend_rows - 1)))
    n_arms = max(1, len(summary))
    for i, (name, points) in enumerate(summary.items()):
        color = colors[i % len(colors)]
        # Nudge the arms apart on x, since at a solve rate of 0 their intervals
        # would coincide exactly. The offset is in points, not in parameters: the
        # arms sit at the same x by design, and a data-space offset would look
        # like a real parameter difference once the axis is zoomed to one width.
        shift = ScaledTranslation((i - (n_arms - 1) / 2) * 5 / 72, 0, fig.dpi_scale_trans)
        params = np.array([p["params"] for p in points], dtype=float)

        mean = np.array([p["mean"] for p in points])
        sd = np.array([p["std"] for p in points])
        ax_acc.plot(params, mean, "o-", color=color, lw=1.8, ms=5, label=label_of[name])
        ax_acc.fill_between(params, mean - sd, mean + sd, color=color, alpha=0.18, lw=0)

        rate = np.array([p["solve_rate"] for p in points])
        err = np.stack([rate - np.array([p["ci_lo"] for p in points]),
                        np.array([p["ci_hi"] for p in points]) - rate])
        ax_rate.errorbar(params, rate, yerr=err, fmt="o-", color=color, capsize=3,
                         lw=1.6, alpha=0.9, transform=ax_rate.transData + shift,
                         label=f"{label_of[name]}  (lr {points[0]['lr']:g}"
                               + ("" if len({p['lr'] for p in points}) == 1 else "+")
                               + ")")
        # only label the arms that ever solve: a rate of 0 is already unambiguous
        # on the axis, and with six arms those labels would all collide at y = 0
        dx = (i - (n_arms - 1) / 2) * 5
        for p in points:
            if p["solved"]:
                ax_rate.annotate(f"{p['solved']}/{p['n']}", (p["params"], p["solve_rate"]),
                                 textcoords="offset points", xytext=(dx, 8),
                                 ha="center", fontsize=8, color=color)

        runs = curves[name]
        ax_runs.scatter([r["params"] for r in runs], [r["final_acc"] for r in runs],
                        transform=ax_runs.transData + ScaledTranslation(
                            (i - (n_arms - 1) / 2) * 5 / 72, 0, fig.dpi_scale_trans),
                        zorder=3, s=16, color=color, alpha=0.55, edgecolors="none",
                        label=label_of[name])

    # the arm artists carry a point-offset transform, which stops matplotlib from
    # inferring data limits from them, so set the x range from the values directly
    all_params = sorted({p["params"] for pts in summary.values() for p in pts})
    span = (all_params[-1] - all_params[0]) or all_params[-1]
    xlim = (all_params[0] - 0.08 * span, all_params[-1] + 0.08 * span)
    ax_rate.set_xlim(*xlim)
    ax_acc.set_xlim(*xlim)
    ax_runs.set_xlim(min(xlim[0], ceiling["params"] - 0.08 * span) if ceiling else xlim[0],
                     max(xlim[1], ceiling["params"] + 0.08 * span) if ceiling else xlim[1])

    ax_acc.axhline(0.5, ls="--", c="grey", lw=1)
    ax_acc.axhline(args.solve_threshold, ls=":", c="black", lw=1)
    ax_acc.set_xlabel("number of parameters (matched budgets)")
    ax_acc.set_ylabel(r"accuracy of $y_L$")
    seen_n = {p["n"] for pts in summary.values() for p in pts}
    n_text = f"{seen_n.pop()} seeds" if len(seen_n) == 1 else "seeds"
    ax_acc.set_title(f"Accuracy vs parameters\n(mean $\\pm$ s.d. over {n_text}, "
                     "at the selected lr)", fontsize=10)
    ax_acc.set_ylim(0.45, 1.03)
    ax_acc.grid(alpha=0.3)

    ax_rate.axhline(0.0, ls=":", c="grey", lw=1)
    ax_rate.set_xlabel("number of parameters (matched budgets)")
    ax_rate.set_ylabel(f"fraction of runs reaching $y_L$ acc $\\geq$ {args.solve_threshold:g}")
    ax_rate.set_title("Solve rate at the best learning rate")
    ax_rate.set_ylim(-0.12, 1.12)
    ax_rate.grid(alpha=0.3)

    if ceiling:
        ax_runs.plot([ceiling["params"]], [ceiling["acc"]], "*", ms=16, color="black",
                     zorder=2, label=f"construction by hand ({ceiling['params']} params)")
    ax_runs.axhline(0.5, ls="--", c="grey", lw=1, label="chance")
    ax_runs.axhline(args.solve_threshold, ls=":", c="black", lw=1,
                    label=f"solved ($\\geq$ {args.solve_threshold:g})")
    ax_runs.set_xlabel("number of parameters")
    ax_runs.set_ylabel(r"accuracy of $y_L$")
    ax_runs.set_title("Every run (all learning rates, all seeds)")
    ax_runs.set_ylim(0.45, 1.06)
    ax_runs.grid(alpha=0.3)

    # one legend under both panels, so nothing can hide a successful run
    handles = ax_rate.get_legend_handles_labels()[0] + ax_runs.get_legend_handles_labels()[0][
        len(summary):]
    labels = ax_rate.get_legend_handles_labels()[1] + ax_runs.get_legend_handles_labels()[1][
        len(summary):]
    fig.legend(handles, labels, loc="lower center", ncol=legend_cols, fontsize=8,
               frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.055 * legend_rows + 0.02, 1, 0.94))
    path = os.path.join(args.out_dir, "params_vs_acc.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # -- figure 2: where the jump happens ------------------------------------
    if not any(r.get("history") for runs in curves.values() for r in runs):
        print("wrote", path, "and runs.json (no learning curves recorded)")
        return
    # Panels are parameter budgets, not widths: once arms stop being permutations
    # of one another they hit a budget at different widths, so grouping by d would
    # scatter the arms across panels that cannot be compared.
    n_panels = max(len(p) for p in summary.values())
    ncol = min(4, n_panels)
    nrow = math.ceil(n_panels / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.3 * nrow),
                             squeeze=False, sharey=True)
    for k, ax in enumerate(axes.flat[:n_panels]):
        budget = []
        for i, (name, points) in enumerate(summary.items()):
            if k >= len(points):
                continue
            p = points[k]
            budget.append(p["params"])
            color = colors[i % len(colors)]
            shown = [r for r in curves[name]
                     if r["d"] == p["d"] and r["lr"] == p["lr"] and r.get("history")]
            for j, r in enumerate(shown):
                hist = np.array(r["history"])
                ax.plot(hist[:, 0], hist[:, 1], color=color, lw=0.9, alpha=0.55,
                        label=label_of[name] if j == 0 else None)
        ax.axhline(0.5, ls="--", c="grey", lw=0.8)
        ax.axhline(args.solve_threshold, ls=":", c="black", lw=0.8)
        ax.set_title(f"budget $\\approx$ {np.median(budget) / 1000:.1f}k params "
                     f"({min(budget)}-{max(budget)})", fontsize=9)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    axes[0][0].set_ylabel(r"accuracy of $y_L$")
    axes[0][0].set_ylim(0.45, 1.03)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=8,
               frameon=False)
    fig.suptitle("Learning curves at the best learning rate, one line per seed",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.10 if len(labels) > 3 else 0.06, 1, 0.94))
    curve_path = os.path.join(args.out_dir, "learning_curves.png")
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    print("wrote", path, ",", curve_path, "and runs.json")


def print_summary(summary, args):
    print(f"\nsolve rate = fraction of runs with y_L accuracy >= {args.solve_threshold:g}, "
          f"at the learning rate that solves most often")
    header = (f"  {'arm':<32}{'d':>4}{'params':>8}{'lr':>8}{'solved':>9}"
              f"{'rate':>7}{'95% CI':>14}{'>=' + format(args.partial_threshold, 'g'):>7}"
              f"{'mean':>7}{'max':>7}{'jump@':>8}")
    print(header)
    for name, points in summary.items():
        for p in points:
            step = "-" if p["median_solve_step"] is None else f"{p['median_solve_step']:.0f}"
            print(f"  {name[:32]:<32}{p['d']:>4}{p['params']:>8}{p['lr']:>8g}"
                  f"{p['solved']:>5}/{p['n']:<3}{p['solve_rate']:>7.2f}"
                  f"{'[' + format(p['ci_lo'], '.2f') + ',' + format(p['ci_hi'], '.2f') + ']':>14}"
                  f"{p['partial']:>7}{p['mean']:>7.3f}{p['max']:>7.3f}{step:>8}")


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

    p.add_argument("--length", type=int, default=DEFAULT_L)
    p.add_argument("--vocab", type=int, default=DEFAULT_VOCAB, help="|M|, content tokens")
    p.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="block length l")
    p.add_argument("--reps", type=int, default=DEFAULT_REPS, help="repetition count c")
    p.add_argument("--hard-neg-frac", type=float, default=1.0,
                   help="fraction of negatives built by planting then breaking a copy; "
                        "1.0 closes the token-frequency shortcut, 0.0 is plain noise")

    p.add_argument("--configs", type=str, nargs="+",
                   default=["construction:TF,SSM,SSM", "construction:SSM,SSM,TF"],
                   help="arms to run; the default pair differs only in layer order")
    p.add_argument("--variant", choices=["construction", "generic"], default="construction")
    p.add_argument("--layers", type=str, default="TF,SSM,SSM")
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
    p.add_argument("--seeds", type=int, default=8)
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
    p.add_argument("--verbose", action="store_true")
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


def main():
    global SCAN_MODE
    args = parse_args()
    SCAN_MODE = args.scan
    device = torch.device(args.device)
    cap_gpu_memory(device, args.max_gpu_gb)
    if args.self_test:
        self_test(args, device)
    elif args.replot:
        replot(args, device)
    elif args.acc_figures:
        acc_figures(args, device)
    elif args.verify:
        verify(args, device)
    elif args.params_table:
        params_table(args, device)
    elif args.compare:
        compare(args, device)
    elif args.sweep:
        sweep(args, device)
    else:
        print("nothing to do; pass --self-test, --verify, --params-table, "
              "--compare, --sweep or --replot")


if __name__ == "__main__":
    main()
