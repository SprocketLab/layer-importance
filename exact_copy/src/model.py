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

# The stack the construction is written into. "The construction" always means
# this one -- --verify, self-test [11], the star on the sweep figures -- and its
# depth is the one the layer-count ablations vary away from.
CONSTRUCTION_LAYERS = ["TF", "SSM", "SSM"]
CONSTRUCTION_DEPTH = len(CONSTRUCTION_LAYERS)

# The pair of arms every parameter-matched figure starts from: the same three
# blocks, in the two orders the theory separates.
DEFAULT_CONFIGS = ["construction:TF,SSM,SSM", "construction:SSM,SSM,TF"]

# The four orders of the equal-memory figure: the two hybrids and the two pure
# stacks. Plain "SSM" makes the first SSM of a stack two-gated, which is the block
# the construction latches with, so TF->SSM here is the depth-2 stack that has a
# chance of expressing the task rather than a deliberately weakened one. Pass the
# SSM1 spellings instead for the single-gated family.
MEMORY_ARMS = ["construction:TF,SSM", "construction:SSM,TF",
               "construction:TF,TF", "construction:SSM,SSM"]


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
# Inference memory: what a window and a state dimension cost
# --------------------------------------------------------------------------- #
#
# The parameter-matched figures ask what an architecture can learn per weight.
# This is the other budget: what it can learn per *unit of state carried along the
# sequence*, which is the quantity attention and recurrence actually trade against
# each other. Attention keeps every key and value inside its window, so it pays
# 2 W d floats per layer and pays nothing at training time to reach further back.
# A selective SSM keeps one matrix S in R^{d x N}, so it pays N d floats per layer
# no matter how long the sequence is, and buys reach with compression instead.
#
# Fixing that total is what makes "TF -> SSM beats SSM -> TF" a claim about order
# rather than about one arm being handed a bigger cache.

# 1536 floats, with a grid of token dimensions that divides it exactly (see
# equal_memory_setting: a layer's share of 768 has to be a multiple of both 2d and
# d, which for 768 admits 8, 12, 16, 24, 32, 48, 64, 96, 128). The budget also
# places the W = l+1 cliff at d = 1536/24 = 64, inside the range where these models
# can still be trained, instead of out past d = 256 where a single run costs 16
# core-hours and the figure would be about compute rather than about memory.
DEFAULT_TOKEN_DIMS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
DEFAULT_MEMORY_BUDGET = 1536        # floats of inference state for the whole stack


def kv_cache_floats(d: int, window: int, length: int) -> int:
    """Keys and values one windowed attention layer must keep: 2 min(W, L) d.

    Capped at L because a window past the end of the sequence caches nothing more.
    """
    return min(window, length) * d
    #return 2 * min(window, length) * d


def ssm_state_floats(d: int, state_dim: int) -> int:
    """The recurrent state of one selective SSM layer: S in R^{d x N} is N d floats."""
    return state_dim * d


def stack_memory(layers, d: int, window: int, state_dim: int, length: int) -> int:
    """Inference state of a whole stack, in floats."""
    return sum(kv_cache_floats(d, window, length) if kind == "TF"
               else ssm_state_floats(d, state_dim)
               for kind in layers)


def equal_memory_setting(d: int, layers: int, budget: int, length: int, w_to_s: float):
    depth_tf = sum([1 if itr == "TF" else 0 for itr in layers])
    depth_ssm = sum([1 if itr == "SSM" else 0 for itr in layers])
    depth = depth_tf + depth_ssm

    window = max(1, min(length, budget / (d * (depth_tf + depth_ssm / w_to_s))))
    state_dim = max(1, window / w_to_s)

    window = int(window)
    state_dim = int(state_dim)

    return window, state_dim


#def memory_grid_dims(depth: int, budget: int, hi: int = 1024):
#    """The token dimensions that split `budget` exactly at this depth."""
#    share = budget // depth
#    return [d for d in range(1, hi + 1)
#            if share % (2 * d) == 0 and share % d == 0 and share // (2 * d) >= 1]


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
    model = build_model(args, d, "construction", CONSTRUCTION_LAYERS).to(device)
    set_construction_weights(model, args, verbose=verbose)
    model.eval()
    x, y = fixed_eval_set(args, n_examples)
    final, dense = evaluate(model, x.to(device), y.to(device), args.batch)
    return count_params(model), final, dense

