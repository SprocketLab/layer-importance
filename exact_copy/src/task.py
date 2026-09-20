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
