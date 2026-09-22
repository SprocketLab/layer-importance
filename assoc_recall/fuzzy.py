#!/usr/bin/env python3
"""Fuzzy recall: 2-token key AND 2-token value on the 8-layer stack.

<bos> key key #0 value value <eos>.  The harder 2+2 task is used to separate the hybrids that are
nearly tied on MQAR, so only the competitive hybrids run (no pure baselines).  Same protocol as
mqar.py / mkar.py (d=384, window 20, 48 epochs, auto-LR per config, 7 seeds); results are
namespaced under results/fuzzy/.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os
import decode_recall as RE

RE.TASK = "fuzzy"
RE.RESULTS_DIR = os.path.join(RE.RESULTS_DIR, "fuzzy")

SUBSET = [
    "4s4t_alternate",                          # alternating (decode-recall winner)
    "4s4t_ssm_first", "5s3t_end", "6s2t_end",  # TF-at-end family
    "5s3t_start", "6s2t_start",                # TF-at-start family
]
SEEDS = [0, 1, 2, 3, 4, 5, 6]

print(f"=== FUZZY [{RE.cond_tag()}] @ {RE.RESULTS_DIR} "
      f"— {len(SUBSET)} configs x {len(SEEDS)} seeds ===", flush=True)
print("Step 1: auto-LR search per config on fuzzy recall", flush=True)
lrs = {c: RE.find_best_lr(c) for c in SUBSET}
print("\nResolved fuzzy auto-LR:", flush=True)
for c in SUBSET:
    print(f"  {c:16s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

plan = [(c, lrs[c], s) for c in SUBSET for s in SEEDS]
print(f"\nStep 2: {len(plan)} training runs", flush=True)
RE.run_plan(plan, "fuzzy")
print("=== FUZZY done ===", flush=True)
