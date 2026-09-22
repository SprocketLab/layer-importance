#!/usr/bin/env python3
"""MKAR: multi-key associative recall (2-token key, 1-token value) on the 8-layer stack.

The sibling of mqar.py with the key and value lengths swapped:  <bos> key key #0 value <eos>.
Recall difficulty then sits in matching a longer key rather than emitting a longer value.
Same protocol (d=384, window 20, 48 epochs, auto-LR per config); 7 seeds because this task has
the widest error bars.  Results are namespaced under results/mkar/.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os
import decode_recall as RE

RE.TASK = "mkar"
RE.RESULTS_DIR = os.path.join(RE.RESULTS_DIR, "mkar")

# Same subset as MQAR so the two tasks stay directly comparable.
SUBSET = [
    "pure_tf", "pure_ssm",
    "4s4t_ssm_first", "5s3t_end", "6s2t_end", "7s1t_end",
    "4s4t_tf_first", "5s3t_start", "6s2t_start", "7s1t_start",
    "4s4t_alternate", "5s3t_alt", "3s5t_alt",
]
SEEDS = [0, 1, 2, 3, 4, 5, 6]

print(f"=== MKAR [{RE.cond_tag()}] @ {RE.RESULTS_DIR} "
      f"— {len(SUBSET)} configs x {len(SEEDS)} seeds ===", flush=True)
print("Step 1: auto-LR search per config on MKAR", flush=True)
lrs = {c: RE.find_best_lr(c) for c in SUBSET}
print("\nResolved MKAR auto-LR:", flush=True)
for c in SUBSET:
    print(f"  {c:16s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

plan = [(c, lrs[c], s) for c in SUBSET for s in SEEDS]
print(f"\nStep 2: {len(plan)} training runs", flush=True)
RE.run_plan(plan, "mkar")
print("=== MKAR done ===", flush=True)
