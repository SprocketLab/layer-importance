#!/usr/bin/env python3
"""MQAR: multi-query associative recall (1-token key, 2-token value) on the 8-layer stack.

Items use the explicit format  <bos> key #0 value value <eos>.  The first occurrence of a key
defines its value; a later occurrence is a query whose value span is scored.  Everything else
follows decode_recall.py: d=384, window 20, 48 epochs, auto-LR per config.  Results are
namespaced under results/mqar/ (the run-condition tag has no task field).
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os
import decode_recall as RE

RE.TASK = "mqar"
RE.RESULTS_DIR = os.path.join(RE.RESULTS_DIR, "mqar")

# Two ordering families swept over #SSM in {4..7} (TF-at-end = SSM-first, TF-at-start = SSM-last),
# alternating variants, and both pure anchors.
SUBSET = [
    "pure_tf", "pure_ssm",
    "4s4t_ssm_first", "5s3t_end", "6s2t_end", "7s1t_end",
    "4s4t_tf_first", "5s3t_start", "6s2t_start", "7s1t_start",
    "4s4t_alternate", "5s3t_alt", "3s5t_alt",
]
SEEDS = [0, 1, 2, 3, 4]

print(f"=== MQAR [{RE.cond_tag()}] @ {RE.RESULTS_DIR} "
      f"— {len(SUBSET)} configs x {len(SEEDS)} seeds ===", flush=True)
print("Step 1: auto-LR search per config on MQAR", flush=True)
lrs = {c: RE.find_best_lr(c) for c in SUBSET}
print("\nResolved MQAR auto-LR:", flush=True)
for c in SUBSET:
    print(f"  {c:16s} {RE.CONFIGS[c]}  lr={lrs[c]:.2e}", flush=True)

plan = [(c, lrs[c], s) for c in SUBSET for s in SEEDS]
print(f"\nStep 2: {len(plan)} training runs", flush=True)
RE.run_plan(plan, "mqar")
print("=== MQAR done ===", flush=True)
