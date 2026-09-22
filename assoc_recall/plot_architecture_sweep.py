#!/usr/bin/env python3
"""Redraw the all-22-config d=384 comparison figure, but with each model's FULL
configuration spelled out on the label: the shorthand name AND the actual 8-layer
S/T arrangement (e.g. 4s4t_alternate -> S-T-S-T-S-T-S-T). x = #SSM layers, color =
last-layer type (green TF / red SSM). One point per config = mean over its seeds at
the LR present in finals."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import decode_recall as RE

COND = "d384"
ROOT = os.path.join(RE.RESULTS_DIR, COND)

rows = []
for name, layout in RE.CONFIGS.items():
    cdir = os.path.join(ROOT, name)
    files = glob.glob(os.path.join(cdir, "*.json"))
    if not files:
        continue
    # group by lr-tag, keep the lr with the most seeds (the finals run)
    by_lr = {}
    for f in files:
        lrtag = os.path.basename(f).split("__")[1]
        by_lr.setdefault(lrtag, []).append(f)
    best = max(by_lr.values(), key=len)
    accs = [json.load(open(f))["mean_acc"] for f in best]
    rows.append({
        "name": name,
        "layout": layout,
        "ssm": layout.count("S"),
        "last_tf": layout[-1] == "T",
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "n": len(accs),
    })

rows.sort(key=lambda r: (r["ssm"], -r["mean"]))

fig, ax = plt.subplots(figsize=(17, 9))
# jitter configs sharing an SSM count so labels/points don't overlap
from collections import defaultdict
buckets = defaultdict(list)
for r in rows:
    buckets[r["ssm"]].append(r)

for ssm, group in buckets.items():
    k = len(group)
    offsets = np.linspace(-0.22, 0.22, k) if k > 1 else [0.0]
    for off, r in zip(offsets, group):
        r["x"] = ssm + off

for r in rows:
    color = "#2ca02c" if r["last_tf"] else "#d62728"
    ax.errorbar(r["x"], r["mean"], yerr=r["std"], fmt="o", ms=8,
                color=color, ecolor=color, elinewidth=1.2, capsize=3, zorder=3)
    pretty = "-".join(r["layout"])           # S-T-S-T-S-T-S-T
    label = f"{r['name']}\n{pretty}"
    va = "bottom" if r["last_tf"] else "top"
    dy = 0.012 if r["last_tf"] else -0.012
    ax.annotate(label, (r["x"], r["mean"] + dy), ha="center", va=va,
                fontsize=6.5, color="#222", zorder=4)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0], [0], marker="o", ls="", color="#2ca02c", label="last layer = TF"),
    Line2D([0], [0], marker="o", ls="", color="#d62728", label="last layer = SSM"),
], loc="lower left", fontsize=11)

ax.set_xlabel("SSM layers (out of 8)   —   left = TF-heavy,  right = SSM-heavy", fontsize=12)
ax.set_ylabel("Char Accuracy", fontsize=12)
ax.set_title(f"Decode-recall: ALL {len(rows)} configs at d=384, n_bits=5  "
             f"(label = name + S/T layout, 3–5 seeds)", fontsize=13)
ax.set_xticks(range(0, 9))
ax.grid(True, alpha=0.3)
ax.margins(y=0.10)

out = os.path.join(ROOT, "..", "architecture_sweep.png")
out = os.path.abspath(out)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out, "with", len(rows), "configs")
