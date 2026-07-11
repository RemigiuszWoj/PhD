"""Figure: QPU quantum neighborhoods at n=20 — RPD vs time budget.

Originals (solid): tranche P1, results/experiments/20260703_083952.
Enhanced (dashed): tranches P2/T3, results/experiments/20260707_203922.
Real Advantage_system4.1 runs, ILS+SA pooled; points with >=50 runs only.
Labels on the original series show the median iteration count.

Run from repo root:
    .venv311/bin/python3 articles/windowed_qubo/figures/qpu_convergence_n20.py
"""

import json
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIRS = {
    "orig": "results/experiments/20260703_083952",
    "enh": "results/experiments/20260707_203922",
}
OUT = Path(__file__).parent / "qpu_convergence_n20"

data = {}
for kind, d in DIRS.items():
    by = defaultdict(list)
    for f in glob.glob(f"{d}/*/result.json"):
        r = json.load(open(f))
        c = r["config"]
        by[(c["neighborhood"], c["time_limit_ms"])].append(r)
    data[kind] = by

SERIES = {
    "quantum_motzkin": ("Motzkin", "#d62728", "o"),
    "quantum_fibonacci": ("Fibonacci", "#9467bd", "s"),
    "quantum_adjacent": ("Adjacent", "#1f77b4", "^"),
    "quantum_dynasearch": ("Dynasearch", "#ff7f0e", "D"),
}

fig, ax = plt.subplots(figsize=(5.8, 3.6))
for base, (label, color, marker) in SERIES.items():
    for kind, suffix, ls, lw in (("orig", "", "-", 1.5), ("enh", "_enhanced", "--", 1.2)):
        neigh = base + suffix
        tls, rpds, iters = [], [], []
        for tl in (100, 500, 1000, 5000):
            rs = data[kind].get((neigh, tl), [])
            if len(rs) < 50:
                continue
            gaps = [x["gap_percent"] for x in rs if x["gap_percent"] is not None]
            its = sorted((x.get("iterations") or 0) for x in rs)
            tls.append(tl)
            rpds.append(sum(gaps) / len(gaps))
            iters.append(its[len(its) // 2])
        if not tls:
            continue
        ax.plot(tls, rpds, marker=marker, color=color, ls=ls, lw=lw,
                ms=4.5, mfc=(color if kind == "orig" else "white"),
                label=(f"{label}" if kind == "orig" else f"{label} Enh."))
        if kind == "orig":
            for tl, rpd, it in zip(tls, rpds, iters):
                ax.annotate(f"{it} it.", (tl, rpd), textcoords="offset points",
                            xytext=(6, 4), fontsize=6.5, color=color)

ax.set_xscale("log")
ax.set_xticks([100, 500, 1000, 5000])
ax.set_xticklabels(["100", "500", "1000", "5000"])
ax.set_xlabel(r"time budget $t_{\max}$ [ms]")
ax.set_ylabel("mean RPD [%]")
ax.grid(True, alpha=0.25, lw=0.5)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}.png", dpi=200)
fig.savefig(f"{OUT}.pdf")
print(f"saved {OUT}.png/.pdf")
