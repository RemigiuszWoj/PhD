"""Figure: QPU quantum neighborhoods at n=20 — RPD vs time budget.

Data: tranche P1 (results/experiments/20260703_083952), real
Advantage_system4.1 runs, ILS+SA pooled. Point labels show the median
iteration count — the budget buys iterations, and iterations buy RPD.

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

D = "results/experiments/20260703_083952"
OUT = Path(__file__).parent / "qpu_convergence_n20"

rows = [json.load(open(f)) for f in glob.glob(f"{D}/*/result.json")]
by = defaultdict(list)
for r in rows:
    c = r["config"]
    by[(c["neighborhood"], c["time_limit_ms"])].append(r)

SERIES = {
    "quantum_motzkin": ("Q-Motzkin", "#d62728", "o"),
    "quantum_fibonacci": ("Q-Fibonacci", "#9467bd", "s"),
    "quantum_adjacent": ("Q-Adjacent", "#1f77b4", "^"),
    "quantum_dynasearch": ("Q-Dynasearch", "#ff7f0e", "D"),
}

fig, ax = plt.subplots(figsize=(5.4, 3.4))
for neigh, (label, color, marker) in SERIES.items():
    tls, rpds, iters = [], [], []
    for tl in (500, 1000, 5000):
        rs = by.get((neigh, tl), [])
        if len(rs) < 20:
            continue
        gaps = [x["gap_percent"] for x in rs if x["gap_percent"] is not None]
        its = sorted((x.get("iterations") or 0) for x in rs)
        tls.append(tl)
        rpds.append(sum(gaps) / len(gaps))
        iters.append(its[len(its) // 2])
    ax.plot(tls, rpds, marker=marker, color=color, label=label, lw=1.4, ms=5)
    for tl, rpd, it in zip(tls, rpds, iters):
        ax.annotate(f"{it} it.", (tl, rpd), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=color)

ax.set_xscale("log")
ax.set_xticks([500, 1000, 5000])
ax.set_xticklabels(["500", "1000", "5000"])
ax.set_xlabel(r"time budget $t_{\max}$ [ms]")
ax.set_ylabel("mean RPD [%]")
ax.grid(True, alpha=0.25, lw=0.5)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(f"{OUT}.png", dpi=200)
fig.savefig(f"{OUT}.pdf")
print(f"saved {OUT}.png/.pdf")
