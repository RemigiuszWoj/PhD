"""PRD per time budget for the four neighborhoods under four solver back-ends.

Classical, quantum-annealing and windowed-annealing rows are the measurements of
the windowed QUBO study on the same Taillard 20x5 instances; the gate row is
measured here on ibm_fez and is budget-invariant, since a run completes exactly
one move at every budget of the protocol.

Run: python solver_comparison.py   ->  solver_comparison.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = [100, 500, 1000, 2000, 5000, 10000]

# ILS rows. Classical / Q / Q-enhanced from the windowed annealing study.
DATA = {
    "Adjacent": {
        "Classical":        [6.60, 4.59, 3.89, 3.68, 3.52, 3.41],
        "Annealer":         [25.09, 25.04, 25.07, 23.53, 22.25, 20.17],
        "Annealer, windowed": [25.04, 25.04, 25.04, 24.11, 22.27, 20.03],
        "Gate QAOA":        [25.04] * 6,
    },
    "Fibonacci": {
        "Classical":        [5.96, 4.25, 4.09, 3.99, 3.87, 3.82],
        "Annealer":         [23.71, 23.44, 23.79, 21.49, 19.97, 17.94],
        "Annealer, windowed": [23.57, 23.51, 23.60, 22.74, 19.62, 18.20],
        "Gate QAOA":        [23.41] * 6,
    },
    "Dynasearch": {
        "Classical":        [3.92, 3.47, 3.32, 3.08, 2.83, 2.68],
        "Annealer":         [25.77, 26.21, 26.66, 26.13, 26.11, 25.85],
        "Annealer, windowed": [25.76, 25.68, 25.56, 25.76, 24.57, 26.24],
        "Gate QAOA":        [22.12] * 6,
    },
    "Motzkin": {
        "Classical":        [4.33, 3.42, 3.19, 2.92, 2.75, 2.64],
        "Annealer":         [21.33, 21.68, 21.66, 21.40, 19.05, 19.02],
        "Annealer, windowed": [24.82, 25.33, 25.98, 25.07, 24.68, 25.22],
        "Gate QAOA":        [23.12] * 6,
    },
}

STYLE = {
    "Classical":          dict(color="0.25", marker="o", ls="-",  lw=1.4, ms=4),
    "Annealer":           dict(color="#1f6fb4", marker="s", ls="--", lw=1.4, ms=4),
    "Annealer, windowed": dict(color="#7aa9d6", marker="^", ls=":",  lw=1.4, ms=4),
    "Gate QAOA":          dict(color="#c1452b", marker="D", ls="-",  lw=1.8, ms=4.5),
}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True, sharey=True)
for ax, (nb, series) in zip(axes.flat, DATA.items()):
    for name, ys in series.items():
        ax.plot(BUDGETS, ys, label=name, **STYLE[name])
    ax.set_xscale("log")
    ax.set_title(nb, fontsize=10)
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=7)
    ax.tick_params(axis="y", labelsize=8)

for ax in axes[1]:
    ax.set_xlabel("time budget [ms]", fontsize=9)
for ax in axes[:, 0]:
    ax.set_ylabel("PRD [%]", fontsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(f"solver_comparison.{ext}", dpi=200, bbox_inches="tight")
print("wrote solver_comparison.pdf / .png")
