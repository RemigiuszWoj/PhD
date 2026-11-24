"""Quantum vs classical k-bonacci moves (ILS, 10 s), figures and numbers for the ICAPS paper.

Uses only instances 0-4 and seeds 0-2, the set computed on all three platforms
(IQM: 100x10 has seed 0 only). Writes figures/fig_quantum_time.pdf and prints the numbers quoted in the text.
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "articles" / "kbonacci" / "figures"
PAT = re.compile(r"neigh=(?:quantum|gate)_kbonacci_k(\d+)__file=(tai\d+_\d+)__inst=(\d+)__n(\d+)__m(\d+)__tl=10000ms__seed=(\d)")
INST, SEEDS = range(5), range(3)


def load(pattern: str) -> dict:
    out = {}
    for f in glob.glob(str(ROOT / pattern)):
        mt = PAT.search(f)
        if not mt:  # pilot runs at 5 s
            continue
        k, fi, i, n, m, s = mt.groups()
        if int(i) in INST and int(s) in SEEDS:
            out[(fi, int(k), int(i), int(s))] = json.loads(Path(f).read_text())
    return out


def classical(fi: str, k: int, i: int, s: int) -> dict:
    n, m = fi[3:].split("_")
    p = ROOT / f"results/kbonacci/main_budgets/algo=ils__neigh=kbonacci_k{k}__file={fi}__inst={i}__n{n}__m{m}__tl=10000ms__seed={s}/result.json"
    return json.loads(p.read_text())


def rpd(c: float, lb: float) -> float:
    return 100.0 * (c - lb) / lb


def best_at(start: int, log: list, t: np.ndarray) -> np.ndarray:
    """Best Cmax found by time t (a move counts once it has ended)."""
    ends = np.array([e for e, _ in log] or [np.inf])
    vals = np.minimum.accumulate([c for _, c in log] or [start])
    idx = np.searchsorted(ends, t, side="right")
    return np.where(idx == 0, start, vals[np.maximum(idx - 1, 0)])


def class_key(fi: str) -> tuple:
    n, m = fi[3:].split("_")
    return int(n), int(m)


def main() -> None:
    dw, iqm = load("results/quantum_tiers/dwave/*/result.json"), load("results/quantum_tiers/iqm_ils_sirius/*/result.json")
    assert iqm and all(key in dw for key in iqm), "every IQM run must have its D-Wave twin"

    # numbers for the text
    dw_wins_iqm = sum(dw[key]["cmax_best"] < r["cmax_best"] for key, r in iqm.items())
    iqm_impr = sum(r["cmax_best"] < r["cmax_history"][0] for r in iqm.values())
    cl_wins = sum(classical(*key)["cmax_best"] < r["cmax_best"] for key, r in dw.items())
    empty = sum(r["qpu_stats"]["quantum_empty_moves"] for r in dw.values())
    moves = sum(r["iterations"] for r in dw.values())
    per_class = defaultdict(list)
    for (fi, k, i, s), r in dw.items():
        per_class[(fi, k)].append(r["iterations"])
    mv = [np.mean(v) for v in per_class.values()]
    print(f"IQM runs {len(iqm)}, improve start {iqm_impr}; D-Wave below IQM {dw_wins_iqm}")
    print(f"D-Wave runs {len(dw)}, classical better in {cl_wins}; empty moves {empty} of {moves}; "
          f"mean moves per class/k {min(mv):.1f}-{max(mv):.1f}")

    # Figure: best RPD against time, k = 3, on the runs made on all three platforms
    panels = ["tai20_10", "tai50_10", "tai100_10"]
    t = np.logspace(-2, np.log10(60), 400) * 1000
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=False)
    for ax, fi in zip(axes, panels):
        keys = [key for key in iqm if key[0] == fi and key[1] == 3]
        curves = {"IQM": [], "D-Wave": [], "classical": []}
        for key in keys:
            q, d = iqm[key], dw[key]
            c = json.loads(Path(classical(*key)["source_run"]).read_text())  # full run with move_log
            lb, start = q["lower_bound"], q["cmax_history"][0]
            for name, r in (("IQM", q), ("D-Wave", d), ("classical", c)):
                curves[name].append(rpd(best_at(start, r["move_log"], t), lb))
        for (name, cs), ls, col in zip(curves.items(), [":", "--", "-"], ["0.45", "0.25", "0"]):
            ax.plot(t / 1000, np.mean(cs, axis=0), ls=ls, color=col, lw=1.1, label=name)
        ax.axvline(10, color="0.6", lw=0.6)
        ax.set_xscale("log")
        n, m = class_key(fi)
        ax.set_title(f"{n}$\\times${m}, k = 3 ({len(keys)} runs)", fontsize=8)
        ax.set_xlabel("time [s]")
    axes[0].set_ylabel("mean RPD [%]")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_quantum_time.pdf")


if __name__ == "__main__":
    main()
