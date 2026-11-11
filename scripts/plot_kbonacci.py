"""Figures of the k-bonacci paper (PDF, grayscale-safe).

Inputs (each one optional; a figure is skipped when its input is missing):
  --tables   directory written by scripts/analyze_kbonacci.py (rpd_ils.csv, iters_ils.csv, ...)
  --anatomy  JSON written by scripts/kbonacci_anatomy.py

Figures:
  fig_rpd_vs_k.pdf      mean RPD vs k, one panel per n, one line per budget; reference
                        neighborhoods as horizontal marks at the longest budget
  fig_iters_vs_k.pdf    ILS iterations within the longest budget vs k (log-log), one line per n
  fig_best_k.pdf        best k for every (n, budget)
  fig_best_k_rel.pdf    best k / n for every Taillard class, x = n (needs --runs)
  fig_additivity.pdf    predicted vs real gain and share of optimistic predictions vs k
  fig_distances.pdf     distances of the swaps selected at k = n, per instance class
  fig_cost.pdf          evaluation time and candidates vs k for both filters

Usage:
    python scripts/plot_kbonacci.py --tables articles/kbonacci/tables \
        --anatomy articles/kbonacci/anatomy/anatomy.json --out articles/kbonacci/figures
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42})
GRAYS = ["0.75", "0.6", "0.45", "0.3", "0.15", "0.0"]
MARKS = ["o", "s", "^", "D", "v", "P"]
REFS = {"adjacent": "Adj", "fibonacci": "Fib", "dynasearch": "Dyn", "motzkin": "Motz", "insert": "Ins"}


def k_of(name: str, n: int):
    """x-coordinate of a k-bonacci row; k = n is drawn at x = n."""
    if not name.startswith("kbonacci_k") or name.endswith("_cp"):
        return None
    tail = name[len("kbonacci_k"):]
    return n if tail == "n" else int(tail)


def read_table(path: Path):
    """{n: {neighborhood: {budget_ms: value}}}"""
    out = defaultdict(dict)
    with open(path) as fh:
        rd = csv.reader(fh)
        budgets = [int(h[:-2]) for h in next(rd)[2:]]
        for row in rd:
            out[int(row[0])][row[1]] = {b: float(v) for b, v in zip(budgets, row[2:]) if v != ""}
    return out, budgets


def fig_rpd(tables: Path, out: Path, algo: str) -> None:
    data, budgets = read_table(tables / f"rpd_{algo}.csv")
    ns = sorted(data)
    fig, axes = plt.subplots(1, len(ns), figsize=(1.75 * len(ns), 2.2), sharey=False)
    axes = [axes] if len(ns) == 1 else list(axes)
    for ax, n in zip(axes, ns):
        for bi, b in enumerate(budgets):
            pts = sorted((k_of(nb, n), v[b]) for nb, v in data[n].items()
                         if k_of(nb, n) is not None and b in v and k_of(nb, n) <= n)
            if pts:
                ax.plot(*zip(*pts), marker=MARKS[bi % 6], ms=2.5, lw=0.9, color=GRAYS[bi % 6],
                        label=f"{b / 1000:g} s")
        for nb, lab in REFS.items():
            v = data[n].get(nb, {}).get(budgets[-1])
            if v is not None:
                ax.axhline(v, lw=0.5, ls=":", color="0.4")
                ax.text(1.0, v, lab, transform=ax.get_yaxis_transform(), fontsize=5.5, va="bottom", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_title(f"$n={n}$")
        ax.set_xlabel("$k$")
    axes[0].set_ylabel("mean RPD [%]")
    axes[-1].legend(fontsize=6, title="budget", title_fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_rpd_vs_k.pdf")
    plt.close(fig)


def fig_iters(tables: Path, out: Path, algo: str) -> None:
    data, budgets = read_table(tables / f"iters_{algo}.csv")
    b = budgets[-1]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for i, n in enumerate(sorted(data)):
        pts = sorted((k_of(nb, n), v[b]) for nb, v in data[n].items()
                     if k_of(nb, n) is not None and b in v and v[b] > 0 and k_of(nb, n) <= n)
        if pts:
            ax.plot(*zip(*pts), marker=MARKS[i % 6], ms=2.5, lw=0.9, color=GRAYS[i % 6], label=f"$n={n}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(f"iterations in {b / 1000:g} s")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_iters_vs_k.pdf")
    plt.close(fig)


def fig_best_k(tables: Path, out: Path, algo: str) -> None:
    data, budgets = read_table(tables / f"rpd_{algo}.csv")
    ns = sorted(data)
    fig, ax = plt.subplots(figsize=(3.3, 0.45 * len(ns) + 0.8))
    for yi, n in enumerate(ns):
        for xi, b in enumerate(budgets):
            cand = [(v[b], k_of(nb, n), nb) for nb, v in data[n].items() if k_of(nb, n) is not None and b in v]
            if not cand:
                continue
            _, k, nb = min(cand)
            lab = "$n$" if nb.endswith("kn") else str(k)
            # darker cell = larger k among the tested values
            shade = 1.0 - 0.75 * sorted(c[1] for c in cand).index(k) / max(1, len(cand) - 1)
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, color=str(shade), ec="white"))
            ax.text(xi + 0.5, yi + 0.5, lab, ha="center", va="center",
                    color="white" if shade < 0.5 else "black")
    ax.set_xlim(0, len(budgets))
    ax.set_ylim(0, len(ns))
    ax.set_xticks([i + 0.5 for i in range(len(budgets))], [f"{b / 1000:g}" for b in budgets])
    ax.set_yticks([i + 0.5 for i in range(len(ns))], [str(n) for n in ns])
    ax.set_xlabel("budget [s]")
    ax.set_ylabel("$n$")
    ax.set_title("best $k$ by mean RPD")
    fig.tight_layout()
    fig.savefig(out / "fig_best_k.pdf")
    plt.close(fig)


def fig_best_k_rel(dirs, out: Path, algo: str, budgets=(100, 1000, 10000)) -> None:
    """Best k as a fraction of n, per Taillard class: x = n, y = k/n (log-log).

    Grey dots: every tested k. Filled marker: best mean RPD of the class (marker = m).
    Open marker: k not distinguishable from the best (paired two-sided Wilcoxon over
    instance x seed, p >= 0.05). Dashed: (log2 n + 2) / n, where N_k already holds
    most of Dynasearch (share ~ exp(-n / 2^(k+1)))."""
    import math
    import sys

    from scipy.stats import wilcoxon

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_kbonacci import load

    runs = defaultdict(dict)  # (file, tl) -> {k: {(inst, seed): rpd}}
    for r in load(dirs, None):
        k = k_of(r["neigh"], r["n"])
        if r["algo"] == algo and k is not None:
            runs[(r["file"], r["tl"])].setdefault(min(k, r["n"]), {})[(r["inst"], r["seed"])] = r["rpd"]
    budgets = [b for b in budgets if any(t == b for _, t in runs)]
    if not budgets:
        return
    ms = sorted({int(f.split("_")[1]) for f, _ in runs})
    mark = dict(zip(ms, MARKS))
    shade = dict(zip(ms, ["0.0", "0.35", "0.6"]))
    fig, axes = plt.subplots(1, len(budgets), figsize=(2.4 * len(budgets), 2.6), sharey=True, squeeze=False)
    for ax, b in zip(axes[0], budgets):
        for (f, t), by_k in sorted(runs.items()):
            if t != b:
                continue
            n, m = (int(x) for x in f[3:].split("_"))
            dx = {ms[0]: 0.92, ms[-1]: 1.08}.get(m, 1.0)  # spread the m classes of one n
            means = {k: statistics.mean(v.values()) for k, v in by_k.items()}
            kbest = min(means, key=means.get)
            ax.scatter([n * dx] * len(means), [k / n for k in means], s=4, color="0.8", zorder=1)
            for k, v in by_k.items():
                if k == kbest:
                    continue
                keys = sorted(v.keys() & by_k[kbest].keys())
                a_, b_ = [v[x] for x in keys], [by_k[kbest][x] for x in keys]
                tie = a_ == b_ or (len(keys) >= 6 and wilcoxon(a_, b_).pvalue >= 0.05)
                if tie:
                    ax.scatter(n * dx, k / n, s=16, marker=mark[m], facecolors="none",
                               edgecolors=shade[m], linewidths=0.6, zorder=2)
            ax.scatter(n * dx, kbest / n, s=30, marker=mark[m], color=shade[m], zorder=3)
        ns = sorted({int(f[3:].split("_")[0]) for f, _ in runs})
        xs = [ns[0] * 0.85] + ns + [ns[-1] * 1.15]
        ax.plot(xs, [min(1.0, (math.log2(x) + 2) / x) for x in xs], "k--", lw=0.7, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("$n$")
        ax.set_title(f"{b / 1000:g} s")
    axes[0][0].set_ylabel("$k / n$")
    axes[0][0].set_yticks([0.005, 0.01, 0.03, 0.1, 0.3, 1.0], ["0.5%", "1%", "3%", "10%", "30%", "100%"])
    handles = [plt.Line2D([], [], ls="", marker=mark[m], color=shade[m], label=f"$m={m}$") for m in ms]
    handles.append(plt.Line2D([], [], ls="--", color="k", lw=0.7, label=r"$(\log_2 n+2)/n$"))
    axes[0][-1].legend(handles=handles, loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_best_k_rel.pdf")
    plt.close(fig)


def class_key(f: str):
    return tuple(int(v) for v in f[3:].split("_"))


def fig_anatomy(path: Path, out: Path) -> None:
    rows = json.loads(path.read_text())
    files = sorted({r["file"] for r in rows}, key=class_key)
    cp = [r for r in rows if r["filter"] == "critical_path"]
    kx = lambda r: r["n"] if r["k"] == "n" else int(r["k"])  # noqa: E731

    # additivity: predicted and real gain, share of optimistic predictions
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for i, f in enumerate(files):
        g = defaultdict(list)
        for r in cp:
            if r["file"] == f:
                g[kx(r)].append(r)
        ks = sorted(g)
        # both means are negative (gains), so the ratio is the realized share of the predicted gain
        ratio = [statistics.mean(r["real"] for r in g[k]) / min(-1e-9, statistics.mean(r["pred"] for r in g[k]))
                 for k in ks]
        optim = [sum(r["real"] > r["pred"] for r in g[k]) / len(g[k]) for k in ks]
        style = dict(marker=MARKS[i % 6], ms=2.5, lw=0.9, color=GRAYS[(i // 3 * 2 + 1) % 6],
                     ls=["-", "--", ":"][i % 3], label=f.replace("tai", "").replace("_", r"$\times$"))
        a1.plot(ks, ratio, **style)
        a2.plot(ks, optim, **style)
    for ax, lab in ((a1, "real gain / predicted gain"), (a2, "share of optimistic predictions")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("$k$")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.05)
    a2.legend(fontsize=5.5, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig_additivity.pdf")
    plt.close(fig)

    # distances of the swaps selected by the largest k
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    data, labels = [], []
    for f in files:
        d = [x / r["n"] for r in cp if r["file"] == f and r["k"] == "n" for x in r["dist"]]
        if d:
            data.append(d)
            labels.append(f.replace("tai", "").replace("_", "x"))
    if data:
        ax.boxplot(data, tick_labels=labels, showfliers=False, medianprops={"color": "black"})
        ax.set_ylabel("swap distance / $n$ at $k=n$")
        ax.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        fig.savefig(out / "fig_distances.pdf")
    plt.close(fig)

    # cost: evaluation time vs k for both filters (largest class of every n)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    largest = {}
    for f in files:
        largest[class_key(f)[0]] = f
    for i, f in enumerate(largest.values()):
        for flt, ls in (("last_machine", "--"), ("critical_path", "-")):
            g = defaultdict(list)
            for r in rows:
                if r["file"] == f and r["filter"] == flt:
                    g[kx(r)].append(r["t_ms"])
            ks = sorted(g)
            ax.plot(ks, [statistics.mean(g[k]) for k in ks], ls=ls, marker=MARKS[i % 6], ms=2.5, lw=0.9,
                    color=GRAYS[(2 * i + 1) % 6],
                    label=f.replace("tai", "").replace("_", r"$\times$") if ls == "-" else None)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel("evaluation time [ms]")
    ax.legend(fontsize=6, title="solid: exact filter, dashed: last-machine", title_fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out / "fig_cost.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=None)
    ap.add_argument("--anatomy", type=Path, default=None)
    ap.add_argument("--runs", nargs="*", default=[],
                    help="result dirs with per-run data, e.g. results/kbonacci/main_budgets "
                         "results/kbonacci/relative_budgets (for fig_best_k_rel)")
    ap.add_argument("--algo", default="ils")
    ap.add_argument("--out", type=Path, default=Path("articles/kbonacci/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.runs:
        fig_best_k_rel(a.runs, a.out, a.algo)
        print("relative-k figure written")
    if a.tables and (a.tables / f"rpd_{a.algo}.csv").exists():
        fig_rpd(a.tables, a.out, a.algo)
        fig_best_k(a.tables, a.out, a.algo)
        if (a.tables / f"iters_{a.algo}.csv").exists():
            fig_iters(a.tables, a.out, a.algo)
        print("table figures written")
    if a.anatomy and a.anatomy.exists():
        fig_anatomy(a.anatomy, a.out)
        print("anatomy figures written")


if __name__ == "__main__":
    main()
