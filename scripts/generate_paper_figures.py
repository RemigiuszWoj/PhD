"""Generate LNCS-ready figures from experiment summaries.

This script is intentionally dependency-light (stdlib + matplotlib).
It reads `results/experiments/<run_id>/summary.csv` and produces:

1) Quality vs time-limit curves (mean over instances) for each neighborhood.
2) Win-rate bar chart (how often a neighborhood wins on instances), per algorithm and TL.

Outputs are written to `figures/paper/<run_id>/`.

Usage:
    python scripts/generate_paper_figures.py --run-id 20251206_181012
    python scripts/generate_paper_figures.py --run-id 20251206_181012 --format png
    python scripts/generate_paper_figures.py --run-id 20251206_181012 --format png

Notes:
- Assumes minimization of `cmax_best`.
- Uses relative gap (%) as the main metric for cross-instance comparison.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless


@dataclass(frozen=True)
class Row:
    algorithm: str
    neighborhood: str
    instance_file: str
    instance_number: int
    jobs: int
    machines: int
    seed: int
    time_limit_ms: int
    cmax_best: int
    lower_bound: int
    gap_percent: float


NEIGHBOR_ORDER = [
    "adjacent",
    "fibonahi_neighborhood",
    "dynasearch_neighborhood",
    "motzkin_neighborhood",
]

NEIGHBOR_LABEL = {
    "adjacent": "Adjacent",
    "fibonahi_neighborhood": "Fibonacci",
    "dynasearch_neighborhood": "Dynasearch",
    "motzkin_neighborhood": "Motzkin",
}

ALGO_LABEL = {
    "tabu": "Tabu Search",
    "sa": "Simulated Annealing",
}


def read_summary_csv(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    algorithm=r["algorithm"],
                    neighborhood=r["neighborhood"],
                    instance_file=r["instance_file"],
                    instance_number=int(r["instance_number"]),
                    jobs=int(r["jobs"]),
                    machines=int(r["machines"]),
                    seed=int(r["seed"]),
                    time_limit_ms=int(r["time_limit_ms"]),
                    cmax_best=int(r["cmax_best"]),
                    lower_bound=int(r["lower_bound"]),
                    gap_percent=float(r["gap_percent"]),
                )
            )
    return rows


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def stdev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_iter_log(path: Path) -> tuple[list[int], list[int]]:
    """Read iter_log_*.csv.

    Returns:
        (elapsed_ms, best_cmax)
    """
    elapsed: list[int] = []
    best: list[int] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            elapsed.append(int(r["elapsed_ms"]))
            best.append(int(r["best_cmax"]))
    return elapsed, best


def plot_convergence_from_compare_dir(
    compare_dir: Path,
    out_dir: Path,
    *,
    title: str,
    out_format: str,
    dpi: int,
) -> Path:
    """Plot convergence (best_cmax vs elapsed_ms) from a compare folder.

    The folder is expected to contain:
        iter_log_adjacent.csv
        iter_log_fibonahi_neighborhood.csv
        iter_log_dynasearch_neighborhood.csv
        iter_log_motzkin_neighborhood.csv
    """

    series: dict[str, tuple[list[int], list[int]]] = {}
    missing: list[str] = []
    for neigh in NEIGHBOR_ORDER:
        p = compare_dir / f"iter_log_{neigh}.csv"
        if not p.exists():
            missing.append(neigh)
            continue
        series[neigh] = read_iter_log(p)

    if not series:
        raise FileNotFoundError(f"No iter logs found in: {compare_dir}")

    plt.figure(figsize=(6.0, 3.6))
    for neigh in NEIGHBOR_ORDER:
        if neigh not in series:
            continue
        x, y = series[neigh]
        plt.plot(x, y, linewidth=1.8, label=NEIGHBOR_LABEL[neigh])

    plt.xlabel("Elapsed time [ms]")
    plt.ylabel("Best $C_{max}$")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    if missing:
        plt.gca().text(
            0.99,
            0.01,
            "Missing: " + ", ".join(NEIGHBOR_LABEL.get(m, m) for m in missing),
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            alpha=0.8,
        )
    plt.tight_layout()

    out_path = out_dir / f"convergence_{compare_dir.name}.{out_format}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def plot_gap_vs_time_limit(
    rows: list[Row],
    out_dir: Path,
    *,
    algorithm: str,
    out_format: str,
    dpi: int,
) -> Path:
    # Aggregate: for each TL and neighborhood ⇒ mean gap over all instances.
    bucket: DefaultDict[tuple[int, str], list[float]] = defaultdict(list)
    tls: set[int] = set()

    for r in rows:
        if r.algorithm != algorithm:
            continue
        if r.neighborhood not in NEIGHBOR_LABEL:
            continue
        tls.add(r.time_limit_ms)
        bucket[(r.time_limit_ms, r.neighborhood)].append(r.gap_percent)

    tl_sorted = sorted(tls)

    plt.figure(figsize=(6.0, 3.6))
    for neigh in NEIGHBOR_ORDER:
        ys = [mean(bucket[(tl, neigh)]) for tl in tl_sorted]
        plt.plot(
            tl_sorted,
            ys,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=NEIGHBOR_LABEL[neigh],
        )

    plt.xscale("log")
    plt.xlabel("Time limit [ms] (log scale)")
    plt.ylabel("Relative gap to LB [%] (mean)")
    plt.title(f"{ALGO_LABEL.get(algorithm, algorithm)}: gap vs time limit")
    plt.grid(True, which="both", linestyle=":", linewidth=0.8)
    plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()

    out_path = out_dir / f"gap_vs_time_{algorithm}.{out_format}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def plot_win_rate(
    rows: list[Row],
    out_dir: Path,
    *,
    algorithm: str,
    time_limit_ms: int,
    out_format: str,
    dpi: int,
) -> Path:
    # For each instance, determine the best neighborhood (min gap). Count wins.
    # Tie handling: split point evenly between tied best.
    by_instance: DefaultDict[tuple[str, int], list[Row]] = defaultdict(list)

    for r in rows:
        if r.algorithm != algorithm or r.time_limit_ms != time_limit_ms:
            continue
        if r.neighborhood not in NEIGHBOR_LABEL:
            continue
        key = (r.instance_file, r.instance_number)
        by_instance[key].append(r)

    scores = {neigh: 0.0 for neigh in NEIGHBOR_ORDER}

    for _, group in by_instance.items():
        best = min(g.gap_percent for g in group)
        winners = [g.neighborhood for g in group if abs(g.gap_percent - best) < 1e-12]
        if not winners:
            continue
        share = 1.0 / len(winners)
        for w in winners:
            scores[w] += share

    labels = [NEIGHBOR_LABEL[n] for n in NEIGHBOR_ORDER]
    vals = [scores[n] for n in NEIGHBOR_ORDER]

    plt.figure(figsize=(6.0, 3.2))
    plt.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.ylabel("Win score (ties split)")
    plt.title(f"{ALGO_LABEL.get(algorithm, algorithm)}: win-rate @ {time_limit_ms} ms")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = out_dir / f"win_rate_{algorithm}_{time_limit_ms}ms.{out_format}"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--out-root", default="figures/paper")
    parser.add_argument("--win-tl", type=int, default=5000)
    parser.add_argument(
        "--compare-root",
        default="results",
        help="Root folder that contains `tabu_search_compare_*` directories.",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Also generate convergence plots from `*_search_compare_*` folders.",
    )
    parser.add_argument(
        "--convergence-instances",
        default="tai200_20:0,tai500_20:2",
        help=(
            "Comma-separated instance selectors as file:inst (e.g. tai200_20:0). "
            "Used to locate `tabu_search_compare_<file>_instance<inst>` folders."
        ),
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output format for figures (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output (PNG). Ignored by some backends for PDF.",
    )
    args = parser.parse_args()

    run_dir = Path(args.results_root) / args.run_id
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.csv: {summary_path}")

    out_dir = Path(args.out_root) / args.run_id
    ensure_dir(out_dir)

    rows = read_summary_csv(summary_path)

    created: list[Path] = []
    for algo in ("tabu", "sa"):
        created.append(
            plot_gap_vs_time_limit(
                rows,
                out_dir,
                algorithm=algo,
                out_format=args.format,
                dpi=args.dpi,
            )
        )

    if args.convergence:
        compare_root = Path(args.compare_root)
        selectors = [s.strip() for s in str(args.convergence_instances).split(",") if s.strip()]
        for sel in selectors:
            if ":" not in sel:
                raise SystemExit(
                    f"Bad selector '{sel}'. Expected format like 'tai200_20:0' (file:inst)."
                )
            instance_file, inst_str = sel.split(":", 1)
            folder = compare_root / f"tabu_search_compare_{instance_file}_instance{inst_str}"
            if not folder.exists():
                raise SystemExit(f"Missing compare folder: {folder}")
            created.append(
                plot_convergence_from_compare_dir(
                    folder,
                    out_dir,
                    title=f"Tabu Search convergence: {instance_file}, instance {inst_str}",
                    out_format=args.format,
                    dpi=args.dpi,
                )
            )
        created.append(
            plot_win_rate(
                rows,
                out_dir,
                algorithm=algo,
                time_limit_ms=args.win_tl,
                out_format=args.format,
                dpi=args.dpi,
            )
        )

    # Write a small manifest for convenience
    manifest = out_dir / "manifest.txt"
    with manifest.open("w") as f:
        for p in created:
            f.write(str(p) + os.linesep)

    print(f"Wrote {len(created)} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
