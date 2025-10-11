import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # Must be set before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402


def save_gantt_chart_with_name(
    pi: List[int],
    processing_times: List[List[int]],
    cmax: int,
    name: str,
    show_legend: Optional[bool] = None,
):
    """Create and save Gantt chart for the given permutation with custom filename.

    Improvements:
    - Uses constrained_layout to reduce layout warnings.
    - Disables legend automatically for large n unless forced.
    - Adaptive figure size based on number of machines and jobs.
    """
    m = len(processing_times)
    n = len(pi)
    start_times, completion_times = calculate_schedule(pi, processing_times)

    # Adaptive sizing: width grows slowly with jobs, height with machines
    base_w, base_h = 10, 0.5 * m + 2
    fig, ax = plt.subplots(
        figsize=(min(base_w + n * 0.05, 18), min(base_h, 16)),
        constrained_layout=True,
    )
    cmap = plt.cm.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n)]
    for i in range(m):
        for j in range(n):
            job_id = pi[j]
            start = start_times[i][j]
            duration = processing_times[i][job_id]
            ax.barh(
                i,
                duration,
                left=start,
                height=0.8,
                color=colors[job_id],
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
            )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Machine", fontsize=12)
    ax.set_title(f"Gantt Chart - Cmax = {cmax}", fontsize=14, fontweight="bold")
    ax.set_yticks(range(m))
    ax.set_yticklabels([f"M{i}" for i in range(m)])
    ax.grid(True, alpha=0.25, axis="x", linestyle="--", linewidth=0.7)
    ax.set_ylim(-0.5, m - 0.5)

    # Legend logic
    if show_legend is None:
        # auto policy: only show when jobs <= 40
        show_legend = n <= 40
    if show_legend:
        legend_elements = [
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=colors[i], alpha=0.85, edgecolor="black", label=f"Job {i}"
            )
            for i in range(n)
        ]
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
            frameon=False,
            ncol=1 if n <= 25 else 2 if n <= 50 else 3,
        )

    # If `name` is a full path (has dir), use it; otherwise save under results folder
    if os.path.dirname(name):
        filepath = name
    else:
        filepath = os.path.join("results", name)
    _ensure_dir(os.path.dirname(filepath) or "results")
    fig.savefig(filepath, dpi=180)  # dpi 180 dla kompromisu jakości/czasu
    plt.close(fig)
    print(f"Gantt chart saved as: {filepath}")


def save_multi_convergence_plot(
    histories: dict,
    labels: dict = None,
    colors: dict = None,
    filepath: str = None,
    results_folder: str = "results",
    time_limit_ms: int = None,
):
    """Draw comparison of several convergence histories on one plot and save it.

    Uses constrained_layout to avoid tight_layout warnings and places legend outside.
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if labels is None:
        labels = {}
    if colors is None:
        colors = {}
    for key, (times, cmax_values) in histories.items():
        # Defensive copy to avoid mutating original history structures
        times_local = list(times)
        cmax_local = list(cmax_values)
        # If we have a declared time_limit_ms and the last recorded timestamp is earlier,
        # append a flat segment so that all curves visually reach the same horizon.
        if time_limit_ms is not None and times_local:
            if times_local[-1] < time_limit_ms:
                times_local.append(time_limit_ms)
                cmax_local.append(cmax_local[-1])
        elif time_limit_ms is not None and not times_local:
            # Edge case: empty history (should not happen) – synthesize a flat line
            times_local = [0, time_limit_ms]
            cmax_local = [0, 0]
        label = labels.get(key, key)
        color = colors.get(key, None)
        ax.plot(
            times_local,
            cmax_local,
            label=label,
            linewidth=2,
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            color=color,
        )
        if times_local and cmax_local:
            ax.annotate(
                f"{cmax_local[-1]}",
                xy=(times_local[-1], cmax_local[-1]),
                xytext=(6, -10),
                textcoords="offset points",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.55),
            )
    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_ylabel("Cmax", fontsize=12)
    ax.set_title("Convergence comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    if time_limit_ms is not None:
        try:
            ax.axvline(x=time_limit_ms, color="red", linestyle="--", linewidth=1.2, zorder=5)
            ax.text(
                time_limit_ms,
                ax.get_ylim()[1],
                " time limit",
                color="red",
                fontsize=9,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.6),
            )
        except Exception:
            pass
    # Legend outside on the right
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_convergence_plot_{timestamp}.png"
        filepath = os.path.join(results_folder, filename)
    else:
        _ensure_dir(os.path.dirname(filepath) or results_folder)
    fig.savefig(filepath, dpi=180)
    plt.close(fig)
    print(f"Multi convergence plot saved as: {filepath}")


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def clear_old_plots(results_folder: str = "results"):
    """Remove old plot files from the specified results directory."""
    results_dir = results_folder
    if os.path.exists(results_dir):
        # Remove old gantt charts
        for file in glob.glob(os.path.join(results_dir, "gantt_chart_*.png")):
            os.remove(file)
        # Remove old convergence plots
        for file in glob.glob(os.path.join(results_dir, "convergence_plot_*.png")):
            os.remove(file)
        # Remove old multi convergence plots
        for file in glob.glob(os.path.join(results_dir, "multi_convergence_plot_*.png")):
            os.remove(file)


def calculate_schedule(pi: List[int], processing_times: List[List[int]]):
    """Calculate start and completion times for each job on each machine."""
    m = len(processing_times)  # number of machines
    n = len(pi)  # number of jobs

    # Initialize start and completion time matrices
    start_times = [[0 for _ in range(n)] for _ in range(m)]
    completion_times = [[0 for _ in range(n)] for _ in range(m)]

    # First job on first machine
    start_times[0][0] = 0
    completion_times[0][0] = processing_times[0][pi[0]]

    # First machine, remaining jobs
    for j in range(1, n):
        start_times[0][j] = completion_times[0][j - 1]
        completion_times[0][j] = start_times[0][j] + processing_times[0][pi[j]]

    # First job, remaining machines
    for i in range(1, m):
        start_times[i][0] = completion_times[i - 1][0]
        completion_times[i][0] = start_times[i][0] + processing_times[i][pi[0]]

    # Remaining jobs and machines
    for i in range(1, m):
        for j in range(1, n):
            start_times[i][j] = max(completion_times[i - 1][j], completion_times[i][j - 1])
            completion_times[i][j] = start_times[i][j] + processing_times[i][pi[j]]

    return start_times, completion_times


def save_convergence_plot_to(iterations: List[int], cmax_values: List[int], filepath: str):
    """Save convergence plot to a specific filepath."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        iterations,
        cmax_values,
        "b-o",
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=2,
    )

    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_ylabel("Cmax", fontsize=12)
    ax.set_title("Convergence", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(iterations) > 1:
        ax.annotate(
            f"Start: {cmax_values[0]}",
            xy=(iterations[0], cmax_values[0]),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        ax.annotate(
            f"Best: {cmax_values[-1]}",
            xy=(iterations[-1], cmax_values[-1]),
            xytext=(10, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    plt.tight_layout()
    _ensure_dir(os.path.dirname(filepath) or "results")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved as: {filepath}")


def build_algorithm_multi_convergence_plots(timestamp_dir: str | Path) -> List[Path]:
    """Zbuduj wykresy konwergencji per algorytm z trzema sąsiedztwami równocześnie.

    Szuka JSONów w katalogu `timestamp_dir` (pojedynczy eksperyment) i dla każdej trójki
    (algorithm, instance_number, time_limit_ms) wybiera jeden seed per neighborhood:
      - preferencyjnie seed = 0; jeśli brak, najniższy dostępny.

    Zapisuje pliki: algo={alg}_inst={inst}_tl={limit}ms_multi.png w tym katalogu.
    Zwraca listę ścieżek wygenerowanych plików.
    """
    td = Path(timestamp_dir)
    if not td.exists():
        return []
    json_files = list(td.glob("*.json"))
    if not json_files:
        return []

    # indeks -> (alg, inst, tl, neigh, seed) : path oraz meta per (alg, inst, tl)
    index = {}
    meta_dims = {}  # (alg, inst, tl) -> (jobs, machines, instance_file_stem)
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            alg = cfg.get("algorithm")
            neigh = cfg.get("neighborhood")
            inst = cfg.get("instance_number")
            seed = cfg.get("seed")
            tl = cfg.get("time_limit_ms") or cfg.get("time_limit") or data.get("time_limit_ms")
            if None in (alg, neigh, inst, seed, tl):
                continue
            index[(alg, inst, tl, neigh, seed)] = jf
            # Wyciągnij wymiary tylko raz
            key_meta = (alg, inst, tl)
            if key_meta not in meta_dims:
                jobs = data.get("instance_jobs") or data.get("jobs")
                machines = data.get("instance_machines") or data.get("machines")
                inst_file = cfg.get("instance_file") or ""
                inst_stem = Path(inst_file).stem if inst_file else "inst"
                meta_dims[key_meta] = (jobs, machines, inst_stem)
        except Exception:
            continue

    if not index:
        return []

    seeds_per = defaultdict(list)
    for alg, inst, tl, neigh, seed in index.keys():
        seeds_per[(alg, inst, tl, neigh)].append(seed)

    neigh_order = ["adjacent", "fibonahi_neighborhood", "dynasearch_neighborhood"]
    base_colors = {
        "adjacent": "#00FFFF",
        "fibonahi_neighborhood": "#FF00CC",
        "dynasearch_neighborhood": "#7CFF00",
    }
    group_keys = sorted({(k[0], k[1], k[2]) for k in index.keys()})
    outputs: List[Path] = []
    for alg, inst, tl in group_keys:
        histories = {}
        labels = {}
        colors = {}
        any_curve = False
        for neigh in neigh_order:
            seeds = seeds_per.get((alg, inst, tl, neigh), [])
            if not seeds:
                continue
            selected_seed = 0 if 0 in seeds else min(seeds)
            json_path = index.get((alg, inst, tl, neigh, selected_seed))
            if not json_path:
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                times = data.get("time_history_ms") or []
                cmax_hist = data.get("cmax_history") or []
                if times and cmax_hist:
                    histories[neigh] = (times, cmax_hist)
                    labels[neigh] = f"{alg.upper()} {neigh} (seed {selected_seed})"
                    colors[neigh] = base_colors.get(neigh)
                    any_curve = True
            except Exception:
                continue
        if any_curve:
            jobs, machines, inst_stem = meta_dims.get((alg, inst, tl), (None, None, "inst"))
            dim_part = ""
            if jobs is not None and machines is not None:
                dim_part = f"_n{jobs}_m{machines}"
            base_name = (
                f"algo={alg}_file={inst_stem}_inst={inst}_tl={int(tl)}ms" f"{dim_part}_multi.png"
            )
            out_path = Path(next_unique_path(td / base_name))
            try:
                save_multi_convergence_plot(
                    histories,
                    labels=labels,
                    colors=colors,
                    filepath=str(out_path),
                    time_limit_ms=int(tl),
                )
                outputs.append(out_path)
            except Exception:
                continue
    print(f"[Visualization] Generated {len(outputs)} per-algorithm multi convergence plots.")
    return outputs


def next_unique_path(path: str | Path) -> str:
    """Jeżeli plik istnieje, dodaj sufiks _1, _2 ... aż do znalezienia wolnej nazwy.

    Zwraca ścieżkę jako string.
    """
    p = Path(path)
    if not p.exists():
        return str(p)
    stem = p.stem
    suffix = p.suffix
    parent = p.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return str(candidate)
        counter += 1
