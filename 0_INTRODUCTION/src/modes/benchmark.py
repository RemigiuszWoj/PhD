"""Benchmark (batch) mode logic.

Executes a random sample of Taillard (``ta*``) instance files and for each
instance runs every algorithm (Hill, Tabu, SA) ``runs`` times from an
independent random permutation. Produces per‑algorithm incremental JSON
snapshots (fault tolerant), final summary JSON, individual progress & Gantt
plots, and an aggregated progress plot per instance combining the best curve
of each algorithm.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime

from src.decoder import build_schedule_from_permutation, create_random_permutation
from src.models import DataInstance
from src.modes.common import AlgoParams, run_algorithm
from src.parser import load_instance
from src.visualization import plot_gantt, plot_progress_curves

logger = logging.getLogger("jssp.benchmark")


def _perm_inline(perm: list[tuple[int, int]] | None) -> str | None:
    """Return a compact string representation of a permutation.

    Format: ``[[j0,o0],[j1,o1],...]`` used only in benchmark JSON to keep file
    size modest while still being trivially parseable.
    """
    if perm is None:
        return None
    return "[" + ",".join(f"[{p[0]},{p[1]}]" for p in perm) + "]"


def _collect_instance_files(instances_dir: str, benchmark_sample: int) -> list[str]:
    """Collect candidate Taillard instance file paths.

    Args:
        instances_dir: Directory containing raw instance text files.
        benchmark_sample: Maximum number of random files to sample (ceil).

    Returns:
        List of absolute file paths (possibly shuffled sample). Empty list
        when directory missing or no matching files.
    """
    if not os.path.isdir(instances_dir):
        logger.error("Instances directory not found: %s", instances_dir)
        return []
    files = [
        os.path.join(instances_dir, f)
        for f in sorted(os.listdir(instances_dir))
        if os.path.isfile(os.path.join(instances_dir, f)) and not f.startswith(".")
    ]
    files = [p for p in files if os.path.basename(p).startswith("ta")]
    if not files:
        logger.error("No Taillard (ta*) instance files found in %s", instances_dir)
        return []
    k = min(max(1, benchmark_sample), len(files))
    if k < len(files):
        files = random.sample(files, k)
        logger.info("Benchmark sampling %d random instances", k)
    return files


def run_benchmark(
    instances_dir: str,
    benchmark_dir: str,
    benchmark_sample: int,
    runs: int,
    params: AlgoParams,
    rng: random.Random,
) -> None:
    """Execute batch benchmark across a sample of Taillard instances.

    Side-effects: builds directory tree rooted at ``benchmark_dir`` with one
    subfolder per instance, containing per‑algorithm artefacts and a combined
    progress figure.
    """
    instance_files = _collect_instance_files(instances_dir, benchmark_sample)
    if not instance_files:
        return
    algos = ["hill", "tabu", "sa"]
    logger.info(
        "Benchmark: %d instances, algorithms=%s, runs per algo=%d",
        len(instance_files),
        ",".join(algos),
        runs,
    )
    for inst_path in instance_files:
        try:
            data_inst: DataInstance = load_instance(inst_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load %s: %s", inst_path, e)
            continue
        inst_name = os.path.splitext(os.path.basename(inst_path))[0]
        logger.info(
            "Instance %s: jobs=%d machines=%d",
            inst_name,
            data_inst.jobs_number,
            data_inst.machines_number,
        )
        inst_progress: dict[str, list[int]] = {}
        inst_time_progress: dict[str, list[float]] = {}
        for algo_name in algos:
            out_dir = os.path.join(benchmark_dir, inst_name, algo_name)
            os.makedirs(out_dir, exist_ok=True)
            # Clean old artefacts (JSON / PNG) before new runs
            for fname in os.listdir(out_dir):
                if fname.endswith(".json") or fname.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(out_dir, fname))
                    except OSError:
                        pass
            results_c: list[int] = []
            results_t: list[float] = []
            best_c = None
            best_time = None
            best_perm = None
            best_progress: list[int] = []
            best_time_progress: list[float] = []
            per_run_records: list[dict] = []
            incr_path = os.path.join(out_dir, f"results_incremental_{algo_name}.json")
            algo_start_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for r_i in range(1, runs + 1):
                start_perm = create_random_permutation(data_inst, rng=rng)
                progress_list: list[int] = []
                time_list: list[float] = []
                perm, c, elapsed = run_algorithm(
                    algo_name,
                    data_inst,
                    start_perm,
                    params,
                    rng,
                    progress=progress_list,
                    time_progress=time_list,
                )
                results_c.append(c)
                results_t.append(elapsed)
                if best_c is None or c < best_c:
                    best_c = c
                    best_time = elapsed
                    best_perm = perm
                    best_progress = list(progress_list)
                    best_time_progress = list(time_list)
                per_run_records.append({"run": r_i, "cmax": c, "time": elapsed})
                # incremental JSON
                try:
                    avg_c_inc = sum(results_c) / len(results_c)
                    avg_t_inc = sum(results_t) / len(results_t)
                    incr_payload = {
                        "instance": inst_path,
                        "instance_name": inst_name,
                        "algorithm": algo_name,
                        "timestamp_start": algo_start_stamp,
                        "runs_completed": r_i,
                        "planned_runs": runs,
                        "per_run": per_run_records,
                        "best": {
                            "cmax": best_c,
                            "time": best_time,
                            "permutation": _perm_inline(best_perm),
                        },
                        "average": {
                            "avg_cmax": avg_c_inc,
                            "avg_time": avg_t_inc,
                        },
                    }
                    tmp_path = incr_path + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f_inc:
                        json.dump(incr_payload, f_inc, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, incr_path)
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "Incremental JSON save failed %s run %d: %s",
                        algo_name,
                        r_i,
                        e,
                    )
                if r_i % max(1, runs // 10) == 0:
                    logger.info(
                        "%s %s progress %d/%d best_c=%s",
                        inst_name,
                        algo_name,
                        r_i,
                        runs,
                        best_c,
                    )
            # Wykres progresu
            try:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prog_path = os.path.join(out_dir, f"progress_{algo_name}_{stamp}.png")
                plot_progress_curves(
                    {algo_name: best_progress},
                    {algo_name: best_time_progress},
                    save_path=prog_path,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("Progress plot failed %s %s", algo_name, e)
            # Gantt
            try:
                if best_perm is not None:
                    sched_best = build_schedule_from_permutation(
                        data_inst, best_perm, check_completeness=True
                    )
                    g_path = os.path.join(out_dir, f"gantt_{algo_name}_c{sched_best.cmax}.png")
                    plot_gantt(
                        sched_best,
                        save_path=g_path,
                        algo_name=algo_name,
                    )
            except Exception as e:  # pragma: no cover
                logger.warning("Gantt failed %s %s", algo_name, e)
            # Final JSON
            try:
                stamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_path = os.path.join(out_dir, f"results_{algo_name}_{stamp2}.json")
                per_run = [
                    {
                        "run": i + 1,
                        "cmax": results_c[i],
                        "time": results_t[i],
                    }
                    for i in range(len(results_c))
                ]
                payload = {
                    "instance": inst_path,
                    "instance_name": inst_name,
                    "algorithm": algo_name,
                    "runs": runs,
                    "timestamp": stamp2,
                    "per_run": per_run,
                    "best": {
                        "cmax": best_c,
                        "time": best_time,
                        "permutation": _perm_inline(best_perm),
                    },
                    "average": {
                        "avg_cmax": sum(results_c) / len(results_c) if results_c else None,
                        "avg_time": sum(results_t) / len(results_t) if results_t else None,
                    },
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            except Exception as e:  # pragma: no cover
                logger.warning("JSON save failed %s %s", algo_name, e)
            inst_progress[algo_name] = best_progress
            inst_time_progress[algo_name] = best_time_progress
        # Combined plot
        try:
            inst_root = os.path.join(benchmark_dir, inst_name)
            for fname in os.listdir(inst_root):
                if fname.startswith("progress_all_") and fname.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(inst_root, fname))
                    except OSError:
                        pass
            stamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_path = os.path.join(inst_root, f"progress_all_{stamp_all}.png")
            plot_progress_curves(
                inst_progress,
                inst_time_progress,
                save_path=combined_path,
            )
            logger.info(
                "Saved combined progress plot for %s to %s",
                inst_name,
                combined_path,
            )
        except Exception as e:  # pragma: no cover
            logger.warning(
                "Combined progress plot failed for %s: %s",
                inst_name,
                e,
            )
