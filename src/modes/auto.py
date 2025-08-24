"""Auto mode execution logic.

Runs each algorithm (Hill Climb, Tabu Search, Simulated Annealing)
independently ``runs`` times from fresh random permutations. Best run
statistics and simple aggregates are persisted as JSON while per‑algorithm
best schedules may be rendered to Gantt charts.

This is intentionally decoupled from the CLI entry point to keep ``main.py``
small and enable re-use in notebooks or higher level experiment drivers.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import random

from src.decoder import (
    build_schedule_from_permutation,
    create_random_permutation,
)
from src.visualization import plot_gantt
from src.models import DataInstance
from .common import AlgoParams, run_algorithm

logger = logging.getLogger("jssp.auto")


def run_auto(
    instance: DataInstance,
    instance_path: str,
    runs: int,
    params: AlgoParams,
    rng: random.Random,
    charts_dir: str,
) -> tuple[Optional[list[tuple[int, int]]], Optional[int]]:
    """Run independent multi-start experiments for all algorithms.

    Args:
        instance: Parsed JSSP instance.
        instance_path: Original path (stored in JSON output for traceability).
        runs: Number of independent starts per algorithm.
        params: Shared algorithm hyper-parameters.
        rng: Random generator used for permutation sampling & stochastic moves.
        charts_dir: Directory for output artefacts (created if missing).

    Returns:
    Tuple ``(best_perm, best_cmax)`` across all algorithms or
    ``(None, None)`` when no candidate produced (should not happen unless
    ``runs==0``).
    """
    algo_names = ("hill", "tabu", "sa")
    stats: Dict[str, Dict] = {
        name: {
            "best_c": None,
            "best_perm": None,
            "best_time": None,
            "c": [],
            "t": [],
            "progress": [],
            "time_progress": [],
        }
        for name in algo_names
    }
    for i in range(1, runs + 1):
        start_perm = create_random_permutation(instance, rng=rng)
        for name in algo_names:
            perm, c, elapsed = run_algorithm(
                name,
                instance,
                start_perm,
                params,
                rng,
                progress=stats[name]["progress"],
                time_progress=stats[name]["time_progress"],
            )
            stats[name]["c"].append(c)
            stats[name]["t"].append(elapsed)
            if (
                stats[name]["best_c"] is None
                or c < stats[name]["best_c"]  # type: ignore[operator]
            ):
                stats[name].update(
                    {"best_c": c, "best_perm": perm, "best_time": elapsed}
                )
        if i % max(1, runs // 10) == 0:
            logger.info(
                "Progress %d/%d: HC=%s Tabu=%s SA=%s",
                i,
                runs,
                stats["hill"]["best_c"],
                stats["tabu"]["best_c"],
                stats["sa"]["best_c"],
            )
    
    def _avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")
    for name in algo_names:
        logger.info(
            "Auto summary %-8s: best=%s (%.4fs) avg=%.2f (%.4fs)",
            name,
            stats[name]["best_c"],
            stats[name]["best_time"],
            _avg([float(v) for v in stats[name]["c"]]),
            _avg(stats[name]["t"]),
        )
    candidates = [
        (name, stats[name]["best_c"], stats[name]["best_perm"])
        for name in algo_names
        if stats[name]["best_c"] is not None
        and stats[name]["best_perm"] is not None
    ]
    best_perm = None
    best_c = None
    if candidates:
        best_algo_name, best_c, best_perm = min(
            candidates, key=lambda x: x[1]
        )  # type: ignore
        logger.info(
            "Overall best algorithm=%s cmax=%s", best_algo_name, best_c
        )
    auto_best = {k: stats[k]["best_perm"] for k in algo_names}
    try:
        def _perm_to_list(
            perm: Optional[list[tuple[int, int]]],
        ) -> Optional[list[list[int]]]:
            if perm is None:
                return None
            return [[int(p[0]), int(p[1])] for p in perm]

        def _perm_compact(
            perm: Optional[list[tuple[int, int]]]
        ) -> Optional[str]:
            if perm is None:
                return None
            return ",".join(f"J{p[0]}O{p[1]}" for p in perm)

        def _job_sequence(
            perm: Optional[list[tuple[int, int]]],
        ) -> Optional[list[int]]:
            if perm is None:
                return None
            return [int(p[0]) for p in perm]

        stamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(charts_dir, f"auto_results_{stamp2}.json")
        per_run = {"hill": [], "tabu": [], "sa": []}
        for algo in algo_names:
            c_list = stats[algo]["c"]
            t_list = stats[algo]["t"]
            for idx in range(len(c_list)):
                per_run[algo].append(
                    {"run": idx + 1, "cmax": c_list[idx], "time": t_list[idx]}
                )
        best_block = {}
        for algo in algo_names:
            best_block[algo] = {
                "cmax": stats[algo]["best_c"],
                "time": stats[algo]["best_time"],
                "permutation_pairs": _perm_to_list(stats[algo]["best_perm"]),
                "permutation_compact": _perm_compact(stats[algo]["best_perm"]),
                "job_sequence": _job_sequence(stats[algo]["best_perm"]),
            }
        averages_block = {}
        for algo in algo_names:
            c_vals = stats[algo]["c"]
            t_vals = stats[algo]["t"]
            averages_block[algo] = {
                "avg_cmax": (sum(c_vals) / len(c_vals) if c_vals else None),
                "avg_time": (sum(t_vals) / len(t_vals) if t_vals else None),
            }
        json_payload = {
            "instance": instance_path,
            "runs": runs,
            "timestamp": stamp2,
            "per_run": per_run,
            "best": best_block,
            "averages": averages_block,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved auto mode results JSON to %s", results_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to write results JSON: %s", e)
    try:
        stamp3 = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, perm in auto_best.items():
            if perm is None:
                continue
            sched_best = build_schedule_from_permutation(
                instance, perm, check_completeness=True
            )
            g_path = os.path.join(
                charts_dir, f"gantt_{name}_c{sched_best.cmax}_{stamp3}.png"
            )
            plot_gantt(sched_best, save_path=g_path, algo_name=name)
            logger.info("Saved Gantt chart for %s to %s", name, g_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create Gantt charts: %s", e)
    return best_perm, best_c
