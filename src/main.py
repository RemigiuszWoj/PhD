"""CLI entry-point for Job Shop metaheuristic experiments.

Supported modes (``--algo``):

* ``demo``       – tiny neighborhood sampling then HC -> Tabu -> SA pipeline
* ``hill`` ``tabu`` ``sa`` – run a single algorithm (Tabu warms up via HC)
* ``pipeline``   – multi-start random -> HC -> Tabu -> SA keep best SA run
* ``auto``       – independent multi-start comparison of all algorithms
* ``benchmark``  – batch over a random subset of Taillard instances

The heavy per-mode logic (auto / benchmark) lives in ``src.modes`` modules to
keep this file concise; only lightweight single-instance logic remains here.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime

from src.decoder import (
    build_schedule_from_permutation,
    check_no_machine_overlap,
    create_random_permutation,
    validate_permutation,
)
from src.models import DataInstance
from src.modes.auto import run_auto
from src.modes.benchmark import run_benchmark
from src.modes.common import AlgoParams
from src.neighborhood import generate_neighbors
from src.operations import create_base_permutation as create_spt_permutation  # fallback
from src.parser import load_instance
from src.search import evaluate, hill_climb, simulated_annealing, tabu_search
from src.visualization import plot_gantt

logger = logging.getLogger("jssp")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# Lokalna definicja AlgoParams/run_algorithm przeniesiona do modes.common


def main(
    instance_path: str,
    algo: str,
    seed: int | None,
    neighbor_limit: int,
    max_no_improve: int,
    tabu_iterations: int,
    tabu_tenure: int,
    tabu_candidate_size: int,
    sa_iterations: int,
    sa_initial_temp: float,
    sa_cooling: float,
    sa_neighbor_moves: int,
    pipeline_runs: int,
    gantt: bool,
    gantt_path: str | None,
    runs: int,
    charts_dir: str,
    instances_dir: str = "data/JSPLIB/instances",
    benchmark_dir: str = "research",
    benchmark_sample: int = 5,
) -> None:  # noqa: D401
    """Execute selected experiment mode for the JSSP metaheuristics.

    Modes:
        demo        : Sample a tiny neighborhood of SPT then HC->Tabu->SA.
        hill/tabu/sa: Run a single algorithm (Tabu starts from HC result).
        pipeline    : Multi-start HC->Tabu->SA chain keep best SA.
        auto        : Independent multi-start comparison (one instance).
        benchmark   : Batch over Taillard ta* set; per instance run each
                      algorithm N times, write per‑algo dirs + combined
                      progress plot.

    Args:
        instance_path: Path to instance file (or directory) for non-
            benchmark modes.
        algo: One of {demo,hill,tabu,sa,pipeline,auto,benchmark}.
        seed: RNG seed (affects permutation sampling and stochastic moves).
        neighbor_limit: Sampled neighbors per hill climb iteration.
        max_no_improve: Early stop threshold for hill climb (consecutive
            non‑improvements).
        tabu_iterations: Tabu search iterations.
        tabu_tenure: Tabu tenure (iterations a move stays forbidden).
        tabu_candidate_size: Swap candidates sampled per tabu iteration.
        sa_iterations: Simulated annealing iterations.
        sa_initial_temp: Initial temperature for SA.
        sa_cooling: Geometric cooling factor (0 < factor < 1).
        sa_neighbor_moves: Neighbor trials per SA iteration (best kept).
        pipeline_runs: Multi-start repetitions of HC->Tabu->SA (pipeline).
        gantt: Whether to render a final Gantt chart (auto mode overrides).
        gantt_path: Explicit output path for single Gantt if provided.
        runs: Independent runs per algorithm (auto / benchmark modes).
        charts_dir: Directory to store charts for single-instance modes.

    Returns:
        None. Side-effects: logging + JSON + PNG artifacts on disk.
    """
    # If invoked directly with no extra CLI args (handled in __main__) we may
    # want to ensure charts; the __main__ block already forces gantt=True.

    instance: DataInstance = load_instance(instance_path)
    logger.info(
        "Instance loaded: jobs=%d machines=%d ops=%d",
        instance.jobs_number,
        instance.machines_number,
        instance.jobs_number * instance.machines_number,
    )

    # Initial heuristic schedules for baseline comparison
    base_schedule = build_schedule_from_permutation(
        instance,
        create_spt_permutation(instance),  # baseline heuristic
        check_completeness=True,
    )
    check_no_machine_overlap(base_schedule)
    logger.info("Base schedule cmax=%s", base_schedule.cmax)

    rng = random.Random(seed) if seed is not None else random.Random()
    rand_perm = create_random_permutation(instance, rng=rng)
    validate_permutation(instance, rand_perm)
    rand_schedule = build_schedule_from_permutation(
        instance,
        rand_perm,
        check_completeness=True,
    )
    check_no_machine_overlap(rand_schedule)
    logger.info("Random schedule cmax=%s", rand_schedule.cmax)

    spt_perm = create_spt_permutation(instance)
    validate_permutation(instance, spt_perm)
    spt_schedule = build_schedule_from_permutation(
        instance,
        spt_perm,
        check_completeness=True,
    )
    check_no_machine_overlap(spt_schedule)
    logger.info("SPT schedule cmax=%s", spt_schedule.cmax)

    best = min(
        [
            ("base", base_schedule.cmax),
            ("random", rand_schedule.cmax),
            ("spt", spt_schedule.cmax),
        ],
        key=lambda x: x[1],
    )
    logger.info("Best initial heuristic: %s", best)

    final_perm = None
    final_c = None

    # --- Modes ---
    if algo == "benchmark":
        params = AlgoParams(
            neighbor_limit=neighbor_limit,
            max_no_improve=max_no_improve,
            tabu_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            tabu_candidate_size=tabu_candidate_size,
            sa_iterations=sa_iterations,
            sa_initial_temp=sa_initial_temp,
            sa_cooling=sa_cooling,
            sa_neighbor_moves=sa_neighbor_moves,
        )
        run_benchmark(
            instances_dir=instances_dir,
            benchmark_dir=benchmark_dir,
            benchmark_sample=benchmark_sample,
            runs=runs,
            params=params,
            rng=rng,
        )
        return
    if algo == "demo":
        neighbors = generate_neighbors(spt_perm, limit=5, rng=rng)
        best_c = None
        for p in neighbors:
            c = evaluate(instance, p)
            if best_c is None or c < best_c:
                best_c = c
        if best_c is not None:
            logger.info("Best neighbor (from SPT) cmax: %s", best_c)
        best_perm, best_cmax, iters, evals = hill_climb(
            instance,
            spt_perm,
            neighbor_limit=neighbor_limit,
            max_no_improve=max_no_improve,
            best_improvement=True,
            rng=rng,
        )
        logger.info(
            "HillClimb from SPT: cmax=%s iters=%s evals=%s",
            best_cmax,
            iters,
            evals,
        )
        tabu_best_perm, tabu_best_c, tabu_evals = tabu_search(
            instance,
            best_perm,
            iterations=tabu_iterations,
            tenure=tabu_tenure,
            candidate_size=tabu_candidate_size,
            rng=rng,
        )
        logger.info(
            "Tabu    from HillClimb: cmax=%s evals=%s",
            tabu_best_c,
            tabu_evals,
        )
        sa_best_perm, sa_best_c, sa_evals, sa_iters = simulated_annealing(
            instance,
            tabu_best_perm,
            iterations=sa_iterations,
            initial_temp=sa_initial_temp,
            cooling=sa_cooling,
            neighbor_moves=sa_neighbor_moves,
            rng=rng,
        )
        logger.info(
            "SimAnn  from Tabu: cmax=%s iters=%s evals=%s",
            sa_best_c,
            sa_iters,
            sa_evals,
        )
        final_perm = sa_best_perm
        final_c = sa_best_c
    elif algo == "hill":
        best_perm, best_cmax, iters, evals = hill_climb(
            instance,
            spt_perm,
            neighbor_limit=neighbor_limit,
            max_no_improve=max_no_improve,
            best_improvement=True,
            rng=rng,
        )
        logger.info(
            "HillClimb: cmax=%s iters=%s evals=%s",
            best_cmax,
            iters,
            evals,
        )
        final_perm = best_perm
        final_c = best_cmax
    elif algo == "tabu":
        hc_perm, _, _, _ = hill_climb(
            instance,
            spt_perm,
            neighbor_limit=neighbor_limit,
            max_no_improve=max_no_improve,
            best_improvement=True,
            rng=rng,
        )
        best_perm, best_c, evals = tabu_search(
            instance,
            hc_perm,
            iterations=tabu_iterations,
            tenure=tabu_tenure,
            candidate_size=tabu_candidate_size,
            rng=rng,
        )
        logger.info("Tabu: cmax=%s evals=%s", best_c, evals)
        final_perm = best_perm
        final_c = best_c
    elif algo == "sa":
        best_perm, best_c, evals, iters = simulated_annealing(
            instance,
            spt_perm,
            iterations=sa_iterations,
            initial_temp=sa_initial_temp,
            cooling=sa_cooling,
            neighbor_moves=sa_neighbor_moves,
            rng=rng,
        )
        logger.info(
            "SimAnn: cmax=%s iters=%s evals=%s",
            best_c,
            iters,
            evals,
        )
        final_perm = best_perm
        final_c = best_c
    elif algo == "pipeline":
        ms_results = []
        best_sa_perm = None
        for _ in range(pipeline_runs):
            sp = create_random_permutation(instance, rng=rng)
            hc_perm, hc_c, _, _ = hill_climb(
                instance,
                sp,
                neighbor_limit=neighbor_limit,
                max_no_improve=max_no_improve,
                best_improvement=True,
                rng=rng,
            )
            tb_perm, tb_c, _ = tabu_search(
                instance,
                hc_perm,
                iterations=tabu_iterations,
                tenure=tabu_tenure,
                candidate_size=tabu_candidate_size,
                rng=rng,
            )
            sa_perm, sa_c, _, _ = simulated_annealing(
                instance,
                tb_perm,
                iterations=sa_iterations,
                initial_temp=sa_initial_temp,
                cooling=sa_cooling,
                neighbor_moves=sa_neighbor_moves,
                rng=rng,
            )
            ms_results.append((hc_c, tb_c, sa_c))
            if best_sa_perm is None or final_c is None or sa_c < final_c:
                best_sa_perm = sa_perm
                final_c = sa_c
        best_hc = min(r[0] for r in ms_results)
        best_tb = min(r[1] for r in ms_results)
        best_sa = min(r[2] for r in ms_results)
        logger.info(
            "Pipeline summary: best HC=%s best Tabu=%s best SA=%s",
            best_hc,
            best_tb,
            best_sa,
        )
        final_perm = best_sa_perm
        final_c = best_sa
    elif algo == "auto":
        params = AlgoParams(
            neighbor_limit=neighbor_limit,
            max_no_improve=max_no_improve,
            tabu_iterations=tabu_iterations,
            tabu_tenure=tabu_tenure,
            tabu_candidate_size=tabu_candidate_size,
            sa_iterations=sa_iterations,
            sa_initial_temp=sa_initial_temp,
            sa_cooling=sa_cooling,
            sa_neighbor_moves=sa_neighbor_moves,
        )
        final_perm, final_c = run_auto(
            instance=instance,
            instance_path=instance_path,
            runs=runs,
            params=params,
            rng=rng,
            charts_dir=charts_dir,
        )
        return

    # Common post-processing (non-benchmark & non-auto)
    if final_perm is None:
        logger.warning("No final permutation available for visualization")
        return
    if not gantt and not gantt_path:
        return
    sched = build_schedule_from_permutation(
        instance,
        final_perm,
        check_completeness=True,
    )
    out_path = gantt_path
    if out_path is None:
        os.makedirs(charts_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            charts_dir,
            f"gantt_{algo}_c{sched.cmax}_{stamp}.png",
        )
    logger.info("Saving Gantt chart to %s", out_path)
    plot_gantt(sched, save_path=out_path, algo_name=algo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSSP metaheuristic demo")
    parser.add_argument("--instance", default="data/JSPLIB/instances/ta01")
    parser.add_argument("--algo", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neighbor-limit", type=int, default=40)
    parser.add_argument("--max-no-improve", type=int, default=20)
    parser.add_argument("--tabu-iterations", type=int, default=150)
    parser.add_argument("--tabu-tenure", type=int, default=12)
    parser.add_argument("--tabu-candidate-size", type=int, default=60)
    parser.add_argument("--sa-iterations", type=int, default=800)
    parser.add_argument("--sa-initial-temp", type=float, default=40.0)
    parser.add_argument("--sa-cooling", type=float, default=0.96)
    parser.add_argument("--sa-neighbor-moves", type=int, default=2)
    parser.add_argument("--pipeline-runs", type=int, default=5)
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Repetitions for auto mode",
    )
    # Benchmark mode arguments (duplicated with inner parser for now)
    parser.add_argument(
        "--instances-dir",
        default="data/JSPLIB/instances",
        help="Directory with Taillard instances (for benchmark mode)",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="research",
        help="Root directory for benchmark outputs (benchmark mode)",
    )
    parser.add_argument(
        "--benchmark-sample",
        type=int,
        default=5,
        help="Number of random instances to sample for benchmark (max)",
    )
    parser.add_argument(
        "--charts-dir",
        default="charts",
        help="Directory to save generated Gantt charts",
    )
    parser.add_argument(
        "--gantt",
        action="store_true",
        help="Render Gantt chart of final solution",
    )
    parser.add_argument(
        "--gantt-path",
        help="Path to save chart (PNG) instead of displaying",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # No extra arguments -> force auto charts
    if len(sys.argv) == 1:
        args.gantt = True
        logging.getLogger(__name__).info("No CLI args -> forcing auto mode charts enabled")
    main(
        instance_path=args.instance,
        algo=args.algo,
        seed=args.seed,
        neighbor_limit=args.neighbor_limit,
        max_no_improve=args.max_no_improve,
        tabu_iterations=args.tabu_iterations,
        tabu_tenure=args.tabu_tenure,
        tabu_candidate_size=args.tabu_candidate_size,
        sa_iterations=args.sa_iterations,
        sa_initial_temp=args.sa_initial_temp,
        sa_cooling=args.sa_cooling,
        sa_neighbor_moves=args.sa_neighbor_moves,
        pipeline_runs=args.pipeline_runs,
        gantt=args.gantt,
        gantt_path=args.gantt_path,
        runs=args.runs,
        charts_dir=args.charts_dir,
        instances_dir=args.instances_dir,
        benchmark_dir=args.benchmark_dir,
        benchmark_sample=args.benchmark_sample,
    )
