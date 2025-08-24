"""Executable runner for Job Shop metaheuristic experiments.

Modes:
* demo      – neighborhood sample + HC -> Tabu -> SA pipeline
* hill/tabu/sa – run a single algorithm
* pipeline  – sequential HC -> Tabu -> SA starting from SPT
* auto      – independent multi‑start benchmark of all algorithms with reports
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys
import logging
from datetime import datetime
from typing import Optional

from src.decoder import build_schedule_from_permutation, validate_permutation
from src.search import evaluate
from src.models import DataInstance
from src.neighborhood import generate_neighbors
from src.operations import (
    create_base_permutation as create_spt_permutation,  # fallback
)
from src.decoder import create_random_permutation
from src.search import hill_climb, simulated_annealing, tabu_search
from src.visualization import plot_gantt, plot_progress_curves
from src.decoder import check_no_machine_overlap
from src.parser import load_instance

logger = logging.getLogger("jssp")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def main(
    instance_path: str,
    algo: str,
    seed: Optional[int],
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
    gantt_path: Optional[str],
    runs: int,
    charts_dir: str,
):
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
        "--runs", type=int, default=100, help="Repetitions for auto mode"
    )
    parser.add_argument(
        "--charts-dir", default="charts", help="Directory to save charts"
    )
    parser.add_argument(
        "--gantt", action="store_true", help="Render Gantt of final solution"
    )
    parser.add_argument(
        "--gantt-path", help="Path to save chart (PNG) instead of displaying"
    )
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
    args = parser.parse_args()

    instance_path = args.instance
    algo = args.algo
    seed: Optional[int] = args.seed
    neighbor_limit = args.neighbor_limit
    max_no_improve = args.max_no_improve
    tabu_iterations = args.tabu_iterations
    tabu_tenure = args.tabu_tenure
    tabu_candidate_size = args.tabu_candidate_size
    sa_iterations = args.sa_iterations
    sa_initial_temp = args.sa_initial_temp
    sa_cooling = args.sa_cooling
    sa_neighbor_moves = args.sa_neighbor_moves
    pipeline_runs = args.pipeline_runs
    runs = args.runs
    charts_dir = args.charts_dir
    instances_dir = args.instances_dir
    benchmark_dir = args.benchmark_dir
    benchmark_sample = args.benchmark_sample
    gantt = args.gantt
    gantt_path = args.gantt_path

    # If user supplied no extra args force Gantt generation
    import sys
    if len(sys.argv) == 1:
        gantt = True

    instance: DataInstance = load_instance(instance_path)
    logger.info(
        "Instance loaded: jobs=%d machines=%d ops=%d",
        instance.jobs_number,
        instance.machines_number,
        instance.jobs_number * instance.machines_number,
    )

    # Heurystyki startowe
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
        instance, rand_perm, check_completeness=True
    )
    check_no_machine_overlap(rand_schedule)
    logger.info("Random schedule cmax=%s", rand_schedule.cmax)

    spt_perm = create_spt_permutation(instance)
    validate_permutation(instance, spt_perm)
    spt_schedule = build_schedule_from_permutation(
        instance, spt_perm, check_completeness=True
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

    # --- Tryby ---
    if algo == "benchmark":
        # Batch research mode: iterate Taillard instance files in directory
        # run each algorithm independently 'runs' times and store outputs
        if not os.path.isdir(instances_dir):
            logger.error("Instances directory not found: %s", instances_dir)
            return
        instance_files = [
            os.path.join(instances_dir, f)
            for f in sorted(os.listdir(instances_dir))
            if (
                os.path.isfile(os.path.join(instances_dir, f))
                and not f.startswith(".")
            )
        ]
    # Keep only Taillard instances (prefix 'ta') to avoid incompatible
    # formats (la*, orb*, etc.)
        instance_files = [
            p for p in instance_files if os.path.basename(p).startswith("ta")
        ]
        if not instance_files:
            logger.error(
                "No Taillard (ta*) instance files found in %s", instances_dir
            )
            return
        # Sample N random distinct instances (or fewer if not enough)
        if instance_files:
            k = min(max(1, benchmark_sample), len(instance_files))
            instance_files = random.sample(instance_files, k)
            logger.info("Benchmark sampling %d random instances", k)
        if not instance_files:
            logger.error("No instance files in %s", instances_dir)
            return
        algos = ["hill", "tabu", "sa"]
        logger.info(
            "Benchmark: %d instances, algorithms=%s, runs per algo=%d",
            len(instance_files),
            ",".join(algos),
            runs,
        )

        def _perm_inline(perm):
            if perm is None:
                return None
            return "[" + ",".join(f"[{p[0]},{p[1]}]" for p in perm) + "]"

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
            # Collect best-run progress/time series per algorithm
            # for a combined plot
            inst_progress: dict[str, list[int]] = {}
            inst_time_progress: dict[str, list[float]] = {}
            for algo_name in algos:
                out_dir = os.path.join(benchmark_dir, inst_name, algo_name)
                os.makedirs(out_dir, exist_ok=True)
                # Clean old outputs (json/png)
                for f in os.listdir(out_dir):
                    if f.endswith('.json') or f.lower().endswith('.png'):
                        try:
                            os.remove(os.path.join(out_dir, f))
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
                # Stable incremental results file path (overwritten each run)
                incr_path = os.path.join(
                    out_dir, f"results_incremental_{algo_name}.json"
                )
                algo_start_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for r_i in range(1, runs + 1):
                    start_perm = create_random_permutation(data_inst, rng=rng)
                    progress_list: list[int] = []
                    time_list: list[float] = []
                    t0 = time.perf_counter()
                    if algo_name == "hill":
                        perm, c, _, _ = hill_climb(
                            data_inst,
                            start_perm,
                            neighbor_limit=neighbor_limit,
                            max_no_improve=max_no_improve,
                            best_improvement=True,
                            rng=rng,
                            progress=progress_list,
                            time_progress=time_list,
                        )
                    elif algo_name == "tabu":
                        perm, c, _ = tabu_search(
                            data_inst,
                            start_perm,
                            iterations=tabu_iterations,
                            tenure=tabu_tenure,
                            candidate_size=tabu_candidate_size,
                            rng=rng,
                            progress=progress_list,
                            time_progress=time_list,
                        )
                    else:  # sa
                        perm, c, _, _ = simulated_annealing(
                            data_inst,
                            start_perm,
                            iterations=sa_iterations,
                            initial_temp=sa_initial_temp,
                            cooling=sa_cooling,
                            neighbor_moves=sa_neighbor_moves,
                            rng=rng,
                            progress=progress_list,
                            time_progress=time_list,
                        )
                    elapsed = time.perf_counter() - t0
                    results_c.append(c)
                    results_t.append(elapsed)
                    if best_c is None or c < best_c:  # type: ignore[operator]
                        best_c = c
                        best_time = elapsed
                        best_perm = perm
                        best_progress = list(progress_list)
                        best_time_progress = list(time_list)
                    # Incremental save after this run (fault tolerance)
                    per_run_records.append(
                        {"run": r_i, "cmax": c, "time": elapsed}
                    )
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
                            json.dump(
                                incr_payload,
                                f_inc,
                                ensure_ascii=False,
                                indent=2,
                            )
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
                # Save progress curve (best run)
                try:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    prog_path = os.path.join(
                        out_dir, f"progress_{algo_name}_{stamp}.png"
                    )
                    plot_progress_curves(
                        {algo_name: best_progress},
                        {algo_name: best_time_progress},
                        save_path=prog_path,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning("Progress plot failed %s %s", algo_name, e)
                # Save Gantt for best perm
                try:
                    if best_perm is not None:
                        sched_best = build_schedule_from_permutation(
                            data_inst, best_perm, check_completeness=True
                        )
                        g_path = os.path.join(
                            out_dir,
                            f"gantt_{algo_name}_c{sched_best.cmax}.png",
                        )
                        plot_gantt(
                            sched_best,
                            save_path=g_path,
                            algo_name=algo_name,
                        )
                except Exception as e:  # pragma: no cover
                    logger.warning("Gantt failed %s %s", algo_name, e)
                # JSON summary
                try:
                    stamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_path = os.path.join(
                        out_dir, f"results_{algo_name}_{stamp2}.json"
                    )
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
                            "avg_cmax": (
                                sum(results_c) / len(results_c)
                                if results_c
                                else None
                            ),
                            "avg_time": (
                                sum(results_t) / len(results_t)
                                if results_t
                                else None
                            ),
                        },
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:  # pragma: no cover
                    logger.warning("JSON save failed %s %s", algo_name, e)
                # Store best run progress for combined chart
                inst_progress[algo_name] = best_progress
                inst_time_progress[algo_name] = best_time_progress
            # After all algorithms for this instance -> combined progress plot
            try:
                inst_root = os.path.join(benchmark_dir, inst_name)
                # Optional cleanup of previous combined plots
                for f in os.listdir(inst_root):
                    if (
                        f.startswith("progress_all_")
                        and f.lower().endswith(".png")
                    ):
                        try:
                            os.remove(os.path.join(inst_root, f))
                        except OSError:
                            pass
                stamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_path = os.path.join(
                    inst_root, f"progress_all_{stamp_all}.png"
                )
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
                    "Combined progress plot failed for %s: %s", inst_name, e
                )
        return
    if algo == "demo":
        neighbors = generate_neighbors(spt_perm, limit=5, rng=rng)
        best_c = None
        for p in neighbors:
            c = evaluate(instance, p)
            if best_c is None or c < best_c:  # type: ignore[operator]
                best_c = c  # type: ignore[assignment]
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
            if (
                best_sa_perm is None
                or final_c is None
                or sa_c < final_c  # type: ignore[operator]
            ):
                best_sa_perm = sa_perm
                final_c = sa_c  # type: ignore[assignment]
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
        # Independent multi-run benchmarking; algorithms do NOT seed each
        # other (fair comparison of starts). Each algorithm starts from its
        # own fresh random permutation per run.
        logger.info(
            "Auto mode (independent): running each algorithm %d times", runs
        )

        def _avg(values: list[float]) -> float:
            return sum(values) / len(values) if values else float("nan")

        # Containers per algorithm
        stats = {
            "hill": {
                "best_c": None,
                "best_perm": None,
                "best_time": None,
                "c": [],
                "t": [],
                "progress": [],
                "time_progress": [],
            },
            "tabu": {
                "best_c": None,
                "best_perm": None,
                "best_time": None,
                "c": [],
                "t": [],
                "progress": [],
                "time_progress": [],
            },
            "sa": {
                "best_c": None,
                "best_perm": None,
                "best_time": None,
                "c": [],
                "t": [],
                "progress": [],
                "time_progress": [],
            },
        }
        for i in range(1, runs + 1):
            # Fresh random start for this iteration
            start_perm = create_random_permutation(instance, rng=rng)

            # Hill Climb
            t0 = time.perf_counter()
            hc_perm, hc_c, _, _ = hill_climb(
                instance,
                start_perm,
                neighbor_limit=neighbor_limit,
                max_no_improve=max_no_improve,
                best_improvement=True,
                rng=rng,
                progress=stats["hill"]["progress"],
                time_progress=stats["hill"]["time_progress"],
            )
            t_hc = time.perf_counter() - t0
            stats["hill"]["c"].append(hc_c)  # type: ignore[index]
            stats["hill"]["t"].append(t_hc)  # type: ignore[index]
            if (
                stats["hill"]["best_c"] is None
                or hc_c < stats["hill"]["best_c"]  # type: ignore[operator]
            ):
                stats["hill"].update(
                    {"best_c": hc_c, "best_perm": hc_perm, "best_time": t_hc}
                )

            # Tabu (independent start)
            t1 = time.perf_counter()
            tb_perm, tb_c, _ = tabu_search(
                instance,
                start_perm,
                iterations=tabu_iterations,
                tenure=tabu_tenure,
                candidate_size=tabu_candidate_size,
                rng=rng,
                progress=stats["tabu"]["progress"],
                time_progress=stats["tabu"]["time_progress"],
            )
            t_tb = time.perf_counter() - t1
            stats["tabu"]["c"].append(tb_c)  # type: ignore[index]
            stats["tabu"]["t"].append(t_tb)  # type: ignore[index]
            if (
                stats["tabu"]["best_c"] is None
                or tb_c < stats["tabu"]["best_c"]  # type: ignore[operator]
            ):
                stats["tabu"].update(
                    {"best_c": tb_c, "best_perm": tb_perm, "best_time": t_tb}
                )

            # Simulated Annealing (independent start)
            t2 = time.perf_counter()
            sa_perm, sa_c, _, _ = simulated_annealing(
                instance,
                start_perm,
                iterations=sa_iterations,
                initial_temp=sa_initial_temp,
                cooling=sa_cooling,
                neighbor_moves=sa_neighbor_moves,
                rng=rng,
                progress=stats["sa"]["progress"],
                time_progress=stats["sa"]["time_progress"],
            )
            t_sa = time.perf_counter() - t2
            stats["sa"]["c"].append(sa_c)  # type: ignore[index]
            stats["sa"]["t"].append(t_sa)  # type: ignore[index]
            if (
                stats["sa"]["best_c"] is None
                or sa_c < stats["sa"]["best_c"]  # type: ignore[operator]
            ):
                stats["sa"].update(
                    {"best_c": sa_c, "best_perm": sa_perm, "best_time": t_sa}
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

        # Summaries
        for name in ("hill", "tabu", "sa"):
            b_c = stats[name]["best_c"]
            b_t = stats[name]["best_time"]
            avg_c = _avg([float(v) for v in stats[name]["c"]])
            avg_t = _avg(stats[name]["t"])  # type: ignore[arg-type]
            logger.info(
                "Auto summary %-8s: best_c=%s best_time=%.4fs | "
                "avg_c=%.2f avg_time=%.4fs",
                name,
                b_c,
                b_t,
                avg_c,
                avg_t,
            )

        # Determine overall best (lowest cmax)
        candidates = [
            (
                name,
                stats[name]["best_c"],
                stats[name]["best_perm"],
            )
            for name in ("hill", "tabu", "sa")
        ]
        candidates = [
            c for c in candidates if c[1] is not None and c[2] is not None
        ]
        if candidates:
            best_algo_name, final_c, final_perm = min(
                candidates, key=lambda x: x[1]
            )  # type: ignore
            logger.info(
                "Overall best algorithm=%s cmax=%s", best_algo_name, final_c
            )

        # Post‑processing for auto mode: plots, JSON, per‑algorithm Gantty
        auto_best = {k: stats[k]["best_perm"] for k in ("hill", "tabu", "sa")}
        try:
            os.makedirs(charts_dir, exist_ok=True)
            # Clean old PNGs so only the newest batch remains
            for f in os.listdir(charts_dir):
                if f.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(charts_dir, f))
                    except OSError:
                        pass
            # Progress scatter
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_path = os.path.join(
                charts_dir, f"progress_curves_{stamp}.png"
            )
            plot_progress_curves(
                {k: stats[k]["progress"] for k in ("hill", "tabu", "sa")},
                {k: stats[k]["time_progress"] for k in ("hill", "tabu", "sa")},
                save_path=progress_path,
            )
            logger.info("Saved progress curves figure to %s", progress_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to plot progress curves: %s", e)

        # JSON export
        try:
            def _perm_to_list(perm):
                if perm is None:
                    return None
                return [[int(p[0]), int(p[1])] for p in perm]

            def _perm_compact(perm):
                if perm is None:
                    return None
                return ",".join(f"J{p[0]}O{p[1]}" for p in perm)

            def _job_sequence(perm):
                if perm is None:
                    return None
                return [int(p[0]) for p in perm]

            stamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(
                charts_dir, f"auto_results_{stamp2}.json"
            )
            per_run = {"hill": [], "tabu": [], "sa": []}
            for idx in range(len(stats["hill"]["c"])):
                per_run["hill"].append(
                    {
                        "run": idx + 1,
                        "cmax": stats["hill"]["c"][idx],
                        "time": stats["hill"]["t"][idx],
                    }
                )
            for idx in range(len(stats["tabu"]["c"])):
                per_run["tabu"].append(
                    {
                        "run": idx + 1,
                        "cmax": stats["tabu"]["c"][idx],
                        "time": stats["tabu"]["t"][idx],
                    }
                )
            for idx in range(len(stats["sa"]["c"])):
                per_run["sa"].append(
                    {
                        "run": idx + 1,
                        "cmax": stats["sa"]["c"][idx],
                        "time": stats["sa"]["t"][idx],
                    }
                )
            json_payload = {
                "instance": instance_path,
                "runs": runs,
                "timestamp": stamp2,
                "per_run": per_run,
                "best": {
                    algo: {
                        "cmax": stats[algo]["best_c"],
                        "time": stats[algo]["best_time"],
                        "permutation_pairs": _perm_to_list(
                            stats[algo]["best_perm"]
                        ),
                        "permutation_compact": _perm_compact(
                            stats[algo]["best_perm"]
                        ),
                        "job_sequence": _job_sequence(
                            stats[algo]["best_perm"]
                        ),
                    }
                    for algo in ("hill", "tabu", "sa")
                },
                "averages": {
                    algo: {
                        "avg_cmax": (
                            sum(stats[algo]["c"]) / len(stats[algo]["c"])
                            if stats[algo]["c"]
                            else None
                        ),
                        "avg_time": (
                            sum(stats[algo]["t"]) / len(stats[algo]["t"])
                            if stats[algo]["t"]
                            else None
                        ),
                    }
                    for algo in ("hill", "tabu", "sa")
                },
            }
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, ensure_ascii=False, indent=2)
            logger.info("Saved auto mode results JSON to %s", results_path)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to write results JSON: %s", e)

        # Per‑algorithm Gantty zawsze generowane w auto
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

    # Return early for auto mode (single‑algorithm block below not needed)
        return
        if final_perm is None:
            logger.warning("Brak finalnej permutacji do wizualizacji")
            return
        # Decode final schedule
        sched = build_schedule_from_permutation(
            instance,
            final_perm,
            check_completeness=True,
        )
        # Determine output path
        out_path = gantt_path
        if out_path is None:
            os.makedirs(charts_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_tag = algo
            if algo == "auto" and 'algo' in locals():  # best algo reused
                algo_tag = locals().get('algo', 'auto')
            out_path = os.path.join(
                charts_dir,
                f"gantt_{algo_tag}_c{sched.cmax}_{stamp}.png",
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
        "--gantt-path", help="Path to save chart (PNG) instead of displaying"
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
        logging.getLogger(__name__).info(
            "No CLI args -> forcing auto mode charts enabled"
        )
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
    )
