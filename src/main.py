import argparse
import logging
import os
import random
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decoder import build_schedule_from_permutation, create_random_permutation  # noqa: E402
from src.parser import load_instance  # noqa: E402
from src.search import simulated_annealing, tabu_search  # noqa: E402
from src.visualization import plot_gantt  # noqa: E402


def run_single(
    inst_path,
    algo,
    runs,
    seed,
    tabu_iterations,
    tabu_tenure,
    tabu_candidate_size,
    sa_iterations,
    sa_initial_temp,
    sa_cooling,
    sa_neighbor_moves,
    charts_dir,
    logger,
):
    instance = load_instance(inst_path)
    logger.info(
        "Instance: %s jobs=%d machines=%d ops=%d",
        inst_path,
        instance.jobs_number,
        instance.machines_number,
        instance.jobs_number * instance.machines_number,
    )
    rng = random.Random(seed) if seed is not None else random.Random()
    best_c = None
    best_perm = None
    for _ in range(runs):
        perm = create_random_permutation(instance, rng=rng)
        if algo == "tabu":
            p, c, _ = tabu_search(
                instance,
                perm,
                iterations=tabu_iterations,
                tenure=tabu_tenure,
                candidate_size=tabu_candidate_size,
                rng=rng,
            )
        elif algo == "sa":
            p, c, *_ = simulated_annealing(
                instance,
                perm,
                iterations=sa_iterations,
                initial_temp=sa_initial_temp,
                cooling=sa_cooling,
                neighbor_moves=sa_neighbor_moves,
                rng=rng,
            )
        else:
            logger.error(f"Unknown algorithm: {algo}")
            return
        if best_c is None or c < best_c:
            best_c = c
            best_perm = p
    sched = build_schedule_from_permutation(instance, best_perm, check_completeness=True)
    os.makedirs(charts_dir, exist_ok=True)
    out_path = os.path.join(
        charts_dir,
        f"gantt_{algo}_c{sched.cmax}_"
        f"{os.path.basename(inst_path)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plot_gantt(sched, save_path=out_path, algo_name=algo)
    logger.info(f"Saved Gantt chart to {out_path}")


def main(
    instance_path,
    algo,
    runs,
    seed,
    tabu_iterations,
    tabu_tenure,
    tabu_candidate_size,
    sa_iterations,
    sa_initial_temp,
    sa_cooling,
    sa_neighbor_moves,
    charts_dir,
):
    logger = logging.getLogger("jssp")
    logger.setLevel(logging.INFO)

    # Clean charts directory at the beginning (remove old image files)
    if os.path.exists(charts_dir):
        for f in os.listdir(charts_dir):
            fp = os.path.join(charts_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
    else:
        os.makedirs(charts_dir, exist_ok=True)

    algos = []
    if algo == "both":
        algos = ["tabu", "sa"]
    else:
        algos = [algo]

    if os.path.isdir(instance_path):
        files = [
            os.path.join(instance_path, f)
            for f in os.listdir(instance_path)
            if not f.startswith(".")
        ]
        for f in files:
            for a in algos:
                run_single(
                    f,
                    a,
                    runs,
                    seed,
                    tabu_iterations,
                    tabu_tenure,
                    tabu_candidate_size,
                    sa_iterations,
                    sa_initial_temp,
                    sa_cooling,
                    sa_neighbor_moves,
                    charts_dir,
                    logger,
                )
    else:
        for a in algos:
            run_single(
                instance_path,
                a,
                runs,
                seed,
                tabu_iterations,
                tabu_tenure,
                tabu_candidate_size,
                sa_iterations,
                sa_initial_temp,
                sa_cooling,
                sa_neighbor_moves,
                charts_dir,
                logger,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSSP metaheuristic demo (tabu/sa)")
    parser.add_argument("--instance", default="data/JSPLIB/instances/ta01")
    parser.add_argument("--algo", choices=["tabu", "sa", "both"], default="tabu")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of independent runs per instance"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tabu-iterations", type=int, default=150)
    parser.add_argument("--tabu-tenure", type=int, default=12)
    parser.add_argument("--tabu-candidate-size", type=int, default=60)
    parser.add_argument("--sa-iterations", type=int, default=800)
    parser.add_argument("--sa-initial-temp", type=float, default=40.0)
    parser.add_argument("--sa-cooling", type=float, default=0.96)
    parser.add_argument("--sa-neighbor-moves", type=int, default=2)
    parser.add_argument("--charts-dir", default="charts")
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(
        instance_path=args.instance,
        algo=args.algo,
        runs=args.runs,
        seed=args.seed,
        tabu_iterations=args.tabu_iterations,
        tabu_tenure=args.tabu_tenure,
        tabu_candidate_size=args.tabu_candidate_size,
        sa_iterations=args.sa_iterations,
        sa_initial_temp=args.sa_initial_temp,
        sa_cooling=args.sa_cooling,
        sa_neighbor_moves=args.sa_neighbor_moves,
        charts_dir=args.charts_dir,
    )
