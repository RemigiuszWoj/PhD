import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.decoder import build_schedule_from_permutation, create_sequential_permutation  # noqa: E402
from src.parser import load_instance  # noqa: E402
from src.search import simulated_annealing, tabu_search  # noqa: E402
from src.visualization import plot_gantt, plot_iteration_progress_multi  # noqa: E402


def run_single(
    inst_path,
    algo_name,
    runs,
    seed,
    tabu_iterations,
    tabu_tenure,
    tabu_neighborhood,
    sa_iterations,
    sa_initial_temp,
    sa_cooling,
    sa_neighbor_moves,
    sa_neighborhood,
    charts_dir,
    time_limit_ms,
    logger,
    traces_dir,
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
    all_current: list[list[int]] = []
    # przygotowanie trace (zawsze włączone)
    trace_dir = os.path.join(traces_dir, "traces") if traces_dir else None
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    for r_idx in range(runs):
        # deterministyczna permutacja startowa (blokami jobów)
        perm = create_sequential_permutation(instance)
        trace_file = None
        if trace_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_file = os.path.join(
                trace_dir,
                f"trace_{algo_name}_run{r_idx}_" + os.path.basename(inst_path) + f"_{ts}.txt",
            )
            with open(trace_file, "w", encoding="utf-8") as tf:
                if algo_name == "tabu":
                    tf.write("iter;current;best;move;evals\n")
                else:
                    tf.write("iter;T;current;best;delta;accepted;evals;move\n")

        if algo_name == "tabu":
            p, c, _, current_hist = tabu_search(
                instance,
                perm,
                iterations=tabu_iterations,
                tenure=tabu_tenure,
                neighborhood=tabu_neighborhood,
                rng=rng,
                time_limit_ms=time_limit_ms,
                gantt_dir=os.path.join(charts_dir, "tabu_iter_gantts"),
                trace_file=trace_file,
            )
        elif algo_name == "sa":
            p, c, *_rest = simulated_annealing(
                instance,
                perm,
                iterations=sa_iterations,
                initial_temp=sa_initial_temp,
                cooling=sa_cooling,
                neighbor_moves=sa_neighbor_moves,
                rng=rng,
                neighborhood=sa_neighborhood,
                time_limit_ms=time_limit_ms,
                gantt_dir=os.path.join(charts_dir, "sa_iter_gantts"),
                trace_file=trace_file,
            )
            # _rest = (evals, it, current_hist)
            current_hist = _rest[-1]
        else:
            logger.error(f"Unknown algorithm: {algo_name}")
            return
        all_current.append(current_hist)
        if best_c is None or c < best_c:
            best_c = c
            best_perm = p
    sched = build_schedule_from_permutation(instance, best_perm, check_completeness=True)
    os.makedirs(charts_dir, exist_ok=True)
    out_path = os.path.join(
        charts_dir,
        f"gantt_{algo_name}_c{sched.cmax}_"
        f"{os.path.basename(inst_path)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    plot_gantt(sched, save_path=out_path, algo_name=algo_name)
    logger.info(f"Saved Gantt chart to {out_path}")
    # wykres wieloseriiowy (wszystkie raw cmax)
    if all_current:
        # wyrównanie długości (dłuższe mogą być skrócone do minimalnej długości)
        min_len = min(len(lst) for lst in all_current if lst)
        trimmed = {
            f"run_{i}": lst[:min_len] for i, lst in enumerate(all_current) if len(lst) >= min_len
        }
        if trimmed:
            multi_path = os.path.join(charts_dir, f"{algo_name}_multi_progress.png")
            plot_iteration_progress_multi(trimmed, save_path=multi_path)
            logger.info(f"Saved multi-series progress to {multi_path}")


def main(
    instance_path,
    algo,
    runs,
    seed,
    tabu_iterations,
    tabu_tenure,
    tabu_neighborhood,
    sa_iterations,
    sa_initial_temp,
    sa_cooling,
    sa_neighbor_moves,
    sa_neighborhood,
    charts_dir,
    time_limit_ms,
    traces_dir,
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
    elif algo == "sa_compare":  # wewnętrzny tryb porównania dwóch sąsiedztw SA
        algos = ["sa_compare"]
    elif algo == "tabu_compare":  # wewnętrzny tryb porównania dwóch sąsiedztw Tabu
        algos = ["tabu_compare"]
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
                if a == "sa_compare":
                    from src.decoder import create_sequential_permutation

                    instance = load_instance(f)
                    base_perm = create_sequential_permutation(instance)
                    rng = random.Random(seed) if seed is not None else random.Random()
                    histories: dict[str, list[int]] = {}
                    for neigh in ["swap_adjacent_neighborhood", "fibonachi_neighborhood"]:
                        _p, c, _evals, _it, hist = simulated_annealing(
                            instance,
                            base_perm,
                            iterations=sa_iterations,
                            initial_temp=sa_initial_temp,
                            cooling=sa_cooling,
                            neighbor_moves=sa_neighbor_moves,
                            rng=rng,
                            neighborhood=neigh,
                            time_limit_ms=None,
                            gantt_dir=None,
                            trace_file=None,
                        )
                        histories[neigh] = hist
                        logger.info(
                            "SA compare file=%s neigh=%s best_c=%s final_current=%s",
                            os.path.basename(f),
                            neigh,
                            c,
                            (hist[-1] if hist else "n/a"),
                        )
                    min_len = min(len(h) for h in histories.values() if h)
                    trimmed = {k: v[:min_len] for k, v in histories.items()}
                    os.makedirs(charts_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_name = f"sa_compare_progress_{os.path.basename(f)}_{ts}.png"
                    out_path = os.path.join(charts_dir, out_name)
                    plot_iteration_progress_multi(trimmed, save_path=out_path)
                    logger.info(f"Saved SA neighborhood comparison to {out_path}")
                elif a == "tabu_compare":
                    from src.decoder import create_sequential_permutation

                    instance = load_instance(f)
                    base_perm = create_sequential_permutation(instance)
                    rng = random.Random(seed) if seed is not None else random.Random()
                    histories: dict[str, list[int]] = {}
                    for neigh in ["swap_adjacent_neighborhood", "fibonachi_neighborhood"]:
                        _p, c, _evals, hist = tabu_search(
                            instance,
                            base_perm,
                            iterations=tabu_iterations,
                            tenure=tabu_tenure,
                            neighborhood=neigh,
                            rng=rng,
                            time_limit_ms=None,
                            gantt_dir=None,
                            trace_file=None,
                        )
                        histories[neigh] = hist
                        logger.info(
                            "TABU compare file=%s neigh=%s best_c=%s final_current=%s",
                            os.path.basename(f),
                            neigh,
                            c,
                            (hist[-1] if hist else "n/a"),
                        )
                    min_len = min(len(h) for h in histories.values() if h)
                    trimmed = {k: v[:min_len] for k, v in histories.items()}
                    os.makedirs(charts_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_name = f"tabu_compare_progress_{os.path.basename(f)}_{ts}.png"
                    out_path = os.path.join(charts_dir, out_name)
                    plot_iteration_progress_multi(trimmed, save_path=out_path)
                    logger.info(f"Saved TABU neighborhood comparison to {out_path}")
                else:
                    run_single(
                        f,
                        a,
                        runs,
                        seed,
                        tabu_iterations,
                        tabu_tenure,
                        tabu_neighborhood,
                        sa_iterations,
                        sa_initial_temp,
                        sa_cooling,
                        sa_neighbor_moves,
                        sa_neighborhood,
                        charts_dir,
                        time_limit_ms,
                        logger,
                        charts_dir,
                    )
    else:
        for a in algos:
            if a == "sa_compare":
                from src.decoder import create_sequential_permutation

                instance = load_instance(instance_path)
                base_perm = create_sequential_permutation(instance)
                rng = random.Random(seed) if seed is not None else random.Random()
                histories: dict[str, list[int]] = {}
                for neigh in ["swap_adjacent_neighborhood", "fibonachi_neighborhood"]:
                    _p, c, _evals, _it, hist = simulated_annealing(
                        instance,
                        base_perm,
                        iterations=sa_iterations,
                        initial_temp=sa_initial_temp,
                        cooling=sa_cooling,
                        neighbor_moves=sa_neighbor_moves,
                        rng=rng,
                        neighborhood=neigh,
                        time_limit_ms=None,
                        gantt_dir=None,
                        trace_file=None,
                    )
                    histories[neigh] = hist
                    logger.info(
                        "SA compare instance=%s neigh=%s best_c=%s final_current=%s",
                        os.path.basename(instance_path),
                        neigh,
                        c,
                        (hist[-1] if hist else "n/a"),
                    )
                min_len = min(len(h) for h in histories.values() if h)
                trimmed = {k: v[:min_len] for k, v in histories.items()}
                os.makedirs(charts_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"sa_compare_progress_{os.path.basename(instance_path)}_{ts}.png"
                out_path = os.path.join(charts_dir, out_name)
                plot_iteration_progress_multi(trimmed, save_path=out_path)
                logger.info(f"Saved SA neighborhood comparison to {out_path}")
            elif a == "tabu_compare":
                from src.decoder import create_sequential_permutation

                instance = load_instance(instance_path)
                base_perm = create_sequential_permutation(instance)
                rng = random.Random(seed) if seed is not None else random.Random()
                histories: dict[str, list[int]] = {}
                for neigh in ["swap_adjacent_neighborhood", "fibonachi_neighborhood"]:
                    _p, c, _evals, hist = tabu_search(
                        instance,
                        base_perm,
                        iterations=tabu_iterations,
                        tenure=tabu_tenure,
                        neighborhood=neigh,
                        rng=rng,
                        time_limit_ms=None,
                        gantt_dir=None,
                        trace_file=None,
                    )
                    histories[neigh] = hist
                    logger.info(
                        "TABU compare instance=%s neigh=%s best_c=%s final_current=%s",
                        os.path.basename(instance_path),
                        neigh,
                        c,
                        (hist[-1] if hist else "n/a"),
                    )
                min_len = min(len(h) for h in histories.values() if h)
                trimmed = {k: v[:min_len] for k, v in histories.items()}
                os.makedirs(charts_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"tabu_compare_progress_{os.path.basename(instance_path)}_{ts}.png"
                out_path = os.path.join(charts_dir, out_name)
                plot_iteration_progress_multi(trimmed, save_path=out_path)
                logger.info(f"Saved TABU neighborhood comparison to {out_path}")
            else:
                run_single(
                    instance_path,
                    a,
                    runs,
                    seed,
                    tabu_iterations,
                    tabu_tenure,
                    tabu_neighborhood,
                    sa_iterations,
                    sa_initial_temp,
                    sa_cooling,
                    sa_neighbor_moves,
                    sa_neighborhood,
                    charts_dir,
                    time_limit_ms,
                    logger,
                    charts_dir,
                )


if __name__ == "__main__":
    # Teraz uruchomienie wyłącznie z pliku konfiguracyjnego.
    parser = argparse.ArgumentParser(description="JSSP metaheuristic demo (config only)")
    parser.add_argument(
        "--config",
        required=True,
        help="Ścieżka do pliku konfiguracyjnego YAML/JSON (wymagane)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        text = f.read()

    if args.config.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Brak pakietu pyyaml do parsowania YAML") from e
        cfg: Dict[str, Any] = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)

    # Sekcje
    tabu_cfg = cfg.get("tabu", {}) if isinstance(cfg.get("tabu"), dict) else {}
    sa_cfg = cfg.get("sa", {}) if isinstance(cfg.get("sa"), dict) else {}
    charts_cfg = cfg.get("charts", {}) if isinstance(cfg.get("charts"), dict) else {}

    log_level = cfg.get("log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Odczyt wartości (bez nadpisywania przez CLI)
    instance_path = cfg.get("instance")
    if not instance_path:
        raise ValueError("Brak klucza 'instance' w configu")
    algo = cfg.get("algo", "tabu")
    runs = int(cfg.get("runs", 1))
    seed = cfg.get("seed")
    time_limit_ms = cfg.get("time_limit_ms")

    main(
        instance_path=instance_path,
        algo=algo,
        runs=runs,
        seed=seed,
        tabu_iterations=int(tabu_cfg.get("iterations", 100)),
        tabu_tenure=int(tabu_cfg.get("tenure", 10)),
        tabu_neighborhood=tabu_cfg.get("neighborhood", "swap_adjacent_neighborhood"),
        sa_iterations=int(sa_cfg.get("iterations", 500)),
        sa_initial_temp=float(sa_cfg.get("initial_temp", 30.0)),
        sa_cooling=float(sa_cfg.get("cooling", 0.95)),
        sa_neighbor_moves=int(sa_cfg.get("neighbor_moves", 2)),
        sa_neighborhood=sa_cfg.get("neighborhood", "swap_adjacent_neighborhood"),
        charts_dir=charts_cfg.get("dir", "charts"),
        time_limit_ms=time_limit_ms,
        traces_dir=charts_cfg.get("dir", "charts"),
    )
