#!/usr/bin/env python3


import os
import sys

import yaml

from src.experiments.aggregate import (
    write_matrix_gap_table,
    write_matrix_per_seed_table,
    write_summary_csv,
    write_wide_gap_table,
)
from src.experiments.runner import (
    ExperimentRunner,
    generate_plan_for_files,
)
from src.parser import parser
from src.serach import simulated_annealing, tabu_search
from src.taillard_gen import generate_taillard_instance
from src.visualization import (
    build_algorithm_multi_convergence_plots,
    clear_old_plots,
    next_unique_path,
    save_gantt_chart_with_name,
    save_multi_convergence_plot,
)

# main is now compare-only; run_algorithm wrapper removed to keep code minimal


def _pad_histories_and_store(
    results_dict,
    best_pis_dict,
    cmax_summary_dict,
    neigh_mode,
    iteration_history,
    cmax_history,
    best_pi,
    best_cmax,
):
    """Pad shorter history to match lengths, store results and summaries."""
    max_len = max(len(iteration_history), len(cmax_history))
    if len(iteration_history) < max_len:
        last_iter = iteration_history[-1] if iteration_history else 0
        iteration_history += [last_iter] * (max_len - len(iteration_history))
    if len(cmax_history) < max_len:
        last_cmax = cmax_history[-1] if cmax_history else 0
        cmax_history += [last_cmax] * (max_len - len(cmax_history))

    results_dict[neigh_mode] = (iteration_history, cmax_history)
    best_pis_dict[neigh_mode] = best_pi
    cmax_summary_dict[neigh_mode] = best_cmax


def run_compare_mode(
    algorithm,
    processing_times,
    config,
    algorithm_common,
    labels,
    colors,
    out_dir: str = None,
    results_folder: str = "results",
):
    clear_old_plots(results_folder)
    results = {}
    best_pis = {}
    cmax_summary = {}

    # Validate algorithm choice
    if algorithm not in ("tabu_search_compare", "simulated_annealing_compare"):
        raise ValueError(f"Unknown compare algorithm: {algorithm}")

    # Preload algorithm-specific config
    ts_config = config.get("tabu_search", {}) if algorithm == "tabu_search_compare" else None
    sa_config = (
        config.get("simulated_annealing", {})
        if algorithm == "simulated_annealing_compare"
        else None
    )

    for neigh_mode in [
        "adjacent",
        "fibonahi_neighborhood",
        "dynasearch_neighborhood",
    ]:
        if algorithm == "tabu_search_compare":
            best_pi, best_cmax, iteration_history, cmax_history = tabu_search(
                processing_times,
                max_time_ms=algorithm_common.get("time_limit_ms", 100000),
                tabu_tenure=ts_config.get("tabu_tenure"),
                neigh_mode=neigh_mode,
                iter_log_path=(
                    os.path.join(out_dir, f"iter_log_{neigh_mode}.csv") if out_dir else None
                ),
            )
        else:
            best_pi, best_cmax, iteration_history, cmax_history = simulated_annealing(
                processing_times,
                time_limit_ms=algorithm_common.get("time_limit_ms", 100000),
                initial_temp=sa_config.get("initial_temp"),
                final_temp=sa_config.get("final_temp"),
                alpha=sa_config.get("alpha"),
                neigh_mode=neigh_mode,
                reheat_factor=sa_config.get("reheat_factor"),
                stagnation_ms=sa_config.get("stagnation_ms"),
                temp_floor_factor=sa_config.get("temp_floor_factor"),
                iter_log_path=(
                    os.path.join(out_dir, f"iter_log_{neigh_mode}.csv") if out_dir else None
                ),
            )

        _pad_histories_and_store(
            results,
            best_pis,
            cmax_summary,
            neigh_mode,
            iteration_history,
            cmax_history,
            best_pi,
            best_cmax,
        )

    # Save multi convergence into optional out_dir
    if out_dir:
        base_multi = os.path.join(out_dir, "multi_convergence.png")
        multi_path = next_unique_path(base_multi)
    else:
        multi_path = None
    save_multi_convergence_plot(
        results,
        labels=labels,
        colors=colors,
        filepath=multi_path,
        time_limit_ms=algorithm_common.get("time_limit_ms"),
    )
    for neigh_mode in [
        "adjacent",
        "fibonahi_neighborhood",
        "dynasearch_neighborhood",
    ]:
        # Save gantt into neighborhood-specific file under out_dir if provided
        if out_dir:
            base_gantt = os.path.join(out_dir, f"gantt_chart_{neigh_mode}.png")
            gantt_name = next_unique_path(base_gantt)
        else:
            gantt_name = f"gantt_chart_{neigh_mode}.png"
        save_gantt_chart_with_name(
            best_pis[neigh_mode], processing_times, cmax_summary[neigh_mode], gantt_name
        )
    print("Comparison plot and Gantt charts saved.")
    for neigh_mode in [
        "adjacent",
        "fibonahi_neighborhood",
        "dynasearch_neighborhood",
    ]:
        print(f"Final cmax for {labels[neigh_mode]}: {cmax_summary[neigh_mode]}")


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main() -> None:
    # Load configuration from file
    config = load_config()

    # Conditional experiment batch mode
    exp_cfg = config.get("experiment", {})
    if exp_cfg.get("enabled"):
        instance_files = exp_cfg.get("instance_files") or []
        if not instance_files:
            raise ValueError("experiment.instance_files list must be set and non-empty")
        repeats = exp_cfg.get("repeats", 1)
        # Multi-limit list ONLY (no single fallback)
        time_limits_list = exp_cfg.get("time_limits_s") or []
        if not time_limits_list:
            raise ValueError("experiment.time_limits_s must be a non-empty list of seconds")
        try:
            time_limits_ms = [int(float(x) * 1000) for x in time_limits_list]
        except Exception:
            time_limits_ms = [int(float(x)) for x in time_limits_list]
        print(
            "[Main] Experiment batch mode: full enumeration over files / algorithms / "
            "neighborhoods."
        )
        plan = generate_plan_for_files(
            instance_files=instance_files,
            repeats=repeats,
            time_limits_ms=time_limits_ms,
        )
        # We no longer delete historical results – each run gets its own timestamped directory
        runner = ExperimentRunner()
        runner.run(plan)
        try:
            summary_path = write_summary_csv(runner.timestamp_dir)
            write_wide_gap_table(runner.timestamp_dir, summary_path)
            write_matrix_gap_table(runner.timestamp_dir, summary_path)
            write_matrix_per_seed_table(runner.timestamp_dir, summary_path)
            build_algorithm_multi_convergence_plots(runner.timestamp_dir)
        except Exception as e:
            print(f"[Main] Failed to write summary: {e}")
        print("[Main] Experiment batch completed.")
        return

    # Extract configuration parameters
    general_config = config["general"]
    viz_config = config["visualization"]
    base_results = viz_config.get("results_folder", "results")

    # Allow generating a synthetic instance via config.generator
    gen_cfg = config.get("generator", {})
    if gen_cfg.get("enabled"):
        m = gen_cfg.get("m")
        n = gen_cfg.get("n")
        seed = gen_cfg.get("seed")
        if m is None or n is None:
            raise ValueError("Generator enabled but 'm' or 'n' not provided in config.generator")
        # Taillard-only generator
        processing_times = generate_taillard_instance(n, m, seed)
        data = {
            "info": {"jobs": n, "machines": m, "seed": seed, "generator": "taillard"},
            "processing_times": processing_times,
        }
    else:
        data = parser(
            file_path=general_config["input_file"],
            instance_number=general_config["instance_number"],
        )
        processing_times = data["processing_times"]

    algorithm = general_config["algorithm"]

    algorithm_common = config.get("algorithm_common", {})
    # Normalize time limit: accept either `time_limit_ms` or `time_limit_s` in config
    if "time_limit_ms" not in algorithm_common:
        if "time_limit_s" in algorithm_common:
            try:
                algorithm_common["time_limit_ms"] = int(algorithm_common["time_limit_s"] * 1000)
            except Exception:
                algorithm_common["time_limit_ms"] = int(
                    float(algorithm_common["time_limit_s"]) * 1000
                )
        else:
            algorithm_common["time_limit_ms"] = 100000
    # Only run comparison flows (neighborhood comparisons) — simplify main.
    if algorithm == "tabu_search_compare":
        labels = {
            "adjacent": "Tabu: adjacent",
            "fibonahi_neighborhood": "Tabu: fibonahi_neigh",
            "dynasearch_neighborhood": "Tabu: dynasearch",
        }
        colors = {
            "adjacent": "#00FFFF",  # neon cyan
            "fibonahi_neighborhood": "#FF00CC",  # neon magenta (kept)
            "dynasearch_neighborhood": "#7CFF00",  # neon lime
        }
    elif algorithm == "simulated_annealing_compare":
        labels = {
            "adjacent": "SA: adjacent",
            "fibonahi_neighborhood": "SA: fibonahi_neigh",
            "dynasearch_neighborhood": "SA: dynasearch",
        }
        colors = {
            "adjacent": "#00FFFF",  # neon cyan
            "fibonahi_neighborhood": "#FF00CC",  # neon magenta (kept)
            "dynasearch_neighborhood": "#7CFF00",  # neon lime
        }
    else:
        raise ValueError(
            "Unknown algorithm: use 'tabu_search_compare' or 'simulated_annealing_compare'"
        )

    # run_compare_mode will save plots to results_folder/out_dir
    # If generator produced the instance, use a generated-friendly name
    if gen_cfg.get("enabled"):
        try:
            data_name = f"generated_taillard_m{m}_n{n}_seed{seed}"
        except Exception:
            data_name = "generated_instance"
        data_file = data_name
    else:
        data_file = general_config.get("input_file", "unknown")
        data_name = data_file.split("/")[-1].split(".")[0]
    instance_num = general_config.get("instance_number", 0)
    out_dir = os.path.join(base_results, f"{algorithm}_{data_name}_instance{instance_num}")
    os.makedirs(out_dir, exist_ok=True)
    run_compare_mode(
        algorithm,
        processing_times,
        config,
        algorithm_common,
        labels,
        colors,
        out_dir=out_dir,
        results_folder=base_results,
    )
    return


if __name__ == "__main__":
    # Allow deeper recursion for the recursive neighborhoods.
    # Set a higher recursion limit first (process-wide).
    try:
        sys.setrecursionlimit(2000000)
    except Exception:
        pass

    # Run main() in a dedicated thread with a larger stack to avoid
    # hitting the OS thread stack size when recursion is deep.
    try:
        import threading

        # Request a 128MB stack for the new thread. This may be ignored on some platforms
        # or restricted by system limits; it's a best-effort increase.
        try:
            threading.stack_size(254 * 1024 * 1024)
        except Exception:
            pass

        t = threading.Thread(target=main, name="main_runner")
        t.start()
        t.join()
    except Exception:
        # Fallback: call directly (with increased recursionlimit above)
        main()
