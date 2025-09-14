#!/usr/bin/env python3


import yaml

from src.parser import parser
from src.serach import simulated_annealing, tabu_search
from src.visualization import (
    clear_old_plots,
    save_convergence_plot,
    save_gantt_chart,
    save_gantt_chart_with_name,
    save_multi_convergence_plot,
)


def run_algorithm(algorithm, processing_times, config, algorithm_common):
    if algorithm == "tabu_search":
        ts_config = config["tabu_search"]
        return tabu_search(
            processing_times,
            max_time_ms=algorithm_common.get("time_limit_ms", 100000),
            tabu_tenure=ts_config["tabu_tenure"],
            neigh_mode=algorithm_common.get("neigh_mode", "adjacent"),
        )
    elif algorithm == "simulated_annealing":
        sa_config = config["simulated_annealing"]
        return simulated_annealing(
            processing_times,
            time_limit_ms=algorithm_common.get("time_limit_ms", 100000),
            initial_temp=sa_config["initial_temp"],
            final_temp=sa_config["final_temp"],
            alpha=sa_config["alpha"],
            neigh_mode=algorithm_common.get("neigh_mode", "adjacent"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_compare_mode(algorithm, processing_times, config, algorithm_common, labels, colors):
    clear_old_plots()
    results = {}
    best_pis = {}
    cmax_summary = {}
    if algorithm == "tabu_search_compare":
        ts_config = config["tabu_search"]
        for neigh_mode in ["adjacent", "fibonahi_neighborhood"]:
            best_pi, best_cmax, iteration_history, cmax_history = tabu_search(
                processing_times,
                max_time_ms=algorithm_common.get("time_limit_ms", 100000),
                tabu_tenure=ts_config["tabu_tenure"],
                neigh_mode=neigh_mode,
            )
            max_len = max(len(iteration_history), len(cmax_history))
            if len(iteration_history) < max_len:
                last_iter = iteration_history[-1] if iteration_history else 0
                iteration_history += [last_iter] * (max_len - len(iteration_history))
            if len(cmax_history) < max_len:
                last_cmax = cmax_history[-1] if cmax_history else 0
                cmax_history += [last_cmax] * (max_len - len(cmax_history))
            results[neigh_mode] = (iteration_history, cmax_history)
            best_pis[neigh_mode] = best_pi
            cmax_summary[neigh_mode] = best_cmax
    elif algorithm == "simulated_annealing_compare":
        sa_config = config["simulated_annealing"]
        for neigh_mode in ["adjacent", "fibonahi_neighborhood"]:
            best_pi, best_cmax, iteration_history, cmax_history = simulated_annealing(
                processing_times,
                time_limit_ms=algorithm_common.get("time_limit_ms", 100000),
                initial_temp=sa_config["initial_temp"],
                final_temp=sa_config["final_temp"],
                alpha=sa_config["alpha"],
                neigh_mode=neigh_mode,
            )
            max_len = max(len(iteration_history), len(cmax_history))
            if len(iteration_history) < max_len:
                last_iter = iteration_history[-1] if iteration_history else 0
                iteration_history += [last_iter] * (max_len - len(iteration_history))
            if len(cmax_history) < max_len:
                last_cmax = cmax_history[-1] if cmax_history else 0
                cmax_history += [last_cmax] * (max_len - len(cmax_history))
            results[neigh_mode] = (iteration_history, cmax_history)
            best_pis[neigh_mode] = best_pi
            cmax_summary[neigh_mode] = best_cmax
    else:
        raise ValueError(f"Unknown compare algorithm: {algorithm}")
    save_multi_convergence_plot(results, labels=labels, colors=colors)
    for neigh_mode in ["adjacent", "fibonahi_neighborhood"]:
        gantt_name = f"gantt_chart_{neigh_mode}.png"
        save_gantt_chart_with_name(
            best_pis[neigh_mode], processing_times, cmax_summary[neigh_mode], gantt_name
        )
    print("Comparison plot and Gantt charts saved.")
    for neigh_mode in ["adjacent", "fibonahi_neighborhood"]:
        print(f"Final cmax for {labels[neigh_mode]}: {cmax_summary[neigh_mode]}")


def load_config(config_file: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main() -> None:
    # Load configuration from file
    config = load_config()

    # Extract configuration parameters
    general_config = config["general"]
    viz_config = config["visualization"]

    data = parser(
        file_path=general_config["input_file"], instance_number=general_config["instance_number"]
    )
    processing_times = data["processing_times"]

    algorithm = general_config["algorithm"]

    algorithm_common = config.get("algorithm_common", {})
    if algorithm in ("tabu_search", "simulated_annealing"):
        best_pi, best_cmax, iteration_history, cmax_history = run_algorithm(
            algorithm, processing_times, config, algorithm_common
        )
        algorithm_name = "Tabu Search" if algorithm == "tabu_search" else "Simulated Annealing"
    elif algorithm in ("tabu_search_compare", "simulated_annealing_compare"):
        if algorithm == "tabu_search_compare":
            labels = {"adjacent": "Tabu: adjacent", "fibonahi_neighborhood": "Tabu: fibonahi_neigh"}
            colors = {"adjacent": "blue", "fibonahi_neighborhood": "magenta"}
        else:
            labels = {"adjacent": "SA: adjacent", "fibonahi_neighborhood": "SA: fibonahi_neigh"}
            colors = {"adjacent": "green", "fibonahi_neighborhood": "orange"}
        run_compare_mode(algorithm, processing_times, config, algorithm_common, labels, colors)
        return

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. " f"Use 'tabu_search' or 'simulated_annealing'"
        )

    print(f"{algorithm_name} - Best sequence: {best_pi}")
    print(f"{algorithm_name} - Cmax: {best_cmax}")

    # Generate and save plots if enabled
    if viz_config["save_plots"]:
        save_gantt_chart(best_pi, processing_times, best_cmax)
        save_convergence_plot(iteration_history, cmax_history)
        print(f"Plots saved to '{viz_config['results_folder']}' folder")

    print(data["info"])


if __name__ == "__main__":
    main()
