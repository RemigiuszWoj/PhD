#!/usr/bin/env python3

import yaml

from src.parser import parser
from src.serach import simulated_annealing, tabu_search
from src.visualization import save_convergence_plot, save_gantt_chart


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

    if algorithm == "tabu_search":
        print("Running Tabu Search...")
        ts_config = config["tabu_search"]
        # Run tabu search algorithm
        best_pi, best_cmax, iteration_history, cmax_history = tabu_search(
            processing_times,
            max_time_ms=ts_config["max_time_ms"],
            tabu_tenure=ts_config["tabu_tenure"],
        )
        algorithm_name = "Tabu Search"
    elif algorithm == "simulated_annealing":
        print("Running Simulated Annealing...")
        sa_config = config["simulated_annealing"]
        # Run simulated annealing algorithm
        best_pi, best_cmax, iteration_history, cmax_history = simulated_annealing(
            processing_times,
            time_limit_ms=sa_config["time_limit_ms"],
            initial_temp=sa_config["initial_temp"],
            final_temp=sa_config["final_temp"],
            alpha=sa_config["alpha"],
        )
        algorithm_name = "Simulated Annealing"
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
