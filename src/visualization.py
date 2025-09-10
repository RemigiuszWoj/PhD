import glob
import os
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt


def clear_old_plots():
    """Remove old plot files from results directory."""
    results_dir = "results"
    if os.path.exists(results_dir):
        # Remove old gantt charts
        for file in glob.glob(os.path.join(results_dir, "gantt_chart_*.png")):
            os.remove(file)
        # Remove old convergence plots
        for file in glob.glob(os.path.join(results_dir, "convergence_plot_*.png")):
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


def save_gantt_chart(pi: List[int], processing_times: List[List[int]], cmax: int):
    """Create and save Gantt chart for the given permutation."""
    # Clear old plots first
    clear_old_plots()

    m = len(processing_times)  # number of machines
    n = len(pi)  # number of jobs

    start_times, completion_times = calculate_schedule(pi, processing_times)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for different jobs
    colors = [plt.cm.tab20(i / n) for i in range(n)]

    # Plot each job on each machine
    for i in range(m):
        for j in range(n):
            job_id = pi[j]
            start = start_times[i][j]
            duration = processing_times[i][job_id]

            # Draw the bar
            ax.barh(
                i,
                duration,
                left=start,
                height=0.8,
                color=colors[job_id],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

            # # Add job label
            # ax.text(
            #     start + duration / 2, i, f"J{job_id}", ha="center", va="center", fontweight="bold"
            # )

    # Customize the plot
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Machine", fontsize=12)
    ax.set_title(f"Gantt Chart - Cmax = {cmax}", fontsize=14, fontweight="bold")
    ax.set_yticks(range(m))
    ax.set_yticklabels([f"M{i}" for i in range(m)])
    ax.grid(True, alpha=0.3, axis="x")

    # Set y-axis limits
    ax.set_ylim(-0.5, m - 0.5)

    # Add legend
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=colors[i], alpha=0.8, edgecolor="black", label=f"Job {i}"
        )
        for i in range(n)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gantt_chart_{timestamp}.png"
    filepath = os.path.join("results", filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gantt chart saved as: {filepath}")


def save_convergence_plot(iterations: List[int], cmax_values: List[int]):
    """Create and save convergence plot showing Cmax improvement over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the convergence
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

    # Customize the plot
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Cmax", fontsize=12)
    ax.set_title("Tabu Search Convergence", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Add annotations for first and last values
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

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"convergence_plot_{timestamp}.png"
    filepath = os.path.join("results", filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Convergence plot saved as: {filepath}")
