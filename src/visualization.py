"""Gantt schedule visualization.

Function ``plot_gantt(schedule, save_path=None)`` renders one bar per
operation grouped by machine on the y-axis and time on the x-axis.

Default style (can be switched to white background) originally targeted a
dark / cyberpunk theme with a purple/pink palette and outlined labels. The
function now exposes parameters to switch background, add legend, watermark
etc. For further theming simply copy and adapt.

Matplotlib is imported lazily so users not needing visualization don't need
the dependency. A clear ImportError message is raised when missing.
"""

from __future__ import annotations

from typing import Any  # noqa: F401 (placeholder for potential future types)

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import patheffects

from src.models import Schedule


def plot_gantt(
    schedule: Schedule,
    save_path: str | None = None,
    *,
    width: float = 12.0,
    row_height: float = 0.8,
    extra_height: float = 2.2,
    dpi: int = 220,
    white_background: bool = True,
    add_legend: bool = True,
    show_labels: bool = False,
    min_bar_width_full: float = 3.0,
    min_bar_width_short: float = 1.2,
    min_label_gap: float = 5.0,
    smart_collision: bool = True,
    watermark_cmax: bool = False,
    algo_name: str | None = None,
) -> None:
    """Render a Gantt chart for the given schedule.

    Parameters tuned for a reasonably large, high-DPI output. Label drawing
    is disabled by default (legend only) to avoid clutter; enable if needed.

    Args:
        schedule: `Schedule` instance with operation list.
        save_path: Output path to store image (PNG etc.); when None show it.
        width: Figure width in inches.
        row_height: Height per machine lane in inches.
        extra_height: Extra vertical padding (title/legend space) in inches.
        dpi: Resolution when saving.
        white_background: If True use white background; else dark theme.
        add_legend: Whether to include job color legend and short format note.
        show_labels: Draw per-operation labels on bars (default False).
        min_bar_width_full: Minimum bar width (time units) for full label
            J<job>O<op>.
        min_bar_width_short: Minimum width for short label J<job>; narrower
            bars receive no label.
        min_label_gap: Minimum distance between label centers on same machine.
        smart_collision: If True attempt to downgrade/omit overlapping labels.
        watermark_cmax: If True add translucent makespan watermark in
            background.
        algo_name: Optional algorithm name to inject into chart title.
    """
    ops = schedule.operations
    # Group operations by machine
    # Store operation rows per machine
    from src.models import ScheduleOperationRow  # local import to avoid cycle

    machines: dict[int, list[ScheduleOperationRow]] = {}
    for op in ops:
        machines.setdefault(op.machine, []).append(op)

    machine_ids = sorted(machines.keys())
    # Figure / background
    height = row_height * len(machine_ids) + extra_height
    fig, ax = plt.subplots(figsize=(width, height))
    if white_background:
        bg_color = "#ffffff"
        axis_label_color = "#222222"
        tick_color = "#333333"
        title_color = "#000000"
        grid_color = "#888888"
        spine_color = "#555555"
        text_color = "#000000"
        use_stroke = False
    else:  # dark theme
        # retain previous dark style when white_background is False
        bg_color = "#0d0b1e"
        axis_label_color = "#e0d7ff"
        tick_color = "#c2b8ff"
        title_color = "#ffffff"
        grid_color = "#ff00c8"
        spine_color = "#4a3b73"
        text_color = "#ffffff"
        use_stroke = True
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    y_ticks = []
    y_labels = []
    y = 0
    # Neon purple/pink palette (cycled); adjust or extend as desired
    palette = [
        "#ff00c8",  # vivid magenta
        "#d500f9",  # neon violet
        "#ff4dd2",  # light pink
        "#b000e6",  # deep violet
        "#ff1493",  # neon pink (alt tone)
        "#9c27b0",  # classic purple
        "#e040fb",  # bright pinkish violet
        "#c51162",  # crimson magenta
        "#b388ff",  # pastel lavender
        "#f500f5",  # bright neon magenta
    ]
    colors = {}

    text_effect = [patheffects.withStroke(linewidth=1.6, foreground="black")] if use_stroke else []

    # For collision detection store centers of already placed labels per row
    for m in machine_ids:
        y_ticks.append(y + 0.4)
        y_labels.append(f"M{m}")
        placed_centers: list[float] = []
        for op in machines[m]:
            # color per job from cyclic neon palette
            if op.job not in colors:
                colors[op.job] = palette[op.job % len(palette)]
            ax.add_patch(
                patches.Rectangle(
                    (op.start, y),
                    op.end - op.start,
                    0.8,
                    facecolor=colors[op.job],
                    edgecolor=colors[op.job],
                    linewidth=1.2,
                    alpha=0.92,
                )
            )
            if show_labels:
                bar_width = op.end - op.start
                center = op.start + bar_width / 2
                full_label = f"J{op.job}O{op.operation_index}"
                short_label = f"J{op.job}"

                # Decide label text by width thresholds
                if bar_width >= min_bar_width_full:
                    chosen = full_label
                elif bar_width >= min_bar_width_short:
                    chosen = short_label
                else:
                    chosen = ""  # skip

                if chosen and smart_collision:
                    # Collision: center too close to an existing label
                    too_close = any(abs(center - c) < min_label_gap for c in placed_centers)
                    if too_close:
                        # Attempt to shorten if still full label
                        if chosen == full_label and bar_width >= min_bar_width_short:
                            chosen = short_label
                            # second collision -> drop entirely
                            if any(abs(center - c) < min_label_gap for c in placed_centers):
                                chosen = ""  # drop entirely
                        else:
                            chosen = ""  # drop

                if chosen:
                    ax.text(
                        center,
                        y + 0.4,
                        chosen,
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        fontweight="bold",
                        color=text_color,
                        path_effects=text_effect,
                    )
                    placed_centers.append(center)
        y += 1

    ax.set_ylim(-0.1, len(machine_ids) + 0.1)
    ax.set_xlim(0, schedule.cmax * 1.02)
    ax.set_xlabel("Time", color=axis_label_color)
    ax.set_ylabel("Machine", color=axis_label_color)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, color=tick_color)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)
    title_core = f"Gantt chart (cmax={schedule.cmax})"
    if algo_name:
        title_core = f"Gantt chart - {algo_name} (cmax={schedule.cmax})"
    ax.set_title(
        title_core,
        color=title_color,
        pad=12,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.55, color=grid_color)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.9)

    # Legend (job color mapping) + optional label format note
    if add_legend:
        from matplotlib.patches import Patch

        job_ids_sorted = sorted(colors.keys())
        # Avoid unwieldy legend (soft limit)
        max_legend_jobs = 25
        if len(job_ids_sorted) <= max_legend_jobs:
            handles = [
                Patch(
                    facecolor=colors[j],
                    edgecolor=colors[j],
                    label=f"J{j}",
                )
                for j in job_ids_sorted
            ]
            ax.legend(
                handles=handles,
                title="Jobs",
                loc="upper right",
                fontsize=8,
                title_fontsize=9,
                framealpha=0.9,
            )
        # Label format description (bottom left)
        if show_labels:
            label_note = "Label format: J<job>O<op>; in narrow bars shortened."
            fig.text(
                0.01,
                0.01,
                label_note,
                fontsize=8,
                color=axis_label_color,
                ha="left",
                va="bottom",
            )

    # (watermark disabled by default; enable via watermark_cmax=True)
    if watermark_cmax:
        ax.text(
            0.5,
            0.5,
            f"cmax={schedule.cmax}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=120,
            color="#000000" if white_background else "#ffffff",
            alpha=0.06,
            weight="bold",
            zorder=0,
        )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_progress_curves(
    progresses: dict[str, list[int]],
    time_axes: dict[str, list[float]],
    save_path: str | None = None,
    *,
    dpi: int = 220,
    white_background: bool = True,
    minimalist: bool = True,
) -> None:
    """Plot (time, best_cmax) progress points (no connecting lines).

    Only colored markers are shown (no lines) per user preference. Neon
    violet / magenta palette consistent with Gantt styling.

    Args:
        progresses: algo -> list of best cmax values.
        time_axes: algo -> list of elapsed times (seconds).
        save_path: Optional output path.
        dpi: Resolution.
        white_background: Theme toggle.
        minimalist: If True lighter grid & hides top/right spines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bg = "#ffffff" if white_background else "#0d0b1e"
    fg = "#111111" if white_background else "#f2eeff"
    grid = "#dddddd" if white_background else "#3c3259"
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    palette = {
        "hill": "#9c27b0",  # purple
        "tabu": "#ff00c8",  # neon magenta
        "sa": "#b388ff",  # soft lavender
    }
    marker_map = {"hill": "o", "tabu": "s", "sa": "^"}
    for algo, series in progresses.items():
        if not series or algo not in time_axes:
            continue
        t = time_axes[algo]
        ax.scatter(
            t,
            series,
            s=22,
            marker=marker_map.get(algo, "o"),
            color=palette.get(algo, "#888888"),
            edgecolors=bg,
            linewidths=0.5,
            alpha=0.95,
            zorder=3,
        )
    ax.set_xlabel("Time (s)", color=fg)
    ax.set_ylabel("Best cmax", color=fg)
    ax.set_title("Best cmax vs time (progress)", color=fg, pad=10)
    ax.grid(True, linestyle=":", alpha=0.45, color=grid)
    for spine in ax.spines.values():
        spine.set_color(fg)
    if minimalist:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(colors=fg, labelsize=9)
    # Custom legend with markers
    handles = []
    for k in ("hill", "tabu", "sa"):
        if k in progresses and progresses[k]:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker_map.get(k, "o"),
                    color=palette.get(k, "#777777"),
                    label=k,
                    linestyle="-",
                    linewidth=1.2,
                    markersize=5,
                    markerfacecolor=palette.get(k, "#777777"),
                )
            )
    if handles:
        ax.legend(
            handles=handles,
            framealpha=0.85,
            facecolor=bg,
            edgecolor=grid,
            fontsize=8.5,
            title="Algorithms",
            title_fontsize=9,
            loc="best",
        )
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_iteration_progress(
    progress: list[int],
    save_path: str | None = None,
    *,
    algo_name: str | None = None,
    dpi: int = 180,
    white_background: bool = True,
    line: bool = True,
) -> None:
    """Plot cmax vs iteration for a single algorithm.

    Args:
        progress: List of best cmax values per iteration.
        save_path: Output path (PNG) or None to display.
        algo_name: Optional algorithm name for title.
        dpi: Resolution.
        white_background: Theme toggle.
        line: If True draw connecting line, else markers only.
    """
    import matplotlib.pyplot as plt  # lokalny import

    if not progress:
        return
    bg = "#ffffff" if white_background else "#0d0b1e"
    fg = "#111111" if white_background else "#f2eeff"
    grid = "#dddddd" if white_background else "#3c3259"
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    x = list(range(len(progress)))
    if line:
        ax.plot(x, progress, color="#ff00c8", linewidth=1.4, alpha=0.95)
    ax.scatter(x, progress, color="#ff00c8", s=18, edgecolors=bg, linewidths=0.4)
    ax.set_xlabel("Iteration", color=fg)
    ax.set_ylabel("Best cmax", color=fg)
    title = "cmax Progress" if not algo_name else f"cmax Progress - {algo_name}"
    ax.set_title(title, color=fg, pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, color=grid)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.tick_params(colors=fg, labelsize=9)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()


def plot_iteration_progress_multi(
    series: dict[str, list[int]],
    save_path: str | None = None,
    *,
    dpi: int = 180,
    white_background: bool = True,
) -> None:
    """Multi-series cmax vs iteration plot.

    Args:
        series: Mapping name -> list of cmax values (uniform length).
        save_path: Optional output path.
        dpi: Resolution.
        white_background: Theme toggle.
    """
    import matplotlib.pyplot as plt

    if not series:
        return
    length = len(next(iter(series.values())))
    if length == 0:
        return
    bg = "#ffffff" if white_background else "#0d0b1e"
    fg = "#111111" if white_background else "#f2eeff"
    grid = "#dddddd" if white_background else "#3c3259"
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    x = list(range(length))
    palette = [
        "#ff00c8",
        "#9c27b0",
        "#b388ff",
        "#d500f9",
        "#ff4dd2",
        "#c51162",
    ]
    for i, (name, vals) in enumerate(series.items()):
        color = palette[i % len(palette)]
        ax.plot(x, vals, label=name, color=color, linewidth=1.3, alpha=0.95)
    ax.set_xlabel("Iteration", color=fg)
    ax.set_ylabel("cmax", color=fg)
    ax.set_title("cmax series vs iteration", color=fg, pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, color=grid)
    for spine in ax.spines.values():
        spine.set_color(fg)
    ax.tick_params(colors=fg, labelsize=9)
    ax.legend(framealpha=0.85, fontsize=8.5)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
    else:
        plt.show()
