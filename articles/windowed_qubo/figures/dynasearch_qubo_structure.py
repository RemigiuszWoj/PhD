"""Visualisation of the Dynasearch QUBO envelope structure.

Idea (from the whiteboard sketch):
    The Dynasearch QUBO has K = O(n^2) variables indexed by interval
    pairs (i, j), i < j. Storing the full K x K matrix takes O(K^2)
    memory. However, when pairs are ordered lexicographically, every
    row's non-zero entries form a *contiguous interval* of columns
    (the matrix has an "envelope" / skyline structure). Hence the
    matrix can be reconstructed from O(K) numbers — the start and end
    column of each row's envelope — and the rest of the cells (white
    in the figure) need not be stored at all.

The figure renders, side by side:
    (a) the dense K x K matrix coloured by entry type (diagonal,
        overlap penalty, structural zero), with thin block outlines
        grouping pairs that share the same left endpoint i;
    (b) the same matrix with the envelope boundary overlaid as a
        single closed curve and the diagonal traversal highlighted —
        graphically reading off the O(K) information that suffices to
        rebuild Q.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------
# Dynasearch QUBO mask
# --------------------------------------------------------------------------
def build_pairs(n: int) -> list[tuple[int, int]]:
    """All (i, j) with 0 <= i < j <= n-1 in lexicographic order."""
    return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]


def overlap(p: tuple[int, int], q: tuple[int, int]) -> bool:
    """Interval pairs overlap when they share at least one index."""
    i1, j1 = p
    i2, j2 = q
    return max(i1, i2) <= min(j1, j2)


def build_qubo_mask(n: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """mask[k, l] is 1 on the diagonal, 2 on overlap penalty cells, 0 elsewhere."""
    pairs = build_pairs(n)
    K = len(pairs)
    mask = np.zeros((K, K), dtype=int)
    for k in range(K):
        mask[k, k] = 1
        for l in range(k + 1, K):
            if overlap(pairs[k], pairs[l]):
                mask[k, l] = 2
                mask[l, k] = 2
    return mask, pairs


def envelope_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row, return the (l_min, l_max) of non-zero columns."""
    K = mask.shape[0]
    l_min = np.empty(K, dtype=int)
    l_max = np.empty(K, dtype=int)
    for k in range(K):
        nz = np.flatnonzero(mask[k])
        l_min[k] = nz[0]
        l_max[k] = nz[-1]
    return l_min, l_max


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
COLOUR_ZERO = "#F4F4F4"
COLOUR_DIAG = "#1f77b4"
COLOUR_PEN = "#ff7f0e"
COLOUR_PEN_OUT = "#FCE7CC"    # softer overlap penalty for (b)
COLOUR_ENVELOPE = "#2ca02c"
COLOUR_BLOCK = "#777777"


def _row_to_y(k: int, K: int) -> float:
    """Convert matrix row index k (0=top) into matplotlib y-coord (origin bottom)."""
    return K - 1 - k


def draw_grid(
    ax,
    mask: np.ndarray,
    pairs: list[tuple[int, int]],
    *,
    pen_colour: str = COLOUR_PEN,
) -> None:
    K = mask.shape[0]
    palette = {0: COLOUR_ZERO, 1: COLOUR_DIAG, 2: pen_colour}

    for k in range(K):
        for l in range(K):
            ax.add_patch(
                mpatches.Rectangle(
                    (l, _row_to_y(k, K)), 1, 1,
                    facecolor=palette[mask[k, l]],
                    edgecolor="white", linewidth=0.6,
                )
            )

    # block outlines: contiguous groups of equal left endpoint i
    sizes: list[int] = []
    cur_i, count = pairs[0][0], 0
    for i, _ in pairs:
        if i == cur_i:
            count += 1
        else:
            sizes.append(count)
            cur_i, count = i, 1
    sizes.append(count)

    offset = 0
    for size in sizes:
        ax.add_patch(
            mpatches.Rectangle(
                (offset, _row_to_y(offset + size - 1, K)),
                size, size,
                facecolor="none", edgecolor=COLOUR_BLOCK, linewidth=1.5,
            )
        )
        offset += size

    ax.set_xlim(0, K)
    ax.set_ylim(0, K)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_envelope(ax, mask: np.ndarray) -> None:
    """Trace the envelope boundary of non-zero entries as one closed curve."""
    K = mask.shape[0]
    l_min, l_max = envelope_bounds(mask)

    # Build the polygon outline: down the right boundary, then back up the left.
    right = [(l_max[k] + 1, _row_to_y(k, K) + 1) for k in range(K)]
    right += [(l_max[k] + 1, _row_to_y(k, K)) for k in range(K)]
    # Sort right boundary in actual visiting order:
    pts_right: list[tuple[float, float]] = []
    for k in range(K):
        pts_right.append((l_max[k] + 1, _row_to_y(k, K) + 1))
        pts_right.append((l_max[k] + 1, _row_to_y(k, K)))
    pts_left: list[tuple[float, float]] = []
    for k in reversed(range(K)):
        pts_left.append((l_min[k], _row_to_y(k, K)))
        pts_left.append((l_min[k], _row_to_y(k, K) + 1))
    polygon = pts_right + pts_left
    xs, ys = zip(*polygon)
    ax.plot(xs + (xs[0],), ys + (ys[0],),
            color=COLOUR_ENVELOPE, linewidth=2.2, zorder=5)

    # Mark per-row bounds with small dots so the O(K) data is visible.
    for k in range(K):
        y = _row_to_y(k, K) + 0.5
        ax.scatter([l_min[k] + 0.5, l_max[k] + 0.5], [y, y],
                   color=COLOUR_ENVELOPE, s=18, zorder=6,
                   edgecolor="white", linewidth=0.6)


def label_axes(ax, pairs: list[tuple[int, int]]) -> None:
    K = len(pairs)
    boundaries = [k for k in range(1, K) if pairs[k][0] != pairs[k - 1][0]]
    ticks = [0] + boundaries + [K]
    centres = [(ticks[i] + ticks[i + 1]) / 2 for i in range(len(ticks) - 1)]
    labels = [rf"$i={pairs[ticks[i]][0]}$" for i in range(len(centres))]
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([K - c for c in centres])
    ax.set_yticklabels(labels, fontsize=8)


# --------------------------------------------------------------------------
# Main figure
# --------------------------------------------------------------------------
def main() -> None:
    n = 6
    mask, pairs = build_qubo_mask(n)
    K = len(pairs)
    l_min, l_max = envelope_bounds(mask)
    nnz = int((mask != 0).sum())
    width = (l_max - l_min + 1).sum()
    full_storage = K * K
    envelope_storage = 2 * K   # two ints per row

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    # ---- (a) Full block view ----
    draw_grid(axes[0], mask, pairs)
    label_axes(axes[0], pairs)
    axes[0].set_title(
        rf"(a) Full Dynasearch QUBO,  $n={n}$,  $K={K}$",
        fontsize=11, pad=12,
    )
    axes[0].set_xlabel("column $l$", fontsize=9)
    axes[0].set_ylabel("row $k$", fontsize=9)

    legend_a = [
        mpatches.Patch(facecolor=COLOUR_DIAG, edgecolor="white",
                       label=r"diagonal $\delta_k$"),
        mpatches.Patch(facecolor=COLOUR_PEN, edgecolor="white",
                       label=r"overlap penalty $\lambda$"),
        mpatches.Patch(facecolor=COLOUR_ZERO, edgecolor="white",
                       label="structural zero"),
        mpatches.Patch(facecolor="none", edgecolor=COLOUR_BLOCK,
                       label=r"block of fixed left endpoint $i$"),
    ]
    axes[0].legend(handles=legend_a, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08), ncol=2,
                   frameon=False, fontsize=8)

    # ---- (b) Envelope view ----
    draw_grid(axes[1], mask, pairs, pen_colour=COLOUR_PEN_OUT)
    overlay_envelope(axes[1], mask)
    label_axes(axes[1], pairs)
    axes[1].set_title(
        r"(b) Envelope curve: $\mathcal{O}(K)$ memory suffices",
        fontsize=11, pad=12,
    )
    axes[1].set_xlabel("column $l$", fontsize=9)
    axes[1].set_ylabel("row $k$", fontsize=9)

    legend_b = [
        mpatches.Patch(facecolor=COLOUR_DIAG, edgecolor="white",
                       label=r"diagonal $\delta_k$"),
        mpatches.Patch(facecolor=COLOUR_PEN_OUT, edgecolor="white",
                       label=r"$\lambda$ inside envelope"),
        plt.Line2D([0], [0], color=COLOUR_ENVELOPE, marker="o",
                   linewidth=2.2, markersize=5,
                   label=r"envelope $(l_{\min}(k),\, l_{\max}(k))$"),
    ]
    axes[1].legend(handles=legend_b, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08), ncol=3,
                   frameon=False, fontsize=8)

    # Footer caption
    compression = full_storage / envelope_storage
    info = (
        rf"Non-zero entries: ${nnz}$ out of $K \cdot K = {full_storage}$.  "
        rf"Envelope needs $2K = {envelope_storage}$ integers — "
        rf"compression about ${compression:.1f}$x at $n={n}$, "
        r"growing as $\Theta(K)$ for larger $n$."
    )
    fig.text(0.5, -0.02, info, ha="center", fontsize=9)

    plt.tight_layout()
    out = Path(__file__).parent / "dynasearch_qubo_structure.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.png')}")
    print(f"K = {K},  nnz = {nnz},  full = {full_storage},  "
          f"envelope = {envelope_storage}")


if __name__ == "__main__":
    main()
