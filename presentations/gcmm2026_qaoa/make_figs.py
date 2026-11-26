"""Figures for the GCMM 2026 talk (Paper 59, qaoa_gcmm).

Numbers come from the tables of articles/qaoa_gcmm/main.tex; per-instance
points from the ibm_fez runs (git archive origin/gcmm-rev
articles/qaoa_gcmm/data/qaoa_gate_hw_n20 -> HW_DIR). The angle landscape is
recomputed with the calibration code on the same training set (identity
permutation, window 6, step 3).

Run from the repo root:  python3 presentations/gcmm2026_qaoa/make_figs.py HW_DIR TAILLARD_REPO
"""
import glob
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

OUT = Path(__file__).parent / "img"
HW_DIR = Path(sys.argv[1])
TAI_REPO = Path(sys.argv[2])  # repo with data/tai20_*.txt

PWR = "#9A342D"
NB = ["Adjacent", "Fibonacci", "Dynasearch", "Motzkin"]
COL = {"Adjacent": "#5B7DB1", "Fibonacci": PWR, "Dynasearch": "#E08A1E", "Motzkin": "#2A9D8F"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name, transparent=False, facecolor="white")
    plt.close(fig)


# ---------- 1. conflict graphs on a 6-job window ----------
def conflict_graphs(grid=(1, 4), size=(16, 4.4), name="conflict_graphs.png"):
    n = 6
    adj = list(range(n - 1))
    ivs = [(i, j) for i, j in itertools.combinations(range(n), 2)]
    overl = lambda a, b: max(a[0], b[0]) <= min(a[1], b[1])
    def motz(a, b):
        if {a[0], a[1]} & {b[0], b[1]}:
            return True
        return a[0] < b[0] < a[1] < b[1] or b[0] < a[0] < b[1] < a[1]
    graphs = {
        "Adjacent": (adj, lambda a, b: True),
        "Fibonacci": (adj, lambda a, b: abs(a - b) == 1),
        "Dynasearch": (ivs, overl),
        "Motzkin": (ivs, motz),
    }
    fig, axes = plt.subplots(*grid, figsize=size)
    for ax, nb in zip(np.ravel(axes), NB):
        nodes, conf = graphs[nb]
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from((a, b) for a, b in itertools.combinations(range(len(nodes)), 2)
                         if conf(nodes[a], nodes[b]))
        pos = nx.circular_layout(G)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COL[nb], alpha=.55, width=1.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=COL[nb], node_size=110)
        dens = 2 * G.number_of_edges() / (len(nodes) * (len(nodes) - 1))
        ax.set_title(f"{nb}\n{len(nodes)} qubits, {G.number_of_edges()} gates\n{dens:.0%} of pairs",
                     color=COL[nb], fontweight="bold", fontsize="medium")
        ax.axis("off")
    save(fig, name)


# ---------- 2. p=1 angle landscape ----------
def landscape(grid=(1, 4), size=(17, 4.2), name="landscape.png"):
    import src.experiments.qaoa_calibrate_angles as C
    from src.neighborhoods.gate_qaoa.angles import _objective, _prepare
    C.REPO = TAI_REPO
    inst = C._load_instances(C.DEFAULT_INSTANCES)
    angles = json.load(open(TAI_REPO / "data/qaoa_angles.json"))
    G = np.linspace(0, np.pi, 49)
    B = np.linspace(0, np.pi / 2, 25)
    fig, axes = plt.subplots(*grid, figsize=size, constrained_layout=True)
    for ax, nb in zip(np.ravel(axes), NB):
        P = _prepare(C.build_training(nb.lower(), inst, window=6, step=3, perms=1))
        Z = np.array([[_objective(P, [g], [b]) for g in G] for b in B])
        im = ax.pcolormesh(G / np.pi, B / np.pi, Z, cmap="RdBu_r", shading="auto",
                           vmin=-abs(Z).max(), vmax=abs(Z).max())
        a = angles[nb.lower()]["1"]
        ax.plot(a["gamma"][0] / np.pi, a["beta"][0] / np.pi, marker="*", ms=20,
                mec="black", mfc="gold")
        ax.set_title(nb, color=COL[nb], fontweight="bold")
        ax.set_xlabel(r"$\gamma/\pi$")
        ax.set_ylabel(r"$\beta/\pi$")
    fig.colorbar(im, ax=axes, shrink=.85, label=r"mean $\langle H_C\rangle$ (normalized)")
    save(fig, name)


def landscape_one(nb="Fibonacci", size=(10, 6.5), name="p_landscape1.png"):
    """One annotated panel: what the angle map means (poster)."""
    import src.experiments.qaoa_calibrate_angles as C
    from src.neighborhoods.gate_qaoa.angles import _objective, _prepare
    C.REPO = TAI_REPO
    inst = C._load_instances(C.DEFAULT_INSTANCES)
    a = json.load(open(TAI_REPO / "data/qaoa_angles.json"))[nb.lower()]["1"]
    G = np.linspace(0, np.pi, 49)
    B = np.linspace(0, np.pi / 2, 25)
    P = _prepare(C.build_training(nb.lower(), inst, window=6, step=3, perms=1))
    Z = np.array([[_objective(P, [g], [b]) for g in G] for b in B])
    # the frozen angles must be the minimum of the paper's 25x13 calibration grid (every 2nd point here)
    bi, gi = np.unravel_index(Z[::2, ::2].argmin(), Z[::2, ::2].shape)
    assert abs(G[::2][gi] - a["gamma"][0]) < 1e-6 and abs(B[::2][bi] - a["beta"][0]) < 1e-6, (gi, bi, a)
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    im = ax.pcolormesh(G / np.pi, B / np.pi, Z, cmap="RdBu_r", shading="auto",
                       vmin=-abs(Z).max(), vmax=abs(Z).max())
    ax.plot(a["gamma"][0] / np.pi, a["beta"][0] / np.pi, marker="*", ms=34, mec="black", mfc="gold")
    ax.annotate("frozen angles,\nused for every move", (a["gamma"][0] / np.pi, a["beta"][0] / np.pi),
                xytext=(0.47, 0.43), fontsize="small", ha="center",
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.2, 0.06, "worse than\nrandom", ha="center", fontsize="small", color="#7a0019")
    ax.text(0.88, 0.04, "better", ha="center", fontsize="small", color="#08306b")
    ax.set_xlabel(r"cost angle $\gamma/\pi$")
    ax.set_ylabel(r"mixer angle $\beta/\pi$")
    ax.set_title(f"{nb}: expected cost of the QAOA state", fontsize="medium")
    cb = fig.colorbar(im, ax=ax, shrink=.9)
    cb.set_label("mean expected cost (0 = random guess)")
    save(fig, name)


# ---------- 3. compiled depth across coupling maps (Table 1) ----------
MAPS = ["Heron r2\n(heavy-hex)", "Crystal 20\n(square)", "Crystal 54\n(square)", "Star 16\n(resonator)"]
DEPTH = {"Fibonacci": [58, 22, 22, 38], "Adjacent": [337, 200, 188, 227],
         "Motzkin": [2208, 1280, 1177, 1438], "Dynasearch": [2509, 1466, 1383, 1758]}
TWOQ = {"Fibonacci": [14, 14, 14, 28], "Adjacent": [195, 153, 159, 180],
        "Motzkin": [1455, 1080, 1077, 1180], "Dynasearch": [1668, 1287, 1275, 1438]}


def depth(grid=(1, 2), size=(16, 5.2), name="depth.png"):
    order = ["Fibonacci", "Adjacent", "Motzkin", "Dynasearch"]
    x = np.arange(len(MAPS))
    w = .2
    fig, axes = plt.subplots(*grid, figsize=size)
    for ax, data, lab in zip(axes, [DEPTH, TWOQ], ["compiled depth", "two-qubit gates"]):
        for k, nb in enumerate(order):
            bars = ax.bar(x + (k - 1.5) * w, data[nb], w, color=COL[nb], label=nb)
            ax.bar_label(bars, fontsize="x-small", padding=3, rotation=90)
        ax.set_yscale("log")
        ax.set_xticks(x, MAPS)
        ax.set_ylabel(lab + " (log)")
        ax.set_ylim(8, 15000)
    axes[0].legend(ncol=4 if grid == (1, 2) else 2, loc="upper left", frameon=False, fontsize="small")
    if grid != (1, 2):
        axes[0].set_ylim(8, 90000)
    save(fig, name)


# ---------- 4. coupling maps ----------
def coupling_maps(size=(16, 5), name="coupling_maps.png"):
    from qiskit.transpiler import CouplingMap
    fig, axes = plt.subplots(1, 3, figsize=size)
    hh = nx.Graph(list(CouplingMap.from_heavy_hex(3).get_edges()))
    sq = nx.grid_2d_graph(4, 5)
    st = nx.star_graph(16)
    specs = [(hh, nx.kamada_kawai_layout(hh), "heavy-hex (Heron r2)"),
             (sq, {v: v for v in sq}, "square (Crystal 20, 54)"),
             (st, nx.spring_layout(st, seed=1), "star (Star 16)")]
    for ax, (G, pos, t) in zip(axes, specs):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#888", width=1.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=PWR, node_size=80)
        if G is st:
            nx.draw_networkx_nodes(G, pos, nodelist=[0], ax=ax, node_color="#E08A1E",
                                   node_size=600, node_shape="s")
        ax.set_title(t, fontsize="large")
        ax.axis("off")
        ax.set_aspect("equal")
    save(fig, name)


# ---------- 5. hardware results (ibm_fez, per instance) ----------
def load_hw():
    rows = []
    for f in glob.glob(str(HW_DIR / "*/result.json")):
        r = json.load(open(f))
        c = r["config"]
        lb = r["lower_bound"]
        rows.append(dict(algo=c["algorithm"], nb=c["neighborhood"].replace("gate_", "").capitalize(),
                         inst=c["instance_number"], prd=r["gap_percent"],
                         start=(r["cmax_history"][0] - lb) / lb * 100, t=r["total_time_ms"] / 1000))
    assert len(rows) == 80, len(rows)
    return rows


def hw_prd(rows, grid=(1, 2), size=(16, 5.4), name="hw_prd.png"):
    kw = {"width_ratios": [1, 1.6]} if grid == (1, 2) else {}
    fig, axes = plt.subplots(*grid, figsize=size, gridspec_kw=kw)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    x = np.arange(4)
    # dot plot: a bar axis cut at 15 would exaggerate the differences
    for a, off, mk in [("ils", -.12, "o"), ("sa", .12, "s")]:
        m = [np.mean([r["prd"] for r in rows if r["nb"] == nb and r["algo"] == a]) for nb in NB]
        ax.vlines(x + off, m, 27.90, color=[COL[n] for n in NB], lw=2, alpha=.5)
        ax.scatter(x + off, m, s=160, marker=mk, color=[COL[n] for n in NB], zorder=3,
                   edgecolor="black" if a == "sa" else "none", label=a.upper())
        for xi, v in zip(x + off, m):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(-12 if a == "ils" else 12, -6), va="top",
                        ha="right" if a == "ils" else "left", fontsize="small")
    ax.axhline(27.90, color="black", ls="--", lw=1.5)
    ax.text(3.45, 28.25, "identity start 27.9", ha="right", fontsize="small")
    ax.set_xticks(x, NB)
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(19.5, 29.5)
    ax.set_ylabel("mean PRD [%]")
    leg = ax.legend(frameon=False, loc="lower left", ncol=2)
    for h in leg.legend_handles:
        h.set_color("gray")
    ax.set_title("mean of 10 instances: ILS (circle), SA (square)")
    if len(axes) == 1:
        fig.tight_layout()
        save(fig, name)
        return
    ax = axes[1]
    ils = [r for r in rows if r["algo"] == "ils"]
    for i in range(10):
        s = next(r["start"] for r in ils if r["inst"] == i)
        ax.plot([i - .3, i + .3], [s, s], color="black", lw=2.5)
        for k, nb in enumerate(NB):
            p = next(r["prd"] for r in ils if r["inst"] == i and r["nb"] == nb)
            ax.plot([i - .22 + .147 * k] * 2, [s, p], color=COL[nb], lw=1.2, alpha=.6)
            ax.plot(i - .22 + .147 * k, p, "o", color=COL[nb], ms=7, label=nb if i == 0 else None)
    ax.set_xticks(range(10), [f"ta{i + 1:03d}" for i in range(10)], rotation=30)
    ax.set_ylabel("PRD [%]")
    ax.set_title("per instance (ILS): start (bar) → one move")
    ax.legend(ncol=4, frameon=False, fontsize="small", loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.tight_layout()
    save(fig, name)


def hw_time(rows, size=(11, 5), name="hw_time.png"):
    fig, ax = plt.subplots(figsize=size)
    rng = np.random.default_rng(0)
    qpu = {"Adjacent": 3, "Fibonacci": 3, "Dynasearch": 7, "Motzkin": 7}
    for k, nb in enumerate(NB):
        t = [r["t"] for r in rows if r["nb"] == nb]
        ax.bar(k, qpu[nb], .6, color=COL[nb], alpha=.35)
        ax.scatter(k + rng.uniform(-.2, .2, len(t)), t, color=COL[nb], s=40, zorder=3)
        ax.plot([k - .3, k + .3], [np.median(t)] * 2, color="black", lw=2.5, zorder=4)
    ax.axhline(10, color=PWR, ls="--", lw=2)
    ax.text(3.45, 10.6, "budget 10 s", color=PWR, ha="right")
    ax.set_yscale("log")
    ax.set_xticks(range(4), NB)
    ax.set_ylabel("wall-clock time of one move [s] (log)")
    ax.set_title("dots: runs · bar: billed quantum time · line: median", fontsize="medium")
    save(fig, name)


# ---------- 6. annealer vs gate (Table 3, ILS) ----------
def compare(size=(14, 5.4), name="compare.png"):
    data = {"Classical (10 s)": [3.41, 3.82, 2.68, 2.64], "Annealer (10 s)": [20.17, 17.94, 25.85, 19.02],
            "Annealer, split into windows (10 s)": [20.03, 18.20, 26.24, 25.22], "Gate QAOA (one move, 10–288 s)": [25.04, 23.41, 22.12, 23.12]}
    cols = ["#BBBBBB", "#B4A7D6", "#5E3C99", "#F72585"]
    x = np.arange(4)
    w = .2
    fig, ax = plt.subplots(figsize=size)
    for k, (lab, v) in enumerate(data.items()):
        b = ax.bar(x + (k - 1.5) * w, v, w, color=cols[k], label=lab)
        ax.bar_label(b, fmt="%.1f", fontsize="x-small", padding=3, rotation=90)
    ax.set_xticks(x, NB)
    ax.set_ylabel("PRD [%]")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.22))
    ax.set_ylim(0, 32)
    save(fig, name)


# ---------- 7. three coupling maps on hardware + threshold ----------
def arch(grid=(1, 2), size=(16, 5.4), name="arch.png"):
    maps = ["Heron r2", "Crystal 20", "Star 16"]
    prd = {"Adjacent": [12.82, 12.82, 12.82], "Fibonacci": [12.26, 12.01, 12.01],
           "Dynasearch": [12.42, 15.10, 12.09], "Motzkin": [16.96, 14.85, 13.31]}
    fig, axes = plt.subplots(*grid, figsize=size)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for nb in NB:
        ax.plot(maps, prd[nb], "o--", color=COL[nb], lw=1.2, ms=13, label=nb)  # chips are categories, not a trend
    ax.axhline(17.53, color="black", ls="--", lw=1.2)
    ax.text(2, 17.2, "identity start 17.5", ha="right", fontsize="small")
    ax.set_ylabel("PRD after one move [%]")
    ax.set_title("first instance: same QUBOs, angles, shots")
    ax.legend(frameon=False, ncol=4, fontsize="small", loc="upper center", bbox_to_anchor=(.5, -.1))
    if len(axes) == 1:
        fig.tight_layout()
        save(fig, name)
        return
    ax = axes[1]
    for nb in NB:
        cz = TWOQ[nb][0]
        spread = max(prd[nb]) - min(prd[nb])
        ax.scatter(cz, spread, s=260, color=COL[nb], zorder=3)
        ax.annotate(nb, (cz, spread), textcoords="offset points", xytext=(10, 8), color=COL[nb],
                    fontweight="bold")
    ax.axvspan(8, 400, color="#2A9D8F", alpha=.08)
    ax.axvspan(400, 4000, color=PWR, alpha=.08)
    ax.text(40, 3.3, "map invisible", ha="center", fontsize="medium")
    ax.text(1300, 1.0, "map decides", ha="center", fontsize="medium")
    ax.set_xscale("log")
    ax.set_xlim(8, 4000)
    ax.set_ylim(-.3, 4.2)
    ax.set_xlabel("two-qubit gates per move on Heron r2 (log)")
    ax.set_ylabel("PRD spread across 3 maps [pp]")
    ax.set_title("threshold, not gradient (ta001)")
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    conflict_graphs()
    depth()
    coupling_maps()
    compare()
    arch()
    rows = load_hw()
    hw_prd(rows)
    hw_time(rows)
    # poster variants: 2x2 grids, stacked panels, fonts readable on A0
    with plt.rc_context({"font.size": 20}):
        conflict_graphs((2, 2), (9, 9.5), "p_conflict_graphs.png")
        depth((2, 1), (10, 11), "p_depth.png")
        arch((1, 1), (10, 6.5), "p_arch1.png")
        hw_prd(rows, (1, 1), (10, 6.5), "p_hw_prd1.png")
        hw_time(rows, (10, 6), "p_hw_time.png")
        compare((10, 6.5), "p_compare.png")
        coupling_maps((14, 5.5), "p_coupling_maps.png")
    if "--no-landscape" not in sys.argv:
        landscape()
        with plt.rc_context({"font.size": 20}):
            landscape((2, 2), (10, 9), "p_landscape.png")
            landscape_one()
