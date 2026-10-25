"""Figures and data tables of the depth study, in English and Polish.

Every number comes from the measurement files in ../measurements; nothing is
typed in by hand. Style as in the ACIIDS paper: black ink, the neighborhood
carried by the marker shape, TrueType fonts (Springer rejects Type 3).

Outputs (figures/ and tables/):
  fig1_e1_<lang>.pdf     E1: P_opt relative to uniform sampling vs p, three angle tables
  fig2_hits_<lang>.pdf   E1: chance that the best of S shots is optimal, vs S, per p
  fig4_moves_<lang>.pdf  E5: moves in the 10 s budget and time per move, vs p
  tab1_e3.tex            E3: CZ gates per window on the star and the lattice (body only)
  tab2_prd.tex           E5: median PRD by p, plus rows to fill (E5n, Sirius) (body only)

Usage: ../../../.venv311/bin/python3 make_figures.py
"""
import gzip
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 7.5
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent / "measurements"
TAB = HERE.parent / "tables"
INK = "#111111"
NB = ["adjacent", "fibonacci", "dynasearch", "motzkin"]
NAME = {"adjacent": "Adjacent", "fibonacci": "Fibonacci", "dynasearch": "Dynasearch", "motzkin": "Motzkin"}
MARK = {"adjacent": "o", "fibonacci": "s", "dynasearch": "^", "motzkin": "D"}
TABLE_STYLE = {"small": ("-", INK), "big": ("--", "#555555"), "ils": (":", "#888888")}
WIDTH = 4.8                                    # LNCS text width, inches
TXT = {
    "en": {"ratio": r"$P_{\mathrm{opt}}$ / uniform (median)", "p": "QAOA depth $p$",
           "hit": "P(best of $S$ optimal)", "S": "shots $S$",
           "moves": "moves in 10 s", "ms": "ms per move",
           "tables": {"small": "small windows", "big": "big windows", "ils": "ILS windows"},
           "uniform": "uniform ($p=0$)"},
    "pl": {"ratio": r"$P_{\mathrm{opt}}$ / losowanie (mediana)", "p": "głębokość QAOA $p$",
           "hit": "P(najlepsza z $S$ opt.)", "S": "liczba shotów $S$",
           "moves": "ruchów w 10 s", "ms": "ms na ruch",
           "tables": {"small": "małe okna", "big": "duże okna", "ils": "okna z ILS"},
           "uniform": "losowanie ($p=0$)"},
}


def rows(name):
    """Rows of a jsonl measurement file, plain or gzipped."""
    path = MEAS / name
    opener = gzip.open if path.suffix == ".gz" else open
    if not path.exists() and (MEAS / (name + ".gz")).exists():
        path, opener = MEAS / (name + ".gz"), gzip.open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_e1():
    """{(neigh, table, p): [rows]} for windows with an improving move."""
    out = defaultdict(list)
    for r in rows("e1_window_quality.jsonl"):
        if r["E_min"] < 0:
            out[(r["neigh"], r["table"], r["p"])].append(r)
    return out


def fig1(e1, lang):
    t = TXT[lang]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.8), sharey=True)
    uni = {nb: {(r["file"], r["inst"], r["perm"], r["window"]): r["P_opt"]
                for r in e1[(nb, "uniform", 0)]} for nb in NB}
    for ax, nb in zip(axes, NB):
        for table, (ls, col) in TABLE_STYLE.items():
            ps = sorted(p for (n, tb, p) in e1 if n == nb and tb == table)
            if table == "big" and nb in ("dynasearch", "motzkin"):
                continue                       # identical to "ils" for the dense neighborhoods
            ys = [statistics.median(r["P_opt"] / uni[nb][(r["file"], r["inst"], r["perm"], r["window"])]
                                    for r in e1[(nb, table, p)]) for p in ps]
            ax.plot(ps, ys, ls=ls, color=col, marker=MARK[nb], ms=2.5, lw=0.9, label=t["tables"][table])
        ax.axhline(1.0, color="#999999", lw=0.6)
        ax.set_yscale("log")
        ax.set_title(NAME[nb], fontsize=7.5)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_xlabel(t["p"])
    axes[0].set_ylabel(t["ratio"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / f"fig1_e1_{lang}.pdf")
    plt.close(fig)


def fig2(e1, lang):
    """Mean probability that the best of S samples is optimal, from the exact P_opt."""
    t = TXT[lang]
    best = {"adjacent": "big", "fibonacci": "small", "dynasearch": "small", "motzkin": "small"}
    S = np.unique(np.round(np.logspace(0, np.log10(4096), 60)).astype(int))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.8), sharey=True)
    grays = {0: "#aaaaaa", 1: "#777777", 2: "#444444", 3: INK, 5: INK}
    styles = {0: ":", 1: "--", 2: "-.", 3: "-", 5: (0, (1, 0.6))}
    for ax, nb in zip(axes, NB):
        for p in (0, 1, 2, 3, 5):
            key = (nb, "uniform", 0) if p == 0 else (nb, best[nb], p)
            if key not in e1:
                continue
            P = np.array([r["P_opt"] for r in e1[key]])
            hit = [float(np.mean(1.0 - (1.0 - P) ** s)) for s in S]
            ax.plot(S, hit, ls=styles[p], color=grays[p], lw=0.9,
                    label=t["uniform"] if p == 0 else f"$p={p}$")
        ax.set_xscale("log")
        ax.set_xticks([1, 16, 256, 4096])
        ax.set_xticklabels(["1", "16", "256", "4096"])
        ax.set_title(NAME[nb], fontsize=7.5)
        ax.set_xlabel(t["S"])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(t["hit"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, ncol=5, loc="upper center")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / f"fig2_hits_{lang}.pdf")
    plt.close(fig)


def fig3(e1_all, lang):
    """E4: share of windows where the lowest-energy of 4096 samples is optimal, versus p:
    noiseless (exact, from E1), noisy simulated Sirius (calibrated) and thermal (instance 0 only)."""
    t = TXT[lang]
    best = {"adjacent": "big", "fibonacci": "small", "dynasearch": "small", "motzkin": "small"}
    e4 = defaultdict(list)
    for r in rows("e4_window_noise.jsonl"):
        e4[(r["neigh"], r["table"], r["p"], r["mode"])].append(r)
    lab = {"en": ("noiseless", "noisy (calibrated)", "noisy (thermal, inst. 0)", "best of 4096 optimal"),
           "pl": ("bez szumu", "z szumem (skalibrowany)", "z szumem (termiczny, inst. 0)", "najlepsza z 4096 optymalna")}[lang]
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.8), sharey=True)
    for ax, nb in zip(axes, NB):
        tb = best[nb]
        ps = [0] + [p for p in range(1, 7) if (nb, tb, p, "calibrated") in e4]
        def key(p, mode):
            return (nb, "uniform", 0, mode) if p == 0 else (nb, tb, p, mode)
        noisy = [np.mean([r["pick_is_opt"] for r in e4[key(p, "calibrated")]]) for p in ps]
        ideal = []
        for p in ps:
            rs = e4[key(p, "calibrated")]
            ideal.append(np.mean([e1_all[(r["file"], r["inst"], nb, r["window"], "uniform" if p == 0 else tb, p)]["hit_4096"]
                                  for r in rs]))
        ax.plot(ps, ideal, ls="--", color="#777777", marker=MARK[nb], ms=2.3, lw=0.9, label=lab[0])
        ax.plot(ps, noisy, ls="-", color=INK, marker=MARK[nb], ms=2.3, lw=0.9, label=lab[1])
        th = [(p, e4[key(p, "thermal")]) for p in (1, 3, 6) if e4.get(key(p, "thermal"))]
        if th:
            ax.plot([p for p, _ in th], [np.mean([r["pick_is_opt"] for r in rs]) for _, rs in th],
                    ls="none", color=INK, marker="x", ms=3.5, label=lab[2])
        ax.set_title(NAME[nb], fontsize=7.5)
        ax.set_xticks(range(0, 7))
        ax.set_xlabel(t["p"])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(lab[3])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, ncol=3, loc="upper center")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / f"fig3_noise_{lang}.pdf")
    plt.close(fig)


def load_e1_all():
    out = {}
    for r in rows("e1_window_quality.jsonl"):
        if r["perm"] == "identity" and r["inst"] <= 3:
            out[(r["file"], r["inst"], r["neigh"], r["window"], r["table"], r["p"])] = r
    return out


def load_e5():
    """{(neigh, table, p, shots): {(file, inst): (prd, moves, ms_per_move)}} averaged over repeats."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows("e5_ils_ideal.jsonl"):
        acc[(r["neigh"], r["table"], r["p"], r["shots"])][(r["file"], r["inst"])].append(
            (r["prd"], r["moves"], r["ms_per_move"]))
    return {k: {i: tuple(np.mean(v, axis=0)) for i, v in d.items()} for k, d in acc.items()}


def fig4(e5, lang):
    t = TXT[lang]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.1))
    for nb in NB:
        for shots, ls in ((256, "--"), (4096, "-")):
            ps = list(range(0, 11))
            keys = [(nb, "uniform", 0, shots)] + [(nb, "small", p, shots) for p in range(1, 11)]
            mv = [np.mean([v[1] for v in e5[k].values()]) for k in keys]
            ms = [np.mean([v[2] for v in e5[k].values()]) for k in keys]
            lab = f"{NAME[nb]}, $S={shots}$"
            a1.plot(ps, mv, ls=ls, color=INK, marker=MARK[nb], ms=2.3, lw=0.8, label=lab)
            a2.plot(ps, ms, ls=ls, color=INK, marker=MARK[nb], ms=2.3, lw=0.8)
    for ax, yl in ((a1, t["moves"]), (a2, t["ms"])):
        ax.set_yscale("log")
        ax.set_xlabel(t["p"])
        ax.set_ylabel(yl)
        ax.set_xticks(range(0, 11, 2))
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.5, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(HERE / f"fig4_moves_{lang}.pdf")
    plt.close(fig)


def tab1():
    """E3: median CZ count per window, star (Sirius) and square lattice (Apollo)."""
    cz = defaultdict(list)
    for r in rows("e3_circuit_cost.jsonl"):
        for c in r["cost"]:
            cz[(r["neigh"], c["topology"], c["p"])].append(c["cz"])
    lines = []
    for nb in NB:
        cells = []
        for p in (1, 3, 6, 10):
            s = statistics.median(cz[(nb, "star", p)])
            l = statistics.median(cz[(nb, "lattice", p)])
            cells.append(f"{s:.0f} & {l:.0f}")
        lines.append(f"{NAME[nb]} & " + " & ".join(cells) + r" \\")
    (TAB / "tab1_e3.tex").write_text("\n".join(lines) + "\n")


def tab2(e5):
    """E5 median PRD (small table) by p at S = 256 and 4096, classical DP; rows to fill marked."""
    fill = r"\textbf{?}"
    ps = (0, 1, 2, 3, 6, 10)
    lines = []
    for nb in NB:
        dp = statistics.median(v[0] for v in e5[(nb, "classical", None, None)].values())
        for shots in (256, 4096):
            cells = []
            for p in ps:
                k = (nb, "uniform", 0, shots) if p == 0 else (nb, "small", p, shots)
                cells.append(f"{statistics.median(v[0] for v in e5[k].values()):.1f}")
            first = f"\\multirow{{2}}{{*}}{{{NAME[nb]}}}" if shots == 256 else ""
            dpc = f"\\multirow{{2}}{{*}}{{{dp:.1f}}}" if shots == 256 else ""
            lines.append(f"{first} & {shots} & " + " & ".join(cells) + f" & {dpc}" + r" \\")
        lines.append(r"\addlinespace[1pt]")
    body = "\n".join(lines[:-1])
    # E5n (noisy simulator, same moves as Sirius) and Sirius rows: data still to come.
    noisy = [f"{NAME[nb]} & 4096 & " + " & ".join([fill] * len(ps)) + r" & \\" for nb in NB]
    (TAB / "tab2_prd.tex").write_text(body + "\n")
    (TAB / "tab2_prd_pl.tex").write_text(re.sub(r"(\d)\.(\d)", r"\1{,}\2", body) + "\n")
    (TAB / "tab2_prd_noisy.tex").write_text("\n".join(noisy) + "\n")


def tab3(e5):
    """Instances 0-3 (where Sirius ran): PRD at S = 4096 for the noiseless simulator, the noisy
    simulator (E5n, same moves as Sirius) and Sirius itself. Missing data is marked."""
    sirius = json.loads((HERE.parents[1] / "gate_qaoa/measurements/sirius_full_taillard_ils.json").read_text())
    e5n = defaultdict(list)
    if (MEAS / "e5n_ils_noisy.jsonl").exists():
        for r in rows("e5n_ils_noisy.jsonl"):
            e5n[(r["neigh"], r["p"])].append(r)
    h2 = defaultdict(list)
    if (MEAS / "h2_sirius_ils.jsonl").exists():
        for r in rows("h2_sirius_ils.jsonl"):
            h2[(r["neigh"], r["p"])].append(r)
    keep = {(f, i) for f in ("tai20_5", "tai20_10", "tai20_20") for i in range(4)}
    ps = (0, 1, 2, 3, 6)
    fill, none = r"\textbf{?}", "--"

    def med(vals):
        return f"{statistics.median(vals):.1f}" if vals else fill

    # Noiseless at the SAME number of moves as Sirius (prefix of the E5 move log): Sirius makes
    # 1-2 moves in 10 s, the simulator hundreds, so the 10 s PRD would compare nothing.
    hw_moves = {(v["plik"], v["inst"], v["neigh"][5:]): v["ruchow"] for v in sirius.values()
                if isinstance(v, dict) and v.get("powt") == 0 and (v.get("plik"), v.get("inst")) in keep}
    pref = defaultdict(lambda: defaultdict(list))
    for r in rows("e5_ils_ideal.jsonl"):
        k_hw = hw_moves.get((r["file"], r["inst"], r["neigh"]))
        if r["shots"] != 4096 or k_hw is None or r["table"] not in ("small", "uniform"):
            continue
        ml = r["move_log"]
        if len(ml) >= k_hw:
            pref[(r["neigh"], r["p"])][(r["file"], r["inst"])].append(100.0 * (ml[k_hw - 1][1] - r["lb"]) / r["lb"])
    lines = []
    for nb in NB:
        ideal = [med([np.mean(v) for v in pref[(nb, p)].values()]) for p in ps]
        noisy = []
        for p in ps:                               # one table per neighborhood: small
            rs = [r for r in e5n[(nb, p)] if r["table"] in ("small", "uniform")]
            noisy.append(med([r["prd"] for r in rs]) if len(rs) >= 12 else fill)
        hw = []
        for p in ps:
            if p == 1:
                hw.append(med([v["prd"] for v in sirius.values() if isinstance(v, dict)
                               and v.get("neigh") == f"gate_{nb}" and v.get("powt") == 0
                               and (v.get("plik"), v.get("inst")) in keep]))
            elif p in (2, 3):
                hw.append(med([r["prd"] for r in h2[(nb, p)]]) if len(h2[(nb, p)]) >= 12 else fill)
            else:
                hw.append(none)
        for lab, cells in (("ideal", ideal), ("noisy", noisy), ("sirius", hw)):
            first = f"\\multirow{{3}}{{*}}{{{NAME[nb]}}}" if lab == "ideal" else ""
            lines.append(f"{first} & {{LAB_{lab}}} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1pt]")
    body = "\n".join(lines[:-1])
    for lang, labs in (("en", {"ideal": "noiseless", "noisy": "noisy sim.", "sirius": "Sirius"}),
                       ("pl", {"ideal": "bez szumu", "noisy": "z szumem", "sirius": "Sirius"})):
        out = body
        for k, v in labs.items():
            out = out.replace(f"{{LAB_{k}}}", v)
        if lang == "pl":
            out = re.sub(r"(\d)\.(\d)", r"\1{,}\2", out)
        (TAB / f"tab3_hw_{lang}.tex").write_text(out + "\n")


if __name__ == "__main__":
    TAB.mkdir(exist_ok=True)
    e1 = load_e1()
    e5 = load_e5()
    e1_all = load_e1_all() if (MEAS / "e4_window_noise.jsonl").exists() else None
    for lang in ("en", "pl"):
        if e1_all:
            fig3(e1_all, lang)
        fig1(e1, lang)
        fig2(e1, lang)
        fig4(e5, lang)
    tab1()
    tab2(e5)
    tab3(e5)
    print("ok:", sorted(p.name for p in HERE.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))
