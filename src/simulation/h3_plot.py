"""H3 for the paper: simulator vs Sirius on the circuits that ran (figure and table, EN and PL).

Reads h3_replay_tvd.jsonl (written by ``h3_replay --analyse``) and writes
figures/h3_energy{,_pl}.{pdf,png} and measurements/h3_table{,_pl}.tex. The
figure plots, per circuit and day, the mean normalised QUBO energy on the
simulator against the one on Sirius; the grey band is the mean difference
between the two device days. Qulacs gives the same values as qsim within
0.002, so the figure shows qsim and the table both.

Usage (any venv with matplotlib, e.g. .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.h3_plot
"""
from __future__ import annotations

import json

import numpy as np

from .h3_replay import OUT
from .t6_plot import FIGS, PROGRAMS

SCEN_STYLE = {"S1": ("none", 0.9), "S2": ("full", 0.9)}       # S3 matches S2 within 0.002 (table)

TXT = {
    "en": {"x": "mean energy on Sirius", "y": "mean energy on the simulator", "band": "Sirius day 1 vs day 2",
           "caption": r"The table compares the mean normalised QUBO energy (0 is the optimum) of the 84 H3 "
                      r"circuits on two days. $|\Delta E|$ is the mean per-circuit difference from Sirius, "
                      r"which differs from itself by 0.006--0.008 between the days.",
           "head": r"$p$ & Source & $E$ qsim & $E$ Qulacs & $|\Delta E|$ & optimum sampled & ones \\",
           "sirius": "Sirius", "noiseless": "S1 (noiseless)"},
    "pl": {"x": "średnia energia na Siriusie", "y": "średnia energia na symulatorze",
           "band": "Sirius dzień 1 vs dzień 2",
           "caption": r"Tabela porównuje średnią znormalizowaną energię QUBO (0 to optimum) 84 obwodów H3 z dwóch "
                      r"dni. $|\Delta E|$ to średnia różnica od Siriusa na obwód, a sam Sirius różni się między "
                      r"dniami o 0{,}006--0{,}008.",
           "head": r"$p$ & Źródło & $E$ qsim & $E$ Qulacs & $|\Delta E|$ & optimum w próbce & jedynki \\",
           "sirius": "Sirius", "noiseless": "S1 (bez szumu)"},
}


def load():
    return [json.loads(line) for line in OUT.with_name("h3_replay_tvd.jsonl").read_text().splitlines()]


def num(x, spec, lang):
    s = format(x, spec)
    return s.replace(".", "{,}") if lang == "pl" else s


def write_tex(rows, path, lang):
    T = TXT[lang]
    out = [r"\begin{table}[t]", r"\caption{" + T["caption"] + "}", r"\label{tab:h3}", r"\centering", r"\small",
           r"\begin{tabular}{clccccc}", r"\toprule", T["head"], r"\midrule"]
    for p in (1, 2):
        g = [r for r in rows if r["p"] == p and r["engine"] == "qsim" and r["scenario"] == "S1"]
        n = len(g)
        out.append(f"{p} & {T['sirius']} & \\multicolumn{{2}}{{c}}{{{num(np.mean([r['e_hw'] for r in g]), '.3f', lang)}}}"
                   f" & {num(np.mean([r['de_floor'] for r in g]), '.3f', lang)}"
                   f" & {sum(r['found_hw'] for r in g)}/{n} & {num(np.mean([r['w_hw'] for r in g]), '.2f', lang)} \\\\")
        for sc in ("S1", "S2", "S3"):
            q = [r for r in rows if r["p"] == p and r["engine"] == "qsim" and r["scenario"] == sc]
            u = [r for r in rows if r["p"] == p and r["engine"] == "qulacs_loop" and r["scenario"] == sc]
            name = T["noiseless"] if sc == "S1" else sc
            out.append(f" & {name} & {num(np.mean([r['e_sim'] for r in q]), '.3f', lang)}"
                       f" & {num(np.mean([r['e_sim'] for r in u]), '.3f', lang)}"
                       f" & {num(np.mean([abs(r['e_sim'] - r['e_hw']) for r in q]), '.3f', lang)}"
                       f" & {sum(r['found_sim'] for r in q)}/{len(q)}"
                       f" & {num(np.mean([r['w_sim'] for r in q]), '.2f', lang)} \\\\")
        if p == 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def plot(rows, path, lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = TXT[lang]
    label, col, marker = PROGRAMS["cirq"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
    band = np.mean([r["de_floor"] for r in rows])
    lim = [0, max(max(r["e_sim"], r["e_hw"]) for r in rows) * 1.05]
    for ax, p in zip(axes, (1, 2)):
        ax.fill_between(lim, [x - band for x in lim], [x + band for x in lim], color="0.88", zorder=-1,
                        label=T["band"])
        ax.plot(lim, lim, color="0.5", lw=0.8, zorder=0)
        for sc, (fill, a) in SCEN_STYLE.items():
            g = [r for r in rows if r["p"] == p and r["engine"] == "qsim" and r["scenario"] == sc]
            name = T["noiseless"] if sc == "S1" else sc
            ax.scatter([r["e_hw"] for r in g], [r["e_sim"] for r in g], s=16, marker=marker, alpha=a,
                       facecolors=col if fill == "full" else "none", edgecolors=col, linewidths=0.8,
                       label=f"{label}, {name}")
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(T["x"])
        ax.set_title(f"$p = {p}$", fontsize=10)
    axes[0].set_ylabel(T["y"])
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load()
    for lang, sfx in (("en", ""), ("pl", "_pl")):
        write_tex(rows, OUT.with_name(f"h3_table{sfx}.tex"), lang)
        plot(rows, FIGS / f"h3_energy{sfx}", lang)
    print(OUT.with_name("h3_table.tex").read_text())


if __name__ == "__main__":
    main()
