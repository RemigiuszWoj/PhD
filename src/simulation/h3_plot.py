"""H3 for the paper: simulator vs Sirius on the circuits that ran (figure and table, EN and PL).

Reads h3_replay_tvd.jsonl (written by ``h3_replay --analyse``) and writes
figures/h3_energy{,_pl}.{pdf,png} and measurements/h3_table{,_pl}.tex. The
figure plots, per circuit and day, the mean normalised QUBO energy on qsim in
S1-S3 against the one on Sirius; the grey band is the mean difference between
the two device days. Qulacs gives the same values as qsim within 0.002, so the
figure shows qsim and the table both. The baseline rows of the table (empty
move, uniform sampler, Bernoulli sampler fitted per histogram) come from
h3_baselines.jsonl, also written by ``h3_replay --analyse``.

Usage (any venv with matplotlib, e.g. .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.h3_plot
"""
from __future__ import annotations

import json

import numpy as np

from .h3_replay import OUT
from .t6_plot import FIGS

# scenario: (marker, filled, colour); one hue (qsim's), darker with more detail, marker as second cue
SCEN_STYLE = {"S1": ("o", False, "#eb6834"), "S2": ("s", True, "#eb6834"), "S3": ("^", True, "#8f3a12")}

TXT = {
    "en": {"x": "mean energy on Sirius", "y": "mean energy on the simulator", "band": "Sirius day 1 vs day 2",
           "caption": r"Mean normalized QUBO energy $E$ (0 is the optimum) over the 84 histograms per depth, with "
                      r"the mean absolute difference $|\Delta E|$ from Sirius on the same day (for Sirius, between "
                      r"the days), the number of histograms that contain the optimal move, and the mean number of "
                      r"ones per sample. The last three columns refer to qsim for the simulators and give expected "
                      r"values for the baselines and the Bernoulli sampler fitted to each histogram.",
           "head": r"$p$ & Source & $E$ qsim & $E$ Qulacs & $|\Delta E|$ & optimum sampled & ones \\",
           "sirius": "Sirius", "noiseless": "S1 (noiseless)",
           "base": ("empty move", "uniform sampler", "Bernoulli, fitted")},
    "pl": {"x": "średnia energia na Siriusie", "y": "średnia energia na symulatorze",
           "band": "Sirius dzień 1 vs dzień 2",
           "caption": r"Średnia znormalizowana energia QUBO $E$ (0 to optimum) z 84 histogramów na głębokość, ze "
                      r"średnią bezwzględną różnicą $|\Delta E|$ od Siriusa z tego samego dnia (dla Siriusa między "
                      r"dniami), liczbą histogramów zawierających ruch optymalny i średnią liczbą jedynek w próbce. "
                      r"Trzy ostatnie kolumny dotyczą qsim dla symulatorów, a dla prób bazowych i próbnika "
                      r"Bernoulliego dopasowanego do każdego histogramu podają wartości oczekiwane.",
           "head": r"$p$ & Źródło & $E$ qsim & $E$ Qulacs & $|\Delta E|$ & optimum w próbce & jedynki \\",
           "sirius": "Sirius", "noiseless": "S1 (bez szumu)",
           "base": ("pusty ruch", "próbnik jednostajny", "Bernoulli, dopasowany")},
}


def load(name="h3_replay_tvd.jsonl"):
    return [json.loads(line) for line in OUT.with_name(name).read_text().splitlines()]


def num(x, spec, lang):
    s = format(x, spec)
    return s.replace(".", "{,}") if lang == "pl" else s


def write_tex(rows, base, path, lang):
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
        b = [r for r in base if r["p"] == p]
        mean = lambda f: np.mean([f(r) for r in b])
        for name, e, found, ones in zip(T["base"], ("e_empty", "e_uniform", "e_coin"),
                                        (None, "found_uniform", "found_coin"),
                                        (lambda r: 0, lambda r: r["K"] / 2, lambda r: r["w_hw"])):
            hits = f"{round(sum(r[found] for r in b))}/{len(b)}" if found else "---"
            ones_s = num(mean(ones), ".2f", lang) if found else "0"
            out.append(f" & {name} & \\multicolumn{{2}}{{c}}{{{num(mean(lambda r: r[e]), '.3f', lang)}}}"
                       f" & {num(mean(lambda r: abs(r[e] - r['e_hw'])), '.3f', lang)} & {hits} & {ones_s} \\\\")
        if p == 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n")


def plot(rows, path, lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = TXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), layout="constrained")
    band = np.mean([r["de_floor"] for r in rows])
    lim = [0, max(max(r["e_sim"], r["e_hw"]) for r in rows) * 1.05]
    for ax, p in zip(axes, (1, 2)):
        ax.fill_between(lim, [x - band for x in lim], [x + band for x in lim], color="0.88", zorder=-1,
                        label=T["band"])
        ax.plot(lim, lim, color="0.5", lw=0.8, zorder=0)
        for sc, (marker, filled, col) in SCEN_STYLE.items():
            g = [r for r in rows if r["p"] == p and r["engine"] == "qsim" and r["scenario"] == sc]
            name = T["noiseless"] if sc == "S1" else sc
            ax.scatter([r["e_hw"] for r in g], [r["e_sim"] for r in g], s=16, marker=marker, alpha=0.85,
                       facecolors=col if filled else "none", edgecolors=col, linewidths=0.8,
                       label=f"qsim, {name}")
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
    rows, base = load(), load("h3_baselines.jsonl")
    for lang, sfx in (("en", ""), ("pl", "_pl")):
        write_tex(rows, base, OUT.with_name(f"h3_table{sfx}.tex"), lang)
        plot(rows, FIGS / f"h3_energy{sfx}", lang)
    print(OUT.with_name("h3_table.tex").read_text())


if __name__ == "__main__":
    main()
