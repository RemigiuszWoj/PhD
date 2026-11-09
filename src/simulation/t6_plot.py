"""T6 for the paper: time per circuit versus qubits (figure) and the largest feasible window (table).

Reads t6_scaling.jsonl (p = 1, one thread, as measured by t6.py run) and writes
figures/t6_cost.{pdf,png}, measurements/t6_table.tex and measurements/t6_table.md;
the figure and the LaTeX table also in Polish (*_pl) for main_pl.tex.
Programs without gate-level noise (DDSIM, QuTiP, Stim) appear only in the table.
One colour and marker per program, shared with the T5 figure (PROGRAMS).

Usage (any venv with matplotlib, e.g. .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.t6_plot
"""
from __future__ import annotations

import json
from collections import defaultdict

from .t6 import MEAS, OUT as T6

FIGS = MEAS.parent / "figures"
CAP_S = 300.0
MAX_QUBITS = 16                     # largest window at n = 20 (K = 15) plus the resonator
# program: (label, colour, marker); colours are slots 1-7 of the validated categorical palette
PROGRAMS = {"aer": ("Qiskit Aer", "#2a78d6", "o"), "cirq": ("Cirq / qsim", "#eb6834", "s"),
            "qulacs": ("Qulacs", "#1baf7a", "^"), "qibo": ("Qibo", "#eda100", "D"),
            "braket": ("Braket", "#e87ba4", "v"), "pennylane": ("PennyLane", "#008300", "P"),
            "cudaq": ("CUDA-Q", "#6250d6", "X")}
# engine: (program, method label, dashed)
ENGINES = {"aer_sv": ("aer", "trajectories", False), "qsim": ("cirq", "trajectories", False),
           "qulacs_loop": ("qulacs", "shot loop", False), "qulacs": ("qulacs", "NoiseSimulator", True),
           "qibo": ("qibo", "trajectories", False),
           "aer_dm": ("aer", "DM", False), "cirq_dm": ("cirq", "DM", False), "qulacs_dm": ("qulacs", "DM", False),
           "qibo_dm": ("qibo", "DM", False), "braket": ("braket", "DM", False),
           "pennylane": ("pennylane", "DM", False), "cudaq": ("cudaq", "DM", False)}
TRAJ = ["aer_sv", "qsim", "qulacs_loop", "qulacs", "qibo"]
DM = ["aer_dm", "cirq_dm", "qulacs_dm", "qibo_dm", "braket", "pennylane", "cudaq"]
TABLE = TRAJ + DM + ["ddsim", "ddsim_dm", "qutip", "stim"]
SCENARIOS = ["S1", "S2", "S3"]
# series that stop on a library error or give a wrong distribution, not on the time cap (stage 1-2 report)
BROKEN = {("aer_sv", "S3"): "Aer error \"Kraus is empty\" from K = 5",
          ("qulacs", "S3"): "NoiseSimulator never decays in AmplitudeDampingNoise (wrong distribution)"}


TXT = {
    "en": {"caption": r"Largest window $K$ that runs within 300\,s per circuit (density matrices, DM, also within "
                      r"13 qubits), median time over three windows at $K = 11$ in S2 for all 4096 shots, and peak "
                      r"resident memory (RSS) in S2 ($p = 1$, one thread). The dagger marks a library error or a "
                      r"wrong distribution, and --- an unsupported scenario or a run over the limit; in S1, Braket, "
                      r"PennyLane and CUDA-Q use their statevector simulators; DDSIM and QuTiP run S1 only, up to "
                      r"$K = 16$.",
           "head": r"Program & Method & \multicolumn{3}{c}{max.\ $K$} & $t$ [s], S2, $K = 11$ & RSS S2 [MB] \\",
           "methods": {}, "traj": "trajectories", "dm": "density matrix", "y": "time per circuit [s]",
           "x": "qubits (window size K + resonator)", "cap": "cap 300 s"},
    "pl": {"caption": r"Największe okno $K$, które mieści się w 300\,s na obwód (macierze gęstości, DM, także "
                      r"w 13 kubitach), mediana czasu z trzech okien przy $K = 11$ w S2 dla wszystkich 4096 shotów "
                      r"i szczytowa pamięć rezydentna (RSS) w S2 ($p = 1$, jeden wątek). Krzyżyk oznacza błąd "
                      r"biblioteki lub zły rozkład, a --- scenariusz bez obsługi lub przekroczenie limitu; w S1 "
                      r"Braket, PennyLane i CUDA-Q używają symulatorów wektora stanu, a DDSIM i QuTiP liczą tylko "
                      r"S1, do $K = 16$.",
           "head": r"Program & Metoda & \multicolumn{3}{c}{maks.\ $K$} & $t$ [s], S2, $K = 11$ & RSS S2 [MB] \\",
           "methods": {"trajectories": "trajektorie", "shot loop": "pętla po shotach"}, "traj": "trajektorie",
           "dm": "macierz gęstości", "y": "czas na obwód [s]", "x": "kubity (rozmiar okna K + rezonator)",
           "cap": "limit 300 s"},
}


def method(engine, lang):
    m = ENGINES[engine][1]
    return TXT[lang]["methods"].get(m, m)


def style(engine, lang="en"):
    prog, _, dashed = ENGINES[engine]
    label, colour, marker = PROGRAMS[prog]
    return f"{label}, {method(engine, lang)}", colour, marker, "--" if dashed else "-"


def series():
    g = defaultdict(dict)
    for r in map(json.loads, T6.read_text().splitlines()):
        if r["p"] == 1 and r.get("threads", 1) == 1:
            g[(r["engine"], r["scenario"])][r["K"]] = r
    return g


def plot(g, path, lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = TXT[lang]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True, sharey=True, layout="constrained")
    for row, (group, title) in enumerate([(TRAJ, T["traj"]), (DM, T["dm"])]):
        for col, s in enumerate(["S2", "S3"]):
            ax = axes[row, col]
            for e in group:
                ks = sorted(K for K, r in g.get((e, s), {}).items() if r.get("ok"))
                if not ks or (e, s) in BROKEN:
                    continue
                label, colour, marker, ls = style(e, lang)
                ax.plot([g[(e, s)][K]["n_qubits"] for K in ks], [g[(e, s)][K]["t_median"] for K in ks],
                        ls=ls, lw=1.5, color=colour, marker=marker, ms=5, label=label)
            ax.axhline(CAP_S, color="0.6", lw=0.8, ls=":")
            ax.axvline(MAX_QUBITS, color="0.6", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_xticks(range(5, 18, 2))
            ax.set_title(f"{title}, {s}", fontsize=10)
            ax.grid(axis="y", color="0.92", lw=0.6)
            if col == 0:
                ax.set_ylabel(T["y"])
            if row == 1:
                ax.set_xlabel(T["x"])
        handles = {}
        for ax in axes[row]:
            handles.update(zip(*ax.get_legend_handles_labels()[::-1]))
        order = [style(e, lang)[0] for e in group if style(e, lang)[0] in handles]
        axes[row, 1].legend([handles[k] for k in order], order, fontsize=7, loc="upper left",
                            bbox_to_anchor=(1.02, 1), frameon=False)
    axes[0, 0].annotate(T["cap"], (5, CAP_S), xytext=(0, 3), textcoords="offset points", fontsize=7, color="0.4")
    axes[0, 0].annotate("K = 15", (MAX_QUBITS, 1e-2), xytext=(-3, 0), textcoords="offset points", fontsize=7,
                        color="0.4", ha="right")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def cell(g, e, s):
    ks = g.get((e, s), {})
    ok = [K for K, r in ks.items() if r.get("ok")]
    return max(ok) if ok else None


def t_median(g, e, K):
    r = g.get((e, "S2"), {}).get(K)
    return f"{r['t_median']:#.3g}".rstrip(".") if r and r.get("ok") else None


def rss_s2(g, e):
    return max((r.get("peak_rss_mb") or 0 for r in g.get((e, "S2"), {}).values() if r.get("ok")), default=0)


def mark(g, e, s, dash):
    k = cell(g, e, s)
    return (str(k) if k else dash) + ("†" if (e, s) in BROKEN else "")


def write_md(g, path):
    md = ["# T6: largest window under the cap (p = 1, one thread, 300 s per circuit)", "",
          "| engine | max K S1 | max K S2 | max K S3 | t [s] S2, K = 11 | t [s] S2, K = 15 | peak RSS S2 [MB] | note |",
          "|---|---|---|---|---|---|---|---|"]
    for e in TABLE:
        note = next((r.get("skipped") for sc in SCENARIOS for r in g.get((e, sc), {}).values()
                     if r.get("skipped") and not r.get("above_cap")), "")
        note = "; ".join([note] * bool(note) + [f"{s}: {BROKEN[(e, s)]}" for s in SCENARIOS if (e, s) in BROKEN])
        md.append(f"| {e} | " + " | ".join(mark(g, e, s, "–") for s in SCENARIOS) +
                  f" | {t_median(g, e, 11) or '–'} | {t_median(g, e, 15) or '–'} | {rss_s2(g, e):.0f} | {note[:110]} |")
    path.write_text("\n".join(md) + "\n")


def write_tex(g, path, lang):
    tex = [r"% generated by src/simulation/t6_plot.py", r"\begin{table}[t]", rf"\caption{{{TXT[lang]['caption']}}}",
           r"\label{tab:t6}", r"\centering", r"\small", r"\begin{tabular}{llccccr}", r"\toprule",
           TXT[lang]["head"], r" & & S1 & S2 & S3 & & \\", r"\midrule"]
    for e in (e for e in TABLE if e in ENGINES):
        t = t_median(g, e, 11)
        t = (t.replace(".", ",") if lang == "pl" else t) if t else "---"
        tex.append(f"{PROGRAMS[ENGINES[e][0]][0]} & {method(e, lang)} & " +
                   " & ".join(mark(g, e, s, "---").replace("†", r"$^\dagger$") for s in SCENARIOS) +
                   f" & {t} & {rss_s2(g, e):.0f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(tex) + "\n")


def main():
    g = series()
    FIGS.mkdir(parents=True, exist_ok=True)
    write_md(g, MEAS / "t6_table.md")
    for lang, sfx in (("en", ""), ("pl", "_pl")):
        plot(g, FIGS / f"t6_cost{sfx}", lang)
        write_tex(g, MEAS / f"t6_table{sfx}.tex", lang)
    print((MEAS / "t6_table.md").read_text())


if __name__ == "__main__":
    main()
