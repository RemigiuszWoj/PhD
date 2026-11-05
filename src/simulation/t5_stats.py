"""T5 statistics: is a simulator distinguishable from Sirius at the level of whole ILS runs?

Every T5 run replays one Sirius run (same instance, neighbourhood, repetition and
move budget), so the comparison is paired. Per engine x scenario:
  * mean, median PRD difference simulator - Sirius, wins / ties / losses;
  * Wilcoxon signed-rank test (ties dropped, the scipy default), Holm-corrected
    over the engine x scenario cells;
  * bootstrap 95 % CI of the mean difference, over runs and over instances
    (file, inst; 12 clusters), the latter because runs of one instance share
    the instance and are not independent.
The floor is Sirius against itself: inst. 0 with Dynasearch and Motzkin was run
five times on Sirius (6 configurations x 5 runs). The differences between two
Sirius runs of one configuration show how far one Sirius run lies from another;
the simulator runs of the same 30 Sirius runs are put next to them.

Writes measurements/t5_stats.md, the table t5_table.tex and figures/t5_prd.{pdf,png},
the last two also in Polish (*_pl) for main_pl.tex.

Usage (any venv with scipy and matplotlib, e.g. .venv-annsim):
    PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.t5_stats
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from .noise import REPO
from .t5_ils import OUT as T5, SIRIUS_ILS

MEAS = REPO / "articles" / "annsim_simulators" / "measurements"
FIGS = REPO / "articles" / "annsim_simulators" / "figures"
ENGINES = {"qsim": "qsim", "qulacs_loop": "Qulacs (shot loop)"}
SCENARIOS = ["S2", "S3"]
N_BOOT = 20000


def load():
    rows = [r for r in map(json.loads, T5.read_text().splitlines()) if r.get("ok")]
    sirius = [v for k, v in json.loads(SIRIUS_ILS.read_text()).items() if not k.startswith("_")]
    return rows, sirius


def run_id(r):
    return (r["file"], r["inst"], r["neigh"], r["rep"])


def boot_mean_ci(diff, clusters, rng):
    """95 % percentile CIs of the mean of ``diff``: resampling runs, and resampling whole clusters."""
    diff = np.asarray(diff)
    by_run = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(axis=1)
    groups = defaultdict(list)
    for d, c in zip(diff, clusters):
        groups[c].append(d)
    sums = np.array([sum(g) for g in groups.values()])
    sizes = np.array([len(g) for g in groups.values()])
    pick = rng.integers(0, len(sums), (N_BOOT, len(sums)))
    by_cluster = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return np.percentile(by_run, [2.5, 97.5]), np.percentile(by_cluster, [2.5, 97.5])


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_cells(rows, rng):
    cells = []
    for e in ENGINES:
        for s in SCENARIOS:
            d = sorted((r for r in rows if r["engine"] == e and r["scenario"] == s), key=run_id)
            sim = np.array([r["prd"] for r in d])
            sir = np.array([r["prd_sirius"] for r in d])
            diff = sim - sir
            ci_run, ci_inst = boot_mean_ci(diff, [(r["file"], r["inst"]) for r in d], rng)
            cells.append({"engine": e, "scenario": s, "n": len(d), "prd_sim": sim.mean(), "prd_sirius": sir.mean(),
                          "mean_diff": diff.mean(), "median_diff": float(np.median(diff)),
                          "ci_run": ci_run, "ci_inst": ci_inst, "p": wilcoxon(sim, sir).pvalue,
                          "wins": int(np.sum(diff < 0)), "ties": int(np.sum(diff == 0)),
                          "losses": int(np.sum(diff > 0)), "diff": diff, "sim": sim, "sir": sir})
    for c, a in zip(cells, holm(np.array([c["p"] for c in cells]))):
        c["p_holm"] = a
    return cells


def engine_pairs(rows, rng):
    out = []
    for s in SCENARIOS:
        a, b = ({run_id(r): r for r in rows if r["engine"] == e and r["scenario"] == s} for e in ENGINES)
        ks = sorted(set(a) & set(b))
        diff = np.array([a[k]["prd"] - b[k]["prd"] for k in ks])
        ci_run, ci_inst = boot_mean_ci(diff, [k[:2] for k in ks], rng)
        out.append({"scenario": s, "n": len(ks), "mean_diff": diff.mean(), "ci_run": ci_run, "ci_inst": ci_inst,
                    "p": wilcoxon(diff).pvalue, "ties": int(np.sum(diff == 0))})
    return out


def floor(rows, sirius):
    """Sirius-vs-Sirius differences on the repeated configurations, and the simulators on the same runs."""
    reps = defaultdict(dict)
    for v in sirius:
        reps[(v["plik"], v["inst"], v["neigh"].replace("gate_", ""))][v["powt"]] = v["prd"]
    reps = {c: reps[c] for c in sorted(reps, key=config_order) if len(reps[c]) >= 2 and c[0].startswith("tai20")}
    sir_pairs = np.array([x[i] - x[j] for x in reps.values() for i, j in itertools.permutations(x, 2)])
    sir_sd = np.array([np.std(list(x.values()), ddof=1) for x in reps.values()])
    out = {"configs": list(reps),"n_runs": sum(len(x) for x in reps.values()),
           "sirius_abs_pair": np.abs(sir_pairs).mean(), "sirius_pair_q95": np.percentile(np.abs(sir_pairs), 95),
           "sirius_sd_pooled": float(np.sqrt(np.mean(sir_sd ** 2))), "sirius_sd": sir_sd, "sim": {}}
    for e in ENGINES:
        for s in SCENARIOS:
            d = defaultdict(dict)
            for r in rows:
                c = (r["file"], r["inst"], r["neigh"])
                if r["engine"] == e and r["scenario"] == s and c in reps:
                    d[c][r["rep"]] = r["prd"]
            vs_sir = np.array([d[c][i] - reps[c][i] for c in reps for i in d[c]])
            sd = np.array([np.std(list(d[c].values()), ddof=1) for c in reps])
            means = np.array([np.mean(list(d[c].values())) - np.mean(list(reps[c].values())) for c in reps])
            mw = np.array([mannwhitneyu(list(d[c].values()), list(reps[c].values())).pvalue for c in reps])
            out["sim"][(e, s)] = {"abs_vs_sirius": np.abs(vs_sir).mean(), "sd_pooled": float(np.sqrt(np.mean(sd ** 2))),
                                  "config_mean_diff": means, "config_p": mw}
    return out


def config_order(c):
    n, m = map(int, c[0][3:].split("_"))
    return n, m, c[1], c[2]


def fmt_ci(ci):
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]"


def write_md(cells, pairs, fl, path):
    L = ["# T5: ILS with a simulator in place of Sirius (tai20, inst. 0-3, p = 1)", "",
         "PRD in percentage points (pp); difference = simulator - Sirius, paired by run (each simulator run",
         "replays one Sirius run with its move budget). Wilcoxon signed-rank (ties dropped), Holm over the",
         f"four cells. Bootstrap 95 % CI of the mean ({N_BOOT} resamples) over runs and over the 12 instances.", "",
         "| engine | sc | n | PRD sim | PRD Sirius | mean diff | median | CI runs | CI instances | better / tie / worse | p | p Holm |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        L.append(f"| {c['engine']} | {c['scenario']} | {c['n']} | {c['prd_sim']:.2f} | {c['prd_sirius']:.2f} | "
                 f"{c['mean_diff']:+.2f} | {c['median_diff']:+.2f} | {fmt_ci(c['ci_run'])} | {fmt_ci(c['ci_inst'])} | "
                 f"{c['wins']} / {c['ties']} / {c['losses']} | {c['p']:.2g} | {c['p_holm']:.2g} |")
    L += ["", "## qsim - Qulacs (shot loop), same runs", "",
          "| sc | n | mean diff | CI runs | CI instances | ties | p |", "|---|---|---|---|---|---|---|"]
    for q in pairs:
        L.append(f"| {q['scenario']} | {q['n']} | {q['mean_diff']:+.2f} | {fmt_ci(q['ci_run'])} | "
                 f"{fmt_ci(q['ci_inst'])} | {q['ties']} | {q['p']:.2g} |")
    L += ["", "## Floor: Sirius against itself", "",
          f"Repeated configurations: {', '.join('/'.join(map(str, c)) for c in fl['configs'])} "
          f"({fl['n_runs']} Sirius runs, 1 move each).", "",
          f"- Sirius vs Sirius, two runs of one configuration: mean |diff| = {fl['sirius_abs_pair']:.2f} pp, "
          f"95th percentile {fl['sirius_pair_q95']:.2f} pp; pooled within-configuration SD = {fl['sirius_sd_pooled']:.2f} pp "
          f"(per configuration: {', '.join(f'{x:.2f}' for x in fl['sirius_sd'])}).", "",
          "Per configuration (same order as above): mean of the 5 simulator runs minus mean of the 5 Sirius runs,",
          "with the Mann-Whitney p of the two sets of 5 (smallest attainable two-sided p = 0.008).", "",
          "| engine | sc | mean abs(sim - Sirius), same run | pooled SD of the simulator | config. mean diff (p) |",
          "|---|---|---|---|---|"]
    for (e, s), x in fl["sim"].items():
        L.append(f"| {e} | {s} | {x['abs_vs_sirius']:.2f} | {x['sd_pooled']:.2f} | "
                 f"{', '.join(f'{m:+.2f} ({q:.2g})' for m, q in zip(x['config_mean_diff'], x['config_p']))} |")
    path.write_text("\n".join(L) + "\n")


TXT = {
    "en": {"caption": r"The table compares the mean PRD of 72 paired ILS runs on a simulator and on Sirius. "
                      r"$\Delta$ is the simulator minus Sirius in pp, CI is the 95\,\% interval over instances, "
                      r"and $p$ comes from the Wilcoxon test with Holm correction. Sirius reaches a mean PRD of {sir}.",
           "head": r"Simulator & Sc. & PRD & $\Delta$ & 95\,\% CI & better / tie / worse & $p$ \\",
           "qulacs_loop": "Qulacs (shot loop)", "scenario": "scenario", "x": "PRD on Sirius [%]",
           "y": "PRD on the simulator [%]", "band": "Sirius vs Sirius, mean |Δ|",
           "ci_x": "mean Δ PRD, sim − Sirius [pp]", "ci_title": "mean, 95 % CI over instances"},
    "pl": {"caption": r"Tabela porównuje średni PRD 72 sparowanych biegów ILS na symulatorze i na Siriusie. "
                      r"$\Delta$ to różnica symulator minus Sirius w pp, CI to 95\,\% przedział po instancjach, "
                      r"a $p$ pochodzi z testu Wilcoxona z poprawką Holma. Sirius osiąga średni PRD {sir}.",
           "head": r"Symulator & Sc. & PRD & $\Delta$ & 95\,\% CI & lepszy / remis / gorszy & $p$ \\",
           "qulacs_loop": "Qulacs (pętla po shotach)", "scenario": "scenariusz", "x": "PRD na Siriusie [%]",
           "y": "PRD na symulatorze [%]", "band": "Sirius vs Sirius, średnia |Δ|",
           "ci_x": "średnia Δ PRD, sym. − Sirius [pp]", "ci_title": "średnia, 95% CI po instancjach"},
}


def num(x, spec, lang, math=False):
    s = format(x, spec)
    return s.replace(".", "{,}" if math else ",") if lang == "pl" else s


def label(engine, lang):
    return TXT[lang].get(engine, ENGINES[engine])


def write_tex(cells, path, lang):
    caption = TXT[lang]["caption"].replace("{sir}", num(cells[0]["prd_sirius"], ".2f", lang))
    L = [r"% generated by src/simulation/t5_stats.py", r"\begin{table}[t]", rf"\caption{{{caption}}}",
         r"\label{tab:t5}", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccccc}",
         r"\toprule",
         TXT[lang]["head"], r"\midrule"]
    for c in cells:
        lo, hi = (num(x, "+.2f", lang, True) for x in c["ci_inst"])
        L.append(f"{label(c['engine'], lang)} & {c['scenario']} & {num(c['prd_sim'], '.2f', lang)} & "
                 f"${num(c['mean_diff'], '+.2f', lang, True)}$ & "
                 f"$[{lo}; {hi}]$ & {c['wins']} / {c['ties']} / {c['losses']} & {num(c['p_holm'], '.2g', lang)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(L) + "\n")


def plot(cells, fl, path, lang):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .t6_plot import PROGRAMS

    T = TXT[lang]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1, 0.9]},
                             layout="constrained")
    marker = {e: PROGRAMS[prog][:0:-1] for e, prog in (("qsim", "cirq"), ("qulacs_loop", "qulacs"))}
    band = fl["sirius_abs_pair"]
    for ax, s in zip(axes[:2], SCENARIOS):
        for c in (c for c in cells if c["scenario"] == s):
            m, col = marker[c["engine"]]
            ax.scatter(c["sir"], c["sim"], s=14, marker=m, facecolors="none", edgecolors=col, linewidths=0.8,
                       label=label(c["engine"], lang))
        lim = [0, max(max(c["sim"].max(), c["sir"].max()) for c in cells) * 1.05]
        ax.plot(lim, lim, color="0.5", lw=0.8, zorder=0)
        ax.fill_between(lim, [x - band for x in lim], [x + band for x in lim], color="0.88", zorder=-1,
                        label=T["band"])
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_xlabel(T["x"])
        ax.set_title(f"{T['scenario']} {s}", fontsize=10)
        ax.set_aspect("equal")
    axes[0].set_ylabel(T["y"])
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    ax = axes[2]
    for i, c in enumerate(cells):
        m, col = marker[c["engine"]]
        lo, hi = c["ci_inst"]
        ax.errorbar(c["mean_diff"], i, xerr=[[c["mean_diff"] - lo], [hi - c["mean_diff"]]], fmt=m, color=col,
                    ms=5, capsize=3)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(cells)), [f"{c['engine'].split('_')[0]} {c['scenario']}" for c in cells], fontsize=8)
    ax.set_ylim(len(cells) - 0.5, -0.5)
    ax.set_xlabel(T["ci_x"])
    ax.set_title(T["ci_title"], fontsize=10)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, sirius = load()
    rng = np.random.default_rng(0)
    cells, pairs, fl = paired_cells(rows, rng), engine_pairs(rows, rng), floor(rows, sirius)
    FIGS.mkdir(parents=True, exist_ok=True)
    write_md(cells, pairs, fl, MEAS / "t5_stats.md")
    for lang, sfx in (("en", ""), ("pl", "_pl")):
        write_tex(cells, MEAS / f"t5_table{sfx}.tex", lang)
        plot(cells, fl, FIGS / f"t5_prd{sfx}", lang)
    print((MEAS / "t5_stats.md").read_text())


if __name__ == "__main__":
    main()
