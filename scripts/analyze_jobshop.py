"""Analysis of the job shop campaign (articles/jobshop/PLAN.md §6-§7).

Reads result.json files (normally the *_budgets directories written by extract_budgets.py)
and prints / writes:
  T1  mean RPD to BKS per Taillard class x mode, for every budget (Markdown + LaTeX)
  T2  mean number of moves per class x mode at the longest budget
  T3  Friedman test + average ranks + Nemenyi critical difference per budget (blocks =
      instances, values averaged over seeds), globally and per class
  T4  Wilcoxon signed-rank + Holm for the key pairs (H1, H6, H9)
  T5  continuum: size x rho x mode, RPD to the best makespan of any run on that instance
  T6  bridge: PFSP instances solved as job shops (mean makespan per file x mode)

Usage:
    python scripts/analyze_jobshop.py --dirs results/jobshop/min_budgets results/jobshop/opt_budgets \
        --out articles/jobshop/tables
    python scripts/analyze_jobshop.py --dirs results/jobshop/continuum_budgets --out articles/jobshop/tables
    python scripts/analyze_jobshop.py ... --bks data/bks_jssp_current.json   # optional {name: bks} override
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.jobshop.instance import TA_CLASSES, bks as ta_bks, class_of  # noqa: E402

PAIRS = [("js_n5", "js_fib"), ("js_n5", "js_k3"), ("js_n5", "js_kn"), ("js_n6", "js_c6"), ("js_n7", "js_c7"),
         ("js_k3", "js_kn"), ("js_n6", "js_k3"), ("js_n1", "js_n5")]
# Studentized range / sqrt(2), alpha = 0.05 (Demsar 2006), k = 2..12
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268}


def load(dirs, bks_override):
    rows = []
    for d in dirs:
        for p in Path(d).rglob("result.json"):
            r = json.loads(p.read_text())
            c = r["config"]
            f = c["instance_file"]
            ref = bks_override.get(f) or (ta_bks(f) if f.startswith("ta") else None)
            rows.append({"algo": c["algorithm"], "mode": c["neighborhood"], "file": f, "seed": c["seed"],
                         "t": c["time_limit_ms"], "C": r["cmax_best"], "lb": r.get("lower_bound"),
                         "bks": ref, "it": r.get("iterations") or 0, "cls": class_of(f)})
    return rows


def rpd(x, ref):
    return 100.0 * (x - ref) / ref if ref else float("nan")


def mean(v):
    v = [x for x in v if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(v) if v else float("nan")


def table_rpd(rows, out: Path | None):
    ta = [r for r in rows if r["cls"]]
    if not ta:
        return
    lines_md, lines_tex = [], []
    for algo in sorted({r["algo"] for r in ta}):
        for t in sorted({r["t"] for r in ta if r["algo"] == algo}):
            sel = [r for r in ta if r["algo"] == algo and r["t"] == t]
            modes = sorted({r["mode"] for r in sel})
            g = defaultdict(list)
            for r in sel:
                g[(r["cls"], r["mode"])].append(rpd(r["C"], r["bks"]))
            head = f"| {algo} t={t} ms | " + " | ".join(m.replace("js_", "") for m in modes) + " |"
            lines_md += ["", head, "|" + "---|" * (len(modes) + 1)]
            lines_tex += ["", f"% {algo}, t = {t} ms: mean RPD to BKS [%]",
                          "\\begin{tabular}{l" + "r" * len(modes) + "}", "\\toprule",
                          "Class & " + " & ".join(m.replace("js_", "").replace("_", "\\_") for m in modes) + " \\\\",
                          "\\midrule"]
            for cls in TA_CLASSES:
                vals = [mean(g.get((cls, m), [])) for m in modes]
                if all(math.isnan(v) for v in vals):
                    continue
                best = min(v for v in vals if not math.isnan(v))
                lines_md.append(f"| {cls} | " + " | ".join(("**%.2f**" if v == best else "%.2f") % v for v in vals) + " |")
                lines_tex.append(cls.replace("x", "$\\times$") + " & " + " & ".join(
                    ("\\textbf{%.2f}" if v == best else "%.2f") % v for v in vals) + " \\\\")
            allv = [mean([rpd(r["C"], r["bks"]) for r in sel if r["mode"] == m]) for m in modes]
            lines_md.append("| all | " + " | ".join("%.2f" % v for v in allv) + " |")
            lines_tex += ["\\midrule", "All & " + " & ".join("%.2f" % v for v in allv) + " \\\\",
                          "\\bottomrule", "\\end{tabular}"]
    print("\n## T1. Mean RPD to BKS [%]" + "\n".join(lines_md))
    if out:
        (out / "T1_rpd.md").write_text("\n".join(lines_md) + "\n")
        (out / "T1_rpd.tex").write_text("\n".join(lines_tex) + "\n")


def table_moves(rows, out: Path | None):
    ta = [r for r in rows if r["cls"]]
    if not ta:
        return
    t = max(r["t"] for r in ta)
    lines = []
    for algo in sorted({r["algo"] for r in ta}):
        sel = [r for r in ta if r["algo"] == algo and r["t"] == t]
        modes = sorted({r["mode"] for r in sel})
        lines += ["", f"| {algo} moves at {t} ms | " + " | ".join(m.replace("js_", "") for m in modes) + " |",
                  "|" + "---|" * (len(modes) + 1)]
        for cls in TA_CLASSES:
            v = [mean([r["it"] for r in sel if r["cls"] == cls and r["mode"] == m]) for m in modes]
            if not all(math.isnan(x) for x in v):
                lines.append(f"| {cls} | " + " | ".join("%.0f" % x for x in v) + " |")
    print("\n## T2. Mean number of moves" + "\n".join(lines))
    if out:
        (out / "T2_moves.md").write_text("\n".join(lines) + "\n")


def per_instance(sel, modes):
    """instance -> {mode: mean RPD over seeds}; only instances with every mode present."""
    g = defaultdict(lambda: defaultdict(list))
    for r in sel:
        g[r["file"]][r["mode"]].append(rpd(r["C"], r["bks"] or r["lb"]))
    return {f: {m: statistics.mean(v[m]) for m in modes} for f, v in g.items() if all(m in v for m in modes)}


def friedman(rows, out: Path | None):
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        print("\n(scipy missing: Friedman skipped)")
        return
    ta = [r for r in rows if r["cls"]]
    lines = []
    for algo in sorted({r["algo"] for r in ta}):
        for t in sorted({r["t"] for r in ta if r["algo"] == algo}):
            sel = [r for r in ta if r["algo"] == algo and r["t"] == t]
            modes = sorted({r["mode"] for r in sel})
            for scope in ["all"] + list(TA_CLASSES):
                s2 = sel if scope == "all" else [r for r in sel if r["cls"] == scope]
                pi = per_instance(s2, modes)
                if len(pi) < 5 or len(modes) < 3:
                    continue
                data = [[pi[f][m] for f in pi] for m in modes]
                stat, p = friedmanchisquare(*data)
                ranks = defaultdict(list)
                for f in pi:
                    vals = sorted(pi[f][m] for m in modes)
                    for m in modes:
                        pos = [i + 1 for i, v in enumerate(vals) if v == pi[f][m]]
                        ranks[m].append(sum(pos) / len(pos))
                k, nb = len(modes), len(pi)
                cd = Q05.get(k, float("nan")) * math.sqrt(k * (k + 1) / (6.0 * nb))
                avg = sorted(((statistics.mean(ranks[m]), m) for m in modes))
                lines.append(f"{algo} t={t} {scope:<7} N={nb:<3} Friedman p={p:.2e}  CD={cd:.2f}  ranks: "
                             + ", ".join(f"{m.replace('js_', '')} {r:.2f}" for r, m in avg))
    print("\n## T3. Friedman, average ranks, Nemenyi CD (alpha = 0.05)\n" + "\n".join(lines))
    if out:
        (out / "T3_friedman.txt").write_text("\n".join(lines) + "\n")


def wilcoxon_holm(rows, out: Path | None):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return
    ta = [r for r in rows if r["cls"]]
    lines = []
    for algo in sorted({r["algo"] for r in ta}):
        for t in sorted({r["t"] for r in ta if r["algo"] == algo}):
            sel = [r for r in ta if r["algo"] == algo and r["t"] == t]
            res = []
            for a, b in PAIRS:
                pi = per_instance([r for r in sel if r["mode"] in (a, b)], [a, b])
                if len(pi) < 6:
                    continue
                x = [pi[f][a] for f in pi]
                y = [pi[f][b] for f in pi]
                if all(u == v for u, v in zip(x, y)):
                    continue
                p = wilcoxon(x, y).pvalue
                res.append((p, a, b, statistics.mean(x), statistics.mean(y), len(pi)))
            res.sort()
            mcount = len(res)
            for i, (p, a, b, mx, my, nb) in enumerate(res):
                ph = min(1.0, p * (mcount - i))
                lines.append(f"{algo} t={t} {a.replace('js_', ''):>5} vs {b.replace('js_', ''):<5} N={nb:<3} "
                             f"{mx:6.2f} vs {my:6.2f}  p={p:.2e}  Holm p={ph:.2e}  {'*' if ph < 0.05 else ''}")
    print("\n## T4. Wilcoxon + Holm (key pairs)\n" + "\n".join(lines))
    if out:
        (out / "T4_wilcoxon.txt").write_text("\n".join(lines) + "\n")


def continuum(rows, out: Path | None):
    gen = [r for r in rows if r["file"].startswith("gen:")]
    if not gen:
        return
    lines = []
    for algo in sorted({r["algo"] for r in gen}):
        for t in sorted({r["t"] for r in gen if r["algo"] == algo}):
            sel = [r for r in gen if r["algo"] == algo and r["t"] == t]
            best = defaultdict(lambda: 10**12)
            for r in sel:
                best[r["file"]] = min(best[r["file"]], r["C"])
            modes = sorted({r["mode"] for r in sel})
            g = defaultdict(list)
            for r in sel:
                _, size, rho, _k = r["file"].split(":")
                g[(size, rho, r["mode"])].append(rpd(r["C"], best[r["file"]]))
            lines += ["", f"| {algo} t={t} ms: RPD to best | rho | " + " | ".join(m.replace("js_", "") for m in modes) + " |",
                      "|" + "---|" * (len(modes) + 2)]
            for size, rho in sorted({(k[0], k[1]) for k in g}, key=lambda x: (int(x[0].split("x")[0]), int(x[0].split("x")[1]), x[1])):
                v = [mean(g[(size, rho, m)]) for m in modes]
                b = min(v)
                lines.append(f"| {size} | {rho.split('=')[1]} | " + " | ".join(("**%.2f**" if x == b else "%.2f") % x for x in v) + " |")
    print("\n## T5. Continuum (RPD to the best run on the instance)" + "\n".join(lines))
    if out:
        (out / "T5_continuum.md").write_text("\n".join(lines) + "\n")


def bridge(rows, out: Path | None):
    pf = [r for r in rows if r["file"].startswith("pfsp:")]
    if not pf:
        return
    lines = []
    t = max(r["t"] for r in pf)
    sel = [r for r in pf if r["t"] == t]
    modes = sorted({r["mode"] for r in sel})
    lines += ["", f"| file (t={t} ms) | " + " | ".join(m.replace("js_", "") for m in modes) + " |", "|" + "---|" * (len(modes) + 1)]
    for f in sorted({r["file"].split(":")[1] for r in sel}):
        v = [mean([r["C"] for r in sel if r["file"].split(":")[1] == f and r["mode"] == m]) for m in modes]
        lines.append(f"| {f} | " + " | ".join("%.1f" % x for x in v) + " |")
    lines.append("Mean makespan of non-permutation (job shop) schedules; compare with the PFSP results of the same "
                 "instances (e.g. results/kbonacci/main_budgets, kbonacci_kn ILS 10 s).")
    print("\n## T6. Bridge: Taillard PFSP instances as job shops" + "\n".join(lines))
    if out:
        (out / "T6_bridge.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bks", default=None, help="optional JSON {instance: best known makespan}")
    a = ap.parse_args()
    over = json.loads(Path(a.bks).read_text()) if a.bks else {}
    rows = load(a.dirs, over)
    print(f"{len(rows)} results from {len(a.dirs)} directories")
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    neg = [r for r in rows if r["bks"] and r["C"] < r["bks"]]
    if neg:
        print(f"NOTE: {len(neg)} runs beat the stored BKS (update the BKS file!): "
              + ", ".join(sorted({"%s=%s" % (r["file"], r["C"]) for r in neg})[:10]))
    bad = [r for r in rows if r["lb"] and r["C"] < r["lb"]]
    if bad:
        print(f"ERROR: {len(bad)} runs below the lower bound -- inspect before using the data")
    table_rpd(rows, out)
    table_moves(rows, out)
    friedman(rows, out)
    wilcoxon_holm(rows, out)
    continuum(rows, out)
    bridge(rows, out)


if __name__ == "__main__":
    main()
