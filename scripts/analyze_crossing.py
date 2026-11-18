"""Tables and tests for the crossing paper (articles/crossings/PLAN.md, v3: E1, E2, E3, E5).

Endpoint: RPD to the Taillard lower bound (gap_percent) at 10 s. Blocks are instances: the five seeds are
averaged first, so a class gives 10 pairs. Confirmatory families fixed before the campaign (PLAN 7.4a, v3):
  P1 (H3)  inv vs c0, plain, 12 classes, Holm;  decisive classes n <= 100, criterion >= 7/9 significant;
  P2 (H9') dyn vs c0, plain, two-sided, 12 classes, Holm;
  Page trend test along dyn < c0 < c1 < pa1 < inv, per class and pooled over the classes n <= 100.
Everything else is exploratory and reported without correction: dyn vs swap, every family vs c0, H3 at equal
iterations (move_log), shorter budgets, safe vs plain (E5), exact filter (E3), anatomy (E1).

Usage (repo root):
    python scripts/analyze_crossing.py --out articles/crossings/tables
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "crossings"
CHAIN = ["dyn", "c0", "c1", "pa1", "inv"]
PLAIN = CHAIN + ["swap"]
FILES = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20", "tai100_5", "tai100_10",
         "tai100_20", "tai200_10", "tai200_20", "tai500_20"]
DECISIVE = [f for f in FILES if int(f[3:].split("_")[0]) <= 100]
KB_REF = ["kbonacci_k2", "kbonacci_k3", "kbonacci_k4", "kbonacci_k8", "kbonacci_kn"]   # PLAN E2: k in {2, 3, 4, 8, n}


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [1.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def load(dirs, base=RES):
    """{(variant, file, inst, seed, tl): record}; the variant drops the 'cross_' prefix."""
    out = {}
    for d in dirs:
        for p in (base / d).glob("algo=ils__*/result.json"):  # ILS only (k-bonacci dirs also hold SA runs)
            r = json.loads(p.read_text())
            c = r["config"]
            f = Path(c["instance_file"]).stem
            v = c["neighborhood"].removeprefix("cross_")
            out[(v, f, c["instance_number"], c["seed"], c["time_limit_ms"])] = r
    return out


def inst_means(runs, v, f, tl, key="gap_percent"):
    """Mean over seeds per instance, as a list over instances 0..9 (None if a seed is missing)."""
    res = []
    for i in range(10):
        xs = [runs[(v, f, i, s, tl)][key] for s in range(5) if (v, f, i, s, tl) in runs]
        res.append(statistics.mean(xs) if len(xs) == 5 else None)
    return res


def wilcoxon_p(x, y):
    from scipy.stats import wilcoxon
    d = [a - b for a, b in zip(x, y)]
    return 1.0 if all(abs(u) < 1e-12 for u in d) else float(wilcoxon(x, y).pvalue)


def a12(runs, a, b, f, tl):
    """Vargha-Delaney A12 that a is better (smaller RPD) than b: computed per instance over the 5 x 5 seed
    pairs, so instances of different difficulty are not mixed, then averaged; 0.5 = no effect."""
    vals = []
    for i in range(10):
        x = [runs[(a, f, i, s, tl)]["gap_percent"] for s in range(5) if (a, f, i, s, tl) in runs]
        y = [runs[(b, f, i, s, tl)]["gap_percent"] for s in range(5) if (b, f, i, s, tl) in runs]
        if x and y:
            vals.append(sum((u < v) + 0.5 * (u == v) for u in x for v in y) / (len(x) * len(y)))
    return statistics.mean(vals)


def compare(runs, a, b, files, tl=10000):
    """One row per class: mean RPD of a and b, difference, wins of a, A12 of a, raw p."""
    rows = []
    for f in files:
        x, y = inst_means(runs, a, f, tl), inst_means(runs, b, f, tl)
        if None in x or None in y:
            continue
        rows.append({"file": f, "a": a, "b": b, "rpd_a": statistics.mean(x), "rpd_b": statistics.mean(y),
                     "diff": statistics.mean(x) - statistics.mean(y),
                     "a_better": sum(u < v for u, v in zip(x, y)), "a12": a12(runs, a, b, f, tl), "p": wilcoxon_p(x, y)})
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def table(lines, title, cols, rows_by_file, fmt="{:.2f}"):
    lines.append(f"\n{title}\n" + f"{'file':>10} " + " ".join(f"{c:>8}" for c in cols))
    for f, vals in rows_by_file:
        lines.append(f"{f:>10} " + " ".join(f"{fmt.format(v):>8}" if v is not None else f"{'-':>8}" for v in vals))


def family(lines, name, rows, decisive_rule=True):
    adj = holm([r["p"] for r in rows])
    lines.append(f"\n{name}  (diff = a - b in RPD pp; wins = instances where a is better, of 10)")
    lines.append(f"{'file':>10} {'rpd_a':>7} {'rpd_b':>7} {'diff':>7} {'wins':>5} {'A12':>5} {'p':>8} {'p_holm':>8}")
    for r, pa in zip(rows, adj):
        r["p_holm"] = pa
        mark = "*" if pa < 0.05 else ""
        a = f"{r['a12']:5.2f}" if "a12" in r else f"{'-':>5}"
        lines.append(f"{r['file']:>10} {r['rpd_a']:7.2f} {r['rpd_b']:7.2f} {r['diff']:+7.2f} {r['a_better']:>5} {a} "
                     f"{r['p']:8.4f} {pa:8.4f}{mark}")
    if decisive_rule:
        dec = [r for r in rows if r["file"] in DECISIVE]
        lines.append(f"decisive classes (n <= 100): significant after Holm {sum(r['p_holm'] < 0.05 for r in dec)}/"
                     f"{len(dec)}, of them b better {sum(r['p_holm'] < 0.05 and r['diff'] > 0 for r in dec)}, "
                     f"a better {sum(r['p_holm'] < 0.05 and r['diff'] < 0 for r in dec)}")
    return rows


def equal_iterations(runs, a, b, files):
    """H3 at equal iterations: per run pair (same file, inst, seed) take I = the smaller iteration count and
    compare the best Cmax after I moves (move_log), as RPD; seeds averaged per instance."""
    rows = []
    for f in files:
        x_i, y_i = [], []
        for i in range(10):
            xs, ys = [], []
            for s in range(5):
                ra, rb = runs.get((a, f, i, s, 10000)), runs.get((b, f, i, s, 10000))
                if not ra or not rb or not ra["move_log"] or not rb["move_log"]:
                    continue
                I = min(len(ra["move_log"]), len(rb["move_log"]))
                lb = ra["lower_bound"]
                best = lambda r: min(c for _, c in r["move_log"][:I])
                xs.append(100 * (best(ra) - lb) / lb)
                ys.append(100 * (best(rb) - lb) / lb)
            if xs:
                x_i.append(statistics.mean(xs))
                y_i.append(statistics.mean(ys))
        if len(x_i) >= 6:
            rows.append({"file": f, "a": a, "b": b, "rpd_a": statistics.mean(x_i), "rpd_b": statistics.mean(y_i),
                         "diff": statistics.mean(x_i) - statistics.mean(y_i),
                         "a_better": sum(u < v for u, v in zip(x_i, y_i)), "p": wilcoxon_p(x_i, y_i)})
    return rows


def page(runs, files, tl=10000):
    """Page's L test for RPD increasing along CHAIN, blocks = instances."""
    from scipy.stats import page_trend_test
    data = []
    for f in files:
        cols = [inst_means(runs, v, f, tl) for v in CHAIN]
        data += [list(r) for r in zip(*cols) if None not in r]
    res = page_trend_test(data, ranked=False, predicted_ranks=list(range(1, len(CHAIN) + 1)))
    return len(data), float(res.statistic), float(res.pvalue)


def friedman(runs, files, variants, tl=10000):
    """Friedman test over `variants` per class, blocks = instances; mean ranks (1 = best)."""
    from scipy.stats import friedmanchisquare, rankdata
    rows = []
    for f in files:
        cols = [inst_means(runs, v, f, tl) for v in variants]
        blocks = [r for r in zip(*cols) if None not in r]
        ranks = [rankdata(b) for b in blocks]
        mr = [statistics.mean(r[j] for r in ranks) for j in range(len(variants))]
        rows.append({"file": f, **{f"rank_{v}": m for v, m in zip(variants, mr)},
                     "p": float(friedmanchisquare(*zip(*blocks)).pvalue)})
    return rows


def anatomy(lines):
    from scipy.stats import wilcoxon
    fam_names = ["dyn", "c0", "c1", "pa1", "inv"]
    by_file = defaultdict(list)
    for p in sorted((RES / "anatomy").glob("*.json")):
        d = json.loads(p.read_text())
        by_file[d["file"]] += [s for s in d["states"] if "fam" in s]
    rows_vs, rows_ratio, rows_pred, rows_pairs, rows_minc, csv_rows = [], [], [], [], [], []
    for f in FILES:
        st = by_file.get(f)
        if not st:
            continue
        vs = [100 * statistics.mean(s["fam"][k]["vs_single"] == "worse" for s in st) for k in fam_names]
        ratio = []
        for k in fam_names:
            pr = sum(s["fam"][k]["pred"] for s in st)
            ratio.append(sum(s["fam"][k]["real"] for s in st) / pr if pr else None)
        pred = [100 * statistics.mean(s["fam"][k]["pred"] / s["fam"]["inv"]["pred"] for s in st
                                      if s["fam"]["inv"]["pred"] < 0) for k in fam_names]
        geo = {}
        for g in ("disjoint", "nested", "crossing"):
            pr = [q for s in st for q in s["pairs"][g]]
            geo[g] = sum(q[1] for q in pr) / sum(q[0] for q in pr) if pr else None
        # per state ratio real/pred of pairs, disjoint vs overlapping (nested + crossing), paired by state
        dj, ov = [], []
        for s in st:
            a = [q for q in s["pairs"]["disjoint"]]
            b = s["pairs"]["nested"] + s["pairs"]["crossing"]
            if a and b:
                dj.append(sum(q[1] for q in a) / sum(q[0] for q in a))
                ov.append(sum(q[1] for q in b) / sum(q[0] for q in b))
        p_geo = float(wilcoxon(dj, ov).pvalue) if len(dj) >= 6 else None
        minc = [100 * statistics.mean(s["min_c"] == c for s in st) for c in (0, 1, "2+")]
        rows_vs.append((f, vs))
        rows_ratio.append((f, ratio))
        rows_pred.append((f, pred))
        rows_pairs.append((f, [geo["disjoint"], geo["nested"], geo["crossing"], p_geo]))
        rows_minc.append((f, minc + [len(st)]))
        csv_rows.append({"file": f, "states": len(st), **{f"worse_{k}": v for k, v in zip(fam_names, vs)},
                         **{f"real_over_pred_{k}": v for k, v in zip(fam_names, ratio)},
                         **{f"pred_share_{k}": v for k, v in zip(fam_names, pred)},
                         "pairs_disjoint": geo["disjoint"], "pairs_nested": geo["nested"],
                         "pairs_crossing": geo["crossing"], "p_disjoint_vs_overlap": p_geo,
                         "minc0": minc[0], "minc1": minc[1], "minc2plus": minc[2]})
    lines.append("\n=== E1 anatomy (states from plain ILS runs, P-cores) ===")
    table(lines, "Compound move worse than the best single swap [% of states]", fam_names, rows_vs, "{:.0f}")
    table(lines, "Realised / predicted gain (sum over states)", fam_names, rows_ratio)
    table(lines, "H1: predicted gain as % of the involution optimum (states with inv gain < 0)", fam_names,
          rows_pred, "{:.0f}")
    table(lines, "H2: pair additivity real/pred by geometry; p = Wilcoxon disjoint vs overlapping per state",
          ["disj", "nest", "cross", "p"], rows_pairs, "{:.3g}")
    table(lines, "Smallest c with OPT_c = OPT_inv [% of states]", ["c=0", "c=1", "c>=2", "states"], rows_minc,
          "{:.0f}")
    return csv_rows


CARD_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10**9)]


def cardinality(lines):
    """Realised/predicted ratio by move cardinality (swaps per move), pooled over anatomy states with an
    improving single swap; separates geometry from the number of swaps (reviewer objection, 18.11)."""
    fam_names = ["dyn", "c0", "pa1", "inv"]
    rows = []  # (fam, swaps, ratio, nest, worse)
    for p in sorted((RES / "anatomy").glob("*.json")):
        for s in json.loads(p.read_text())["states"]:
            if "fam" not in s or s["best_single"] >= 0:
                continue
            for k in fam_names:
                x = s["fam"][k]
                if x["pred"] < 0:
                    rows.append((k, x["swaps"], x["real"] / x["pred"], x["nest"], x["vs_single"] == "worse"))
    med = lambda v: statistics.median(v) if v else None
    lab = lambda a, b: f"{a}" if a == b else (f"{a}-{b}" if b < 10**9 else f"{a}+")
    t_ratio, t_worse, csv_rows = [], [], []
    for a, b in CARD_BINS:
        r = [med([q[2] for q in rows if q[0] == k and a <= q[1] <= b]) for k in fam_names]
        w = [None if not (v := [q[4] for q in rows if q[0] == k and a <= q[1] <= b]) else 100 * statistics.mean(v)
             for k in fam_names]
        n = [sum(1 for q in rows if q[0] == k and a <= q[1] <= b) for k in fam_names]
        t_ratio.append((lab(a, b), r + n))
        t_worse.append((lab(a, b), w))
        csv_rows.append({"swaps": lab(a, b), **{f"ratio_{k}": v for k, v in zip(fam_names, r)},
                         **{f"worse_{k}": v for k, v in zip(fam_names, w)}, **{f"n_{k}": v for k, v in zip(fam_names, n)}})
    sw = [(k, sorted(q[1] for q in rows if q[0] == k)) for k in fam_names]
    nest = [(lab(a, b), [med([q[2] for q in rows if q[0] == "c0" and 3 <= q[1] <= 8 and a <= q[3] <= b]),
                         sum(1 for q in rows if q[0] == "c0" and 3 <= q[1] <= 8 and a <= q[3] <= b)])
            for a, b in [(0, 0), (1, 2), (3, 6), (7, 10**9)]]
    lines.append("\n=== E1b cardinality: swaps per move vs additivity (anatomy states, pooled) ===")
    lines.append("Swaps per move, median [q1, q3]: " + ", ".join(
        f"{k} {v[len(v)//2]} [{v[len(v)//4]}, {v[3*len(v)//4]}]" for k, v in sw if v))
    table(lines, "Median realised/predicted by swaps per move (then n states per family)",
          fam_names + [f"n_{k}" for k in fam_names], t_ratio, "{:.2f}")
    table(lines, "Compound move worse than best single swap [%] by swaps per move", fam_names, t_worse, "{:.0f}")
    table(lines, "N0, 3-8 swaps: median realised/predicted by number of nestings", ["ratio", "n"], nest, "{:.2f}")
    return csv_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "articles" / "crossings" / "tables")
    ap.add_argument("--kbonacci", type=Path, default=Path.home() / "Desktop" / "PhD-kbonacci" / "results" / "kbonacci",
                    help="k-bonacci campaign (same protocol and machine) for the reference rows; skipped if missing")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    full = load(["main", "safe", "ablation"])                       # 10 s runs with move_log and stats
    bud = load(["main_budgets", "safe_budgets", "ablation_budgets"])  # extracted shorter budgets
    lines = [f"{len(full)} runs of 10 s, {len(bud)} budget records"]

    # E2 main table, iterations, budgets
    table(lines, "E2: mean RPD to LB [%] at 10 s (plain)", PLAIN,
          [(f, [statistics.mean(inst_means(full, v, f, 10000)) for v in PLAIN]) for f in FILES])
    table(lines, "E2: mean iterations in 10 s (plain)", PLAIN,
          [(f, [statistics.mean(inst_means(full, v, f, 10000, "iterations")) for v in PLAIN]) for f in FILES],
          "{:.0f}")
    tls = sorted({k[4] for k in bud})
    for tl in tls:
        if tl == 10000:
            continue
        table(lines, f"E2 (descriptive): mean RPD at {tl / 1000:g} s (plain)", PLAIN,
              [(f, [statistics.mean(inst_means(bud, v, f, tl)) for v in PLAIN]) for f in FILES])
    rows_tbl = [{"file": f, "tl_ms": tl, "variant": v, "rpd": statistics.mean(inst_means(bud, v, f, tl)),
                 "iters": statistics.mean(inst_means(bud, v, f, tl, "iterations"))}
                for tl in tls for f in FILES for v in sorted({k[0] for k in bud})
                if None not in inst_means(bud, v, f, tl)]
    write_csv(a.out / "rpd_by_file_budget.csv", rows_tbl)

    # confirmatory families
    lines.append("\n=== Confirmatory (PLAN 7.4a, v3) ===")
    p1 = family(lines, "P1 (H3): inv vs c0, plain, Holm over 12 classes", compare(full, "inv", "c0", FILES))
    p2 = family(lines, "P2 (H9'): dyn vs c0, plain, two-sided, Holm over 12 classes", compare(full, "dyn", "c0", FILES))
    pg = []
    for f in FILES:
        nb, L, p = page(full, [f])
        pg.append({"file": f, "blocks": nb, "L": L, "p": p})
    nb, L, p = page(full, DECISIVE)
    pg.append({"file": "pooled n<=100", "blocks": nb, "L": L, "p": p})
    lines.append("\nPage trend test, RPD increasing along dyn < c0 < c1 < pa1 < inv (blocks = instances)")
    lines += [f"{r['file']:>14}  blocks {r['blocks']:>3}  L = {r['L']:8.1f}  p = {r['p']:.3g}" for r in pg]
    write_csv(a.out / "p1_inv_vs_c0.csv", p1)
    write_csv(a.out / "p2_dyn_vs_c0.csv", p2)
    write_csv(a.out / "page.csv", pg)

    # exploratory
    lines.append("\n=== Exploratory (no correction across comparisons; Holm only inside each block) ===")
    expl = []
    for x, y in [("dyn", "swap"), ("c0", "swap"), ("c1", "c0"), ("pa1", "c0"), ("pa1", "c1"), ("inv", "swap")]:
        expl += family(lines, f"{x} vs {y}, plain", compare(full, x, y, FILES), decisive_rule=False)
    eq = family(lines, "H3 at equal iterations: inv vs c0 (best Cmax after the same number of moves)",
                equal_iterations(full, "inv", "c0", FILES))
    eq += family(lines, "Equal iterations: dyn vs c0", equal_iterations(full, "dyn", "c0", FILES), False)
    lines.append("\n--- E5: safeguarded variants (exploratory) ---")
    table(lines, "E5: mean RPD at 10 s (safe)", [f"{v}-s" for v in CHAIN],
          [(f, [statistics.mean(inst_means(full, f"{v}_safe", f, 10000)) for v in CHAIN]) for f in FILES])
    safe = []
    for v in CHAIN:
        safe += family(lines, f"E5: {v}-safe vs {v}-plain", compare(full, f"{v}_safe", v, FILES), False)
    safe += family(lines, "E5: c0-safe vs dyn-plain", compare(full, "c0_safe", "dyn", FILES), False)
    lines.append("\n--- E3: exact critical-path filter (exploratory) ---")
    abl = []
    for v in ("c0", "inv"):
        abl += family(lines, f"E3: {v}_cp vs {v} (both plain)", compare(full, f"{v}_cp", v, FILES), False)
    fr = friedman(full, FILES, PLAIN)
    lines.append("\nFriedman over the six plain variants, mean rank per class (1 = best)")
    lines.append(f"{'file':>10} " + " ".join(f"{v:>6}" for v in PLAIN) + f" {'p':>9}")
    lines += [f"{r['file']:>10} " + " ".join(f"{r['rank_' + v]:6.2f}" for v in PLAIN) + f" {r['p']:9.2g}" for r in fr]
    write_csv(a.out / "friedman.csv", fr)

    ref = []
    if (a.kbonacci / "main").is_dir():
        kb = {k: r for k, r in load(["main"], a.kbonacci).items() if k[0] in KB_REF and k[4] == 10000}
        both = {**full, **kb}
        lines.append("\n--- Reference: k-bonacci campaign (same protocol, same Mac; exploratory) ---")
        table(lines, "Mean RPD at 10 s: dyn and c0 here, k-bonacci k = 2, 3, 4, 8, n",
              ["dyn", "c0"] + [k.removeprefix("kbonacci_") for k in KB_REF],
              [(f, [statistics.mean(m) if None not in (m := inst_means(both, v, f, 10000)) else None
                    for v in ["dyn", "c0"] + KB_REF]) for f in FILES])
        ref += family(lines, "dyn vs k-bonacci k = n (two codes of the disjoint-swap family)",
                      compare(both, "dyn", "kbonacci_kn", FILES), False)
        ref += family(lines, "dyn vs k-bonacci k = 2 (Fibonacci)", compare(both, "dyn", "kbonacci_k2", FILES), False)
    write_csv(a.out / "exploratory.csv", expl + eq + safe + abl + ref)

    write_csv(a.out / "anatomy.csv", anatomy(lines))
    write_csv(a.out / "cardinality.csv", cardinality(lines))
    text = "\n".join(lines) + "\n"
    (a.out / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
