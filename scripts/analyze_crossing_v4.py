"""Analysis of the crossing stage v4 (articles/crossings/PLAN.md, v4): equal cardinality (card), exact
acceptance among several moves (top), NEH/IG context, pair-corrected QUBO in the ILS (qubo), candidate caps
(lim) and the anatomy extras (anatomy_v4: restricted c2 optimum, QUBO prediction against real gain).

Same endpoint and blocks as analyze_crossing.py: RPD to the Taillard lower bound at 10 s, instance = block
(mean over 5 seeds), Wilcoxon signed-rank with Holm inside each comparison. Everything here is exploratory.
Writes tables/v4_*.csv and tables/summary_v4.txt.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_crossing import RES, ROOT, compare, family, inst_means, table, write_csv

OUT = ROOT / "articles" / "crossings" / "tables"
SMALL = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20", "tai100_5", "tai100_10", "tai100_20"]


def load_any(dirs):
    """Like analyze_crossing.load, but for every algorithm; the variant is '<algo>:<neigh>' unless algo = ils."""
    out = {}
    for d in dirs:
        for p in (RES / d).glob("algo=*/result.json"):
            r = json.loads(p.read_text())
            c = r["config"]
            v = c["neighborhood"].removeprefix("cross_")
            if c["algorithm"] != "ils":
                v = f"{c['algorithm']}:{v}"
            out[(v, Path(c["instance_file"]).stem, c["instance_number"], c["seed"], c["time_limit_ms"])] = r
    return out


def mean_table(lines, runs, title, variants, files):
    table(lines, title, variants,
          [(f, [statistics.mean(m) if None not in (m := inst_means(runs, v, f, 10000)) else None for v in variants])
           for f in files])


def block(lines, runs, pairs, files):
    rows = []
    for a, b in pairs:
        rows += family(lines, f"{a} vs {b}", compare(runs, a, b, files), decisive_rule=False)
    return rows


def anatomy_v4(lines):
    """Restricted optimum (top 60 improving swaps): how often c <= 2 reaches the involution optimum; QUBO
    model on the top 16: predicted vs real gain of the pair-corrected optimum and of dyn/c0/inv."""
    agg = defaultdict(lambda: defaultdict(list))
    minc = defaultdict(lambda: defaultdict(int))
    for p in sorted((RES / "anatomy_v4").glob("*.json")):
        d = json.loads(p.read_text())
        f = d["file"]
        for s in d["states"]:
            v4 = s.get("v4")
            if not v4:
                continue
            r = v4["restricted"]
            minc[f][str(r["min_c"])] += 1
            q = v4["q"]
            b1 = q["best_single"]
            for fam, x in q["fam"].items():
                if x["pred"] < 0:
                    agg[f][f"{fam}_ratio"].append(x["real"] / x["pred"])
                agg[f][f"{fam}_worse_single"].append(float(x["real"] > b1))
                agg[f][f"{fam}_real"].append(x["real"])
                agg[f][f"{fam}_swaps"].append(x["swaps"])
    rows = []
    lines.append("\n--- anatomy_v4: restricted (top 60) min c with OPT_c = OPT_inv; share of states ---")
    lines.append(f"{'file':>10} {'states':>6} {'c=0':>6} {'c=1':>6} {'c=2':>6} {'3+':>6}")
    for f in sorted(minc, key=lambda x: (int(x[3:].split('_')[0]), int(x.split('_')[1]))):
        t = sum(minc[f].values())
        sh = [minc[f].get(k, 0) / t for k in ("0", "1", "2", "3+")]
        lines.append(f"{f:>10} {t:>6} " + " ".join(f"{x:6.2f}" for x in sh))
        rows.append({"file": f, "kind": "min_c", "states": t, **{f"c{k}": x for k, x in zip(("0", "1", "2", "3p"), sh)}})
    lines.append("\n--- anatomy_v4: top-16 QUBO; median real/pred, share of moves worse than the best single swap ---")
    fams = ["dyn", "c0", "inv", "q"]
    lines.append(f"{'file':>10} " + " ".join(f"{x + '_r/p':>9}" for x in fams) + " " + " ".join(f"{x + '_w':>7}" for x in fams))
    for f in sorted(agg, key=lambda x: (int(x[3:].split('_')[0]), int(x.split('_')[1]))):
        a = agg[f]
        rp = [statistics.median(a[f"{x}_ratio"]) if a[f"{x}_ratio"] else None for x in fams]
        w = [statistics.mean(a[f"{x}_worse_single"]) for x in fams]
        lines.append(f"{f:>10} " + " ".join(f"{v:9.2f}" if v is not None else f"{'-':>9}" for v in rp) + " "
                     + " ".join(f"{v:7.2f}" for v in w))
        rows.append({"file": f, "kind": "qubo16", **{f"{x}_real_over_pred": v for x, v in zip(fams, rp)},
                     **{f"{x}_worse_than_single": v for x, v in zip(fams, w)},
                     **{f"{x}_swaps_median": statistics.median(a[f"{x}_swaps"]) for x in fams}})
    return rows


def main():
    base = load_any(["main", "safe"])
    lines = ["Stage v4, exploratory. RPD to LB [%] at 10 s, blocks = instances (5 seeds), Holm inside each block."]

    card = {**base, **load_any(["card"])}
    lines.append("\n=== card: moves cut to the K most improving swaps ===")
    v = ["dyn_k2", "c0_k2", "pa1_k2", "inv_k2", "c0_k4", "pa1_k4", "inv_k4", "dyn", "c0", "inv", "swap"]
    mean_table(lines, card, "mean RPD", v, SMALL)
    rc = block(lines, card, [("inv_k2", "c0_k2"), ("inv_k2", "dyn_k2"), ("c0_k2", "dyn_k2"), ("inv_k4", "c0_k4"),
                             ("c0_k2", "c0"), ("inv_k2", "inv"), ("dyn_k2", "dyn"), ("dyn_k2", "swap")], SMALL)

    top = {**base, **load_any(["top"])}
    lines.append("\n=== top: exact acceptance among the first move, its alternatives and the best single swap ===")
    v = ["dyn_top8", "c0_top8", "inv_top8", "dyn_safe", "c0_safe", "inv_safe", "dyn", "swap"]
    mean_table(lines, top, "mean RPD", v, SMALL)
    rt = block(lines, top, [("c0_top8", "c0_safe"), ("inv_top8", "inv_safe"), ("dyn_top8", "dyn"),
                            ("c0_top8", "dyn"), ("inv_top8", "dyn"), ("inv_top8", "c0_top8")], SMALL)

    ctx = {**base, **load_any(["context"])}
    lines.append("\n=== context: ILS from NEH and Iterated Greedy (IG_RS) ===")
    v = ["ils_neh:dyn", "ils_neh:swap", "ig:neh_insert", "dyn", "swap"]
    mean_table(lines, ctx, "mean RPD", v, SMALL)
    rx = block(lines, ctx, [("ils_neh:dyn", "ils_neh:swap"), ("ils_neh:dyn", "dyn"), ("ig:neh_insert", "ils_neh:dyn")],
               SMALL)

    qb = {**base, **load_any(["qubo", "card", "top"])}
    files_q = SMALL[:6]
    lines.append("\n=== qubo: pair-corrected QUBO on the 16 most improving swaps, solved exactly, inside the ILS ===")
    v = ["q16", "dyn", "c0", "inv", "swap", "c0_safe", "inv_safe", "c0_top8", "inv_top8"]
    mean_table(lines, qb, "mean RPD", v, files_q)
    rq = block(lines, qb, [("q16", "inv"), ("q16", "c0"), ("q16", "dyn"), ("q16", "swap"), ("q16", "inv_safe"),
                           ("q16", "inv_top8")], files_q)

    lm = {**base, **load_any(["lim"])}
    files_l = sorted({k[1] for k in load_any(["lim"])}, key=lambda x: (int(x[3:].split('_')[0]), int(x.split('_')[1])))
    lines.append("\n=== lim: candidate caps T x C at n = 100 and 200 ===")
    v = ["c0_lim10000x2000", "c0_lim1000x100", "inv_lim10000x2000", "inv_lim1000x100", "c0", "inv", "dyn"]
    mean_table(lines, lm, "mean RPD", v, files_l)
    table(lines, "mean iterations in 10 s", v,
          [(f, [statistics.mean(m) if None not in (m := inst_means(lm, x, f, 10000, "iterations")) else None for x in v])
           for f in files_l], "{:.0f}")
    rl = block(lines, lm, [("c0_lim1000x100", "c0"), ("inv_lim1000x100", "inv"), ("c0_lim10000x2000", "c0"),
                           ("inv_lim10000x2000", "inv")], files_l)

    write_csv(OUT / "v4_tests.csv", rc + rt + rx + rq + rl)
    an = anatomy_v4(lines)
    write_csv(OUT / "v4_anatomy_minc.csv", [r for r in an if r["kind"] == "min_c"])
    write_csv(OUT / "v4_anatomy_qubo16.csv", [r for r in an if r["kind"] == "qubo16"])
    text = "\n".join(lines) + "\n"
    (OUT / "summary_v4.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
