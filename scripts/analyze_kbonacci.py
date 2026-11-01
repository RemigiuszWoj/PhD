"""Tables and tests for the k-bonacci paper (same metric and test as the previous papers).

Main metric: RPD to the Taillard lower bound (gap_percent), two-sided Wilcoxon
signed-rank test, alpha = 0.05 -- as before. Added because they change no
computation: Holm-adjusted p-values, Friedman test with Nemenyi critical
difference over all neighborhoods, and RPD to best-known solutions if a BKS
file is given.

Usage:
    python scripts/analyze_kbonacci.py \
        --dirs results/kbonacci/main_budgets results/kbonacci/insert_budgets results/kbonacci/ablation_budgets \
        --reference results/experiments/<old campaign dir> [...] \
        [--bks data/taillard_bks.json] --out articles/kbonacci/tables

BKS file format: {"tai20_5": [bks of instance 0, ..., instance 9], ...}
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiments.aggregate import _normalize_result  # noqa: E402

REF_ORDER = ["adjacent", "fibonacci", "dynasearch", "motzkin", "insert"]
# Nemenyi q_{0.05} (Demsar 2006, extended): number of compared methods -> q
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
       10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426}


def neigh_sort_key(name: str):
    if name in REF_ORDER:
        return (0, REF_ORDER.index(name), 0)
    if name.startswith("kbonacci_k"):
        tail = name[len("kbonacci_k"):].replace("_cp", "")
        k = 10**6 if tail == "n" else int(tail)
        return (1, k, name.endswith("_cp"))
    return (2, 0, name)


def load(dirs, bks):
    recs = []
    for d in dirs:
        for p in Path(d).rglob("result.json"):
            r = _normalize_result(json.loads(p.read_text()))
            c = r["config"]
            f = Path(c["instance_file"]).stem
            n, m = (int(x) for x in f[3:].split("_"))
            rec = {"algo": c["algorithm"], "neigh": c["neighborhood"], "file": f, "n": n, "m": m,
                   "inst": c["instance_number"], "seed": c["seed"], "tl": c["time_limit_ms"],
                   "rpd": r.get("gap_percent"), "cmax": r.get("cmax_best"),
                   "iters": r.get("iterations")}
            if bks and f in bks:
                b = bks[f][c["instance_number"]]
                rec["rpd_bks"] = 100.0 * (rec["cmax"] - b) / b
            if rec["rpd"] is not None:
                recs.append(rec)
    return recs


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [1.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--bks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "articles" / "kbonacci" / "tables")
    a = ap.parse_args()
    from scipy.stats import friedmanchisquare, wilcoxon

    bks = json.loads(a.bks.read_text()) if a.bks else None
    recs = load(a.dirs + a.reference, bks)
    a.out.mkdir(parents=True, exist_ok=True)
    metrics = ["rpd"] + (["rpd_bks"] if bks else [])
    neighs = sorted({r["neigh"] for r in recs}, key=neigh_sort_key)
    tls = sorted({r["tl"] for r in recs})
    ns = sorted({r["n"] for r in recs})
    algos = sorted({r["algo"] for r in recs})
    print(f"{len(recs)} results, neighborhoods: {neighs}")

    # 1) mean RPD and iterations per (algo, n, neigh, tl)  -- the main tables
    cell = defaultdict(list)
    for r in recs:
        cell[(r["algo"], r["n"], r["neigh"], r["tl"])].append(r)
    for metric in metrics + ["iters"]:
        for algo in algos:
            path = a.out / f"{metric}_{algo}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["n", "neighborhood"] + [f"{t}ms" for t in tls])
                for n in ns:
                    for nb in neighs:
                        row = []
                        for t in tls:
                            v = [x[metric] for x in cell.get((algo, n, nb, t), []) if x.get(metric) is not None]
                            row.append(f"{statistics.mean(v):.2f}" if v else "")
                        if any(row):
                            w.writerow([n, nb] + row)
            print(f"wrote {path}")

    # 2) Wilcoxon: every k-bonacci variant vs every reference neighborhood (paired on file, inst, seed)
    ref_present = [nb for nb in neighs if nb in REF_ORDER]
    kb = [nb for nb in neighs if nb.startswith("kbonacci_")]
    by = {(r["algo"], r["neigh"], r["file"], r["inst"], r["seed"], r["tl"]): r["rpd"] for r in recs}
    rows = []
    for algo in algos:
        for t in tls:
            fam = []
            for k in kb:
                for ref in ref_present:
                    pairs = [(v, by[(algo, ref, f, i, s, t)]) for (al, nb, f, i, s, tt), v in by.items()
                             if al == algo and nb == k and tt == t and (algo, ref, f, i, s, t) in by]
                    if len(pairs) < 6:
                        continue
                    x, y = zip(*pairs)
                    p = wilcoxon(x, y).pvalue if any(u != v for u, v in pairs) else 1.0
                    fam.append([algo, t, k, ref, len(pairs), statistics.mean(x) - statistics.mean(y), p])
            for row, padj in zip(fam, holm([f[-1] for f in fam])):
                rows.append(row + [padj])
    with open(a.out / "wilcoxon.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "kbonacci", "reference", "pairs", "mean_rpd_diff", "p", "p_holm"])
        w.writerows(rows)
    print(f"wrote {a.out / 'wilcoxon.csv'} ({len(rows)} tests)")

    # 3) Friedman + Nemenyi CD over neighborhoods with complete blocks (file, inst; seed-averaged)
    frows = []
    for algo in algos:
        for t in tls:
            blocks = defaultdict(dict)
            for r in recs:
                if r["algo"] == algo and r["tl"] == t:
                    blocks[(r["file"], r["inst"])].setdefault(r["neigh"], []).append(r["rpd"])
            common = [nb for nb in neighs if all(nb in b for b in blocks.values())]
            if len(common) < 3 or len(blocks) < 5:
                continue
            data = [[statistics.mean(blocks[b][nb]) for b in blocks] for nb in common]
            stat, p = friedmanchisquare(*data)
            ranks = defaultdict(float)
            for j in range(len(blocks)):
                vals = sorted((data[i][j], i) for i in range(len(common)))
                pos = 0
                while pos < len(vals):  # average ranks for ties
                    end = pos
                    while end + 1 < len(vals) and vals[end + 1][0] == vals[pos][0]:
                        end += 1
                    for q in range(pos, end + 1):
                        ranks[vals[q][1]] += (pos + end) / 2 + 1
                    pos = end + 1
            kk, N = len(common), len(blocks)
            cd = Q05.get(kk, float("nan")) * math.sqrt(kk * (kk + 1) / (6 * N))
            for i, nb in enumerate(common):
                frows.append([algo, t, nb, round(ranks[i] / N, 3), N, round(stat, 2), p, round(cd, 3)])
    with open(a.out / "friedman.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "neighborhood", "avg_rank", "blocks", "chi2", "p", "nemenyi_cd"])
        w.writerows(frows)
    print(f"wrote {a.out / 'friedman.csv'}")

    # 4) best k per (algo, n, tl)
    print("\nbest k-bonacci variant per (algo, n, budget) by mean RPD:")
    for algo in algos:
        for n in ns:
            line = []
            for t in tls:
                best = min(((statistics.mean(x["rpd"] for x in cell[(algo, n, k, t)]), k)
                            for k in kb if cell.get((algo, n, k, t))), default=None)
                line.append(f"{t}ms:{best[1].replace('kbonacci_', '')}" if best else f"{t}ms:-")
            print(f"  {algo} n={n:<3} " + "  ".join(line))


if __name__ == "__main__":
    main()
