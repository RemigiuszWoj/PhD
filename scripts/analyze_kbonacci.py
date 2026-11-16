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
        --bks data/taillard_pfsp_bks.json --out articles/kbonacci/tables

BKS file format: {"tai20_5": [bks of instance 0, ..., instance 9], ..., "_meta": {...}}
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


def load(dirs, bks, reference=()):
    """Runs of `dirs` and of the `reference` campaign, one record per configuration.

    A configuration found twice keeps the run read last. Both campaigns check the time limit
    between iterations, so a run always finishes the move in progress and may exceed its budget by
    one move; such runs are kept, as in the earlier papers. A reference cell (neighborhood, class,
    budget) is dropped only when it is incomplete.
    """
    recs = {}
    for d in list(dirs) + list(reference):
        is_ref = d in reference
        for p in Path(d).rglob("result.json"):
            try:
                r = _normalize_result(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                print(f"[load] CORRUPT, skipped: {p}")
                continue
            c = r["config"]
            f = Path(c["instance_file"]).stem
            n, m = (int(x) for x in f[3:].split("_"))
            rec = {"algo": c["algorithm"], "neigh": c["neighborhood"], "file": f, "n": n, "m": m,
                   "inst": c["instance_number"], "seed": c["seed"], "tl": c["time_limit_ms"],
                   "rpd": r.get("gap_percent"), "cmax": r.get("cmax_best"),
                   "iters": r.get("iterations")}
            ub93 = r.get("upper_bound")
            if ub93:
                rec["rpd_ub93"] = 100.0 * (rec["cmax"] - ub93) / ub93
            if bks and f in bks:
                b = bks[f][c["instance_number"]]
                rec["rpd_bks"] = 100.0 * (rec["cmax"] - b) / b
            if rec["rpd"] is not None:
                recs[(rec["algo"], rec["neigh"], f, rec["inst"], rec["seed"], rec["tl"])] = rec
    # reference runs: only the algorithms of the campaign, only complete cells
    algos = {r["algo"] for r in recs.values() if r["neigh"] not in REF_ORDER[:4]}
    cells = _count(recs.values(), ("algo", "neigh", "file", "tl"))
    full = max(cells.values())
    short = sorted({(c[1], c[2], c[3]) for c, v in cells.items()
                    if v < full and c[1] in REF_ORDER[:4] and c[0] in algos})
    if short:
        print(f"[load] incomplete reference cells dropped: {short}")
    return [r for r in recs.values() if r["neigh"] not in REF_ORDER[:4]
            or (r["algo"] in algos and cells[(r["algo"], r["neigh"], r["file"], r["tl"])] == full)]


def _count(recs, fields):
    c = defaultdict(int)
    for r in recs:
        c[tuple(r[f] for f in fields)] += 1
    return c


def nemenyi_q(k: int) -> float:
    """q_{0.05} of the Nemenyi test for k methods (Demsar 2006): studentized range / sqrt(2)."""
    if k in Q05:
        return Q05[k]
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(0.95, k, 10**6)) / math.sqrt(2)


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
    recs = load(a.dirs, bks, a.reference)
    a.out.mkdir(parents=True, exist_ok=True)
    metrics = ["rpd", "rpd_ub93"] + (["rpd_bks"] if bks else [])
    neighs = sorted({r["neigh"] for r in recs}, key=neigh_sort_key)
    tls = sorted({r["tl"] for r in recs})
    ns = sorted({r["n"] for r in recs})
    algos = sorted({r["algo"] for r in recs})
    print(f"{len(recs)} results, neighborhoods: {neighs}")

    # 1) mean RPD and iterations per (algo, n, neigh, tl)  -- the main tables
    cell = defaultdict(list)
    for r in recs:
        cell[(r["algo"], r["n"], r["neigh"], r["tl"])].append(r)
    cell_f = defaultdict(list)
    for r in recs:
        cell_f[(r["algo"], r["file"], r["neigh"], r["tl"])].append(r)
    files_present = sorted({r["file"] for r in recs}, key=lambda f: tuple(int(x) for x in f[3:].split("_")))
    for metric in metrics + ["iters"]:
        for algo in algos:
            path = a.out / f"{metric}_{algo}_by_file.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["file", "neighborhood"] + [f"{t}ms" for t in tls])
                for f in files_present:
                    for nb in neighs:
                        v_row = []
                        for t in tls:
                            v = [x[metric] for x in cell_f.get((algo, f, nb, t), []) if x.get(metric) is not None]
                            v_row.append(f"{statistics.mean(v):.2f}" if v else "")
                        if any(v_row):
                            w.writerow([f, nb] + v_row)
    for metric in metrics + ["iters"]:
        for algo in algos:
            path = a.out / f"{metric}_{algo}.csv"
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["n", "neighborhood"] + [f"{t}ms" for t in tls])
                for n in ns:
                    n_files = len({r["file"] for r in recs if r["n"] == n})
                    for nb in neighs:
                        row = []
                        for t in tls:
                            got = cell.get((algo, n, nb, t), [])
                            v = [x[metric] for x in got if x.get(metric) is not None]
                            if nb in REF_ORDER[:4] and len({x["file"] for x in got}) < n_files:
                                v = []  # a reference row is averaged only over all classes of this n
                            row.append(f"{statistics.mean(v):.2f}" if v else "")
                        if any(row):
                            w.writerow([n, nb] + row)
            print(f"wrote {path}")

    # 2) Wilcoxon: every k-bonacci variant vs every reference neighborhood (paired on file, inst, seed)
    ref_present = [nb for nb in neighs if nb in REF_ORDER]
    kb = [nb for nb in neighs if nb.startswith("kbonacci_")]
    by = {(r["algo"], r["neigh"], r["file"], r["inst"], r["seed"], r["tl"]): r["rpd"] for r in recs}
    n_of = {f: int(f[3:].split("_")[0]) for f in {r["file"] for r in recs}}
    rows = []
    for algo in algos:
        for t in tls:
            for n_sel in ["all"] + ns:
                fam = []
                for k in kb:
                    for ref in ref_present:
                        pairs = [(v, by[(algo, ref, f, i, s, t)]) for (al, nb, f, i, s, tt), v in by.items()
                                 if al == algo and nb == k and tt == t and (n_sel == "all" or n_of[f] == n_sel)
                                 and (algo, ref, f, i, s, t) in by]
                        if len(pairs) < 6:
                            continue
                        x, y = zip(*pairs)
                        p = wilcoxon(x, y).pvalue if any(u != v for u, v in pairs) else 1.0
                        fam.append([algo, t, n_sel, k, ref, len(pairs), statistics.mean(x) - statistics.mean(y), p])
                for row, padj in zip(fam, holm([f[-1] for f in fam])):  # Holm within (algo, budget, n)
                    rows.append(row + [padj])
    with open(a.out / "wilcoxon.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "n", "kbonacci", "reference", "pairs", "mean_rpd_diff", "p", "p_holm"])
        w.writerows(rows)
    print(f"wrote {a.out / 'wilcoxon.csv'} ({len(rows)} tests)")

    # 2b) the same tests on instance means (10 instances per class): seeds of one instance are not independent
    inst = defaultdict(list)
    for (al, nb, f, i, s, t), v in by.items():
        inst[(al, nb, f, i, t)].append(v)
    inst = {key: statistics.mean(v) for key, v in inst.items()}

    def paired(algo, t, n_sel, x_nb, y_nb):
        pairs = [(v, inst[(algo, y_nb, f, i, t)]) for (al, nb, f, i, tt), v in inst.items()
                 if al == algo and nb == x_nb and tt == t and (n_sel == "all" or n_of[f] == n_sel)
                 and (algo, y_nb, f, i, t) in inst]
        if len(pairs) < 6:
            return None
        x, y = zip(*pairs)
        p = wilcoxon(x, y).pvalue if any(u != v for u, v in pairs) else 1.0
        return [len(pairs), statistics.mean(x) - statistics.mean(y), p]

    rows = []
    for algo in algos:
        for t in tls:
            for n_sel in ["all"] + ns:
                fam = [[algo, t, n_sel, k, ref] + res for k in kb for ref in ref_present
                       if (res := paired(algo, t, n_sel, k, ref))]
                for row, padj in zip(fam, holm([f[-1] for f in fam])):
                    rows.append(row + [padj])
    with open(a.out / "wilcoxon_inst.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "n", "kbonacci", "reference", "pairs", "mean_rpd_diff", "p", "p_holm"])
        w.writerows(rows)
    print(f"wrote {a.out / 'wilcoxon_inst.csv'} ({len(rows)} tests)")

    # 2c) between values of k: the best k of each filter (by mean RPD) against every other k of that filter
    rows = []
    for algo in algos:
        for t in tls:
            for n_sel in ns:
                for exact in (False, True):
                    ks = [k for k in kb if k.endswith("_cp") == exact]
                    means = {}
                    for k in ks:
                        v = [x for (al, nb, f, i, tt), x in inst.items()
                             if al == algo and nb == k and tt == t and n_of[f] == n_sel]
                        if v:
                            means[k] = statistics.mean(v)
                    if len(means) < 2:
                        continue
                    best = min(means, key=means.get)
                    fam = [[algo, t, n_sel, "exact" if exact else "block", best, k] + res
                           for k in means if k != best and (res := paired(algo, t, n_sel, k, best))]
                    for row, padj in zip(fam, holm([f[-1] for f in fam])):
                        rows.append(row + [padj])
    with open(a.out / "wilcoxon_k.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "n", "filter", "best_k", "other_k", "pairs", "mean_rpd_diff", "p", "p_holm"])
        w.writerows(rows)
    print(f"wrote {a.out / 'wilcoxon_k.csv'} ({len(rows)} tests)")

    # 3) Friedman + Nemenyi CD over neighborhoods with complete blocks (file, inst; seed-averaged)
    frows = []
    for algo, t, n_sel in [(al, tt, nn) for al in algos for tt in tls for nn in ["all"] + ns]:
        if True:
            blocks = defaultdict(dict)
            for r in recs:
                if r["algo"] == algo and r["tl"] == t and (n_sel == "all" or r["n"] == n_sel):
                    blocks[(r["file"], r["inst"])].setdefault(r["neigh"], []).append(r["rpd"])
            common = [nb for nb in neighs if all(nb in b for b in blocks.values())]
            if len(common) < 3 or len(blocks) < 5:  # "all": only neighborhoods present for every n
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
            cd = nemenyi_q(kk) * math.sqrt(kk * (kk + 1) / (6 * N))
            for i, nb in enumerate(common):
                frows.append([algo, t, n_sel, nb, round(ranks[i] / N, 3), N, round(stat, 2), p, round(cd, 3)])
    with open(a.out / "friedman.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algo", "tl_ms", "n", "neighborhood", "avg_rank", "blocks", "chi2", "p", "nemenyi_cd"])
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
