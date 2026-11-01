"""Print the pilot tables quoted in ../PLAN.md from results/*.json.

Usage: python analyze_pilots.py [anatomy|descent|pairq|ils|qubo|all]
"""
import glob, json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
RES = Path(__file__).resolve().parent / "results"

def anatomy():
    R = json.load(open(RES / "anatomy.json"))
    g = lambda kind: "start" if kind in ("identity", "random") else "traj"
    grp = defaultdict(list)
    for r in R: grp[(r["n"], r["m"], g(r["kind"]))].append(r)
    print("class  scope  states |C|med shared cross | inv crossings med/mean/max, share(maxcr_arc>1) | "
          "pred gain c0/c1/pa1 rel. to inv | real/pred c0 c1 pa1 inv | t_sel ms c0 c1 pa1 inv")
    for key in sorted(grp):
        L = grp[key]
        inv_cr = [r["sel"]["inv"]["cr"] for r in L]
        def predrel(nm):
            v = [r["sel"][nm]["pred"] / r["sel"]["inv"]["pred"] for r in L if r["sel"]["inv"]["pred"] < 0]
            return st.mean(v) if v else float("nan")
        def rr(nm):
            den = sum(-r["sel"][nm]["pred"] for r in L)
            return sum(-r["sel"][nm]["real"] for r in L) / den if den else float("nan")
        t = {nm: st.median(r["sel"][nm]["t_ms"] for r in L) for nm in ("c0", "c1", "pa1", "inv")}
        print(f"{key[0]}x{key[1]:<3} {key[2]:5} {len(L):4d} {st.median(r['n_improving'] for r in L):6.0f} "
              f"{st.mean(r['conf_shared'] for r in L):7.0f} {st.mean(r['conf_cross'] for r in L):7.0f} | "
              f"{st.median(inv_cr):4.0f}/{st.mean(inv_cr):5.1f}/{max(inv_cr):3d} {sum(1 for r in L if r['sel']['inv']['maxcr_arc'] > 1)/len(L):.2f} | "
              f"{predrel('c0'):.3f}/{predrel('c1'):.3f}/{predrel('pa1'):.3f} | {rr('c0'):.2f} {rr('c1'):.2f} {rr('pa1'):.2f} {rr('inv'):.2f} | "
              f"{t['c0']:.1f} {t['c1']:.1f} {t['pa1']:.1f} {t['inv']:.1f}")
    print("\nc2 (MILP), where |C| <= 120:")
    for key in sorted(grp):
        L = [r for r in grp[key] if "c2" in r["sel"] and r["sel"]["inv"]["pred"] < 0]
        if L:
            pr = st.mean(r["sel"]["c2"]["pred"] / r["sel"]["inv"]["pred"] for r in L)
            rr2 = sum(-r["sel"]["c2"]["real"] for r in L) / sum(-r["sel"]["c2"]["pred"] for r in L)
            print(f"  {key}: states={len(L)} pred c2/inv={pr:.3f} real/pred={rr2:.2f} t_med={st.median(r['sel']['c2']['t_ms'] for r in L):.0f} ms")
    print("\npair additivity by geometry (pooled): real/pred, P(pair not improving), P(exactly additive), mean |err|")
    for scope in ("start", "traj"):
        for n in (20, 50, 100):
            acc = defaultdict(lambda: defaultdict(float))
            for r in R:
                if r["n"] == n and g(r["kind"]) == scope and "pairs" in r:
                    for geo, s in r["pairs"].items():
                        for k, v in s.items(): acc[geo][k] += v
            for geo in ("disjoint", "nested", "crossing"):
                s = acc[geo]
                if s["N"]:
                    print(f"  {scope:5} n={n:3d} {geo:8s} N={int(s['N']):5d} real/pred={s['sum_real']/s['sum_pred']:5.2f} "
                          f"P(not impr.)={s['nonimproving']/s['N']:.2f} P(exact)={s['exact']/s['N']:.2f} mean|err|={s['sum_abs_err']/s['N']:5.1f}")

def descent():
    R = json.load(open(RES / "descent.json"))
    V = ["c0", "c1", "pa1", "c2", "inv"]
    grp = defaultdict(list)
    for r in R: grp[(r["n"], r["m"])].append(r)
    print("class runs | final Cmax vs c0 [%] | steps | share of steps with fallback to the best single swap")
    for key in sorted(grp):
        L = grp[key]; vs = [v for v in V if all(v in r for r in L)]
        rel = {v: st.mean(100 * (r[v]["final"] - r["c0"]["final"]) / r["c0"]["final"] for r in L) for v in vs}
        steps = {v: st.mean(r[v]["steps"] for r in L) for v in vs}
        fb = {v: sum(r[v]["fallbacks"] for r in L) / max(1, sum(r[v]["steps"] for r in L)) for v in vs}
        f = lambda d, fmt: " ".join(f"{v}:{format(d[v], fmt)}" for v in vs)
        print(f"{key[0]}x{key[1]:<3} {len(L):3d} | {f(rel, '+.2f')} | {f(steps, '.0f')} | {f(fb, '.2f')}")

def pairq():
    R = json.load(open(RES / "pairq.json"))
    g = defaultdict(list)
    for r in R: g[r["n"]].append(r)
    for n, L in sorted(g.items()):
        print(f"n={n}: states={len(L)} |C| median={st.median(r['nC'] for r in L)}; best single swap mean real={st.mean(r['best_single'] for r in L):.1f}")
        for nm in ("c0", "c1", "inv", "pq_c0", "pq_inv"):
            print(f"   {nm:7s} mean real={st.mean(r[nm]['real'] for r in L):7.1f} pred={st.mean(r[nm]['pred'] for r in L):7.1f} "
                  f"P(not improving)={sum(1 for r in L if r[nm]['real'] >= 0)/len(L):.2f} "
                  f"P(better than best single)={sum(1 for r in L if r[nm]['real'] < r['best_single'])/len(L):.2f} "
                  f"swaps={st.mean(r[nm]['swaps'] for r in L):.1f} cr={st.mean(r[nm]['cr'] for r in L):.1f}")
        print(f"   pair-corrected MILP time median: {st.median(r['pq_inv']['t_ms'] for r in L):.0f} ms")

def ils():
    from scipy.stats import wilcoxon
    for tl in sorted({int(Path(p).name.split('_')[1][:-2]) for p in glob.glob(str(RES / "ils_*ms_*.json"))}):
        # several files per budget may repeat a mode (e.g. cr_c0_safe); keep the first file in name order
        res, seen = [], set()
        for p in sorted(glob.glob(str(RES / f"ils_{tl}ms_*.json"))):
            for r in json.load(open(p)):
                key = (r["file"], r["mode"], r["inst"], r["seed"])
                if key not in seen:
                    seen.add(key); res.append(r)
        g = defaultdict(list)
        for r in res: g[(r["file"], r["mode"])].append(r)
        files = sorted({r["file"] for r in res}, key=["tai20_5", "tai20_10", "lcg50_10"].index)
        order = ["motzkin", "dynasearch", "cr_swap", "cr_c0", "cr_c1", "cr_pa1", "cr_inv", "cr_c0_safe", "cr_c1_safe",
                 "cr_pa1_safe", "cr_inv_safe", "cr_pq12_c0", "cr_pq12_inv", "cr_pq_c0", "cr_pq_inv"]
        modes = sorted({r["mode"] for r in res}, key=order.index)
        print(f"\nILS {tl} ms: mean RPD to LB [%] +- std (mean iterations); 4 instances x 3 seeds")
        for mo in modes:
            print(f"  {mo:12s} " + "".join(
                f"{st.mean(r['rpd'] for r in g[(f, mo)]):7.2f} ±{st.stdev(r['rpd'] for r in g[(f, mo)]):4.2f} ({st.mean(r['iters'] for r in g[(f, mo)]):5.0f} it)  "
                if g[(f, mo)] else " " * 30 for f in files))
        def paired(a, b, f):
            A = {(r["inst"], r["seed"]): r["rpd"] for r in g[(f, a)]}; B = {(r["inst"], r["seed"]): r["rpd"] for r in g[(f, b)]}
            d = [A[k] - B[k] for k in sorted(set(A) & set(B))]
            if not d or all(x == 0 for x in d): return None
            return f"diff {st.mean(d):+.2f} pp, wins {sum(1 for x in d if x < 0)}/{sum(1 for x in d if x > 0)}, p={wilcoxon(d).pvalue:.3f}"
        for f in files:
            for a, b in [("cr_inv", "cr_c0"), ("cr_c0_safe", "cr_c0"), ("cr_inv_safe", "cr_inv"), ("cr_inv_safe", "cr_c0_safe"),
                         ("cr_c1_safe", "cr_c0_safe"), ("cr_c0_safe", "cr_swap"), ("cr_inv_safe", "cr_swap"), ("cr_c0_safe", "dynasearch")]:
                s = paired(a, b, f)
                if s: print(f"  {f}: {a} vs {b}: {s}")

def qubo():
    for r in json.load(open(RES / "qubo_structure.json")):
        print(r)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("anatomy", anatomy), ("descent", descent), ("pairq", pairq), ("ils", ils), ("qubo", qubo)):
        if what in (name, "all"):
            print(f"===== {name} ====="); fn(); print()
