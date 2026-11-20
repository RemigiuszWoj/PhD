"""D-Wave ILS pilot vs classical ILS at the same 5 s budget (19.11.2026).

Pairs each D-Wave run (results/quantum_tiers/dwave) with the classical run of the same
k-bonacci neighborhood, instance and seed, cut at 5 s (results/kbonacci/main_budgets).
RPD is measured to the instance lower bound stored in result.json.
"""
from __future__ import annotations

import collections
import glob
import json
import re
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DW = ROOT / "results" / "quantum_tiers" / "dwave"
CL = ROOT / "results" / "kbonacci" / "main_budgets"
PAT = re.compile(r"neigh=quantum_kbonacci_k(\d+)__file=(tai\d+_\d+)__inst=(\d+)__n(\d+)__m(\d+)__tl=5000ms__seed=(\d+)")


def rpd(d: dict) -> float:
    lb = d.get("lower_bound") or d.get("upper_bound")
    return 100.0 * (d["cmax_best"] - lb) / lb


def main() -> None:
    rows = collections.defaultdict(list)
    failed = collections.Counter()
    for p in glob.glob(str(DW / "*")):
        m = PAT.search(p)
        if not m:
            continue
        k, file, inst, n, mm, seed = m.groups()
        key = (int(n), int(mm), int(k))
        res = Path(p) / "result.json"
        if not res.exists():
            failed[key] += 1
            continue
        dq = json.loads(res.read_text())
        cl = CL / f"algo=ils__neigh=kbonacci_k{k}__file={file}__inst={inst}__n{n}__m{mm}__tl=5000ms__seed={seed}" / "result.json"
        if not cl.exists():
            continue
        dc = json.loads(cl.read_text())
        rows[key].append((rpd(dq), rpd(dc), dq.get("iterations", 0), dc.get("iterations", 0),
                          dq["qpu_stats"]["qpu_success"]))
    print(f"{'n':>4} {'m':>3} {'k':>2} {'pairs':>5} {'fail':>4} {'RPD DW':>7} {'RPD CL':>7} "
          f"{'it DW':>6} {'it CL':>7} {'QPU':>4} {'W/T/L':>8} {'p':>7}")
    for key in sorted(set(rows) | set(failed)):
        v = rows.get(key, [])
        if not v:
            print(f"{key[0]:>4} {key[1]:>3} {key[2]:>2} {0:>5} {failed[key]:>4}   (no successful run)")
            continue
        a = [x[0] for x in v]
        b = [x[1] for x in v]
        w = sum(x < y for x, y in zip(a, b))
        t = sum(x == y for x, y in zip(a, b))
        l_ = len(v) - w - t
        try:
            pv = wilcoxon(a, b).pvalue if w + l_ > 0 else 1.0
        except ValueError:
            pv = float("nan")
        mean = lambda xs: sum(xs) / len(xs)
        print(f"{key[0]:>4} {key[1]:>3} {key[2]:>2} {len(v):>5} {failed[key]:>4} {mean(a):7.2f} {mean(b):7.2f} "
              f"{mean([x[2] for x in v]):6.1f} {mean([x[3] for x in v]):7.0f} {mean([x[4] for x in v]):4.1f} "
              f"{w:>2}/{t}/{l_:<2} {pv:7.1e}")


if __name__ == "__main__":
    main()
