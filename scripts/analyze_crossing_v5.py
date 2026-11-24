"""Analysis of the crossing scale v5 (articles/crossings/PLAN.md, v5): ILS with cross_inv_cf<F>, F = 10..90 %
of the involution move's crossings, against c0 (noncrossing, exact) and inv (F = 100). Same endpoint as
analyze_crossing.py: RPD to the Taillard lower bound at 10 s, instance = block (mean over 5 seeds), Wilcoxon
with Holm inside each comparison family. Writes tables/v5_*.csv and tables/summary_v5.txt.
"""

import statistics

from analyze_crossing import ROOT, compare, family, inst_means, table, write_csv
from analyze_crossing_v4 import load_any

OUT = ROOT / "articles" / "crossings" / "tables"
FILES = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20", "tai100_5", "tai100_10",
         "tai100_20", "tai200_10", "tai200_20"]
CF = [f"inv_cf{f}" for f in range(10, 100, 10)]


def main():
    runs = load_any(["main", "safe", "cf", "cf0"])
    lines = ["Stage v5, crossing scale. RPD to LB [%] at 10 s, blocks = instances (5 seeds), Holm per family."]
    v = ["c0"] + CF + ["inv"]
    files = [f for f in FILES if None not in inst_means(runs, "c0", f, 10000)]
    table(lines, "mean RPD (c0 = F 0 exact, inv = F 100)", [x.removeprefix("inv_") for x in v],
          [(f, [statistics.mean(m) if None not in (m := inst_means(runs, x, f, 10000)) else None for x in v])
           for f in files])
    rows = []
    for x in CF:
        rows += family(lines, f"{x} vs c0", compare(runs, x, "c0", files), decisive_rule=False)
    for x in CF:
        rows += family(lines, f"{x} vs inv", compare(runs, x, "inv", files), decisive_rule=False)
    # control: the same greedy repair down to zero crossings (n >= 100 only)
    f0 = [f for f in files if None not in inst_means(runs, "inv_cf0", f, 10000)]
    if f0:
        for a, b in [("inv_cf0", "c0"), ("inv_cf10", "inv_cf0"), ("inv_cf20", "inv_cf0")]:
            rows += family(lines, f"control: {a} vs {b}", compare(runs, a, b, f0), decisive_rule=False)
    write_csv(OUT / "v5_tests.csv", rows)
    text = "\n".join(lines) + "\n"
    (OUT / "summary_v5.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
