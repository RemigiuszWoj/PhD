from itertools import combinations

import pandas as pd
from scipy import stats

df = pd.read_csv("summary.csv")

neighborhoods = [
    "adjacent",
    "motzkin_neighborhood",
    "fibonahi_neighborhood",
    "dynasearch_neighborhood",
]
neigh_labels = {
    "adjacent": "Adj",
    "motzkin_neighborhood": "Mot",
    "fibonahi_neighborhood": "Fib",
    "dynasearch_neighborhood": "Dyn",
}
algorithms = ["tabu", "sa"]
time_limits = [100, 500, 1000, 2000, 5000, 10000]

all_pairs = list(combinations(neighborhoods, 2))

# ============================================================
# 1. Srednie globalne i odchylenia standardowe
# ============================================================
print("=== Srednie i odchylenia standardowe ===\n")
result = (
    df.groupby(["algorithm", "neighborhood", "instance_file", "time_limit_ms"])["gap_percent"]
    .mean()
    .reset_index()
)

summary = (
    result.groupby(["algorithm", "neighborhood", "time_limit_ms"])["gap_percent"]
    .agg(["mean", "std"])
    .round(3)
)
print(summary.to_string())

# ============================================================
# 2. Test Wilcoxona dla wszystkich par
# ============================================================
print("\n=== Wilcoxon signed-rank: wszystkie pary ===")
print("Legenda: ** p<0.01  * p<0.05  ~ p<0.10  ns p>=0.10")
print("         < pierwsza sasiedztwo lepsze (nizszy gap)")
print("         > drugie sasiedztwo lepsze (nizszy gap)")

for algo in algorithms:
    print(f"\nAlgorytm: {algo.upper()}")
    header = f"{'Para':<14}"
    for tl in time_limits:
        header += f" {'TL='+str(tl):>11}"
    print(header)
    print("-" * 82)

    for n1, n2 in all_pairs:
        label = f"{neigh_labels[n1]} vs {neigh_labels[n2]}"
        row = f"{label:<14}"
        for tl in time_limits:
            sub = df[(df["algorithm"] == algo) & (df["time_limit_ms"] == tl)]
            g1 = sub[sub["neighborhood"] == n1].groupby("instance_file")["gap_percent"].mean()
            g2 = sub[sub["neighborhood"] == n2].groupby("instance_file")["gap_percent"].mean()
            common = g1.index.intersection(g2.index)

            try:
                w, p = stats.wilcoxon(g1[common], g2[common])
                diff = (g1[common] - g2[common]).median()
                direction = "<" if diff < 0 else ">"
                if p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                elif p < 0.10:
                    sig = "~"
                else:
                    sig = "ns"
                row += f"  {p:>6.3f}{sig:<2}{direction}"
            except Exception as e:
                row += f'  {"err":>9}'
        print(row)

# ============================================================
# 3. Szczegolowe wyniki dla wszystkich par
# ============================================================
print("\n=== Szczegoly: wszystkie pary (W i p-value) ===")
print(f"{'algo':<6} {'para':<20} {'TL':>6} {'W':>8} {'p-value':>10} {'istotne?':>10}")
print("-" * 65)

for algo in algorithms:
    for n1, n2 in all_pairs:
        label = f"{neigh_labels[n1]} vs {neigh_labels[n2]}"
        for tl in time_limits:
            sub = df[(df["algorithm"] == algo) & (df["time_limit_ms"] == tl)]
            g1 = sub[sub["neighborhood"] == n1].groupby("instance_file")["gap_percent"].mean()
            g2 = sub[sub["neighborhood"] == n2].groupby("instance_file")["gap_percent"].mean()
            common = g1.index.intersection(g2.index)

            try:
                w, p = stats.wilcoxon(g1[common], g2[common])
                if p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                elif p < 0.10:
                    sig = "~"
                else:
                    sig = "ns"
                print(f"{algo:<6} {label:<20} {tl:>6} {w:>8.1f} {p:>10.4f} {sig:>10}")
            except Exception as e:
                print(f'{algo:<6} {label:<20} {tl:>6} {"err":>8}')
