"""PRD wobec rozmiaru instancji, osobna linia na kazda liczbe maszyn.

Pierwsza wersja byla rozrzutem wobec m/n i okazala sie nieczytelna: os
logarytmiczna miala dwa podpisy, a dwanascie punktow bylo nie do rozroznienia.
Ten wykres pokazuje strukture, ktora te zaleznosc wytwarza. Przy stalym m
odchylenie spada z n, a linie ida w kolejnosci m -- i stad bierze sie
korelacja z m/n, ktora podajemy w tekscie.

Druk LNAI bywa czarno-bialy, wiec seria ma wlasny styl linii, wlasny ksztalt
znacznika i podpis przy linii zamiast legendy w ramce. Zadnego koloru.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# TrueType zamiast Type 3: kontrola PDF Springera odrzuca fonty Type 3.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

KORZEN = Path(__file__).resolve().parents[3]
DANE = KORZEN / "articles/gate_qaoa/measurements/sirius_full_taillard_ils.json"
NEIGH = ["gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"]
META = {"tai20_5": (20, 5), "tai20_10": (20, 10), "tai20_20": (20, 20),
        "tai50_5": (50, 5), "tai50_10": (50, 10), "tai50_20": (50, 20),
        "tai100_5": (100, 5), "tai100_10": (100, 10), "tai100_20": (100, 20),
        "tai200_10": (200, 10), "tai200_20": (200, 20), "tai500_20": (500, 20)}
SERIE = {20: ("-", "^", 0), 10: ("--", "s", -11), 5: (":", "o", -11)}
TUSZ = "#111111"

rek = [v for k, v in json.loads(DANE.read_text()).items() if not k.startswith("_")]
per = defaultdict(lambda: defaultdict(list))
for v in rek:
    per[(v["plik"], v["neigh"])][v["inst"]].append(v["prd"])

dane = defaultdict(list)
for plik, (n, m) in META.items():
    best = min(statistics.mean([statistics.mean(x) for x in per[(plik, ne)].values()])
               for ne in NEIGH)
    dane[m].append((n, best))

fig, ax = plt.subplots(figsize=(4.7, 3.0))
for m in (20, 10, 5):
    ls, mk, dy = SERIE[m]
    pk = sorted(dane[m])
    xs = [p[0] for p in pk]
    ys = [p[1] for p in pk]
    ax.plot(xs, ys, ls=ls, color=TUSZ, lw=1.2, zorder=2)
    ax.plot(xs, ys, ls="none", marker=mk, ms=6, markerfacecolor="white",
            markeredgecolor=TUSZ, markeredgewidth=1.2, zorder=3)
    # podpis przy linii zamiast legendy w ramce
    ax.annotate(f"$m = {m}$", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(9, dy + 4), fontsize=8.5, color=TUSZ, va="center")

ax.set_xscale("log")
ax.set_xticks([20, 50, 100, 200, 500])
ax.set_xticklabels(["20", "50", "100", "200", "500"])
ax.minorticks_off()
ax.set_xlabel("jobs $n$", fontsize=9)
ax.set_ylabel("PRD of the best neighborhood [%]", fontsize=9)
ax.set_xlim(17, 760)
ax.set_ylim(8, 47)
ax.tick_params(labelsize=8.5)
ax.grid(True, axis="y", lw=0.4, alpha=0.25)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_suffix("." + ext), dpi=300)

x = [m / n for n, m in ((v[0], v[1]) for v in META.values())]
y = [min(statistics.mean([statistics.mean(z) for z in per[(p, ne)].values()])
         for ne in NEIGH) for p in META]
mx, my = statistics.mean(x), statistics.mean(y)
r = (sum((a - mx) * (b - my) for a, b in zip(x, y))
     / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5)
print(f"zapisane. korelacja PRD z m/n = {r:.3f}")
for m in (5, 10, 20):
    pk = sorted(dane[m])
    print(f"  m={m:>2}: " + "  ".join(f"n={n} {v:.1f}%" for n, v in pk))
