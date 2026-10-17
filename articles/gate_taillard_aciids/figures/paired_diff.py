"""Odchylenie kazdego sasiedztwa od sredniej jego pary, 47 par.

Czterech sasiedztw nie da sie pokazac jako czterech krzywych bezwzglednego PRD:
dzielilyby je srednio 2,0 punktu na osi o zakresie 27, wiec by sie nalozyly.
Odniesienie do sredniej z pary usuwa trudnosc instancji, ktora jest wspolna dla
calej czworki, i dopiero wtedy uklad staje sie widoczny.

Punktem odniesienia jest srednia pary, nie Adjacent -- inaczej Adjacent bylby
niewidoczna linia zerowa, a to wlasnie on wypada najgorzej i trzeba to widziec.

Bez koloru: ksztalt znacznika niesie sasiedztwo, gruba kreska to mediana.
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
ETYK = {"gate_adjacent": ("Adjacent", "o"), "gate_fibonacci": ("Fibonacci", "s"),
        "gate_dynasearch": ("Dynasearch", "^"), "gate_motzkin": ("Motzkin", "D")}
TUSZ = "#111111"

rek = [v for k, v in json.loads(DANE.read_text()).items() if not k.startswith("_")]
per = defaultdict(dict)
for v in rek:
    per[(v["plik"], v["inst"])].setdefault(v["neigh"], []).append(v["prd"])
pary = [k for k, v in per.items() if all(ne in v for ne in NEIGH)]

fig, ax = plt.subplots(figsize=(4.9, 2.9))
ax.set_ylim(-5.2, 7.4)
ax.axhline(0, color=TUSZ, lw=1.0, zorder=1)
for i, ne in enumerate(NEIGH):
    nazwa, mk = ETYK[ne]
    r = []
    for k in pary:
        sr = statistics.mean(statistics.mean(per[k][x]) for x in NEIGH)
        r.append(statistics.mean(per[k][ne]) - sr)
    xs = [i + (j % 7 - 3) * 0.028 for j in range(len(r))]
    ax.scatter(xs, r, marker=mk, s=20, facecolors="none", edgecolors=TUSZ,
               linewidths=0.9, alpha=0.7, zorder=2)
    med = statistics.median(r)
    ax.plot([i - 0.27, i + 0.27], [med, med], color=TUSZ, lw=2.2, zorder=3)
    ax.annotate(f"{sum(1 for x in r if x < 0)}/{len(r)}", (i, 5.5), ha="center",
                fontsize=8.5, color=TUSZ)

ax.annotate("pairs below the mean of their own pair", (1.5, 6.8), ha="center",
            fontsize=7.5, color=TUSZ, style="italic")
ax.set_xticks(range(4))
ax.set_xticklabels([ETYK[n][0] for n in NEIGH], fontsize=9)
ax.set_ylabel("PRD minus the mean of its pair\n[points, negative is better]",
              fontsize=8.5)
ax.set_xlim(-0.5, 3.5)
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", length=0)
ax.grid(True, axis="y", lw=0.4, alpha=0.25)
ax.set_axisbelow(True)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_suffix("." + ext), dpi=300)
print(f"par: {len(pary)}")
for ne in NEIGH:
    r = [statistics.mean(per[k][ne]) - statistics.mean(statistics.mean(per[k][x])
         for x in NEIGH) for k in pary]
    print(f"  {ETYK[ne][0]:>11}: mediana {statistics.median(r):+.2f}, "
          f"ponizej zera {sum(1 for x in r if x < 0)}/{len(r)}")
