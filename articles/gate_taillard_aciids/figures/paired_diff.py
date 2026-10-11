"""Roznice parowane wobec Adjacent, po jednej na kazda z 47 par (plik, instancja).

Czterech sasiedztw nie da sie pokazac jako czterech krzywych bezwzglednego PRD:
dzielilyby je srednio 2,0 punktu na osi o zakresie 27, wiec by sie nalozyly.
Roznica liczona w obrebie pary usuwa trudnosc instancji, ktora jest wspolna dla
wszystkich czterech, i dopiero wtedy uklad staje sie widoczny.

Bez koloru: ksztalt znacznika niesie sasiedztwo, pozioma kreska to mediana.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KORZEN = Path(__file__).resolve().parents[3]
DANE = KORZEN / "articles/gate_qaoa/measurements/sirius_full_taillard_ils.json"
NEIGH = ["gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"]
ETYK = {"gate_fibonacci": ("Fibonacci", "s"), "gate_dynasearch": ("Dynasearch", "^"),
        "gate_motzkin": ("Motzkin", "D")}
TUSZ = "#111111"

rek = [v for k, v in json.loads(DANE.read_text()).items() if not k.startswith("_")]
per = defaultdict(dict)
for v in rek:
    per[(v["plik"], v["inst"])].setdefault(v["neigh"], []).append(v["prd"])
pary = [k for k, v in per.items() if all(ne in v for ne in NEIGH)]

fig, ax = plt.subplots(figsize=(4.7, 2.9))
ax.axhline(0, color=TUSZ, lw=1.0, zorder=1)
ax.set_ylim(-11.5, 7.2)          # ustalone z gory, zeby podpisy staly rowno
for i, ne in enumerate(["gate_fibonacci", "gate_dynasearch", "gate_motzkin"]):
    nazwa, mk = ETYK[ne]
    r = [statistics.mean(per[k][ne]) - statistics.mean(per[k]["gate_adjacent"])
         for k in pary]
    # lekki rozrzut poziomy, zeby punkty o tej samej wartosci sie nie zlewaly
    xs = [i + (j % 7 - 3) * 0.028 for j in range(len(r))]
    ax.scatter(xs, r, marker=mk, s=22, facecolors="none", edgecolors=TUSZ,
               linewidths=0.9, alpha=0.75, zorder=2)
    med = statistics.median(r)
    ax.plot([i - 0.26, i + 0.26], [med, med], color=TUSZ, lw=2.0, zorder=3)
    lep = sum(1 for x in r if x < 0)
    ax.annotate(f"{lep}/{len(r)} below zero", (i, 6.0), ha="center",
                fontsize=8.5, color=TUSZ)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([ETYK[n][0] for n in
                    ["gate_fibonacci", "gate_dynasearch", "gate_motzkin"]], fontsize=9)
ax.set_ylabel("PRD difference vs Adjacent\n[points, negative is better]", fontsize=8.5)
ax.set_xlim(-0.5, 2.5)
ax.tick_params(axis="y", labelsize=8.5)
ax.tick_params(axis="x", length=0)
ax.grid(True, axis="y", lw=0.4, alpha=0.25)
ax.set_axisbelow(True)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_suffix("." + ext), dpi=300)
print(f"zapisane, par: {len(pary)}")
for ne in ["gate_fibonacci", "gate_dynasearch", "gate_motzkin"]:
    r = [statistics.mean(per[k][ne]) - statistics.mean(per[k]["gate_adjacent"]) for k in pary]
    print(f"  {ETYK[ne][0]:>11}: mediana {statistics.median(r):+.2f}, "
          f"srednia {statistics.mean(r):+.2f}, ponizej zera {sum(1 for x in r if x<0)}/{len(r)}")
