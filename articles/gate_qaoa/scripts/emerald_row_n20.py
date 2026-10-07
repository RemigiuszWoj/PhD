"""Jeden pelny przebieg protokolu na emeraldzie, dla kazdego z 4 sasiedztw.

Pelny protokol, nie pilot: ILS, budzet 10 s, tai20_5 instancja 0, seed 0
(= start identycznosciowy), p=1, 4096 shotow, okno 10 dla sasiedztw gestych
(45 zmiennych, maksimum na 54 kubitach emeralda). Wynik jest wiec gotowa
komorka tabeli, a nie pomiarem kalibracyjnym do wyrzucenia.

move_log z ils.py daje czas konca i best_cmax po KAZDYM ruchu, wiec z tego
jednego przebiegu odtwarzamy kolumny 100 ms - 10 s, replikujac warunek stopu
petli: ruch k wchodzi, gdy jego czas startu < budzet.

Zapis po kazdym przebiegu, bo kazdy jest oplacony.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# skrypt lezy w articles/gate_qaoa/scripts/, a importuje src/ z korzenia repo
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

load_dotenv(Path("/Users/remigiuszwojewodzki/Desktop/PhD/.env"))
os.environ["IQM_COMPUTER"] = "emerald"

from src.parser import parser
from src.permutation_procesing import c_max
from src.algorithms.ils import iterated_local_search
from src.neighborhoods.gate_qaoa import solve as S

OUT = Path(__file__).resolve().parents[1] / "measurements" / "emerald_row_n20_ils.json"
BUDZET_MS = 10_000
BUDZETY = [100, 500, 1000, 2000, 5000, 10000]
OBWODOW = {"gate_adjacent": 1, "gate_fibonacci": 1, "gate_dynasearch": 4, "gate_motzkin": 4}
QC = {"qaoa_backend": "iqm", "qaoa_p": 1, "qaoa_shots": 4096,
      "qaoa_window_size": 10, "qaoa_overlap_ratio": 0.5}

S._IQM_CACHE = None
hw = S._get_iqm()
inst = parser("data/tai20_5.txt", 0)
pt = inst["processing_times"]
LB = inst["info"]["lower_bound"]
c0 = c_max(list(range(len(pt[0]))), pt)
print(f"{hw.name}, {hw.num_qubits} kubitow | tai20_5 inst 0 | LB={LB} start={c0} "
      f"| budzet {BUDZET_MS} ms\n")

zapis = {"_meta": {"backend": hw.name, "qubits": hw.num_qubits, "algo": "ils",
                   "instancja": "tai20_5#0", "lb": LB, "cmax_start": c0,
                   "budzet_ms": BUDZET_MS, "shots": 4096, "p": 1,
                   "okno_geste": 10, "overlap": 0.5, "seed": 0}}

def kolumny(move_log):
    """Odtworz wynik dla kazdego budzetu protokolu, replikujac warunek stopu petli."""
    out = {}
    for t in BUDZETY:
        k = 0
        while k < len(move_log) and (0 if k == 0 else move_log[k - 1][0]) < t:
            k += 1
        out[str(t)] = move_log[k - 1][1] if k else None
    return out

print(f"{'sasiedztwo':>16} {'obw':>4} {'ruchow':>7} {'cmax':>6} {'PRD':>7} {'czas':>8}")
for ne in ("gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"):
    t0 = time.time()
    _, cmax, _, _, st = iterated_local_search(
        pt, max_time_ms=BUDZET_MS, tabu_tenure=10, neigh_mode=ne, quantum_config=QC)
    dt = time.time() - t0
    ml = [list(x) for x in st["move_log"]]
    prd = 100 * (cmax - LB) / LB
    zapis[ne] = {"neigh": ne, "obwodow_na_ruch": OBWODOW[ne], "cmax": cmax, "prd": prd,
                 "iteracji": st["iterations"], "wall_s": dt,
                 "neigh_time_ms": st["neigh_time_ms"], "move_log": ml,
                 "kolumny_budzetow": kolumny(ml),
                 "obwodow_oplaconych": OBWODOW[ne] * st["iterations"]}
    OUT.write_text(json.dumps(zapis, indent=1))          # zapis po kazdym oplaconym przebiegu
    print(f"{ne:>16} {OBWODOW[ne]:>4} {st['iterations']:>7} {cmax:>6} {prd:>6.2f}% {dt:>7.1f} s")

print(f"\n{'sasiedztwo':>16} | " + " | ".join(f"{str(t)+'ms':>8}" for t in BUDZETY))
for ne in OBWODOW:
    k = zapis[ne]["kolumny_budzetow"]
    print(f"{ne:>16} | " + " | ".join(
        f"{(100*(k[str(t)]-LB)/LB):>7.2f}%" if k[str(t)] else f"{'—':>8}" for t in BUDZETY))

obw = sum(zapis[ne]["obwodow_oplaconych"] for ne in OBWODOW)
ruch = sum(zapis[ne]["iteracji"] for ne in OBWODOW)
zapis["_koszt"] = {"ruchow": ruch, "obwodow": obw,
                   "uwaga": "stawke odczytac z panelu IQM; f*ruchow + p*obwodow = delta"}
OUT.write_text(json.dumps(zapis, indent=1))
print(f"\nzaplacono: {ruch} ruchow, {obw} obwodow")
print(f"stawka: delta_kredytow = f*{ruch} + p*{obw}  -> odczytaj delte z panelu")
print(f"zapisane -> {OUT}")
