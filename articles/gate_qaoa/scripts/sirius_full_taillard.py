"""Pelny Taillard na siriusie: 4 sasiedztwa bramkowe w ILS, budzet 10 s.

Protokol (za windowed_qubo, zeby liczby byly wprost porownywalne):
  12 plikow Taillarda x 10 instancji, ILS, budzet 10 s, p=1, 4096 shotow,
  PRD od dolnego ograniczenia Taillarda. Tyle ruchow, ile zmiesci sie
  w budzecie -- bez sztucznego limitu.

Okna sa rozne per klasa sasiedztw, bo w pozycji daje w-1 zmiennych dla QUBO
trojdiagonalnego i w(w-1)/2 dla parowego:
  liniowe (adjacent, fibonacci)  okno W_LIN  -> W_LIN-1 zmiennych
  geste   (dynasearch, motzkin)  okno W_GEST -> W_GEST*(W_GEST-1)/2 zmiennych
Wszystkie okna jednego ruchu ida w JEDNYM zadaniu, a rozliczenie jest za
zadanie, nie za obwod -- wiec n=500 kosztuje tyle co n=20.

Powtorzenia per sasiedztwo z pomiarow rozrzutu na symulatorze: adjacent i
fibonacci sa deterministyczne (odch. 0.00 i 0.03 pkt PRD), geste maja 1.5-2.5
pkt, czyli wiecej niz szukany efekt, wiec dostaja 5 powtorzen.

Kolejnosc WSZERZ po instancjach: przerwanie w dowolnym momencie zostawia
zbalansowana tabele na mniejszej liczbie instancji, a nie polowe plikow.
Zapis po KAZDYM przebiegu, wznowienie pomija to, co juz policzone.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Konto personalne jest wyczerpane -- liczymy z uczelnianego, jesli jest w .env.
if os.environ.get("IQM_PWR_TOKEN"):
    os.environ["IQM_TOKEN"] = os.environ["IQM_PWR_TOKEN"]
os.environ["IQM_COMPUTER"] = os.environ.get("CAMPAIGN_COMPUTER", "sirius")

from src.parser import parser
from src.permutation_procesing import c_max
from src.algorithms.ils import iterated_local_search
from src.neighborhoods.gate_qaoa import solve as S

BUDZET_MS = 10_000
BUDZETY = [100, 500, 1000, 2000, 5000, 10000]
W_LIN, W_GEST, OVERLAP = 17, 6, 0.5
SHOTS, P = 4096, 1
# Jedno powtorzenie wszedzie: powtorzenia kosztuja 3x wiecej niz instancje i
# probkuja tylko szum shotow (zmierzone odchylenie 0,00 pkt dla adjacenta,
# 0,03 dla fibonacciego). Piec powtorzen dla gestych na instancji 0 pieciu
# plikow juz mamy -- to wystarczy na oszacowanie szumu sprzetowego.
POWT = {"gate_adjacent": 1, "gate_fibonacci": 1, "gate_dynasearch": 1, "gate_motzkin": 1}
SASIEDZTWA = ["gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"]
PLIKI = ["tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20",
         "tai100_5", "tai100_10", "tai100_20", "tai200_10", "tai200_20", "tai500_20"]
# Partie po poziomach instancji; budzet 2000 kredytow na caly artykul miesci
# cztery poziomy (~1706 kr) plus 150 rezerwy. Podnosic po kazdej partii.
INSTANCJI = int(os.environ.get("CAMPAIGN_INSTANCES", "1"))
# Pliki pominiete w tej partii. tai500_20 kosztuje 190 kredytow za instancje,
# tyle co 27 instancji tai20_5, wiec przy koncowce budzetu zostaje w tyle.
POMIN = {x for x in os.environ.get("CAMPAIGN_SKIP", "").split(",") if x}
# Sasiedztwa w tej partii. Przy koncowce budzetu liniowe (1 obwod na okno) sa
# rzedy wielkosci tansze od gestych, wiec czasem miesci sie tylko ich para.
TYLKO = {x for x in os.environ.get("CAMPAIGN_NEIGH", "").split(",") if x}

OUT = Path(__file__).resolve().parents[1] / "measurements" / "sirius_full_taillard_ils.json"
QC = {"qaoa_backend": "iqm", "qaoa_p": P, "qaoa_shots": SHOTS,
      "qaoa_window_size_linear": W_LIN, "qaoa_window_size": W_GEST,
      "qaoa_overlap_ratio": OVERLAP}


def kolumny(move_log):
    """Wynik dla kazdego budzetu protokolu, replikujac warunek stopu petli ILS:
    ruch k wchodzi, gdy jego czas STARTU (koniec ruchu k-1) byl < budzet."""
    out = {}
    for t in BUDZETY:
        k = 0
        while k < len(move_log) and (0 if k == 0 else move_log[k - 1][0]) < t:
            k += 1
        out[str(t)] = move_log[k - 1][1] if k else None
    return out


S._IQM_CACHE = None
hw = S._get_iqm()
dane = json.loads(OUT.read_text()) if OUT.exists() else {}
dane.setdefault("_meta", {
    "backend": hw.name, "qubits": hw.num_qubits, "algo": "ils",
    "budzet_ms": BUDZET_MS, "shots": SHOTS, "p": P,
    "okno_liniowe": W_LIN, "okno_geste": W_GEST, "overlap": OVERLAP,
    "powtorzenia": POWT, "instancji_na_plik": INSTANCJI,
    "protokol": "windowed_qubo: 12 plikow x 10 instancji, PRD od dolnego ograniczenia",
})

zadania = [(i, f, ne, r) for i in range(INSTANCJI) for f in PLIKI if f not in POMIN
           for ne in SASIEDZTWA if not TYLKO or ne in TYLKO
           for r in range(POWT[ne])]
zrobione = sum(1 for k in dane if not k.startswith("_"))
print(f"{hw.name}, {hw.num_qubits} kubitow | budzet {BUDZET_MS} ms | "
      f"okna {W_LIN} lin / {W_GEST} gest")
print(f"{len(zadania)} przebiegow w planie, {zrobione} juz policzonych"
      + (f"  (pominiete: {', '.join(sorted(POMIN))})" if POMIN else "")
      + (f"  (tylko: {', '.join(sorted(TYLKO))})" if TYLKO else "") + "\n")
print(f"{'inst':>4} {'plik':>10} {'sasiedztwo':>16} {'powt':>4} {'ruchow':>6} "
      f"{'cmax':>7} {'PRD':>7} {'czas':>7} {'zadan':>6}")

cache = {}
zadan_oplaconych = 0
for i, f, ne, r in zadania:
    key = f"{f}|{i}|{ne}|{r}"
    if key in dane:
        continue
    if f not in cache:
        inst = parser(f"data/{f}.txt", i)
        cache.clear()
        cache[f] = True
    inst = parser(f"data/{f}.txt", i)
    pt = inst["processing_times"]
    LB = inst["info"]["lower_bound"]
    c0 = c_max(list(range(len(pt[0]))), pt)
    t0 = time.time()
    try:
        _, cmax, _, _, st = iterated_local_search(
            pt, max_time_ms=BUDZET_MS, tabu_tenure=10, neigh_mode=ne, quantum_config=QC)
    except Exception as e:
        print(f"\nPRZERWANE na {key}: {type(e).__name__}: {e}")
        print("(zapisane do tej pory zostaje; wznow ten sam skrypt po doladowaniu)")
        break
    dt = time.time() - t0
    ml = [list(x) for x in st["move_log"]]
    dane[key] = {"plik": f, "inst": i, "neigh": ne, "powt": r, "lb": LB,
                 "cmax_start": c0, "cmax": cmax, "prd": 100 * (cmax - LB) / LB,
                 "ruchow": st["iterations"], "wall_s": dt, "move_log": ml,
                 "kolumny_budzetow": kolumny(ml),
                 "kiedy": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    OUT.write_text(json.dumps(dane, indent=1))      # zapis po KAZDYM oplaconym przebiegu
    zadan_oplaconych += st["iterations"]
    print(f"{i:>4} {f:>10} {ne:>16} {r:>4} {st['iterations']:>6} {cmax:>7} "
          f"{dane[key]['prd']:>6.2f}% {dt:>6.1f}s {zadan_oplaconych:>6}")

print(f"\noplaconych zadan w tej sesji: {zadan_oplaconych} "
      f"(~{zadan_oplaconych * 0.633:.0f} kredytow przy stawce siriusa)")
print(f"policzonych przebiegow: {sum(1 for k in dane if not k.startswith('_'))}/{len(zadania)}")
print(f"zapisane -> {OUT}")
