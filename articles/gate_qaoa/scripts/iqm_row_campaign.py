"""Ten-instance rows for the sparse neighborhoods, for the paper's gate column.

tai20_5, instances 0-9, seed 0 (= identity start, as the ibm_fez campaign used),
ILS, one move per run, p=1, 4096 shots, calibrated angles. Adjacent and
Fibonacci only: one circuit per move, so the whole sweep is 20 circuits.

Results are written after EVERY run, because each one costs credits and a crash
must not throw away what was already paid for. Re-running skips what is done.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path("/Users/remigiuszwojewodzki/Desktop/PhD/.env"))

MACHINE = sys.argv[1] if len(sys.argv) > 1 else "sirius"
os.environ["IQM_COMPUTER"] = MACHINE
SHOTS = 4096
FILE = "data/tai20_5.txt"

from src.parser import parser
from src.permutation_procesing import c_max
from src.neighborhoods.gate_qaoa import solve as S
from src.neighborhoods.gate_qaoa.adjacent import gate_adjacent_neighborhood
from src.neighborhoods.gate_qaoa.fibonacci import gate_fibonacci_neighborhood

S._IQM_CACHE = None
hw = S._get_iqm()
OUT = Path("/private/tmp/claude-501/-Users-remigiuszwojewodzki-Desktop-PhD/"
           "ad15d383-75e9-40c9-90fc-df0f2cea6500/scratchpad") / f"row_{MACHINE}.json"
done = json.loads(OUT.read_text()) if OUT.exists() else {}

print(f"{hw.name} ({hw.num_qubits} kubitow) | {FILE} inst 0-9 | seed 0 = identycznosc")
print(f"ILS, 1 ruch/bieg, p=1, {SHOTS} shotow\n")
print(f"{'inst':>4} {'LB':>5} {'start':>6} | {'adjacent':>18} | {'fibonacci':>18}")

FN = {"adjacent": gate_adjacent_neighborhood, "fibonacci": gate_fibonacci_neighborhood}
spent = 0.0
RATE = 0.25 + (0.43 if MACHINE != "garnet" else 0.70)

for i in range(10):
    inst = parser(FILE, i)
    pt = inst["processing_times"]
    LB = inst["info"]["lower_bound"]
    pi0 = list(range(len(pt[0])))
    c0 = c_max(pi0, pt)
    line = f"{i:>4} {LB:>5} {c0:>6} |"
    for name, fn in FN.items():
        key = f"{i}_{name}"
        if key in done:
            r = done[key]
            line += f" {r['cmax']:>5} {r['prd']:>6.2f}% (cache) |"
            continue
        t0 = time.time()
        try:
            _, cmax, swaps = fn(pi0, pt, p=1, backend="iqm", shots=SHOTS)
        except Exception as e:
            done[key] = {"error": f"{type(e).__name__}: {e}"}
            OUT.write_text(json.dumps(done, indent=1))
            line += f" {'BLAD':>13} |"
            continue
        wall = time.time() - t0
        prd = 100 * (cmax - LB) / LB
        done[key] = {"inst": i, "neigh": name, "machine": hw.name, "lb": LB,
                     "cmax_start": c0, "cmax": cmax, "prd": prd,
                     "swaps": swaps, "wall_s": wall}
        OUT.write_text(json.dumps(done, indent=1))      # save after every paid run
        spent += RATE
        line += f" {cmax:>5} {prd:>6.2f}% {wall:>5.1f}s |"
    print(line, flush=True)

print(f"\n{'=' * 68}")
PUB = {"adjacent": 25.04, "fibonacci": 23.41}        # ibm_fez, Tabela 7, 10 inst, ILS
print(f"{'sasiedztwo':>11} {'srednia':>9} {'std':>7} {'min':>7} {'max':>7} | "
      f"{'ibm_fez (publ.)':>15}")
summary = {}
for name in FN:
    vals = [done[f"{i}_{name}"]["prd"] for i in range(10)
            if f"{i}_{name}" in done and "prd" in done[f"{i}_{name}"]]
    if len(vals) < 2:
        print(f"{name:>11} za malo punktow ({len(vals)})")
        continue
    m, sd = statistics.mean(vals), statistics.stdev(vals)
    summary[name] = {"n": len(vals), "mean": m, "std": sd,
                     "min": min(vals), "max": max(vals)}
    print(f"{name:>11} {m:>9.2f} {sd:>7.2f} {min(vals):>7.2f} {max(vals):>7.2f} | "
          f"{PUB[name]:>15.2f}")
done["_summary"] = summary
OUT.write_text(json.dumps(done, indent=1))
print(f"\nzapisane -> {OUT.name}   nowych biegow oplaconych: ~{spent:.2f} kredytow")
