#!/usr/bin/env bash
# Sekwencyjne uruchomienie wszystkich 10 instancji motzkin dla n=500.
# Analogiczne do run_n500_dyn_all.sh.
set -e
cd /Users/remigiuszwojewodzki/Desktop/PhD

echo "================================================================"
echo "MOTZKIN n=500 — chunk 1 (inst 0-2)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh motzkin --inst 0-2

echo "================================================================"
echo "MOTZKIN n=500 — chunk 2 (inst 3-5)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh motzkin --inst 3-5

echo "================================================================"
echo "MOTZKIN n=500 — chunk 3 (inst 6-9)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh motzkin --inst 6-9

echo "================================================================"
echo "ALL MOTZKIN n=500 COMPLETED  $(date '+%H:%M:%S')"
echo "================================================================"
