#!/usr/bin/env bash
# Sekwencyjne uruchomienie wszystkich 10 instancji dynasearch dla n=500.
# Każdy chunk pisze do tego samego experiment dir przez resume_dir.
set -e
cd /Users/remigiuszwojewodzki/Desktop/PhD

echo "================================================================"
echo "DYNASEARCH n=500 — chunk 1 (inst 0-2)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh dynasearch --inst 0-2

echo "================================================================"
echo "DYNASEARCH n=500 — chunk 2 (inst 3-5)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh dynasearch --inst 3-5

echo "================================================================"
echo "DYNASEARCH n=500 — chunk 3 (inst 6-9)  $(date '+%H:%M:%S')"
echo "================================================================"
.venv311/bin/python3 -u scripts/run_n500_chunk.py --neigh dynasearch --inst 6-9

echo "================================================================"
echo "ALL DYNASEARCH n=500 COMPLETED  $(date '+%H:%M:%S')"
echo "================================================================"
