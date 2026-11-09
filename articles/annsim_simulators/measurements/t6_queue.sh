#!/bin/zsh
# T6 after the night script: the whole grid (p = 1-6, S1-S3, all engines, 300 s cap per window), one cell at a
# time and nothing else on the machine; then a multi-thread pass (8 threads) at p = 1 in S2.
cd /Users/remigiuszwojewodzki/Desktop/PhD/.claude/worktrees/annsim-simulators
PY=/Users/remigiuszwojewodzki/Desktop/PhD/.venv-annsim/bin/python
echo "=== $(date '+%d.%m %H:%M') T6, 1 thread"
PYTHONPATH=. $PY -m src.simulation.t6 run --cap 300 --budget-h 60 --threads 1 2>&1 | grep -v -i "runtimewarning\|row\["
echo "=== $(date '+%d.%m %H:%M') T6, 8 threads, p = 1, S2"
PYTHONPATH=. $PY -m src.simulation.t6 run --cap 300 --budget-h 12 --threads 8 --p 1 --scenarios S2 2>&1 | grep -v -i "runtimewarning\|row\["
echo "=== $(date '+%d.%m %H:%M') T6 done"
