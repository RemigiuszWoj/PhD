#!/bin/zsh
# After the night script: T5 for n = 20 first (PRD next to Sirius, the paper's main result), then the full T6.
cd /Users/remigiuszwojewodzki/Desktop/PhD/.claude/worktrees/annsim-simulators
PY=/Users/remigiuszwojewodzki/Desktop/PhD/.venv-annsim/bin/python
echo "=== $(date '+%d.%m %H:%M') T5, n = 20, qsim + qulacs_loop, S2 + S3, p = 1"
PYTHONHASHSEED=0 PYTHONPATH=. $PY -m src.simulation.t5_ils --engines qsim qulacs_loop --scenarios S2 S3 \
    --files tai20_5 tai20_10 tai20_20 --p 1 --jobs 8 2>&1 | grep -v -i "runtimewarning\|row\["
./articles/annsim_simulators/measurements/t6_queue.sh
