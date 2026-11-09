#!/bin/zsh
# Night 27/28.10.2026: track B (p = 1 first, the depth Sirius has data for), the rest of stage 2, then
# track B at p = 2-3. Every phase skips cells already in its output file, so a rerun resumes.
# No DDSIM noise cells (engines.UNSUPPORTED), so no crash dialogs.
cd /Users/remigiuszwojewodzki/Desktop/PhD/.claude/worktrees/annsim-simulators
PY=/Users/remigiuszwojewodzki/Desktop/PhD/.venv-annsim/bin/python
M=articles/annsim_simulators/measurements
run() { echo "=== $(date +%H:%M) phase: $*"; PYTHONPATH=. $PY -m src.simulation.stage2 --jobs 8 "$@" 2>&1 | grep -v -i "runtimewarning\|row\["; }
# track B, p = 1: qiskit-iqm windows not yet in S2 (K 12-15) and every cirq-iqm window; qsim for all,
# exact Aer density matrix as reference up to 13 qubits
run --kmin 12 --kmax 15 --p 1 --scenarios S2 --batch 4 --engines qsim --out $M/stage2_trackB_qiskit.jsonl
run --kmax 15 --p 1 --scenarios S2 --batch 4 --engines qsim --circuits $M/circuits_tai20_cirq.json.gz --out $M/stage2_trackB_cirq.jsonl
# rest of stage 2 (p = 3 on the slow engines, PennyLane p = 2-3 up to K = 9)
run --p 3 --scenarios S2 S3 --batch 4 --engines cirq_dm qulacs_dm
run --p 3 --scenarios S2 --batch 4 --engines qibo
run --p 2 3 --scenarios S2 S3 --batch 2 --kmax 9 --engines pennylane
# track B, p = 2-3
run --kmin 12 --kmax 15 --p 2 3 --scenarios S2 --batch 4 --engines qsim --out $M/stage2_trackB_qiskit.jsonl
run --kmax 15 --p 2 3 --scenarios S2 --batch 4 --engines qsim --circuits $M/circuits_tai20_cirq.json.gz --out $M/stage2_trackB_cirq.jsonl
echo "=== $(date +%H:%M) night done"
