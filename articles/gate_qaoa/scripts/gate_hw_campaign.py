"""Hardware column for the gate block of the windowed-article table (tab:rpd_n20_tl).

Starts with the LAST budget column (10000 ms) for all four neighborhoods and
both algorithms on real ibm_fez. If a run completes only one move at 10000 ms,
every shorter budget necessarily does too, so the remaining columns are the same
numbers and need no QPU time -- that is what this run establishes.

Scope: tai20_5, instances 0..9, seed 0, ils + sa, gate_{adjacent,fibonacci,
dynasearch,motzkin}, p=1, window 6, 4096 shots.

Safety: checks the IBM usage quota before every run and stops when less than
MIN_REMAINING seconds are left.
"""
import os, sys, time

for _l in open(".env"):
    _l = _l.strip()
    if _l and not _l.startswith("#") and "=" in _l:
        k, v = _l.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

from qiskit_ibm_runtime import QiskitRuntimeService
from src.experiments.runner import ExperimentRunner, RunConfig

FILE = "data/tai20_5.txt"
NEIGH = ["gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"]
ALGOS = ["ils", "sa"]
TL = 10000
RESULTS_DIR = "results/experiments/qaoa_gate_hw_n20"
MIN_REMAINING = 60          # seconds of QPU time to leave untouched

_svc = QiskitRuntimeService(channel="ibm_quantum_platform",
                            token=os.environ["IBM_TOKEN"],
                            instance=os.environ.get("IBM_CRN"))


def remaining():
    try:
        return _svc.usage()["usage_remaining_seconds"]
    except Exception as exc:
        print(f"[hw] usage check failed ({exc}); assuming exhausted", flush=True)
        return 0


quantum_config = {
    "qaoa_backend": "ibm",
    "qaoa_p": 1,
    "qaoa_window_size": 6,
    "qaoa_overlap_ratio": 0.5,
    "qaoa_shots": 4096,
}

runner = ExperimentRunner(base_results_dir="results/experiments",
                          quantum_config=quantum_config,
                          resume_dir=RESULTS_DIR)

configs = []
for neigh in NEIGH:
    for algo in ALGOS:
        for inst in range(10):
            configs.append(RunConfig(algorithm=algo, neighborhood=neigh,
                                     instance_file=FILE, instance_number=inst,
                                     seed=0, time_limit_ms=TL))

print(f"[hw] {len(configs)} runs at tl={TL} ms on ibm_fez; quota left {remaining()} s",
      flush=True)

for i, cfg in enumerate(configs, 1):
    left = remaining()
    if left < MIN_REMAINING:
        print(f"[hw] STOP: only {left}s of QPU time left, {len(configs)-i+1} runs skipped",
              flush=True)
        break
    print(f"[hw] ({i}/{len(configs)}) {cfg.neighborhood} {cfg.algorithm} inst={cfg.instance_number} "
          f"| quota {left}s", flush=True)
    t0 = time.time()
    runner.run([cfg])
    print(f"[hw]   wall {time.time()-t0:.0f}s", flush=True)

print(f"[hw] done; quota left {remaining()} s", flush=True)
