"""Targeted embedding-retry for the 4 tl=10000 cells that failed QPU embedding.

minorminer is stochastic, so deleting the stale failed.json and re-running the
exact config can find an embedding an earlier attempt missed (same trick as P7).
A miss is budget-free (the guard checks the budget before submission, an
embedding miss never reaches the QPU); a hit is ~1 QPU call.

Run (on the windowed_qubo branch, .env sourced for the D-Wave token):
    set -a; . ./.env; set +a
    .venv311/bin/python3 scripts/retry_embed_stragglers.py
"""
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

INST = "data/tai20_5.txt"
ORIG = "results/experiments/20260703_083952"
ENH = "results/experiments/20260707_203922"

# (resume_dir, algo, neighborhood, instance, seed, tabu_tenure)
CELLS = [
    (ORIG, "ils", "quantum_dynasearch", 2, 2, 10),
    (ORIG, "ils", "quantum_dynasearch", 6, 3, 10),
    (ORIG, "sa",  "quantum_dynasearch", 6, 1, None),
    (ENH,  "sa",  "quantum_motzkin_enhanced", 7, 1, None),
]
MAX_ROUNDS = 5


def cell_dir(base, algo, ng, inst, seed):
    return (f"{base}/algo={algo}__neigh={ng}__file=tai20_5"
            f"__inst={inst}__n20__m5__tl=10000ms__seed={seed}")


def done(base, algo, ng, inst, seed):
    return os.path.exists(cell_dir(base, algo, ng, inst, seed) + "/result.json")


def main():
    base_q = dict((yaml.safe_load(open("config.yaml")).get("quantum") or {}))
    for rnd in range(1, MAX_ROUNDS + 1):
        by_dir = {}
        for base, algo, ng, inst, seed, tt in CELLS:
            if done(base, algo, ng, inst, seed):
                continue
            fj = cell_dir(base, algo, ng, inst, seed) + "/failed.json"
            if os.path.exists(fj):
                os.remove(fj)  # force a fresh embedding search
            by_dir.setdefault(base, []).append(
                RunConfig(algorithm=algo, neighborhood=ng, instance_file=INST,
                          instance_number=inst, seed=seed, time_limit_ms=10000,
                          tabu_tenure=tt))
        if not by_dir:
            print("[retry] KOMPLET — nic do dobicia", flush=True)
            break
        for base, plan in by_dir.items():
            q = dict(base_q); q["backend"] = "dwave"; q["qpu_budget_s"] = 8.0
            print(f"[retry r{rnd}] {base}: {len(plan)} do dobicia", flush=True)
            ExperimentRunner(quantum_config=q, resume_dir=base).run(plan)
        left = sum(1 for base, algo, ng, inst, seed, tt in CELLS
                   if not done(base, algo, ng, inst, seed))
        print(f"[retry r{rnd}] po rundzie zostało {left}", flush=True)
        if left == 0:
            print("[retry] KOMPLET 800/800", flush=True)
            break
    else:
        left = [f"{a} {ng} i{i} s{s}" for base, a, ng, i, s, tt in CELLS
                if not done(base, a, ng, i, s)]
        print(f"[retry] po {MAX_ROUNDS} rundach wciąż fail: {left}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
