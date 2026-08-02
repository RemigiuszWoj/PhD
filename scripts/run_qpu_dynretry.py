"""Targeted retry of the residual dynasearch embedding failures.

After P7, four dynasearch runs (all instance 6, tl in {100,1000,2000})
still find no minor-embedding for the ~190-variable QUBO. minorminer is
stochastic, so a fresh search may still succeed. Retries are budget-free
when they miss (the guard is checked before submission) and only ~32 ms
each when they embed, so a tight cap keeps the global 90 s allowance
safe. The runner skips runs that already have result.json and retries
only the failed.json ones.

Usage:
    .venv311/bin/python3 scripts/run_qpu_dynretry.py --cap 0.7
"""
import argparse, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

DIR = "results/experiments/20260703_083952"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.7)
    args = ap.parse_args()
    with open("config.yaml") as f:
        q = dict((yaml.safe_load(f).get("quantum") or {}))
    q["backend"] = "dwave"
    q["qpu_budget_s"] = args.cap
    plan = [
        RunConfig(algorithm=a, neighborhood="quantum_dynasearch",
                  instance_file="data/tai20_5.txt", instance_number=i,
                  seed=s, time_limit_ms=tl, tabu_tenure=10 if a == "ils" else None)
        for tl in (100, 500, 1000, 2000) for a in ("ils", "sa")
        for i in range(10) for s in range(5)
    ]
    print(f"[dynretry] cap={args.cap}s, dir={DIR}, planned={len(plan)} "
          "(runner retries only the failed.json ones)", flush=True)
    ExperimentRunner(quantum_config=q, resume_dir=DIR).run(plan)
    print("[dynretry] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
