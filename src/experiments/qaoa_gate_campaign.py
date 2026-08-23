"""Gate-model QAOA RPD campaign on Taillard n=20, matching the windowed_qubo
methodology exactly (same instances, seeds, time budgets, ILS/SA frameworks,
RPD-from-lower-bound metric). Only the gate neighborhoods are run; the
classical / D-Wave numbers are reused from the windowed_qubo article.

Scope (identical to windowed_qubo for the n=20 size class):
  instances : tai20_5, tai20_10, tai20_20, indices 0..9   (30 instances)
  seeds     : 0..4                                          (5 per instance)
  algorithms: ils, sa
  budgets   : 1000 and 5000 ms   (the two the QPU class reports)
  gate      : gate_adjacent / fibonacci / dynasearch / motzkin, p=1, statevector

Results are written under results/experiments/qaoa_gate_n20/ and the run is
resumable (finished result.json files are skipped).

Usage:  python -m src.experiments.qaoa_gate_campaign [--tl 1000 5000] [--p 1]
        [--backend statevector] [--window 6]
"""
from __future__ import annotations

import argparse
from src.experiments.runner import ExperimentRunner, RunConfig

FILES = ["data/tai20_5.txt", "data/tai20_10.txt", "data/tai20_20.txt"]
NEIGH = ["gate_adjacent", "gate_fibonacci", "gate_dynasearch", "gate_motzkin"]
ALGOS = ["ils", "sa"]
RESULTS_DIR = "results/experiments/qaoa_gate_n20"


def build_configs(tls, instances, seeds):
    configs = []
    for f in FILES:
        for inst in instances:
            for algo in ALGOS:
                for neigh in NEIGH:
                    for tl in tls:
                        for seed in seeds:
                            configs.append(RunConfig(
                                algorithm=algo, neighborhood=neigh,
                                instance_file=f, instance_number=inst,
                                seed=seed, time_limit_ms=tl))
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tl", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--instances", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--backend", type=str, default="statevector")
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args()

    quantum_config = {
        "qaoa_backend": args.backend,
        "qaoa_p": args.p,
        "qaoa_window_size": args.window,
        "qaoa_overlap_ratio": 0.5,
    }
    configs = build_configs(args.tl, args.instances, args.seeds)
    print(f"[campaign] {len(configs)} gate runs "
          f"(files={len(FILES)}, neigh={len(NEIGH)}, algos={len(ALGOS)}, "
          f"tl={args.tl}, backend={args.backend}, p={args.p}) -> {RESULTS_DIR}")
    runner = ExperimentRunner(
        base_results_dir="results/experiments",
        quantum_config=quantum_config,
        resume_dir=RESULTS_DIR,
    )
    runner.run(configs)
    print("[campaign] done")


if __name__ == "__main__":
    main()
