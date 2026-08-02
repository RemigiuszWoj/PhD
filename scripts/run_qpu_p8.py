"""QPU campaign P8 --- complete the tl=10000 column of Table 6 (tab:rpd_n20_tl).

Run this when the monthly Leap quota refreshes. It is the deferred layer
from P6/P7: every neighborhood at n=20, m=5, tl=10000 ms that is not yet
measured.

Already done (skipped automatically via result.json):
  - quantum_fibonacci (orig): 100/100  --> excluded from the plan
  - quantum_adjacent  (orig):  11/100  --> resumes the missing 89

Still to measure (target 100 = 50 ILS + 50 SA each):
  orig: quantum_adjacent (resume), quantum_motzkin, quantum_dynasearch
  enh : quantum_{adjacent,fibonacci,dynasearch,motzkin}_enhanced

Estimated cost (from measured calls/run x ~32 ms access):
  ~73 s of QPU access total -- about 18 % of the ~400 s monthly quota,
  so it fits comfortably in one month's allowance.
  Wall clock is the real bottleneck, NOT the budget: dynasearch orig
  runs ~140 s of minorminer search per call (~4 h for its 100 runs),
  so the whole column is ~6--7 h of wall time. Run it in the background
  over one or two nights.

Phase order is cheapest-wall-clock first so the light neighborhoods
finish quickly and dynasearch orig (the 4 h grind) goes last:
  P1 orig adjacent (resume)         ~2 s/call
  P2 enh  adjacent + fibonacci      ~2 s/call
  P3 orig motzkin                   ~18 s/call
  P4 enh  dynasearch + motzkin      windowed, minutes/call
  P5 orig dynasearch                ~140 s/call  <-- last

Each phase caps itself at (--budget - access_used_so_far), measured from
results/qpu_timing.jsonl, so the global cap holds across the per-phase
budget resets in ExperimentRunner. Everything is --resume, so a session
that dies mid-column just continues on the next run.

IMPORTANT --- the background process does NOT inherit the D-Wave token.
Launch with the gitignored .env sourced:

    cd <repo>
    set -a; . ./.env; set +a
    nohup .venv311/bin/python3 scripts/run_qpu_p8.py --budget 90 \
        > results/p8.log 2>&1 &

(never print the token value). --budget is how much QPU access [s] this
whole campaign may spend; 90 covers the ~73 s estimate with margin.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.runner import ExperimentRunner, RunConfig  # noqa: E402

INSTANCE_FILE = "data/tai20_5.txt"
ENH = "results/experiments/20260707_203922"
ORIG = "results/experiments/20260703_083952"
TIMING_LOG = "results/qpu_timing.jsonl"
TL = 10000

T0 = time.time()


def used_since_start() -> float:
    if not os.path.exists(TIMING_LOG):
        return 0.0
    tot = 0.0
    with open(TIMING_LOG) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) >= T0:
                tot += (r.get("qpu_access_time_us") or 0) / 1e6
    return tot


def cfgs(neighs, insts=range(10), seeds=range(5)):
    plan = []
    for ng in neighs:
        for algo in ("ils", "sa"):
            for inst in insts:
                for seed in seeds:
                    plan.append(RunConfig(
                        algorithm=algo, neighborhood=ng,
                        instance_file=INSTANCE_FILE, instance_number=inst,
                        seed=seed, time_limit_ms=TL,
                        tabu_tenure=10 if algo == "ils" else None,
                    ))
    return plan


def run_phase(name, resume_dir, neighs, base_qconf, budget) -> None:
    remaining = budget - used_since_start()
    if remaining <= 0.15:
        print(f"[P8:{name}] SKIP -- budget spent "
              f"({used_since_start():.2f}/{budget}s)", flush=True)
        return
    qconf = dict(base_qconf)
    qconf["backend"] = "dwave"
    qconf["qpu_budget_s"] = remaining
    plan = cfgs(neighs)
    print(f"[P8:{name}] cap={remaining:.2f}s (used "
          f"{used_since_start():.2f}/{budget}s), dir={resume_dir}, "
          f"neighs={neighs}, planned={len(plan)}", flush=True)
    ExperimentRunner(quantum_config=qconf, resume_dir=resume_dir).run(plan)
    print(f"[P8:{name}] done -- used {used_since_start():.2f}/{budget}s",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0,
                    help="total QPU access budget for the whole column [s]")
    ap.add_argument("--fast", action="store_true",
                    help="only the wall-clock-cheap phases (P1,P2); skip the "
                         "slow motzkin/dynasearch phases (P3-P5)")
    args = ap.parse_args()
    with open("config.yaml", "r", encoding="utf-8") as f:
        base_qconf = dict((yaml.safe_load(f).get("quantum") or {}))

    b = args.budget
    run_phase("P1-orig-adjacent", ORIG, ["quantum_adjacent"], base_qconf, b)
    run_phase("P2-enh-adj-fib", ENH,
              ["quantum_adjacent_enhanced", "quantum_fibonacci_enhanced"],
              base_qconf, b)
    if args.fast:
        print("[P8] --fast: skipping slow phases P3-P5 "
              "(motzkin/dynasearch); run without --fast next month.",
              flush=True)
    else:
        run_phase("P3-orig-motzkin", ORIG, ["quantum_motzkin"], base_qconf, b)
        run_phase("P4-enh-dyn-motz", ENH,
                  ["quantum_dynasearch_enhanced", "quantum_motzkin_enhanced"],
                  base_qconf, b)
        run_phase("P5-orig-dynasearch", ORIG, ["quantum_dynasearch"],
                  base_qconf, b)

    print(f"[P8] ALL PHASES DONE -- global qpu_access used "
          f"{used_since_start():.2f}/{b}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
