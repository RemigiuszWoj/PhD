"""QPU campaign — P7: recover failed runs + spend the residual ~18 s budget.

Context: after P6 the 90 s allowance stands at 71.1 s used (18.9 s left).
Two jobs, one hard global budget of 18 s of NEW qpu_access_time:

  (1) Retry the 13 genuinely-failed runs (the 800 "budget exhausted"
      failures are the intentionally-deferred tl=10000 column, handled
      separately below):
        - 2x quantum_motzkin_enhanced tl=2000 (transient SAPI/other) --
          cheap, almost certainly recovers the cell to n=50.
        - 11x quantum_dynasearch tl=100/500/1000/2000 (embedding, 189/181
          vars). minorminer is stochastic, so a fresh search may embed
          what an earlier attempt missed. Budget-free when it fails (the
          guard is checked before submission, an embedding miss never
          reaches the QPU).
  (2) Use the remaining budget on the tl=10000 column for the only
      neighborhoods that are wall-clock-feasible (adjacent, fibonacci;
      ~2 s/call, embed trivially). motzkin (18.5 s/call) and dynasearch
      (140 s/call) at tl=10000 stay deferred to next month.

Phase order (each phase gets cap = 18 - access_used_so_far, measured
from results/qpu_timing.jsonl so the global cap is honoured across the
per-phase budget resets in ExperimentRunner.__init__):

  P1  enh  motzkin_enh tl=2000 retry          (guaranteed, ~0.1 s)
  P2  orig dynasearch tl=100/500/1000/2000     (embedding retries, ~free)
  P3  orig fibonacci + adjacent tl=10000       (budget-productive)
  P4  enh  fibonacci + adjacent tl=10000       (whatever budget remains)

Whole cells complete in order (neigh -> algo -> inst -> seed), so a
mid-phase budget cut leaves clean n=50 cells plus one ragged cell that
resumes next month. --skip-dyn drops P2 if minor-embedding stalls.

Usage:
    .venv311/bin/python3 scripts/run_qpu_p7.py            # all phases
    .venv311/bin/python3 scripts/run_qpu_p7.py --skip-dyn # drop P2
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
GLOBAL_BUDGET_S = 18.0          # 71.1 already used -> total <= ~89.2 < 90
TIMING_LOG = "results/qpu_timing.jsonl"

T0 = time.time()                # only P7's access counts against the cap


def used_since_start() -> float:
    """Seconds of qpu_access_time logged since this process began."""
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


def cfgs(neighs, tls, insts=range(10), seeds=range(5)):
    plan = []
    for tl in tls:
        for ng in neighs:
            for algo in ("ils", "sa"):
                for inst in insts:
                    for seed in seeds:
                        plan.append(RunConfig(
                            algorithm=algo,
                            neighborhood=ng,
                            instance_file=INSTANCE_FILE,
                            instance_number=inst,
                            seed=seed,
                            time_limit_ms=tl,
                            tabu_tenure=10 if algo == "ils" else None,
                        ))
    return plan


def run_phase(name, resume_dir, plan, base_qconf) -> None:
    remaining = GLOBAL_BUDGET_S - used_since_start()
    if remaining <= 0.15:
        print(f"[P7:{name}] SKIP — global budget spent "
              f"({used_since_start():.2f}/{GLOBAL_BUDGET_S}s)", flush=True)
        return
    qconf = dict(base_qconf)
    qconf["backend"] = "dwave"
    qconf["qpu_budget_s"] = remaining
    print(f"[P7:{name}] cap={remaining:.2f}s (global used "
          f"{used_since_start():.2f}/{GLOBAL_BUDGET_S}s), dir={resume_dir}, "
          f"planned={len(plan)}", flush=True)
    runner = ExperimentRunner(quantum_config=qconf, resume_dir=resume_dir)
    runner.run(plan)
    print(f"[P7:{name}] done — global used {used_since_start():.2f}/"
          f"{GLOBAL_BUDGET_S}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-dyn", action="store_true",
                    help="drop the dynasearch embedding retries (P2)")
    args = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        base_qconf = dict((yaml.safe_load(f).get("quantum") or {}))

    # P1: transient recovery (cheap, high value) ---------------------------
    run_phase("P1-motz-enh-retry", ENH,
              cfgs(["quantum_motzkin_enhanced"], [2000]), base_qconf)

    # P2: dynasearch embedding retries (budget-free on miss) ---------------
    if not args.skip_dyn:
        run_phase("P2-dyn-retry", ORIG,
                  cfgs(["quantum_dynasearch"], [100, 500, 1000, 2000]),
                  base_qconf)

    # P3: budget-productive tl=10000 fill, original formulations -----------
    run_phase("P3-orig-tl10000", ORIG,
              cfgs(["quantum_fibonacci", "quantum_adjacent"], [10000]),
              base_qconf)

    # P4: tl=10000 fill, enhanced formulations (leftover budget) -----------
    run_phase("P4-enh-tl10000", ENH,
              cfgs(["quantum_fibonacci_enhanced", "quantum_adjacent_enhanced"],
                   [10000]),
              base_qconf)

    print(f"[P7] ALL PHASES DONE — global qpu_access used "
          f"{used_since_start():.2f}/{GLOBAL_BUDGET_S}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
