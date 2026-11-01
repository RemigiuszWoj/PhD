"""Staged quantum campaign for the k-bonacci paper: MINIMUM -> OPTIMUM -> MAXIMUM.

Each tier runs only if the remaining budget covers its estimated cost; the
campaign stops cleanly at the first tier it cannot afford. Everything is
resumable: finished runs are skipped, so the script can simply be started again
next month (new allocation) and it continues where it stopped.

Protocols are the ones of the previous quantum papers (nothing new):

  D-Wave (as windowed_qubo): ExperimentRunner, ILS/SA unchanged, identity start,
      instances 0-9, seeds 0-4, num_reads = 100, t_max = 5000 ms
      (smaller budgets are extracted from the move log, see extract_budgets.py).
      Neighborhood: quantum_kbonacci_k<K> (interval QUBO, L_max = k on the QPU).
  IQM (as gate_qaoa): one move from the identity permutation, seed 0, p = 1,
      4096 shots, window 6, overlap 0.5, calibrated 'dynasearch' angles.
      Neighborhood: gate_kbonacci (windowed, L_max = k).

Usage (on the Mac, from the repo root, tokens in .env):
    python scripts/run_quantum_tiers.py --backend dwave --dry-run
    python scripts/run_quantum_tiers.py --backend dwave --budget-s 400
    python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits 30 --dry-run
    python scripts/run_quantum_tiers.py --backend iqm --machine sirius --credits 30
Smoke test without hardware (no budget used):
    python scripts/run_quantum_tiers.py --backend dwave --simulate --limit 2 --tl-ms 500
"""

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:  # tokens (DWAVE_API_TOKEN, IQM_TOKEN) live in the gitignored .env
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import yaml  # noqa: E402

TAILLARD = [
    "tai20_5", "tai20_10", "tai20_20", "tai50_5", "tai50_10", "tai50_20",
    "tai100_5", "tai100_10", "tai100_20", "tai200_10", "tai200_20", "tai500_20",
]
INSTANCES = list(range(10))
SEEDS = list(range(5))
OUT = ROOT / "results" / "quantum_tiers"
LEDGER = OUT / "ledger.json"


@dataclass
class DWaveBlock:
    files: Sequence[str]
    algos: Sequence[str]
    ks: Sequence[int]


@dataclass
class IQMBlock:
    files: Sequence[str]
    ks: Sequence[int]
    instances: Sequence[int] = field(default_factory=lambda: INSTANCES)


# Tiers are cumulative: OPTIMUM adds to MINIMUM, MAXIMUM adds to OPTIMUM.
DWAVE_TIERS: Dict[str, List[DWaveBlock]] = {
    # Table "n = 20" of windowed_qubo: tai20_5, ILS + SA.
    "min": [DWaveBlock(["tai20_5"], ["ils", "sa"], [3, 4])],
    # + k = 8 at n = 20, + scaling rows of windowed_qubo (ILS, m = 10) for n = 20..100.
    "opt": [
        DWaveBlock(["tai20_5"], ["ils", "sa"], [8]),
        DWaveBlock(["tai20_10", "tai50_10", "tai100_10"], ["ils"], [3, 4]),
    ],
    # + rest of the scaling table (n = 200, 500), then the full Taillard benchmark.
    "max": [
        DWaveBlock(["tai200_10", "tai500_20"], ["ils"], [3, 4]),
        DWaveBlock(TAILLARD, ["ils", "sa"], [3, 4]),
    ],
}
IQM_TIERS: Dict[str, List[IQMBlock]] = {
    "min": [IQMBlock(["tai20_5"], [3, 4], instances=[0])],
    "opt": [IQMBlock(["tai20_5"], [3, 4])],
    "max": [IQMBlock(["tai20_5"], [8]), IQMBlock(["tai20_10", "tai20_20"], [3, 4])],
}
TIER_ORDER = ["min", "opt", "max"]

# IQM Resonance price list measured in gate_qaoa (credits): fee per job + per circuit.
IQM_JOB_FEE = 0.25
IQM_RATE = {"sirius": 0.43, "garnet": 0.70, "emerald": 0.70}  # emerald: not measured yet
IQM_QUBITS = {"sirius": 16, "garnet": 20, "emerald": 54}
IQM_MAX_CIRCUITS = 100
CREDIT_USD = 0.60


# ----------------------------------------------------------------------------- ledger
def month_key() -> str:
    return dt.date.today().strftime("%Y-%m")


def load_ledger() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"dwave_s": {}, "iqm_credits": {}}


def save_ledger(ledger: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    tmp.replace(LEDGER)


# ----------------------------------------------------------------------------- D-Wave
def dwave_tier_configs(tier: str, tl_ms: int):
    from src.experiments.runner import RunConfig

    seen, out = set(), []
    for t in TIER_ORDER[: TIER_ORDER.index(tier) + 1]:
        for b in DWAVE_TIERS[t]:
            for f in b.files:
                for algo in b.algos:
                    for k in b.ks:
                        for i in INSTANCES:
                            for s in SEEDS:
                                key = (f, algo, k, i, s)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append((t, RunConfig(
                                    algorithm=algo,
                                    neighborhood=f"quantum_kbonacci_k{k}",
                                    instance_file=f"data/{f}.txt",
                                    instance_number=i,
                                    seed=s,
                                    time_limit_ms=tl_ms,
                                    tabu_tenure=10 if algo == "ils" else None,
                                )))
    return [cfg for t, cfg in out if t == tier]


def run_dwave(args) -> None:
    from src.experiments.runner import ExperimentRunner
    from src.neighborhoods.common import get_qpu_budget

    res_dir = OUT / ("dwave_sim" if args.simulate else "dwave")
    res_dir.mkdir(parents=True, exist_ok=True)
    with open("config.yaml") as fh:
        qc = dict((yaml.safe_load(fh) or {}).get("quantum", {}))
    qc.update({"backend": "simulator" if args.simulate else "dwave", "num_reads": args.num_reads})

    calls_per_run = math.ceil(args.tl_ms / (1000 * args.call_wall_s))
    est_run_s = calls_per_run * args.billed_ms_per_call / 1000
    ledger = load_ledger()
    month = month_key()

    for tier in args.tiers:
        cfgs = dwave_tier_configs(tier, args.tl_ms)
        probe = ExperimentRunner(base_results_dir=str(OUT), quantum_config=dict(qc, qpu_budget_s=None),
                                 resume_dir=str(res_dir))
        todo = [c for c in cfgs if not probe._already_done(c)]
        n_left = len(todo)
        if args.limit:
            todo = todo[: args.limit]
        est = len(todo) * est_run_s
        used = ledger["dwave_s"].get(month, 0.0)
        remaining = args.budget_s - used
        wall_h = len(todo) * (args.tl_ms / 1000 + args.call_wall_s) / 3600
        print(f"\n=== D-Wave tier {tier.upper()}: {len(cfgs)} runs, {n_left} left"
              f"{f' (this call: {len(todo)})' if args.limit else ''} | "
              f"estimate <= {est:.1f} s QPU ({calls_per_run} calls/run x {args.billed_ms_per_call} ms), "
              f"~{wall_h:.1f} h wall | month {month}: used {used:.1f} s of {args.budget_s:.0f} s")
        if not todo:
            print("    nothing to do (tier complete)")
            continue
        if args.dry_run:
            continue
        if not args.simulate and remaining <= 0:  # a cap of 0 would mean "unlimited" in solve_qubo
            print(f"    STOP: no QPU time left this month ({used:.1f} of {args.budget_s:.0f} s used).")
            return
        if not args.simulate and est > remaining and not args.partial:
            print(f"    STOP: tier needs ~{est:.1f} s but only {remaining:.1f} s left this month. "
                  "Re-run next month (it resumes) or pass --partial to spend what is left.")
            return
        cap = None if args.simulate else max(0.0, remaining)
        runner = ExperimentRunner(base_results_dir=str(OUT), quantum_config=dict(qc, qpu_budget_s=cap),
                                  resume_dir=str(res_dir))
        runner.run(todo)
        spent = get_qpu_budget()["used_ms"] / 1000.0
        if not args.simulate:
            ledger["dwave_s"][month] = used + spent
            save_ledger(ledger)
        print(f"    tier {tier.upper()} done: QPU access {spent:.1f} s (ledger {month}: "
              f"{ledger['dwave_s'].get(month, 0.0):.1f} s)")
        if not args.simulate and cap is not None and spent >= cap:
            print("    budget cap reached; stopping (resume next month).")
            return


# ----------------------------------------------------------------------------- IQM
def iqm_tier_moves(tier: str):
    seen, out = set(), []
    for t in TIER_ORDER[: TIER_ORDER.index(tier) + 1]:
        for b in IQM_TIERS[t]:
            for f in b.files:
                for k in b.ks:
                    for i in b.instances:
                        if (f, k, i) not in seen:
                            seen.add((f, k, i))
                            out.append((t, (f, k, i)))
    return [m for t, m in out if t == tier]


def move_price(sizes: List[int], rate: float) -> float:
    jobs = max(1, math.ceil(len(sizes) / IQM_MAX_CIRCUITS))
    return jobs * IQM_JOB_FEE + len(sizes) * rate


def run_iqm(args) -> None:
    from src.neighborhoods.gate_qaoa.kbonacci import gate_kbonacci_neighborhood, window_plan
    from src.parser import parser
    from src.permutation_procesing import c_max

    os.environ["IQM_COMPUTER"] = args.machine
    rate, qubits = IQM_RATE[args.machine], IQM_QUBITS[args.machine]
    res_path = OUT / f"iqm_{'sim' if args.simulate else args.machine}.json"
    done = json.loads(res_path.read_text()) if res_path.exists() else {}
    ledger = load_ledger()
    month = month_key()

    for tier in args.tiers:
        moves = [m for m in iqm_tier_moves(tier) if f"{m[0]}|{m[1]}|{m[2]}" not in done]
        n_left = len(moves)
        if args.limit:
            moves = moves[: args.limit]
        priced = []
        for f, k, i in moves:
            pt = parser(f"data/{f}.txt", i)["processing_times"]
            sizes = window_plan(list(range(len(pt[0]))), pt, k, args.window, args.overlap)
            priced.append(((f, k, i), sizes, move_price(sizes, rate)))
        est = sum(p for _, _, p in priced)
        too_big = [(m, max(s)) for m, s, _ in priced if s and max(s) > qubits]
        used = ledger["iqm_credits"].get(month, 0.0)
        remaining = args.credits - used
        print(f"\n=== IQM tier {tier.upper()} ({args.machine}, {qubits} qubits): {n_left} moves left"
              f"{f' (this call: {len(priced)})' if args.limit else ''} | "
              f"{sum(len(s) for _, s, _ in priced)} circuits | estimate {est:.1f} credits "
              f"(~{est * CREDIT_USD:.0f} USD) | month {month}: used {used:.1f} of {args.credits:.0f}")
        if too_big:
            print(f"    WARNING: {len(too_big)} moves need more qubits than {args.machine} has, "
                  f"e.g. {too_big[0]} -> use a larger machine or a smaller window")
        if not priced:
            print("    nothing to do (tier complete)")
            continue
        if args.dry_run:
            continue
        if not args.simulate and est > remaining and not args.partial:
            print(f"    STOP: tier needs ~{est:.1f} credits but only {remaining:.1f} left. "
                  "Ask for credits / wait for the monthly pool, then re-run (it resumes).")
            return
        for (f, k, i), sizes, price in priced:
            if not args.simulate and ledger["iqm_credits"].get(month, 0.0) + price > args.credits:
                print("    credit cap reached; stopping (re-run resumes).")
                return
            inst = parser(f"data/{f}.txt", i)
            pt, lb = inst["processing_times"], inst["info"]["lower_bound"]
            pi0 = list(range(len(pt[0])))
            t0 = time.time()
            new_pi, new_c, swaps = gate_kbonacci_neighborhood(
                pi0, pt, k, p=1, backend="aer_noisy" if args.simulate else "iqm",
                shots=4096, window_size=args.window, overlap_ratio=args.overlap)
            rec = {"file": f, "k": k, "instance": i, "lb": lb, "cmax_start": c_max(pi0, pt),
                   "cmax": new_c, "prd": 100.0 * (new_c - lb) / lb, "swaps": swaps,
                   "circuits": len(sizes), "vars_per_circuit": sizes, "est_credits": price,
                   "wall_s": time.time() - t0, "machine": "sim" if args.simulate else args.machine,
                   "tier": tier, "date": dt.datetime.now().isoformat(timespec="seconds")}
            done[f"{f}|{k}|{i}"] = rec
            res_path.write_text(json.dumps(done, indent=1))  # after EVERY paid move
            if not args.simulate:
                ledger["iqm_credits"][month] = ledger["iqm_credits"].get(month, 0.0) + price
                save_ledger(ledger)
            print(f"    {f} inst {i} k={k}: Cmax {rec['cmax_start']} -> {new_c} "
                  f"(PRD {rec['prd']:.2f}%), {len(sizes)} circuits, ~{price:.2f} cr")
    if done:
        print("\nIQM summary (mean PRD [%] per file and k):")
        groups: Dict[tuple, List[float]] = {}
        for r in done.values():
            groups.setdefault((r["file"], r["k"]), []).append(r["prd"])
        for (f, k), v in sorted(groups.items()):
            print(f"    {f:<9} k={k}: {statistics.mean(v):6.2f}  (n={len(v)})")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["dwave", "iqm"], required=True)
    ap.add_argument("--tiers", default="min,opt,max", help="comma list, in order (default min,opt,max)")
    ap.add_argument("--dry-run", action="store_true", help="only print plan and cost estimates")
    ap.add_argument("--simulate", action="store_true",
                    help="no hardware: D-Wave -> dimod simulator, IQM -> aer_noisy; ledger untouched")
    ap.add_argument("--partial", action="store_true", help="start a tier even if the estimate exceeds the budget")
    ap.add_argument("--limit", type=int, default=0, help="at most N runs/moves per tier (smoke tests)")
    # D-Wave
    ap.add_argument("--budget-s", type=float, default=400.0, help="monthly D-Wave QPU allocation [s]")
    ap.add_argument("--tl-ms", type=int, default=5000)
    ap.add_argument("--num-reads", type=int, default=100)
    ap.add_argument("--call-wall-s", type=float, default=2.1, help="wall time of one QPU call (windowed_qubo)")
    ap.add_argument("--billed-ms-per-call", type=float, default=30.0, help="billed QPU access per call")
    # IQM
    ap.add_argument("--machine", choices=sorted(IQM_RATE), default="sirius")
    ap.add_argument("--credits", type=float, default=30.0, help="IQM credits available this month")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--overlap", type=float, default=0.5)
    args = ap.parse_args()
    args.tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    bad = [t for t in args.tiers if t not in TIER_ORDER]
    if bad:
        ap.error(f"unknown tiers {bad}; use {TIER_ORDER}")
    OUT.mkdir(parents=True, exist_ok=True)
    (run_dwave if args.backend == "dwave" else run_iqm)(args)


if __name__ == "__main__":
    main()
