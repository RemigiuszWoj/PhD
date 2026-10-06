"""Offline calibration of fixed QAOA angles for the gate_qaoa neighborhoods.

Builds a training set of small windowed QUBOs per neighborhood from a few
Taillard instances, finds one angle schedule per depth p (via the gate_qaoa
angle engine), and writes them to data/qaoa_angles.json. This is the offline
half of fixed-angle QAOA: run once; solve_qaoa then loads the frozen angles so
every move costs a single circuit execution.

The angles are calibrated on SMALL windows and applied to full-size runtime
QUBOs -- this relies on angle transferability (the optimal p=1 angles of a
neighborhood are near-constant across instances/sizes once the Ising is
scale-normalized).

Usage:
    python -m src.experiments.qaoa_calibrate_angles [--p-max 5] [--window 6]
        [--step 3] [--perms 3] [--out data/qaoa_angles.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.parser import parser
from src.neighborhoods.common import compute_deltas
from src.neighborhoods.common_qubo import (
    assemble_onehot_qubo,
    assemble_pairwise_qubo,
    assemble_tridiagonal_qubo,
    enumerate_interval_candidates,
)
from src.neighborhoods.accelerator import intervals_overlap, motzkin_conflict
from src.neighborhoods.gate_qaoa.angles import optimize_neighborhood

REPO = Path(__file__).resolve().parents[2]

# Taillard instances used for calibration: (file relative to repo, index).
DEFAULT_INSTANCES: List[Tuple[str, int]] = [
    ("data/tai20_5.txt", 0),
    ("data/tai20_5.txt", 1),
    ("data/tai20_10.txt", 0),
]
NEIGHBORHOODS = ("adjacent", "fibonacci", "dynasearch", "motzkin")


def _load_instances(specs: List[Tuple[str, int]]):
    out = []
    for rel, idx in specs:
        data = parser(str(REPO / rel), idx)
        out.append((data["info"]["jobs"], data["processing_times"]))
    return out


def _windows(n: int, w: int, step: int):
    for s in range(0, max(1, n - 1), step):
        e = min(s + w, n)
        if e - s >= 2:
            yield s, e


def _window_qubo(neighborhood: str, pi, pt, s: int, e: int):
    """Build the small windowed QUBO for one neighborhood over positions [s, e)."""
    if neighborhood in ("adjacent", "fibonacci"):
        deltas = compute_deltas(pi, pt)
        cands = [(pos, deltas[pos]) for pos in range(s, min(e - 1, len(deltas)))]
        if not cands:
            return None
        return (assemble_onehot_qubo(cands) if neighborhood == "adjacent"
                else assemble_tridiagonal_qubo(cands))
    # interval neighborhoods: filter_delta=True keeps K small on the simulator
    cands = enumerate_interval_candidates(pi, pt, window=(s, e), filter_delta=True)
    conflict = intervals_overlap if neighborhood == "dynasearch" else motzkin_conflict
    return assemble_pairwise_qubo(cands, conflict)


def build_training(neighborhood: str, instances, *, window: int, step: int,
                   perms: int, seed: int = 0) -> List[Dict]:
    """Collect windowed QUBOs for one neighborhood across instances and a few
    seeded random permutations (for diversity)."""
    rng = np.random.default_rng(seed)
    qubos: List[Dict] = []
    for n, pt in instances:
        base_perms = [list(range(n))] + [list(rng.permutation(n)) for _ in range(max(0, perms - 1))]
        for pi in base_perms:
            for s, e in _windows(n, window, step):
                Q = _window_qubo(neighborhood, pi, pt, s, e)
                if Q:
                    qubos.append(Q)
    return qubos


def calibrate(instances, *, p_max: int, window: int, step: int, perms: int,
              n_gamma: int, n_beta: int, maxiter: int) -> Dict[str, Dict]:
    table: Dict[str, Dict] = {}
    for nb in NEIGHBORHOODS:
        training = build_training(nb, instances, window=window, step=step, perms=perms)
        res = optimize_neighborhood(training, p_max=p_max, n_gamma=n_gamma,
                                    n_beta=n_beta, maxiter=maxiter)
        # store as {str(p): {"gamma":[...], "beta":[...], "objective": float}}
        table[nb] = {str(p): res[p] for p in res}
        best = min(res, key=lambda p: res[p]["objective"])
        print(f"{nb:11s} trained on {len(training):3d} windows | "
              f"p=1 obj {res[1]['objective']:+.4f} ... p={p_max} obj {res[p_max]['objective']:+.4f}")
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-max", type=int, default=5)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--n-gamma", type=int, default=25)
    ap.add_argument("--n-beta", type=int, default=13)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--out", type=str, default="data/qaoa_angles.json")
    args = ap.parse_args()

    instances = _load_instances(DEFAULT_INSTANCES)
    print(f"calibrating on {len(instances)} Taillard instances, p=1..{args.p_max}")
    table = calibrate(instances, p_max=args.p_max, window=args.window, step=args.step,
                      perms=args.perms, n_gamma=args.n_gamma, n_beta=args.n_beta,
                      maxiter=args.maxiter)
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=1, sort_keys=True))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
