"""First-ever validation of quantum_qubo_enhanced neighborhoods (simulator only).

Part 3 of the windowed_qubo restart plan (branch 2026-07-02-windowed-qubo-restart).

For every enhanced neighborhood and a range of instance sizes this script:
  1. calls the neighborhood ONCE on a shuffled permutation,
  2. asserts the result is a valid permutation (no lost/duplicated jobs),
  3. asserts the returned Cmax matches a from-scratch recomputation,
  4. applies the neighborhood REPEATEDLY (5 steps) to catch corruption
     that only shows up after chained windowed merges,
  5. reports wall time per call — input for the QPU budget estimate
     (hard limit: max 10% of the monthly D-Wave quota for all research).

Backend is hard-coded to 'simulator' — this script must never touch the QPU.

Usage:
    .venv311/bin/python3 scripts/validate_enhanced.py
"""

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.algorithms.base import get_neighbor  # noqa: E402
from src.parser import parser  # noqa: E402
from src.permutation_procesing import c_max  # noqa: E402

BACKEND = "simulator"  # NEVER change to 'dwave' in this script
NUM_READS = 5
CHAIN_STEPS = 5

# (neighborhood, instance file, n) — sizes per the article's feasibility bounds
PLAN = [
    ("quantum_adjacent_enhanced", "data/tai20_5.txt", 20),
    ("quantum_adjacent_enhanced", "data/tai50_5.txt", 50),
    ("quantum_adjacent_enhanced", "data/tai100_5.txt", 100),
    ("quantum_fibonacci_enhanced", "data/tai20_5.txt", 20),
    ("quantum_fibonacci_enhanced", "data/tai50_5.txt", 50),
    ("quantum_fibonacci_enhanced", "data/tai100_5.txt", 100),
    ("quantum_fibonacci_enhanced", "data/tai200_10.txt", 200),
    ("quantum_dynasearch_enhanced", "data/tai20_5.txt", 20),
    ("quantum_dynasearch_enhanced", "data/tai50_5.txt", 50),
    ("quantum_motzkin_enhanced", "data/tai20_5.txt", 20),
    ("quantum_motzkin_enhanced", "data/tai50_5.txt", 50),
]

QCONF = {"backend": BACKEND, "num_reads": NUM_READS, "num_reads_enhanced": NUM_READS}


def check_permutation(pi, n, label):
    if sorted(pi) != list(range(n)):
        raise AssertionError(f"{label}: INVALID PERMUTATION (lost/duplicated jobs)")


def main() -> int:
    failures = 0
    print(f"{'neighborhood':<30} {'n':>4} {'base_cmax':>9} {'new_cmax':>9} "
          f"{'delta':>7} {'call_ms':>8}  chain")
    for neigh, inst_file, n in PLAN:
        data = parser(inst_file, 0)
        pt = data["processing_times"]
        random.seed(42)
        pi = list(range(n))
        random.shuffle(pi)
        base = c_max(pi, pt)

        label = f"{neigh}@n={n}"
        try:
            t0 = time.time()
            new_pi, new_c, _, _ = get_neighbor(neigh, pi, pt, n, None, QCONF)
            call_ms = int((time.time() - t0) * 1000)

            check_permutation(new_pi, n, label)
            recomputed = c_max(new_pi, pt)
            if recomputed != new_c:
                raise AssertionError(
                    f"{label}: CMAX MISMATCH returned={new_c} recomputed={recomputed}"
                )

            # Chained application — corruption after repeated windowed merges
            cur = new_pi
            for step in range(CHAIN_STEPS):
                cur, cur_c, _, _ = get_neighbor(neigh, cur, pt, n, None, QCONF)
                check_permutation(cur, n, f"{label} chain-step {step + 1}")
                if c_max(cur, pt) != cur_c:
                    raise AssertionError(f"{label} chain-step {step + 1}: CMAX MISMATCH")

            print(f"{neigh:<30} {n:>4} {base:>9} {new_c:>9} "
                  f"{new_c - base:>7} {call_ms:>8}  OK({CHAIN_STEPS} steps, final={cur_c})")
        except AssertionError as e:
            failures += 1
            print(f"{neigh:<30} {n:>4}  *** FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"{neigh:<30} {n:>4}  *** ERROR: {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"VALIDATION FAILED: {failures} case(s).")
        return 1
    print("ALL ENHANCED NEIGHBORHOODS VALID (simulator).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
