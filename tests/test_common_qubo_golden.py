"""Golden characterization test for the quantum neighborhood QUBO families.

Guards the behaviour-preserving refactor onto ``common_qubo``. For a fixed set
of Taillard instances it deterministically captures, for each neighborhood:
  * every QUBO matrix passed to ``solve_qubo`` (the build side), and
  * the returned ``(new_pi, new_cmax, moves)`` (the map-back side),
by replacing ``solve_qubo`` with a spy that logs ``Q`` and returns a fixed,
Q-only-dependent solution. The capture is asserted against a committed
reference (``tests/golden_neighborhoods.json``). Instance processing times
are embedded in ``tests/golden_instances.json`` so the test does not depend on
the gitignored ``data/`` directory. Any change to QUBO construction or solution
map-back — in ``quantum_qubo``, ``quantum_qubo_enhanced`` or the shared
``common_qubo`` — will fail this test.

To regenerate the reference after an *intended* change:
    python -m tests.test_common_qubo_golden   # writes the JSON, prints a summary
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REFERENCE_PATH = REPO / "tests" / "golden_neighborhoods.json"
INSTANCES_PATH = REPO / "tests" / "golden_instances.json"
_INSTANCES = json.loads(INSTANCES_PATH.read_text())

# (label, module path, function name)
NEIGH = [
    ("qq_adjacent",    "src.neighborhoods.quantum_qubo.adjacent",            "quantum_adjacent_neighborhood"),
    ("qq_fibonacci",   "src.neighborhoods.quantum_qubo.fibonacci",           "quantum_fibonacci_neighborhood"),
    ("qq_dynasearch",  "src.neighborhoods.quantum_qubo.dynasearch",          "quantum_dynasearch_neighborhood"),
    ("qq_motzkin",     "src.neighborhoods.quantum_qubo.motzkin",             "quantum_motzkin_neighborhood"),
    ("enh_adjacent",   "src.neighborhoods.quantum_qubo_enhanced.adjacent",   "quantum_adjacent_enhanced"),
    ("enh_fibonacci",  "src.neighborhoods.quantum_qubo_enhanced.fibonacci",  "quantum_fibonacci_enhanced"),
    ("enh_dynasearch", "src.neighborhoods.quantum_qubo_enhanced.dynasearch", "quantum_dynasearch_enhanced"),
    ("enh_motzkin",    "src.neighborhoods.quantum_qubo_enhanced.motzkin",    "quantum_motzkin_enhanced"),
]
# instance names; the processing_times live in golden_instances.json so the
# test does not depend on the gitignored data/ directory
INSTANCE_NAMES = ["tai20_5#0", "tai20_5#1", "tai20_10#0"]


def _qkey(Q):
    return sorted([f"{i}|{j}={round(float(v), 9)}" for (i, j), v in Q.items()])


def _det_solution(Q):
    """Deterministic solution: pick every variable with a negative diagonal."""
    variables = sorted({v for pair in Q for v in pair})
    return {v: (1 if Q.get((v, v), 0.0) < 0 else 0) for v in variables}


def _make_spy(record):
    def spy(Q, *args, **kwargs):
        record.append(_qkey(Q))
        return _det_solution(Q)
    return spy


def _norm(x):
    return json.loads(json.dumps(x, default=list))


def _run_case(modpath, fname, iname):
    func = getattr(importlib.import_module(modpath), fname)
    pt = _INSTANCES[iname]["processing_times"]
    n = _INSTANCES[iname]["n"]
    pi = list(range(n))
    rec: list = []
    spy = _make_spy(rec)
    saved = {}
    patch_targets = [importlib.import_module(m) for _, m, _ in NEIGH]
    for mod in patch_targets:
        if hasattr(mod, "solve_qubo"):
            saved[mod] = mod.solve_qubo
            mod.solve_qubo = spy
    try:
        new_pi, new_c, moves = func(pi, pt)
    finally:
        for mod, orig in saved.items():
            mod.solve_qubo = orig
    return {
        "n_qubo_calls": len(rec),
        "qubos": rec,
        "new_pi": list(new_pi),
        "new_cmax": int(new_c),
        "moves": _norm(moves),
    }


def _snapshot():
    out = {}
    for label, modpath, fname in NEIGH:
        for iname in INSTANCE_NAMES:
            out[f"{label}::{iname}"] = _run_case(modpath, fname, iname)
    return out


_CASES = {
    f"{label}::{iname}": (modpath, fname, iname)
    for label, modpath, fname in NEIGH
    for iname in INSTANCE_NAMES
}


@pytest.mark.parametrize("case_key", sorted(_CASES))
def test_neighborhood_matches_golden(case_key):
    reference = json.loads(REFERENCE_PATH.read_text())
    modpath, fname, iname = _CASES[case_key]
    result = _run_case(modpath, fname, iname)
    assert result == reference[case_key], (
        f"{case_key}: QUBO or map-back changed vs golden reference"
    )


if __name__ == "__main__":  # regenerate the reference after an intended change
    snap = _snapshot()
    REFERENCE_PATH.write_text(json.dumps(snap, indent=1, sort_keys=True))
    print(f"wrote {REFERENCE_PATH} ({len(snap)} cases)")
