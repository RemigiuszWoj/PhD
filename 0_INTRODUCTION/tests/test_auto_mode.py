"""Tests for auto mode orchestration (run_auto).

Covers:

Creates a temporary charts output directory under pytest tmp_path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.models import DataInstance
from src.modes.auto import run_auto
from src.modes.common import AlgoParams
from src.parser import parse_taillard_data


def _load_small_instance() -> DataInstance:
    # Use a tiny Taillard instance (ta01) to keep runtime low
    return parse_taillard_data("tests/fixtures/ta01")


def test_run_auto_creates_results_and_returns_best(tmp_path: Path) -> None:
    instance = _load_small_instance()
    rng = random.Random(123)
    params = AlgoParams(
        neighbor_limit=10,
        max_no_improve=5,
        tabu_iterations=15,
        tabu_tenure=5,
        tabu_candidate_size=15,
        sa_iterations=30,
        sa_initial_temp=10.0,
        sa_cooling=0.9,
        sa_neighbor_moves=2,
    )
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()

    best_perm, best_c = run_auto(
        instance=instance,
        instance_path="tests/fixtures/ta01",
        runs=3,
        params=params,
        rng=rng,
        charts_dir=str(charts_dir),
    )
    assert best_perm is not None and best_c is not None

    # Find produced auto_results_*.json
    json_files = list(charts_dir.glob("auto_results_*.json"))
    assert json_files, "Expected auto_results JSON file"
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text())

    # Top-level keys
    for key in [
        "instance",
        "runs",
        "timestamp",
        "per_run",
        "best",
        "averages",
    ]:
        assert key in data
    assert data["runs"] == 3

    # Best block sanity
    best_block = data["best"]
    for algo in ("hill", "tabu", "sa"):
        assert algo in best_block
        entry = best_block[algo]
        # cmax may be None if algo produced no result (should not happen)
        if entry["cmax"] is not None:
            assert isinstance(entry["cmax"], int)
        # permutation representations are consistent in length
        pairs = entry["permutation_pairs"]
        compact = entry["permutation_compact"]
        if pairs is not None and compact is not None:
            # number of comma segments equals length of pairs
            assert len(compact.split(",")) == len(pairs)

    # Verify returned best matches one of the best algo entries
    all_best_cs = [
        best_block[a]["cmax"] for a in ("hill", "tabu", "sa") if best_block[a]["cmax"] is not None
    ]
    if all_best_cs:
        assert best_c in all_best_cs


def test_run_auto_determinism_seed(tmp_path: Path) -> None:
    instance = _load_small_instance()
    params = AlgoParams(
        neighbor_limit=5,
        max_no_improve=3,
        tabu_iterations=8,
        tabu_tenure=4,
        tabu_candidate_size=8,
        sa_iterations=12,
        sa_initial_temp=8.0,
        sa_cooling=0.85,
        sa_neighbor_moves=2,
    )
    charts_a = tmp_path / "charts_a"
    charts_b = tmp_path / "charts_b"
    charts_a.mkdir()
    charts_b.mkdir()

    seed = 999
    rng1 = random.Random(seed)
    rng2 = random.Random(seed)

    best_perm_a, best_c_a = run_auto(
        instance=instance,
        instance_path="tests/fixtures/ta01",
        runs=2,
        params=params,
        rng=rng1,
        charts_dir=str(charts_a),
    )
    best_perm_b, best_c_b = run_auto(
        instance=instance,
        instance_path="tests/fixtures/ta01",
        runs=2,
        params=params,
        rng=rng2,
        charts_dir=str(charts_b),
    )
    # Both best cmax values should match with identical seeds & params
    assert best_c_a == best_c_b
    # Permutations (sequence of tuples) should also match
    assert best_perm_a == best_perm_b
