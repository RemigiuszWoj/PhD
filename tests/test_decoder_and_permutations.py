import random

import pytest

from src.decoder import (
    build_schedule_from_permutation,
    check_no_machine_overlap,
    create_random_permutation,
    create_spt_permutation,
)
from src.models import DataInstance
from src.operations import create_base_permutation, validate_permutation


def small_instance() -> DataInstance:
    """Create a tiny 2x2 synthetic instance for focused unit tests."""
    # 2 jobs x 2 machines synthetic
    jobs = [
        [(0, 3), (1, 2)],
        [(1, 2), (0, 1)],
    ]
    return DataInstance(jobs=jobs, jobs_number=2, machines_number=2)


def test_base_decode_and_overlap():
    data = small_instance()
    perm = create_base_permutation(data)
    validate_permutation(data, perm)
    sched = build_schedule_from_permutation(data, perm, check_completeness=True)
    assert sched.cmax > 0
    assert check_no_machine_overlap(sched)


def test_random_and_spt_permutations_valid_and_different():
    data = small_instance()
    rng = random.Random(123)
    rand_perm = create_random_permutation(data, rng=rng)
    spt_perm = create_spt_permutation(data)
    validate_permutation(data, rand_perm)
    validate_permutation(data, spt_perm)
    assert rand_perm != spt_perm  # highly likely different for this toy case


def test_incomplete_permutation_raises():
    data = small_instance()
    # Missing last operation of job 1
    incomplete = [(0, 0), (0, 1), (1, 0)]
    with pytest.raises(ValueError):
        validate_permutation(data, incomplete)


def test_precedence_violation_in_decoder():
    data = small_instance()
    # reversed operations of job 0
    bad = [(0, 1), (0, 0), (1, 0), (1, 1)]
    with pytest.raises(ValueError):
        build_schedule_from_permutation(data, bad)
