import random

from src.neighborhood import (
    generate_neighbors,
    insertion,
    swap_adjacent,
    swap_any,
)
from src.operations import create_base_permutation, validate_permutation
from src.parser import parse_taillard_data


def test_swap_adjacent_preserves_order() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    perm = create_base_permutation(inst)
    # swap first two positions (likely same job -> no change)
    newp = swap_adjacent(perm, 0)
    validate_permutation(inst, newp)


def test_swap_any_legal_and_illegal() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    perm = create_base_permutation(inst)
    # pick two positions from different jobs:
    # last op of job0 and first op of job1
    i = inst.machines_number - 1
    j = inst.machines_number
    moved = swap_any(perm, i, j)
    validate_permutation(inst, moved)


def test_insertion_and_multiset() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    perm = create_base_permutation(inst)
    newp = insertion(perm, 0, len(perm) - 1)
    validate_permutation(inst, newp)
    assert sorted(perm) == sorted(newp)


def test_generate_neighbors() -> None:
    inst = parse_taillard_data("data/JSPLIB/instances/ta01")
    perm = create_base_permutation(inst)
    neighs = generate_neighbors(perm, 25, rng=random.Random(0))
    assert 0 < len(neighs) <= 25
    # all unique and valid
    seen: set[tuple] = set()
    for p in neighs:
        validate_permutation(inst, p)
        t = tuple(p)
        assert t not in seen
        seen.add(t)
