"""Neighborhood (local move) operators for permutations.

All moves preserve feasibility (job operation order). Illegal moves silently
return a copy of the original permutation (design choice for easy filtering).
"""

import random
from typing import Optional

from models import OperationKey


def _job_order_ok(perm: list[OperationKey], job_id: int) -> bool:
    """Return True if a job's operations appear in canonical order 0..k.

    Extracts op_index sequence for the given job and compares to range(len).
    """
    seq = [op_idx for (j, op_idx) in perm if j == job_id]
    return seq == list(range(len(seq)))


def swap_adjacent(perm: list[OperationKey], i: int) -> list[OperationKey]:
    """Swap consecutive positions i and i+1 if different jobs.

    If they belong to the same job the within-job order would break, so the
    original permutation (copy) is returned.
    """
    if i < 0 or i >= len(perm) - 1:
        raise IndexError("i out of range")
    a, b = perm[i], perm[i + 1]
    if a[0] == b[0]:
        return perm[:]  # would violate within‑job order
    newp = perm[:]
    newp[i], newp[i + 1] = newp[i + 1], newp[i]
    return newp


def swap_any(perm: list[OperationKey], i: int, j: int) -> list[OperationKey]:
    """Swap two arbitrary positions while preserving each job's order.

    After swapping only the two involved jobs can have altered order so only
    they are checked.
    """
    if not (0 <= i < len(perm) and 0 <= j < len(perm)):
        raise IndexError("index out of range")
    if i == j:
        return perm[:]
    if i > j:
        i, j = j, i
    newp = perm[:]
    newp[i], newp[j] = newp[j], newp[i]
    touched = {perm[i][0], perm[j][0]}
    for job in touched:
        if not _job_order_ok(newp, job):
            return perm[:]  # illegal move
    return newp


def insertion(perm: list[OperationKey], i: int, j: int) -> list[OperationKey]:
    """Remove element at index i and insert before index j.

    If the job's internal order breaks the original permutation is returned.
    """
    n = len(perm)
    if not (0 <= i < n and 0 <= j <= n):
        raise IndexError("index out of range")
    if i == j or i + 1 == j:
        return perm[:]  # no effective change
    elem = perm[i]
    newp = perm[:]
    del newp[i]
    if i < j:
        j -= 1
    newp.insert(j, elem)
    if not _job_order_ok(newp, elem[0]):
        return perm[:]
    return newp


def generate_neighbors(
    perm: list[OperationKey],
    limit: int,
    rng: Optional[random.Random] = None,
) -> list[list[OperationKey]]:
    """Generate up to `limit` distinct feasible permutations.

    Stops early if not enough unique moves can be produced.
    """
    if rng is None:
        rng = random
    n = len(perm)
    result: list[list[OperationKey]] = []
    seen: set[tuple[OperationKey, ...]] = {tuple(perm)}
    attempts = 0
    while len(result) < limit and attempts < limit * 10:
        attempts += 1
        move = rng.choice(("adj", "swap", "ins"))
        if move == "adj":
            idx = rng.randrange(0, n - 1)
            cand = swap_adjacent(perm, idx)
        elif move == "swap":
            a, b = rng.sample(range(n), 2)
            cand = swap_any(perm, a, b)
        else:
            a, b = rng.sample(range(n), 2)
            cand = insertion(perm, a, b)
        t = tuple(cand)
        if t not in seen:
            seen.add(t)
            result.append(cand)
    return result
