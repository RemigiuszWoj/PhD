"""Neighborhood (local move) operators for JSSP permutations.

Design choices
--------------
Feasibility preservation:
    Each move attempts to keep the technological (within‑job) order intact.
    When a requested move would violate this invariant the function returns a
    *copy* of the original permutation instead of raising. This makes it easy
    to generate candidate move lists without guarding every call. Callers that
    need to filter out no‑ops can compare object identity or content.

Return semantics:
    All operators return a new list (never mutate the input) so upstream code
    can safely cache/evaluate permutations.
"""

import random
from typing import Dict, List, Tuple, Optional, Union

from .evaluation import evaluate
from .models import OperationKey


def _job_order_ok(perm: list[OperationKey], job_id: int) -> bool:
    """Check whether a job's operations appear in canonical order ``0..k``.

    Args:
        perm: Permutation under test.
        job_id: Job whose subsequence to verify.

    Returns:
        True if the extracted sequence of operation indices equals
        ``list(range(len(seq)))``; False otherwise.
    """
    seq = [op_idx for (j, op_idx) in perm if j == job_id]
    return seq == list(range(len(seq)))


def swap_adjacent(perm: list[OperationKey], i: int) -> list[OperationKey]:
    """Swap consecutive positions ``i`` and ``i+1`` if jobs differ.

    Args:
        perm: Source permutation.
        i: Index of the left element in the swap (must satisfy ``0 <= i < n-1``).

    Returns:
        New permutation with the two elements swapped if they belong to
        different jobs; otherwise a shallow copy of the original permutation.

    Raises:
        IndexError: If ``i`` is out of range.
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
    """Swap two arbitrary positions while preserving feasibility.

    After performing the swap only the two touched jobs can have their order
    affected; only those are revalidated for efficiency.

    Args:
        perm: Source permutation.
        i: First index.
        j: Second index.

    Returns:
        New feasible permutation if the swap maintains job ordering, else a
        copy of the original permutation.

    Raises:
        IndexError: If either index is out of bounds.
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
    """Remove element at index ``i`` and insert before index ``j``.

    Args:
        perm: Source permutation.
        i: Index of element to remove.
        j: Target insertion index (interpreted after removal if ``i < j``).

    Returns:
        New feasible permutation if within‑job order remains valid; otherwise
        a copy of the original permutation (including cases with no effective
        change such as moving next to itself).

    Raises:
        IndexError: If either index is out of range (``j`` may equal ``n`` for
            appending semantics).
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
    """Stochastically generate up to ``limit`` distinct feasible neighbors.

    The generator randomly samples among three move types (adjacent swap,
    arbitrary swap, insertion). Infeasible attempts silently yield no new
    neighbor (because the move functions return the original permutation)
    and are filtered via a set of seen tuples.

    Args:
        perm: Base permutation to perturb.
        limit: Maximum number of distinct neighbors to return.
        rng: Optional random number generator (for reproducibility in tests);
            when absent the module-level ``random`` is used.

    Returns:
        List of up to ``limit`` feasible neighbor permutations (order not
        guaranteed). The original permutation is never included.
    """
    if rng is None:
        # Create dedicated RNG instance for reproducibility isolation
        rng = random.Random()
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


# Wersja DP create_fibonachi_neighborhood (stara placeholder usunięta)
def create_fibonachi_neighborhood(
    perm: list[OperationKey],
    data=None,
    cache: Optional[dict] = None,
    allow_equal: bool = False,
    debug: bool = False,
    return_cost: bool = False,
) -> Union[list[OperationKey], tuple[list[OperationKey], int]]:
    """Zbuduj wielokrotny ruch z kilku niekolidujących adjacent swapów.

    Mechanika:
    1. Wyznacz wszystkich dopuszczalnych kandydatów (zamiany (i,i+1) dwóch różnych jobów).
    2. Dla każdej zamiany oblicz delta = cmax_after - cmax_base.
        3. Rozwiąż problem wyboru podzbioru indeksów bez dwóch sąsiadujących
            minimalizujących sumę delt (DP).
        4. Jeśli łączna delta < 0 (lub allow_equal=True i delta==0) zastosuj
            wszystkie zamiany (nie kolidują).

    Parametry:
        perm: wejściowa permutacja.
        data: instancja danych (wymagana do evaluate).
        cache: współdzielony cache ocen.
        allow_equal: zezwól zwrócić ruch bez poprawy gdy delta==0.
    debug: dodatkowe printy.
    return_cost: jeżeli True zwróć (permutacja, cmax_nowego) zamiast samej permutacji.
    """
    if data is None:
        raise ValueError("create_fibonachi_neighborhood: wymagane 'data'")
    if cache is None:
        cache = {}

    n = len(perm)
    if n < 2:
        return (perm, evaluate(data, perm, cache=cache)) if return_cost else perm

    base_c = evaluate(data, perm, cache=cache)

    candidates: List[Tuple[int, int]] = []  # (i, delta)
    for i in range(n - 1):
        a, b = perm[i], perm[i + 1]
        if a[0] == b[0]:
            continue
        newp = perm[:]
        newp[i], newp[i + 1] = newp[i + 1], newp[i]
        c_after = evaluate(data, newp, cache=cache)
        delta = c_after - base_c
        candidates.append((i, delta))

    if not candidates:
        return (perm, base_c) if return_cost else perm

    m = len(candidates)
    memo: Dict[int, Tuple[int, Tuple[int, ...]]] = {}

    def solve(pos: int) -> Tuple[int, Tuple[int, ...]]:
        if pos >= m:
            return 0, ()
        if pos in memo:
            return memo[pos]
        idx, delta = candidates[pos]
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set
        next_pos = pos + 1
        while next_pos < m and candidates[next_pos][0] == idx + 1:
            next_pos += 1
        take_rest_val, take_rest_set = solve(next_pos)
        take_val = delta + take_rest_val
        if take_val < best_val:
            best_val = take_val
            best_set = (idx,) + take_rest_set
        memo[pos] = (best_val, best_set)
        return memo[pos]

    total_delta, chosen = solve(0)
    if debug:
        print(f"[fibo-neigh] base={base_c} best_delta={total_delta} chosen={chosen}")

    # Brak poprawy (lub brak wybranych swapów) -> zwrot oryginału bez kopiowania
    if (not allow_equal and total_delta >= 0) or not chosen:
        return (perm, base_c) if return_cost else perm

    new_perm = perm[:]
    for idx in sorted(chosen):
        new_perm[idx], new_perm[idx + 1] = new_perm[idx + 1], new_perm[idx]
    final_c = evaluate(data, new_perm, cache=cache)
    if debug:
        print(f"[fibo-neigh] final_c={final_c} improvement={base_c - final_c}")
    return (new_perm, final_c) if return_cost else new_perm
