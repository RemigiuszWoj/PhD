from typing import Dict, Iterator, List, Tuple

from src.permutation_procesing import c_max


def swap_jobs(pi: List[int], i: int, j: int) -> List[int]:
    """Return a new permutation with jobs at positions i and j swapped."""
    neighbor = pi.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def generate_neighbors_swap(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = swap_jobs(pi, i, i + 1)
            yield neighbor, (i, j)


def generate_neighbors_adjacent(pi: List[int]) -> Iterator[Tuple[List[int], Tuple[int, int]]]:
    """Generate neighbors by swapping any two jobs in pi (lazy generator)."""
    n = len(pi)
    for i in range(n - 1):
        neighbor = swap_jobs(pi, i, i + 1)
        yield neighbor, (i, i + 1)


def fibonahi_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[int], int]:
    """Build composite move of several non-overlapping adjacent swaps using DP."""

    n = len(pi)
    if n < 2:
        return pi, c_max(pi, processing_times)

    base_c = c_max(pi, processing_times)
    candidates: List[Tuple[int, int]] = []  # (i, delta)

    # wszystkie możliwe ruchy (i,i+1)
    for i in range(n - 1):
        newp = pi[:]
        newp[i], newp[i + 1] = newp[i + 1], newp[i]
        c_after = c_max(newp, processing_times)
        delta = c_after - base_c
        candidates.append((i, delta))

    if not candidates:
        return pi, base_c

    m = len(candidates)
    memo: Dict[int, Tuple[int, Tuple[int, ...]]] = {}

    # DP: wybór ruchów niekolidujących
    def solve(pos: int) -> Tuple[int, Tuple[int, ...]]:
        if pos >= m:
            return 0, ()
        if pos in memo:
            return memo[pos]

        idx, delta = candidates[pos]

        # pomijamy ten ruch
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set

        # bierzemy ten ruch → pomijamy wszystkie nachodzące (czyli pos+1 jeśli idx+1)
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

    # # jeśli nic nie wybrano, wybieramy najmniejszą deltę
    if not chosen:
        best_idx = min(candidates, key=lambda x: x[1])[0]
        chosen = (best_idx,)

    # wykonujemy wybrane ruchy
    new_pi = pi[:]
    for idx in sorted(chosen):
        new_pi[idx], new_pi[idx + 1] = new_pi[idx + 1], new_pi[idx]

    final_c = c_max(new_pi, processing_times)
    return new_pi, final_c


def naiv_dynasearch_neighborhood(
    pi: List[int],
    processing_times: List[List[int]],
) -> Tuple[List[int], int]:
    """Dynasearch neighborhood: best set of non-overlapping swaps (i,j), any distance."""

    n = len(pi)
    if n < 2:
        return pi, c_max(pi, processing_times)

    base_c = c_max(pi, processing_times)
    candidates: List[Tuple[int, int, int]] = []  # (i, j, delta)

    # all possible swaps (i, j), j > i
    for i in range(n - 1):
        for j in range(i + 1, n):
            newp = pi[:]
            newp[i], newp[j] = newp[j], newp[i]
            c_after = c_max(newp, processing_times)
            delta = c_after - base_c
            candidates.append((i, j, delta))

    if not candidates:
        return pi, base_c

    m = len(candidates)
    memo: Dict[int, Tuple[int, Tuple[Tuple[int, int], ...]]] = {}

    def solve(pos: int) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        """DP: pick non-overlapping moves with minimum total delta."""
        if pos >= m:
            return 0, ()
        if pos in memo:
            return memo[pos]

        i, j, delta = candidates[pos]

        # skip this move
        skip_val, skip_set = solve(pos + 1)
        best_val, best_set = skip_val, skip_set

        # take this move → skip all overlapping moves
        next_pos = pos + 1
        while next_pos < m:
            i2, j2, _ = candidates[next_pos]
            if not (j2 >= i and i2 <= j):  # overlap test
                break
            next_pos += 1

        take_rest_val, take_rest_set = solve(next_pos)
        take_val = delta + take_rest_val
        if take_val < best_val:
            best_val = take_val
            best_set = ((i, j),) + take_rest_set

        memo[pos] = (best_val, best_set)
        return memo[pos]

    total_delta, chosen = solve(0)

    if not chosen:
        # enforce at least one move
        best_move = min(candidates, key=lambda x: x[2])
        chosen = ((best_move[0], best_move[1]),)

    # apply moves
    new_pi = pi[:]
    for i, j in sorted(chosen):
        new_pi[i], new_pi[j] = new_pi[j], new_pi[i]

    final_c = c_max(new_pi, processing_times)
    return new_pi, final_c
