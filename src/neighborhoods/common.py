"""Common functions for Flow Shop neighborhoods.

Shared by classical/, quantum_qubo/, and quantum_qubo_enhanced/.

Functions:
    swap_jobs               - swap two elements in a permutation
    compute_head            - forward completion time matrix
    compute_tail            - backward remaining time matrix
    compute_head_and_tail   - both matrices together
    compute_deltas          - Cmax deltas for all adjacent swaps (Head+Tail)
    compute_endpoint_swap_delta - delta for endpoint swap of segment [i..j]
    apply_swaps             - apply adjacent swaps to permutation
    validate_no_overlap     - remove overlapping adjacent swaps
    solve_qubo              - solve QUBO via D-Wave or SimulatedAnnealingSampler
"""

from typing import Dict, List, Tuple


def swap_jobs(pi: List[int], i: int, j: int) -> List[int]:
    neighbor = pi.copy()
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def compute_head(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """Head[i][j] = completion time of job at position j on machine i. O(m·n)"""
    m = len(processing_times)
    n = len(pi)
    Head = [[0] * n for _ in range(m)]
    Head[0][0] = processing_times[0][pi[0]]
    for j in range(1, n):
        Head[0][j] = Head[0][j - 1] + processing_times[0][pi[j]]
    for i in range(1, m):
        Head[i][0] = Head[i - 1][0] + processing_times[i][pi[0]]
    for i in range(1, m):
        for j in range(1, n):
            Head[i][j] = max(Head[i - 1][j], Head[i][j - 1]) + processing_times[i][pi[j]]
    return Head


def compute_tail(pi: List[int], processing_times: List[List[int]]) -> List[List[int]]:
    """Tail[i][j] = remaining time from position j on machine i to end. O(m·n)"""
    m = len(processing_times)
    n = len(pi)
    Tail = [[0] * n for _ in range(m)]
    Tail[m - 1][n - 1] = processing_times[m - 1][pi[n - 1]]
    for i in range(m - 2, -1, -1):
        Tail[i][n - 1] = Tail[i + 1][n - 1] + processing_times[i][pi[n - 1]]
    for j in range(n - 2, -1, -1):
        Tail[m - 1][j] = Tail[m - 1][j + 1] + processing_times[m - 1][pi[j]]
    for j in range(n - 2, -1, -1):
        for i in range(m - 2, -1, -1):
            Tail[i][j] = max(Tail[i + 1][j], Tail[i][j + 1]) + processing_times[i][pi[j]]
    return Tail


def compute_head_and_tail(
    pi: List[int], processing_times: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    return compute_head(pi, processing_times), compute_tail(pi, processing_times)


def compute_deltas(pi: List[int], processing_times: List[List[int]]) -> List[float]:
    """Cmax delta for each adjacent swap (i, i+1). Uses Head+Tail → O(m·n)."""
    n = len(pi)
    if n < 2:
        return []
    m = len(processing_times)
    Head = compute_head(pi, processing_times)
    Tail = compute_tail(pi, processing_times)
    base_cmax = Head[m - 1][n - 1]
    deltas: List[float] = []
    for j in range(n - 1):
        job_a, job_b = pi[j], pi[j + 1]
        C_j = [0] * m
        C_j1 = [0] * m
        for i in range(m):
            left = Head[i][j - 1] if j > 0 else 0
            top = C_j[i - 1] if i > 0 else 0
            C_j[i] = max(top, left) + processing_times[i][job_b]
            top_j1 = C_j1[i - 1] if i > 0 else 0
            C_j1[i] = max(top_j1, C_j[i]) + processing_times[i][job_a]
        if j + 2 < n:
            new_cmax = max(C_j1[i] + Tail[i][j + 2] for i in range(m))
        else:
            new_cmax = C_j1[m - 1]
        deltas.append(new_cmax - base_cmax)
    return deltas


def compute_endpoint_swap_delta(
    pi: List[int],
    i: int,
    j: int,
    Head: List[List[int]],
    Tail: List[List[int]],
    processing_times: List[List[int]],
    base_cmax: int,
) -> int:
    """Delta for swapping endpoints of segment [i..j]. O(m·(j-i))."""
    m = len(processing_times)
    n = len(pi)
    col_prev = [Head[r][i - 1] for r in range(m)] if i > 0 else [0] * m
    col = [0] * m

    job = pi[j]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
    col_prev, col = col, col_prev

    for t in range(i + 1, j):
        job = pi[t]
        col[0] = col_prev[0] + processing_times[0][job]
        for r in range(1, m):
            col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
        col_prev, col = col, col_prev

    job = pi[i]
    col[0] = col_prev[0] + processing_times[0][job]
    for r in range(1, m):
        col[r] = max(col[r - 1], col_prev[r]) + processing_times[r][job]
    col_prev, col = col, col_prev

    if j + 1 < n:
        new_cmax = max(col_prev[r] + Tail[r][j + 1] for r in range(m))
    else:
        new_cmax = col_prev[m - 1]
    return new_cmax - base_cmax


def apply_swaps(pi: List[int], indices: List[int]) -> List[int]:
    """Apply adjacent swaps (each index i means swap positions i and i+1)."""
    new_pi = pi.copy()
    for idx in sorted(indices):
        new_pi[idx], new_pi[idx + 1] = new_pi[idx + 1], new_pi[idx]
    return new_pi


def validate_no_overlap(indices: List[int]) -> List[int]:
    """Remove overlapping adjacent swaps (keep non-overlapping)."""
    valid: List[int] = []
    last_idx = -2
    for idx in sorted(indices):
        if idx > last_idx + 1:
            valid.append(idx)
            last_idx = idx
    return valid


def solve_qubo(
    Q: Dict[Tuple[str, str], float],
    num_reads: int = 5,
    backend: str = "simulator",
    dwave_token: str | None = None,
    solver: str | None = None,
    annealing_time_us: int | None = None,
    chain_strength: float | None = None,
    num_spin_reversal_transforms: int | None = None,
) -> Dict[str, int]:
    """Solve QUBO via SimulatedAnnealingSampler or real D-Wave QPU.

    Falls back to SimulatedAnnealingSampler if no QPU embedding is found.
    """
    if not Q:
        return {}

    from dimod import BinaryQuadraticModel
    bqm = BinaryQuadraticModel.from_qubo(Q)

    if backend == "dwave":
        if not dwave_token:
            raise ValueError("dwave_token required when backend='dwave'")
        from dwave.system import DWaveSampler, EmbeddingComposite
        import logging

        sample_kwargs = {"num_reads": num_reads}
        if annealing_time_us is not None:
            sample_kwargs["annealing_time"] = annealing_time_us
        if chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength
        if num_spin_reversal_transforms is not None:
            sample_kwargs["num_spin_reversal_transforms"] = num_spin_reversal_transforms

        try:
            ctx = DWaveSampler(token=dwave_token, solver=solver) if solver \
                else DWaveSampler(token=dwave_token)
            with ctx as dw_sampler:
                sampler = EmbeddingComposite(dw_sampler)
                result = sampler.sample(bqm, **sample_kwargs)
        except ValueError as e:
            if "no embedding found" in str(e):
                logging.warning(
                    "[solve_qubo] No QPU embedding found (%d vars). "
                    "Falling back to SimulatedAnnealingSampler.", bqm.num_variables
                )
                from dimod import SimulatedAnnealingSampler
                result = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
            else:
                raise
    else:
        from dimod import SimulatedAnnealingSampler
        result = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)

    return dict(result.first.sample)
