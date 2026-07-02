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
    reset_qpu_stats         - zero the per-run QPU call counters
    get_qpu_stats           - snapshot the per-run QPU call counters
"""

from typing import Dict, List, Tuple


# Per-run QPU stats (module-level singleton; reset_qpu_stats() at start of each run).
# Fields:
#   calls               - total solve_qubo() invocations
#   qpu_success         - calls answered by real QPU (timing dict populated)
#   fallback_quota      - "insufficient remaining solver access time" → SA fallback
#   fallback_embedding  - "no embedding found" → SA fallback
#   fallback_other      - any other exception → SA fallback
#   simulator_calls     - backend != "dwave" (intentional SA, not a failure)
_qpu_stats: Dict[str, int] = {
    "calls": 0,
    "qpu_success": 0,
    "fallback_quota": 0,
    "fallback_embedding": 0,
    "fallback_other": 0,
    "simulator_calls": 0,
}


class QPUError(RuntimeError):
    """Raised when backend='dwave' and the QPU call fails.

    No silent fallback: a run that cannot be answered by the real QPU
    must FAIL LOUDLY, otherwise 'quantum' results silently become
    simulator results and the experiment is scientifically worthless.

    Attributes:
        category: 'quota' | 'embedding' | 'other'
    """

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def reset_qpu_stats() -> None:
    """Zero the per-run QPU counters. Call before each experiment run."""
    for k in _qpu_stats:
        _qpu_stats[k] = 0


def get_qpu_stats() -> Dict[str, int]:
    """Return a copy of the per-run QPU counters."""
    return dict(_qpu_stats)


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


def _log_qpu_timing(result, num_variables: int) -> None:
    """Extract QPU timing breakdown from D-Wave result and append to results/qpu_timing.jsonl."""
    import json, logging, os, time as _time
    timing = result.info.get("timing", {})
    if not timing:
        return

    solver_id = result.info.get("solver_id") or result.info.get("problem_id", "unknown")
    record = {
        "ts": _time.time(),
        "solver_id": solver_id,
        "num_variables": num_variables,
        "qpu_sampling_time_us":            timing.get("qpu_sampling_time"),
        "qpu_anneal_time_per_sample_us":   timing.get("qpu_anneal_time_per_sample"),
        "qpu_readout_time_per_sample_us":  timing.get("qpu_readout_time_per_sample"),
        "qpu_access_time_us":              timing.get("qpu_access_time"),
        "qpu_access_overhead_time_us":     timing.get("qpu_access_overhead_time"),
        "qpu_programming_time_us":         timing.get("qpu_programming_time"),
        "total_real_time_us":              timing.get("total_real_time"),
        "charge_time_us":                  timing.get("charge_time"),
        "post_processing_overhead_time_us": timing.get("post_processing_overhead_time"),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/qpu_timing.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")

    logging.info(
        "[QPU timing] vars=%d  access=%s µs  total=%s µs  overhead=%s µs  "
        "anneal=%s µs/read  readout=%s µs/read  programming=%s µs",
        num_variables,
        timing.get("qpu_access_time"),
        timing.get("total_real_time"),
        timing.get("qpu_access_overhead_time"),
        timing.get("qpu_anneal_time_per_sample"),
        timing.get("qpu_readout_time_per_sample"),
        timing.get("qpu_programming_time"),
    )


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

    backend='simulator': explicit classical mode (SimulatedAnnealingSampler).
    backend='dwave': real QPU only — any failure (quota exhausted, no
    embedding, network error) raises QPUError. There is NO silent fallback
    to the simulator: mixed-provenance results are scientifically worthless.
    Per-call disposition is recorded in module-level _qpu_stats.
    """
    if not Q:
        return {}

    from dimod import BinaryQuadraticModel
    bqm = BinaryQuadraticModel.from_qubo(Q)

    _qpu_stats["calls"] += 1

    # Intentional simulator backend — not a failure, just classical.
    if backend != "dwave":
        from dimod import SimulatedAnnealingSampler
        _qpu_stats["simulator_calls"] += 1
        result = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads)
        return dict(result.first.sample)

    # backend == "dwave"
    if not dwave_token:
        # Fall back to the DWAVE_API_TOKEN environment variable
        # (loaded from the gitignored .env file; see .env.example).
        import os
        dwave_token = os.environ.get("DWAVE_API_TOKEN")
    if not dwave_token:
        raise ValueError(
            "D-Wave token required for backend='dwave': set it in config.yaml "
            "or, preferably, in the DWAVE_API_TOKEN environment variable."
        )
    from dwave.system import DWaveSampler, EmbeddingComposite
    import logging

    sample_kwargs = {"num_reads": num_reads}
    if annealing_time_us is not None:
        sample_kwargs["annealing_time"] = annealing_time_us
    if chain_strength is not None:
        sample_kwargs["chain_strength"] = chain_strength
    if num_spin_reversal_transforms is not None:
        sample_kwargs["num_spin_reversal_transforms"] = num_spin_reversal_transforms

    def _classify_and_raise(exc: Exception) -> None:
        """Classify the QPU failure, bump the right counter, raise QPUError.

        NO fallback to the simulator — a failed QPU call must kill this run
        (the runner records it as FAILED and continues with the batch).
        """
        msg = str(exc).lower()
        if "insufficient remaining solver access time" in msg or ("insufficient" in msg and "access time" in msg):
            _qpu_stats["fallback_quota"] += 1
            category = "quota"
            logging.error("[solve_qubo] QPU quota exhausted (%d vars): %s", bqm.num_variables, exc)
        elif "no embedding found" in msg:
            _qpu_stats["fallback_embedding"] += 1
            category = "embedding"
            logging.error("[solve_qubo] No QPU embedding found (%d vars).", bqm.num_variables)
        else:
            _qpu_stats["fallback_other"] += 1
            category = "other"
            logging.error("[solve_qubo] QPU request failed: %s", exc)
        raise QPUError(category, f"QPU call failed ({category}, {bqm.num_variables} vars): {exc}") from exc

    # Submit to QPU (lazy SampleSet — failures may surface here OR during resolution).
    result = None
    submission_error: Exception | None = None
    try:
        ctx = DWaveSampler(token=dwave_token, solver=solver) if solver \
            else DWaveSampler(token=dwave_token)
        with ctx as dw_sampler:
            sampler = EmbeddingComposite(dw_sampler)
            result = sampler.sample(bqm, **sample_kwargs)
    except Exception as e:
        submission_error = e

    if submission_error is not None:
        _classify_and_raise(submission_error)

    # Resolve the SampleSet — this is where "insufficient solver access time" typically surfaces.
    try:
        sample = dict(result.first.sample)
    except Exception as e:
        _classify_and_raise(e)

    # Real QPU success: timing dict present iff response came from QPU.
    timing = (result.info or {}).get("timing")
    if timing:
        _qpu_stats["qpu_success"] += 1
        _log_qpu_timing(result, bqm.num_variables)
    else:
        # No timing → response did not come from a real QPU annealer.
        _qpu_stats["fallback_other"] += 1
        raise QPUError(
            "other",
            f"QPU response missing timing info ({bqm.num_variables} vars) — "
            "not answered by a real QPU annealer.",
        )
    return sample
