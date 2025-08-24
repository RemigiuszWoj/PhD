"""Permutation utilities: creation and validation of permutations.

Concepts
--------
Permutation
    A list of ``(job_id, operation_index)`` tuples (``OperationKey``) that
    respects each job's internal technological order (operation indices of a
    given job appear in strictly increasing sequence without gaps). The
    decoder relies on this invariant; functions here help produce and verify
    such permutations.
"""

from src.models import DataInstance, OperationKey


def create_base_permutation(data_instance: DataInstance) -> list[OperationKey]:
    """Create the canonical job‑major permutation.

    Args:
        data_instance: Parsed problem instance.

    Returns:
        List with operations ordered by ascending ``job_id`` and, within each
        job, by ascending operation index (0..m-1). This is a trivial feasible
        baseline useful for correctness tests and as a deterministic start
        solution.
    """
    permutation: list[OperationKey] = []
    for i in range(data_instance.jobs_number):
        for j in range(len(data_instance.jobs[i])):
            permutation.append((i, j))
    return permutation


def validate_permutation(
    data_instance: DataInstance,
    permutation: list[OperationKey],
) -> bool:
    """Validate a permutation's completeness and per‑job order.

    Args:
        data_instance: Problem instance supplying numbers of jobs/machines and
            each job's operation list.
        permutation: Candidate permutation to check.

    Returns:
        True if the permutation is valid. (Return value mostly for convenience
        so the function can be used inside assertions / conditional flows.)

    Raises:
        ValueError: If length is incorrect, a job id is out of range, or the
            sequence of operation indices for any job deviates from the
            canonical increasing sequence (e.g. repeats, skips, or reorders).
    """
    expected_permutation_length = data_instance.machines_number * data_instance.jobs_number
    if len(permutation) != expected_permutation_length:
        raise ValueError("Incomplete permutation (missing operations)")
    next_operational_index = [0] * data_instance.jobs_number
    for job_id, operation_index in permutation:
        if not (0 <= job_id < data_instance.jobs_number):
            raise ValueError(f"Job index out of range: {job_id}")
        if operation_index != next_operational_index[job_id]:
            raise ValueError("Operation index out of order for job " f"{job_id}: {operation_index}")
        next_operational_index[job_id] += 1
    return True
