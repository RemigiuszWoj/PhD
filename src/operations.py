"""Permutation utilities (generation + validation).

Functions here create or validate operation permutations (list of pairs
(job_id, operation_index)) preserving intra‑job technological order.
"""

from models import DataInstance, OperationKey


def create_base_permutation(data_instance: DataInstance) -> list[OperationKey]:
    """Return canonical permutation: all ops of job 0, then job 1, etc.

    Useful as a simple baseline and for tests.
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
    """Validate completeness and per‑job order of a permutation.

    Conditions:
      - length == J * M (every operation appears exactly once),
      - for each job, operation indices appear 0,1,2,... without gaps.
    Returns True or raises ValueError with a diagnostic message.
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
