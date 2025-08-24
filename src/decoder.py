import random
from typing import Iterable, Optional

from .models import DataInstance, OperationKey, Schedule, ScheduleOperationRow
from .operations import validate_permutation


def build_schedule_from_permutation(
    data_instance: DataInstance,
    permutation: Iterable[OperationKey],
    validate: bool = False,
    check_completeness: bool = False,
) -> Schedule:
    """Decode a permutation into a schedule (Serial SGS, non-delay).

    The Serial Schedule Generation Scheme (non-delay variant) places each
    operation as early as possible obeying two constraints: (1) job
    precedence (operation j,k starts after j,k-1 finishes) and (2) machine
    capacity (one operation at a time per machine). Order of consideration
    is exactly the given permutation.

    Args:
        data_instance: Problem data (jobs with (machine, proc_time) tuples).
        permutation: Sequence of (job_id, op_index) specifying insertion
            order for SGS.
        validate: When True perform full permutation validation (length
            and per-job order) before decoding.
        check_completeness: When True verify number of decoded operations
            equals jobs_number * machines_number; raise if not.

    Returns:
        Schedule: Object containing list of scheduled operations with start
        and end times plus computed makespan (cmax).

    Raises:
        ValueError: If job id out of range, precedence violated, duplicate
            operation encountered, or schedule incomplete when
            check_completeness=True.
    """
    jobs_number = data_instance.jobs_number
    machines_number = data_instance.machines_number
    ready_job = [0] * jobs_number
    ready_machine = [0] * machines_number  # completion time per machine
    # next expected operation index per job
    next_operational_index = [0] * jobs_number
    operations: list[ScheduleOperationRow] = []

    if validate:
        validate_permutation(data_instance, list(permutation))

    count = 0
    for i, j in permutation:
        if not (0 <= i < jobs_number):
            raise ValueError("job id out of range")
        if j != next_operational_index[i]:
            raise ValueError("precedence violation or duplicate")
        machine, processing_time = data_instance.jobs[i][j]
        start = max(ready_job[i], ready_machine[machine])
        end = start + processing_time
        ready_job[i] = end
        ready_machine[machine] = end
        next_operational_index[i] += 1
        operations.append(
            ScheduleOperationRow(
                start=start,
                end=end,
                job=i,
                operation_index=j,
                machine=machine,
                processing_time=processing_time,
            )
        )
        count += 1

    if check_completeness:
        expected_elements = jobs_number * machines_number
        if count != expected_elements:
            raise ValueError(
                "Incomplete schedule: " f"{count}/{expected_elements} operations decoded"
            )

    cmax = max((row.end for row in operations), default=0)
    return Schedule(operations=operations, cmax=cmax)


def check_no_machine_overlap(schedule: Schedule) -> bool:
    """Ensure no two operations overlap on the same machine.

    Iterates operations grouped by machine, ordered by start, verifying
    that each starts no earlier than the previous one ended.

    Args:
        schedule: Schedule returned by the decoder.

    Returns:
        True if no overlaps are found.

    Raises:
        AssertionError: On the first detected temporal overlap for a
        machine.
    """
    by_machine: dict[int, list[ScheduleOperationRow]] = {}
    for op in schedule.operations:
        by_machine.setdefault(op.machine, []).append(op)
    for machine_ops in by_machine.values():
        machine_ops.sort(key=lambda r: r.start)
        prev_end = -1
        for r in machine_ops:
            if r.start < prev_end:
                raise AssertionError(
                    "Overlap on machine " f"{r.machine} between end {prev_end} and start {r.start}"
                )
            prev_end = r.end
    return True


def create_random_permutation(
    data_instance: DataInstance,
    *,
    rng: Optional[random.Random] = None,
) -> list[OperationKey]:
    """Generate a random feasible permutation.

    At each step uniformly chooses among jobs with remaining operations and
    appends that job's next operation, preserving intra-job order.

    Args:
        data_instance: Problem data.
        rng: Optional random.Random instance (for reproducibility). If
            None uses module-level random.

    Returns:
        List of (job_id, op_index) forming a feasible permutation.
    """
    if rng is None:
        rng = random
    remaining = [len(job_ops) for job_ops in data_instance.jobs]
    next_idx = [0] * data_instance.jobs_number
    eligible = [j for j, r in enumerate(remaining) if r > 0]
    perm: list[OperationKey] = []
    while eligible:
        job = rng.choice(eligible)
        op = next_idx[job]
        perm.append((job, op))
        next_idx[job] += 1
        remaining[job] -= 1
        if remaining[job] == 0:
            eligible.remove(job)
    return perm


def create_spt_permutation(data_instance: DataInstance) -> list[OperationKey]:
    """Construct a Shortest Processing Time (SPT) permutation.

    Repeatedly selects the job whose *next* operation has minimal processing
    time (ties broken by lower job id), then appends that operation. This
    yields a static permutation heuristic (not a dynamic dispatch rule).

    Args:
        data_instance: Problem data.

    Returns:
        List of (job_id, op_index) chosen under SPT criterion.
    """
    remaining = [len(job_ops) for job_ops in data_instance.jobs]
    next_idx = [0] * data_instance.jobs_number
    perm: list[OperationKey] = []
    while True:
        candidates: list[tuple[int, int]] = []
        for j, rem in enumerate(remaining):
            if rem > 0:
                _, p = data_instance.jobs[j][next_idx[j]]
                candidates.append((p, j))
        if not candidates:
            break
        _, chosen = min(candidates)  # tie-break przez job id
        op = next_idx[chosen]
        perm.append((chosen, op))
        next_idx[chosen] += 1
        remaining[chosen] -= 1
    return perm
