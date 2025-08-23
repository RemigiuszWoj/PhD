import random
from typing import Iterable, Optional

from models import DataInstance, OperationKey, Schedule, ScheduleOperationRow
from operations import validate_permutation


def build_schedule_from_permutation(
    data_instance: DataInstance,
    permutation: Iterable[OperationKey],
    validate: bool = False,
    check_completeness: bool = False,
) -> Schedule:
    """Decode a permutation into a schedule (Serial SGS, non-delay).

    If validate=True perform full permutation validation first. If
    check_completeness=True ensure number of decoded operations == J*M.
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
    """Ensure no overlapping operations on the same machine.

    Sorts operations per machine by start time and checks start >= prev_end.
    Raises AssertionError on the first detected overlap.
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
    """Random permutation preserving intra-job order.

    Repeatedly selects uniformly a job with remaining operations.
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
    """SPT permutation: always pick job with shortest next operation.

    Ties broken by lower job id.
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
