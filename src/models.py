"""Core data structures for Taillard Job Shop instances.

This module defines:
    Job          -- alias describing a single operation (machine, proc_time).
    DataInstance -- immutable container with all jobs for one instance.
"""

from dataclasses import dataclass

Job = tuple[int, int]  # (machine, processing_time)
OperationKey = tuple[int, int]  # OperationKey = (job_id, operation_index)


@dataclass(frozen=True)
class DataInstance:
    """Immutable representation of a Taillard JSSP instance.

    Attributes:
        jobs: Nested list: jobs[j][k] -> (machine, processing_time).
        jobs_number: Number of jobs (J).
        machines_number: Number of machines (M).
    """

    jobs: list[list[Job]]
    jobs_number: int
    machines_number: int


@dataclass(frozen=True)
class ScheduleOperationRow:
    """Single scheduled operation with timing and identification data.

    Fields:
        start: Start time of the operation.
        end: Completion time (start + processing_time).
        job: Job identifier.
        operation_index: Index of the operation inside its job (0-based).
        machine: Machine on which the operation is processed.
        processing_time: Duration of the operation.
    """
    start: int
    end: int
    job: int
    operation_index: int
    machine: int
    processing_time: int


@dataclass(frozen=True)
class Schedule:
    """Full schedule plus objective value (cmax).

    Fields:
        operations: Flat list of all scheduled operations (length J*M).
        cmax: Makespan (maximum completion time across all operations).
    """
    operations: list[ScheduleOperationRow]
    cmax: int
