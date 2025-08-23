"""Core data structures for Taillard Job Shop instances.

This module defines:
    Job          -- alias describing a single operation (machine, proc_time).
    DataInstance -- immutable container with all jobs for one instance.
"""

from dataclasses import dataclass

Job = tuple[int, int]  # (machine, processing_time)


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
