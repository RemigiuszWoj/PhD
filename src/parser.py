"""Parser for Taillard pair-format job shop instances."""

from models import DataInstance, Job


def parse_taillard_data(file_path: str) -> DataInstance:
    """Parse a Taillard instance file and return a DataInstance.

    Format:
        First non-empty line: J M
        Next J lines: 2*M integers alternating (machine processing_time).
        Machine indices may be 0- or 1-based; result is always 0-based.
    Raises:
        FileNotFoundError: file not present.
        ValueError: malformed structure or invalid numeric values.
    """
    lines: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    if not lines:
        raise ValueError("Empty file")

    headers = lines[0].split()
    if len(headers) < 2:
        raise ValueError("invalid header")
    try:
        jobs_number = int(headers[0])
        machines_number = int(headers[1])
    except ValueError as exc:
        raise ValueError("header must contain two integers") from exc

    if len(lines) - 1 < jobs_number:
        raise ValueError("Not enough job lines")

    jobs: list[list[Job]] = []
    expected_elements_number = 2 * machines_number
    for i in range(jobs_number):
        elements = lines[1 + i].split()
        if len(elements) != expected_elements_number:
            raise ValueError(
                f"Job line {i} has {len(elements)} ints != 2*{machines_number}"
            )
        values = list(map(int, elements))
        one_job: list[Job] = [
            (values[j], values[j + 1]) for j in range(0, len(values), 2)
        ]
        jobs.append(one_job)

        machines_numbers = [m for job in jobs for (m, _) in job]
        min_machine = min(machines_numbers)
        max_machine = max(machines_numbers)

    # Normalize 1-based indices if needed
    if min_machine == 1 and max_machine == machines_number:
        jobs = [[(m - 1, p) for (m, p) in job] for job in jobs]
        max_machine -= 1  # now max_machine should be machines_number - 1
        min_machine = 0

    if min_machine < 0 or max_machine >= machines_number:
        raise ValueError("Machine index out of range")

    for job in jobs:
        if len(job) != machines_number:
            raise ValueError("unexpected job length (non-uniform)")
        for m, p in job:
            if p <= 0:
                raise ValueError("non-positive processing time")

    return DataInstance(
        jobs=jobs,
        jobs_number=jobs_number,
        machines_number=machines_number,
    )
