"""Simple demo of parsing + permutation heuristics + decoding.

Adjust the path if needed (expects Taillard instance format).
Prints makespan (cmax) for base, random and SPT permutations.
"""

import random

from .decoder import (
    build_schedule_from_permutation,
    check_no_machine_overlap,
    create_random_permutation,
    create_spt_permutation,
)
from .operations import create_base_permutation, validate_permutation
from .parser import parse_taillard_data


def format_schedule(schedule):  # small helper for potential future detail
    return f"cmax={schedule.cmax} ops={len(schedule.operations)}"


def main() -> None:
    instance = parse_taillard_data("data/JSPLIB/instances/ta01")
    print(
        "Instance:",
        instance.jobs_number,
        "jobs x",
        instance.machines_number,
        "machines",
    )

    # Base permutation (job-major order)
    base_perm = create_base_permutation(instance)
    validate_permutation(instance, base_perm)
    base_schedule = build_schedule_from_permutation(instance, base_perm, check_completeness=True)
    check_no_machine_overlap(base_schedule)
    print("Base   ", format_schedule(base_schedule))

    # Random permutation
    rng = random.Random(42)
    rand_perm = create_random_permutation(instance, rng=rng)
    validate_permutation(instance, rand_perm)
    rand_schedule = build_schedule_from_permutation(instance, rand_perm, check_completeness=True)
    check_no_machine_overlap(rand_schedule)
    print("Random ", format_schedule(rand_schedule))

    # SPT permutation
    spt_perm = create_spt_permutation(instance)
    validate_permutation(instance, spt_perm)
    spt_schedule = build_schedule_from_permutation(instance, spt_perm, check_completeness=True)
    check_no_machine_overlap(spt_schedule)
    print("SPT    ", format_schedule(spt_schedule))

    # Quick ranking
    best = min(
        [
            ("base", base_schedule.cmax),
            ("random", rand_schedule.cmax),
            ("spt", spt_schedule.cmax),
        ],
        key=lambda x: x[1],
    )
    print("Best initial heuristic:", best)


if __name__ == "__main__":
    main()
