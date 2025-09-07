"""Funkcje oceny (evaluate) wydzielone z search.py aby uniknąć importów cyklicznych.

Zawiera jedną funkcję evaluate do liczenia cmax (z opcją zwrotu harmonogramu)
+ prosty cache.
"""

from __future__ import annotations

from .decoder import build_schedule_from_permutation
from .models import DataInstance, OperationKey, Schedule


def evaluate(
    data: DataInstance,
    permutation: list[OperationKey],
    cache: dict | None = None,
    return_schedule: bool = False,
    validate: bool = False,
) -> int | tuple[int, Schedule]:
    key = tuple(permutation)
    if cache is not None and key in cache:
        cmax, sched = cache[key]
        if return_schedule:
            return cmax, sched
        return cmax
    sched = build_schedule_from_permutation(
        data,
        key,
        validate=validate,
        check_completeness=True,
    )
    if cache is not None:
        cache[key] = (sched.cmax, sched)
    if return_schedule:
        return sched.cmax, sched
    return sched.cmax
