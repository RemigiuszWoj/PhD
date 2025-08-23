"""Pytest tests for `parse_taillard_data`.

Each test creates a temporary Taillard-format instance file and asserts either
successful parsing (structure + normalization) or the correct exception.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from parser import parse_taillard_data

import pytest


@contextmanager
def temp_instance(content: str):
    fd, path = tempfile.mkstemp(text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:  # pragma: no cover
            pass


def test_parse_simple_zero_based():
    with temp_instance("""2 2\n0 5 1 3\n1 4 0 2\n""") as path:
        inst = parse_taillard_data(path)
        assert inst.jobs_number == 2
        assert inst.machines_number == 2
        assert len(inst.jobs) == 2
        for job in inst.jobs:
            assert len(job) == 2
        assert {m for job in inst.jobs for (m, _) in job} == {0, 1}


def test_parse_simple_one_based_normalization():
    with temp_instance("""1 3\n1 10 2 5 3 7\n""") as path:
        inst = parse_taillard_data(path)
        assert inst.jobs_number == 1
        assert inst.machines_number == 3
        machines = [m for (m, _) in inst.jobs[0]]
        assert machines == [0, 1, 2]


@pytest.mark.parametrize(
    "content",
    [
        """2\n0 5 1 3\n""",  # invalid header (only one int)
        # insufficient job lines (declares 2 jobs, provides 1)
        """2 1\n0 5\n""",
        """1 2\n0 5 1\n""",  # invalid token count (3 instead of 4)
        """1 1\n0 0\n""",  # non-positive processing time
        """1 2\n5 3 1 2\n""",  # machine index out of range
    ],
)
def test_parse_errors(content: str):
    with temp_instance(content) as path:
        with pytest.raises(ValueError):
            parse_taillard_data(path)
