"""Core package for PhD JSSP experiments.

Exports base data structures and parsing utilities.
"""

from .models import DataInstance, Job  # noqa: F401
from .parser import parse_taillard_data  # type: ignore  # noqa: F401

__all__ = [
    "DataInstance",
    "Job",
    "parse_taillard_data",
]
