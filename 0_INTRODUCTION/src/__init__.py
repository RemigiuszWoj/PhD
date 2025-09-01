"""Core package for PhD JSSP experiments.

Exports base data structures and parsing utilities.
"""

from src.models import DataInstance, Job  # noqa: F401
from src.parser import parse_taillard_data  # noqa: F401

__all__ = [
    "DataInstance",
    "Job",
    "parse_taillard_data",
]
