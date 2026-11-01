"""Job shop instances: JSPLIB / Taillard formats, Taillard's generator, bounds.

Operation id o = j*m + t (job j, t-th operation of its route); mach[o], dur[o].
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "taillard_jssp.json"
JSPLIB_DIR = Path("data") / "JSPLIB" / "instances"

TA_CLASSES = {  # Taillard job shop classes, 10 instances each
    "15x15": range(1, 11), "20x15": range(11, 21), "20x20": range(21, 31), "30x15": range(31, 41),
    "30x20": range(41, 51), "50x15": range(51, 61), "50x20": range(61, 71), "100x20": range(71, 81),
}


class Instance:
    def __init__(self, name: str, n: int, m: int, route: list[list[int]], p: list[list[int]]):
        self.name, self.n, self.m = name, n, m
        self.route, self.p = route, p
        self.N = n * m
        self.mach = [route[o // m][o % m] for o in range(self.N)]
        self.dur = [p[o // m][o % m] for o in range(self.N)]
        loads = [0] * m
        for o in range(self.N):
            loads[self.mach[o]] += self.dur[o]
        self.lb = max(max(sum(row) for row in p), max(loads))  # job length / machine load bound


def read_jsplib(path: str | os.PathLike, name: str | None = None) -> Instance:
    toks: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.strip().startswith("#"):
                continue
            toks.extend(line.split())
    n, m = int(toks[0]), int(toks[1])
    vals = list(map(int, toks[2 : 2 + 2 * n * m]))
    route = [[vals[2 * (j * m + t)] for t in range(m)] for j in range(n)]
    p = [[vals[2 * (j * m + t) + 1] for t in range(m)] for j in range(n)]
    return Instance(name or Path(path).name, n, m, route, p)


def load(name: str, jsplib_dir: str | os.PathLike = JSPLIB_DIR) -> Instance:
    return read_jsplib(Path(jsplib_dir) / name, name)


def from_pfsp(processing_times: list[list[int]], name: str) -> Instance:
    """PFSP instance (m x n matrix) as a job shop with identical routes 0..m-1 (F||Cmax)."""
    m, n = len(processing_times), len(processing_times[0])
    return Instance(name, n, m, [list(range(m)) for _ in range(n)],
                    [[processing_times[t][j] for t in range(m)] for j in range(n)])


# ---------------------------------------------------------------------------
# Taillard (1993) generator
# ---------------------------------------------------------------------------


def _lehmer(seed: int) -> int:
    a, b, c, M = 16807, 127773, 2836, 2**31 - 1
    k = seed // b
    seed = a * (seed % b) - k * c
    return seed + M if seed < 0 else seed


def _unif(seed: int, low: int, high: int) -> tuple[int, int]:
    seed = _lehmer(seed)
    return seed, low + int(seed / (2**31 - 1) * (high - low + 1))


def taillard_generate(time_seed: int, machine_seed: int, n: int, m: int, rho: float = 1.0,
                      name: str = "") -> Instance:
    """Taillard's job shop generator (exact for rho = 1: reproduces ta01-ta80).

    rho < 1 keeps each route shuffle step with probability rho (the coin uses its own
    stream, so the times p_ij do not depend on rho). rho = 0 gives identical routes
    0, 1, ..., m-1, i.e. a general (non-permutation) flow shop.
    """
    p = [[0] * m for _ in range(n)]
    s = time_seed
    for j in range(n):
        for t in range(m):
            s, p[j][t] = _unif(s, 1, 99)
    route = [list(range(m)) for _ in range(n)]
    s = machine_seed
    coin = random.Random(machine_seed ^ 0x5EED)
    for j in range(n):
        for t in range(m):
            s, r = _unif(s, t, m - 1)
            if rho >= 1.0 or coin.random() < rho:
                route[j][t], route[j][r] = route[j][r], route[j][t]
    return Instance(name or f"gen{n}x{m}_rho{rho}", n, m, route, p)


@lru_cache(maxsize=1)
def taillard_table() -> dict:
    with open(DATA) as fh:
        return json.load(fh)["instances"]


def bks(name: str) -> int | None:
    """Best known upper bound for ta01-ta80 (see the _doc field of the data file)."""
    e = taillard_table().get(name)
    return None if e is None else e["bks"]


def ta_name(i: int) -> str:
    return f"ta{i:02d}"


def class_of(name: str) -> str | None:
    if not name.startswith("ta"):
        return None
    i = int(name[2:])
    for c, r in TA_CLASSES.items():
        if i in r:
            return c
    return None
