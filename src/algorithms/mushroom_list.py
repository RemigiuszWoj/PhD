"""Mushroom list — elite solution pool with double-bridge perturbation.

Maintains the k best distinct permutations seen during search.
When local search stagnates, perturbs the least-recently-used elite
solution with a double-bridge move instead of a fully random restart.

Double-bridge (4-opt):
    Split π into four consecutive segments A|B|C|D and reconnect
    as A|C|B|D.  The cut points are chosen uniformly at random from
    {1..n-1}, ensuring each segment is non-empty.  This move is
    outside the 2-opt neighbourhood and cannot be undone by a single
    adjacent or endpoint swap, making it an effective diversification
    step for FSP/TSP-class problems.
"""

import random
from typing import List, Optional, Tuple


def double_bridge(pi: List[int]) -> List[int]:
    """Apply a double-bridge (4-opt) perturbation to pi.

    Returns a new permutation.  The original list is not modified.
    """
    n = len(pi)
    if n < 4:
        return pi[:]

    # Three distinct cut points in 1..n-1, sorted
    cuts = sorted(random.sample(range(1, n), 3))
    a, b, c = cuts
    # Segments: [0,a) [a,b) [b,c) [c,n)
    seg_A = pi[:a]
    seg_B = pi[a:b]
    seg_C = pi[b:c]
    seg_D = pi[c:]
    return seg_A + seg_C + seg_B + seg_D


class MushroomList:
    """Elite solution pool of fixed capacity k.

    Keeps the k best distinct permutations encountered during search.
    Provides a diversification oracle: given a stagnation signal,
    returns a perturbed copy of the best unvisited elite solution.

    Usage in ILS:
        ml = MushroomList(k=10)
        ml.offer(pi, cmax)          # after every improving move
        restart_pi = ml.perturb()   # when stagnating
    """

    def __init__(self, k: int = 10) -> None:
        self.k = k
        # List of (cmax, pi) sorted ascending by cmax
        self._pool: List[Tuple[int, List[int]]] = []
        # Index of the next elite to use for perturbation (round-robin)
        self._cursor: int = 0

    # ------------------------------------------------------------------
    def offer(self, pi: List[int], cmax: int) -> bool:
        """Offer a solution to the pool.

        Returns True if it was added (new or better than the worst).
        """
        key = tuple(pi)
        # Check for duplicates
        for _, existing in self._pool:
            if tuple(existing) == key:
                return False

        if len(self._pool) < self.k:
            self._pool.append((cmax, pi[:]))
            self._pool.sort(key=lambda x: x[0])
            return True

        # Replace worst if this is better
        if cmax < self._pool[-1][0]:
            self._pool[-1] = (cmax, pi[:])
            self._pool.sort(key=lambda x: x[0])
            # Reset cursor so we always try the best first after a refresh
            self._cursor = 0
            return True

        return False

    # ------------------------------------------------------------------
    def perturb(self) -> Optional[List[int]]:
        """Return a double-bridge perturbation of the next elite solution.

        Cycles through the pool in round-robin order so that repeated
        calls to perturb() diversify across all k elites, not just the best.

        Returns None if the pool is empty.
        """
        if not self._pool:
            return None
        idx = self._cursor % len(self._pool)
        self._cursor += 1
        _, elite = self._pool[idx]
        return double_bridge(elite)

    # ------------------------------------------------------------------
    def best(self) -> Optional[Tuple[int, List[int]]]:
        """Return (cmax, pi) of the best solution in the pool, or None."""
        return self._pool[0] if self._pool else None

    def __len__(self) -> int:
        return len(self._pool)
