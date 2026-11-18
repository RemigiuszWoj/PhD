"""NEH and Iterated Greedy (IG) for the PFSP / C_max -- context rows of the crossing paper (plan v4, item 3).

NEH (Nawaz, Enscore, Ham 1983): jobs by decreasing total processing time, each inserted at its best
position (Taillard's acceleration, ties -> first position). IG (Ruiz and Stuetzle 2007, the IG_RS
variant): destruction of d = 4 random jobs, greedy reinsertion, iterative-improvement insertion local
search, and a constant-temperature acceptance with T = 0.4 * sum(p) / (10 n m). Same wall-clock budget
and return format as iterated_local_search, so the experiment runner treats IG like ILS. Pure Python
like the rest of the code base: IG here is a reference point, not a tuned state-of-the-art implementation.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Tuple

from src.neighborhoods.classical.insert import _heads_tails
from src.permutation_procesing import c_max


def best_insertion(seq: List[int], job: int, pt: List[List[int]]) -> Tuple[int, int]:
    """Best position t (0..len(seq)) to insert `job` into `seq` and the resulting C_max (Taillard)."""
    m, L = len(pt), len(seq)
    if L == 0:
        return 0, sum(pt[r][job] for r in range(m))
    e, q = _heads_tails(seq, pt)
    best_t, best_c = 0, None
    for t in range(L + 1):
        f_prev, cmax = 0, 0
        for r in range(m):
            left = e[r][t - 1] if t else 0
            f = (left if left > f_prev else f_prev) + pt[r][job]
            tail = q[r][t] if t < L else 0
            if f + tail > cmax:
                cmax = f + tail
            f_prev = f
        if best_c is None or cmax < best_c:
            best_t, best_c = t, cmax
    return best_t, best_c


def neh(pt: List[List[int]]) -> List[int]:
    """NEH permutation (deterministic: ties in the job order by index, ties in position -> first)."""
    m, n = len(pt), len(pt[0])
    order = sorted(range(n), key=lambda j: (-sum(pt[r][j] for r in range(m)), j))
    seq: List[int] = []
    for j in order:
        t, _ = best_insertion(seq, j, pt)
        seq.insert(t, j)
    return seq


def _local_search_insertion(seq: List[int], cmax: int, pt: List[List[int]], deadline: float) -> Tuple[List[int], int]:
    """Iterative improvement by insertion (IG_RS): remove each job in random order, reinsert at the best
    position, repeat while it improves."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for job in random.sample(seq, len(seq)):
            if time.time() >= deadline:
                break
            rest = [x for x in seq if x != job]
            t, c = best_insertion(rest, job, pt)
            if c < cmax:
                seq, cmax, improved = rest[:t] + [job] + rest[t:], c, True
    return seq, cmax


def iterated_greedy(pt: List[List[int]], max_time_ms: int = 10000, d: int = 4, temp_factor: float = 0.4,
                    start: str = "neh"):
    """IG_RS. Returns (best_pi, best_cmax, time_history_ms, cmax_history, stats) like the ILS."""
    t0 = time.time()
    deadline = t0 + max_time_ms / 1000.0
    m, n = len(pt), len(pt[0])
    temp = temp_factor * sum(map(sum, pt)) / (10.0 * n * m)
    cur = neh(pt) if start == "neh" else list(range(n))
    cur_c = c_max(cur, pt)
    best, best_c = cur.copy(), cur_c
    t_hist, c_hist, move_log = [int((time.time() - t0) * 1000)], [best_c], []
    cur, cur_c = _local_search_insertion(cur, cur_c, pt, deadline)
    if cur_c < best_c:
        best, best_c = cur.copy(), cur_c
        t_hist.append(int((time.time() - t0) * 1000))
        c_hist.append(best_c)
    move_log.append((int((time.time() - t0) * 1000), best_c))
    iters = 0
    while time.time() < deadline:
        iters += 1
        seq = cur.copy()
        removed = random.sample(seq, min(d, n - 1))
        seq = [x for x in seq if x not in removed]
        for job in removed:
            t, _ = best_insertion(seq, job, pt)
            seq.insert(t, job)
        c = c_max(seq, pt)
        seq, c = _local_search_insertion(seq, c, pt, deadline)
        if c < cur_c or random.random() <= math.exp(-(c - cur_c) / temp):
            cur, cur_c = seq, c
        if cur_c < best_c:
            best, best_c = cur.copy(), cur_c
            t_hist.append(int((time.time() - t0) * 1000))
            c_hist.append(best_c)
        move_log.append((int((time.time() - t0) * 1000), best_c))
    return best, best_c, t_hist, c_hist, {"iterations": iters, "neigh_time_ms": None, "move_log": move_log}
