"""ILS and SA for the job shop: the SAME control flow and constants as the PFSP code.

Mirrors src/algorithms/ils.py, src/algorithms/sa.py and src/algorithms/mushroom_list.py
line by line (tabu on the move id with aspiration, elite-pool double-bridge kick; SA with
Boltzmann acceptance of the best neighbor, geometric cooling to a floor, reset of T to T0
plus an elite kick after a wall-clock stagnation). Only the problem layer differs:
start = round-robin dispatch (identity analog), kick = double bridge on the operation
sequence, neighbor = src/jobshop/neighborhoods.best_move. The PFSP code is not touched
(PLAN.md D12), so running PFSP campaigns are not affected.

Both return the fields the runner and scripts/extract_budgets.py expect, including the
move log [(end of move in ms, best C_max)] used to read smaller budgets off one long run.
"""

from __future__ import annotations

import math
import random
import time

from src.jobshop.instance import Instance
from src.jobshop.neighborhoods import best_move, parse_mode
from src.jobshop.schedule import cmax, double_bridge, identity_start, random_start, seq_key


class EliteList:
    """MushroomList for machine sequences: k best distinct solutions, round-robin kick."""

    def __init__(self, k: int, inst: Instance, rng: random.Random):
        self.k, self.inst, self.rng = k, inst, rng
        self._pool: list[tuple[int, list[list[int]]]] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._pool)

    def offer(self, seq, c: int) -> bool:
        key = seq_key(seq)
        if any(seq_key(s) == key for _, s in self._pool):
            return False
        if len(self._pool) < self.k:
            self._pool.append((c, [s[:] for s in seq]))
            self._pool.sort(key=lambda x: x[0])
            return True
        if c < self._pool[-1][0]:
            self._pool[-1] = (c, [s[:] for s in seq])
            self._pool.sort(key=lambda x: x[0])
            self._cursor = 0
            return True
        return False

    def perturb(self):
        if not self._pool:
            return None
        idx = self._cursor % len(self._pool)
        self._cursor += 1
        return double_bridge(self.inst, self._pool[idx][1], self.rng)


def _result(best_C, best_seq, c_hist, t_hist, it, t0, move_log, neigh_s):
    return {"cmax_best": best_C, "best_sequences": best_seq, "cmax_history": c_hist,
            "time_history_ms": t_hist, "iterations": it,
            "total_time_ms": int((time.time() - t0) * 1000), "neigh_time_ms": int(neigh_s * 1000),
            "move_log": move_log}


def iterated_local_search(inst: Instance, mode: str, time_limit_ms: int, seed: int,
                          tabu_tenure: int = 10, mushroom_k: int = 10, diversification: str = "mushroom") -> dict:
    rng = random.Random(seed)
    spec = parse_mode(mode)
    cur = identity_start(inst)
    cur_C = cmax(inst, cur)
    best_seq, best_C = [s[:] for s in cur], cur_C
    c_hist, t_hist = [cur_C], [0]
    tabu: dict = {}
    tenure = tabu_tenure if tabu_tenure else 10
    elite = EliteList(mushroom_k, inst, rng)
    elite.offer(cur, cur_C)
    it = 0
    neigh_s = 0.0
    move_log: list[tuple[int, int]] = []
    t0 = time.time()
    limit = time_limit_ms / 1000.0
    while time.time() - t0 < limit:
        _t = time.time()
        new, new_C, move_id = best_move(inst, cur, spec)
        neigh_s += time.time() - _t
        tabu_active = move_id is not None and move_id in tabu and tabu[move_id] > it
        if new is None or (tabu_active and new_C >= best_C):
            # diversify (ILS kick): elite double bridge, else uniform random restart
            alt = elite.perturb() if diversification == "mushroom" else None
            if alt is None:
                alt = random_start(inst, rng)
            new, new_C, move_id = alt, cmax(inst, alt), ("kick", it)
        cur, cur_C = new, new_C
        tabu[move_id] = it + tenure
        if cur_C < best_C:
            best_C, best_seq = cur_C, [s[:] for s in cur]
            c_hist.append(best_C)
            t_hist.append(int((time.time() - t0) * 1000))
            elite.offer(best_seq, best_C)
        it += 1
        move_log.append((int((time.time() - t0) * 1000), best_C))
    return _result(best_C, best_seq, c_hist, t_hist, it, t0, move_log, neigh_s)


def simulated_annealing(inst: Instance, mode: str, time_limit_ms: int, seed: int,
                        initial_temp: float = 1000.0, final_temp: float = 100.0, alpha: float = 0.99,
                        reheat_factor: float | None = 1.5, stagnation_ms: int | None = 100,
                        temp_floor_factor: float | None = 1.0, mushroom_k: int = 10) -> dict:
    rng = random.Random(seed)
    spec = parse_mode(mode)
    cur = identity_start(inst)
    cur_C = cmax(inst, cur)
    best_seq, best_C = [s[:] for s in cur], cur_C
    c_hist, t_hist = [cur_C], [0]
    T = initial_temp
    stagnation = stagnation_ms if stagnation_ms and stagnation_ms > 0 else None
    reheat = reheat_factor if reheat_factor and reheat_factor > 1.0 else None
    floor = final_temp * (temp_floor_factor if temp_floor_factor and temp_floor_factor >= 1.0 else 1.0)
    elite = EliteList(mushroom_k, inst, rng)
    elite.offer(cur, cur_C)
    it = 0
    neigh_s = 0.0
    move_log: list[tuple[int, int]] = []
    t0 = time.time()
    last_improve = t0
    limit = time_limit_ms / 1000.0
    while time.time() - t0 < limit:
        _t = time.time()
        new, new_C, _ = best_move(inst, cur, spec)
        neigh_s += time.time() - _t
        if new is None:  # no candidate (single block or all infeasible): forced kick
            new = double_bridge(inst, cur, rng)
            new_C = cmax(inst, new)
        delta = new_C - cur_C
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur, cur_C = new, new_C
        if cur_C < best_C:
            best_C, best_seq = cur_C, [s[:] for s in cur]
            c_hist.append(best_C)
            t_hist.append(int((time.time() - t0) * 1000))
            last_improve = time.time()
            elite.offer(best_seq, best_C)
        it += 1
        T = max(T * alpha, floor)
        if stagnation is not None and reheat is not None:
            if (time.time() - last_improve) * 1000.0 >= stagnation:
                T = initial_temp
                kicked = elite.perturb()
                cur = kicked if kicked is not None else double_bridge(inst, cur, rng)
                cur_C = cmax(inst, cur)
                last_improve = time.time()
        move_log.append((int((time.time() - t0) * 1000), best_C))
    return _result(best_C, best_seq, c_hist, t_hist, it, t0, move_log, neigh_s)
