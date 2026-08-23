"""Common structures and helper functions for search algorithms.

Neighborhood classes:
    classical/          - CPU-only, fully classical
    quantum_qubo/       - D-Wave / SimulatedAnnealingSampler, original formulations
    quantum_qubo_enhanced/ - D-Wave, large-n extensions (windowed / no delta filter)
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Tuple

# --- classical ---
from src.neighborhoods.classical.adjacent import generate_neighbors_adjacent
from src.neighborhoods.classical.dynasearch import dynasearch_full
from src.neighborhoods.classical.fibonacci import (
    fibonacci_neighborhood_topk,
)
from src.neighborhoods.classical.motzkin import motzkin_neighborhood_full

# --- quantum_qubo ---
from src.neighborhoods.quantum_qubo.adjacent import quantum_adjacent_neighborhood
from src.neighborhoods.quantum_qubo.dynasearch import quantum_dynasearch_neighborhood
from src.neighborhoods.quantum_qubo.fibonacci import (
    quantum_fibonacci_neighborhood,
)
from src.neighborhoods.quantum_qubo.motzkin import quantum_motzkin_neighborhood

# --- quantum_qubo_enhanced ---
from src.neighborhoods.quantum_qubo_enhanced.adjacent import quantum_adjacent_enhanced
from src.neighborhoods.quantum_qubo_enhanced.dynasearch import quantum_dynasearch_enhanced
from src.neighborhoods.quantum_qubo_enhanced.fibonacci import quantum_fibonacci_enhanced
from src.neighborhoods.quantum_qubo_enhanced.motzkin import quantum_motzkin_enhanced

# gate_qaoa (gate-model QAOA)
from src.neighborhoods.gate_qaoa.adjacent import gate_adjacent_neighborhood
from src.neighborhoods.gate_qaoa.dynasearch import gate_dynasearch_neighborhood
from src.neighborhoods.gate_qaoa.fibonacci import gate_fibonacci_neighborhood
from src.neighborhoods.gate_qaoa.motzkin import gate_motzkin_neighborhood
from src.permutation_procesing import c_max

# ---------------------------------------------------------------------------
# Valid neighborhood mode names
# ---------------------------------------------------------------------------
CLASSICAL_MODES = ("adjacent", "fibonacci", "dynasearch", "motzkin")
QUANTUM_QUBO_MODES = (
    "quantum_adjacent",
    "quantum_fibonacci",
    "quantum_dynasearch",
    "quantum_motzkin",
)
QUANTUM_ENHANCED_MODES = (
    "quantum_adjacent_enhanced",
    "quantum_fibonacci_enhanced",
    "quantum_dynasearch_enhanced",
    "quantum_motzkin_enhanced",
)
GATE_QAOA_MODES = (
    "gate_adjacent",
    "gate_fibonacci",
    "gate_dynasearch",
    "gate_motzkin",
)
ALL_MODES = CLASSICAL_MODES + QUANTUM_QUBO_MODES + QUANTUM_ENHANCED_MODES + GATE_QAOA_MODES


@dataclass
class SearchState:
    """Shared state for search algorithms."""

    current_pi: List[int]
    current_cmax: int
    best_pi: List[int]
    best_cmax: int
    cmax_history: List[int] = field(default_factory=list)
    iteration_history: List[int] = field(default_factory=list)
    start_time: float = 0.0
    iteration: int = 0

    def update_best(self) -> bool:
        if self.current_cmax < self.best_cmax:
            self.best_cmax = self.current_cmax
            self.best_pi = self.current_pi.copy()
            self.cmax_history.append(self.best_cmax)
            elapsed_ms = int((time.time() - self.start_time) * 1000)
            self.iteration_history.append(elapsed_ms)
            return True
        return False

    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)


@contextmanager
def open_log_file(path: str | None, algo_name: str) -> Iterator[Any]:
    log_file = None
    if path:
        try:
            log_file = open(path, "w", encoding="utf-8")
            log_file.write("iteration,elapsed_ms,current_cmax,best_cmax,permutation\n")
        except Exception as e:
            print(f"[{algo_name}] Failed to open log file {path}: {e}")
            log_file = None
    try:
        yield log_file
    finally:
        if log_file:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass


def log_iteration(log_file: Any, state: SearchState) -> None:
    if log_file:
        try:
            permutation_str = " ".join(map(str, state.current_pi))
            log_file.write(
                f"{state.iteration},{state.elapsed_ms()},{state.current_cmax},"
                f'{state.best_cmax},"{permutation_str}"\n'
            )
        except Exception:
            pass


def _extract_quantum_params(quantum_config: dict | None, mode: str) -> dict:
    """Extract per-mode quantum parameters from quantum_config dict."""
    # gate_qaoa modes use their own parameter set (QAOA depth/backend/shots),
    # not the D-Wave annealer parameters; angles are loaded by solve_qaoa from
    # the calibration table.
    if mode.startswith("gate_"):
        cfg = quantum_config or {}
        params = {
            "p": cfg.get("qaoa_p", 1),
            "backend": cfg.get("qaoa_backend", "statevector"),
            "shots": cfg.get("qaoa_shots", 4096),
        }
        if mode in ("gate_dynasearch", "gate_motzkin"):
            params["window_size"] = cfg.get("qaoa_window_size")     # None = single QUBO
            params["overlap_ratio"] = cfg.get("qaoa_overlap_ratio", 0.5)
            params["L_max"] = cfg.get(
                "L_max_dynasearch" if mode == "gate_dynasearch" else "L_max_motzkin")
        return params
    if not quantum_config:
        return {}
    params = {
        "num_reads": quantum_config.get("num_reads", 5),
        "backend": quantum_config.get("backend", "simulator"),
        "dwave_token": quantum_config.get("dwave_token"),
        "solver": quantum_config.get("solver"),
        "annealing_time_us": quantum_config.get("annealing_time_us"),
        "chain_strength": quantum_config.get("chain_strength"),
        "num_spin_reversal_transforms": quantum_config.get("num_spin_reversal_transforms"),
    }
    # Enhanced modes: higher default num_reads
    if "enhanced" in mode:
        params["num_reads"] = quantum_config.get("num_reads_enhanced", 100)
        params["window_size"] = quantum_config.get("window_size")  # None = auto
        params["overlap_ratio"] = quantum_config.get("overlap_ratio", 0.5)
    # mode-specific L_max for non-enhanced dynasearch/motzkin
    if mode == "quantum_dynasearch":
        params["L_max"] = quantum_config.get("L_max_dynasearch")
    if mode == "quantum_motzkin":
        params["L_max"] = quantum_config.get("L_max_motzkin")
    return params


def get_neighbor(
    neigh_mode: str,
    current_pi: List[int],
    processing_times: List[List[int]],
    n: int,
    tabu_len: int | None = None,
    quantum_config: dict | None = None,
) -> Tuple[List[int], int, Any, List[dict] | None]:
    """Generate the best neighbor for a given neighborhood mode.

    Returns:
        (new_pi, new_cmax, move_id, top_moves)
        top_moves: only for adjacent/fibonacci in ILS (tabu_len provided)
    """
    # ------------------------------------------------------------------
    # CLASSICAL
    # ------------------------------------------------------------------
    if neigh_mode == "adjacent":
        neighbors_with_cmax = []
        for neighbor, move in generate_neighbors_adjacent(current_pi):
            c = c_max(neighbor, processing_times)
            neighbors_with_cmax.append({"pi": neighbor, "cmax": c, "move": move})
        neighbors_with_cmax.sort(key=lambda x: x["cmax"])
        best = neighbors_with_cmax[0] if neighbors_with_cmax else None
        if best is None:
            return current_pi, c_max(current_pi, processing_times), None, None
        top_moves = None
        if tabu_len is not None:
            top_moves = neighbors_with_cmax[: tabu_len + 1]
        return best["pi"], best["cmax"], best["move"], top_moves

    elif neigh_mode == "fibonacci":
        k = (tabu_len + 1) if tabu_len is not None else 1
        top_moves = fibonacci_neighborhood_topk(current_pi, processing_times, k)
        if top_moves:
            best = top_moves[0]
            return best["pi"], best["cmax"], best["move"], top_moves if tabu_len else None
        return current_pi, c_max(current_pi, processing_times), None, None

    elif neigh_mode == "dynasearch":
        new_pi, new_c, _ = dynasearch_full(current_pi, processing_times)
        return new_pi, new_c, tuple(new_pi), None

    elif neigh_mode == "motzkin":
        if n > 150:
            print(f"[motzkin] Warning: n={n} may be slow.")
        new_pi, new_c, selected = motzkin_neighborhood_full(current_pi, processing_times)
        move_id = tuple(selected) if selected else tuple(new_pi)
        return new_pi, new_c, move_id, None

    # ------------------------------------------------------------------
    # QUANTUM QUBO (original formulations)
    # ------------------------------------------------------------------
    elif neigh_mode == "quantum_adjacent":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, move = quantum_adjacent_neighborhood(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 5),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, move, None

    elif neigh_mode == "quantum_fibonacci":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_fibonacci_neighborhood(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 5),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "quantum_dynasearch":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_dynasearch_neighborhood(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 5),
            L_max=p.get("L_max"),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "quantum_motzkin":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_motzkin_neighborhood(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 5),
            L_max=p.get("L_max"),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    # ------------------------------------------------------------------
    # QUANTUM QUBO ENHANCED (large-n extensions)
    # ------------------------------------------------------------------
    elif neigh_mode == "quantum_adjacent_enhanced":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, move = quantum_adjacent_enhanced(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 100),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, move, None

    elif neigh_mode == "quantum_fibonacci_enhanced":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_fibonacci_enhanced(
            current_pi,
            processing_times,
            num_reads=p.get("num_reads", 100),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "quantum_dynasearch_enhanced":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_dynasearch_enhanced(
            current_pi,
            processing_times,
            window_size=p.get("window_size"),
            overlap_ratio=p.get("overlap_ratio", 0.5),
            num_reads=p.get("num_reads", 100),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "quantum_motzkin_enhanced":
        p = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = quantum_motzkin_enhanced(
            current_pi,
            processing_times,
            window_size=p.get("window_size"),
            overlap_ratio=p.get("overlap_ratio", 0.5),
            num_reads=p.get("num_reads", 100),
            backend=p.get("backend", "simulator"),
            dwave_token=p.get("dwave_token"),
            solver=p.get("solver"),
            annealing_time_us=p.get("annealing_time_us"),
            chain_strength=p.get("chain_strength"),
            num_spin_reversal_transforms=p.get("num_spin_reversal_transforms"),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    # ------------------------------------------------------------------
    # GATE-MODEL QAOA
    # ------------------------------------------------------------------
    elif neigh_mode == "gate_adjacent":
        qp = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, move = gate_adjacent_neighborhood(
            current_pi, processing_times,
            p=qp["p"], backend=qp["backend"], shots=qp["shots"],
        )
        return new_pi, new_c, move, None

    elif neigh_mode == "gate_fibonacci":
        qp = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = gate_fibonacci_neighborhood(
            current_pi, processing_times,
            p=qp["p"], backend=qp["backend"], shots=qp["shots"],
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "gate_dynasearch":
        qp = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = gate_dynasearch_neighborhood(
            current_pi, processing_times,
            p=qp["p"], backend=qp["backend"], shots=qp["shots"], L_max=qp.get("L_max"),
            window_size=qp.get("window_size"), overlap_ratio=qp.get("overlap_ratio", 0.5),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    elif neigh_mode == "gate_motzkin":
        qp = _extract_quantum_params(quantum_config, neigh_mode)
        new_pi, new_c, swaps = gate_motzkin_neighborhood(
            current_pi, processing_times,
            p=qp["p"], backend=qp["backend"], shots=qp["shots"], L_max=qp.get("L_max"),
            window_size=qp.get("window_size"), overlap_ratio=qp.get("overlap_ratio", 0.5),
        )
        return new_pi, new_c, tuple(swaps) if swaps else tuple(new_pi), None

    else:
        raise ValueError(f"Unknown neigh_mode='{neigh_mode}'. " f"Valid modes: {ALL_MODES}")
