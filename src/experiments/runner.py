from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

from src.algorithms import iterated_local_search, simulated_annealing
from src.neighborhoods.common import QPUError, get_qpu_stats, reset_qpu_stats, set_qpu_budget
from src.parser import parser

ALGORITHMS_ALL = ("ils", "sa")

# Trzy klasy sąsiedztw:
#   classical          - CPU only
#   quantum_qubo       - D-Wave, oryginalne sformułowania (małe n)
#   quantum_qubo_enhanced - D-Wave, duże n (windowed / bez filtrów delta)
_DEFAULT_NEIGHBORHOODS = (
    # classical
    "adjacent",
    "fibonacci",
    "dynasearch",
    "motzkin",
    # quantum_qubo
    "quantum_adjacent",
    "quantum_fibonacci",
    "quantum_dynasearch",
    "quantum_motzkin",
    # quantum_qubo_enhanced
    "quantum_adjacent_enhanced",
    "quantum_fibonacci_enhanced",
    "quantum_dynasearch_enhanced",
    "quantum_motzkin_enhanced",
)


def _load_neighborhoods_from_config() -> tuple | None:
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        neigh = cfg.get("experiment", {}).get("neighborhoods")
        if isinstance(neigh, list) and neigh:
            return tuple(neigh)
    except Exception:
        pass
    return None


NEIGHBORHOODS_ALL = _load_neighborhoods_from_config() or _DEFAULT_NEIGHBORHOODS


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    neighborhood: str
    instance_file: str
    instance_number: int
    seed: int
    time_limit_ms: int
    tabu_tenure: int | None = None


@dataclass
class RunResult:
    config: RunConfig
    cmax_best: int
    time_to_best_ms: int
    total_time_ms: int
    cmax_history: List[int]
    time_history_ms: List[int]
    best_permutation: List[int]
    instance_jobs: int
    instance_machines: int
    upper_bound: int | None
    lower_bound: int | None
    qpu_stats: dict | None = None
    # --- measurement instrumentation (added 2026-07) ---
    iterations: int | None = None       # metaheuristic loop passes
    tl_exceeded: bool = False           # total_time_ms > time_limit_ms
    overrun_ms: int = 0                 # max(0, total_time_ms - time_limit_ms)
    avg_iter_ms: float | None = None    # total_time_ms / iterations
    neigh_time_ms: int | None = None    # cumulative wall time inside get_neighbor()

    def gap_percent(self) -> float | None:
        if self.lower_bound is None:
            return None
        try:
            return (self.cmax_best - self.lower_bound) / self.lower_bound * 100.0
        except ZeroDivisionError:
            return None

    def used_qpu_fallback(self) -> bool:
        """True if any QPU call fell back to SA (quota / embedding / other)."""
        s = self.qpu_stats or {}
        return bool(
            s.get("fallback_quota", 0)
            or s.get("fallback_embedding", 0)
            or s.get("fallback_other", 0)
        )

    def to_dict(self):
        d = asdict(self)
        d["gap_percent"] = self.gap_percent()
        d["used_qpu_fallback"] = self.used_qpu_fallback()
        d["config"] = asdict(self.config)
        return d


class ExperimentRunner:
    def __init__(
        self,
        base_results_dir: str = "results/experiments",
        quantum_config: dict | None = None,
        resume_dir: str | None = None,
        generate_plots: bool = False,
    ):
        self.base_dir = Path(base_results_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if resume_dir is not None:
            self.timestamp_dir = Path(resume_dir)
            self._resume_from = self.timestamp_dir
        else:
            self.timestamp_dir = self.base_dir / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self._resume_from = None
        self.timestamp_dir.mkdir(parents=True, exist_ok=True)
        if quantum_config is None:
            try:
                with open("config.yaml", "r") as f:
                    cfg = yaml.safe_load(f)
                quantum_config = cfg.get("quantum", {})
            except Exception:
                quantum_config = {}
        self.quantum_config = quantum_config
        # Campaign-wide QPU budget (self-metering; see common.set_qpu_budget).
        budget_s = (quantum_config or {}).get("qpu_budget_s")
        set_qpu_budget(budget_s * 1000.0 if budget_s else None)
        self.generate_plots = generate_plots
        # Read SA + ILS params from config.yaml (fallback to defaults if missing)
        try:
            with open("config.yaml", "r") as f:
                _cfg = yaml.safe_load(f) or {}
            self.sa_params = _cfg.get("simulated_annealing", {}) or {}
            self.ils_params = _cfg.get("iterated_local_search", {}) or {}
        except Exception:
            self.sa_params = {}
            self.ils_params = {}
        self._dispatch = {
            "ils": self._run_ils,
            "sa": self._run_sa,
        }

    def run(self, configs: Sequence[RunConfig]) -> List[RunResult]:
        results: List[RunResult] = []
        failed = 0
        for idx, cfg in enumerate(configs, start=1):
            if self._already_done(cfg):
                print(f"[Experiment] ({idx}/{len(configs)}) Skipping (done): {cfg}")
                continue
            print(f"[Experiment] ({idx}/{len(configs)}) Running: {cfg}")
            try:
                result = self._run_single(cfg)
            except QPUError as e:
                # No silent fallback: record the failure and move on with the
                # batch. Failed runs are retried on resume (_already_done only
                # honours result.json, not failed.json).
                failed += 1
                self._persist_failure(cfg, e)
                print(
                    f"[Experiment] ({idx}/{len(configs)}) FAILED (QPU {e.category}): {cfg} — {e}"
                )
                continue
            results.append(result)
            self._persist_result(result)
        if failed:
            print(f"[Experiment] WARNING: {failed} run(s) FAILED on QPU — see failed.json files.")
        if self.generate_plots:
            try:
                from src import visualization as viz
                viz.build_algorithm_multi_convergence_plots(self.timestamp_dir)
            except Exception as e:
                print(f"[Experiment] Failed to build multi-convergence plots: {e}")
        return results

    def _already_done(self, cfg: RunConfig) -> bool:
        if self._resume_from is None:
            return False
        from pathlib import Path as _Path
        stem = _Path(cfg.instance_file).stem
        pattern = (
            f"algo={cfg.algorithm}__neigh={cfg.neighborhood}"
            f"__file={stem}__inst={cfg.instance_number}"
            f"__*__tl={cfg.time_limit_ms}ms__seed={cfg.seed}"
        )
        return any(self._resume_from.glob(f"{pattern}/result.json"))

    def _run_single(self, cfg: RunConfig) -> RunResult:
        random.seed(cfg.seed)
        data = parser(cfg.instance_file, cfg.instance_number)
        processing_times = data["processing_times"]
        jobs = data["info"]["jobs"]
        machines = data["info"]["machines"]
        upper = data["info"].get("upper_bound")
        lower = data["info"].get("lower_bound")

        # Per-run QPU bookkeeping (covers all solve_qubo() invocations during this run)
        reset_qpu_stats()

        start = time.time()
        run_fn = self._dispatch.get(cfg.algorithm)
        if run_fn is None:
            raise ValueError(f"Unknown algorithm {cfg.algorithm}")
        best_pi, best_cmax, t_hist, c_hist, algo_stats = run_fn(processing_times, cfg)
        total_time_ms = int((time.time() - start) * 1000)
        time_to_best_ms = t_hist[-1] if t_hist else total_time_ms

        iterations = algo_stats.get("iterations")
        neigh_time_ms = algo_stats.get("neigh_time_ms")
        overrun_ms = max(0, total_time_ms - cfg.time_limit_ms)
        avg_iter_ms = (total_time_ms / iterations) if iterations else None

        qpu_stats = get_qpu_stats()

        # Loud, single-line summary if this quantum run silently fell back.
        is_quantum = cfg.neighborhood.startswith("quantum_")
        if is_quantum and (
            qpu_stats["fallback_quota"]
            or qpu_stats["fallback_embedding"]
            or qpu_stats["fallback_other"]
        ):
            print(
                f"[Experiment] WARNING: {cfg.algorithm}/{cfg.neighborhood} "
                f"seed={cfg.seed} tl={cfg.time_limit_ms}ms "
                f"used QPU fallback — qpu_success={qpu_stats['qpu_success']}/"
                f"{qpu_stats['calls']}, quota={qpu_stats['fallback_quota']}, "
                f"embed={qpu_stats['fallback_embedding']}, "
                f"other={qpu_stats['fallback_other']}"
            )

        return RunResult(
            config=cfg,
            cmax_best=best_cmax,
            time_to_best_ms=time_to_best_ms,
            total_time_ms=total_time_ms,
            cmax_history=c_hist,
            time_history_ms=t_hist,
            best_permutation=best_pi,
            instance_jobs=jobs,
            instance_machines=machines,
            upper_bound=upper,
            lower_bound=lower,
            qpu_stats=qpu_stats,
            iterations=iterations,
            tl_exceeded=total_time_ms > cfg.time_limit_ms,
            overrun_ms=overrun_ms,
            avg_iter_ms=avg_iter_ms,
            neigh_time_ms=neigh_time_ms,
        )

    def _run_dir_name(self, cfg: RunConfig, jobs: int | None = None, machines: int | None = None) -> str:
        base = (
            f"algo={cfg.algorithm}__neigh={cfg.neighborhood}"
            f"__file={Path(cfg.instance_file).stem}"
            f"__inst={cfg.instance_number}"
        )
        if jobs is not None and machines is not None:
            base += f"__n{jobs}__m{machines}"
        return base + f"__tl={cfg.time_limit_ms}ms__seed={cfg.seed}"

    def _persist_failure(self, cfg: RunConfig, exc: QPUError) -> None:
        """Record a QPU-failed run as failed.json (retryable on resume).

        Uses the SAME directory name as a successful run (n/m parsed from the
        instance), so a later successful retry lands in this directory and
        removes the failed.json marker — no orphaned failure dirs.
        """
        try:
            data = parser(cfg.instance_file, cfg.instance_number)
            jobs = data["info"]["jobs"]
            machines = data["info"]["machines"]
        except Exception:
            jobs = machines = None
        run_dir = self.timestamp_dir / self._run_dir_name(cfg, jobs, machines)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(cfg),
            "failed_at": datetime.utcnow().isoformat() + "Z",
            "category": exc.category,
            "error": str(exc),
            "qpu_stats": get_qpu_stats(),
        }
        with open(run_dir / "failed.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Experiment] Saved {run_dir / 'failed.json'}")

    def _persist_result(self, result: RunResult) -> None:
        cfg = result.config
        run_dir = self.timestamp_dir / self._run_dir_name(
            cfg, result.instance_jobs, result.instance_machines
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "result.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        # Successful (re)run supersedes an earlier QPU failure of the same config.
        (run_dir / "failed.json").unlink(missing_ok=True)
        print(f"[Experiment] Saved {path}")

        if self.generate_plots:
            try:
                from src import visualization as viz
                times = result.time_history_ms or []
                c_hist = result.cmax_history or []
                if times and c_hist:
                    viz.save_convergence_plot_to(times, c_hist, str(run_dir / "convergence.png"))
            except Exception as e:
                print(f"[Experiment] Failed to create convergence plot: {e}")

    def _run_ils(self, processing_times, cfg: RunConfig):
        return iterated_local_search(
            processing_times,
            max_time_ms=cfg.time_limit_ms,
            tabu_tenure=cfg.tabu_tenure or self.ils_params.get("tabu_tenure", 10),
            neigh_mode=cfg.neighborhood,
            iter_log_path=None,
            quantum_config=self.quantum_config,
            mushroom_k=self.ils_params.get("mushroom_k", 10),
        )

    def _run_sa(self, processing_times, cfg: RunConfig):
        sp = self.sa_params
        return simulated_annealing(
            processing_times,
            time_limit_ms=cfg.time_limit_ms,
            initial_temp=sp.get("initial_temp", 1000.0),
            final_temp=sp.get("final_temp", 1.0),
            alpha=sp.get("alpha", 0.95),
            neigh_mode=cfg.neighborhood,
            reheat_factor=sp.get("reheat_factor"),
            stagnation_ms=sp.get("stagnation_ms"),
            temp_floor_factor=sp.get("temp_floor_factor"),
            iter_log_path=None,
            quantum_config=self.quantum_config,
        )


def generate_basic_plan(
    instance_file: str,
    instance_number: int,
    repeats: int,
    time_limit_ms: int,
    algorithms: Iterable[str] | None = None,
    neighborhoods: Iterable[str] | None = None,
) -> List[RunConfig]:
    algorithms = tuple(algorithms) if algorithms is not None else ALGORITHMS_ALL
    neighborhoods = tuple(neighborhoods) if neighborhoods is not None else NEIGHBORHOODS_ALL
    configs: List[RunConfig] = []
    for algo in algorithms:
        for neigh in neighborhoods:
            for seed in range(repeats):
                configs.append(
                    RunConfig(
                        algorithm=algo,
                        neighborhood=neigh,
                        instance_file=instance_file,
                        instance_number=instance_number,
                        seed=seed,
                        time_limit_ms=time_limit_ms,
                        tabu_tenure=10 if algo == "ils" else None,
                    )
                )
    return configs


def count_instances_in_file(instance_file: str) -> int:
    instances = 0
    try:
        with open(instance_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        it = iter(lines)
        for line in it:
            if line.startswith("number of jobs"):
                try:
                    header = next(it)
                except StopIteration:
                    break
                try:
                    next(it)
                except StopIteration:
                    break
                parts = header.split()
                if len(parts) >= 5:
                    machines = int(parts[1])
                    for _ in range(machines):
                        try:
                            next(it)
                        except StopIteration:
                            break
                    instances += 1
    except FileNotFoundError:
        return 0
    return instances


def generate_plan_all_instances(
    instance_file: str,
    repeats: int,
    time_limit_ms: int | None = None,
    time_limits_ms: Iterable[int] | None = None,
    algorithms: Iterable[str] | None = None,
    neighborhoods: Iterable[str] | None = None,
) -> List[RunConfig]:
    total = count_instances_in_file(instance_file)
    if total == 0:
        return []
    limits: List[int] = (
        list(time_limits_ms)
        if time_limits_ms
        else ([time_limit_ms] if time_limit_ms is not None else [1000])
    )
    configs: List[RunConfig] = []
    for inst_num in range(total):
        for tl in limits:
            configs.extend(
                generate_basic_plan(
                    instance_file=instance_file,
                    instance_number=inst_num,
                    repeats=repeats,
                    time_limit_ms=tl,
                    algorithms=algorithms,
                    neighborhoods=neighborhoods,
                )
            )
    return configs


def generate_plan_for_files(
    instance_files: Iterable[str],
    repeats: int,
    time_limits_ms: Iterable[int],
    algorithms: Iterable[str] | None = None,
    neighborhoods: Iterable[str] | None = None,
) -> List[RunConfig]:
    all_configs: List[RunConfig] = []
    for inst_file in instance_files:
        all_configs.extend(
            generate_plan_all_instances(
                instance_file=inst_file,
                repeats=repeats,
                time_limits_ms=time_limits_ms,
                algorithms=algorithms,
                neighborhoods=neighborhoods,
            )
        )
    return all_configs
