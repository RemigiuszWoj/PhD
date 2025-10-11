from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from src.parser import parser
from src.serach import simulated_annealing, tabu_search  # TODO: rename serach -> search later

# Full canonical sets used for experiments (always applied regardless of config lists)
ALGORITHMS_ALL = ("tabu", "sa")
NEIGHBORHOODS_ALL = (
    "adjacent",
    "fibonahi_neighborhood",
    "dynasearch_neighborhood",
)


@dataclass(frozen=True)
class RunConfig:
    """Minimal konfiguracja pojedynczego uruchomienia.

    Uproszczono: usunięto wszystkie parametry SA poza limitem czasu.
    Parametry SA używają stałych domyślnych, Tabu ma opcjonalny tabu_tenure.
    """

    algorithm: str  # 'tabu' | 'sa'
    neighborhood: str  # 'adjacent' | 'fibonahi_neighborhood' | 'dynasearch_neighborhood'
    instance_file: str  # ścieżka do pliku instancji
    instance_number: int  # numer instancji w pliku
    seed: int  # seed RNG
    time_limit_ms: int  # budżet czasu
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
    # Placeholder for future: evals_total, evals_to_best, accept_rate

    def gap_percent(self) -> float | None:
        if self.lower_bound is None:
            return None
        try:
            return (self.cmax_best - self.lower_bound) / self.lower_bound * 100.0
        except ZeroDivisionError:
            return None

    def to_dict(self):
        d = asdict(self)
        d["gap_percent"] = self.gap_percent()
        # Convert config to nested dict
        d["config"] = asdict(self.config)
        return d


class ExperimentRunner:
    def __init__(self, base_results_dir: str = "results/experiments"):
        """Runner nie czyści teraz całego katalogu bazowego.

        Każdy batch dostaje własny katalog timestamp, stare pozostają.
        """
        self.base_dir = Path(base_results_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp_dir = self.base_dir / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.timestamp_dir.mkdir(parents=True, exist_ok=True)
        # Predefiniowana mapa dispatcherów
        self._dispatch = {
            "tabu": self._run_tabu,
            "sa": self._run_sa,
        }

    def run(self, configs: Sequence[RunConfig]) -> List[RunResult]:
        results: List[RunResult] = []
        for idx, cfg in enumerate(configs, start=1):
            print(f"[Experiment] ({idx}/{len(configs)}) Running: {cfg}")
            result = self._run_single(cfg)
            results.append(result)
            self._persist_result(result)
        return results

    def _run_single(self, cfg: RunConfig) -> RunResult:
        random.seed(cfg.seed)
        # load instance
        data = parser(cfg.instance_file, cfg.instance_number)
        processing_times = data["processing_times"]
        jobs = data["info"]["jobs"]
        machines = data["info"]["machines"]
        upper = data["info"].get("upper_bound")
        lower = data["info"].get("lower_bound")

        start = time.time()
        run_fn = self._dispatch.get(cfg.algorithm)
        if run_fn is None:
            raise ValueError(f"Unknown algorithm {cfg.algorithm}")
        best_pi, best_cmax, t_hist, c_hist = run_fn(processing_times, cfg)
        total_time_ms = int((time.time() - start) * 1000)
        if t_hist:
            time_to_best_ms = t_hist[-1]
        else:
            time_to_best_ms = total_time_ms

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
        )

    def _persist_result(self, result: RunResult) -> None:
        cfg = result.config
        filename = (
            f"algo={cfg.algorithm}_neigh={cfg.neighborhood}_file={Path(cfg.instance_file).stem}"
            f"_instnum={cfg.instance_number}_n{result.instance_jobs}_m{result.instance_machines}"
            f"_tl={cfg.time_limit_ms}ms_seed={cfg.seed}.json"
        )
        path = self.timestamp_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"[Experiment] Saved {path}")

    # --- prywatne implementacje algorytmów dla dispatcher'a ---
    def _run_tabu(self, processing_times, cfg: RunConfig):
        return tabu_search(
            processing_times,
            max_time_ms=cfg.time_limit_ms,
            tabu_tenure=cfg.tabu_tenure or 10,
            neigh_mode=cfg.neighborhood,
            iter_log_path=None,
        )

    def _run_sa(self, processing_times, cfg: RunConfig):
        return simulated_annealing(
            processing_times,
            time_limit_ms=cfg.time_limit_ms,
            initial_temp=1000.0,
            final_temp=1.0,
            alpha=0.95,
            neigh_mode=cfg.neighborhood,
            reheat_factor=None,
            stagnation_ms=None,
            temp_floor_factor=None,
            iter_log_path=None,
        )


def generate_basic_plan(
    instance_file: str,
    instance_number: int,
    repeats: int,
    time_limit_ms: int,
    algorithms: Iterable[str] | None = None,
    neighborhoods: Iterable[str] | None = None,
) -> List[RunConfig]:
    """Generate plan; if algorithms/neighborhoods omitted, use full canonical sets.

    NOTE: For research mode we now ALWAYS override to the full sets (ALGORITHMS_ALL,
    NEIGHBORHOODS_ALL) to guarantee comprehensive coverage, ignoring any restricted
    lists passed in config. This makes experiments comparable across runs.
    """
    # Force full coverage regardless of user-provided subsets.
    algorithms = ALGORITHMS_ALL
    neighborhoods = NEIGHBORHOODS_ALL
    configs: List[RunConfig] = []
    base_seeds = list(range(repeats))
    for algo in algorithms:
        for neigh in neighborhoods:
            for seed in base_seeds:
                cfg = RunConfig(
                    algorithm=algo,
                    neighborhood=neigh,
                    instance_file=instance_file,
                    instance_number=instance_number,
                    seed=seed,
                    time_limit_ms=time_limit_ms,
                    tabu_tenure=10 if algo == "tabu" else None,
                )
                configs.append(cfg)
    return configs


def count_instances_in_file(instance_file: str) -> int:
    """Count how many instances are present in a Taillard-format file using parser logic.

    We iterate parser until exhaustion; cheap because file small vs runtime.
    """
    # Re-parse similarly to parser but counting; reuse parser by reading all then len(list)
    instances: List[int] = []
    try:
        with open(instance_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        it = iter(lines)
        for line in it:
            if line.startswith("number of jobs"):
                # skip header lines matching parser expectations
                try:
                    header = next(it)
                except StopIteration:
                    break
                # skip 'processing times :' label
                try:
                    next(it)
                except StopIteration:
                    break
                # header holds: jobs machines seed upper lower
                parts = header.split()
                if len(parts) >= 5:
                    machines = int(parts[1])
                    # consume machines lines of processing times
                    for _ in range(machines):
                        try:
                            next(it)
                        except StopIteration:
                            break
                    instances.append(1)
    except FileNotFoundError:
        return 0
    return len(instances)


def generate_plan_all_instances(
    instance_file: str,
    repeats: int,
    time_limit_ms: int | None = None,
    time_limits_ms: Iterable[int] | None = None,
    algorithms: Iterable[str] | None = None,
    neighborhoods: Iterable[str] | None = None,
) -> List[RunConfig]:
    total = count_instances_in_file(instance_file)
    configs: List[RunConfig] = []
    if total == 0:
        return configs
    # Determine list of time limits
    limits: List[int]
    if time_limits_ms:
        limits = list(time_limits_ms)
    elif time_limit_ms is not None:
        limits = [time_limit_ms]
    else:
        limits = [1000]
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
