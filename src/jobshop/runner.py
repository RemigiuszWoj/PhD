"""One job shop run -> result.json in the format of src/experiments/runner.py.

Same directory naming (algo=...__neigh=...__file=...__inst=...__n..__m..__tl=..ms__seed=..)
and the same fields, so scripts/extract_budgets.py and the aggregation work unchanged.
`instance_file` is a JSPLIB name (ta01..ta80, ft10, ...) or "gen:<n>x<m>:rho=<r>:<k>" for
generated continuum instances (Taillard generator, seeds derived from <k>), or
"pfsp:<file>:<i>" for a Taillard PFSP instance read as a job shop with identical routes.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.jobshop import instance as I
from src.jobshop.algorithms import iterated_local_search, simulated_annealing


@dataclass(frozen=True)
class JSRunConfig:
    algorithm: str          # "ils" | "sa"
    neighborhood: str       # js_* mode
    instance_file: str      # see module docstring
    instance_number: int    # 0 for single-instance files
    seed: int
    time_limit_ms: int
    tabu_tenure: int | None = None
    diversification: str = "mushroom"


SA_DEFAULTS = dict(initial_temp=1000.0, final_temp=100.0, alpha=0.99, reheat_factor=1.5,
                   stagnation_ms=100, temp_floor_factor=1.0)  # = config.yaml simulated_annealing
ILS_DEFAULTS = dict(tabu_tenure=10, mushroom_k=10)             # = config.yaml iterated_local_search


def gen_seeds(n: int, m: int, k: int) -> tuple[int, int]:
    """Deterministic Taillard-style seeds for the k-th generated instance of size n x m."""
    rng = random.Random(1_000_003 * n + 7_919 * m + k)
    return rng.randrange(1, 2**31 - 1), rng.randrange(1, 2**31 - 1)


def load_instance(spec: str, number: int = 0, jsplib_dir=I.JSPLIB_DIR) -> I.Instance:
    if spec.startswith("gen:"):
        _, size, rho, k = spec.split(":")
        n, m = map(int, size.split("x"))
        ts, ms = gen_seeds(n, m, int(k))
        return I.taillard_generate(ts, ms, n, m, float(rho.split("=")[1]), name=spec)
    if spec.startswith("pfsp:"):
        from src.parser import parser
        _, f, i = spec.split(":")
        d = parser(f"data/{f}.txt", int(i))
        return I.from_pfsp(d["processing_times"], spec)
    return I.load(spec, jsplib_dir)


def file_stem(spec: str) -> str:
    return spec.replace(":", "_").replace("=", "").replace(".", "p")


def run_dir_name(cfg: JSRunConfig, n: int, m: int) -> str:
    return (f"algo={cfg.algorithm}__neigh={cfg.neighborhood}__file={file_stem(cfg.instance_file)}"
            f"__inst={cfg.instance_number}__n{n}__m{m}__tl={cfg.time_limit_ms}ms__seed={cfg.seed}")


def already_done(out_dir: Path, cfg: JSRunConfig) -> bool:
    pat = (f"algo={cfg.algorithm}__neigh={cfg.neighborhood}__file={file_stem(cfg.instance_file)}"
           f"__inst={cfg.instance_number}__*__tl={cfg.time_limit_ms}ms__seed={cfg.seed}")
    return any(out_dir.glob(f"{pat}/result.json"))


def run_one(cfg: JSRunConfig, out_dir: Path | None = None, keep_solution: bool = True) -> dict:
    random.seed(cfg.seed)
    inst = load_instance(cfg.instance_file, cfg.instance_number)
    t0 = time.time()
    if cfg.algorithm == "ils":
        res = iterated_local_search(inst, cfg.neighborhood, cfg.time_limit_ms, cfg.seed,
                                    tabu_tenure=cfg.tabu_tenure or ILS_DEFAULTS["tabu_tenure"],
                                    mushroom_k=ILS_DEFAULTS["mushroom_k"], diversification=cfg.diversification)
    elif cfg.algorithm == "sa":
        res = simulated_annealing(inst, cfg.neighborhood, cfg.time_limit_ms, cfg.seed, **SA_DEFAULTS)
    else:
        raise ValueError(cfg.algorithm)
    total_ms = int((time.time() - t0) * 1000)
    upper = I.bks(inst.name) if inst.name.startswith("ta") else None
    lower = inst.lb
    t_hist = res["time_history_ms"]
    d = {
        "config": asdict(cfg),
        "problem": "jssp",
        "cmax_best": res["cmax_best"],
        "time_to_best_ms": t_hist[-1] if t_hist else total_ms,
        "total_time_ms": total_ms,
        "cmax_history": res["cmax_history"],
        "time_history_ms": t_hist,
        "best_sequences": res["best_sequences"] if keep_solution else None,
        "instance_jobs": inst.n,
        "instance_machines": inst.m,
        "upper_bound": upper,
        "lower_bound": lower,
        "qpu_stats": None,
        "iterations": res["iterations"],
        "tl_exceeded": total_ms > cfg.time_limit_ms,
        "overrun_ms": max(0, total_ms - cfg.time_limit_ms),
        "avg_iter_ms": (total_ms / res["iterations"]) if res["iterations"] else None,
        "neigh_time_ms": res["neigh_time_ms"],
        "move_log": [[t, c] for t, c in res["move_log"]],
        "gap_percent": 100.0 * (res["cmax_best"] - lower) / lower,
        "used_qpu_fallback": False,
    }
    if out_dir is not None:
        p = Path(out_dir) / run_dir_name(cfg, inst.n, inst.m) / "result.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    return d
