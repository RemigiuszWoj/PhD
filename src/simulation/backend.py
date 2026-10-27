"""A simulator as the "device" of solve.py, so that ILS runs on it unchanged.

``SimDevice(engine, scenario).counts_batch`` has the signature of
``solve._device_counts_batch(circuits, backend, shots)`` and returns one
shot histogram per circuit, in Qiskit's bit order, exactly as a device would:
every logical circuit is compiled by qiskit-iqm for the Sirius star (in a
compile worker in .venv-iqm, the compiler of the hardware runs), executed by
the engine under scenario S1-S3 and read out with the scenario's readout
errors. ``install(device)`` puts it in place of the device call of solve.py,
the way ``qaoa_depth_study.run_h2`` counts circuits; solve.py itself is not
changed, so ILS is configured with a backend name solve.py accepts ("ideal"),
which the device then ignores.

Only trajectory methods can run every window of a move (up to K = 15, i.e.
16 qubits); a density-matrix engine fails on the first large window.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .noise import REPO, NoiseParams, apply_readout_counts

def iqm_python() -> Path:
    """.venv-iqm of this checkout or, from a worktree, of the main checkout (ANNSIM_IQM_PYTHON overrides)."""
    if os.environ.get("ANNSIM_IQM_PYTHON"):
        return Path(os.environ["ANNSIM_IQM_PYTHON"])
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                            capture_output=True, text=True).stdout.strip()
    for root in (REPO, Path(common).parent if common else REPO):
        p = root / ".venv-iqm" / "bin" / "python"
        if p.exists():
            return p
    raise FileNotFoundError("no .venv-iqm found; set ANNSIM_IQM_PYTHON")


class CompileWorker:
    """One compile_worker process, kept open for the whole run."""

    def __init__(self, python: Path | None = None):
        python = python or iqm_python()
        env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONHASHSEED": "0"}
        self.proc = subprocess.Popen([str(python), "-m", "src.simulation.compile_worker"], cwd=REPO, env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        if not json.loads(self.proc.stdout.readline() or "{}").get("ready"):
            raise RuntimeError(f"compile worker did not start ({python})")

    def compile(self, logical: dict) -> dict:
        self.proc.stdin.write(json.dumps({"logical": logical}) + "\n")
        self.proc.stdin.flush()
        out = json.loads(self.proc.stdout.readline())
        if "error" in out:
            raise RuntimeError(f"compile worker: {out['error']}")
        return out["compiled"]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


class SimDevice:
    def __init__(self, engine: str, scenario: str, worker: CompileWorker | None = None):
        self.engine, self.scenario = engine, scenario
        self.noise = NoiseParams(scenario)
        self.worker = worker or CompileWorker()
        self.stats = {"circuits": 0, "compile_s": 0.0, "sim_s": 0.0}

    def counts_batch(self, circuits, backend=None, shots=4096):
        """Histograms of the measured logical circuits (``backend`` is ignored, see the module doc)."""
        from src.neighborhoods.gate_qaoa import solve as S

        from .engines import run
        from .export import _gate_list

        out = []
        for qc in circuits:
            ops, meas = _gate_list(qc)
            t0 = time.time()
            comp = self.worker.compile({"n": qc.num_qubits, "ops": ops, "measure": meas})
            t1 = time.time()
            seed = int(S._SAMPLER_RNG.integers(2 ** 31)) if S._SAMPLER_RNG is not None else \
                int(np.random.default_rng().integers(2 ** 31))
            r = run(self.engine, comp, self.scenario, shots, seed)
            names = comp["names"]
            errors = self.noise.readout_errors([names[q] for q, _ in sorted(comp["measure"], key=lambda m: m[1])])
            samples = apply_readout_counts(np.asarray(r["samples"], dtype=np.int64), errors,
                                           np.random.default_rng(seed + 1))
            K = len(comp["measure"])
            vals, cnt = np.unique(samples, return_counts=True)
            out.append({format(int(v), f"0{K}b"): int(c) for v, c in zip(vals, cnt)})   # bit k = char K-1-k
            self.stats["circuits"] += 1
            self.stats["compile_s"] += t1 - t0
            self.stats["sim_s"] += time.time() - t1
        return out


def install(device: SimDevice):
    """Route every device call of solve.py to ``device``; returns a function that undoes it."""
    from src.neighborhoods.gate_qaoa import solve as S

    real = S._device_counts_batch
    S._device_counts_batch = device.counts_batch
    return lambda: setattr(S, "_device_counts_batch", real)
