"""One measurement cell: engine x circuit x scenario, run in its own process.

Prints one JSON line with the agreement and cost of the cell. The reference
distribution is read from ``--ref-dir`` (written by ``--make-ref``):

  S1      exact statevector of the LOGICAL circuit (what backend ``ideal``
          samples from), so a compiled-circuit cell also checks the compiler
  S2, S3  exact density matrix of the compiled circuit (Aer, same channels),
          readout applied; only up to ``REF_MAX_QUBITS``

Metrics (distributions over the classical bits, readout applied in S2/S3):
  tvd_exact   total variation distance of the engine's exact distribution
  tvd_shots   TVD of the empirical distribution of ``shots`` samples
  P_opt       probability mass on optimal assignments (exact, and from shots)
  pick_opt    the lowest-QUBO-energy sample is optimal (the ILS readout)
  tvd_E_shots TVD over QUBO energy levels (meaningful at large K, where the
              floor of the TVD over 2^K assignments approaches 1)
  floor_*     median / 95th percentile of the same TVD for ``shots`` samples
              of the reference itself (shot-noise floor)

Raw samples (readout applied) and exact distributions go to
``<measurements>/samples/*.npz``, so any other metric can be computed later.

Time is wall clock in one thread; peak memory is the max RSS of this process.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import sys
import time
import traceback
import zlib
from pathlib import Path

# One thread everywhere (ANNSIM_THREADS for T6). NUMBA_NUM_THREADS stays unset: qibojit asks numba for one thread per
# physical core at start-up and fails under NUMBA_NUM_THREADS=1; the qibo adapter pins it to 1.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("ANNSIM_THREADS", "1"))

import numpy as np  # noqa: E402

from .export import OUT as CIRCUITS  # noqa: E402
from .noise import NoiseParams, apply_readout_counts, apply_readout_probs  # noqa: E402

REF_MAX_QUBITS = 13


def load_circuit(key: str, path: Path = CIRCUITS) -> dict:
    with gzip.open(path, "rt") as fh:
        for c in json.load(fh)["circuits"]:
            if c["key"] == key:
                return c
    raise KeyError(key)


def energies(c) -> np.ndarray:
    K = c["K"]
    pos = {v: k for k, v in enumerate(c["variables"])}
    idx = np.arange(2 ** K)
    x = ((idx[:, None] >> np.arange(K)) & 1).astype(float)
    E = np.zeros(2 ** K)
    for a, b, w in c["Q"]:
        E += w * (x[:, pos[a]] if a == b else x[:, pos[a]] * x[:, pos[b]])
    return E


def readout_errors(c, noise: NoiseParams):
    names = c["compiled"]["names"]
    return noise.readout_errors([names[q] for q, _ in sorted(c["compiled"]["measure"], key=lambda m: m[1])])


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def ref_path(ref_dir: Path, key: str, scenario: str) -> Path:
    return ref_dir / (key.replace("|", "_") + f"_{scenario}.npy")


def make_ref(c, scenario: str, ref_dir: Path) -> dict:
    from .engines import run

    out = ref_path(ref_dir, c["key"], scenario)
    t0 = time.time()
    if scenario == "S1":
        from qiskit.quantum_info import Statevector
        from .engines import _qiskit_circuit, to_clbits_probs

        qc = _qiskit_circuit(c["logical"], None, measure=False)
        probs = to_clbits_probs(Statevector(qc).probabilities(), c["logical"]["n"], c["logical"]["measure"])
    else:
        if c["compiled"]["n"] > REF_MAX_QUBITS:
            return {"key": c["key"], "scenario": scenario, "ref": None,
                    "reason": f"{c['compiled']['n']} qubits > {REF_MAX_QUBITS}"}
        r = run("aer_dm", c["compiled"], scenario, 16, 0)
        probs = apply_readout_probs(r["probs"], readout_errors(c, NoiseParams(scenario)))
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(out, probs)
    return {"key": c["key"], "scenario": scenario, "ref": out.name, "ref_s": time.time() - t0}


def shot_floor(ref, shots, rng, draws=200):
    """Median and 95th percentile TVD of ``shots`` samples drawn from ``ref`` itself."""
    t = [0.5 * np.abs(rng.multinomial(shots, ref) / shots - ref).sum() for _ in range(draws)]
    return float(np.median(t)), float(np.percentile(t, 95))


def energy_levels(E):
    """Index of the QUBO energy level of every assignment (levels sorted ascending)."""
    levels, inv = np.unique(np.round(E, 9), return_inverse=True)
    return levels, inv


def measure_cell(engine, c, which, scenario, shots, ref_dir: Path) -> dict:
    from .engines import run

    seed = seed_of(engine, c["key"], which, scenario)
    row = {"engine": engine, "key": c["key"], "which": which, "scenario": scenario, "K": c["K"],
           "p": c["p"], "neigh": c["neigh"], "n_qubits": c[which]["n"], "shots": shots, "seed": seed}
    if which == "compiled":
        row.update({k: c["compiled"][k] for k in ("cz", "move", "r", "depth")})
    t0 = time.time()
    try:
        r = run(engine, c[which], scenario, shots, seed)
    except Exception as exc:                                    # recorded, not raised: a cell may fail
        row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500],
                    "trace": traceback.format_exc()[-1500:], "wall_s": time.time() - t0})
        return row
    noise = NoiseParams(scenario)
    rng = np.random.default_rng(seed + 1)
    errors = readout_errors(c, noise) if noise.noisy else None
    probs = None if r["probs"] is None else apply_readout_probs(np.asarray(r["probs"], float), errors)
    samples = apply_readout_counts(np.asarray(r["samples"], dtype=np.int64), errors, rng)
    E = energies(c)
    opt = np.abs(E - E.min()) < 1e-9
    emp = np.bincount(samples, minlength=2 ** c["K"]) / len(samples)
    row.update({"ok": True, "method": r["method"], "t_build": r["t_build"], "t_run": r["t_run"],
                "wall_s": time.time() - t0, "n_samples": int(len(samples)),
                "P_opt_shots": float(emp[opt].sum()), "pick_opt": bool(opt[samples[np.argmin(E[samples])]]),
                "P_opt_exact": None if probs is None else float(probs[opt].sum())})
    # Over 2^K assignments the shot-noise floor of the TVD grows towards 1 at K = 15 (32768 outcomes,
    # 4096 shots), so the same comparison is also made over QUBO energy levels, which is what the
    # neighborhood reads out anyway.
    levels, inv = energy_levels(E)
    emp_E = np.bincount(inv, weights=emp, minlength=len(levels))
    row["mean_E_shots"] = float(emp @ E)
    sample_dir = ref_dir.parent / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_dir / (f"{engine}_{c['key']}_{which}_{scenario}".replace("|", "_") + ".npz"),
                        samples=samples.astype(np.uint16 if c["K"] <= 16 else np.uint32),
                        **({} if probs is None else {"probs": probs}))
    rp = ref_path(ref_dir, c["key"], scenario)
    if rp.exists():
        ref = np.load(rp)
        ref_E = np.bincount(inv, weights=ref, minlength=len(levels))
        rng2 = np.random.default_rng(seed + 2)
        row["P_opt_ref"] = float(ref[opt].sum())
        row["mean_E_ref"] = float(ref @ E)
        row["tvd_shots"] = float(0.5 * np.abs(emp - ref).sum())
        row["floor_median"], row["floor_p95"] = shot_floor(ref, len(samples), rng2)
        row["tvd_E_shots"] = float(0.5 * np.abs(emp_E - ref_E).sum())
        row["floor_E_median"], row["floor_E_p95"] = shot_floor(ref_E, len(samples), rng2)
        if probs is not None:
            row["tvd_exact"] = float(0.5 * np.abs(probs - ref).sum())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", help="one circuit")
    ap.add_argument("--keys-file", type=Path, help="several circuits, one key per line, run in this process "
                    "(peak_rss_mb is then the running maximum of the batch)")
    ap.add_argument("--engine")
    ap.add_argument("--which", choices=("logical", "compiled"), default="compiled")
    ap.add_argument("--scenario", choices=("S1", "S2", "S3"), default="S1")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--make-ref", action="store_true")
    ap.add_argument("--circuits", type=Path, default=CIRCUITS, help="export of export.py or compile_cirq.py")
    args = ap.parse_args()
    keys = [args.key] if args.key else [k for k in args.keys_file.read_text().splitlines() if k]
    with gzip.open(args.circuits, "rt") as fh:
        cs = {c["key"]: c for c in json.load(fh)["circuits"] if c["key"] in set(keys)}
    for key in keys:
        c = cs[key]
        if args.make_ref:
            row = make_ref(c, args.scenario, args.ref_dir)
        else:
            row = measure_cell(args.engine, c, args.which, args.scenario, args.shots, args.ref_dir)
        row["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20   # bytes on macOS
        sys.stdout.write("@@ROW " + json.dumps(row) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
