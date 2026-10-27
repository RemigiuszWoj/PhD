"""Noise scenarios S1-S3 from one Sirius calibration snapshot (numpy only).

Every engine reads its channel parameters from here, so the scenarios are
defined once:

  S1  no noise
  S2  after every gate a depolarizing channel with the measured fidelity of
      that gate on that locus, lambda = d (1 - F) / (d - 1); readout errors
      (the ``calibrated`` mode of fake_sirius.py)
  S3  after every gate thermal relaxation (T1, T2 over the gate duration) of
      each component it acts on, then depolarizing for the rest of the
      infidelity, lambda = d max(0, (1 - F) - (1 - F_relax)) / (d - 1);
      readout errors (the ``thermal`` mode of fake_sirius.py)

Unlike the fake backend, the MOVE gate does get its channel here (the fake
backend replaces MOVE by an unnamed unitary before Aer runs it, so the "move"
errors of its noise model are never applied). Gate durations are the median
over loci per gate name, as in fake_sirius.py. No idle noise in any scenario.

Readout is a classical channel on the outcome and is applied the same way for
every engine (``apply_readout_probs`` on exact distributions,
``apply_readout_counts`` on shots), so it cannot differ between engines.

Depolarizing ``lambda`` is the parameter of E(rho) = (1 - lambda) rho +
lambda I / d (Aer's ``depolarizing_error``). Engines that take the total Pauli
error probability get ``pauli_total(lambda, n) = lambda (4^n - 1) / 4^n``.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("S1", "S2", "S3")
GATES = {"r": "prx", "cz": "cz", "move": "move"}


def latest_snapshot() -> Path:
    files = sorted((REPO / "data").glob("iqm_sirius_calibration_*.json"))
    if not files:
        raise FileNotFoundError("no Sirius calibration snapshot in data/")
    return files[-1]


def relaxation_kraus(t1: float, t2: float, t: float):
    """Kraus operators of amplitude damping (T1) followed by pure dephasing, so
    that populations relax with T1 and coherences decay as exp(-t / T2)."""
    if t2 > 2 * t1:
        raise ValueError(f"T2 = {t2} > 2 T1 = {2 * t1}")
    gamma = 1.0 - np.exp(-t / t1)
    rate_phi = 1.0 / t2 - 1.0 / (2.0 * t1)
    lam = 1.0 - np.exp(-2.0 * t * rate_phi)            # phase damping: coherence x sqrt(1 - lam)
    ad = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]]), np.array([[0, np.sqrt(gamma)], [0, 0]])]
    pd = [np.array([[1, 0], [0, np.sqrt(1 - lam)]]), np.array([[0, 0], [0, np.sqrt(lam)]])]
    return [b @ a for a in ad for b in pd], gamma, lam


def average_gate_fidelity(kraus) -> float:
    d = kraus[0].shape[0]
    f_pro = sum(abs(np.trace(k)) ** 2 for k in kraus) / d ** 2
    return (d * f_pro + 1) / (d + 1)


def pauli_total(lam: float, n: int) -> float:
    """Total Pauli error probability of a depolarizing channel with parameter ``lam``."""
    return lam * (4 ** n - 1) / 4 ** n


class NoiseParams:
    """Channel parameters of one scenario, looked up by gate name and qubit names."""

    def __init__(self, scenario: str, snapshot: Path | None = None):
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario {scenario!r} (one of {SCENARIOS})")
        self.scenario = scenario
        self.snapshot = Path(snapshot or latest_snapshot())
        snap = json.loads(self.snapshot.read_text())
        self.duration = {g: median(e["duration"] for e in snap[g].values()) for g in ("prx", "cz", "move")}
        self.readout = snap["readout"]
        self.t1, self.t2 = snap["t1"], snap["t2"]
        self._depol = {}
        for g in ("prx", "cz", "move"):
            for locus, e in snap[g].items():
                names = tuple(locus.split("|"))
                d = 2 ** len(names)
                relax_inf = 0.0
                if scenario == "S3":
                    ks = [relaxation_kraus(self.t1[c], self.t2[c], self.duration[g])[0] for c in names]
                    full = ks[0] if len(ks) == 1 else [np.kron(a, b) for a in ks[0] for b in ks[1]]
                    relax_inf = 1.0 - average_gate_fidelity(full)
                self._depol[(g, names)] = d * max(0.0, (1.0 - e["fidelity"]) - relax_inf) / (d - 1)

    @property
    def noisy(self) -> bool:
        return self.scenario != "S1"

    def depolarizing(self, gate: str, names) -> float:
        """lambda of the depolarizing channel after ``gate`` (r | cz | move) on ``names``."""
        if not self.noisy:
            return 0.0
        g = GATES[gate]
        key = (g, tuple(names))
        if key not in self._depol:                 # cz / move loci are stored as (qubit, resonator)
            key = (g, tuple(reversed(names)))
        return self._depol[key]

    def relaxation(self, gate: str, name: str):
        """(gamma, lam): amplitude- and phase-damping parameters of ``name`` during ``gate``,
        or None outside S3."""
        if self.scenario != "S3":
            return None
        _, gamma, lam = relaxation_kraus(self.t1[name], self.t2[name], self.duration[GATES[gate]])
        return gamma, lam

    def readout_errors(self, names):
        """[(P(1|0), P(0|1)), ...] for the measured qubits, None in S1."""
        if not self.noisy:
            return None
        return [(self.readout[q]["0"], self.readout[q]["1"]) for q in names]


def apply_readout_probs(probs: np.ndarray, errors) -> np.ndarray:
    """Exact readout channel on a distribution indexed by bit k = classical bit k."""
    if errors is None:
        return probs
    K = len(errors)
    p = probs.reshape([2] * K)                       # axis j = bit K-1-j (C order)
    for k, (e0, e1) in enumerate(errors):
        m = np.array([[1 - e0, e1], [e0, 1 - e1]])   # m[read, true]
        p = np.moveaxis(np.tensordot(m, p, axes=([1], [K - 1 - k])), 0, K - 1 - k)
    return p.reshape(-1)


def apply_readout_counts(samples: np.ndarray, errors, rng) -> np.ndarray:
    """Readout channel on sampled outcomes (integers, bit k = classical bit k)."""
    if errors is None:
        return samples
    out = samples.copy()
    for k, (e0, e1) in enumerate(errors):
        bit = (samples >> k) & 1
        flip = rng.random(samples.shape) < np.where(bit == 1, e1, e0)
        out ^= (flip.astype(samples.dtype) << k)
    return out
