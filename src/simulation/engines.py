"""One adapter per simulator: JSON gate list (export.py) -> outcome distribution.

``run(engine, circ, scenario, shots, seed)`` takes one circuit of the export,
``circ = {"n", "ops", "measure", ["names"]}`` (logical: h, rz, rzz, rx;
compiled: r, cz, move), and returns

  probs     exact distribution over the classical bits (bit k = classical bit
            k = variables[k]), or None when the method only samples
  samples   ``shots`` outcomes as integers with the same bit convention
  method    what actually ran (statevector, density matrix, trajectories, DD)
  t_build   seconds to build the engine's circuit (and noise)
  t_run     seconds to simulate and sample

Readout errors are not given to the engines; the driver applies them to
``probs`` and ``samples`` with the same code for all (noise.py).

Gate conventions (all checked against the logical statevector in S1):
R(theta, phi) = RZ(phi) RX(theta) RZ(-phi) up to a global phase,
RZZ(theta) = exp(-i theta/2 Z Z), MOVE = SWAP of qubit and resonator (exact
for these circuits: the resonator is |0> whenever a state moves in, and the
undefined MOVE phase cancels between the two MOVEs of a pair).

Every engine builds its noise from ``NoiseParams``: after each gate, in S3,
amplitude damping and phase damping of every component it touches, then the
depolarizing channel of that gate and locus.
"""
from __future__ import annotations

import time

import numpy as np

from .noise import NoiseParams

ENGINES = ("aer_sv", "aer_dm", "qsim", "cirq_dm", "qulacs", "qulacs_loop", "qulacs_dm", "ddsim", "ddsim_dm",
           "qibo", "qibo_dm", "cudaq", "pennylane", "braket", "qutip", "stim")


# (engine, scenario) cells that cannot run (found in stage 1 and 2); the drivers record the reason
# instead of running them. DDSIM's noise simulators fail the p = 0 check (wrong distributions) and
# crash from K ~ 10 (each crash is a macOS "Python quit unexpectedly" dialog), so they are not rerun.
_DDSIM_NOISE = ("MQT DDSIM 2.7.0 noise simulators are wrong already at p = 0 (stochastic: a second "
                "parametrised gate of the same type on a qubit reuses the first angle; deterministic: "
                "outcomes below ~0.4 % are dropped when sampling) and crash from K ~ 10")
UNSUPPORTED = {
    ("stim", "*"): "Stim simulates Clifford circuits only; QAOA rotations by generic angles have no Clifford form",
    ("qutip", "S2"): "qutip-qip circuits carry no gate-level noise channels (noise only in pulse-level Processor models)",
    ("qutip", "S3"): "qutip-qip circuits carry no gate-level noise channels (noise only in pulse-level Processor models)",
    ("qibo", "S3"): "qibo state vectors take unitary and reset channels only; amplitude damping needs a density matrix",
    ("ddsim", "S2"): _DDSIM_NOISE, ("ddsim", "S3"): _DDSIM_NOISE,
    ("ddsim_dm", "S2"): _DDSIM_NOISE, ("ddsim_dm", "S3"): _DDSIM_NOISE,
}


def unsupported(engine, scenario):
    """Why (engine, scenario) is not run, or None."""
    return UNSUPPORTED.get((engine, scenario)) or UNSUPPORTED.get((engine, "*"))


# ----------------------------------------------------------------- helpers
def r_matrix(theta, phi):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * np.exp(-1j * phi) * s], [-1j * np.exp(1j * phi) * s, c]])


def depol2_kraus(lam):
    """Kraus operators of the two-qubit depolarizing channel with parameter lam."""
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    w = lam / 16
    return [np.sqrt(1 - 15 * w) * np.eye(4)] + [np.sqrt(w) * np.kron(a, b)
                                                for i, a in enumerate(P) for j, b in enumerate(P) if i or j]


def damping_kraus(gamma, lam):
    ad = [np.array([[1, 0], [0, np.sqrt(1 - gamma)]]), np.array([[0, np.sqrt(gamma)], [0, 0]])]
    pd = [np.array([[1, 0], [0, np.sqrt(1 - lam)]]), np.array([[0, 0], [0, np.sqrt(lam)]])]
    return ad, pd


def noise_after(op, names, noise: NoiseParams):
    """[(kind, qubits, param)] applied after ``op``: ('relax', [q], (gamma, lam)), ('depol', qs, lam)."""
    name, qs = op[0], op[1]
    if not noise.noisy or name not in ("r", "cz", "move"):
        return []
    out = []
    for q in qs:
        rel = noise.relaxation(name, names[q])
        if rel is not None:
            out.append(("relax", [q], rel))
    lam = noise.depolarizing(name, [names[q] for q in qs])
    if lam > 0:
        out.append(("depol", list(qs), lam))
    return out


def to_clbits_probs(full, n, measure, big_endian=False):
    """Probability vector over the n circuit qubits -> over the classical bits."""
    p = np.asarray(full, dtype=float).reshape([2] * n)
    # axis a of the C-order reshape holds qubit n-1-a (little endian) or qubit a (big endian)
    axis = {q: (q if big_endian else n - 1 - q) for q in range(n)}
    keep = [q for q, _ in sorted(measure, key=lambda m: m[1])]           # clbit order 0..K-1
    drop = tuple(a for a in range(n) if a not in {axis[q] for q in keep})
    p = p.sum(axis=drop) if drop else p
    remaining = sorted(axis[q] for q in keep)                            # axes left, in order
    order = [remaining.index(axis[q]) for q in reversed(keep)]           # C order: clbit K-1 first
    return np.transpose(p, order).reshape(-1)


def to_clbits_samples(ints, measure):
    """Integers over circuit qubits (bit q = qubit q) -> integers over classical bits."""
    ints = np.asarray(ints, dtype=np.int64)
    out = np.zeros_like(ints)
    for q, c in measure:
        out |= ((ints >> q) & 1) << c
    return out


def bitstrings_to_ints(counts: dict, order):
    """{bitstring: count} -> integers (bit q = qubit q); ``order[i]`` is the qubit of character i."""
    vals = []
    for bs, c in counts.items():
        bs = bs.replace(" ", "")
        v = 0
        for i, ch in enumerate(bs):
            if ch == "1":
                v |= 1 << order[i]
        vals += [v] * int(c)
    return np.array(vals, dtype=np.int64)


def sample_probs(probs, shots, rng):
    probs = np.clip(np.asarray(probs, float), 0, None)
    return rng.choice(len(probs), size=shots, p=probs / probs.sum()).astype(np.int64)


# ---------------------------------------------------------------- engines
def _qiskit_circuit(circ, noise: NoiseParams | None, measure=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RGate

    n = circ["n"]
    qc = QuantumCircuit(n, len(circ["measure"]))
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            qc.h(qs[0])
        elif name == "rz":
            qc.rz(par[0], qs[0])
        elif name == "rx":
            qc.rx(par[0], qs[0])
        elif name == "rzz":
            qc.rzz(par[0], qs[0], qs[1])
        elif name == "r":
            qc.append(RGate(par[0], par[1]), [qs[0]])
        elif name == "cz":
            qc.cz(qs[0], qs[1])
        elif name == "move":
            qc.swap(qs[0], qs[1])
        else:
            raise ValueError(f"gate {name}")
    if measure:
        for q, c in circ["measure"]:
            qc.measure(q, c)
    return qc


def _aer_noise_model(circ, noise: NoiseParams):
    """Aer NoiseModel with the same channels, per instruction locus (r, cz, swap = move).

    Relaxation uses Aer's own ``thermal_relaxation_error`` (the same channel as
    ``noise.relaxation_kraus``: T1 amplitude damping plus dephasing to T2), as a
    tensor product over the gate's qubits, as IQMFakeBackend builds it. Expanding
    a single-qubit Kraus set to two qubits by hand instead made the statevector
    method fail on some loci ("QuantumError: Kraus is empty").
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

    from .noise import GATES

    names = circ["names"]
    nm = NoiseModel(basis_gates=["r", "cz", "swap"])
    seen = set()
    for op in circ["ops"]:
        key = (op[0], tuple(op[1]))
        if key in seen or op[0] not in GATES:
            continue
        seen.add(key)
        err = None
        if noise.scenario == "S3":
            t = noise.duration[GATES[op[0]]]
            rel = [thermal_relaxation_error(noise.t1[names[q]], noise.t2[names[q]], t) for q in op[1]]
            err = rel[0] if len(rel) == 1 else rel[1].tensor(rel[0])     # qubit 0 = right factor
        lam = noise.depolarizing(op[0], [names[q] for q in op[1]])
        if lam > 0:
            dep = depolarizing_error(lam, len(op[1]))
            err = dep if err is None else err.compose(dep)
        if err is not None:
            nm.add_quantum_error(err, {"r": "r", "cz": "cz", "move": "swap"}[op[0]], list(op[1]))
    return nm


def run_aer(circ, noise, shots, seed, method):
    from qiskit_aer import AerSimulator

    t0 = time.time()
    nm = _aer_noise_model(circ, noise) if noise.noisy else None
    if method == "density_matrix":
        qc = _qiskit_circuit(circ, noise, measure=False)
        qc.save_probabilities(qubits=[q for q, _ in sorted(circ["measure"], key=lambda m: m[1])])
    else:
        qc = _qiskit_circuit(circ, noise)
    sim = AerSimulator(method=method, noise_model=nm, max_parallel_threads=1, seed_simulator=seed)
    t1 = time.time()
    res = sim.run(qc, shots=None if method == "density_matrix" else shots).result()
    if method == "density_matrix":
        probs = np.asarray(res.data(0)["probabilities"])     # bit k = k-th listed qubit = clbit k
        samples = sample_probs(probs, shots, np.random.default_rng(seed))
        return {"probs": probs, "samples": samples, "method": "density matrix",
                "t_build": t1 - t0, "t_run": time.time() - t1}
    counts = res.get_counts()
    K = len(circ["measure"])
    samples = bitstrings_to_ints(counts, list(range(K - 1, -1, -1)))  # clbit strings, little endian
    return {"probs": None, "samples": samples,
            "method": "statevector" if not noise.noisy else "trajectories (statevector)",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def _cirq_circuit(circ, noise: NoiseParams):
    import cirq

    n = circ["n"]
    q = cirq.LineQubit.range(n)
    names = circ.get("names")
    ops = []
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            ops.append(cirq.H(q[qs[0]]))
        elif name == "rz":
            ops.append(cirq.rz(par[0])(q[qs[0]]))
        elif name == "rx":
            ops.append(cirq.rx(par[0])(q[qs[0]]))
        elif name == "rzz":
            ops.append(cirq.ZZPowGate(exponent=par[0] / np.pi, global_shift=-0.5)(q[qs[0]], q[qs[1]]))
        elif name == "r":
            ops.append(cirq.PhasedXPowGate(phase_exponent=par[1] / np.pi, exponent=par[0] / np.pi)(q[qs[0]]))
        elif name == "cz":
            ops.append(cirq.CZ(q[qs[0]], q[qs[1]]))
        elif name == "move":
            ops.append(cirq.SWAP(q[qs[0]], q[qs[1]]))
        if names is not None:
            for kind, nq, par2 in noise_after(op, names, noise):
                if kind == "relax":
                    ops.append(cirq.amplitude_damp(par2[0])(q[nq[0]]))
                    ops.append(cirq.phase_damp(par2[1])(q[nq[0]]))
                elif len(nq) == 1:
                    ops.append(cirq.depolarize(3 * par2 / 4)(q[nq[0]]))
                else:
                    ops.append(cirq.depolarize(15 * par2 / 16, n_qubits=2)(q[nq[0]], q[nq[1]]))
    meas_q = [q[m] for m, _ in sorted(circ["measure"], key=lambda m: m[1])]
    return cirq.Circuit(ops), q, meas_q


def run_qsim(circ, noise, shots, seed):
    import cirq
    import qsimcirq

    t0 = time.time()
    c, q, meas_q = _cirq_circuit(circ, noise)
    opts = qsimcirq.QSimOptions(cpu_threads=1)
    sim = qsimcirq.QSimSimulator(qsim_options=opts, seed=seed)
    t1 = time.time()
    if not noise.noisy:
        sv = sim.simulate(c, qubit_order=q).final_state_vector
        probs = to_clbits_probs(np.abs(sv) ** 2, circ["n"], circ["measure"], big_endian=True)
        cm = c + cirq.Circuit(cirq.measure(*meas_q, key="m"))
        res = sim.run(cm, repetitions=shots)
        bits = res.measurements["m"]                          # column k = clbit k
        samples = (bits.astype(np.int64) << np.arange(bits.shape[1])).sum(axis=1)
        return {"probs": probs, "samples": samples, "method": "statevector",
                "t_build": t1 - t0, "t_run": time.time() - t1}
    cm = c + cirq.Circuit(cirq.measure(*meas_q, key="m"))
    res = sim.run(cm, repetitions=shots)
    bits = res.measurements["m"]
    samples = (bits.astype(np.int64) << np.arange(bits.shape[1])).sum(axis=1)
    return {"probs": None, "samples": samples, "method": "trajectories (statevector)",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_cirq_dm(circ, noise, shots, seed):
    import cirq

    t0 = time.time()
    c, q, _ = _cirq_circuit(circ, noise)
    sim = cirq.DensityMatrixSimulator(seed=seed, dtype=np.complex128)
    t1 = time.time()
    rho = sim.simulate(c, qubit_order=q).final_density_matrix
    probs = to_clbits_probs(np.real(np.diag(rho)), circ["n"], circ["measure"], big_endian=True)
    samples = sample_probs(probs, shots, np.random.default_rng(seed))
    return {"probs": probs, "samples": samples, "method": "density matrix",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def _qulacs_circuit(circ, noise: NoiseParams, density=False):
    from qulacs import QuantumCircuit
    from qulacs import gate as G

    n = circ["n"]
    qc = QuantumCircuit(n)
    names = circ.get("names")
    for op in circ["ops"]:
        name, qs, *par = op
        # Qulacs rotations are exp(+i theta/2 P), the opposite sign of Qiskit and Cirq, hence -theta.
        # A circuit made only of such rotations (the logical one) hides the flip: every gate is
        # conjugated, so all probabilities stay the same; a mix with exact matrices (cirq-iqm output:
        # R as a matrix next to virtual RZ) shows it (TVD 0.36 before this fix).
        if name == "h":
            qc.add_gate(G.H(qs[0]))
        elif name == "rz":
            qc.add_gate(G.RZ(qs[0], -par[0]))
        elif name == "rx":
            qc.add_gate(G.RX(qs[0], -par[0]))
        elif name == "rzz":
            qc.add_gate(G.PauliRotation([qs[0], qs[1]], [3, 3], -par[0]))
        elif name == "r":
            qc.add_gate(G.DenseMatrix(qs[0], r_matrix(par[0], par[1])))
        elif name == "cz":
            qc.add_gate(G.CZ(qs[0], qs[1]))
        elif name == "move":
            qc.add_gate(G.SWAP(qs[0], qs[1]))
        if names is not None:
            for kind, nq, par2 in noise_after(op, names, noise):
                if kind == "relax":
                    qc.add_gate(G.AmplitudeDampingNoise(nq[0], par2[0]))
                    # phase damping lam = Z flip with probability (1 - sqrt(1 - lam)) / 2
                    qc.add_gate(G.DephasingNoise(nq[0], (1 - np.sqrt(1 - par2[1])) / 2))
                elif len(nq) == 1:
                    qc.add_gate(G.DepolarizingNoise(nq[0], 3 * par2 / 4))
                else:
                    qc.add_gate(G.TwoQubitDepolarizingNoise(nq[0], nq[1], 15 * par2 / 16))
    return qc


def run_qulacs(circ, noise, shots, seed, density=False, loop=False):
    import qulacs
    from qulacs import NoiseSimulator, QuantumState

    t0 = time.time()
    qc = _qulacs_circuit(circ, noise, density)
    n = circ["n"]
    t1 = time.time()
    if density:
        st = qulacs.DensityMatrix(n)
        st.set_zero_state()
        qc.update_quantum_state(st)
        full = np.real(np.diag(st.get_matrix()))
        probs = to_clbits_probs(full, n, circ["measure"], big_endian=False)
        samples = sample_probs(probs, shots, np.random.default_rng(seed))
        return {"probs": probs, "samples": samples, "method": "density matrix",
                "t_build": t1 - t0, "t_run": time.time() - t1}
    st = QuantumState(n)
    st.set_zero_state()
    if not noise.noisy:
        qc.update_quantum_state(st)
        full = np.abs(st.get_vector()) ** 2
        probs = to_clbits_probs(full, n, circ["measure"], big_endian=False)
        raw = np.array(st.sampling(shots, seed), dtype=np.int64)
        return {"probs": probs, "samples": to_clbits_samples(raw, circ["measure"]),
                "method": "statevector", "t_build": t1 - t0, "t_run": time.time() - t1}
    if loop:
        # One update_quantum_state per shot: every noise gate picks its Kraus branch by its norm on
        # the current state. NoiseSimulator does not: in qulacs 0.6.14 it never takes the decay
        # branch of AmplitudeDampingNoise (|1> stays |1>), so it is wrong in S3 and right in S2.
        raw = np.empty(shots, dtype=np.int64)
        for k in range(shots):
            st.set_zero_state()
            qc.update_quantum_state(st)
            raw[k] = st.sampling(1, seed + k)[0]
        return {"probs": None, "samples": to_clbits_samples(raw, circ["measure"]),
                "method": "trajectories (state vector, one run per shot)",
                "t_build": t1 - t0, "t_run": time.time() - t1}
    raw = np.array(NoiseSimulator(qc, st).execute(shots), dtype=np.int64)
    return {"probs": None, "samples": to_clbits_samples(raw, circ["measure"]),
            "method": "trajectories (NoiseSimulator)", "t_build": t1 - t0, "t_run": time.time() - t1}


def _ddsim_global(circ, noise: NoiseParams):
    """DDSIM takes one probability per effect for the whole circuit: means over the gates of
    this circuit, multi-qubit gates via multi_qubit_gate_factor (scenario marked S2~/S3~)."""
    names = circ["names"]
    one, two, gam, inf2, inf1 = [], [], [], [], []
    for op in circ["ops"]:
        if op[0] not in ("r", "cz", "move"):
            continue
        lam = noise.depolarizing(op[0], [names[q] for q in op[1]])
        rels = [noise.relaxation(op[0], names[q]) for q in op[1]]
        gam += [r[0] for r in rels if r is not None]
        if len(op[1]) == 1:
            one.append(lam)                # DDSIM depolarizes as (1 - p) rho + p I/2: p = lambda
        else:
            two.append(5 * lam / 8)        # per-qubit p with the 2q gate's average infidelity 3 lambda / 4
    p1 = float(np.mean(one)) if one else 0.0
    p2 = float(np.mean(two)) if two else 0.0
    g = float(np.mean(gam)) if gam else 0.0
    return p1, (p2 / p1 if p1 > 0 else 2.0), g


def run_ddsim(circ, noise, shots, seed, deterministic=False):
    from mqt.ddsim import DDSIMProvider

    t0 = time.time()
    qc = _qiskit_circuit(circ, noise)
    prov = DDSIMProvider()
    if not noise.noisy:
        be = prov.get_backend("qasm_simulator")
        t1 = time.time()
        counts = be.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
        method = "DD (statevector)"
    else:
        p1, factor, g = _ddsim_global(circ, noise)
        effects = "D" if noise.scenario == "S2" else "AD"
        be = prov.get_backend("deterministic_noise_simulator" if deterministic else "stochastic_noise_simulator")
        t1 = time.time()
        counts = be.run(qc, shots=shots, seed_simulator=seed, noise_effects=effects,
                        noise_probability=p1, amp_damping_probability=g,
                        multi_qubit_gate_factor=factor).result().get_counts()
        method = ("DD density matrix" if deterministic else "DD trajectories") + \
            f" (global {effects}: p={p1:.2e}, x{factor:.2f}, amp={g:.2e})"
    K = len(circ["measure"])
    samples = bitstrings_to_ints(counts, list(range(K - 1, -1, -1)))
    return {"probs": None, "samples": samples, "method": method,
            "t_build": t1 - t0, "t_run": time.time() - t1}


def _qibo_circuit(circ, noise: NoiseParams, density):
    from qibo import Circuit, gates

    n = circ["n"]
    c = Circuit(n, density_matrix=density)
    names = circ.get("names")
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            c.add(gates.H(qs[0]))
        elif name == "rz":
            c.add(gates.RZ(qs[0], par[0]))
        elif name == "rx":
            c.add(gates.RX(qs[0], par[0]))
        elif name == "rzz":
            c.add(gates.RZZ(qs[0], qs[1], par[0]))
        elif name == "r":
            c.add(gates.Unitary(r_matrix(par[0], par[1]), qs[0]))
        elif name == "cz":
            c.add(gates.CZ(qs[0], qs[1]))
        elif name == "move":
            c.add(gates.SWAP(qs[0], qs[1]))
        if names is not None:
            for kind, nq, par2 in noise_after(op, names, noise):
                if kind == "relax":
                    # Not gates.ThermalRelaxationChannel: in qibo 0.3.6 its T2 <= T1 form (reset + Z
                    # mixture) is wrong on any qubit but 0 of an entangled density matrix (max error
                    # 4e-2 in a 2-qubit test; the T2 > T1 form and ResetChannel alone are right).
                    c.add(gates.AmplitudeDampingChannel(nq[0], par2[0]))
                    c.add(gates.PhaseDampingChannel(nq[0], par2[1]))
                else:
                    c.add(gates.DepolarizingChannel(tuple(nq), par2))
    return c


def run_qibo(circ, noise, shots, seed, density):
    import qibo

    qibo.set_backend("qibojit", platform="numba")
    qibo.set_threads(1)
    qibo.get_backend().set_seed(seed)
    if noise.scenario == "S3" and not density:
        raise NotImplementedError("qibo: amplitude damping needs a density matrix (repeated execution "
                                  "on state vectors takes unitary and reset channels only)")
    t0 = time.time()
    c = _qibo_circuit(circ, noise, density)
    n = circ["n"]
    t1 = time.time()
    if density or not noise.noisy:
        res = c()
        st = np.asarray(res.state())
        full = np.real(np.diag(st)) if density else np.abs(st) ** 2
        probs = to_clbits_probs(full, n, circ["measure"], big_endian=True)
        samples = sample_probs(probs, shots, np.random.default_rng(seed))
        method = "density matrix" if density else "statevector"
        return {"probs": probs, "samples": samples, "method": method,
                "t_build": t1 - t0, "t_run": time.time() - t1}
    from qibo import gates

    meas = [m for m, _ in sorted(circ["measure"], key=lambda m: m[1])]
    c.add(gates.M(*meas))
    res = c(nshots=shots)
    bits = np.asarray(res.samples(binary=True), dtype=np.int64)    # column k = clbit k
    samples = (bits << np.arange(bits.shape[1])).sum(axis=1)
    return {"probs": None, "samples": samples, "method": "trajectories (repeated execution)",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_cudaq(circ, noise, shots, seed):
    """CUDA-Q builder kernel; on a Mac (no GPU) noise runs only on density-matrix-cpu.

    Channels go in with ``apply_noise`` right after each gate. A NoiseModel keyed
    by gate name cannot carry the MOVE channel ("swap" is refused by
    ``add_channel``), and ``apply_noise`` is silently ignored unless some noise
    model is set (before the kernel is built), so an empty one is installed.
    """
    import cudaq

    cudaq.set_random_seed(seed)
    cudaq.set_target("density-matrix-cpu" if noise.noisy else "qpp-cpu")
    model = cudaq.NoiseModel() if noise.noisy else None   # keep the reference: set_noise does not,
    if model is not None:                                   # and a collected model segfaults get_state
        cudaq.set_noise(model)
    t0 = time.time()
    n = circ["n"]
    names = circ.get("names")
    k = cudaq.make_kernel()
    q = k.qalloc(n)
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            k.h(q[qs[0]])
        elif name == "rz":
            k.rz(par[0], q[qs[0]])
        elif name == "rx":
            k.rx(par[0], q[qs[0]])
        elif name == "rzz":
            k.cx(q[qs[0]], q[qs[1]])
            k.rz(par[0], q[qs[1]])
            k.cx(q[qs[0]], q[qs[1]])
        elif name == "r":
            k.rz(-par[1], q[qs[0]])
            k.rx(par[0], q[qs[0]])
            k.rz(par[1], q[qs[0]])
        elif name == "cz":
            k.cz(q[qs[0]], q[qs[1]])
        elif name == "move":
            k.swap(q[qs[0]], q[qs[1]])
        if names is not None:
            for kind, nq, par2 in noise_after(op, names, noise):
                if kind == "relax":
                    k.apply_noise(cudaq.AmplitudeDampingChannel, par2[0], q[nq[0]])
                    k.apply_noise(cudaq.PhaseDamping, par2[1], q[nq[0]])
                elif len(nq) == 1:
                    k.apply_noise(cudaq.DepolarizationChannel, 3 * par2 / 4, q[nq[0]])
                else:
                    k.apply_noise(cudaq.Depolarization2, 15 * par2 / 16, q[nq[0]], q[nq[1]])
    t1 = time.time()
    try:
        st = np.array(cudaq.get_state(k))
        res = cudaq.sample(k, shots_count=shots)
    finally:
        cudaq.unset_noise()
    full = np.real(np.diag(st)) if noise.noisy else np.abs(st.reshape(-1)) ** 2
    counts = {bs: c for bs, c in res.items()}
    raw = bitstrings_to_ints(counts, list(range(n)))                  # character i = qubit i
    probs = to_clbits_probs(full, n, circ["measure"], big_endian=False)   # state index: qubit 0 = LSB
    return {"probs": probs, "samples": to_clbits_samples(raw, circ["measure"]),
            "method": "density matrix" if noise.noisy else "statevector",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_pennylane(circ, noise, shots, seed):
    import pennylane as qml

    n = circ["n"]
    names = circ.get("names")
    t0 = time.time()
    dev = qml.device("default.mixed" if noise.noisy else "default.qubit", wires=n)

    def body():
        for op in circ["ops"]:
            name, qs, *par = op
            if name == "h":
                qml.Hadamard(qs[0])
            elif name == "rz":
                qml.RZ(par[0], qs[0])
            elif name == "rx":
                qml.RX(par[0], qs[0])
            elif name == "rzz":
                qml.IsingZZ(par[0], wires=[qs[0], qs[1]])
            elif name == "r":
                qml.RZ(-par[1], qs[0])
                qml.RX(par[0], qs[0])
                qml.RZ(par[1], qs[0])
            elif name == "cz":
                qml.CZ(wires=[qs[0], qs[1]])
            elif name == "move":
                qml.SWAP(wires=[qs[0], qs[1]])
            if names is not None:
                for kind, nq, par2 in noise_after(op, names, noise):
                    if kind == "relax":
                        qml.AmplitudeDamping(par2[0], wires=nq[0])
                        qml.PhaseDamping(par2[1], wires=nq[0])
                    elif len(nq) == 1:
                        qml.DepolarizingChannel(3 * par2 / 4, wires=nq[0])
                    else:
                        qml.QubitChannel(depol2_kraus(par2), wires=nq)
        return qml.probs(wires=range(n))

    node = qml.QNode(body, dev)
    t1 = time.time()
    full = np.asarray(node())
    probs = to_clbits_probs(full, n, circ["measure"], big_endian=True)
    samples = sample_probs(probs, shots, np.random.default_rng(seed))
    return {"probs": probs, "samples": samples,
            "method": "density matrix" if noise.noisy else "statevector",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_braket(circ, noise, shots, seed):
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator

    n = circ["n"]
    names = circ.get("names")
    t0 = time.time()
    c = Circuit()
    for q in range(n):
        c.i(q)                                    # keep all n qubits in the register
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            c.h(qs[0])
        elif name == "rz":
            c.rz(qs[0], par[0])
        elif name == "rx":
            c.rx(qs[0], par[0])
        elif name == "rzz":
            c.zz(qs[0], qs[1], par[0])
        elif name == "r":
            c.rz(qs[0], -par[1]).rx(qs[0], par[0]).rz(qs[0], par[1])
        elif name == "cz":
            c.cz(qs[0], qs[1])
        elif name == "move":
            c.swap(qs[0], qs[1])
        if names is not None:
            for kind, nq, par2 in noise_after(op, names, noise):
                if kind == "relax":
                    c.amplitude_damping(nq[0], par2[0])
                    c.phase_damping(nq[0], par2[1])
                elif len(nq) == 1:
                    c.depolarizing(nq[0], 3 * par2 / 4)
                else:
                    c.two_qubit_depolarizing(nq[0], nq[1], 15 * par2 / 16)
    c.probability()
    dev = LocalSimulator("braket_dm" if noise.noisy else "braket_sv")
    t1 = time.time()
    full = np.asarray(dev.run(c, shots=0).result().values[0])
    probs = to_clbits_probs(full, n, circ["measure"], big_endian=True)
    samples = sample_probs(probs, shots, np.random.default_rng(seed))
    return {"probs": probs, "samples": samples,
            "method": "density matrix" if noise.noisy else "statevector",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_qutip(circ, noise, shots, seed):
    """S1 only: qutip-qip gate-level circuit; noise in QuTiP lives at the pulse level."""
    if noise.noisy:
        raise NotImplementedError("QuTiP: gate-level channels are not part of qutip-qip circuits "
                                  "(noise only through pulse-level Processor models)")
    import qutip
    from qutip_qip.circuit import QubitCircuit
    from qutip_qip.operations import Gate

    n = circ["n"]
    t0 = time.time()
    qc = QubitCircuit(n)
    for op in circ["ops"]:
        name, qs, *par = op
        if name == "h":
            qc.add_gate("SNOT", targets=[qs[0]])
        elif name == "rz":
            qc.add_gate("RZ", targets=[qs[0]], arg_value=par[0])
        elif name == "rx":
            qc.add_gate("RX", targets=[qs[0]], arg_value=par[0])
        elif name == "rzz":
            qc.add_gate("CNOT", controls=[qs[0]], targets=[qs[1]])
            qc.add_gate("RZ", targets=[qs[1]], arg_value=par[0])
            qc.add_gate("CNOT", controls=[qs[0]], targets=[qs[1]])
        elif name == "r":
            qc.add_gate("RZ", targets=[qs[0]], arg_value=-par[1])
            qc.add_gate("RX", targets=[qs[0]], arg_value=par[0])
            qc.add_gate("RZ", targets=[qs[0]], arg_value=par[1])
        elif name == "cz":
            qc.add_gate("CSIGN", controls=[qs[0]], targets=[qs[1]])
        elif name == "move":
            qc.add_gate("SWAP", targets=[qs[0], qs[1]])
    t1 = time.time()
    psi = qc.run(qutip.basis([2] * n, [0] * n))
    full = np.abs(psi.full().reshape(-1)) ** 2
    probs = to_clbits_probs(full, n, circ["measure"], big_endian=True)
    samples = sample_probs(probs, shots, np.random.default_rng(seed))
    return {"probs": probs, "samples": samples, "method": "statevector (qutip-qip)",
            "t_build": t1 - t0, "t_run": time.time() - t1}


def run_stim(circ, noise, shots, seed):
    """Stim simulates stabilizer circuits only; a QAOA rotation by a generic angle has no
    Clifford form, so the circuit cannot be written in Stim at all."""
    import stim

    c = stim.Circuit()
    for op in circ["ops"]:
        name, qs, *par = op
        clifford = {"h": "H", "cz": "CZ", "move": "SWAP"}
        if name in clifford:
            c.append(clifford[name], qs)
            continue
        angle = par[0]
        if abs((angle / (np.pi / 2)) - round(angle / (np.pi / 2))) > 1e-9:
            raise NotImplementedError(f"Stim: {name}({angle:.4f}) is not a Clifford rotation")
    raise NotImplementedError("Stim: circuit is Clifford (unexpected for QAOA)")


def run(engine: str, circ: dict, scenario: str, shots: int, seed: int) -> dict:
    noise = NoiseParams(scenario)
    if noise.noisy and "names" not in circ:
        raise ValueError("noise needs the compiled circuit (calibration names)")
    if engine == "aer_sv":
        return run_aer(circ, noise, shots, seed, "statevector")
    if engine == "aer_dm":
        return run_aer(circ, noise, shots, seed, "density_matrix")
    if engine == "qsim":
        return run_qsim(circ, noise, shots, seed)
    if engine == "cirq_dm":
        return run_cirq_dm(circ, noise, shots, seed)
    if engine == "qulacs":
        return run_qulacs(circ, noise, shots, seed)
    if engine == "qulacs_dm":
        return run_qulacs(circ, noise, shots, seed, density=True)
    if engine == "qulacs_loop":
        return run_qulacs(circ, noise, shots, seed, loop=True)
    if engine == "ddsim":
        return run_ddsim(circ, noise, shots, seed)
    if engine == "ddsim_dm":
        return run_ddsim(circ, noise, shots, seed, deterministic=True)
    if engine == "qibo":
        return run_qibo(circ, noise, shots, seed, density=False)
    if engine == "qibo_dm":
        return run_qibo(circ, noise, shots, seed, density=True)
    if engine == "cudaq":
        return run_cudaq(circ, noise, shots, seed)
    if engine == "pennylane":
        return run_pennylane(circ, noise, shots, seed)
    if engine == "braket":
        return run_braket(circ, noise, shots, seed)
    if engine == "qutip":
        return run_qutip(circ, noise, shots, seed)
    if engine == "stim":
        return run_stim(circ, noise, shots, seed)
    raise ValueError(f"engine {engine!r} (one of {ENGINES})")
