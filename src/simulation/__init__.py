"""Gate-model simulators against IQM Sirius (ANNSIM 2027, articles/annsim_simulators).

Track A: one circuit, compiled once by qiskit-iqm for the Sirius star, runs on
every simulator under the same noise scenario, so the differences come from the
simulation engine alone. Track B: the same logical circuit compiled by
qiskit-iqm and by cirq-iqm, both run on one engine (effect of the compiler).

  export          (.venv-iqm)        windows -> logical + compiled circuits, neutral JSON
  noise           (any venv)         scenarios S1-S3 from the calibration snapshot, numpy only
  engines         (.venv-annsim)     one adapter per simulator and method, plus UNSUPPORTED
  cell            (.venv-annsim)     one measurement cell (or a batch) in its own process
  stage1, stage2  (.venv-annsim)     campaign drivers, summaries, Markdown tables
  compile_cirq    (.venv-annsim-cirq) track B: cirq-iqm compilation, offline from the snapshot
  fake_move_check (.venv-iqm)        what the missing MOVE noise of the fake Sirius changes

Run every module from the repository root with ``PYTHONPATH=.``, e.g.
``PYTHONPATH=. .venv-annsim/bin/python -m src.simulation.stage1 --preset stage1``.
"""
