# T6: largest window under the cap (p = 1, one thread, 300 s per circuit)

| engine | max K S1 | max K S2 | max K S3 | t [s] S2, K = 11 | t [s] S2, K = 15 | peak RSS S2 [MB] | note |
|---|---|---|---|---|---|---|---|
| aer_sv | 16 | 14 | 4† | 6.62 | – | 113 | S3: Aer error "Kraus is empty" from K = 5 |
| qsim | 16 | 16 | 15 | 5.8 | 106 | 239 |  |
| qulacs_loop | 16 | 16 | 13 | 2.26 | 57.6 | 106 |  |
| qulacs | 16 | 16 | 13† | 1.69 | 48.2 | 9092 | S3: NoiseSimulator never decays in AmplitudeDampingNoise (wrong distribution) |
| qibo | 16 | 15 | – | 95.6 | 222 | 284 | qibo state vectors take unitary and reset channels only; amplitude damping needs a density matrix |
| aer_dm | 12 | 11 | 11 | 62.9 | – | 366 | density matrix > 13 qubits |
| cirq_dm | 12 | 10 | 11 | – | – | 719 | density matrix > 13 qubits |
| qulacs_dm | 12 | 10 | 11 | – | – | 277 | density matrix > 13 qubits |
| qibo_dm | 12 | 12 | 11 | 30.1 | – | 4630 | density matrix > 13 qubits |
| braket | 16 | 12 | 12 | 26 | – | 6064 | density matrix > 13 qubits |
| pennylane | 16 | 9 | 10 | – | – | 310 | density matrix > 13 qubits |
| cudaq | 16 | 8 | 8 | – | – | 225 | density matrix > 13 qubits |
| ddsim | 16 | – | – | – | – | 0 | MQT DDSIM 2.7.0 noise simulators are wrong already at p = 0 (stochastic: a second parametrised gate of the sam |
| ddsim_dm | 16 | – | – | – | – | 0 | MQT DDSIM 2.7.0 noise simulators are wrong already at p = 0 (stochastic: a second parametrised gate of the sam |
| qutip | 16 | – | – | – | – | 0 | qutip-qip circuits carry no gate-level noise channels (noise only in pulse-level Processor models) |
| stim | – | – | – | – | – | 0 | Stim simulates Clifford circuits only; QAOA rotations by generic angles have no Clifford form |
