# ANNSIM 2027 — przegląd symulatorów (stan 25.10.2026, z dokumentacji, bez instalacji)

Cel artykułu: wszystkie wymienione programy × 3 scenariusze szumu (S1 bez szumu; S2 depolaryzacja z wierności
kalibracji Siriusa + odczyt; S3 = S2 + relaksacja T1/T2), porównane z wynikami IQM Sirius (ACIIDS p = 1, H2 p = 2, 3).
Okna QUBO: 1–16 zmiennych (mediana 10 liniowe, 14 gęste). Macierz gęstości dla 16 kubitów ~64 GB → z szumem przy
15–16 kubitach tylko trajektorie albo diagramy decyzyjne.

| Program | Metoda z szumem | S1 | S2 | S3 | Uwagi | Źródło |
|---|---|---|---|---|---|---|
| Qiskit Aer | trajektorie (statevector), density_matrix, MPS | ✅ | ✅ | ✅ | już działa (fałszywy sirius) | https://qiskit.org/documentation/stubs/qiskit_aer.AerSimulator.html |
| Cirq + qsim | trajektorie (C++) | ✅ | ✅ | ✅ | kompilator IQM w iqm-client[cirq] (cirq-iqm wycofany) | https://quantumai.google/qsim/tutorials/noisy_qsimcirq , https://pypi.org/p/cirq-iqm |
| Qulacs | kanały probabilistyczne (DepolarizingNoise, TwoQubitDepolarizingNoise, DephasingNoise) i CPTP (AmplitudeDampingNoise) | ✅ | ✅ | ✅ | kompilacja przez pytket-iqm | https://docs.qulacs.org/en/latest/guide/2.0_python_advanced.html , https://docs.quantinuum.com/tket/extensions/pytket-iqm |
| MQT DDSIM | diagramy decyzyjne, stochastycznie / deterministycznie | ✅ | ⚠️ | ⚠️ | szum: JEDNO globalne prawdopodobieństwo (A/D/P), nie per bramka | https://mqt.readthedocs.io/projects/ddsim/en/latest/simulators/NoiseAwareSimulator.html |
| Qibo | macierz gęstości przy kanałach | ✅ | ⚠️ | ⚠️ | trajektorie do sprawdzenia | https://arxiv.org/pdf/2009.01845 |
| CUDA-Q | density-matrix-cpu | ✅ | ⚠️ | ⚠️ | trajektorie głównie GPU (brak na Macu) | https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html |
| PennyLane | default.mixed (< ~15 kubitów) | ✅ | ⚠️ | ⚠️ | import modeli szumu z Qiskit | https://pennylane.ai/devices/default-mixed |
| Amazon Braket (lokalnie) | braket_dm (~12 kubitów) | ✅ | ⚠️ | ⚠️ | DM1 w chmurze płatny | https://docs.aws.amazon.com/braket/latest/developerguide/choose-a-simulator.html |
| QuTiP | równania master (poziom impulsów) | ⚠️ | ⚠️ | ⚠️ | inny poziom opisu, bardzo ciężkie przy 16 kubitach | — |
| Stim | tylko obwody Clifforda | ❌ | ❌ | ❌ | QAOA nie jest cliffordowski — odpada | — |

Licencje: wszystkie open source i darmowe (z pamięci, potwierdzić przy instalacji); płatny tylko chmurowy DM1 Braket.

## Proponowany układ
1. Wszystkie programy (bez Stim) na podzbiorze okien K ≤ 12: S1–S3 → pełne porównanie 9 programów.
2. Pełny zakres (okna do 16 kubitów, ILS z ruchami jak Sirius) tylko Aer, qsim, Qulacs, DDSIM.
3. QuTiP: jedno małe okno na poziomie impulsów (ciekawostka). Stim: wymieniony jako odrzucony, z uzasadnieniem.

## Do sprawdzenia przy instalacji (po ICAISC)
Trajektorie w Qibo i CUDA-Q na CPU; star/MOVE w kompilatorach Cirq i TKET dla Siriusa; Python API szumu w DDSIM;
konflikty wersji (osobne venv); licencje.
