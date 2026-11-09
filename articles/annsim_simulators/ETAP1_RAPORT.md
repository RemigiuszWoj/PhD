# ANNSIM — etap 1 i początek etapu 2: raport z nocy 26/27.10.2026

Praca na gałęzi `annsim-simulators` (od `icaisc-p-study`, `1bbbeee`) w worktree
`.claude/worktrees/annsim-simulators`. Nic nie jest zacommitowane ani wypchnięte. Nie wysłałem żadnych zadań
na sprzęt. Główny katalog i jego gałąź są nietknięte.

## W skrócie

Zainstalowałem i sprawdziłem w praktyce wszystkie 10 programów. Dziewięć liczy nasz obwód QAOA; Stim odpada,
bo obsługuje tylko obwody Clifforda. W S1, przy tym samym skompilowanym obwodzie, każdy program zgadza się
z backendem `ideal` na całym rdzeniu n = 20 (213 obwodów). Przy szumie wszystkie silniki macierzy gęstości
dają ten sam rozkład co do 1e-14, a trajektorie mieszczą się w szumie shotów. Wyjątkiem są błędy samych
bibliotek, które opisuję niżej. Kompilator zmienia rozkład o rząd wielkości więcej niż silnik (tor B), co
potwierdza hipotezę RQ2.

Najważniejsze dla ICAISC: **fałszywy Sirius w ogóle nie nakłada szumu na bramki MOVE**, a w naszych obwodach
MOVE jest tyle co CZ albo więcej. W E4/E5n szum dwukubitowy był więc zaniżony; rozkład okna przesuwa się
o TVD 0,016–0,044, a P_opt o 11–16% (sprawdzone na trzech oknach).

## Co działa, a co nie (program po programie)

| Program (wersja) | Metody z szumem | S1 | S2 | S3 | Czas S2: K=9 / 12 / 15 | Uwagi |
|---|---|---|---|---|---|---|
| Qiskit Aer 0.18.2 | trajektorie, macierz gęstości (DM) | tak | tak | DM tak, trajektorie **nie** | traj. 2 / 13 / 305 s; DM 2,5 / 304 s / — | trajektorie w S3: „Kraus is empty” (błąd Aer, niżej) |
| Cirq 1.7 + qsim 0.22.1 | qsim: trajektorie; Cirq: DM | tak (qsim float32) | tak | tak | qsim 1,9 / 10 / 135 s; Cirq DM 30 s / > 1 h / — | najlepszy kandydat do pełnego zakresu |
| Qulacs 0.6.14 | trajektorie (NoiseSimulator), DM | tak | tak | NoiseSimulator **nie**, pętla po shotach tak, DM tak | 0,3 / 3,8 / 70 s; DM 21 s / > 1 h / — | najszybszy; błąd NoiseSimulator i odwrotny znak obrotów |
| MQT DDSIM 2.7.0 | szum tylko globalny (S2≈, S3≈) | tak | **nie** | **nie** | — | oba symulatory z szumem błędne już przy p = 0; wywrotki od K ≈ 10 |
| Qibo 0.3.6 + qibojit | trajektorie, DM | tak | tak | tylko DM | traj. 70 / 120 / 427 s; DM 1,6 / 1048 s / — | trajektorie nie biorą tłumienia amplitudy; błąd ThermalRelaxationChannel |
| CUDA-Q 0.16 (CPU, Mac) | tylko DM | tak | tak | tak | DM 1860 s (!) / — / — | brak trajektorii na CPU; dwie pułapki API |
| PennyLane 0.45.1 | DM (default.mixed) | tak | tak | tak | DM 111 s / > 1 h / — | wolny |
| Amazon Braket (lokalny) | DM (braket_dm) | tak | tak | tak | DM 2,4 / 163 s / — | najszybsza macierz gęstości |
| QuTiP 5.3.1 + qutip-qip | szum tylko na poziomie impulsów | tak | — | — | — | demo impulsowe: SCQubits, K = 4, 45 s |
| Stim 1.16 | — | **nie** | — | — | — | tylko obwody Clifforda |

„Tak” znaczy: zgodność z referencją sprawdzona komórka po komórce, czyli dokładnie (DM, TVD ≤ 1e-14) albo
w granicy 95. percentyla podłogi szumu shotów (trajektorie). Czasy to jeden wątek, 4096 shotów, p = 1.
Pełna tabela z pamięcią jest w `measurements/stage1_table.md`. Macierze gęstości ograniczyłem do 13 kubitów
(K ≤ 12); przy 16 kubitach potrzebowałyby 64 GB.

## Etap 2: rdzeń n = 20 (tai20_5/10/20, inst. 0–3, 71 okien z K ≤ 11)

Liczyłem od 23:20 do 07:10, w sumie około 7000 komórek. Pełne tabele są w `measurements/stage2_table.md`.

- **S1**, 213 obwodów (p = 1–3) × 14 silników, plus `qulacs_loop` dla p = 1: wszystkie komórki przeszły.
  Silniki dokładne różnią się od referencji o ≤ 4e-14 (qsim o 3e-6, bo liczy we float32). Silniki
  próbkujące mieszczą się w podłodze w 93–97% komórek, czyli tyle, ile wynika z 95. percentyla.
- **S2**: wszystkie silniki macierzy gęstości zgadzają się co do ≤ 5e-14. Trajektorie Aer, qsim, Qulacs
  (oba warianty) i Qibo mieszczą się w podłodze w 65–70 z 71 okien przy każdym p. Stochastyczny DDSIM
  daje zły rozkład (1 z 64) i wywraca się w części okien.
- **S3**: macierze gęstości zgadzają się co do ≤ 2e-8; to rozbieżność `thermal_relaxation_error` z Aer
  wobec postaci Krausa, bez znaczenia. qsim mieści się w podłodze w 67–68 z 71 okien, a Qulacs z pętlą po
  shotach w 67 z 71. Wbudowany NoiseSimulator Qulacs tylko w 20–31 z 71 (błąd biblioteki), a trajektorie
  Aer liczą się tylko w 6–16 z 71 okien („Kraus is empty”).
- **Czas** (mediana, S2, p = 1, K = 9–11): Qulacs 0,45 s, qsim 2,4 s, Aer (trajektorie) 3,0 s, Braket DM
  3,7 s, Aer DM 3,8 s, Qibo DM 3,9 s, Qulacs DM 27 s, Cirq DM 33 s, PennyLane 98 s (K ≤ 9), Qibo
  (trajektorie) 196 s. CUDA-Q liczyłem tylko do K ≤ 6, bo przy K = 9 potrzebuje 31 min na okno.
- **Niedokończone** (przerwałem o 07:10, żeby procesor był wolny na H2): S2/S3 dla p = 2–3 w CUDA-Q,
  PennyLane i `qulacs_loop`; S3 dla p = 2 w Cirq DM i Qulacs DM częściowo; p = 3 w Cirq DM, Qulacs DM
  i Qibo w ogóle. Driver po wznowieniu pomija gotowe komórki.

Wniosek dla RQ1: przy identycznym obwodzie i modelu szumu programy, które działają poprawnie, zgadzają się
ze sobą do błędu zaokrągleń (macierze gęstości) albo szumu shotów (trajektorie). Każda różnica większa niż
to okazała się błędem biblioteki albo adaptera.

## Tor B: kompilatory

- **qiskit-iqm** (jak dotąd) i **cirq-iqm** (iqm-client 36, urządzenie zbudowane offline z migawki
  kalibracji, bez łączenia z IQM) skompilowały wszystkie 483 obwody. Liczby CZ i PRX są identyczne, ale
  cirq-iqm wstawia więcej MOVE: mediana ×1,5–1,75 dla sąsiedztw gęstych przy p = 2–3.
- W S2 ten sam obwód logiczny po dwóch kompilatorach daje rozkłady różniące się o TVD 0,010 / 0,043 /
  0,050 (K = 4 / 9 / 12). Silniki przy tym samym obwodzie różnią się o 1e-15. Rozkład z cirq-iqm jest dalej
  od idealnego (TVD 0,322 wobec 0,298 przy K = 12).
- **TKET odpada**: pytket-iqm 0.18 (najnowszy) jawnie odmawia urządzeń z rezonatorem obliczeniowym i wymaga
  połączenia z IQM już przy tworzeniu backendu.

## Znalezione błędy i pułapki

W naszym kodzie (ICAISC, nic nie poprawiałem):
1. `IQMFakeBackend.run` zamienia `move` na nienazwaną `UnitaryGate`, więc błędy „move” z modelu szumu nigdy nie
   są nakładane. 200 bramek MOVE zostawia P(1) bez zmian, a 200 CZ depolaryzuje stan. Mój S2 to dokładnie
   fałszywy Sirius z włączonym MOVE (TVD 1e-12). Ilościowo: `measurements/fake_sirius_move_check.jsonl`.
2. `IQMFakeBackend.run` przekazuje do Aer tylko `shots`, a `seed_simulator` gubi, więc E4/E5n nie są
   powtarzalne co do ziarna.
3. `iqm_sirius_snapshot.py` trzyma jedną implementację na locus: dla CZ na QB2 i QB10 zapisała się niedomyślna
   `tgss`.
4. Sprawdzone i w porządku: `find_bit(q).index` w obwodach z `transpile_to_IQM` bywa przesunięty, ale Aer używa
   pozycji w `qc.qubits`, więc szum CZ w E4/E5n trafiał na właściwe kubity.

W bibliotekach (kandydaci na zgłoszenie upstream, decyzja Twoja):
5. **MQT DDSIM 2.7.0**: StochasticNoiseSimulator przy dwóch bramkach parametrycznych tego samego typu na
   kubicie bierze kąt pierwszej (rx(0,9) rx(1,3) daje rx(1,8)). DeterministicNoiseSimulator przy próbkowaniu
   gubi wyniki o p < ~0,4%. Oba wywracają się od K ≈ 10.
6. **Qulacs 0.6.14**: NoiseSimulator nigdy nie wybiera gałęzi rozpadu w AmplitudeDampingNoise (|1⟩ zostaje
   |1⟩). Do tego RX, RZ i PauliRotation mają znak exp(+iθ/2), odwrotny niż Qiskit i Cirq. Mój adapter miał
   przez to błąd, który wyszedł dopiero na obwodach z cirq-iqm; poprawiony.
7. **Qibo 0.3.6**: ThermalRelaxationChannel w formie T2 ≤ T1 jest zły na kubicie ≠ 0 splątanej macierzy
   gęstości (błąd do 4e-2).
8. **Qiskit Aer 0.17.2 i 0.18.2**: trajektorie w S3 kończą się „Kraus is empty”, gdy na rezonatorze spotkają
   się relaksacja w postaci Krausa (QB2, T2 > T1) i relaksacja złożona z depolaryzacją. Każda osobno działa;
   macierz gęstości działa. Reprodukcja: `measurements/aer_kraus_empty_repro.txt`.
9. **CUDA-Q 0.16**: `apply_noise` jest po cichu ignorowane bez ustawionego modelu szumu; `set_noise` z obiektem
   tymczasowym daje segfault; `add_channel` nie przyjmuje „swap” (MOVE).

Wniosek do metodyki: test S1 na samym obwodzie logicznym nie wykrywa globalnego odwrócenia znaku obrotów,
bo prawdopodobieństwa się nie zmieniają. Potrzebny jest obwód, który miesza bramki natywne z jawnymi
macierzami, a tor B daje taki obwód za darmo.

## Zmiany w kodzie do przejrzenia (nic nie zacommitowane)

Same nowe pliki w `src/simulation/`; istniejącego kodu nie zmieniałem.

| Plik | Venv | Rola |
|---|---|---|
| `__init__.py` | — | opis pakietu |
| `export.py` | `.venv-iqm` | okna E4 → obwód jak w `solve.py` → `transpile_to_IQM` → JSON (logiczny i skompilowany) |
| `noise.py` | dowolny | S1–S3 z migawki (numpy), odczyt jako kanał klasyczny, wspólny dla wszystkich |
| `engines.py` | `.venv-annsim` | 16 adapterów (10 programów, warianty metod) |
| `cell.py` | `.venv-annsim` | jedna komórka (lub wsad) w osobnym procesie: czas, RSS, TVD, podłogi, próbki do `.npz` |
| `stage1.py`, `stage2.py` | `.venv-annsim` | drivery etapów 1 i 2, podsumowania, tabela Markdown |
| `compile_cirq.py` | `.venv-annsim-cirq` | tor B: kompilacja cirq-iqm offline z migawki |
| `fake_move_check.py` | `.venv-iqm` | ile znaczy brak szumu MOVE w fałszywym Siriusie |

Dane i notatki: `articles/annsim_simulators/measurements/` (obwody, referencje, próbki, JSONL, logi,
`env/requirements-*.txt`, notatki o błędach). W worktree dodałem dowiązania `data/tai*.txt` (ignorowane przez
git). Nowe venv w katalogu głównym: `.venv-annsim`, `.venv-annsim-cirq`, `.venv-annsim-tket`. Każdy ma własny
`.gitignore` z `*`, więc `.gitignore` repo został bez zmian.

## Decyzje do potwierdzenia

1. **Format toru A**: neutralna lista bramek w JSON zamiast OpenQASM z planu. MOVE nie ma w QASM, a import QASM
   też różni się między programami.
2. **Definicja S3**: tak jak tryb `thermal` (relaksacja plus depolaryzacja tylko na resztę błędu), a nie
   „S2 + relaksacja” dodawane wprost.
3. **MOVE z szumem w S2 i S3**: tak (inaczej niż fałszywy Sirius). Czy w ICAISC poprawiamy `fake_sirius` i
   powtarzamy część E4/E5n, czy opisujemy to jako ograniczenie? Najlepiej w osobnej sesji.
4. **Odczyt** nakładam klasycznie, jednakowo dla wszystkich programów, zamiast używać ich własnych modeli
   odczytu.
5. **Wiersze tabeli** to pary program–metoda (16 wariantów). Tytuł „Nine Simulators” trzeba policzyć od nowa,
   bo z szumem poprawnie działa 8 programów, a DDSIM i Stim odpadają.
6. **Zgłoszenia upstream** (DDSIM, Qulacs, Qibo, Aer, CUDA-Q): wysłać czy nie? To publiczne działanie, więc
   wymaga Twojej zgody.
7. **TKET** usunąć z toru B czy emulować gwiazdę ogólnym routingiem? Taka emulacja nie byłaby wierna (SWAP
   zamiast MOVE).
8. **Miary dla dużych okien**: TVD po 2^K wyników ma przy K ≥ 12 podłogę 0,3–0,7, a nawet TVD po poziomach
   energii przy K = 15 (7273 poziomy) ma podłogę 0,4. Proponuję jako miary główne P_opt, średnią energię
   z błędem standardowym i energię najlepszej z S próbek, a kryteria wyboru do etapu 3 ustalić teraz: zdane
   S1–S3, czas na obwód przy K = 15 w S2, pamięć.
9. **Protokół czasu**: dzisiejsze czasy mierzyłem przy 4–8 komórkach jednowątkowych naraz (M4 Pro, 8 rdzeni
   wydajnościowych + 4 efektywnościowe). Tabela T6 powinna powstać sekwencyjnie.
10. **Zakres wolnych silników** w etapie 2 (CUDA-Q, PennyLane, Cirq DM, Qulacs DM): K ≤ 9 czy pełne K ≤ 11
    przy wielu godzinach liczenia?
11. **Dane w gicie**: referencje i próbki to dziesiątki MB. Commitować czy trzymać poza repo?

## Proponowany następny krok

Przejrzyj kod i decyzje 1–4 i 8, bo od nich zależy reszta etapu 2. Potem dokończę S2/S3 dla p = 2–3 na
wolniejszych silnikach (około 10 h CPU, mogę puścić, kiedy procesor będzie wolny po H2) i wybiorę 2–3
programy do etapu 3 według ustalonych kryteriów. Kandydaci po tej nocy: qsim i Qulacs (pętla po shotach),
a Aer DM jako dokładna referencja do K = 12. Poprawkę MOVE w `fake_sirius` dla ICAISC warto zrobić osobno.
