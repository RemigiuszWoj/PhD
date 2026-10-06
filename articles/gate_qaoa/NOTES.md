# Gate-based (QAOA) quantum neighborhoods — notatki do artykułu

Żywy plik z faktami i decyzjami do artykułu o **bramkowych** sąsiedztwach
(QAOA na IBM) — rozszerzenie linii D-Wave (windowed_qubo) na model bramkowy.
Uzupełniać na bieżąco.

## Architektura / sprzęt (do rozdziału "Setup")
- **Platforma:** IBM Quantum Platform (IBM Cloud), plan **Open (darmowy)** —
  10 min QPU / 28 dni okno kroczące + promocyjne 180 min / 12 mies.
- **Backendy dostępne (stan 2026-08):** `ibm_fez`, `ibm_marrakesh`,
  `ibm_kingston` — wszystkie **IBM Heron r2**, **156 kubitów**, topologia
  **heavy-hex** (ibm_fez: 352 krawędzie sprzężeń).
- **Bramki natywne ibm_fez:** dwukubitowa **CZ**; jednokubitowe **RZ, SX, X**
  (+ delay, id, measure, reset, if_else). Implikacja: człony `Z_iZ_j` QUBO
  kompilują się do CZ + rotacji; pary NIEsąsiednie na heavy-hex wymagają
  sieci SWAP → głębszy obwód (patrz "wyzwania").
- **Software (pinned, requirements.txt):** qiskit 2.5.1,
  qiskit-ibm-runtime 0.48.0.
- **Domyślny backend na finalne runy:** `ibm_fez` (least busy przy teście);
  identyfikacja programowa przez `QiskitRuntimeService.backend(...)`.

## Decyzje metodologiczne (do opisania w artykule)
- **Kodowanie: swap-selection QUBO** — TA SAMA macierz Q co na D-Wave
  (analogicznie do poprzednich badań), NIE one-hot permutacji. Zmienne =
  "zastosuj swap k". Małe, rzadkie → gra pod mocne strony bramek. To
  utrzymuje spójność z linią annealingową (te same sąsiedztwa, trzy
  backendy kwantowe: anneal / QAOA / enhanced-windowed).
- **★ Fixed-angle QAOA (WKŁAD METODOLOGICZNY — koniecznie opisać):**
  kąty (γ, β) pre-optymalizowane OFFLINE na symulatorze (lub uniwersalne
  kąty z literatury dla p=1), a na realnym sprzęcie **tylko 1 wywołanie
  QPU na ruch** (sampling gotowego obwodu). Powód: free tier (10 min/28 dni)
  nie udźwignie pełnego wariacyjnego QAOA (dziesiątki–setki wywołań QPU na
  jeden ruch). To bezpośredni analog "1 wywołanie/ruch" z D-Wave i klucz do
  uczciwego porównania wall-clock.
- **Głębokość p:** start p=1 (szum NISQ), p=2 jeśli symulator pokaże zysk.
- **Mikser:** standardowy transverse-field `H_M = Σ X_i` (RX na kubit).
- **Zakres:** wszystkie 4 sąsiedztwa (Adjacent, Fibonacci, Dynasearch,
  Motzkin); zaczynamy od **Fibonacciego** (tridiagonalny → płytki obwód).
- **Rozwój:** symulator-first (Aer, za darmo) → walidacja na zaszumowanym
  fake-Heron → 1 finalny run na `ibm_fez`. Realny QPU tylko na finał.

## Wyzwania do dyskusji (paralela do windowed_qubo)
- **Głębokość obwodu = bramkowy analog "embedding limitu" D-Wave.** Człony
  `Z_iZ_j` na heavy-hex: pary niesąsiednie → SWAP-y → głębiej → więcej szumu.
  Gęste QUBO (dynasearch ~190 zmiennych) trudne; rzadkie (Fibonacci
  tridiagonalny) płytkie i wykonalne. **Kolejność wykonalności jak na D-Wave,
  ale z innego powodu** (głębokość/szum, nie pojemność embeddingu).
- **Metodyka uczciwego wall-clock** przenosi się z windowed_qubo, ale koszt
  bramkowy ma inną strukturę: transpilacja, głębokość, kolejka Leap.
- **Windowing (mamy!)** można reużyć dla gęstych — QAOA na małych oknach =
  płytkie obwody.

## Hipoteza badawcza (wątek na artykuł)
Czy QAOA lepiej radzi sobie z **globalnym zagnieżdżeniem Motzkina** niż
D-Wave? Model bramkowy realizuje dowolne sprzężenia (przez SWAP), bez
sztywnej topologii Pegasusa — więc nesting Motzkina może być tu
naturalniejszy… albo głębokość go zabija. **Do zmierzenia.**

## Figury do artykułu
- **Diagram obwodu QAOA (p=1) dla tridiagonalnego Fibonacciego** —
  `figures/qaoa_circuit_fibonacci.{pdf,png}`. Pokazuje: warstwa startowa H
  (równa superpozycja) → warstwa kosztu RZ(h_i·γ) + ZZ(J·γ, tylko sąsiedzi)
  → mikser RX(2β). Tridiagonalność = cecha Fibonacciego (płytko).
  *(Do dopracowania na wersję finalną: etykieta "H" zamiast U(π/2,0,π).)*

## ★ Tabela międzytopologiczna (2026-09-30) — DARMOWA, kompletna

Powód: konto IBM Cloud zablokowane (IAM: "account is blocked"), więc doszedł drugi
dostawca — **IQM Resonance** (30 kredytów/mies., 1 s runtime = 0,50 kredytu, narzut
kolejki i kompilacji **nie jest** rozliczany). Szczegóły wpięcia: pamięć
`project_iqm_resonance`, patch w `src/neighborhoods/gate_qaoa/solve.py` (backend `iqm`),
osobny venv `.venv-iqm` (bo `iqm-client[qiskit]` wymaga qiskit<2.2, a repo pinuje 2.5.1).

Głębokość i liczba bramek po kompilacji, `tai20_5` inst 0, permutacja startowa
identycznościowa, p=1, window 6, optimization_level=1, transpiler każdego producenta.
Wszystko policzone **bez ani jednego zadania na QPU** — transpilacja jest darmowa,
a topologię prawdziwych maszyn pobiera się z metadanych.

| Sąsiedztwo | Zmienne | heavy-hex `fez` 156q/352kr. | krata `garnet` 20q/30kr. | krata `emerald` 54q/85kr. | gwiazda `sirius` 16q/240kr. |
|---|---|---|---|---|---|
| fibonacci  | 10      | 58 · CZ 14                 | **22** · CZ 14           | **22** · CZ 14            | 38 · CZ 14 + MOVE 14        |
| adjacent   | 10      | 337 · CZ 195               | 200 · CZ 153             | **188** · CZ 159          | 227 · **CZ 90** + MOVE 90   |
| dynasearch | 6 okien | Σ2509 (max 681) · CZ 1668  | Σ1466 (max 391) · CZ 1287| **Σ1383** · CZ 1275       | Σ1758 · **CZ 696** + MOVE 742 |
| motzkin    | 6 okien | Σ2208 (max 601) · CZ 1455  | Σ1280 (max 320) · CZ 1080| **Σ1177** · CZ 1077       | Σ1438 · **CZ 576** + MOVE 604 |

Rozmiary okien dla gęstych: [15, 12, 12, 9, 15, 4]. `fez` = `FakeFez` z
`qiskit_ibm_runtime.fake_provider` (offline, bez konta).

**Wnioski do §3.6 — sekcja przestaje być o jednym sprzęcie:**

1. **Kolejność wykonalności jest identyczna na wszystkich czterech architekturach:**
   fibonacci ≪ adjacent < motzkin < dynasearch. Teza „depth assumes the role that
   embedding capacity plays on annealers" przestaje zależeć od wyboru heavy-hex.
2. **Heavy-hex jest przypadkiem pesymistycznym**, nie reprezentatywnym — krata
   kwadratowa jest 1,7–2,6× płytsza dla każdego sąsiedztwa. Obecny tekst §3.6 trzeba
   przeformułować: opisuje najgorszy przypadek, nie typowy.
3. **Przypadek kontrolny: fibonacci ma CZ = 14 na wszystkich czterech maszynach.**
   Tridiagonalny QUBO nie potrzebuje routingu nigdzie, więc różnice głębokości biorą
   się z routingu, nie z samego obwodu. To domyka argument przyczynowy.
4. **Najmocniejszy pojedynczy wynik: adjacent vs fibonacci na tym samym sprzęcie** —
   te same 10 zmiennych, głębokość **337 vs 58** na heavy-hex, blisko 6×, wyłącznie
   z gęstości grafu konfliktów. Teza wyizolowana w jednej liczbie.
5. **Gwiazda potwierdza hipotezę, ale z niuansem.** Globalna łączność przez centralny
   rezonator redukuje CZ dla gęstych o połowę (696 vs 1668, 576 vs 1455) — nie ma sieci
   SWAP. Płaci jednak MOVE-ami (742, 604), które serializują się przez jeden rezonator,
   więc głębokość wychodzi w środku stawki. Do opisania jako:
   **connectivity buys gate count, not depth.**

Skrypty (scratchpad, do przeniesienia jeśli mają zostać): `depth_capture.py`
(monkeypatch solvera → zrzut prawdziwych Q do JSON + transpilacja na FakeFez),
`depth_iqm.py` (transpilacja tych samych Q na garnet/sirius/emerald/FakeDeneb).
`IQMFakeDeneb` ma tylko 6 kubitów, więc gwiazdy w pełnej skali nie da się zasymulować
lokalnie — liczby dla `sirius` (16 kubitów) pochodzą z metadanych prawdziwej maszyny.

## ★ Faza 2: PRD na trzech architekturach (2026-09-30) — WYNIKI SPRZĘTOWE

`tai20_5` inst 0, start identycznościowy (Cmax 1448, PRD 17,53%), LB 1232,
**dokładnie jeden ruch**, p=1, 4096 shotów, window 6, kąty z `data/qaoa_angles.json`
(ta sama zamrożona tabela co kampania IBM). Jeden ruch, nie pełny budżet 10 s —
koszt jest wtedy deterministyczny, a to i tak reżim, w którym są opublikowane
wiersze `ibm_fez` (tam jeden ruch przekraczał budżet).

| Sąsiedztwo | `ibm_fez` heavy-hex | `garnet` krata | `sirius` gwiazda | CZ (fez→garnet→sirius) |
|---|---|---|---|---|
| adjacent   | 1390 · **12,82%** | 1390 · **12,82%** | 1390 · **12,82%** | 195 → 153 → 90 |
| fibonacci  | 1383 · 12,26%     | 1380 · **12,01%** | 1380 · **12,01%** | 14 → 14 → 14 |
| dynasearch | 1385 · **12,42%** | 1418 · 15,10%     | 1381 · **12,09%** | 1668 → 1287 → 696 |
| motzkin    | 1441 · 16,96%     | 1415 · 14,85%     | 1396 · **13,31%** | 1455 → 1080 → 576 |

Zegar na ruch: `ibm_fez` 13,9 / 27,7 / 27,1 / **169,8** s (fib/adj/dyn/motz),
`garnet` 3,6 / 5,3 / 11,0 / 11,9 s, `sirius` — / — / 12,1 / 11,6 s.

### Co z tego wynika — uczciwie

1. **Motzkin: monotoniczna poprawa zgodna z przewidywaniem.** 16,96 → 14,85 → 13,31
   dokładnie w kolejności malejącej liczby CZ (1455 → 1080 → 576). To jedyne miejsce,
   gdzie predykcja z tabeli głębokości sprawdza się wprost.
2. **Sąsiedztwa rzadkie są niewrażliwe na architekturę** — adjacent daje identyczne 1390
   na obu maszynach, fibonacci 1383 vs 1380 (w granicach szumu). **To nie jest wynik
   zerowy, a przypadek kontrolny, który działa:** fibonacci ma CZ = 14 na każdej
   architekturze, więc nic nie powinno się różnić — i nie różni się.
3. **Dynasearch łamie prostą historię.** `ibm_fez` 12,42%, `garnet` 15,10%, `sirius`
   12,09% — krata wypada NAJGORZEJ, choć jest płytsza od heavy-hex. Przy n=1 instancji
   to może być zwykły szum urządzenia, ale **nie wolno tego zamiatać**: przy jednej
   instancji nie mamy prawa twierdzić, że głębokość przekłada się na jakość monotonicznie.
4. **Gwiazda jest najlepsza albo równa najlepszej na obu gęstych sąsiedztwach.** To
   zgadza się z tezą, że zniesienie sieci SWAP pomaga dokładnie tam, gdzie sieć SWAP
   była potrzebna.
5. **Zegar: IQM jest 2,4–14× szybszy.** Ruch motzkina to 169,8 s na `ibm_fez` przeciw
   11,9 s na `garnet`. Wzmacnia akapit §4.2 o niepewnej konwencji pomiaru: 30 s/ruch
   raportowane dla IBM było własnością obłożenia współdzielonego urządzenia, nie metody.

### POPRAWKA 2026-10-02: efekt jest progowy, nie gradientowy

Uzupełnione wiersze `sirius` dla adjacent i fibonacci (koszt 1,36 kredytu) zmieniają
interpretację i trzeba to poprawić w stosunku do notatki z 30.09.

**Adjacent daje Cmax = 1390 na wszystkich trzech architekturach, co do jednostki**, choć
liczba bramek CZ spada 195 → 153 → 90, czyli ponad dwukrotnie. Fibonacci daje 1380 na
garnecie i siriusie oraz 1383 na fezie. Czyli **liczba bramek dwukubitowych sama z siebie
nie napędza jakości rozwiązania**.

Właściwe sformułowanie: zależność jest **progowa, nie gradientowa**. Poniżej pewnej
głębokości obwodu architektura jest nieistotna — adjacent przy głębokości 200–337 siedzi
po bezpiecznej stronie progu na każdej maszynie. Powyżej progu zaczyna decydować —
motzkin przy sumie głębokości 1280–2208 jest po drugiej stronie i tam zmiana topologii
daje 16,96 → 14,85 → 13,31.

To jest ostrożniejsza teza niż „mniej CZ, lepszy PRD" z notatki z 30.09 i **trzeba pisać
tę, nie tamtą**. Dwa wiersze kontrolne ją skorygowały, zamiast potwierdzić.

Tabela `tab:iqm` nie ma już kresek: cztery sąsiedztwa × trzy architektury, komplet.

### Ograniczenie, które musi trafić do tekstu

**Jedna instancja, jeden ruch, jeden seed.** Wiersze `garnet` i `sirius` są demonstracją
wykonania na drugiej i trzeciej architekturze, nie wynikiem statystycznym. Ciężar
ilościowy niesie tabela głębokości, która jest darmowa i pełna (4 sąsiedztwa × 4 topologie).
Rozszerzenie na 3–10 instancji wymaga odnowienia puli kredytów (30/miesiąc).

### Koszt

0,50 kredytu za sekundę, potwierdzone na trzech zadaniach. **Koszt skaluje się z liczbą
obwodów, nie z głębokością** (~1,5 s = 0,75 kredytu na obwód przy 4096 shotach): motzkin
ma 55× większą sumaryczną głębokość niż fibonacci, a kosztuje 4,5× więcej, bo ma 6 okien
zamiast 1. Wniosek praktyczny: **schodzenie z 4096 shotów nic nie da**, więc
porównywalność z wierszami IBM jest darmowa.
Zużyte w fazach 1–2: ~25,5 z 30 kredytów.

## TODO
- [ ] **Noisy-sim (odłożone; 2026-08-24):** §3.6 artykułu zmiękczone — usunięto
      claim o walidacji na "noisy fake-Heron backend", bo tego kroku nie zrobiliśmy.
      Do dorobienia: aer + model szumu Herona (`qiskit_ibm_runtime.fake_provider`),
      figura/tabela **noiseless → noisy → sprzęt** do §4 lub rozszerzonej wersji.
      Wymaga instalacji `qiskit-aer`. Szczegóły: pamięć `project_gate_noisy_figure`.
- [ ] Szkielet solvera QAOA (`_solve_qaoa` jako backend w common.py) —
      pokazać w czacie przed commitem
- [ ] Faza 0: QAOA dla Fibonacci/Adjacent na Aer — czy znajduje najlepszy ruch
- [ ] Faza 1: pre-optymalizacja kątów (fixed-angle), walidacja fake-Heron
- [ ] Faza 2: 1 run na ibm_fez, porównanie z D-Wave
- [ ] Faza 3: dynasearch/motzkin przez windowing

## GCMM 2026 — cel publikacyjny (SS3), deadline abstraktu 20.08.2026
Sesja SS3 „Quantum-Enhanced Optimization and Intelligent Control" (org. m.in.
Bożejko, Gnatowski, Idzikowski, Kucharski, Rudy — nasza katedra). Tematy SS3
trafiające w nas: „Quantum-Enhanced Scheduling" + „Compilation Techniques for
Gate-Based Superconducting Quantum Computers" → akcentować kompilację/głębokość
na heavy-hex. Springer LNME/LNNS, artykuł 12–15 str., blind. Scope: teoria +
framework, BEZ pisania o testach/symulatorze w abstrakcie.

**Tytuł:** Gate-Model Quantum Neighborhood Search for the Permutation Flow Shop
Problem via Fixed-Angle QAOA

**Abstrakt (EN, ~215 słów, zaakceptowany kierunek):**
The permutation flow shop problem (PFSP) is a canonical NP-hard scheduling
problem whose local-search solvers spend most of their effort evaluating
neighborhood moves. Recent work has offloaded this evaluation to quantum
annealers by casting neighborhoods as QUBO problems, yet the gate-model
paradigm — despite its flexible connectivity — remains largely unexplored for
scheduling. This paper introduces a gate-model framework that evaluates PFSP
neighborhoods with the Quantum Approximate Optimization Algorithm (QAOA)
embedded in classical Iterated Local Search and Simulated Annealing
metaheuristics, giving a hybrid quantum–classical scheme for quantum-enhanced
scheduling. We reuse the swap-selection QUBO encoding from the annealing line,
so a single problem matrix drives either an annealer or a QAOA circuit, and we
adopt a fixed-angle QAOA whose variational angles are pre-optimized offline; on
hardware each move then costs a single circuit execution, keeping the method
viable under NISQ constraints. We formulate all four neighborhood structures
(Adjacent, Fibonacci, Dynasearch, Motzkin) and analyze how their conflict-graph
density governs circuit depth once the QUBO is compiled onto heavy-hex
superconducting hardware. We show that depth assumes the role that embedding
capacity plays on annealers, producing the same feasibility ordering for a
different, compilation-grounded reason, and we discuss windowed decomposition as
a means of taming dense neighborhoods. The paper sets out the evaluation
methodology and the open challenges on the path to on-hardware execution.

**Keywords:** quantum-enhanced scheduling; hybrid quantum–classical
optimization; QUBO; iterated local search; circuit compilation on heavy-hex
hardware; NISQ. Corresponding author: gmail użytkownika.
