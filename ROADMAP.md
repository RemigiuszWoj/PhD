# Roadmapa badań — H2 2026

Stan wyjściowy (2026-07-14): windowed_qubo kompletny (czeka na review
promotora + decyzję o dziurach Table 6 — patrz articles/windowed_qubo/TODO.md).
Pipeline: 12 sąsiedztw × ILS/SA × QUBO na D-Wave, instrumentacja iteracji,
budżetowanie QPU transzami, Taillard do n=500.

## Kotwica: cele projektu doktorskiego (szkola_doktorska_2026/projekt_doktorski)
1. Hipoteza delta-filter + okienkowa dekompozycja — ✅ ZWERYFIKOWANA
   (windowed_qubo, Gantt poz. 1 domknięta przed startem S1).
2. Empiryczne porównanie wyżarzania (D-Wave) i QAOA na identycznych
   instancjach PFSP — Gantt poz. 2 (S1–S2) → front A + artykuły D/E.
3. Granica przewagi kwantowej (rozmiar × struktura sąsiedztwa) — częściowo
   dostarczona (tabela skalowania, luka 20.4→4.6 pp); pełna "mapa" wymaga QAOA.
4. Otwarty framework benchmarkowy — upublicznienie repo (obiecane w Data
   availability windowed_qubo) + protokół transz/anatomia kosztu QPU.
Gantt dalej: 3. środowisko Odra 5 (S2–S3), 4. nowe sąsiedztwa (S3–S4),
5. luka BQP, 6. hybrydy QC, 7. benchmarking i złożoność, 8. rozprawa.

## Fronty

### A. QAOA / komputery bramkowe — PRIORYTET (Gantt 2, cel 2)
Te same QUBO/Ising (H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j — wzór już w artykule)
jako obwody QAOA.
- Stan: src/qaoa skasowane ~3 maja (jest w historii gita); decyzja
  restore-vs-rewrite otwarta od audytu — przy obecnym stanie pipeline'u
  sensowniejszy rewrite pod interfejs solve_qubo.
- Start od Fibonacciego: tridiagonalny Hamiltonian → płytki obwód,
  topologia łańcuchowa, n−1 kubitów (n=20 → 19; n=100 → 99 — w zasięgu
  IBM Eagle/Heron).
- Sprzęt: symulator (qiskit-aer/pennylane) → IBM Quantum open plan →
  środowisko Odra 5 (Gantt poz. 3, gdy dostępne na PWr). W tekstach pisać
  "gate-based", nie "Willow" (niedostępny publicznie; projekt mówi "np. Willow").
- Kroki: (1) szkielet qaoa/ w src z tym samym interfejsem co solve_qubo,
  ansatz p=1..3, symulator; (2) QAOA vs annealer vs SA-sampler na tych
  samych QUBO (cel 2 wprost); (3) mały bieg sprzętowy.

### C. Nowe otoczenia — CZTERY osobne implementacje (Gantt 4 — do przodu planu)
Pełny reuse pipeline'u; równolegle z A, bo nie dotyka QPU (symulator +
klasyka). W projekcie doktorskim wymienione: Hamming i de Montmort;
użytkownik dokłada rodzinę k-bonacci z tribonaccim na czele.
1. **Tribonacci** (trudność NISKA, kilka dni): obok swapów sąsiednich
   3-cykle na pozycjach (i,i+1,i+2); niekolidujące podzbiory →
   T(n)=T(n−1)+T(n−2)+T(n−3); DP liniowe; QUBO wstęgowe (szer. 2) —
   embeduje jak fibonacci, skaluje do n=500. Pierwsza konkretna instancja
   rodziny k-bonacci.
2. **k-bonacci** (NISKA/ŚREDNIA): uogólnienie — ruchy/cykle do długości k
   (k=2 ⇒ Fibonacci, k=3 ⇒ Tribonacci); jedno DP, jeden dowód zliczania,
   strojony parametr bogactwa ruchu vs koszt. Osobny moduł parametryzowany
   k, nie kopia per k. Oś artykułu: "sequence-counted neighborhood family".
3. **Hamming** (ŚREDNIA): ruchy o ograniczonym dystansie Hamminga d
   (wybór d pozycji do optymalnej repermutacji); QUBO przypisaniowe
   ~d² zmiennych (gęstsze — sprawdzić granicę embeddingu).
4. **de Montmort / deranżacje** (PROJEKTOWA — najpierw sesja koncepcyjna
   z promotorem): ruchy deranżacyjne na wybranym podzbiorze pozycji
   (żadna pozycja nie zostaje na miejscu); zliczanie = podsilnia !d;
   struktura QUBO nieoczywista (przypisanie z zakazem diagonali) —
   zaprojektować przed wyceną.
- Publikacja: ICAISC/SOCO 2027 albo sekcje w artykule D.

### D. Głęboki artykuł o Fibonaccim = wehikuł celu 2
Jedno otoczenie, wszystkie realizacje: klasyczne DP (+top-k), QUBO na
annealerze (dane już są), QAOA (z A), teoria (F_{n+1}, tridiagonalność,
trywialny embedding), skalowanie n=500. To jest naturalne miejsce
pierwszego porównania annealer-vs-QAOA na identycznych instancjach.
Target: journal (Quantum Information Processing / COR).

### E. Głęboki artykuł o Motzkinie
Analogicznie: kombinatoryka (M_n, Catalan), DP O(n^3) + akceleracja,
gęstość grafu konfliktów vs embedding, strata okienkowania (zmierzona!),
QAOA dla gęstszych Hamiltonianów (koszt SWAP-ów — ciekawy kontrast z D).

### F. Środowisko Odra 5 / IQM Spark (Gantt 3, S2–S3)
Lokalna 5-kubitowa maszyna IQM Spark na PWr — bez latencji sieciowej.
- Implementacja ŁATWA: adapter qiskit-iqm, te same obwody co w A,
  transpilacja do bramek natywnych; okna w=6 (K=5) przez istniejący
  parametr window_size.
- Wartość naukowa: znika 2.1 s latencji Leap → pierwszy pomiar
  QPU-in-the-loop z realną liczbą iteracji (bezpośredni test modelu
  "budżet kupuje iteracje" z windowed_qubo). 5 kubitów ogranicza do
  mikro-okien — to feature dla metodyki, nie bug.
- Blokada: dostęp/formalności, nie kod. Czekać na udostępnienie.

### G. Metodyka złożoności i benchmarkowania QPU (Gantt 5+7, rezultat 2)
Trudność NISKA — embrion już istnieje: instrumentacja iteracji per-run,
14k+ rekordów qpu_timing.jsonl, tabela anatomii kosztu (30 ms billed vs
2.1–140 s wall), protokół transz, reżim 1-iteracji, stochastyczność
embeddingu.
- Do sformalizowania: model kosztu oraculum QPU
  (t_build + t_embed + t_queue + t_anneal, amortyzacja embeddingu);
  trzy reżimy uczciwego porównania (równy wall / równy billed /
  równe iteracje); notacja złożoności pętli hybrydowych; uczciwe
  ujęcie BQP (oraculum heurystyczne ≠ teza o klasie złożoności).
- Kandydat na samodzielny artykuł metodyczny prawie bez nowych
  eksperymentów — najtańsza wysokowartościowa pozycja; dostarcza
  wprost oczekiwany rezultat 2 projektu.

### B. Job Shop (POZA projektem doktorskim — opcjonalne rozszerzenie)
Duży lift: graf dysjunktywny, Cmax = najdłuższa ścieżka w DAG, ruchy na
blokach ścieżki krytycznej, nowy parser (ta01–80, ft/la/orb). Uogólnienie
Fibonacciego/Motzkina: niekolidujące swapy w blokach krytycznych.
W projekcie doktorskim PFSP jest centralny — JSSP robić dopiero, gdy
A/C/D domknięte, albo przeformułować z promotorem (najbliższy planowi
byłby hybrid flow shop, wymieniony w future work windowed_qubo).

## Rekomendowana kolejność (po uzgodnieniu z celami projektu)
1. **A (QAOA, symulator)** — zgodne z Gantt S1–S2 i celem 2; zero quoty;
   trudność ŚREDNIA (~1–2 tyg. na szkielet + walidację).
2. **C1/C2 (tribonacci → k-bonacci)** równolegle jako szybki wynik —
   wyprzedza Gantt poz. 4; trudność NISKA.
3. **G (metodyka benchmarkowania)** — pisanie/formalizacja z istniejących
   danych; można wpleść między eksperymenty; dostarcza rezultat 2.
4. **D (Fibonacci deep)** — konsumuje A + istniejące dane; dostarcza cel 2.
5. **C3/C4 (Hamming, de Montmort)** — Hamming po C2; de Montmort po sesji
   projektowej z promotorem.
6. **E (Motzkin deep)**; **F (Odra 5)** gdy pojawi się dostęp.
7. **B (Job Shop / hybrid flow shop)** — po domknięciu powyższych i po
   rozmowie z promotorem o zakresie.

Zasoby: quota D-Wave ~205 s do końca lipca, reset ~1 sierpnia (~400 s/mies).
Tranże 40 s, mikro-test przed kampanią, protokół jak w run_qpu_p4.py.
QAOA na symulatorze i klasyka (C) nie zużywają quoty.
