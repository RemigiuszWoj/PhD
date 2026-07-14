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

### C. Nowe otoczenia: Tribonacci, k-bonacci, Hamming (Gantt 4 — do przodu planu)
Najtańszy nowy wynik, pełny reuse pipeline'u; można robić równolegle z A,
bo nie dotyka QPU (symulator + klasyka), a wyprzedza harmonogram (S3–S4).
- Tribonacci: obok swapów sąsiednich 3-cykle na pozycjach (i,i+1,i+2);
  niekolidujące podzbiory → T(n)=T(n−1)+T(n−2)+T(n−3); DP liniowe; QUBO
  wstęgowe (szer. 2) — embeduje jak fibonacci, skaluje do n=500.
- k-bonacci: ruchy do długości k (k=2 ⇒ Fibonacci, k=3 ⇒ Tribonacci);
  jedno DP, jeden dowód, strojony parametr bogactwa ruchu. Oś artykułu:
  "sequence-counted neighborhood family".
- Hamming: ruchy o ograniczonym dystansie Hamminga d (wybór d pozycji do
  optymalnej repermutacji); QUBO przypisaniowe (gęstsze — sprawdzić granicę
  embeddingu). Semantykę doprecyzować z promotorem.
- Publikacja: ICAISC/SOCO 2027 albo sekcja w artykule D.

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

### B. Job Shop (POZA projektem doktorskim — opcjonalne rozszerzenie)
Duży lift: graf dysjunktywny, Cmax = najdłuższa ścieżka w DAG, ruchy na
blokach ścieżki krytycznej, nowy parser (ta01–80, ft/la/orb). Uogólnienie
Fibonacciego/Motzkina: niekolidujące swapy w blokach krytycznych.
W projekcie doktorskim PFSP jest centralny — JSSP robić dopiero, gdy
A/C/D domknięte, albo przeformułować z promotorem (najbliższy planowi
byłby hybrid flow shop, wymieniony w future work windowed_qubo).

## Rekomendowana kolejność (po uzgodnieniu z celami projektu)
1. **A (QAOA, symulator)** — zgodne z Gantt S1–S2 i celem 2; zero quoty.
2. **C (tribonacci/k-bonacci)** równolegle jako szybki wynik publikacyjny —
   wyprzedza Gantt poz. 4.
3. **D (Fibonacci deep)** — konsumuje A + istniejące dane; dostarcza cel 2.
4. **E (Motzkin deep)**.
5. **B (Job Shop / hybrid flow shop)** — po domknięciu powyższych i po
   rozmowie z promotorem o zakresie.

Zasoby: quota D-Wave ~205 s do końca lipca, reset ~1 sierpnia (~400 s/mies).
Tranże 40 s, mikro-test przed kampanią, protokół jak w run_qpu_p4.py.
QAOA na symulatorze i klasyka (C) nie zużywają quoty.
