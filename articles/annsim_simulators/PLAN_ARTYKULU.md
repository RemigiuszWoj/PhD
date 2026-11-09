# ANNSIM 2027 — plan artykułu

Wersja po 10 rundach poprawek (25.10.2026). Ścieżka **Quantum Computing**, pełny artykuł.
**Wytyczne robocze z ANNSIM 2026** (oficjalny szablon Overleaf „2026 ANNSIM Template”, klasa `scspaperproc`;
dla 2027 jeszcze nieopublikowane, zakładamy to samo): **5–12 stron**; **recenzja dwustronnie ślepa** — bez
autorów i biogramów w zgłoszeniu; w wersji końcowej **biogram każdego autora obowiązkowy** (jeden akapit,
kończy się e-mailem); cytowania numeryczne w nawiasach, bibliografia w kolejności cytowania (styl `scsproc`).
Anonimizacja jak w ICAISC: własne prace w 3. osobie.

## Tytuł (propozycje)

1. **How Faithful Are Quantum Circuit Simulators? Nine Simulators and Three Noise Scenarios against a
   16-Qubit Star Processor**
2. Simulating a Star-Topology Quantum Processor: A Cross-Simulator Validation on QAOA Neighborhood Search
3. One Calibration, Nine Simulators: Predicting IQM Sirius on a Scheduling Workload

Rekomendacja: (1) — mówi wprost, co porównujemy i z czym.

## Teza (do potwierdzenia)

Przy tych samych danych kalibracyjnych programy zgadzają się ze sobą w granicy szumu shotów, jeśli dostają ten
sam obwód; większe różnice wprowadza kompilator niż silnik. Najprostszy realistyczny model (depolaryzacja
z wierności bramek + odczyt) przewiduje sprzęt najlepiej, a dodanie relaksacji T1/T2 przesadza z szumem.
Wybór programu decyduje głównie o koszcie: macierz gęstości kończy się ok. 12 kubitów, trajektorie i diagramy
decyzyjne sięgają 17. Bez tez o przewadze kwantowej.

## Struktura i budżet stron (~12)

Zasady jak w ICAISC: bez nagłówka sekcji tuż nad podrozdziałem; wstęp i wkład prozą; pełne zdania;
humanizer; formatowanie na końcu.

| § | Tytuł | Strony | Treść |
|---|---|---|---|
| — | Abstract | 0,3 | ≤ 250 słów |
| 1 | Introduction | 1,0 | rola symulatorów w NISQ, luka: brak walidacji na realnym zadaniu i topologii gwiazdy |
| 2 | Background | 1,2 | zadanie (QAOA w oknach PFSP, krótko z cytatem ICAISC/ACIIDS), procesor gwiazdy i MOVE, kanały szumu |
| 3 | Simulators | 1,5 | 10 programów bramkowych: metoda, szum, kompilator IQM, licencja (Tabela 1, z pomiarów etapu 1); dlaczego Stim odpada |
| 4 | Methodology | 1,8 | scenariusze S1–S3 z jednej migawki, tory A i B, dane ze sprzętu (ACIIDS, H2, H3), miary, podłogi zgodności |
| 5 | Experimental Setup | 0,8 | okna (71 z K ≤ 11; 40 do 16 zmiennych), p, shoty, sprzęt obliczeniowy, wersje |
| 6 | Results | 3,5 | 5 figur/tabel |
| 7 | Discussion | 0,8 | zalecenia: który program do czego; ograniczenia |
| 8 | Conclusions | 0,4 | jeden akapit |
| — | References | ~1,0 | ~20–25 pozycji |

### 1. Introduction (prozą)
Symulatory są głównym narzędziem pracy z NISQ, bo czas sprzętu jest drogi; ich wierność zwykle sprawdza się na
benchmarkach syntetycznych, a nie na zadaniu, którego wynik oceniamy (tu: PRD harmonogramu). Topologia gwiazdy
z rezonatorem i bramką MOVE jest nowa i słabo wspierana w ekosystemach. Wkład prozą: (i) przegląd 10 programów
pod kątem symulacji z szumem przy 12–17 kubitach, (ii) rozdzielenie wpływu silnika i kompilatora, (iii)
porównanie trzech scenariuszy szumu ze sprzętem na poziomie rozkładów i PRD, z podłogą zgodności sprzęt–sprzęt,
(iv) koszt obliczeń i granice wykonalności.

### 2. Background
Okienkowe QAOA w ILS (2 akapity, odesłanie do ICAISC/ACIIDS); Sirius: 16 kubitów, rezonator, MOVE, CZ, PRX
[gnatowski2026routing]; kanały: depolaryzacja, tłumienie amplitudy i fazy, błąd odczytu.

### 3. Simulators
**Tabela 1** — program | metoda z szumem | max kubitów z szumem (zmierzone T6) | kompilator na IQM | licencja.
Akapit na grupę (pełny zakres / macierz gęstości / impulsy / odrzucony).

### 4. Methodology
Jedna migawka kalibracji → parametry kanałów (wzory: depolaryzacja z wierności, λ = d(1−F)/(d−1)); S1–S3;
tor A (wspólny QASM, MOVE → SWAP z rezonatorem jako kubitem) i tor B (trzy kompilatory, wspólny silnik);
miary: TVD, Hellinger, P_opt, trafienie optimum; podłoga szumu shotów (bootstrap) i podłoga sprzęt–sprzęt (H3
w dwa dni); PRD przy tej samej liczbie ruchów co Sirius.

### 5. Experimental Setup
Instancje tai20_5/10/20 inst. 0–3 (okna pierwszego ruchu), K ≤ 11 dla wszystkich, do 16 zmiennych dla czterech;
p = 1…3; 4096 shotów (T3: 1024); ILS n = 20; maszyna (CPU, RAM), wersje pakietów.

### 6. Results (bez nagłówka podrozdziału tuż pod „6”)
- **Fig. 1** — S1: TVD każdego programu do statevectora, na tle podłogi szumu shotów (weryfikacja, RQ1).
- **Tab. 2** — tor B: CZ, MOVE, głębokość dla trzech kompilatorów; TVD do sprzętu w S2 (RQ2).
- **Fig. 2** — mapa ciepła TVD do Siriusa: program × scenariusz (S1–S3), z podłogą sprzęt–sprzęt (RQ3).
- **Fig. 3** — PRD symulatora vs PRD Siriusa (ILS, ta sama liczba ruchów), punkty = instancje, p = 1…3 (RQ3).
- **Fig. 4** — czas na obwód i pamięć vs liczba kubitów, wszystkie programy, skala log (RQ4).

### 7. Discussion
Zalecenia praktyczne (który program do weryfikacji, który do szumu, który do skali); ograniczenia: rezonator
jako qubit, brak szumu bezczynności, jedna migawka kalibracji, n = 20, jeden procesor.

### 8. Conclusions
Jeden akapit z odpowiedziami na RQ1–RQ4.

## Bibliografia (~20–25, każdy DOI do weryfikacji w Crossref)

- Programy: Qiskit [javadi2024]; qsim (Isakov i in. 2021, arXiv 2111.02396); Qulacs (Suzuki i in. 2021,
  Quantum); DDSIM (Zulehner, Wille 2019; Grurl i in. 2022 — szum w DD); Qibo (Efthymiou i in. 2022); CUDA-Q;
  PennyLane (Bergholm i in.); Braket; QuTiP (Johansson i in.); Stim (Gidney 2021); TKET (Sivarajah i in. 2020).
- Benchmarki symulatorów i modelowanie szumu: do znalezienia (np. benchmarki pakietów symulacyjnych, modele
  szumu urządzeń NISQ).
- Zadanie i sprzęt: Farhi QAOA, Taillard, gnatowski2026routing; nasze: ACIIDS, GCMM, ICAISC 2027 (under review).

## Rozdzielenie od ICAISC
ICAISC: p jako parametr algorytmu. ANNSIM: wierność i koszt symulatorów. Wspólne dane (E4/E5n, H2) — z cytatem;
tekst metody od nowa; kontrola wspólnych 8-wyrazowych fraz przed wysyłką.

## Kontrola przed wysłaniem
humanizer → spójność liczb z danymi → Crossref → wytyczne SCS 2027 (porównać z 2026: 5–12 s., double-blind,
biogramy) → anonimizacja → `pdffonts`.

## Dziennik poprawek (10 rund + uzupełnienie)

11. (25.10) Tor wyżarzaczy przeniesiony do osobnego artykułu; ANNSIM tylko symulatory bramkowe.


1. Struktura z ICAISC przeniesiona na temat symulatorów.
2. Tytuł zmieniony z opisowego na pytanie z liczbami (9 programów, 3 scenariusze, 16 kubitów).
3. Dodana sekcja „Simulators” jako osobny wkład przeglądowy (Tabela 1).
4. Metodologia rozbita na tory A/B, bo inaczej wyniki nie rozdzielają silnika od kompilatora.
5. Fig. 1 jako weryfikacja S1 — bez niej różnice w S2/S3 nie mają punktu odniesienia.
6. Dodana podłoga sprzęt–sprzęt w Fig. 2 (H3 w dwa dni).
7. Teza złagodzona do hipotez sprawdzalnych (kompilator > silnik; S2 najlepszy; koszt).
8. Budżet stron: Results 3,5 s.; Background skrócony, bo zadanie jest opisane w ICAISC/ACIIDS.
9. Bibliografia: lista programów do Crossref; uzupełnić 2–3 prace o benchmarkach symulatorów.
10. Otwarte sprawy jawnie: szablon i limit stron SCS, tryb recenzji, rozdzielenie od ICAISC.

Uzupełnienie 25.10: wytyczne 2026 przyjęte jako robocze (5–12 s., double-blind, biogramy w wersji końcowej).
