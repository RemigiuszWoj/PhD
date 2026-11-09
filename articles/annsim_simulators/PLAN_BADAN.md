# ANNSIM 2027 — plan badań: symulatory kwantowe a procesor IQM Sirius

Wersja po 10 rundach poprawek (25.10.2026). Termin: **18.12.2026** (ścieżka Quantum Computing, pełny
artykuł; Scopus/DBLP/ACM). Realizacja **po wysłaniu ICAISC (13.11)**. Przegląd programów:
`PRZEGLAD_SYMULATOROW.md`. Nacisk na symulacje; sprzęt minimalnie.

## 0. Etapy (decyzja użytkownika 25.10.2026)

1. **Przegląd wszystkich** — każdy program bramkowy zainstalowany i sprawdzony w praktyce
   na jednym oknie: działa / nie działa, jakie scenariusze szumu obsługuje, czas, pamięć, maks. rozmiar.
   Wynik: Tabela 1 artykułu oparta na pomiarach, nie na dokumentacji (T0, T6).
2. **Podstawowy Taillard (n = 20) na wszystkich, na których się da** — tai20_5/10/20, rdzeń porównania ze
   sprzętem (Sirius: ACIIDS p = 1 + H2), wszystkie scenariusze wykonalne dla danego programu (T1–T5). Programy z macierzą gęstości — na oknach K ≤ 11.
3. **Pełny Taillard (n = 20…500) na najbardziej obiecujących** — 2–3 programy wybrane po etapie 2 według
   zgodności ze sprzętem i kosztu (kandydaci: szybkie trajektorie — qsim / Qulacs); porównanie z pełnymi
   danymi sprzętowymi (Sirius p = 1 dla 12 klas).

Kryteria wyboru do etapu 3 ustalić przed etapem 2 (np. TVD/|ΔPRD| do sprzętu w S2/A3 i czas na obwód), żeby
wybór nie był dopasowany do wyników post hoc.

## 1. Pytanie i hipotezy

**Pytanie główne:** na ile różne programy do symulacji obwodów kwantowych, przy tych samych danych
kalibracyjnych, przewidują zachowanie 16-kubitowego procesora o topologii gwiazdy (IQM Sirius) w realnym
zadaniu — okienkowym QAOA w przeszukiwaniu sąsiedztwa dla PFSP — i ile to kosztuje.

| | Pytanie szczegółowe | Hipoteza robocza |
|---|---|---|
| RQ1 | Czy programy zgadzają się ze sobą przy identycznym obwodzie i identycznym modelu szumu? | S1: zgodność w granicy szumu shotów (weryfikacja); S2/S3: małe różnice wynikające z implementacji kanałów |
| RQ2 | Czy kompilator (qiskit-iqm, iqm-client[cirq], pytket-iqm) zmienia wynik z szumem? | tak, przez różną liczbę CZ/MOVE na gwieździe |
| RQ3 | Który scenariusz szumu najlepiej przewiduje Siriusa (rozkłady okien, PRD w ILS)? | S2 ≈ sprzęt; S3 zbyt pesymistyczny (relaksacja przekracza zmierzone błędy bramek, W3 ICAISC) |
| RQ4 | Jak czas i pamięć rosną z liczbą kubitów i p w każdym programie? | trajektorie i DD skalują się do 17 kubitów; macierz gęstości kończy się ok. 12–13 |

## 2. Programy (decyzja użytkownika: wszystkie przejrzane)

| Grupa | Programy | Zakres |
|---|---|---|
| **Pełny zakres** (trajektorie / DD) | Qiskit Aer, Cirq + qsim, Qulacs, MQT DDSIM | okna do 16 zmiennych (17 kubitów z rezonatorem) |
| **Macierz gęstości** | Qibo, CUDA-Q (CPU), PennyLane `default.mixed`, Amazon Braket lokalny | okna K ≤ 11 (12 kubitów z rezonatorem) |
| **Poziom impulsów** | QuTiP | jedno okno, demonstracja innego poziomu opisu |
| **Odrzucony** | Stim | tylko obwody Clifforda; w artykule z uzasadnieniem |

Jeśli Qibo albo CUDA-Q mają trajektorie na CPU (do sprawdzenia w tygodniu 1), przechodzą do pełnego zakresu.

## 3. Scenariusze szumu (identyczne definicje w każdym programie)

Źródło parametrów: migawka kalibracji Siriusa (`src/experiments/iqm_sirius_snapshot.py`), ta sama dla
wszystkich programów; przy biegach sprzętowych świeża migawka z dnia biegu.

| | Definicja | Kanały |
|---|---|---|
| **S1** | bez szumu | — |
| **S2** | depolaryzacja z wierności każdej bramki (PRX, CZ, MOVE, per locus) + błędy odczytu per kubit | depolaryzacja 1q/2q, macierz pomyłek odczytu |
| **S3** | S2 + relaksacja T1/T2 w czasie bramek (tylko w czasie bramek, bez bezczynności — tak samo w każdym programie) | + tłumienie amplitudy i fazy |

Wyjątek: MQT DDSIM ma jedno globalne prawdopodobieństwo szumu → S2/S3 z uśrednionymi parametrami, oznaczone
w tabelach jako przybliżone („S2≈”, „S3≈”).

## 4. Dwa tory doświadczenia (rozdzielenie efektów)

**Tor A — silnik symulacji.** Ten sam obwód po kompilacji (qiskit-iqm na Siriusa: bramki r/PRX, CZ, MOVE),
eksportowany do OpenQASM; MOVE tłumaczony na SWAP kubit–rezonator, rezonator jako dodatkowy kubit. Ten sam
model szumu. Różnice = tylko silnik i implementacja kanałów → RQ1, RQ4.

**Tor B — kompilator.** Ten sam obwód logiczny skompilowany przez qiskit-iqm, iqm-client[cirq] i pytket-iqm;
zliczenia bramek i głębokości, potem symulacja we wspólnym silniku (Aer) w S2 → RQ2. Jeśli któryś kompilator
nie obsługuje MOVE/gwiazdy, tor B obejmuje tylko te, które obsługują (ustalić w tygodniu 1).

## 5. Dane referencyjne ze sprzętu

| Źródło | Co | Koszt |
|---|---|---|
| ACIIDS (istnieje) | ILS na Siriusie p = 1, 12 klas Taillarda n = 20…500, inst. 0–3, 10 s | 0 |
| H2 z ICAISC (planowane) | ILS p = 2, 3, n = 20, inst. 0–3 | już w budżecie ICAISC (~215 kr) |
| **H3 — histogramy okien (nowe, małe)** | okna pierwszego ruchu tai20_5/10/20 inst. 0 (~40 okien z ruchem), p = 1, 2, 4096 shotów, **dwa biegi w różne dni** | ~160 obwodów ≈ **70 kr** (0,445 kr/obw.); minimum 1 bieg ≈ 36 kr |

Powtórzenie H3 w drugi dzień daje **zgodność sprzętu z samym sobą** — dolną granicę, poniżej której żaden
symulator nie może „lepiej” przewidzieć sprzętu. Bez niej porównanie rozkładów jest nieinterpretowalne.

## 6. Eksperymenty

| # | Co | Programy | Zakres | Szac. czas (8 rdzeni) |
|---|---|---|---|---|
| **T0** | instalacja w osobnych venv, test jednego okna S1 we wszystkich programach | wszystkie | 1 okno | 1–2 dni pracy |
| **T1** | S1: zgodność rozkładów (tor A), wszystkie okna K ≤ 11 | wszystkie 9 | 71 okien × p = 1…3 | < 1 h |
| **T2** | S2, S3 na oknach K ≤ 11 (tor A) | wszystkie 9 | 71 × 3 p × 2 scen. | ~3–6 h łącznie |
| **T3** | S2, S3 na wszystkich oknach do 16 zm. (tor A), 1024 shoty | 4 pełne | 40 okien (inst. 0) × 3 p × 2 scen. | ~5 h/program, ~20 h |
| **T4** | tor B: kompilatory, zliczenia + Aer S2 | 3 kompilatory | 161 okien × p = 1…3 | ~2 h |
| **T5** | ILS z ruchami jak Sirius, S2 | Aer (z E5n), qsim, Qulacs | n = 20, inst. 0–3, p = 1…3 | ~5 h/program |
| **T6** | skalowanie czasu i pamięci vs K (4…16) i p (1…6) | wszystkie | po 3 okna na K | ~2 h |
| **T7** | porównanie z H3 (rozkłady) i z ACIIDS/H2 (PRD) | wszystkie | — | analiza |

Wyniki Aera w S2 dla n = 20 można częściowo wziąć z E4/E5n (ICAISC) — z cytatem; reszta liczona od nowa.

## 7. Miary

- **Rozkłady okien:** odległość całkowitej zmienności (TVD), Hellinger, różnica P_opt, trafienie optimum przez
  najlepszą z S próbek; **podłoga szumu shotów** = TVD dwóch niezależnych prób 4096 z tego samego rozkładu
  (bootstrap); dla sprzętu — TVD między dwoma dniami H3.
- **ILS:** PRD przy tej samej liczbie ruchów co Sirius; |PRD_sym − PRD_hw|; τ Kendalla po instancjach; test
  znaków.
- **Koszt:** czas na obwód (ścienny, 1 wątek i wielowątkowo), szczytowa pamięć (RSS), maks. wykonalne K.
- **Statystyka:** mediany z przedziałami bootstrap 95 %; Wilcoxon par po oknach dla par programów; korekta
  Holma.

## 8. Harmonogram (14.11–18.12.2026)

| Tydzień | Co |
|---|---|
| 14–20.11 | T0 (venv, test 1 okna), decyzje: trajektorie Qibo/CUDA-Q, MOVE w kompilatorach; **H3 bieg 1** (świeża migawka) |
| 21–27.11 | T1, T2, T4; **H3 bieg 2** (inny dzień); start T3 |
| 28.11–4.12 | T3, T5, T6 |
| 5–11.12 | analiza, figury, tabele; tekst §3–§5 |
| 12–17.12 | tekst §1, §6–§8, humanizer, przegląd, wysyłka **16.12** (dzień zapasu) |

Kolizje: 5.11 decyzja GCMM (camera-ready w listopadzie), H2 ICAISC — sprzęt i procesor w jednym czasie dla
jednej kampanii z budżetem czasowym.

## 9. Ryzyka i plan B

| Ryzyko | Plan B |
|---|---|
| Kompilator Cirq/TKET nie obsługuje MOVE (gwiazda) | tor B tylko dla działających; w artykule jako wynik (wsparcie gwiazdy w ekosystemach) |
| Konflikty wersji pakietów | osobny venv na program (jak `.venv-iqm`); obwody przez QASM |
| DDSIM bez szumu per bramka | S2≈/S3≈ jawnie oznaczone; porównanie jakościowe |
| Rezonator jako qubit to przybliżenie | jak w fałszywym backendzie IQM; opisane w ograniczeniach |
| Brak kredytów na H3 | porównanie tylko na poziomie PRD (ACIIDS + H2); rozkłady wyłącznie symulator–symulator |
| Za mało czasu | T3 tylko inst. 0 i p = 1, 2; T5 tylko qsim |
| Pokrywanie z ICAISC | ICAISC = p jako parametr; ANNSIM = wierność symulatorów; cytat ICAISC (under review), tekst od nowa, kontrola wspólnych fraz |

## 10. Odtwarzalność

Wersje pakietów zamrożone w `requirements-<program>.txt`; ziarna z klucza przebiegu; migawki kalibracji
w repo; wyniki w `measurements/*.jsonl(.gz)`; figury i tabele generowane skryptem z danych (jak w ICAISC).

## Dziennik poprawek (10 rund + uzupełnienie)

11. (25.10) Tor wyżarzaczy dodany, a potem — decyzją użytkownika — przeniesiony do osobnego artykułu
    (`articles/annealing_simulators/PLAN.md`); ANNSIM zostaje przy symulatorach bramkowych.


1. Szkic: 3 programy × 3 scenariusze, porównanie PRD z ACIIDS.
2. Rozszerzenie na wszystkie przejrzane programy (decyzja użytkownika); podział na grupy wykonalności.
3. Identyczny obwód i model nie różnicują programów → dwa tory: A (silnik) i B (kompilator).
4. Obwód po kompilacji ma rezonator → 17 kubitów; próg dla macierzy gęstości przesunięty na K ≤ 11 (71 okien,
   zrównoważone po sąsiedztwach).
5. S3 bez bezczynności, jednakowo we wszystkich programach (Aer i tak nie modeluje bezczynności).
6. DDSIM: szum globalny → S2≈/S3≈ zamiast cichego pominięcia.
7. Porównanie rozkładów wymaga sprzętu → mały H3; dodane powtórzenie w drugi dzień jako podłoga zgodności.
8. Podłoga szumu shotów (bootstrap) — bez niej TVD nieinterpretowalna.
9. Koszt obliczeń ograniczony: T3 na 1024 shotach i inst. 0; reuse E4/E5n z cytatem.
10. Harmonogram z dniem zapasu, plan B dla każdego ryzyka, rozdzielenie od ICAISC.
