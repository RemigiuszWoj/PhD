# ICAISC 2027 — plan artykułu (studium p)

Format: Springer LNCS/LNAI (`llncs`, `splncs04`), „preferably around 12 pages” razem z bibliografią,
**zgłoszenie zanonimizowane** (PDF bez nazwisk + źródła z nazwiskami), system icaiscsystem.icaisc.eu.
Bibliografia: `references.bib` (76 pozycji w puli, DOI zweryfikowane w Crossref 23.10.2026); w artykule
**20 pozycji** (lista niżej, decyzja 24.10.2026).

## Tytuł (propozycje)

1. **Depth or Shots? The Number of QAOA Layers in Quantum Neighborhood Search for the Permutation Flow Shop Problem**
2. How Many QAOA Layers Does a Scheduling Neighborhood Need? A Case Study on the Permutation Flow Shop Problem
3. Circuit Depth in Fixed-Angle QAOA Neighborhood Search for the Permutation Flow Shop Problem: A Case Study

Rekomendacja: (1). E1 to potwierdza: od p = 2, przy dobrej tabeli kątów, najlepsza z 4096 próbek trafia
optimum prawie zawsze (best4096 ≥ 0,999), a różnice między p widać przy 256 shotach. Ostatecznie decyduje E5.

## Teza (E1 zrobione; E4/E5 do potwierdzenia)

(a) p = 1 jest słabe, a dla Fibonacciego i Dynasearcha gorsze od losowania jednostajnego (0,33× i 0,15×).
(b) Od p = 2, przy dobrej tabeli kątów, odczyt z 4096 shotów prawie zawsze trafia optimum, więc dalsze
warstwy prawie nic nie dają, a shoty mogą zastąpić głębokość. (c) To, skąd pochodzą kąty, waży więcej niż p,
a średnia energia, którą minimalizuje kalibracja, nie przekłada się na trafianie optimum. (d) Przy stałych
kątach P_opt nie rośnie monotonicznie z p na oknach spoza treningu. (e) Z szumem optimum leży przy małym p*
(E4 — do potwierdzenia). Koszt obwodu (bramki dwukubitowe, MOVE na gwieździe) rośnie liniowo w p.
**Nie pisać „przewaga kwantowa”** ani „pierwsi”.

## Struktura i budżet stron

Zasada: nigdy nagłówek sekcji i od razu podrozdział (bez tekstu między nimi); przy dwóch podrozdziałach
usuwamy oba nagłówki. Wstęp i wkład **ciągłą prozą, bez wypunktowań**. Zdania pełne, z orzeczeniem.

| § | Tytuł | Strony | Treść |
|---|---|---|---|
| — | Abstract | 0,3 | ≤ 250 słów, 4–5 słów kluczowych |
| 1 | Introduction | 1,2 | kontekst, luka, wkład prozą |
| 2 | Related Work | 1,0 | głębokość QAOA, szum, kąty stałe, kwantowe sąsiedztwa |
| 3 | Quantum Neighborhood Search | 1,2 | PFSP, QUBO wyboru zamian, 4 sąsiedztwa, okna |
| 4 | QAOA at Depth p | 1,2 | ansatz, trzy tabele kątów, odczyt, koszt na gwieździe |
| 5 | Experimental Setup | 0,8 | instancje, poziomy, warianty, miary, statystyka |
| 6 | Results | 4,0 | 4 figury + 2 tabele |
| 7 | Discussion | 0,6 | ograniczenia, czego nie twierdzimy |
| 8 | Conclusions | 0,4 | jeden akapit |
| — | References | ~1,2 | 20 pozycji |

## Bibliografia (20 pozycji)

| Grupa | Klucze |
|---|---|
| Nasze prace (7, w 3. osobie) | wojewodzki2026soco, wojewodzki2026icaisc, wojewodzki2025exit, wojewodzki2026bpasts, wojewodzki2026windowed, wojewodzki2026gcmm, wojewodzki2026aciids |
| PFSP i metaheurystyki (2) | taillard1993, lourenco2003 |
| QUBO i QAOA (5) | lucas2014ising, farhi2014qaoa, zhou2020, wurtz2021fixed, basso2022 |
| Szum (2) | franca2021, harrigan2021 |
| Grupa (4, w 3. osobie) | bozejko2026new, bozejko2024cinti, dudkiewicz2026depth, gnatowski2026routing |

### 1. Introduction (prozą)
Akapit 1: PFSP i benchmarki Taillarda [taillard1993]; ILS [lourenco2003]; sąsiedztwa wykładnicze
przeszukiwane jako problem optymalizacyjny — Fibonacci [bozejko2026new], Motzkin [wojewodzki2026soco,
wojewodzki2025exit], hybrydy [wojewodzki2026icaisc].
Akapit 2: wybór ruchu jako QUBO [lucas2014ising] na annealerze [wojewodzki2026bpasts, wojewodzki2026windowed]
i na komputerze bramkowym przez QAOA [farhi2014qaoa, bozejko2024cinti, wojewodzki2026gcmm,
wojewodzki2026aciids].
Akapit 3 — luka: prace o sąsiedztwach bramkowych stosują p = 1 (bez ujawniania autorstwa); literatura o p
dotyczy MaxCut i modelu SK [zhou2020, basso2022, wurtz2021fixed], a nie odczytu energetycznego w pętli
lokalnego przeszukiwania; studium p w problemie kombinatorycznym grupy to plecak z COBYLA
[dudkiewicz2026depth] — inna miara, bez sprzętu z gwiazdą.
Akapit 4 — wkład prozą: (i) p = 1…10 bez szumu na czterech sąsiedztwach, (i′) porównanie trzech sposobów
kalibracji kątów, różniących się szerokością okna treningowego, (ii) głębokość vs shoty, (iii) p* z szumem
i na 16-kubitowym procesorze o topologii gwiazdy, (iv) przełożenie na Cmax/PRD w ILS, z punktami odniesienia
— losowaniem (p = 0) i klasycznym DP.

### 2. Related Work
Bez nagłówków podrozdziałów; trzy akapity tematyczne.
- **Głębokość QAOA:** [farhi2014qaoa, zhou2020 (INTERP), basso2022 (duże p), wurtz2021fixed (kąty stałe)].
- **Szum ogranicza głębokość:** [franca2021, harrigan2021].
- **Kwantowe sąsiedztwa i sprzęt (3. osoba):** [bozejko2026new, bozejko2024cinti, dudkiewicz2026depth,
  gnatowski2026routing (routing na gwieździe), wojewodzki2026soco, wojewodzki2026icaisc,
  wojewodzki2026windowed, wojewodzki2026gcmm, wojewodzki2026aciids].

### 3. Quantum Neighborhood Search
PFSP i Cmax (1 równanie); ruch jako zbiór rozłącznych zamian; QUBO wyboru zamian (przekątna = delta
Cmax, kara za konflikt) [lucas2014ising]. Cztery sąsiedztwa w jednej tabeli: liczność, struktura QUBO,
zmienne na okno [bozejko2026new, wojewodzki2026soco]. Okienkowanie (w = 17 / w = 6, nakładanie 0,5)
z odwołaniem do wcześniejszych prac w 3. osobie [wojewodzki2026windowed, wojewodzki2026aciids], bez
przepisywania ich tekstu.

### 4. QAOA at Depth p
Ansatz (1 równanie), normalizacja Isinga. Trzy tabele kątów stałych, różniące się szerokością okna
treningowego: małe (w = 6, p ≤ 10), duże bez akceleratora (p ≤ 5), okna z ILS (p ≤ 5); siatka dla p = 1,
Nelder–Mead z INTERP dla p ≥ 2 [zhou2020], budżet skalowany z liczbą parametrów (200 · 2p); monotoniczność
w p **tylko na zbiorze treningowym**. Odczyt energetyczny i dlaczego argmax jest błędny. Koszt obwodu w p:
RZZ na krawędź × p; na gwieździe każda bramka dwukubitowa idzie przez rezonator (MOVE)
[gnatowski2026routing]. Wzór na trafienie optimum przy S shotach.

### 5. Experimental Setup
Poziomy: bez szumu (`ideal`), z szumem (fałszywy sirius w trybach `calibrated` i `thermal`), sirius.
Instancje tai20_5/10/20 [taillard1993]; start z **permutacji tożsamościowej** (jak we wszystkich poprzednich
pracach). Warianty A–D: tabela „small”, najlepsza tabela z E1, p = 0, klasyczne DP; 256 i 4096 shotów.
ILS bez szumu: budżet 10 s; ILS z szumem: **budżet w ruchach równy liczbie ruchów siriusa** w odpowiadającym
przebiegu. PRD od dolnego ograniczenia Taillarda, testy statystyczne, ziarna liczone z klucza przebiegu.
Wprost: dane p = 1 na sprzęcie pochodzą z wcześniejszej kampanii [wojewodzki2026aciids].

### 6. Results (bez nagłówka 6.1 bezpośrednio pod „6”)
- **Fig. 1** — P_opt względem losowania i best-of-S (S = 256, 4096) w funkcji p, dla trzech tabel kątów,
  cztery panele (sąsiedztwa) (E1); E2 opcjonalnie jako linia przerywana.
- **Fig. 2** — trafienie optimum w funkcji liczby shotów S dla różnych p: ile shotów zastępuje warstwę.
- **Tab. 1** — bramki dwukubitowe i głębokość dla p = 1…10 na gwieździe i kracie (E3).
- **Fig. 3** — P_opt(p) bez szumu / z szumem / sirius dla p = 0…6 (sirius 1…3): punkt p* (E4, H1).
- **Tab. 2** — PRD sąsiedztwo × p dla E5 (bez szumu) i E5n (z szumem, te same ruchy co sirius) + sirius
  p = 1, 2, 3 (H2); wiersze odniesienia: p = 0 i klasyczne DP; testy par.
- **Fig. 4** — liczba ruchów w 10 s i czas ruchu w funkcji p (E5).

### 7. Discussion
Ograniczenia: tylko n = 20 (świadomie); kąty stałe; 1–2 ruchy w 10 s na sprzęcie; energia okna ≠ Cmax przy
nieaddytywnych deltach; średnia energia, którą minimalizuje kalibracja, nie przekłada się na trafianie
optimum; akcelerator bloków ogranicza kandydatów w oknach liniowych z ILS: 1–16 zmiennych (mediana 10) zamiast stałych 16; fałszywy sirius nie
modeluje szumu w czasie bezczynności kubitów, a migawka kalibracji jest z 23.10, podczas gdy dane p = 1 ze
sprzętu pochodzą z 7–20.10. Czego nie twierdzimy: przewagi nad klasyką — klasyczne DP jest wyraźnie lepsze
(pierwszy test: PRD 5,3 % wobec 7,5–12,8 %).

### 8. Conclusions
Jeden akapit: wynik o p*, zalecenie (shoty zamiast warstw albo przeciwnie — zależnie od wyników), dalsze
prace (adaptacyjne p, inne topologie).

## Anonimizacja (recenzja dwustronnie ślepa)

- Bez nazwisk, afiliacji, e-maili, ORCID, podziękowań za dostęp do sprzętu (PWr/IQM), linku do repo.
- Własne prace w **3. osobie** („Bożejko et al. [x] proposed…”, „the windowed decomposition of [y]”).
  Nie: „our previous work”.
- **Prace w recenzji (`wojewodzki2026gcmm`, `wojewodzki2026aciids`, `wojewodzki2026bpasts`): decyzja
  24.10.2026 — wariant (b):** cytowane w 3. osobie z nazwiskami, jak opublikowane, z adnotacją „under review”.
- Metadane PDF: puste `pdfauthor`, brak nazw w ścieżkach figur; sprawdzić `pdfinfo` przed wysłaniem.
- Oświadczenie o użyciu modelu językowego zostaje, ale bez nazwisk (jak w ACIIDS).

## Autoplagiat
Tekst metody (§3–4) napisać od nowa; po napisaniu kontrola 8-wyrazowych wspólnych fraz z ACIIDS,
GCMM, windowed_qubo, SOCO, ICAISC 2026 (jak przy ACIIDS: cel 0).

## Kontrola jakości przed wysłaniem
humanizer (reguły z CLAUDE.md) → przegląd spójności liczb z JSON-ami → Crossref dla każdego DOI →
anonimizacja → `pdffonts` (brak Type 3) → liczba stron.
