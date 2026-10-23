# ICAISC 2027 — plan artykułu (studium p)

Format: Springer LNCS/LNAI (`llncs`, `splncs04`), „preferably around 12 pages” razem z bibliografią,
**zgłoszenie zanonimizowane** (PDF bez nazwisk + źródła z nazwiskami), system icaiscsystem.icaisc.eu.
Bibliografia: `references.bib` (76 pozycji, DOI zweryfikowane w Crossref 23.10.2026).

## Tytuł (propozycje)

1. **Depth or Shots? The Number of QAOA Layers in Quantum Neighborhood Search for the Permutation Flow Shop Problem**
2. How Many QAOA Layers Does a Scheduling Neighborhood Need? A Case Study on the Permutation Flow Shop Problem
3. Circuit Depth in Fixed-Angle QAOA Neighborhood Search for the Permutation Flow Shop Problem: A Case Study

Rekomendacja: (1), jeśli E1/E5 potwierdzą, że przy odczycie energetycznym shoty zastępują warstwy. Jeśli nie,
(3) — bezpieczny i opisowy.

## Teza (do potwierdzenia wynikami)

Bez szumu jakość rozkładu QAOA rośnie z p i nasyca się około p ≈ 4, szybciej dla sąsiedztw rzadkich niż
gęstych. Odczyt energetyczny z 4096 shotów zjada większość tego zysku na poziomie ILS. Na sprzęcie
i z szumem optimum leży przy małym p*, a koszt obwodu (bramki dwukubitowe, MOVE na gwieździe) rośnie
liniowo w p. Praktyczny wniosek: w sąsiedztwach okienkowanych zamiast dokładać warstwy lepiej dokładać shoty.
**Nie pisać „przewaga kwantowa”** ani „pierwsi”.

## Struktura i budżet stron

Zasada: nigdy nagłówek sekcji i od razu podrozdział (bez tekstu między nimi); przy dwóch podrozdziałach
usuwamy oba nagłówki. Wstęp i wkład **ciągłą prozą, bez wypunktowań**. Zdania pełne, z orzeczeniem.

| § | Tytuł | Strony | Treść |
|---|---|---|---|
| — | Abstract | 0,3 | ≤ 250 słów, 4–5 słów kluczowych |
| 1 | Introduction | 1,2 | kontekst, luka, wkład prozą |
| 2 | Related Work | 1,3 | głębokość QAOA, szum, kąty stałe, kwantowe szeregowanie |
| 3 | Quantum Neighborhood Search | 1,2 | PFSP, QUBO wyboru zamian, 4 sąsiedztwa, okna |
| 4 | QAOA at Depth p | 1,2 | ansatz, kalibracja p ≤ 10, odczyt, koszt na gwieździe |
| 5 | Experimental Setup | 0,8 | instancje, trzy poziomy, miary, statystyka |
| 6 | Results | 3,0 | 4 figury/tabele |
| 7 | Discussion | 0,6 | ograniczenia, czego nie twierdzimy |
| 8 | Conclusions | 0,4 | jeden akapit |
| — | References | ~2,5 | ~40–45 pozycji |

### 1. Introduction (prozą)
Akapit 1: PFSP jest NP-trudny dla m ≥ 3 [garey1976]; benchmarki Taillarda [taillard1993]; metaheurystyki
lokalne [nowicki1996, ruiz2007, ruizmaroto2005, lourenco2003, wodecki2002parallel]; sąsiedztwa wykładnicze przeszukiwane jako problem
optymalizacyjny [ahuja2002, congram2002].
Akapit 2: przeniesienie wyboru ruchu na komputer kwantowy jako QUBO [lucas2014ising, glover2019]; annealing
[kadowaki1998] i prace grupy na D-Wave (3. osoba); QAOA [farhi2014qaoa] na urządzeniach NISQ [preskill2018].
Akapit 3 — luka: wcześniejsze prace o sąsiedztwach bramkowych stosują p = 1 (wskazać prace bez
ujawniania autorstwa); literatura o p dotyczy MaxCut i modelu SK [zhou2020, basso2022, wurtz2021fixed],
a nie odczytu energetycznego w pętli lokalnego przeszukiwania. Jedyne znane nam studium p w problemie
kombinatorycznym grupy to plecak z COBYLA [dudkiewicz2026depth] — inna miara, bez sprzętu z gwiazdą.
Akapit 4 — wkład prozą: (i) p = 1…10 bez szumu na czterech sąsiedztwach, (ii) głębokość vs shoty,
(iii) p* z szumem i na 16-kubitowym procesorze o topologii gwiazdy, (iv) przełożenie na Cmax/PRD w ILS.

### 2. Related Work
Bez nagłówków podrozdziałów; trzy–cztery akapity tematyczne.
- **Głębokość QAOA, teoria:** [farhi2014qaoa, zhou2020 (INTERP), crooks2018, akshay2020 (reachability
  deficits), basso2022, farhi2022sk, blekos2024 (przegląd), hadfield2019 (QAOA jako ansatz operatorowy)].
- **Szum ogranicza głębokość:** [franca2021, wang2021noise, marshall2020, xue2021, harrigan2021,
  pagano2020, weidenfeller2022, lotshaw2022, bharti2022, cerezo2021]; skalowanie [shaydulin2024].
- **Kąty stałe i przenoszenie:** [brandao2018, wurtz2021fixed, galda2021, streif2020, sack2021, egger2021];
  optymalizacja kątów [nelder1965].
- **Kwantowe szeregowanie i prace grupy (3. osoba):**
  - annealing w szeregowaniu: [venturelli2015, bozejko2022distributed, bozejko2023determination,
    bozejko2023twomachine, bozejko2024comparison, bozejko2024bnb, bozejko2025sumcost, bozejko2026huge,
    bozejko2026flexible, bozejko2026twttsp, bozejko2024knapsack, choptiany2026knapsack];
  - sąsiedztwa wykładnicze jako QUBO: [bozejko2026new (Fibonacci), bozejko2025noncrossing,
    wojewodzki2026soco (Motzkin), wojewodzki2025exit, wojewodzki2026icaisc, wojewodzki2026windowed];
  - bramkowe: [bozejko2024cinti (QAOA flow shop), kurowski2023, amaro2022, dudkiewicz2026tsp,
    dudkiewicz2026depth];
  - kompilacja i sprzęt IQM: [gnatowski2026mapping, gnatowski2026routing (gwiazda!), gnatowski2026spark,
    biskup2026frameworks, gnatowski2026volumetric, choptiany2026bitflip (mitygacja błędów)];
  - przeglądy/metody: [jaroszczuk2025vrp, jaroszczuk2026extrema] — tylko jeśli zdanie je uzasadnia.

### 3. Quantum Neighborhood Search
PFSP i Cmax (1 równanie); ruch jako zbiór rozłącznych zamian; QUBO wyboru zamian (przekątna = delta
Cmax, kara za konflikt). Cztery sąsiedztwa w jednej tabeli: liczność, struktura QUBO, zmienne na okno
[aigner1998 dla Motzkina, congram2002 dla Dynasearch]. Okienkowanie (w = 17 / w = 6, nakładanie 0,5)
z odwołaniem do wcześniejszych prac w 3. osobie, bez przepisywania ich tekstu.

### 4. QAOA at Depth p
Ansatz (1 równanie), normalizacja Isinga, kalibracja kątów stałych p = 1…10 (siatka dla p = 1,
Nelder–Mead z INTERP dla p ≥ 2, monotoniczność z konstrukcji) [zhou2020, nelder1965]. Odczyt energetyczny
i dlaczego argmax jest błędny. Koszt obwodu w p: RZZ na krawędź × p; na gwieździe każda bramka dwukubitowa
idzie przez rezonator (MOVE) [gnatowski2026routing]. Wzór na S95.

### 5. Experimental Setup
Trzy poziomy (bez szumu / z szumem / sirius), instancje, start z permutacji NEH [nawaz1983], sąsiedztwa, p, shoty, limit ruchów ILS, budżet
10 s na sprzęcie, PRD od dolnego ograniczenia Taillarda, testy statystyczne. Środowisko: Qiskit
[javadi2024], Aer, iqm-client. Wprost: dane p = 1 na sprzęcie pochodzą z wcześniejszej kampanii (cytat).

### 6. Results (bez nagłówka 6.1 bezpośrednio pod „6”)
- **Fig. 1** — `P_opt(p)` i `r(p)` dla p = 1…10, cztery sąsiedztwa, mediany + IQR (E1); linia przerywana:
  kąty optymalne per okno (E2).
- **Fig. 2** — S95 w funkcji p (log), bez szumu i z szumem: ile shotów zastępuje warstwę (Q2).
- **Tab. 1** — bramki dwukubitowe i głębokość dla p = 1…10 na gwieździe i kracie (E3).
- **Fig. 3** — `P_opt(p)` bez szumu / z szumem / sirius dla p = 1…3(6): punkt p* (E4, H1).
- **Tab. 2** — ILS: PRD (mediana) sąsiedztwo × p na symulatorze (E5) + wiersz sirius p = 1, 2, 3 (H2),
  z testami par.

### 7. Discussion
Ograniczenia: tylko n = 20 (świadomie), kąty stałe z małych okien, model szumu zastępczy (jeśli plan B),
1–2 ruchy w 10 s na sprzęcie, energia okna ≠ Cmax przy nieaddytywnych deltach. Czego nie twierdzimy:
przewagi nad klasyką; ILS z dokładnym DP jest szybszy.

### 8. Conclusions
Jeden akapit: wynik o p*, zalecenie (shoty zamiast warstw albo przeciwnie — zależnie od wyników), dalsze
prace (adaptacyjne p, inne topologie).

## Anonimizacja (recenzja dwustronnie ślepa)

- Bez nazwisk, afiliacji, e-maili, ORCID, podziękowań za dostęp do sprzętu (PWr/IQM), linku do repo.
- Własne prace w **3. osobie** („Bożejko et al. [x] proposed…”, „the windowed decomposition of [y]”).
  Nie: „our previous work”.
- **Prace w recenzji (`wojewodzki2026gcmm`, `wojewodzki2026aciids`, `wojewodzki2026bpasts`):** ACIIDS jest źródłem metody okienkowej na sirius i danych
  p = 1, więc nie wolno go pominąć. Opcje: (a) cytować jako „Anonymous: … (under review)” i wysłać PDF
  przewodniczącym, jeśli poproszą; (b) cytować preprint SSRN (windowed QUBO, D-Wave) + 3. osoba.
  GCMM — decyzja 5.11: jeśli przyjęte, cytować jako „to appear” w 3. osobie. **Do decyzji z promotorem.**
- Metadane PDF: puste `pdfauthor`, brak nazw w ścieżkach figur; sprawdzić `pdfinfo` przed wysłaniem.
- Oświadczenie o użyciu modelu językowego zostaje, ale bez nazwisk (jak w ACIIDS).

## Autoplagiat
Tekst metody (§3–4) napisać od nowa; po napisaniu kontrola 8-wyrazowych wspólnych fraz z ACIIDS,
GCMM, windowed_qubo, SOCO, ICAISC 2026 (jak przy ACIIDS: cel 0).

## Kontrola jakości przed wysłaniem
humanizer (reguły z CLAUDE.md) → przegląd spójności liczb z JSON-ami → Crossref dla każdego DOI →
anonimizacja → `pdffonts` (brak Type 3) → liczba stron.
