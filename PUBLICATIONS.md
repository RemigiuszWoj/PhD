# Publikacje — Remigiusz Wojewódzki

Stan na: **2026-11-22**. Wszystkie prace współautorskie z Wojciechem Bożejką
(Katedra Sterowania i Obliczeń Kwantowych, WIT, Politechnika Wrocławska).
Źródła stanu: repo (README zgłoszeń, gałęzie), notatka DOKTORAT, Crossref (15.11.2026).
Pozycje oznaczone **(?)** wymagają potwierdzenia przez autora.

| # | Praca (skrót) | Miejsce | Stan | Następny krok / termin |
|---|---|---|---|---|
| 1 | Motzkin – rozdział | monografia EXIT 2025 | opublikowany | druk egzemplarza |
| 2 | Motzkin ILS/SA | SOCO 2026 (CCIS 3046) | opublikowany, DOI | druk egzemplarza |
| 3 | Hybrid neighborhoods | ICAISC 2026 (LNAI) | w druku, korekta wydawnictwa | DOI → ORCID, Scholar, druk |
| 4 | Quantum and classical annealing (VLSN, PFSP) | BPASTS (BPASTS-05673-2026-02) | rewizja wysłana 18.11.2026 (major revision z 5.11) | ponowna ocena redakcji i recenzentów |
| 5 | Windowed QUBO | C&OR; preprint SSRN | preprint; w recenzji (data zgłoszenia i nr rękopisu (?)) | — |
| 6 | Fixed-angle QAOA | GCMM 2026 (Paper 59) | warunkowo przyjęty (9.11, minor revisions), rewizja wgrana 13.11.2026, czeka na potwierdzenie | konferencja 2–4.12.2026 (rejestracja przez katedrę); korekta |
| 7 | Windowed QAOA on IQM Sirius | ACIIDS 2027 (nr 316) | w recenzji od 23.10.2026 | decyzja: styczeń 2027 |
| 8 | Depth or Shots? | ICAISC 2027 (Paper 12373) | w recenzji od 09.11.2026 | decyzja od 28.02.2027 |
| 9 | Simulators vs IQM Sirius | ANNSIM 2027 | gotowy szkic 12 s. (EN i PL, 5 rund krytyka) | pokazać promotorowi; zgłoszenie 18.12.2026 |
| 10 | k-bonacci | ICAPS 2027 (abstrakt 7.12, artykuł 14.12.2026) | pisanie; kampania klasyczna i piloty D-Wave/IQM zakończone 20.11 | abstrakt 7.12.2026 |
| 11 | Crossings / inwolucje | IJCAI 2027 (abstrakt 4.01, artykuł 11.01.2027) | plan v4 + kampania, szkic | abstrakt 4.01.2027 |
| 12 | Job shop | 2027/28 (plan) | plan v3 + szkic EN/PL, czeka na decyzję promotora | rozmowa G0 |

## Opublikowane

### 1. Rozdział w monografii (2025)
- **Tytuł:** Motzkin Neighborhood for the Permutation Flow Shop Problem
- **W:** Idzikowski R., Kucharski M.A. (red.), *Applications of Advanced Classical and Quantum Computing
  in Science and Technology: Current Research Developments*, Akademicka Oficyna Wydawnicza EXIT, Warszawa 2025
- **Strony:** 29–42 · **ISBN:** 978-83-7837-162-5
- Źródła: `articles/motzkin_chapter_pl` · klucz w puli: `wojewodzki2025exit`

### 2. SOCO 2026 (czerwiec 2026)
- **Tytuł:** Motzkin Neighborhood Evaluation in Iterated Local Search and Simulated Annealing for the
  Permutational Flow Shop Problem
- **W:** *Soft Computing Models in Industrial and Environmental Applications (SOCO 2026)*, Marbella,
  18–19.06.2026; CCIS vol. 3046, Springer, s. 144–155
- **DOI:** [10.1007/978-3-032-29254-4_12](https://doi.org/10.1007/978-3-032-29254-4_12)
- Offprint: `articles/motzkin_soco/SOCO2026_Motzkin_Wojewodzki_offprint.pdf` · klucz: `wojewodzki2026soco`

## W druku / przyjęte

### 3. ICAISC 2026 (w druku)
- **Tytuł:** Hybrid Neighborhood Evaluation in Tabu Search and Simulated Annealing for the Permutation
  Flow Shop Problem
- **W:** *Artificial Intelligence and Soft Computing — ICAISC 2026*, Zakopane; Springer LNAI (rozdział 30)
- **DOI/tom/strony:** brak w Crossref na 15.11.2026 — uzupełnić po publikacji tomu
- Źródła: `articles/hybrid_neighborhoods_icaisc` · poster: `presentations/icaisc2026_hybrid` · klucz: `wojewodzki2026icaisc`

### 6. GCMM 2026 (warunkowo przyjęty, Paper 59, sesja SS3)
- **Tytuł:** Gate-Model Quantum Neighborhood Search for the Permutation Flow Shop Problem via Fixed-Angle QAOA
- **Konferencja:** GCMM 2026, Wrocław, 2–4.12.2026; Springer proceedings; rejestracja przez katedrę
- **Stan:** akceptacja warunkowa 9.11.2026 (obaj recenzenci: minor revisions) → rewizja, odpowiedź recenzentom,
  Licence to Publish i nowy abstrakt wgrane do EasyChair 13.11.2026; czeka na potwierdzenie przyjęcia i korektę
  wydawnictwa
- **Kod i dane:** Zenodo, DOI 10.5281/zenodo.24106982; surowe przebiegi przechodzą do osobnego repozytorium danych
- Źródła: `articles/qaoa_gcmm`, gałąź `gcmm-rev` (scalona z `main` 22.11.2026, bez katalogu `data/`) ·
  klucz: `wojewodzki2026gcmm`

## W recenzji

### 4. BPASTS (BPASTS-05673-2026-02, rewizja wysłana)
- **Tytuł (po rewizji):** Quantum and Classical Annealing for Very Large-Scale Neighborhood Search in the
  Permutation Flow Shop Problem (pierwotnie: Classical and Quantum Neighborhoods Evaluation in Iterated Local
  Search and Simulated Annealing for the Permutation Flow Shop Problem)
- **Czasopismo:** Bulletin of the Polish Academy of Sciences: Technical Sciences
- **Stan:** major revision 5.11.2026; rewizja z listem do recenzentów wysłana 18.11.2026 (potwierdzenie redakcji
  17:52); czeka na ponowną ocenę redakcji i recenzentów. Gałąź `pan-bpasts`, commit `fd092a1`
- Źródła: `articles/bpasts_classical_quantum` · klucz: `wojewodzki2026bpasts`

### 5. Windowed QUBO (C&OR)
- **Tytuł:** Scalable Quantum Neighborhood Search for the Permutation Flow Shop Problem via Windowed QUBO
  Decomposition
- **Preprint:** SSRN 7411334, DOI [10.2139/ssrn.7411334](https://doi.org/10.2139/ssrn.7411334)
- **Czasopismo:** Computers & Operations Research (Elsevier); w recenzji; data zgłoszenia i numer rękopisu **(?)**
- Źródła: `articles/windowed_qubo` · klucz: `wojewodzki2026windowed`

### 7. ACIIDS 2027 (nr 316, Main track)
- **Tytuł:** Windowed QAOA Neighborhoods for the Permutation Flow Shop Problem on a Gate-Based Quantum Computer
- **Stan:** zgłoszony 23.10.2026 (EasyChair), decyzja: styczeń 2027; Springer LNAI
- Źródła: `articles/gate_taillard_aciids`, gałąź `gate-all-instances` · klucz: `wojewodzki2026aciids`

### 8. ICAISC 2027 (Paper 12373)
- **Tytuł:** Depth or Shots? The Number of QAOA Layers in Quantum Neighborhood Search for the Permutation
  Flow Shop Problem, Tested on a Real Quantum Computer
- **Stan:** zgłoszony 09.11.2026; decyzja od 28.02.2027; wersja finalna 25.03.2027; konferencja
  20–24.06.2027, Zakopane; Springer LNAI
- Źródła: `articles/qaoa_depth_icaisc`, gałąź `icaisc-camera-ready` na wersję finalną

## W przygotowaniu

### 9. ANNSIM 2027
- **Tytuł:** How Faithful Are Quantum Circuit Simulators? Nine Simulators and Three Noise Scenarios Checked
  Against the Physical 16-Qubit IQM Sirius
- **Termin zgłoszenia:** 18.12.2026 (ścieżka Quantum Computing, recenzja double-blind)
- **Stan:** gotowy szkic, 12 stron w wersji EN i PL po pięciu rundach krytyki; czeka na przejrzenie przez promotora
- Źródła: `articles/annsim_simulators` (scalone do `main`), gałąź `annsim-submission`

### 10. k-bonacci
- **Temat:** rodzina sąsiedztw k-bonacci (Fibonacci → Dynasearch), studium parametru k
- **Miejsce:** ICAPS 2027 (abstrakt 7.12.2026, artykuł 14.12.2026; decyzja z 18.11.2026)
- **Stan:** pisanie; kampania klasyczna (ILS, SA) i piloty D-Wave oraz IQM zakończone 20.11.2026
- Źródła: gałąź `kbonacci-icaps`, `articles/kbonacci`

### 11. Crossings / inwolucje · 12. Job shop
- **Crossings:** cel IJCAI 2027 (abstrakt 4.01.2027, artykuł 11.01.2027; decyzja z 18.11.2026); plan v4, kampania
  i szkic na gałęzi `crossings` (zdalnie `claude/tender-thompson-ql1r4e`).
- **Job shop:** plan v3 i szkic EN/PL na gałęzi `jobshop-plan-v2`; czeka na rozmowę z promotorem, czy wchodzi do
  doktoratu.

---
*Aktualizować skillem `/domkniecie` przy każdej zmianie stanu. Lista prac katedry z punktami MNiSW i IF:
cytownik (arkusz prywatny, link poza repo).*
