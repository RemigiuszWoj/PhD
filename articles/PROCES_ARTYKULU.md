# Proces: od pomysłu do wydrukowanego egzemplarza

Spisane 14.11.2026 z historii prac: SOCO 2026, ICAISC 2026, BPASTS, windowed QUBO (C&OR), GCMM 2026, ACIIDS 2027,
ICAISC 2027 (studium p), ANNSIM 2027, k-bonacci (ICAPS/IJCAI). Każdy etap ma skill w `.claude/skills/`; wejściem do całości jest `/doktorat`.
Stan konkretnego artykułu trzymamy w jego `articles/<slug>/STATUS.md` (szablon w `articles/_szablon/`).

## Etap 0 – nowe badanie (`/nowe-badanie`)

1. Slug: `<temat>_<konf>` (np. `qaoa_depth_icaisc`, `annsim_simulators`); gałąź z myślnikami (`icaisc-p-study`).
2. Gałąź od aktualnego `main`, worktree `~/Desktop/PhD-<slug>`.
3. `cp -r articles/_szablon articles/<slug>`; szablon LaTeX konferencji (llncs/svproc/scspaperproc/IEEE) do folderu.
4. Wymagania konferencji spisane w `STATUS.md`: limit stron, ślepa recenzja, limit abstraktu (słowa **i znaki**
   pola formularza), system (EasyChair, własny), terminy.
5. Wpis w notatce DOKTORAT (tabela konferencji) i w kalendarzu PWR: termin zgłoszenia, decyzja, wersja finalna.
6. Wpis w pamięci projektu (`project_<slug>.md`).

## Etap 1 – badania (`/badania`)

`PLAN_BADAN.md`: pytanie badawcze → Q1…Qn, „co już mamy (sprawdzone <data>)”, usterki wykryte przy planowaniu,
zakres (instancje, sąsiedztwa, parametry, ziarna), etapy E0…E5 z kosztem czasu, decyzje z datami.
Pilot przed kampanią, pomiar czasu, kampania w tle, wyniki do `measurements/` (JSON/CSV z ustawieniami w nazwie),
tabele i figury generowane skryptem, `WYNIKI.md` z liczbami i testem. Raport etapu do odsłuchu.

## Etap 2 – bibliografia (`/bibliografia`)

Pula `bibliography/pula.bib` (+ eksport z cytownika przez `--extra`). Artykuł bierze ~20–30 pozycji: nasze prace,
grupa promotora, klasyka PFSP, QUBO/QAOA/annealing, tematyczne. Każdy DOI sprawdzony w Crossref/DataCite.
Przy ślepej recenzji własne prace w 3. osobie.

## Etap 3 – pisanie (`/artykul`)

`PLAN_ARTYKULU.md`: 3 tytuły z rekomendacją, teza (a)…(e), tabela sekcji z budżetem stron, bibliografia
pogrupowana, treść każdej sekcji akapit po akapicie. Potem `main.tex` (EN) i wersja PL.

## Etap 4 – krytyka (`/przeglad-krytyczny`)

5–10 rund: matematyka, wyniki od zera z danych, tezy vs dowody, styl, PL vs EN; humanizer; formatowanie na końcu.
Runda promotora (Overleaf) jako osobny commit.

## Etap 5 – zgłoszenie (`/zgloszenie`)

Anonimizacja, `pdfinfo`/`pdffonts`, liczba stron, abstrakt w limicie słów i znaków, iThenticate/nakładanie z naszymi
pracami, pliki do formularza w `submission_<KONF>/`, snapshot `submitted_v1/`. Właściciel wysyła sam.
Po wysyłce: numer pracy w README, merge do `main`, push (z ostrzeżeniem o ślepej recenzji), kalendarz, notatka.

## Etap 6 – rewizja (`/rewizja`)

`response/plan_rewizji_pl.tex` (dla nas), `response/response_to_reviewers.tex` (punkt po punkcie),
zmiany na niebiesko → `main_highlighted.pdf`, czysta wersja, źródła, Licence to Publish (podpis odręczny),
Alt Text, Zenodo. Paczka w `~/Downloads/<KONF><ROK>_Paper<NR>_upload/`.

## Etap 7 – domknięcie (`/domkniecie`)

Akceptacja → korekta wydawnictwa (proofs) → DOI. W każdym kroku aktualizacja: `PUBLICATIONS.md`, `STATUS.md`,
notatka DOKTORAT (obie tabele), kalendarz PWR, pamięć projektu, ORCID i Google Scholar po DOI,
scalenie gałęzi, offprint do repo. **Na końcu przypomnienie o druku.**

## Etap 8 – druk (`/druk`)

Gdy jest DOI, wersja wydawcy i porządek w folderach: PDF do druku (LNCS → A5, dwie kolumny → A4), okładka,
strony podzielne przez 4, zamówienie. Stan druku w `STATUS.md` i w tabeli publikacji.
