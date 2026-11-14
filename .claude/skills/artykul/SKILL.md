---
name: artykul
description: Pisanie artykułu doktoratu – plan artykułu (tytuły, teza, budżet stron), standardowy układ sekcji, zasady stylu (proza, pełne zdania, bez nagłówka tuż nad podrozdziałem, bez „przewagi kwantowej”), wersja EN + PL, figury i tabele ze skryptów, Overleaf dla promotora. Użyj przy planowaniu i pisaniu tekstu artykułu.
---

# Artykuł

## Najpierw plan (`PLAN_ARTYKULU.md`, szablon w `articles/_szablon/`)

1. Format: klasa, limit stron (z bibliografią?), ślepa recenzja, styl cytowań.
2. **3 propozycje tytułu** + rekomendacja z uzasadnieniem (tytuł mówi wprost, co porównujemy; przy sprzęcie
   kwantowym eksponujemy fizyczny procesor, np. „Tested on a Real Quantum Computer”).
3. **Teza** jako (a)…(e), z zaznaczeniem, co potwierdzone, a co czeka na etap badań.
4. Tabela sekcji z budżetem stron. 5. Bibliografia pogrupowana (klucze). 6. Treść każdej sekcji akapit po akapicie
   z cytowaniami. Plan pokazujemy właścicielowi przed pisaniem.

## Standardowy układ (konferencja ~12 stron LNCS)

| § | Sekcja | Strony | Treść |
|---|---|---|---|
| — | Abstract | 0,3 | ≤ limit słów i znaków formularza; 4–5 słów kluczowych; główny wynik liczbą |
| 1 | Introduction | 1,0–1,2 | 4 akapity: kontekst (PFSP, Taillard, ILS, sąsiedztwa) → QUBO/annealer/QAOA → luka → wkład prozą (i)…(iv) |
| 2 | Related Work | 1,0 | 3 akapity tematyczne, bez podrozdziałów |
| 3 | Problem / Neighborhood Search | 1,2 | PFSP i Cmax (1 równanie), ruch jako zbiór rozłącznych zamian, QUBO, sąsiedztwa, okna |
| 4 | Metoda (QAOA / annealer / nowe sąsiedztwo) | 1,2 | to, co nowe w tej pracy |
| 5 | Experimental Setup | 0,8 | instancje, warianty, miary (PRD = (C−C*)/C*·100%), statystyka, sprzęt, wersje pakietów |
| 6 | Results | 3,5–4 | 4 figury + 2 tabele, każda odpowiada na jedno Q |
| 7 | Discussion | 0,6–0,8 | ograniczenia, czego nie twierdzimy, zalecenia |
| 8 | Conclusions | 0,4 | jeden akapit |
| — | Acknowledgements / Disclosure | — | oświadczenie o AI (wymóg konferencji), Disclosure of Interests (Springer), Data availability → Zenodo |
| — | References | ~1,2 | patrz `/bibliografia` |

Czasopismo (BPASTS, C&OR): to samo, szerszy przegląd literatury, więcej tabel, cover letter.

## Styl (wymagania promotora i recenzentów, zebrane z rund poprawek)

- Nigdy nagłówek sekcji i od razu podrozdział – zawsze akapit wprowadzający. Przy dwóch podrozdziałach usuń oba.
- Wstęp i wkład **ciągłą prozą**, bez wypunktowań. Pełne zdania z orzeczeniem.
- Bez „quantum advantage”, „first”, „pierwsi”, „przewaga kwantowa”. Ostrożne tezy, ograniczenia nazwane wprost.
- Abstrakt prosty i opisowy, wynik liczbą („1% to 26%”, PRD w procentach). Akronimy rozwinięte przy pierwszym
  użyciu (API = adjacent pairwise interchange).
- Podpisy figur/tabel 1–2 linie, objaśnienia pod tabelą; najlepszy w kolumnie pogrubiony; średnia ± odchylenie.
- Legenda figur min. 6 pt (Springer). Fonty osadzone, bez Type 3.
- Spójna pisownia (flow shop/flow-shop – jedna forma), jedna notacja; kolizje oznaczeń poprawiać od razu.
- Wersja PL: wyrównana treściowo z EN (te same zdania, tabele, pozycje bibliografii), bez kalek z angielskiego.
  Synchronizację robi tańszy model, ale kontrola różnic jest częścią rundy krytyka.

## Technika

- Figury: `figures/<nazwa>_en.pdf` i `_pl.pdf`, generowane skryptem z `measurements/`.
- Tabele: `tables/*.tex` ze skryptu. Nigdy ręcznie przepisane liczby.
- Kompilacja: `latexmk -pdf main.tex` (i `main_pl.tex`), sprawdź liczbę stron po każdej większej zmianie.
- Overleaf: paczka `overleaf_source.zip` (main.tex, .bib, .cls/.bst, figures, tables) albo `overleaf_sync.sh`.
  Zmiany promotora z Overleafa → osobny commit „as edited by the supervisor on Overleaf”.
- `\usepackage{microtype}` i `\emergencystretch` na przepełnione linie; `\detokenize` dla DOI z podkreślnikami.

Gdy szkic jest kompletny → `/przeglad-krytyczny`.
