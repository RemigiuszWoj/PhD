# Lista zmian do artykułu Fibonaci (ICAISC 2026)

**Paper ID:** 12054  
**Title:** Hybrid Neighborhood Evaluation in Tabu Search and Simulated Annealing for the Permutation Flow Shop Problem  
**Decision:** CONDITIONALLY ACCEPTED  
**Deadline:** 7 kwietnia 2026  

---

## Wymagania formalne (z emaila ICAISC)

### 1. Format Springer LNCS ✓
- Artykuł MUSI być ściśle zgodny z formatem LaTeX LNCS
- Link: https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines
- Status: **Już zrobione** (używamy `\documentclass[runningheads]{llncs}`)

### 2. Tytuł — kapitalizacja TODO
- Każde słowo znaczące w tytule musi zaczynać się wielką literą
- Obecny: "Motzkin Neighborhood Evaluation in Iterated Local Search and Simulated Annealing for the Permutational Flow Shop Problem"
- Poprawny format: Każde Content Word z Wielką Literą
- Status: **DO SPRAWDZENIA**

### 3. Afiliacje po angielsku ✓
- Nazwy instytucji muszą być w języku angielskim
- Status: **Już zrobione** ("Wrocław University of Science and Technology, Poland")

### 4. Emaile w afiliacji ✓
- Dodać adresy email autorów
- Status: **Już zrobione** (remigiusz.wojewodzki@gmail.com, wojciech.bozejko@pwr.edu.pl)

### 5. Brak słów w innych językach ✓
- Cały tekst musi być po angielsku
- Status: **Już zrobione** (artykuł po angielsku)

### 6. Referencje w innych językach TODO
- Przetłumaczyć tytuły na angielski i dodać oznaczenie (In Polish) itp.
- Przykład:
  ```
  Tadeusiewicz R.: Neural Networks. RM Academic Publishing House, Warsaw (1993) (In Polish)
  ```
- Status: **DO ZROBIENIA** — sprawdzić bibliografię

### 7. Korekta angielskiego TODO
- Tekst musi być zweryfikowany przez native speakera
- Zalecane narzędzie: www.grammarly.com
- Status: **DO ZROBIENIA** — przepuścić przez Grammarly

### 8. Formularz copyright TODO
- Zeskanowany, osobny plik ZIP (nazwa: `12054_cp.zip`)
- Status: **DO ZROBIENIA**

### 9. Lista poprawek TODO
- Osobny plik wg szablonu: http://icaiscsystem.icaisc.eu/listofimprovements.pdf
- Nazwa pliku: `12054_cor.zip`
- Status: **DO ZROBIENIA**

### 10. Deadline: 7 kwietnia 2026 ⚠️
- Po tym terminie artykuł NIE zostanie opublikowany
- Także płatność musi być przed tym terminem

---

## Poprawki merytoryczne (z recenzji Rev. 1)

### 11. Dynasearch — pełniejszy opis algorytmu HIGH PRIORITY
**Problem:** Recenzent wskazuje, że dynasearch jest "under-specified" ("details omitted for brevity")

**Do zrobienia:**
- Dodać szczegółowy pseudokod algorytmu dynasearch w sekcji 3 lub appendixie
- Opisać dokładnie jak działa wariant rekurencyjny
- Wyjaśnić dekompozycję na segmenty i transformacje
- Dodać przykład działania na małej instancji

**Gdzie:** Sekcja o dynasearch neighborhood (obecnie bardzo ogólnikowa)

### 12. Porównanie ze state-of-the-art HIGH PRIORITY
**Problem:** Brak odniesienia do najlepszych znanych metod flow-shop; badanie jest "wewnętrzne"

**Do zrobienia:**
- Dodać tabelę z Best Known Solutions (BKS) z literatury
- Porównać z wynikami:
  - Nowicki & Smutnicki (1996) — TSAB algorithm
  - Taillard (1993) — upper bounds
  - Inne state-of-the-art ILS/VNS variants
- Dodać kolumnę w tabelach z % odchylenia od BKS
- Dodać dyskusję: "nasze wyniki vs najlepsze znane"

**Gdzie:** 
- Nowa sekcja "Comparison with State-of-the-Art" przed Discussion
- Rozszerzyć tabele wyników

### 13. Dynasearch w tabelach PRD MEDIUM PRIORITY
**Problem:** Dynasearch wyłączony z tabel — recenzent chce choć częściowe statystyki

**Do zrobienia:**
- Dodać tabelę z wynikami dynasearch dla dłuższych budżetów (5000ms, 10000ms)
- Pokazać, że dla większych limitów dynasearch działa
- Dodać kolumnę "avg iteration time" żeby uzasadnić wyłączenie dla krótkich limitów
- Dodać footnote: "Dynasearch excluded from short time limits due to iteration cost exceeding budget"

**Gdzie:** 
- Nowa tabela w sekcji Results
- Footnote w istniejących tabelach

### 14. Pozycjonowanie w literaturze MEDIUM PRIORITY
**Problem:** Słabe pozycjonowanie względem literatury VLSN i flow-shop

**Do zrobienia:**
- Rozszerzyć sekcję Introduction/Related Work
- Dodać więcej referencji do:
  - VLSN (Ahuja et al. 2002 — już jest, rozwinąć dyskusję)
  - Flow shop metaheuristics (Nowicki & Smutnicki, Ruiz & Stützle survey)
  - Composite neighborhoods (więcej przykładów z literatury)
- Jasno powiedzieć: "nasza praca nie wprowadza nowej metaheurystyki, ale systematycznie porównuje struktury sąsiedztw"
- Podkreślić: wartość dodana = empiryczna charakteryzacja trade-offs

**Gdzie:** Sekcja 1 Introduction — dodać paragraf "Related Work and Positioning"

---

## Uwagi dodatkowe

### Z recenzji:
- **Mocne strony (do zachowania):**
  - Jasna i dobrze opisana idea porównania
  - Metodologiczna uczciwość (wyłączenie dynasearch z uzasadnieniem)
  - Skromne, ale uzasadnione wnioski
  
- **Słabe strony (uwzględnione w punktach 11-14):**
  - Dynasearch niedospecyfikowany
  - Brak porównania z SOTA
  - "Internal" comparison
  - Weak positioning vs literature

### Uwaga recenzenta:
> "Rather a borderline paper. The paper could become suitable with stronger positioning against the literature, more detailed algorithmic descriptions, and a more convincing experimental comparison to recognized state-of-the-art methods."

**Wniosek:** Artykuł zaakceptowany warunkowo, ale wymaga solidnej poprawy zwłaszcza w punktach 11-12.

---

## Pliki do przygotowania (deadline: 7 kwietnia 2026)

1. **12054.zip** — główny plik artykułu:
   - `fibonaci.tex` (poprawiony)
   - Wszystkie pliki źródłowe (.tex, .eps, figures/)
   - PDF wygenerowany z .tex
   
2. **12054_cp.zip** — zeskanowany formularz copyright

3. **12054_cor.zip** — lista poprawek wg szablonu ICAISC

4. **Płatność:**
   - 6100 PLN (on-site) lub 2800 PLN (online)
   - Deadline: również 7 kwietnia 2026
   - W przelewie podać numer paperu: 12054

---

## Kolejność działań (propozycja)

### Faza 1: Poprawki merytoryczne (2-3 tygodnie)
1. Punkt 11 — dodać opis dynasearch
2. Punkt 12 — dodać porównanie z BKS/SOTA
3. Punkt 13 — dodać tabelę dynasearch dla długich limitów
4. Punkt 14 — rozszerzyć Related Work

### Faza 2: Formatowanie i review (1 tydzień)
5. Punkt 2 — sprawdzić kapitalizację tytułu
6. Punkt 6 — sprawdzić bibliografię (tłumaczenia)
7. Punkt 7 — Grammarly full check
8. Wygenerować PDF, sprawdzić format LNCS

### Faza 3: Dokumentacja (3 dni)
9. Punkt 8 — przygotować copyright form
10. Punkt 9 — przygotować list of improvements
11. Zapakować wszystko do 3 plików ZIP

### Faza 4: Submission (przed 7 kwietnia)
12. Upload 12054.zip, 12054_cp.zip, 12054_cor.zip
13. Zapłacić conference fee
14. ✓ DONE

---

## Notatki

- Recenzent podkreśla, że to "borderline paper" ale ACCEPT
- Kluczowe są punkty 11-12 (opis dynasearch + porównanie z SOTA)
- Mamy dużo czasu (do 7 kwietnia), nie spieszyć się
- Lepiej zrobić solidnie niż szybko

