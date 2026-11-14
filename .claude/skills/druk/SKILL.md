---
name: druk
description: Druk egzemplarza opublikowanego artykułu doktoratu do własnej kolekcji – kiedy drukować, przygotowanie PDF (A5 dla Springer LNCS/CCIS, A4 dla dwukolumnowych czasopism), okładka, strony podzielne przez 4, gdzie zamówić i za ile, stan w STATUS.md i tabeli publikacji. Użyj, gdy praca ma DOI i recenzje są zamknięte, albo gdy właściciel mówi „drukujemy”.
---

# Druk artykułów

Cel: każda opublikowana praca jako osobny zeszyt z okładką (jak cienka broszura konferencyjna) na półkę.
Ustalenia z wątku „PDFy artykułów do druku i oprawy” (14.11.2026) – **jeśli ten wątek ustalił coś nowego
(drukarnia, format, druk samodzielny), zaktualizuj ten plik.**

## Kiedy

Warunki: jest DOI, wersja wydawcy (offprint) albo finalna wersja autorska jest w `articles/<slug>/`, recenzje
i korekta zamknięte, `STATUS.md` kompletny. Przypomnienie wychodzi z `/domkniecie`. Drukować można też partiami
(kilka prac naraz – taniej, jedna wysyłka).

## Przygotowanie (`articles/<slug>/druk/`, nie commitować dużych PDF-ów poza finalnym `do_druku.pdf`, jeśli mały)

1. **Źródło:** offprint wydawcy (najlepiej, np. SOCO), inaczej finalny PDF z nazwiskami.
2. **Format:**
   - Springer LNCS/LNAI/CCIS i svproc (oryginał 155×235 mm) → **A5**, wychodzi jak rozdział z tomu.
   - Dwie kolumny (BPASTS) i czasopisma Elsevier (C&OR) → **A4**.
3. **Okładka** (`okladka.tex`, szablon w `articles/_szablon/druk/`): tytuł, autorzy, konferencja/czasopismo,
   miejsce i rok, seria/tom/strony, DOI; opcjonalnie rysunek z artykułu.
4. **Liczba stron podzielna przez 4** (zeszyt szyty) – dopełnij pustymi stronami (`pdfjam`/`qpdf`).
5. Sprawdź podgląd: marginesy po skalowaniu, czytelność wzorów i legend w A5.

## Gdzie (ceny z 14.11.2026, 1 egz., A5, 16 stron, brutto)

| Opcja | Wariant | Cena |
|---|---|---|
| Doxdirect (online, od 1 szt., wysyłka gratis > 50 zł) | zeszyt szyty, środek cz-b, kolorowa okładka 300 g | 40,96 zł |
| Doxdirect | zeszyt szyty, cały w kolorze | 52,40 zł |
| Doxdirect | oprawa klejona, okładka laminowana mat | 64,88 zł |
| Grunwald24, Norwida 11, Wrocław (blisko kampusu) | broszury szyte/klejone, folia mat | brak cennika – mail lub tel. 536 141 024 |
| Cyfrus (Wrocław, kalkulator online) | druk szyty zeszytowo | nie odczytano |

Przy kilku pracach zapytaj o wycenę zbiorczą. Zamówienia nie składam sam – przygotowuję pliki i treść maila,
właściciel zamawia i płaci.

## Po druku

`STATUS.md`: „wydrukowany <data>, <gdzie>, <koszt>”; w notatce DOKTORAT dopisek w kolumnie stanu albo
osobnej kolumnie „druk”. Jeśli druk odłożony – zadanie w kalendarzu PWR.
