---
name: wiki
description: Baza wiedzy doktoratu (prywatne repo PhD-wiki w ~/Desktop/Doktorat/PhD-wiki) – odpowiadanie na pytania z bazy, dodawanie nowych źródeł (praca promotora, PDF, notatka, link), aktualizacja stron po zmianie stanu artykułu, przegląd spójności. Użyj, gdy właściciel pyta o cokolwiek z doktoratu („co wiemy o…”, „jak działa…”, „gdzie jest…”), mówi „dodaj do bazy/wiki”, „zaktualizuj bazę”, „przegląd bazy”, albo gdy inny skill każe zajrzeć do bazy.
---

# Baza wiedzy doktoratu (`/wiki`)

Baza: **`~/Desktop/Doktorat/PhD-wiki`** – osobne, **prywatne** repo git (wzór „LLM wiki” Karpathy'ego: źródła
w `raw/`, z nich strony markdown utrzymywane przez Claude'a). Zasady i format stron: `CLAUDE.md` w bazie –
przeczytaj go przed pierwszą zmianą w sesji. Brak folderu → powiedz właścicielowi (repo trzeba sklonować).

Układ: `doktorat/` (przegląd, plan publikacji, ludzie, pomysły, historia decyzji, worktree), `publikacje/`
(strona na każdy nasz artykuł), `pojecia/` (PFSP, sąsiedztwa, QUBO, QAOA, sprzęt), `kod/`, `proces/` (cykl artykułu,
skille, styl, uwagi promotora, narzędzia, zasady), `literatura/` (cytownik, prace promotora, kluczowe źródła), `raw/`.

## Zapytanie

1. `index.md` albo `python3 ~/Desktop/Doktorat/PhD-wiki/tools/wiki.py szukaj "<słowa>"` → 1–3 strony.
2. Czytaj tylko te strony, idź po `[[linkach]]` (plik `<nazwa>.md` w dowolnym folderze bazy).
3. Liczby i stany przed użyciem w artykule lub decyzji sprawdź w źródle z `sources:` (wiki to streszczenie).
4. Odpowiedź, która się jeszcze przyda (analiza, porównanie, ustalenie), zapisz jako stronę albo dopisz do
   istniejącej – zapytaj właściciela jednym zdaniem, jeśli to coś więcej niż drobny dopisek.

## Dodanie źródła (ingest)

1. Plik do `raw/<rodzaj>/<nazwa>_<RRRR-MM-DD>.<ext>` (PDF, .tex, eksport). **Usuń sekrety** (tokeny, hasła) –
   sprawdź `grep -n -i -E 'token|hasl|password|key|olp_'`. Klon Overleafa: `~/Desktop/Doktorat/overleaf/<nazwa>`
   (git, token w pęku kluczy), do `raw/` kopia z datą.
2. Napisz stronę źródła (zwykle `literatura/`) i uzupełnij strony, których dotyczy (pojęcia, publikacje) – jedno
   źródło zwykle zmienia kilka stron. Sprzeczności z istniejącą treścią zapisz wprost (obie wersje + skąd).
3. Wpis w `log.md` (`## [RRRR-MM-DD] ingest | …`), `python3 tools/wiki.py index`, `python3 tools/wiki.py lint`.

## Aktualizacja po zmianie stanu (woła ją też `/domkniecie`)

Strona artykułu w `publikacje/`: pola `status`, `updated`, sekcja „Stan (data)”; tabela w
`doktorat/plan-publikacji.md`; decyzja z datą w `doktorat/historia-decyzji.md`; wpis w `log.md`.

## Przegląd (lint)

`python3 tools/wiki.py lint` (zepsute linki, brak pól, strony bez linków przychodzących), potem ręcznie:
stany w `plan-publikacji.md` vs `~/Desktop/PhD/PUBLICATIONS.md` i pamięć `project_publikacje.md`;
strony `updated` starsze niż 30 dni przy aktywnych artykułach; pojęcia używane bez własnej strony;
punkty z `doktorat/do-wyjasnienia.md`. Raport: krótka lista „co poprawić”, poprawki po akceptacji.

## SKRZYNKA w notatce DOKTORAT (przy każdym przeglądzie)

Notatka „DOKTORAT” w Notatkach Apple ma sekcję **SKRZYNKA** na luźne zapiski właściciela (z telefonu, ze spotkań
z promotorem). Przy przeglądzie:
1. Odczyt tylko do czytania: `osascript -e 'tell application "Notes" to get plaintext of (first note whose name is "DOKTORAT")'`;
   zapiski to linie między nagłówkiem SKRZYNKA a „KONFERENCJE”.
2. Każdy zapisek rozłóż na właściwą stronę bazy (pomysł → `doktorat/pomysly.md`, uwaga promotora →
   `proces/uwagi-promotora.md`, termin → `doktorat/plan-publikacji.md`, sprawa → `doktorat/do-wyjasnienia.md`), wpis w `log.md`.
   Niejasny zapisek: zapytaj właściciela, nie zgaduj.
3. Pokaż właścicielowi listę „zapisek → strona” i dopiero po akceptacji usuń zapiski z notatki – **tylko w aplikacji,
   w trybie pełnego ekranu** (przeciągnięcie po zapiskach, Delete). Nigdy `set body` (kasuje zdjęcia) ani edycja w tle
   (podmienia zły zakres). Przed i po: `get body` – liczba `<table` (4) i `<img` (2) bez zmian.

## Commit i prywatność

- Po sesji zmian: `git -C ~/Desktop/Doktorat/PhD-wiki add -A && git commit -m "wiki: <co>"`. Push tylko na polecenie.
- Baza jest prywatna, repo PhD **publiczne**: treści z bazy (stany recenzji, cytownik, szkice promotora, notatki)
  nie trafiają do repo PhD bez sprawdzenia. Ścieżki i ogólny opis bazy mogą być w repo PhD.
- Notatki Apple czytamy tylko te o doktoracie; prywatnych nie otwieramy.
