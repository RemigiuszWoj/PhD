---
name: doktorat
description: Główne wejście do pracy nad doktoratem – rozpoznaje, o który artykuł i etap chodzi, i prowadzi przez cały zestaw skilli po kolei (nowe badanie → badania → bibliografia → artykuł → krytyka → zgłoszenie → rewizja → domknięcie → druk). Użyj, gdy właściciel mówi „robimy doktorat”, „lecimy z artykułem”, „nowe badanie”, „co dalej z <artykuł>”, albo nie wiadomo, od którego skilla zacząć.
argument-hint: "[slug artykułu albo temat nowego badania] [etap]"
---

# Doktorat – dyspozytor

Argumenty: `$ARGUMENTS` (mogą być puste).

## 1. Ustal, co robimy

- **Bez argumentów albo „stan”** → wykonaj `/stan-doktoratu` i zapytaj, którą pracę bierzemy.
- **Nowy temat** (nie ma `articles/<slug>`) → `/nowe-badanie`.
- **Istniejący artykuł** → przeczytaj `articles/<slug>/STATUS.md` (gdy go nie ma: `submission_*/README.md`,
  `PLAN*.md`, `PUBLICATIONS.md`, gałąź i ostatnie commity) i wybierz etap z tabeli. Starsze artykuły nie mają
  `STATUS.md` – załóż go z `articles/_szablon/STATUS.md` i wypełnij tym, co wynika z repo.

Sprawdź też, gdzie jesteś: `git branch --show-current` i `pwd`. Praca nad artykułem idzie w jego folderze
`~/Desktop/PhD-<slug>` na jego gałęzi, nie w `~/Desktop/PhD` na `main`.

## 2. Etap → skill

| Stan w STATUS.md (pierwszy nieodhaczony punkt) | Skill | Koniec etapu, gdy |
|---|---|---|
| 0. brak folderu / gałęzi / terminów | `nowe-badanie` | szkielet, wymagania, kalendarz, notatka |
| 1. brak planu badań lub wyników | `badania` | `WYNIKI.md` kompletne, teza oceniona |
| 2. bibliografia niezweryfikowana | `bibliografia` | `check` bez problemów |
| 3. brak tekstu | `artykul` | pełny szkic EN + PL |
| 4. szkic bez rund krytyka | `przeglad-krytyczny` | runda bez istotnych uwag, promotor zatwierdził |
| 5. gotowy, niewysłany | `zgloszenie` | numer pracy w README, merge, tabele |
| 6. są recenzje / akceptacja warunkowa | `rewizja` | paczka wgrana przez właściciela |
| 7. przyjęty / korekta / DOI | `domkniecie` | wszystkie miejsca ze stanem zgodne |
| 8. DOI jest, druku nie było | `druk` | egzemplarz zamówiony lub odłożony z datą |

Wywołaj odpowiedni skill narzędziem Skill i pracuj według niego. Po zakończeniu etapu:
1. odhacz punkt w `STATUS.md`, zrób commit,
2. uruchom `domkniecie` w trybie aktualizacji stanu, jeśli zmienił się stan widoczny na zewnątrz
   (wysłany, decyzja, przyjęty, DOI),
3. powiedz jednym zdaniem, co jest następne, i zapytaj, czy iść dalej – **nie przechodź sam** przez granice,
   które wymagają decyzji właściciela: akceptacja planu badań, akceptacja planu artykułu, wysyłka, push, druk.

## 3. Zawsze na początku sesji z artykułem

- Terminy: jeśli do terminu zgłoszenia/rewizji jest mniej niż 7 dni, powiedz to w pierwszym zdaniu.
- Liczenie w tle: sprawdź, czy na Macu coś się liczy dla tego artykułu (postęp, godzina końca).
- Niezacommitowane zmiany i pliki spoza rozmowy (`overleaf_source.zip`, `result.json`) – nie commituj cudzych.
- Baza wiedzy: przeczytaj stronę artykułu w `~/Desktop/Doktorat/PhD-wiki/publikacje/` (spis w `index.md`); brak strony → utwórz ją (`/wiki`).

## 4. Przypomnienia przekrojowe (przy każdym użyciu, jedno zdanie, jeśli dotyczy)

- praca z DOI bez wydruku → „do druku czeka: …” (`druk`),
- rozjazd `PUBLICATIONS.md` ↔ notatka DOKTORAT ↔ kalendarz PWR → zaproponuj `domkniecie`,
- nasza opublikowana praca, której nie ma w cytowniku katedry → przypomnij o dopisaniu.
