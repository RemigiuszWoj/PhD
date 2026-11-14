---
name: nowe-badanie
description: Start nowego badania / artykułu w doktoracie – osobna gałąź, osobny folder roboczy ~/Desktop/PhD-<slug>, szkielet articles/<slug> z szablonu, wymagania konferencji, wpisy w kalendarzu PWR i notatce DOKTORAT. Użyj, gdy właściciel mówi „zaczynamy nowe badanie/artykuł”, „robimy coś na <konferencję>”.
---

# Nowe badanie

Zanim cokolwiek założysz, ustal z właścicielem w 2–3 zdaniach: temat (jedno pytanie badawcze), docelowe miejsce
(konferencja/czasopismo) i termin. Jeśli miejsce nie jest wybrane, zaproponuj 2–3 z uzasadnieniem
(ranga CORE, dopasowanie tematu, termin, ślepa recenzja, koszt) i zapytaj.

## Kroki

1. **Slug i gałąź.** Folder `articles/<temat>_<konf>` (podkreślniki), gałąź `<konf>-<temat>` albo krótka nazwa
   tematu (myślniki). Sprawdź, że nazwa nie jest zajęta: `git branch -a | grep <nazwa>`.
2. **Osobny folder roboczy** (nie ruszamy `~/Desktop/PhD`, bo tam mogą pracować inne sesje):
   ```bash
   cd ~/Desktop/PhD && git fetch origin && git worktree add -b <galaz> ~/Desktop/PhD-<slug> origin/main
   ```
   Jeśli worktree się nie nadaje (np. potrzebny jest osobny venv), sklonuj repo do `~/Desktop/PhD-<slug>`.
   Venv: skopiuj sposób uruchomienia z `README.md`/`RUN_LOCAL.md`, sekrety z `~/Desktop/PhD/.env` (nie commitować).
3. **Szkielet z szablonu:** `cp -r articles/_szablon articles/<slug>` i usuń to, czego nie będzie
   (np. `druk/` zostaje na koniec). Dołóż klasę LaTeX konferencji (llncs + splncs04 dla Springer LNCS/LNAI,
   svproc dla Springer proceedings, scspaperproc dla SCS/ANNSIM, IEEEtran/bpasts dla czasopism) – najlepiej
   skopiuj z poprzedniego artykułu w tym samym formacie.
4. **Wymagania konferencji** sprawdź na stronie (nie z pamięci) i wpisz do `STATUS.md`: limit stron (z bibliografią
   czy bez), ślepa recenzja i co z autocytowaniami, limit abstraktu w słowach **i w znakach pola formularza**,
   słowa kluczowe, system zgłoszeń, oświadczenie o AI, terminy (zgłoszenie – podaj strefę, AoE = rano następnego
   dnia w PL; decyzja; wersja finalna; konferencja). Sprzeczne źródła – trzymamy się ostrzejszego i to zapisujemy.
5. **Kalendarz PWR** (aplikacja Kalendarz na Macu): termin zgłoszenia (z alarmem 3 dni i dzień przed),
   własny termin wysyłki (zwykle 2–3 dni wcześniej), decyzja, wersja finalna, konferencja (całodniowo).
   W opisie: nazwa artykułu, link do strony. Pokaż listę i dodaj po akceptacji.
6. **Notatka DOKTORAT**: wiersz w tabeli konferencji (konferencja, termin, artykuł). Patrz `/domkniecie` – sekcja
   o edycji notatki.
7. **Pamięć projektu**: nowy `project_<slug>.md` (temat, gałąź, folder, terminy, decyzje) + wpis w `MEMORY.md`.
8. Pierwszy commit na gałęzi: `<slug>: skeleton from template, venue requirements`. Bez pusha, chyba że właściciel
   chce.
9. Przejdź do `/badania` (plan badań).

## Pułapki z historii

- Sesja dostała kiedyś pusty plan przez błędne przekazanie – nowa sesja ma dostać pełny brief: slug, gałąź,
  folder, termin, link do `PLAN_BADAN.md`.
- Nie pracuj w `~/Desktop/PhD` na gałęzi innego artykułu; sprawdź `git -C ~/Desktop/PhD branch --show-current`.
