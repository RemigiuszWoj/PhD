# windowed_qubo — status brancha `2026-07-02-windowed-qubo-restart`

## Po co ten branch
Artykuł **„Scalable Quantum Neighborhood Search for the Permutation Flow Shop
Problem via Windowed QUBO Decomposition"** — cel wydawniczy **Computers &
Operations Research** (Elsevier, `elsarticle`).
EN: `main.tex`; PL (wersja do czytania): `wersja_pl.tex`.

## Na czym stanęło (2026-08-09)
- **Tabela 6 (`tab:rpd_n20_tl`) KOMPLETNA** — cała kolumna `tl=10000` wypełniona:
  8 wariantów kwantowych (adjacent / fibonacci / dynasearch / motzkin ×
  orig / enhanced) × ILS + SA × **50 runów**, realny QPU D-Wave Advantage. **800/800**.
- Layout tabeli naprawiony (`\footnotesize` + `\arraystretch 0.95` — wcześniej
  stopka wchodziła na numer strony).
- **EN i PL zsynchronizowane** (te same liczby i pogrubienia w Tabeli 6).
- Kompiluje się czysto (0 błędów / Overfull / undefined ref-cite); humanizer czysto.

## Metoda liczenia RPD (żeby nie zgadywać za miesiąc)
Komórka = średnia `gap_percent` po 50 runach, gdzie
RPD = (Cmax − C*)/C* · 100, a **C\* = dolna granica Taillarda** (`lower_bound`
w `result.json`). Średnie kolumnowe = po 4 sąsiedztwach; wierszowe „Mean/Śr." =
po 6 budżetach; narożnik = grand-mean po wszystkich komórkach bloku.
Zwalidowane odtworzeniem Fibonacci 17.94/17.82 z `result.json`.

## Dane / skrypty
- Wyniki orig: `results/experiments/20260703_083952/`
- Wyniki enhanced: `results/experiments/20260707_203922/`
- `scripts/run_qpu_p8.py` — kampanie budżetowe QPU (`--budget X` s QPU-access,
  `--resume`, fazy od najlżejszych; dynasearch orig ~140 s/run wall-clock).
- `scripts/retry_embed_stragglers.py` — retry embeddingu 4 komórek, które padły
  na minorminerze (usuwa `failed.json` → wymusza świeże szukanie; miss = za darmo).

## Otwarte / do przejrzenia
- Proza EN ~linia 641 (+ odpowiednik PL): „the QPU campaigns report 1000 and
  5000 ms" — możliwie nieaktualne (jest już 6 budżetów); sprawdzić, czy nie
  odnosi się do Tabeli 2 (ta z `±std`). **Nie zmieniane.**
- PL `\date{... 17.07.2026 ...}` (~linia 29) — datownik nieaktualny (treść
  sierpniowa). Do odświeżenia.
- Niespójny case nagłówków sekcji (kosmetyka).

## Bezpieczeństwo
`pwr_recruitment/` (prywatne: CV, dyplomy, formularze) i `szkola_doktorska_2026/`
są w `.gitignore` — **NIGDY nie commitować/pushować.**
