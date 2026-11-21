---
name: rewizja
description: Rewizja artykułu doktoratu po recenzjach i wersja finalna (camera-ready) – plan rewizji po polsku, odpowiedź recenzentom punkt po punkcie, zmiany na niebiesko, wersja czysta i źródła, Licence to Publish / copyright, Alt Text, Zenodo, paczka do wgrania. Użyj po decyzji (accept / minor / major / conditional) i przy wersji finalnej.
---

# Rewizja i wersja finalna

Wzorzec: GCMM 2026 Paper 59 (gałąź `gcmm-rev`), BPASTS (`pan-bpasts`).

**Baza wiedzy:** na start strona artykułu (`~/Desktop/Doktorat/PhD-wiki/publikacje/`) i `proces/uwagi-promotora.md` (np. w odpowiedzi teksty
recenzentów kursywą, nasze prosto). Po wgraniu streść uwagi recenzentów i nasze odpowiedzi w sekcji „Recenzje”
strony artykułu.

## Start

1. Gałąź `<konf>-rev` (albo `<konf>-camera-ready` założona po wysyłce) w osobnym folderze `~/Desktop/PhD-<konf>`.
2. Recenzje do `response/reviews.txt` (bez danych osobowych recenzentów). Termin rewizji do kalendarza PWR.
3. `response/plan_rewizji_pl.tex` – **dla nas i promotora**: każda uwaga (R1.1, R1.2, …) → co robimy, czy
   potrzebne nowe obliczenia, ile to kosztuje, gdzie w tekście. Do akceptacji właściciela przed pracą.

## Praca

- Zmiany w `main.tex` oznaczane na niebiesko (makro `\rev{...}` z `xcolor`), z tego `main_highlighted.pdf`;
  czysta wersja przez przełączenie makra. Porównanie z `submitted_v1/`.
- `response/response_to_reviewers.tex` – punkt po punkcie: cytat uwagi, odpowiedź, co zmieniono i gdzie
  (sekcja/strona/rysunek). Uprzejmie, konkretnie, bez kłótni; gdy się nie zgadzamy – argument + co mimo to
  doprecyzowano. Promotor zwykle redaguje list na Overleafie → osobny commit.
- Nowe obliczenia jak w `/badania`; dane za nowymi tabelami do repo.
- Kod i dane: snapshot w **Zenodo** (z promotorem jako współautorem w metadanych), DOI do artykułu
  (Data availability) i do listu.
- Po zmianach: `/przeglad-krytyczny` (krótsza runda) i kontrola z `/zgloszenie`.
- Wersja finalna zawsze **z nazwiskami**, normalne autocytowania, podziękowania, biogramy jeśli wymagane (ANNSIM).

## Paczka do wgrania: `~/Downloads/<KONF><ROK>_Paper<NR>_upload/`

Ponumerowane podfoldery wg pól systemu, np. `1_revised_paper/` (PDF, LaTeX), `2_responses_to_reviewers/`,
`3_highlighted/`, `4_LtP/`, `5_sources/` (main.tex, main.bbl, klasy, figures/) + `CO_GDZIE_WGRAC.txt` z mapą
pole → plik. Uwaga na pola wymagające .docx (pandoc ze źródeł LaTeX, nie konwersja z PDF), Alt Text (arkusz
wydawcy), Licence to Publish (**podpis odręczny** autora korespondencyjnego – przygotuj PDF „do podpisu”
i powiedz, którą stronę podpisać). Abstrakt w metadanych systemu podmień na nowy, jeśli się zmienił.

## Po wgraniu

Mail do organizatorów/redakcji – przygotuj do wklejenia, nie wysyłaj. Commit i push gałęzi (zapytać),
`/domkniecie`: stan „przyjęty, rewizja wgrana <data>”, kalendarz (konferencja, korekta), pamięć.
