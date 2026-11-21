---
name: zgloszenie
description: Przygotowanie zgłoszenia artykułu doktoratu do konferencji/czasopisma – wersja zanonimizowana, kontrola PDF (pdfinfo, pdffonts, strony), abstrakt w limicie słów i znaków, pola formularza, paczka źródeł, snapshot submitted_v1, a po wysyłce numer pracy, merge, kalendarz i tabele. Użyj przed wysyłką i tuż po niej.
---

# Zgłoszenie

**Nie wysyłam sam.** Przygotowuję pliki i pola; właściciel klika „Submit”. Jeśli właściciel prosi o wgranie
przez przeglądarkę – każdy krok potwierdzam przed zatwierdzeniem formularza.

**Baza wiedzy:** wymagania konferencji są też na stronie artykułu (`~/Desktop/Doktorat/PhD-wiki/publikacje/`). Tuż po wysyłce zaktualizuj
tam `status` i `paper_id` i dopisz zdarzenie do `doktorat/historia-decyzji.md` (`/wiki`, aktualizacja).

## Pliki (`articles/<slug>/submission_<KONF>/`)

- `README.md` – gdzie (URL systemu), terminy, wymagania ze strony (sprawdzone <data>), autorzy do formularza,
  co wgrać do którego pola, wynik kontroli. Szablon: `articles/_szablon/submission/README.md`.
- `tytul.txt`, `abstrakt.txt`, `slowa_kluczowe.txt` – czysty tekst bez LaTeX-a, gotowy do wklejenia.
  Abstrakt w formularzu **identyczny** z artykułem.
- Ślepa recenzja: anonimowy PDF (`\author{Anonymous}`, puste metadane) + jeśli wymagane, źródła z nazwiskami
  (`zrodla_z_nazwiskami.zip`: main.tex z blokiem autorów, .bib, .cls/.bst, figures/*_en.pdf, tables/, bez PL).
  Archiwów i PDF-ów zgłoszenia nie commitujemy (poza `submitted_v1/`).

Blok autorów (afiliacja):
```latex
\author{Remigiusz Wojew\'{o}dzki \and Wojciech Bo\.{z}ejko}
\institute{Department of Control and Quantum Computing,
  Faculty of Information and Communication Technology,
  Wroc\l{}aw University of Science and Technology, Wroc\l{}aw, Poland\\
  \email{remigiusz.wojewodzki@pwr.edu.pl, wojciech.bozejko@pwr.edu.pl}}
```
Autor korespondencyjny: Remigiusz Wojewódzki.

## Kontrola przed wysyłką (wynik wpisz do README z datą i commitem)

- [ ] kompilacja EN bez błędów, liczba stron w limicie (PL osobno, informacyjnie)
- [ ] `pdfinfo` – puste Title/Author/Subject/Keywords w wersji ślepej; `pdffonts` – wszystko osadzone, brak Type 3
- [ ] `grep` w tekście i w PDF: nazwiska, afiliacje, e-maile, „our previous”, linki do GitHuba/Zenodo z nazwiskiem,
      podziękowania (ślepa wersja)
- [ ] abstrakt: liczba słów (dwie metody liczenia) i znaków vs limit pola formularza
- [ ] liczby w tekście = tabele = dane
- [ ] DOI wszystkich pozycji sprawdzone (`/bibliografia`)
- [ ] nakładanie z naszymi pracami / iThenticate
- [ ] oświadczenie o AI i Disclosure of Interests wg wymagań konferencji

## Tuż po wysyłce

1. Numer pracy, data, kategoria i wgrane pliki do `submission_<KONF>/README.md` („WYSŁANE <data>: Paper <nr>”).
2. Snapshot wysłanej wersji: `submitted_v1/` (main.tex, klasy, figury, PDF) – baza do porównania przy rewizji.
3. Commit `<slug>: submitted to <KONF> as Paper <nr> (<data>)`, merge gałęzi do `main`, nowa gałąź
   `<konf>-camera-ready` od `main` na wersję po recenzjach.
4. Push – zapytaj; przy ślepej recenzji ostrzeż, że repo jest publiczne (źródła z nazwiskami i historia).
5. `/domkniecie` w trybie aktualizacji: tabela publikacji („w recenzji, zgł. <data>, nr <nr>, decyzja od <data>”),
   usuń wiersz z tabeli konferencji, kalendarz (decyzja, wersja finalna, konferencja), pamięć projektu.
