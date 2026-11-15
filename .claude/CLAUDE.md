# Doktorat – Remigiusz Wojewódzki (PWr, WIT, Katedra Sterowania i Obliczeń Kwantowych)

Promotor: **prof. Wojciech Bożejko** (współautor wszystkich prac). Rozmawiamy po polsku, artykuły piszemy po angielsku
i trzymamy wersję polską obok (`main_pl.tex` albo `wersja_pl.tex`), wyrównaną treściowo z angielską.

## O czym są badania

Sąsiedztwa w przeszukiwaniu lokalnym dla **permutacyjnego problemu przepływowego (PFSP, Cmax)**, benchmarki Taillarda,
ILS i SA startujące z permutacji tożsamościowej. Rodzina sąsiedztw z rozłącznych zamian:
Adjacent ⊂ Fibonacci ⊂ k-bonacci ⊂ Dynasearch ⊂ Motzkin ⊂ inwolucje (katalog: `articles/katalog_sasiedztw.md`).
Wybór ruchu zapisujemy jako **QUBO** i rozwiązujemy klasycznie (DP, SA-sampler), na **annealerze D-Wave**
(okienkowa dekompozycja QUBO) i na **komputerach bramkowych przez QAOA** (IQM Sirius – 16 kubitów, gwiazda; IBM).
Miary: Cmax, PRD/RPD (do BKS lub dolnego ograniczenia), testy Wilcoxona. Cele doktoratu i fronty: `ROADMAP.md`,
pomysły: `research_ideas.md`, lista publikacji: `PUBLICATIONS.md`.

Nie piszemy „przewaga kwantowa”, „pierwsi”, „quantum advantage”. Piszemy „gate-based”, nie nazwy niedostępnego sprzętu.

## Jak pracujemy (cykl życia artykułu)

Pełny opis i lista kontrolna: **`articles/PROCES_ARTYKULU.md`**. Wejście: **`/doktorat`** – sam rozpoznaje etap i prowadzi dalej.
W skrócie, każdy etap ma swój skill:

| Etap | Skill | Wynik |
|---|---|---|
| 0. Nowe badanie: gałąź, osobny folder, szkielet | `/nowe-badanie` | `articles/<slug>/` z szablonu, `~/Desktop/PhD-<slug>` |
| 1. Badania: pytania, etapy E0…, kampanie, wyniki | `/badania` | `PLAN_BADAN.md`, `measurements/`, `WYNIKI.md` |
| 2. Bibliografia: pula, Crossref, autocytowania | `/bibliografia` | `references.bib` z puli `bibliography/pula.bib` |
| 3. Pisanie: układ, styl, EN+PL, figury | `/artykul` | `main.tex`, `main_pl.tex`, `PLAN_ARTYKULU.md` |
| 4. Rundy krytyka, humanizer, promotor | `/przeglad-krytyczny` | poprawki, commit po każdej rundzie |
| 5. Zgłoszenie: anonimizacja, kontrola PDF, formularz | `/zgloszenie` | `submission_<KONF>/`, `submitted_v1/` |
| 6. Rewizja po recenzjach, camera-ready | `/rewizja` | `response/`, `main_highlighted.pdf`, LtP |
| 7. Domknięcie: DOI, tabele, kalendarz, ORCID, druk | `/domkniecie` | wszystkie miejsca ze stanem zaktualizowane |
| 8. Druk egzemplarza do półki | `/druk` | `druk/` z PDF do drukarni |
| przekrojowo: stan wszystkich prac | `/stan-doktoratu` | przegląd terminów i tabel |
| **wejście do całości** | **`/doktorat`** | rozpoznaje etap i woła właściwy skill |

## Gdzie co jest (Mac)

- Repo główne: `~/Desktop/PhD` (gałąź `main`). **Każdy artykuł ma osobną gałąź i osobny folder roboczy**
  `~/Desktop/PhD-<slug>` (git worktree), żeby sesje nie deptały sobie po plikach. Nazwy gałęzi jak dotąd:
  `icaisc-p-study`, `annsim-simulators`, `gate-all-instances`, `gcmm-rev`, `pan-bpasts`, `kbonacci-icaps`,
  `icaisc-camera-ready`. Po wysyłce gałąź artykułu scalamy do `main`; rewizja i wersja finalna idą na nowej gałęzi
  (`<konf>-rev`, `<konf>-camera-ready`).
- Pliki do wgrania do systemu konferencji: `~/Downloads/<KONF><ROK>_Paper<NR>_upload/` z ponumerowanymi podfolderami
  (np. `2_responses_to_reviewers/`, `4_LtP/`).
- **Notatka „DOKTORAT”** w Notatkach Apple: tabela publikacji (nr, tytuł, stan, konferencja/czasopismo, daty, link)
  i tabela konferencji (termin, artykuł). Edytować ostrożnie: sprawdzić po edycji, że załączniki i zdjęcia zostały.
- **Kalendarz „PWR”**: terminy zgłoszeń, decyzji, wersji finalnych, konferencje (całodniowe, z alarmem tydzień
  i dzień przed, w opisie numer pracy i link).
- Pamięć projektu Claude: `~/.claude/projects/-Users-remigiuszwojewodzki-Desktop-PhD/memory/` (`project_*.md`,
  m.in. `project_konferencje_plan.md`). Po każdej zmianie stanu artykułu aktualizować odpowiednią notatkę.
- **Cytownik katedry** (Google Sheets): lista prac grupy z BibTeX-em, punktami MNiSW i IF –
  arkusz prywatny, link poza repo (identyfikator w zmiennej `CYTOWNIK_ID` w `.env`);
  `python tools/bib_pool.py cytownik` → `bibliography/cytownik.bib` (lokalnie, poza gitem).
  Synchronizacja w obie strony: skill `/cytownik`.
- Overleaf: promotor czyta i poprawia na Overleafie; synchronizacja skryptem `overleaf_sync.sh` albo paczką
  `overleaf_source.zip`. Zmiany promotora z Overleafa wracają do repo osobnym commitem.
- Kod i dane do artykułu: snapshot w **Zenodo** (DOI w artykule i w odpowiedzi recenzentom).

## Zasady stałe

- Commity po angielsku, prefiks folderu artykułu: `qaoa_gcmm: revision for GCMM 2026 Paper 59`.
  Na końcu wiadomości linia `Co-Authored-By` zgodnie z bieżącą instrukcją środowiska.
- **Nie wysyłam niczego za właściciela** (formularze konferencji, maile, EasyChair) bez wyraźnego „wyślij”.
  Mail do organizatorów przygotowuję do wklejenia. Push na GitHub: zapytać, chyba że właściciel już kazał.
- Repo jest **publiczne**: przy recenzji dwustronnie ślepej ostrzec przed wypchnięciem źródeł z nazwiskami.
  Sekretów (token D-Wave, IBM, IQM z `.env`) nigdy nie commitować.
- Odpowiedzi krótkie, wyniki słownie i w tabelce. Długie teksty (plan, raport) właściciel lubi odsłuchać –
  zaproponować wersję audio.
- Modele: przeglądy merytoryczne (matematyka, wyniki, tezy) na najmocniejszym modelu; mechaniczne prace
  (synchronizacja PL z EN, formatowanie, przepisywanie tabel) na tańszym.
- Długie obliczenia liczą się w tle na Macu łańcuchem etapów; przed pełną kampanią zmierzyć czas na małym
  wycinku i podać godzinę końca.
- Po każdej zmianie stanu artykułu: `/domkniecie` w trybie „aktualizacja stanu” (tabele, kalendarz, pamięć).
