# Inwolucje i krzyżowania: plan badań i plan artykułu (do przejrzenia, 11.11.2026)

Wersja 3 po decyzji o bezpieczniku, wprowadzona 11.11 do `PLAN.md` (nota „Zmiany v3”) i do `main.tex`.
Dane liczbowe: pilot z chmury i powtórzenie na Macu (`pilot/measurements_mac/`), 4 instancje × 3 ziarna, bez filtra blokowego.

---

## Część I. Plan badań

### 1. Pytanie i teza

Pytanie: czy zamiany w ruchu złożonym dla PFSP powinny móc się krzyżować? Oś nachodzenia: Dynasearch ⊂ Motzkin (N₀) ⊂ N₁ ⊂ P₁ ⊂ inwolucje.

Teza robocza (do potwierdzenia kampanią): przy wyborze addytywnym jakość ILS spada wzdłuż osi, bo nachodzenie psuje addytywność delt. Bezpiecznik nie należy do tezy: zmienia regułę wyboru ruchu, a nie otoczenie, i psuje porównywalność z wcześniejszymi pracami.

### 2. Zasady protokołu

- Badamy otoczenia, nie metaheurystyki: ILS, SA, stałe i budżety jak w poprzednich pracach, ruch złożony przyjmowany bez sprawdzenia.
- Jeden kod kandydatów, delt i filtra blokowego dla całej osi (`crossing.py`, delty z `kbonacci.compute_candidates` przy k = n).
- 120 instancji Taillarda, 5 ziaren, jeden przebieg 10 s, budżety 0,1–5 s z `move_log`.
- Wiersze główne tylko bez bezpiecznika. Bezpiecznik jest osobnym, oznaczonym wariantem dodatkowym (osobna sekcja i tabela).

### 3. Warianty

| Grupa | Warianty | Rola |
|---|---|---|
| Główne (6) | Dynasearch, Motzkin (N₀), N₁, P₁, inwolucje, Swap (najlepsza pojedyncza zamiana) | cała oś plus kontrola |
| Dodatkowe, osobny etap (5) | Dynasearch, N₀, N₁, P₁, inwolucje z bezpiecznikiem | czy kontrola rzeczywistego C_max zmienia obraz |
| Odniesienia | stary Motzkin i Dynasearch, k‑bonacci, Insert | z wcześniejszych kampanii |
| Ablacje | dokładny filtr ścieżki krytycznej dla N₀ i inwolucji; limit top‑K | trafność wyników |

Warianty `_prefix` zostają w kodzie, ale nie wchodzą do żadnego etapu. Wariantów `_safe` nie usuwamy z kodu.

### 4. Hipotezy

| ID | Hipoteza | Eksperyment | Dane z pilota |
|---|---|---|---|
| H1 | Przewidywany zysk rośnie z c, ale się nasyca: N₀ ma ≥ 60% zysku inwolucji | E1 | N₀ 64–90% na stanach z trajektorii |
| H2 | Addytywność zależy od geometrii pary: rozłączne > zagnieżdżone, krzyżujące się | E1 | realny/przewidywany 0,52–0,66 wobec 0,12–0,37 |
| H3 | **Główna.** RPD rośnie wzdłuż łańcucha Dyn ⊂ N₀ ⊂ N₁ ⊂ P₁ ⊂ inw. | E2 | Mac 50×10, 5 s: 4,93 / 13,69 / 16,18 / 16,89 / 22,82 |
| H9 | **Główna.** Dynasearch jest nie gorszy od N₀ i lepszy od Swap | E2 | 50×10: 6,24 i 4,93 wobec 17,84 i 13,69 (N₀); przy n = 20 remis |
| H8 | Liczba krzyżowań optimum inwolucyjnego rośnie z liczbą kandydatów | E1 | mediana 0–3 (n = 20) do 76 (100×10) |
| H6 | Model drugiego rzędu (poprawki par) poprawia zysk na krok | E1, E3 | niepoprawiające ruchy 31–62% → 0–4% |
| H7 | QUBO inwolucji jest rzadsze niż QUBO Motzkina | E4 | 15–58% mniej kubitów, 11 stanów |
| H4′ | Dodatkowa: bezpiecznik usuwa większość straty wariantów z nachodzeniem | E5 | 50×10: N₀ 17,84 → 5,99 (Mac, 1 s) |

H5 (równoważność z bezpiecznikiem, TOST) i margines ±0,3 pp wypadają. H9 jest teraz mocniejsza niż w wersji 2, bo nie porównujemy Dynasearch z Motzkinem z bezpiecznikiem (tam przewaga była niestabilna: Mac 1 s dał 6,24 wobec 5,99).

### 5. Eksperymenty

| E | Co | Czas | Wynik |
|---|---|---|---|
| E0 | testy, równoważność N₀ ze starym Motzkinem, model kosztu wyboru, **pilot z filtrem blokowym** na tai20_5, tai20_10, 50×10 | ~1,5 h | zgoda kierunków efektów z pilotem bez filtra (warunek startu kampanii) |
| E1 | anatomia: stany z trajektorii ILS (Dynasearch, N₀, inwolucje bez bezpiecznika), zysk przewidywany i realny, krzyżowania, pary wg geometrii, **licznik offline: w ilu stanach ruch złożony jest gorszy od najlepszej pojedynczej zamiany** | ~2 h | rys. zysku wzdłuż osi, rys. addytywności, histogram krzyżowań |
| E2 | kampania główna ILS, 6 wariantów × 120 instancji × 5 ziaren | ~14 h | tabela główna, trend wzdłuż osi |
| E3 | ablacje: dokładny filtr, limit top‑K | ~4 h | trafność |
| E5 | osobny etap: 5 wariantów z bezpiecznikiem | ~11,5 h | tabela dodatkowa |
| E4 | kwanty: wyrocznia na zapisanych stanach (sekcja 8) | zależnie od wariantu | tabela QUBO i sprzętu |

Razem klasyka: ok. 22 h bez etapu E5, ok. 33 h z nim. SA po decyzji, jak w k‑bonaccim. Kampania rusza po kampanii k‑bonacciego (konkurencja o Maca).

### 6. Statystyka (ustalona przed kampanią)

- Punkt końcowy: RPD do dolnego ograniczenia Taillarda przy 10 s; krótsze budżety opisowo.
- Klasy rozstrzygające: te, w których 10 s daje co najmniej 30 ruchów (spodziewane n ≤ 100, 9 klas). Przy n = 200 i 500 wynik opisowy (n = 500 to jeden ruch).
- Rodzina P1 (H3): inwolucje vs N₀, Wilcoxon + Holm, bloki = instancje, kryterium ≥ 7 z 9 klas.
- Rodzina P2 (H9): Dynasearch vs N₀, dwustronnie, to samo kryterium.
- Test trendu Page’a wzdłuż łańcucha Dyn ⊂ N₀ ⊂ N₁ ⊂ P₁ ⊂ inw.
- Wszystko inne jest eksploracyjne i tak opisane. H3 dodatkowo przy równej liczbie iteracji, bo warianty z krzyżowaniami robią mniej iteracji.

### 7. Ryzyka i otwarte sprawy

- **Reżim pilota.** Przy 50–300 ruchach od permutacji identycznościowej przebieg to głównie zejście. Wnioski dotyczą tego reżimu; wyjaśnienie „ruch pogarszający jako perturbacja” zostaje hipotezą.
- **Mac robi mniej iteracji niż chmura** (57 wobec 69 przy 50×10, 1 s). Do wyjaśnienia przed kampanią.
- **Wynik negatywny.** Rekomendacja „zamiany nie powinny się krzyżować” to wynik negatywny; wartością artykułu są mechanizm (H2), algorytmy wyboru i oś.
- **Ruchy złożone dają ok. 1 pp nad Swap** przy n = 50 i 0,1–0,3 pp przy n = 20. Piszemy to wprost.
- **Lemat Poissona** może być znany; sprawdzić literaturę i OEIS.
- **Terminy:** GECCO 2027 niepotwierdzony; w kalendarzu GCMM 16.11, BPASTS 3.12, ICAPS 14.12, ANNSIM 18.12.

### 8. Zasoby kwantowe

**Co mierzymy.** QPU jako wyrocznia na zapisanych stanach: dla każdego stanu trzy QUBO na tych samych kandydatach (Motzkin, inwolucje, inwolucje z poprawkami par), porównane z optimum MILP. QPU w pętli ILS daje 1–4 ruchy na przebieg, więc nie wchodzi do głównego eksperymentu.

**D-Wave (Advantage, Leap, projekt fdMX).**
- Realna kwota to **ok. 287 s QPU miesięcznie**, nie 400 s, jak zakładał plan v2 (korekta z lipca 2026). Twardy limit badań: 10% = **ok. 29 s** na kampanię.
- Koszt: ok. 30 ms QPU na wywołanie przy 100 odczytach.

| Wariant | Stany × QUBO | Wywołania | QPU | Udział miesięcznej kwoty |
|---|---|---|---|---|
| MIN | 30 × 3 (tai20_5) | 90 | ~3 s | ~1% |
| OPT | 90 × 3 (tai20_5/10/20) | 270 | ~8 s | ~3% |
| MAX | OPT + 50×10, λ, pętla tai20_5 | ~1 000 | ~30 s | ~10,5%, ponad limit: rozłożyć na dwa miesiące |

**IQM Resonance (sirius 16 kubitów, garnet 20).**
- Rozliczenie za obwód, nie za zadanie. Zmierzone stawki: 0,445 kr (p = 1), 0,54 kr (p = 2), 0,58 kr (p = 3); pomiar z października 0,75 kr. Do prognoz biorę zakres 0,45–0,75 kr za obwód dla p = 1.
- K ≤ 16 na sirius i ≤ 20 na garnet, więc bez okienkowania.

| Wariant | Obwody | Koszt (p = 1, 4096 strzałów) |
|---|---|---|
| MIN | 5 stanów × 2 QUBO = 10 | 4,5–7,5 kr |
| OPT | 10 stanów × 3 QUBO = 30 | 13–22 kr |
| MAX | 30 stanów × 3 QUBO = 90 | 40–68 kr |

Plan v2 podawał 7–10 / 20–29 / 60–86 kr (stawki 0,7–0,95 za obwód, przestarzałe); poprawiam go na powyższe.

**Który dostęp jest luźniejszy.** D-Wave. Cały wariant OPT to ok. 3% jednego miesiąca kwoty, a limit 10% daje zapas ponad trzykrotny. IQM jest ciaśniejszy: konto personalne ma 30 kr miesięcznie (OPT 13–22 kr mieści się w jednym miesiącu), a MAX wymaga konta uczelnianego (200 kr) albo dwóch miesięcy. Saldo obu kont trzeba sprawdzić na resonance.iqm.tech/jobs przed startem; po kampaniach ICAISC stan jest nieznany. Fake backendy IQM (Deneb, Adonis, Apollo, Aphrodite) dają za darmo tabelę głębokości i liczby CZ.

---

## Część II. Plan artykułu

Cel: GECCO 2027 (zwykle 8 stron plus bibliografia, do potwierdzenia w CFP); zapas SOCO 2027 albo czasopismo. Tytuł roboczy: *Should Swaps Cross? Compound-Swap Neighborhoods from Motzkin to Involutions for the Permutation Flow Shop*. Szkic: `articles/crossings/main.tex` (6 stron).

| § | Sekcja | Strony | Zawartość | Rysunki i tabele | Stan |
|---|---|---|---|---|---|
| 1 | Introduction | 0,8 | ruchy złożone, zakaz krzyżowań, pytanie, 4 wkłady, bez wypunktowań | — | napisane, wkład 4 bez bezpiecznika |
| 2 | Background | 0,7 | Dynasearch, Motzkin, VLSN przez skojarzenia, RNA, krzyżowania w skojarzeniach | — | napisane |
| 3 | The crossing family | 1,2 | definicje, dwa łańcuchy, ułamek łańcuchowy, lemat Poissona, P₁, wspólne optima lokalne | rys. 1 (geometrie), tab. 1 (rozmiary) | napisane; dowody tylko szkice |
| 4 | Exact move selection | 0,9 | DP przedziałowe, DP z rdzeniami dwułukowymi, skojarzenie, koszt, reguła plain | alg. 1 | napisane; opis DP N₁ i P₁ do sprawdzenia z kodem |
| 5 | Additivity by pair geometry | 0,9 | człon drugiego rzędu Q_ab, pomiar według geometrii, zysk przewidywany i realny | tab. 2, rys. 2–3 | napisane na danych pilota; rysunki do zrobienia |
| 6 | Experiments | 2,4 | protokół, tabela główna (6 wariantów, 12 klas), trend wzdłuż osi, Dynasearch vs Swap | tab. 3–4, rys. 4 | szkielet plus tabela pilota z Maca (bez bezpiecznika) |
| 7 | A safeguarded variant | 0,5 | reguła bezpiecznika, osobna tabela, co zmienia | tab. 5 | napisana 11.11 (tabela z Maca) |
| 8 | Quantum structure | 0,6 | QUBO trzech modeli, wyrocznia na stanach | tab. 6 | napisane krótko; zależy od E4 |
| 9 | Conclusions | 0,3 | „zamiany nie powinny się krzyżować”, ruchy złożone a Swap, model par jako kierunek | — | do napisania po kampanii |

Zmiany wprowadzone 11.11 po decyzji o bezpieczniku: abstrakt i wkład 4 bez bezpiecznika; akapit „Acceptance rules” skrócony do reguły plain; tabela pilota i szkielet tabeli głównej tylko z wariantami bez bezpiecznika (liczby z Maca, z N₁ i P₁); dwie rodziny testów zamiast czterech; nowa sekcja 7.

Co jeszcze brakuje do wysłania: dowód lematu Poissona i część o addytywności (dowody, kontrprzykłady), bibliografia zweryfikowana w Crossref, prace własne w trzeciej osobie, wersja polska, decyzja D11 (osobny artykuł czy wspólny z k‑bonaccim).
