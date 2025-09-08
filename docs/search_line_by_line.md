# Tabu Search & Simulated Annealing – Line‑by‑Line Explanation

Plik: `src/search.py`

Legenda:
- Lx: numer linii w aktualnej wersji pliku (zsynchronizowano z wersją o łącznej długości 443 linii).
- Kod skracany gdy blok wielokrotny (… oznacza niezmienione / pominięte dla zwięzłości).
- Opis koncentruje się na semantyce i roli w algorytmie.

> Uwaga: Zmiana pliku wymaga ponownej regeneracji – w przeciwnym razie numeracja się zdezaktualizuje.

---
## Sekcja importów i zależności

L1: logging – logowanie przebiegu.
L2: math – exp w SA.
L3: os – katalogi (gantt / trace).
L4: random – RNG.
L5: time – limity czasu.
L6: Optional – adnotacje typów opcjonalnych.
L8: evaluate – obliczenie kosztu / harmonogramu.
L9: DataInstance, OperationKey – typy modeli.
L10: create_fibonachi_neighborhood, swap_adjacent – operatory ruchów.
L11: plot_gantt – wykres Gantta.

---
## Funkcja `tabu_search`

L14–26: Nagłówek i sygnatura wraz z typem zwrotu.
L27–43: Docstring (mechanizmy: aspiracja, częstotliwości, dywersyfikacja, fallback).
L44–46: Inicjalizacja RNG jeśli None.
L47–48: Inicjalizacja cache jeśli None.
L49–53: Ustawienie stanu początkowego i licznik ewaluacji.
L55–56: Struktura tabu (klucz ruch -> wygaśnięcie).
L57–58: Pamięć częstotliwości par.
L59–62: Parametry bazowe (tenure, ostatnia poprawa).
L63–68: Parametry stagnacji, dywersyfikacji i kary częstotliwości.
L70–75: Start czasu, historia, limit czasu.
L76–78: Tworzenie katalogu Gantt jeśli aktywny.
L79: Pętla główna iteracji.
L80–84: Sprawdzenie limitu czasu i ewentualny break.
L85–89: Reset zmiennych ruchu w tej iteracji.
L91–93: Wyliczenie stagnacji i warunek dywersyfikacji.
L94–105: Log + losowe realne swapy podczas dywersyfikacji.
L106–111: Aktualizacja po dywersyfikacji (czyszczenie tabu, lifting tenure, historia, trace, continue).
L125–129: Aktywacja / dezaktywacja kary częstotliwości.
L131–132: Start bloku generacji sąsiedztwa.
L133–138: Lista indeksów swapów i ew. wczesne wyjście gdy brak.
L139–154: Pętla ocen ruchów adjacent (koszt, adjusted, klasyfikacja admissible vs tabu_only).
L155–160: Wybór najlepszego admissible.
L161–168: Fallback do najlepszego tabu (sort po (adjusted, expiry)).
L169–176: Brak ruchów – log i break.
L178–186: Gałąź fibonachi – wygenerowanie multi‑swap perm oraz kosztu.
L187–197: Detekcja zamienionych par.
L198–205: Budowa klucza ruchu i sprawdzenie tabu.
L206–215: Obliczenie kary częstotliwości (multi vs single para).
L216–223: Selekcja dopuszczalnego lub fallback jeśli wszystko tabu.
L231: Wyjątek dla nieznanego typu sąsiedztwa.
L233–237: Brak wybranego ruchu – log i break.
L238–239: Aplikacja ruchu + koszt.
L240–254: Aktualizacja tabu + częstotliwości (multi-swap rozpisany na pary).
L255–257: Czyszczenie wygasłych wpisów.
L258–263: Aktualizacja best + reset stagnacji i tenure.
L264–265: Aktualizacja historii.
L266–273: Log iteracyjny (fallback_tabu gdy zastosowano ruch tabu).
L274–281: Zapis do trace (dodaje znacznik FALLBACK).
L282–293: Opcjonalny Gantt (reewaluacja z harmonogramem i zapis PNG).
L295: Zwrócenie wyników Tabu.

---
## Funkcja `simulated_annealing`

L298–312: Nagłówek i parametry SA (temperatura, chłodzenie, neighbor_moves, min_temp, itd.).
L314–324: Inicjalizacje RNG, cache, bieżący stan, koszt, best, temperatura, licznik ewaluacji.
L326–333: Start czasu, historia, limit, katalog Gantt.
L333–338: Warunek limitu czasu wewnątrz pętli.
L339–341: Inkrement iteracji + reset opisu ruchu.
L342–365: Gałąź swap_adjacent (wyszukiwanie realnego swapu do neighbor_moves prób).
L366–391: Gałąź fibonachi (multi‑swap + rekonstrukcja indeksów dla trace).
L392: Wyjątek nieznanego sąsiedztwa.
L393–401: Obliczenie delty i decyzja akceptacji (Metropolis).
L402–407: Aktualizacja stanu jeśli ruch zaakceptowany (oraz best).
L407–408: Chłodzenie temperatury.
L409–421: Log iteracyjny SA.
L422–431: Zapis do trace (T; current; best; delta; accepted; evals; move_repr).
L432–442: Opcjonalny Gantt snapshot.
L443: Zwrócenie wyników SA.

---
## Podsumowanie mechanizmów

Tabu Search: pełny skan sąsiedztwa, tabu lista (krótkoterminowa pamięć), aspiracja, kara częstotliwości (długoterminowa), dywersyfikacja po stagnacji, fallback przy całkowitym zablokowaniu.

Simulated Annealing: pojedynczy (lub kilka prób) losowy sąsiad na iterację, klasyczna akceptacja Metropolis, chłodzenie geometryczne, brak pamięci ruchów.

---
Jeśli potrzebujesz wersji równoległej kod | opis w tabeli – napisz.
