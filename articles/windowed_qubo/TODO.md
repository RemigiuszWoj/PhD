# TODO — windowed_qubo (Computers & Operations Research)

Stan: 2026-07-13. Tabele rpd_n20 i rpd_scaling w 100% wypełnione realnymi
danymi QPU (kampanie P1/P2/T3 lipiec + P4 11–13.07). Wilcoxon policzony
i wpisany. Wnioski/abstract/highlights zaktualizowane o wyniki.

## Przed wysyłką do promotora
- [ ] /humanizer pass na całym tekście (stały wymóg przed finalizacją)
- [ ] Ostatnie czytanie PDF strona po stronie (formatowanie, overfull boxy,
      podpisy figur) — formatowanie robimy NA KOŃCU

## Przed submisją (decyzje podjęte, pilnować spójności)
- [x] BTJL: ablacja ZMIERZONA 14.07 (zmiana decyzji z 13.07): 3600 runów
      klasycznych, dir results/experiments/20260713_225226. BTJL bije
      restart w każdej komórce m×tl o 1.1–2.6 pp (p<1e-17); per sąsiedztwo:
      adjacent +4.07, fibonacci +3.51, dynasearch +0.43, motzkin +0.25 pp.
      Wpisane do akapitu "BackTrackJumpList contribution" i wniosków.
- [ ] Siatka 6 budżetów dla quantum: tabele jej NIE wymagają; Setup
      doprecyzowany (klasyczne 6 budżetów, QPU: tabele tl=1000/5000,
      konwergencja 100–5000 enh). Opcjonalne doliczenie warstw
      tl=100/500/2000/10000 (P1 orig 1525 runów + P2 enh 914 runów,
      ~170 s quoty = 4–5 transz, ~30–40 h wall) TYLKO jeśli recenzent
      poprosi o pełne krzywe konwergencji dla klasy oryginalnej.

## PUNKT POWROTU (stan na 2026-07-14, pauza — priorytet przeszedł na nowe tematy)
- Artykuł KOMPLETNY do wysłania promotorowi: wszystkie tabele z realnymi
  danymi QPU, Wilcoxony, ablacja BTJL, humanizer+formatowanie zrobione
  (commit 8720e63, 23 strony, floaty na miejscu).
- **Otwarta decyzja: dziury "--" w Table 6 (per-limit)** — brakujące
  warstwy QPU: orig tl=100/500/2000/10000 (1525 runów), enh tl=2000/10000
  + resztki tl=100 (914 runów). Koszt: ~170 s quoty (≈ reszta lipca albo
  sierpień) + ~30–40 h wall. Skrypt gotowy do napisania na wzór
  run_qpu_p4.py; kolejność tanie-najpierw, dyn orig na końcu.
  Alternatywa: zostawić kreski (caption już to uczciwie opisuje).
- Budżet lipca po wszystkim: 194.5/~400 s zużyte, ~205 s wolne.

## Badania na przyszłość (sierpniowa quota lub później)
- [ ] **Domknąć kolumnę tl=10000 w Tab. 6 (tab:rpd_n20_tl).** Zrobione
      w P7: fibonacci orig = 17.94 (ILS) / 17.82 (SA). **P8 25.07:
      quota lipca WYCZERPANA** po dorobieniu adjacent orig ILS 11→32/50
      (SA 0/50) — żadna nowa komórka kompletna. D-Wave: "insufficient
      remaining solver access time in project fdMX". Zostaje: adjacent
      orig (wznawia 32/100), motzkin orig, dynasearch orig oraz WSZYSTKIE
      enhanced. **SKRYPT GOTOWY: scripts/run_qpu_p8.py** (--resume,
      kolejność tanie-najpierw, guard budżetu; flaga --fast = tylko
      szybkie sąsiedztwa). Szacunek pełnej kolumny: ~73 s access,
      ~6-7 h wall (dynasearch orig ~140 s/wyw.). Odpalić po odnowieniu:
      `set -a; . ./.env; set +a; nohup .venv311/bin/python3
      scripts/run_qpu_p8.py --budget 90 > results/p8.log 2>&1 &`
      (NIE pokazywać tokena). 268 runów ma failed.json(quota) —
      p8 --resume je ponowi. Po komplecie: średnie (lower_bound) do
      Tab. 6 + odświeżyć „Mean/Śr.".
- [x] Porażki embeddingu dynasearch — WSZYSTKIE odzyskane przez ponawianie
      (scripts/run_qpu_dynretry.py; loteria minorminer, ostatnia instancja 6
      tl=2000 wpadła 25.07). Wszystkie komórki dynasearch = 50/50, 0 porażek.
- [ ] Czułość na overlap ρ: dyn/motz enhanced, ρ ∈ {0.25, 0.75}
      (ρ=0.5 zmierzone), n=20 i n=50, ILS tl=5000 → ~400 runów,
      ~25–30 s quoty (1 transza), ~1 noc wall. Kod: przepuścić
      overlap_ratio przez config/runner (parametr już istnieje
      w funkcjach okienkowania). Po pomiarze wymienić zdanie
      "sensitivity to ρ remains untested".
- [ ] Future work z artykułu: adaptive window sizing, QAOA/Willow,
      cross-window interactions, hybrid flow shop.

## Administracyjne przy submisji
- [ ] Data availability: upublicznić repo kodu i wyników (obiecane
      "upon acceptance")
- [ ] Sprawdzić dane afiliacji/ORCID w elsarticle
- [ ] Highlights: limit Elsevier 85 znaków/bullet — zweryfikować długości

## Zasoby / fakty
- Budżet QPU lipca: zużyte 194.5 s / ~400 s (P1 40 + P2 40 + T3 40 +
  P4T1 40 + P4T2 34.5); reszta ~205 s.
- Katalogi wyników: 20260703_083952 (P1 orig m=5),
  20260707_203922 (P2+T3 enh m=5), 20260711_094938 (P4 scaling m=10/20).
- Skrypt kampanii: scripts/run_qpu_p4.py (--resume, --limit).
