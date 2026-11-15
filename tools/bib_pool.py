#!/usr/bin/env python3
"""Wspólna pula bibliografii doktoratu.

Trzy polecenia (tylko biblioteka standardowa):

  cytownik pobiera arkusz „cytownik” katedry (Google Sheets, eksport CSV) i zapisuje
           bibliography/cytownik.bib: jeden wpis na prace, klucze ASCII i bez powtorzen.
  build    zbiera wszystkie articles/**/*.bib (i pliki podane w --extra, np. eksport
           z cytownika) do bibliography/pula.bib; duplikaty po kluczu i po DOI,
           konflikty wypisuje zamiast zgadywać.
  check    sprawdza DOI z pliku .bib w Crossref (a gdy brak, w DataCite) i porównuje
           tytuł; wypisuje rozjazdy. Wymaga sieci (api.crossref.org, api.datacite.org).
  extract  bierze klucze z \\cite{...} w pliku .tex i zapisuje podzbiór puli do
           references.bib artykułu (z kolejnością jak w puli).

Przykłady:
  python tools/bib_pool.py build
  python tools/bib_pool.py build --extra ~/Documents/cytownik.bib
  python tools/bib_pool.py check bibliography/pula.bib
  python tools/bib_pool.py extract articles/nowy/main.tex -o articles/nowy/references.bib
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POOL = REPO / "bibliography" / "pula.bib"
CYTOWNIK_BIB = REPO / "bibliography" / "cytownik.bib"
# arkusz katedry jest prywatny: identyfikator tylko ze zmiennej srodowiskowej (np. z .env), nie w repo
CYTOWNIK_ID = os.environ.get("CYTOWNIK_ID", "")
CYTOWNIK_URL = f"https://docs.google.com/spreadsheets/d/{CYTOWNIK_ID}/edit#gid=0"
CYTOWNIK_CSV = f"https://docs.google.com/spreadsheets/d/{CYTOWNIK_ID}/export?format=csv&gid=0"


def parse_bib(text: str) -> list[dict]:
    """Prosty parser: dzieli plik na wpisy @typ{klucz, ...} liczac nawiasy."""
    entries = []
    i = 0
    while True:
        m = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.S).search(text, i)
        if not m:
            break
        etype, key = m.group(1).lower(), m.group(2)
        if etype in ("comment", "preamble", "string"):
            i = m.end()
            continue
        depth, j = 1, text.index("{", m.start()) + 1
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        raw = text[m.start():j]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", raw):
            name = fm.group(1).lower()
            k = fm.end()
            if k < len(raw) and raw[k] == "{":
                d, s = 1, k + 1
                k += 1
                while k < len(raw) and d:
                    d += {"{": 1, "}": -1}.get(raw[k], 0)
                    k += 1
                fields.setdefault(name, raw[s:k - 1])
            elif k < len(raw) and raw[k] == '"':
                e = raw.index('"', k + 1)
                fields.setdefault(name, raw[k + 1:e])
            else:
                e = re.compile(r"[,}\n]").search(raw, k)
                fields.setdefault(name, raw[k:e.start() if e else len(raw)].strip())
        entries.append({"type": etype, "key": key, "raw": raw.strip(), "fields": fields})
        i = j
    return entries


def norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.replace("\\_", "_")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\.|[{}]", "", t.lower()))


def ascii_key(key: str) -> str:
    import unicodedata
    key = unicodedata.normalize("NFKD", key.replace("\u0142", "l").replace("\u0141", "L"))
    return re.sub(r"[^A-Za-z0-9_:-]", "", key)


def cmd_cytownik(args) -> int:
    import csv
    import io
    if args.csv:
        text = Path(args.csv).expanduser().read_text(encoding="utf-8")
    else:
        if not CYTOWNIK_ID:
            print("Brak CYTOWNIK_ID w srodowisku (identyfikator arkusza katedry) – ustaw go albo podaj --csv.")
            return 1
        req = urllib.request.Request(CYTOWNIK_CSV, headers={"User-Agent": "PhD-bib/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    seen_ids, used, out, missing = set(), {}, [], []
    for r in rows[1:]:
        if len(r) < 8 or not r[0].strip() or r[0] in seen_ids:
            continue
        seen_ids.add(r[0])
        rid, year, title, topic, pts, bib = r[0], r[1], r[3].strip(), r[4], r[5], r[7].strip()
        if not bib:
            if title:
                missing.append(f"{rid}: {title}")
            continue
        entries = parse_bib(bib)
        if not entries:
            missing.append(f"{rid}: {title} (nieczytelny BibTeX)")
            continue
        e = entries[0]
        key = ascii_key(e["key"])
        t = norm_title(e["fields"].get("title", ""))
        base, n = key, 0
        while key in used and used[key] != t:      # ten sam klucz, inna praca
            n += 1
            key = base + "abcdefgh"[n]
        used[key] = t
        raw = re.sub(r"\{\s*[^,\s]+\s*,", "{" + key + ",", e["raw"], count=1)
        out.append(f"% cytownik ID {rid} | rok {year} | {topic} | MNiSW {pts or '-'}\n{raw}")
    CYTOWNIK_BIB.parent.mkdir(exist_ok=True)
    CYTOWNIK_BIB.write_text(
        "% Eksport z cytownika katedry (Google Sheets). NIE edytowac recznie.\n"
        "% Odtwarzanie: python tools/bib_pool.py cytownik && python tools/bib_pool.py build\n"
        "% Wpisy pochodza z Google Scholar: przed cytowaniem uzupelnic DOI, tom i strony (check).\n\n"
        + "\n\n".join(out) + "\n", encoding="utf-8")
    print(f"Zapisano {CYTOWNIK_BIB.relative_to(REPO)}: {len(out)} prac.")
    if missing:
        print(f"W cytowniku bez BibTeX-a ({len(missing)}):")
        for m in missing:
            print("  -", m)
    return 0


def cmd_build(args) -> int:
    sources = sorted((REPO / "articles").rglob("*.bib"))
    sources = [p for p in sources if "submitted_v" not in str(p) and "_szablon" not in str(p)]
    if CYTOWNIK_BIB.exists():
        sources.append(CYTOWNIK_BIB)      # po artykulach: wpisy z DOI z artykulow maja pierwszenstwo
    sources += [Path(p).expanduser() for p in (args.extra or [])]
    by_key: dict[str, dict] = {}
    by_doi: dict[str, str] = {}
    by_title: dict[str, str] = {}
    conflicts, aliases, in_pool = [], [], []
    for src in sources:
        for e in parse_bib(src.read_text(encoding="utf-8", errors="replace")):
            e["src"] = str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)
            doi = norm_doi(e["fields"].get("doi"))
            if e["key"] in by_key:
                old = by_key[e["key"]]
                old_doi = norm_doi(old["fields"].get("doi"))
                if (old_doi and doi and old_doi != doi) or \
                        norm_title(old["fields"].get("title", "")) != norm_title(e["fields"].get("title", "")):
                    conflicts.append(f"klucz {e['key']}: {old['src']} vs {e['src']}")
                continue
            if doi and doi in by_doi:
                # ta sama praca pod innym kluczem: zostaje (artykuly sie do niego odwoluja),
                # ale nowe artykuly maja uzywac pierwszego klucza
                aliases.append(f"{e['key']} = {by_doi[doi]} (DOI {doi})")
            else:
                t = norm_title(e["fields"].get("title", ""))
                if t and t in by_title:
                    if src == CYTOWNIK_BIB:      # praca z cytownika jest juz w puli w lepszej wersji
                        in_pool.append(f"{e['key']} -> {by_title[t]}")
                        continue
                    aliases.append(f"{e['key']} = {by_title[t]} (ten sam tytul)")
                elif t:
                    by_title[t] = e["key"]
            by_key[e["key"]] = e
            if doi:
                by_doi.setdefault(doi, e["key"])
    POOL.parent.mkdir(exist_ok=True)
    out = [
        "% Wspolna pula bibliografii doktoratu. NIE edytowac recznie kluczy istniejacych wpisow,",
        "% bo artykuly sie do nich odwoluja. Odtwarzana: python tools/bib_pool.py build",
        f"% Zrodla: {len(sources)} plikow, {len(by_key)} wpisow.",
        "",
    ]
    for e in sorted(by_key.values(), key=lambda x: x["key"].lower()):
        out += [f"% z: {e['src']}", e["raw"], ""]
    POOL.write_text("\n".join(out), encoding="utf-8")
    print(f"Zapisano {POOL.relative_to(REPO)}: {len(by_key)} wpisow z {len(sources)} plikow.")
    no_doi = [k for k, e in by_key.items() if not e["fields"].get("doi")]
    if no_doi:
        print(f"Bez DOI ({len(no_doi)}): {', '.join(sorted(no_doi))}")
    if in_pool:
        print(f"\nZ cytownika pominieto {len(in_pool)} prac, ktore pula juz ma (klucz cytownika -> klucz w puli):")
        print("  " + ", ".join(in_pool))
    if aliases:
        print(f"\nALIASY ({len(aliases)}) – ta sama praca pod dwoma kluczami; w nowych tekstach uzywaj drugiego:")
        for a in aliases:
            print("  -", a)
    if conflicts:
        print(f"\nKONFLIKTY ({len(conflicts)}) – zostawiony pierwszy wpis, sprawdz recznie:")
        for c in conflicts:
            print("  -", c)
    return 0


def fetch_json(url: str) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "PhD-bib-check/1.0 (mailto:remigiusz.wojewodzki@pwr.edu.pl)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def cmd_check(args) -> int:
    entries = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    bad = 0
    for e in entries:
        doi = norm_doi(e["fields"].get("doi"))
        if not doi:
            print(f"[BRAK DOI] {e['key']}")
            continue
        q = urllib.parse.quote(doi)
        title, where = None, None
        d = fetch_json(f"https://api.crossref.org/works/{q}")
        if d and d.get("message", {}).get("title"):
            title, where = d["message"]["title"][0], "Crossref"
        else:
            d = fetch_json(f"https://api.datacite.org/dois/{q}")
            if d and d.get("data"):
                ts = d["data"]["attributes"].get("titles") or [{}]
                title, where = ts[0].get("title"), "DataCite"
        if not title:
            print(f"[DOI NIE ISTNIEJE] {e['key']}: {doi}")
            bad += 1
        elif norm_title(title)[:40] != norm_title(e["fields"].get("title", ""))[:40]:
            print(f"[INNY TYTUL] {e['key']} ({where}):\n    bib:  {e['fields'].get('title')}\n    {where}: {title}")
            bad += 1
        else:
            print(f"[OK] {e['key']} ({where})")
        time.sleep(0.2)
    print(f"\n{len(entries)} wpisow, problemow: {bad}")
    return 1 if bad else 0


def cmd_extract(args) -> int:
    tex = Path(args.tex).read_text(encoding="utf-8")
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    keys = []
    for m in re.finditer(r"\\(?:no)?cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    pool = {e["key"]: e for e in parse_bib(POOL.read_text(encoding="utf-8"))}
    missing = [k for k in keys if k not in pool]
    found = [pool[k] for k in keys if k in pool]
    Path(args.output).write_text(
        f"% Wyciag z bibliography/pula.bib dla {args.tex} ({len(found)} pozycji).\n\n"
        + "\n\n".join(e["raw"] for e in found) + "\n", encoding="utf-8")
    print(f"Zapisano {args.output}: {len(found)} pozycji.")
    if missing:
        print(f"BRAK W PULI ({len(missing)}): {', '.join(missing)} – dodaj je do references.bib artykulu "
              "i po weryfikacji DOI przebuduj pule (build).")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cy = sub.add_parser("cytownik")
    cy.add_argument("--csv", help="lokalny plik CSV zamiast pobierania z Google Sheets")
    b = sub.add_parser("build")
    b.add_argument("--extra", nargs="*", help="dodatkowe pliki .bib (np. eksport z cytownika)")
    c = sub.add_parser("check")
    c.add_argument("bib")
    x = sub.add_parser("extract")
    x.add_argument("tex")
    x.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    return {"cytownik": cmd_cytownik, "build": cmd_build, "check": cmd_check, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
