"""Download the JSPLIB job shop instances (ta01-80, ft, la, orb, abz, swv, yn) into data/JSPLIB/.

data/ is gitignored, so the instances are fetched once per machine. Afterwards run
scripts/verify_taillard_jssp.py (ta01-ta80 regenerated from Taillard's seeds).

Usage: python scripts/fetch_jsplib.py [--dest data/JSPLIB]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/JSPLIB")
    a = ap.parse_args()
    dest = Path(a.dest)
    (dest / "instances").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE + "instances.json", timeout=60) as r:
        index = json.loads(r.read().decode())
    (dest / "instances.json").write_text(json.dumps(index, indent=1))
    got = skipped = 0
    for e in index:
        target = dest / "instances" / e["name"]
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        with urllib.request.urlopen(BASE + e["path"], timeout=60) as r:
            target.write_bytes(r.read())
        got += 1
    print(f"JSPLIB: {len(index)} instances in index, {got} downloaded, {skipped} already present -> {dest}")
    sys.exit(0 if len(list((dest / 'instances').iterdir())) >= len(index) else 1)


if __name__ == "__main__":
    main()
