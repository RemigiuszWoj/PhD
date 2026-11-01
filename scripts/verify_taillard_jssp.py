"""Check that data/JSPLIB/instances/ta01..ta80 are Taillard's job shop instances.

Each instance is regenerated with Taillard's (1993) generator from the time and machine
seeds stored in src/jobshop/data/taillard_jssp.json and compared with the JSPLIB file
(processing times and routes). Expected output: 80/80 present, 0 mismatches.

Usage: python scripts/verify_taillard_jssp.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.jobshop.instance import JSPLIB_DIR, load, taillard_generate, taillard_table  # noqa: E402


def main() -> None:
    tab = taillard_table()
    missing, bad = [], []
    for name, e in tab.items():
        if not (JSPLIB_DIR / name).exists():
            missing.append(name)
            continue
        inst = load(name)
        g = taillard_generate(e["time_seed"], e["machine_seed"], e["n"], e["m"], 1.0)
        if g.p != inst.p or g.route != inst.route:
            bad.append(name)
    print(f"Taillard JSSP: present {len(tab) - len(missing)}/{len(tab)}, missing: {missing or 'none'}, "
          f"mismatches: {len(bad)} {bad if bad else ''}")
    sys.exit(1 if (missing or bad) else 0)


if __name__ == "__main__":
    main()
