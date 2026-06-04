#!/usr/bin/env bash
# Sekwencyjne uruchomienie 2b (n=100) → 2c (n=200) classical baseline.
# Każda faza modyfikuje aktywny blok instance_files w config.yaml,
# a następnie wywołuje main.py. Komentowane bloki configu nie są ruszane
# (regex zastępuje tylko pierwsze wystąpienie).

set -e
cd /Users/remigiuszwojewodzki/Desktop/PhD

set_instance_files() {
  .venv311/bin/python3 - "$@" <<'PY'
import re, sys
files = sys.argv[1:]
new_block = "  instance_files:\n" + "\n".join(f'    - "{f}"' for f in files) + "\n"
text = open("config.yaml").read()
text2 = re.sub(r'  instance_files:\n(    - "[^"]+"\n)+', new_block, text, count=1)
if text2 == text:
    print("WARNING: no instance_files block matched", file=sys.stderr); sys.exit(1)
open("config.yaml", "w").write(text2)
print(f"Active phase set to: {files}")
PY
}

echo "================================================================"
echo "STAGE 2b: n=100 (tai100_5/10/20)  $(date '+%H:%M:%S')"
echo "================================================================"
set_instance_files data/tai100_5.txt data/tai100_10.txt data/tai100_20.txt
.venv311/bin/python3 main.py
echo "STAGE 2b DONE  $(date '+%H:%M:%S')"

echo "================================================================"
echo "STAGE 2c: n=200 (tai200_10/20)  $(date '+%H:%M:%S')"
echo "================================================================"
set_instance_files data/tai200_10.txt data/tai200_20.txt
.venv311/bin/python3 main.py
echo "STAGE 2c DONE  $(date '+%H:%M:%S')"

echo "================================================================"
echo "ALL PHASES COMPLETED  $(date '+%H:%M:%S')"
echo "================================================================"
