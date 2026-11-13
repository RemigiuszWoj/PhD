#!/bin/zsh
# Push the k-bonacci article to its Overleaf project (git integration).
# Copies only what the article needs to compile: main.tex (EN), wersja_pl.tex (PL)
# and figures/*.pdf. On Overleaf, Menu -> Main document switches between the two.
# Notes, results and drafts stay in this repo.
#
# Usage: scripts/overleaf_sync.sh ["commit message"]
# Overleaf clone: ~/Desktop/overleaf/kbonacci (token in the macOS keychain).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/articles/kbonacci"
OL="${OVERLEAF_KBONACCI:-$HOME/Desktop/overleaf/kbonacci}"
MSG="${1:-Sync from PhD-kbonacci ($(git -C "$SRC" rev-parse --short HEAD))}"

git -C "$OL" pull --ff-only          # never overwrite edits made in the browser
mkdir -p "$OL/figures"
cp "$SRC/main.tex" "$SRC/wersja_pl.tex" "$OL/"
cp "$SRC"/figures/*.pdf "$OL/figures/"

git -C "$OL" add main.tex wersja_pl.tex figures
if git -C "$OL" diff --cached --quiet; then
    echo "Overleaf already up to date."
    exit 0
fi
git -C "$OL" diff --cached --stat
git -C "$OL" commit -q -m "$MSG"
git -C "$OL" push -q
echo "Pushed to Overleaf."
