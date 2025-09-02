# PhD JSSP Workspace

[![CI](https://github.com/RemigiuszWoj/PhD/actions/workflows/ci.yml/badge.svg)](https://github.com/RemigiuszWoj/PhD/actions/workflows/ci.yml)

Experimental sandbox for Job Shop Scheduling Problem (JSSP) metaheuristics
on Taillard benchmark instances.

> CURRENT STATE (trimmed): Repository został uproszczony – aktywne są tylko dwa algorytmy: `tabu` oraz `sa` (plus tryb `both` uruchamiający oba po kolei). Poprzednie tryby (`demo`, `hill`, `pipeline`, `auto`, `benchmark`) zostały usunięte z kodu i poniższe sekcje oznaczone jako LEGACY nie dotyczą już bieżącej wersji. Zachowano je jedynie referencyjnie.

## 🔥 Szybki start (nowa uproszczona wersja)

Nowy terminal / świeża sesja:

```bash
# 1. (Opcjonalnie) utwórz / użyj środowiska wirtualnego
python3 -m venv .venv311  # jeśli jeszcze nie istnieje
source .venv311/bin/activate  # macOS / Linux (zsh/bash)

# 2. Instalacja zależności (podstawowe + matplotlib do wykresów)
pip install -r requirements.txt
pip install matplotlib  # jeśli nie jest w requirements

# 3. Uruchomienie jednego algorytmu
python src/main.py --algo tabu --instance data/JSPLIB/instances/ta01 --runs 5
python src/main.py --algo sa   --instance data/JSPLIB/instances/ta01 --runs 5

# 4. Uruchomienie obu algorytmów sekwencyjnie
python src/main.py --algo both --instance data/JSPLIB/instances/ta01 --runs 5

# 5. Uruchomienie na wszystkich instancjach (podaj katalog zamiast pliku)
python src/main.py --algo both --instance data/JSPLIB/instances --runs 3
```

Generowane pliki Gantta trafiają do katalogu `charts/`. Na początku każdego uruchomienia katalog jest czyszczony (tylko pliki – katalog pozostaje). Nazwa pliku zawiera algorytm, uzyskane `cmax`, nazwę instancji i timestamp.

### Parametry kluczowe

| Parametr | Opis | Domyślnie |
|----------|------|-----------|
| `--algo` | `tabu`, `sa` lub `both` | `tabu` |
| `--instance` | Ścieżka do pojedynczej instancji lub katalogu z wieloma | `data/JSPLIB/instances/ta01` |
| `--runs` | Ile niezależnych losowych startów (najlepszy zostaje na wykres) | `10` |
| `--seed` | Seed RNG (powtarzalność) | `42` |
| `--tabu-iterations` | Iteracje Tabu | `150` |
| `--tabu-tenure` | Tenure listy tabu | `12` |
| `--tabu-candidate-size` | Rozmiar puli sąsiadów | `60` |
| `--sa-iterations` | Iteracje SA | `800` |
| `--sa-initial-temp` | Temperatura początkowa | `40.0` |
| `--sa-cooling` | Współczynnik chłodzenia | `0.96` |
| `--sa-neighbor-moves` | Liczba ruchów sąsiedztwa / iterację | `2` |
| `--charts-dir` | Katalog na wykresy | `charts` |
| `--log-level` | Poziom logowania | `INFO` |

Przykład z własnym katalogiem wykresów:

```bash
python src/main.py --algo both --instance data/JSPLIB/instances/ta01 \
	--runs 8 --charts-dir out_charts --log-level INFO
```

### Typowe problemy

1. `ModuleNotFoundError: No module named 'src'` – upewnij się, że uruchamiasz z katalogu głównego repo (`pwd` powinno kończyć się na `PhD`).
2. Brak polecenia `python` – użyj `python3` (macOS / Linux) lub pełnej ścieżki do środowiska: `./.venv311/bin/python`.
3. Puste wykresy / brak plików – sprawdź czy instancja istnieje (`ls data/JSPLIB/instances`).
4. Duplikaty logów – jeśli dopisujesz własne loggery, unikaj wielokrotnego konfigurowania `basicConfig`.

---

## (LEGACY) Dokumentacja poprzedniej, rozbudowanej wersji
Poniższe sekcje dotyczą wcześniejszej wersji projektu i mogą nie odzwierciedlać aktualnego kodu.

## Environment setup

Recommended: local virtual environment pinned to project directory.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\\Scripts\\activate       # Windows PowerShell
python -m pip install --upgrade pip
```

Install development dependencies (pytest etc.) if not already present:

```bash
python -m pip install pytest
```

If you prefer an explicit requirements file later, create `requirements-dev.txt` and freeze:

```bash
python -m pip freeze > requirements-dev.txt
```

Deactivate environment when done:

```bash
deactivate
```

## Running tests

Pytest is configured via `pyproject.toml` with options:

```
-ra  # show summary of skipped/xfailed/xpassed/failed
-vv  # verbose test names
--durations=5 --durations-min=0.05  # slow test report
--color=yes
```

Basic run:

```bash
pytest
```

Quiet run:

```bash
pytest -q
```

Module invocation (equivalent):

```bash
python -m pytest
```

Run only tests matching substring:

```bash
pytest -k parser
```

Stop after first failure:

```bash
pytest -x
```

## Custom test summary

A custom `pytest_terminal_summary` hook (see `tests/conftest.py`) prints a compact counts line at the end, plus the slowest tests.

## Parser

Implemented Taillard parser (`src/parser.py`) producing `DataInstance` (`src/models.py`) with automatic normalization of 1-based machine indices.

## Visualization (Gantt)

Optional Gantt chart generation (after running search algorithms) is available via:

```bash
python -m src.main --algo demo --gantt --gantt-path gantt.png
```

To enable plotting install the extra dependency:

```bash
pip install .[viz]
```

If you only want to add matplotlib to an existing environment:

```bash
pip install matplotlib
```

When `--gantt-path` is omitted a window will pop up (interactive backend permitting). The file name extension determines the output format (e.g. `.png`, `.pdf`).

## CLI Modes (LEGACY – usunięte w aktualnym kodzie)

`src/main.py` exposes multiple modes via `--algo`:

- `demo` – heuristic comparison + hill climb + tabu + SA + (optional) Gantt
- `hill` – only hill climbing from SPT permutation
- `tabu` – hill climb warm‑start then tabu search
- `sa` – simulated annealing from SPT
- `pipeline` – multi‑start random → hill → tabu → SA summary
- `auto` – independent multi‑start (all three algorithms)
- `benchmark` – batch over a random sample of Taillard `ta*` files

Common useful flags:

```bash
--neighbor-limit 40 --max-no-improve 20 \
--tabu-iterations 150 --tabu-tenure 12 --tabu-candidate-size 60 \
--sa-iterations 800 --sa-initial-temp 40 --sa-cooling 0.96 --sa-neighbor-moves 2 \
--pipeline-runs 5 --gantt
```

Logging level can be adjusted:

```bash
python -m src.main --algo hill --log-level DEBUG
```

## Auto Mode (LEGACY – usunięty)

Auto mode runs each algorithm (`hill`, `tabu`, `sa`) independently `--runs` times starting every run from a fresh random permutation (algorithms do NOT seed one another). Artifacts saved into `--charts-dir` (default `charts/`):

- `progress_curves_<timestamp>.png` – overlay of best‐so‑far Cmax vs iteration/time for all algorithms (scatter / step look).
- `gantt_<algo>_c<bestC>_<timestamp>.png` – per‑algorithm Gantt for its best run.
- `auto_results_<timestamp>.json` – structured summary:
	* `per_run` per algorithm list with `cmax` and `time` (seconds) per run
	* `best` per algorithm: several permutation encodings (`permutation_pairs`, `permutation_compact`, `job_sequence`)
	* `averages` per algorithm (mean cmax/time)

Example:

```bash
python -m src.main --algo auto --instance data/JSPLIB/instances/ta01 --runs 50 \
	--neighbor-limit 40 --max-no-improve 20 \
	--tabu-iterations 150 --tabu-tenure 12 --tabu-candidate-size 60 \
	--sa-iterations 800 --sa-initial-temp 40 --sa-cooling 0.96 --sa-neighbor-moves 2
```

## Benchmark Mode (LEGACY – usunięty)

Benchmark mode iterates a RANDOM sample of Taillard instances (`ta*`) from `--instances-dir` and for each instance runs each algorithm `--runs` times, storing per‑algorithm artifacts and a combined progress plot.

Key flags:

```bash
--algo benchmark \
--instances-dir data/JSPLIB/instances \
--benchmark-dir research \
--benchmark-sample 8   # how many random instances to include (max)
--runs 60
```

Directory layout produced inside `research/` (example for two instances):

```
research/
	ta01/
		hill/
			results_incremental_hill.json          # overwritten after each run (fault‑tolerant)
			results_hill_<timestamp>.json          # final full summary
			progress_hill_<timestamp>.png          # best run progress
			gantt_hill_c<bestC>.png
		tabu/
			... (analogiczne pliki)
		sa/
			...
		progress_all_<timestamp>.png             # combined (one best curve per algo)
	ta17/
		...
```

### Incremental JSON (fault tolerance)

During benchmark each algorithm directory maintains `results_incremental_<algo>.json` rewritten atomically (temporary file + replace) after every run so that partial progress survives interruptions (e.g. crash / Ctrl+C / power loss). When all runs complete a timestamped `results_<algo>_<timestamp>.json` is also written (immutable record).

Incremental schema (subset):

```jsonc
{
	"instance": "data/JSPLIB/instances/ta01",
	"algorithm": "tabu",
	"runs_completed": 37,
	"planned_runs": 60,
	"per_run": [{"run":1,"cmax":1234,"time":0.231}, ...],
	"best": {"cmax": 1178, "time": 0.812, "permutation": "[[0,0],[1,0],...]"},
	"average": {"avg_cmax": 1210.4, "avg_time": 0.95}
}
```

The final (non‑incremental) JSON adds a `timestamp` and retains full per‑run list.

### Sampling

`--benchmark-sample K` randomly picks at most `K` distinct `ta*` instance files (default 5). This accelerates iterative research cycles while still giving variety. Set it larger (or to the total number of Taillard files) for exhaustive benchmarking.

### Reproducibility Tips

- Provide `--seed` for deterministic permutation sampling; note that stochastic acceptance (SA) and candidate choices (Tabu) depend on this RNG state too.
- Capture the exact CLI plus the contents of each `results_incremental_*.json` for resuming context.

## JSON Permutation Representations

In auto mode (`auto_results_*.json`) three complementary encodings help downstream analysis:

- `permutation_pairs`: list of `[job, operation]` integer pairs
- `permutation_compact`: `J{job}O{op}` comma separated tokens
- `job_sequence`: only the job id sequence (useful for certain decoders)

Benchmark incremental JSON uses a compact flattened pair format inside a single string for brevity.

## Ignoring Research Artifacts

The `.gitignore` now excludes the entire `research/` hierarchy (and PNG charts) to avoid polluting version control with large experimental outputs. If you need to version specific aggregated summaries, copy or move those JSON files into a tracked directory before committing, or add a curated script that extracts only the necessary metrics.

If you want to keep directory structure without data, add empty `.gitkeep` files manually (ignored rule exempts them).

## Developer / Lint Extras

Install dev & viz extras in one step:

```bash
pip install .[dev,viz]
```

Where `dev` includes tooling (pytest, ruff, black, mypy) and `viz` brings `matplotlib`.

## Continuous Integration (GitHub Actions)

This repository uses a single workflow at `.github/workflows/ci.yml` triggered on every
`push` and `pull_request` to `main`. It enforces:

1. Black (format check, 100 line length, preview mode)
2. isort (import ordering check) limited to `src` and `tests`
3. flake8 (style / basic lint) aligned with Black config
4. Ruff (additional lint rules: pyflakes, bugbear, annotations, etc.)
5. mypy (static typing) against `src` + `tests`
6. pytest (unit tests, timing summary enabled)

Artifacts (charts PNG/JSON) from test runs are uploaded if present.

### How to Require Passing CI Before Merge

1. Open GitHub repo: Settings → Branches.
2. Under Branch protection rules click “Add rule”.
3. Set “Branch name pattern” to `main`.
4. Enable:
	- “Require a pull request before merging”.
	- “Require status checks to pass before merging”.
	- In the status checks list select the job name: `CI / build-test` (it appears after the first workflow run on a PR or push).
	- (Optional) “Require branches to be up to date before merging” (forces rebase/merge with latest `main` before final green).
5. (Optional) Enable “Require signed commits” or “Require linear history” for stricter policy.
6. Save rule.

After this, any pull request must pass the CI workflow (all tools/tests green) prior to merge.

### Running Locally Before Opening a PR

You can mimic CI locally:

```bash
pip install .[dev,viz]
black --check .
isort --check-only src tests
flake8 src tests
ruff check .
mypy src tests
pytest
```

If formatting fails, apply:

```bash
black . && isort src tests
```

Then re-run the checks.

## Quick Reference

Run single algorithm once:

```bash
python -m src.main --algo tabu --instance data/JSPLIB/instances/ta01
```

Run auto comparison (30 runs each):

```bash
python -m src.main --algo auto --runs 30 --instance data/JSPLIB/instances/ta01
```

Run benchmark over 10 random Taillard instances (40 runs each):

```bash
python -m src.main --algo benchmark --benchmark-sample 10 --runs 40 \
	--instances-dir data/JSPLIB/instances --benchmark-dir research
```

## Development Convenience Script

Pełny lokalny zestaw kontroli (format, lint, typy, testy) jednym poleceniem:

```bash
./scripts/check_all.sh
```

Flagi wyłączające wybrane kroki (przyspieszenie iteracji):

```bash
./scripts/check_all.sh --no-flake8       # Ruff pokrywa większość
./scripts/check_all.sh --no-mypy         # tylko lint
./scripts/check_all.sh --no-tests        # szybka weryfikacja stylu
./scripts/check_all.sh --no-format --no-mypy --no-tests  # same Ruff/flake8
```

Skrypt automatycznie wykrywa lokalną wirtualną env `.venv311` lub `.venv`.

## License

See `LICENSE` (MIT unless changed).


