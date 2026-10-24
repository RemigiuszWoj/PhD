#!/bin/bash
# Lancuch po E5: testy dymne -> E3 -> E4 -> E5n. Zatrzymuje sie na pierwszym bledzie.
cd /Users/remigiuszwojewodzki/Desktop/PhD
L=articles/qaoa_depth_icaisc/logs
M=articles/qaoa_depth_icaisc/measurements
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. PYTHONUNBUFFERED=1
PY=.venv-iqm/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a $L/chain.log; }
step() {  # step <nazwa> <plik wyniku> <argumenty...>
  name=$1; out=$2; shift 2
  log "START $name"
  if ! $PY -m src.experiments.qaoa_depth_study "$@" > $L/$name.log 2>&1; then
    log "BLAD $name (kod $?), zatrzymuje lancuch; zob. $L/$name.log"; exit 1; fi
  if [ ! -s "$out" ]; then log "BLAD $name: pusty $out, zatrzymuje lancuch"; exit 1; fi
  log "OK $name ($(wc -l < $out) wierszy w $out)"
}
log "czekam na koniec E5"
while pgrep -f "qaoa_depth_study e5 --workers" > /dev/null; do sleep 60; done
log "E5 zakonczone ($(wc -l < $M/e5_ils_ideal.jsonl) wierszy)"
step smoke_e3  $M/e3_smoke.jsonl  e3  --smoke --workers 2
step smoke_e4  $M/e4_smoke.jsonl  e4  --smoke --workers 8
step smoke_e5n $M/e5n_smoke.jsonl e5n --smoke --workers 8
step e3  $M/e3_circuit_cost.jsonl e3  --workers 8
step e4  $M/e4_window_noise.jsonl e4  --workers 8
step e5n $M/e5n_ils_noisy.jsonl   e5n --workers 8
log "LANCUCH ZAKONCZONY"
