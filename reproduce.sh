#!/usr/bin/env bash
# Reproduce every number and figure of the preprint from the released trajectories.
#
#   bash reproduce.sh [DATA_DIR]      # DATA_DIR defaults to data/runs
#
# Writes results/*.txt|json and figures/*.pdf|png, then compares the printed tables
# with expected/. The run order matters: de-duplication keeps the first copy.
set -euo pipefail
cd "$(dirname "$0")"

DATA="${1:-data/runs}"
RUNS=("$DATA/20260806T212541Z-768873" "$DATA/20260820T220044Z-9e47ab")
for r in "${RUNS[@]}"; do
  [[ -f "$r/config.yaml" ]] || { echo "missing run directory: $r (see README, Data)" >&2; exit 1; }
done

mkdir -p results figures
python scripts/analyze_corpus.py "${RUNS[@]}" --json results/analyze_corpus.json \
  | tee results/analyze_corpus.txt
python scripts/paper_numbers.py "${RUNS[@]}" --json results/paper_numbers.json \
  | tee results/paper_numbers.txt
python scripts/make_figures.py "${RUNS[@]}" --results results --out figures

status=0
for name in analyze_corpus paper_numbers; do
  if diff <(grep -v '^wrote ' "results/$name.txt") "expected/$name.txt" > "results/$name.diff"; then
    echo "$name: matches expected/$name.txt"
    rm "results/$name.diff"
  else
    echo "$name: differs from expected/$name.txt, see results/$name.diff" >&2
    status=1
  fi
done
exit $status
