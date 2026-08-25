#!/usr/bin/env bash
# Score everything once the runs are done: the LLM-ACES shards (which were
# launched with SKIP_EVAL=1), then symbolic accuracy, then the tables.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPT_MODEL="${GPT_MODEL:-openai/gpt-4o-mini-2024-07-18}"
QWEN_MODEL="${QWEN_MODEL:-qwen/qwen3-30b-a3b-instruct-2507}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"

for spec in "gpt $GPT_MODEL" "qwen $QWEN_MODEL"; do
  tag="${spec%% *}"; model="${spec#* }"
  for b in odebench odebase; do
    out="outputs/$b/llm_aces_$tag"
    if [[ ! -d "$out" ]] || [[ -z "$(ls -A "$out" 2>/dev/null)" ]]; then
      echo "[skip] $b/llm_aces_$tag — no outputs"
      continue
    fi
    echo "== scoring $b/llm_aces_$tag ($(ls "$out"/*.json 2>/dev/null | wc -l | tr -d ' ') systems)"
    python -m baselines.eval_llm_aces \
      --benchmark "$b" --outputs_dir "$out" --logs_dir "logs/$b/llm_aces_$tag" \
      --results_root "$RESULTS_ROOT" --method_name "llm_aces_$tag" --model "$model"
  done
done

declare -i SCORED=0
for b in odebench odebase; do
  if [[ ! -d "$RESULTS_ROOT/$b" ]]; then
    echo "[skip] $b — $RESULTS_ROOT/$b does not exist yet (nothing has finished)"
    continue
  fi
  echo "== symbolic accuracy: $b"
  python -m baselines.symbolic_accuracy --benchmark "$b" --quiet
  SCORED+=1
done

if (( SCORED == 0 )); then
  echo "No results to aggregate."
  exit 1
fi

[[ -d "$RESULTS_ROOT/odebench" ]] && {
  python -m baselines.aggregate_results --benchmark odebench --csv "$RESULTS_ROOT/table2.csv"
  python -m baselines.aggregate_results --benchmark odebench --metric traj
}
[[ -d "$RESULTS_ROOT/odebase" ]] && {
  python -m baselines.aggregate_results --benchmark odebase --csv "$RESULTS_ROOT/table3.csv"
  python -m baselines.aggregate_results --benchmark odebase --metric traj
}

echo
echo "Tables: $RESULTS_ROOT/table2.csv (ODEBench), $RESULTS_ROOT/table3.csv (ODEBase)"
