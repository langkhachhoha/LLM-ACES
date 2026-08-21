#!/usr/bin/env bash
# Run LLM-ACES (the paper's own method) over a whole benchmark and score it with
# the shared evaluator, so its numbers land in results/ next to the baselines.
#
# Paper defaults (Appendix B.3): 10 iterations, up to 3 operator priors per
# round, PySR with 20 iterations / 15 populations, temperature 0.8, a pool of 10
# candidate initial conditions per acquisition step.
#
# Usage:
#   bash scripts/run_llm_aces.sh <benchmark> <model_id> <tag> [extra args...]
#
#   bash scripts/run_llm_aces.sh odebench openai/gpt-4o-mini-2024-07-18 gpt
#   bash scripts/run_llm_aces.sh odebase  qwen/qwen3-30b-a3b-instruct-2507 qwen
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BENCH="${1:?usage: run_llm_aces.sh <odebench|odebase> <model_id> <tag> [extra args]}"
MODEL="${2:?missing model id}"
TAG="${3:?missing tag, e.g. gpt or qwen}"
shift 3
EXTRA=("$@")

case "$BENCH" in
  odebench) DATA_ROOT="$ROOT/data/ode" ;;
  odebase)  DATA_ROOT="$ROOT/data/odebase" ;;
  *) echo "benchmark must be odebench or odebase"; exit 1 ;;
esac

METHOD="llm_aces_${TAG}"
LOG_ROOT="$ROOT/logs/$BENCH/$METHOD"
OUT_DIR="$ROOT/outputs/$BENCH/$METHOD"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
mkdir -p "$LOG_ROOT" "$OUT_DIR" "$RESULTS_ROOT"

# Paper defaults
N_ITERATIONS="${N_ITERATIONS:-10}"
MAX_CONCEPTS="${MAX_CONCEPTS:-3}"
N_VIRTUAL="${N_VIRTUAL:-10}"
BO_INIT_POINTS="${BO_INIT_POINTS:-3}"
PYSR_NITER="${PYSR_NITER:-20}"
PYSR_POPS="${PYSR_POPS:-15}"
CONCEPT_TEMPERATURE="${CONCEPT_TEMPERATURE:-0.8}"

set -a; [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"; set +a

# portable (works on bash 3.2 too): no mapfile
# ACES_SYSTEMS="a b c" restricts the run to those NPZ stems (used for ablations).
DATASETS=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  if [[ "$BENCH" == "odebase" && "$line" != *odebase_vars2_prog* && "$line" != *odebase_vars3_prog* ]]; then
    continue
  fi
  if [[ -n "${ACES_SYSTEMS:-}" ]]; then
    stem="$(basename "$line" .npz)"
    keep=0
    for want in ${ACES_SYSTEMS}; do [[ "$stem" == "$want" ]] && keep=1 && break; done
    [[ $keep -eq 0 ]] && continue
  fi
  DATASETS+=("$line")
done < <(find "$DATA_ROOT" -name "*.npz" ! -name "*_snr_*" | sort)
echo "LLM-ACES ($MODEL) on $BENCH: ${#DATASETS[@]} systems"
echo "  iterations=$N_ITERATIONS  concepts/round=$MAX_CONCEPTS  n_virtual=$N_VIRTUAL  pysr=$PYSR_NITER/$PYSR_POPS"

for data_path in "${DATASETS[@]}"; do
  system="$(basename "$data_path" .npz)"
  if [[ -f "$OUT_DIR/${system}.json" ]]; then
    echo "[SKIP] $system (already done)"
    continue
  fi
  echo "=============================================================="
  echo "  $system"
  echo "=============================================================="
  mkdir -p "$LOG_ROOT/$system"
  python llm-aces/active_llm_aces.py \
    --data_path              "$data_path" \
    --log_path               "$LOG_ROOT/$system" \
    --output_dir             "$OUT_DIR" \
    --n_iterations           "$N_ITERATIONS" \
    --max_concepts_per_round "$MAX_CONCEPTS" \
    --n_virtual              "$N_VIRTUAL" \
    --bo_init_points         "$BO_INIT_POINTS" \
    --concept_temperature    "$CONCEPT_TEMPERATURE" \
    --pysr_niterations       "$PYSR_NITER" \
    --pysr_populations       "$PYSR_POPS" \
    --use_api                true \
    --api_provider           openrouter \
    --api_model              "$MODEL" \
    "${EXTRA[@]}" \
    2>&1 | tee "$LOG_ROOT/$system/stdout.log"
  echo "[DONE] $system"
done

echo "Scoring LLM-ACES outputs with the shared evaluator..."
python -m baselines.eval_llm_aces \
  --benchmark "$BENCH" \
  --outputs_dir "$OUT_DIR" \
  --logs_dir "$LOG_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --method_name "$METHOD" \
  --model "$MODEL"

echo "All done -> $RESULTS_ROOT/$BENCH/$METHOD"
