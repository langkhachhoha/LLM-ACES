#!/usr/bin/env bash
# One entry point for every method in the reproduction.
#
#   bash run_baseline.sh <method> <benchmark> [extra args passed to the driver]
#
# Everything runs in the single `llm-aces` conda environment; activate it once
# before calling this script. Methods:
#
#   sindy      PySINDy / STLSQ                     (paper Table 10)
#   pysr       PySR, expanded operators            (paper Table 10)
#   operon     pyoperon, MDL Pareto selection      (paper Table 10)
#   odeformer  pretrained ODEFormer checkpoint     (paper Table 10)
#   e2e        facebookresearch/symbolicregression (paper Table 10)
#   llm_only   NewtonBench-style iterative LLM     (125 calls / 1000 eqs)
#   llm_ode    gryaklab/llm-ode evolutionary core  (125 calls / 1000 eqs)
#   apps_ode   jiangnanhugo/APPS-ODE grammar-RL    (50 epochs, BFGS)
#   qbc        PySR + Query-by-Committee ICs       (10 acquisition rounds)
#   bo         PySR + GP/EI over ICs               (10 acquisition rounds)
#
# LLM-ACES itself is run with scripts/run_llm_aces.sh.
#
# Environment overrides:
#   RESULTS_ROOT  (default: results)
#   MODEL         LLM id for llm_only / llm_ode (default: openai/gpt-4o-mini-2024-07-18)
#   METHOD_NAME   result sub-folder name (default: the method id)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

METHOD="${1:?usage: run_baseline.sh <method> <odebench|odebase> [extra args]}"
BENCH="${2:?missing benchmark}"
shift 2
EXTRA=("$@")

RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"
MODEL="${MODEL:-openai/gpt-4o-mini-2024-07-18}"
NAME_ARGS=()
[[ -n "${METHOD_NAME:-}" ]] && NAME_ARGS=(--method_name "$METHOD_NAME")

set -a; [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"; set +a
mkdir -p "$RESULTS_ROOT"

COMMON=(--benchmark "$BENCH" --results_root "$RESULTS_ROOT" ${NAME_ARGS[@]+"${NAME_ARGS[@]}"})

case "$METHOD" in
  sindy)     python -m baselines.run_sindy     "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  pysr)      python -m baselines.run_pysr      "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  operon)    python -m baselines.run_operon    "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  odeformer) python -m baselines.run_odeformer "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  e2e)       python -m baselines.run_e2e       "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  qbc)       python -m baselines.run_qbc       "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  bo)        python -m baselines.run_bo        "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  llm_only)  python -m baselines.run_llm_only  "${COMMON[@]}" --model "$MODEL" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  llm_ode)   python -m baselines.run_llm_ode   "${COMMON[@]}" --model "$MODEL" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  apps_ode)  python -m baselines.run_apps_ode  "${COMMON[@]}" ${EXTRA[@]+"${EXTRA[@]}"} ;;
  *) echo "unknown method: $METHOD"; sed -n '5,25p' "$0"; exit 1 ;;
esac
