#!/usr/bin/env bash
# Launch the whole reproduction as a set of sharded tmux sessions, sized to the
# machine. This is the one command RUN_TMUX.md tells you to run.
#
#   bash scripts/launch_all.sh                 # size to `nproc`
#   bash scripts/launch_all.sh --cores 64      # force a budget
#   bash scripts/launch_all.sh --dry-run       # print what it would start
#   bash scripts/launch_all.sh --only apps,bo  # a subset of the lanes
#   bash scripts/launch_all.sh --restart       # resume: kill old sessions, start again
#
# Lanes: apps  bo  qbc  aces  pysr  quick  llm
#
# Everything is resumable: a finished system is skipped on the next run, so a
# lane that dies (or that you kill) can simply be launched again.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPT_MODEL="${GPT_MODEL:-openai/gpt-4o-mini-2024-07-18}"
QWEN_MODEL="${QWEN_MODEL:-qwen/qwen3-30b-a3b-instruct-2507}"
LLMONLY_MODEL="${LLMONLY_MODEL:-$GPT_MODEL}"

detect_cores() {
  if command -v nproc >/dev/null 2>&1; then nproc
  elif command -v sysctl >/dev/null 2>&1; then sysctl -n hw.ncpu
  else echo 8
  fi
}

CORES=""
DRY=0
RESTART=0
ONLY="apps,bo,qbc,aces,pysr,quick,llm"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cores)   CORES="$2"; shift 2 ;;
    --only)    ONLY="$2";  shift 2 ;;
    --dry-run) DRY=1;      shift ;;
    --restart) RESTART=1;  shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -z "$CORES" ]] && CORES="$(detect_cores)"
if [[ ! "$CORES" =~ ^[0-9]+$ ]] || (( CORES < 1 )); then
  echo "--cores wants a positive integer, got '$CORES'" >&2; exit 2
fi

wants() { [[ ",$ONLY," == *",$1,"* ]]; }

# The plan is tuned for 64 cores and scaled linearly from there. Shards scale
# linearly, Julia threads do not (8 threads = ~4x on these fits), so the budget
# buys shards first and gives each one 1-2 threads.
scale() {  # scale <value-at-64-cores> -> at least 1
  awk -v v="$1" -v c="$CORES" 'BEGIN { n = int(v * c / 64 + 0.5); print (n < 1 ? 1 : n) }'
}
# Every one of these can be overridden from the environment, which is how you
# give a single lane the whole machine (see RUN_TMUX.md, "four groups").
APPS_SHARDS="${APPS_SHARDS:-$(scale 8)}"      # per benchmark, 1 core each
ACT_SHARDS="${ACT_SHARDS:-$(scale 4)}"        # BO / QBC, per benchmark
ACT_THREADS="${ACT_THREADS:-2}"
ACES_SHARDS="${ACES_SHARDS:-$(scale 3)}"      # per LLM-ACES run (4 runs)
ACES_THREADS="${ACES_THREADS:-1}"
PYSR_SHARDS="${PYSR_SHARDS:-$(scale 2)}"      # passive PySR, per benchmark
PYSR_THREADS="${PYSR_THREADS:-2}"
QUICK_SHARDS="${QUICK_SHARDS:-1}"             # ODEFormer / E2E, per benchmark
QUICK_THREADS="${QUICK_THREADS:-2}"           # torch threads per shard

declare -i BUDGET=0
wants apps && BUDGET+=$(( APPS_SHARDS * 2 ))
wants bo   && BUDGET+=$(( ACT_SHARDS * ACT_THREADS * 2 ))
wants qbc  && BUDGET+=$(( ACT_SHARDS * ACT_THREADS * 2 ))
wants aces && BUDGET+=$(( ACES_SHARDS * ACES_THREADS * 4 ))
wants pysr && BUDGET+=$(( PYSR_SHARDS * PYSR_THREADS * 2 ))
wants quick && BUDGET+=$(( QUICK_SHARDS * QUICK_THREADS * 4 ))

echo "LLM-ACES full run — $CORES cores"
echo "  lanes      : $ONLY"
wants apps  && echo "  apps       : $APPS_SHARDS shards x 2 benchmarks x 1 core"
wants bo    && echo "  bo         : $ACT_SHARDS shards x 2 benchmarks x $ACT_THREADS threads"
wants qbc   && echo "  qbc        : $ACT_SHARDS shards x 2 benchmarks x $ACT_THREADS threads"
wants aces  && echo "  aces       : $ACES_SHARDS shards x 4 runs x $ACES_THREADS threads (also blocked on the API)"
wants pysr  && echo "  pysr       : $PYSR_SHARDS shards x 2 benchmarks x $PYSR_THREADS threads"
wants quick && echo "  quick      : SINDy + ODEFormer/E2E on $QUICK_SHARDS shards x 2 benchmarks x $QUICK_THREADS torch threads"
wants llm   && echo "  llm        : LLM-only, LLM-ODE — one session each"
echo "  nominal load: $BUDGET cores of $CORES; the LLM lanes idle on the network"
echo

run() {  # run <session> <command...>
  if (( DRY )); then
    printf '  tmux_run %-26s %s\n' "$1" "${*:2}"
    return
  fi
  # tmux_run refuses to reuse a live session name. --restart clears it first:
  # every driver skips systems that already have a result, so this resumes.
  (( RESTART )) && tmux kill-session -t "$1" 2>/dev/null
  bash scripts/tmux_run.sh "$@" || true
}

if wants apps; then
  echo "-- APPS-ODE (the long pole: ~70 min/system, 122 systems)"
  for b in odebench odebase; do
    for ((i = 0; i < APPS_SHARDS; i++)); do
      run "appsode_${b}_$i" bash run_baseline.sh apps_ode "$b" --n_cores 1 --shard "$i/$APPS_SHARDS"
    done
  done
fi

for m in bo qbc; do
  wants "$m" || continue
  echo "-- $m (one full PySR search per dimension per acquisition round, 10 rounds)"
  for b in odebench odebase; do
    for ((i = 0; i < ACT_SHARDS; i++)); do
      run "${m}_${b}_$i" bash run_baseline.sh "$m" "$b" --pysr_procs "$ACT_THREADS" --shard "$i/$ACT_SHARDS"
    done
  done
done

if wants aces; then
  echo "-- LLM-ACES (the paper's method, 4 runs)"
  for spec in "gpt $GPT_MODEL" "qwen $QWEN_MODEL"; do
    tag="${spec%% *}"; model="${spec#* }"
    for b in odebench odebase; do
      for ((i = 0; i < ACES_SHARDS; i++)); do
        run "aces_${tag}_${b}_$i" \
          env "PYSR_PROCS=$ACES_THREADS" "ACES_SHARD=$i/$ACES_SHARDS" SKIP_EVAL=1 \
          bash scripts/run_llm_aces.sh "$b" "$model" "$tag"
      done
    done
  done
fi

if wants pysr; then
  echo "-- PySR, passive (paper Table 10 budget: 100 iterations, 20 populations, 1000 cycles)"
  for b in odebench odebase; do
    for ((i = 0; i < PYSR_SHARDS; i++)); do
      run "pysr_${b}_$i" bash run_baseline.sh pysr "$b" \
        --pysr_procs "$PYSR_THREADS" --shard "$i/$PYSR_SHARDS"
    done
  done
fi

if wants quick; then
  echo "-- quick lane (SINDy ~4 min, then the two transformer baselines)"
  run sindy bash -c "
    bash run_baseline.sh sindy odebench && bash run_baseline.sh sindy odebase"
  for m in odeformer e2e; do
    for b in odebench odebase; do
      if (( QUICK_SHARDS <= 1 )); then
        run "${m}_${b}" env "OMP_NUM_THREADS=$QUICK_THREADS" "MKL_NUM_THREADS=$QUICK_THREADS" \
          bash run_baseline.sh "$m" "$b"
      else
        for ((i = 0; i < QUICK_SHARDS; i++)); do
          # torch would otherwise grab every core in every shard
          run "${m}_${b}_$i" env "OMP_NUM_THREADS=$QUICK_THREADS" "MKL_NUM_THREADS=$QUICK_THREADS" \
            bash run_baseline.sh "$m" "$b" --shard "$i/$QUICK_SHARDS"
        done
      fi
    done
  done
fi

if wants llm; then
  echo "-- LLM-only / LLM-ODE (API-bound, negligible CPU)"
  run llmonly bash -c "
    MODEL=$LLMONLY_MODEL bash run_baseline.sh llm_only odebench &&
    MODEL=$LLMONLY_MODEL bash run_baseline.sh llm_only odebase"
  run llmode bash -c "
    bash run_baseline.sh llm_ode odebench && bash run_baseline.sh llm_ode odebase"
fi

echo
if (( DRY )); then
  echo "dry run — nothing was started."
else
  echo "Started. Watch:  tmux ls  |  tail -f logs/tmux/<session>.log"
fi
cat <<EOF

When every session prints 'finished ... with exit 0', score once:

  bash scripts/score_all.sh
EOF
