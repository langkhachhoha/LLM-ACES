#!/usr/bin/env bash
# One screen: how far every method is, and which tmux sessions are still working.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results}"

count() { ls -1 "$@" 2>/dev/null | wc -l | tr -d ' '; }

for b in odebench odebase; do
  case "$b" in
    odebench) root=data/ode ;;
    odebase)  root=data/odebase ;;
  esac
  # same filter the drivers use: the _snr_ files are the noise ablation
  total="$(find "$root" -name '*.npz' ! -name '*_snr_*' 2>/dev/null | wc -l | tr -d ' ')"
  echo "== $b  ($total systems)"
  for d in "$RESULTS_ROOT/$b"/*/; do
    [[ -d "$d" ]] || continue
    m="$(basename "$d")"
    read -r done_n fail_n < <(python - "$d/systems" <<'PYEOF'
import json, sys, pathlib
ok = bad = 0
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    try:
        s = json.loads(f.read_text()).get("status")
    except Exception:
        s = "unreadable"
    ok += s == "ok"
    bad += s != "ok"
print(ok + bad, bad)
PYEOF
)
    if (( fail_n > 0 )); then
      printf '   %-16s %3s/%-3s systems  (%s FAILED — rerun the lane to retry)\n' "$m" "$done_n" "$total" "$fail_n"
    else
      printf '   %-16s %3s/%-3s systems\n' "$m" "$done_n" "$total"
    fi
  done
  for t in gpt qwen; do
    o="outputs/$b/llm_aces_$t"
    [[ -d "$o" ]] || continue
    printf '   %-16s %3s/%-3s systems  (not scored until scripts/score_all.sh)\n' \
      "llm_aces_$t" "$(count "$o"/*.json)" "$total"
  done
  echo
done

echo "== tmux"
if ! command -v tmux >/dev/null 2>&1 || ! tmux ls >/dev/null 2>&1; then
  echo "   no tmux sessions"
  exit 0
fi
# A session whose pane is back at an interactive shell has finished its job.
declare -i working=0 finished=0
while read -r name cmd; do
  if [[ "$cmd" == "bash" || "$cmd" == "-bash" || "$cmd" == "zsh" ]]; then
    finished+=1
  else
    working+=1
    printf '   working  %-26s %s\n' "$name" "$cmd"
  fi
done < <(tmux list-panes -a -F '#{session_name} #{pane_current_command}' 2>/dev/null)
echo "   $working still working, $finished finished (sessions are kept alive on purpose)"
echo
echo "Detail:  tail -f logs/tmux/<session>.log   |   tail -f $RESULTS_ROOT/<bench>/<method>/run.log"
