#!/usr/bin/env bash
# Launch a command in a detached tmux session that
#
#   1. inherits the environment you are in right now (the already-activated
#      conda env -- no `conda activate` or `conda run` inside the session), and
#   2. STAYS ALIVE when the command finishes: the pane drops into an interactive
#      shell instead of exiting, so the session and its scrollback remain.
#
# Usage (from inside the activated env, at the repo root):
#
#   bash scripts/tmux_run.sh sindy bash run_baseline.sh sindy odebench
#   bash scripts/tmux_run.sh pysr  bash run_baseline.sh pysr odebench --pysr_procs 8
#
# Everything is also tee'd to logs/tmux/<session>.log.
set -uo pipefail

SESSION="${1:-}"
shift || true
if [[ -z "$SESSION" || $# -eq 0 ]]; then
  echo "usage: bash scripts/tmux_run.sh <session-name> <command...>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed (apt-get install tmux / yum install tmux)" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists." >&2
  echo "  attach : tmux attach -t $SESSION" >&2
  echo "  replace: tmux kill-session -t $SESSION && <re-run this command>" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "warning: no conda env is active; the session will inherit the shell as-is." >&2
fi

LOG_DIR="$ROOT/logs/tmux"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$SESSION.log"

# Preserve argv exactly: printf %q quotes each argument so that when bash
# re-parses the line inside the pane it rebuilds the same argument vector.
# (Plain "$*" would flatten `bash -c 'a && b'` into three separate words.)
USER_CMD=$(printf '%q ' "$@")
DISPLAY_CMD="$*"

# The command the pane runs. Note the trailing `exec bash`: that is what keeps
# the session alive after the work is done. PIPESTATUS preserves the real exit
# code through the `tee`.
PANE_SCRIPT=$(cat <<EOF
echo "[$SESSION] started \$(date '+%F %T')  env=\${CONDA_DEFAULT_ENV:-none}"
echo "[$SESSION] \$ $DISPLAY_CMD"
echo
{ $USER_CMD ; } 2>&1 | tee -a '$LOG_FILE'
status=\${PIPESTATUS[0]}
echo
echo "[$SESSION] finished \$(date '+%F %T') with exit \$status"
echo "[$SESSION] log: $LOG_FILE"
echo "[$SESSION] session kept alive - detach with Ctrl-b d, close with 'exit'"
exec bash
EOF
)

# Pass the current environment explicitly rather than relying on the tmux
# server's environment: if a tmux server was already running before you
# activated the env, new sessions would otherwise inherit the *server's* PATH
# and silently use the wrong Python.
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "env PATH=$(printf %q "$PATH") \
       CONDA_PREFIX=$(printf %q "${CONDA_PREFIX:-}") \
       CONDA_DEFAULT_ENV=$(printf %q "${CONDA_DEFAULT_ENV:-}") \
       PYTHONPATH=$(printf %q "${PYTHONPATH:-}") \
       bash --noprofile --norc -c $(printf %q "$PANE_SCRIPT")"

# Belt and braces: if the pane ever does exit, keep it visible instead of
# tearing the window down.
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null 2>&1 || true

echo "[launched] tmux session '$SESSION'"
echo "  command : $DISPLAY_CMD"
echo "  log     : $LOG_FILE"
echo "  watch   : tmux attach -t $SESSION      (detach: Ctrl-b then d)"
