#!/usr/bin/env bash
# Fetch the upstream repositories the paper's baselines are taken from.
#
#   LLM-ODE   -> https://github.com/gryaklab/llm-ode           (Appendix B.2)
#   APPS-ODE  -> https://github.com/jiangnanhugo/APPS-ODE      (Appendix B.2)
#   E2E       -> https://github.com/facebookresearch/symbolicregression + model1.pt
#   MDBench   -> https://github.com/gryaklab/mdbench           (reference implementations)
#
# Usage:
#   bash scripts/setup_third_party.sh            # all of them
#   bash scripts/setup_third_party.sh llm-ode    # just one
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

clone() {  # clone <url> <dir>
  local url="$1" dir="$2"
  if [[ -d "$TP/$dir/.git" ]]; then
    echo "[skip] $dir already cloned"
  else
    echo "[clone] $url -> third_party/$dir"
    git clone --depth 1 "$url" "$TP/$dir"
  fi
}

setup_llm_ode() {
  clone https://github.com/gryaklab/llm-ode.git llm-ode
  # Upstream pins requires-python == 3.13.5 purely because of two calls to
  # str.replace(..., count=1); the keyword argument only exists from CPython
  # 3.13. On an older interpreter every candidate equation raises and the island
  # initialiser spins forever. Making the argument positional is behaviour-
  # preserving on every Python version and lets LLM-ODE share the single env.
  local patched=0
  for f in "$TP/llm-ode/llmode/equation.py" "$TP/llm-ode/llmode/system.py"; do
    if grep -q "count=1" "$f" 2>/dev/null; then
      perl -pi -e "s/\\.replace\\('C', new_coeff, count=1\\)/.replace('C', new_coeff, 1)/" "$f"
      patched=1
    fi
  done
  if [[ $patched -eq 1 ]]; then
    echo "[patch] llm-ode: str.replace(count=1) -> positional (drops the Python 3.13 requirement)"
  else
    echo "[skip] llm-ode compat patch already applied"
  fi
  echo "[ok] LLM-ODE ready. baselines/run_llm_ode.py imports llmode.* from third_party/llm-ode"
  echo "     (the upstream OpenAI/vLLM transport is replaced by our OpenRouter client)."
}

setup_apps_ode() {
  clone https://github.com/jiangnanhugo/APPS-ODE.git APPS-ODE
  cat <<'EOF'
[ok] APPS-ODE ready. Its `scibench` and `grammar` packages are pure Python and
     are put on PYTHONPATH by baselines/run_apps_ode.py -- nothing to install,
     and no risk of shadowing this repo's own scripts/scibench.
     Runtime deps (numba, cython, click, commentjson, PyYAML, pathos, psutil)
     are part of the single llm-aces environment; see SETUP_SERVER.md step 3.
EOF
}

setup_e2e() {
  clone https://github.com/facebookresearch/symbolicregression.git symbolicregression
  local ckpt="$TP/symbolicregression/model.pt"
  if [[ -f "$ckpt" ]]; then
    echo "[skip] E2E checkpoint already present"
  else
    echo "[download] E2E pretrained checkpoint (~700MB)"
    curl -L --fail -o "$ckpt" https://dl.fbaipublicfiles.com/symbolicregression/model1.pt
  fi
  echo "[ok] E2E ready at third_party/symbolicregression"
}

setup_mdbench() {
  clone https://github.com/gryaklab/mdbench.git mdbench
  echo "[ok] MDBench cloned for reference (our SINDy/PySR/Operon/ODEFormer drivers mirror it)."
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then
  targets=(llm-ode apps-ode e2e mdbench)
fi
for t in "${targets[@]}"; do
  case "$t" in
    llm-ode)  setup_llm_ode ;;
    apps-ode) setup_apps_ode ;;
    e2e)      setup_e2e ;;
    mdbench)  setup_mdbench ;;
    *) echo "unknown target: $t (llm-ode|apps-ode|e2e|mdbench)"; exit 1 ;;
  esac
done
echo "Done."
