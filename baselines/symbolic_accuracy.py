"""LLM-judged symbolic accuracy (paper, Appendix A.2).

    "Following LLMSRBench, we adopt an LLM-based evaluation methodology to
     assess symbolic equivalence between discovered and ground-truth ODE
     equations. ... First, all equations are pre-processed to produce
     constant-free structural skeletons.  For ground-truth equations, symbolic
     placeholder parameters are removed, and for predicted equations, fitted
     numerical constants are stripped.  This pre-processing is performed
     analytically via SymPy, where the expression tree is traversed recursively,
     and all free scalar terms are replaced with unity while structural elements
     like variables, operators, and integer/rational exponents are preserved.
     Second, the resulting skeletons are passed to GPT-4o-mini, which is
     prompted to assess whether the two expressions share the same mathematical
     structure, variables, and operations.  For ODE systems with multiple state
     variables, equivalence is assessed independently per dimension, and the
     per-problem score is computed as the fraction of dimensions correctly
     recovered (partial credit).  The final symbolic accuracy across a benchmark
     is the mean per-problem score."

Usage::

    python -m baselines.symbolic_accuracy --results_root results \\
        --benchmark odebench --methods sindy pysr llm_aces_gpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines import common, oracle as oracle_lib
from baselines.llm_client import GPT_MODEL, LLMClient

JUDGE_PROMPT = """You are judging whether two mathematical expressions have the SAME STRUCTURE.

Both expressions are constant-free "skeletons": every fitted numerical coefficient has
already been replaced by 1, so differences in numeric values must be IGNORED.

Ground-truth skeleton:
  {gt}

Predicted skeleton:
  {pred}

Decide whether the two expressions share the same mathematical structure: the same
variables, the same operations, and the same functional form (after algebraic
simplification such as expanding, factoring or reordering terms).

Answer with exactly one word: EQUIVALENT or DIFFERENT."""


# ---------------------------------------------------------------------------
# Skeletonisation
# ---------------------------------------------------------------------------
def skeletonize(expr):
    """Replace every free scalar with 1, keeping integer/rational exponents."""
    import sympy as sp

    if expr is None:
        return None

    def walk(e):
        if e.is_Number:
            return sp.Integer(1)
        if e.is_Symbol:
            return e
        if e.is_Pow:
            base, exponent = e.args
            new_exp = exponent if exponent.is_Rational else walk(exponent)
            return sp.Pow(walk(base), new_exp, evaluate=False)
        if e.args:
            return e.func(*[walk(a) for a in e.args], evaluate=False)
        return e

    try:
        out = walk(expr)
        return sp.simplify(sp.sympify(out))
    except Exception:
        try:
            return walk(expr)
        except Exception:
            return expr


def skeleton_str(eq_str: str, dim: int) -> str | None:
    expr = common.sympify_equation(eq_str, dim)
    if expr is None:
        return None
    sk = skeletonize(expr)
    return None if sk is None else str(sk)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------
class Judge:
    def __init__(self, client: LLMClient, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.cache: dict[str, bool] = {}
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def equivalent(self, gt: str, pred: str) -> bool:
        if gt is None or pred is None:
            return False
        key = f"{gt} ||| {pred}"
        if key in self.cache:
            return bool(self.cache[key])
        # exact structural match short-circuits the API call
        if gt.replace(" ", "") == pred.replace(" ", ""):
            self.cache[key] = True
            self._flush()
            return True
        try:
            out = self.client.chat(JUDGE_PROMPT.format(gt=gt, pred=pred),
                                   n=1, temperature=0.0, max_tokens=8)[0]
        except Exception:
            return False
        verdict = "equivalent" in out.strip().lower()
        self.cache[key] = verdict
        self._flush()
        return verdict

    def _flush(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)


def score_method(result_dir: Path, judge: Judge, verbose: bool = True) -> dict:
    per_system: dict[str, dict] = {}
    for path in sorted((result_dir / "systems").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        system = payload["system"]
        orc = oracle_lib.get_oracle(system)
        if orc is None:
            continue
        dim = orc.dim
        eqs = payload.get("equations") or []
        gt_sk = [skeleton_str(e, dim) for e in orc.gt_equations]
        pred_sk = [skeleton_str(e, dim) for e in eqs] + [None] * max(0, dim - len(eqs))

        correct = [bool(judge.equivalent(g, p)) for g, p in zip(gt_sk, pred_sk[:dim])]
        score = float(sum(correct)) / dim if dim else 0.0
        per_system[system] = {
            "score": score, "per_dim": correct,
            "gt_skeletons": gt_sk, "pred_skeletons": pred_sk[:dim],
        }
        if verbose:
            print(f"  {system}: {score:.2f}  {correct}")

    scores = [v["score"] for v in per_system.values()]
    summary = {
        "n_systems": len(scores),
        "symbolic_accuracy": float(sum(scores) / len(scores)) if scores else 0.0,
        "per_system": per_system,
    }
    with open(result_dir / "symbolic_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judged symbolic accuracy")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--benchmark", choices=["odebench", "odebase"], required=True)
    parser.add_argument("--methods", type=str, nargs="*", default=None,
                        help="Method folders to score (default: all under results/<benchmark>/).")
    parser.add_argument("--judge_model", type=str, default=GPT_MODEL,
                        help="Paper uses GPT-4o-mini as the judge.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root) / args.benchmark
    methods = args.methods or [p.name for p in sorted(root.iterdir()) if (p / "systems").is_dir()]
    client = LLMClient(model=args.judge_model, temperature=0.0, max_tokens=8)

    overall = {}
    for method in methods:
        result_dir = root / method
        if not (result_dir / "systems").is_dir():
            print(f"skip {method}: no systems/ folder")
            continue
        print(f"\n=== symbolic accuracy: {args.benchmark}/{method} ===")
        judge = Judge(client, result_dir / "symbolic_accuracy_cache.json")
        summary = score_method(result_dir, judge, verbose=not args.quiet)
        overall[method] = summary["symbolic_accuracy"]
        print(f"  -> symbolic accuracy = {summary['symbolic_accuracy'] * 100:.1f}% "
              f"over {summary['n_systems']} systems")

    print("\n=== summary ===")
    for method, acc in sorted(overall.items(), key=lambda kv: -kv[1]):
        print(f"  {method:24s} {acc * 100:5.1f}%")
    print(f"\njudge calls: {client.n_calls}")


if __name__ == "__main__":
    main()
