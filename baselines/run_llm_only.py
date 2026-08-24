"""LLM-only iterative-refinement baseline (paper, Appendix B.2).

    "LLM-only follows the protocol from NewtonBench, prompting GPT-4o-mini
     iteratively with feedback from the previous round to guide the future
     equations.  The LLM is implemented using a temperature tau = 1.0 to
     maximize generative diversity."

    "All LLM-based baselines use GPT-4o-mini and run for 125 LLM calls,
     generating 1000 candidate equations."

So: 125 calls x 8 candidate systems per call = 1000 candidates.  Each round the
prompt carries the best and worst candidates seen so far (with their scores),
exactly the feedback loop NewtonBench uses.  Candidates are scored by NMSE on a
held-out validation split of the reconstruction window; the best-scoring system
is returned.

Following the paper (Section 2.4 / 4.3) the prompt contains *no* semantic
description of the system -- only the anonymised state variables and data.
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

from baselines import common
from baselines.llm_client import GPT_MODEL, LLMClient

PROMPT = """You are an expert in discovering governing equations of dynamical systems from data.

## TASK
Find the right-hand side of an autonomous ordinary differential equation system of dimension {dim}:
{dim_lines}

You are given observations of the state variables and their time derivatives.

## DATA (state -> derivative)
{data_block}

## SUMMARY STATISTICS
{stats_block}

## FEEDBACK FROM PREVIOUS ROUNDS
### Best candidates so far (lower NMSE is better)
{best_block}
### Worst candidates so far -- avoid these functional forms
{worst_block}

## RULES
- Use only the variables {var_list} and numeric constants.
- Allowed operators: +, -, *, /, **, sin, cos, tan, exp, log, sqrt, abs, tanh, sinh, cosh.
- Write concrete numeric constants (no symbolic placeholders); they are NOT optimised for you.
- Propose {n_candidates} DIFFERENT candidate systems that improve on the best candidates above.

## OUTPUT FORMAT (strict)
Return exactly {n_candidates} candidates, ONE PER LINE, and nothing else:
{output_format}
Do not number the lines, do not add commentary, do not use code fences.
"""


def _output_format(dim: int) -> str:
    if dim == 1:
        return ("CANDIDATE: <rhs for dx0/dt>\n"
                "One candidate per line. Do NOT put several candidates on the same line "
                "and do NOT use the '|' character -- this system has a single dimension.")
    rhs = " | ".join(f"<rhs for dx{i}/dt>" for i in range(dim))
    return (f"CANDIDATE: {rhs}\n"
            f"Exactly {dim} right-hand sides per line, separated by '|'. "
            f"One candidate per line.")


def _data_block(u: np.ndarray, du: np.ndarray, n: int = 20) -> str:
    idx = np.linspace(0, len(u) - 1, min(n, len(u))).astype(int)
    dim = u.shape[1]
    head = "  ".join([f"x{i}" for i in range(dim)] + [f"dx{i}/dt" for i in range(dim)])
    rows = [head]
    for i in idx:
        rows.append("  ".join(f"{v: .6g}" for v in list(u[i]) + list(du[i])))
    return "\n".join(rows)


def _stats_block(u: np.ndarray, du: np.ndarray) -> str:
    lines = []
    for i in range(u.shape[1]):
        lines.append(
            f"x{i}: min={u[:, i].min():.4g} max={u[:, i].max():.4g} mean={u[:, i].mean():.4g} | "
            f"dx{i}/dt: min={du[:, i].min():.4g} max={du[:, i].max():.4g} mean={du[:, i].mean():.4g}"
        )
    return "\n".join(lines)


# A prefix is only a prefix when it is followed by whitespace. Without those
# lookaheads "-4.8 * x0" loses its sign (the "-" reads as a bullet and "4." as
# list numbering) and "CANDIDATE -0.3*x0" loses its minus to the trailing
# punctuation class -- both silently corrupt the equation.
_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*+\u2022](?=\s)"                      # bullet: "- expr"
    r"|\d+\s*[\.\):](?=\s)"                  # numbering: "3. expr" / "3) expr"
    r"|candidate\s*\d*\s*[:\.]?"             # label: "CANDIDATE:" / "candidate 3."
    r"|system\s*\d*\s*[:\.]?"
    r"|eq(?:uation)?\s*\d*\s*[:\.]?"
    r")\s*",
    re.IGNORECASE,
)


def _strip_prefixes(line: str) -> str:
    """Remove any stack of list markers / labels, e.g. '3. CANDIDATE: ...'."""
    body = line.strip().strip("`").strip()
    for _ in range(4):
        stripped = _PREFIX_RE.sub("", body, count=1).strip()
        if stripped == body:
            break
        body = stripped
    return body.strip("`").strip()


def parse_candidates(text: str, dim: int) -> list[list[str]]:
    """Extract candidate systems from a raw completion.

    Handles the shapes GPT-4o-mini actually emits: bare lines, numbered lines,
    'CANDIDATE:'-labelled lines, both at once, and -- for 1-D systems -- several
    candidates crammed onto one '|'-separated line.
    """
    out: list[list[str]] = []
    for raw_line in text.splitlines():
        body = _strip_prefixes(raw_line)
        if not body or body.startswith("#"):
            continue
        parts = [p.strip() for p in body.split("|")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if len(parts) == dim:
            out.append(parts)
        elif len(parts) > dim and len(parts) % dim == 0:
            # several candidates on one line (common when dim == 1)
            for i in range(0, len(parts), dim):
                out.append(parts[i:i + dim])
    return out


def fit(data: common.Dataset, logger, client: LLMClient, n_calls: int,
        n_candidates: int) -> dict:
    dim = data.dim
    train, val = data.train_split(0.2)
    var_list = ", ".join(f"x{i}" for i in range(dim))
    dim_lines = "\n".join(f"  dx{i}/dt = f{i}({var_list})" for i in range(dim))

    scored: list[tuple[float, int, list[str]]] = []   # (val nmse, complexity, equations)
    seen: set[str] = set()
    n_parsed = 0

    for call in range(n_calls):
        ranked = sorted(scored, key=lambda e: e[0])
        best_block = "\n".join(
            f"  NMSE={e[0]:.4e}  " + " | ".join(e[2]) for e in ranked[:3]
        ) or "  (none yet)"
        worst_block = "\n".join(
            f"  NMSE={e[0]:.4e}  " + " | ".join(e[2]) for e in ranked[-3:][::-1]
        ) or "  (none yet)"

        prompt = PROMPT.format(
            dim=dim, dim_lines=dim_lines, var_list=var_list,
            data_block=_data_block(train["u"], train["du"]),
            stats_block=_stats_block(train["u"], train["du"]),
            best_block=best_block, worst_block=worst_block,
            n_candidates=n_candidates, output_format=_output_format(dim),
        )
        try:
            raw = client.chat(prompt, n=1, temperature=1.0, max_tokens=1500)[0]
        except Exception as exc:
            logger.info(f"      [LLM-only] call {call + 1} failed: {exc}")
            continue

        for cand in parse_candidates(raw, dim)[:n_candidates]:
            key = " | ".join(cand)
            if key in seen:
                continue
            seen.add(key)
            n_parsed += 1
            f = common.equations_to_callable(cand, dim)
            if f is None:
                continue
            try:
                with np.errstate(all="ignore"):
                    v = common.nmse(val["du"], f(val["u"]))
            except Exception:
                continue
            scored.append((v, common.expression_complexity(cand, dim), cand))

        if (call + 1) % 25 == 0:
            best = min(scored, key=lambda e: e[0])[0] if scored else float("nan")
            logger.info(f"      [LLM-only] {call + 1}/{n_calls} calls, "
                        f"{n_parsed} candidates, best val NMSE={best:.4e}")

    if not scored:
        raise RuntimeError(
            f"LLM produced no evaluable candidate equations after {client.n_calls} calls "
            f"({n_parsed} parsed, none numerically usable). "
            f"Raw responses are in the llm_calls/ log for this system."
        )
    best = min(scored, key=lambda e: (e[0], e[1]))
    return {
        "equations": best[2],
        "val_nmse": float(best[0]),
        "n_candidates": n_parsed,
        "llm_calls": client.n_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-only iterative refinement baseline")
    common.add_common_args(parser)
    parser.add_argument("--model", type=str, default=GPT_MODEL)
    parser.add_argument("--n_calls", type=int, default=125,
                        help="LLM calls per system (paper: 125).")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Candidate systems requested per call (125 x 8 = 1000).")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    common.set_thread_env(1)
    common.silence_numeric_warnings()

    paths = common.resolve_data_paths(args)
    method = args.method_name or "llm_only"
    result_dir = common.make_result_dir(args.results_root, args.benchmark, method)
    logger = common.setup_logger(result_dir, "llm_only")
    logger.info(f"LLM-only ({args.model}) on {args.benchmark}: {len(paths)} systems -> {result_dir}")
    logger.info(f"  budget: {args.n_calls} calls x {args.n_candidates} candidates")

    def _fit(data, log):
        client = LLMClient(model=args.model, temperature=args.temperature,
                           log_path=result_dir / "llm_calls" / f"{data.name}.jsonl")
        t0 = time.time()
        out = fit(data, log, client, args.n_calls, args.n_candidates)
        out["llm_seconds"] = round(time.time() - t0, 2)
        out.update(client.stats())
        return out

    common.run_over_datasets(method, args.benchmark, paths, result_dir, _fit, logger,
                             resume=not args.no_resume, extra_meta={"model": args.model})
    logger.info("LLM-only done.")


if __name__ == "__main__":
    main()
