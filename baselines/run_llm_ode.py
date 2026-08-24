"""LLM-ODE baseline (paper, Appendix B.2).

    "LLM-ODE uses the original implementation from llm-ode with LLM-proposed
     structural templates and numerical parameter optimization.  The multi-island
     evolutionary strategy and dynamic experience buffer are used as in the
     original implementation."

This driver imports the *official* evolutionary core from
https://github.com/gryaklab/llm-ode (``llmode.llmode.LlmOdeEquation``, islands,
programs, BFGS coefficient optimisation) and only replaces two things:

1. the transport layer -- the upstream ``Llm`` talks to a local vLLM server via
   the OpenAI *responses* API; we route the identical prompts through
   OpenRouter's chat-completions endpoint instead;
2. the data -- upstream generates its own trajectories on ``t in [0, 10]``,
   whereas we feed the benchmark's reconstruction window so that every method in
   this reproduction sees exactly the same observations.

Budget follows the paper: 125 LLM calls generating 1000 candidate equations per
system.  LLM-ODE issues ``n_islands`` prompts per dimension per iteration and
requests ``b`` hypotheses per prompt, so the number of iterations is
``125 / (n_islands * dim)`` with ``b = 8``.

Run ``scripts/setup_third_party.sh llm-ode`` first to clone the upstream repo
into ``third_party/llm-ode``.

``scripts/setup_third_party.sh`` also applies a one-line compat patch: upstream
calls ``str.replace(..., count=1)``, a keyword argument that only exists from
CPython 3.13 (their ``pyproject.toml`` pins ``requires-python == 3.13.5``).
Making it positional is behaviour-preserving and lets LLM-ODE run in the same
environment as everything else.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import types
from pathlib import Path

import numpy as np

from baselines import common
from baselines.llm_client import GPT_MODEL, LLMClient

DEFAULT_REPO = os.environ.get("LLM_ODE_REPO", "third_party/llm-ode")


def _install_stubs() -> list[str]:
    """``llmode.llm`` imports torch/openai at module level but we never use them.

    The stubs exist only for the duration of the import: SymPy inspects
    ``sys.modules`` for a real ``torch`` and raises ``module 'torch' has no
    attribute 'Tensor'`` on every ``sympify`` call if a fake one is left behind,
    which makes upstream's ``make_random_program`` retry forever.
    """
    installed = []
    for name in ("torch", "openai"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            if name == "openai":
                stub.OpenAI = object  # type: ignore[attr-defined]
            sys.modules[name] = stub
            installed.append(name)
    return installed


def _remove_stubs(installed: list[str]) -> None:
    for name in installed:
        sys.modules.pop(name, None)


def load_upstream(repo: Path):
    repo = repo.resolve()
    if sys.version_info < (3, 13):
        # Without the compat patch, upstream raises on every generated equation
        # and Island.__init__ loops forever instead of failing loudly.
        eq_src = repo / "llmode" / "equation.py"
        if eq_src.is_file() and "count=1" in eq_src.read_text(encoding="utf-8"):
            raise RuntimeError(
                "third_party/llm-ode still uses str.replace(count=...), which needs "
                f"Python 3.13 (this interpreter is {sys.version.split()[0]}). "
                "Run: bash scripts/setup_third_party.sh llm-ode"
            )
    if not (repo / "llmode").is_dir():
        raise FileNotFoundError(
            f"llm-ode not found at {repo}. Run: bash scripts/setup_third_party.sh llm-ode"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # Upstream evaluates hypotheses in a ProcessPoolExecutor. Under the "spawn"
    # start method the children do not inherit sys.path, so they must be able to
    # import llmode.* (and this package) from PYTHONPATH or they die on unpickle.
    root = Path(__file__).resolve().parent.parent
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(repo), str(root)] + ([existing] if existing else [])
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    installed = _install_stubs()
    try:
        from llmode import llmode as llmode_mod  # noqa: E402
        from llmode.llm import generate_prompt  # noqa: E402
    finally:
        _remove_stubs(installed)

    return llmode_mod, generate_prompt


class OpenRouterLlm:
    """Drop-in replacement for ``llmode.llm.Llm`` backed by our LLM client."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.n_queries = 1

    def query(self, prompts: list[list[dict]]) -> list[str]:
        outs = []
        for messages in prompts:
            system = next((m["content"] for m in messages if m["role"] == "system"), None)
            user = "\n".join(m["content"] for m in messages if m["role"] != "system")
            try:
                outs.append(self.client.chat(user, n=1, system=system, max_tokens=1024)[0])
            except Exception:
                outs.append("")
        self.n_queries += len(prompts)
        return outs


def fit(data: common.Dataset, logger, llmode_mod, client: LLMClient, args) -> dict:
    dim = data.dim
    train, val = data.train_split(0.2)
    llm = OpenRouterLlm(client)

    config = {
        "n_islands": args.n_islands,
        "island_size": args.island_size,
        "k": args.k,
        "b": args.b,
        "iters_per_refine": args.iters_per_refine,
        "num_mixing": args.num_mixing,
    }
    n_iterations = max(1, args.llm_calls // (args.n_islands * dim))
    logger.info(f"      [LLM-ODE] {n_iterations} iterations x {args.n_islands} islands x {dim} dims "
                f"= {n_iterations * args.n_islands * dim} calls, b={args.b}")

    pareto_dir = Path(args.results_root) / args.benchmark / (args.method_name or "llm_ode") / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)

    equations_per_dim = []
    for d in range(dim):
        eq_search = llmode_mod.LlmOdeEquation(
            llm=llm,
            t_train=train["t"], X_train=train["u"], y_train=train["du"][:, d],
            t_val=val["t"], X_val=val["u"], y_val=val["du"][:, d],
            pareto_level_file=pareto_dir / f"{data.name}_x{d}.complexity_pf.jsonl",
            config=config,
        )
        for it in range(n_iterations):
            try:
                eq_search.step()
            except Exception as exc:  # upstream can raise on degenerate populations
                logger.info(f"      [LLM-ODE] dim {d} iteration {it + 1} failed: {exc}")

        # Preferred selection: the complexity/MSE Pareto frontier upstream keeps.
        candidates = []
        try:
            pf = eq_search.get_pareto_frontier()
            candidates = [row["program"] for _, row in pf.iterrows()]
        except Exception as exc:
            logger.info(f"      [LLM-ODE] dim {d} Pareto frontier unavailable: {exc}")
        if not candidates:
            # Upstream's frontier bookkeeping can end up empty (it drops rows on
            # any evaluation error); fall back to the surviving island programs.
            candidates = [p for island in eq_search.islands for p in island.get_programs()]

        best_eq, best_v = None, np.inf
        for program in candidates:
            try:
                eq_str = str(program.equation.to_string(precision=20))
            except Exception:
                continue
            v = _dim_nmse(eq_str, dim, val["u"], val["du"][:, d])
            if v < best_v:
                best_v, best_eq = v, eq_str
        if best_eq is None:
            raise RuntimeError(f"LLM-ODE found no usable equation for dimension {d}")
        equations_per_dim.append(best_eq)
        logger.info(f"      [LLM-ODE] dx{d}/dt = {best_eq}  (val NMSE={best_v:.4e})")

    return {"equations": equations_per_dim, "llm_calls": client.n_calls,
            "n_iterations": n_iterations}


def _dim_nmse(eq_str: str, n_vars: int, u: np.ndarray, y: np.ndarray) -> float:
    f = common.equations_to_callable([eq_str], n_vars)
    if f is None:
        return float("inf")
    try:
        with np.errstate(all="ignore"):
            return common.nmse(y, f(u)[:, 0])
    except Exception:
        return float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-ODE baseline (upstream implementation)")
    common.add_common_args(parser)
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--model", type=str, default=GPT_MODEL)
    parser.add_argument("--llm_calls", type=int, default=125)
    parser.add_argument("--n_islands", type=int, default=4)
    parser.add_argument("--island_size", type=int, default=2)
    parser.add_argument("--k", type=int, default=8, help="in-context examples per prompt")
    parser.add_argument("--b", type=int, default=8, help="hypotheses requested per prompt")
    parser.add_argument("--iters_per_refine", type=int, default=5)
    parser.add_argument("--num_mixing", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    common.set_thread_env(1)
    common.silence_numeric_warnings()

    # Upstream validates hypotheses in a ProcessPoolExecutor. It is written for
    # Linux, where "fork" is the default and the children inherit sys.path and
    # the already-imported llmode modules. Under "spawn" (macOS default) the
    # children re-bootstrap __main__ and die, which silently empties every
    # island. Force "fork" wherever it exists.
    try:
        multiprocessing.set_start_method("fork", force=True)
    except (ValueError, RuntimeError):
        pass

    paths = common.resolve_data_paths(args)
    method = args.method_name or "llm_ode"
    result_dir = common.make_result_dir(args.results_root, args.benchmark, method)
    logger = common.setup_logger(result_dir, "llm_ode")
    common.quiet_third_party_logging(result_dir, "llmode_upstream.log")
    llmode_mod, _ = load_upstream(Path(args.repo))
    logger.info(f"LLM-ODE ({args.model}) on {args.benchmark}: {len(paths)} systems -> {result_dir}")

    def _fit(data, log):
        client = LLMClient(model=args.model, temperature=args.temperature,
                           log_path=result_dir / "llm_calls" / f"{data.name}.jsonl")
        out = fit(data, log, llmode_mod, client, args)
        out.update(client.stats())
        return out

    common.run_over_datasets(method, args.benchmark, paths, result_dir, _fit, logger,
                             resume=not args.no_resume, extra_meta={"model": args.model})
    logger.info("LLM-ODE done.")


if __name__ == "__main__":
    main()
