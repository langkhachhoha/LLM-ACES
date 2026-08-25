"""PySR baseline (paper, Appendix B.2 + Table 10).

    "PySR uses MDBench's implementation of PySR with multi-population
     evolutionary search over expression trees, with an expanded operator set.
     Nested constraints prevent pathological compositions (e.g. exp(exp(.))),
     and the complexity-fitness tradeoff is scored via a Pareto fitness metric."

Table 10 settings: 100 iterations, 1000 cycles per iteration, 20 populations of
size 100, maxsize 40, maxdepth 20, expanded unary/binary operator sets.

One PySRRegressor is fitted per state dimension; the returned equation is the
Pareto-front member maximising MDBench's ``fitness(nmse, complexity)``.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from baselines import common, ops


def resolve_parallelism(mode: str, procs: int) -> str:
    """``auto`` -> serial for one process, Julia *threads* above that.

    Threads are also PySR's own default. The alternative, "multiprocessing",
    runs the populations in ``procs`` separate Julia processes over Distributed,
    and when PySR tears those workers down at the end of a fit the search loop's
    monitoring tasks are still fetching from them, so every run ends with a
    stack of ``UNHANDLED TASK ERROR: Distributed.ProcessExitedException(n)``.
    That noise is harmless -- the equations are already selected by then -- but
    it is indistinguishable from a real worker crash, and threads share one heap
    so they are lighter on memory-limited machines. Both modes were verified to
    return the same equation on rc-circuit.
    """
    if mode != "auto":
        return mode
    return "serial" if procs <= 1 else "multithreading"


def build_regressor(niterations: int, populations: int, population_size: int,
                    ncycles: int, maxsize: int, maxdepth: int, procs: int,
                    seed: int, timeout_s: float | None = None,
                    unary: list[str] | None = None, binary: list[str] | None = None,
                    parallelism: str = "auto"):
    mode = resolve_parallelism(parallelism, procs)
    if mode == "multithreading" and procs > 1:
        # juliacall reads this when Julia starts, i.e. on the first pysr import.
        os.environ.setdefault("PYTHON_JULIACALL_THREADS", str(procs))

    from pysr import PySRRegressor

    common.quiet_julia_logging()
    unary = list(unary if unary is not None else ops.PYSR_UNARY)
    binary = list(binary if binary is not None else ops.PYSR_BINARY)
    nested = {k: {kk: vv for kk, vv in v.items() if kk in unary}
              for k, v in ops.PYSR_NESTED_CONSTRAINTS.items() if k in unary}
    constraints = {k: v for k, v in ops.PYSR_CONSTRAINTS.items() if k in binary}

    kwargs = dict(
        niterations=niterations,
        ncycles_per_iteration=ncycles,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        maxdepth=maxdepth,
        binary_operators=binary,
        unary_operators=unary,
        constraints=constraints,
        nested_constraints=nested,
        adaptive_parsimony_scaling=1000.0,
        weight_optimize=0.001,
        verbosity=0,
        progress=False,
        random_state=seed,
        deterministic=mode == "serial",
        parallelism=mode,
        temp_equation_file=True,
    )
    if mode == "multiprocessing" and procs > 1:
        kwargs["procs"] = procs
    if timeout_s and timeout_s > 0:
        kwargs["timeout_in_seconds"] = timeout_s
    return PySRRegressor(**kwargs)


def pareto_best_equation(model, X: np.ndarray, y: np.ndarray) -> str:
    """Pick the Pareto-front equation with the best MDBench fitness."""
    import sympy as sp

    eqs = model.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    best_str, best_score = None, -np.inf
    for _, row in eqs.iterrows():
        try:
            with np.errstate(all="ignore"):
                pred = row["lambda_format"](X)
            score = common.fitness(common.nmse(y, np.asarray(pred, dtype=float).reshape(-1)),
                                   int(row["complexity"]))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_str = str(sp.sympify(row["sympy_format"]))
    if best_str is None:
        best_str = str(model.sympy())
    return best_str


def fit_pysr_system(u: np.ndarray, du: np.ndarray, dim: int, args, seed: int,
                    logger=None, unary=None, binary=None) -> list[str]:
    """Fit one PySR model per dimension and return the chosen RHS strings."""
    eqs = []
    for d in range(dim):
        if logger:
            logger.info(f"      [PySR] fitting dx{d}/dt (of {dim}) — "
                        f"{args.pysr_niterations} iterations x {args.pysr_populations} "
                        f"populations x {args.pysr_ncycles} cycles, no output until it lands")
        t_fit = time.time()
        model = build_regressor(
            niterations=args.pysr_niterations, populations=args.pysr_populations,
            population_size=args.pysr_population_size, ncycles=args.pysr_ncycles,
            maxsize=args.pysr_maxsize, maxdepth=args.pysr_maxdepth,
            procs=args.pysr_procs, seed=seed + d, timeout_s=args.pysr_timeout,
            unary=unary, binary=binary,
            parallelism=getattr(args, "pysr_parallelism", "auto"),
        )
        model.fit(u, du[:, d])
        eq = pareto_best_equation(model, u, du[:, d])
        eqs.append(eq)
        if logger:
            logger.info(f"      [PySR] dx{d}/dt = {eq}   ({time.time() - t_fit:.0f}s)")
    return eqs


def add_pysr_args(parser):
    parser.add_argument("--pysr_niterations", type=int, default=100)
    parser.add_argument("--pysr_populations", type=int, default=20)
    parser.add_argument("--pysr_population_size", type=int, default=100)
    parser.add_argument("--pysr_ncycles", type=int, default=1000)
    parser.add_argument("--pysr_maxsize", type=int, default=40)
    parser.add_argument("--pysr_maxdepth", type=int, default=20)
    parser.add_argument("--pysr_procs", type=int, default=1)
    parser.add_argument("--pysr_parallelism", type=str, default="auto",
                        choices=["auto", "serial", "multithreading", "multiprocessing"],
                        help="auto = serial for --pysr_procs 1, multiprocessing above. "
                             "Use multithreading if Julia workers get OOM-killed "
                             "(Distributed.ProcessExitedException).")
    parser.add_argument("--pysr_timeout", type=float, default=0.0,
                        help="Per-dimension PySR wall-clock cap in seconds (0 = no cap).")
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(description="PySR baseline")
    common.add_common_args(parser)
    add_pysr_args(parser)
    args = parser.parse_args()
    common.set_thread_env(1 if args.pysr_procs <= 1 else args.pysr_procs)
    common.silence_numeric_warnings()

    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or "pysr")
    logger = common.setup_logger(result_dir, "pysr")
    logger.info(f"PySR on {args.benchmark}: {len(paths)} systems -> {result_dir}")

    def fit(data: common.Dataset, log):
        eqs = fit_pysr_system(data.u, data.du, data.dim, args, args.seed, log)
        return {"equations": eqs}

    common.run_over_datasets("pysr", args.benchmark, paths, result_dir, fit, logger,
                             resume=not args.no_resume)
    logger.info("PySR done.")


if __name__ == "__main__":
    main()
