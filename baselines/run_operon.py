"""Operon baseline (paper, Appendix B.2 + Table 10).

    "Operon uses the expanded operator set for operon on top of implementation
     given in [MDBench].  Model selection uses minimum description length on
     the Pareto front."

Hyperparameters follow MDBench's ``default_hyper_params`` with the paper's
Table 10 overrides (brood 10, max depth 10, max length 50, pool/population
1000, tournament 3, mutation 0.25, Levenberg-Marquardt local search) and the
expanded ``allowed_symbols``.
"""
from __future__ import annotations

import argparse
from multiprocessing import cpu_count

import numpy as np

from baselines import common, ops

DEFAULT_HYPER_PARAMS = {
    "allowed_symbols": ops.OPERON_SYMBOLS,
    "brood_size": 10,
    "comparison_factor": 0,
    "crossover_internal_probability": 0.9,
    "crossover_probability": 1.0,
    "epsilon": 1e-05,
    "female_selector": "tournament",
    "male_selector": "tournament",
    "generations": 1000,
    "initialization_max_depth": 5,
    "initialization_max_length": 10,
    "initialization_method": "btc",
    "irregularity_bias": 0.0,
    "local_search_probability": 1.0,
    "lamarckian_probability": 1.0,
    "optimizer_iterations": 1,
    "optimizer": "lm",
    "max_depth": 10,
    "max_evaluations": 1_000_000,
    "max_length": 50,
    "max_selection_pressure": 100,
    "model_selection_criterion": "minimum_description_length",
    "mutation_probability": 0.25,
    "objectives": ["r2", "length"],
    "offspring_generator": "os",
    "pool_size": 1000,
    "population_size": 1000,
    "random_state": 42,
    "reinserter": "keep-best",
    "max_time": 43200,
    "tournament_size": 3,
    "add_model_intercept_term": True,
    "add_model_scale_term": True,
}


def fit(data: common.Dataset, logger, n_threads: int = 1, max_time: int = 43200) -> dict:
    import sympy as sp
    from pyoperon.sklearn import SymbolicRegressor

    u = np.array(data.u, dtype=np.float64, order="F")
    eqs = []
    for d in range(data.dim):
        params = dict(DEFAULT_HYPER_PARAMS)
        params["max_time"] = max_time
        model = SymbolicRegressor(**params, n_threads=n_threads)
        y = data.du[:, d]
        model.fit(u, y)

        # MDL selection on the Pareto front, tie-broken by MDBench fitness.
        pareto = model.pareto_front_
        best_idx, best_score = 0, -np.inf
        for i, sol in enumerate(pareto):
            try:
                pred = model.evaluate_model(sol["tree"], u)
                score = common.fitness(common.nmse(y, np.asarray(pred, dtype=float)),
                                       int(sol["complexity"]))
            except Exception:
                continue
            if score > best_score:
                best_score, best_idx = score, i
        model.model_ = pareto[best_idx]["tree"]

        s = model.get_model_string(model.model_, precision=10,
                                   names=[f"x{i}" for i in range(data.dim)])
        s = s.replace("^", "**")
        try:
            s = str(sp.parse_expr(s))
        except Exception:
            pass
        eqs.append(s)
        logger.info(f"      [Operon] dx{d}/dt = {s}")
    return {"equations": eqs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Operon baseline")
    common.add_common_args(parser)
    parser.add_argument("--n_threads", type=int, default=1)
    parser.add_argument("--max_time", type=int, default=43200,
                        help="Per-dimension Operon wall-clock cap in seconds.")
    args = parser.parse_args()
    common.set_thread_env(args.n_threads)

    n_threads = cpu_count() if args.n_threads == -1 else args.n_threads
    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or "operon")
    logger = common.setup_logger(result_dir, "operon")
    logger.info(f"Operon on {args.benchmark}: {len(paths)} systems -> {result_dir}")

    common.run_over_datasets(
        "operon", args.benchmark, paths, result_dir,
        lambda d, log: fit(d, log, n_threads=n_threads, max_time=args.max_time),
        logger, resume=not args.no_resume,
    )
    logger.info("Operon done.")


if __name__ == "__main__":
    main()
