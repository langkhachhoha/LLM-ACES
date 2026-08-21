"""SINDy baseline (paper, Appendix B.2 + Table 10).

    "SINDy uses the PySINDy implementation with STLSQ (sequential thresholding
     least squares) optimizer.  The sparsity threshold is swept over a
     log-uniform grid from 1e-7 to 1 (16 values), and l2 regularization
     strength alpha in {1e-5, 1e-4}.  We extend the basis library beyond
     MDBench's polynomial-only setting to also search over the full expanded
     nonlinear library above, combined with polynomials up to degree 4."

Hyperparameter selection follows MDBench: an 80/20 chronological split of the
training window, pick the configuration with the highest complexity-aware
fitness on the validation part, then refit on the whole training window.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from baselines import common, ops

HYPERPARAMS = {
    "threshold": list(np.logspace(-7, 0, 16)),
    "alpha": [1e-5, 1e-4],
    "poly_order": [1, 2, 3, 4],
    "basis": ["polynomial", "expanded"],
}
MAX_ITER = 200


def _build_library(basis: str, poly_order: int, u_train: np.ndarray):
    """CustomLibrary with polynomials (+ the expanded unary set when asked).

    Unary features that are not finite on the training states (e.g. ``log`` of a
    negative trajectory) are dropped -- STLSQ cannot fit NaN columns, and the
    paper's library is meant to be *searched over*, not to poison the design
    matrix.
    """
    import pysindy as ps

    fns, names = [], []
    for p in range(1, poly_order + 1):
        fns.append((lambda q: (lambda x: x ** q))(p))
        names.append((lambda q: (lambda s: f"({s})**{q}"))(p))

    if basis == "expanded":
        u_fns, u_names = ops.numpy_unary_library()
        for f, n in zip(u_fns, u_names):
            with np.errstate(all="ignore"):
                try:
                    vals = f(u_train)
                except Exception:
                    continue
            if not np.all(np.isfinite(vals)):
                continue
            fns.append(f)
            names.append(n)

    return ps.feature_library.CustomLibrary(
        library_functions=fns, function_names=names, include_bias=True
    )


def _fit_one(u, du, t, threshold, alpha, poly_order, basis):
    import pysindy as ps

    lib = _build_library(basis, poly_order, u)
    opt = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=MAX_ITER)
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    with np.errstate(all="ignore"):
        model.fit(u, x_dot=du, t=t)
    return model


def _equations(model, dim: int) -> list[str]:
    """Rebuild RHS strings in terms of x0..x{dim-1} from the coefficient matrix."""
    coefs = np.asarray(model.coefficients())
    feats = list(model.get_feature_names())
    eqs = []
    for d in range(dim):
        terms = []
        for c, feat in zip(coefs[d], feats):
            if abs(c) <= 1e-12:
                continue
            f = feat.strip()
            # repr(np.float64) is "np.float64(...)" on NumPy 2.x -- cast first.
            if f == "1":
                terms.append(f"({float(c)!r})")
            else:
                terms.append(f"({float(c)!r})*({f})")
        eqs.append(" + ".join(terms) if terms else "0")
    return eqs


def fit(data: common.Dataset, logger) -> dict:
    train, val = data.train_split(0.2)
    best = None
    grid = list(itertools.product(
        HYPERPARAMS["threshold"], HYPERPARAMS["alpha"],
        HYPERPARAMS["poly_order"], HYPERPARAMS["basis"],
    ))
    for threshold, alpha, poly_order, basis in grid:
        try:
            model = _fit_one(train["u"], train["du"], train["t"], threshold, alpha, poly_order, basis)
            eqs = _equations(model, data.dim)
            f = common.equations_to_callable(eqs, data.dim)
            if f is None:
                continue
            with np.errstate(all="ignore"):
                v = common.nmse(val["du"], f(val["u"]))
            comp = common.expression_complexity(eqs, data.dim)
            score = common.fitness(v, comp)
        except Exception:
            continue
        if best is None or score > best[0]:
            best = (score, dict(threshold=threshold, alpha=alpha, poly_order=poly_order, basis=basis))

    if best is None:
        raise RuntimeError("no SINDy configuration produced a usable model")

    hp = best[1]
    logger.info(f"    best SINDy hyperparameters: {hp}")
    model = _fit_one(data.u, data.du, data.t, hp["threshold"], hp["alpha"], hp["poly_order"], hp["basis"])
    return {"equations": _equations(model, data.dim), "hyperparameters": hp}


def main() -> None:
    parser = argparse.ArgumentParser(description="SINDy baseline")
    common.add_common_args(parser)
    args = parser.parse_args()
    common.set_thread_env(1)

    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or "sindy")
    logger = common.setup_logger(result_dir, "sindy")
    logger.info(f"SINDy on {args.benchmark}: {len(paths)} systems -> {result_dir}")
    common.run_over_datasets("sindy", args.benchmark, paths, result_dir, fit, logger,
                             resume=not args.no_resume)
    logger.info("SINDy done.")


if __name__ == "__main__":
    main()
