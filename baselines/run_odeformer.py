"""ODEFormer baseline (paper, Appendix B.2 + Table 10).

    "ODEFormer is run in inference mode using the pretrained checkpoint,
     without fine-tuning.  As a fixed pretrained model, its internal operator
     vocabulary cannot be modified."

Table 10: beam size 50, beam temperature swept over {0.05, 0.1, 0.2, 0.3, 0.5}.
Following MDBench, the temperature is selected on a held-out 20% tail of the
reconstruction window using the complexity-aware fitness, then the model is
refit on the full window.

ODEFormer consumes a *trajectory* ``(t, X)`` and emits a symbolic system, which
we hand back to the shared evaluator as RHS strings.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np

from baselines import common

BEAM_TEMPERATURES = [0.05, 0.1, 0.2, 0.3, 0.5]
BEAM_SIZE = 50


def _predicted_equations(model, dim: int) -> list[str] | None:
    try:
        pred = model.predictions[0][0]
    except Exception:
        return None
    parts = str(pred).split("|")
    if len(parts) != dim:
        return None
    return [p.strip() for p in parts]


@lru_cache(maxsize=1)
def _regressor():
    """Load the pretrained checkpoint once per process.

    The temperature sweep plus the final refit means 6 fits per system; loading
    the checkpoint each time would cost ~700 model loads over a benchmark.
    Only `set_model_args` differs between fits, so one instance is enough.
    """
    from odeformer.model import SymbolicTransformerRegressor

    return SymbolicTransformerRegressor(from_pretrained=True)


def _fit_once(t: np.ndarray, u: np.ndarray, beam_temperature: float, dim: int):
    model = _regressor()
    model.set_model_args({"beam_size": BEAM_SIZE, "beam_temperature": beam_temperature})
    model.fit(t, u)
    return _predicted_equations(model, dim)


def fit(data: common.Dataset, logger) -> dict:
    train, val = data.train_split(0.2)

    best_temp, best_score = BEAM_TEMPERATURES[0], -np.inf
    for temp in BEAM_TEMPERATURES:
        try:
            eqs = _fit_once(train["t"], train["u"], temp, data.dim)
            if eqs is None:
                continue
            f = common.equations_to_callable(eqs, data.dim)
            if f is None:
                continue
            with np.errstate(all="ignore"):
                v = common.nmse(val["du"], f(val["u"]))
            score = common.fitness(v, common.expression_complexity(eqs, data.dim))
        except Exception as exc:
            logger.info(f"      [ODEFormer] beam_temperature={temp} failed: {exc}")
            continue
        logger.info(f"      [ODEFormer] beam_temperature={temp}: val nmse={v:.3e} score={score:.4f}")
        if score > best_score:
            best_score, best_temp = score, temp

    logger.info(f"    best beam_temperature = {best_temp}")
    eqs = _fit_once(data.t, data.u, best_temp, data.dim)
    if eqs is None:
        raise RuntimeError("ODEFormer produced no parsable prediction")
    return {"equations": eqs, "hyperparameters": {"beam_size": BEAM_SIZE, "beam_temperature": best_temp}}


def main() -> None:
    parser = argparse.ArgumentParser(description="ODEFormer baseline")
    common.add_common_args(parser)
    args = parser.parse_args()
    common.set_thread_env(1)

    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or "odeformer")
    logger = common.setup_logger(result_dir, "odeformer")
    logger.info(f"ODEFormer on {args.benchmark}: {len(paths)} systems -> {result_dir}")
    common.run_over_datasets("odeformer", args.benchmark, paths, result_dir, fit, logger,
                             resume=not args.no_resume)
    logger.info("ODEFormer done.")


if __name__ == "__main__":
    main()
