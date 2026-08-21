"""End2End (E2E) transformer baseline (paper, Appendix B.2 + Table 10).

Kamienny et al. 2022, "End-to-end symbolic regression with transformers".  The
paper follows MDBench's wrapper around
https://github.com/facebookresearch/symbolicregression, with Table 10 settings:
``max input points = 200``, ``#trees to refine = 10``, ``rescale = True``.

E2E is a *static* symbolic regressor: one model per state dimension, mapping
states u -> du.  It needs the upstream repo on ``sys.path`` and the pretrained
``model.pt`` checkpoint; see ``--sr_repo`` / ``--checkpoint``
(``scripts/setup_e2e.sh`` downloads both).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from baselines import common

MAX_INPUT_POINTS = 200
N_TREES_TO_REFINE = 10
RESCALE = True


def _load_base_model(repo: Path, checkpoint: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"E2E checkpoint not found at {checkpoint}. Run scripts/setup_e2e.sh first."
        )
    map_location = None if torch.cuda.is_available() else torch.device("cpu")
    try:
        model = torch.load(checkpoint, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        model = torch.load(checkpoint, map_location=map_location)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def make_fit(repo: Path, checkpoint: Path):
    base_model = _load_base_model(repo, checkpoint)
    from symbolicregression.model import SymbolicTransformerRegressor  # noqa: E402

    def fit(data: common.Dataset, logger) -> dict:
        eqs = []
        for d in range(data.dim):
            model = SymbolicTransformerRegressor(
                model=base_model,
                max_input_points=MAX_INPUT_POINTS,
                n_trees_to_refine=N_TREES_TO_REFINE,
                rescale=RESCALE,
            )
            model.fit(data.u, data.du[:, d])
            tree = model.retrieve_tree(refinement_type="BFGS", with_infos=True)["relabed_predicted_tree"]
            rhs = model.model.env.simplifier.tree_to_sympy_expr(tree)
            s = str(rhs)
            for j in range(data.dim):
                s = s.replace(f"x_{j}", f"x{j}")
            eqs.append(s)
            logger.info(f"      [E2E] dx{d}/dt = {s}")
        return {"equations": eqs}

    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description="End2End (E2E) transformer baseline")
    common.add_common_args(parser)
    parser.add_argument("--sr_repo", type=str, default=os.environ.get("E2E_REPO", "third_party/symbolicregression"))
    parser.add_argument("--checkpoint", type=str, default=os.environ.get("E2E_CHECKPOINT", "third_party/symbolicregression/model.pt"))
    args = parser.parse_args()
    common.set_thread_env(1)

    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or "e2e")
    logger = common.setup_logger(result_dir, "e2e")
    logger.info(f"E2E on {args.benchmark}: {len(paths)} systems -> {result_dir}")

    fit = make_fit(Path(args.sr_repo).resolve(), Path(args.checkpoint).resolve())
    common.run_over_datasets("e2e", args.benchmark, paths, result_dir, fit, logger,
                             resume=not args.no_resume)
    logger.info("E2E done.")


if __name__ == "__main__":
    main()
