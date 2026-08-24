"""APPS-ODE baseline driver (paper, Appendix B.2).

    "APPS-ODE implementation uses the original grammar-RL pipeline with grammar
     function set {+, -, *, /, sin, exp, poly, const}.  We train APPS-ODE for 50
     policy-gradient epochs querying initial conditions with 100 observations as
     given in their official repository.  Coefficients are optimized with BFGS;
     the reward signal is inverse NMSE.  For both datasets, trajectories are
     queried from fresh initial conditions each epoch using the same oracle used
     for LLM-ACES."

APPS-ODE addresses systems by their *scibench* identifiers (``vars2_prog7``,
``odebase_vars3_prog2``), not by our NPZ stems, so ``scripts/build_scibench_map.py``
builds the name mapping by numerically matching ground-truth vector fields.

Each system is run in a subprocess inside the APPS-ODE checkout (it needs its own
conda environment because of the cython/numba ``grammar`` package); the resulting
equations are then scored by the shared evaluator so the numbers are comparable
with every other method here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from baselines import common

DEFAULT_REPO = os.environ.get("APPS_ODE_REPO", "third_party/APPS-ODE")
MAP_PATH = Path(__file__).resolve().parent / "scibench_map.json"


def load_map() -> dict:
    if not MAP_PATH.exists():
        raise FileNotFoundError(
            f"{MAP_PATH} missing. Run: python scripts/build_scibench_map.py"
        )
    with open(MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_fit(repo: Path, python_bin: str, args, result_dir: Path, name_map: dict):
    runner = Path(__file__).resolve().parent / "apps_ode_runner.py"
    pkg = repo / "apps_ode_pytorch"
    config = pkg / "config_regression.json"

    def fit(data: common.Dataset, logger) -> dict:
        eq_name = name_map.get(data.name)
        if not eq_name:
            raise RuntimeError(f"no scibench equation id for {data.name}; rebuild scibench_map.json")

        # The subprocess runs with cwd=apps_ode_pytorch, so every path handed to
        # it must be absolute or the output lands inside the vendored checkout.
        out_json = (result_dir / "raw" / f"{data.name}.json").resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin, str(runner), str(config),
            "--equation_name", eq_name,
            "--out", str(out_json),
            "--optimizer", args.optimizer,
            "--metric_name", args.metric_name,
            "--num_init_conds", str(args.num_init_conds),
            "--total_iterations", str(args.total_iterations),
            "--n_cores", str(args.n_cores),
            "--trajectory_time_steps", str(args.trajectory_time_steps),
            "--t_end", str(args.t_end),
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(pkg), str(repo / "src" / "grammar"),
             str(repo / "data_oracle" / "scibench"), env.get("PYTHONPATH", "")]
        )
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
        logger.info(f"      [APPS-ODE] {eq_name}: {' '.join(cmd)}")

        log_file = (result_dir / "raw" / f"{data.name}.log").resolve()
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(pkg), env=env, stdout=lf,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
        if proc.returncode != 0 or not out_json.exists():
            raise RuntimeError(f"APPS-ODE failed (exit {proc.returncode}); see {log_file}")

        with open(out_json, encoding="utf-8") as f:
            payload = json.load(f)
        eqs = payload.get("equations")
        used_topk = False
        if not eqs or len(eqs) != data.dim:
            # `model.train()` returns best_expression=None whenever the reward
            # threshold is never reached (common on short runs and on hard
            # systems); the ranked population is still there, so use its head.
            for cand in payload.get("topk") or []:
                if cand and len(cand) == data.dim:
                    eqs, used_topk = cand, True
                    break
        if not eqs or len(eqs) != data.dim:
            raise RuntimeError(
                f"APPS-ODE returned no {data.dim}-D system (best={payload.get('equations')!r}, "
                f"topk empty); see {log_file}"
            )
        return {
            "equations": eqs,
            "from_topk": used_topk,
            "scibench_equation": eq_name,
            "best_reward": payload.get("best_reward"),
            "apps_ode_seconds": payload.get("train_seconds"),
        }

    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description="APPS-ODE baseline")
    common.add_common_args(parser)
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--python_bin", type=str, default=sys.executable,
                        help="Python interpreter of the apps-ode conda environment.")
    parser.add_argument("--total_iterations", type=int, default=50, help="policy-gradient epochs")
    parser.add_argument("--optimizer", type=str, default="BFGS")
    parser.add_argument("--metric_name", type=str, default="inv_nmse")
    parser.add_argument("--num_init_conds", type=int, default=10)
    parser.add_argument("--trajectory_time_steps", type=int, default=100)
    parser.add_argument("--t_end", type=float, default=1.0)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3 * 60 * 60)
    args = parser.parse_args()
    common.silence_numeric_warnings()

    paths = common.resolve_data_paths(args)
    method = args.method_name or "apps_ode"
    result_dir = common.make_result_dir(args.results_root, args.benchmark, method)
    logger = common.setup_logger(result_dir, "apps_ode")
    repo = Path(args.repo).resolve()
    if not (repo / "apps_ode_pytorch").is_dir():
        raise SystemExit(f"APPS-ODE not found at {repo}. Run: bash scripts/setup_third_party.sh apps-ode")

    logger.info(f"APPS-ODE on {args.benchmark}: {len(paths)} systems -> {result_dir}")
    fit = make_fit(repo, args.python_bin, args, result_dir, load_map())
    common.run_over_datasets(method, args.benchmark, paths, result_dir, fit, logger,
                             resume=not args.no_resume)
    logger.info("APPS-ODE done.")


if __name__ == "__main__":
    main()
