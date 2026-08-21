"""Runner executed *inside* the APPS-ODE repository.

This mirrors ``apps_ode_pytorch/main.py`` from
https://github.com/jiangnanhugo/APPS-ODE one-for-one, with the paper's
Appendix B.2 settings hard-wired

    * grammar function set {+, -, *, /, sin, exp, poly, const}
    * 50 policy-gradient epochs
    * BFGS coefficient optimisation
    * inverse-NMSE reward
    * trajectories queried from fresh initial conditions each epoch,
      100 observations per trajectory

and writes the discovered system to JSON instead of printing it, so the outer
driver (``baselines/run_apps_ode.py``) can score it with the shared evaluator.

Invoked as::

    python apps_ode_runner.py <config.json> --equation_name odebase_vars2_prog1 \\
        --out result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_template")
    ap.add_argument("--equation_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--optimizer", default="BFGS")
    ap.add_argument("--metric_name", default="inv_nmse")
    ap.add_argument("--num_init_conds", type=int, default=10)
    ap.add_argument("--num_regions", type=int, default=10)
    ap.add_argument("--noise_type", default="normal")
    ap.add_argument("--noise_scale", type=float, default=0.0)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--total_iterations", type=int, default=50)
    ap.add_argument("--n_cores", type=int, default=1)
    ap.add_argument("--use_gpu", type=int, default=-1)
    ap.add_argument("--active_mode", default="default")
    ap.add_argument("--trajectory_time_steps", type=int, default=100)
    ap.add_argument("--t_end", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from scibench.symbolic_equation_evaluator import Equation_evaluator
    from scibench.symbolic_data_generator import DataX

    from grammar.grammar import ContextFreeGrammar
    from grammar.grammar_regress_task import RegressTask
    from grammar.production_rules import (get_production_rules,
                                          construct_non_terminal_nodes_and_start_symbols)
    from grammar.grammar_program import grammarProgram
    from active_deep_symbolic_regression import ActDeepSymbolicRegression

    threshold_values = {
        "neg_mse": {"reward_threshold": -1e-6},
        "neg_nmse": {"reward_threshold": -1e-6},
        "neg_nrmse": {"reward_threshold": -1e-3},
        "neg_rmse": {"reward_threshold": -1e-3},
        "inv_mse": {"reward_threshold": 1 / (1 + 1e-6)},
        "inv_nmse": {"reward_threshold": 1 / (1 + 1e-6)},
        "inv_nrmse": {"reward_threshold": 1 / (1 + 1e-6)},
    }

    data_query_oracle = Equation_evaluator(
        args.equation_name, args.noise_type, args.noise_scale,
        metric_name=args.metric_name, time_sequence_drop_rate=0,
    )
    dataXgen = DataX(data_query_oracle.vars_range_and_types_to_json)
    nvars = data_query_oracle.get_nvars()

    # Paper, Appendix B.2: APPS-ODE keeps its own (smaller) grammar function set.
    function_set = ["add", "sub", "mul", "div", "sin", "exp", "poly", "const"]

    # Reconstruction window of the benchmark: t in [0, 1] with 100 samples.
    time_span = (1e-4, args.t_end)
    t_eval = np.linspace(time_span[0], time_span[1], args.trajectory_time_steps)

    task = RegressTask(args.num_init_conds, nvars, dataXgen, data_query_oracle,
                       time_span, t_eval, num_of_regions=args.num_regions, width=0.1)

    non_terminal_nodes, start_symbols = construct_non_terminal_nodes_and_start_symbols(nvars)
    production_rules = []
    for node in non_terminal_nodes:
        production_rules.extend(get_production_rules(nvars, function_set, node))

    program = grammarProgram(
        non_terminal_nodes=non_terminal_nodes, optimizer=args.optimizer,
        metric_name=args.metric_name, n_cores=args.n_cores, max_opt_iter=100,
    )
    grammar_model = ContextFreeGrammar(
        nvars=nvars, production_rules=production_rules, start_symbols=start_symbols,
        non_terminal_nodes=non_terminal_nodes, max_length=args.max_len,
        topK_size=10, reward_threhold=threshold_values[args.metric_name],
    )
    grammar_model.task = task
    grammar_model.program = program

    model = ActDeepSymbolicRegression(args.config_template, grammar_model)
    device = (torch.device(f"cuda:{args.use_gpu}")
              if args.use_gpu >= 0 and torch.cuda.is_available() else torch.device("cpu"))
    model.setup(device)

    t0 = time.time()
    epoch_best_rewards, epoch_best_expressions, best_reward, best_expression = model.train(
        threshold_values[args.metric_name]["reward_threshold"], args.total_iterations, args.active_mode
    )
    elapsed = time.time() - t0

    def _as_list(expr):
        if expr is None:
            return None
        fitted = getattr(expr, "fitted_eq", None)
        if fitted is None:
            return None
        return [str(e) for e in fitted]

    topk = getattr(grammar_model, "best_predicted_equations", []) or []
    best_equations = _as_list(best_expression)
    if best_equations is None:
        # train() yields None unless the reward threshold is hit; fall back to
        # the ranked population so the run still produces a system.
        for expr in topk:
            best_equations = _as_list(expr)
            if best_equations:
                break

    payload = {
        "equation_name": args.equation_name,
        "equations": best_equations,
        "best_reward": float(best_reward) if best_reward is not None else None,
        "train_seconds": round(elapsed, 2),
        "topk": [_as_list(e) for e in topk[:10]],
        "epoch_best_rewards": [float(r) for r in np.asarray(epoch_best_rewards).ravel().tolist()],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[apps_ode_runner] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
