"""Score LLM-ACES runs with the same evaluator as every baseline.

``llm-aces/active_llm_aces.py`` writes one JSON per system into ``outputs/``
containing the best equation per state dimension, but it does not compute the
reconstruction / generalization / OOD numbers, the expression complexity, or the
symbolic-accuracy inputs that Tables 2 and 3 report.  This converter reads those
outputs and re-scores them through ``baselines.common``, producing the same
``results/<benchmark>/<method>/systems/*.json`` layout the baselines use.

    python -m baselines.eval_llm_aces --benchmark odebench \\
        --outputs_dir outputs/odebench --method_name llm_aces_gpt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines import common


def _equations_from_output(payload: dict, dim: int) -> list[str] | None:
    best = payload.get("best_equations") or []
    by_dim: dict[int, str] = {}
    for entry in best:
        d = entry.get("dimension")
        if d is None:
            continue
        by_dim[int(d)] = str(entry.get("equation", ""))
    if not by_dim:
        return None
    return [by_dim.get(d, "0") for d in range(dim)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score LLM-ACES outputs with the shared evaluator")
    parser.add_argument("--benchmark", choices=["odebench", "odebase"], required=True)
    parser.add_argument("--outputs_dir", type=str, required=True,
                        help="Directory of per-system JSON files written by active_llm_aces.py")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--method_name", type=str, default="llm_aces")
    parser.add_argument("--logs_dir", type=str, default=None,
                        help="Optional logs/<benchmark> dir; iteration counts are copied into the result.")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    data_root = Path(args.data_root or ("data/ode" if args.benchmark == "odebench" else "data/odebase"))
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name)
    logger = common.setup_logger(result_dir, "eval_llm_aces")

    npz_by_stem = {p.stem: p for p in common.discover_datasets(data_root, args.benchmark)}
    outputs = sorted(Path(args.outputs_dir).glob("*.json"))
    if not outputs:
        raise SystemExit(f"no LLM-ACES outputs found in {args.outputs_dir}")
    logger.info(f"Scoring {len(outputs)} LLM-ACES outputs -> {result_dir}")

    n_ok = 0
    for out_path in outputs:
        with open(out_path, encoding="utf-8") as f:
            payload_in = json.load(f)
        system = payload_in.get("problem") or out_path.stem
        npz = npz_by_stem.get(system)
        if npz is None:
            logger.info(f"  SKIP {system}: no NPZ under {data_root}")
            continue

        data = common.Dataset(npz)
        payload = {
            "system": system, "benchmark": args.benchmark, "method": args.method_name,
            "model": args.model, "data_path": str(npz), "dim": data.dim,
            "status": "error", "error": "",
            "queried_initial_conditions": payload_in.get("queried_initial_conditions", []),
            "n_oracle_queries": len(payload_in.get("queried_initial_conditions", [])) + 1,
        }
        eqs = _equations_from_output(payload_in, data.dim)
        if eqs is None:
            payload["error"] = "no best_equations in output"
        else:
            payload.update(common.evaluate_equations(eqs, data))
            payload["status"] = "ok"
            n_ok += 1
            logger.info(f"  {system}: recon={payload['recon_nmse']:.3e} gen={payload['gen_nmse']:.3e} "
                        f"ood={payload['ood_nmse']:.3e} complexity={payload['complexity']}")

        if args.logs_dir:
            jsonl = Path(args.logs_dir) / system / "active_llm_pysr_results.jsonl"
            if jsonl.exists():
                try:
                    lines = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
                    payload["n_iterations_logged"] = len(lines)
                    payload["final_train_size"] = lines[-1].get("train_size") if lines else None
                    payload["total_wall_time_s"] = round(sum(l.get("wall_time_s", 0) or 0 for l in lines), 2)
                except Exception:
                    pass

        common.save_system_result(result_dir, payload)

    logger.info(f"Done: {n_ok}/{len(outputs)} systems scored into {result_dir}")


if __name__ == "__main__":
    main()
