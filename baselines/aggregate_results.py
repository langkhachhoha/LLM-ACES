"""Aggregate per-system results into the paper's Table 2 / Table 3.

Reports, per method: median reconstruction / generalization / OOD NMSE, mean
expression complexity, and mean symbolic accuracy (if
``symbolic_accuracy.json`` has been produced by ``baselines.symbolic_accuracy``).

    python -m baselines.aggregate_results --benchmark odebench
    python -m baselines.aggregate_results --benchmark odebase --metric traj
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Paper ordering of Tables 2 and 3.
GROUPS = [
    ("Passive Symbolic Discovery", ["sindy", "esindy", "operon", "pysr", "e2e", "odeformer"]),
    ("LLM-guided Symbolic Discovery", ["llm_only", "llm_ode"]),
    ("Active Symbolic Discovery", ["qbc", "bo", "apps_ode", "llm_aces"]),
]
GT_COMPLEXITY = {"odebench": 19.3, "odebase": 35.2}


def load_method(result_dir: Path) -> list[dict]:
    rows = []
    for p in sorted((result_dir / "systems").glob("*.json")):
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    return rows


def summarize(result_dir: Path, metric: str = "deriv") -> dict | None:
    rows = load_method(result_dir)
    if not rows:
        return None
    ok = [r for r in rows if r.get("status") == "ok"]
    suffix = "_traj_nmse" if metric == "traj" else "_nmse"

    def col(key):
        vals = []
        for r in rows:  # failures count as worst-case, they are still attempts
            v = r.get(f"{key}{suffix}")
            vals.append(float(v) if v is not None and np.isfinite(float(v)) else 1e10)
        return np.asarray(vals, dtype=float)

    complexities = [r.get("complexity") for r in ok if r.get("complexity")]
    sym_path = result_dir / "symbolic_accuracy.json"
    sym_acc = None
    if sym_path.exists():
        try:
            with open(sym_path, encoding="utf-8") as f:
                sym_acc = json.load(f).get("symbolic_accuracy")
        except Exception:
            sym_acc = None

    return {
        "method": result_dir.name,
        "n_systems": len(rows),
        "n_ok": len(ok),
        "recon": float(np.median(col("recon"))),
        "gen": float(np.median(col("gen"))),
        "ood": float(np.median(col("ood"))),
        "recon_q1": float(np.percentile(col("recon"), 25)),
        "recon_q3": float(np.percentile(col("recon"), 75)),
        "complexity": float(np.mean(complexities)) if complexities else float("nan"),
        "symbolic_accuracy": (sym_acc * 100.0) if sym_acc is not None else None,
        "mean_train_time_s": float(np.mean([r.get("train_time_s", 0) or 0 for r in rows])),
    }


def _order(methods: list[str]) -> list[str]:
    ordered, seen = [], set()
    for _, group in GROUPS:
        for prefix in group:
            for m in sorted(methods):
                if (m == prefix or m.startswith(prefix + "_")) and m not in seen:
                    ordered.append(m)
                    seen.add(m)
    ordered += [m for m in sorted(methods) if m not in seen]
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results into the paper's tables")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--benchmark", choices=["odebench", "odebase"], required=True)
    parser.add_argument("--metric", choices=["deriv", "traj"], default="deriv",
                        help="deriv: vector-field NMSE (default). traj: integrated-trajectory NMSE.")
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.results_root) / args.benchmark
    if not root.is_dir():
        raise SystemExit(f"{root} does not exist")
    methods = [p.name for p in sorted(root.iterdir()) if (p / "systems").is_dir()]
    if not methods:
        raise SystemExit(f"no method folders with systems/ under {root}")

    summaries = []
    for m in _order(methods):
        s = summarize(root / m, args.metric)
        if s:
            summaries.append(s)

    header = (f"{'Method':<22}{'Recon NMSE':>13}{'Gen NMSE':>13}{'OOD NMSE':>13}"
              f"{'Complexity':>12}{'Sym.Acc(%)':>12}{'n':>6}")
    print(f"\n=== {args.benchmark.upper()} ({'trajectory' if args.metric == 'traj' else 'vector-field'} NMSE) ===")
    print(f"Ground-truth mean complexity: {GT_COMPLEXITY.get(args.benchmark, float('nan'))}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        sym = f"{s['symbolic_accuracy']:.1f}" if s["symbolic_accuracy"] is not None else "-"
        print(f"{s['method']:<22}{s['recon']:>13.2e}{s['gen']:>13.2e}{s['ood']:>13.2e}"
              f"{s['complexity']:>12.1f}{sym:>12}{s['n_ok']:>4}/{s['n_systems']:<2}")

    out = Path(args.results_root) / f"summary_{args.benchmark}_{args.metric}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nwrote {out}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
