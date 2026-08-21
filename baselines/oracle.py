"""Ground-truth ODE oracle Omega for both benchmarks.

The paper's active baselines (LLM-ACES, BO, QBC, APPS-ODE) all query the same
oracle: a SciPy ``solve_ivp`` integration of the *true* system from a chosen
initial condition (Section 2.4).  This module exposes one lookup that works for

* **ODEBench**  -- the 63 Strogatz systems in ``scripts/strogatz_ode.py``
* **ODEBase**   -- the 60 biological systems in
  ``scripts/scibench/data/equation_odes_odebase.py``

and returns, for a given NPZ stem:

* ``rhs(t, x) -> np.ndarray``      the oracle
* ``gt_equations``                 ground-truth RHS strings (for symbolic accuracy)
* ``ic_bounds``                    per-dimension feasible IC box U
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Integration settings identical to generate_ode.py / generate_odebase.py so
# that oracle-queried trajectories are drawn from the same distribution as the
# pre-generated benchmark data.
SOLVER_KWARGS = dict(method="LSODA", rtol=1e-5, atol=1e-7, first_step=1e-6, min_step=1e-10)


@dataclass
class Oracle:
    name: str
    dim: int
    benchmark: str                      # "odebench" | "odebase"
    rhs: Callable                       # rhs(t, x) -> (dim,)
    gt_equations: list[str]             # in terms of x0, x1, ...
    ic_bounds: list[list[float]] = field(default_factory=list)
    description: str = ""


def _slug(s: str) -> str:
    return s.lower().replace(" ", "-").replace("_", "-")


# ---------------------------------------------------------------------------
# ODEBench (Strogatz)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _strogatz_index() -> dict:
    import strogatz_ode  # noqa: E402  (vendored in scripts/)

    return {_slug(e["name"]): e for e in strogatz_ode.equations}


@lru_cache(maxsize=1)
def _odebench_ic_bounds() -> dict:
    path = _ROOT / "llm-aces" / "ic_bounds.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_odebench(system_name: str) -> Oracle | None:
    import sympy as sp

    entry = _strogatz_index().get(_slug(system_name))
    if entry is None:
        return None

    dim = entry["dim"]
    consts = entry["consts"][0]
    var_symbols = sp.symbols([f"x_{i}" for i in range(dim)])
    const_symbols = sp.symbols([f"c_{i}" for i in range(len(consts))])
    const_subs = dict(zip(const_symbols, consts))

    exprs, lambdas = [], []
    for eq_str in entry["eq"].split("|"):
        expr = sp.sympify(eq_str).subs(const_subs)
        exprs.append(expr)
        lambdas.append(sp.lambdify(var_symbols, expr, "numpy"))

    def rhs(t, x):
        return np.array([float(f(*x)) for f in lambdas])

    gt = [str(e).replace("x_", "x") for e in exprs]
    bounds = _odebench_ic_bounds().get(_slug(system_name))
    if bounds is None:
        # Fall back to a box around the two reference ICs.
        inits = np.asarray(entry["init"], dtype=float)
        lo, hi = inits.min(axis=0), inits.max(axis=0)
        span = np.maximum(np.abs(hi - lo), np.maximum(np.abs(hi), 1.0))
        bounds = [[float(l - 0.5 * s), float(h + 0.5 * s)] for l, h, s in zip(lo, hi, span)]

    return Oracle(
        name=system_name,
        dim=dim,
        benchmark="odebench",
        rhs=rhs,
        gt_equations=gt,
        ic_bounds=bounds,
        description=entry.get("eq_description", ""),
    )


# ---------------------------------------------------------------------------
# ODEBase (scibench)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _odebase_index() -> dict:
    import importlib

    importlib.import_module("scibench.data.equation_odes_odebase")  # registers the classes
    from scibench.data.base import EQUATION_CLASS_DICT

    return {cls._eq_name: cls
            for cls in EQUATION_CLASS_DICT.values()
            if str(getattr(cls, "_eq_name", "")).startswith("odebase_")}


def _load_odebase(system_name: str) -> Oracle | None:
    cls = _odebase_index().get(system_name)
    if cls is None:
        return None
    eq = cls()
    dim = eq.num_vars

    def rhs(t, x, _eq=eq):
        return np.asarray(_eq.np_eq(t, np.asarray(x, dtype=float)), dtype=float).reshape(-1)

    # sympy_eq strings use x[0], x[1], ... -> convert to x0, x1, ...
    gt = []
    for s in eq.sympy_eq:
        out = s
        for i in range(dim):
            out = out.replace(f"x[{i}]", f"x{i}")
        gt.append(out)

    bounds = []
    for sampler in eq.vars_range_and_types:
        lo, hi = sampler.range
        bounds.append([float(lo), float(hi)])

    return Oracle(
        name=system_name,
        dim=dim,
        benchmark="odebase",
        rhs=rhs,
        gt_equations=gt,
        ic_bounds=bounds,
        description=getattr(eq, "_description", ""),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_oracle(system_name: str) -> Oracle | None:
    """Look up the ground-truth system for an NPZ stem, or ``None``."""
    system_name = system_name.split("_snr_")[0]
    if system_name.startswith("odebase_"):
        return _load_odebase(system_name)
    return _load_odebench(system_name)


def benchmark_of(system_name: str) -> str:
    return "odebase" if system_name.startswith("odebase_") else "odebench"


def query_oracle(rhs: Callable, u0, t_eval) -> dict | None:
    """Integrate the true system from ``u0`` on ``t_eval``; return t/u/du."""
    from scipy.integrate import solve_ivp

    u0 = np.asarray(u0, dtype=float).reshape(-1)
    t_eval = np.asarray(t_eval, dtype=float)
    try:
        sol = solve_ivp(rhs, [float(t_eval[0]), float(t_eval[-1])], u0.tolist(),
                        t_eval=t_eval, **SOLVER_KWARGS)
        if not sol.success or sol.y.shape[1] != len(t_eval):
            return None
        u = sol.y.T
        if not np.all(np.isfinite(u)):
            return None
        du = np.stack([np.asarray(rhs(float(t_eval[i]), u[i]), dtype=float) for i in range(len(t_eval))])
        if not np.all(np.isfinite(du)):
            return None
        return {"t": t_eval.copy(), "u": u, "du": du}
    except Exception:
        return None


def sample_ics(bounds: list[list[float]], n: int, rng: np.random.Generator) -> np.ndarray:
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    return rng.uniform(lows, highs, size=(n, len(bounds)))
