"""Map benchmark NPZ stems to scibench equation ids (needed by APPS-ODE).

APPS-ODE addresses systems through scibench's ``Equation_evaluator``, which
knows them as ``vars1_prog1`` ... ``vars4_prog10`` (Strogatz / ODEBench) and
``odebase_vars2_prog1`` ... (ODEBase).  Our NPZ stems use human-readable names
for ODEBench, so this script matches each ODEBench system to its scibench twin
by evaluating both vector fields on random states and comparing numerically.

Writes ``baselines/scibench_map.json``:

    {"rc-circuit": "vars1_prog1", ..., "odebase_vars2_prog1": "odebase_vars2_prog1"}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from baselines import oracle as oracle_lib  # noqa: E402

N_PROBE = 64
RTOL = 1e-6
ATOL = 1e-8


def scibench_systems() -> dict:
    """All scibench equations, keyed by ``_eq_name``."""
    import importlib

    importlib.import_module("scibench.data.equation_odes_odebase")
    importlib.import_module("scibench.data.equation_odes_strogatz")
    from scibench.data.base import EQUATION_CLASS_DICT

    return {cls._eq_name: cls for cls in EQUATION_CLASS_DICT.values()}


def probe(rhs, points: np.ndarray) -> np.ndarray | None:
    out = []
    with np.errstate(all="ignore"):
        for p in points:
            try:
                v = np.asarray(rhs(0.0, p), dtype=float).reshape(-1)
            except Exception:
                return None
            out.append(v)
    arr = np.asarray(out)
    return arr if np.all(np.isfinite(arr)) else None


def main() -> None:
    data_ode = _ROOT / "data" / "ode"
    data_odebase = _ROOT / "data" / "odebase"
    sci = scibench_systems()
    rng = np.random.default_rng(0)

    mapping: dict[str, str] = {}
    unmatched: list[str] = []

    # ODEBase: identity mapping (same generator, same names)
    if data_odebase.is_dir():
        for p in sorted(data_odebase.glob("*/*.npz")):
            if "_snr_" in p.name:
                continue
            if p.stem in sci:
                mapping[p.stem] = p.stem
            else:
                unmatched.append(p.stem)

    # ODEBench: numerical vector-field matching against the strogatz scibench set
    strogatz_ids = [k for k in sci if k.startswith("vars")]
    if data_ode.is_dir():
        for p in sorted(data_ode.glob("*/*.npz")):
            if "_snr_" in p.name:
                continue
            orc = oracle_lib.get_oracle(p.stem)
            if orc is None:
                unmatched.append(p.stem)
                continue
            pts = rng.uniform(0.3, 2.0, size=(N_PROBE, orc.dim))
            ref = probe(orc.rhs, pts)
            if ref is None:
                unmatched.append(p.stem)
                continue
            hit = None
            for eq_name in strogatz_ids:
                cls = sci[eq_name]
                try:
                    eq = cls()
                except Exception:
                    continue
                if eq.num_vars != orc.dim:
                    continue
                cand = probe(lambda t, x, _e=eq: _e.np_eq(t, x), pts)
                if cand is None:
                    continue
                if np.allclose(ref, cand, rtol=RTOL, atol=ATOL):
                    hit = eq_name
                    break
            if hit is None:
                # Second pass: scibench sometimes uses different parameter values
                # for the same textbook system, so fall back to its description.
                target = orc.description.strip().lower()
                for eq_name in strogatz_ids:
                    cls = sci[eq_name]
                    if str(getattr(cls, "_description", "")).strip().lower() == target and target:
                        try:
                            if cls().num_vars == orc.dim:
                                hit = eq_name
                                break
                        except Exception:
                            continue
            if hit:
                mapping[p.stem] = hit
            else:
                unmatched.append(p.stem)

    out = _ROOT / "baselines" / "scibench_map.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=2)

    print(f"Wrote {out} with {len(mapping)} mappings.")
    if unmatched:
        print(f"{len(unmatched)} systems have no scibench twin "
              f"(APPS-ODE will skip them): {sorted(unmatched)}")


if __name__ == "__main__":
    main()
