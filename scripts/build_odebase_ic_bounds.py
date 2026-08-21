"""Generate ``llm-aces/ic_bounds_odebase.json``.

LLM-ACES's active acquisition needs a feasible initial-condition box U per
system.  The repo ships ``llm-aces/ic_bounds.json`` for the 63 ODEBench systems
only; for ODEBase the box is read straight off the scibench sampler ranges each
system was generated from (``vars_range_and_types``), which is exactly the
region ``generate_odebase.py`` draws its initial conditions from.

Without this file, ``active_llm_aces.py`` prints "not found in ic_bounds.json"
and silently skips acquisition on every ODEBase system.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))


def main() -> None:
    importlib.import_module("scibench.data.equation_odes_odebase")
    from scibench.data.base import EQUATION_CLASS_DICT

    bounds: dict[str, list[list[float]]] = {}
    for cls in EQUATION_CLASS_DICT.values():
        name = str(getattr(cls, "_eq_name", ""))
        if not name.startswith("odebase_"):
            continue
        try:
            eq = cls()
        except Exception:
            continue
        bounds[name] = [[float(s.range[0]), float(s.range[1])] for s in eq.vars_range_and_types]

    out = _ROOT / "llm-aces" / "ic_bounds_odebase.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(bounds.items())), f, indent=2)
    print(f"Wrote {out} with {len(bounds)} ODEBase systems.")


if __name__ == "__main__":
    main()
