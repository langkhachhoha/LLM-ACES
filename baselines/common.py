"""Shared data loading, metrics and result IO for every baseline.

Evaluation protocol (paper, Section 3.1):

* **Reconstruction** -- training initial condition, ``t in [0, 1]``, 100 samples.
* **Generalization** -- held-out initial condition, ``t in [0, 1]``, 100 samples.
* **Out-of-distribution** -- training initial condition, ``t in (1, 10]``, 150 samples.

The generated NPZ stores state *and* true derivative for all three windows
(``u``/``du``, ``u_gen``/``du_gen``, ``u_ood``/``du_ood``), so we report both

* ``*_nmse``       -- vector-field (derivative) NMSE, the quantity every method
  in this code base is actually optimising, and
* ``*_traj_nmse``  -- trajectory NMSE obtained by integrating the discovered
  system from the window's initial condition (the ODEFormer-style reading of
  "trajectory-level").

Both are stored for every method so the aggregation step can produce the
paper's tables under either reading.  NMSE follows MDBench:
``sum((y - yhat)^2) / (sum(y^2) + 1e-10)``.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

EPS = 1e-10
# Diverged trajectories produce astronomically large NMSE; cap so medians and
# means stay meaningful (only ever *increases* a method's reported error).
NMSE_CAP = 1e10
# A discovered system that diverges can keep LSODA busy indefinitely; cap both
# the state magnitude and the wall-clock of every integration.
BLOWUP_THRESHOLD = 1e12
INTEGRATE_TIMEOUT_S = 20.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.shape != y_true.shape:
        return float(NMSE_CAP)
    if not np.all(np.isfinite(y_pred)):
        return float(NMSE_CAP)
    val = float(np.sum((y_true - y_pred) ** 2) / (np.sum(y_true ** 2) + EPS))
    if not np.isfinite(val):
        return float(NMSE_CAP)
    return min(val, NMSE_CAP)


def fitness(nmse_value: float, complexity: int, lam: float = 1.0, L: float = 200.0) -> float:
    """MDBench's complexity-aware fitness (higher is better)."""
    return 1.0 / (1.0 + nmse_value) + lam * float(np.exp(-complexity / L))


# ---------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------
def sympify_equation(eq_str: str, dim: int):
    """Parse a RHS string in terms of x0..x{dim-1} into a sympy expression."""
    import sympy as sp

    if eq_str is None:
        return None
    s = str(eq_str).strip()
    if not s or s.lower() in {"nan", "none", "(unknown)"}:
        return None
    # normalise common variable spellings to x0, x1, ...
    for i in range(max(dim, 10) - 1, -1, -1):
        s = s.replace(f"x_{{{i}}}", f"x{i}").replace(f"x_{i}", f"x{i}")
        s = s.replace(f"u{i}", f"x{i}").replace(f"X{i}", f"x{i}")
        s = s.replace(f"x[{i}]", f"x{i}")
    s = s.replace("^", "**")
    local = {f"x{i}": sp.Symbol(f"x{i}", real=True) for i in range(dim)}
    local["t"] = sp.Symbol("t", real=True)
    try:
        return sp.sympify(s, locals=local)
    except Exception:
        return None


def expression_complexity(eq_strings: Sequence[str], dim: int) -> int:
    """Total expression-tree node count over all dimensions (paper, App. A.2)."""
    import sympy as sp

    total = 0
    for eq in eq_strings:
        expr = sympify_equation(eq, dim)
        if expr is None:
            continue
        total += sum(1 for _ in sp.preorder_traversal(expr))
    return int(total)


def equations_to_callable(eq_strings: Sequence[str], n_vars: int):
    """Return ``f(U) -> dU`` for a batch of states, or ``None`` if unparsable.

    ``U`` has ``n_vars`` columns; the output has one column per entry of
    ``eq_strings`` (so a single dimension can be evaluated on its own).
    """
    import sympy as sp

    syms = [sp.Symbol(f"x{i}", real=True) for i in range(n_vars)]
    fns = []
    for eq in eq_strings:
        expr = sympify_equation(eq, n_vars)
        if expr is None:
            return None
        try:
            fns.append(sp.lambdify(syms, expr, "numpy"))
        except Exception:
            return None
    if not fns:
        return None

    def f(U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(np.asarray(U, dtype=float))
        cols = []
        with np.errstate(all="ignore"):
            for fn in fns:
                out = fn(*[U[:, i] for i in range(n_vars)])
                out = np.broadcast_to(np.asarray(out, dtype=float), (U.shape[0],))
                cols.append(out)
        return np.stack(cols, axis=1)

    return f


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Dataset:
    """One benchmark system loaded from its NPZ file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.stem
        raw = np.load(self.path)
        self.t = np.asarray(raw["t"], dtype=float).ravel()
        self.u = np.atleast_2d(np.asarray(raw["u"], dtype=float))
        self.du = np.atleast_2d(np.asarray(raw["du"], dtype=float))
        if self.u.shape[0] == 1 and self.u.shape[1] == len(self.t):
            self.u, self.du = self.u.T, self.du.T
        if self.u.ndim == 1:
            self.u = self.u[:, None]
            self.du = self.du[:, None]
        self.u_gen = np.asarray(raw["u_gen"], dtype=float).reshape(len(self.t), -1)
        self.du_gen = np.asarray(raw["du_gen"], dtype=float).reshape(len(self.t), -1)
        self.t_ood = np.asarray(raw["t_ood"], dtype=float).ravel()
        self.u_ood = np.asarray(raw["u_ood"], dtype=float).reshape(len(self.t_ood), -1)
        self.du_ood = np.asarray(raw["du_ood"], dtype=float).reshape(len(self.t_ood), -1)
        self.dim = self.u.shape[1]

    # -- convenience ------------------------------------------------------
    @property
    def u0(self) -> np.ndarray:
        return self.u[0]

    @property
    def u0_gen(self) -> np.ndarray:
        return self.u_gen[0]

    def train_split(self, val_ratio: float = 0.2):
        """Chronological train/val split of the reconstruction window."""
        n_val = max(int(len(self.t) * val_ratio), 1)
        n_tr = len(self.t) - n_val
        return (
            dict(t=self.t[:n_tr], u=self.u[:n_tr], du=self.du[:n_tr]),
            dict(t=self.t[n_tr:], u=self.u[n_tr:], du=self.du[n_tr:]),
        )


def discover_datasets(data_root: str | Path, benchmark: str, include_noisy: bool = False,
                      limit: int | None = None) -> list[Path]:
    """All clean NPZ files for a benchmark, sorted deterministically."""
    root = Path(data_root)
    paths = sorted(root.rglob("*.npz"))
    if not include_noisy:
        paths = [p for p in paths if "_snr_" not in p.name]
    if benchmark == "odebase":
        paths = [p for p in paths if p.stem.startswith("odebase_vars2_") or p.stem.startswith("odebase_vars3_")]
    if limit:
        paths = paths[:limit]
    return paths


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class _Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """Hard wall-clock cap for a block (Unix main thread only; else a no-op).

    Discovered equations are frequently stiff or divergent, and LSODA can spend
    unbounded time on them; without this a single pathological system stalls a
    whole benchmark run.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _Timeout()

    try:
        previous = signal.signal(signal.SIGALRM, _handler)
    except ValueError:      # not the main thread
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _integrate(f: Callable, u0: np.ndarray, t_eval: np.ndarray,
               timeout_s: float = INTEGRATE_TIMEOUT_S) -> np.ndarray | None:
    from scipy.integrate import solve_ivp

    def rhs(t, x):
        return f(np.asarray(x, dtype=float)[None, :])[0]

    def blowup(t, x):
        return BLOWUP_THRESHOLD - float(np.max(np.abs(x)))

    blowup.terminal = True
    blowup.direction = -1

    try:
        with time_limit(timeout_s), np.errstate(all="ignore"):
            sol = solve_ivp(rhs, [float(t_eval[0]), float(t_eval[-1])], np.asarray(u0, dtype=float).tolist(),
                            t_eval=t_eval, method="LSODA", rtol=1e-5, atol=1e-7, events=blowup)
        if not sol.success or sol.y.shape[1] != len(t_eval):
            return None
        return sol.y.T
    except Exception:   # includes _Timeout
        return None


def evaluate_equations(eq_strings: Sequence[str], data: Dataset) -> dict:
    """Compute all five reported quantities for a discovered ODE system."""
    dim = data.dim
    result = {
        "equations": [str(e) for e in eq_strings],
        "recon_nmse": float(NMSE_CAP), "gen_nmse": float(NMSE_CAP), "ood_nmse": float(NMSE_CAP),
        "recon_traj_nmse": float(NMSE_CAP), "gen_traj_nmse": float(NMSE_CAP), "ood_traj_nmse": float(NMSE_CAP),
        "complexity": 0, "parse_ok": False,
    }
    if len(eq_strings) != dim:
        return result
    f = equations_to_callable(eq_strings, dim)
    if f is None:
        return result
    result["parse_ok"] = True
    result["complexity"] = expression_complexity(eq_strings, dim)

    # derivative-level NMSE on each window
    with np.errstate(all="ignore"):
        result["recon_nmse"] = nmse(data.du, f(data.u))
        result["gen_nmse"] = nmse(data.du_gen, f(data.u_gen))
        result["ood_nmse"] = nmse(data.du_ood, f(data.u_ood))

    # trajectory-level NMSE (integrate the discovered system)
    traj = _integrate(f, data.u0, data.t)
    if traj is not None:
        result["recon_traj_nmse"] = nmse(data.u, traj)
    traj_gen = _integrate(f, data.u0_gen, data.t)
    if traj_gen is not None:
        result["gen_traj_nmse"] = nmse(data.u_gen, traj_gen)
    t_full = np.concatenate([data.t, data.t_ood])
    traj_full = _integrate(f, data.u0, t_full)
    if traj_full is not None:
        result["ood_traj_nmse"] = nmse(data.u_ood, traj_full[len(data.t):])
    return result


def evaluate_predictor(predict_fn: Callable, data: Dataset) -> dict:
    """Fallback evaluation for methods without a parsable symbolic form."""
    out = {
        "recon_nmse": float(NMSE_CAP), "gen_nmse": float(NMSE_CAP), "ood_nmse": float(NMSE_CAP),
        "recon_traj_nmse": float(NMSE_CAP), "gen_traj_nmse": float(NMSE_CAP), "ood_traj_nmse": float(NMSE_CAP),
    }
    try:
        with np.errstate(all="ignore"):
            out["recon_nmse"] = nmse(data.du, predict_fn(data.u))
            out["gen_nmse"] = nmse(data.du_gen, predict_fn(data.u_gen))
            out["ood_nmse"] = nmse(data.du_ood, predict_fn(data.u_ood))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Result IO / logging
# ---------------------------------------------------------------------------
def make_result_dir(results_root: str | Path, benchmark: str, method: str) -> Path:
    d = Path(results_root) / benchmark / method
    (d / "systems").mkdir(parents=True, exist_ok=True)
    return d


def setup_logger(result_dir: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(f"{name}:{result_dir}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(result_dir / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def system_result_path(result_dir: Path, system: str) -> Path:
    return result_dir / "systems" / f"{system}.json"


def already_done(result_dir: Path, system: str) -> bool:
    p = system_result_path(result_dir, system)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("status") == "ok"
    except Exception:
        return False


def save_system_result(result_dir: Path, payload: dict) -> Path:
    p = system_result_path(result_dir, payload["system"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(result_dir / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return p


def run_over_datasets(
    method: str,
    benchmark: str,
    data_paths: Sequence[Path],
    result_dir: Path,
    fit_fn: Callable[[Dataset, logging.Logger], dict],
    logger: logging.Logger,
    resume: bool = True,
    extra_meta: dict | None = None,
) -> None:
    """Drive ``fit_fn`` over every system, saving one JSON per system.

    ``fit_fn(dataset, logger)`` must return a dict with at least ``equations``
    (list of RHS strings in terms of x0..xd-1); anything else it returns is
    merged into the stored payload (e.g. ``n_oracle_queries``).
    """
    for i, path in enumerate(data_paths, 1):
        system = path.stem
        if resume and already_done(result_dir, system):
            logger.info(f"[{i}/{len(data_paths)}] SKIP {system} (already done)")
            continue
        logger.info(f"[{i}/{len(data_paths)}] RUN  {system}")
        t0 = time.time()
        payload = {
            "system": system, "benchmark": benchmark, "method": method,
            "data_path": str(path), "status": "error", "error": "",
        }
        if extra_meta:
            payload.update(extra_meta)
        try:
            data = Dataset(path)
            payload["dim"] = data.dim
            out = fit_fn(data, logger)
            eqs = out.pop("equations", None)
            payload.update(out)
            if eqs is None:
                raise RuntimeError("fit_fn returned no equations")
            payload.update(evaluate_equations(eqs, data))
            payload["status"] = "ok"
            logger.info(
                f"    {system}: recon={payload['recon_nmse']:.3e} "
                f"gen={payload['gen_nmse']:.3e} ood={payload['ood_nmse']:.3e} "
                f"complexity={payload['complexity']}"
            )
            for d, eq in enumerate(payload["equations"]):
                logger.info(f"      dx{d}/dt = {eq}")
        except Exception as exc:  # keep going on failures, like MDBench does
            payload["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception(f"    {system} FAILED: {payload['error']}")
        payload["train_time_s"] = round(time.time() - t0, 2)
        save_system_result(result_dir, payload)


def add_common_args(parser):
    parser.add_argument("--benchmark", choices=["odebench", "odebase"], required=True)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Defaults to data/ode (odebench) or data/odebase (odebase).")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--method_name", type=str, default=None,
                        help="Sub-folder name under results/<benchmark>/. Defaults to the method id.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N systems (smoke tests).")
    parser.add_argument("--systems", type=str, nargs="*", default=None, help="Explicit system stems to run.")
    parser.add_argument("--include_noisy", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_data_paths(args) -> list[Path]:
    default_root = "data/ode" if args.benchmark == "odebench" else "data/odebase"
    root = Path(args.data_root or default_root)
    paths = discover_datasets(root, args.benchmark, include_noisy=args.include_noisy)
    if args.systems:
        wanted = set(args.systems)
        paths = [p for p in paths if p.stem in wanted]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No NPZ datasets found under {root}. Run the data generation step first.")
    return paths


def silence_numeric_warnings() -> None:
    """Mute the numerical noise third-party search loops make while scoring
    candidate equations.

    Upstream evolutionary searches lambdify every hypothesis and call it without
    an ``np.errstate`` guard, so each ``log`` of a negative number or overflowing
    ``exp`` in a *bad* candidate prints a RuntimeWarning -- hundreds of lines per
    system, all of them expected (a candidate that produces NaNs is supposed to
    score badly and be discarded). Set ``NUMERIC_WARNINGS=1`` to get them back.
    """
    if os.environ.get("NUMERIC_WARNINGS"):
        return
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Worker processes: fork inherits the filters above, spawn does not.
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def quiet_third_party_logging(result_dir: Path, filename: str = "third_party.log") -> None:
    """Send third-party ``logging.warning`` chatter to a file instead of stdout.

    Upstream search loops log every failed hypothesis on the *root* logger
    ("Error making random program: Score is NaN or Inf", ...). That is useful
    diagnostic material but it does not belong between our per-system result
    lines, and our own loggers set ``propagate = False`` so redirecting root is
    safe. Set ``NUMERIC_WARNINGS=1`` to keep it on the console as well.
    """
    root = logging.getLogger()
    root.handlers.clear()
    fh = logging.FileHandler(result_dir / filename)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)
    if os.environ.get("NUMERIC_WARNINGS"):
        root.addHandler(logging.StreamHandler(sys.stdout))
    root.setLevel(logging.WARNING)


def set_thread_env(n: int = 1) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
