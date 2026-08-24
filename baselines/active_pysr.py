"""Active symbolic-discovery baselines built on PySR (paper, Appendix B.2).

Two acquisition strategies share one closed loop, so that the only difference
between them is *how the next initial condition is chosen*:

**Bayesian Optimization (BO)**
    "BO combines PySR with a Gaussian process surrogate over initial condition
     space.  Each iteration, the GP predicts the expected value of the NMSE of
     the PySR fit from a candidate IC and selects the next query using Expected
     Improvement (EI) from a pool of 256 uniformly sampled ICs.  After querying
     the oracle, PySR is refit on all accumulated data.  The GP is updated with
     the observed NMSE as the reward."

**Query-by-Committee (QBC)**
    "QBC adapts QBC active learning to ODE initial condition selection, using
     PySR operating on gradient-matched data, where the committee is drawn from
     the Pareto front.  QBC also uses the same oracle as LLM-ACES to query new
     initial conditions."

Budget matches LLM-ACES exactly (Appendix B.3): the run starts from 20 oracle
samples at the dataset's own initial condition (10 train / 10 validation via
interleaved splitting) and performs ``--n_iterations`` acquisition rounds; each
round queries one initial condition on 20 uniformly spaced points in t in [0, 1]
and adds 10 training + 10 validation samples.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from baselines import common, oracle as oracle_lib
from baselines.run_pysr import add_pysr_args, build_regressor

T_QUERY = np.linspace(0.0, 1.0, 20)
BO_POOL = 256


# ---------------------------------------------------------------------------
# PySR fitting that also exposes the Pareto front (needed by QBC)
# ---------------------------------------------------------------------------
def _fit_dimension(u, y, args, seed):
    """Fit PySR for one dimension; return (best_eq_str, pareto_eq_strs)."""
    import sympy as sp

    model = build_regressor(
        niterations=args.pysr_niterations, populations=args.pysr_populations,
        population_size=args.pysr_population_size, ncycles=args.pysr_ncycles,
        maxsize=args.pysr_maxsize, maxdepth=args.pysr_maxdepth,
        procs=args.pysr_procs, seed=seed, timeout_s=args.pysr_timeout,
        parallelism=getattr(args, "pysr_parallelism", "auto"),
    )
    model.fit(u, y)
    eqs = model.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]

    pareto, best_str, best_score = [], None, -np.inf
    for _, row in eqs.iterrows():
        try:
            s = str(sp.sympify(row["sympy_format"]))
        except Exception:
            continue
        pareto.append(s)
        try:
            with np.errstate(all="ignore"):
                pred = np.asarray(row["lambda_format"](u), dtype=float).reshape(-1)
            score = common.fitness(common.nmse(y, pred), int(row["complexity"]))
        except Exception:
            continue
        if score > best_score:
            best_score, best_str = score, s
    if best_str is None and pareto:
        best_str = pareto[-1]
    return best_str, pareto


def _dim_val_nmse(eq: str, dim: int, u: np.ndarray, y: np.ndarray) -> float:
    """Validation NMSE of a single dimension's RHS, evaluated independently."""
    f = common.equations_to_callable([eq], dim)
    if f is None:
        return float("inf")
    with np.errstate(all="ignore"):
        return common.nmse(y, f(u)[:, 0])


def _rollout(f, u0: np.ndarray, n_steps: int = 10, dt: float = 0.1) -> np.ndarray | None:
    """Short forward-Euler rollout, as used for LLM-ACES's divergence score."""
    u = np.asarray(u0, dtype=float).reshape(-1).copy()
    traj = []
    with np.errstate(all="ignore"):
        for _ in range(n_steps):
            traj.append(u.copy())
            du = f(u[None, :])[0]
            if not np.all(np.isfinite(du)):
                return None
            u = u + dt * du
            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e12:
                return None
    return np.concatenate(traj)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------
def qbc_scores(committee, candidates: np.ndarray) -> np.ndarray:
    """Mean pairwise rollout disagreement of the Pareto-front committee."""
    scores = np.zeros(len(candidates))
    for m, u0 in enumerate(candidates):
        trajs = [t for t in (_rollout(f, u0) for f in committee) if t is not None]
        if len(trajs) < 2:
            continue
        norm = np.linalg.norm(np.mean(trajs, axis=0)) + 1e-8
        z = [t / norm for t in trajs]
        pair = [float(np.mean((z[i] - z[j]) ** 2))
                for i in range(len(z)) for j in range(i + 1, len(z))]
        scores[m] = float(np.mean(pair)) if pair else 0.0
    return scores


def _expected_improvement(mu, sigma, best_y, xi=0.01):
    from scipy.stats import norm as _norm

    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best_y - xi
    z = imp / sigma
    return imp * _norm.cdf(z) + sigma * _norm.pdf(z)


def bo_select(observed_x, observed_y, candidates_unit, rng: np.random.Generator) -> int:
    """Expected-Improvement pick over the candidate pool; returns pool index."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    if len(observed_y) < 2 or len(np.unique(observed_y)) <= 1:
        return int(rng.integers(0, len(candidates_unit)))
    dim = candidates_unit.shape[1]
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(dim), nu=2.5)
              + WhiteKernel(noise_level=1e-6))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    gp.fit(np.asarray(observed_x), np.asarray(observed_y))
    mu, sigma = gp.predict(candidates_unit, return_std=True)
    ei = _expected_improvement(mu, sigma, float(np.max(observed_y)))
    return int(np.argmax(ei))


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
def run_active(data: common.Dataset, args, logger, acq: str) -> dict:
    orc = oracle_lib.get_oracle(data.name)
    if orc is None:
        raise RuntimeError(f"no ground-truth oracle for {data.name}; cannot run active acquisition")
    dim = data.dim
    rng = np.random.default_rng(args.seed)
    bounds = orc.ic_bounds
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)

    # --- initial data: 20 oracle samples at the dataset IC, interleaved 10/10
    init = oracle_lib.query_oracle(orc.rhs, data.u0, T_QUERY)
    if init is None:
        raise RuntimeError("oracle failed at the dataset initial condition")
    idx = np.arange(len(T_QUERY))
    train = {"u": init["u"][idx[0::2]], "du": init["du"][idx[0::2]]}
    val = {"u": init["u"][idx[1::2]], "du": init["du"][idx[1::2]]}

    queried_ics: list[list[float]] = []
    bo_x: list[np.ndarray] = []
    bo_y: list[float] = []
    best_eq = [None] * dim
    best_val = [np.inf] * dim
    history = []

    for it in range(args.n_iterations):
        t0 = time.time()
        best_this, pareto_this = [], []
        for d in range(dim):
            # One PySR fit at the paper's budget takes minutes, and there are
            # n_iterations x dim of them per system, so say what is happening
            # instead of leaving the log silent for an hour.
            logger.info(f"      iter {it + 1}/{args.n_iterations} fitting dx{d}/dt "
                        f"(n_train={len(train['u'])}) ...")
            td = time.time()
            eq, pareto = _fit_dimension(train["u"], train["du"][:, d], args,
                                        seed=int(rng.integers(0, 2 ** 31)))
            logger.info(f"      iter {it + 1}/{args.n_iterations} dx{d}/dt done in "
                        f"{time.time() - td:.1f}s -> {eq}")
            best_this.append(eq)
            pareto_this.append(pareto)
            if eq is None:
                continue
            v = _dim_val_nmse(eq, dim, val["u"], val["du"][:, d])
            if v < best_val[d]:
                best_val[d], best_eq[d] = v, eq

        logger.info(f"    iter {it + 1}/{args.n_iterations}  n_train={len(train['u'])}  "
                    f"({time.time() - t0:.1f}s)  eqs={best_this}")

        if it == args.n_iterations - 1:
            history.append({"iteration": it + 1, "n_train": int(len(train["u"])),
                            "equations": best_this, "wall_time_s": round(time.time() - t0, 2)})
            break

        # --- acquisition -------------------------------------------------
        candidates = rng.uniform(lows, highs, size=(BO_POOL, dim))
        cand_unit = (candidates - lows) / (highs - lows + 1e-12)

        if acq == "qbc":
            # committee = Pareto-front members, combined across dimensions
            committee = []
            n_committee = min(args.committee_size, min(len(p) for p in pareto_this) if all(pareto_this) else 0)
            for c in range(n_committee):
                eqs_c = [pareto_this[d][-(c + 1)] for d in range(dim)]
                f = common.equations_to_callable(eqs_c, dim)
                if f is not None:
                    committee.append(f)
            if len(committee) < 2:
                chosen = int(rng.integers(0, BO_POOL))
                score = float("nan")
            else:
                s = qbc_scores(committee, candidates)
                chosen = int(np.argmax(s))
                score = float(s[chosen])
        else:  # bo
            chosen = bo_select(bo_x, bo_y, cand_unit, rng) if bo_x else int(rng.integers(0, BO_POOL))
            score = float("nan")

        u0_next = candidates[chosen]
        acquired = oracle_lib.query_oracle(orc.rhs, u0_next, T_QUERY)
        if acquired is None:
            logger.info(f"    iter {it + 1}: oracle failed at {np.round(u0_next, 3).tolist()}, resampling")
            continue
        queried_ics.append([float(v) for v in u0_next])

        # GP reward = NMSE of the current fit on the freshly acquired trajectory
        if acq == "bo":
            f_now = common.equations_to_callable([e for e in best_this if e is not None], dim) \
                if all(e is not None for e in best_this) else None
            reward = 1.0
            if f_now is not None:
                with np.errstate(all="ignore"):
                    reward = common.nmse(acquired["du"], f_now(acquired["u"]))
            # NMSE spans many orders of magnitude; the GP models log10(NMSE) so
            # the Matern kernel sees a well-scaled target. EI then maximises the
            # predicted error, i.e. it queries where the current fit is worst.
            bo_x.append(cand_unit[chosen])
            bo_y.append(float(np.log10(max(reward, 1e-30))))
            score = float(reward)

        aidx = np.arange(len(T_QUERY))
        train = {"u": np.concatenate([train["u"], acquired["u"][aidx[0::2]]]),
                 "du": np.concatenate([train["du"], acquired["du"][aidx[0::2]]])}
        val = {"u": np.concatenate([val["u"], acquired["u"][aidx[1::2]]]),
               "du": np.concatenate([val["du"], acquired["du"][aidx[1::2]]])}

        logger.info(f"    iter {it + 1}: queried IC={np.round(u0_next, 3).tolist()} "
                    f"acq_score={score:.4e} -> n_train={len(train['u'])}")
        history.append({"iteration": it + 1, "n_train": int(len(train["u"])),
                        "equations": best_this, "queried_ic": [float(v) for v in u0_next],
                        "acq_score": score, "wall_time_s": round(time.time() - t0, 2)})

    equations = [best_eq[d] if best_eq[d] is not None else (best_this[d] or "0") for d in range(dim)]
    return {
        "equations": equations,
        "queried_initial_conditions": queried_ics,
        "n_oracle_queries": len(queried_ics) + 1,
        "final_train_size": int(len(train["u"])),
        "per_dim_val_nmse": [float(v) for v in best_val],
        "iterations": history,
    }


def build_parser(method: str):
    parser = argparse.ArgumentParser(description=f"{method.upper()} active symbolic discovery baseline")
    common.add_common_args(parser)
    add_pysr_args(parser)
    parser.set_defaults(pysr_niterations=20, pysr_populations=15)
    parser.add_argument("--n_iterations", type=int, default=10,
                        help="Acquisition rounds (same budget as LLM-ACES).")
    parser.add_argument("--committee_size", type=int, default=5,
                        help="QBC committee members drawn from the PySR Pareto front.")
    return parser


def main(acq: str) -> None:
    parser = build_parser(acq)
    args = parser.parse_args()
    common.set_thread_env(1)
    common.silence_numeric_warnings()

    paths = common.resolve_data_paths(args)
    result_dir = common.make_result_dir(args.results_root, args.benchmark, args.method_name or acq)
    logger = common.setup_logger(result_dir, acq)
    logger.info(f"{acq.upper()} on {args.benchmark}: {len(paths)} systems -> {result_dir}")
    logger.info(f"  n_iterations={args.n_iterations}  PySR(niter={args.pysr_niterations}, "
                f"pop={args.pysr_populations})")

    common.run_over_datasets(
        acq, args.benchmark, paths, result_dir,
        lambda d, log: run_active(d, args, log, acq),
        logger, resume=not args.no_resume,
    )
    logger.info(f"{acq.upper()} done.")
