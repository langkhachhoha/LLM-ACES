# Reproducing LLM-ACES on a server — setup and run guide

This document takes a bare Linux server from "nothing installed" to the full
result tables of **LLM-ACES: Closed-Loop Discovery of Dynamical Systems with
LLM-Guided Adaptive Search** (Tables 2 and 3), on both benchmarks
(**ODEBench**, 63 systems; **ODEBase**, 59 systems) with both LLM backbones.

Everything runs through **OpenRouter**, with the two models

| role in the paper | model id used here |
|---|---|
| GPT-4o-mini | `openai/gpt-4o-mini-2024-07-18` |
| Qwen3 (LLM-ACES second backbone) | `qwen/qwen3-30b-a3b-instruct-2507` |

---

## 0. What gets reproduced

Everything runs in **one conda environment, `llm-aces`**.

| Paper row | Driver | Source |
|---|---|---|
| SINDy | `baselines/run_sindy.py` | PySINDy, MDBench protocol |
| PySR | `baselines/run_pysr.py` | PySR, MDBench protocol |
| Operon | `baselines/run_operon.py` | pyoperon, MDBench protocol |
| ODEFormer | `baselines/run_odeformer.py` | `odeformer` pip package + pretrained ckpt |
| E2E | `baselines/run_e2e.py` | `facebookresearch/symbolicregression` + `model1.pt` |
| LLM-only | `baselines/run_llm_only.py` | NewtonBench protocol (re-implemented) |
| LLM-ODE | `baselines/run_llm_ode.py` | `gryaklab/llm-ode` (upstream core, our transport) |
| APPS-ODE | `baselines/run_apps_ode.py` | `jiangnanhugo/APPS-ODE` (upstream grammar-RL) |
| Query-by-Committee | `baselines/run_qbc.py` | implemented here (PySR + Pareto committee) |
| Bayesian Optimization | `baselines/run_bo.py` | implemented here (PySR + GP/EI over ICs) |
| **LLM-ACES (GPT / Qwen)** | `scripts/run_llm_aces.sh` | this repo |

All of them write into one shared tree:

```
results/
  odebench/
    sindy/        systems/<system>.json   results.jsonl  run.log  symbolic_accuracy.json
    pysr/         ...
    operon/  odeformer/  e2e/  llm_only/  llm_ode/  apps_ode/  qbc/  bo/
    llm_aces_gpt/  llm_aces_qwen/
  odebase/
    ... same layout ...
  summary_odebench_deriv.json
  summary_odebase_deriv.json
```

Each `systems/<system>.json` carries the discovered equations, all six NMSE
figures, expression complexity, wall-clock time and (for active methods) the
queried initial conditions — everything needed to rebuild the paper's tables or
run further analysis later.

---

## 1. Install conda

```bash
cd ~
curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
exec bash          # reload the shell
conda config --set auto_activate_base false
```

## 2. Get the repo and the API key

```bash
cd ~
git clone <your-fork-or-copy-of-this-repo> LLM-ACES
cd LLM-ACES

cat > .env <<'EOF'
OPENAI_API_KEY=sk-or-v1-...............................
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EOF
chmod 600 .env
```

Every LLM path in this repo (LLM-ACES, LLM-only, LLM-ODE, and the symbolic
accuracy judge) reads `.env` automatically — no `export` needed.

## 3. Create the one environment

A single env holds every dependency of every method. Python 3.11 is the version
that all of PySR, PySINDy, pyoperon, ODEFormer/torch and the APPS-ODE stack have
wheels for.

```bash
conda create -y -n llm-aces python=3.11
conda activate llm-aces
pip install --upgrade pip

# core numerics + the shared evaluator
pip install "numpy==1.26.4" "scipy==1.16.0" "sympy==1.11.1" "pandas==2.2.3" \
            "scikit-learn==1.7.0" "setuptools<81" tqdm findiff matplotlib requests

# symbolic regression backends (SINDy / PySR / Operon)
pip install "pysr==1.5.10" "pysindy==2.1.0" "pyoperon==0.6.1"

# transformer baselines (ODEFormer + E2E share this torch)
pip install "torch==2.4.1" "odeformer==0.1.7" "sympytorch==0.1.4"

# APPS-ODE runtime
pip install numba cython click commentjson PyYAML pathos psutil

# PySR needs a Julia backend; this downloads and precompiles it once (~5 min)
python -c "from pysr import PySRRegressor; import numpy as np; \
m=PySRRegressor(niterations=2,populations=2,population_size=10,\
binary_operators=['+','*'],verbosity=0,progress=False,parallelism='serial',\
temp_equation_file=True); m.fit(np.random.rand(20,2), np.random.rand(20)); print('PySR OK')"
```

Verify in one shot:

```bash
python -c "import numpy,scipy,sympy,sklearn,pandas,pysindy,pysr,pyoperon,torch,odeformer,numba,click,commentjson,pathos,psutil; print('env complete')"
```

Two pip warnings are expected and harmless:

* `pysindy 2.1.0 requires numpy>=2.0, but you have numpy 1.26.4` — metadata only;
  PySINDy 2.1.0 runs correctly on NumPy 1.26.4, which torch 2.4.1 and ODEFormer
  require. All 122 SINDy systems were run on this exact combination.
* `odeformer` may pin an older sympy/scipy; the pins above win and work.

If you install anything else into the env later, re-check the numpy version --
`pysindy` and several other packages will happily pull numpy 2 in behind you:

```bash
python -c "import numpy; print(numpy.__version__)"   # expect 1.26.4
pip install "numpy==1.26.4"                          # if it drifted
```

Everything here runs on numpy 2 as well (E2E included, see the notes at the
bottom), but 1.26.4 is the combination all the reported numbers were produced on.

Nothing else needs installing. In particular the APPS-ODE `scibench` and
`grammar` packages are **not** pip-installed — they are pure Python and the
driver puts them on `PYTHONPATH`, which keeps them from shadowing this repo's
own vendored `scripts/scibench`.

## 4. Generate both benchmarks

```bash
conda activate llm-aces
python generate_ode.py                  # ODEBench -> data/ode/      (63 systems)
python generate_odebase.py --no_snr     # ODEBase  -> data/odebase/  (59 systems saved, 1 IC failure)

# Helper tables the baselines need
python scripts/build_odebase_ic_bounds.py   # IC boxes U for ODEBase (LLM-ACES/BO/QBC acquisition)
python scripts/build_scibench_map.py        # NPZ stem -> scibench equation id (APPS-ODE)
```

Both scripts produce the paper's protocol per system: reconstruction window
`t ∈ [0,1]` with 100 samples, a generalization trajectory from a second initial
condition on the same window, and an OOD window `t ∈ (1,10]` with 150 samples.

`generate_odebase.py` reports `59 saved, 1 failed` — `odebase_vars2_prog4` has no
integrable initial condition. That is why the paper evaluates **59** ODEBase
systems out of the 60 defined.

## 5. Fetch the third-party baselines

```bash
bash scripts/setup_third_party.sh            # llm-ode, APPS-ODE, symbolicregression(+ckpt), mdbench
# or one at a time:
# bash scripts/setup_third_party.sh llm-ode
```

This clones into `third_party/`, downloads the E2E checkpoint
(`third_party/symbolicregression/model.pt`, ~700 MB) and applies one compat
patch to LLM-ODE (see the notes at the end). No extra environments are created.

Cache the ODEFormer checkpoint once so the first run does not stall on a
download:

```bash
python -c "from odeformer.model import SymbolicTransformerRegressor as S; S(from_pretrained=True); print('ODEFormer ckpt cached')"
```

## 6. Smoke test before committing the machine

```bash
conda activate llm-aces          # the one env; everything below assumes it
export T=/tmp/aces_smoke

bash run_baseline.sh sindy odebench --systems rc-circuit --no_resume            # ~2 s
RESULTS_ROOT=$T bash run_baseline.sh pysr odebench --systems rc-circuit \
    --pysr_niterations 10 --pysr_populations 8 --no_resume                       # ~30 s, checks Julia
RESULTS_ROOT=$T bash run_baseline.sh llm_only odebench --systems rc-circuit \
    --n_calls 2 --n_candidates 4 --no_resume                                     # checks OpenRouter
RESULTS_ROOT=$T bash run_baseline.sh odeformer odebench --systems rc-circuit --no_resume
RESULTS_ROOT=$T bash run_baseline.sh llm_ode odebench --systems rc-circuit \
    --llm_calls 8 --n_islands 2 --b 4 --no_resume                                # checks the llm-ode patch
RESULTS_ROOT=$T bash run_baseline.sh e2e odebench --systems rc-circuit --no_resume
RESULTS_ROOT=$T bash run_baseline.sh operon odebench --systems rc-circuit --no_resume
```

`rc-circuit` is `dx0/dt = 0.3030 - 0.3608*x0`; every method should print
something close to that. If they all do, the machine is ready.

Then check the tmux launcher itself, since every long run goes through it:

```bash
bash scripts/tmux_run.sh smoketest bash run_baseline.sh sindy odebench --systems rc-circuit --no_resume
tmux attach -t smoketest     # detach with Ctrl-b then d
```

The pane should print the SINDy result, then
`[smoketest] finished ... with exit 0`, and then **stay open at a shell prompt**
— that is the intended behaviour, the session is not torn down. Inside it,
`which python` must point at `.../envs/llm-aces/bin/python`. Close it with
`tmux kill-session -t smoketest` when satisfied.

---

## 7. Running everything (tmux)

See **[RUN_TMUX.md](RUN_TMUX.md)** for the copy-paste tmux command set.

---

## 8. Collecting the results

```bash
# still in the same llm-aces env

# Symbolic accuracy (GPT-4o-mini judge, paper Appendix A.2). Cached per method,
# so re-running is cheap.
python -m baselines.symbolic_accuracy --benchmark odebench --quiet
python -m baselines.symbolic_accuracy --benchmark odebase  --quiet

# Paper Table 2 and Table 3
python -m baselines.aggregate_results --benchmark odebench
python -m baselines.aggregate_results --benchmark odebase

# Same tables under the trajectory-NMSE reading (see "Two NMSE readings" below)
python -m baselines.aggregate_results --benchmark odebench --metric traj
python -m baselines.aggregate_results --benchmark odebase  --metric traj
```

Output looks like:

```
=== ODEBENCH (vector-field NMSE) ===
Ground-truth mean complexity: 19.3
Method                   Recon NMSE     Gen NMSE     OOD NMSE  Complexity  Sym.Acc(%)     n
-------------------------------------------------------------------------------------------
sindy                      5.07e-04     5.09e-01     1.41e+00        11.7        18.5  63/63
...
llm_aces_gpt               1.33e-17     8.28e-17     2.46e-16        17.1        46.2  63/63
```

Machine-readable copies land in `results/summary_<benchmark>_<metric>.json`.

---

## 9. Sanity check: does the harness land on the paper's numbers?

The SINDy row is the cheapest complete row to produce (≈4 min for all 122
systems), which makes it the natural check that the protocol is wired up
correctly before spending hours on the rest. Running it end-to-end here gave:

| | Recon NMSE | Gen NMSE | OOD NMSE | Complexity | Sym. Acc |
|---|---|---|---|---|---|
| **ODEBench — this harness** | 2.45e-04 | 6.50e-01 | 1.57e+00 | 12.9 | 24.1% |
| ODEBench — paper Table 2 | 5.07e-04 | 5.09e-01 | 1.41e+00 | 11.7 | 18.5% |
| **ODEBase — this harness** | 5.82e-04 | 1.07e+00 | 1.96e-01 | 12.3 | 9.6% |
| ODEBase — paper Table 3 | 1.06e-03 | 1.12e+00 | 3.47e-01 | 12.5 | 5.9% |

Every figure is within a small factor of the published one, on a metric that
spans twenty orders of magnitude across methods — which is the main evidence
that the vector-field NMSE reading and the data generation are right. If your
SINDy row comes out very differently, fix that before launching anything else.

That run is committed under `reference_results/` so you can diff against it
per system. It is deliberately *not* under `results/`: every driver resumes by
skipping systems already stored with `status: "ok"`, so a shipped run inside
`results/` would make your own first SINDy launch skip all 122 systems and exit
in three seconds.

Individual spot checks on `rc-circuit` (a system every method should solve):

| method | recon NMSE | discovered equation |
|---|---|---|
| SINDy | 1.47e-32 | `0.30303 - 0.36075*x0` (exact) |
| ODEFormer | 2.45e-04 | `-0.3294*x0` |
| LLM-ODE | 1.23e-13 | `0.30304 - 0.360752*x0` |
| LLM-ACES (GPT-4o-mini) | 6.26e-16 | `0.3030307 - 0.3607504*x0` (exact) |

---

## Implementation notes and deliberate choices

These are the places where the paper leaves a degree of freedom, and what this
reproduction does about it.

**Two NMSE readings.** Section 3.1 says the three settings are evaluated
"trajectory-level", while every method in this code base (and in MDBench, whose
protocol the paper follows) optimises the *vector field* `u -> du`. The generated
NPZ files store the true derivative for all three windows, so each result JSON
records both: `recon_nmse` / `gen_nmse` / `ood_nmse` are vector-field NMSE, and
`recon_traj_nmse` / `gen_traj_nmse` / `ood_traj_nmse` integrate the discovered
system from that window's initial condition and compare states. The default
tables use the vector-field numbers, which is the reading that matches the
magnitudes reported in the paper; `--metric traj` produces the other one.

**Missing ODEBase support in the released code.** The released repo's
`generate_odebase.py` imports `scibench.data.*`, which was not bundled; those
definitions are now vendored into `scripts/scibench/data/` from the APPS-ODE
release. Likewise `aces_utils.load_true_ode` only knew the Strogatz systems, so
LLM-ACES silently disabled its acquisition loop on ODEBase; it now falls back to
the ODEBase catalogue, and `llm-aces/ic_bounds_odebase.json` supplies the IC
boxes.

**Active-baseline PySR budget.** The paper equalises the *acquisition* budget of
BO/QBC with LLM-ACES (10 rounds, 20 oracle samples per round split 10 train / 10
validation) but does not state their PySR budget. Since BO and QBC refit PySR
every round, they default to LLM-ACES's own PySR setting (20 iterations, 15
populations) so total search compute is comparable; `--pysr_niterations` /
`--pysr_populations` override this.

**LLM call budgets.** "125 LLM calls, generating 1000 candidate equations" is
implemented as 125 calls × 8 candidates for LLM-only, and for LLM-ODE as
`125 / (n_islands × dim)` iterations with `b = 8` hypotheses per prompt
(`--llm_calls` controls both). LLM-ACES uses its own budget: 10 iterations ×
up to 3 operator priors = 30 calls per system.

**SINDy library filtering.** The expanded basis includes `log`, `sqrt` and
`inv`, which are undefined on trajectories that go non-positive. Features that
evaluate to non-finite values on the training states are dropped for that
system, since STLSQ cannot fit NaN columns.

**Bugs this test round surfaced (all fixed).** Running every method for real on a
2-system subset, plus the first run on the Linux server, caught six things that
no amount of code reading had:

1. `sympytorch` is imported by the E2E checkpoint's unpickling path and was
   missing from the install list — E2E died on `torch.load`.
2. The LLM-only response parser stripped only one prefix, so
   `1. CANDIDATE: -4.8*x0 | ...` kept the `CANDIDATE:` label and every candidate
   failed to sympify; the method returned zero usable equations on *both* test
   systems. The rewrite also fixed a worse variant of the same bug: a bullet /
   numbering regex without lookaheads turned `-4.8 * x0` into `8 * x0`,
   silently dropping minus signs.
3. For 1-D systems the model puts several candidates on one `|`-separated line;
   the parser now splits those into separate candidates.
4. `run_apps_ode.py` passed a *relative* `--out` path to a subprocess that runs
   with `cwd=third_party/APPS-ODE/apps_ode_pytorch`, so results were written
   inside the vendored checkout and the driver reported "failed (exit 0)".
5. APPS-ODE's `model.train()` returns `best_expression=None` whenever the reward
   threshold is never reached (i.e. on most real systems); the driver now falls
   back to the ranked top-k population.
6. E2E's upstream `envs/generators.py` opens with
   `from numpy.compat.py3k import npy_load_module`, and `numpy.compat` was
   removed in NumPy 2 — so on any env that drifted to numpy>=2 the checkpoint
   failed to unpickle with `ModuleNotFoundError: No module named 'numpy.compat'`.
   The symbol is never used upstream, so `baselines/run_e2e.py` now re-supplies
   it before `torch.load` (`_ensure_numpy_compat`) and `setup_third_party.sh`
   deletes the dead import from fresh clones. E2E runs on numpy 1.x and 2.x.

Plus one efficiency fix: ODEFormer reloaded the pretrained checkpoint for each
of its 6 fits per system (~730 loads over a full benchmark); it is now cached
per process.

**What was verified before shipping.** In this single environment: SINDy (all
122 systems), PySR, BO, QBC, LLM-only, LLM-ODE, LLM-ACES (both benchmarks' data
generation and a full GPT-4o-mini run on `rc-circuit`), ODEFormer, the
symbolic-accuracy judge, the aggregation step and `scripts/tmux_run.sh` were all
run end-to-end, as was E2E (`rc-circuit`, pretrained checkpoint). The APPS-ODE
runner was verified for one epoch on `odebase_vars2_prog1` (its output flows
through the shared evaluator correctly). **Only Operon was never executed
here**: pyoperon publishes no wheel for macOS 13 (Linux x86_64 wheels exist, so
the server is fine). Its driver mirrors MDBench's implementation line for line,
but run the smoke test first
(`bash run_baseline.sh operon odebench --systems rc-circuit --no_resume`,
expect roughly `dx0/dt = 0.3030 - 0.3608*x0`) before launching the full sweep.

**APPS-ODE coverage.** APPS-ODE addresses systems by scibench ids.
`scripts/build_scibench_map.py` matches 120 of the 122 systems by comparing
vector fields numerically (with a description-based fallback);
`binocular-rivalry-model` and `refined-language-death-model` have no scibench
twin and are skipped, which is reported in the run log.

**Resuming, and re-running a method.** Every driver writes one JSON per system
and, on the next launch, skips only the systems stored with `status: "ok"` —
so a run that died halfway, or one where a few systems raised, is resumed
simply by launching the same command again. To force a full redo, either pass
`--no_resume` or delete that method's folder:

```bash
python -c "
import json, glob, collections
c = collections.Counter(json.load(open(f))['status']
                        for f in glob.glob('results/odebench/pysr/systems/*.json'))
print(c)"                                    # how many ok / error
rm -rf results/odebench/pysr                 # or: --no_resume
```

**PySR parallelism.** `--pysr_procs N` now runs N Julia *threads*
(`--pysr_parallelism auto`, which is also PySR's own default). The previous
default, `multiprocessing`, runs N separate Julia worker processes over
Distributed, and PySR tears those workers down while the search loop is still
fetching from them, so a perfectly successful run ends in a wall of

```
UNHANDLED TASK ERROR: Distributed.ProcessExitedException(6)
```

That message is teardown noise, not a failed fit — it is printed *after* the
equations are selected, the per-system JSON still says `status: "ok"`, and the
process exits 0. Threads avoid it entirely and share one heap, which also
matters on memory-capped containers. `--pysr_parallelism multiprocessing`
restores the old behaviour if you want it.

**Why BO/QBC/APPS-ODE look hung.** They are the only methods whose unit of work
is longer than a log line. BO and QBC run a *complete* PySR search inside every
acquisition round -- `n_iterations=10` rounds x one fit per state dimension --
and APPS-ODE runs 50 policy-gradient epochs in a subprocess that returns
nothing until it exits. Measured here (laptop, one core per fit): a BO round on
a 1-D system takes ~60 s once Julia is warm (~105 s for the first, which pays
Julia startup), so ~11 min for a 1-D system and ~35 min for a 3-D one;
APPS-ODE is ~80 s per epoch, i.e. ~70 min per system. Nothing is stuck.

Both now log per fit rather than per round, and APPS-ODE prints the path of the
live subprocess log, so `tail -f results/<bench>/<method>/run.log` always shows
movement. `--pysr_procs 8` (Julia threads) cuts BO/QBC substantially, and
`--shard i/n` splits any sweep over several tmux sessions round-robin.

**LLM-ACES itself is the same shape, only bigger.** Every system runs
`1 + n_iterations x max_concepts` = up to 31 operator concepts, and each concept
is one full PySR search *per state dimension* -- 31 x dim searches, on top of up
to 30 sequential LLM calls. Measured at `pysr=20/15` on 10 training points:
7.3 s per search serial, 1.8 s with 8 Julia threads, so a whole benchmark is
~7-10 h of PySR single-threaded and ~2 h at `PYSR_PROCS=8`. `fit_pysr_concept`
used to hardcode `parallelism="serial"`, which made `--pysr_procs` a no-op;
it now honours it (`procs > 1` -> `multithreading`, `procs == 1` keeps the
deterministic serial search the paper numbers were produced with). The old
`--fit_pause_seconds 5.0` default also slept 31 x 5 s per system -- ~2.7 h per
benchmark of doing nothing -- and is now `0.0`. The API calls are network-serial
and CPU cannot help them; `ACES_SHARD=i/n` (see RUN_TMUX.md §1) overlaps them by
running several systems at once.

The `Baseline PySR run (no LLM operator restriction)` block printed for every
system is deliberate: it fits PySR once with the full operator set and no LLM
prior, as the control the LLM-guided rounds are compared against.

**Console noise.** Third-party search loops lambdify every candidate equation
and evaluate it without an `np.errstate` guard, so a normal LLM-ODE / APPS-ODE
run used to bury its result lines under hundreds of
`RuntimeWarning: invalid value encountered in log` messages (plus a pandas
`FutureWarning` from `llmode.py:212`). SINDy is the same story: the Table 10
sweep tries 16 thresholds per system, so PySINDy's `UserWarning: Sparsity
parameter is too big ... eliminated all coefficients` fires on most of them by
construction. PySR is a third case: the 18 unary operators of Appendix B.2 make
DynamicExpressions.jl print "You have passed over 15 unary operators ..." on
every fit, and that one arrives on *Julia's* stderr, so only Julia's logger can
mute it (`common.quiet_julia_logging()` raises its level to `Error`; Julia
errors still propagate as exceptions). Those are all expected -- a candidate that
produces NaNs is *supposed* to score badly and be dropped -- so every driver now
calls `common.silence_numeric_warnings()`, and LLM-ODE additionally routes
upstream's root-logger chatter ("Error making random program: ...") into
`<result_dir>/llmode_upstream.log` instead of stdout. Nothing is lost from
`run.log`. To get the noise back for debugging:

```bash
NUMERIC_WARNINGS=1 bash run_baseline.sh llm_ode odebench --systems rc-circuit --no_resume
```

**LLM-ODE transport and interpreter.** The upstream `Llm` class calls a local
vLLM server through the OpenAI *responses* API and picks the model with
`models.list()[0]`.
Neither works against OpenRouter, so `baselines/run_llm_ode.py` substitutes a
chat-completions client and feeds the upstream evolutionary core our benchmark
data; islands, experience buffer, BFGS coefficient optimisation and Pareto
selection are the upstream implementation untouched.

**Anonymised prompts.** Section 4.3 anonymises both benchmarks by "replacing
all state variable names, time derivatives and semantic identifiers with generic
labels (x0, x1, ..., dx0, dx1, ...), removing any domain-specific terminology
from prompts". Every LLM path here is already anonymised by construction: the
LLM-ACES specs, the LLM-only prompt and the upstream LLM-ODE prompt all address
the system purely as `x0 ... x{d-1}` and never receive the system name or its
textual description.

**One run per system.** As in the paper (Section 3.1), each system is run once
and results are summarised by medians; `--seed` is fixed at 42 throughout.
