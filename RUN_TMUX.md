# tmux command set — full reproduction

Everything runs in the **single `llm-aces` environment**, activated once up
front. The commands below assume you are already inside it — no
`conda activate` or `conda run` appears anywhere in them.

```bash
cd ~/LLM-ACES
conda activate llm-aces        # <- ONCE, before anything below
```

Sessions are launched through `scripts/tmux_run.sh`, which does two things:

1. **inherits the environment you are in right now**, passing `PATH` /
   `CONDA_PREFIX` explicitly. (If a tmux server was already running from before
   you activated, a plain `tmux new` would inherit the *server's* PATH and
   silently use the wrong Python.)
2. **keeps the session alive when the job finishes** — the pane drops into an
   interactive shell instead of exiting, so the session, its scrollback and its
   working directory all remain. Nothing kills your session.

```bash
bash scripts/tmux_run.sh <session-name> <command...>
```

Output is also tee'd to `logs/tmux/<session-name>.log`.

Prerequisites: [SETUP_SERVER.md](SETUP_SERVER.md) steps 1–5 are done, and
`data/ode/` (63 NPZ) plus `data/odebase/` (59 NPZ) exist.

---

## 0. Session map

| session | what runs in it |
|---|---|
| `aces-gpt-bench` / `aces-gpt-base` | LLM-ACES, GPT-4o-mini |
| `aces-qwen-bench` / `aces-qwen-base` | LLM-ACES, Qwen3 |
| `sindy` `pysr` `operon` | passive symbolic regression |
| `odeformer` `e2e` | transformer baselines |
| `llmonly` `llmode` | LLM-guided discovery |
| `bo` `qbc` `appsode` | active symbolic discovery |
| `score` | symbolic accuracy + final tables |

Useful at any time:

```bash
tmux ls                        # what exists (finished sessions stay listed)
tmux attach -t pysr            # watch a run   (detach: Ctrl-b then d)
tail -f logs/tmux/pysr.log     # or just follow the log
tail -f results/odebench/pysr/run.log
```

Sessions persist after their job completes, by design. Close one only when you
are done with it — type `exit` inside it, or `tmux kill-session -t <name>`.

---

## 1. LLM-ACES — the paper's method (4 runs)

Paper defaults: 10 iterations, ≤3 operator priors per round (30 LLM calls per
system), PySR 20 iterations / 15 populations, temperature 0.8, 10 candidate ICs
per acquisition step, `bo_init_points=3`.

```bash
bash scripts/tmux_run.sh aces-gpt-bench \
  bash scripts/run_llm_aces.sh odebench openai/gpt-4o-mini-2024-07-18 gpt

bash scripts/tmux_run.sh aces-gpt-base \
  bash scripts/run_llm_aces.sh odebase  openai/gpt-4o-mini-2024-07-18 gpt

bash scripts/tmux_run.sh aces-qwen-bench \
  bash scripts/run_llm_aces.sh odebench qwen/qwen3-30b-a3b-instruct-2507 qwen

bash scripts/tmux_run.sh aces-qwen-base \
  bash scripts/run_llm_aces.sh odebase  qwen/qwen3-30b-a3b-instruct-2507 qwen
```

Per-system artefacts: `logs/<bench>/llm_aces_<tag>/<system>/` holds
`active_llm_pysr_results.jsonl` (per-iteration concepts, NMSEs, acquired ICs,
wall time), `pysr_equations.txt` (every Pareto front PySR produced) and
`stdout.log`. Scored results land in `results/<bench>/llm_aces_<tag>/`.

---

## 2. Passive symbolic discovery

```bash
# SINDy — 16 thresholds x 2 alphas x 4 poly orders x 2 libraries  (~4 min total)
bash scripts/tmux_run.sh sindy \
  bash -c 'bash run_baseline.sh sindy odebench && bash run_baseline.sh sindy odebase'

# PySR — paper Table 10 budget (100 iterations, 20 populations, 1000 cycles)
bash scripts/tmux_run.sh pysr \
  bash -c 'bash run_baseline.sh pysr odebench --pysr_procs 8 && bash run_baseline.sh pysr odebase --pysr_procs 8'

# Operon — 1000 generations, MDL selection on the Pareto front
bash scripts/tmux_run.sh operon \
  bash -c 'bash run_baseline.sh operon odebench --n_threads 8 --max_time 1800 && bash run_baseline.sh operon odebase --n_threads 8 --max_time 1800'

# ODEFormer — pretrained checkpoint, beam 50, temperature swept over 5 values
bash scripts/tmux_run.sh odeformer \
  bash -c 'bash run_baseline.sh odeformer odebench && bash run_baseline.sh odeformer odebase'

# E2E — 200 input points, 10 trees to refine, rescale
bash scripts/tmux_run.sh e2e \
  bash -c 'bash run_baseline.sh e2e odebench && bash run_baseline.sh e2e odebase'
```

> PySR at the paper's full budget is the slowest of these: roughly 2–10 min per
> state dimension, so budget ~6 h for ODEBench and ~10 h for ODEBase on 8 cores.
> To scout faster first, add `--pysr_niterations 20 --pysr_populations 15`
> (LLM-ACES's own PySR budget) and re-run at full budget later with `--no_resume`.
>
> `--pysr_procs 8` means 8 Julia **threads**. If you switch to
> `--pysr_parallelism multiprocessing`, expect a wall of
> `UNHANDLED TASK ERROR: Distributed.ProcessExitedException(n)` at the end of
> every fit — that is PySR tearing down its Distributed workers *after* the
> equations are chosen, not a failure (the per-system JSON still says
> `status: "ok"`).

> Re-launching any of these resumes: a system is skipped only if its stored
> status is `ok`, so systems that errored are retried automatically. Count them
> with
> `python -c "import json,glob,collections; print(collections.Counter(json.load(open(f))['status'] for f in glob.glob('results/odebench/pysr/systems/*.json')))"`.

---

## 3. LLM-guided symbolic discovery

Both use 125 LLM calls / 1000 candidate equations per system with GPT-4o-mini,
exactly as the paper specifies for all LLM baselines.

```bash
# LLM-only (NewtonBench protocol, temperature 1.0)
bash scripts/tmux_run.sh llmonly \
  bash -c 'MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_only odebench && MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_only odebase'

# LLM-ODE (upstream multi-island evolutionary core)
bash scripts/tmux_run.sh llmode \
  bash -c 'MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_ode odebench && MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_ode odebase'
```

Every prompt and response is kept in
`results/<bench>/<method>/llm_calls/<system>.jsonl` for later inspection.

---

## 4. Active symbolic discovery

```bash
# Bayesian Optimization: GP over IC space, EI over a pool of 256 ICs, 10 rounds
bash scripts/tmux_run.sh bo \
  bash -c 'bash run_baseline.sh bo odebench --pysr_procs 8 && bash run_baseline.sh bo odebase --pysr_procs 8'

# Query-by-Committee: committee drawn from the PySR Pareto front, 10 rounds
bash scripts/tmux_run.sh qbc \
  bash -c 'bash run_baseline.sh qbc odebench --pysr_procs 8 && bash run_baseline.sh qbc odebase --pysr_procs 8'

# APPS-ODE: 50 policy-gradient epochs, BFGS, inverse-NMSE reward
bash scripts/tmux_run.sh appsode \
  bash -c 'bash run_baseline.sh apps_ode odebench --n_cores 8 && bash run_baseline.sh apps_ode odebase --n_cores 8'
```

> **These three are the slow ones, and they log per system, not per second.**
> Expect long silences — that is the search running, not a hang:
>
> | method | per system | why |
> |---|---|---|
> | BO / QBC | ~2 min per PySR fit x 10 rounds x dim → 20 min (1-D) to 1 h (3-D) | a full PySR search inside every acquisition round |
> | APPS-ODE | ~70 min (50 policy-gradient epochs, ~80 s each) | one subprocess per system, silent until it exits |
>
> Check they are alive rather than waiting for the next line:
>
> ```bash
> tail -f results/odebench/bo/run.log            # BO/QBC: a line per fit
> tail -f results/odebench/apps_ode/raw/*.log    # APPS-ODE: the live subprocess log
> ```
>
> APPS-ODE at ~70 min/system is ~6 days for all 122 sequentially, so split it
> over sessions with `--shard i/n` (round-robin, so the 3-D systems spread out):
>
> ```bash
> for i in 0 1 2 3; do
>   bash scripts/tmux_run.sh appsode$i \
>     bash run_baseline.sh apps_ode odebench --n_cores 8 --shard $i/4
> done
> ```
>
> The same flag works for every method (BO and QBC benefit just as much).
> Reducing `--total_iterations` below 50 departs from the paper; say so if you do.

BO and QBC record every queried initial condition and the per-iteration
equations in their result JSON (`queried_initial_conditions`, `iterations`), so
the acquisition trajectory can be replotted later without re-running anything.

---

## 5. Scoring and tables

Run once the work sessions have printed `finished ... with exit 0`.

```bash
bash scripts/tmux_run.sh score bash -c '
  python -m baselines.symbolic_accuracy --benchmark odebench --quiet &&
  python -m baselines.symbolic_accuracy --benchmark odebase  --quiet &&
  python -m baselines.aggregate_results --benchmark odebench --csv results/table2.csv &&
  python -m baselines.aggregate_results --benchmark odebase  --csv results/table3.csv &&
  python -m baselines.aggregate_results --benchmark odebench --metric traj &&
  python -m baselines.aggregate_results --benchmark odebase  --metric traj'

tmux attach -t score
```

---

## 6. Recommended launch order

The API-bound and CPU-bound jobs do not compete, so start one of each first.

```bash
# Wave 1 — start together
bash scripts/tmux_run.sh aces-gpt-bench bash scripts/run_llm_aces.sh odebench openai/gpt-4o-mini-2024-07-18 gpt
bash scripts/tmux_run.sh pysr  bash -c 'bash run_baseline.sh pysr odebench --pysr_procs 8'
bash scripts/tmux_run.sh sindy bash -c 'bash run_baseline.sh sindy odebench && bash run_baseline.sh sindy odebase'

# Wave 2 — once wave 1's API job finishes
bash scripts/tmux_run.sh aces-qwen-bench bash scripts/run_llm_aces.sh odebench qwen/qwen3-30b-a3b-instruct-2507 qwen
bash scripts/tmux_run.sh llmonly bash -c 'MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_only odebench'

# Wave 3
bash scripts/tmux_run.sh aces-gpt-base  bash scripts/run_llm_aces.sh odebase openai/gpt-4o-mini-2024-07-18 gpt
bash scripts/tmux_run.sh aces-qwen-base bash scripts/run_llm_aces.sh odebase qwen/qwen3-30b-a3b-instruct-2507 qwen
bash scripts/tmux_run.sh bo  bash -c 'bash run_baseline.sh bo odebench && bash run_baseline.sh bo odebase'
bash scripts/tmux_run.sh qbc bash -c 'bash run_baseline.sh qbc odebench && bash run_baseline.sh qbc odebase'

# Wave 4
bash scripts/tmux_run.sh odeformer bash -c 'bash run_baseline.sh odeformer odebench && bash run_baseline.sh odeformer odebase'
bash scripts/tmux_run.sh e2e       bash -c 'bash run_baseline.sh e2e odebench && bash run_baseline.sh e2e odebase'
bash scripts/tmux_run.sh appsode   bash -c 'bash run_baseline.sh apps_ode odebench --n_cores 8'
```

Rough wall-clock on an 8-core machine (both benchmarks, 122 systems total):

| method | time |
|---|---|
| SINDy | ~4 min (measured) |
| ODEFormer | 1–2 h |
| E2E | 2–4 h |
| LLM-only | 3–6 h |
| BO / QBC | 3–6 h each |
| LLM-ACES (per model) | 6–12 h |
| LLM-ODE | 6–12 h |
| Operon | 4–8 h |
| PySR (full Table 10 budget) | 12–20 h |
| APPS-ODE | ~70 min/system — split across sessions (see above) |

---

## 7. Partial / scouting runs

Every driver takes `--limit N` (first N systems) and `--systems a b c`
(explicit stems), which is the fastest way to sanity-check a configuration.
These are quick enough to run in the foreground:

```bash
bash run_baseline.sh bo odebench --systems lotka-volterra-simple duffing-equation \
     --n_iterations 3 --pysr_niterations 10 --no_resume

RESULTS_ROOT=/tmp/probe bash run_baseline.sh pysr odebase --limit 3 \
     --pysr_niterations 20 --pysr_populations 15
```

For LLM-ACES, the same is done with environment overrides:

```bash
N_ITERATIONS=3 PYSR_NITER=10 bash scripts/run_llm_aces.sh odebench \
    openai/gpt-4o-mini-2024-07-18 gpt-probe
```

---

## 8. Ablations (Section 4.1, Figure 2)

The paper ablates on **15 stratified ODEBench systems** spanning 1D, 2D and 3D.
Two of the three variants are reachable with existing flags:

```bash
export STRAT="rc-circuit population-growth-carrying-capacity logistic-equation-allee-effect \
gompertz-law-tumor-growth language-death-model harmonic-oscillator duffing-equation \
van-der-pol-oscillator lotka-volterra-simple sir-infection brusselator glycolytic-oscillator \
lorenz-equations-chaotic rossler-attractor sprott-attractor"

# w/o Predictive Divergence: keep the LLM priors, sample ICs uniformly at random
bash scripts/tmux_run.sh abl-nodiv \
  env ACES_SYSTEMS="$STRAT" bash scripts/run_llm_aces.sh odebench \
      openai/gpt-4o-mini-2024-07-18 nodiv --acq_method random

# w/o LLM Priors: the unconstrained operator vocabulary (single fallback concept)
bash scripts/tmux_run.sh abl-nopriors \
  env ACES_SYSTEMS="$STRAT" MAX_CONCEPTS=1 bash scripts/run_llm_aces.sh odebench \
      openai/gpt-4o-mini-2024-07-18 nopriors --concept_stop_token FORCE_FALLBACK
```

The third variant (**w/o Diversity** — asking for all priors in one call instead
of one call per prior) needs a prompt change in
`llm-aces/prompts/ode_concept_more.txt`; it is not wired to a flag.
