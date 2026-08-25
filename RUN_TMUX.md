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

## The whole run — two commands

```bash
cd ~/LLM-ACES && conda activate llm-aces
bash scripts/launch_all.sh            # start everything, sized to this machine
# ... hours later, when every session says 'finished ... with exit 0':
bash scripts/score_all.sh             # scores LLM-ACES + symbolic accuracy + tables
```

`launch_all.sh` reads `nproc` and lays the machine out as sharded tmux sessions:
APPS-ODE (the long pole) gets the most shards, BO and QBC get 2 Julia threads
each, LLM-ACES gets 3 shards per run because it is blocked on the API more than
on the CPU, and everything cheap shares one sequential lane. On 64 cores that is
**~12 h for the whole reproduction**; the same work run unsharded is over a week.

```bash
bash scripts/launch_all.sh --dry-run          # print the plan, start nothing
bash scripts/launch_all.sh --cores 32         # override the detected core count
bash scripts/launch_all.sh --only apps,bo     # one lane at a time (small boxes)
```

| lane | sessions at 64 cores | cores | expected |
|---|---|---|---|
| `apps` — APPS-ODE | 16 (8 per benchmark) | 16 | ~9 h |
| `bo` — Bayesian optimization | 8 (4 per benchmark) | 16 | ~7 h ODEBench, ~9 h ODEBase |
| `qbc` — query-by-committee | 8 | 16 | ~7 h / ~9 h |
| `aces` — LLM-ACES x 4 runs | 12 | 12 | ~4-5 h per run |
| `quick` — SINDy, ODEFormer, E2E, PySR | 1 | 8 | ~6 h |
| `llm` — LLM-only, LLM-ODE | 2 | ~0 | API-bound |

That is 68 nominal on 64 cores, deliberately: the `aces` and `llm` sessions
spend most of their wall time waiting on the network, so real load sits near 60.
Check RAM first — ~45 Julia+Python processes at ~1.5 GB each want 64 GB+:

```bash
free -g
```

If RAM is tight, or the box is small, run one lane at a time with `--only`.
Nothing is lost by stopping and restarting: every driver skips systems that
already have a result, so a killed shard picks up where it left off.

The rest of this file is the reference — what each session actually runs, and
how to run any single piece by hand.

---

## 0. Session map

These are the names `launch_all.sh` uses (`_<i>` is the shard index).

| session | what runs in it |
|---|---|
| `aces_gpt_<bench>_<i>` | LLM-ACES, GPT-4o-mini |
| `aces_qwen_<bench>_<i>` | LLM-ACES, Qwen3 |
| `bo_<bench>_<i>` `qbc_<bench>_<i>` `appsode_<bench>_<i>` | active symbolic discovery |
| `quick` | SINDy -> ODEFormer -> E2E -> PySR, sequentially |
| `llmonly` `llmode` | LLM-guided discovery |
| `operon` | Operon (optional; needs the pyoperon build) |
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
  env PYSR_PROCS=8 bash scripts/run_llm_aces.sh odebench openai/gpt-4o-mini-2024-07-18 gpt

bash scripts/tmux_run.sh aces-gpt-base \
  env PYSR_PROCS=8 bash scripts/run_llm_aces.sh odebase  openai/gpt-4o-mini-2024-07-18 gpt

bash scripts/tmux_run.sh aces-qwen-bench \
  env PYSR_PROCS=8 bash scripts/run_llm_aces.sh odebench qwen/qwen3-30b-a3b-instruct-2507 qwen

bash scripts/tmux_run.sh aces-qwen-base \
  env PYSR_PROCS=8 bash scripts/run_llm_aces.sh odebase  qwen/qwen3-30b-a3b-instruct-2507 qwen
```

**What one system costs.** Each system runs `1 + n_iterations x max_concepts`
= up to **31 operator concepts**, and every concept is one complete PySR search
*per state dimension* — 31 x dim searches, plus up to 30 sequential LLM calls.
Measured at `pysr=20/15` on 10 training points: **7.3 s per search serial, 1.8 s
with `PYSR_PROCS=8`** (the first search of a process adds ~11 s of Julia
compilation). Over a whole benchmark:

| | searches | PySR time, serial | PySR time, `PYSR_PROCS=8` |
|---|---|---|---|
| ODEBench (117 dims) | 3,627 | ~7.4 h | ~1.8 h |
| ODEBase (154 dims) | 4,774 | ~9.7 h | ~2.4 h |

On top of that sits the API: ≤30 calls per system, issued one after another, so
a 10 s model adds ~5 min per system (~5 h per benchmark) that no amount of CPU
will remove. Only running several systems at once hides it — `ACES_SHARD="i/n"`
splits the benchmark round-robin over n sessions:

```bash
for i in 0 1 2 3; do
  bash scripts/tmux_run.sh aces-qwen-base-$i \
    env PYSR_PROCS=2 ACES_SHARD=$i/4 SKIP_EVAL=1 \
    bash scripts/run_llm_aces.sh odebase qwen/qwen3-30b-a3b-instruct-2507 qwen
done
```

`SKIP_EVAL=1` stops each shard from scoring a half-finished `outputs/` folder;
score once, after the last shard exits, with the command the shard prints. The
shards share `outputs/<bench>/llm_aces_<tag>/` — one JSON per system, nothing to
merge. A shard that dies can just be restarted: finished systems are skipped.

**`Baseline PySR run (no LLM operator restriction)` in the log is expected.**
Before iteration 1, every system is fitted once with PySR's full operator set
and no LLM prior at all, and its train/test NMSE is printed and stored. That is
the control arm: it is what the LLM-guided rounds are measured against, so the
per-system JSON can say whether the operator priors helped. It is one concept
out of the 31, i.e. ~3% of the run — it is not a retry or a fallback.

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

Six sessions, one per (method, benchmark). They are kept separate rather than
chained with `&&` so that each benchmark starts immediately, a failure in one
cannot hold up the other, and you can see at a glance in `tmux ls` which half
is still running.

```bash
# --- Bayesian Optimization: GP over IC space, EI over a pool of 256 ICs, 10 rounds
bash scripts/tmux_run.sh bo_odebench \
  bash run_baseline.sh bo odebench --pysr_procs 8

bash scripts/tmux_run.sh bo_odebase \
  bash run_baseline.sh bo odebase --pysr_procs 8

# --- Query-by-Committee: committee drawn from the PySR Pareto front, 10 rounds
bash scripts/tmux_run.sh qbc_odebench \
  bash run_baseline.sh qbc odebench --pysr_procs 8

bash scripts/tmux_run.sh qbc_odebase \
  bash run_baseline.sh qbc odebase --pysr_procs 8

# --- APPS-ODE: 50 policy-gradient epochs, BFGS, inverse-NMSE reward
bash scripts/tmux_run.sh appsode_odebench \
  bash run_baseline.sh apps_ode odebench --n_cores 8

bash scripts/tmux_run.sh appsode_odebase \
  bash run_baseline.sh apps_ode odebase --n_cores 8
```

Running all six at once on one 8-core box makes them fight for cores. Either
start the two benchmarks of one method and add the next method when they
finish, or shard instead (below), which gives you the same parallelism with
control over how many cores are in play.

> **These three are the slow ones, and they log per system, not per second.**
> Long silences are the search running, not a hang. The unit that matters is a
> *dimension*: BO and QBC run one complete PySR search per state dimension per
> acquisition round, and there are 10 rounds.
>
> | | measured | per system |
> |---|---|---|
> | BO / QBC, `--pysr_procs 1` | ~150 s per dimension per round | ~25 min x dim (so ~75 min for a 3-D system) |
> | APPS-ODE | ~80 s per policy-gradient epoch | ~70 min (50 epochs) |
>
> ODEBench is 117 state dimensions over 63 systems (23 1-D, 28 2-D, 10 3-D,
> 2 4-D) and ODEBase is 154 over 59 (23 2-D, 36 3-D), so single-core BO or QBC
> is ~49 h and ~64 h respectively. `--pysr_procs 8` runs each PySR search on 8
> Julia threads and is the cheapest way to cut that; `--shard i/n` splits a
> sweep over several sessions round-robin:
>
> ```bash
> for i in 0 1 2 3; do
>   bash scripts/tmux_run.sh qbc_odebench_$i \
>     bash run_baseline.sh qbc odebench --pysr_procs 2 --shard $i/4
> done
> ```
>
> APPS-ODE at ~70 min/system is ~3 days per benchmark sequentially, so it wants
> the same treatment:
>
> ```bash
> for i in 0 1 2 3; do
>   bash scripts/tmux_run.sh appsode_odebench_$i \
>     bash run_baseline.sh apps_ode odebench --n_cores 2 --shard $i/4
> done
> ```
>
> Shards write into the same `results/<bench>/<method>/` folder — one JSON per
> system, so there is nothing to merge afterwards. Reducing `--total_iterations`
> below 50, or `--n_iterations` below 10, departs from the paper; say so if you do.
>
> Check progress rather than waiting for the next line:
>
> ```bash
> tail -f results/odebench/qbc/run.log            # a line per dimension fit
> tail -f results/odebench/apps_ode/raw/*.log     # APPS-ODE's live subprocess log
> ```

BO and QBC record every queried initial condition and the per-iteration
equations in their result JSON (`queried_initial_conditions`, `iterations`), so
the acquisition trajectory can be replotted later without re-running anything.

---

## 5. Scoring and tables

Run once the work sessions have printed `finished ... with exit 0`.
`bash scripts/score_all.sh` does all of it — the four LLM-ACES runs (whose
shards were launched with `SKIP_EVAL=1`, so nothing has scored them yet), then
symbolic accuracy, then the tables. The pieces, if you want them separately:

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

**Shard first, thread second.** Splitting a sweep over n sessions is linear —
n shards do n times the work. Threads are not: 8 Julia threads made one PySR
search 4x faster, not 8x (populations, not data, is what gets parallelised, and
these fits see 10-110 points). So spend the machine on shards and give each one
1-2 threads. The exception is APPS-ODE, whose per-system cost is a serial
policy-gradient loop: shard it, do not thread it.

### On a 64-core box — everything at once, ~12 h

`bash scripts/launch_all.sh` does exactly the following; run the block by hand
only if you want to change the split.

```bash
# --- APPS-ODE: the long pole (~70 min/system, 122 systems). 16 shards.
for i in 0 1 2 3 4 5 6 7; do
  bash scripts/tmux_run.sh appsode_odebench_$i \
    bash run_baseline.sh apps_ode odebench --n_cores 1 --shard $i/8
  bash scripts/tmux_run.sh appsode_odebase_$i \
    bash run_baseline.sh apps_ode odebase  --n_cores 1 --shard $i/8
done

# --- BO and QBC: 4 shards x 2 threads per benchmark
for m in bo qbc; do
  for b in odebench odebase; do
    for i in 0 1 2 3; do
      bash scripts/tmux_run.sh ${m}_${b}_$i \
        bash run_baseline.sh $m $b --pysr_procs 2 --shard $i/4
    done
  done
done

# --- LLM-ACES: 3 shards per run. These block on the API more than on the CPU.
for tag_model in "gpt openai/gpt-4o-mini-2024-07-18" "qwen qwen/qwen3-30b-a3b-instruct-2507"; do
  set -- $tag_model
  for b in odebench odebase; do
    for i in 0 1 2; do
      bash scripts/tmux_run.sh aces-$1-$b-$i \
        env PYSR_PROCS=1 ACES_SHARD=$i/3 SKIP_EVAL=1 \
        bash scripts/run_llm_aces.sh $b "$2" "$1"
    done
  done
done

# --- Everything cheap, one lane, sequential (skip what is already done)
bash scripts/tmux_run.sh quick bash -c '
  bash run_baseline.sh sindy     odebench && bash run_baseline.sh sindy     odebase &&
  bash run_baseline.sh odeformer odebench && bash run_baseline.sh odeformer odebase &&
  bash run_baseline.sh e2e       odebench && bash run_baseline.sh e2e       odebase &&
  bash run_baseline.sh pysr      odebench --pysr_procs 8 &&
  bash run_baseline.sh pysr      odebase  --pysr_procs 8'

bash scripts/tmux_run.sh llmonly bash -c '
  MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_only odebench &&
  MODEL=openai/gpt-4o-mini-2024-07-18 bash run_baseline.sh llm_only odebase'

bash scripts/tmux_run.sh llmode bash -c '
  bash run_baseline.sh llm_ode odebench && bash run_baseline.sh llm_ode odebase'
```

| lane | sessions | cores each | cores | expected |
|---|---|---|---|---|
| APPS-ODE | 16 | 1 | 16 | ~9 h |
| BO | 8 | 2 | 16 | ~7 h ODEBench, ~9 h ODEBase |
| QBC | 8 | 2 | 16 | ~7 h / ~9 h |
| LLM-ACES | 12 | 1 | 12 | ~4-5 h per run |
| quick lane (SINDy, ODEFormer, E2E, PySR) | 1 | 8 | 8 | ~6 h |
| LLM-only, LLM-ODE | 2 | ~0 | - | API-bound |

That is 68 nominal on 64 cores, which is fine: the LLM-ACES and LLM-only/LLM-ODE
sessions spend most of their wall time blocked on the network, so real load sits
around 60. Two things to check before launching:

```bash
free -g     # ~45 Julia+Python processes, budget ~1.5 GB each -> want 64 GB+
nproc
```

If RAM is tight, halve the APPS-ODE and BO/QBC shard counts and run them in two
passes — restarting a shard is free, finished systems are skipped. If OpenRouter
starts returning 429s, drop LLM-ACES to 2 shards per run.

Score once, at the end (the LLM-ACES shards ran with `SKIP_EVAL=1`):

```bash
for b in odebench odebase; do
  for t in gpt qwen; do
    python -m baselines.eval_llm_aces --benchmark $b \
      --outputs_dir outputs/$b/llm_aces_$t --logs_dir logs/$b/llm_aces_$t \
      --results_root results --method_name llm_aces_$t --model $t
  done
done
```

then §5.

### On an 8-core box

Run one lane at a time, longest first — `launch_all.sh --only apps`, then
`--only bo`, `--only qbc`, `--only aces`, `--only quick,llm`. Rough wall-clock for both
benchmarks (122 systems) at 8 cores total:

| method | time |
|---|---|
| SINDy | ~4 min (measured) |
| ODEFormer | 1-2 h |
| E2E | 2-4 h |
| LLM-only | 3-6 h |
| BO / QBC | ~49 h (ODEBench) + ~64 h (ODEBase) each at `--pysr_procs 1`; shard them |
| LLM-ACES (per model) | ~7 h ODEBench + ~10 h ODEBase of PySR at `PYSR_PROCS=1`, plus ~5 h of API |
| LLM-ODE | 6-12 h |
| Operon | 4-8 h |
| PySR (full Table 10 budget) | 12-20 h |
| APPS-ODE | ~70 min/system - split across sessions (see above) |

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
