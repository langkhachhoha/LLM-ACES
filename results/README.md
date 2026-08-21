# Results tree

One folder per method per benchmark:

```
results/<benchmark>/<method>/
  systems/<system>.json           discovered equations + all metrics for one system
  results.jsonl                   append-only log of the same payloads
  run.log                         full stdout/stderr of the run
  symbolic_accuracy.json          per-dimension LLM-judge verdicts (after scoring)
  symbolic_accuracy_cache.json    judge cache, so re-scoring is nearly free
  llm_calls/<system>.jsonl        every prompt + response (LLM methods only)
  raw/                            upstream artefacts (APPS-ODE)
  pareto/                         per-dimension Pareto frontiers (LLM-ODE)
```

Each `systems/<system>.json` contains:

| field | meaning |
|---|---|
| `equations` | discovered RHS per state dimension, in terms of `x0..x{d-1}` |
| `recon_nmse` / `gen_nmse` / `ood_nmse` | vector-field NMSE on the three windows |
| `recon_traj_nmse` / `gen_traj_nmse` / `ood_traj_nmse` | trajectory NMSE after integrating the discovered system |
| `complexity` | expression-tree node count summed over dimensions |
| `train_time_s` | wall-clock for that system |
| `status` / `error` | `ok`, or the exception that ended the run |
| `queried_initial_conditions`, `iterations` | acquisition trace (active methods) |
| `llm_calls`, `prompt_tokens`, `completion_tokens` | API accounting (LLM methods) |

Summaries produced by `baselines.aggregate_results` land in
`results/summary_<benchmark>_<metric>.json`.
