# Reference run

The SINDy sweep over all 122 systems that was used to validate this
reproduction against the paper's Table 1/2 (macOS, Python 3.11, numpy 1.26.4).
It lives here rather than in `results/` for one practical reason: every driver
resumes by skipping systems whose stored `status` is `ok`, so a shipped run
inside `results/` would make *your* first `run_baseline.sh sindy ...` skip all
122 systems and exit immediately.

Nothing reads this directory. It is here so you can diff your own numbers
against a known-good run:

```bash
python -c "
import json, glob
for src in ('reference_results', 'results'):
    v = [json.load(open(f))['recon_nmse'] for f in glob.glob(src+'/odebench/sindy/systems/*.json')]
    print(src, len(v), sorted(v)[len(v)//2] if v else '-')
"
```

Vector-field NMSE (median) in this run vs the paper:

| benchmark | recon | gen | OOD | complexity | symbolic |
|---|---|---|---|---|---|
| ODEBench (here) | 2.45e-04 | 6.50e-01 | 1.57e+00 | 12.9 | 24.1% |
| ODEBench (paper) | 5.07e-04 | 5.09e-01 | 1.41e+00 | 11.7 | 18.5% |
| ODEBase (here) | 5.82e-04 | 1.07e+00 | 1.96e-01 | 12.3 | 9.6% |
| ODEBase (paper) | 1.06e-03 | 1.12e+00 | 3.47e-01 | 12.5 | 5.9% |
