# Experiment Status

Validated locally on CPU on 2026-08-12 with VMAS 1.5.2 and PettingZoo 1.26.1.
These runs verify the reconstructed execution path; their short episodes and
small budgets are not paper-quality performance measurements.

## Automated checks

- `python -m pytest -q test/test_matdd_core.py`: 14 tests passed.
- Ruff E/F/I and format checks passed for every changed Python file.
- `python -m compileall -q hmacl` and both shell syntax checks passed.
- VMAS target interfaces created and stepped successfully at 18 agents
  (`obs=296`, `state=5328`) and 25 agents (`obs=408`, `state=10200`).
- Pursuit target interfaces created and stepped successfully at 60x60 with 20
  pursuers and 80x80 with 30 pursuers (`obs=7x7x3`, five actions).

## Paper-scale interface smoke runs

| Target | Student | Runner | Total steps | Final mean abs TD error | PPO teacher update |
|---|---|---:|---:|---:|---:|
| Football-18 | QMIX | episode | 216 | 1.1427 | yes |
| Football-18 | IPPO | parallel (2) | 312 | 0.5449 | yes |
| Football-18 | MAPPO | parallel (2) | 312 | 0.2943 | yes |
| Football-25 | QMIX | episode | 96 | 0.2858 | n/a (random designer) |

All runs used the paper-shaped `matdd_rnn` student. TD-error values were finite
and nonzero, and parallel workers exited cleanly.
Run `scripts/run_matdd_football_smoke.sh` to reproduce these checks. Full
five-seed commands are in `scripts/run_matdd_football.sh`; those long-running
experiments have not been executed in this recovery pass.

## Pursuit interface smoke runs

| Target | Student | State path | Total steps | Final mean abs TD error | PPO teacher update |
|---|---|---|---:|---:|---:|
| Pursuit-60-20 | QMIX | DyNA mixer | 156 | 0.3511 | yes |
| Pursuit-60-20 | IPPO | local critic | 156 | 0.4236 | yes |
| Pursuit-60-20 | MAPPO | DyNA critic | 156 | 0.4231 | yes |
| Pursuit-80-30 | QMIX | DyNA mixer | 156 | 0.8500 | yes |
| Pursuit-80-30 | IPPO | local critic | 156 | 0.3558 | yes |
| Pursuit-80-30 | MAPPO | DyNA critic | 156 | 0.3418 | yes |

All six runs used the paper-shaped `matdd_conv` student, two curriculum
iterations, six-step episodes, and the actual target interface for final
training/evaluation. JSON outputs contained no NaN or infinity. Run
`scripts/run_matdd_pursuit_smoke.sh` to reproduce them; full five-seed commands
are in `scripts/run_matdd_pursuit.sh`.

## Remaining experiment gap

Both environment families and reported target interfaces are executable, but
the long five-seed studies, paper baselines, and ablations have not yet been
rerun. Until those complete, the manuscript's comparative performance claims
remain historical rather than reproduced evidence.
