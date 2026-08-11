# Experiment Status

Validated locally on CPU on 2026-08-12 with VMAS 1.5.2. These runs verify the
reconstructed execution path; their 12-step episodes and small budgets are not
paper-quality performance measurements.

## Automated checks

- `python -m unittest discover -s test -v`: 9 tests passed.
- Ruff E/F/I and format checks passed for every changed Python file.
- `python -m compileall -q hmacl` and both shell syntax checks passed.
- VMAS target interfaces created and stepped successfully at 18 agents
  (`obs=296`, `state=5328`) and 25 agents (`obs=408`, `state=10200`).

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

## Remaining experiment gap

Football is executable for the two paper target sizes. Pursuit-60-20 and
Pursuit-80-30 remain blocked on restoring a current PettingZoo/SISL adapter and
the paper's convolutional/DyNA model path. Until those are implemented and full
five-seed runs complete, the manuscript's original comparative claims should
be treated as historical rather than reproduced evidence.
