# Experiment Status

Validated locally on CPU on 2026-08-12 with VMAS 1.5.2 and PettingZoo 1.26.1.
These runs verify the reconstructed execution path; their short episodes and
small budgets are not paper-quality performance measurements.

## Automated checks

- `python -m unittest test.test_matdd_core -v`: 19 tests passed.
- Ruff E/F/I and format checks passed for every changed Python file.
- `python -m compileall -q hmacl` and all changed shell syntax checks passed.
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

## Baseline and dispatcher smoke runs

All comparison paths were exercised with QMIX on Pursuit-60-20. Rule-based and
Minimax runs disable replay in this smoke test so two newly generated tasks can
be inspected; the full comparison script enables TV+SR replay by default.

| Path | Observed task/source behavior | Total steps | Final mean abs TD error |
|---|---|---:|---:|
| LI | 16-8 -> 38-14, both designed | 180 | 0.9797 |
| DR | two independent uniform tasks | 180 | 0.8547 |
| Minimax | PPO update from negative student return | 180 | 0.6584 |
| Direct | target 60-20 twice, no antagonist | 180 | 0.1352 |
| TV | second task replayed, effective `rho=0` | 276 | 0.2069 |
| SR | second task replayed, effective `rho=1` | 276 | 0.2069 |
| No Replay | both tasks designed | 276 | 0.8038 |

All losses were finite. Reproduce with
`scripts/run_matdd_baseline_smoke.sh`; launch five-seed comparisons with, for
example, `scripts/run_matdd_baseline.sh li qmix pursuit60-20 tv_sr`.
The same Minimax path also completed on Football-18 with `matdd_rnn`,
`dyna_qmix`, a PPO teacher update, 180 total steps, and final mean absolute TD
error 0.9574.

## Remaining experiment gap

Both environment families, all reported target interfaces, curriculum
baselines, and dispatcher ablations are executable. The long five-seed studies
have not yet been rerun, so the manuscript's comparative performance claims
remain historical rather than reproduced evidence.
