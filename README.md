# HMACL and Recovered MATDD

This repository contains the original HMACL code and a reconstructed
implementation of **Multi-agent Automatic Task Design and Dispatch (MATDD)**.
The original MATDD experiment source was lost; the recovery follows the ECAI
manuscript and records every underspecified design choice in
[`docs/MATDD_RECOVERY.md`](docs/MATDD_RECOVERY.md).
Current validation evidence and remaining experiment gaps are tracked in
[`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).

## What is implemented

- bounded, normalized task parameter spaces;
- a Beta-policy PPO curriculum designer;
- parallel-team regret from independent protagonist and antagonist learners;
- historical curriculum replay ranked by TD error and target proximity;
- a framework-independent training state machine with total interaction
  accounting;
- HMACL episode and multi-process parallel backends with dynamic task updates;
- M2ALE map-size, VMAS Football, and PettingZoo Pursuit task adapters;
- target-shaped zero padding and active-agent loss masks;
- the paper's Football student network (`FC-256/GRU-256/FC-128`);
- the paper's Pursuit convolutional student (`Conv-16/Conv-16`, then
  `FC-256/GRU-256/FC-128`);
- DyNA-style shared observation encoding and masked team aggregation for QMIX
  and MAPPO;
- paper comparison strategies (LI, DR, Minimax, and direct target training)
  plus TV/SR/no-replay dispatcher ablations;
- deterministic unit tests and a runnable end-to-end smoke path.

Football curricula scale both teams from 3 agents to a configurable target.
The paper's 18-agent and 25-agent target interfaces are supported. Pursuit
curricula vary square map size and pursuer count, including the paper's 60-20
and 80-30 targets.

## Setup

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify

Run the dependency-light core tests:

```bash
python -m unittest test.test_matdd_core -v
```

Run a small end-to-end M2ALE smoke experiment:

```bash
python -m hmacl.matdd_run \
  --name qmix --designer random \
  --curriculum_iterations 1 --steps_per_curriculum 20 \
  --final_target_steps 0 --curriculum_eval_episodes 1 \
  --eval_nepisode 1 --test_nepisode 1 \
  --batch_size 1 --buffer_size 2 \
  --use_cuda false --use_wandb false --exp matdd-smoke
```

Run Football with a small target and parallel IPPO workers:

```bash
python -m hmacl.matdd_run \
  --name ippo --env vmas --runner parallel \
  --batch_size_run 2 --batch_size 2 --target_agents 5 \
  --episode_limit 16 --designer random \
  --curriculum_iterations 1 --steps_per_curriculum 1 \
  --final_target_steps 1 --curriculum_eval_episodes 2 \
  --eval_nepisode 2 --use_cuda false --use_wandb false \
  --exp matdd-football-smoke
```

Results, dispatcher state, step accounting, and model checkpoints are written
under `runs/<experiment>/`. For five-seed configurations, run
`scripts/run_matdd_m2ale.sh [qmix|ippo|mappo]` or
`scripts/run_matdd_football.sh [qmix|ippo|mappo] [18|25]` from the repository
root. Pursuit uses:

```bash
scripts/run_matdd_pursuit.sh qmix 60-20
scripts/run_matdd_pursuit_smoke.sh
scripts/run_matdd_baseline.sh li qmix pursuit60-20 tv_sr
scripts/run_matdd_baseline_smoke.sh
```

## Reproducibility note

MATDD counts protagonist training, antagonist training, current-task
evaluation, and target-task evaluation interactions. HMACL's conservative
model rollback is not enabled in the recovered method; evaluate it separately
if it is reintroduced.
