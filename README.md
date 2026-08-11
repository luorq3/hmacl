# HMACL and Recovered MATDD

This repository contains the original HMACL code and a reconstructed
implementation of **Multi-agent Automatic Task Design and Dispatch (MATDD)**.
The original MATDD experiment source was lost; the recovery follows the ECAI
manuscript and records every underspecified design choice in
[`docs/MATDD_RECOVERY.md`](docs/MATDD_RECOVERY.md).

## What is implemented

- bounded, normalized task parameter spaces;
- a Beta-policy PPO curriculum designer;
- parallel-team regret from independent protagonist and antagonist learners;
- historical curriculum replay ranked by TD error and target proximity;
- a framework-independent training state machine with total interaction
  accounting;
- an HMACL/M2ALE backend for map-size curricula;
- deterministic unit tests and a runnable end-to-end smoke path.

The M2ALE adapter varies map size only. It does **not** yet reproduce the
paper's variable-agent Pursuit and Football experiments.

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

Results, dispatcher state, step accounting, and model checkpoints are written
under `runs/<experiment>/`. For a five-seed configuration, run
`scripts/run_matdd_m2ale.sh [qmix|ippo|mappo]` from the repository root.

## Reproducibility note

MATDD counts protagonist training, antagonist training, current-task
evaluation, and target-task evaluation interactions. HMACL's conservative
model rollback is not enabled in the recovered method; evaluate it separately
if it is reintroduced.
