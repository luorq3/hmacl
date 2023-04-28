#!/bin/bash

name=vdn_ns
env=m2ale
use_wandb=True
use_rnn=True
wandb_job_type=Training

for seed in {1..3}
do
  echo "Started with seed=${seed}."
  # python hmacl/run.py --name=vdn_ns --env=m2ale --seed=3 --use_wandb=True --wandb_job_type=Training --use_rnn=True
  python hmacl/run.py --name=$name --env=$env --seed="$seed" --use_wandb=$use_wandb --wandb_job_type=$wandb_job_type --use_rnn=$use_rnn
  echo "Done with seed=${seed}."
done
