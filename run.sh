#!/bin/bash

name=vdn_ns
env=m2ale
use_wandb=True

for seed in {1..3}
do
  echo "Started with seed=${seed}."
  python hmacl/run.py --name=$name --env=$env --seed="$seed" --use_wandb=$use_wandb
  echo "Done with seed=${seed}."
done
