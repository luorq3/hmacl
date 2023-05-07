#!/bin/bash

# required
project="exp-v5"
name=vdn_ns
env=m2ale
# experiment name and dir
exp=$(date +%Y-%m-%d_%T)
exp_exist_ok=True
# hmacl
loss_coef=0.1
temperature=0.1
# time
t_max=4000000
t_update=20000
# algo
buffer_size=5000
lr=0.0005
hidden_dim=128
epsilon_anneal_time=100000
use_rnn=True
# log and file path
wandb_job_type=Training
use_wandb=True


for seed in {1..5}
do
  echo "Started with seed=${seed}."
  echo hmacl/run.py --project=$project --name=$name --env=$env --seed="$seed" --exp="$exp" --exp_exist_ok=$exp_exist_ok --loss_coef=$loss_coef --temperature=$temperature --t_max=$t_max --t_update=$t_update --buffer_size=$buffer_size --lr=$lr --hidden_dim=$hidden_dim --epsilon_anneal_time=$epsilon_anneal_time --use_rnn=$use_rnn --wandb_job_type=$wandb_job_type --use_wandb=$use_wandb
  echo "Done with seed=${seed}."
done
