#!/usr/bin/env bash
set -euo pipefail

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

for strategy in li dr minimax direct; do
  python -m hmacl.matdd_run \
    --name qmix \
    --env pursuit \
    --runner parallel \
    --batch_size_run 2 \
    --batch_size 2 \
    --buffer_size 4 \
    --lr 0.0005 \
    --student_gamma 0.99 \
    --student_grad_norm_clip 10 \
    --state_encoder dyna \
    --designer "${strategy}" \
    --dispatcher_mode none \
    --teacher_update_interval 2 \
    --teacher_epochs 1 \
    --teacher_minibatch_size 2 \
    --teacher_gamma 0.995 \
    --seed 91 \
    --target_map_size 60 \
    --target_agents 20 \
    --episode_limit 6 \
    --curriculum_iterations 2 \
    --steps_per_curriculum 12 \
    --final_target_steps 12 \
    --eval_nepisode 2 \
    --use_wandb false \
    --save_model false \
    --exp "baseline-${strategy}-pursuit60-20-qmix-short" \
    --exp_exist_ok true
done

python -m hmacl.matdd_run \
  --name qmix \
  --env vmas \
  --runner parallel \
  --batch_size_run 2 \
  --batch_size 2 \
  --buffer_size 4 \
  --lr 0.0005 \
  --student_gamma 0.99 \
  --student_grad_norm_clip 10 \
  --state_encoder dyna \
  --designer minimax \
  --dispatcher_mode none \
  --teacher_update_interval 2 \
  --teacher_epochs 1 \
  --teacher_minibatch_size 2 \
  --teacher_gamma 0.999 \
  --seed 93 \
  --target_agents 18 \
  --episode_limit 6 \
  --curriculum_iterations 2 \
  --steps_per_curriculum 12 \
  --final_target_steps 12 \
  --eval_nepisode 2 \
  --use_wandb false \
  --save_model false \
  --exp baseline-minimax-football18-qmix-short \
  --exp_exist_ok true

for dispatcher_mode in tv sr none; do
  python -m hmacl.matdd_run \
    --name qmix \
    --env pursuit \
    --runner parallel \
    --batch_size_run 2 \
    --batch_size 2 \
    --buffer_size 4 \
    --state_encoder dyna \
    --designer matdd \
    --dispatcher_mode "${dispatcher_mode}" \
    --dispatcher_capacity 1 \
    --teacher_update_interval 2 \
    --teacher_epochs 1 \
    --teacher_minibatch_size 2 \
    --seed 92 \
    --target_map_size 60 \
    --target_agents 20 \
    --episode_limit 6 \
    --curriculum_iterations 2 \
    --steps_per_curriculum 12 \
    --final_target_steps 12 \
    --eval_nepisode 2 \
    --use_wandb false \
    --save_model false \
    --exp "ablation-${dispatcher_mode}-pursuit60-20-qmix-short" \
    --exp_exist_ok true
done
