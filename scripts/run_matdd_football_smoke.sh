#!/usr/bin/env bash
set -euo pipefail

for algorithm in qmix ippo mappo; do
  if [[ "${algorithm}" == "qmix" ]]; then
    runner="episode"
    batch_size_run=1
    batch_size=1
    student_gamma=0.99
    grad_norm_clip=10
  else
    runner="parallel"
    batch_size_run=2
    batch_size=2
    student_gamma=0.995
    grad_norm_clip=0.5
  fi

  python -m hmacl.matdd_run \
    --name "${algorithm}" \
    --env vmas \
    --runner "${runner}" \
    --batch_size_run "${batch_size_run}" \
    --batch_size "${batch_size}" \
    --lr 0.0005 \
    --student_gamma "${student_gamma}" \
    --student_grad_norm_clip "${grad_norm_clip}" \
    --designer ppo \
    --teacher_update_interval 2 \
    --teacher_epochs 1 \
    --teacher_minibatch_size 2 \
    --teacher_gamma 0.999 \
    --seed 61 \
    --target_agents 18 \
    --episode_limit 12 \
    --curriculum_iterations 2 \
    --steps_per_curriculum 24 \
    --final_target_steps 24 \
    --curriculum_eval_episodes "${batch_size_run}" \
    --eval_nepisode "${batch_size_run}" \
    --use_wandb false \
    --save_model false \
    --exp "matdd-football18-${algorithm}-short" \
    --exp_exist_ok true
done

python -m hmacl.matdd_run \
  --name qmix \
  --env vmas \
  --runner episode \
  --batch_size_run 1 \
  --batch_size 1 \
  --lr 0.0005 \
  --student_gamma 0.99 \
  --student_grad_norm_clip 10 \
  --designer random \
  --seed 62 \
  --target_agents 25 \
  --episode_limit 12 \
  --curriculum_iterations 1 \
  --steps_per_curriculum 12 \
  --final_target_steps 12 \
  --curriculum_eval_episodes 1 \
  --eval_nepisode 1 \
  --use_wandb false \
  --save_model false \
  --exp matdd-football25-qmix-short \
  --exp_exist_ok true
