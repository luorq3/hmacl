#!/usr/bin/env bash
set -euo pipefail

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

for target in 60-20 80-30; do
  case "${target}" in
    60-20)
      map_size=60
      agents=20
      ;;
    80-30)
      map_size=80
      agents=30
      ;;
  esac

  for algorithm in qmix ippo mappo; do
    if [[ "${algorithm}" == "qmix" ]]; then
      student_gamma=0.99
      grad_norm_clip=10
    else
      student_gamma=0.995
      grad_norm_clip=0.5
    fi

    python -m hmacl.matdd_run \
      --name "${algorithm}" \
      --env pursuit \
      --runner parallel \
      --batch_size_run 2 \
      --batch_size 2 \
      --buffer_size 4 \
      --lr 0.0005 \
      --student_gamma "${student_gamma}" \
      --student_grad_norm_clip "${grad_norm_clip}" \
      --state_encoder dyna \
      --designer ppo \
      --teacher_update_interval 2 \
      --teacher_epochs 1 \
      --teacher_minibatch_size 2 \
      --teacher_gamma 0.995 \
      --seed 81 \
      --target_map_size "${map_size}" \
      --target_agents "${agents}" \
      --episode_limit 6 \
      --curriculum_iterations 2 \
      --steps_per_curriculum 12 \
      --final_target_steps 12 \
      --curriculum_eval_episodes 2 \
      --eval_nepisode 2 \
      --use_wandb false \
      --save_model false \
      --exp "matdd-pursuit${target}-${algorithm}-short" \
      --exp_exist_ok true
  done
done
