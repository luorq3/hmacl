#!/usr/bin/env bash
set -euo pipefail

algorithm="${1:-qmix}"
target="${2:-60-20}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

case "${algorithm}" in
  qmix)
    buffer_size=5000
    student_gamma=0.99
    grad_norm_clip=10
    ;;
  ippo|mappo)
    buffer_size=32
    student_gamma=0.995
    grad_norm_clip=0.5
    ;;
  *)
    echo "Usage: $0 [qmix|ippo|mappo] [60-20|80-30]" >&2
    exit 2
    ;;
esac

case "${target}" in
  60-20)
    map_size=60
    agents=20
    ;;
  80-30)
    map_size=80
    agents=30
    ;;
  *)
    echo "Pursuit target must be 60-20 or 80-30" >&2
    exit 2
    ;;
esac

for seed in 1 2 3 4 5; do
  python -m hmacl.matdd_run \
    --name "${algorithm}" \
    --env pursuit \
    --runner parallel \
    --batch_size_run 16 \
    --batch_size 32 \
    --buffer_size "${buffer_size}" \
    --lr 0.0005 \
    --hidden_dim 256 \
    --student_gamma "${student_gamma}" \
    --student_grad_norm_clip "${grad_norm_clip}" \
    --state_encoder dyna \
    --target_update_interval_or_tau 2000 \
    --epsilon_anneal_time 1000000 \
    --target_map_size "${map_size}" \
    --target_agents "${agents}" \
    --designer ppo \
    --teacher_gamma 0.995 \
    --teacher_epochs 4 \
    --seed "${seed}" \
    --curriculum_iterations 80 \
    --steps_per_curriculum 50000 \
    --final_target_steps 50000 \
    --dispatcher_capacity 8 \
    --dispatcher_rho 0.5 \
    --dispatcher_temperature 0.1 \
    --eval_nepisode 32 \
    --use_wandb false \
    --exp "matdd-pursuit${target}-${algorithm}-seed${seed}"
done
