#!/usr/bin/env bash
set -euo pipefail

algorithm="${1:-qmix}"
target_agents="${2:-18}"

case "${algorithm}" in
  qmix)
    runner="parallel"
    buffer_size=5000
    student_gamma=0.99
    grad_norm_clip=10
    ;;
  ippo|mappo)
    runner="parallel"
    buffer_size=32
    student_gamma=0.995
    grad_norm_clip=0.5
    ;;
  *)
    echo "Usage: $0 [qmix|ippo|mappo] [18|25]" >&2
    exit 2
    ;;
esac

if [[ "${target_agents}" != "18" && "${target_agents}" != "25" ]]; then
  echo "Football target must be 18 or 25 agents" >&2
  exit 2
fi

for seed in 1 2 3 4 5; do
  python -m hmacl.matdd_run \
    --name "${algorithm}" \
    --env vmas \
    --runner "${runner}" \
    --batch_size_run 16 \
    --batch_size 32 \
    --buffer_size "${buffer_size}" \
    --lr 0.0005 \
    --hidden_dim 256 \
    --student_gamma "${student_gamma}" \
    --student_grad_norm_clip "${grad_norm_clip}" \
    --target_update_interval_or_tau 2000 \
    --epsilon_anneal_time 1000000 \
    --target_agents "${target_agents}" \
    --designer ppo \
    --teacher_gamma 0.999 \
    --teacher_epochs 5 \
    --seed "${seed}" \
    --curriculum_iterations 80 \
    --steps_per_curriculum 50000 \
    --final_target_steps 50000 \
    --dispatcher_capacity 8 \
    --dispatcher_rho 0.5 \
    --dispatcher_temperature 0.1 \
    --eval_nepisode 32 \
    --use_wandb false \
    --exp "matdd-football${target_agents}-${algorithm}-seed${seed}"
done
