#!/usr/bin/env bash
set -euo pipefail

algorithm="${1:-qmix}"
case "${algorithm}" in
  qmix)
    runner="episode"
    batch_size_run=1
    batch_size=64
    ;;
  ippo|mappo)
    runner="parallel"
    batch_size_run=10
    batch_size=10
    ;;
  *)
    echo "Usage: $0 [qmix|ippo|mappo]" >&2
    exit 2
    ;;
esac

for seed in 1 2 3 4 5; do
  python -m hmacl.matdd_run \
    --name "${algorithm}" \
    --env m2ale \
    --runner "${runner}" \
    --batch_size_run "${batch_size_run}" \
    --batch_size "${batch_size}" \
    --designer ppo \
    --seed "${seed}" \
    --curriculum_iterations 80 \
    --steps_per_curriculum 50000 \
    --final_target_steps 50000 \
    --dispatcher_capacity 8 \
    --dispatcher_rho 0.5 \
    --dispatcher_temperature 0.1 \
    --eval_nepisode 32 \
    --use_wandb false \
    --exp "matdd-${algorithm}-seed${seed}"
done
