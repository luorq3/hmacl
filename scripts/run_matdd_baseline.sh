#!/usr/bin/env bash
set -euo pipefail

strategy="${1:-li}"
algorithm="${2:-qmix}"
benchmark="${3:-pursuit60-20}"
dispatcher_mode="${4:-tv_sr}"

case "${strategy}" in
  matdd|li|dr|minimax|direct) ;;
  *)
    echo "Usage: $0 [matdd|li|dr|minimax|direct] [qmix|ippo|mappo] [pursuit60-20|pursuit80-30|football18|football25] [tv_sr|tv|sr|none]" >&2
    exit 2
    ;;
esac

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
    echo "Student must be qmix, ippo, or mappo" >&2
    exit 2
    ;;
esac

case "${dispatcher_mode}" in
  tv_sr|tv|sr|none) ;;
  *)
    echo "Dispatcher mode must be tv_sr, tv, sr, or none" >&2
    exit 2
    ;;
esac

target_args=()
case "${benchmark}" in
  pursuit60-20)
    environment=pursuit
    target_args=(--target_map_size 60 --target_agents 20)
    teacher_gamma=0.995
    teacher_epochs=4
    ;;
  pursuit80-30)
    environment=pursuit
    target_args=(--target_map_size 80 --target_agents 30)
    teacher_gamma=0.995
    teacher_epochs=4
    ;;
  football18)
    environment=vmas
    target_args=(--target_agents 18)
    teacher_gamma=0.999
    teacher_epochs=5
    ;;
  football25)
    environment=vmas
    target_args=(--target_agents 25)
    teacher_gamma=0.999
    teacher_epochs=5
    ;;
  *)
    echo "Unknown benchmark: ${benchmark}" >&2
    exit 2
    ;;
esac

if [[ "${strategy}" == "direct" ]]; then
  dispatcher_mode=none
fi

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

for seed in 1 2 3 4 5; do
  python -m hmacl.matdd_run \
    --name "${algorithm}" \
    --env "${environment}" \
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
    --designer "${strategy}" \
    --dispatcher_mode "${dispatcher_mode}" \
    --teacher_gamma "${teacher_gamma}" \
    --teacher_epochs "${teacher_epochs}" \
    --seed "${seed}" \
    --curriculum_iterations 80 \
    --steps_per_curriculum 50000 \
    --final_target_steps 50000 \
    --dispatcher_capacity 8 \
    --dispatcher_rho 0.5 \
    --dispatcher_temperature 0.1 \
    --eval_nepisode 32 \
    --use_wandb false \
    --exp "${strategy}-${benchmark}-${algorithm}-${dispatcher_mode}-seed${seed}" \
    "${target_args[@]}"
done
