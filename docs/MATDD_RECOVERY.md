# MATDD Recovery Specification

This implementation reconstructs MATDD from the ECAI manuscript because the
original experiment code is unavailable. It preserves the existing HMACL code
and places the recovered algorithm under `hmacl/matdd/`.

## Executable interpretation

1. A `TaskParameterSpace` maps explicitly bounded environment parameters to
   `[0, 1]^d`. Generated curricula are clipped component-wise to the target.
2. At curriculum iteration `i`, replay is selected with probability `k / K`,
   where `k` is the dispatcher pool size and `K` is its capacity. An empty pool
   always invokes the designer.
3. The PPO designer observes the current and target task vectors, normalized
   training progress, target-task return, and pool fill ratio. Its bounded Beta
   policy emits the next task vector.
4. Protagonist and antagonist train independently on the same curriculum and
   with the same interaction budget. Designer reward is parallel-team regret:
   `return_antagonist - return_protagonist` on that curriculum.
5. The dispatcher retains curricula after replay. Its distribution mixes the
   descending rank of mean absolute TD error and the ascending rank of
   normalized Euclidean distance to the target. FIFO eviction is used when the
   pool is full.
6. The protagonist is evaluated on the target after every curriculum, and all
   protagonist, antagonist, and evaluation interactions are counted. HMACL's
   conservative rollback is deliberately excluded; it should be a separate
   ablation rather than an undocumented part of MATDD.
7. Football uses one integer free parameter. It controls the blue learners and
   scales the heuristic red team symmetrically. VMAS worlds are rebuilt between
   curricula, while observations, state, actions, and agent slots remain fixed
   at the configured 18-agent or 25-agent target interface.
8. Football students default to the manuscript's `FC(256) -> GRU(256) ->
   FC(128) -> output` policy network. `--student_architecture legacy` retains
   the original single-width HMACL agent for controlled comparisons.

The run entry point saves protagonist, antagonist, and PPO teacher checkpoints
alongside `matdd_result.json` so the recovered training state is inspectable.

## Known gaps

- The manuscript did not define the teacher observation, replay eviction, or
  PPO rollout/update schedule. The choices above are explicit recovery
  decisions and must be reported as such.
- `M2ALETaskAdapter` remains a map-size integration environment. VMAS Football
  now supports variable agent counts, target-shaped padding, active-agent loss
  masks, dynamic parallel-worker updates, return/win-rate metrics, and both
  paper target sizes.
- SISL/PettingZoo Pursuit has not been restored. Its grid-size and pursuer-count
  task space, image observations, and convolutional encoder remain necessary
  for the Pursuit-60-20 and Pursuit-80-30 experiments.
- The manuscript describes DyNA aggregation for changing team sizes. The
  recovered implementation currently uses target-shaped zero padding plus
  active-agent masks, which is testable but is not an implementation of DyNA.
- The exact historical VMAS release is unknown. Recovery targets VMAS 1.5.x;
  its Boolean `dense_reward` option replaces the removed `dense_reward_ratio`.
- The legacy environments and action selectors use process-global random
  generators. Runs are deterministic for a fixed command and seed, but fully
  isolated per-role RNG streams require a later environment refactor.
