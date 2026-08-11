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
9. Pursuit uses two integer free parameters: square map size and pursuer count.
   The current SISL environment is rebuilt between curricula, while a padded
   target interface keeps the agent slots and spatial state shape fixed.
   Evaders remain at the historical default of 30 unless
   `scale_n_evaders=true` is set explicitly.
10. Pursuit students use two valid 3x3, 16-channel convolutions over each
    7x7x3 observation, producing 144 spatial features before the same
    `FC(256) -> GRU(256) -> FC(128)` stack. For variable teams, `dyna_qmix`
    conditions its monotonic mixer on a masked mean of shared observation
    embeddings; `dyna_cv_critic` combines each agent's embedding with the
    masked mean embedding of its teammates.
11. The comparison strategies share the same student backend and task space.
    LI interpolates all normalized parameters linearly, DR samples each
    parameter uniformly, Minimax trains the PPO teacher with negative
    protagonist return, and Direct always trains on the target. Only MATDD's
    parallel-team-regret objective trains an antagonist. The dispatcher is
    enabled for LI, DR, Minimax, and MATDD in full comparison runs.
12. Dispatcher ablations map directly to the paper: `tv_sr` uses configurable
    `rho` (0.5 by default), `tv` forces `rho=0`, `sr` forces `rho=1`, and
    `none` disables replay while continuing to design curricula.

The run entry point saves every trained component alongside `matdd_result.json`.
MATDD stores protagonist, antagonist, and PPO teacher checkpoints; Minimax
omits the antagonist, and rule-based strategies omit the teacher checkpoint.

## Known gaps

- The manuscript did not define the teacher observation, replay eviction, or
  PPO rollout/update schedule. The choices above are explicit recovery
  decisions and must be reported as such.
- `M2ALETaskAdapter` remains a map-size integration environment. Football and
  Pursuit support target-shaped padding, active-agent loss masks, dynamic
  parallel-worker updates, and all target interfaces reported in the paper.
- PettingZoo does not expose a public Pursuit `state()` implementation. The
  adapter therefore reads the current SISL model's first three spatial state
  channels. This dependency on a PettingZoo internal is isolated in
  `hmacl/algo/envs/pursuit.py` and covered by an interface test.
- The manuscript specifies shared DyNA extraction and fixed-size aggregation,
  but not the aggregation operator or embedding widths. Masked mean pooling,
  a 128-unit encoder, and a 64-dimensional embedding are explicit recovery
  choices. `--state_encoder padded` retains the earlier global-state path for
  ablation.
- The exact historical VMAS release is unknown. Recovery targets VMAS 1.5.x;
  its Boolean `dense_reward` option replaces the removed `dense_reward_ratio`.
- The legacy environments and action selectors use process-global random
  generators. Runs are deterministic for a fixed command and seed, but fully
  isolated per-role RNG streams require a later environment refactor.
- The precise historical LI step schedule and Minimax teacher observation were
  not retained. Linear interpolation over curriculum progress and the same PPO
  observation used by MATDD are explicit, controlled recovery choices.
