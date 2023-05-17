from run import search
# from ray.air import session
import ray
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining


# def search(config):
#     print(config)
#     session.report({"win_rate": 0.5},
#                    checkpoint=air.Checkpoint.from_dict({
#                        "step": 0,
#                        "model_state_dict": None,
#                        "optimizer_state_dict": None,
#                    }))
#     print(session.get_checkpoint())


if ray.is_initialized():
    ray.shutdown()
ray.init()

search_space = {
    "project": ["search-cpi"],
    "name": ["vdn_cs"],
    "env": ["m2ale"],
    "exp": ["vdn_cs-5u_vs_5u"],
    "loss_coef": [0.05, 0.1, 0.2],
    "temperature": [0.05, 0.1, 0.15],
    "buffer_size": [2000, 3000, 4000],
    "lr": [0.0003, 0.0005, 0.0006],
    "hidden_dim": [128],
    "epsilon_anneal_time": tune.choice([100000, 200000, 300000]),
    "t_update": [20000, 30000, 40000],
    "t_max": [2040000],
    "wandb_job_type": ["tune-search"],
    "use_wandb": [False],
    "use_rnn": [False],
    "use_cuda": [False],
    "tuning": [True]
}

# trainable_with_cpu_gpu = tune.with_resources(search, {"cpu": 20, "gpu": 1})
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="win_rate",
    mode="max",
    perturbation_interval=5,
    hyperparam_mutations=search_space
)

tuner = tune.Tuner(
    search,
    run_config=air.RunConfig(
        name="pbt_vdn_cs-5u_vs_5u",
        stop={"win_rate": 0.7},
        verbose=1,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_score_attribute="win_rate",
            num_to_keep=8
        ),
        local_dir="/home/luorq3/ray_results"
    ),
    tune_config=tune.TuneConfig(
        num_samples=500,
        scheduler=pbt_scheduler
    )
)

results = tuner.fit()
