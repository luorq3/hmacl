from run import search
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

ray.init()

search_space = {
    "project": ["tune-search"],
    "name": ["vdn_ns"],
    "loss_coef": [0.1, 0.2, 0.3],
    "temperature": tune.quniform(0.0, 0.7, 0.1),
    "buffer_size": [2000, 5000],
    "lr": tune.choice([0.0001, 0.0003, 0.0005]),
    "hidden_dim": tune.choice([128, 64]),
    "epsilon_anneal_time": tune.choice([100000, 200000, 300000]),
    "t_update": [10000, 20000, 30000, 40000],
    "t_max": [2040000],
    "wandb_job_type": ["tune-search"],
    "use_wandb": [True]
}

# trainable_with_cpu_gpu = tune.with_resources(search, {"cpu": 20, "gpu": 1})
pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric="win_rate",
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations=search_space
)

tuner = tune.Tuner(
    search,
    tune_config=tune.TuneConfig(
        num_samples=50,
        scheduler=pbt_scheduler
    )
)

results = tuner.fit()
