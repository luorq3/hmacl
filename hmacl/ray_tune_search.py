from run import search
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

ray.init()

search_space = {
    "project": "tune-search",
    "name": tune.grid_search(["vdn", "vdn_ns"]),
    "loss_coef": tune.choice([0.1, 0.2, 0.3]),
    "temperature": tune.quniform(0.0, 0.7, 0.1),
    "buffer_size": tune.grid_search([2000, 5000]),
    "lr": tune.choice([0.0001, 0.0003, 0.0005]),
    "hidden_dim": tune.choice([128, 64]),
    "epsilon_anneal_time": tune.choice([100000, 200000, 300000]),
    "t_update": tune.choice([1000, 20000, 30000, 40000]),
    "t_max": 2040000,
    "wandb_job_type": "tune-search",
    "use_wandb": True
}

tuner = tune.Tuner(
    search,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=50,
        scheduler=ASHAScheduler(
            metric="win_rate",
            mode="max"
        )
    )
)

results = tuner.fit()
