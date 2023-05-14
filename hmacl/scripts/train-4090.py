import copy
import datetime
import multiprocessing

from run import search


_CPU_COUNT = min(multiprocessing.cpu_count() - 1, 8)

num_seeds = 20
num_parallel = min(num_seeds, _CPU_COUNT)

config = {
    # required
    "project": "exp-v6",
    "name": "vdn",
    "env": "m2ale",
    # experiment name and dir
    "exp": datetime.datetime.now().strftime('%m-%d_%H-%M-%S'),
    "exp_exist_ok": True,
    # hmacl
    "loss_coef": 0.1,
    "temperature": 0.1,
    # time
    "t_max": 4000000,
    "t_update": 30000,
    # algo
    "buffer_size": 2000,
    "lr": 0.0005,
    "hidden_dim": 128,
    "epsilon_anneal_time": 200000,
    "use_rnn": True,
    # log and file path
    "wandb_job_type": "Training",
    "use_wandb": True
}

configs = []
for i in range(1, 1 + num_seeds):
    config_ = copy.deepcopy(config)
    config_["seed"] = i
    configs.append(config_)


start = 0
num_config = len(configs)
num_iters = num_seeds // num_parallel if num_seeds % num_parallel == 0 else num_seeds // num_parallel + 1
for iter in range(num_iters):
    end = min(num_config, start + num_parallel)
    with multiprocessing.Pool(processes=num_parallel) as p:
        p.map(search, configs[start: end])
    start = end

