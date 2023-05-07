import copy
import datetime
import multiprocessing

from run import search


_CPU_COUNT = multiprocessing.cpu_count() - 1

cpus = _CPU_COUNT
num_seeds = 5

config = {
    # required
    "project": "exp-v5",
    "name": "vdn_ns",
    "env": "m2ale",
    # experiment name and dir
    "exp": datetime.datetime.now().strftime('%m-%d_%H-%M-%s'),
    "exp_exist_ok": True,
    # hmacl
    "loss_coef": 0.1,
    "temperature": 0.1,
    # time
    "t_max": 4000000,
    "t_update": 20000,
    # algo
    "buffer_size": 5000,
    "lr": 0.0005,
    "hidden_dim": 128,
    "epsilon_anneal_time": 100000,
    "use_rnn": True,
    # log and file path
    "wandb_job_type": "Training",
    "use_wandb": True
}

configs = []
for i in range(1, 6):
    config_ = copy.deepcopy(config)
    config_["seed"] = i
    configs.append(config_)
with multiprocessing.Pool(processes=cpus) as p:
    p.map(search, configs)

