from collections import defaultdict
import logging
import numpy as np

import wandb
import socket


project_name = "hmacl"


class Logger(logging.Logger):
    def __init__(self, name: str):
        super().__init__(name)
        self.console_logger = set_logger()
        self.stats = defaultdict(lambda: [])

        self.use_wandb = False
        self.use_tb = False

    def setup_wandb(self, args, wandb_logs_direc):
        self.wandb_run = wandb.init(
            config=args,
            project=project_name + "-" + args.env,
            group=args.wandb_group,
            entity=args.wandb_user_name,
            notes=socket.gethostname(),
            name=args.unique_token,
            dir=wandb_logs_direc,
            job_type=args.wandb_job_type,
            reinit=True
        )
        self.use_wandb = True

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_wandb:
            wandb.log({key: value}, step=t)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def set_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger


def get_logger():
    log = Logger(project_name)
    return log


