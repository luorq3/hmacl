import argparse

from hmacl.algo import Algo
from hmacl.cpi import CPI
from hmacl.stg import STG
from hmacl.utils.logging_ import get_logger


def main(args):
    _log = get_logger()

    cur_task = 0
    target_task = None

    stg = STG(args.env, args.scenario, **args)
    cpi = CPI(args.target_task, args.ps_type, args.update_type, **args)

    n_timesteps = args.n_timesteps
    t = 0
    # steps_per_algo
    steps_per_algo = None
    while t < n_timesteps:
        loss = Algo.run(**args)
        metrics = cpi.improve()
        next_task = stg.generate(cur_task, metrics, loss)
        cur_task = next_task
        t += steps_per_algo

    # Train on target_task
    Algo.run(**args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
