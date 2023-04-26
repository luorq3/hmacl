import argparse

from hmacl.algo import Algo
from hmacl.cpi import CPI
from hmacl.stg import STG


def main(args):

    env = args.env
    scenario = args.scenario

    stg = STG(env, scenario, **args)

    n_timesteps = args.n_timesteps
    t = 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
