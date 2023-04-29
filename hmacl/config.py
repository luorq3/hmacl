import random
from distutils.util import strtobool
from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser("HMACL")
    # required
    parser.add_argument("--project", type=str, default="hmacl", help="", required=True)
    parser.add_argument("--env", type=str, default="m2ale", help="", required=True)
    parser.add_argument("--name", type=str, default="vdn", help="", required=True)
    # experiment name and dir
    parser.add_argument("--seed", type=int, default=random.randint(1000, 99999), help="")
    parser.add_argument("--exp", type=str, default='exp', help="")
    parser.add_argument("--exp_exist_ok", default=False, type=lambda x: bool(strtobool(x)), help="")
    # hmacl
    parser.add_argument("--improve_type", type=str, default="hard", help="")
    parser.add_argument("--metrics", type=str, default="win_rate", help="")
    parser.add_argument("--loss_coef", type=float, default=0.5, help="")
    parser.add_argument("--temperature", type=float, default=1, help="")
    parser.add_argument("--save_cpi_model", default=True, type=lambda x: bool(strtobool(x)), help="")
    # time
    parser.add_argument("--t_max", type=int, default=2040000, help="")
    parser.add_argument("--t_update", type=int, default=5000, help="")
    parser.add_argument("--test_nepisode", type=int, default=100, help="")
    parser.add_argument("--eval_nepisode", type=int, default=100, help="")
    parser.add_argument("--test_interval", type=int, default=5000, help="")
    parser.add_argument("--eval_interval", type=int, default=5000, help="")
    parser.add_argument("--log_interval", type=int, default=25000, help="")
    parser.add_argument("--runner_log_interval", type=int, default=25000, help="")
    parser.add_argument("--learner_log_interval", type=int, default=25000, help="")
    parser.add_argument("--save_model_interval", type=int, default=25000, help="")
    # algo
    parser.add_argument("--save_model", default=False, type=lambda x: bool(strtobool(x)), help="")
    parser.add_argument("--runner", type=str, default='episode', help="")
    parser.add_argument("--buffer_size", type=int, default=5000, help="")
    parser.add_argument("--epsilon_anneal_time", type=int, default=100000, help="")
    parser.add_argument("--hidden_dim", type=int, default=128, help="")
    parser.add_argument("--lr", type=float, default=0.0003, help="")
    parser.add_argument("--evaluation_epsilon", type=float, default=0.05, help="")
    parser.add_argument("--target_update_interval_or_tau", type=float, default=200, help="")
    parser.add_argument("--use_rnn", default=True, type=lambda x: bool(strtobool(x)), help="")
    parser.add_argument("--standardise_rewards", default=True, type=lambda x: bool(strtobool(x)), help="")
    # log and file path
    parser.add_argument("--wandb_user_name", type=str, default="luorq3", help="")
    parser.add_argument("--wandb_job_type", type=str, default="debug", help="")
    parser.add_argument("--use_wandb", default=True, type=lambda x: bool(strtobool(x)), help="")
    # env_args
    parser.add_argument("--env_args.scenario", type=str, default="5u_vs_5u", help="")
    parser.add_argument("--env_args.time_limit", type=int, default=100, help="")
    parser.add_argument("--env_args.camp", type=str, default="offensive", help="")
    parser.add_argument("--env_args.reward_type", type=str, default="dense", help="")
    # parser.add_argument("--env_args.map_size", type=list, default=[8, 8], nargs=2, help="")
    # parser.add_argument("--env_args.tag_map_size", type=list, default=[40, 40], nargs=2, help="")
    parser.add_argument("--env_args.expand_d", type=int, default=2, help="")
    parser.add_argument("--env_args.pos_init_type", type=str, default="random", help="")

    return parser
