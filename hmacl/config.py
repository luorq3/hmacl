from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser("HMACL")
    # required
    parser.add_argument("--env", type=str, default="m2ale", help="")
    parser.add_argument("--name", type=str, default="vdn", help="")
    # global setting
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--exp", type=str, default='exp', help="")
    parser.add_argument("--exp_exist_ok", default=True, action="store_false", help="")
    # hmacl
    parser.add_argument("--improve_type", type=str, default="hard", help="")
    parser.add_argument("--metrics", type=str, default="win_rate", help="")
    parser.add_argument("--loss_coef", type=float, default=0.5, help="")
    parser.add_argument("--temperature", type=float, default=1, help="")
    parser.add_argument("--save_cpi_model", default=True, action="store_false", help="")
    # algo
    parser.add_argument("--save_model", default=False, action="store_true", help="")
    parser.add_argument("--runner", type=str, default='episode', help="")
    parser.add_argument("--use_rnn", default=True, action="store_false", help="")
    # log and file path
    parser.add_argument("--wandb_user_name", type=str, default="luorq3", help="")
    parser.add_argument("--wandb_job_type", type=str, default="debug", help="")
    parser.add_argument("--use_wandb", default=False, action="store_true", help="")
    # env_args
    parser.add_argument("--env_args.obs_with_agent_id", default=True, action="store_false", help="")
    return parser
