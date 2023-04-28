from argparse import ArgumentParser


parser = ArgumentParser("HMACL", add_help=False)
parser.add_argument("--env", type=str, default="m2ale", help="")
parser.add_argument("--name", type=str, default="vdn_ns", help="")
parser.add_argument("--improve_type", type=str, default="hard", help="")
parser.add_argument("--metrics", type=str, default="win_rate", help="")
parser.add_argument("--loss_coef", type=float, default=0.5, help="")
parser.add_argument("--temperature", type=float, default=1, help="")
parser.add_argument("--exp", type=str, default='exp', help="")
parser.add_argument("--save_model", type=bool, default=True, help="")
parser.add_argument("--save_cpi_model", type=bool, default=True, help="")
