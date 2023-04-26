import os
from argparse import Namespace
from copy import deepcopy

import yaml


def load_scenario_conf(config):
    scenario = config.scenario
    conf_path = os.path.join(os.path.dirname(__file__), 'scenarios', scenario + '.yml')
    dict_conf = load_dict_config(conf_path)
    for k, v in vars(config).items():
        dict_conf[k] = v
    return dict_to_ns_recursively(dict_conf)

def load_dict_config(file_path):
    with open(file_path) as s:
        conf = yaml.safe_load(s)
    return conf


def dict_to_ns_recursively(args):
    assert isinstance(args, dict), "args should be a dict or it's subclass"
    new_args = deepcopy(args)
    for k, v in args.items():
        if isinstance(v, dict):
            new_args[k] = dict_to_ns_recursively(v)

    return Namespace(**new_args)
