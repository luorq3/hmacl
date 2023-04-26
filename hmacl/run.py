
import numpy as np


class Task:

    def __init__(self, task_id, env_name, scenario, **kwargs):
        self.id = task_id
        self.env = env_name
        self.scenario = scenario
        for k, w in kwargs.items():
            setattr(self, k, w)


def main():
    pass


if __name__ == "__main__":
    main()
