import numpy as np


def merge_visible_units(vu1, vu2):
    return np.logical_or(vu1, vu2).astype(np.float32)
