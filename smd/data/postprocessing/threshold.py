import numpy as np


def apply_threshold(output):
    return np.around(output).astype(int)
