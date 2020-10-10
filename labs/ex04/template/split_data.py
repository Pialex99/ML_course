# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    # split the data based on the given ratio
    split = int(ratio * len(y))
    indexes = np.random.permutation(len(y))
    
    idx_tr = indexes[:split]
    idx_te = indexes[split:]
    
    return x[idx_tr], x[idx_te], y[idx_tr], y[idx_te]
