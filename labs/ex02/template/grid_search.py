# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
from ex02 import costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i0, w0_tmp in enumerate(w0) :
        for i1, w1_tmp in enumerate(w1) :
            losses[i0, i1] = costs.compute_loss(y, tx, np.array([w0_tmp, w1_tmp]))
    return losses
