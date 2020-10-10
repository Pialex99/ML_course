# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

from costs import compute_mse


def least_squares(y, tx):
    """calculate the least squares."""
    
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    return compute_mse(y, tx, w), w
