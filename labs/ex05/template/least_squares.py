# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    mse = np.mean(e ** 2) / 2
    return mse, w
