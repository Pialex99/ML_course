# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    N, D = tx.shape
    A = tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    return w
