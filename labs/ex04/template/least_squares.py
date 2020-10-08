# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

import costs


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    # ***************************************************
    
    
    "    w = np.linalg.solve(tx.T @ tx, tx.T @ y)\n",
    "    e = y - tx @ w\n",
    "    mse = e.T @ e / (2 * len(y))\n",
    "    return mse, w"
    A = tx.T.dot(tx)
    b = tx.T.dot(w)
    w = np.linalg.solve(A, b)
    return compute_mse(y, tx, w), w
