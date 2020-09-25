# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx @ w
    return - tx.T @ e / len(y)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for i in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size): 
            w = w - gamma * compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        ws.append(w)
    return losses, ws