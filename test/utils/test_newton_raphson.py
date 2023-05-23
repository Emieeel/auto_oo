#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

# import pytest

from pennylane import math
from torch.autograd.functional import jacobian, hessian
import auto_oo


def function(a, b, c):
    """
    a: m x n tensor
    b: m tensor
    c: n tensor
    """
    def f(x):
        return math.abs(math.transpose(c) @ x - math.sum(math.log(math.abs(b - a @ x))))
    return f


# def function(a, b, c):
#     def f(x):
#         return -4 * math.log(math.abs(x)) + math.abs(x) - 4 + 4 * math.log(4)
#     return f


def optimize(a, b, c, theta_init, max_iterations, conv_tol, verbose=1):
    opt = auto_oo.NewtonStep(verbose=verbose)

    cost = function(a, b, c)

    energy_init = cost(theta_init).item()
    if verbose is not None:
        print(f"iter = 000, energy = {energy_init:.12f}")

    theta_l = []
    energy_l = []
    hess_eig_l = []

    theta = theta_init
    for n in range(max_iterations):

        grad = jacobian(cost, theta)
        hess = hessian(cost, theta)

        theta_tup, hess_eig = opt.damped_newton_step(
            cost, (theta,), grad, hess
        )
        if math.shape(theta_tup) == (1):
            theta = theta_tup[0]
        else:
            theta = theta_tup

        hess_eig_l.append(hess_eig)

        theta_l.append(theta)

        # import pdb
        # pdb.set_trace()
        energy = cost(theta).item()
        energy_l.append(energy)

        if verbose is not None:
            print(f"iter = {n+1:03}, energy = {energy:.12f}")
        if n > 1:
            if abs(energy_l[-1] - energy_l[-2]) < conv_tol:
                if verbose is not None:
                    print("optimization finished.")
                    print("E_fin =", energy_l[-1])
                break

    return energy_l, theta_l, hess_eig_l


# def test_optimization(a, b, c, theta_init, max_iterations, conv_tol, verbose=1):
#     opt = auto_oo.NewtonStep(verbose=0)

#     energy_l, theta_l, hess_eig_l = optimize(a, b, c, theta_init, max_iterations, conv_tol)
#     pass


if __name__ == '__main__':
    import torch
    dim1 = 800
    dim2 = 140
    a = torch.rand(dim1, dim2)
    b = torch.rand(dim1)
    c = torch.rand(dim2)
    theta_init = torch.rand(dim2)
    # theta_init = torch.Tensor([1067950.])
    e_l, theta_l, hess_eig_l = optimize(a, b, c, theta_init, 100, 1e-12)

    import matplotlib.pyplot as plt
    plt.plot(e_l)
    plt.yscale('log')
    plt.show()
