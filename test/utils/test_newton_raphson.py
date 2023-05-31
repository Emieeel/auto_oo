#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import auto_oo
from torch.autograd.functional import jacobian, hessian
import pytest


from pennylane import math
import torch
from jax.config import config

torch.set_default_dtype(torch.float64)
config.update("jax_enable_x64", True)


def vec_to_expmat(x):
    xmat = auto_oo.oo_energy.vector_to_skew_symmetric(x)
    return math.expm(-xmat)


def function_type_a(a):
    va, wa = math.linalg.eigh(a)

    def f(x):
        expmat = vec_to_expmat(x)
        zeromat = (expmat.T @ a @ expmat - math.diag(va))
        return math.sum(zeromat**2)
    return f


def function_type_b(t):
    def f(x):
        cost = -t * math.log(math.abs(x)) + math.abs(x) - t + t * math.log(t)
        return math.sum(cost)
    return f


def optimize(cost, params_init, max_iterations, conv_tol, verbose=1, **nr_kwargs):
    opt = auto_oo.NewtonStep(verbose=verbose, **nr_kwargs)

    energy_init = cost(*params_init).item()
    if verbose is not None:
        print(f"iter = 000, energy = {energy_init:.12f}")

    theta_l = [params_init]
    energy_l = [energy_init]
    hess_eig_l = ['nan']

    theta = params_init
    for n in range(max_iterations):

        grad = jacobian(cost, theta)
        hess = hessian(cost, theta)

        if len(params_init) > 1:
            grad = math.concatenate((grad[0], grad[1]))
            hess = math.concatenate(
                (
                    math.concatenate((hess[0][0], hess[0][1]), axis=1),
                    math.concatenate((hess[1][0], hess[1][1]), axis=1),
                ),
                axis=0,
            )
        else:
            grad = grad[0]
            hess = hess[0][0]

        newp, hess_eig = opt.damped_newton_step(
            cost, theta, grad, hess
        )

        hess_eig_l.append(hess_eig)

        theta = (newp,)

        theta_l.append(theta)

        energy = cost(*theta).item()
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


@pytest.mark.parametrize(
    ("dim", "max_iterations", "conv_tol", "lambda_min", "rho", "mu"),
    [(2, 20, 1e-12, 1e-6, 2, 1e-4),
     (4, 20, 1e-12, 1e-6, 2, 1e-4),
     (8, 50, 1e-10, 1e-6, 3, 1e-4),
     (16, 200, 1e-10, 1e-6, 5, 1e-5)],
)
def test_optimization_type_a(dim, max_iterations, conv_tol, lambda_min, rho, mu, verbose=1):
    a = (torch.rand(dim, dim)-.5)
    a = math.transpose(a) + a
    theta_init = 0.00001 * (torch.rand((dim*(dim-1)//2))-.5)
    energy_l, theta_l, hess_eig_l = optimize(
        function_type_a(a), (theta_init,), max_iterations, conv_tol, aug=True,
        lambda_min=lambda_min, rho=rho, mu=mu)
    assert math.allclose(energy_l[-1], 0.)

    va, wa = math.linalg.eigh(a)
    finmat = vec_to_expmat(theta_l[-1][0])
    assert math.allclose(finmat.T @ a @ finmat, math.diag(va))


@pytest.mark.parametrize(
    ("t", "max_iterations", "conv_tol"),
    [(4, 10, 1e-12),
     (3, 10, 1e-12),
     (0.00004, 100, 1e-12)],
)
def test_optimization_type_b(t, max_iterations, conv_tol, verbose=1):
    theta_init = torch.Tensor([10])
    energy_l, theta_l, hess_eig_l = optimize(
        function_type_b(t), (theta_init,), max_iterations, conv_tol, aug=False)
    assert math.allclose(energy_l[-1], 0.)


if __name__ == '__main__':
    torch.set_num_threads(12)
    max_iterations = 10000
    conv_tol = 1e-12
    dim = 2
    a = (torch.rand(dim, dim)-.5)
    a = math.transpose(a) + a

    x_init = 0.00001 * (torch.rand((dim*(dim-1)//2))-.5)

    # energy_l, theta_l, hess_eig_l = optimize(function_type_a(a), (x_init,), max_iterations,
    #                                          conv_tol=conv_tol, aug=True, lambda_min=1e-6,
    #                                          rho=10, mu=1e-4)
    # va, wa = math.linalg.eigh(a)
    # print(energy_l[-1])
    # finmat = vec_to_expmat(theta_l[-1][0])
    # print(math.allclose(finmat.T @ a @ finmat,
    #                     math.diag(va)))

    t = 0.0004
    theta_init = torch.Tensor([10])
    energy_l, theta_l, hess_eig_l = optimize(
        function_type_b(t), (theta_init,), max_iterations, conv_tol, aug=False)

    import matplotlib.pyplot as plt
    plt.plot(energy_l)
    plt.yscale('log')
    plt.show()
