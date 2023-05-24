#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import auto_oo
from torch.autograd.functional import jacobian, hessian
import pytest

import numpy as np
from pennylane import math
import torch
from jax.config import config

torch.set_default_dtype(torch.float64)
config.update("jax_enable_x64", True)


def function_type_a(a):
    def f(x):
        xmat = x.reshape(a.shape)
        i = math.eye(a.shape[0], like='torch')
        ix = i + xmat
        vix, wix = math.linalg.eigh(ix)
        ix_inv = wix @ math.diag(1/vix) @ wix.T
        # ymat = y.reshape(a.shape)
        # (a @ (x**2) + (c @ x) - b)**2)
        return math.sum((((i - xmat) @ ix_inv) - a)**2)
    return f


def function_type_b(t):
    def f(x):
        return -t * math.log(math.abs(x)) + math.abs(x) - t + t * math.log(t)
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

        # import pdb
        # pdb.set_trace()

        if len(params_init) == 1:
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
    ("dim", "max_iterations", "conv_tol"),
    [(10, 10, 1e-12),
     (50, 50, 1e-10),
     (100, 100, 1e-8),
     (500, 150, 1e-8),
     (1000, 200, 1e-8)],
)
def test_optimization_type_a(dim, max_iterations, conv_tol, verbose=1):
    a = (torch.rand(dim, dim)-.5)
    a = math.transpose(a) + a
    b = (torch.rand(dim) - .5)
    theta_init = 20*(torch.rand(dim)-.5)
    energy_l, theta_l, hess_eig_l = optimize(
        function_type_a(a, b), theta_init, max_iterations, conv_tol, aug=False)
    assert math.allclose(energy_l[-1], 0.)

    va, wa = math.linalg.eigh(a)
    a_inv = wa @ math.diag(1/va) @ wa.T

    assert math.allclose(theta_l[-1]**2, a_inv @ b)


def test_optimization_type_b(t, max_iterations, conv_tol, verbose=1):
    theta_init = torch.rand([10])
    energy_l, theta_l, hess_eig_l = optimize(
        function_type_b(t), theta_init, max_iterations, conv_tol, aug=False)
    assert math.allclose(energy_l[-1], 0.)


if __name__ == '__main__':
    import torch
    from scipy.stats import ortho_group
    torch.set_num_threads(12)
    dim = 3
    a = (torch.rand(dim, dim)-.5)
    # a = math.transpose(a) + a
    a = a.T - a

    ia = math.eye(dim, like='torch') + a
    via, wia = math.linalg.eigh(ia)
    ia_inv = wia @ math.diag(1/via) @ wia.T

    r = (math.eye(dim, like='torch') - a) @ ia_inv

    ir = math.eye(dim, like='torch') + r
    vir, wir = math.linalg.eigh(ir)
    ir_inv = wir @ math.diag(1/vir) @ wir.T

    vr, wr = math.linalg.eigh(r)
    r_inv = wr @ math.diag(1/vr) @ wr.T

    # a = ortho_group.rvs(dim=dim)
    # a = torch.from_numpy(a)

    # b = ortho_group.rvs(dim=dim)
    # b = torch.from_numpy(b)

    b = (torch.rand(dim))
    c = (torch.rand(dim, dim)-.5)
    c = math.transpose(c) + c
    x_init = (torch.rand(dim*dim)-.5)
    y_init = torch.zeros(1)
    # y_init = 20*(torch.rand(dim*dim)-.5)
    max_iterations = 100
    conv_tol = 1e-12
    energy_l, theta_l, hess_eig_l = optimize(function_type_a(a), (x_init,), max_iterations,
                                             conv_tol=conv_tol, aug=True, lambda_min=1e-10,
                                             rho=1.01, mu=1e-5)

    # vc, wc = math.linalg.eigh(c)
    # c_inv = wc @ math.diag(1/vc) @ wc.T

    print(energy_l[-1])
    print(math.allclose(theta_l[-1][0].reshape(dim, dim), r))

    # print(math.allclose(theta_l[-1][0].reshape(dim, dim) @ a @ theta_l[-1][0].reshape(dim, dim),
    #                   math.diag(b)))

    import matplotlib.pyplot as plt
    plt.plot(energy_l)
    plt.yscale('log')
    plt.show()

    t = 3.5
    # test_optimization_type_b(t, 100, 1e-10)

    # test_optimization_type_a(dim1, 100, 1e-12)
