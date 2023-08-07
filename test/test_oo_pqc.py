#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import pytest

import pennylane as qml

from pennylane import math
import auto_oo

from jax import jacobian as jjacobian
from jax import hessian as jhessian
from torch.autograd.functional import jacobian as tjacobian
from torch.autograd.functional import hessian as thessian

import torch
from jax.config import config

torch.set_default_dtype(torch.float64)
config.update("jax_enable_x64", True)


# @pytest.mark.parametrize(
#     ()


# )
# def test_energy_from_parameters(geometry, basis, ncas, nelecas, n_layers, freeze_active,
#                                 theta, kappa, e_ref):
#     pass


@pytest.mark.parametrize(
    ("geometry", "basis", "ncas", "nelecas", "n_layers", "freeze_active", "oao_mo_coeff", "theta"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g', 2, 2, 1, True,
         math.array([[9.8972e-01, -3.0733e-04, -1.1116e-01,  5.1563e-02,  5.3295e-03,
                      -1.5372e-02,  1.4414e-02, -3.2024e-02, -1.8512e-02, -5.7299e-02,
                      -1.7001e-02, -5.2446e-04,  5.8430e-03],
                     [1.4170e-01,  1.3605e-03,  7.0577e-01, -3.1654e-01, -5.1869e-02,
                      1.1233e-01, -1.1934e-01,  2.4548e-01,  1.5418e-01,  4.8462e-01,
                      1.6492e-01,  8.2957e-03, -7.6820e-02],
                     [9.9281e-04,  9.9522e-03,  8.2237e-02,  2.7922e-01, -2.2398e-02,
                      6.0103e-01,  1.7126e-01, -1.8756e-01, -1.9025e-01,  2.2193e-01,
                      -7.2464e-03, -9.4638e-02,  6.2806e-01],
                     [1.5624e-03,  1.7944e-04,  1.2011e-01, -1.4078e-01,  4.8091e-02,
                      -2.0103e-01,  2.3748e-01, -6.6814e-01, -5.1516e-01,  2.6657e-01,
                      1.0478e-01,  2.4466e-02, -2.6869e-01],
                     [6.9554e-04, -2.8820e-04, -3.4335e-03, -2.2587e-02,  2.3251e-01,
                      -7.0129e-02,  9.0415e-01,  2.8272e-01,  1.5874e-01,  4.7639e-02,
                      1.3697e-02, -1.0813e-01, -6.1108e-02],
                     [-3.8506e-04,  9.8831e-01, -7.8486e-02, -9.9812e-02,  4.4399e-04,
                      -9.1425e-04, -7.1268e-03,  7.7768e-03,  5.0307e-03,  6.0887e-02,
                      -5.2187e-02, -2.5037e-03,  2.3177e-02],
                     [3.0732e-03,  1.4996e-01,  4.5928e-01,  5.5104e-01, -2.5532e-03,
                      1.5108e-02,  4.2091e-02, -4.2326e-02, -2.3899e-02, -4.6079e-01,
                      4.6109e-01,  1.9291e-02, -1.8236e-01],
                     [-9.7189e-03, -6.1456e-04, -1.9661e-01,  2.4685e-01,  4.2773e-02,
                      -5.6830e-01, -9.4738e-02,  6.8660e-02,  8.0524e-02,  3.9788e-01,
                      4.9271e-01, -6.0858e-02,  3.8804e-01],
                     [3.8251e-04, -1.6496e-04,  4.4556e-02, -3.7514e-02,  1.5628e-02,
                      1.3470e-02,  5.1823e-02, -6.0438e-01,  7.9082e-01, -1.1952e-02,
                      -1.2028e-02,  1.0792e-02,  4.9158e-02],
                     [1.3475e-03, -1.8168e-04,  1.9591e-02, -7.6597e-03,  7.0661e-01,
                      6.0325e-02, -8.8876e-02,  1.1018e-02, -1.7219e-02,  4.4925e-03,
                      1.2360e-02,  6.9238e-01,  9.3505e-02],
                     [1.3014e-03,  1.7867e-02,  2.0091e-01,  3.8281e-01,  4.8289e-01,
                      -1.6897e-01, -1.8892e-01, -1.7374e-02,  7.4842e-03,  1.6724e-01,
                      -4.9407e-01, -4.8120e-01, -1.0657e-01],
                     [-6.1181e-04,  1.8054e-02,  1.7092e-01,  3.9175e-01, -4.5298e-01,
                      -2.6166e-01,  1.5017e-01,  3.6132e-02,  2.7464e-02,  1.8542e-01,
                      -4.7479e-01,  5.0840e-01,  2.1180e-02],
                     [1.6515e-02,  1.9801e-04,  3.7499e-01, -3.5048e-01,  2.5467e-02,
                      -3.9566e-01,  5.2081e-02, -6.0235e-02, -1.2454e-01, -4.4992e-01,
                      -1.8269e-01, -7.1269e-02,  5.6245e-01]]),
         math.array([0.8324, 0.2490])
         ),
    ],
)
def test_full_derivatives(geometry, basis, ncas, nelecas, n_layers,
                          freeze_active, oao_mo_coeff, theta):
    mol = auto_oo.Moldata_pyscf(geometry, basis)

    dev = qml.device('default.qubit', wires=2*ncas)

    pqc_torch = auto_oo.Parameterized_circuit(ncas, nelecas, dev,
                                              ansatz='np_fabric', n_layers=n_layers,
                                              interface='torch')
    oo_pqc_torch = auto_oo.OO_pqc(pqc_torch, mol, ncas, nelecas,
                                  oao_mo_coeff=oao_mo_coeff, freeze_active=freeze_active,
                                  interface='torch')

    theta_torch = math.array(theta, like='torch')
    kappa_torch = math.zeros(oo_pqc_torch.n_kappa, like='torch')

    grad_auto_torch = tjacobian(
        oo_pqc_torch.energy_from_parameters, (theta_torch, kappa_torch))

    grad_exact_C_torch = oo_pqc_torch.circuit_gradient(theta_torch)
    grad_exact_O_torch = oo_pqc_torch.orbital_gradient(theta_torch)

    assert math.allclose(grad_auto_torch[0], grad_exact_C_torch)
    assert math.allclose(grad_auto_torch[1], grad_exact_O_torch)

    hess_auto_torch = thessian(
        oo_pqc_torch.energy_from_parameters, (theta_torch, kappa_torch))

    hessian_C_C_torch = oo_pqc_torch.circuit_circuit_hessian(theta_torch)
    hessian_C_O_torch = oo_pqc_torch.orbital_circuit_hessian(theta_torch)
    hessian_O_O_torch = oo_pqc_torch.orbital_orbital_hessian(theta_torch)

    assert math.allclose(hess_auto_torch[0][0], hessian_C_C_torch)
    assert math.allclose(hess_auto_torch[1][0], hessian_C_O_torch)
    assert math.allclose(hess_auto_torch[1][1], hessian_O_O_torch)

    pqc_jax = auto_oo.Parameterized_circuit(ncas, nelecas, dev,
                                            ansatz='np_fabric', n_layers=n_layers,
                                            interface='jax')
    oo_pqc_jax = auto_oo.OO_pqc(pqc_jax, mol, ncas, nelecas, interface='jax',
                                oao_mo_coeff=oao_mo_coeff, freeze_active=freeze_active)

    theta_jax = math.array(theta, like='jax')
    kappa_jax = math.zeros(oo_pqc_jax.n_kappa, like='jax')

    grad_auto_jax = jjacobian(
        oo_pqc_jax.energy_from_parameters, argnums=(0, 1))(theta_jax, kappa_jax)

    grad_exact_C_jax = oo_pqc_jax.circuit_gradient(theta_jax)
    grad_exact_O_jax = oo_pqc_jax.orbital_gradient(theta_jax)

    assert math.allclose(grad_auto_jax[0], grad_exact_C_jax)
    assert math.allclose(grad_auto_jax[1], grad_exact_O_jax)

    hess_auto_jax = jhessian(
        oo_pqc_jax.energy_from_parameters, argnums=(0, 1))(theta_jax, kappa_jax)

    hessian_C_C_jax = oo_pqc_jax.circuit_circuit_hessian(theta_jax)
    hessian_C_O_jax = oo_pqc_jax.orbital_circuit_hessian(theta_jax)
    hessian_O_O_jax = oo_pqc_jax.orbital_orbital_hessian(theta_jax)

    assert math.allclose(hess_auto_jax[0][0], hessian_C_C_jax)
    assert math.allclose(hess_auto_jax[1][0], hessian_C_O_jax)
    assert math.allclose(hess_auto_jax[1][1], hessian_O_O_jax)


@pytest.mark.parametrize(
    ("geometry", "basis", "ncas", "nelecas", "n_layers", "freeze_active"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g', 2, 2, 1, True),
        # (auto_oo.get_formal_geo(140, 80), 'sto-3g', 3, 4, 2, True),
        # (auto_oo.get_formal_geo(140, 80), 'sto-3g', 3, 2, 2, True),
        (auto_oo.get_formal_geo(140, 80), 'cc-pvdz', 2, 2, 1, True)
    ],
)
def test_full_optimization(geometry, basis, ncas, nelecas, n_layers, freeze_active):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    mol.run_casscf(ncas, nelecas)

    dev = qml.device('default.qubit', wires=2*ncas)

    pqc_torch = auto_oo.Parameterized_circuit(ncas, nelecas, dev,
                                              ansatz='np_fabric', n_layers=n_layers,
                                              interface='torch')
    oo_pqc_torch = auto_oo.OO_pqc(pqc_torch, mol, ncas, nelecas, freeze_active=freeze_active)

    theta_zero_torch = pqc_torch.init_zeros()
    energy_l_torch, _, _, _, _ = oo_pqc_torch.full_optimization(theta_zero_torch)

    assert math.allclose(energy_l_torch[-1], mol.casscf.e_tot)

    pqc_jax = auto_oo.Parameterized_circuit(ncas, nelecas, dev,
                                            ansatz='np_fabric', n_layers=n_layers,
                                            interface='jax')
    oo_pqc_jax = auto_oo.OO_pqc(pqc_jax, mol, ncas, nelecas,
                                freeze_active=freeze_active, interface='jax')

    theta_zero_jax = pqc_jax.init_zeros()
    energy_l_jax, _, _, _, _ = oo_pqc_jax.full_optimization(theta_zero_jax)

    assert math.allclose(energy_l_jax[-1], mol.casscf.e_tot)
