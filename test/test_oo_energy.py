#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import pytest

import torch
import numpy as np
import auto_oo

from pennylane import math
from pyscf import ao2mo
from jax.config import config

torch.set_default_dtype(torch.float64)
config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("geometry, basis, hf_oao_coeff_ref"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g',
         math.array([[9.89775942e-01, -3.29898279e-04, -1.12359168e-01,
                      5.24659469e-02,  3.18702648e-03, -1.65763271e-02,
                      2.64432430e-02, -1.22156121e-03, -2.78584833e-02,
                      -5.37690011e-02, -1.67894464e-02,  6.15929969e-04,
                      5.57514036e-03],
                     [1.41355839e-01,  1.55429771e-03,  7.20192172e-01,
                      -3.24912537e-01, -2.38695621e-02,  1.25440160e-01,
                      -2.12658203e-01,  7.95376922e-03,  2.37356609e-01,
                      4.53685613e-01,  1.62790407e-01, -1.71446883e-03,
                      -7.42089942e-02],
                     [6.23150396e-04,  1.00798752e-02,  6.64026157e-02,
                      2.88178301e-01, -2.34087416e-02,  5.81306808e-01,
                      2.29093217e-01, -3.14698682e-03, -2.13788489e-01,
                      2.62375443e-01, -3.61354102e-03, -8.82647206e-02,
                      6.31769482e-01],
                     [1.15162345e-03,  1.31230715e-04,  6.86229663e-02,
                      -1.12963416e-01,  1.98234497e-02, -2.57113424e-01,
                      6.09051196e-01, -1.93689576e-01, -5.35768916e-01,
                      3.66839807e-01,  1.12759363e-01,  3.26187079e-02,
                      -2.68646773e-01],
                     [4.32092382e-04,  3.30763858e-05,  2.23804477e-02,
                      -2.13447541e-02,  2.32761527e-01, -2.69250493e-02,
                      3.06266000e-01,  9.07146586e-01,  9.67061568e-02,
                      3.04217628e-02,  1.35648644e-02, -1.14751272e-01,
                      -6.16216470e-02],
                     [-3.53002105e-04,  9.88315264e-01, -7.83216003e-02,
                      -1.00263834e-01, -2.63061992e-04, -7.17236664e-04,
                      -1.18835272e-02,  2.00662151e-03,  1.47891981e-02,
                      5.85390217e-02, -5.20549922e-02, -3.14885679e-03,
                      2.31236171e-02],
                     [2.84621848e-03,  1.49942270e-01,  4.58197180e-01,
                      5.54188037e-01,  4.03559166e-04,  1.39405577e-02,
                      7.25535092e-02, -1.37306915e-02, -9.37836719e-02,
                      -4.47357651e-01,  4.60421314e-01,  2.37863350e-02,
                      -1.82524825e-01],
                     [-9.31917367e-03, -6.28167943e-04, -1.92453332e-01,
                      2.44285760e-01,  3.34529824e-02, -5.66886523e-01,
                      -1.60534335e-01,  2.01339527e-02,  1.64349267e-01,
                      3.67998881e-01,  4.88383721e-01, -6.51612168e-02,
                      3.86450978e-01],
                     [-4.37804312e-05, -4.34862994e-04,  1.56855749e-02,
                      -2.32936181e-02,  3.16013025e-03, -1.47000634e-02,
                      6.24303099e-01, -2.81867458e-01,  7.16969922e-01,
                      -9.80131580e-02, -1.20398175e-02, -9.19733025e-03,
                      7.68080812e-02],
                     [-2.52164870e-05,  6.02058485e-06,  4.19292746e-03,
                      -3.13703884e-03,  7.09653690e-01,  4.66092979e-02,
                      -3.59506545e-02, -7.58174384e-02,  4.09483431e-04,
                      1.24401831e-02,  1.25883729e-02,  6.92065034e-01,
                      8.87899366e-02],
                     [3.65335772e-04,  1.79673158e-02,  1.86923653e-01,
                      3.84246286e-01,  4.81430952e-01, -1.81600023e-01,
                      -8.82467898e-02, -1.64326320e-01,  3.29076574e-02,
                      1.57529557e-01, -4.96688884e-01, -4.83909410e-01,
                      -1.03825702e-01],
                     [3.62808491e-04,  1.79625440e-02,  1.84555615e-01,
                      3.86576128e-01, -4.55076038e-01, -2.46041525e-01,
                      3.03721883e-02,  1.63188700e-01,  6.45586101e-02,
                      1.81274605e-01, -4.76678444e-01,  5.06380300e-01,
                      1.78896488e-02],
                     [1.62748031e-02,  2.16566362e-04,  3.69725967e-01,
                      -3.47742158e-01,  2.63358558e-02, -4.02220603e-01,
                      7.69288056e-02,  8.09866367e-03, -2.13209455e-01,
                      -4.24881413e-01, -1.81283983e-01, -5.93680812e-02,
                      5.57927976e-01]]))
    ],
)
def test_mo_ao_to_oao(geometry, basis, hf_oao_coeff_ref):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    assert math.allclose(auto_oo.mo_ao_to_mo_oao(mol.oao_coeff, mol.overlap), math.eye(mol.nao))
    mol.run_rhf()
    assert math.allclose(auto_oo.mo_ao_to_mo_oao(mol.hf.mo_coeff, mol.overlap), hf_oao_coeff_ref)


@pytest.mark.parametrize(
    ("geometry, basis"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g'),
        (auto_oo.get_formal_geo(140, 80), 'cc-pvdz'),
        (auto_oo.get_formal_geo(120, 125), 'sto-3g'),
        (auto_oo.get_formal_geo(120, 125), 'cc-pvdz')
    ],
)
def test_int_transforms(geometry, basis):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    mol.run_rhf()

    int1e_ao = mol.int1e_ao
    int2e_ao = mol.int2e_ao

    hf_mo_coeff = mol.hf.mo_coeff
    mol.run_casscf(2, 2)
    casscf_2_2_mo_coeff = mol.casscf.mo_coeff
    mol.run_casscf(4, 4)
    casscf_4_4_mo_coeff = mol.casscf.mo_coeff

    int1e_mo_hf = hf_mo_coeff.T @ int1e_ao @ hf_mo_coeff
    int2e_mo_hf = ao2mo.kernel(int2e_ao, hf_mo_coeff)

    int1e_mo_casscf_2_2 = casscf_2_2_mo_coeff.T @ int1e_ao @ casscf_2_2_mo_coeff
    int2e_mo_casscf_2_2 = ao2mo.kernel(int2e_ao, casscf_2_2_mo_coeff)

    int1e_mo_casscf_4_4 = casscf_4_4_mo_coeff.T @ int1e_ao @ casscf_4_4_mo_coeff
    int2e_mo_casscf_4_4 = ao2mo.kernel(int2e_ao, casscf_4_4_mo_coeff)

    assert math.allclose(auto_oo.int1e_transform(int1e_ao, hf_mo_coeff), int1e_mo_hf)
    assert math.allclose(auto_oo.int2e_transform(int2e_ao, hf_mo_coeff), int2e_mo_hf)

    assert math.allclose(auto_oo.int1e_transform(int1e_ao, casscf_2_2_mo_coeff),
                         int1e_mo_casscf_2_2)
    assert math.allclose(auto_oo.int2e_transform(int2e_ao, casscf_2_2_mo_coeff),
                         int2e_mo_casscf_2_2)

    assert math.allclose(auto_oo.int1e_transform(int1e_ao, casscf_4_4_mo_coeff),
                         int1e_mo_casscf_4_4)
    assert math.allclose(auto_oo.int2e_transform(int2e_ao, casscf_4_4_mo_coeff),
                         int2e_mo_casscf_4_4)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='torch'), math.array(hf_mo_coeff, like='torch')), int1e_mo_hf)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='torch'), math.array(hf_mo_coeff, like='torch')), int2e_mo_hf)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='torch'), math.array(casscf_2_2_mo_coeff, like='torch')),
        int1e_mo_casscf_2_2)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='torch'), math.array(casscf_2_2_mo_coeff, like='torch')),
        int2e_mo_casscf_2_2)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='torch'), math.array(casscf_4_4_mo_coeff, like='torch')),
        int1e_mo_casscf_4_4)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='torch'), math.array(casscf_4_4_mo_coeff, like='torch')),
        int2e_mo_casscf_4_4)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='jax'), math.array(hf_mo_coeff, like='jax')), int1e_mo_hf)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='jax'), math.array(hf_mo_coeff, like='jax')), int2e_mo_hf)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='jax'), math.array(casscf_2_2_mo_coeff, like='jax')),
        int1e_mo_casscf_2_2)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='jax'), math.array(casscf_2_2_mo_coeff, like='jax')),
        int2e_mo_casscf_2_2)

    assert math.allclose(auto_oo.int1e_transform(
        math.array(int1e_ao, like='jax'), math.array(casscf_4_4_mo_coeff, like='jax')),
        int1e_mo_casscf_4_4)
    assert math.allclose(auto_oo.int2e_transform(
        math.array(int2e_ao, like='jax'), math.array(casscf_4_4_mo_coeff, like='jax')),
        int2e_mo_casscf_4_4)


@pytest.mark.parametrize(
    ("vector, matrix_ref"),
    [
        (math.array([1., 2., 3., 4., 5., 6.], like='numpy'),
         math.array([[0., -1., -2., -4.],
                     [1., 0., -3., -5.],
                     [2., 3., 0., -6.],
                     [4., 5., 6., 0.]
                     ])),
        (math.array([1., 2., 3., 4., 5., 6.], like='torch'),
            math.array([[0., -1., -2., -4.],
                        [1., 0., -3., -5.],
                        [2., 3., 0., -6.],
                        [4., 5., 6., 0.]
                        ])),
        (math.array([1., 2., 3., 4., 5., 6.], like='jax'),
            math.array([[0., -1., -2., -4.],
                        [1., 0., -3., -5.],
                        [2., 3., 0., -6.],
                        [4., 5., 6., 0.]
                        ]))
    ],
)
def test_vector_to_skew_symmetric(vector, matrix_ref):
    assert math.allclose(auto_oo.oo_energy.vector_to_skew_symmetric(vector), matrix_ref)
    assert math.allclose(vector, auto_oo.oo_energy.skew_symmetric_to_vector(matrix_ref))


@pytest.mark.parametrize(
    ("occ_idx", "act_idx", "virt_idx", "freeze_active", "idx_ref"),
    [
        ([0, 1], [2, 3], [4, 5], False,
         np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13])),
        ([0, 1], [2, 3], [4, 5], True,
         np.array([1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13])),
        ([0, 1, 2], [3, 4], [5, 6], False,
            np.array([3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])),
        ([0, 1, 2], [3, 4], [5, 6], True,
            np.array([3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))
    ],
)
def test_non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active, idx_ref):
    assert math.allclose(
        auto_oo.oo_energy.non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active), idx_ref)

    # @pytest.mark.parametrize(
    #     ("inputs", "references"),
    #     [(np, reference),
    #       (inpt, reference),
    #       (inut, reference),
    #       (iput, reference)
    #       ]
    # )
    # def test_energy_from_mo_coeff(inputs, reference_values,):
    #     assert blabla == blabla
