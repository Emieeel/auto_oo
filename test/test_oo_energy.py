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


def get_oo_energy(geometry, basis, ncas, nelecas, freeze_active, interface):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    return auto_oo.OO_energy(mol, ncas, nelecas,
                             freeze_active=freeze_active, interface=interface)


@pytest.mark.parametrize(
    ("geometry", "basis", "ncas", "nelecas", "freeze_active",
     "mo_coeff", "one_rdm", "two_rdm", "e_ref"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g', 2, 2, True,
         math.array([[9.9333e-01,  5.3492e-04, -1.9943e-01,  9.8055e-02,  8.1665e-03,
                      -3.3549e-02,  1.1721e-02, -6.4373e-02, -4.6523e-02, -1.5654e-01,
                      -4.7053e-02, -2.4485e-03,  2.1614e-02],
                     [3.0339e-02, -4.9511e-03,  6.7346e-01, -3.5360e-01, -3.5112e-02,
                      1.5469e-01, -4.7991e-02,  2.9737e-01,  2.4426e-01,  9.7493e-01,
                      3.2404e-01,  2.2811e-02, -2.1115e-01],
                     [-1.9460e-03, -3.1147e-03,  2.1914e-02,  1.8732e-01, -8.2563e-03,
                      4.7294e-01,  2.0194e-01, -1.6444e-01, -2.1537e-01,  2.8919e-01,
                      -4.4163e-02, -1.3997e-01,  1.0266e+00],
                     [-3.9878e-05,  6.6075e-04,  7.2801e-02, -1.0095e-01,  4.0295e-02,
                      -1.7616e-01,  2.2665e-01, -5.7484e-01, -5.9265e-01,  4.0076e-01,
                      1.6334e-01,  4.4681e-02, -4.5303e-01],
                     [-1.1902e-02,  1.6472e-03, -1.3235e-01,  3.2205e-02,  1.7079e-01,
                      -1.5286e-01,  8.9586e-01,  2.7477e-01,  1.8450e-01,  7.1456e-02,
                      2.2596e-02, -1.7450e-01, -1.0140e-01],
                     [7.2868e-04,  9.9261e-01, -1.2727e-01, -1.7835e-01, -1.7647e-03,
                      -9.6190e-04, -2.4979e-02,  1.7113e-02,  1.1409e-02,  1.6007e-01,
                      -1.5328e-01, -5.4942e-03,  6.1968e-02],
                     [-5.2938e-03,  3.3104e-02,  3.5673e-01,  5.3595e-01,  4.6614e-03,
                      2.1574e-02,  8.0988e-02, -7.1101e-02, -3.8002e-02, -8.8977e-01,
                      1.0173e+00,  2.9676e-02, -3.5273e-01],
                     [5.1827e-03,  3.7286e-04, -7.6705e-02,  1.5253e-01,  1.8357e-02,
                      -4.3958e-01, -1.5356e-01,  8.5878e-02,  8.5815e-02,  5.2312e-01,
                      7.8272e-01, -9.0428e-02,  6.0311e-01],
                     [-2.9170e-04, -9.0678e-05,  3.0389e-02, -2.5348e-02,  1.3448e-02,
                      2.3416e-02,  4.3243e-02, -6.1446e-01,  7.9580e-01, -3.6630e-02,
                      -2.1887e-02,  8.6047e-03,  7.5446e-02],
                     [4.5959e-04, -6.1325e-05,  9.7942e-03, -3.6090e-03,  5.4424e-01,
                      4.5110e-02, -6.1893e-02,  9.0112e-03, -2.7601e-02,  7.9414e-04,
                      1.9797e-02,  1.1837e+00,  1.5692e-01],
                     [2.5188e-03, -7.2010e-03,  1.1455e-01,  2.2841e-01,  4.0786e-01,
                      -1.3306e-01, -2.1778e-01, -2.1842e-02,  1.0349e-02,  3.1104e-01,
                      -8.7774e-01, -8.6456e-01, -1.6929e-01],
                     [-2.0047e-03, -6.6356e-03,  5.9476e-02,  2.4810e-01, -3.8739e-01,
                      -2.3379e-01,  1.3225e-01,  4.9314e-02,  2.6779e-02,  3.2789e-01,
                      -8.4479e-01,  9.0883e-01,  5.5081e-02],
                     [-8.1451e-03, -1.5492e-04,  1.7996e-01, -2.3228e-01,  2.7867e-02,
                      -3.6551e-01,  7.2975e-02, -7.7523e-02, -1.6133e-01, -7.2801e-01,
                      -3.1231e-01, -1.1850e-01,  9.8801e-01]]),
         math.array([[1.6686, -0.0778],
                     [-0.0778,  0.3314]]),
         math.array([[[[1.6569, -0.1387],
                       [-0.1387,  0.0116]],

                      [[-0.1387, -0.7280],
                       [0.0116,  0.0609]]],


                     [[[-0.1387,  0.0116],
                       [-0.7280,  0.0609]],

                      [[0.0116,  0.0609],
                         [0.0609,  0.3198]]]]),
         math.array([-92.74923236954386])),
    ],
)
def test_energy_from_mo_coeff(geometry, basis, ncas, nelecas, freeze_active,
                              mo_coeff, one_rdm, two_rdm, e_ref):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    oo_energy = auto_oo.OO_energy(mol, ncas, nelecas,
                                  freeze_active=freeze_active, interface='torch')
    assert np.allclose(oo_energy.energy_from_mo_coeff(
        math.array(mo_coeff, like='torch'), math.array(one_rdm, like='torch'),
        math.array(two_rdm, like='torch')), e_ref)


@pytest.mark.parametrize(
    ("geometry", "basis", "ncas", "nelecas", "freeze_active",
     "mo_coeff", "one_rdm", "two_rdm", "e_ref"),
    [
        (auto_oo.get_formal_geo(140, 80), 'sto-3g', 2, 2, False,
         math.array([[1.02410942e+00, -1.44485996e-01, -1.22283337e-03,
                      -6.92105527e-03, -1.22191185e-03, -1.68737940e-03,
                      1.75420166e-02, -1.64976921e-02,  3.63410363e-04,
                      9.10179123e-05,  7.02693079e-04,  7.69242606e-04,
                      2.45601209e-02],
                     [-1.44485996e-01,  1.27102203e+00, -8.35510237e-03,
                      8.33090765e-02,  1.47040840e-02,  2.05491933e-02,
                      -1.74022090e-01,  2.16821224e-01, -3.37367753e-03,
                      -8.62524345e-04, -1.09749430e-02, -1.16666054e-02,
                      -3.66189921e-01],
                     [-1.22283337e-03, -8.35510237e-03,  1.17080424e+00,
                      -6.09411627e-02, -1.07561034e-02,  1.55985716e-02,
                      -2.00656514e-01,  2.13767917e-01,  2.46589718e-03,
                      6.30139174e-04, -1.02027449e-02, -9.69784783e-03,
                      2.67483185e-01],
                     [-6.92105527e-03,  8.33090765e-02, -6.09411627e-02,
                      1.05594126e+00,  8.42234637e-03, -5.58826900e-04,
                      6.92144928e-03, -7.04309680e-03, -7.63897614e-02,
                      -4.93211825e-04,  8.70467891e-04,  4.74942082e-04,
                      -2.09761320e-01],
                     [-1.22191185e-03,  1.47040840e-02, -1.07561034e-02,
                      8.42234637e-03,  1.01025279e+00, -9.86413949e-05,
                      1.22157318e-03, -1.24292348e-03, -3.40658645e-04,
                      -8.40429606e-02,  1.11163141e-02, -1.08789589e-02,
                      -3.70164397e-02],
                     [-1.68737940e-03,  2.05491933e-02,  1.55985716e-02,
                      -5.58826900e-04, -9.86413949e-05,  1.02845157e+00,
                      -1.66022513e-01, -7.48021951e-04,  2.51443819e-05,
                      6.34381847e-06,  2.62311797e-02,  2.62360254e-02,
                      2.23010467e-03],
                     [1.75420166e-02, -1.74022090e-01, -2.00656514e-01,
                      6.92144928e-03,  1.22157318e-03, -1.66022513e-01,
                      1.41719899e+00,  3.75451001e-02, -2.64145217e-04,
                      -6.79178174e-05, -3.68521259e-01, -3.68577010e-01,
                      -3.16218915e-02],
                     [-1.64976921e-02,  2.16821224e-01,  2.13767917e-01,
                      -7.04309680e-03, -1.24292348e-03, -7.48021951e-04,
                      3.75451001e-02,  1.16950164e+00,  2.42905654e-04,
                      6.29243560e-05, -1.67318064e-01, -1.67264548e-01,
                      3.37769658e-02],
                     [3.63410363e-04, -3.37367753e-03,  2.46589718e-03,
                      -7.63897614e-02, -3.40658645e-04,  2.51443819e-05,
                      -2.64145217e-04,  2.42905654e-04,  1.00832990e+00,
                      2.67703703e-05, -2.43181136e-05, -5.12373901e-06,
                      6.41676147e-03],
                     [9.10179123e-05, -8.62524345e-04,  6.30139174e-04,
                      -4.93211825e-04, -8.40429606e-02,  6.34381847e-06,
                      -6.79178174e-05,  6.29243560e-05,  2.67703703e-05,
                      1.22811066e+00, -3.27055897e-01,  3.27047885e-01,
                      1.71464547e-03],
                     [7.02693079e-04, -1.09749430e-02, -1.02027449e-02,
                      8.70467891e-04,  1.11163141e-02,  2.62311797e-02,
                      -3.68521259e-01, -1.67318064e-01, -2.43181136e-05,
                      -3.27055897e-01,  1.29124706e+00, -3.95728266e-02,
                      -4.59431442e-03],
                     [7.69242606e-04, -1.16666054e-02, -9.69784783e-03,
                      4.74942082e-04, -1.08789589e-02,  2.62360254e-02,
                      -3.68577010e-01, -1.67264548e-01, -5.12373901e-06,
                      3.27047885e-01, -3.95728266e-02,  1.29123882e+00,
                      -3.00271688e-03],
                     [2.45601209e-02, -3.66189921e-01,  2.67483185e-01,
                      -2.09761320e-01, -3.70164397e-02,  2.23010467e-03,
                      -3.16218915e-02,  3.37769658e-02,  6.41676147e-03,
                      1.71464547e-03, -4.59431442e-03, -3.00271688e-03,
                      1.27359252e+00]]),
            math.array([[2., 0.],
                        [0., 0.]]),
            math.array([[[[2., 0.],
                       [0., 0.]],
                [[0., 0.],
                 [0., 0.]]],
                [[[0., 0.],
                  [0., 0.]],
                 [[0., 0.],
                  [0., 0.]]]]), math.array([-92.66372193556138])),
    ],
)
def test_orbital_optimization(geometry, basis, ncas, nelecas, freeze_active,
                              mo_coeff, one_rdm, two_rdm, e_ref):
    mol = auto_oo.Moldata_pyscf(geometry, basis)
    oo_energy = auto_oo.OO_energy(mol, ncas, nelecas,
                                  freeze_active=freeze_active, interface='torch')
    energy_l = oo_energy.orbital_optimization(math.array(one_rdm, like='torch'),
                                              math.array(two_rdm, like='torch'))
    assert math.allclose(e_ref, energy_l[-1])
