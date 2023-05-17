#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import pytest

from pyscf import gto, scf, fci, mcscf
import auto_oo


# @pytest.mark.parametrize(
#     ("symbols, geometry"),
#     [

#     ],
# )
# def test_mo_ao_to_oao(geometry, basis, oao_coeff_ref, **kwargs):
#     mol = gto.Mole(geometry, basis, **kwargs)
#     oao_coeff = auto_oo.mo_ao_to_mo_oao(mol.mo_coeff, mol.overlap)


# def test_int_transforms():
#     pass


# @pytest.mark.parametrize(
#     ("inputs", "references"),
#     [(np, reference),
#      (inpt, reference),
#      (inut, reference),
#      (iput, reference)
#      ]
# )
# def test_energy_from_mo_coeff(inputs, reference_values,):
#     assert blabla == blabla
