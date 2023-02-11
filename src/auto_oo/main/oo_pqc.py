#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:10:18 2023

@author: emielkoridon
"""

import numpy as np

import pennylane as qml
import torch

from auto_oo.oo_energy.oo_energy import OO_energy
from auto_oo.ansatz.pqc import Parameterized_circuit
from auto_oo.moldata_pyscf import Moldata_pyscf

def vec_to_skew_symmetric(vector):
    r"""
    Map a vector to an anti-symmetric matrix with np.tril_indices.
    
    For example, the resulting matrix for `np.Tensor([1,2,3,4,5,6])` is:

    .. math::
        \begin{pmatrix}
            0 & -1 & -2 & -4\\
            1 &  0 & -3 & -5\\
            2 &  3 &  0 & -6\\
            4 &  5 &  6 &  0
        \end{pmatrix}

    Args:
        vector (torch.Tensor): 1d tensor
    """
    size = int(np.sqrt(8 * len(vector) + 1) + 1)//2
    matrix = torch.zeros((size,size))
    tril_indices = torch.tril_indices(row=size,col=size, offset=-1)
    matrix[tril_indices[0],tril_indices[1]] = vector
    matrix[tril_indices[1],tril_indices[0]] = - vector
    return matrix

class OO_pqc_cost(OO_energy):
    def __init__(self, pqc : Parameterized_circuit, mol : Moldata_pyscf,
                 ncas, nelecas, mo_coeff=None, freeze_active=False):
        super().__init__(mol, ncas, nelecas,
                     mo_coeff=None, freeze_active=False)
        self.pqc = pqc
        
        rotation_sizes = [len(self.occ_idx)*len(self.act_idx),
                          len(self.act_idx)*len(self.virt_idx),
                          len(self.occ_idx)*len(self.virt_idx)]
        if not freeze_active:
            rotation_sizes.append(
                len(self.act_idx) * (len(self.act_idx) - 1)//2)
        self.n_kappa = sum(rotation_sizes)

        # Save non-redundant kappa indices
        self.params_idx = np.array([], dtype=int)

        num = 0
        for l_idx, r_idx in zip(*np.tril_indices(self.nao,-1)):
            if not(
            ((l_idx in self.act_idx and r_idx in self.act_idx
              ) and freeze_active) or (
                  l_idx in self.occ_idx and r_idx in self.occ_idx) or (
                    l_idx in self.virt_idx and r_idx in self.virt_idx)):
                self.params_idx = np.append(self.params_idx, [num])
            num +=1
        
    def energy_from_parameters(self, theta, kappa=None):
        if kappa is None:
            mo_coeff = self.mo_coeff
        else:
            mo_coeff = self.get_transformed_mo(kappa)
        state = self.pqc.ansatz_state(theta)
        one_rdm, two_rdm = self.pqc.get_rdms_from_state(state)
        return self.energy_from_mo_coeff(mo_coeff, one_rdm, two_rdm)

    def kappa_vector_to_matrix(self, kappa):
        kappa_total_vector = torch.zeros(self.nao * (self.nao - 1)//2)
        kappa_total_vector[self.params_idx] = kappa
        return vec_to_skew_symmetric(kappa_total_vector)

    def kappa_to_mo_coeff(self, kappa):
        kappa_matrix = self.kappa_vector_to_matrix(kappa)
        return torch.linalg.matrix_exp(-kappa_matrix)

    def get_transformed_mo(self, kappa, update=True):
        mo_coeff = self.mo_coeff @ self.kappa_to_mo_coeff(kappa)
        if update:
            self.mo_coeff = mo_coeff
            self.update_integrals(mo_coeff)
        return mo_coeff

if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt
    
    def get_formal_geo(alpha,phi):
        variables = [1.498047, 1.066797, 0.987109, 118.359375] + [alpha, phi]
        geom = """
                        N
                        C 1 {0}
                        H 2 {1}  1 {3}
                        H 2 {1}  1 {3} 3 180
                        H 1 {2}  2 {4} 3 {5}
                        """.format(*variables)
        return geom
    
    geometry = get_formal_geo(140, 80)
    basis = 'sto-3g'
    mol = Moldata_pyscf(geometry, basis)

    ncas = 2
    nelecas = 2
    dev = qml.device('default.qubit', wires=2*ncas)
    pqc = Parameterized_circuit(ncas, nelecas, dev, vqe_singles=False)
    theta = torch.rand_like(pqc.init_zeros())
    # theta = pqc.init_zeros()
    oo_pqc = OO_pqc_cost(pqc, mol, ncas, nelecas)
    kappa = torch.zeros(oo_pqc.n_kappa)
    energy_test = oo_pqc.energy_from_parameters(theta)
    print("theta:", theta)
    print("state:", dirac_notation(pqc.ansatz_state(theta).detach().numpy()))
    print('Expectation value of Hamiltonian:', energy_test.item())
    mol.run_rhf()
    print('HF energy:', mol.hf.e_tot)
