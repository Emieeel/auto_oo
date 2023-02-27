#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:39:19 2023

@author: emielkoridon
"""

import itertools

import torch
import pennylane as qml

from auto_oo.ansatz.pqc import Parameterized_circuit


class Noisy_parameterized_circuit(Parameterized_circuit):
    def __init__(self, ncas, nelecas, dev, variance, ansatz_state_fn=None, add_singles=False):
        super().__init__(ncas, nelecas, dev,
                         ansatz_state_fn=ansatz_state_fn,
                         add_singles=add_singles)
        self.variance = variance
        
    def get_rdms_from_state(self, state, restricted=True):
        r"""
        Generate one- and two-particle reduced density matrices from
        a state :math:`| \Psi \rangle` with some :math:`\epsilon` noise:
        
        .. math::
            \gamma = \langle \Psi | E_{pq} | \Psi \rangle + \Epsilon_{pq}\\
            \Gamma = \langle \Psi | e_{pqrs} | \Psi \rangle + \epsilon_{pqrs}
        
        where the single (double) excitation operators :math:`E_{pq}`
        (:math:`e_{pqrs}`) can be either restricted (summed over spin)
        or unrestricted.
        
        :math:`\epsilon` is a random tensor with a gaussian distribution
        around zero.
        """
        self.init_e_pq(restricted)
        self.init_e_pqrs(restricted)
        if restricted:
            rdm_size = self.ncas
        else:
            rdm_size = 2 * self.ncas
        one_rdm = torch.zeros((rdm_size, rdm_size))
        two_rdm = torch.zeros((rdm_size, rdm_size, rdm_size, rdm_size))
        for p, q in itertools.product(range(rdm_size),repeat=2):
            e_pq = self.e_pq[p][q]
            one_rdm[p,q] = torch.matmul(state, torch.matmul(e_pq, state)).real
            for r, s in itertools.product(range(rdm_size),repeat=2):
                e_pqrs = self.e_pqrs[p][q][r][s]
                two_rdm[p,q,r,s] = torch.matmul(state, torch.matmul(e_pqrs, state)).real
        
        noisy_one_rdm = one_rdm + (self.variance**0.5)*torch.randn((rdm_size,
                                                                    rdm_size))
        noisy_two_rdm = two_rdm + (self.variance**0.5)*torch.randn((rdm_size,
                                                                    rdm_size,
                                                                    rdm_size,
                                                                    rdm_size))
        return noisy_one_rdm, noisy_two_rdm

            
if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt
    ncas = 2
    nelecas = 2
    dev = qml.device('default.qubit', wires=2*ncas)
    pqc = Parameterized_circuit(ncas, nelecas, dev, add_singles=False)
    theta = torch.rand_like(pqc.init_zeros())
    state = pqc.ansatz_state(theta)
    print("theta = ", theta)
    print("state:", dirac_notation(state.detach().numpy()))
    one_rdm, two_rdm = pqc.get_rdms_from_state(state)
    plt.imshow(one_rdm)
    plt.colorbar()
    plt.show()
    plt.imshow(two_rdm.reshape(ncas**2,ncas**2))
    plt.colorbar()
    plt.show()
