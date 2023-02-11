#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:39:19 2023

@author: emielkoridon
"""
import warnings
import itertools

import torch
import pennylane as qml
import openfermion

from auto_oo.ansatz.uccd import UCCD

torch.set_default_tensor_type(torch.DoubleTensor)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


def e_pq(p, q, restricted=True):
    r"""
    Can generate either spin-unrestricted single excitation operator:
    
    .. math::
        E_{pq} = a_{p}^\dagger a_{q}
    where :math:`p` and :math:`q` are composite spatial/spin indices,
    or spin-restricted single excitation operator:
    
    .. math::
            E_{pq} = \sum_\sigma a_{p \sigma}^\dagger a_{q \sigma}
    where :math:`p` and :math:`q` are spatial indices.
    """
    if restricted:
        return (openfermion.FermionOperator(f'{2*p}^ {2*q}') + 
                openfermion.FermionOperator(f'{2*p+1}^ {2*q+1}'))

    else:
        return openfermion.FermionOperator(f'{p}^ {q}')

def e_pqrs(p, q, r, s, restricted=True):
    r"""
    Can generate either spin-unrestricted double excitation operator:
    
    .. math::
        e_{pqrs} = a_{p}^\dagger a_{q}^\dagger a_{r} a_{s}
    where :math:`p` and :math:`q` are composite spatial/spin indices,
    or spin-restricted double excitation operator:
    
    .. math::
        e_{pqrs} = \sum_{\sigma \tau} a_{p \sigma}^\dagger a_{
            r \tau}^\dagger a_{s \tau} a_{q \sigma}
                 = E_{pq}E_{rs} -\delta_{qr}E_{ps}
    where the indices are spatial indices.
    
    Indices in the restricted case are in chemist order, meant to be 
    contracted with two-electron integrals in chemist order to obtain the
    Hamiltonian or to obtain the two-electron RDM.
    """
    if restricted:
        operator = e_pq(p, q, restricted) * e_pq(r, s, restricted)
        if q == r:
            operator += - e_pq(p, s, restricted)
        return operator
    else:
        return openfermion.FermionOperator(f'{p}^ {q}^ {r} {s}')


def scipy_csc_to_torch(scipy_csc, dtype=torch.complex128):
    """ Convert a scipy sparse CSC matrix to pytorch sparse tensor.
    
    TODO: Newton-Raphson only works if I cast them to dense matrices for some reason....
          so now it has a similar runtime as the jordan-wigner intops functions, but
          reduced because it is in the restricted formalism."""
    ccol_indices = scipy_csc.indptr
    row_indices = scipy_csc.indices
    values = scipy_csc.data
    size = scipy_csc.shape
    return torch.sparse_csc_tensor(
        torch.tensor(ccol_indices, dtype=torch.int64),
        torch.tensor(row_indices, dtype=torch.int64),
        torch.tensor(values), dtype=dtype, size=size).to_dense().detach()

def initialize_e_pq(ncas, restricted = False):
    """Initialize full e_pq operator in pytorch CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    return [[scipy_csc_to_torch(
        openfermion.get_sparse_operator(
            e_pq(p,q,restricted), n_qubits=2*ncas))
        for q in range(num_ind)] for p in range(num_ind)]

def initialize_e_pqrs(ncas, restricted = False):
    """Initialize full e_pqrs operator in pytorch CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    return [[[[scipy_csc_to_torch(
        openfermion.get_sparse_operator(
            e_pqrs(p,q,r,s,restricted), n_qubits=2*ncas))
        for s in range(num_ind)] for r in range(num_ind)]
        for q in range(num_ind)] for p in range(num_ind)]

def uccd_state(theta, dev, wires, s_wires, d_wires, hfstate, vqe_singles):
    @qml.qnode(dev, interface='torch', diff_method="backprop")
    def uccd_circuit():
        if vqe_singles:
            qml.UCCSD(theta, wires, s_wires=s_wires,
                      d_wires=d_wires, init_state=hfstate)
        else:
            UCCD(theta, wires, d_wires, init_state=hfstate)
        return qml.state()
    return uccd_circuit()

class Parameterized_circuit():
    def __init__(self, ncas, nelecas, dev, ansatz_state_fn=None, vqe_singles=False):
        self.ncas = ncas
        self.nelecas = nelecas
        self.n_qubits = 2 * ncas
        
        self.dev = dev
        
        self.vqe_singles = vqe_singles
        
        self.e_pq = None
        self.e_pqrs = None
        
        if ansatz_state_fn is None:
            self.singles, self.doubles = qml.qchem.excitations(nelecas,
                                                               self.n_qubits)
            if vqe_singles:
                self.n_theta = len(self.doubles) + len(self.singles)
            else:
                self.n_theta = len(self.doubles)
            self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(
                self.singles, self.doubles)
            self.hfstate = qml.qchem.hf_state(nelecas, self.n_qubits)
            self.wires = range(self.n_qubits)
            self.ansatz_state = self.uccd_state
        else:
            self.ansatz_state = ansatz_state_fn

    def uccd_state(self, theta):
        return uccd_state(theta, 
                          self.dev, self.wires, self.s_wires,
                          self.d_wires, self.hfstate, self.vqe_singles)
    
    def init_zeros(self):
        return torch.zeros(self.n_theta)
    
    def get_rdms_from_state(self, state, restricted=True):
        r"""
        Generate one- and two-particle reduced density matrices from
        a state :math:`| \Psi \rangle`:
        
        .. math::
            \gamma = \langle \Psi | E_{pq} | \Psi \rangle\\
            \Gamma = \langle \Psi | e_{pqrs} | \Psi \rangle
        
        where the single (double) excitation operators :math:`E_{pq}`
        (:math:`e_{pqrs}`) can be either restricted (summed over spin)
        or unrestricted.
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
        return one_rdm, two_rdm
        
    def init_e_pq(self, restricted=True):
        if self.e_pq is None:
            self.e_pq = initialize_e_pq(self.ncas, restricted)

    def init_e_pqrs(self, restricted=True):
        if self.e_pqrs is None:
            self.e_pqrs = initialize_e_pqrs(self.ncas, restricted)
            
if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt
    ncas = 2
    nelecas = 2
    dev = qml.device('default.qubit', wires=2*ncas)
    pqc = Parameterized_circuit(ncas, nelecas, dev, vqe_singles=False)
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
