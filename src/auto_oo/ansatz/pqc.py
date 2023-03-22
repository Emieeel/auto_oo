#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:39:19 2023

@author: emielkoridon
"""

import itertools

import torch
import numpy as np
import pennylane as qml
import openfermion

from auto_oo.ansatz.uccd import UCCD


def e_pq(p, q, n_modes, restricted=True, up_then_down=False):
    r"""
    Can generate either spin-unrestricted single excitation operator:

    .. math::
        E_{pq} = a_{p}^\dagger a_{q}
    where :math:`p` and :math:`q` are composite spatial/spin indices,
    or spin-restricted single excitation operator:

    .. math::
            E_{pq} = \sum_\sigma a_{p \sigma}^\dagger a_{q \sigma}
    where :math:`p` and :math:`q` are spatial indices. For the spin-
    restricted case, One can either select up-then-down convention,
    or up-down-up-down.
    """
    if restricted:
        if up_then_down:
            operator = (openfermion.FermionOperator(f'{p}^ {q}') +
                        openfermion.FermionOperator(f'{p+n_modes}^ {q + n_modes}'))
        else:
            operator = (openfermion.FermionOperator(f'{2*p}^ {2*q}') +
                        openfermion.FermionOperator(f'{2*p+1}^ {2*q+1}'))
    else:
        operator = openfermion.FermionOperator(f'{p}^ {q}')

    return operator


def e_pqrs(p, q, r, s, n_modes, restricted=True, up_then_down=False):
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
        operator = e_pq(p, q, n_modes, restricted, up_then_down) * e_pq(
            r, s, n_modes, restricted, up_then_down)
        if q == r:
            operator += - e_pq(p, s, n_modes, restricted, up_then_down)
    else:
        operator = openfermion.FermionOperator(f'{p}^ {q}^ {r} {s}')
    return operator


def scipy_csc_to_torch(scipy_csc, dtype=torch.double):
    """ Convert a scipy sparse CSC matrix to pytorch sparse CSC tensor."""
    ccol_indices = scipy_csc.indptr
    row_indices = scipy_csc.indices
    values = scipy_csc.data
    size = scipy_csc.shape
    return torch.sparse_csc_tensor(
        torch.tensor(ccol_indices, dtype=torch.int64),
        torch.tensor(row_indices, dtype=torch.int64),
        torch.tensor(values.real), dtype=dtype, size=size)


def initialize_e_pq(ncas, restricted=True, up_then_down=False):
    """Initialize full e_pq operator in pytorch CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    return [[scipy_csc_to_torch(
        openfermion.get_sparse_operator(
            e_pq(p, q, num_ind, restricted, up_then_down), n_qubits=2*ncas))
        for q in range(num_ind)] for p in range(num_ind)]


def initialize_e_pqrs(ncas, restricted=True, up_then_down=False):
    """Initialize full e_pqrs operator in pytorch CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    return [[[[scipy_csc_to_torch(
        openfermion.get_sparse_operator(
            e_pqrs(p, q, r, s, num_ind, restricted, up_then_down), n_qubits=2*ncas))
        for s in range(num_ind)] for r in range(num_ind)]
        for q in range(num_ind)] for p in range(num_ind)]


def uccd_circuit(theta, wires, s_wires, d_wires, hfstate, add_singles=False):
    """Outputs UCC(S)D ansatz state, in up-down-up-down JW ordering"""
    if add_singles:
        qml.UCCSD(theta, wires, s_wires=s_wires,
                  d_wires=d_wires, init_state=hfstate)
    else:
        UCCD(theta, wires, d_wires, init_state=hfstate)
    return qml.state()


def gatefabric_circuit(theta, wires, hfstate):
    """ Outputs NP fabric ansatz state, adapted to up-then-down
    JW ordering"""
    # l2 = list(range(1, len(wires), 2))
    # l1 = list(range(0, len(wires), 2))
    qml.GateFabric(theta,
                   wires=wires, init_state=hfstate, include_pi=False)
    # qml.Permute(l1 + l2, wires)
    return qml.state()


class Parameterized_circuit():
    """ Parameterized quantum circuit class. Defined by an active space of
        nelecas electrons in ncas orbitals. Defines a qnode that outputs a
        quantum state. Can output one and two-RDMs."""

    def __init__(self, ncas, nelecas, dev, ansatz='ucc', n_layers=3,
                 add_singles=False):
        """
        Args:
            ncas: Number of active orbitals

            nelecas: Number of active electrons

            dev: Pennylane device on which to run the quantum circuit

            ansatz (default 'ucc'): Either 'ucc' or 'np_fabric', one of the
                native ansatzes implemented, or a custom Pennylane QNode that
                outputs a computational basis state.

            n_layers (default 3): Number of layers in an 'np_fabric' ansatz.

            add_singles (default: False): Add UCC single excitations to a 'ucc'
                ansatz
        """
        self.ncas = ncas
        self.nelecas = nelecas
        self.n_qubits = 2 * ncas

        self.dev = dev
        self.add_singles = add_singles

        self.e_pq = None
        self.e_pqrs = None

        if ansatz == 'ucc':
            self.up_then_down = False
            self.singles, self.doubles = qml.qchem.excitations(nelecas,
                                                               self.n_qubits)
            if add_singles:
                self.theta_shape = len(self.doubles) + len(self.singles)
            else:
                self.theta_shape = len(self.doubles)
            self.s_wires, self.d_wires = qml.qchem.excitations_to_wires(
                self.singles, self.doubles)
            self.hfstate = qml.qchem.hf_state(nelecas, self.n_qubits)
            self.wires = range(self.n_qubits)
            self.qnode = qml.qnode(dev, interface='torch', diff_method='backprop')(
                self.uccd_state)

        elif ansatz == 'np_fabric':
            self.n_layers = n_layers
            self.up_then_down = False
            self.wires = list(range(self.n_qubits))
            self.hfstate = qml.qchem.hf_state(nelecas, self.n_qubits)
            self.full_theta_shape = qml.GateFabric.shape(self.n_layers, len(
                self.wires))

            # Calculate the redundant indices of theta, describing initial rotations
            # between all-occupied or all-virtual states. THESE ARE ONLY REDUNDANT
            # WHEN STARTING WITH THE HF STATE!
            if self.n_qubits > 4:
                self.redundant_idx = [x for x in range(0, 2*(self.nelecas//4))]
                if self.ncas % 2 == 0:
                    self.redundant_idx += [x for x in range(2*((self.n_qubits-self.nelecas)//4),
                                                            2*(self.n_qubits//4))]
            else:
                self.redundant_idx = []

            self.params_idx = [x for x in range(np.prod(self.full_theta_shape))
                               if x not in self.redundant_idx]
            self.theta_shape = len(self.params_idx)
            self.qnode = qml.qnode(dev, interface='torch', diff_method='backprop')(
                self.gatefabric_state)

        else:
            self.qnode = ansatz

    def uccd_state(self, theta):
        """Return UCC(S)D state"""
        return uccd_circuit(theta,
                            self.wires, self.s_wires,
                            self.d_wires, self.hfstate, self.add_singles)

    def gatefabric_state(self, theta):
        """The first few parameters of the GateFabric ansatz are redundant,
        as they are rotations between active all-occupied or all-virtual
        orbitals. Set them to zero and then return the state."""
        theta_full = torch.zeros(len(self.redundant_idx) + self.theta_shape)
        theta_full[self.params_idx] = theta
        theta_full = theta_full.reshape(self.full_theta_shape)
        return gatefabric_circuit(theta_full, self.wires, self.hfstate)

    def init_zeros(self):
        """Initialize thetas in all-zero"""
        return torch.zeros(self.theta_shape)

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
        for p, q in itertools.product(range(rdm_size), repeat=2):
            e_pq = self.e_pq[p][q]
            one_rdm[p, q] = torch.matmul(state.real, torch.matmul(e_pq, state.real))
            for r, s in itertools.product(range(rdm_size), repeat=2):
                e_pqrs = self.e_pqrs[p][q][r][s]
                two_rdm[p, q, r, s] = torch.matmul(
                    state.real, torch.matmul(e_pqrs, state.real))
        return one_rdm, two_rdm

    def draw_circuit(self, theta):
        """Draw the qnode circuit for a given theta."""
        return qml.draw(self.qnode, expansion_strategy='device')(theta)

    def init_e_pq(self, restricted=True):
        if self.e_pq is None:
            self.e_pq = initialize_e_pq(self.ncas, restricted,
                                        self.up_then_down)

    def init_e_pqrs(self, restricted=True):
        if self.e_pqrs is None:
            self.e_pqrs = initialize_e_pqrs(self.ncas, restricted,
                                            self.up_then_down)


if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt

    ncas = 3
    nelecas = 2
    dev = qml.device('default.qubit', wires=2*ncas)
    pqc = Parameterized_circuit(ncas, nelecas, dev,
                                ansatz='np_fabric', n_layers=2, add_singles=False)
    print(pqc.redundant_idx)

    theta = torch.rand_like(pqc.init_zeros())
    # theta = pqc.init_zeros()
    # theta = torch.Tensor([[[-0.2,0.3]],
    #                       [[-0.2,0.1]]])
    state = pqc.qnode(theta)
    print("theta = ", theta)
    print("state:", dirac_notation(state.detach().numpy()))
    # # pqc.up_then_down = False
    one_rdm, two_rdm = pqc.get_rdms_from_state(state)
    plt.imshow(one_rdm)
    plt.colorbar()
    plt.show()
    plt.imshow(two_rdm.reshape(ncas**2, ncas**2))
    plt.colorbar()
    plt.show()

    print(pqc.draw_circuit(theta))

    # # def cost(theta):
