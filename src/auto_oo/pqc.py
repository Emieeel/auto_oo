#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:39:19 2023

@author: emielkoridon
"""

import itertools

import numpy as np
import pennylane as qml
from pennylane import math
import openfermion

from auto_oo.ansatze.uccd import UCCD
from auto_oo.utils.active_space import e_pq, e_pqrs
from auto_oo.utils.miscellaneous import scipy_csc_to_torch, scipy_csc_to_jax


def initialize_e_pq(ncas, restricted=True, up_then_down=False, interface='scipy'):
    """Initialize full e_pq operator in CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    if interface == 'torch':
        return [[scipy_csc_to_torch(
            openfermion.get_sparse_operator(
                e_pq(p, q, num_ind, restricted, up_then_down), n_qubits=2*ncas))
            for q in range(num_ind)] for p in range(num_ind)]
    elif interface == 'jax':
        return [[scipy_csc_to_jax(
            openfermion.get_sparse_operator(
                e_pq(p, q, num_ind, restricted, up_then_down), n_qubits=2*ncas))
            for q in range(num_ind)] for p in range(num_ind)]
    else:
        return [[openfermion.get_sparse_operator(
                e_pq(p, q, num_ind, restricted, up_then_down), n_qubits=2*ncas)
            for q in range(num_ind)] for p in range(num_ind)]


def initialize_e_pqrs(ncas, restricted=True, up_then_down=False, interface='scipy'):
    """Initialize full e_pqrs operator in CSC sparse format"""
    if restricted:
        num_ind = ncas
    else:
        num_ind = 2 * ncas
    if interface == 'torch':
        return [[[[scipy_csc_to_torch(
            openfermion.get_sparse_operator(
                e_pqrs(p, q, r, s, num_ind, restricted, up_then_down), n_qubits=2*ncas))
            for s in range(num_ind)] for r in range(num_ind)]
            for q in range(num_ind)] for p in range(num_ind)]
    elif interface == 'jax':
        return [[[[scipy_csc_to_jax(
            openfermion.get_sparse_operator(
                e_pqrs(p, q, r, s, num_ind, restricted, up_then_down), n_qubits=2*ncas))
            for s in range(num_ind)] for r in range(num_ind)]
            for q in range(num_ind)] for p in range(num_ind)]
    else:
        return [[[[openfermion.get_sparse_operator(
                e_pqrs(p, q, r, s, num_ind, restricted, up_then_down), n_qubits=2*ncas)
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
    """ Outputs NP fabric ansatz state"""
    qml.GateFabric(theta,
                   wires=wires, init_state=hfstate, include_pi=False)
    return qml.state()


class Parameterized_circuit():
    """ Parameterized quantum circuit class. Defined by an active space of
        nelecas electrons in ncas orbitals. Defines a qnode that outputs a
        quantum state. Can output one and two-RDMs."""

    def __init__(self, ncas, nelecas, dev, ansatz='ucc', n_layers=3,
                 add_singles=False, interface='torch', diff_method='backprop'):
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
        self.interface = interface

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
            self.qnode = qml.qnode(dev, interface=interface, diff_method='backprop')(
                self.uccd_state)

        elif ansatz == 'np_fabric':
            self.n_layers = n_layers
            self.up_then_down = False
            self.wires = list(range(self.n_qubits))
            self.hfstate = math.convert_like(
                qml.qchem.hf_state(nelecas, self.n_qubits), math.zeros(1, like=interface))
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

            self.params_idx = math.convert_like(
                np.array([x for x in range(np.prod(self.full_theta_shape))
                          if x not in self.redundant_idx]),
                math.zeros(1, like=interface))
            self.theta_shape = len(self.params_idx)
            self.qnode = qml.qnode(dev, interface=interface, diff_method='backprop')(
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
        theta_full = math.zeros(len(self.redundant_idx) + self.theta_shape,
                                like=math.get_interface(theta))
        theta_full = math.set_index(theta_full, self.params_idx, theta)
        theta_full = math.reshape(theta_full, self.full_theta_shape)
        # import pdb
        # pdb.set_trace()
        return gatefabric_circuit(theta_full, self.wires, self.hfstate)

    def init_zeros(self):
        """Initialize thetas in all-zero"""
        return math.zeros(self.theta_shape, like=self.interface)

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
        one_rdm = math.convert_like(math.zeros((rdm_size, rdm_size)), state)
        two_rdm = math.convert_like(math.zeros((rdm_size, rdm_size, rdm_size, rdm_size)), state)
        for p, q in itertools.product(range(rdm_size), repeat=2):
            one_rdm = math.set_index(one_rdm, (p, q), (state @ (self.e_pq[p][q] @ state)).real)
            for r, s in itertools.product(range(rdm_size), repeat=2):
                two_rdm = math.set_index(two_rdm, (p, q, r, s),
                                         (state @ (self.e_pqrs[p][q][r][s] @ state)).real)
        return one_rdm, two_rdm

    def get_rdms(self, theta, restricted=True):
        return self.get_rdms_from_state(self.qnode(theta), restricted=restricted)

    def draw_circuit(self, theta):
        """Draw the qnode circuit for a given theta."""
        return qml.draw(self.qnode, expansion_strategy='device')(theta)

    def init_e_pq(self, restricted=True):
        if self.e_pq is None:
            self.e_pq = initialize_e_pq(self.ncas, restricted,
                                        self.up_then_down, self.interface)

    def init_e_pqrs(self, restricted=True):
        if self.e_pqrs is None:
            self.e_pqrs = initialize_e_pqrs(self.ncas, restricted,
                                            self.up_then_down, self.interface)


if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt

    ncas = 3
    nelecas = 4
    dev = qml.device('default.qubit', wires=2*ncas)
    interface = 'torch'

    np.random.seed(30)

    pqc = Parameterized_circuit(ncas, nelecas, dev,
                                ansatz='np_fabric', n_layers=2,
                                add_singles=False, interface=interface)
    # print(pqc.redundant_idx)

    theta = math.convert_like(
        math.random.rand(*math.shape(pqc.init_zeros())), math.zeros(1, like=interface))
    theta = math.cast(theta, np.float32)
    # theta = pqc.init_zeros()
    # theta = torch.Tensor([[[-0.2,0.3]],
    #                       [[-0.2,0.1]]])
    state = pqc.qnode(theta)
    print("theta = ", theta)
    print("state:", dirac_notation(math.convert_like(math.detach(state),
                                                     math.zeros(1, like='numpy'))))
    # # pqc.up_then_down = False
    one_rdm, two_rdm = pqc.get_rdms_from_state(state)
    plt.imshow(one_rdm)
    plt.colorbar()
    plt.show()
    plt.imshow(two_rdm.reshape(ncas**2, ncas**2))
    plt.colorbar()
    plt.show()

    print(pqc.draw_circuit(theta))

    def rdms_from_theta(param):
        return pqc.get_rdms_from_state(pqc.qnode(param))

    theta.requires_grad = True
    if interface == 'torch':
        import torch
        grad = torch.autograd.functional.jacobian(rdms_from_theta, theta)
        # rdms = rdms_from_theta(theta)
        # rdms.backward()
        # grad = theta.grad
    elif interface == 'autograd':
        grad = qml.grad(rdms_from_theta, argnum=0)(theta)

    #

    # grad_check = qml.jacobian(rdms_from_theta, argnum=0)(theta)

    # # def cost(theta):
