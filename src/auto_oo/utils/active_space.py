#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:58:49 2022

@author: emielkoridon
"""

import itertools

import numpy as np

import pennylane as qml
from pennylane import math

import openfermion

# construct the combinations of delta functions in einstein notation
# p=q=r=s (same spin):
_eye4d = math.einsum('ia, ib, ic, id', *[math.eye(2)]*4)

# p=q!=r=s (mix):
_X = math.array([[0., 1.], [1., 0.]])
_mix4d = math.einsum('ia, ib, ic, id', math.eye(2), _X, _X, math.eye(2))

_spin_comp_tensor = (_eye4d + _mix4d) / 2


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


def restricted_to_unrestricted(tensor, alpha_then_beta=False):
    '''
    Transform the one- or two-electron integral tensor from restricred to
    unrestricted representation, i.e. adding the spin.
    Interleaved indices spin convention (even for alpha, odd for beta).

    NB: Physicst's ordering convention required for the two-body integrals
    '''
    s = tensor.size()
    if len(s) == 2:  # two-electron integrals
        if alpha_then_beta:
            tensor = math.einsum('pq, ab -> apbq', tensor, math.convert_like(
                math.eye(2), tensor))
        else:
            tensor = math.einsum('pq, ab -> paqb', tensor, math.convert_like(
                math.eye(2), tensor))
    elif len(s) == 4:
        tensor = math.einsum('ijkl, abcd -> iajbkcld', tensor, math.convert_like(
            _spin_comp_tensor, tensor))
    else:
        raise ValueError("Only shape 2- or 4-dimensional tensors supported.")

    return math.reshape(tensor, [i*2 for i in s])


def active_space_integrals(one_body_integrals,
                           two_body_integrals,
                           occ_idx,
                           act_idx):
    """
    Restricts a molecule at a spatial orbital level to an active space
    This active space may be defined by a list of active indices and
    doubly occupied indices. Note that one_body_integrals and
    two_body_integrals must be defined in an orthonormal basis set (MO like).

    NB: Chemists's ordering convention required for the two-body integrals

    ---

    Args:
         - one_body_integrals: spatial [p^ q] integrals
         - two_body_integrals: spatial [p^ r^ s q] integrals (chemist's order)
         - occ_idx: A list of spatial orbital indices
           indicating which orbitals should be considered doubly occupied.
         - act_idx: A list of spatial orbital indices indicating
           which orbitals should be considered active.

    Returns:
        `tuple`: Tuple with the following entries:
            - core_constant: Adjustment to constant shift in Hamiltonian
                from integrating out core orbitals
            - as_one_body_integrals: one-electron integrals over active space.
            - as_two_body_integrals: two-electron integrals over active space.
    """
    # --- Determine core constant ---
    core_constant = (
        2 * math.sum(one_body_integrals[occ_idx, occ_idx]) +  # i^ j
        2 * math.sum(two_body_integrals[
            occ_idx, occ_idx, :, :][
            :, occ_idx, occ_idx])  # i^ j^ j i
        - math.sum(two_body_integrals[
            occ_idx, :, :, occ_idx][
            :, occ_idx, occ_idx])  # i^ j^ i j
    )

    # restrict range to active indices only
    as_two_body_integrals = two_body_integrals[np.ix_(*[act_idx]*4)]

    # --- Modified one electron integrals ---
    # sum over i in occ_idx
    as_one_body_integrals = (
        one_body_integrals[np.ix_(*[act_idx]*2)]
        + 2 * math.sum(two_body_integrals[:, :, occ_idx, occ_idx
                                          ][act_idx, :, :][:, act_idx, :],
                       axis=2)  # i^ p^ q i
        - math.sum(two_body_integrals[:, occ_idx, occ_idx, :][
            act_idx, :, :][:, :, act_idx], axis=1)  # i^ p^ i q
    )

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            as_one_body_integrals,
            as_two_body_integrals)


def molecular_hamiltonian_coefficients(nuclear_repulsion,
                                       one_body_integrals,
                                       two_body_integrals,
                                       occ_idx=None,
                                       act_idx=None):
    '''
    Transform full-space restricted orbitals to CAS restricted Hamiltonian
    coefficient's in chemist notation.

    The resulting tensors are ready for openfermion.InteractionOperator, and
    follow the same conventions

    Returns: `tuple` consisting of
        - E_constant (one dimensional tensor)
        - one_body_coefficients (2-dimensional tensor)
        - two_body_coefficients (4-dimensional tensor)

    '''

    # Build CAS
    if occ_idx is None and act_idx is None:
        E_constant = nuclear_repulsion
    else:
        (core_adjustment,
         one_body_integrals,
         two_body_integrals) = active_space_integrals(one_body_integrals,
                                                      two_body_integrals,
                                                      occ_idx,
                                                      act_idx)
        E_constant = core_adjustment + nuclear_repulsion

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = one_body_integrals
    two_body_coefficients = 0.5 * two_body_integrals

    return E_constant, one_body_coefficients, two_body_coefficients


def fermionic_cas_hamiltonian(c0, c1, c2, restricted=True, up_then_down=False):
    r"""
    Generate active space Hamiltonian in FermionOperator form. For now, only works with
    restricted e_pq and e_pqrs, where p,q,r,s are active indices.
    The total Hamiltonian is thus:

    .. math::
        H = E_{\rm nuc} + E_{\rm core} +
        \sum_{pq}\tilde{h}_{pq} E_{pq} +
        \sum_{pqrs} g_{pqrs} e_{pqrs}

    where :math:`E_{core}` is the mean-field energy of the core (doubly-occupied) orbitals,
    :math:`\tilde{h}_{pq}` is contains the active one-body terms plus the mean-field
    interaction of core-active orbitals and :math:`g_{pqrs}` are the active integrals
    in chemist ordering.
    """
    ncas = c1.shape[0]
    hamiltonian = openfermion.FermionOperator('', c0.item())
    one_body_op = openfermion.FermionOperator()
    for p, q in itertools.product(range(ncas), repeat=2):
        one_body_op += c1[p, q].item() * e_pq(p, q, ncas, restricted, up_then_down)
    two_body_op = openfermion.FermionOperator()
    for p, q, r, s in itertools.product(range(ncas), repeat=4):
        two_body_op += c2[p, q, r, s].item() * e_pqrs(p, q, r, s, ncas, restricted, up_then_down)
    hamiltonian += one_body_op + two_body_op
    return hamiltonian


def s2(ncas, nelecas):
    nqubits = 2 * ncas
    s2_ham = qml.qchem.spin2(nelecas, nqubits)
    s2 = qml.matrix(s2_ham)
    return s2


def sz(ncas):
    nqubits = 2 * ncas
    sz_ham = qml.qchem.spinz(nqubits)
    return qml.matrix(sz_ham)
