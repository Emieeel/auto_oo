#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:58:49 2022

@author: emielkoridon
"""

import itertools

import numpy as np
import pennylane as qml

import torch

import openfermion
import pyscf

# construct the combinations of delta functions in einstein notation
# p=q=r=s (same spin):
_eye4d = torch.einsum('ia, ib, ic, id', *[torch.eye(2)]*4)

# p=q!=r=s (mix):
_X = torch.Tensor([[0., 1.], [1., 0.]])
_mix4d = torch.einsum('ia, ib, ic, id', torch.eye(2), _X, _X, torch.eye(2))

_spin_comp_tensor = (_eye4d + _mix4d) / 2

def restricted_to_unrestricted(tensor):
    '''
    Transform the one- or two-electron integral tensor from restricred to 
    unrestricted representation, i.e. adding the spin. 
    Interleaved indices spin convention (even for alpha, odd for beta).
    
    NB: Physicst's ordering convention required for the two-body integrals
    '''
    s = tensor.size()
    if len(s) == 2: # two-electron integrals
        tensor = torch.einsum('pq, ab -> paqb', tensor, torch.eye(2))
    elif len(s) == 4:
        tensor = torch.einsum('ijkl, abcd -> iajbkcld', tensor, _spin_comp_tensor)
    else:
        raise ValueError("Only shape 2- or 4-dimensional tensors supported.")
    return torch.reshape(tensor, [i*2 for i in s])

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
        2 * torch.sum(one_body_integrals[occ_idx, occ_idx]) + # i^ j
        2 * torch.sum(two_body_integrals[
            occ_idx, occ_idx, :, :][
            :, occ_idx, occ_idx])  # i^ j^ j i 
        - torch.sum(two_body_integrals[
            occ_idx, :, :, occ_idx][
            :, occ_idx, occ_idx]) # i^ j^ i j 
    )
    
    # restrict range to active indices only
    as_two_body_integrals = two_body_integrals[np.ix_(*[act_idx]*4)]
    
    # --- Modified one electron integrals ---            
    # sum over i in occ_idx
    as_one_body_integrals = (
        one_body_integrals[np.ix_(*[act_idx]*2)]
        + 2 * torch.sum(two_body_integrals[:, :, occ_idx, occ_idx
                                       ][act_idx, :, :][:, act_idx, :],
                     dim = 2) # i^ p^ q i
        - torch.sum(two_body_integrals[:, occ_idx, occ_idx, :][
            act_idx, :, :][:, :, act_idx], dim = 1) # i^ p^ i q
    )

    # Restrict integral ranges and change M appropriately
    return (core_constant,
            as_one_body_integrals,
            as_two_body_integrals)

def molecular_hamiltonian_coefficients(nuclear_repulsion,
                                       one_body_integrals,
                                       two_body_integrals,
                                       occ_idx = None,
                                       act_idx = None):
    '''
    Transform full-space restricted orbitals in chemist's notation 
    to CAS (un)restricted Hamiltonian coefficient's in 
    (physicist) chemist notation.
    
    
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

def molecular_hamiltonian(nuclear_repulsion,
                          one_body_integrals,
                          two_body_integrals,
                          occ_idx = None,
                          act_idx = None):
    '''
    Creates an electronic structure Hamiltonian from zero- one- and two-electron
    integrals (in restricted form i.e. no spin, and with chemist's index
    ordering convention)
    If indices are given, the CAS active space Hamiltonian is constructed.
    
    returns: `openfermion.InteractionOperator`: AS (or full-space) Hamiltonian 
        according to the interleaved-spin convention (alpha-even, beta-odd)
    '''
    # Initialize Hamiltonian coefficients.
    (E_constant,
     one_body_coefficients,
     two_body_coefficients) = molecular_hamiltonian_coefficients(
        nuclear_repulsion,
        one_body_integrals,
        two_body_integrals,
        occ_idx,
        act_idx)
                 
    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = openfermion.InteractionOperator(
        E_constant.item(), 
        one_body_coefficients.detach().numpy(), 
        two_body_coefficients.detach().numpy())

    return molecular_hamiltonian

def fermionic_cas_hamiltonian(c0, c1, c2, restricted = True):
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
    hamiltonian = openfermion.FermionOperator('',c0.item())
    one_body_op = openfermion.FermionOperator()
    for p,q in itertools.product(range(ncas), repeat=2):
        one_body_op += c1[p,q].item() * e_pq(p,q, restricted)
    two_body_op = openfermion.FermionOperator()
    for p,q,r,s in itertools.product(range(ncas), repeat=4):
        two_body_op += c2[p,q,r,s].item() * e_pqrs(p,q,r,s,restricted)
    hamiltonian += one_body_op + two_body_op
    return hamiltonian

def jw_intop_1body(p, q, n_qubits):
    '''OLD
    '''
    return np.real(openfermion.get_sparse_operator(
                   openfermion.FermionOperator(((p, 1),(q,0)),), 
                   n_qubits=n_qubits).A)

def jw_intop_2body(p, q, r, s, n_qubits):
    '''OLD
    in physicist order'''
    return np.real(openfermion.get_sparse_operator(
                openfermion.FermionOperator(((p, 1),(q,1),(r, 0),(s,0)),), 
                n_qubits=n_qubits).A)

def jordan_wigner_intop_matrices(ncas):
    '''
    OLD
    in physicist order
    '''
    nq = 2*ncas
    _eye = torch.eye(2**nq)
    _e_pq = torch.zeros((nq, nq, 2**nq, 2**nq),)
    _e_pqrs = torch.zeros((nq, nq, nq, nq, 2**nq, 2**nq),)
    
    for p in range(nq):
        for q in range(nq):
            _e_pq[p,q,:,:] = torch.from_numpy(
                jw_intop_1body(p, q, nq))
            
    for p in range(nq):
        for q in range(nq):
            for r in range(nq):
                for s in range(nq):
                    _e_pqrs[p,q,r,s,:,:] = torch.from_numpy(
                        jw_intop_2body(p, q, r, s, nq))
                    
    return _eye, _e_pq, _e_pqrs

def s2(ncas, nelecas):
    nqubits = 2 * ncas
    s2_ham = qml.qchem.spin2(nelecas,nqubits)
    s2 = qml.matrix(s2_ham)
    return s2

def sz(ncas):
    nqubits = 2 * ncas
    sz_ham = qml.qchem.spinz(nqubits)
    return qml.matrix(sz_ham)

def pyscf_ci_to_psi(mc, ncas, nelecas):
    psi = np.zeros(2**(2*ncas))
    occslst = pyscf.fci.cistring._gen_occslst(range(ncas), nelecas//2)
    for i,occsa in enumerate(occslst):
        for j,occsb in enumerate(occslst):
            alpha_bin = [1 if x in occsa else 0 for x in range(ncas)]
            beta_bin = [1 if y in occsb else 0 for y in range(ncas)]
            alpha_bin.reverse()
            beta_bin.reverse()
            idx = 0
            for spatorb in range(ncas):
                if alpha_bin[spatorb] == 1:
                    idx += 2**(2*spatorb+1)
                if beta_bin[spatorb] == 1:
                    idx += 2**(2*spatorb)
            psi[idx] = mc.ci[i,j]
    return psi


def e_pq(p, q, restricted = False):
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

def e_pqrs(p, q, r, s, restricted = False):
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
  
