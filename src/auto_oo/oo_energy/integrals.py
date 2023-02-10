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
        params (np.Tensor): 1d tensor of parameters
·
    """
    size = int(np.sqrt(8 * len(vector) + 1) + 1)//2
    matrix = np.zeros((size,size))
    tril_indices = np.tril_indices(row=size,col=size, offset=-1)
    matrix[tril_indices[0],tril_indices[1]] = vector
    matrix[tril_indices[1],tril_indices[0]] = - vector
    return matrix


def general_4index_transform(M, C0, C1, C2, C3):
    """
    M is a rank-4 tensor, Cs are rank-2 tensors representing ordered index 
    transformations of M
    """
    M = np.einsum('pi, pqrs', C0, M)
    M = np.einsum('qj, iqrs', C1, M)
    M = np.einsum('rk, ijrs', C2, M)
    M = np.einsum('sl, ijks', C3, M)
    return M

def uniform_4index_transform(M, C):
    """
    Autodifferentiable index transformation for two-electron tensor.
    
    Note: on a test case (dimension 13) this is 3x faster than optimized einsum,
        and >1000x more efficient than unoptimized einsum.
        Computing the Jacobian is also very efficient.
    """
    return general_4index_transform(M, C, C, C, C)

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

def fermionic_cas_hamiltonian(c0, c1, c2, e_pq, e_pqrs, restricted = True):
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

def s2(ncas, nelecas):
    nqubits = 2 * ncas
    s2_ham = qml.qchem.spin2(nelecas,nqubits)
    s2 = qml.matrix(s2_ham)
    return s2

def sz(ncas):
    nqubits = 2 * ncas
    sz_ham = qml.qchem.spinz(nqubits)
    return qml.matrix(sz_ham)

def fock_core(one_body_integrals, two_body_integrals, occ_idx):
    g_tilde = (
        2 * torch.sum(two_body_integrals[:, :, occ_idx, occ_idx], dim=-1) # p^ i^ i q 
          - torch.sum(two_body_integrals[:, occ_idx, occ_idx, :], dim=1)) # p^ i^ q i
    return one_body_integrals + g_tilde

def fock_active(one_body_integrals, two_body_integrals, one_rdm, act_idx):
    g_tilde = (
    two_body_integrals[:,:,:,act_idx][:,:,act_idx,:]
    -.5 * torch.permute(two_body_integrals[:,:,act_idx,:][:,act_idx,:,:],(0,3,2,1)))
    return torch.einsum('wx, pqwx', one_rdm, g_tilde)

def fock_generalized(one_body_integrals, two_body_integrals,
                     one_rdm, two_rdm, occ_idx, act_idx):
    fock_C = fock_core(one_body_integrals, two_body_integrals, occ_idx)
    fock_A = fock_active(one_rdm, two_body_integrals, act_idx)
    fock_general = torch.zeros(one_body_integrals.shape)
    fock_general[occ_idx,:] = 2 * torch.t(fock_C[:,occ_idx] + fock_A[:,occ_idx])
    fock_general[act_idx,:] = torch.einsum('qw,vw->vq',fock_C[:,act_idx],one_rdm) + torch.einsum(
        'vwxy,qwxy->vq',two_rdm,two_body_integrals[:,:,:,act_idx][:,:,act_idx,:][:,act_idx,:,:])
    return fock_general



  
