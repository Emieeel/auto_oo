#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:05:15 2022

@author: emielkoridon
"""
import itertools

import numpy as np
import openfermion

import torch

from auto_oo.moldata import moltools

def vec_to_skew_symmetric(vector):
    r"""
    Map a vector to an anti-symmetric matrix with torch.tril_indices.
    
    For example, the resulting matrix for `torch.Tensor([1,2,3,4,5,6])` is:

    .. math::
        \begin{pmatrix}
            0 & -1 & -2 & -4\\
            1 &  0 & -3 & -5\\
            2 &  3 &  0 & -6\\
            4 &  5 &  6 &  0
        \end{pmatrix}

    Args:
        params (torch.Tensor): 1d tensor of parameters
·
    """
    size = int(np.sqrt(8 * len(vector) + 1) + 1)//2
    matrix = torch.zeros((size,size))
    tril_indices = torch.tril_indices(row=size,col=size, offset=-1)
    matrix[tril_indices[0],tril_indices[1]] = vector
    matrix[tril_indices[1],tril_indices[0]] = - vector
    return matrix


def general_4index_transform(M, C0, C1, C2, C3):
    """
    M is a rank-4 tensor, Cs are rank-2 tensors representing ordered index 
    transformations of M
    """
    M = torch.einsum('pi, pqrs', C0, M)
    M = torch.einsum('qj, iqrs', C1, M)
    M = torch.einsum('rk, ijrs', C2, M)
    M = torch.einsum('sl, ijks', C3, M)
    return M

def uniform_4index_transform(M, C):
    """
    Autodifferentiable index transformation for two-electron tensor.
    
    Note: on a test case (dimension 13) this is 3x faster than optimized einsum,
        and >1000x more efficient than unoptimized einsum.
        Computing the Jacobian is also very efficient.
    """
    return general_4index_transform(M, C, C, C, C)

def ao_to_oao(ovlp):
    """orthogonal atomic orbitals in terms of atomic orbitals"""
    S_eigval, S_eigvec = np.linalg.eigh(ovlp)
    return S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ S_eigvec.T


class Parameterized_active_space():
    #%% Initialize attributes of a Parameterized Active Space class:
    def __init__(self, mol, ncas, nelecas, freeze_active=False):
        """
        Class that can generate energies, orbital gradients and orbital
        Hessians of an active space Hamiltonian from RDMs.

        Args:
            molecule: MolecularData instance (openfermion or pyscf ? )
                specifiying geometry, basisset and other molecular properties.
            mo_coeff: Reference OAO-MO coefficients (ndarray)
            ncas: Number of active orbitals
            nelecas: Number of active electrons
            freeze_active (default: False):
                Freeze active-active rotations in the kappa matrix
        
        """
        self.int1e_AO = torch.from_numpy(
            mol.intor('int1e_kin') + self.mol.intor('int1e_nuc'))
        self.int2e_AO = torch.from_numpy(
            mol.intor('int2e'))
        self.overlap = mol.intor('int1e_ovlp')
        self.oao_coeff = ao_to_oao(self.overlap)
        self.nuc = self.mol.get_enuc()
        self.nao = self.overlap.shape[0]

        # Set active space parameters
        self.ncas = ncas
        self.nelecas = nelecas
        self.nelecore = self.mol.nelectron - nelecas
        if self.nelecore % 2 == 1:
            raise ValueError('odd number of core electrons')
            
        self.occ_idx = np.arange(self.nelecore // 2)
        self.act_idx = (self.occ_idx[-1] + 1 + np.arange(ncas)
                               if len(self.occ_idx) > 0 
                               else np.arange(ncas))
        self.virt_idx = np.arange(self.act_idx[-1]+1,self.mol.nao) 
        
        rotation_sizes = [len(self.occ_idx)*len(self.act_idx),
                          len(self.act_idx)*len(self.virt_idx),
                          len(self.occ_idx)*len(self.virt_idx)]
        if not freeze_active:
            rotation_sizes.append(
                len(self.act_idx) * (len(self.act_idx) - 1)//2)
        self.kappa_len = sum(rotation_sizes)

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
    

    def energy_from_rdms(self, one_rdm, two_rdm, kappa):
        r"""
        Get total energy given the one- and two-particle reduced density matrices
        :math:`\gamma_{pq}` and :math:`\Gamma_{pqrs}`.
        Total energy is thus:

        .. math::
            E = E_{\rm nuc} + E_{\rm core} +
            \sum_{pq}\tilde{h}_{pq} \gamma_{pq} + 
            \sum_{pqrs} g_{pqrs} \Gamma_{pqrs}

        where :math:`E_{core}` is the mean-field energy of the core (doubly-occupied) orbitals,
        :math:`\tilde{h}_{pq}` is contains the active one-body terms plus the mean-field
        interaction of core-active orbitals and :math:`g_{pqrs}` are the active integrals
        in chemist ordering.
        """
        
        if not len(kappa) == self.kappa_len:
            ValueError(
                f'length of kappa is {len(kappa)}, while it should be {self.kappa_len}')
        mo_coeff = self.transform_mo_coeff(kappa)
        int1e = self.int1e_mo(mo_coeff)
        int2e = self.int2e_mo(mo_coeff)
        c0, c1, c2 =  moltools.molecular_hamiltonian_coefficients(
            self.nuc, int1e, int2e, self.occ_idx, self.act_idx)
        return sum(c0,
                   torch.einsum('pq, pq', c1, one_rdm),
                   torch.einsum('pqrs, pqrs', c2, two_rdm))
    
    def transform_mo_coeff(self, kappa):
        kappa_vec = torch.zeros(size=(self.nao*(self.nao-1)//2))
        kappa_vec[self.params_idx] = kappa 
        kappa_matrix = vec_to_skew_symmetric(kappa_vec)
        return self.mo_coeff @ torch.linalg.matrix_exp(-kappa_matrix)

    def oao_mo_coeff(self, mo_coeff):
        """ Convert AO-MO coefficients to OAO-MO coefficients
        """
        if type(mo_coeff) is torch.Tensor:
            mo_np = torch.clone(mo_coeff).detach().numpy()
        else:
            mo_np = np.copy(mo_coeff)
        
        # Calculate the square root of the overlap matrix to estimate 
        # OAO-MO coefficients from AO-MO coefficients
        S_eigval, S_eigvec = np.linalg.eigh(self.overlap)
        S_half = S_eigvec @ np.diag((S_eigval)**(1./2.)) @ S_eigvec.T
        return S_half @ mo_np

    #%% Functions for 4-index transformations
    def int1e_mo(self, mo_coeff):
        """1e MO-integrals in chemists order"""
        return mo_coeff.T @ self.int1e_AO @ mo_coeff

    def int2e_mo(self, mo_coeff):
        """2e MO-integrals in chemists order"""
        return uniform_4index_transform(self.int2e_AO, mo_coeff)
