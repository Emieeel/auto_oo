#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:06:01 2023

@author: emielkoridon
"""

import itertools

import numpy as np

import torch

from auto_oo.oo_energy import integrals

torch.set_default_tensor_type(torch.DoubleTensor)

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

def int1e_transform(int1e_ao, mo_coeff):
    """Transform 1e MO-integrals"""
    return mo_coeff.T @ int1e_ao @ mo_coeff

def int2e_transform(int2e_ao, mo_coeff):
    """Transform 2e MO-integrals"""
    return uniform_4index_transform(int2e_ao, mo_coeff)


def mo_ao_to_mo_oao(mo_coeff, overlap):
    """ Convert AO-MO coefficients to OAO-MO coefficients (numpy arrays)
    """
    # Calculate the square root of the overlap matrix to estimate 
    # OAO-MO coefficients from AO-MO coefficients
    S_eigval, S_eigvec = np.linalg.eigh(overlap)
    S_half = S_eigvec @ np.diag((S_eigval)**(1./2.)) @ S_eigvec.T
    return S_half @ mo_coeff
    

class OO_energy():
    def __init__(self, mol, ncas, nelecas,
                 oao_mo_coeff=None, freeze_active=False):
        """
        Orbital Optimized energy class for extracting energies for any given set of
        RDMs. Can compute orbital gradients and hessians.

        Args:
            mol: Moldata_pyscf class containing molecular information like
                geometry, AO basis, MO basis and 1e- and 2e-integrals
            ncas: Number of active orbitals
            nelecas: Number of active electrons
            oao_mo_coeff (default None): Reference OAO-MO coefficients (ndarray)
            freeze_active (default: False):
                Freeze active-active oo indices
        """
        # Set molecular data
        self.int1e_ao = torch.from_numpy(mol.int1e_ao)
        self.int2e_ao = torch.from_numpy(mol.int2e_ao)
        self.overlap = mol.overlap
        self.oao_coeff = torch.from_numpy(mol.oao_coeff)
        self.nuc = mol.nuc
        self.nao = mol.nao

        if oao_mo_coeff is None:
            print("Starting with canonical HF MOs")
            mol.run_rhf()
            self.oao_mo_coeff = torch.from_numpy(
                mo_ao_to_mo_oao(mol.hf.mo_coeff, mol.overlap))
        else:
            if type(oao_mo_coeff) == np.ndarray:
                self.oao_mo_coeff = torch.from_numpy(oao_mo_coeff)
            else:
                self.oao_mo_coeff = oao_mo_coeff

        # Set active space parameters
        self.ncas = ncas
        self.nelecas = nelecas
        
        self.occ_idx, self.act_idx, self.virt_idx = mol.get_active_space_idx(
            ncas, nelecas)
    
    @property
    def mo_coeff(self):
        return self.oao_coeff @ self.oao_mo_coeff
    
    def energy_from_mo_coeff(self, mo_coeff, one_rdm, two_rdm):
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
        c0, c1, c2 = self.get_active_integrals(mo_coeff)
        return sum((c0,
                    torch.einsum('pq, pq', c1, one_rdm),
                    torch.einsum('pqrs, pqrs', c2, two_rdm)))

    
    def get_active_integrals(self, mo_coeff):
        """Transform full-space restricted orbitals to CAS restricted Hamiltonian 
        coefficient's in chemist notation."""
        int1e_mo = int1e_transform(self.int1e_ao, mo_coeff)
        int2e_mo = int2e_transform(self.int2e_ao, mo_coeff)
        return integrals.molecular_hamiltonian_coefficients(
            self.nuc, int1e_mo, int2e_mo, self.occ_idx, self.act_idx)

    def fock_core(self, int1e_mo, int2e_mo):
        g_tilde = (
            2 * torch.sum(int2e_mo[:, :, self.occ_idx, self.occ_idx],
                          dim=-1) # p^ i^ i q 
              - torch.sum(int2e_mo[:, self.occ_idx, self.occ_idx, :],
                          dim=1)) # p^ i^ q i
        return int1e_mo + g_tilde

    def fock_active(self, int2e_mo, one_rdm):
        g_tilde = (
        int2e_mo[:,:,:,self.act_idx][:,:,self.act_idx,:]
        -.5 * torch.permute(int2e_mo[:,:,self.act_idx,:][:,self.act_idx,:,:],
                            (0,3,2,1)))
        return torch.einsum('wx, pqwx', one_rdm, g_tilde)

    def fock_generalized(self, int1e_mo, int2e_mo, one_rdm, two_rdm):
        fock_C = self.fock_core(int1e_mo, int2e_mo)
        fock_A = self.fock_active(int2e_mo, one_rdm)
        fock_general = torch.zeros(int1e_mo.shape)
        fock_general[self.occ_idx,:] = 2 * torch.t(
            fock_C[:,self.occ_idx] + fock_A[:,self.occ_idx])
        fock_general[self.act_idx,:] = torch.einsum(
            'qw,vw->vq',fock_C[:,self.act_idx],one_rdm) + torch.einsum(
            'vwxy,qwxy->vq',
            two_rdm,
            int2e_mo[:,:,:,self.act_idx][:,:,self.act_idx,:][:,self.act_idx,:,:])
        return fock_general

    def analytic_gradient(self, one_rdm, two_rdm, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            mo_coeff = mo_coeff

        int1e_mo = int1e_transform(self.int1e_ao, mo_coeff)
        int2e_mo = int2e_transform(self.int2e_ao, mo_coeff)
        
        fock_general = self.fock_generalized(int1e_mo, int2e_mo, one_rdm, two_rdm)
        return 2 * (fock_general - torch.t(fock_general))
    
    def analytic_hessian(self, one_rdm, two_rdm, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            mo_coeff = mo_coeff

        int1e_mo = int1e_transform(self.int1e_ao, mo_coeff)
        int2e_mo = int2e_transform(self.int2e_ao, mo_coeff)

        one_full, two_full = self.full_rdms(one_rdm, two_rdm)
        y_matrix = self.y_matrix(int2e_mo, two_full)
        fock_general = self.fock_generalized(int1e_mo, int2e_mo, one_rdm, two_rdm)
        fock_general_symm =  fock_general + torch.t(fock_general)

        hess0 = 2 * torch.einsum('pr, qs->pqrs', one_full, int1e_mo)
        hess1 = - torch.einsum('pr, qs->pqrs', fock_general_symm, torch.eye(self.nao))
        hess2 = 2 * y_matrix

        hess_permuted0 = hess0 + hess1 + hess2
        hess_permuted1 = torch.permute(hess_permuted0, (0,1,3,2))
        hess_permuted2 = torch.permute(hess_permuted0, (1,0,2,3))
        hess_permuted3 = torch.permute(hess_permuted0, (1,0,3,2))

        return hess_permuted0 - hess_permuted1 - hess_permuted2 + hess_permuted3

    def full_rdms(self, one_rdm, two_rdm):
        one_full = torch.zeros((self.nao,self.nao))
        two_full = torch.zeros((self.nao,self.nao,self.nao,self.nao))
    
        one_full[self.occ_idx,self.occ_idx] = 2 * torch.ones(len(self.occ_idx))
        one_full[np.ix_(self.act_idx,self.act_idx)] = one_rdm
    
        two_full[np.ix_(*[self.occ_idx]*4)] = 4 * torch.einsum(
            'ij,kl->ijkl',*[torch.eye(len(self.occ_idx))]*2) - 2 * torch.einsum(
            'il,jk->ijkl',*[torch.eye(len(self.occ_idx))]*2)
        two_full[np.ix_(self.occ_idx,self.occ_idx,self.act_idx,self.act_idx)] = 2 * torch.einsum(
            'wx,ij->ijwx',one_rdm,torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.act_idx,self.act_idx,self.occ_idx,self.occ_idx)] = 2 * torch.einsum(
            'wx,ij->wxij',one_rdm,torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.occ_idx,self.act_idx,self.act_idx,self.occ_idx)] = -torch.einsum(
            'wx,ij->iwxj',one_rdm,torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.act_idx,self.occ_idx,self.occ_idx,self.act_idx)] = -torch.einsum(
            'wx,ij->xjiw',one_rdm,torch.eye(len(self.occ_idx)))
        two_full[np.ix_(*[self.act_idx]*4)] = two_rdm
        return one_full, two_full
    
    def y_matrix(self, int2e_mo, two_full):
        y0 = torch.einsum('pmrn, qmns->pqrs', two_full, int2e_mo)
        y1 = torch.einsum('pmnr, qmns->pqrs', two_full, int2e_mo)
        y2 = torch.einsum('prmn, qsmn->pqrs', two_full, int2e_mo)
        return y0 + y1 + y2
        
        
        
        
        
        
