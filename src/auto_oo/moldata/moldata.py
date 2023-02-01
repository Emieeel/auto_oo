#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:05:15 2022

@author: emielkoridon
"""
import itertools

import numpy as np
import pennylane as qml
import pyscf
import openfermion

import torch

from auto_oo.moldata import moltools
# from auto_oo.orbital import givens_rotations
from auto_oo.orbital import orbital_rotations


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


class Moldata():
    #%% Initialize attributes of Moldata class:
    def __init__(self, geometry, basis, freeze_redundant=True,
                 vqe_singles=False, **kwargs):
        """
        Moldata class to capture all properties of the molecule in one
        specific geometry

        Args:
            geometry: geometry of molecule
            basis: name of basis set
            freeze_redundant (default: True):
                Split oo indices in redundant and non-redundant.
            vqe_singles (default: False):
                Absorb UCC single excitations in the circuit, instead of a
                classical orbital rotation.
        
        """
        self.mol = pyscf.gto.Mole(atom=geometry,basis=basis, **kwargs)
        self.mol.build()
        self.int1e_AO = torch.from_numpy(
            self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc'))
        self.int2e_AO = torch.from_numpy(
            self.mol.intor('int2e'))
        self.overlap = self.mol.intor('int1e_ovlp')
        self.oao_coeff = ao_to_oao(self.overlap)
        self.nuc = self.mol.get_enuc()
        self.nao = self.overlap.shape[0]
        self.freeze_redundant = freeze_redundant
        self.vqe_singles = vqe_singles
        self.hf = None
        self.occ_idx = None
        self.act_idx = None
        self.virt_idx = None
        self.cas_idx = None
        self.jw_stencils = None
        self.e_pq = None
        self.e_pqrs = None

    #%% Functions for initializing active space parameters
    def define_CAS(self, ncas, nelecas):
        """
        Define active space variables. This also generates the division between
        redundant oo parameters, non-redundant oo parameters and vqe parameters,
        depending on the __init__ input.
        
        Args:
            ncas: number of active space orbitals
            
            nelecas: number of active space electrons
        """
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
        
        # Generate active and redundant orbital rotation indices
        self.gen_redunt_idx()
        
        # Set number of vqe parameters
        singles, doubles = qml.qchem.excitations(nelecas, 2*ncas)
        self.nsingles = len(singles)
        self.ndoubles = len(doubles)
        if self.vqe_singles:
            self.noosingles = 0
            self.nvqeparam = self.nsingles + self.ndoubles
        else:
            self.noosingles = len(self.params_act_act_idx)
            self.nvqeparam = self.ndoubles + self.noosingles
            

    def check_cas(self):
        if self.occ_idx is None:
            raise ValueError('CAS not defined. First run '+
                             'self.define_CAS(as_orbitals, as_electron_pairs)')
    
    #%% Functions for initializing oo and vqe parameters
    def gen_redunt_idx(self):
        """Split oo parameters in two sets, characterized by params_idx and 
        redunt_idx corresponding to non-redundant and redundant rotations
        respectively. Occupied  to occupied, active to active (dealt with in
        the ansatz, either classically or quantumly) and virtual to virtual
        rotations are seen as redundant here.
        """
        self.params_idx = np.array([], dtype=int)
        self.redunt_idx = np.array([], dtype=int)
        self.params_act_act_idx = np.array([], dtype=int)

        num = 0
        l_indices, r_indices = np.tril_indices(self.nao,-1)
        for l_idx, r_idx in zip(l_indices,r_indices):
            if l_idx in self.act_idx and r_idx in self.act_idx:
                self.params_act_act_idx = np.append(self.params_act_act_idx, [num])
                self.redunt_idx = np.append(self.redunt_idx, [num])
            elif (l_idx in self.occ_idx and r_idx in self.occ_idx) or (
                    l_idx in self.virt_idx and r_idx in self.virt_idx):
                self.redunt_idx = np.append(self.redunt_idx, [num])
            else:
                self.params_idx = np.append(self.params_idx, [num])
            num +=1

    def init_full(self, start_from_hf=0):
        """Initialize zero-parameters for orbital-optimized VQE. If start_from_hf=True,
        decompose the canonical HF MOs into orbital rotation parameters.
        """
        if start_from_hf:
            init_oo = self.init_hf()
        else:
            paramsize = int((self.nao*(self.nao-1))/2)
            init_oo = torch.zeros(paramsize)

        if self.freeze_redundant:
            init_oo_params = init_oo[self.params_idx]
            self.redunt_params = init_oo[self.redunt_idx]
        else:
            init_oo_params = init_oo
            self.redunt_params = None
        
        return init_oo_params, torch.zeros(self.nvqeparam)

    def init_hf(self):
        """Initialize parameters by QR decomposition of HF mo_coeff"""
        if self.hf is None:
             self.run_rhf()
        init_oo = self.mo_to_param(self.hf.mo_coeff)
        return torch.Tensor(init_oo)

    def mo_to_param(self, mo_coeff):
        """ Convert AO to MO-coefficient matrix to oo parameters
        """
        if type(mo_coeff) is torch.Tensor:
            mo_np = torch.clone(mo_coeff).detach().numpy()
        else:
            mo_np = np.copy(mo_coeff)
        
        # Calculate the square root of the overlap matrix to estimate 
        # OAO-MO coefficients from AO-MO coefficients
        S_eigval, S_eigvec = np.linalg.eigh(self.overlap)
        S_half = S_eigvec @ np.diag((S_eigval)**(1./2.)) @ S_eigvec.T
        oao_mo_coeff = S_half @ mo_np
        
        # Check if the mo_coeff is in SO(N) or O(N). If the determinant is 1,
        # use freedom in minus sign of MOs to cast the MOs to SO(N).
        if np.allclose(np.linalg.det(oao_mo_coeff),-1):
            print("MOs have determinant -1. Put minus sign on the last MO.")
            oao_mo_coeff_aug = np.copy(oao_mo_coeff)
            oao_mo_coeff_aug[:,-1] = -oao_mo_coeff_aug[:,-1]
            # init_oo = givens_rotations.givens_decomposition(oao_mo_coeff_aug)
            init_oo = orbital_rotations.orbital_decomposition(oao_mo_coeff_aug)
        elif np.allclose(np.linalg.det(oao_mo_coeff),1):
            # init_oo = givens_rotations.givens_decomposition(oao_mo_coeff)
            init_oo = orbital_rotations.orbital_decomposition(oao_mo_coeff)
        else:
            ValueError("Input matrix is not orthogonal!")
        return init_oo
    
    def init_e_pq(self, restricted=False):
        if self.e_pq is None:
            self.e_pq = moltools.initialize_e_pq(self.ncas, restricted)

    def init_e_pqrs(self, restricted=False):
        if self.e_pqrs is None:
            self.e_pqrs = moltools.initialize_e_pqrs(self.ncas, restricted)
    
    
    #%% Function for initializing e_pqrs
    def init_jw_stencils(self):
        """
        Precompute the Jordan-Wigner representation of one and two-body operators
        in the active space, needed for the differentiable CASSCF implementation.
        New implementation with RDMs does NOT need this.
        """
        self.check_cas()
        if self.jw_stencils is None:
            self.jw_stencils = moltools.jordan_wigner_intop_matrices(self.ncas)

    #%% Functions to compute active space Hamiltonians
    def int1e_mo(self, mo_coeff):
        """1e MO-integrals in chemists order"""
        return mo_coeff.T @ self.int1e_AO @ mo_coeff

    def int2e_mo(self, mo_coeff):
        """2e MO-integrals in chemists order"""
        return uniform_4index_transform(self.int2e_AO, mo_coeff)

    def ham_from_mo(self, mo_coeff):
        """Calculate the Hamiltonian from MO-coefficients as an OpenFermion
        interaction operator (non-autodifferentiable)"""
        self.run_rhf()
        int1e = self.int1e_mo(mo_coeff)
        int2e = self.int2e_mo(mo_coeff)
        return moltools.molecular_hamiltonian(self.nuc, int1e, int2e, 
                                     self.occ_idx,
                                     self.act_idx)

    def ham_coeff_from_mo(self, mo_coeff):
        """Calculate the molecular Hamiltonian coefficients (integrals) in spin-orbital basis
        given MO-coefficients"""
        self.run_rhf()
        int1e = self.int1e_mo(mo_coeff)
        int2e = self.int2e_mo(mo_coeff)
        return moltools.molecular_hamiltonian_coefficients(self.nuc, int1e, int2e, 
                                     self.occ_idx,
                                     self.act_idx)

    def cas_ham_from_mo(self, mo_coeff):
        """Generate active space Hamiltonian matrix in Jordan-Wigner representation
        given MO_coefficients"""
        self.init_jw_stencils()
        c0, c1, c2 = self.ham_coeff_from_mo(mo_coeff)
        m0, m1, m2 = self.jw_stencils
        return (c0 * m0,
                torch.einsum('pq, pqij', c1, m1),
                torch.einsum('pqrs, pqrsij', c2, m2))
    
    def fermionic_cas_hamiltonian(self, mo_coeff, restricted=True):
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
        self.init_e_pq(restricted)
        self.init_e_pqrs(restricted)
        int1e = self.int1e_mo(mo_coeff)
        int2e = self.int2e_mo(mo_coeff)
        c0, c1, c2 =  moltools.molecular_hamiltonian_coefficients(
            self.nuc, int1e, int2e, self.occ_idx, self.act_idx,
            restricted = restricted)
        return moltools.fermionic_cas_hamiltonian(c0, c1 , c2, restricted)

    #%% Functions to extract and compute with RDMs

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
            one_rdm[p,q] = state @ e_pq @ state
            for r, s in itertools.product(range(rdm_size),repeat=2):
                    e_pqrs = self.e_pqrs[p][q][r][s]
                    two_rdm[p,q,r,s] = state @ e_pqrs @ state
        return one_rdm, two_rdm

    def get_energy_from_rdms(self, one_rdm, two_rdm, mo_coeff, restricted=True):
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
        self.run_rhf()
        int1e = self.int1e_mo(mo_coeff)
        int2e = self.int2e_mo(mo_coeff)
        c0, c1, c2 =  moltools.molecular_hamiltonian_coefficients(
            self.nuc, int1e, int2e, self.occ_idx, self.act_idx,
            restricted = restricted)
        return (c0,
                torch.einsum('pq, pq', c1, one_rdm),
                torch.einsum('pqrs, pqrs', c2, two_rdm))
                        
    #%% Functions to run PySCF methods on molecule
    def run_rhf(self):
        """Run PySCF RHF"""
        if self.hf is None:
            self.hf = self.mol.RHF(verbose=0).run()
    
    def run_fci(self, n_roots=1, fix_singlet=1):
        """Run PySCF FCI"""
        self.run_rhf()
        if fix_singlet:
            self.fci = pyscf.fci.addons.fix_spin_(
                pyscf.fci.FCI(self.hf), ss=0.)
        else:
            self.fci = pyscf.fci.FCI(self.hf)
        self.fci.nroots = n_roots                                    
        self.fci.kernel(verbose = 0)
        
    def run_casci(self, n_roots=1, mo=None, fix_singlet=1):
        """Run PySCF CASCI"""
        self.check_cas()
        self.run_rhf()
        self.casci = self.hf.CASCI(self.ncas, self.nelecas)
        self.casci.fcisolver.nroots = n_roots
        if fix_singlet:
            self.casci.fix_spin_(shift=1.5,ss=0)
        if mo is not None:
            self.casci.kernel(mo)
        else:
            self.casci.kernel()
        
    def run_casscf(self, fix_singlet=1):
        """Run PySCF CASSCF"""
        self.check_cas()
        self.run_rhf()
        self.casscf = pyscf.mcscf.CASSCF(self.hf, self.ncas, self.nelecas)
        if fix_singlet:
            self.casscf.fix_spin_(ss=0)
        self.casscf.kernel()
    
    def run_sa_casscf(self, fix_singlet=1):
        """Run PySCF SA-CASSCF"""
        self.check_cas()
        self.run_rhf()
        self.sa_casscf = pyscf.mcscf.CASSCF(self.hf,self.ncas,self.nelecas)
        self.sa_casscf.state_average_([0.5,0.5])
        if fix_singlet:
            self.sa_casscf.fcisolver.spin = 0
            self.sa_casscf.fix_spin_(ss=0)
        self.sa_casscf.kernel()
    
    #%% Functions for openfermion exact diagonalization
    def diagonalize_hamiltonian_mo(self, mo_coeff):
        """diagonalize self.ham_from_mo(mo_coeff) in Jordan-Wigner representation
        non-autodifferentiable"""
        intop = self.ham_from_mo(mo_coeff)
        sop = openfermion.get_sparse_operator(intop)
        assert(openfermion.is_hermitian(sop.A))
        return np.linalg.eigh(sop.A)
    
    def cas_ground_energy_MO(self, mo_coeff):
        """Diagonalize active space hamiltonian"""
        eigvals, _ = np.linalg.eigh(self.cas_ham_from_mo(mo_coeff))
        return eigvals[0]
