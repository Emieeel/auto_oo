#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:05:15 2022

@author: emielkoridon
"""

import numpy as np
from pyscf import gto, fci, mcscf


def ao_to_oao(ovlp):
    """orthogonal atomic orbitals in terms of atomic orbitals"""
    S_eigval, S_eigvec = np.linalg.eigh(ovlp)
    return S_eigvec @ np.diag((S_eigval)**(-1./2.)) @ S_eigvec.T


class Moldata_pyscf():
    def __init__(self, geometry, basis, **kwargs):
        """
        Class that can import molecular info from PySCF

        Args:
            geometry: geometry of molecule
            basis: name of basis set
        """
        self.mol = gto.Mole(atom=geometry, basis=basis, **kwargs)
        self.mol.build()
        self.int1e_ao = self.mol.intor('int1e_kin') + self.mol.intor('int1e_nuc')
        self.int2e_ao = self.mol.intor('int2e')
        self.overlap = self.mol.intor('int1e_ovlp')
        self.oao_coeff = ao_to_oao(self.overlap)
        self.nuc = self.mol.get_enuc()
        self.nao = self.overlap.shape[0]
        self.hf = None
        self.fci = None
        self.casci = None
        self.casscf = None
        self.sa_casscf = None

    def get_active_space_idx(self, ncas, nelecas):
        """Calculate the active spaces indices based on the amount of active
        orbitals and electrons. Returns three list with orbital indices."""
        # Set active space parameters
        nelecore = self.mol.nelectron - nelecas
        if nelecore % 2 == 1:
            raise ValueError('odd number of core electrons')

        occ_idx = np.arange(nelecore // 2)
        act_idx = (occ_idx[-1] + 1 + np.arange(ncas)
                   if len(occ_idx) > 0
                   else np.arange(ncas))
        virt_idx = np.arange(act_idx[-1]+1, self.mol.nao)

        return occ_idx, act_idx, virt_idx

    def run_rhf(self, verbose=0):
        """Run PySCF RHF"""
        if self.hf is None:
            self.hf = self.mol.RHF(verbose=verbose).run()

    def run_fci(self, n_roots=1, fix_singlet=1, verbose=0):
        """Run PySCF FCI"""
        self.run_rhf()
        if fix_singlet:
            self.fci = fci.addons.fix_spin_(
                fci.FCI(self.hf), ss=0.)
        else:
            self.fci = fci.FCI(self.hf)
        self.fci.nroots = n_roots
        self.fci.kernel(verbose=verbose)

    def run_casci(self, ncas, nelecas, n_roots=1, mo=None, fix_singlet=1, verbose=0):
        """Run PySCF CASCI"""
        self.run_rhf()
        self.casci = self.hf.CASCI(ncas, nelecas)
        self.casci.fcisolver.nroots = n_roots
        if fix_singlet:
            self.casci.fix_spin_(shift=1.5, ss=0)
        self.casci.verbose = verbose
        if mo is not None:
            self.casci.kernel(mo)
        else:
            self.casci.kernel()

    def run_casscf(self, ncas, nelecas, fix_singlet=1, verbose=0):
        """Run PySCF CASSCF"""
        self.run_rhf()
        self.casscf = mcscf.CASSCF(self.hf, ncas, nelecas)
        if fix_singlet:
            self.casscf.fix_spin_(ss=0)
        self.casscf.verbose = verbose
        self.casscf.kernel()

    def run_sa_casscf(self, ncas, nelecas, fix_singlet=1, verbose=0):
        """Run PySCF SA-CASSCF"""
        self.run_rhf()
        self.sa_casscf = mcscf.CASSCF(self.hf, ncas, nelecas)
        self.sa_casscf.state_average_([0.5, 0.5])
        if fix_singlet:
            self.sa_casscf.fcisolver.spin = 0
            self.sa_casscf.fix_spin_(ss=0)
        self.sa_casscf.verbose = verbose
        self.sa_casscf.kernel()
