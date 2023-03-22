#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:06:01 2023

@author: emielkoridon
"""

from functools import partial

import numpy as np

import torch

from auto_oo.oo_energy import integrals
from auto_oo.moldata_pyscf import Moldata_pyscf
from auto_oo.newton_raphson import NewtonStep

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


def vector_to_skew_symmetric(vector):
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
        vector (torch.Tensor): 1d tensor
    """
    size = int(np.sqrt(8 * len(vector) + 1) + 1)//2
    matrix = torch.zeros((size, size))
    tril_indices = torch.tril_indices(row=size, col=size, offset=-1)
    matrix[tril_indices[0], tril_indices[1]] = vector
    matrix[tril_indices[1], tril_indices[0]] = - vector
    return matrix


def skew_symmetric_to_vector(kappa_matrix):
    """Return 1D tensor of parameters given anti-symmetric matrix `kappa`"""
    size = kappa_matrix.size(dim=0)
    tril_indices = torch.tril_indices(row=size, col=size, offset=-1)
    return kappa_matrix[tril_indices[0], tril_indices[1]]


def non_redundant_indices(occ_idx, act_idx, virt_idx, freeze_active):
    """ Calculate non-redundant indices for indexing kappa vectors
    for a given active space"""
    no, na, nv = len(occ_idx), len(act_idx), len(virt_idx)
    nao = no + na + nv
    rotation_sizes = [no*na, na*nv, no*nv]
    if not freeze_active:
        rotation_sizes.append(na * (na - 1)//2)
    n_kappa = sum(rotation_sizes)
    params_idx = np.array([], dtype=int)
    num = 0
    for l_idx, r_idx in zip(*np.tril_indices(nao, -1)):
        if not(
            ((l_idx in act_idx and r_idx in act_idx
              ) and freeze_active) or (
                l_idx in occ_idx and r_idx in occ_idx) or (
                l_idx in virt_idx and r_idx in virt_idx)):  # or (
            # l_idx in virt_idx and r_idx in occ_idx)):
            params_idx = np.append(params_idx, [num])
        num += 1
    assert(n_kappa == len(params_idx))
    return params_idx


class OO_energy():
    """ Orbital Optimized energy class for extracting energies for any given set of
        RDMs. Can compute analytical orbital gradients and hessians."""

    def __init__(self, mol: Moldata_pyscf, ncas, nelecas,
                 oao_mo_coeff=None, freeze_active=False):
        """
        Args:
            mol: Moldata_pyscf class containing molecular information like
                geometry, AO basis, MO basis and 1e- and 2e-integrals

            ncas: Number of active orbitals

            nelecas: Number of active electrons

            oao_mo_coeff (default None): Reference OAO-MO
                coefficients (ndarray)

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
            # print("Initialized with canonical HF MOs")
            mol.run_rhf()
            self.oao_mo_coeff = torch.from_numpy(
                mo_ao_to_mo_oao(mol.hf.mo_coeff, mol.overlap))
        else:
            if type(oao_mo_coeff) == np.ndarray:
                self.oao_mo_coeff = torch.from_numpy(oao_mo_coeff)
            else:
                self.oao_mo_coeff = oao_mo_coeff.detach().clone()

        # Set active space parameters
        self.ncas = ncas
        self.nelecas = nelecas

        self.occ_idx, self.act_idx, self.virt_idx = mol.get_active_space_idx(
            ncas, nelecas)

        # Calculate non-redundant orbital rotations
        self.params_idx = non_redundant_indices(self.occ_idx, self.act_idx, self.virt_idx,
                                                freeze_active)
        self.n_kappa = len(self.params_idx)

    @property
    def mo_coeff(self):
        """ Set mo_coeff automatically when changing oao_mo_coeff attribute."""
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

    def energy_from_kappa(self, kappa, one_rdm, two_rdm):
        """Get total energy after transforming the MOs with kappa"""
        mo_coeff = self.mo_coeff @ self.kappa_to_mo_coeff(kappa)
        return self.energy_from_mo_coeff(mo_coeff, one_rdm, two_rdm)

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
                          dim=-1)  # p^ i^ i q
            - torch.sum(int2e_mo[:, self.occ_idx, self.occ_idx, :],
                        dim=1))  # p^ i^ q i
        return int1e_mo + g_tilde

    def fock_active(self, int2e_mo, one_rdm):
        g_tilde = (
            int2e_mo[:, :, :, self.act_idx][:, :, self.act_idx, :]
            - .5 * torch.permute(int2e_mo[:, :, self.act_idx, :][:, self.act_idx, :, :],
                                 (0, 3, 2, 1)))
        return torch.einsum('wx, pqwx', one_rdm, g_tilde)

    def fock_generalized(self, int1e_mo, int2e_mo, one_rdm, two_rdm):
        fock_C = self.fock_core(int1e_mo, int2e_mo)
        fock_A = self.fock_active(int2e_mo, one_rdm)
        fock_general = torch.zeros(int1e_mo.shape)
        fock_general[self.occ_idx, :] = 2 * torch.t(
            fock_C[:, self.occ_idx] + fock_A[:, self.occ_idx])
        fock_general[self.act_idx, :] = torch.einsum(
            'qw,vw->vq', fock_C[:, self.act_idx], one_rdm) + torch.einsum(
            'vwxy,qwxy->vq',
            two_rdm,
            int2e_mo[:, :, :, self.act_idx][:, :, self.act_idx, :][:, self.act_idx, :, :])
        return fock_general

    def analytic_gradient(self, one_rdm, two_rdm, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            mo_coeff = mo_coeff

        int1e_mo = int1e_transform(self.int1e_ao, mo_coeff)
        int2e_mo = int2e_transform(self.int2e_ao, mo_coeff)

        fock_general = self.fock_generalized(
            int1e_mo, int2e_mo, one_rdm, two_rdm)
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
        fock_general = self.fock_generalized(
            int1e_mo, int2e_mo, one_rdm, two_rdm)
        fock_general_symm = fock_general + torch.t(fock_general)

        hess0 = 2 * torch.einsum('pr, qs->pqrs', one_full, int1e_mo)
        hess1 = - torch.einsum('pr, qs->pqrs',
                               fock_general_symm, torch.eye(self.nao))
        hess2 = 2 * y_matrix

        hess_permuted0 = hess0 + hess1 + hess2
        hess_permuted1 = torch.permute(hess_permuted0, (0, 1, 3, 2))
        hess_permuted2 = torch.permute(hess_permuted0, (1, 0, 2, 3))
        hess_permuted3 = torch.permute(hess_permuted0, (1, 0, 3, 2))

        return hess_permuted0 - hess_permuted1 - hess_permuted2 + hess_permuted3

    def full_rdms(self, one_rdm, two_rdm):
        """Generate RDMs in the full space.
        TODO: Remove this function and incorporate only active space RDMS"""
        one_full = torch.zeros((self.nao, self.nao))
        two_full = torch.zeros((self.nao, self.nao, self.nao, self.nao))

        one_full[self.occ_idx, self.occ_idx] = 2 * \
            torch.ones(len(self.occ_idx))
        one_full[np.ix_(self.act_idx, self.act_idx)] = one_rdm

        two_full[np.ix_(*[self.occ_idx]*4)] = 4 * torch.einsum(
            'ij,kl->ijkl', *[torch.eye(len(self.occ_idx))]*2) - 2 * torch.einsum(
            'il,jk->ijkl', *[torch.eye(len(self.occ_idx))]*2)
        two_full[np.ix_(self.occ_idx, self.occ_idx,
                        self.act_idx, self.act_idx)] = 2 * torch.einsum(
            'wx,ij->ijwx', one_rdm, torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.act_idx, self.act_idx,
                        self.occ_idx, self.occ_idx)] = 2 * torch.einsum(
            'wx,ij->wxij', one_rdm, torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.occ_idx, self.act_idx,
                        self.act_idx, self.occ_idx)] = -torch.einsum(
            'wx,ij->iwxj', one_rdm, torch.eye(len(self.occ_idx)))
        two_full[np.ix_(self.act_idx, self.occ_idx,
                        self.occ_idx, self.act_idx)] = -torch.einsum(
            'wx,ij->xjiw', one_rdm, torch.eye(len(self.occ_idx)))
        two_full[np.ix_(*[self.act_idx]*4)] = two_rdm
        return one_full, two_full

    def y_matrix(self, int2e_mo, two_full):
        y0 = torch.einsum('pmrn, qmns->pqrs', two_full, int2e_mo)
        y1 = torch.einsum('pmnr, qmns->pqrs', two_full, int2e_mo)
        y2 = torch.einsum('prmn, qsmn->pqrs', two_full, int2e_mo)
        return y0 + y1 + y2

    def kappa_vector_to_matrix(self, kappa):
        """Generate a skew-symmetric matrix from orbital rotation parameters"""
        kappa_total_vector = torch.zeros(self.nao * (self.nao - 1)//2)
        kappa_total_vector[self.params_idx] = kappa
        return vector_to_skew_symmetric(kappa_total_vector)

    def kappa_matrix_to_vector(self, kappa_matrix):
        """Generate orbital rotation parameters from a skew-symmetric matrix"""
        kappa_total_vector = skew_symmetric_to_vector(kappa_matrix)
        return kappa_total_vector[self.params_idx]

    def kappa_to_mo_coeff(self, kappa):
        """ Generate a skew_symmetric matrix from orbital rotation parameters
        and exponentiate it to get an orbital transformation"""
        kappa_matrix = self.kappa_vector_to_matrix(kappa)
        return torch.linalg.matrix_exp(-kappa_matrix)

    def get_transformed_mo(self, mo_coeff, kappa):
        """ Transform a general matrix mo_coeff with orbital rotation parameters
        kappa"""
        mo_coeff_transformed = mo_coeff @ self.kappa_to_mo_coeff(kappa)
        return mo_coeff_transformed

    def full_hessian_to_matrix(self, full_hess):
        """Convert the full Hessian (nao,nao,nao,nao) torch.Tensor to a matrix with only
        non-redundant indices."""
        tril_indices = torch.tril_indices(
            row=self.nao, col=self.nao, offset=-1)
        partial_hess = full_hess[tril_indices[0], tril_indices[1], :, :]
        reduced_hess = partial_hess[:, tril_indices[0], tril_indices[1]]
        nonredundant_hess = reduced_hess[self.params_idx,
                                         :][:, self.params_idx]
        return nonredundant_hess

    def orbital_optimization(self, one_rdm, two_rdm,
                             conv_tol=1e-8, max_iterations=100, verbose=0, **kwargs):
        r"""
        Optimize the orbitals using the damped Newton method. The MO coefficients
        are automatically transformed throughout the procedure as the attribute of
        the OO_energy class.

        Args:
            one_rdm (2D torch.Tensor): One-particle reduced density matrix

            two_rdm (4D torch.Tensor): Two-particle reduced density matrix

        Returns:
            energy_l (list): Energy trajectory of the procedure
        """

        objective_fn = partial(self.energy_from_kappa,
                               one_rdm=one_rdm, two_rdm=two_rdm)
        opt = NewtonStep(verbose=verbose, **kwargs)

        energy_l = []

        if verbose:
            energy = self.energy_from_mo_coeff(
                self.mo_coeff, one_rdm, two_rdm).item()
            print(f'Starting energy: {energy:.12f}')

        for n in range(max_iterations):
            kappa = torch.zeros(self.n_kappa)
            gradient = self.kappa_matrix_to_vector(
                self.analytic_gradient(one_rdm, two_rdm))
            hessian = self.full_hessian_to_matrix(
                self.analytic_hessian(one_rdm, two_rdm))

            kappa, lowest_eigenvalue = opt.damped_newton_step(objective_fn, (kappa,),
                                                              gradient, hessian)
            self.oao_mo_coeff = self.oao_mo_coeff @ self.kappa_to_mo_coeff(
                kappa)

            energy = self.energy_from_mo_coeff(
                self.mo_coeff, one_rdm, two_rdm).item()
            energy_l.append(energy)

            if verbose is not None:
                print(f'iter = {n:03}, energy = {energy:.12f}')

            if n > 1:
                if (abs(energy_l[-1] - energy_l[-2]) < conv_tol):
                    if verbose:
                        print("Orbital optimization finished.")
                        print("E_fin =", energy_l[-1])
                    break
        return energy_l


if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt

    # torch.set_num_threads(12)

    def get_formal_geo(alpha, phi):
        variables = [1.498047, 1.066797, 0.987109, 118.359375] + [alpha, phi]
        geom = """
                        N
                        C 1 {0}
                        H 2 {1}  1 {3}
                        H 2 {1}  1 {3} 3 180
                        H 1 {2}  2 {4} 3 {5}
                        """.format(*variables)
        return geom

    geometry = get_formal_geo(140, 80)
    basis = 'cc-pvdz'
    mol = Moldata_pyscf(geometry, basis)

    ncas = 3
    nelecas = 4

    one_rdm = torch.Tensor(
        [[1.9947e+00,  2.9425e-02, -1.4976e-17],
         [2.9425e-02,  1.7815e+00,  1.0134e-16],
         [-1.4976e-17,  1.0134e-16,  2.2377e-01]])

    two_rdm = torch.Tensor(
        [[[[1.9914e+00,  2.6885e-02, -1.5717e-17],
          [2.6885e-02,  3.5558e+00,  2.0367e-16],
          [-1.5717e-17,  2.0367e-16,  4.3698e-01]],

         [[2.6885e-02,  2.0326e-02,  5.1605e-18],
          [-1.7729e+00,  2.5402e-03,  8.1501e-18],
          [-1.0036e-16, -7.8452e-18,  5.8850e-02]],

         [[-1.5717e-17,  5.1605e-18, -5.8406e-02],
          [-1.0036e-16, -3.6777e-17,  7.7252e-02],
          [-2.1849e-01, -2.9425e-02,  7.5669e-18]]],


         [[[2.6885e-02, -1.7729e+00, -1.0036e-16],
          [2.0326e-02,  2.5402e-03, -7.8452e-18],
          [5.1605e-18,  8.1501e-18,  5.8850e-02]],

         [[3.5558e+00,  2.5402e-03, -3.6777e-17],
          [2.5402e-03,  1.7782e+00,  1.5596e-16],
          [-3.6777e-17,  1.5596e-16,  1.0561e-02]],

         [[2.0367e-16,  8.1501e-18,  7.7252e-02],
          [-7.8452e-18,  1.5596e-16, -6.1816e-01],
          [-2.9425e-02, -5.2803e-03, -5.5598e-17]]],


         [[[-1.5717e-17, -1.0036e-16, -2.1849e-01],
          [5.1605e-18, -3.6777e-17, -2.9425e-02],
          [-5.8406e-02,  7.7252e-02,  7.5669e-18]],

         [[2.0367e-16, -7.8452e-18, -2.9425e-02],
          [8.1501e-18,  1.5596e-16, -5.2803e-03],
          [7.7252e-02, -6.1816e-01, -5.5598e-17]],

         [[4.3698e-01,  5.8850e-02,  7.5669e-18],
          [5.8850e-02,  1.0561e-02, -5.5598e-17],
          [7.5669e-18, -5.5598e-17,  2.2377e-01]]]])

    oo_energy = OO_energy(mol, ncas, nelecas)

    # mo_coeff = torch.from_numpy(mol.oao_coeff)
    # from scipy.stats import ortho_group
    # mo_transform = torch.from_numpy(ortho_group.rvs(mol.nao))
    # oao_mo_coeff = mo_transform
    # oo_energy.oao_mo_coeff = oao_mo_coeff
    # print("check if property works:",
    #       torch.allclose(oo_energy.mo_coeff, torch.from_numpy(mol.oao_coeff) @ oao_mo_coeff)     )

    plt.title('one rdm')
    plt.imshow(one_rdm)
    plt.colorbar()
    plt.show()
    plt.title('two rdm')
    plt.imshow(two_rdm.reshape(ncas**2, ncas**2))
    plt.colorbar()
    plt.show()

    energy_l = oo_energy.orbital_optimization(one_rdm, two_rdm,
                                              conv_tol=1e-9, max_iterations=150,
                                              mu=1e-4, rho=1.05, verbose=0)
    mol.run_casscf(ncas, nelecas)

    print(f'Final OO energy         = {energy_l[-1]:.8f}')
    print(f'Reference CASSCF energy = {mol.casscf.e_tot:.8f}')

    plt.title('Energy over orbital optimization')
    plt.plot(energy_l)
    plt.show()
