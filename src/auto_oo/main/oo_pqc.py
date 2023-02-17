#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:10:18 2023

@author: emielkoridon
"""

import pennylane as qml
import torch
from functorch import jacfwd
from functorch import hessian
# from torch.autograd.functional import jacobian, hessian

from auto_oo.oo_energy.oo_energy import OO_energy
from auto_oo.ansatz.pqc import Parameterized_circuit
from auto_oo.moldata_pyscf.moldata_pyscf import Moldata_pyscf
from auto_oo.newton_raphson.newton_raphson import NewtonStep


class OO_pqc_cost(OO_energy):
    def __init__(self, pqc : Parameterized_circuit, mol : Moldata_pyscf,
                 ncas, nelecas, oao_mo_coeff=None, freeze_active=False):
        """
        Orbital Optimized energy class for extracting energies by computing RDMs
        from a quantum state generated by a quantum circuit. Can compute composite
        gradients and hessians with respect to both orbital and quantum circuit 
        parameters.

        Args:
            pqc: Parameterized_circuit class containing the ansatz and the method
                to extract the state and the RDMs
            mol: Moldata_pyscf class containing molecular information like
                geometry, AO basis, MO basis and 1e- and 2e-integrals
            ncas: Number of active orbitals
            nelecas: Number of active electrons
            oao_mo_coeff (default None): Reference OAO-MO coefficients (ndarray)
            freeze_active (default: False):
                Freeze active-active orbital rotations
        """
        super().__init__(mol, ncas, nelecas,
                     oao_mo_coeff=oao_mo_coeff, freeze_active=freeze_active)
        self.pqc = pqc
        
    def energy_from_parameters(self, theta, kappa=None):
        r"""
        Get total energy given quantum circuit parameters and orbital transformation parameters.
        Total energy is computed as:

        .. math::
            E = E_{\rm nuc} + E_{\rm core} +
            \sum_{pq}\tilde{h}_{pq} \gamma_{pq} + 
            \sum_{pqrs} g_{pqrs} \Gamma_{pqrs}

        where :math:`E_{core}` is the mean-field energy of the core (doubly-occupied) orbitals,
        :math:`\tilde{h}_{pq}` is contains the active one-body terms plus the mean-field
        interaction of core-active orbitals and :math:`g_{pqrs}` are the active integrals
        in chemist ordering.
        """
        if kappa is None:
            mo_coeff = self.mo_coeff
        else:
            mo_coeff = self.get_transformed_mo(self.mo_coeff, kappa)
        state = self.pqc.ansatz_state(theta)
        one_rdm, two_rdm = self.pqc.get_rdms_from_state(state)
        return self.energy_from_mo_coeff(mo_coeff, one_rdm, two_rdm)
    
    def circuit_gradient(self, theta):
        """Calculate the electronic gradient w.r.t. circuit parameters"""
        # return jacobian(self.energy_from_parameters, theta)
        return jacfwd(self.energy_from_parameters)(theta)
    
    def orbital_gradient(self, theta):
        """Generate analytically the flattened electronic gradient w.r.t. orbital rotation
        parameters for a given set of circuit parameters"""
        state = self.pqc.ansatz_state(theta)
        one_rdm, two_rdm = self.pqc.get_rdms_from_state(state)
        return self.kappa_matrix_to_vector(
            self.analytic_gradient(one_rdm, two_rdm))
    
    def circuit_circuit_hessian(self, theta):
        """Calculate the electronic hessian w.r.t circuit parameters"""
        # return hessian(self.energy_from_parameters,theta)
        return hessian(self.energy_from_parameters)(theta)

    def orbital_circuit_hessian(self, theta):
        """Generate the mixed orbital-pqc parameter hessian by automatic differentation
        of the analytic orbital gradient"""
        # return jacobian(self.orbital_gradient,theta)
        return jacfwd(self.orbital_gradient)(theta)
    
    def orbital_orbital_hessian(self, theta):
        """Generate the electronic Hessian w.r.t. orbital rotations"""
        state = self.pqc.ansatz_state(theta)
        one_rdm, two_rdm = self.pqc.get_rdms_from_state(state)
        return self.full_hessian_to_matrix(
            self.analytic_hessian(one_rdm, two_rdm))
    
    def full_gradient(self, theta):
        """Return the composite gradient of circuit parameters and orbital rotations"""
        return torch.cat((self.circuit_gradient(theta), self.orbital_gradient(theta)))
    
    def full_hessian(self, theta):
        """Return the composte hessian of circuit parameters and orbital rotations"""
        hessian_vqe_vqe = self.circuit_circuit_hessian(theta)
        hessian_vqe_oo = self.orbital_circuit_hessian(theta)
        hessian_oo_oo = self.orbital_orbital_hessian(theta)
        hessian = torch.cat((
                    torch.cat((hessian_vqe_vqe, hessian_vqe_oo.t()), dim=1),
                    torch.cat((hessian_vqe_oo, hessian_oo_oo), dim=1)), dim = 0)
        return hessian
    
    def full_optimization(self, theta_init, max_iterations=50, conv_tol=1e-10, verbose=0, **kwargs):
        opt = NewtonStep(verbose=verbose, **kwargs)
        energy_init = self.energy_from_parameters(theta_init).item()
        if verbose is not None:
            print(f'iter = 000, energy = {energy_init:.12f}')
    
        theta_l = []
        kappa_l = []
        oao_mo_coeff_l = []
        energy_l = []
        hess_eig_l = []
        
        theta = theta_init.detach().clone()
        for n in range(max_iterations):
    
            kappa = torch.zeros(self.n_kappa)
    
            gradient = self.full_gradient(theta)
            hessian = self.full_hessian(theta)
    
            new_theta_kappa, hess_eig = opt.damped_newton_step(
                self.energy_from_parameters, (theta, kappa), gradient, hessian)
            
            hess_eig_l.append(hess_eig)
            
            theta = new_theta_kappa[0]
            kappa = new_theta_kappa[1]
    
            theta_l.append(theta.detach().clone())
            kappa_l.append(kappa.detach().clone())
    
            self.oao_mo_coeff = self.oao_mo_coeff @ self.kappa_to_mo_coeff(kappa)
    
            oao_mo_coeff_l.append(self.oao_mo_coeff.detach().clone())

            energy = self.energy_from_parameters(theta).item()
            energy_l.append(energy)
            
            if verbose is not None:
                print(f'iter = {n+1:03}, energy = {energy:.12f}')
            if n > 1:
                if (abs(energy_l[-1] - energy_l[-2]) < conv_tol):
                    if verbose is not None:
                        print("optimization finished.")
                        print("E_fin =", energy_l[-1])
                    break
        
        return energy_l, theta_l, kappa_l, oao_mo_coeff_l, hess_eig_l

if __name__ == '__main__':
    from cirq import dirac_notation
    import matplotlib.pyplot as plt
    
    torch.set_num_threads(12)
    
    def get_formal_geo(alpha,phi):
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
    basis = 'sto-3g'
    mol = Moldata_pyscf(geometry, basis)

    ncas = 3
    nelecas = 4
    dev = qml.device('default.qubit', wires=2*ncas)
    pqc = Parameterized_circuit(ncas, nelecas, dev, add_singles=False)
    theta = torch.rand_like(pqc.init_zeros())
    # theta = pqc.init_zeros()
    state = pqc.ansatz_state(theta)
    one_rdm, two_rdm = pqc.get_rdms_from_state(state)

    
    oo_pqc = OO_pqc_cost(pqc, mol, ncas, nelecas)#, oao_mo_coeff = oao_mo_coeff)
    
    
    # mo_coeff = torch.from_numpy(mol.oao_coeff)
    # from scipy.stats import ortho_group
    # mo_transform = torch.from_numpy(ortho_group.rvs(mol.nao))
    # oao_mo_coeff = mo_transform
    # oao_mo_coeff = torch.eye(mol.nao)
    # oo_pqc.oao_mo_coeff = oao_mo_coeff
    # print("check if property works:",
    #       torch.allclose(oo_pqc.mo_coeff, torch.from_numpy(mol.oao_coeff) @ oao_mo_coeff)     )
    

    
    kappa = torch.zeros(oo_pqc.n_kappa)
    energy_test = oo_pqc.energy_from_parameters(theta, kappa)
    print("theta:", theta)
    print("state:", dirac_notation(state.detach().numpy()))
    print('Expectation value of Hamiltonian:', energy_test.item())
    mol.run_rhf()
    print('HF energy:', mol.hf.e_tot)
    
    plt.title('one rdm')
    plt.imshow(one_rdm)
    plt.colorbar()
    plt.show()
    plt.title('two rdm')
    plt.imshow(two_rdm.reshape(ncas**2,ncas**2))
    plt.colorbar()
    plt.show()



    import time
    t0 = time.time()
    # grad_auto = jacobian(oo_pqc.energy_from_parameters, (
    #     theta,kappa))
    grad_auto = jacfwd(oo_pqc.energy_from_parameters, argnums=(0,1))(
        theta,kappa)
    hess_auto = hessian(oo_pqc.energy_from_parameters,(theta,kappa))
    # hess_auto = hessian(oo_pqc.energy_from_parameters,
    #                     argnums=(0,1))(theta, kappa)
    print("time took to generate everything with auto-differentation:", time.time()-t0)
    
    t1 = time.time()
    print("should all be True:",
          torch.allclose(grad_auto[0], oo_pqc.circuit_gradient(theta)),
          torch.allclose(grad_auto[1], oo_pqc.orbital_gradient(theta)),
          torch.allclose(hess_auto[0][0], oo_pqc.circuit_circuit_hessian(theta)),
          torch.allclose(hess_auto[1][0], oo_pqc.orbital_circuit_hessian(theta)),
          torch.allclose(hess_auto[1][1], oo_pqc.orbital_orbital_hessian(theta)))
    print("time took to generate full hessian but orbital part analytically:",
          time.time()-t1)
    
    
    
    # orbgrad_auto_2d = oo_pqc.kappa_vector_to_matrix(orbgrad_auto[1])
    # orbgrad_exact = oo_pqc.orbital_gradient(one_rdm, two_rdm)
    
    # plt.title('automatic diff orbital gradient')
    # plt.imshow(orbgrad_auto_2d)
    # plt.colorbar()
    # plt.show()
    # plt.title('exact orbital gradient')
    # plt.imshow(orbgrad_exact)
    # plt.colorbar()
    # plt.show()
    
    # orbgrad_auto_flat = orbgrad_auto[1]
    # orbgrad_exact_flat = oo_pqc.kappa_matrix_to_vector(orbgrad_exact)
    

    # t0 = time.time()
    # orbhess_auto_comp = hessian(oo_pqc.energy_from_parameters,
    #                             argnums=(0,1))(theta, kappa)
    # # orbhess_auto_comp = torch.autograd.functional.hessian(oo_pqc.energy_from_parameters,
    # #                                                       (theta, kappa))
    # print("time took to calc hess with automatic diff:", time.time()-t0)
    # orbhess_auto_kappa_theta = orbhess_auto_comp[0][1]
    
    # t1 = time.time()
    # orbhess_exact_kappa_theta = oo_pqc.orbital_circuit_hessian(theta)
    # print("time took to calc mixed hess with exact/automatic took:", time.time()-t1)
    
    # plt.title('kappa-theta hessian automatic diff')
    # plt.imshow(orbhess_auto_kappa_theta)
    # plt.colorbar()
    # plt.show()
    # plt.title('kappa-theta hessian exact autodiff')
    # plt.imshow(orbhess_exact_kappa_theta.t())
    # plt.colorbar()
    # plt.show()
    
    # orborbhess_auto = orbhess_auto_comp[1][1]
    # t2 = time.time()
    # orborbhess_exact_full = oo_pqc.orbital_hessian(one_rdm, two_rdm)
    # orborbhess_exact = oo_pqc.full_hessian_to_matrix(orborbhess_exact_full)
    # print("time took to calc orb-orb hess with exact method took:", time.time()-t2)
    
    # plt.title('kappa-kappa hessian automatic diff')
    # plt.imshow(orborbhess_auto)
    # plt.colorbar()
    # plt.show()
    # plt.title('kappa-kappa hessian exact')
    # plt.imshow(orborbhess_exact)
    # plt.colorbar()
    # plt.show()
    
    # orborb_diff = torch.abs(orborbhess_auto - orborbhess_exact)
    # plt.title('kappa-kappa auto exact diff')
    # plt.imshow(orborb_diff)
    # plt.colorbar()
    # plt.show()
    
    
    
    
    
    
