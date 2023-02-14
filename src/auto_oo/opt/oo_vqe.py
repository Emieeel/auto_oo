#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:32:21 2022

@author: emielkoridon
"""

import functools

import pennylane as qml
import numpy as np

import torch

from auto_oo.main.oo_pqc import OO_pqc_cost


class Orb_vqe():
    """
    OO-VQE class for a specific geometry. Can perform orbital optimized
    vqe with autograd in PyTorch.

    Args:
        oo_pqc: Orbital-optimized Parameterized Quantum Circuit class
            containing the cost_function and methods to create gradients
            and hessians

        init_params (torch.Tensor, default=None): Initial parameters,
        tuple of theta

    """
    def __init__(self, oo_pqc : OO_pqc_cost, init_params=None, stepsize=0.0314,
                 max_iterations=300, conv_tol=1e-6, opt=None,
                 print_per_it=50, verbose=0, start_from_hf=False,
                 do_oo=True):
        self.oo_pqc = oo_pqc
        self.stepsize = stepsize

        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.print_per_it = print_per_it
        self.verbose = verbose

        if init_params is None:
            self.init_vqe = oo_pqc.pqc.init_zeros()
        else:
            self.init_vqe = init_params

        self.theta = self.init_vqe.detach().clone()
        self.params_oo.requires_grad = do_oo
        self.theta.requires_grad = True
        self.opt_trajectory = []
        if opt is None:
            if self.max_iterations == 1:
                self.opt = \
                    auto_oo.newton_raphson.newton_raphson.NewtonOptimizer(
                        verbose=self.verbose)
            else:
                self.opt = torch.optim.Adam([self.theta],lr=stepsize)
        else:
            self.opt = opt

    def kernel(self):
        self.e_min = 1e99

        if self.max_iterations == 0:
            energy = self.oo_pqc.energy_from_parameters(self.theta).item()
            self.opt_trajectory.append(energy)
            self.e_min = energy
            self.n_min = 0
            self.n_fin = 0
            self.params_oo_min = self.params_oo.detach().clone()
            self.theta_min = self.theta.detach().clone()
            self.params_min = (self.params_oo_min, self.theta_min)
            if self.verbose:
                print("Computed energy from params. (max_iterations = 0)")

        elif type(self.opt) == auto_oo.opt.newton_raphson.NewtonOptimizer:
            if self.verbose:
                print(f'Doing max {self.max_iterations} steps with Newton-Raphson!')
                print('conv_tol:', self.conv_tol)
                print('starting e:', self.oo_pqc.energy_from_parameters(self.init_oo, self.init_vqe).item())
            self.params = (self.params_oo,self.theta)
            self.hess_eig = []
            for n in range(self.max_iterations):

                curr_params, curr_energy, hess_eig = self.opt.step_and_cost(self.oo_pqc.energy_from_parameters,
                                                                            *self.params)

                curr_params_oo, curr_theta = self.params = curr_params
                self.hess_eig.append(hess_eig)
                
                if n%self.print_per_it == 0 and self.verbose:
                    print('iter, e =', n+1, curr_energy)
                    print('\n')
                self.opt_trajectory.append(curr_energy)
                
                if curr_energy < self.e_min:
                    self.e_min = curr_energy
                    self.n_min = n
                    self.params_oo_min = curr_params_oo.detach().clone()
                    self.theta_min = curr_theta.detach().clone()
                    self.params_min = (self.params_oo_min, self.theta_min)

                # Single convergence condition for NR:
                if n > 1:
                    if (np.abs(self.opt_trajectory[-1] - self.opt_trajectory[-2]) < self.conv_tol):
                        if self.verbose:
                            print("VQE optimization finished.")
                            print("E_fin =", self.opt_trajectory[-1])
                        break
            self.n_fin = n + 1

        else:
            if self.verbose:
                print('conv_tol:', self.conv_tol)
                print('starting e:', self.oo_pqc.energy_from_parameters(self.init_oo, self.init_vqe).item())
            def closure():
                self.opt.zero_grad()
                loss = self.oo_pqc.energy_from_parameters(self.params_oo, self.theta)
                loss.backward()
                return loss
            
            for n in range(self.max_iterations):

                self.opt.step(closure)
                
                curr_params_oo, curr_theta = self.opt.param_groups[0]['params']
                curr_energy = self.oo_pqc.energy_from_parameters(curr_params_oo, curr_theta).item()

                if n%self.print_per_it == 0 and self.verbose:
                    print('iter, e =', n+1, curr_energy)
                self.opt_trajectory.append(curr_energy)

                if curr_energy < self.e_min:
                    self.e_min = curr_energy
                    self.n_min = n
                    self.params_oo_min = curr_params_oo.detach().clone()
                    self.theta_min = curr_theta.detach().clone()
                    self.params_min = (self.params_oo_min, self.theta_min)
                    
                # I implemented a "double" convergence condition:
                if n > 2:
                    if (np.abs(self.opt_trajectory[-1] - self.opt_trajectory[-2])\
                        < self.conv_tol) and (
                            np.abs(self.opt_trajectory[-2] - self.opt_trajectory[-3])\
                                < self.conv_tol):
                            if self.verbose:
                                print("VQE converged")
                                print("E_fin =", self.opt_trajectory[-1])
                            break
            self.n_fin = n + 1
            
if __name__ == '__main__':
    import auto_oo
    # geom ='''H 0 0 0; Li 0 0 1'''
    # geom = '''O -0.01 0.005 -0.14; H 0 .49 .51; H 0.02 -0.5 0.505'''
    alpha, phi = 130, 85
    variables = [1.498047, 1.066797, 0.987109, 118.359375] + [alpha, phi]
    geom = """
                    N
                    C 1 {0}
                    H 2 {1}  1 {3}
                    H 2 {1}  1 {3} 3 180
                    H 1 {2}  2 {4} 3 {5}
                    """.format(*variables)

    ncas = 4
    nelecas = 4
    basis = 'sto-3g'    

    mol = auto_oo.Moldata(geom, basis,
                          freeze_redundant=True, vqe_singles=True)
    mol.define_CAS(ncas,nelecas)

    mol.run_rhf()
    mol.run_casscf()

    init_oo_tot = torch.Tensor(
        mol.mo_to_param(mol.casscf.mo_coeff))

    init_oo = init_oo_tot[mol.params_idx]
    mol.redunt_params = init_oo_tot[mol.redunt_idx]

    init_params = (init_oo, torch.zeros(mol.nvqeparam))


    print('HF energy:', mol.hf.e_tot)
    print('CASSCF energy:', mol.casscf.e_tot)
    print('\n\n')

    stepsize = 0.01
    max_iterations=3000
    conv_tol = .5e-9
    ppi = 1
    do_oo = False

    vqe = Orb_vqe(mol, init_params=init_params, verbose=1, print_per_it=ppi,
                  conv_tol=conv_tol, stepsize=stepsize,
                  max_iterations=max_iterations, do_oo=do_oo)
    # initp = vqe.params
    # cost = vqe.cost_fn(initp)
    vqe.kernel()
    print(f'Final vqe energy: {vqe.e_min}.')
    print(f'CASSCF energy:    {mol.casscf.e_tot}')