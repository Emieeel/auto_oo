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

import auto_oo.opt.newton_raphson
from auto_oo.orbital import orbital_rotations
from auto_oo.vqe.uccd import UCCD

torch.set_default_tensor_type(torch.DoubleTensor)

def uccdstate(theta, ncas, nelecas, vqe_singles=False):
    """
    Extract state-vector of a Unitary Coupled-Cluster Doubles (UCCD) state. 
    Optionally, add the singles to give UCCSD.
    
    Args:
        theta (torch.Tensor): 1D tensor of parameters with length ndoubles (+ nsingles if UCCSD)
        
        ncas (int): number of active orbitals
        
        nelecas (int): number of active electrons
        
        vqe_singles (bool, default=False): Choose to add the singles in the quantum ansatz (UCCSD)
    """
    nq = 2*ncas
    singles, doubles = qml.qchem.excitations(nelecas, nq)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    hfstate = qml.qchem.hf_state(nelecas, nq)
    wires = range(nq)
    
    dev = qml.device('default.qubit', wires=nq)
    @qml.qnode(dev, interface='torch', diff_method="backprop")
    def uccd_state():
        if vqe_singles:
            qml.UCCSD(theta, wires, s_wires=s_wires, d_wires=d_wires, init_state=hfstate)
        else:
            UCCD(theta, wires, d_wires, init_state=hfstate)
        return qml.state()
    return uccd_state()


def uccs_mo(params_singles, mol):
    """Extract MO-MO transformation of active-active rotations"""
    size = (mol.nao)*(mol.nao-1)//2
    mask = np.ones(size,bool)
    mask[mol.params_act_act_idx] = False
    redunt_singles_idx = np.arange(size)[mask]
    singles_kappa = orbital_rotations.kappa_matr(
        params_singles, mol.nao, params_idx=mol.params_act_act_idx, redunt_idx=redunt_singles_idx)
    singles_coeff = orbital_rotations.kappa_to_mo(singles_kappa)
    return singles_coeff
    
def rdms_from_params(params_vqe, mol, restricted=True):
    """Extract one- and two-particle density matrices from vqe parameters"""
    state = uccdstate(
        params_vqe[mol.noosingles:],
        mol.ncas, mol.nelecas, mol.vqe_singles)
    return mol.get_rdms_from_state(state, restricted)    

def oo_uccd_cost(params_oo, params_vqe, mol, restricted = True):
    r"""
    Total cost function.
    
    Computes molecular coefficients based on the non-redundant orbital parameters
    params_oo, taking :math:`\mathbf{C}^{\rm OAO-MO} = e^{-\mathbf{\kappa}}`.
    Then transforms Hamiltonian to corresponding active space using the Moldata class.
    
    Computes one- and two-particle reduced density matrices with params_vqe.
    
    Finally computes the energy by contracting the active space molecular coefficients with the
    one- and two-particle reduced density matrices.
    """
    oo_coeff = orbital_rotations.orbital_full_transform(params_oo, mol)
    if not mol.vqe_singles:
        # Treat singles classically
        oo_coeff = oo_coeff @ uccs_mo(params_vqe[:mol.noosingles],mol)
    one_rdm, two_rdm = rdms_from_params(params_vqe, mol, restricted)
    c0, c1, c2 = mol.get_energy_from_rdms(one_rdm, two_rdm, oo_coeff, restricted)
    return c0 + c1 + c2


class Orb_vqe():
    """
    OO-VQE class for a specific geometry. Can perform orbital optimized
    vqe with autograd in PyTorch.

    Args:
        mol: Moldata class containing the molecular info en methods to create
        the Hamiltonian

        init_params (tuple of torch.Tensors, default=None): Initial parameters,
        tuple of params_oo and params_vqe

    """
    def __init__(self, mol, init_params=None, stepsize=0.0314,
                 max_iterations=300, conv_tol=1e-6, opt=None,
                 print_per_it=50, verbose=0, start_from_hf=False,
                 do_oo=True):
        self.mol = mol
        self.stepsize = stepsize

        self.max_iterations = max_iterations
        self.conv_tol = conv_tol
        self.print_per_it = print_per_it
        self.verbose = verbose

        if init_params is None:
            self.init_oo, self.init_vqe = mol.init_full(start_from_hf)
        else:
            self.init_oo, self.init_vqe = init_params

        self.params_oo, self.params_vqe = self.init_oo.detach().clone(), self.init_vqe.detach().clone()
        self.params_oo.requires_grad = do_oo
        self.params_vqe.requires_grad = True
        self.cost_fn = functools.partial(oo_uccd_cost,
                                         mol=self.mol)
        self.opt_trajectory = []
        if opt is None:
            if self.max_iterations == 1:
                self.opt = auto_oo.opt.newton_raphson.NewtonOptimizer(verbose=self.verbose)
            else:
                self.opt = torch.optim.Adam([self.params_oo, self.params_vqe],lr=stepsize)
        else:
            self.opt = opt

    def kernel(self):
        self.e_min = 1e99

        if self.max_iterations == 0:
            energy = self.cost_fn(self.params_oo, self.params_vqe).item()
            self.opt_trajectory.append(energy)
            self.e_min = energy
            self.n_min = 0
            self.n_fin = 0
            self.params_oo_min = self.params_oo.detach().clone()
            self.params_vqe_min = self.params_vqe.detach().clone()
            self.params_min = (self.params_oo_min, self.params_vqe_min)
            if self.verbose:
                print("Computed energy from params. (max_iterations = 0)")

        elif type(self.opt) == auto_oo.opt.newton_raphson.NewtonOptimizer:
            if self.verbose:
                print(f'Doing max {self.max_iterations} steps with Newton-Raphson!')
                print('conv_tol:', self.conv_tol)
                print('starting e:', self.cost_fn(self.init_oo, self.init_vqe).item())
            self.params = (self.params_oo,self.params_vqe)
            self.hess_eig = []
            for n in range(self.max_iterations):

                curr_params, curr_energy, hess_eig = self.opt.step_and_cost(self.cost_fn,
                                                                            *self.params)

                curr_params_oo, curr_params_vqe = self.params = curr_params
                self.hess_eig.append(hess_eig)
                
                if n%self.print_per_it == 0 and self.verbose:
                    print('iter, e =', n+1, curr_energy)
                    print('\n')
                self.opt_trajectory.append(curr_energy)
                
                if curr_energy < self.e_min:
                    self.e_min = curr_energy
                    self.n_min = n
                    self.params_oo_min = curr_params_oo.detach().clone()
                    self.params_vqe_min = curr_params_vqe.detach().clone()
                    self.params_min = (self.params_oo_min, self.params_vqe_min)

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
                print('starting e:', self.cost_fn(self.init_oo, self.init_vqe).item())
            def closure():
                self.opt.zero_grad()
                loss = self.cost_fn(self.params_oo, self.params_vqe)
                loss.backward()
                return loss
            
            for n in range(self.max_iterations):

                self.opt.step(closure)
                
                curr_params_oo, curr_params_vqe = self.opt.param_groups[0]['params']
                curr_energy = self.cost_fn(curr_params_oo, curr_params_vqe).item()

                if n%self.print_per_it == 0 and self.verbose:
                    print('iter, e =', n+1, curr_energy)
                self.opt_trajectory.append(curr_energy)

                if curr_energy < self.e_min:
                    self.e_min = curr_energy
                    self.n_min = n
                    self.params_oo_min = curr_params_oo.detach().clone()
                    self.params_vqe_min = curr_params_vqe.detach().clone()
                    self.params_min = (self.params_oo_min, self.params_vqe_min)
                    
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