#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:37:49 2022

@author: emielkoridon
"""

import torch
from functorch import jacfwd, jacrev, hessian

def wolfe(t, grad, dp, alpha=1e-4):
    return alpha * t * torch.dot(grad,dp)

class NewtonOptimizer():
    r"""Newton optimizer for pytorch. Supports augmented Hessian and backtracking line search.

    Steepest descent in the direction determined by Hessian norm:

    .. math::
        x^{(t+1)} = x^{(t)} - H^{-1}(x^{(t)})G(x^{(t)}),

    where :math:`H(x^{(t)}) = \nabla^2 f(x^{(t)})` denotes the Hessian, and 
    :math:`G(x^{(t)}) = \nabla f(x^{(t)})` the gradient.
    
    If aug = True, check if :math:`\lambda_0 < \lambda_{\rm min}`, and define augmented Hessian:
    
    .. math::
        H(x) = H(x) + \nu I
    
    where :math:`\nu=\rho|\lambda_0| + \mu` with :math:`\lambda_0` the lowest eigenvalue of
    the Hessian.
    
    While positive definiteness of the Hessian guarantees a descent direction, it could be that
    the step is too large. Backtracking line search is implemented in that case.
    :math:`\alpha \in (0, 0.5)`, :math:`\beta \in (0,1)` are the hyperparameters
    of the backtracking line search with termination condition:
        
    .. math::
        f(x + t\Delta x) < f(x) + \alpha t G(x)^T \Delta x
    
    where starting at :math:`t:=1` at each step, :math:`t:=\beta t`.

    Args:
        alpha (float): the user-defined hyperparameter :math:`\alpha`
        
        beta (float): the user-defined hyperparameter :math:`\beta`
        
        mu (float): the user-defined hyperparameter :math:`\mu`
        
        rho (float): the user-defined hyperparameter :math:`\rho`
        
        lambda_min (float): the user-defined hyperparameter 
            :math:`\lambda_{\rm min}`
        
        lmax (int): maximal line search steps
        
        aug (bool): if True, use augmented Hessian (recommended)
        
        back (bool): if True, use backtracking line search (recommended)

    """
    def __init__(self,
                 alpha=0.0001, beta=.5, mu=1e-6, rho=2., lmax=20, lambda_min=1e-6,
                 aug=True, back=True, verbose=1):
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.rho = rho
        self.lmax = lmax
        self.lambda_min = lambda_min
        self.aug = aug
        self.back = back
        self.verbose = verbose
    
    def step_and_cost(self, objective_fn, *params):
        """Update trainable arguments with one step of the optimizer and return the corresponding
        objective function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            *params : argument for objective function

        Returns:
            tuple[array, float]: the new variable values :math:`x^{(t+1)}` and the objective
            function output prior to the step.
        """

        # Compute initial energy, gradient and Hessian
        energy = objective_fn(*params).item()
        
        if self.verbose:
            print("old energy:", energy)

        nargs = len(params)
        paramshapes = [params[i].shape[-1] for i in range(nargs)]

        params_tot = torch.cat(params)

        if nargs > 1:
            grad_temp = jacfwd(objective_fn, argnums=tuple(range(nargs)))
            grad_comp = grad_temp(*params)
            grad = torch.cat(grad_comp).detach()


            hess_temp = hessian(objective_fn,argnums=tuple(range(nargs)))
            hess_comp = hess_temp(*params)
            hess = torch.cat(tuple(
                (torch.cat(tuple(
                    (hess_comp[i][j].detach()) for j in range(nargs)), dim=-1))
                    for i in range(nargs)), dim=0)

        else:
            grad_temp = jacfwd(objective_fn)
            grad = grad_temp(*params)
            grad_comp = None
            hess_temp = hessian(objective_fn)
            hess = hess_temp(*params)
            hess_comp = None

        vhess, whess = torch.linalg.eigh(hess)

        low_eig = vhess[0].item()
        if self.verbose:
            print("lowest eigval hess =",low_eig)
        
        # augment Hessian if not positive definite
        if low_eig < self.lambda_min and self.aug:
            if self.verbose:
                print("augmenting hess...")
            hess = hess + (
                self.mu + self.rho * abs(low_eig))*torch.eye(
                    hess.shape[0])
            vhess, whess = torch.linalg.eigh(hess)
            if self.verbose:
                print("Lowest eigenvalue of augmented hess:", vhess[0].item())
        
        # Invert hessian
        hess_inv = whess @ torch.diag(1/vhess) @ torch.t(whess)
        
        dp = - (hess_inv @ grad)

        t = 1.
        newp = params_tot + (t * dp)
        test_energy = objective_fn(*split_list_shapes(newp, paramshapes))

        # do backtracking line search
        if test_energy > energy + wolfe(t, grad, dp, alpha=self.alpha) and self.back:
            num = 0
            if self.verbose:
                print("test_energy:", test_energy.item(),"... old energy:", energy)
                print("do backtracking line search...")
            while test_energy > energy + wolfe(t, grad, dp, alpha=self.alpha):
                t = self.beta * t
                if self.verbose:
                    print("t =", t)
                newp = params_tot + (t * dp)
                test_energy = objective_fn(*split_list_shapes(newp, paramshapes))
                num += 1
                if num > self.lmax:
                    t = 0.
                    test_energy = objective_fn(*params)
                    if self.verbose:
                        print("Warning: line search failed. Output previous parameters.")
                    break

        new_energy = test_energy.item()
        newp = params_tot + (t * dp)
        if nargs > 1:
            new_params = tuple(split_list_shapes(newp, paramshapes))
        else:
            new_params = newp
        del grad_comp, grad_temp, grad, hess_comp, hess_temp, hess, hess_inv

        return new_params, new_energy, low_eig
    
def split_list_shapes(l, paramshapes):
    """Divide list l into parts with given shapes."""
    if not sum(paramshapes) == len(l):
        raise ValueError('sum of paramshapes has to be equal to length of list!')
    chunks = []
    num = 0
    for shape in paramshapes:
        chunks.append(l[num:num+shape])
        num+=shape
    return chunks