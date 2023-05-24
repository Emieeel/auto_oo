#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:37:49 2022

@author: emielkoridon
"""

from pennylane import math


def wolfe(t, grad, dp, alpha=1e-4):
    return alpha * t * math.dot(grad, dp)


class NewtonStep():
    r"""Newton step and cost.
    Supports augmented Hessian and backtracking line search (damped Newton step).

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
    """

    def __init__(self,
                 alpha=0.0001, beta=.5, mu=1e-6, rho=1.1, lmax=20, lambda_min=1e-6,
                 aug=True, verbose=1):
        r"""
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
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.rho = rho
        self.lmax = lmax
        self.lambda_min = lambda_min
        self.aug = aug
        self.verbose = verbose

    def newton_step(self, gradient, hessian):
        r"""
        Calculate the newton step:

        .. math::
            x^{(t+1)} = x^{(t)} - H^{-1}(x^{(t)})G(x^{(t)}),

        where :math:`H(x^{(t)}) = \nabla^2 f(x^{(t)})` denotes the Hessian, and
        :math:`G(x^{(t)}) = \nabla f(x^{(t)})` the gradient.

        If aug = True, check if :math:`\lambda_0 < \lambda_{\rm min}`, and define augmented Hessian:

        .. math::
            H(x) = H(x) + \nu I

        where :math:`\nu=\rho|\lambda_0| + \mu` with :math:`\lambda_0` the lowest eigenvalue of
        the Hessian.

        Args:
            gradient (1D tensor): Gradient of a function

            hessian (2D tensor): Hessian of a function

        Returns:
            dp (1D tensor): Newton step

            lowest_eigenvalue (float): Lowest eigenvalue of the Hessian
        """

        vhessian, whessian = math.linalg.eigh(hessian)

        lowest_eigenvalue = vhessian[0].item()
        if self.verbose:
            print("lowest eigval hessian =", lowest_eigenvalue)

        # augment Hessian if not positive definite
        if lowest_eigenvalue < self.lambda_min and self.aug:
            if self.verbose:
                print("augmenting hessian...")
            hessian = hessian + (
                self.mu + self.rho * abs(lowest_eigenvalue)) * math.eye(
                    hessian.shape[0], like=math.get_interface(hessian))
            vhessian, whessian = math.linalg.eigh(hessian)
            if self.verbose:
                print("Lowest eigenvalue of augmented hessian:",
                      vhessian[0].item())

        # Invert hessian
        hessian_inv = whessian @ math.diag(1/vhessian) @ math.transpose(whessian)

        dp = - (hessian_inv @ gradient)
        return dp, lowest_eigenvalue

    def backtracking(self, objective_fn, parameters, dp, gradient):
        r"""
        While positive definiteness of the Hessian guarantees a descent direction, it could be that
        the step is too large. Backtracking line search is to be advised in that case.
        :math:`\alpha \in (0, 0.5)`, :math:`\beta \in (0,1)` are the hyperparameters
        of the backtracking line search with termination condition:

        .. math::
            f(x + t\Delta x) < f(x) + \alpha t G(x)^T \Delta x

        where starting at :math:`t:=1` at each step, :math:`t:=\beta t`.
        """
        nargs = len(parameters)

        t = 1.

        energy = objective_fn(*parameters).item()

        parameters_tot = math.concatenate([math.flatten(parameter) for parameter in parameters])

        paramshapes = [math.shape(parameter) for parameter in parameters]

        newp = parameters_tot + (t * dp)
        test_energy = objective_fn(*split_list_shapes(newp, paramshapes))

        # do backtracking line search
        if test_energy > energy + wolfe(t, gradient, dp, alpha=self.alpha):
            assert (wolfe(t, gradient, dp, alpha=self.alpha) < 0)
            num = 0
            if self.verbose:
                print("test_energy:", test_energy.item(),
                      "... old energy:", energy)
                print("do backtracking line search...")
            while test_energy > energy + wolfe(
                    t, gradient, dp, alpha=self.alpha):
                t = self.beta * t
                if self.verbose:
                    print("t =", t)
                newp = parameters_tot + (t * dp)
                test_energy = objective_fn(*split_list_shapes(newp, paramshapes))
                num += 1
                if num > self.lmax:
                    t = 0.
                    test_energy = objective_fn(*parameters)
                    if self.verbose:
                        print("Warning: line search failed. Output previous parameters.")
                    break

        new_energy = test_energy.item()
        newp = parameters_tot + (t * dp)
        if self.verbose:
            print("new energy:", new_energy)
            print("old energy:", energy)
            # print("wolfe:", wolfe(t, gradient, dp, alpha=self.alpha))
        if nargs > 1:
            new_parameters = tuple(split_list_shapes(newp, paramshapes))
        else:
            new_parameters = newp
        # import pdb
        # pdb.set_trace()

        return new_parameters, new_energy

    def damped_newton_step(self, objective_fn, parameters, gradient, hessian):
        """Update trainable arguments with one step of the optimizer and
        return the corresponding objective function value prior to the step.

        Args:
            objective_fn (function): the objective function for optimization
            parameters (tuple): tuple of arguments for objective function
            gradient (1D tensor): the gradient w.r.t. composite parameters
            hessian (2D tensor): the hessian w.r.t. composite parameters

        Returns:
            tuple[array, float]: the new variable values :math:`x^{(t+1)}`
            and the lowest hessian eigenvalue prior to the step.
        """
        dp, lowest_eigenvalue = self.newton_step(gradient, hessian)
        new_parameters, new_energy = self.backtracking(
            objective_fn, parameters, dp, gradient)
        return new_parameters, lowest_eigenvalue


def split_list_shapes(parameters, paramshapes):
    """Divide list parameters into parts with given shapes."""
    # if not sum(paramshapes) == len(l):
    #     raise ValueError('sum of paramshapes has to be equal to length of list!')
    chunks = []
    num = 0
    for shape in paramshapes:
        shapesize = math.prod(shape)
        chunks.append(parameters[num:num+shapesize].reshape(shape))
        num += shapesize
    return chunks
