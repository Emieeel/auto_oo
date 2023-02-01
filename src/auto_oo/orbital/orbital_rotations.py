#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:17:52 2022

@author: emielkoridon
"""

import numpy as np
import scipy
import torch

    
def vec_to_mat(params):
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
    size = int(np.sqrt(8 * len(params) + 1) + 1)//2
    kappa = torch.zeros((size,size))
    tril_indices = torch.tril_indices(row=size,col=size, offset=-1)
    kappa[tril_indices[0],tril_indices[1]] = params
    kappa[tril_indices[1],tril_indices[0]] = - params
    return kappa


def kappa_matr(params, size,
               freeze_redundant=True, redunt_params=None, params_idx=None, redunt_idx=None):
    """
    Generate anti-symmetric kappa matrix. This matrix can be exponentiated to
    get an orthogonal mo_coeff rotation matrix.
    """
    assert(size*(size-1)//2 == len(params_idx) + len(redunt_idx))
    if freeze_redundant:
        params_tot = torch.zeros(len(params_idx) + len(redunt_idx))
        if redunt_params is None:
            redunt_params = torch.zeros(len(redunt_idx))
        params_tot[params_idx] = params
        params_tot[redunt_idx] = redunt_params
    else:
        params_tot = params
    return vec_to_mat(params_tot)

def kappa_to_mo(kappa):
    r""" Return :math:`e^{-\kappa}` for anti-symmetric matrix :math:`\kappa`"""
    return torch.linalg.matrix_exp(-kappa)

def orbital_full_transform(params, mol, no_oao=False):
    """
    Generate the AO-MO coefficients, given orbital parameters
    
    Args:
        params: Vector of OO parameters, to be mapped to a matrix and exponentiated
        mol: Instance of Moldata class, with pre-defined redundant indices and active space
        no_oao (bool, default=False): 
            Return just the OAO - MO coefficients without applying them to AO-OAO matrix
    """
    size = mol.nao
    if no_oao:
        inp_mo = torch.eye(size)
    else:
        inp_mo = torch.from_numpy(mol.oao_coeff)
    kappa = kappa_matr(params, size, 
                       freeze_redundant=mol.freeze_redundant,
                       redunt_params=mol.redunt_params,
                       params_idx=mol.params_idx,
                       redunt_idx=mol.redunt_idx)
    mo_transform = kappa_to_mo(kappa)
    return inp_mo @ mo_transform

def kappa_to_params(kappa):
    """Return 1D tensor of parameters given anti-symmetric matrix `kappa`"""
    size = kappa.size(dim=0)
    tril_indices = torch.tril_indices(row=size,col=size, offset=-1)
    return kappa[tril_indices[0],tril_indices[1]]

def orbital_decomposition(orbital_rotation_matrix):
    r"""Convert OAO-MO coefficients into orbital rotation by taking
    the natural logarithm:
     
    .. math::
        \mathbf{\kappa} = - \log \mathbf{C}^{\rm OAO-MO}
    """
    kappa = - scipy.linalg.logm(orbital_rotation_matrix)
    params = kappa_to_params(torch.from_numpy(kappa))
    return params.detach().numpy()
    
