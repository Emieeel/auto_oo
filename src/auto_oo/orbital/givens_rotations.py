#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:04:26 2022

@author: emielkoridon

Pytorch rework
"""

import functools
import torch
import numpy as np

import cirq
import openfermion

from auto_oo.moldata.moltools import restricted_to_unrestricted


def givens_matrix(s, c, left_idx, right_idx, size):
    '''Define givens rotation matrix'''
    givmat = torch.eye(size)
    givmat[left_idx,   left_idx] = c
    givmat[right_idx, right_idx] = c
    givmat[left_idx,  right_idx] = -s
    givmat[right_idx,  left_idx] = s
    return givmat

def givens_rotation(theta, left_idx, right_idx, size):
    '''Define one givens rotation'''
    s, c = torch.sin(theta), torch.cos(theta)
    return givens_matrix(s, c, left_idx, right_idx, size)

def givens_rotation_np(theta, left_idx, right_idx, size):
    '''Define one givens rotation in numpy'''
    s, c = np.sin(theta), np.cos(theta)
    C = np.zeros((size, size))
    S = np.zeros((size, size))
    C[ left_idx,  left_idx] = 1.
    C[right_idx, right_idx] = 1.
    S[ left_idx, right_idx] = 1.
    S[right_idx, left_idx] = -1.
    return (c - 1) * C + s * S + np.eye(size)



def givens_full_transform(mo_coeff_inp, params, cas_idx=None, only_cas=False, redunt_params=None):
    '''s
    Return new mo_coeff transformed by givens rotations with input parameters
    between corresponding left_indices and right_indices.
    '''
    size = mo_coeff_inp.size(dim=0)
    
    left_indices, right_indices = np.tril_indices(size,-1)
    
    if cas_idx is None:
        if len(params) != size*(size-1)//2:
            print('wrong size:', size)
            print('len(params:)', len(params))
            raise ValueError('full givens parametrization needs size*(size-1)/2 params')
        if redunt_params is not None:
            print("Redundant parameters not used. Full givens rotations.")
        # left_indices, right_indices = zip(*((l_idx, r_idx)
        #                                     for l_idx in range(size)
        #                                     for r_idx in range(size-1,l_idx,-1)))
        
        givens_rots = [mo_coeff_inp] + [givens_rotation(theta, li, ri, size)
                        for theta, li, ri in zip(params, left_indices, right_indices)]
    elif only_cas:
        act_idx = cas_idx[1]
        na = len(act_idx)
        newsize = int((na*(na-1))/2)
        if len(params) != newsize:
            print('wrong newsize:', newsize)
            print('len(params:)', len(params))
            raise ValueError(
                'CASCI index givens parametrization needs (na*(na-1))/2 params')
        givens_rots = [mo_coeff_inp]
        num_1, num_2 = (0,0)
        for l_idx, r_idx in zip(left_indices,right_indices):
            if l_idx in act_idx and r_idx in act_idx:
                givens_rots += [givens_rotation(params[num_1], l_idx, r_idx, size)]
                num_1+=1
            else:
                givens_rots += [givens_rotation(redunt_params[num_2], l_idx, r_idx, size)]
                num_2+=1
    else:
        fr_idx, act_idx, virt_idx = cas_idx
        nf, na, nv = (len(fr_idx),len(act_idx),len(virt_idx))
        newsize = int(nf*na + nf*nv + na*nv)
        if len(params) != newsize:
            print('wrong newsize:', newsize)
            print('len(params:)', len(params))
            raise ValueError(
                'cas index givens parametrization needs nf*na + nf*nv + na*nv params')
        elif redunt_params is None:
            redunt_params = np.zeros(int(size*(size-1)//2 - newsize))
        givens_rots = [mo_coeff_inp]
        num_1, num_2 = (0,0)
        for l_idx in range(size):
            for r_idx in range(size-1,l_idx,-1):
                if not ((l_idx in fr_idx and r_idx in fr_idx) or (
                        l_idx in virt_idx and r_idx in virt_idx) or (
                            l_idx in act_idx and r_idx in act_idx)):
                    givens_rots += [givens_rotation(params[num_1], l_idx, r_idx, size)]
                    num_1+=1
                else:
                    givens_rots += [givens_rotation(redunt_params[num_2], l_idx, r_idx, size)]
                    num_2+=1
    mo_coeff = functools.reduce(torch.matmul,givens_rots)
    return mo_coeff

def givens_decomposition(A, cas_idx=None, verbose=0):
    '''Decompose special orthogonal matrix A into givens rotations in numpy'''
    size = A.shape[0]

    if verbose:
        print('input:\n', A)

    # Check whether A is special orthogonal
    assert(np.allclose( np.linalg.det(A), 1. ))
    assert(np.allclose( A @ A.T, np.eye(size)))
    assert(np.allclose( A.T @ A, np.eye(size)))
    
    # Initialize active space rotations if needed
    if cas_idx is not None:
        fr_idx, act_idx, virt_idx = cas_idx
    
    # Perform QR decomposition and save givens rotations
    givens_rots = []
    angles = []
    matQ = np.eye(size)
    matR = np.copy(A)
    given_indices = []
    l_indices, r_indices = np.tril_indices(size,-1)
    for row, col in zip(l_indices,r_indices):
        if cas_idx is not None:
            if (row in fr_idx and col in fr_idx) or (
                    row in virt_idx and col in virt_idx):
                if verbose:
                    print(f'skipping index ({row}, {col}')
                continue
        x1 = matR[col, col]
        x2 = matR[row, col]
        r = np.sqrt(x1**2 + x2**2)
        c = x1/r
        s = x2/r
        
        if s > 0:
            angle = np.arccos(c)
        elif c > 0:
            angle = 2*np.pi + np.arcsin(s)
        elif c < 0 and s < 0:
            angle = np.pi + np.arccos(-c)
        
        if verbose:
            print('row, col:', row, col)
            print('i, j:', row, col)
            print('x1, x2:', x1, x2)
            print('s, c:', s, c)
        

        givens_mat = givens_rotation_np(angle, col, row, size)
        given_indices.append((col,row))
        givens_rots.append(givens_mat.T)
        angles.append(-angle)
        matR = givens_mat @ matR
        matQ = matQ @ givens_mat.T
        if verbose:
            print("--- Next step ---")
            print("Rotation matrix for col {}, row {}; x{}: {}, x{}: {}"\
                  .format(col, row, col, x1, row, x2))
            print(givens_mat)
            print("New R")
            print(matR)
            print("New Q")
            print(matQ)
                    
    if cas_idx is None:
        assert(np.allclose(matR, np.eye(size)))
    
    return angles, given_indices

def bogoliubov_atob_cas(givens_a, givens_b, active_indices):
    '''
    Compute bogliubov transformation of G^dag_a G_b on state level.
    WARNING: input is torch.Tensor, output is numpy ndarray!
    '''
    givens_atob = givens_b @ givens_a.T
    givens_atob_R_as = givens_atob[np.ix_(*[active_indices]*2)]
    givens_atob_U_as = restricted_to_unrestricted(givens_atob_R_as).detach().numpy()
    bogoliubov_atob = cirq.Circuit(
        openfermion.bogoliubov_transform(
            cirq.LineQubit.range(2*len(active_indices)),givens_atob_U_as)
    ).unitary()
    # use |0000> to set a gauge phase reference for the Bogoliubov unitary
    return bogoliubov_atob / bogoliubov_atob[0,0]


if __name__ == '__main__':
    # Test QR decomposition:
    from scipy.stats import ortho_group
    m = ortho_group.rvs(4)
    if np.isclose(np.linalg.det(m),-1):
        m[:,-1] = -m[:,-1]
    angles, given_indices = givens_decomposition(m,verbose=1)
    