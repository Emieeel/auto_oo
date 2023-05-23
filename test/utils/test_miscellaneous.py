#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:45:23 2023

@author: emielkoridon
"""

import pytest

import scipy
import jax.numpy as jnp
import torch
import auto_oo


def test_scipy_csc_to_jax():
    scipy_csc = scipy.sparse.random(10, 10, density=0.5, format='csc')
    assert jnp.allclose(jnp.array(scipy_csc.A), auto_oo.scipy_csc_to_jax(scipy_csc).todense())


def test_scipy_csc_to_torch():
    scipy_csc = scipy.sparse.random(10, 10, density=0.5, format='csc')
    assert torch.allclose(torch.from_numpy(scipy_csc.A),
                          auto_oo.scipy_csc_to_torch(scipy_csc, dtype=torch.float64).to_dense())
