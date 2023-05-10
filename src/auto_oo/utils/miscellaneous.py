#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:28:17 2023

@author: emielkoridon
"""

import torch


def scipy_csc_to_torch(scipy_csc, dtype=torch.complex128):
    """ Convert a scipy sparse CSC matrix to pytorch sparse CSC tensor."""
    ccol_indices = scipy_csc.indptr
    row_indices = scipy_csc.indices
    values = scipy_csc.data
    size = scipy_csc.shape
    return torch.sparse_csc_tensor(
        torch.tensor(ccol_indices, dtype=torch.int64),
        torch.tensor(row_indices, dtype=torch.int64),
        torch.tensor(values), dtype=dtype, size=size)


def get_formal_geo(alpha, phi):
    variables = [1.498047, 1.066797, 0.987109, 118.359375] + [alpha, phi]
    geom = """
                    N
                    C 1 {0}
                    H 2 {1}  1 {3}
                    H 2 {1}  1 {3} 3 180
                    H 1 {2}  2 {4} 3 {5}
                    """.format(
        *variables
    )
    return geom
