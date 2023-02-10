#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:13:30 2022

@author: emielkoridon

Contains the kUpCCD template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires

class kUpCCD(Operation):
    r"""Implements the k-Unitary Pair Coupled-Cluster Generalized Doubles (k-UpCCGSD) ansatz.
    
    Adapted from kUpCCGSD class of PennyLane, removing the single excitations.
    See https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/templates/subroutines/kupccgsd.py


    Args:
        weights (tensor_like): Tensor containing the parameters :math:`\theta_{pqrs}`
            entering the Z rotation in :func:`~.FermionicDoubleExcitation`.
            These parameters are the coupled-cluster amplitudes that need to be optimized for each
            double excitation term.
        wires (Iterable): wires that the template acts on
        k (int): Number of times UpCCGD unitary is repeated.
        init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``init_state`` is used to initialize the wires.

    """

    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires, k=1, init_state=None, do_queue=True, id=None):

        if len(wires) < 4:
            raise ValueError(f"Requires at least four wires; got {len(wires)} wires.")
        if len(wires) % 2:
            raise ValueError(f"Requires even number of wires; got {len(wires)} wires.")

        if k < 1:
            raise ValueError(f"Requires k to be at least 1; got {k}.")

        d_wires = generalized_pair_doubles(list(wires))

        shape = qml.math.shape(weights)
        if shape != (
            k,
            len(d_wires),
        ):
            raise ValueError(
                f"Weights tensor must be of shape {(k, len(d_wires),)}; got {shape}."
            )

        init_state = qml.math.toarray(init_state)
        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {
            "init_state": init_state,
            "d_wires": d_wires,
            "k": k,
        }
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(
        weights, wires, d_wires, k, init_state
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.kUpCCGSD.decomposition`.

        Args:
            weights (tensor_like): tensor containing the parameters entering the Z rotation
            wires (Any or Iterable[Any]): wires that the operator acts on
            k (int): number of times UpCCGSD unitary is repeated
            s_wires (Iterable[Any]): single excitation wires
            d_wires (Iterable[Any]): double excitation wires
            init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
                HF state.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        op_list.append(qml.BasisEmbedding(init_state, wires=wires))

        for layer in range(k):
            for i, (w1, w2) in enumerate(d_wires):
                op_list.append(
                    qml.FermionicDoubleExcitation(
                        weights[layer][i], wires1=w1, wires2=w2
                    )
                )

        return op_list

    @staticmethod
    def shape(k, n_wires):
        r"""Returns the shape of the weight tensor required for this template.
        Args:
            k (int): Number of layers
            n_wires (int): Number of qubits
        Returns:
            tuple[int]: shape
        """

        if n_wires < 4:
            raise ValueError(
                f"This template requires the number of qubits to be greater than four; got 'n_wires' = {n_wires}"
            )

        if n_wires % 2:
            raise ValueError(
                f"This template requires an even number of qubits; got 'n_wires' = {n_wires}"
            )

        d_wires = generalized_pair_doubles(range(n_wires))

        return k, len(d_wires)
