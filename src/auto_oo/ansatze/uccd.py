#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:13:30 2022

@author: emielkoridon

Contains the UCCD template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import BasisState


class UCCD(Operation):
    r"""Implements the Unitary Coupled-Cluster Doubles (UCCD) ansatz.
    
    Adapted from UCCSD class of PennyLane, removing the single excitations.
    See https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/templates/subroutines/uccsd.py

    Args:
        weights (tensor_like): Size ``len(d_wires)`` tensor containing the parameters
        :math:`\theta_{pqrs}` entering the Z rotation in
            :func:`~.FermionicDoubleExcitation`. These parameters are the coupled-cluster
            amplitudes that need to be optimized for each single and double excitation generated
            with the :func:`~.excitations` function.
        wires (Iterable): wires that the template acts on
        d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
            specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation
            :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r
            \hat{c}_s \vert \mathrm{HF} \rangle`. The entries ``s`` and ``r`` are wires
            representing two occupied orbitals where the two electrons are annihilated
            while the entries ``q`` and ``p`` correspond to the wires representing two unoccupied
            orbitals where the electrons are created. Wires in-between represent the occupied
            and unoccupied orbitals in the intervals ``[s, r]`` and ``[q, p]``, respectively.
        init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
            HF state. ``init_state`` is used to initialize the wires.
    """

    num_wires = AnyWires
    grad_method = None

    def __init__(
        self, weights, wires, d_wires=None, init_state=None, do_queue=True, id=None
    ):

        if not d_wires:
            raise ValueError(
                f"d_wires lists can not be empty; got pphh={d_wires}"
            )

        for d_wires_ in d_wires:
            if len(d_wires_) != 2:
                raise ValueError(
                    f"expected entries of d_wires to be of size 2; got {d_wires_} of length {len(d_wires_)}"
                )

        shape = qml.math.shape(weights)
        if shape != (len(d_wires),):
            raise ValueError(
                f"Weights tensor must be of shape {(len(d_wires),)}; got {shape}."
            )

        init_state = qml.math.toarray(init_state)

        if init_state.dtype != np.dtype("int"):
            raise ValueError(f"Elements of 'init_state' must be integers; got {init_state.dtype}")

        self._hyperparameters = {"init_state": init_state, "d_wires": d_wires}

        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1


    @staticmethod
    def compute_decomposition(
        weights, wires, d_wires, init_state
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.UCCSD.decomposition`.

        Args:
            weights (tensor_like): Size ``(len(s_wires) + len(d_wires),)`` tensor containing the parameters
                entering the Z rotation in :func:`~.FermionicSingleExcitation` and :func:`~.FermionicDoubleExcitation`.
            wires (Any or Iterable[Any]): wires that the operator acts on
            s_wires (Sequence[Sequence]): Sequence of lists containing the wires ``[r,...,p]``
                resulting from the single excitation.
            d_wires (Sequence[Sequence[Sequence]]): Sequence of lists, each containing two lists that
                specify the indices ``[s, ...,r]`` and ``[q,..., p]`` defining the double excitation.
            init_state (array[int]): Length ``len(wires)`` occupation-number vector representing the
                HF state. ``init_state`` is used to initialize the wires.

        Returns:
            list[.Operator]: decomposition of the operator
        """
        op_list = []

        op_list.append(BasisState(init_state, wires=wires))

        for i, (w1, w2) in enumerate(d_wires):
            op_list.append(
                qml.FermionicDoubleExcitation(weights[i], wires1=w1, wires2=w2)
            )

        return op_list


def generalized_pair_doubles(wires):
    r"""Return pair coupled-cluster double excitations

    .. math::
        \hat{T_2} = \sum_{pq} t_{p_\alpha p_\beta}^{q_\alpha, q_\beta}
               \hat{c}^{\dagger}_{q_\alpha} \hat{c}^{\dagger}_{q_\beta} \hat{c}_{p_\beta} \hat{c}_{p_\alpha}

    """
    pair_gen_doubles_wires = [
        [
            wires[r : r + 2],
            wires[p : p + 2],
        ]  # wires for [wires[r], wires[r+1], wires[p], wires[p+1]] terms
        for r in range(0, len(wires) - 1, 2)
        for p in range(0, len(wires) - 1, 2)
        if p != r  # remove redundant terms
    ]
    return pair_gen_doubles_wires