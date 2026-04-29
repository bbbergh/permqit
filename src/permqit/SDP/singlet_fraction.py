import dataclasses
import math

import numpy as np
from typing import cast, Sequence, Collection
import picos as pc
from ..algebra import EndSnOrbitBasis, EndSnIrrepBasis, MatrixTensorProductBasis, EndSnOrbitBasisSubset, BasisSubset
from ..utilities.sdp_result import SDPResult
from .variables.choi import choi_variable
from ..representation.isomorphism import (
    EndSnAlgebraIsomorphism,
    EndSnBlockDiagonalization,
    tensor_product_block_diagonalization,
    tensor_product_block_diagonalization_basis, tensor_product_inverse_block_diagonalization, )

from .solve_SDP import solve_sdp
from ..utilities.timing import MaybeExpensiveComputation


def maximal_singlet_fraction(
    state: np.ndarray,
    bases: Sequence[EndSnOrbitBasis|EndSnOrbitBasisSubset],
    system_indices_to_optimize: Collection[int],
    *,
    solver: str = "SCS",
    sdp_accuracy: float = 1e-6,
    verbose: bool = False,
    return_optimizers: bool = True,
    timing_analysis: bool = True,
):
    """
    Given a state ρ on End^{S_μ_1}(A_1^μ_1) ⊗ ... ⊗ End^{S_μ_k}(A_k^μ_k), calculate the maximal singlet fraction of that state
    as an SDP, which is optimizing Tr(Φ (id ⊗ D)(ρ)) over all channels: D: End^{S_μ_{k_1}}}(A_{k_1}^μ_{k_1}) ⊗ ... ⊗ End^{S_μ_{k_s}}}(A_{k_s}^μ_{k_s}) -> End^{S_μ_{k_1}}}(A_{k_1}^μ_{k_1}) ⊗ ... ⊗ End^{S_μ_{k_s}}}(A_{k_s}^μ_{k_s})
    where k_1, ..., k_s are the indices in 'system_indices_to_optimize', and Φ is a maximylly entangled state along the split between the systems selected by 'system_indices_to_optimize' and the remaining systems.

    This turns out to be equivalent to optimizing Tr(ρ Γ) over all Γ which are choi matrices of channels 'systems_to_optimize' -> 'complement_of_systems_to_optimize'
    It is easy to see that these choi matrices can be restricted to be S_{μ_1) x ... x S_{μ_k} invariant, which is exploited
    by this function. It is actually this optimization which is performed by this function, and the optimizer returned is
    this choi matrix Γ as coefficients in the TensorProductBasis(EndSnOrbitBasis(μ_1, d_A_1), ..., EndSnOrbitBasis(μ_k, d_A_k)).
    Parameters
    ----------
    state: Coefficient vector of the state ρ in the TensorProductBasis(EndSnOrbitBasis(μ_1, d_A_1), ..., EndSnOrbitBasis(μ_k, d_A_k)) (or subsets thereof)
    bases: [EndSnOrbitBasis(μ_1, d_A_1), ..., EndSnOrbitBasis(μ_k, d_A_k)] (or subsets thereof)
    system_indices_to_optimize: the indices k_1, ..., k_s which define the bipartite splitting of all the systems.
    solver
    sdp_accuracy
    verbose
    return_optimizers
    timing_analysis
    Returns
    -------
    Coefficients of the optimizing choi matrix Γ in the TensorProductBasis specified by 'basis', i.e. if
    the state ρ only lives on a subset, so does Γ and only the subset coefficients are returned.
    """
    problem = pc.Problem()

    isos = [EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(basis.n, basis.d)) for basis in bases]

    assert len(state) == math.prod(b.size() for b in bases)

    choi_input_system_indices = system_indices_to_optimize
    choi_output_system_size = math.prod(bases[idx].size() for idx in (set(range(len(bases))) - set(system_indices_to_optimize)))

    # Main part is here: impose Choi constraints in the SDP
    C_RS_blocks = choi_variable(problem, bases, choi_input_system_indices)

    # Transform state to block basis using forward isomorphism
    # For this, first embed into non-subset basis if basis is a subset
    for idx, basis in enumerate(bases):
        if isinstance(basis, BasisSubset):
            state = basis.index_mapping.apply_to_coefficient_vector(state, axis=idx)
    blocks = tensor_product_block_diagonalization(state, isos)
    blocked_basis = tensor_product_block_diagonalization_basis(isos)

    objective_function = 0
    for basis_block, block, variable_block in zip(blocked_basis.bases, blocks, C_RS_blocks):
        if basis_block.size() == 0:
            continue
        # f_λ weighting for matrix inner product: <C,M>_full = Σ_λ f_λ * Tr(C^λ† @ M^λ)
        weight = math.prod(
            cast(EndSnIrrepBasis, b).partition.count_standard_tableaux()
            for b in cast(MatrixTensorProductBasis, basis_block).bases
        )
        c_M_mult = pc.Constant("rho_AB", block * weight)
        objective_function += (variable_block | c_M_mult).real  # type: ignore[attr-defined]

    # Since we are using the un-normalized chois/maximally entangled states, we need to divide by
    # the relevant dimension, which here is the output system dimension of the 'recovery channel' we constructed
    problem.set_objective(
        "max", objective_function / choi_output_system_size
    )  
     
    

    with MaybeExpensiveComputation("Solving SDP for Maximal Singlet Fraction"):
        F_phi, solver_time = solve_sdp(
            problem,
            solver=solver,
            sdp_accuracy=sdp_accuracy,
            verbose=verbose,
        )

    # Reconstruct the obit basis value of the optimizer
    c_RS_orbit = tensor_product_inverse_block_diagonalization([np.asarray(c.value) for c in C_RS_blocks], isos).ravel()

    optim = (c_RS_orbit, ) if return_optimizers else None
    time_val = solver_time if timing_analysis else None
    return SDPResult(F_phi, time=time_val, optimizers=optim)


def maximal_recovery_coefficient(channel_choi_coefficients: np.ndarray,
                                     bases: Sequence[EndSnOrbitBasis|EndSnOrbitBasisSubset],
                                     input_system_indices: Collection[int] = (0,), **solver_kwargs) -> SDPResult:
    """
    Computes the maximal fidelity of recovery of the given channel, which is specified through the coefficients of its Choi matrix.
    The coefficients are given in the TensorProductBasis(*basses), and the bases at input_system_indices are the input part of the Choi matrix,
    everything else is the output.
    Parameters
    ----------
    channel_choi_coefficients
    bases
    input_system_indices
    solver_kwargs

    Returns
    -------

    """
    return maximal_singlet_fraction(channel_choi_coefficients, bases, set(range(len(bases))) - set(input_system_indices), **solver_kwargs)


def maximal_preparation_coefficient(channel_choi_coefficients: np.ndarray,
                                     bases: Sequence[EndSnOrbitBasis|EndSnOrbitBasisSubset],
                                     input_system_indices: Collection[int] = (0,), **solver_kwargs) -> SDPResult:
    """
    Computes the maximal fidelity of preparation of the given channel, which is specified through the coefficients of its Choi matrix.
    The coefficients are given in the TensorProductBasis(*basses), and the bases at input_system_indices are the input part of the Choi matrix,
    everything else is the output.

    Parameters
    ----------
    channel_choi_coefficients
    bases
    input_system_indices
    solver_kwargs

    Returns
    -------
    """
    # The preparation fidelity is the same as the recovery fidelity, module a renormalization
    recovery = maximal_recovery_coefficient(channel_choi_coefficients, bases, input_system_indices, **solver_kwargs)
    input_system_size = math.prod(bases[idx].size() for idx in input_system_indices)
    output_system_size = math.prod(bases[idx].size() for idx in set(range(len(bases))) - set(input_system_indices))
    return dataclasses.replace(recovery, value=recovery.value *input_system_size/output_system_size)


