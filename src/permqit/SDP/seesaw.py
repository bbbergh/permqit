import dataclasses
from typing import Collection, Sequence

import numpy as np

from permqit.representation.isomorphism import EndSnAlgebraIsomorphism, EndSnBlockDiagonalization

from ..algebra import EndSnOrbitBasis, EndSnOrbitBasisSubset, MatrixTensorProductBasis
from ..utilities.random import random_channel_with_permutation_invariant_output
from ..utilities.sdp_result import SDPResult
from .seesaw_utils import get_coefficient_AlicePOV, get_coefficient_BobPOV
from . import singlet_fraction

DEFAULT_SEESAW_ACCURACY = 1e-6
DEFAULT_SEESAW_REPETITIONS = 1
DEFAULT_SEESAW_ITERATIONS = 50
DEFAULT_ISOMETRY = False
DEFAULT_SEED = 42
DEFAULT_SYMMETRY = None

def compute_tensor_product_fidelity_without_symmetry_seesaw(
        n: int, d_R: int, N: np.ndarray, d_A: int, d_B: int, *, timing_analysis: bool = True
) -> SDPResult:
    basisAB = EndSnOrbitBasis(n, d_A * d_B)

    # Construct the full dense tensor product channel and then feed it into symmetric seesaw with n = 1.
    channel = basisAB.linear_combination(basisAB.coefficients_for_tensor_product(N)).todense()
    return compute_tensor_product_fidelity_seesaw(1, d_R, channel, d_A**n, d_B**n, timing_analysis=timing_analysis)


def compute_tensor_product_fidelity_seesaw(
    n: int, d_R: int, N: np.ndarray, d_A: int, d_B: int, *, timing_analysis: bool = True
) -> SDPResult:
    """Run the seesaw optimization loop to obtain a lower bound on channel fidelity.
    :param n: number of copies of the channel to use
    :param d_R: dimension of the maximally entangled state
    :param N: coefficients of the single-copy Choi operator of the channel in MatrixStandardBasis(d_Axd_B), i.e. a flattened d_Axd_B dimensional array.

    :return: Best fidelity (float) found across seesaw iterations.
    """

    repetitions = DEFAULT_SEESAW_REPETITIONS
    accuracy = DEFAULT_SEESAW_ACCURACY
    iterations = DEFAULT_SEESAW_ITERATIONS

    basisR = EndSnOrbitBasis(1, d_R)

    basisA = EndSnOrbitBasis(n, d_A)
    basisB = EndSnOrbitBasis(n, d_B)
    basisAB = EndSnOrbitBasis(n, d_A * d_B)

    isoA = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d_A))
    isoB = EndSnAlgebraIsomorphism(EndSnBlockDiagonalization(n, d_B))

    total_time = 0.0

    c_N = basisAB.coefficients_for_tensor_product(N)
    fidelities = []
    for rep in range(repetitions):
        c_E = random_channel_with_permutation_invariant_output(d_R, isoA, seed=DEFAULT_SEED)

        # Symmetric encoders disabled for now. I don't actually think it's a good idea, since this is really not very random, and almost always encodes in a maximally mixed state for large n.
        #  if DEFAULT_SYMMETRY is None else random_perm_inv_encoder_symmetric(ctx, symmetry=DEFAULT_SYMMETRY, seed=DEFAULT_SEED)

        F_E = 0.0
        for i in range(iterations):
            # BOB POV: mid = A
            # This returns in [basisR, basisB]
            c_M = get_coefficient_BobPOV(c_E, c_N, basisA, basisB, basisAB, basisR)

            decoder_result = optimize_decoder(c_M, [basisR, basisB], (0,))
            F_D = decoder_result.get_value()

            # Get the decoder in ordered form (B -> R), by default it is returned in the same order as we put in (which we specified as [R, B])
            c_D = decoder_result.assert_get_first_optimizer().reshape([basisR.size(), basisB.size()]).transpose(1, 0)

            # ALICE POV: mid = B
            c_M = get_coefficient_AlicePOV(c_N, c_D, basisA, basisB, basisAB, basisR)
            encoder_result = optimize_encoder(c_M, [basisA, basisR])
            F_E = encoder_result.get_value()
            c_E = encoder_result.assert_get_first_optimizer().reshape([basisA.size(), basisR.size()]).transpose(1, 0)
            print(f"Iteration {i}: Decoder fidelity = {F_D:.6f}, Encoder fidelity = {F_E:.6f}")
            print("Decoder time: ", decoder_result.get_time(), " Encoder time: ", encoder_result.get_time())
            total_time += (decoder_result.get_time() or 0.0) + (encoder_result.get_time() or 0.0)
            if abs(F_E - F_D) < accuracy:
                break
        fidelities.append(F_E)

    max_fidelity = max(fidelities)

    optim = None  # (max_encoder,max_decoder) if ctx.params.return_optimizers else None
    time_val = total_time if timing_analysis else None
    return SDPResult(max_fidelity, time=time_val, optimizers=optim)


def optimize_decoder(
    c_M: np.ndarray,
    bases: Sequence[EndSnOrbitBasis | EndSnOrbitBasisSubset],
    input_system_indices: Collection[int] = (0,),
) -> SDPResult:
    """
    Maximize decoder fidelity using block basis representation.

    :param c_M: 1D coefficients of choi matrix of current concatenated map M: (R->B) from previous step, in the TensorProductBasis(*bases).
    :param input_system_indices: The indices of the systems in the given list bases which correspond to the input system part of the choi matrix.
    :return: SDPResult(
        value: the fidelity achieved with the optimal decoder,
        optimizers: (the coefficients of the optimal decoder in the TensorProductBasis(*bases),)
        Note that the input systems of the decoder are the complement of the input systems of the provided channel (as specified in input_system_indices),
        in particular this will no longer be returned in [input, output] order.
    )
    """

    result = singlet_fraction.maximal_recovery_coefficient(c_M, bases, input_system_indices)
    tensor_product_basis = MatrixTensorProductBasis(tuple(bases))

    # The returned optimizer from the singlet fraction is the choi matrix of the adjoint of the actual decoder.
    # To get the decoder we take the transpose of the choi matrix, as the choi matrix of a channel's adjoint is the transpose of the original channel's choi matrix.
    return dataclasses.replace(
        result, optimizers=(tensor_product_basis.transpose(result.assert_get_first_optimizer().ravel()))
    )


def optimize_encoder(
    c_M: np.ndarray,
    bases: Sequence[EndSnOrbitBasis | EndSnOrbitBasisSubset],
    input_system_indices: Collection[int] = (0,),
):
    """
    Optimize encoder using block basis representation.

    :param c_M: 1D coefficients of the choi matrix of current concatenated map (A->R) from previous step,
              in TensorProductBasis(*bases).
    :param input_system_indices: The indices of the systems in the given list bases which correspond to the input system part of the choi matrix.

    Returns: (
        E_flat: The coefficients of the optimal encoder in the TensorProductBasis(*bases).
        F_enc: The fidelity achieved with the optimal encoder
    )
    """

    tensor_product_basis = MatrixTensorProductBasis(tuple(bases))

    # Since the optimal encoder fidelity is the maximal preparation coefficient of the adjoint map, we first have to transpose the choi matrix (to get the choi matrix of the adjoint map)
    c_M_adj = tensor_product_basis.transpose(c_M.ravel())

    return singlet_fraction.maximal_preparation_coefficient(c_M_adj, bases, input_system_indices)
