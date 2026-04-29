"""
Depolarizing channel utilities.

The qubit depolarizing channel is:
    N_p(ρ) = (1 - p) ρ + p · I/2

Its (normalized) Choi matrix equals the isotropic state with mixing parameter (1-p):
    N_choi_normalized = isotropic_state(1 - p, d=2, fidelity=False)

The seesaw expects the UNNORMALIZED Choi (Tr_out = d_in · I_in):
    N_choi = d · isotropic_state(1 - p, d=2, fidelity=False)

Symmetry: the depolarizing channel is invariant under U⊗U* (CLDUI symmetry).
Its Choi matrix is an isotropic state, so the stronger isotropic equality
constraints hold:
    N_choi[0,0] = N_choi[3,3],   N_choi[1,1] = N_choi[2,2],   N_choi[0,3] = N_choi[3,0]

Orbit structure for the symmetric seesaw
-----------------------------------------
The n-copy orbit basis (EndSnOrbitBasis(n, d_AB=4)) has C(n+15, 15) total orbits,
but the depolarizing channel only has nonzero support on the CLDUI-compatible orbits:
C(n+5, 5) orbits (those whose count matrix has support only on the 6 CLDUI pairs).

Furthermore, the isotropic state has STRONGER symmetry than CLDUI: only 3
independent parameters (alpha, beta, gamma) instead of 6. Each orbit coefficient
decomposes as:
    c[orbit] = alpha^g0 * beta^g1 * gamma^g2
where (g0, g1, g2) are the group totals for the 3 isotropic groups:
    g0 = count((0,0)) + count((3,3))   [alpha group]
    g1 = count((1,1)) + count((2,2))   [beta group]
    g2 = count((0,3)) + count((3,0))   [gamma group]

depolarizing_orbit_bases(n) precomputes orbit indices and group totals once per n
(O(C(n+5,5)) work) so that per-p evaluations are O(C(n+5,5)) vectorized array ops
with no orbit enumeration. This yields speedups of ~380x (n=8), ~1000x (n=10),
~10000x (n=15), ~60000x (n=20) vs the naive get_coefficients_tensor_product_channel.
"""
import numpy as np
from permqit.representation.combinatorics import weak_compositions, weak_composition_to_index
from ..states.isotropic import isotropic_state

__all__ = [
    "depolarizing_choi",
    "depolarizing_orbit_bases",
    "depolarizing_channel_with_support",
]

# Group assignment for each pair (3 groups, 2 pairs each):
#   Group 0: (0,0) and (3,3)  → alpha = N_choi[0,0] = N_choi[3,3]
#   Group 1: (1,1) and (2,2)  → beta  = N_choi[1,1] = N_choi[2,2]
#   Group 2: (0,3) and (3,0)  → gamma = N_choi[0,3] = N_choi[3,0]
_PAIR_TO_GROUP = [0, 1, 1, 0, 2, 2]
# Flat indices of the nonzero entries in the 4x4 Choi matrix (row*4 + col):
#   (0,0)->0, (1,1)->5, (2,2)->10, (3,3)->15, (0,3)->3, (3,0)->12
_PAIR_FLAT_IDX = [0, 5, 10, 15, 3, 12]


def depolarizing_choi(p: float, d: int = 2) -> np.ndarray:
    """Return the unnormalized Choi matrix of the d-dimensional depolarizing channel.

    The depolarizing channel is:
        N_p(ρ) = (1 - p) ρ + p · I/d

    The UNNORMALIZED Choi matrix satisfies Tr_out(N_choi) = d · I_in and equals:
        N_choi = d · isotropic_state(1 - p, d, fidelity=False)
             = d · [(1-p) · |Φ+><Φ+| + p · I/d²]

    This is the convention expected by the seesaw optimization functions.

    Args:
        p: Depolarizing parameter, p ∈ [0, 1].
           p=0: identity channel; p=1: fully depolarizing (N(ρ) = I/d).
        d: Qudit dimension (default: 2 for qubits).

    Returns:
        Unnormalized Choi matrix, shape (d², d²), dtype complex128.
    """
    return d * isotropic_state(1.0 - p, d, fidelity=False)


def depolarizing_orbit_bases(n: int):
    """Precompute CLDUI orbit indices and isotropic group totals for n copies.

    Enumerates all C(n+5, 5) CLDUI-compatible orbits of EndSnOrbitBasis(n, d_AB=4)
    and records, for each orbit:
      - its index in the orbit basis
      - the isotropic group totals (g0, g1, g2) needed to evaluate the channel
        coefficient as alpha^g0 * beta^g1 * gamma^g2

    This is O(C(n+5, 5)) work and needs to be done only once per n. Per-p
    evaluation then reduces to vectorized power computations (no orbit enumeration).

    Args:
        n: Number of channel uses (n >= 1).

    Returns:
        (orbit_indices, group_totals):
            orbit_indices: int64 array of shape (K,) — orbit indices in EndSnOrbitBasis(n, 4).
            group_totals:  int32 array of shape (K, 3) — (g0, g1, g2) per orbit.
            K = C(n+5, 5) is the number of CLDUI-compatible orbits.
    """

    d_AB = 4
    orbit_indices = []
    group_totals = []

    for comp in weak_compositions(n, 6):
        # Build full count vector (length d_AB^2 = 16) for orbit index computation.
        # Only the 6 CLDUI-pair positions are nonzero.
        count_vec = np.zeros(d_AB * d_AB, dtype=np.int_)
        g = [0, 0, 0]
        for k, count in enumerate(comp):
            if count > 0:
                count_vec[_PAIR_FLAT_IDX[k]] = count
            g[_PAIR_TO_GROUP[k]] += count

        orbit_idx = weak_composition_to_index(n, d_AB * d_AB, count_vec)
        orbit_indices.append(orbit_idx)
        group_totals.append(g)

    return (
        np.array(orbit_indices, dtype=np.int64),
        np.array(group_totals, dtype=np.int32),
    )


def depolarizing_channel_with_support(
    n: int,
    p: float,
    tol: float = 1e-14,
    xp=None,
    *,
    _bases=None,
):
    """Compute sparse orbit coefficients for the n-copy depolarizing channel.

    Exploits CLDUI structure: only C(n+5, 5) of the C(n+15, 15) total orbits
    are nonzero. Exploits the stronger isotropic structure to evaluate each
    coefficient as a simple product of three powers:

        c[orbit] = alpha^g0 * beta^g1 * gamma^g2

    where alpha = N[0,0], beta = N[1,1], gamma = N[0,3] are the three independent
    entries of the depolarizing Choi matrix.

    Analogous to superactivation_channels_with_support for a single n-copy channel.

    Args:
        n: Number of channel uses.
        p: Depolarizing parameter.
        tol: Threshold below which a coefficient is treated as zero.
        xp: Array module (numpy or cupy). If None, uses numpy. Pass the GPU
            module when the output will be consumed by GPU seesaw functions.
        _bases: Pre-computed (orbit_indices, group_totals) from
                depolarizing_orbit_bases(n). Pass this when calling in a loop
                over p for the same n to avoid re-enumerating orbits.

    Returns:
        (support_1d, c_N_support):
            support_1d:   1D array of nonzero orbit indices (int64).
            c_N_support:  1D array of corresponding coefficient values (float64).
            Both arrays are on the device specified by xp.
    """
    array_module = xp if xp is not None else np

    # Extract the 3 independent Choi matrix entries (all real for isotropic state)
    N_choi = depolarizing_choi(p, d=2)
    alpha = float(N_choi[0, 0].real)
    beta  = float(N_choi[1, 1].real)
    gamma = float(N_choi[0, 3].real)

    if _bases is None:
        orbit_indices, group_totals = depolarizing_orbit_bases(n)
    else:
        orbit_indices, group_totals = _bases

    # Vectorized coefficient: alpha^g0 * beta^g1 * gamma^g2
    g0 = group_totals[:, 0]
    g1 = group_totals[:, 1]
    g2 = group_totals[:, 2]
    coeffs = (alpha ** g0) * (beta ** g1) * (gamma ** g2)

    nonzero_mask = np.abs(coeffs) > tol
    support_1d   = array_module.asarray(orbit_indices[nonzero_mask], dtype=array_module.int64)
    c_N_support  = array_module.asarray(coeffs[nonzero_mask],        dtype=array_module.float64)

    return support_1d, c_N_support
