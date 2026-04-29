"""Amplitude Damping Channel (ADC) utilities.

The qubit amplitude damping channel (N=0 bath) with damping parameter γ ∈ [0, 1]:

    A_γ(ρ) = K_0 ρ K_0† + K_1 ρ K_1†

    K_0 = [[1,       0      ],    K_1 = [[0,    sqrt(γ)],
            [0, sqrt(1-γ)  ]]             [0,    0      ]]

For γ=0: identity channel. For γ=1: maps everything to |0><0|.

Choi matrix (standard convention, Tr_B(J) = I_in):

    J = [[1,         0, 0, sqrt(1-γ)],
         [0,         0, 0, 0        ],
         [0,         0, γ, 0        ],
         [sqrt(1-γ), 0, 0, 1-γ     ]]

Row/col ordering: (R=0,B=0)=0, (R=0,B=1)=1, (R=1,B=0)=2, (R=1,B=1)=3.

Symmetry: CLDUI (invariant under U⊗U* for diagonal unitaries U = diag(e^iθ, 1)).
The Choi matrix has only 5 nonzero entries at flat positions {0, 3, 10, 12, 15}.

Orbit structure for the symmetric seesaw
-----------------------------------------
The n-copy orbit basis (EndSnOrbitBasis(n, d_AB=4)) has C(n+15, 15) total orbits,
but the ADC only has nonzero support on C(n+4, 4) orbits (those involving only
the 5 ADC pair types).

Each orbit is characterized by a count vector (k0, k1, k2, k3, k4) summing to n,
where k_i counts how many copies of the i-th pair type appear:
    Type 0: pair (0,0) → J=1
    Type 1: pair (0,3) → J=sqrt(1-γ)
    Type 2: pair (2,2) → J=γ
    Type 3: pair (3,0) → J=sqrt(1-γ)
    Type 4: pair (3,3) → J=1-γ

The orbit coefficient is:
    c[orbit] = 1^k0 * sqrt(1-γ)^k1 * γ^k2 * sqrt(1-γ)^k3 * (1-γ)^k4
             = γ^g_C * (1-γ)^(g_B/2 + g_D)
where:
    g_B = k1 + k3   (count of off-diagonal pairs (0,3) and (3,0))
    g_C = k2        (count of diagonal-γ pairs (2,2))
    g_D = k4        (count of diagonal-(1-γ) pairs (3,3))

amplitude_damping_orbit_bases(n) precomputes orbit indices and group totals once
per n (O(C(n+4,4)) work) so that per-γ evaluations are O(C(n+4,4)) vectorized
ops with no orbit enumeration. Speedups vs naive method are ~1000× for n≥10.
"""
import numpy as np
from permqit.representation.combinatorics import weak_compositions, weak_composition_to_index


__all__ = [
    "amplitude_damping_choi",
    "ADC_SUPPORT_GROUPED",
    "amplitude_damping_orbit_bases",
    "amplitude_damping_channel_with_support",
]

# ADC orbit structure constants
# The 5 nonzero pair positions (flat index = i_in * 4 + j_in, d_AB=4):
#   (0,0)->0: J=1, (0,3)->3: J=sqrt(1-γ), (2,2)->10: J=γ, (3,0)->12: J=sqrt(1-γ), (3,3)->15: J=1-γ
_ADC_PAIR_FLAT_IDX = [0, 3, 10, 12, 15]
_ADC_PAIR_TYPES    = 5

# Grouped support for non-symmetric seesaw (d=2):
# Each group contains (i_in, i_out, j_in, j_out) tuples with equal Choi values.
# Mapping: flat row = i_in * 2 + i_out, flat col = j_in * 2 + j_out.
#   Group 0: {(0,0,0,0)}             — J[0,0] = 1          (constant)
#   Group 1: {(0,0,1,1), (1,1,0,0)} — J[0,3] = J[3,0] = sqrt(1-γ)
#   Group 2: {(1,0,1,0)}             — J[2,2] = γ
#   Group 3: {(1,1,1,1)}             — J[3,3] = 1-γ
ADC_SUPPORT_GROUPED = [
    [(0, 0, 0, 0)],
    [(0, 0, 1, 1), (1, 1, 0, 0)],
    [(1, 0, 1, 0)],
    [(1, 1, 1, 1)],
]


def amplitude_damping_choi(gamma: float) -> np.ndarray:
    """Return the Choi matrix of the amplitude damping channel.

    Uses the convention Tr_B(J) = I_in (consistent with J_identity = Σ_{ij} |i><j|⊗N(|i><j|)).

    At γ=0 this equals depolarizing_choi(p=0), the identity channel Choi.

    Args:
        gamma: Damping parameter γ ∈ [0, 1]. γ=0: identity. γ=1: maps to |0><0|.

    Returns:
        4×4 complex128 Choi matrix with row/col ordering (R*2+B).
    """
    J = np.zeros((4, 4), dtype=complex)
    J[0, 0] = 1.0
    J[0, 3] = np.sqrt(1.0 - gamma)
    J[3, 0] = np.sqrt(1.0 - gamma)
    J[2, 2] = gamma
    J[3, 3] = 1.0 - gamma
    return J


def amplitude_damping_orbit_bases(n: int):
    """Precompute ADC orbit indices and group totals for n copies.

    Enumerates all C(n+4, 4) ADC-compatible orbits of EndSnOrbitBasis(n, d_AB=4)
    and records, for each orbit:
      - its index in the orbit basis
      - the group totals (g_B, g_C, g_D) needed to evaluate the channel
        coefficient as γ^g_C * (1-γ)^(g_B/2 + g_D)

    This is O(C(n+4, 4)) work and needs to be done only once per n. Per-γ
    evaluation then reduces to vectorized power computations (no enumeration).

    Args:
        n: Number of channel uses (n >= 1).

    Returns:
        (orbit_indices, group_totals):
            orbit_indices: int64 array of shape (K,) — orbit indices in EndSnOrbitBasis(n, 4).
            group_totals:  int32 array of shape (K, 3) — (g_B, g_C, g_D) per orbit.
            K = C(n+4, 4) is the number of ADC-compatible orbits.
    """

    d_AB = 4
    orbit_indices = []
    group_totals = []

    for comp in weak_compositions(n, _ADC_PAIR_TYPES):
        k0, k1, k2, k3, k4 = comp

        # Build full count vector (length d_AB^2 = 16) for orbit index computation.
        # Only the 5 ADC-pair positions are nonzero.
        count_vec = np.zeros(d_AB * d_AB, dtype=np.int_)
        for k, flat_idx in zip(comp, _ADC_PAIR_FLAT_IDX):
            if k > 0:
                count_vec[flat_idx] = k

        orbit_idx = weak_composition_to_index(n, d_AB * d_AB, count_vec)
        orbit_indices.append(orbit_idx)
        group_totals.append([k1 + k3, k2, k4])  # (g_B, g_C, g_D)

    return (
        np.array(orbit_indices, dtype=np.int64),
        np.array(group_totals, dtype=np.int32),
    )


def amplitude_damping_channel_with_support(
    n: int,
    gamma: float,
    tol: float = 1e-14,
    xp=None,
    *,
    _bases=None,
):
    """Compute sparse orbit coefficients for the n-copy amplitude damping channel.

    Exploits ADC structure: only C(n+4, 4) of the C(n+15, 15) total orbits are
    nonzero. Each coefficient is evaluated as:

        c[orbit] = γ^g_C * (1-γ)^(g_B/2 + g_D)

    Args:
        n: Number of channel uses.
        gamma: Damping parameter γ ∈ [0, 1].
        tol: Threshold below which a coefficient is treated as zero.
        xp: Array module (numpy or cupy). If None, uses numpy.
        _bases: Pre-computed (orbit_indices, group_totals) from
                amplitude_damping_orbit_bases(n). Pass when calling in a loop
                over γ for the same n to avoid re-enumerating orbits.

    Returns:
        (support_1d, c_N_support):
            support_1d:   1D array of nonzero orbit indices (int64).
            c_N_support:  1D array of corresponding coefficient values (float64).
            Both arrays are on the device specified by xp.
    """
    array_module = xp if xp is not None else np

    if _bases is None:
        orbit_indices, group_totals = amplitude_damping_orbit_bases(n)
    else:
        orbit_indices, group_totals = _bases

    g_B = group_totals[:, 0].astype(float)
    g_C = group_totals[:, 1].astype(float)
    g_D = group_totals[:, 2].astype(float)

    # c = γ^g_C * (1-γ)^(g_B/2 + g_D)
    # numpy handles 0^0 = 1 correctly (needed for g_C=0 at γ=0, etc.)
    one_minus_gamma = 1.0 - gamma
    coeffs = (gamma ** g_C) * (one_minus_gamma ** (0.5 * g_B + g_D))

    nonzero_mask = np.abs(coeffs) > tol
    support_1d  = array_module.asarray(orbit_indices[nonzero_mask], dtype=array_module.int64)
    c_N_support = array_module.asarray(coeffs[nonzero_mask],        dtype=array_module.float64)

    return support_1d, c_N_support
