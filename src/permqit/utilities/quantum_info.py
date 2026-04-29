"""Quantum information primitives: entropy, partial trace, channel application, etc.

Von Neumann quantities (entropy, relative_entropy, etc.) are in bits (log base 2).
Renyi quantities (renyi_entropy, sandwiched_renyi_relative, petz_renyi_relative)
are in nats (natural log), matching the convention in error exponent literature.
"""
import numpy as np
from scipy.linalg import eigvalsh

from .general_functions import partial_trace


def entropy(rho, tol=1e-15):
    """Von Neumann entropy S(rho) = -Tr(rho log2 rho), in bits."""
    evals = eigvalsh(rho)
    evals = evals[evals > tol]
    return -float(np.sum(evals * np.log2(evals)))


def partial_trace_subsystem(rho, sys, dims):
    """Trace out subsystem `sys` (0-indexed) from rho with dimensions `dims`.

    Handles arbitrary number of subsystems (not just bipartite).

    Example: partial_trace_subsystem(rho_AB, 0, [d_A, d_B]) traces out A, returns rho_B.
    """
    n_sys = len(dims)
    d_total = int(np.prod(dims))
    assert rho.shape == (d_total, d_total)

    shape = list(dims) + list(dims)
    rho_t = rho.reshape(shape)

    return np.trace(rho_t, axis1=sys, axis2=sys + n_sys).reshape(
        d_total // dims[sys], d_total // dims[sys]
    )


def apply_channel_kraus(rho, kraus_ops, sys, dims):
    """Apply a channel (Kraus operators) to subsystem `sys` of rho.

    Parameters
    ----------
    rho : (d_total, d_total) array
    kraus_ops : list of (d_out, d_in) arrays
    sys : int, 0-indexed subsystem to act on
    dims : list of ints, subsystem dimensions

    Returns
    -------
    rho_out : density matrix with dims[sys] replaced by d_out
    """
    d_in = dims[sys]
    d_out = kraus_ops[0].shape[0]

    d_before = int(np.prod(dims[:sys]))
    d_after = int(np.prod(dims[sys + 1:]))

    rho_out = np.zeros((d_before * d_out * d_after,) * 2, dtype=complex)
    for K in kraus_ops:
        K_full = np.kron(np.kron(np.eye(d_before), K), np.eye(d_after))
        rho_out += K_full @ rho @ K_full.conj().T

    return rho_out


def matrix_log2(A, tol=1e-15):
    """Hermitian matrix logarithm base 2; zero out eigenvalues below tol."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)
    log_vals = np.where(vals > tol, np.log2(vals), 0.0)
    return (vecs * log_vals) @ vecs.conj().T


def relative_entropy(rho, sigma, tol=1e-15):
    """Quantum relative entropy D(rho || sigma) in bits."""
    return -entropy(rho, tol) - float(np.real(np.trace(rho @ matrix_log2(sigma, tol))))


def coherent_information(rho, dims):
    """Coherent information I(A>B) = S(B) - S(AB) for bipartite rho_AB.

    Parameters
    ----------
    rho : density matrix on A ⊗ B
    dims : [d_A, d_B]
    """
    rho_B = partial_trace(rho, dims, 0)
    return entropy(rho_B) - entropy(rho)


def variance_coherent(rho, dims, tol=1e-15):
    """Coherent information variance V(A>B).

    V = Tr[rho (log rho - log(I_A ⊗ rho_B))^2] - I(A>B)^2
    """
    d_A = dims[0]
    rho_B = partial_trace(rho,  dims, 0)
    sigma = np.kron(np.eye(d_A), rho_B)
    return _variance_relative(rho, sigma, tol)


def _variance_relative(rho, sigma, tol=1e-15):
    """Relative entropy variance V(rho || sigma) = Tr[rho (log rho - log sigma)^2] - D^2."""
    D = relative_entropy(rho, sigma, tol)
    diff = matrix_log2(rho, tol) - matrix_log2(sigma, tol)
    return float(np.real(np.trace(rho @ diff @ diff))) - D ** 2


def third_moment_absolute(rho, sigma, tol=1e-15):
    """Third absolute moment T^3 = E[|log(rho/sigma) - D|^3].

    Uses the spectral decomposition approach.
    """
    D = relative_entropy(rho, sigma, tol)

    vals_rho, vecs_rho = np.linalg.eigh(rho)
    vals_sigma, vecs_sigma = np.linalg.eigh(sigma)
    vals_rho = np.maximum(vals_rho, 0.0)
    vals_sigma = np.maximum(vals_sigma, 0.0)

    overlaps = vecs_rho.conj().T @ vecs_sigma
    ov2 = np.abs(overlaps) ** 2

    P = vals_rho[:, None] * ov2

    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = vals_rho[:, None] / np.where(vals_sigma > tol, vals_sigma, 1.0)[None, :]
        log_lh = np.where(
            (vals_rho[:, None] > tol) & (vals_sigma[None, :] > tol),
            np.log2(quotient),
            0.0,
        )

    diff = np.abs(log_lh - D)
    return float(np.real(np.sum(P * diff ** 3)))


# =========================================================================
# Renyi entropies (in nats)
# =========================================================================

def renyi_entropy(rho, alpha, tol=1e-15):
    """Renyi entropy H_alpha(rho) = 1/(1-alpha) * ln(Tr[rho^alpha]), in nats.

    For alpha -> 1, converges to the von Neumann entropy (in nats).
    """
    evals = eigvalsh(rho)
    evals = evals[evals > tol]
    if abs(alpha - 1.0) < 1e-10:
        return -float(np.sum(evals * np.log(evals)))
    return float(1.0 / (1.0 - alpha) * np.log(np.sum(evals ** alpha)))


def sandwiched_renyi_relative(rho, sigma, alpha, tol=1e-15):
    """Sandwiched Renyi relative entropy D_alpha~(rho || sigma), in nats.

    D_alpha~(rho || sigma) = 1/(alpha-1) * ln(Tr[(sigma^{(1-alpha)/(2*alpha)} rho sigma^{(1-alpha)/(2*alpha)})^alpha])

    Defined for alpha in (0, +inf), alpha != 1. For alpha -> 1 converges to D(rho||sigma).
    """
    if abs(alpha - 1.0) < 1e-10:
        return relative_entropy(rho, sigma, tol) * np.log(2)  # convert bits to nats

    vals_s, vecs_s = np.linalg.eigh(sigma)
    vals_s = np.maximum(vals_s, 0.0)

    power = (1.0 - alpha) / (2.0 * alpha)
    sigma_pow_vals = np.where(vals_s > tol, vals_s ** power, 0.0)
    sigma_pow = (vecs_s * sigma_pow_vals) @ vecs_s.conj().T

    M = sigma_pow @ rho @ sigma_pow
    M = (M + M.conj().T) / 2  # enforce Hermitian

    vals_M = eigvalsh(M)
    vals_M = np.maximum(vals_M, 0.0)
    Q_alpha = float(np.sum(vals_M ** alpha))

    if Q_alpha <= 0:
        return float('inf') if alpha > 1 else float('-inf')

    return float(1.0 / (alpha - 1.0) * np.log(Q_alpha))


def petz_renyi_relative(rho, sigma, alpha, tol=1e-15):
    """Petz Renyi relative entropy D_alpha_bar(rho || sigma), in nats.

    D_alpha_bar(rho || sigma) = 1/(alpha-1) * ln(Tr[rho^alpha sigma^{1-alpha}])

    Defined for alpha in (0, 2], alpha != 1. For alpha -> 1 converges to D(rho||sigma).
    """
    if abs(alpha - 1.0) < 1e-10:
        return relative_entropy(rho, sigma, tol) * np.log(2)

    vals_r, vecs_r = np.linalg.eigh(rho)
    vals_r = np.maximum(vals_r, 0.0)
    rho_alpha = (vecs_r * np.where(vals_r > tol, vals_r ** alpha, 0.0)) @ vecs_r.conj().T

    vals_s, vecs_s = np.linalg.eigh(sigma)
    vals_s = np.maximum(vals_s, 0.0)
    sigma_1ma = (vecs_s * np.where(vals_s > tol, vals_s ** (1.0 - alpha), 0.0)) @ vecs_s.conj().T

    Q = float(np.real(np.trace(rho_alpha @ sigma_1ma)))

    if Q <= 0:
        return float('inf') if alpha > 1 else float('-inf')

    return float(1.0 / (alpha - 1.0) * np.log(Q))
