"""
Isotropic states for bipartite systems.
"""

import numpy as np
from typing import Optional

from permqit.utilities.general_functions import partial_transpose
from .max_ent import max_ent
from math import comb
from typing import Sequence, Any

def isotropic_state(p, d, fidelity=True):
    """
    Generates the isotropic state with parameter p on two d-dimensional systems.
    The state is defined as

        rho_Iso = p*|Bell><Bell|+(1-p)*eye(d^2)/d^2,

    where -1/(d^2-1)<=p<=1. Isotropic states are invariant under U ⊗ conj(U)
    for any unitary U, where conj(U) is the complex conjugate of U.

    If fidelity=True, then the function returns a different parameterization of
    the isotropic state in which the parameter p is the fidelity of the state
    with respect to the maximally entangled state.
    """

    Bell = max_ent(d)

    if fidelity:
        return p * Bell + ((1 - p) / (d**2 - 1)) * (np.eye(d**2) - Bell)
    else:
        return p * Bell + (1 - p) * np.eye(d**2) / d**2

def analytical_eigenvalues_transpose(
    n: int,
    coeffs: Sequence[Any],
    d: int
) -> list:
    """
    Compute eigenvalues of the partial transpose of an n-broadcast extension
    of an isotropic state.

    The extension has the form:
        ρ^(n) = Σ_j a_j Σ_{perms} P^{⊗(n-j)} ⊗ Q^{⊗j}

    where coeffs[j] = a_j / C(n,j) are the symmetrized coefficients.

    Eigenvalues group into (n+1) families indexed by k (number of V- indices).

    Args:
        n: Number of copies
        coeffs: Coefficients in symmetric basis (length n+1)
        d: Local dimension

    Returns:
        List of eigenvalues (length n+1)
    """
    # Eigenvalues of P^{T_B} and Q^{T_B} on V+ and V-
    lambda_P_plus = 1.0 / d             # P^{T_B} on V+ (symmetric)
    lambda_P_minus = -1.0 / d           # P^{T_B} on V- (antisymmetric)
    lambda_Q_plus = 1.0 / (d * (d + 1))  # Q^{T_B} on V+
    lambda_Q_minus = 1.0 / (d * (d - 1))  # Q^{T_B} on V-

    eigenvalues = []
    for k in range(n + 1):
        lambda_k = 0
        for j in range(n + 1):
            contribution_j = 0
            for m in range(max(0, j + k - n), min(j, k) + 1):
                if (n - j - k + m) < 0:
                    continue
                combinatorial = comb(k, m) * comb(n - k, j - m)
                factor = (lambda_P_plus ** (n - j - k + m) *
                          lambda_P_minus ** (k - m) *
                          lambda_Q_plus ** (j - m) *
                          lambda_Q_minus ** m)
                contribution_j += combinatorial * factor
            lambda_k += coeffs[j] * contribution_j
        eigenvalues.append(lambda_k)
    return eigenvalues

def analytical_eigenvalues(
    n: int,
    coeffs: Sequence[Any],
    d: int
) -> list:
    """
    Compute eigenvalues (non-partial-transpose) of an n-copy isotropic extension.

    coeffs[j] = a_j / C(n,j)

    Returns:
        List of eigenvalues λ_k, k = 0..n
    """
    eigenvalues = []
    factor = 1.0 / (d**2 - 1)

    for k in range(n + 1):
        lambda_k = 0
        for j in range(k, n + 1):
            lambda_k += coeffs[j] * comb(j, k) * (factor ** k)
        eigenvalues.append(lambda_k)

    return eigenvalues

def get_multiplicities_eigenvalues_transpose(n: int, d: int) -> list:
    """
    Multiplicities of eigenvalue families for isotropic n-extensions.

    μ_k = C(n,k) * [d(d+1)/2]^{n-k} * [d(d-1)/2]^k

    Args:
        n: Number of copies
        d: Local dimension

    Returns:
        List of multiplicities (length n+1)
    """
    dim_plus = d * (d + 1) // 2
    dim_minus = d * (d - 1) // 2
    return [
        comb(n, k) * (dim_plus ** (n - k)) * (dim_minus ** k)
        for k in range(n + 1)
    ]



class IsotropicState:
    """
    Isotropic state on d⊗d system.

    An isotropic state is defined as:
        I(α, d) = α|Φ⁺⟩⟨Φ⁺| + (1-α) * (I - |Φ⁺⟩⟨Φ⁺|) / (d² - 1)

    Equivalently (fidelity parametrization):
        I(F, d) = F|Φ⁺⟩⟨Φ⁺| + (1-F)/(d²-1) * (I - |Φ⁺⟩⟨Φ⁺|)

    Or (standard mixing parametrization):
        I(p, d) = p|Φ⁺⟩⟨Φ⁺| + (1-p) * I/d²

    where |Φ⁺⟩ = (1/√d) Σᵢ |i,i⟩ is the maximally entangled state.

    Properties:
    - Invariant under U⊗U* transformations (U* = complex conjugate)
    - Entangled for α > 1/d (or F > 1/d in fidelity parametrization)
    - PPT for α ≤ 1/d
    - Separable for α ≤ 1/d

    Attributes:
        alpha: Isotropic parameter
        d: Local dimension
        fidelity_param: If True, uses fidelity parametrization
    """

    def __init__(self, alpha: float, d: int = 2, fidelity_param: bool = True, name: Optional[str] = None):
        """
        Initialize an isotropic state.

        Args:
            alpha: Isotropic parameter.
                   If fidelity_param=True: 0 ≤ α ≤ 1 (fidelity with |Φ⁺⟩)
                   If fidelity_param=False: 0 ≤ α ≤ 1 (mixing with maximally mixed)
            d: Local dimension (default: 2 for qubits)
            fidelity_param: If True, use fidelity parametrization (default)
            name: Optional name for the state
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Isotropic parameter α must be in [0, 1], got {alpha}")
        if d < 2:
            raise ValueError(f"Dimension must be at least 2, got {d}")

        self.alpha = alpha
        self.d = d
        self.fidelity_param = fidelity_param
        self._rho = None  # Cache for density matrix

        if name is None:
            param_str = "F" if fidelity_param else "α"
            name = f"Isotropic({param_str}={alpha:.3f}, d={d})"
        self.name = name

    @property
    def rho(self) -> np.ndarray:
        """Get the density matrix, computing it if not cached."""
        if self._rho is None:
            self._rho = self._construct_state()
        return self._rho

    def _construct_state(self) -> np.ndarray:
        """Construct the isotropic state density matrix."""
        # Maximally entangled state |Φ⁺⟩⟨Φ⁺|
        phi_plus = max_ent(self.d, normalized=True, as_matrix=True)

        # Identity
        identity = np.eye(self.d**2, dtype=np.complex128)

        if self.fidelity_param:
            # Fidelity parametrization: F|Φ⁺⟩⟨Φ⁺| + (1-F)/(d²-1) * (I - |Φ⁺⟩⟨Φ⁺|)
            c1 = (1 - self.alpha) / (self.d**2 - 1)
            c2 = self.alpha - c1
            rho = c1 * identity + c2 * phi_plus
        else:
            # Standard mixing: p|Φ⁺⟩⟨Φ⁺| + (1-p) * I/d²
            rho = self.alpha * phi_plus + (1 - self.alpha) * identity / (self.d**2)

        return rho

    def get_fidelity_parameter(self) -> float:
        """
        Get the fidelity parameter F = ⟨Φ⁺|ρ|Φ⁺⟩.

        Returns:
            Fidelity with maximally entangled state
        """
        if self.fidelity_param:
            return self.alpha
        else:
            # Convert from standard to fidelity parametrization
            # F = p + (1-p)/d²
            return self.alpha + (1 - self.alpha) / (self.d**2)

    def get_mixing_parameter(self) -> float:
        """
        Get the mixing parameter p in ρ = p|Φ⁺⟩⟨Φ⁺| + (1-p) I/d².

        Returns:
            Mixing parameter p
        """
        if not self.fidelity_param:
            return self.alpha
        else:
            # Convert from fidelity to standard parametrization
            # p = (F*d² - 1)/(d² - 1)
            return (self.alpha * self.d**2 - 1) / (self.d**2 - 1)

    def is_separable(self) -> bool:
        """
        Check if isotropic state is separable.

        Isotropic states are:
        - Separable for F ≤ 1/d
        - Entangled for F > 1/d

        where F is the fidelity parameter.

        Returns:
            True if separable, False if entangled
        """
        F = self.get_fidelity_parameter()
        threshold = 1.0 / self.d
        return F <= threshold

    def is_ppt(self) -> bool:
        """
        Check if isotropic state is PPT (Positive Partial Transpose).

        Isotropic states are PPT if and only if F ≤ 1/d,
        where F is the fidelity parameter.

        For isotropic states, PPT is equivalent to separability.

        Returns:
            True if PPT, False otherwise
        """
        return self.is_separable()

    def negativity(self) -> float:
        """
        Compute the negativity of the isotropic state analytically.

        For isotropic states:
            N(I(F,d)) = max(0, (d*F - 1) / (d² - 1))

        Returns:
            Negativity value (≥ 0)
        """
        F = self.get_fidelity_parameter()
        return max(0.0, (self.d * F - 1) / (self.d**2 - 1))

    def log_negativity(self) -> float:
        """
        Compute logarithmic negativity analytically.

        E_N(ρ) = log₂(2N(ρ) + 1)

        Returns:
            Log negativity in bits
        """
        neg = self.negativity()
        if neg == 0:
            return 0.0
        return np.log2(1 + 2 * neg)
    
    def log_fidelity_binegativity(self) -> float:
        """
        Compute log-fidelity of binegativity analytically.

        Returns:
            Log-fidelity of binegativity in bits
        """
        f = self.get_fidelity_parameter()
        d = self.d
        if f <= 1/d:
            return 0.0

        fid_bin = np.sqrt(f/d) + np.sqrt((1 - 1/d) * (1 - f))

        return -2 * np.log2(fid_bin)

    def is_k_extendible(self, k: int) -> bool:
        """
        Check if the isotropic state is k-extendible.

        A state ρ_AB is k-extendible if there exists ρ_AB₁...Bₖ such that
        Tr_B₂...Bₖ(ρ_AB₁...Bₖ) = ρ_AB and ρ_AB₁...Bₖ is invariant under
        permutations of B₁,...,Bₖ.

        For isotropic states, the state is k-extendible if:
            F ≤ k/(d(k-1) + k)

        Args:
            k: Extension parameter (k ≥ 2)

        Returns:
            True if k-extendible, False otherwise
        """
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")

        F = self.get_fidelity_parameter()
        threshold = k / (self.d * (k - 1) + k)
        return F <= threshold

    def concurrence(self) -> float:
        """
        Compute the concurrence of the isotropic state (for qubits only).

        For d=2 isotropic states:
            C(ρ) = max(0, 2F - 1)

        where F is the fidelity parameter.

        Returns:
            Concurrence value (0 ≤ C ≤ 1)

        Raises:
            ValueError: If d ≠ 2
        """
        if self.d != 2:
            raise ValueError(f"Concurrence formula only valid for qubits (d=2), got d={self.d}")

        F = self.get_fidelity_parameter()
        return max(0.0, 2 * F - 1)

    def entanglement_of_formation(self) -> float:
        """
        Compute the entanglement of formation (for qubits only).

        For d=2, using concurrence C:
            E_F(ρ) = h((1 + √(1-C²))/2)

        where h(x) = -x log₂(x) - (1-x) log₂(1-x) is the binary entropy.

        Returns:
            Entanglement of formation in ebits

        Raises:
            ValueError: If d ≠ 2
        """
        if self.d != 2:
            raise ValueError(f"E_F formula only valid for qubits (d=2), got d={self.d}")

        C = self.concurrence()
        if C == 0:
            return 0.0

        # Binary entropy function
        def h(x):
            if x <= 0 or x >= 1:
                return 0.0
            return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

        x = (1 + np.sqrt(1 - C**2)) / 2
        return h(x)

    def purity(self) -> float:
        """
        Compute the purity Tr(ρ²) of the isotropic state.

        For isotropic states in fidelity parametrization:
            Tr(ρ²) = F² + 2F(1-F)/(d²-1) + (1-F)²/(d²-1)²

        Returns:
            Purity value (1/d⁴ ≤ Tr(ρ²) ≤ 1)
        """
        F = self.get_fidelity_parameter()
        term1 = F**2
        term2 = 2 * F * (1 - F) / (self.d**2 - 1)
        term3 = (1 - F)**2 / (self.d**2 - 1)**2
        # Simplified but equivalent
        d2 = self.d**2
        return (d2 * F**2 - F + 1) / d2

    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

        Note: This uses numerical diagonalization.

        Returns:
            Entropy in bits
        """
        eigenvalues = np.linalg.eigvalsh(self.rho)
        # Filter out zero/negative eigenvalues (from numerical errors)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def get_partial_transpose(self, system: str = 'B') -> np.ndarray:
        """
        Compute the partial transpose of the isotropic state.

        For isotropic states, the partial transpose has a simple form.

        Args:
            system: Which system to transpose ('A' or 'B')

        Returns:
            Partial transpose density matrix
        """
        dims = [self.d, self.d]
        subsystem = [1] if system == 'A' else [2]
        return partial_transpose(self.rho, subsystem, dims)

    def __repr__(self) -> str:
        """String representation."""
        param_str = "F" if self.fidelity_param else "α"
        return f"IsotropicState({param_str}={self.alpha:.4f}, d={self.d})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.name
