"""
Wang-Duan states demonstrating non-additivity of Rains bound.

These states were discovered by Wang & Duan and studied by Tomamichel-Rubboli,
demonstrating that the Rains bound is not additive: R(ρ⊗ρ) < 2R(ρ).

Reference:
- Wang & Duan (original discovery)
- Tomamichel & Rubboli, "Non-additive Rains information of quantum channels"
"""

import numpy as np
from typing import Optional, Tuple


class WangDuanState:
    """
    Wang-Duan state demonstrating non-additivity of Rains bound.

    This is a specific 2-qubit state that demonstrates:
        R(ρ⊗ρ) < 2R(ρ)

    where R(ρ) is the Rains bound (minimum relative entropy to PPT states).

    The state is NPT (entangled) with Rains bound R(ρ) ≈ 0.389 bits.
    The optimal PPT approximation σ is also provided.

    Properties:
    - Dimension: 2⊗2 (two qubits)
    - NPT entangled state
    - Exhibits zero-binegativity property
    - Rains bound R(ρ) ≈ 0.389 bits for r=0.547

    Attributes:
        r: Parameter controlling the state (default: 0.547)
        state_type: Either 'rho' (the NPT state) or 'sigma' (optimal PPT approximation)
        d: Local dimension (always 2 for qubits)
    """

    def __init__(
        self,
        r: float = 0.547,
        state_type: str = 'rho',
        name: Optional[str] = None
    ):
        """
        Initialize Wang-Duan state.

        Args:
            r: Parameter for the construction (default: 0.547 from Tomamichel-Rubboli)
            state_type: Either 'rho' (NPT state) or 'sigma' (PPT approximation)
            name: Optional name for the state

        Raises:
            ValueError: If state_type is not 'rho' or 'sigma'
        """
        if state_type not in ['rho', 'sigma']:
            raise ValueError(f"state_type must be 'rho' or 'sigma', got {state_type}")

        self.r = r
        self.state_type = state_type
        self.d = 2  # Always qubits
        self._rho = None  # Cache for density matrix

        if name is None:
            name = f"WangDuan_{state_type}(r={r:.3f})"
        self.name = name

    @property
    def rho(self) -> np.ndarray:
        """Get the density matrix, computing it if not cached."""
        if self._rho is None:
            self._rho = self._construct_state()
        return self._rho

    def _construct_state(self) -> np.ndarray:
        """Construct the state density matrix."""
        if self.state_type == 'rho':
            return self._construct_rho()
        else:
            return self._construct_sigma()

    def _construct_rho(self) -> np.ndarray:
        """
        Construct the NPT state ρ.

        This state has:
        - Rains bound R(ρ) ≈ 0.389 bits
        - Is NPT (min eigenvalue of ρ^Γ ≈ -0.276)
        - Satisfies zero-binegativity property

        The construction uses the parameter r to define:
        - y = √(4r² - 5r/2 + 33/64)
        - x = r + (32r² - 10r + 1)/(256r² - 160r + 33) + ...

        Returns:
            4×4 density matrix in computational basis |00⟩, |01⟩, |10⟩, |11⟩
        """
        r = self.r

        # First compute y
        y = np.sqrt(4*r**2 - 5*r/2 + 33/64)

        # Then compute x
        term1 = r
        term2 = (32*r**2 - 10*r + 1) / (256*r**2 - 160*r + 33)
        term3_num = (16*r - 5) * (y**(-1))
        term3_den = 32*np.log(5/8 - y) - 32*np.log(5/8 + y)
        term3 = term3_num / term3_den

        x = term1 + term2 + term3

        # Compute the off-diagonal coefficient
        off_diag_coeff = (32*r**2 - (6 + 32*x)*r + 10*x + 1) / (4*np.sqrt(2))

        # Construct the 4x4 density matrix
        # Basis order: |00⟩, |01⟩, |10⟩, |11⟩
        rho = np.zeros((4, 4), dtype=np.complex128)

        # Diagonal elements
        rho[0, 0] = 1/8                    # |00⟩⟨00|
        rho[1, 1] = x                      # |01⟩⟨01|
        rho[2, 2] = (7 - 8*x) / 8          # |10⟩⟨10|
        rho[3, 3] = 0                      # |11⟩⟨11|

        # Off-diagonal elements (|01⟩⟨10| + |10⟩⟨01|)
        # These are real for this state
        rho[1, 2] = off_diag_coeff
        rho[2, 1] = off_diag_coeff

        return rho

    def _construct_sigma(self) -> np.ndarray:
        """
        Construct the optimal PPT approximation σ.

        This is the minimizer of the Rains bound:
            R(ρ) = min { D(ρ||σ) : σ is PPT }

        where D(ρ||σ) = Tr(ρ log ρ) - Tr(ρ log σ) is the quantum relative entropy.

        For r=0.547, we have D(ρ||σ) ≈ 0.389 bits.

        Returns:
            4×4 density matrix for optimal PPT approximation
        """
        r = self.r

        sigma = np.zeros((4, 4), dtype=np.complex128)

        # Diagonal elements
        sigma[0, 0] = 1/4                   # |00⟩⟨00|
        sigma[1, 1] = r                     # |01⟩⟨01|
        sigma[2, 2] = 5/8 - r               # |10⟩⟨10|
        sigma[3, 3] = 1/8                   # |11⟩⟨11|

        # Off-diagonal elements (|01⟩⟨10| + |10⟩⟨01|)
        sigma[1, 2] = 1/(4*np.sqrt(2))
        sigma[2, 1] = 1/(4*np.sqrt(2))

        return sigma

    def is_ppt(self, threshold: float = 1e-6) -> bool:
        """
        Check if state is PPT (Positive Partial Transpose).

        Args:
            threshold: Numerical tolerance for negative eigenvalues

        Returns:
            True if PPT, False if NPT
        """
        if self.state_type == 'sigma':
            return True  # sigma is always PPT by construction
        else:
            # rho is NPT
            rho_pt = self._partial_transpose_manual()
            eigvals = np.linalg.eigvalsh(rho_pt)
            return np.all(eigvals >= -threshold)  # ty:ignore[invalid-return-type]

    def negativity(self) -> float:
        """
        Compute the negativity N(ρ) = (||ρ^Γ||_1 - 1) / 2.

        For the rho state with r=0.547, negativity ≈ 0.276.
        For the sigma state, negativity = 0 (PPT).

        Returns:
            Negativity value (≥ 0)
        """
        rho_pt = self._partial_transpose_manual()
        eigvals = np.linalg.eigvalsh(rho_pt)
        neg_eig_sum = np.sum(np.abs(eigvals[eigvals < 0]))
        return neg_eig_sum / 2

    def log_negativity(self) -> float:
        """
        Compute logarithmic negativity E_N(ρ) = log₂(2N(ρ) + 1).

        Returns:
            Log negativity in bits
        """
        neg = self.negativity()
        if neg == 0:
            return 0.0
        return np.log2(1 + 2 * neg)

    def _partial_transpose_manual(self) -> np.ndarray:
        """
        Compute partial transpose manually for 2⊗2 system.

        For a 4×4 matrix in basis |00⟩, |01⟩, |10⟩, |11⟩,
        partial transpose on second system swaps indices 1↔2.

        Returns:
            Partial transpose density matrix
        """
        rho_pt = np.zeros_like(self.rho)
        d = self.d

        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        # |i,j⟩⟨k,l| -> |i,l⟩⟨k,j|
                        rho_pt[i * d + l, k * d + j] = self.rho[i * d + j, k * d + l]

        return rho_pt

    def purity(self) -> float:
        """
        Compute the purity Tr(ρ²).

        Returns:
            Purity value (1/16 ≤ Tr(ρ²) ≤ 1)
        """
        return np.trace(self.rho @ self.rho).real

    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

        Returns:
            Entropy in bits
        """
        eigenvalues = np.linalg.eigvalsh(self.rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

    def rains_bound(self) -> float:
        """
        Compute the Rains bound for this state.

        The Rains bound is:
            R(ρ) = min { D(ρ||σ) : σ is PPT }

        For the 'rho' state with r=0.547, R(ρ) ≈ 0.389 bits.
        For the 'sigma' state, R(σ) = 0 (since it's already PPT).

        Note: This returns the known value for r=0.547. For other values of r,
        this would require solving an optimization problem.

        Returns:
            Rains bound in bits (approximate for rho, 0 for sigma)
        """
        if self.state_type == 'sigma':
            return 0.0
        else:
            # For r=0.547, the Rains bound is approximately 0.389 bits
            # This is the relative entropy D(rho || sigma)
            if abs(self.r - 0.547) < 1e-6:
                return 0.389  # Known value from Tomamichel-Rubboli
            else:
                # For other r values, would need to compute
                # For now, return NaN to indicate unknown
                return np.nan

    def __repr__(self) -> str:
        """String representation."""
        return f"WangDuanState(type={self.state_type}, r={self.r:.4f}, PPT={self.is_ppt()})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.name


def create_wang_duan_pair(r: float = 0.547) -> Tuple[WangDuanState, WangDuanState]:
    """
    Create both ρ and σ states for the Wang-Duan non-additive Rains example.

    The pair demonstrates non-additivity of the Rains bound:
        R(ρ⊗ρ) < 2R(ρ)

    Args:
        r: Parameter (default: 0.547 from Tomamichel-Rubboli paper)

    Returns:
        Tuple of (rho_state, sigma_state) where:
        - rho_state is the NPT entangled state
        - sigma_state is the optimal PPT approximation
    """
    rho = WangDuanState(r=r, state_type='rho')
    sigma = WangDuanState(r=r, state_type='sigma')
    return rho, sigma
