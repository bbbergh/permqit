
import numpy as np
from typing import List, Sequence
import math


def partial_transpose(X: np.ndarray, sys: List[int], dim: List[int]) -> np.ndarray:
    """
    Takes the partial transpose on systems given by sys.

    Compatible with qutipy's partial_transpose signature.

    Parameters
    ----------
    X : ndarray
        The operator/matrix to transpose
    sys : list
        List of subsystem indices to transpose (1-indexed)
    dim : list
        Dimensions of each subsystem

    Returns
    -------
    ndarray
        Partially transposed matrix

    Example
    -------
    partial_transpose(rho_AB, [2], [dimA, dimB]) transposes system B.
    partial_transpose(rho_AB, [1], [dimA, dimB]) transposes system A.
    """
    X = np.array(X)
    n = len(dim)  # Number of subsystems

    # Handle vector input (convert to outer product)
    if len(X.shape) == 1 or X.shape[1] == 1:
        X = X.reshape(-1, 1) @ X.reshape(-1, 1).conj().T

    X_reshape = np.reshape(X, dim + dim)
    axes = list(range(2 * n))

    # Swap row and column indices for systems in sys
    for s in sys:
        axes[s - 1], axes[n + s - 1] = axes[n + s - 1], axes[s - 1]

    X_reshape = np.transpose(X_reshape, tuple(axes))
    dim_total = int(np.prod(dim))
    X_new = np.reshape(X_reshape, (dim_total, dim_total))

    return X_new



def choi_representation(kraus_ops: List[np.ndarray], d_in: int) -> np.ndarray:
    """
    Compute the Choi representation of a quantum channel given by Kraus operators.

    The Choi matrix is: J(E) = sum_k (I ⊗ K_k) |Φ+⟩⟨Φ+| (I ⊗ K_k†)
    where |Φ+⟩ = sum_i |ii⟩ / sqrt(d) is the maximally entangled state.

    Parameters
    ----------
    kraus_ops : list of ndarray
        List of Kraus operators [K_1, K_2, ...]
    d_in : int
        Input dimension

    Returns
    -------
    ndarray
        Choi matrix
    """
    if len(kraus_ops) == 0:
        raise ValueError("Kraus operators list cannot be empty")

    d_out = kraus_ops[0].shape[0]

    # Create maximally entangled state |Φ+⟩ (unnormalized)
    phi_plus = np.zeros((d_in * d_in, 1), dtype=complex)
    for i in range(d_in):
        phi_plus[i * d_in + i] = 1.0

    # Choi matrix: sum_k (I ⊗ K_k) |Φ+⟩⟨Φ+| (I ⊗ K_k†)
    choi = np.zeros((d_in * d_out, d_in * d_out), dtype=complex)

    for K in kraus_ops:
        # I ⊗ K acting on |Φ+⟩
        # |Φ+⟩ is reshaped as (d_in, d_in), I acts on first index, K on second
        phi_reshaped = phi_plus.reshape(d_in, d_in)
        IK_phi = (phi_reshaped @ K.T).reshape(-1, 1)  # (d_in * d_out, 1)
        choi += IK_phi @ IK_phi.conj().T

    return choi

def dag(X):
    """
    Takes the numpy array X and returns its complex conjugate transpose.
    """

    return X.conj().T

def bra(dim, *args):
    k = ket(dim, *args)
    return dag(k)

def outer_product_projector(i: int, j: int, dim: int) -> np.ndarray:
    """
    Create projector |i⟩⟨i| onto computational basis state.
    
    Parameters
    ----------
    i : int
        Basis index
    dim : int
        Hilbert space dimension
        
    Returns
    -------
    ndarray
        Projector matrix of shape (dim, dim)
    """
    ket_i = computational_ket(i, dim)
    ket_j = computational_ket(j, dim)
    bra_j = ket_j.conj().T
    return ket_i @ bra_j

def computational_ket(i: int, dim: int) -> np.ndarray:
    """
    Create computational basis ket |i⟩.
    
    Parameters
    ----------
    i : int
        Basis index (0 to dim-1)
    dim : int
        Hilbert space dimension
        
    Returns
    -------
    ndarray
        Column vector of shape (dim, 1)
    """
    ket = np.zeros((dim, 1), dtype=complex)
    ket[i, 0] = 1.0
    return ket

def ket(dim, *args):
    """
    Generates a standard basis vector in dimension dim.

    For example, ket(2,0)=|0> and ket(2,1)=|1>.

    In general, ket(d,j), for j between 0 and d-1, generates a column vector
    (as a numpy matrix) in which the jth element is equal to 1 and the rest
    are equal to zero.

    ket(d,[j1,j2,...,jn]) generates the tensor product |j1>|j2>...|jn> of
    d-dimensional basis vectors.

    If dim is specified as a list, then, e.g., ket([d1,d2],[j1,j2]) generates the
    tensor product |j1>|j2>, with the first tensor factor being d1-dimensional
    and the second tensor factor being d2-dimensional.
    """

    args = np.array(args)

    if type(dim) == list and len(dim) == 1:
        dim = dim[0]

    if args.size == 1:
        num = args[0]
        out = np.zeros([dim, 1])
        out[num] = 1
    else:
        args = args[0]
        if isinstance(dim, int):
            out = ket(dim, args[0])
            for j in range(1, len(args)):
                out = np.kron(out, ket(dim, args[j]))
        elif isinstance(dim, list):
            out = ket(dim[0], args[0])
            for j in range(1, len(args)):
                out = np.kron(out, ket(dim[j], args[j]))

    return out


def permute_tensor_factors(
    matrix: np.ndarray,
    permutation: List[int],
    dimensions: List[int]
) -> np.ndarray:
    """
    Permute tensor factors of a matrix on composite Hilbert space.
    
    Parameters
    ----------
    matrix : ndarray
        Square matrix on systems A1 ⊗ ... ⊗ An with shape (d_total, d_total)
        where d_total = d_A1 * ... * d_An
    permutation : list of int
        Permutation of [1, 2, ..., n] specifying new order of subsystems
    dimensions : list of int
        Dimensions [d_A1, ..., d_An] of each subsystem
        
    Returns
    -------
    ndarray
        Permuted matrix on systems A_π(1) ⊗ ... ⊗ A_π(n)
        
    Examples
    --------
    >>> # Swap two qubits in a 4x4 matrix
    >>> matrix = np.eye(4)
    >>> permuted = permute_tensor_factors(matrix, [2, 1], [2, 2])
    """
    dimensions = list(dimensions)
    permutation = [i - 1 for i in permutation]  # Convert to 0-indexed
    
    # Reshape to tensor form
    reshaped = matrix.reshape(dimensions + dimensions)
    num_dimensions = len(dimensions)
    
    # Permute both input and output indices
    new_order = permutation + [num_dimensions + i for i in permutation]
    permuted = reshaped.transpose(new_order)
    
    # Reshape back to matrix
    total_dim = math.prod(dimensions)
    return permuted.reshape(total_dim, total_dim)

def binary_entropy(x):
    if x <= 0 or x >= 1:
            return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def partial_trace(rho: np.ndarray, dims: Sequence[int]|np.ndarray, subsystem_to_trace: int):
    """Partial trace over one subsystem of a bipartite matrix rho with dims (dA, dB).

    subsystem_to_trace: 0 to trace out A, 1 to trace out B.
    """
    dA, dB = dims
    if subsystem_to_trace == 0:
        d1, d2 = dA, dB
    elif subsystem_to_trace == 1:
        d1, d2 = dB, dA
    else:
        raise ValueError("subsystem_to_trace must be 0 or 1")
    result = 0
    for i in range(d1):
        start = i * d2
        end = (i + 1) * d2
        if subsystem_to_trace == 0:
            result += rho[start:end, start:end]
        else:
            result += rho[i::d1, i::d1]
    return result