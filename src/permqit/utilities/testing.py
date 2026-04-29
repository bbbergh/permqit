import numpy as np
from ..algebra import EndSnOrbitBasis, MatrixStandardBasis, TensorProductBasis, MatrixBasis, MatrixTensorProductBasis
from ..representation.isomorphism import EndSnAlgebraIsomorphism

def assert_is_valid_choi(choi_matrix: np.ndarray, d_R, d_B):
    """Assumes a choi matrix in order RB and tests positivity and trace preserving"""
    assert choi_matrix.shape == (d_R * d_B, d_R * d_B)
    np.testing.assert_allclose(choi_matrix.conj().T, choi_matrix, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(choi_matrix)
    np.testing.assert_array_less(-1e-7, eigenvalues)  # Test for positivity
    np.testing.assert_allclose(np.einsum('rbRb->rR', choi_matrix.reshape(d_R, d_B, d_R, d_B)), np.eye(d_R),
                               atol=1e-10)


def assert_is_hermitian(coeffs: np.ndarray, basis: MatrixBasis):
    np.testing.assert_allclose(coeffs, basis.transpose(coeffs).conj(), atol=1e-10)

def assert_is_valid_choi_via_isomorphism(choi_coeffs: np.ndarray, iso: EndSnAlgebraIsomorphism, d_R, d_B):
    """
    Takes choi_coeffs as coeffs in the TensorProductBasis([MatrixStandardBasis(d_R), EndSnOrbitBasis(n, d_B)]) and verifies
    that these correspond to a valid choi matrix, without constructing the big matrix on V^n.
    :param choi_coeffs:
    :param iso:
    :param d_R:
    :param d_B:
    :return:
    """
    assert choi_coeffs.shape == (d_R**2 * iso.basis_from.size(),)
    assert iso.basis_from.d == d_B

    big_basis = MatrixTensorProductBasis((MatrixStandardBasis(d_R), iso.basis_from))
    assert_is_hermitian(choi_coeffs, big_basis)
    block_coeffs = iso.apply_to_coefficient_vector(choi_coeffs.reshape(d_R ** 2, iso.basis_from.size()).transpose(1, 0)).transpose(1, 0).reshape(d_R, d_R, iso.basis_from.size())
    # block_coeffs now in shape (d_R**2, dim(End^{Sn}(V^n))
    def to_full(blocked_basis_coeffs):
        return iso.basis_to.linear_combination(blocked_basis_coeffs).to_full_matrix()

    total_size = np.sum(iso.basis_to.block_sizes)
    full_matrix = np.apply_along_axis(to_full, 2, block_coeffs).reshape(d_R, d_R, total_size, total_size).transpose(0,2,1,3).reshape(d_R*total_size, d_R*total_size)
    np.testing.assert_array_less(-1e-7, np.linalg.eigvals(full_matrix))

    multiplicities = np.asarray([p.count_standard_tableaux() for p in iso.basis_to.partitions])
    def to_trace(blocked_basis_coeffs):
        return iso.basis_to.linear_combination(blocked_basis_coeffs).trace_with_multiplicities(multiplicities)

    partial_trace = np.apply_along_axis(to_trace, 2, block_coeffs)

    np.testing.assert_almost_equal(partial_trace, np.eye(d_R))