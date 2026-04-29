import numpy as np
from permqit.utilities.general_functions import ket, dag

def max_ent(dim, normalized=True, as_matrix=True):
    """
    Generates the dim-dimensional maximally entangled state, which is defined as

    (1/sqrt(dim))*(|0>|0>+|1>|1>+...+|d-1>|d-1>).

    If normalized=False, then the function returns the unnormalized maximally entangled
    vector.

    If as_matrix=True, then the function returns the state as a density matrix.
    """

    if normalized:
        Bell = (1.0 / np.sqrt(dim)) * np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if as_matrix:
            return Bell @ dag(Bell)
        else:
            return Bell
    else:
        Gamma = np.sum([ket(dim, [i, i]) for i in range(dim)], 0)
        if as_matrix:
            return Gamma @ dag(Gamma)
        else:
            return Gamma

