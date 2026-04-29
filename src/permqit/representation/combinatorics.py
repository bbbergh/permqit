from __future__ import annotations

import itertools
from typing import List, Tuple, Dict, Iterable, Sequence
from collections import Counter
import math

import numpy as np
import scipy
from scipy.special._comb import _comb_int


def multinomial_coeff(ks: Sequence[int]|np.ndarray) -> int:
    """Compute the multinomial coefficient for a sequence of nonnegative integers.

    multinomial_coeff([2,1,1])

    Returns exact integer value of (sum ks)! / (Π k_i!).
    """
    rem = sum(ks)
    res = 1
    for k in ks:
        if k:
            res *= math.comb(rem, k)
            rem -= k
    return res

def get_multinomial_coeff_func_xp():
    from ..utilities import backend
    if not backend.USE_GPU:
        return multinomial_coeff_fast

    from ..utilities.backend import cupyx, cp
    assert cupyx.scipy.special

    def multinomial_coeff_gpu(ks, axis=-1):
        cumsum = cp.cumsum(ks, axis=axis)
        return cp.prod(cupyx.scipy.special.binom(cumsum, ks), axis=axis)

    return multinomial_coeff_gpu

def multinomial_coeff_fast(ks: np.ndarray, *, axis=-1) -> np.ndarray:
    cumsum = np.cumsum(ks, axis=axis)
    return np.prod(scipy.special.binom(cumsum, ks), axis=axis)


def weak_compositions(n, k, dtype=np.int_) -> Iterable[np.ndarray]:
    """
    Yields all arrays of size k with non-negative integer entries that sum to n.
    Yields in the canonical ordering with [n, 0, ..., 0] being the first returned object, and [0, ..., 0, n] the last.
    """
    if k < 1:
        return

    # To obtain non-negative parts summing to n, we find positive parts
    # summing to n + k, then subtract 1 from each part.
    m = n + k

    # We choose k-1 dividers from the range (1, ... , m-1)
    for c in itertools.combinations(range(1, m), k - 1):
        # We add 0 and m as boundaries to calculate the gaps
        yield np.fromiter((b - a - 1 for a, b in itertools.pairwise((0,) + c + (m,))), dtype=dtype, count=k)[::-1]


def _combination_rank_lex(M: int, R: int, comb: Sequence[int]) -> int:
    """Rank of combination (sorted, 1-based elements from 1..M) in lex order. 0-based rank."""
    if R == 0:
        return 0
    rank = 0
    prev = 0
    for i, s in enumerate(comb):
        for x in range(prev + 1, s):
            rank += _comb_int(M - x, R - 1 - i)
        prev = s
    return rank


def _combination_unrank_lex(M: int, R: int, idx: int) -> Tuple[int, ...]:
    """Unrank: the idx-th combination (sorted, 1-based from 1..M) in lex order."""
    if R == 0:
        return ()
    s = 1
    total = 0
    while s <= M - R + 1:
        count = _comb_int(M - s, R - 1)
        if total + count > idx:
            break
        total += count
        s += 1
    rest = _combination_unrank_lex(M - s, R - 1, idx - total)
    return (s,) + tuple(x + s for x in rest)


def weak_composition_to_index(n: int, k: int, comp: np.ndarray) -> int:
    """
    Index of weak composition comp (length k, sums to n) in the same order as weak_compositions(n, k).
    comp is in yielded order: comp[0] is first part (e.g. n for [n,0,...,0]). O(k) time, O(1) space.
    """
    if k <= 1:
        return 0
    m = n + k
    R = k - 1
    # gaps before [::-1] in weak_compositions are (comp[k-1], comp[k-2], ..., comp[0])
    cum = 0
    divs = []
    for i in range(k - 1):
        cum += int(comp[k - 1 - i]) + 1
        divs.append(cum)
    return _combination_rank_lex(m - 1, R, divs)


def weak_composition_from_index(n: int, k: int, idx: int, dtype=np.int_) -> np.ndarray:
    """
    The idx-th weak composition of n into k parts, same order as weak_compositions(n, k).
    O(k) time, O(1) space.
    """
    if k <= 1:
        return np.array([n], dtype=dtype)
    m = n + k
    R = k - 1
    divs = _combination_unrank_lex(m - 1, R, idx)
    parts = np.empty(k, dtype=dtype)
    prev = 0
    for i, d in enumerate(divs):
        parts[i] = d - prev - 1
        prev = d
    parts[k - 1] = m - prev - 1
    return parts[::-1]
