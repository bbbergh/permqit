"""
Generation of the block matrix corresponding to a partition λ:

Functions implement the combinatorial construction shown in the notebook:
- For each partition λ, compute contributions of orbits to the m_λ×m_λ block
  via enumerating κ-maps and evaluating minors/determinants.
- The public functions below are used to precompute per-orbit block contributions
  and to assemble them into constant blocks that are later used in SDP constraints.
"""
import typing
from collections import defaultdict
from typing import Dict, List, Iterator, Tuple, Any
from functools import lru_cache
import itertools
import math
import numpy as np

from . import PairOrbit
from ..algebra.polynomial import Polynomial, MatrixDeterminantPolynomial
from .partition import Partition
from .young_tableau import SSYT

# Type alias for kappa maps: keys are (t, v_tuple, w_tuple) -> multiplicity
Kappa = Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], int]



def calculate_f(lam: Partition, tau: SSYT, gamma: SSYT, d_sys: int, label: str|None=None) -> Polynomial[tuple[int|str, ...], int]:
    """Compute f_{τ,γ}(A) = |C_λ| * Σ_κ N(κ) Π_{(t,v,w)} Δ_t[v,w]^{κ_t(v,w)}.

    Returns a sparse polynomial dict: monomial -> coefficient.
    """
    poly_sum: Polynomial = Polynomial.zero()
    for kappa in enumerate_kappa(lam, tau, gamma, d_sys):
        term = contribution_of_kappa(lam, kappa, label)
        # term is a Polynomial
        poly_sum = poly_sum + term
    f = poly_sum.scale(lam.canonical_tableau.value_column_stabilizer.order())
    return f

def enumerate_kappa(
    lam: Partition,
    tau: SSYT,
    gamma: SSYT,
    d: int
) -> Iterator[Kappa]:
    """
    - enumerate_kappa(lam, tau, gamma, d) generates every admissible κ-map for the given partition λ and tableaux τ, γ, with alphabet size d.
    - - row_symbol_counts(tau) returns an (h x dim) matrix where each entry (i, j) contains the number of times symbol j+1 = {1, ..., dim} appears in row i +1 = {1, ..., h}
    - - flatten_state and unflatten_state convert the residual counts for τ and γ between nested lists and flat tuples
    - - max_multiplicity returns the maximum number of copies of a candidate (v, w) that can still fit given the remaining row counts and how many minors are left to place
    - - dp(t_idx, rt, rg) reconstructs the residual counts dynamically
    """
    h = lam.height
    lam_ext = lam.as_tuple() + (0,)
    N = [lam_ext[t-1] - lam_ext[t] for t in range(1, h+1)]  # N_t

    def row_symbol_counts(tableau: SSYT) -> List[List[int]]:
        cnt = [[0]*d for _ in range(h)]
        for j, L in enumerate(lam):
            row = tableau.rows[j]
            for i in range(L):
                cnt[j][row[i]] += 1
        return cnt

    need_tau   = row_symbol_counts(tau)
    need_gamma = row_symbol_counts(gamma)


    types_by_t = {t: list(itertools.product(
        itertools.product(range(0, d), repeat=t),
        itertools.product(range(0, d), repeat=t))) for t in range(1, h+1)}
    # types_by_t[t] yields ((v_tuple), (w_tuple))

    def flatten_state(rem_tau, rem_gamma):
        rt = tuple(x for i in range(h) for x in rem_tau[i])
        rg = tuple(x for i in range(h) for x in rem_gamma[i])
        return rt, rg

    def unflatten_state(rt: Tuple[int,...], rg: Tuple[int,...]):
        rem_tau = [[0]*d for _ in range(h)]
        rem_gamma = [[0]*d for _ in range(h)]
        it = iter(rt); ig = iter(rg)
        for i in range(h):
            for s in range(d):
                rem_tau[i][s]   = next(it)
                rem_gamma[i][s] = next(ig)
        return rem_tau, rem_gamma

    def max_multiplicity(rem_tau, rem_gamma, t, v, w, remaining):
        ub = remaining
        for i in range(t):
            ub = min(ub, rem_tau[i][v[i]])
            ub = min(ub, rem_gamma[i][w[i]])
            if ub == 0:
                break
        return ub

    @lru_cache(maxsize=None)
    def dp(t_idx: int, rt: Tuple[int,...], rg: Tuple[int,...]) -> Tuple[Dict, ...]:
        rem_tau, rem_gamma = unflatten_state(rt, rg)

        if t_idx == h:
            # success iff all residuals are zero
            if all(all(x == 0 for x in row) for row in rem_tau) and \
               all(all(x == 0 for x in row) for row in rem_gamma):
                return ({},)  # one empty κ to merge upward
            return tuple()     # no solutions

        t = t_idx + 1
        to_place = N[t_idx]

        if to_place == 0:
            return dp(t_idx + 1, rt, rg)

        # candidates feasible at least once
        cand = []
        for v, w in types_by_t[t]:
            ok = True
            for i in range(t):
                if rem_tau[i][v[i]] <= 0 or rem_gamma[i][w[i]] <= 0:
                    ok = False; break
            if ok:
                ub = max_multiplicity(rem_tau, rem_gamma, t, v, w, to_place)
                if ub > 0:
                    cand.append((v, w, ub))
        cand.sort(key=lambda x: x[2])  # tighter bound first

        solutions: List[Kappa] = []

        def distribute(idx, left, rem_tau, rem_gamma, acc_k: Dict):
            if left == 0:
                state = flatten_state(rem_tau, rem_gamma)
                for sub in dp(t_idx + 1, *state):
                    merged = dict(acc_k)
                    merged.update(sub)   # sub is a dict
                    solutions.append(merged)
                return
            if idx == len(cand):
                return

            v, w, ub = cand[idx]
            max_m = min(ub, left)
            for m in range(max_m, -1, -1):
                if m == 0:
                    distribute(idx + 1, left, rem_tau, rem_gamma, acc_k)
                else:
                    # feasibility for m copies
                    feasible = True
                    for i in range(t):
                        if rem_tau[i][v[i]] < m or rem_gamma[i][w[i]] < m:
                            feasible = False; break
                    if not feasible:
                        continue
                    # apply
                    for i in range(t):
                        rem_tau[i][v[i]]   -= m
                        rem_gamma[i][w[i]] -= m
                    acc_k_new = dict(acc_k)
                    acc_k_new[(t, v, w)] = acc_k_new.get((t, v, w), 0) + m
                    distribute(idx + 1, left - m, rem_tau, rem_gamma, acc_k_new)
                    # undo
                    for i in range(t):
                        rem_tau[i][v[i]]   += m
                        rem_gamma[i][w[i]] += m

        distribute(0, to_place, [row[:] for row in rem_tau], [row[:] for row in rem_gamma], {})
        return tuple(solutions)

    rt0, rg0 = flatten_state(need_tau, need_gamma)
    for sol in dp(0, rt0, rg0):
        yield sol


def contribution_of_kappa(lam: Partition, kappa: Kappa, label: str|None):
    """Convert a κ-map into its polynomial contribution N(κ)*Π dets^m.

    contribution_of_kappa takes one k configuratoin and turns it into the scalar or polynomial factor that contributes to f_tau,gamma.
    It computes the weight and then evaluates the corresponding minor.
    """
    weight = N_of_kappa(lam, kappa)
    prod = Polynomial.one()
    for (t, v, w), m in kappa.items():
        if m <= 0:
            continue
        det_val = MatrixDeterminantPolynomial(v, w, common_label=label)
        prod *= (det_val ** m)
    return weight * prod


def N_of_kappa(lam: Partition, kappa: Kappa) -> int:
    """Compute the multinomial weight N(κ) = Π_t ( (λ_t - λ_{t+1})! / Π κ_t(v,w)! ).
    - N_of_kappa(lam, kappa) computes the multinomial weight N(κ) grouped by minor size t.
    """
    h = lam.height
    num = 1
    den = 1
    by_t: Dict[int, List[int]] = {}
    for (t, _, _), m in kappa.items():
        by_t.setdefault(t, []).append(m)
    for t in range(1, h+1):
        block = lam[t-1] - (lam[t] if t < h else 0)
        num *= math.factorial(block)
        for m in by_t.get(t, []):
            den *= math.factorial(m)
    return num // den