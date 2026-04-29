"""
Sparse polynomial operations used in isomorphism calculations.
"""
from __future__ import annotations

from numbers import Number
from typing import TypeVar, Generic, Dict, Tuple, Iterable, Any, Optional, cast, Hashable, Sequence, Self

import sympy.combinatorics


AtomT = TypeVar("AtomT", bound=Hashable)
CoeffT = TypeVar("CoeffT", bound=Number)

# Internal convenience type for monomials: Tuple of atoms
MonomialT = Tuple[AtomT, ...]

__all__ = ["Polynomial", "MatrixDeterminantPolynomial"]

class Polynomial(Generic[AtomT, CoeffT]):
    """Sparse multivariate polynomial with monomials as tuples of atoms.

    Representation:
      - self.coeffs: Dict[MonomialT, CoeffT]
    Notes:
      - CoeffT can be int, float, complex, or any numeric-like type supporting +, *.
      - AtomT is the hashable identifier used for polynomial variables (we mainly use tuples of ints) as atoms.
    """

    def __init__(self, coeffs: Optional[Dict[MonomialT, CoeffT]] = None) -> None:
        """Create a Polynomial from an optional mapping monomial -> coefficient.

        Args:
            coeffs: mapping from monomial tuples to coefficients. If omitted or
                None, constructs the zero polynomial.
        """
        self.coeffs: Dict[MonomialT, CoeffT] = {}
        if coeffs:
            for m, c in coeffs.items():
                if c is None:
                    continue
                # normalize monomial to tuple
                m = tuple(sorted(m, key=lambda x: str(x)))
                self.coeffs[m] = c

    # ---- construction helpers ----
    @classmethod
    def one(cls) -> Self:
        """Return the multiplicative identity polynomial (constant 1).

        The coefficient type CoeffT is not enforced at runtime; callers typically
        use Polynomial[object,int].one() when working with integer weights.
        """
        return cls({(): cast(CoeffT, 1)})  # type: ignore[name-defined]

    @classmethod
    def scalar(cls, k: CoeffT) -> Self:
        """Return the constant polynomial with value k."""
        return cls({(): k})

    @classmethod
    def zero(cls) -> Self:
        """Return the zero polynomial."""
        return cls({})

    @classmethod
    def monomial(cls, *atoms: AtomT):
        return cls.from_dict({atoms: cast(CoeffT, 1)})

    @classmethod
    def from_dict(cls, data: Dict[MonomialT, CoeffT]) -> Self:
        """Alias to construct from a plain dict mapping."""
        return cls(data)

    # ---- conversion ----
    def to_dict(self) -> Dict[MonomialT, CoeffT]:
        """Return a shallow copy of the internal coefficients dict."""
        return dict(self.coeffs)

    # ---- arithmetic (explicit methods) ----
    def add(self, other: Self | CoeffT) -> Self:
        """Return self + other as a new Polynomial.

        Args:
            other: Polynomial to add or a scalar coefficient (treated as constant polynomial).
        """
        if isinstance(other, Polynomial):
            other_poly = other
        else:
            # scalar -> constant polynomial
            other_poly = Polynomial.scalar(other)

        out: Dict[MonomialT, CoeffT] = dict(self.coeffs)
        for m, c in other_poly.coeffs.items():
            out[m] = out.get(m, cast(CoeffT, 0)) + c  # type: ignore[operator]
            if out[m] == 0:  # prune exact-zero coefficients
                del out[m]
        return Polynomial(out)

    def mul(self, other: Self | CoeffT) -> Self:
        """Return product self * other as a new Polynomial.

        Multiplication by a dict is not supported; call `Polynomial.from_dict(...)`
        first if you have a raw mapping.
        """
        if isinstance(other, Number):
            return self.scale(other)
        assert isinstance(other, Polynomial), f"Unsupported type for polynomial multiplication: {type(other)}"

        out: Dict[MonomialT, CoeffT] = {}
        for m1, c1 in self.coeffs.items():
            for m2, c2 in other.coeffs.items():
                # canonicalize monomial by stable ordering of atom string reprs
                m: MonomialT = tuple(sorted(m1 + m2, key=lambda x: str(x)))  # type: ignore[arg-type]
                out[m] = out.get(m, cast(CoeffT, 0)) + (c1 * c2)  # type: ignore[operator]
                if out[m] == 0:
                    del out[m]
        return Polynomial(out)

    def scale(self, k: CoeffT) -> Self:
        """Scale all coefficients by scalar k and return a new Polynomial."""
        if k == cast(CoeffT, 1):
            return Polynomial(self.to_dict())
        if k == cast(CoeffT, 0):
            return Polynomial.zero()
        return Polynomial({m: c * k for m, c in self.coeffs.items()})  # type: ignore[operator]

    def pow(self, e: int) -> Self:
        """Return self**e for nonnegative integer exponent using exponentiation by squaring."""
        assert e >= 0
        if e == 0:
            return Polynomial.one()
        if e == 1:
            return self
        res = Polynomial.one()
        base = self
        ee = e
        while ee > 0:
            if ee & 1:
                res = res.mul(base)
            base = base.mul(base)
            ee >>= 1
        return res

    def derivative(self, atom: AtomT) -> Self:
        """
        Returns the derivative of the polynomial with respect to atom as a new Polynomial.
        :param atom: The atom respective to which to take the derivative.
        :return:
        """
        _new = {}
        for mon, coeff in self.coeffs.items():
            if atom not in mon: # If the monomial does not contain the atom
                continue
            count = sum(1 for _atom in mon if _atom == atom) # The power of the atom in the monomial
            idx = mon.index(atom)
            _new_mon = mon[0:idx] + mon[idx+1:] # Remove one power of the atom we are taking the derivative of
            _new[_new_mon] = coeff * count # Multiply by the power

        return type(self)(_new)


    # ---- dunder operators for Python arithmetic ----
    def __add__(self, other: Self | CoeffT) -> Self:
        return self.add(other)

    def __radd__(self, other: Self | CoeffT) -> Self:
        return self.add(other)

    def __mul__(self, other: Self | CoeffT) -> Self:
        return self.mul(other)

    def __rmul__(self, other: Self | CoeffT) -> Self:
        return self.mul(other)

    def __pow__(self, e: int) -> Self:
        return self.pow(e)

    def __neg__(self) -> Self:
        return Polynomial({m: -c for m, c in self.coeffs.items()})  # type: ignore[operator]

    def __sub__(self, other: Self | CoeffT) -> Self:
        return self.add(-other)

    def __rsub__(self, other: Self | CoeffT) -> Self:
        other_p = other if isinstance(other, Polynomial) else Polynomial({(): other})
        return other_p.add(-self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Polynomial):
            other = other if isinstance(other, Polynomial) else Polynomial({(): other})
        return self.coeffs == other.coeffs

    def __bool__(self) -> bool:
        return bool(self.coeffs)

    def __repr__(self) -> str:
        return f"Polynomial({self.coeffs!r})"

    def __str__(self) -> str:
        """Return a human-friendly string representation of the polynomial.

        Formatting rules:
        - Constant term first (if present), then monomials ordered by total degree descending
          (higher-degree terms first) and lexicographically among equal degrees.
        - Monomials compress repeated atoms as powers (e.g. ('x','x') -> x^2).
        - Coefficient 1 is omitted for non-constant monomials; coefficient -1 is shown as "-".
        - Terms are joined with ` + `, negative coefficients displayed with `-`.
        """
        if not self.coeffs:
            return "0"

        from collections import Counter

        def atom_to_str(atom: AtomT) -> str:
            return str(atom)

        # Build terms list (coeff, monomial_str, degree)
        terms = []
        for m, c in self.coeffs.items():
            if m == ():
                terms.append((c, str(c), 0))
                continue
            # count multiplicities
            counts = Counter(m)
            parts = []
            degree = 0
            for atom in sorted(counts.keys(), key=lambda x: str(x)):
                power = counts[atom]
                degree += power
                a_str = atom_to_str(atom)
                parts.append(a_str if power == 1 else f"{a_str}^{power}")
            mono_str = "*".join(parts)
            terms.append((c, mono_str, degree))

        # Sort: constant last? We put highest degree first, constants (degree 0) last
        terms.sort(key=lambda t: (-t[2], str(t[1])))

        out_parts = []
        for coeff, mono_str, deg in terms:
            if deg == 0:
                out_parts.append(str(coeff))
                continue
            # Non-constant monomial
            # Decide coefficient display
            try:
                coeff_num = coeff
            except Exception:
                coeff_num = coeff
            if coeff_num == 1:
                out_parts.append(mono_str)
            elif coeff_num == -1:
                out_parts.append(f"-{mono_str}")
            else:
                out_parts.append(f"{coeff_num}*{mono_str}")

        # Now combine, but ensure constants (if any) are included in correct position
        # Find if there is a standalone constant term in coeffs
        const = None
        if () in self.coeffs:
            const = str(self.coeffs[()])

        # Join all non-constant parts with ' + ', but normalize signs
        if out_parts:
            expr = " + ".join(out_parts)
        else:
            expr = "0"

        # If constant exists and wasn't included in out_parts (degree 0 handled above but might be first), ensure it's present
        if const is not None and str(self.coeffs.get((), "")) not in out_parts:
            # append constant at end
            if expr == "0":
                expr = const
            else:
                expr = expr + " + " + const
        return expr


def MatrixDeterminantPolynomial(row_indices: Sequence[AtomT], column_indices: Sequence[AtomT], *, common_label=None) -> Polynomial[AtomT, int]:
    """Returns the symbolic determinant det[(x_{a_i, b_j})_(ij)] as a polynomial in the x_{a_i, b_j},
    where (a_i)_i = row_indices, (b_j)_j = column_indices.
    """
    ret = Polynomial.zero()

    k = len(row_indices)
    assert k == len(column_indices), f"{row_indices=} and {column_indices=} must have same length"

    # Calculate the determinant by Leibniz's formula
    for perm in sympy.combinatorics.SymmetricGroup(k).elements:  # type: sympy.combinatorics.Permutation
        if common_label is None:
            monomial_labels = zip(row_indices, perm(column_indices))
        else:
            monomial_labels = zip((common_label,)*k, row_indices, perm(column_indices))
        ret += perm.signature() * Polynomial.monomial(*monomial_labels)

    return ret

