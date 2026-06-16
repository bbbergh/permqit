import unittest

from permqit.algebra.polynomial import Polynomial


class TestPolynomial(unittest.TestCase):
    def test_add_and_mul_polynomials(self):
        # p = 1 + 2x
        p = Polynomial.from_dict({(): 1, ('x',): 2})
        # q = 3 + 4x
        q = Polynomial.from_dict({(): 3, ('x',): 4})

        s = p + q
        assert s.to_dict()[()] == 4
        assert s.to_dict()[('x',)] == 6

        # multiply: (1 + 2x)*(3 + 4x) = 3 + 10x + 8x^2
        r = p * q
        d = r.to_dict()
        assert d[()] == 3
        assert d[('x',)] == 10
        # x^2 monomial represented as ('x','x')
        assert d[('x', 'x')] == 8


    def test_pow_scale_neg_sub_eq(self):
        p = Polynomial.from_dict({(): 2, ('x',): 1})  # 2 + x
        p2 = p ** 2
        # (2 + x)^2 = 4 + 4x + x^2
        d = p2.to_dict()
        assert d[()] == 4
        assert d[('x',)] == 4
        assert d[('x', 'x')] == 1

        s = p.scale(3)
        ds = s.to_dict()
        assert ds[()] == 6
        assert ds[('x',)] == 3

        # negation and subtraction
        n = -p
        nn = (p + n)
        assert not nn  # zero polynomial

        # equality
        assert p == Polynomial.from_dict({(): 2, ('x',): 1})



    def test_multiplication_with_scalar_and_monomial_result(self):
        p = Polynomial.from_dict({('x',): 3})
        # multiply by scalar
        q = p * 2
        assert q.to_dict()[('x',)] == 6

        # scalar times polynomial (rmul)
        r = 4 * p
        assert r.to_dict()[('x',)] == 12

    def test_derivative_simple_univariate(self):
        # p(x) = 1 + 2x + 3x^2  =>  p'(x) = 2 + 6x
        p = Polynomial.from_dict({(): 1, ('x',): 2, ('x', 'x'): 3})
        dp = p.derivative('x')
        d = dp.to_dict()

        # expected polynomial: 2 + 6x
        assert d[()] == 2
        assert d[('x',)] == 6
        # no x^2 term anymore
        assert ('x', 'x') not in d

    def test_derivative_constant_and_zero(self):
        # constant polynomial: derivative = 0
        p_const = Polynomial.from_dict({(): 5})
        dp_const = p_const.derivative('x')
        assert dp_const.to_dict() == {}

        # zero polynomial: derivative = 0
        p_zero = Polynomial.from_dict({})
        dp_zero = p_zero.derivative('x')
        assert dp_zero.to_dict() == {}

    def test_derivative_no_matching_atom(self):
        # p(x) = 3x, derivative w.r.t. y: 0
        p = Polynomial.from_dict({('x',): 3})
        dp = p.derivative('y')
        assert dp.to_dict() == {}

    def test_derivative_multivariate(self):
        # p(x,y) = 2xy + 3x^2 y  with monomials ('x','y') and ('x','x','y')
        # dp/dx = 2y + 6xy
        p = Polynomial.from_dict({('x', 'y'): 2, ('x', 'x', 'y'): 3})
        dp_dx = p.derivative('x')
        d = dp_dx.to_dict()

        # expected monomials: ('y',) with coeff 2 and ('x','y') with coeff 6
        assert d[('y',)] == 2
        assert d[('x', 'y')] == 6
        # no x^2 y term anymore
        assert ('x', 'x', 'y') not in d

    def test_derivative_higher_multiplicity_in_monomial(self):
        # p(x) = 4x^3  =>  p'(x) = 12x^2
        p = Polynomial.from_dict({('x', 'x', 'x'): 4})
        dp = p.derivative('x')
        d = dp.to_dict()

        assert d[('x', 'x')] == 12
        assert ('x',) not in d
        assert () not in d
