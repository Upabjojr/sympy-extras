"""Algebraic number fields and Trager's factorisation against SymPy's
algebraic fields and ``factor_list`` (the oracle)."""
from __future__ import annotations

import random
from fractions import Fraction

from sympy import CRootOf, Poly, Rational, expand, factor_list, sqrt, symbols
from sympy.polys.domains import QQ as SQQ

from sympy_extras._typing import as_expr
from sympy_extras.polys.densearith import dup_mul, dup_pow
from sympy_extras.polys.densetools import dup_monic
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.factortools import dup_ext_factor, dup_ext_norm, dup_ext_sqf_norm

x, t = symbols('x t')

SQRT2 = AlgebraicField([Fraction(1), Fraction(0), Fraction(-2)], sqrt(2))
CBRT2 = AlgebraicField([Fraction(1), Fraction(0), Fraction(0), Fraction(-2)], CRootOf(t**3 - 2, 0))


def _random_element(rng: random.Random, K: AlgebraicField) -> list[Fraction]:
    return K._reduce([Fraction(rng.randint(-4, 4)) for _ in range(K.degree)])


def _random_poly(rng: random.Random, K: AlgebraicField, degree: int) -> list[list[Fraction]]:
    f = [_random_element(rng, K) for _ in range(degree + 1)]
    f[0] = f[0] if f[0] else K.one
    return f


def _to_sympy(f: list[list[Fraction]], K: AlgebraicField) -> Poly:
    expr = as_expr(0)
    n = len(f) - 1
    for i, c in enumerate(f):
        expr = as_expr(expr + K.to_sympy(c)*x**(n - i))
    return Poly(expand(expr), x, domain=SQQ.algebraic_field(K.alpha_expr))


def test_field_arithmetic() -> None:
    rng = random.Random(1)
    for K in (SQRT2, CBRT2):
        field = SQQ.algebraic_field(K.alpha_expr)
        for _ in range(30):
            a, b = _random_element(rng, K), _random_element(rng, K)
            A, B = field.from_sympy(K.to_sympy(a)), field.from_sympy(K.to_sympy(b))
            assert K.to_sympy(K.mul(a, b)) == field.to_sympy(A*B).expand()
            assert K.to_sympy(K.add(a, b)) == field.to_sympy(A + B).expand()
            if b:
                assert K.to_sympy(K.exquo(a, b)) == field.to_sympy(A/B).expand()
                assert K.mul(b, K.invert(b)) == K.one
            # the norm is the product of the conjugates: for sqrt(2),
            # (p + q sqrt 2)(p - q sqrt 2)
            if K is SQRT2 and a:
                p = a[-1]
                q = a[0] if len(a) == 2 else Fraction(0)
                assert K.norm(a) == p*p - 2*q*q
            assert K.from_sympy(K.to_sympy(a)) == a
            # the minimal polynomial vanishes at the element
            m = K.minimal_polynomial(a)
            value = K.zero
            for c in m:
                value = K.add(K.mul(value, a), K.from_rational(c))
            assert value == K.zero
            assert K.degree % max(len(m) - 1, 1) == 0
    assert SQRT2.minimal_polynomial(SQRT2.alpha) == SQRT2.minpoly
    assert SQRT2.minimal_polynomial([Fraction(3)]) == [Fraction(1), Fraction(-3)]


def test_norms() -> None:
    # N(x - sqrt 2) = x**2 - 2
    f = [SQRT2.one, SQRT2.neg(SQRT2.alpha)]
    assert dup_ext_norm(f, SQRT2) == [Fraction(1), Fraction(0), Fraction(-2)]
    # N(x**2 - 2): the product of two copies, not square-free; a shift fixes it
    g = [SQRT2.one, [], SQRT2.from_int(-2)]
    assert dup_ext_norm(g, SQRT2) == [Fraction(1), Fraction(0), Fraction(-4), Fraction(0), Fraction(4)]
    s, shifted, norm = dup_ext_sqf_norm(g, SQRT2)
    assert s != 0 and len(norm) == 5


def test_trager_against_sympy() -> None:
    rng = random.Random(2)
    for K in (SQRT2, CBRT2):
        for _ in range(25):
            f: list[list[Fraction]] = [K.one]
            for _ in range(rng.randint(1, 3)):
                f = dup_mul(f, dup_pow(_random_poly(rng, K, rng.randint(1, 2)), rng.randint(1, 2), K), K)
            lc, factors = dup_ext_factor(f, K)
            product: list[list[Fraction]] = [lc]
            for g, k in factors:
                assert g[0] == K.one
                product = dup_mul(product, dup_pow(g, k, K), K)
            assert product == f, (K, f)
            expected = factor_list(_to_sympy(f, K))[1]
            assert sorted((len(g) - 1, k) for g, k in factors) == sorted((g.degree(), k) for g, k in expected), (K, f)


def test_known_factorisations() -> None:
    # x**2 - 2 splits over QQ(sqrt 2), x**4 - 4 = (x**2 - 2)(x**2 + 2)
    f = [SQRT2.one, [], SQRT2.from_int(-2)]
    lc, factors = dup_ext_factor(f, SQRT2)
    assert lc == SQRT2.one and factors == [([SQRT2.one, SQRT2.neg(SQRT2.alpha)], 1), ([SQRT2.one, SQRT2.alpha], 1)]
    g = dup_mul(f, [SQRT2.one, [], SQRT2.from_int(2)], SQRT2)
    assert [len(h) - 1 for h, _ in dup_ext_factor(g, SQRT2)[1]] == [1, 1, 2]
    # x**3 - 2 over QQ(cbrt 2): a linear and a quadratic factor
    h = [CBRT2.one, [], [], CBRT2.from_int(-2)]
    assert [len(q) - 1 for q, _ in dup_ext_factor(h, CBRT2)[1]] == [1, 2]
    # a non-monic input and a repeated factor
    k = dup_mul([SQRT2.from_int(3)], dup_pow([SQRT2.one, SQRT2.neg(SQRT2.alpha)], 2, SQRT2), SQRT2)
    lc, factors = dup_ext_factor(k, SQRT2)
    assert lc == SQRT2.from_int(3) and factors == [([SQRT2.one, SQRT2.neg(SQRT2.alpha)], 2)]
    assert dup_monic(k, SQRT2)[0] == SQRT2.one
    assert Rational(1, 2) == Fraction(1, 2)
