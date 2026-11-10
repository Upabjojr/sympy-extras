"""The factorisation over ``ZZ_I`` and ``QQ_I`` in several generators
(:mod:`sympy_extras.polys.gaussianfactor`) against SymPy's ``factor`` and
``factor_list`` (the oracle), compared through ``srepr``; the native path
is called directly so that no fallback to SymPy can hide a failure."""
from __future__ import annotations

import random
from fractions import Fraction

import sympy
from sympy.core.cache import clear_cache

from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy, to_sympy
from sympy_extras.polys.domains.gaussiandomains import QQ_I, GaussianRational
from sympy_extras.polys.gaussianfactor import gaussian_ext_factor, gaussian_factor_list, gaussian_sort_key
from sympy_extras.polys.polytools import _gaussian_factor_list, _symbolic_factor
from sympy_extras.polys.rings import PolyRing

s = sympy
x, y, z = s.symbols('x y z')


def native_factor(e: s.Expr) -> s.Basic:
    """The native ``factor`` (``CoercionFailed`` instead of SymPy)."""
    native: Expr = expr_from_sympy(e)
    return to_sympy(_symbolic_factor(native, (), True))


def check(e: s.Expr) -> None:
    clear_cache()
    expected = s.factor(e)
    got = native_factor(e)
    assert s.srepr(got) == s.srepr(expected), (e, got, expected)
    # the oracle itself: the factorisation multiplies back to the input
    assert s.expand(expected - e) == 0 or s.cancel(expected - e) == 0


def test_examples_from_trigsimp_match_sympy() -> None:
    d = s.Dummy('d')
    a = s.sqrt(2)*s.pi/2
    u = s.sin(s.sin(d*x))
    for e in [s.I*x*u + s.sqrt(u**2 + 1),
              (s.I*x*u + s.sqrt(u**2 + 1))/u,
              -(s.tan(a) - s.I*s.tanh(a))**2/(s.tan(a)*s.tanh(a) - s.I)**2,
              s.expand((s.tan(a) - s.I*s.tanh(a))**2),
              s.expand((s.tan(a)*s.tanh(a) - s.I)**2*z)]:
        check(e)


def test_small_examples_match_sympy() -> None:
    for e in [(x + s.I*y)**2*(x - y), s.expand((x + s.I*y)**2*(x - y)), x**4 - y**4 + 0*s.I, x**2 + 2*s.I*x*y - y**2,
              (x**2 + y**2)/3 + s.I*x*y/2, 2*x**3*y + 2*s.I*x*y**3, (1 + s.I)*x*y + (1 - s.I)*z,
              s.expand((x + y + s.I)*(x - y - 2*s.I)**3*(x*y + z + 1)), s.expand(s.I*x**2*y**3*(x + s.I*y)),
              s.expand((3 + 4*s.I)*(x**2 + y**2)*(x + z)), s.expand((x*y - s.I)**2*(x + 2*s.I*z)/(5 - 5*s.I)),
              (x**2 - 2*s.I*x*y - y**2)/(x**2 + y**2)]:
        check(e)


def _random_factor(rng: random.Random, variables: list[s.Symbol], rational: bool) -> s.Expr:
    terms: list[s.Expr] = []
    for _ in range(rng.randint(2, 3)):
        c = rng.randint(-3, 3) + rng.randint(-3, 3)*s.I
        if rational and rng.random() < 0.3:
            c = c / rng.choice([2, 3])
        monomial = s.Mul(*[v**rng.randint(0, 2) for v in variables])
        terms.append(c*monomial)
    return s.Add(*terms)


def random_products(seed: int, count: int) -> list[s.Expr]:
    """Products of small random Gaussian polynomials in two or three
    variables (the first factor possibly squared; three factors in two
    variables only, to keep SymPy's own factorisation fast)."""
    rng = random.Random(seed)
    result: list[s.Expr] = []
    while len(result) < count:
        variables = rng.choice([[x, y], [x, y], [x, y, z]])
        rational = rng.random() < 0.3
        e: s.Expr = _random_factor(rng, variables, rational)**rng.choice([1, 1, 2])
        for _ in range(rng.randint(0, 2 if len(variables) == 2 else 1)):
            e = e * _random_factor(rng, variables, rational)
        e = s.expand(e)
        if e.has(s.I) and len(e.free_symbols) >= 2:
            result.append(e)
    return result


def test_random_products_match_sympy() -> None:
    for e in random_products(2026, 70):
        check(e)


def test_random_irreducible_match_sympy() -> None:
    rng = random.Random(11)
    checked = 0
    for _ in range(40):
        variables = rng.choice([[x, y], [x, y, z]])
        e = s.expand(_random_factor(rng, variables, False) + rng.randint(1, 3)*s.I*x*y)
        if not e.has(s.I) or len(e.free_symbols) < 2:
            continue
        check(e)
        checked += 1
    assert checked >= 30


def test_factor_list_order_matches_sympy() -> None:
    # The univariate factors were sorted with Gaussian numbers compared by
    # their real parts first: (x + 1 + 3*I, x + 2) instead of SymPy's
    # (x + 2, x + 1 + 3*I), which compares the imaginary parts first. The
    # order is that of Poly.factor_list (dmp_factor_list).
    rng = random.Random(3)
    cases: list[s.Expr] = [s.expand((x + 1 + 3*s.I)*(x + 2)), s.expand((x + 1 - s.I)*(x + 2*s.I)*(y - 2)),
                           s.expand((x + y + 2*s.I)*(x + y + 1 + s.I))]
    for _ in range(20):
        cases.append(s.expand(_random_factor(rng, [x, y], False)*_random_factor(rng, [x, y], False)))
    for e in cases:
        if not e.has(s.I) or not e.free_symbols:
            continue
        clear_cache()
        c, poly_factors = s.Poly(e).factor_list()
        factors = [(g.as_expr(), k) for g, k in poly_factors]
        nc, native = _gaussian_factor_list(expr_from_sympy(e))
        assert s.srepr(to_sympy(nc)) == s.srepr(c), e
        assert [s.srepr(to_sympy(g)) for g, _ in native] == [s.srepr(g) for g, _ in factors], e
        assert [k for _, k in native] == [k for _, k in factors], e


def test_gaussian_ext_factor_is_monic_and_multiplies_back() -> None:
    R = PolyRing(QQ_I, ['x', 'y'])
    X, Y = R.gens
    i = R.ground(GaussianRational(Fraction(0), Fraction(1)))
    f = (X*X + Y*Y)*(X*Y - i)*(X*Y - i)
    lc, factors = gaussian_ext_factor(f)
    assert lc == QQ_I.one
    assert all(g.is_monic for g, _ in factors)
    product = R.one
    for g, k in factors:
        product = product * g**k
    assert product == f
    assert sorted(k for _, k in factors) == [1, 1, 2]


def test_gaussian_factor_list_contents() -> None:
    R = PolyRing(QQ_I, ['x', 'y'])
    X, Y = R.gens
    c = R.ground(GaussianRational(Fraction(2), Fraction(2)))
    coefficient, factors = gaussian_factor_list(c*X*(X + Y)*(X*X + Y*Y), True)
    assert QQ_I.to_expr(coefficient) == expr_from_sympy(2 + 2*s.I)
    assert [str(g.to_expr()) for g, _ in factors] == ['x', 'x - I*y', 'x + y', 'x + I*y']  # SymPy's order
    assert factors == sorted(factors, key=gaussian_sort_key)
