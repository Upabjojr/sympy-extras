"""Regression tests of the second pass over the reports against SymPy
(sympy/sympy#31275, #31017) in the simplification: ``combsimp`` of
mirrored binomials and the Meijer G-functions of the modified Bessel
function of the second kind."""
from __future__ import annotations

import mpmath

from sympy_extras.core import I, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import MeijerG, besselk, binomial, factorial, meijerg, sqrt
from sympy_extras.simplify.gammasimp import combsimp
from sympy_extras.simplify.hyperexpand import hyperexpand

n = Symbol('n', integer=True)
k = Symbol('k', integer=True)
z = Symbol('z')


def test_combsimp_does_not_mirror_a_binomial_of_a_negative_upper_index() -> None:
    # (sympy/sympy#31275 V225, sympy/sympy#14612) combsimp(binomial(n, n - k)) was binomial(n, k) for
    # integers of any sign: binomial(-3, -4) is 0 (a negative lower index) and binomial(-3, 1) is -3
    assert binomial(-3, -4) == 0 and binomial(-3, 1) == -3
    assert combsimp(binomial(n, n - k)) == binomial(n, n - k)
    assert combsimp(binomial(n, n - k - 1)) == binomial(n, n - k - 1)
    # nor are two mirrored binomials taken for one
    assert combsimp(binomial(n, n - k)/binomial(n, k)) == binomial(n, n - k)/binomial(n, k)
    assert combsimp(binomial(n, n - k) + binomial(n, k)) == binomial(n, n - k) + binomial(n, k)
    for e in [binomial(n, n - k), binomial(n, n - k)/binomial(n, k) + 1, binomial(n, n - k) + binomial(n, k)]:
        simplified = combsimp(e)
        for upper in range(-4, 5):
            for lower in range(-4, 5):
                given = e.subs(n, upper).subs(k, lower)
                if given.is_finite is True:
                    assert simplified.subs(n, upper).subs(k, lower) == given, (e, upper, lower)
    # a nonnegative upper index: the symmetry holds, and it is used
    m = Symbol('m', integer=True, nonnegative=True)
    assert combsimp(binomial(m, m - k)) == binomial(m, k)
    assert combsimp(binomial(m, m - k)/binomial(m, k)) == 1
    # the simplifications which mirror nothing are the same
    assert combsimp(binomial(n + 1, k + 1)/binomial(n, k)) == (n + 1)/(k + 1)
    assert (combsimp(binomial(2*n + 2, n + 1)/binomial(2*n, n)) - 2*(2*n + 1)/(n + 1)).cancel() == 0
    assert combsimp(factorial(n)/(factorial(k)*factorial(n - k))) == binomial(n, k)
    assert combsimp(binomial(n + 1, n - k)/binomial(n, n - k)) == (n + 1)/(k + 1)


def _g(bm: list[Expr], point: complex) -> complex:
    return complex(mpmath.meijerg([[], []], [[complex(b.evalf(20)) for b in bm], []], point))


def test_hyperexpand_of_the_g_function_of_besselk_of_integer_order() -> None:
    # (sympy/sympy#31017) hyperexpand(meijerg([], [], [0, 0], [], z)) stayed (SymPy's too): the poles of
    # the integrand are double, Slater's theorem has no formula; it is 2*besselk(0, 2*sqrt(z))
    assert hyperexpand(meijerg([], [], [0, 0], [], z)) == 2*besselk(0, 2*sqrt(z))
    assert hyperexpand(meijerg([], [], [0, 1], [], z)) == 2*sqrt(z)*besselk(1, 2*sqrt(z))
    a = Symbol('a')
    assert hyperexpand(meijerg([], [], [a, a], [], z)) == 2*z**a*besselk(0, 2*sqrt(z))
    third = Rational(1, 3)
    cases: list[list[Expr]] = [[Rational(0), Rational(0)], [Rational(0), Rational(1)], [Rational(1), Rational(0)],
                               [Rational(2), Rational(0)], [Rational(-1), Rational(2)], [third, third + 2],
                               [Rational(1, 2), Rational(1, 2)]]
    for bm in cases:
        found = hyperexpand(meijerg([], [], bm, [], z))
        assert not found.has(MeijerG), bm
        for point, value in ((0.7 + 0j, Rational(7, 10)), (2.3 + 0.4j, Rational(23, 10) + 2*I/5),
                             (-0.5 + 1.5j, -Rational(1, 2) + 3*I/2)):
            got = complex(found.subs(z, value).evalf(25))
            want = _g(bm, point)
            assert abs(got - want) < 1e-10*max(1.0, abs(want)), (bm, point, got, want)
    # the functions of noninteger differences are Slater's, as before
    half = hyperexpand(meijerg([], [], [0, Rational(1, 2)], [], z))
    assert abs(complex(half.subs(z, Rational(7, 10)).evalf(25)) - _g([Rational(0), Rational(1, 2)], 0.7)) < 1e-12


def test_combsimp_of_binomials_where_the_gamma_quotient_fails() -> None:
    # combsimp(binomial(2*n + 1, n)/binomial(2*n + 1, n + 1)) was 1 for an integer n of any sign (SymPy's
    # too): at n = -1 it is binomial(-1, -1)/binomial(-1, 0) = 0/1. binomial(n, k) is 0 for k <= n < 0 and
    # the quotient of gamma functions has a nonzero limit there
    from sympy_extras.core import eq
    from sympy_extras.functions.elementary.piecewise import Piecewise
    assert binomial(-1, -1) == 0 and binomial(-1, 0) == 1
    quotient = binomial(2*n + 1, n)/binomial(2*n + 1, n + 1)
    found = combsimp(quotient)
    assert isinstance(found, Piecewise) and found.subs(n, -1) == 0 and found.subs(n, 3) == 1
    assert found.args[0].args[1] == eq(n, -1)
    cases = [quotient, binomial(2*n + 1, n) + binomial(2*n + 1, n + 1), quotient**2,
             binomial(2*n + 3, n + 1)/binomial(2*n + 3, n + 2), binomial(4*n + 6, 2*n)/binomial(4*n + 6, 2*n + 6),
             1/(binomial(2*n + 1, n) - binomial(2*n + 1, n + 1) + 2), binomial(n, 2)/binomial(n, 3),
             binomial(2*n, n)/binomial(2*n, n + 1), binomial(2*n + 1, n)/binomial(2*n, n)]
    for e in cases:
        simplified = combsimp(e)
        assert simplified != e, e
        for value in range(-6, 7):
            given = e.subs(n, value)
            if given.is_finite is True:
                assert simplified.subs(n, value) == given, (e, value)
    # in two symbols the points are not isolated: no rewriting unless the expression is undefined there
    two = [binomial(n, k)/binomial(n, k + 1), binomial(n, k)/binomial(n + 1, k + 1), binomial(n, k + 1)/binomial(n, k),
           binomial(n + 1, k + 1)/binomial(n, k)]
    assert combsimp(two[0]) == two[0] and combsimp(two[1]) == two[1]
    assert (combsimp(two[2]) - (n - k)/(k + 1)).cancel() == 0 and combsimp(two[3]) == (n + 1)/(k + 1)
    for e in two:
        simplified = combsimp(e)
        for upper in range(-5, 6):
            for lower in range(-5, 6):
                given = e.subs(n, upper).subs(k, lower)
                if given.is_finite is True:
                    assert simplified.subs(n, upper).subs(k, lower) == given, (e, upper, lower)
    # the assumptions prove that there is no such point
    m = Symbol('m', integer=True, nonnegative=True)
    assert combsimp(binomial(2*m + 1, m)/binomial(2*m + 1, m + 1)) == 1
    assert (combsimp(binomial(m, k)/binomial(m, k + 1)) - (k + 1)/(m - k)).cancel() == 0
    x = Symbol('x')
    assert combsimp(binomial(2*x + 1, x)/binomial(2*x + 1, x + 1)) == 1
