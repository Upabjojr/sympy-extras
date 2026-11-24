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
    # ((n + 1)/(k + 1) for this one was 0/0 at n = k = -1, where the quotient is binomial(0, 0)/binomial(-1, 0) = 1:
    # it stays for integers of any sign, see test_combsimp_where_a_binomial_is_a_limit_of_gamma_functions)
    assert combsimp(binomial(n + 1, n - k)/binomial(n, n - k)) == binomial(n + 1, n - k)/binomial(n, n - k)
    assert combsimp(binomial(m + 1, m - k)/binomial(m, m - k)) == (m + 1)/(k + 1)


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
    assert combsimp(two[0]) == two[0]
    # (the second one stayed: it is (k + 1)/(n + 1) outside n = -1, where it is 0 at k = -1 and not finite elsewhere)
    second = combsimp(two[1])
    assert isinstance(second, Piecewise) and second.args[0].args[1] == eq(n, -1) and second.args[1].args[0] == (k + 1)/(n + 1)
    assert second.subs(n, -1).subs(k, -1) == 0 == two[1].subs(n, -1).subs(k, -1)
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
    # a lower index which is not an integer has no such point (zeilberger_sum of
    # binomial(n, k)*binomial(n, k + a) lost its closed form)
    assert (combsimp(binomial(n, x + 1)/binomial(n, x)) - (n - x)/(x + 1)).cancel() == 0


def test_combsimp_where_a_binomial_is_a_limit_of_gamma_functions() -> None:
    # combsimp(binomial(n, k)*factorial(k)) was factorial(n)/factorial(n - k) for integers of any sign, zoo/zoo
    # for every negative n, where the product is n*(n - 1)*...*(n - k + 1) (binomial(-2, 1)*factorial(1) is -2);
    # combsimp(binomial(n, k)/binomial(n - 1, k)) was n/(n - k) for nonnegative integers too, 0/0 at n = k = 0,
    # where the quotient is binomial(0, 0)/binomial(-1, 0) = 1. gammasimp gave the same in gamma functions
    from sympy_extras.core import eq
    from sympy_extras.functions.elementary.piecewise import Piecewise
    from sympy_extras.simplify.gammasimp import gammasimp
    assert binomial(-2, 1)*factorial(1) == -2 and binomial(0, 0)/binomial(-1, 0) == 1
    product, quotient = binomial(n, k)*factorial(k), binomial(n, k)/binomial(n - 1, k)
    assert combsimp(product) == product and gammasimp(product) == product
    assert combsimp(quotient) == quotient and gammasimp(quotient) == quotient
    # nonnegative integers: the product has no such point, the quotient the value n = 0 of one symbol
    p, q = Symbol('p', integer=True, nonnegative=True), Symbol('q', integer=True, nonnegative=True)
    assert combsimp(binomial(p, q)*factorial(q)) == factorial(p)/factorial(p - q)
    found = combsimp(binomial(p, q)/binomial(p - 1, q))
    assert isinstance(found, Piecewise) and found.args[0].args[1] == eq(p, 0)
    assert found.subs(p, 3).subs(q, 1) == Rational(3, 2) and found.subs(p, 0).subs(q, 0) == 1
    assert isinstance(gammasimp(binomial(p, q)/binomial(p - 1, q)), Piecewise)
    for e, symbols in ((product, (n, k)), (quotient, (n, k)), (binomial(p, q)*factorial(q), (p, q)),
                       (binomial(p, q)/binomial(p - 1, q), (p, q)), (binomial(p, q)/binomial(p - 1, q - 1), (p, q)),
                       (binomial(n + 2, k + 2)/binomial(n, k), (n, k)), (binomial(n, k)/binomial(n + 1, k + 1), (n, k)),
                       (binomial(n + 1, k)/binomial(n, k), (n, k)), (binomial(n + 1, n - k)/binomial(n, n - k), (n, k))):
        for simplified in (combsimp(e), gammasimp(e)):
            low = 0 if symbols[0].is_nonnegative else -5
            for upper in range(low, 6):
                for lower in range(low, 6):
                    given = e.subs(symbols[0], upper).subs(symbols[1], lower)
                    if given.is_finite is True:
                        assert simplified.subs(symbols[0], upper).subs(symbols[1], lower) == given, (e, upper, lower)
    # a lower index which is a number is a polynomial in the upper one, and a symbol in the upper indices only
    # makes the quotient a rational function of it on every line: these are simplified as before
    assert combsimp(binomial(n, 2)) == n*(n - 1)/2 and combsimp(binomial(n + 1, k + 1)/binomial(n, k)) == (n + 1)/(k + 1)
    assert combsimp(binomial(n + 2, k + 2)/binomial(n, k)) == (n + 1)*(n + 2)/((k + 1)*(k + 2))
    assert combsimp(binomial(n, k)*factorial(k)*factorial(n - k)) == factorial(n)
    # symbols which are not integers are generic, as before
    x, y = Symbol('x'), Symbol('y')
    assert (combsimp(binomial(x, y)/binomial(x - 1, y)) - x/(x - y)).cancel() == 0


def test_combsimp_at_the_points_a_random_search_found() -> None:
    # found by random products of binomials and factorials compared at the integer points:
    # binomial(k - 1, k) became 0 (it is binomial(-1, 0) = 1 at k = 0); binomial(n - 2, 2*n + 1)*binomial(2*n + 1, 3)
    # became factorial(n + 2)/(6*factorial(1 - n)*factorial(2*n - 2)), 1 at n = 1 where the product is
    # binomial(-1, 3) = -1 (the limit of the gamma functions with both indices moving has another sign);
    # binomial(0, n - k)/binomial(0, k - 2) became a product with factorial(-k)*factorial(k - 1), nan at k = 2;
    # binomial(2, k + n)*binomial(2*n - 1, n - 2)*factorial(n - k + 1) a quotient of factorials at their poles,
    # nan at n = k = -4 where the product is 0
    from sympy_extras.simplify.gammasimp import gammasimp
    p = Symbol('p', integer=True, nonnegative=True)
    cases = [(binomial(k - 1, k), (n, k)), (binomial(p - 2, 2*p + 1)*binomial(2*p + 1, 3), (p, k)),
             (binomial(0, n - k)/binomial(0, k - 2), (n, k)),
             (binomial(2, k + n)*binomial(2*n - 1, n - 2)*factorial(n - k + 1), (n, k)),
             (1/binomial(n - 2, n + 1), (n, k)), (binomial(n - 5, 2*n), (n, k))]
    for e, (first, second) in cases:
        for simplified in (combsimp(e), gammasimp(e)):
            for upper in range(0 if first.is_nonnegative else -5, 6):
                for lower in range(-5, 6):
                    given = e.subs(first, upper).subs(second, lower)
                    if given.is_finite is True and e.subs(second, lower).subs(first, upper) == given:
                        assert simplified.subs(first, upper).subs(second, lower) == given, (e, upper, lower)
    assert binomial(-4, 2) == 10 and combsimp(binomial(n - 5, 2*n)).subs(n, 1) == 10
