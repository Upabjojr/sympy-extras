"""Guessing against known recurrences and generating functions (facts of
the literature and of the OEIS) and against random equations."""
from __future__ import annotations

import random
from fractions import Fraction
from math import comb, factorial

from sympy.testing.pytest import raises

from sympy_extras.concrete.guess import (GuessedRecurrence, guess_algebraic_equation, guess_closed_form,
                                         guess_differential_equation, guess_recurrence)
from sympy_extras.core import Function, Integer, Symbol
from sympy_extras.core.function import expand

n = Symbol('n')
x = Symbol('x')


def _catalan(count: int) -> list[int]:
    return [comb(2*k, k)//(k + 1) for k in range(count)]


def _motzkin(count: int) -> list[int]:
    # (n + 2) M(n) = (2 n + 1) M(n - 1) + (3 n - 3) M(n - 2)
    values = [1, 1]
    for k in range(2, count):
        values.append(((2*k + 1)*values[k - 1] + (3*k - 3)*values[k - 2])//(k + 2))
    return values[:count]


def _apery(count: int) -> list[int]:
    # sum_k binomial(n, k)**2 binomial(n + k, k)**2
    return [sum(comb(m, k)**2*comb(m + k, k)**2 for k in range(m + 1)) for m in range(count)]


def test_known_recurrences() -> None:
    # Fibonacci
    r = guess_recurrence([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    assert r is not None and r.coefficients == [[-1], [-1], [1]]
    # Catalan: (n + 2) C(n + 1) = (4 n + 2) C(n)
    r = guess_recurrence(_catalan(12))
    assert r is not None and r.coefficients == [[-4, -2], [1, 2]]
    # Motzkin: (n + 4) M(n + 2) = (2 n + 5) M(n + 1) + (3 n + 3) M(n)
    r = guess_recurrence(_motzkin(16))
    assert r is not None and r.coefficients == [[-3, -3], [-2, -5], [1, 4]]
    # Apery's recurrence for zeta(3) (van der Poorten, 1979):
    # (n + 2)**3 a(n + 2) = (34 n**3 + 153 n**2 + 231 n + 117) a(n + 1) - (n + 1)**3 a(n)
    r = guess_recurrence(_apery(20))
    assert r is not None
    assert r.coefficients == [[1, 3, 3, 1], [-34, -153, -231, -117], [1, 6, 12, 8]]
    # the harmonic numbers
    harmonic = [sum((Fraction(1, j) for j in range(1, k + 1)), Fraction(0)) for k in range(14)]
    r = guess_recurrence(harmonic)
    assert r is not None and r.coefficients == [[1, 1], [-2, -3], [1, 2]]
    # no recurrence within reach for the primes, and none without terms
    assert guess_recurrence([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]) is None
    assert guess_recurrence([]) is None
    assert guess_recurrence([1, 2, 3]) is None


def test_recurrence_object() -> None:
    catalan = _catalan(14)
    r = guess_recurrence(catalan)
    assert r is not None
    assert (r.order, r.degree) == (1, 1)
    assert r.holds(catalan) and not r.holds(catalan[:-1] + [0])
    assert r.extend(catalan[:1], 14) == catalan
    assert r.extend(catalan, 5) == catalan
    a = Function('a')
    assert expand(r.as_expr(a, n) - ((n + 2)*a(n + 1) - (4*n + 2)*a(n))) == Integer(0)
    assert repr(r) == 'GuessedRecurrence([[-4, -2], [1, 2]])'
    # n a(n + 1) = a(n) does not determine a(1)
    stuck = GuessedRecurrence([[-1], [1, 0]])
    raises(ValueError, lambda: stuck.extend([1], 3))
    raises(ValueError, lambda: r.extend([], 3))
    # the zero sequence satisfies a(n) = 0
    zero = guess_recurrence([0]*8)
    assert zero is not None and zero.coefficients == [[1]]


def test_bounds_and_margin() -> None:
    catalan = _catalan(14)
    # no recurrence with constant coefficients
    assert guess_recurrence(catalan, max_degree=0) is None
    assert guess_recurrence(catalan, max_order=0) is None
    r = guess_recurrence(catalan, max_order=1, max_degree=1)
    assert r is not None and r.order == 1
    # order 1, degree 1: four unknowns, hence 4 + 1 + margin terms
    assert guess_recurrence(catalan[:7]) is None
    assert guess_recurrence(catalan[:8]) is not None
    assert guess_recurrence(catalan[:6], margin=1) is not None
    raises(ValueError, lambda: guess_recurrence(catalan, margin=0))


def test_random_recurrences_are_recovered() -> None:
    rng = random.Random(41)
    checked = 0
    for _ in range(40):
        order = rng.randint(1, 3)
        degree = rng.randint(0, 2)
        coefficients = [[rng.randint(-4, 4) for _ in range(degree + 1)] for _ in range(order + 1)]
        # a leading coefficient without non-negative integer root
        coefficients[order] = [rng.randint(1, 4) for _ in range(degree + 1)]
        original = GuessedRecurrence(coefficients)
        initial = [Fraction(rng.randint(-5, 5)) for _ in range(order)]
        count = (order + 1)*(degree + 1) + order + 6
        terms = original.extend(initial, count)
        guessed = guess_recurrence(terms)
        assert guessed is not None
        # an equation on the same terms, with no more coefficients
        assert guessed.holds(terms)
        assert (guessed.order + 1)*(guessed.degree + 1) <= (order + 1)*(degree + 1)
        checked += 1
    assert checked == 40


def test_differential_equations() -> None:
    f = Function('f')
    # exp(x): f' = f
    e = guess_differential_equation([Fraction(1, factorial(k)) for k in range(10)])
    assert e is not None and e.coefficients == [[-1], [1]] and e.inhomogeneous == []
    # 1/sqrt(1 - 4 x): (1 - 4 x) f' = 2 f
    e = guess_differential_equation([comb(2*k, k) for k in range(10)])
    assert e is not None and e.coefficients == [[2], [4, -1]]
    assert (e.order, e.degree) == (1, 1)
    # the Catalan generating function: x (1 - 4 x) f' + (1 - 2 x) f = 1
    e = guess_differential_equation(_catalan(14))
    assert e is not None and e.coefficients == [[2, -1], [4, -1, 0]] and e.inhomogeneous == [1]
    # sum x**n/n!**2 (the Bessel function I_0(2 sqrt(x))): x f'' + f' - f = 0
    e = guess_differential_equation([Fraction(1, factorial(k)**2) for k in range(14)])
    assert e is not None and e.coefficients == [[-1], [1], [1, 0]]
    assert expand(e.as_expr(f, x) - (x*f(x).diff(x, 2) + f(x).diff(x) - f(x))) == Integer(0)
    # sum n! x**n: x**2 f' + (x - 1) f + 1 = 0
    e = guess_differential_equation([factorial(k) for k in range(14)])
    assert e is not None and e.coefficients == [[1, -1], [1, 0, 0]] and e.inhomogeneous == [1]
    assert guess_differential_equation([1, 2]) is None
    assert repr(e) == 'GuessedDifferentialEquation([[1, -1], [1, 0, 0]], [1])'


def test_algebraic_equations() -> None:
    f = Symbol('f')
    # Catalan: x f**2 - f + 1 = 0
    e = guess_algebraic_equation(_catalan(12))
    assert e is not None and e.coefficients == [[1], [-1], [1, 0]]
    assert (e.order, e.degree) == (2, 1)
    # Motzkin: x**2 f**2 + (x - 1) f + 1 = 0
    e = guess_algebraic_equation(_motzkin(14))
    assert e is not None and e.coefficients == [[1], [1, -1], [1, 0, 0]]
    assert expand(e.as_expr(x, f) - (x**2*f**2 + (x - 1)*f + 1)) == Integer(0)
    # Fibonacci: the rational function x/(1 - x - x**2)
    e = guess_algebraic_equation([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    assert e is not None and e.coefficients == [[1, 0], [1, 1, -1]]
    # large Schroeder numbers: x f**2 + (x - 1) f + 1 = 0
    e = guess_algebraic_equation([1, 2, 6, 22, 90, 394, 1806, 8558, 41586, 206098, 1037718, 5293446])
    assert e is not None and e.coefficients == [[1], [1, -1], [1, 0]]
    # exp(x) is not algebraic
    assert guess_algebraic_equation([Fraction(1, factorial(k)) for k in range(16)]) is None
    assert repr(e) == 'GuessedAlgebraicEquation([[1], [1, -1], [1, 0]])'


def test_random_rational_generating_functions() -> None:
    rng = random.Random(42)
    for _ in range(25):
        dq = rng.randint(1, 3)
        dp = rng.randint(0, 3)
        q = [1] + [rng.randint(-3, 3) for _ in range(dq)]          # q(0) = 1, lowest degree first
        p = [rng.randint(-3, 3) for _ in range(dp + 1)]
        count = 2*(max(dp, dq) + 1) + 6
        # the series of p/q
        series: list[Fraction] = []
        for m in range(count):
            value = Fraction(p[m] if m < len(p) else 0)
            for j in range(1, min(m, dq) + 1):
                value -= q[j]*series[m - j]
            series.append(value)
        e = guess_algebraic_equation(series, max_order=1)
        assert e is not None and e.order == 1
        # q_1 f + q_0 = 0 on the series
        q0 = e.coefficients[0][::-1]
        q1 = e.coefficients[1][::-1]
        for m in range(count):
            total = Fraction(q0[m] if m < len(q0) else 0)
            for j, c in enumerate(q1):
                if j <= m:
                    total += c*series[m - j]
            assert total == 0


def test_closed_forms() -> None:
    def values(expression_terms: list[Fraction]) -> None:
        closed = guess_closed_form(expression_terms, n)
        assert closed is not None
        for k, value in enumerate(expression_terms):
            assert expand(closed.subs(n, Integer(k)).doit()) == Integer(value.numerator)/Integer(value.denominator)

    values([Fraction((k + 1)**2) for k in range(10)])
    values([Fraction(2**k - 1) for k in range(10)])
    values([Fraction(comb(2*k, k)) for k in range(10)])
    values([Fraction(c) for c in _catalan(12)])
    values([Fraction(factorial(k)) for k in range(10)])
    # the harmonic numbers and the derangements: d'Alembertian, not hypergeometric
    values([sum((Fraction(1, j) for j in range(1, k + 1)), Fraction(0)) for k in range(14)])
    values([Fraction(c) for c in (1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570)])
    # Fibonacci: Binet's formula, compared after expansion
    values([Fraction(c) for c in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89)])
    assert guess_closed_form([0]*8, n) == Integer(0)
    assert guess_closed_form([3]*8, n) == Integer(3)
    assert guess_closed_form([1, 4, 9, 16], n) is None


def test_initial_values_after_the_zeros_of_the_leading_coefficient() -> None:
    # n/(n + 1) satisfies n (n + 2) a(n + 1) = (n + 1)**2 a(n), whose
    # leading coefficient vanishes at n = 0: with the initial value a(0) = 0
    # the solver returned 0, and no closed form was found
    terms = [Fraction(k, k + 1) for k in range(12)]
    closed = guess_closed_form(terms, n)
    assert closed is not None
    assert [closed.subs(n, Integer(k)) for k in range(1, 6)] == [Integer(k)/Integer(k + 1) for k in range(1, 6)]
