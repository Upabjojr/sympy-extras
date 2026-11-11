"""The native isolation of real roots on objects of the core, checked
against mpmath's root finder and hand computations (the SymPy face is
tested in test_isolation.py)."""
from __future__ import annotations

from fractions import Fraction

import mpmath

from sympy_extras.core import Expr, Integer, Rational, Symbol, oo, sqrt, E
from sympy_extras.core.logic import true
from sympy_extras.core.numbers import NegativeInfinity
from sympy_extras.functions import cos, exp, log, sin, tan
from sympy_extras.sets import interval
from sympy_extras.solvers.isolation import (TranscendentalRoot, compare_points, isolate_real_roots, point_between,
    real_roots_of)

x = Symbol('x')


def _value(r: Expr) -> mpmath.mpf:
    return mpmath.mpf(str(r.evalf(30)))


def _close(r: Expr, near: float, f: object) -> bool:
    """``r`` agrees with mpmath's secant root of ``f`` started at ``near``."""
    assert callable(f)
    return bool(abs(_value(r) - mpmath.findroot(f, near)) < mpmath.mpf(10)**-12)


def test_root_object() -> None:
    r = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert r.is_real and r.is_positive and r.is_number and not r.free_symbols
    assert _close(r, 1.1, lambda t: mpmath.exp(t) - t - 2)
    assert r.refined(4).interval == interval(Rational(9, 8), Rational(19, 16))
    assert r.lower == 1 and r.upper == 2 and r.lower_fraction == Fraction(1)
    assert r.subs(x, 3) == r and r == TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert (r > 1) is true and (r < 2) is true
    assert (r + 1).is_positive is True
    assert str(r) == 'TranscendentalRoot(-x + exp(x) - 2, x, 1, 2)'
    lo, hi = r.bisect()
    assert (lo, hi) == (1, Rational(3, 2))
    for bad in [(exp(x) - x - 2, 2, 3), (exp(x) - x - 2, 2, 1), (exp(x) - Symbol('a'), 0, 1)]:
        try:
            TranscendentalRoot(bad[0], x, bad[1], bad[2])
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_isolated_roots() -> None:
    roots = isolate_real_roots(exp(x) - x - 2, x)
    assert roots is not None and len(roots) == 2
    assert _close(roots[0], -1.8, lambda t: mpmath.exp(t) - t - 2)
    assert _close(roots[1], 1.1, lambda t: mpmath.exp(t) - t - 2)
    roots = isolate_real_roots(sin(x) - x/2, x)
    assert roots is not None and len(roots) == 3 and roots[1] == 0
    assert _close(roots[2], 1.9, lambda t: mpmath.sin(t) - t/2)
    assert isolate_real_roots(log(x) - 1, x, interval(0, oo, True, True)) == [E]
    assert isolate_real_roots(x**2 + cos(x) - 1, x) == [0]
    assert isolate_real_roots(tan(x) - x, x, interval(0, 4)) is None
    assert isolate_real_roots(exp(x) + 1, x) == []
    found = real_roots_of(x - cos(x), x)
    assert found is not None and len(found.args) == 1


def test_polynomials() -> None:
    # SymPy's real_roots: radicals for the trivial factors, with multiplicities
    assert isolate_real_roots(x**4 - 5*x**2 + 6, x) == [-sqrt(3), -sqrt(2), sqrt(2), sqrt(3)]
    assert isolate_real_roots((x - 1)**2*(x + 2), x) == [-2, 1, 1]
    assert isolate_real_roots(2*x**2 - 3*x - 1, x) == [Rational(3, 4) - sqrt(17)/4, Rational(3, 4) + sqrt(17)/4]
    assert isolate_real_roots(x**3 - 2, x) == [Integer(2)**Rational(1, 3)]
    # the roots of an irreducible quintic stay root objects of SymPy (through the face)
    roots = isolate_real_roots(x**5 - x - 1, x)
    assert roots is not None and len(roots) == 1
    assert _close(roots[0], 1.2, lambda t: t**5 - t - 1)


def test_points() -> None:
    r = TranscendentalRoot(x - cos(x), x, 0, 1)
    s = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert compare_points(r, s) == -1 and compare_points(s, r) == 1 and compare_points(r, Integer(0)) == 1
    assert compare_points(s, sqrt(2)) == -1 and compare_points(s, oo) == -1
    between = point_between(r, s)
    assert between is not None and compare_points(r, between) == -1 and compare_points(between, s) == -1
    below = point_between(NegativeInfinity(), r)
    assert below is not None and compare_points(below, r) == -1
    assert point_between(sqrt(2), s) is None
