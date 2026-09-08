from __future__ import annotations

import mpmath

from sympy import (Symbol, exp, cos, sin, log, tan, sqrt, Interval, S, Rational, E, pi, oo, FiniteSet, Eq, Ne,
    LambertW, Union, N, Float)
from sympy.core.expr import Expr
from sympy.testing.pytest import raises

from sympy_extras.solvers.isolation import (TranscendentalRoot, isolate_real_roots, real_roots_of, point_between,
    compare_points)
from sympy_extras.assumptions import solve
from sympy_extras._testing import untyped

x = Symbol('x')


def _oracle(f: Expr, near: float) -> Float:
    """An independent numerical root, by mpmath's secant method."""
    from sympy import lambdify
    g = lambdify(x, f, 'mpmath')
    return Float(mpmath.findroot(g, near), 15)


def _close(root: Expr, value: Float) -> bool:
    return abs(N(root, 15) - value) < 1e-12


def test_transcendental_root() -> None:
    r = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert r.is_real and r.is_positive and r.is_number and not r.free_symbols
    assert _close(r, _oracle(exp(x) - x - 2, 1.1))
    assert abs(r.evalf(40) - (-2 - LambertW(-exp(-2), -1)).evalf(40)) < Float(10)**-38
    assert r.refine(4).interval == Interval(Rational(9, 8), Rational(19, 16))
    assert r > 1 and r < 2 and r.interval == Interval(1, 2)
    assert r.subs(x, 3) == r
    assert r == TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    raises(ValueError, lambda: TranscendentalRoot(exp(x) - x - 2, x, 2, 3))     # no sign change
    raises(ValueError, lambda: TranscendentalRoot(exp(x) - x - 2, x, 2, 1))
    raises(ValueError, lambda: TranscendentalRoot(exp(x) - Symbol('a'), x, 0, 1))
    raises(TypeError, lambda: untyped(TranscendentalRoot)(exp(x) - x - 2, x**2, 1, 2))


def test_isolate_known_roots() -> None:
    # exp(x) = x + 2: the two real roots are -2 - W_0(-e^-2) and -2 - W_{-1}(-e^-2)
    roots = isolate_real_roots(exp(x) - x - 2, x)
    assert roots is not None and len(roots) == 2
    assert _close(roots[0], N(-2 - LambertW(-exp(-2)), 15)) and _close(roots[1], N(-2 - LambertW(-exp(-2), -1), 15))
    # the Dottie number
    roots = isolate_real_roots(x - cos(x), x)
    assert roots is not None and len(roots) == 1 and _close(roots[0], Float('0.7390851332151607', 15))
    # the omega constant
    roots = isolate_real_roots(x*exp(x) - 1, x)
    assert roots is not None and len(roots) == 1 and _close(roots[0], N(LambertW(1), 15))
    # sin(x) = x/2 has three real roots, 0 and +-1.8954942670339809
    roots = isolate_real_roots(sin(x) - x/2, x)
    assert roots is not None and len(roots) == 3 and roots[1] == 0
    assert _close(roots[2], Float('1.8954942670339809', 15)) and _close(-roots[0], Float('1.8954942670339809', 15))
    # exact roots are kept exact
    assert isolate_real_roots(log(x) - 1, x, Interval.open(0, oo)) == [E]
    assert isolate_real_roots(x**3 - 2, x) == [2**Rational(1, 3)]
    assert isolate_real_roots(sin(x) - Rational(1, 2), x, Interval(0, 7)) == [pi/6, 5*pi/6, 13*pi/6]
    assert isolate_real_roots(x**2 - 2, x, Interval(0, 3)) == [sqrt(2)]
    # a double root at a critical point
    assert isolate_real_roots(x**2 + cos(x) - 1, x) == [0]
    assert isolate_real_roots(x**2 + cos(x) - 1, x, Interval.open(-3, oo)) == [0]
    # no roots
    assert isolate_real_roots(x**2 - 3*x + exp(x), x) == []
    assert isolate_real_roots(exp(x) + 1, x) == []
    # singular ends of open intervals
    assert isolate_real_roots(tan(x) - x, x, Interval.open(-pi/2, pi/2)) == [0]
    roots = isolate_real_roots(exp(x) - 1/x, x, Interval.open(0, oo))
    assert roots is not None and len(roots) == 1 and _close(roots[0], _oracle(exp(x) - 1/x, 0.6))
    # two roots on a bounded interval, in order
    roots = isolate_real_roots(exp(x)*sin(x) - 1, x, Interval(0, 5))
    assert roots is not None and len(roots) == 2 and compare_points(roots[0], roots[1]) == -1
    assert _close(roots[0], _oracle(exp(x)*sin(x) - 1, 0.6)) and _close(roots[1], _oracle(exp(x)*sin(x) - 1, 3.1))
    # closed endpoints which are roots are included
    assert isolate_real_roots(sin(x), x, Interval(0, 1)) == [0]
    listed = isolate_real_roots(x - cos(x), x)
    assert listed is not None and real_roots_of(x - cos(x), x) == FiniteSet(*listed)
    raises(ValueError, lambda: isolate_real_roots(exp(x) - Symbol('a'), x))


def test_undecided() -> None:
    # a power with a variable exponent is outside the interval arithmetic
    assert isolate_real_roots(x**x - 2, x, Interval.open(0, oo)) is None
    # a discontinuity in the domain
    assert isolate_real_roots(tan(x) - x, x, Interval(0, 4)) is None


def test_points() -> None:
    r = TranscendentalRoot(x - cos(x), x, 0, 1)
    s = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    assert compare_points(r, s) == -1 and compare_points(s, r) == 1 and compare_points(r, S.Zero) == 1
    between = point_between(r, s)
    assert between is not None and r < between < s
    between = point_between(S.NegativeInfinity, r)
    assert between is not None and between < r


def test_solve_with_isolation() -> None:
    result = solve(Eq(x, cos(x)), x, domain=S.Reals)
    assert isinstance(result, FiniteSet) and len(result) == 1
    [root] = result.args
    assert isinstance(root, TranscendentalRoot) and _close(root, Float('0.7390851332151607', 15))
    result = solve(x**2 + cos(x) > 1, x, domain=S.Reals)
    assert result == Union(Interval.open(-oo, 0), Interval.open(0, oo))
    result = solve(Eq(x**2 + cos(x), 1), x, x > -3)
    assert result == FiniteSet(0)
    result = solve(sin(x) - x/2 >= 0, x, x >= 0)
    assert isinstance(result, Interval) and result.start == 0 and not result.left_open and not result.right_open
    assert _close(result.end, Float('1.8954942670339809', 15))
    result = solve(exp(-x) - x**2 < 0, x, domain=S.Reals)
    assert isinstance(result, Interval) and result.end is oo and _close(result.start, _oracle(exp(-x) - x**2, 0.7))
    result = solve(Ne(x*exp(x), 1), x, domain=S.Reals)
    assert isinstance(result, Union) and len(result.args) == 2
    # both Lambert W branches are found
    result = solve(exp(x) - x - 2, x, domain=S.Reals)
    assert result == FiniteSet(-2 - LambertW(-exp(-2)), -2 - LambertW(-exp(-2), -1))
    assert solve(exp(x) - x - 2, x, x > 0) == FiniteSet(-2 - LambertW(-exp(-2), -1))
