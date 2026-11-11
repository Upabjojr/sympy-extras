"""The rational integrator on the core (no SymPy objects): every
antiderivative checked by differentiation, the forms of SymPy's tests."""
from __future__ import annotations

from sympy_extras.core import I, Rational, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions import atan, log
from sympy_extras.integrals.risch.rationaltools import (log_to_atan, log_to_real, polynomial_part, ratint,
                                                        ratint_logpart, ratint_ratpart)
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.rootoftools import RootSum

x, a, b, t = Symbol('x'), Symbol('a'), Symbol('b'), Symbol('t')


def _is_antiderivative(F: Expr, f: Expr, v: Symbol = x) -> bool:
    return cancel(F.diff(v) - f) == 0


def test_rational_cases() -> None:
    cases = [
        36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), 1/(x**2 + 1), (x**4 - 3*x**2 + 6)/(x**6 - 5*x**4 + 5*x**2 + 4),
        (x**7 - 24*x**4 - 4*x**2 + 8*x - 8)/(x**8 + 6*x**6 + 12*x**4 + 8*x**2), 1/(x**2 + x + 1), 1/(x**3 + 1),
        1/(x - 1)**2, (x**3 + 1)/(x*(x - 1)**3), 1/(x**2 + 1)**3, (x + 3)/(x**2 + 4*x + 13), 1/(x**4 + 4),
        x**5 - 3*x + Rational(1, 2), (x**4 - 1)/(x - 1), 1/(x**4 + 1),
    ]
    for f in cases:
        assert _is_antiderivative(ratint(f, x), f), f
    for f in [1/(x**2 + 1), 1/(x**3 + 1), 1/(x**2 + x + 1)]:
        assert _is_antiderivative(ratint(f, x, real=False), f), f


def test_forms() -> None:
    # the forms of SymPy's tests of ratint
    assert ratint(x**3 - x, x) == x**4/4 - x**2/2
    assert ratint((x**3 - x)/(x - 1), x) == x**3/3 + x**2/2
    assert ratint(x/((x - a)*(x + a)), x) == log(x**2 - a**2)/2
    assert ratint(1/(x**2 + 1), x) == atan(x)
    assert ratint(1/(x**2 + 1), x, real=False) == I*log(x + I)/2 - I*log(x - I)/2
    assert ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x) == \
        -4*log(x + 1) + 4*log(x - 2) + (12*x + 6)/(x**2 - 1)
    assert ratint(1/(x**2 + x + 1), x) == 2*sqrt(3)*atan(sqrt(3)/3 + 2*x*sqrt(3)/3)/3
    assert ratint(1/(x*(a + b*x)**3), x) == \
        (3*a + 2*b*x)/(2*a**4 + 4*a**3*b*x + 2*a**2*b**2*x**2) + (log(x) - log(a/b + x))/a**3
    assert ratint(1/(x**2 + a**2), x) == (-I*log(-I*a + x)/2 + I*log(I*a + x)/2)/a
    assert ratint((x, x**2 - 1), x) == log(x**2 - 1)/2
    d = Dummy('d')
    assert ratint(1/(d**2 + 1), d, symbol=d) == atan(d)


def test_root_sums() -> None:
    F = ratint(1/(x**3 - x - 1), x, real=False)
    assert isinstance(F, RootSum)
    assert _is_antiderivative(F, 1/(x**3 - x - 1))
    # a dummy of the name given
    assert F.variable.name == 't' and isinstance(F.variable, Dummy)
    G = ratint(1/(x**3 - x - 1), x, real=False, symbol='z')
    assert isinstance(G, RootSum) and G.variable.name == 'z'


def test_parameters() -> None:
    p = Symbol('p', positive=True)
    for f in [1/(x + a), 1/(x**2 + a), x/(x**2 + a*x + b), 1/((x - a)*(x - b)), 1/(x**2 + p), 1/(x**2 + p**2),
              p/(x**2 + 1)**2, 1/(x**3 + a)]:
        assert _is_antiderivative(ratint(f, x), f), f
    # real parameters give the arctangent
    assert ratint(1/(x**2 + p**2), x) == atan(x/p)/p


def test_algebraic_coefficients() -> None:
    for f in [(-x**2 + sqrt(2))/(x**4 - sqrt(2)*x**2 + 1), 1/(x - sqrt(3)), 1/(x**2 + sqrt(2)), 1/(x - I)]:
        assert _is_antiderivative(ratint(f, x), f), f


def test_parts() -> None:
    assert ratint_ratpart(36, x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x) == \
        ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))
    assert ratint_ratpart(1, x**2 + a**2, x) == (0, 1/(x**2 + a**2))
    assert ratint_logpart(x, x**2 - 9, x, t) == [(x**2 - 9, 1 - 2*t)]
    assert ratint_logpart(x**2, x**3 - 5, x, t) == [(x**3 - 5, 1 - 3*t)]
    assert ratint_logpart(1, x**2 + x + 1, x, t) == [(x + 3*t/2 + Rational(1, 2), 3*t**2 + 1)]
    assert polynomial_part(x**3 + a, x + 1, x) == (x**3/3 - x**2/2 + x, a - 1)


def test_real_forms() -> None:
    assert log_to_atan(x, 1, x) == 2*atan(x)
    assert log_to_atan(x + Rational(1, 2), sqrt(3)/2, x) == 2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    assert log_to_atan(sqrt(3)/2, x + Rational(1, 2), x) == -2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    y = Symbol('y')
    assert log_to_real(x + 3*y/2 + Rational(1, 2), 3*y**2 + 1, x, y) == 2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
    assert log_to_real(x**2 - 1, 1 - 2*y, x, y) == log(x**2 - 1)/2
    assert log_to_real(x**2 - 1, 1 - 2*y, x, y, complex_only=True) is None


def test_cubic_in_radicals_is_fast() -> None:
    # the real form of 1/(x**3 - x - 1) by Cardano's roots: the port of
    # SymPy's code on SymPy objects took more than five minutes (the
    # conjugates of the roots and the polynomials over EX), the native
    # one a second
    f = 1/(x**3 - x - 1)
    F = ratint(f, x)
    assert not F.has(RootSum)
    from sympy_extras.core.evalf import numeric_value
    for point in [Rational(1, 3), Rational(5, 2), Rational(-7, 3)]:
        d = (F.diff(x) - f).subs(x, point)
        value = numeric_value(d, 120)
        assert value is not None and abs(value) < 1e-25


def test_transcendental_constants_are_parameters() -> None:
    # the root tan(1)**2/(1 + tan(1)**2) of the Risch port's residues was
    # simplified to sin(1)**2 (SymPy's simplify of a root over EX, where
    # SymPy takes tan(1) as a generator of ZZ(tan(1)) and keeps the form),
    # and the antiderivative of sin(x)**2*cot(x - 1) lost its real form
    from sympy_extras.functions import tan
    t0, z = Symbol('t0'), Symbol('z')
    assert log_to_real(t0**2 + 2*t0/tan(1) - 1, z - tan(1)**2/(1 + tan(1)**2), t0, z) == \
        log(t0**2 + 2*t0/tan(1) - 1)*tan(1)**2/(1 + tan(1)**2)
    f = 1/(x**2 + pi)
    assert _is_antiderivative(ratint(f, x), f)
