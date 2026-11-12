"""Tests of Ramanujan's master theorem and the method of brackets on the
objects of the core (:mod:`sympy_extras.integrals.brackets`)."""
from __future__ import annotations

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import Integer, Rational, S, Symbol, oo, pi, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.functions import atan, besselj, cos, erf, exp, gamma, log, sin
from sympy_extras.integrals.brackets import (method_of_brackets, mellin_transform_series, ramanujan_master_theorem,
                                             taylor_coefficient)
from sympy_extras.integrals.mellin import mellin_transform

x, s = Symbol('x'), Symbol('s')


def _agree(u: Expr, v: Expr, point: Expr) -> bool:
    return abs(complex(u.subs(s, point).evalf(20)) - complex(v.subs(s, point).evalf(20))) < 1e-12


def test_taylor_coefficient() -> None:
    c = taylor_coefficient(exp(-x), x)
    assert c is not None and (c.period, c.offset) == (1, 0)
    assert [c.coefficient.subs(c.index, Integer(k)) for k in range(4)] == [1, -1, S.Half, -Rational(1, 6)]
    c = taylor_coefficient(sin(x), x)
    assert c is not None and (c.period, c.offset) == (2, 1)
    assert [c.coefficient.subs(c.index, Integer(k)) for k in range(3)] == [1, -Rational(1, 6), Rational(1, 120)]
    c = taylor_coefficient(log(1 + x), x)
    assert c is not None and c.coefficient.subs(c.index, Integer(2)) == -S.Half
    phi = c.phi()
    assert phi is not None and phi.subs(c.index, Integer(3)) == -2
    # the series of besselj(0, x), sum (-1/4)**j/j!**2 x**(2j), which
    # SymPy's fps did not find (it was None)
    c = taylor_coefficient(besselj(Integer(0), x), x)
    assert c is not None and (c.period, c.offset) == (2, 0)
    assert [c.coefficient.subs(c.index, Integer(k)) for k in range(4)] == \
        [1, -Rational(1, 4), Rational(1, 64), -Rational(1, 2304)]
    # two residue classes, no series at all, a polynomial
    assert taylor_coefficient(exp(x) + sin(x), x) is None
    assert taylor_coefficient(log(x), x) is None
    assert taylor_coefficient(x**2 + 1, x) is None
    c = taylor_coefficient(exp(x), x)
    assert c is not None and c.phi() is None


def test_formulas_which_fail_below_their_start_are_refused() -> None:
    # fps's formula for (exp(4 x) - exp(-4 x))**2 gives 2 at k = 0, where
    # the series has no term, and (-1)**k for 1/(x**2 + x + 1)
    assert taylor_coefficient((exp(4 * x) - exp(-4 * x))**2, x) is None
    assert taylor_coefficient(1 / (x**2 + x + 1), x) is None
    assert taylor_coefficient(log(1 - x), x) is not None


def test_master_theorem_reproduces_the_table() -> None:
    cases = [(exp(-x), Rational(3, 2)), (1 / (1 + x), S.Half), (sin(x), S.Half), (cos(x), S.Half),
             (erf(x), -S.Half), (log(1 + x), -S.Half), (atan(x), -S.Half)]
    for f, point in cases:
        found = ramanujan_master_theorem(f, x, s)
        assert found is not None, f
        table = mellin_transform(f, x, s)
        assert table is not None, f
        assert _agree(found.transform, table.transform, point), f
        assert found.strip == table.strip, f


def test_master_theorem_strips_and_failures() -> None:
    found = ramanujan_master_theorem(exp(-x**2), x, s)
    assert found is not None and found.strip == (0, oo) and found.transform == gamma(s / 2) / 2
    found = ramanujan_master_theorem(cos(sqrt(x)), x, s)
    assert found is not None and found.strip == (0, S.Half)
    assert _agree(found.transform, 2 * gamma(2 * s) * cos(pi * s), Rational(1, 3))
    assert ramanujan_master_theorem(exp(x), x, s) is None


def test_scaled_shifted_and_products() -> None:
    found = mellin_transform_series(x**2 * exp(-3 * x), x, s)
    assert found is not None and found.strip == (-2, oo)
    assert _agree(found.transform, 3**(-s - 2) * gamma(s + 2), S.Half)
    found = mellin_transform_series(log(1 + 2 * x**2), x, s)
    assert found is not None and found.strip == (-2, 0)
    # GR 3.944.5
    found = method_of_brackets(exp(-x) * sin(x), x, s)
    assert found is not None and found.strip == (-1, oo)
    for point in (Rational(1, 3), Rational(3, 2)):
        assert _agree(found.transform, gamma(s) * sin(pi * s / 4) / 2**(s / 2), point)
    found = method_of_brackets(exp(-x**2) * cos(2 * x), x, s)
    assert found is not None and found.strip == (0, oo)
    # three factors, a factor without a series
    assert mellin_transform_series(exp(-x) * sin(x) * cos(x), x, s) is None
    assert method_of_brackets(exp(-x) * besselj(Integer(0), x), x, s) is None


def test_divergent_bracket_series_give_no_transform() -> None:
    # SymPy's summation, used before the port, summed the series left by
    # the bracket of exp(-x)/(1 + x) to E*pi/sin(pi*s); the transform is
    # E*gamma(s)*uppergamma(1 - s, 1) (1.3433 at s = 1/2, not 8.5397)
    found = method_of_brackets(exp(-x) / (1 + x), x, s)
    assert found is None or abs(complex(found.transform.subs(s, S.Half).evalf(20)) - 1.34329342097686) < 1e-10


def test_wrong_types_are_rejected() -> None:
    raises(TypeError, lambda: untyped(mellin_transform_series)([1], x, s))
