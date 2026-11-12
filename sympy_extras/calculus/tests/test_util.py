"""Tests of the port of SymPy's ``periodicity``: the periods of SymPy's
own tests (``sympy/calculus/tests/test_util.py``; the periods are facts,
checked by hand), each period also checked by substitution where the
core recognises the shifted function."""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/calculus/tests/test_util.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the core's objects; the matrix symbols left out.
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core import I, Integer, Mod, Rational, Symbol, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.calculus.util import lcim, periodicity
from sympy_extras.functions import Abs, cos, cot, csc, exp, log, re, sec, sin, tan
from sympy_extras.functions.elementary.piecewise import piecewise

x, y, a = Symbol('x'), Symbol('y'), Symbol('a', real=True)


def _period(f: Basic, v: Symbol = x) -> object:
    return periodicity(f, v)


def test_periodicity_of_trigonometric_functions() -> None:
    assert _period(sin(2*x)) == pi
    assert _period((-2)*tan(4*x)) == pi/4
    assert _period(sin(x)**2) == 2*pi
    assert _period(3**tan(3*x)) == pi/3
    assert _period(tan(x)*cos(x)) == 2*pi
    assert _period(sin(x)**(tan(x))) == 2*pi
    assert _period(tan(x)*sec(x)) == 2*pi
    assert _period(sin(2*x)*cos(2*x) - y) == pi/2
    assert _period(tan(x) + cot(x)) == pi
    assert _period(sin(x) - cos(2*x)) == 2*pi
    assert _period(sin(x) - 1) == 2*pi
    assert _period(sin(4*x) + sin(x)*cos(x)) == pi
    assert _period(exp(sin(x))) == 2*pi
    assert _period(log(cot(2*x)) - sin(cos(2*x))) == pi
    assert _period(sin(2*x)*exp(tan(x) - csc(2*x))) == pi
    assert _period(cos(sec(x) - csc(2*x))) == 2*pi
    assert _period(tan(sin(2*x))) == pi
    assert _period(2*tan(x)**2) == pi
    assert _period(sin(re(x))) == 2*pi
    assert _period(sin(x)**2 + cos(x)**2) == 0
    assert periodicity(tan(x), y) == 0
    assert _period(sin(x) + I*cos(x)) == 2*pi
    assert periodicity(x - sin(2*y), y) == pi


def test_periodicity_of_mod() -> None:
    assert _period(sin(Mod(x, 4))) == 4
    assert _period(Mod(sin(x), 4)) == 2*pi
    assert _period(tan(Mod(3*x - 2, 4))) == Rational(4, 3)
    found = _period(Mod(sqrt(2)*(x + 1) + x, 3))
    assert isinstance(found, Expr) and (found*(sqrt(2) + 1)).expand() == 3
    assert _period(Mod(x**2 + 1, x)) is None


def test_periodicity_of_exponentials() -> None:
    assert _period(exp(x)) is None
    assert _period(exp(I*x)) == 2*pi
    assert periodicity(exp(I*a), a) == 2*pi
    assert periodicity(exp(a), a) is None
    assert periodicity(exp(sin(a)), a) == 2*pi
    assert periodicity(exp(2*I*a), a) == pi
    assert periodicity(exp(a + I*sin(a)), a) is None
    assert periodicity(exp(cos(a/2) + sin(a)), a) == 4*pi
    assert _period(log(x)) is None
    assert _period(exp(x)**sin(x)) is None
    assert periodicity(sin(x)**y, y) is None


def test_periodicity_of_absolute_values() -> None:
    assert _period(Abs(sin(Abs(sin(x))))) == pi
    for f in (cos, sin, sec, csc, tan, cot):
        assert _period(Abs(f(x))) == pi
    assert _period(Abs(sin(tan(x)))) == pi
    assert _period(Abs(sin(sin(x) + tan(x)))) == 2*pi
    assert _period(sin(x) > Rational(1, 2)) == 2*pi


def test_aperiodic() -> None:
    assert _period(x > 2) is None
    assert _period(x**3 - x**2 + 1) is None
    assert _period(Abs(x)) is None
    assert _period(Abs(x**2 - 1)) is None
    assert _period(Mod(x**2 + 4, 2)) is None
    assert _period(Mod(exp(x), 3)) is None
    # returning None for any Piecewise
    assert _period(piecewise((Integer(0), x < -1), (x**2, x <= 1), (log(x), True))) is None


def test_periods_shift_the_function_onto_itself() -> None:
    # the property, not only the value: f(x + T) is f(x)
    for f in (sin(2*x), tan(4*x), sin(x) - cos(2*x), exp(I*x), Abs(sin(x)), sin(Mod(x, 4))):
        period = periodicity(f, x)
        assert isinstance(period, Expr)
        assert as_expr(f.subs(x, x + period)).expand() == f.expand()


def test_lcim() -> None:
    assert lcim([Rational(1, 2), Integer(2), Integer(3)]) == 6
    assert lcim([pi/2, pi/4, pi]) == pi
    assert lcim([2*pi, pi/2]) == 2*pi
    assert lcim([Integer(1), 2*pi]) is None
    assert lcim([Rational(1, 2), Rational(3, 4), Rational(5, 6)]) == Rational(15, 2)
