"""The Rust port of the quadrature of a definite ``Integral``
(``integrals/_evalf.py`` and mpmath's tanh-sinh rule) against the Python
one: the values agree to the precision asked for, or both give up."""
from __future__ import annotations

import mpmath
import pytest

from sympy_extras.core import Integer, Rational, Symbol, oo, pi
from sympy_extras.core.evalf import numeric_value
from sympy_extras.functions import Abs, Piecewise, besselj, cos, exp, floor, log, sin, sqrt
from sympy_extras.integrals.integral import Integral

from ..conftest import load

rc = load("sympy_extras_rs.core")

x, y = Symbol('x'), Symbol('y')

INTEGRALS: list[Integral] = [
    Integral(exp(-x**2), x, Integer(0), oo),
    Integral(sin(x) / x, x, Integer(1), Integer(5)),
    Integral(log(x), x, Integer(0), Integer(1)),
    Integral(1 / (1 + x**2), x, -oo, oo),
    Integral(Abs(x - Rational(1, 3)), x, Integer(0), Integer(1)),
    Integral(floor(x), x, Integer(0), Rational(7, 2)),
    Integral(floor(2 * x), x, Integer(0), Integer(3)),
    Integral(sqrt(x), x, Integer(0), Integer(2)),
    Integral(exp(x), x, Integer(1), Integer(0)),
    Integral(cos(x)**2, x, Integer(0), pi),
    Integral(besselj(0, x), x, Integer(0), Integer(3)),
    Integral(Piecewise((x, x < 1), (1 / x, True)), x, Integer(0), Integer(3)),
    Integral(Piecewise((x, x < Rational(1, 2)), (1 - x, True)), x, Integer(0), Integer(1)),
    Integral(exp(-x) * x**3, x, Integer(0), oo),
    Integral(1 / sqrt(1 - x**2), x, Integer(-1), Integer(1)),
    Integral(Integral(x * y, y, Integer(0), x), x, Integer(0), Integer(1)),
    Integral(exp(-x), x, -oo, Integer(0)),
    Integral(x**2, x, Integer(2), Integer(-1)),
    Integral(1 / x, x, Integer(1), oo),
    Integral(1 / x, x, Integer(1), Rational(1, 2)),
    Integral(sin(x**2), x, Integer(0), Integer(3)),
    Integral(x**x, x, Integer(0), Integer(1)),
    Integral(x**Rational(-3, 2), x, Integer(161), oo),
    Integral(exp(-x**2), x, Integer(-326), Integer(500)),
]

CASES: list[tuple[int, int]] = [(k, prec) for k in range(len(INTEGRALS)) for prec in (53, 80, 120)]


@pytest.mark.parametrize('k, prec', CASES)
def test_agrees(k: int, prec: int) -> None:
    node = INTEGRALS[k]
    py = numeric_value(node, prec)
    rs = rc.numeric_value(node, prec)
    if py is None or rs is None:
        assert py is None and rs is None
        return
    re, im = rs
    with mpmath.workprec(prec + 20):
        value = mpmath.mpc(mpmath.mpf(re.numerator) / re.denominator, mpmath.mpf(im.numerator) / im.denominator)
        assert abs(value - py) <= mpmath.mpf(2) ** (8 - prec) * max(1, abs(py))
