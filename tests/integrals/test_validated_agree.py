"""The Rust port of :mod:`sympy_extras.integrals.validated` against the
Python module. The enclosures are rigorous on both sides but not equal bit
for bit (exact errors and Newton nodes in Rust, mpmath floats in Python),
so the agreement checked is: both find an enclosure or neither does, the
two enclosures overlap, and the Rust radius is below ``10**(-digits)``."""
from __future__ import annotations

from fractions import Fraction

import pytest

from sympy_extras.core import Integer, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.functions import atan, cos, exp, log, sin
from sympy_extras.integrals.validated import validated_integral
from ..conftest import load

rc = load("sympy_extras_rs.core")

x = Symbol('x')
a = Symbol('a')

CASES: list[tuple[Basic, Basic, Basic, int]] = [
    (exp(-x**2), Integer(0), Integer(1), 15),
    (exp(-x**2), Integer(0), Integer(1), 30),
    (atan(x), Integer(0), Integer(1), 15),
    (cos(x)**2, Integer(0), pi, 10),
    (x**Rational(-1, 2)*(1 - x)**Rational(-1, 2), Integer(0), Integer(1), 15),
    (log(x)**2, Integer(0), Integer(1), 15),
    (sqrt(x)*log(x), Integer(0), Integer(1), 15),
    (sin(x)/x, Integer(0), Integer(1), 15),
    (1/x, Integer(0), Integer(1), 15),
    (1/x**2, Integer(0), Integer(1), 15),
    (1/(1 + x**2), Integer(0), oo, 20),
    (x*exp(-x), Integer(0), oo, 15),
    (exp(-x**2), -oo, oo, 12),
    (exp(x), -oo, Integer(0), 12),
    (exp(-x), oo, Integer(0), 10),
    (sin(x)/x, Integer(0), oo, 8),
    (cos(x)/(1 + x**2), Integer(0), oo, 8),
    (sin(-x)/x, Integer(0), oo, 8),
    (sqrt(x), Integer(0), Integer(1), 15),
    (log(1 + x), Integer(0), Integer(1), 12),
    (exp(-a*x), Integer(0), Integer(1), 15),
    (x, Integer(0), a, 15),
    (exp(-x), Integer(1), Integer(1), 15),
    (x, Integer(1), Integer(0), 12),
    (Integer(1), Integer(0), Integer(1), 12),
]


@pytest.mark.parametrize('case', range(len(CASES)))
def test_validated_integral(case: int) -> None:
    f, lo, hi, digits = CASES[case]
    p = validated_integral(f, x, lo, hi, digits)
    r = rc.logic_call('validated', [f, x, lo, hi, Integer(digits)])
    if p is None:
        assert str(r) == 'NONE'
        return
    assert str(r) != 'NONE'
    value, error = p[0].as_fraction(), p[1].as_fraction()
    mid, radius = (Fraction(int(v.p), int(v.q)) for v in r.args)
    assert abs(mid - value) <= radius + error
    assert radius <= Fraction(1, 10**digits)
