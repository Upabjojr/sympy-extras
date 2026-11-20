"""The Rust port of ``concrete/_evalf.py`` (SymPy's ``evalf_sum``:
``hypsum`` in fixed point or by Richardson extrapolation, the direct sum,
Euler-Maclaurin summation) against the Python one on the sums of the
tests (read with the core's parser: plain symbols): the values agree to
the precision asked for, or both give up (Python raises ``ValueError``
for a divergent sum, Rust gives no value)."""
from __future__ import annotations

import mpmath
import pytest

from sympy_extras.concrete.expr_with_limits import Sum
from sympy_extras.core import Symbol
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr
from sympy_extras.core.parsing import parse

from ..conftest import load

rc = load("sympy_extras_rs.core")

#: (the summand, the variable, the bounds, the precision in bits)
CASES: list[tuple[str, str, str, str, int]] = [
    ('1/factorial(i)', 'i', '0', 'oo', 80),
    ('i**(-2)', 'i', '1', 'oo', 80),
    ('(-1)**i/i', 'i', '1', 'oo', 80),
    ('(1/2)**i', 'i', '0', 'oo', 80),
    ('(-1)**i/(2*i + 1)', 'i', '0', 'oo', 80),
    ('binomial(2*i, i)/5**i', 'i', '0', 'oo', 80),
    ('2**i/factorial(i)', 'i', '3', 'oo', 80),
    ('i/3**i', 'i', '1', 'oo', 80),
    ('1', 'i', '1', 'oo', 53),
    ('1/i', 'i', '1', 'oo', 53),
    ('2**i', 'i', '0', 'oo', 53),
    ('factorial(i)', 'i', '0', 'oo', 53),
    ('(-1)**i', 'i', '0', 'oo', 53),
    ('i**2', 'i', '1', '10', 80),
    ('sin(i)', 'i', '1', '5', 80),
    ('1/i', 'i', '5', '4', 80),
    ('pi**i/factorial(i)', 'i', '0', 'oo', 80),
    ('2**(i/2)/factorial(i)', 'i', '0', 'oo', 80),
    ('1/(i**2 + 1)', 'i', '0', 'oo', 80),
    ('1/(i*(i + 1))', 'i', '1', 'oo', 80),
    ('(i + 1/2)**(-2)', 'i', '0', 'oo', 80),
    ('(3/7)**i*(i + 1)**2', 'i', '0', 'oo', 80),
    ('cos(i)', 'i', '2', '9', 80),
    ('i/factorial(i)', 'i', '0', 'oo', 80),
    ('exp(-i**2)', 'i', '0', 'oo', 80),
    ('i**(-3/2)', 'i', '1', 'oo', 80),
    ('harmonic(i)/i**3', 'i', '1', 'oo', 80),
    ('log(i)/i**3', 'i', '1', 'oo', 80),
    ('1/i', 'i', '1', '1000', 80),
    ('1/(i**2 + log(i))', 'i', '1', 'oo', 80),
    ('i**2/(i**4 + 1)', 'i', '0', 'oo', 80),
    ('exp(-i**2)', 'i', '-50', '50', 80),
    ('exp(-i**2)', 'i', '-500', '500', 80),
    ('exp(-(i - 3)**2)', 'i', '-5000', '5000', 53),
    ('sin(i)/i**2', 'i', '1', 'oo', 53),
    ('i', 'i', '1', 'oo', 73),
    ('i**2', 'i', '1', 'oo', 73),
    ('1', 'i', '1', 'oo', 73),
    ('i**(-2)', 'i', '1', 'oo', 73),
    ('2*i', 'i', '1', 'oo', 73),
    ('3', 'i', '1', 'oo', 73),
]


@pytest.mark.parametrize('f, i, a, b, prec', CASES)
def test_agrees(f: str, i: str, a: str, b: str, prec: int) -> None:
    function, var, lo, hi = parse(f), parse(i), parse(a), parse(b)
    assert isinstance(var, Symbol) and isinstance(lo, Expr) and isinstance(hi, Expr)
    node = Sum(function, (var, lo, hi))
    try:
        py = numeric_value(node, prec)
    except ValueError:
        # a divergent sum: Python raises, Rust's numeric_value has no value
        py = None
    rs = rc.numeric_value(node, prec)
    if py is None or rs is None:
        assert py is None and rs is None
        return
    re, im = rs
    with mpmath.workprec(prec + 20):
        value = mpmath.mpc(mpmath.mpf(re.numerator) / re.denominator, mpmath.mpf(im.numerator) / im.denominator)
        assert abs(value - py) <= mpmath.mpf(2) ** (8 - prec) * max(1, abs(py))
