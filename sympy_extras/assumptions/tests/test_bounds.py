from __future__ import annotations

from sympy_extras.core import Symbol, sqrt, Eq, Rational, Dummy
from sympy_extras.functions import exp, log, Abs
from sympy_extras.assumptions.bounds import polynomial_abstraction


def test_native_polynomial_abstraction() -> None:
    x, y = Symbol('x'), Symbol('y')
    a = polynomial_abstraction(exp(x) > 1 + x, {x})
    assert a is not None and len(a.variables) == 1 and len(a.constraints) == 6
    [t] = a.variables
    assert isinstance(t, Dummy) and t.is_real
    assert a.formula == (t > x + 1) and (t >= x + 1) in a.constraints
    b = polynomial_abstraction(sqrt(x) > 1, {x})
    assert b is not None
    [s] = b.variables
    assert b.constraints == [x >= 0, s >= 0, Eq(s**2, x)] and b.requirements == [x >= 0]
    c = polynomial_abstraction(log(x) < x - 1, {x})
    assert c is not None and c.requirements == [x > 0]
    assert polynomial_abstraction(sqrt(x) + Abs(y) > 0, {x, y}) is not None
    assert polynomial_abstraction(x**2 > 1, {x}) is None
    assert polynomial_abstraction(exp(exp(exp(exp(x)))) > 0, {x}) is None
    d = polynomial_abstraction(x**Rational(-1, 3) > 1, {x})
    assert d is not None and d.requirements == [x > 0]
