"""Integrals by reduction-based telescoping against known values
(Gradshteyn–Ryzhik 3.323.2, 3.325, 3.351.3, 3.896.4, 3.249.1, and
Mathematica), and telescopers against the Almkvist–Zeilberger ansatz.

Timings (this machine, one run, ``max_order=6`` for both) of
``reduction_telescoper`` against ``almkvist_zeilberger``; ``order`` is the
order found, ``None`` a failure:

    1/(x**2 + t*x + 1)                     order 1   0.12 s  vs  order 1   0.22 s
    x**2*exp(-x**2)*cos(2*t*x)             order 1   0.25 s  vs  order 1   0.27 s
    1/(x**2 + t**2)**3                     order 1   0.17 s  vs  order 1   0.10 s
    1/(x**2 + t*x + 1)**4                  order 1   1.55 s  vs  order 1   0.61 s
    (x**4 + 1)*exp(-t*x**2)/(x**2 + 1)     order 2   0.21 s  vs  None      0.35 s
    exp(-t*x)/(x**4 + x + 1)               order 4   0.09 s  vs  order 4   0.27 s
    1/((x**3 + t)**2*(x + 1))              order 3   3.34 s  vs  order 3   5.74 s
    (x**2 + 1)**3*exp(-t*x**2)/(x**4 + 1)  order 3   1.86 s  vs  None      1.11 s

The reduction needs no bound on the certificate and no linear system in
the unknown numerator coefficients, so it wins when the order is high or
the ansatz is large; the ansatz fails on the two sums of similar terms
because it telescopes the expanded terms one by one, while the reduction
merges them first.
"""
from sympy import Expr, cancel, cos, exp, oo, pi, sqrt, symbols

from sympy_extras._typing import ExprLike
from sympy_extras.integrals.conditions import ConditionalValue
from sympy_extras.integrals.reduction import reduction_integral, reduction_telescoper
from sympy_extras.integrals.telescoping import almkvist_zeilberger

x = symbols('x')
t = symbols('t', positive=True)


def _check(F: Expr, a: ExprLike, b: ExprLike, expected: Expr) -> None:
    result = reduction_integral(F, x, a, b, t)
    assert result is not None, F
    assert isinstance(result, ConditionalValue)
    assert cancel(result.value - expected) == 0 or (result.value - expected).simplify() == 0, (result.value, expected)


def test_gaussian_family() -> None:
    _check(exp(-x**2) * cos(2 * t * x), 0, oo, sqrt(pi) * exp(-t**2) / 2)          # GR 3.896.4
    _check(exp(-x**2 + 2 * t * x), -oo, oo, sqrt(pi) * exp(t**2))                  # GR 3.323.2
    _check(exp(-x**2 - t**2 / x**2), 0, oo, sqrt(pi) * exp(-2 * t) / 2)            # GR 3.325
    _check(x**2 * exp(-x**2) * cos(2 * t * x), 0, oo, sqrt(pi) * (1 - 2 * t**2) * exp(-t**2) / 4)


def test_rational_family() -> None:
    _check(1 / (x**2 + t**2), 0, oo, pi / (2 * t))
    _check(1 / (x**2 + t**2)**2, 0, oo, pi / (4 * t**3))                          # GR 3.249.1
    _check(1 / (x**2 + t**2)**3, 0, oo, 3 * pi / (16 * t**5))


def test_exponential_family() -> None:
    _check(x**3 * exp(-t * x), 0, oo, 6 / t**4)                                   # GR 3.351.3
    _check(x**2 * exp(-t * x), 0, oo, 2 / t**3)
    _check(exp(-t * x) / sqrt(x), 0, oo, sqrt(pi / t))


def _same_operator(F: Expr) -> None:
    found = reduction_telescoper(F, x, t)
    ansatz = almkvist_zeilberger(F, x, t)
    assert found is not None and ansatz is not None, F
    assert found.check()
    assert [cancel(a - b) for a, b in zip(found.coefficients, ansatz.coefficients)] == [0] * len(found.coefficients)


def test_against_ansatz() -> None:
    # BCCL 2010 rational examples
    _same_operator(1 / (x**2 + t * x + 1))
    _same_operator(1 / ((x**2 + 1) * (x + t)))
    _same_operator(x / (x**3 + t))
    _same_operator(exp(-t * x) / (x**2 + 1))
    _same_operator(exp(-x**2 - t**2 / x**2))


def test_larger_examples() -> None:
    for F in [1 / (x**2 + t * x + 1)**4, (x**4 + 1) * exp(-t * x**2) / (x**2 + 1)]:
        found = reduction_telescoper(F, x, t)
        assert found is not None and found.check(), F
