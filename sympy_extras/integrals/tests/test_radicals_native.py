"""Tests of the native radical table (:mod:`sympy_extras.integrals.radicals`)
on the objects of the core: closed forms of Gradshteyn and Ryzhik 2.26,
every antiderivative checked by differentiation at sample points."""
from __future__ import annotations

from typing import Optional

from sympy_extras.core import Integer, Rational, Symbol, sqrt
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import Boolean
from sympy_extras.functions import Log, asin, asinh, log
from sympy_extras.integrals.conditions import numerically_equal
from sympy_extras.integrals.radicals import (_Table, _coefficients, _odd_power, _quadratic_coefficients,
                                             euler_substitution_antiderivative, quadratic_radical_antiderivative)

x, y = Symbol('x'), Symbol('y')


def _checks(F: Optional[Expr], f: Expr, facts: Optional[list[Boolean]] = None) -> bool:
    return F is not None and numerically_equal(F.diff(x), f, facts)


def test_the_table_is_real() -> None:
    # GR 2.261, 2.262.1: the arcsine and the area sine, not the complex logarithm
    assert quadratic_radical_antiderivative(sqrt(1 - x**2), x) == x * sqrt(1 - x**2) / 2 + asin(x) / 2
    assert quadratic_radical_antiderivative(x**2 / sqrt(x**2 + 1), x) == x * sqrt(x**2 + 1) / 2 - asinh(x) / 2
    assert quadratic_radical_antiderivative(1 / sqrt(2 * x - x**2), x) == asin(x - 1)
    inside: list[Boolean] = [x > Rational(1, 10), x < Rational(1, 2)]
    for f in [x**2 * sqrt(1 - x**2), x**3 / sqrt(x**2 + 1), (2 * x - x**2)**Rational(3, 2),
              x**2 / (1 - x**2)**Rational(3, 2), 1 / sqrt(x**2 + 2 * x + 5), 1 / (x**2 + 1)**Rational(5, 2),
              sqrt(2 * x + 3), x**2 / sqrt(2 * x + 3), x**4 / (1 - x**2)**Rational(5, 2)]:
        assert _checks(quadratic_radical_antiderivative(f, x, euler=False), f, inside), f
    # a positive leading coefficient and two real roots: the logarithm
    F = quadratic_radical_antiderivative(sqrt(x**2 - 1), x, euler=False)
    assert F is not None and F.has(Log) and _checks(F, sqrt(x**2 - 1), [x > 2])
    # GR 2.266: 1/(x sqrt(Q)) with a positive constant term is a logarithm
    assert quadratic_radical_antiderivative(1 / (x * sqrt(x**2 + 1)), x) == -log((2 * sqrt(x**2 + 1) + 2) / x)


def test_signs_from_the_assumptions() -> None:
    a, b, c = Symbol('a'), Symbol('b'), Symbol('c')
    Q = a + b * x + c * x**2
    facts: list[Boolean] = [a > 0, b > 0, c > 0, 4 * a * c - b**2 > 0]
    for f in [sqrt(Q) / x, 1 / (x**2 * sqrt(Q))]:
        F = quadratic_radical_antiderivative(f, x, facts)
        assert _checks(F, f, facts + [x > 0]), f
    assert quadratic_radical_antiderivative(sqrt(Q) / x, x) is None       # the sign of a undecided


def test_products_of_linear_radicals_and_outside_the_table() -> None:
    F = quadratic_radical_antiderivative(2 * sqrt(1 - y) * sqrt(y + 1), y)
    assert F == y * sqrt(1 - y**2) + asin(y)
    assert quadratic_radical_antiderivative(sqrt(x**3 + 1), x) is None
    assert quadratic_radical_antiderivative(sqrt(1 - x**2) * sqrt(x**2 + 1), x) is None
    assert quadratic_radical_antiderivative(1 / (x**2 + 1), x) is None
    assert quadratic_radical_antiderivative(sqrt(-1 - x**2), x) is None


def test_coefficients_beyond_the_native_poly() -> None:
    # sqrt(2) in the radicand: SymPy's Poly takes it over EX, the native Poly
    # does not; the coefficients are read off the expanded polynomial
    assert _coefficients(x**2 + sqrt(2) * x + 1, x) == {2: Integer(1), 1: sqrt(2), 0: Integer(1)}
    assert _coefficients(sqrt(x) + 1, x) is None
    abc = _quadratic_coefficients(x**2 + sqrt(2) * x + 1, x)
    assert abc is not None
    F = _Table(*abc, x).power(-1)
    assert F == asinh(sqrt(2) * (2 * x + sqrt(2)) / 2)
    # (the whole route stays refused, as on SymPy objects before: the
    # sampler of the numerical check finds no point of the fact
    # x**2 + sqrt(2)*x + 1 > 0, with its irrational coefficient)
    assert _checks(F, 1 / sqrt(x**2 + sqrt(2) * x + 1))
    assert _odd_power((x**2 + 1)**3, x) == (x**2 + 1, 3) and _odd_power(x**2 + 1, x) == (x**2 + 1, 1)


def test_euler_substitutions() -> None:
    # the rational function of t goes through SymPy's ratint at run time
    for f in [1 / ((x + 3) * sqrt(x**2 - 1)), sqrt(x**2 + x) / (x**2 + 1)**2, sqrt(x + 1) / (x**2 + 1)]:
        F = euler_substitution_antiderivative(f, x)
        assert _checks(F, f, [x > 2, x < 3]), f
    assert euler_substitution_antiderivative(1 / ((x + 1) * sqrt(x**3 + 1)), x) is None
