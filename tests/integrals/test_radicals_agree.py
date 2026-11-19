"""The Rust port of ``integrals/radicals.py`` against the Python one: the
real antiderivatives of ``x**n*Q**(m/2)`` by the reduction formulas and
Euler's substitutions."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Rational, Symbol, Tuple, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import false, true
from sympy_extras.integrals.radicals import euler_substitution_antiderivative, quadratic_radical_antiderivative

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y, a, b, c = symbols('x y a b c')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


ABC = [a > 0, b > 0, c > 0, 4*a*c - b**2 > 0]
QUADRATIC: list[tuple[Basic, Basic, list[Basic], bool]] = [
    (sqrt(1 - x**2), x, [], True), (x**2/sqrt(x**2 + 1), x, [], True), (1/sqrt(-x**2 + 2*x), x, [], True),
    (x**2*sqrt(1 - x**2), x, [], False), (x**3/sqrt(x**2 + 1), x, [], False), ((2*x - x**2)**Rational(3, 2), x, [], False),
    (x**2/(1 - x**2)**Rational(3, 2), x, [], False), (1/sqrt(x**2 + 2*x + 5), x, [], False),
    ((x**2 + 1)**Rational(-5, 2), x, [], False), (sqrt(2*x + 3), x, [], False), (x**2/sqrt(2*x + 3), x, [], False),
    (sqrt(x**2 - 1), x, [], False), (1/(x*sqrt(x**2 + 1)), x, [], True),
    (sqrt(a + b*x + c*x**2)/x, x, ABC, True), (1/(x**2*sqrt(a + b*x + c*x**2)), x, ABC, True),
    (sqrt(a + b*x + c*x**2)/x, x, [], True), (2*sqrt(1 - y)*sqrt(y + 1), y, [], True), (sqrt(x**3 + 1), x, [], True),
    (sqrt(1 - x**2)*sqrt(x**2 + 1), x, [], True), (sqrt(-x**2 - 1), x, [], True), (sqrt(1 - x**2)/x, x, [], True),
]
EULER: list[Basic] = [
    1/((x + 3)*sqrt(x**2 - 1)), sqrt(x**2 + x)/(x**2 + 1)**2, sqrt(x + 1)/(x**2 + 1), 1/((x + 1)*sqrt(x**3 + 1)),
]


@pytest.mark.parametrize('i', range(len(QUADRATIC)))
def test_quadratic(i: int) -> None:
    f, v, assumptions, euler = QUADRATIC[i]
    got = outcome(lambda: rc.logic_call('radicals', [Symbol('quadratic'), f, v, Tuple(*assumptions), true if euler else false]))
    if got != 'SKIP':
        assert got == outcome(lambda: quadratic_radical_antiderivative(f, v, assumptions or None, euler)), f


@pytest.mark.parametrize('i', range(len(EULER)))
def test_euler(i: int) -> None:
    f = EULER[i]
    got = outcome(lambda: rc.logic_call('radicals', [Symbol('euler'), f, x, Tuple()]))
    if got != 'SKIP':
        assert got == outcome(lambda: euler_substitution_antiderivative(f, x)), f
