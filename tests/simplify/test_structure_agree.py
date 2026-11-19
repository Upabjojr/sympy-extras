"""The Rust port of ``simplify/structure.py`` (the elementary tower of the
Risch–Rosenlicht structure theorem) against the Python one: canonical
forms, the zero test and equality, with and without assumptions."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import E, I, Rational, Symbol, Tuple, pi, sqrt, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.functions import acos, asin, atan, cos, cosh, exp, log, sin, sinh, tan
from sympy_extras.simplify.structure import canonical_form, equal, is_zero

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = symbols('x y')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except ValueError:
        return 'ValueError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


CANONICAL: list[tuple[Basic, list[Basic]]] = [
    (exp(x + y) - exp(x)*exp(y), []), (sin(2*x) - 2*sin(x)*cos(x), []), (exp(I*pi) + 1, []),
    ((sqrt(x) + 1)**2 - x - 1, []), (log(x*y) - log(x) - log(y), [x > 0, y > 0]), (exp(x/2), []),
    (log(x**2) - 2*log(x), [x > 0]), (log(x**2) - 2*log(x), []), (sqrt(2)*sqrt(3) - sqrt(6), []),
    (4*atan(Rational(1, 5)) - atan(Rational(1, 239)), []), (log(8)/log(2), []), (E**2*exp(-2), []),
    (asin(x) - atan(x/sqrt(1 - x**2)), [x > -1, x < 1]), (cosh(x)**2 - sinh(x)**2, []), (tan(x)*cos(x), []),
    (log(-x), [x > 0]), (sqrt(x**2), [x > 0]), (exp(log(x)/3)**3, []), (acos(x) + asin(x), []),
]

ZERO: list[tuple[Basic, list[Basic]]] = [
    (cosh(x)**2 - sinh(x)**2 - 1, []), (exp(x)*exp(-x) - 1, []), (exp(x) - x - 1, []), (sqrt(x)**2 - x, []),
    (sqrt(x**2) - x, []), (sqrt(x**2) - x, [x > 0]), (log(x**2) - 2*log(x), []), (log(x**2) - 2*log(x), [x > 0]),
    (log(2*x) - log(2) - log(x), [x > 0]), (exp(2*log(x)) - x**2, []), (pi - 4*atan(1), []),
]

EQUAL: list[tuple[Basic, Basic]] = [
    (tan(x), sin(x)/cos(x)), (sqrt(x)*sqrt(x), x), (sin(x), x), (exp(x)**2, exp(2*x)), (log(E**3), Rational(3)),
]


@pytest.mark.parametrize('i', range(len(CANONICAL)))
def test_canonical_form(i: int) -> None:
    e, assumptions = CANONICAL[i]
    got = outcome(lambda: rc.logic_call('structure', [Symbol('canonical'), e, Tuple(*assumptions)]))
    if got != 'SKIP':
        assert got == outcome(lambda: canonical_form(e, assumptions or None)), (e, assumptions)


@pytest.mark.parametrize('i', range(len(ZERO)))
def test_is_zero(i: int) -> None:
    e, assumptions = ZERO[i]
    got = outcome(lambda: rc.logic_call('structure', [Symbol('is_zero'), e, Tuple(*assumptions)]))
    if got != 'SKIP':
        assert got == outcome(lambda: is_zero(e, assumptions or None)), (e, assumptions)


@pytest.mark.parametrize('i', range(len(EQUAL)))
def test_equal(i: int) -> None:
    a, b = EQUAL[i]
    got = outcome(lambda: rc.logic_call('structure', [Symbol('equal'), a, b, Tuple()]))
    if got != 'SKIP':
        assert got == outcome(lambda: equal(a, b)), (a, b)
