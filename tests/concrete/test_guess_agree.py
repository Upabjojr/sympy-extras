"""The Rust port of ``concrete/guess.py`` against the Python one on
classical sequences and power series."""
from __future__ import annotations

from fractions import Fraction
from math import comb, factorial
from typing import Callable, Optional

import pytest

from sympy_extras.concrete.guess import (guess_algebraic_equation, guess_differential_equation,
                                         guess_recurrence)
from sympy_extras.core import Integer, Rational, Symbol, Tuple
from sympy_extras.core.basic import Basic

from ..conftest import load

rc = load("sympy_extras_rs.core")


def fibonacci(n: int) -> list[Fraction]:
    a, b, out = 0, 1, []
    for _ in range(n):
        out.append(Fraction(a))
        a, b = b, a + b
    return out


def harmonic(n: int) -> list[Fraction]:
    out, h = [], Fraction(0)
    for k in range(1, n + 1):
        h += Fraction(1, k)
        out.append(h)
    return out


SEQUENCES: list[list[Fraction]] = [
    fibonacci(20),
    [Fraction(comb(2*k, k), k + 1) for k in range(16)],
    [Fraction(factorial(k)) for k in range(14)],
    harmonic(18),
    [Fraction(comb(2*k, k)) for k in range(16)],
    [Fraction(1, factorial(k)) for k in range(16)],
    [Fraction(k**3 - 2*k + 1) for k in range(12)],
    [Fraction(2**k + 3**k) for k in range(14)],
    [Fraction(1)] * 10,
    [Fraction(0)] * 8,
    [Fraction(k % 3) for k in range(15)],
    [Fraction(1), Fraction(2), Fraction(7), Fraction(3)],
]

GUESSERS: list[tuple[str, Callable[..., object]]] = [
    ('recurrence', guess_recurrence), ('differential', guess_differential_equation),
    ('algebraic', guess_algebraic_equation)]


def python(kind: str, found: object) -> str:
    if found is None:
        return 'None'

    def poly(p: list[int]) -> Basic:
        return Tuple(*[Integer(c) for c in p])
    coefficients: list[list[int]] = getattr(found, 'coefficients')
    rows = Tuple(*[poly(p) for p in coefficients])
    if kind == 'differential':
        return str(Tuple(rows, poly(getattr(found, 'inhomogeneous'))))
    return str(rows)


def optional(v: Optional[int]) -> Basic:
    return Symbol('NONE') if v is None else Integer(v)


@pytest.mark.parametrize('kind,i,limits', [(k, i, lim) for k, _ in GUESSERS for i in range(len(SEQUENCES))
                                           for lim in ((None, None), (2, 3))])
def test_guess(kind: str, i: int, limits: tuple[Optional[int], Optional[int]]) -> None:
    terms = SEQUENCES[i]
    guesser = dict(GUESSERS)[kind]
    expected = python(kind, guesser(terms, limits[0], limits[1]))
    found = rc.logic_call('guess', [Symbol(kind), Tuple(*[Rational(t.numerator, t.denominator) for t in terms]),
                                    optional(limits[0]), optional(limits[1]), Integer(3)])
    assert ('None' if str(found) == 'NONE' else str(found)) == expected
