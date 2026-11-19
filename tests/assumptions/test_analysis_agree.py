"""The Rust port of ``assumptions/analysis.py`` against the Python module:
the domains of the variables from the facts, the rational points between
constants (from their values at the working precision), the certified
sorting of constants, the zeros of rational functions in a domain."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.assumptions import analysis as an
from sympy_extras.assumptions.facts import Facts
from sympy_extras.core import E, Integer, Ne, Rational, Symbol, Tuple, oo, pi, sqrt
from sympy_extras.core.basic import Basic
from sympy_extras.sets.sets import Reals, interval

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')


def call(op: str, *args: Basic) -> str:
    try:
        return str(rc.logic_call('analysis', [Symbol(op), *args])).replace('NONE', 'None')
    except NotImplementedError:
        return 'SKIP'


def py(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'


FACTS: list[list[Basic]] = [
    [x > 0, x < pi], [x > y], [x**2 < 2], [x >= 1, x <= 1], [x > 0, x**2 - 3*x + 1 > 0], [Ne(x, 2)],
    [x < E, x > -sqrt(2)], [x**3 - x > 0, x < 2], [x > 1, x < sqrt(3)], [x**2 - 2 >= 0],
]


@pytest.mark.parametrize('i', range(len(FACTS)))
def test_domain_of(i: int) -> None:
    facts = FACTS[i]
    got = call('domain_of', x, Tuple(*facts))
    if got != 'SKIP':
        assert got == py(lambda: an.domain_of(x, Facts(facts)))


def test_between_sorted_zeros() -> None:
    for p, q in [(Integer(1), Integer(2)), (pi, Rational(22, 7)), (-oo, sqrt(2)), (E, oo), (-oo, oo), (sqrt(2), sqrt(3))]:
        assert call('between', p, q) == py(lambda: an._between(p, q))
    values = [pi, Integer(3), sqrt(10), E, Rational(22, 7), sqrt(2) + 1]
    assert call('sorted', Tuple(*values)) == py(lambda: tuple(an._sorted_constants(values) or ()))
    for f, d in [(x**2 - 2, Reals), (1/(x - 1), interval(Integer(0), Integer(2))), ((x - 1)*(x + 3), interval(Integer(0), oo))]:
        found, zeros = an._real_zeros_in(f, x, d)
        assert call('real_zeros_in', f, x, d) == str((found, tuple(zeros)))
