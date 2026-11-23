"""The Rust port of :mod:`sympy_extras.concrete.eulersums` against the
Python one on the calls of the Python tests of the module."""
from __future__ import annotations

from typing import Any

import pytest

from sympy_extras.concrete import eulersums
from sympy_extras.core import Integer, Symbol
from sympy_extras.functions import harmonic, polygamma

from ..conftest import load

rc = load("sympy_extras_rs.core")
n, s = Symbol('n'), Symbol('s')

CASES: list[tuple[str, list[Any]]] = [
    ('euler_sum', [Integer(p), Integer(q)]) for p, q in
    [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (2, 6), (1, 1), (0, 2), (1, 5), (2, 5), (3, 2),
     (3, 4), (3, 5), (4, 2), (4, 3), (4, 4), (4, 5)]
] + [
    ('polygamma_series', [f, n, Integer(lower)]) for f, lower in [
        (harmonic(n)/n**2, 1), (polygamma(0, n)/n**2, 1), (polygamma(1, n)/n**2, 1), (harmonic(n - 1)/n**2, 1),
        (harmonic(n)/n**2, 2), (polygamma(0, n + 2)/n**3, 1), (harmonic(n + 1, 2)/n**3, 1),
        (polygamma(2, n + 1)/n**2, 1), (3*harmonic(n, 3)/n**4, 1), (polygamma(1, n + 3)/n**3, 1),
        (polygamma(1, n - 1)/n**2, 2), (harmonic(n - 2, 2)/n**4, 3), (polygamma(0, n)/n**4, 1),
        (harmonic(n)**2/n**2, 1), (harmonic(n)/n, 1), (harmonic(n, 2)/n**6, 1), (polygamma(1, n - 1)/n**2, 1)]
] + [
    ('polygamma_integral_representation', [polygamma(0, n)/n**s, n]),
    ('polygamma_integral_representation', [polygamma(1, n)/n**s, n]),
]


def _python(name: str, args: list[Any]) -> object:
    if name == 'euler_sum':
        found = eulersums.euler_sum(int(args[0]), int(args[1]))
    elif name == 'polygamma_series':
        found = eulersums.polygamma_series(*args)
    else:
        found = eulersums.polygamma_integral_representation(*args)
    return 'NONE' if found is None else found


@pytest.mark.parametrize('case', range(len(CASES)))
def test_eulersums(case: int) -> None:
    name, args = CASES[case]
    found = rc.logic_call('eulersums', [Symbol(name)] + args)
    assert str(found) == str(_python(name, args))
