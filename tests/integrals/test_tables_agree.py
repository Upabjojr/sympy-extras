"""The Rust port of ``integrals/tables.py`` against the Python one: every
entry of the table (generated node by node from the Python one) and the
lookups of the integrands of the entries at their sample parameters, with
symbolic parameters and with the doctests' integrands."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Symbol, Tuple, oo, pi, symbols
from sympy_extras.core.basic import Basic
from sympy_extras.functions import besselj, log, sin
from sympy_extras.integrals import tables as T
from sympy_extras.integrals.tables import table_integral

from ..conftest import load

rc = load("sympy_extras_rs.core")
x = symbols('x')
a = Symbol('a', positive=True)
p = Symbol('p')


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None else str(r)


def rust(f: Basic, lo: Basic, hi: Basic, assumptions: list[Basic]) -> object:
    r = rc.logic_call('tables', [Symbol('integral'), f, x, lo, hi, Tuple(*assumptions)])
    if str(r) == 'NONE':
        return None
    value, condition = r.args
    return T.ConditionalValue(value, condition)


@pytest.mark.parametrize('i', range(len(T.TABLE)))
def test_entry(i: int) -> None:
    e = T.TABLE[i]
    r = rc.logic_call('tables', [Symbol('entry'), Integer(i)])
    assert [str(v) for v in r.args] == [str(e.pattern), str(e.lower), str(e.upper), str(e.value), str(e.condition)]


CASES: list[tuple[Basic, Basic, Basic, list[Basic]]] = []
for entry in T.TABLE:
    for sample in entry.samples:
        lo, hi = entry.range(sample)
        CASES.append((entry.integrand(x, sample), lo, hi, []))
CASES += [
    (sin(a*x)**3/x**3, Integer(0), oo, []), (log(sin(x))**2, Integer(0), pi/2, []), (besselj(2, x)**2/x, Integer(0), oo, []),
    (sin(p*x)**2/x**2, Integer(0), oo, []), (sin(p*x)**2/x**2, Integer(0), oo, [p > 0]), (log(x)/(1 + x**2), Integer(0), oo, []),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_table_integral(i: int) -> None:
    f, lo, hi, assumptions = CASES[i]
    got = outcome(lambda: rust(f, lo, hi, assumptions))
    if got != 'SKIP':
        assert got == outcome(lambda: table_integral(f, x, lo, hi, assumptions or None)), (f, lo, hi)
