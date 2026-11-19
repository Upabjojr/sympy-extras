"""The Rust ``residue`` (``series/residues.py``) against the Python one."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import I, Integer, Symbol, pi
from sympy_extras.core.basic import Basic
from sympy_extras.functions import cos, exp, log, sin, tan
from sympy_extras.series.residues import residue

from ..conftest import load

rc = load("sympy_extras_rs.core")
x = Symbol('x')


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:
        return type(e).__name__


CASES: list[tuple[Basic, Basic]] = [
    (1/x, Integer(0)), (1/x**2, Integer(0)), (exp(x)/x**3, Integer(0)), (1/(x**2 + 1), I), (1/(x**2 + 1), -I),
    (sin(x)/x**4, Integer(0)), (cos(x)/sin(x), Integer(0)), (1/sin(x)**2, Integer(0)), (exp(1/x), Integer(0)),
    (1/(x*(x - 1)), Integer(1)), (x**2/(x - 2)**3, Integer(2)), (1/tan(x), pi), (log(x)/x, Integer(1)),
    (1/(exp(x) - 1), Integer(0)),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_residue(i: int) -> None:
    e, p = CASES[i]
    assert outcome(lambda: rc.logic_call('series', [Symbol('residue'), e, x, p])) == outcome(lambda: residue(e, x, p))
