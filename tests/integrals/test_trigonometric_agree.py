"""The Rust port of ``integrals/trigonometric.py`` against the Python one,
on a sample of the integrands of its tests (read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.trigonometric import trigonometric_antiderivative

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else str(r)


CASES: list[tuple[str, str]] = [
    ('sin(x)**3', 'x'),
    ('1/(sinh(x)*cosh(x))', 'x'),
    ('1/(sinh(x)**3*cosh(x)**3)', 'x'),
    ('tan(2*x)**2*sec(2*x)**2', 'x'),
    ('sinh(x)**2/cosh(x)**3', 'x'),
    ('x*sin(x)', 'x'),
    ('x*atanh(x)', 'x'),
    ('tan(x)**3', 'x'),
    ('sin(a*x)*sin(b*x)', 'x'),
    ('cosh(x)**2/sinh(x)**3', 'x'),
    ('sin(x)/cos(x)**4', 'x'),
    ('sin(x)*cos(x)**3', 'x'),
    ('sin(x)**3*cos(x)**2', 'x'),
    ('1/(a + cos(x))', 'x'),
    ('sin(x)/(cos(x)**2 + 1)', 'x'),
    ('acos(x)**2', 'x'),
    ('tan(x)**5', 'x'),
    ('1/(sin(x)*cos(x))', 'x'),
    ('sech(x)', 'x'),
    ('1/(tan(x) + 1)', 'x'),
    ('cos(x)**3/sin(x)', 'x'),
    ('sin(x)**3*cos(x)**4', 'x'),
    ('(sin(x) + 1)/(cos(x) + 1)', 'x'),
    ('cosh(x)**2/sinh(x)**2', 'x'),
    ('x**2', 'x'),
    ('1/(sin(x)**3*cos(x))', 'x'),
    ('sin(x)**4/cos(x)', 'x'),
    ('x*sinh(2*x)', 'x'),
    ('sinh(x)**3*cosh(x)**3', 'x'),
    ('csch(x)**4', 'x'),
    ('sin(x)**4*cos(x)**4', 'x'),
    ('sec(x)**4', 'x'),
    ('x**2*acot(x)', 'x'),
    ('1/(sin(x)**3*cos(x)**2)', 'x'),
    ('tan(x)', 'x'),
    ('sin(x)', 'x'),
    ('1/(sin(x)**4*cos(x)**3)', 'x'),
    ('sinh(x)*cosh(x)**3', 'x'),
    ('sec(x)', 'x'),
    ('1/(sin(x)**4*cos(x)**4)', 'x'),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_trigonometric_antiderivative(i: int) -> None:
    text, variable = CASES[i]
    x = Symbol(variable)
    f = parse(text)
    got = outcome(lambda: rc.logic_call('trigonometric', [f, x]))
    if got != 'SKIP':
        assert got == outcome(lambda: trigonometric_antiderivative(f, x)), text
