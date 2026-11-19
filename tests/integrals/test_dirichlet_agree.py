"""The Rust port of ``integrals/dirichlet.py`` against the Python one on
the integrands of its tests (read with the core's parser)."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Ge, Gt, Le, Lt, Symbol, Tuple
from sympy_extras.core.basic import Basic
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.dirichlet import dirichlet_integral

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


def relation(text: str) -> Basic:
    for op, build in ((' >= ', Ge), (' <= ', Le), (' > ', Gt), (' < ', Lt)):
        if op in text:
            left, right = text.split(op)
            return build(parse(left), parse(right))
    raise ValueError(text)


CASES: list[tuple[str, str, str, str, str]] = [
    ('(-cos(x/2) + cos(3*x))/x', 'x', '0', 'oo', 'None'),
    ('(1 - cos(x))/x**2', 'x', '-oo', 'oo', 'None'),
    ('(cos(x) - cos(2*x))/x', 'x', '0', 'oo', 'None'),
    ('135135*sin(x/13)*sin(x/11)*sin(x/9)*sin(x/7)*sin(x/5)*sin(x/3)*sin(x)/x**7', 'x', '0', 'oo', 'None'),
    ('15*sin(x/5)*sin(x/3)*sin(x)/x**3', 'x', '0', 'oo', 'None'),
    ('2027025*sin(x/15)*sin(x/13)*sin(x/11)*sin(x/9)*sin(x/7)*sin(x/5)*sin(x/3)*sin(x)/x**8', 'x', '0', 'oo', 'None'),
    ('cos(x)/x', 'x', '0', 'oo', 'None'),
    ('exp(-x)/x', 'x', '0', 'oo', 'None'),
    ('exp(2*I*x)*sin(x)/x', 'x', '-oo', '0', 'None'),
    ('exp(2*I*x)*sin(x)/x', 'x', '0', 'oo', 'None'),
    ('exp(I*x)*sin(x)/x', 'x', '-oo', 'oo', 'None'),
    ('sin(a*x)/x', 'x', '0', 'oo', 'a < 0'),
    ('sin(a*x)/x', 'x', '0', 'oo', 'a > 0'),
    ('sin(x)**2*cos(x)/x**2', 'x', '0', 'oo', 'None'),
    ('sin(x)**2/(3*x**2)', 'x', '0', 'oo', 'None'),
    ('sin(x)**2/x**2', 'x', '0', 'oo', 'None'),
    ('sin(x)**2/x', 'x', '-oo', 'oo', 'None'),
    ('sin(x)**2/x', 'x', '0', 'oo', 'None'),
    ('sin(x)**3/x**2', 'x', '0', 'oo', 'None'),
    ('sin(x)**3/x**3', 'x', '0', 'oo', 'None'),
    ('sin(x)**4/x**4', 'x', '0', 'oo', 'None'),
    ('sin(x)*sin(2*x)/x**2', 'x', '0', 'oo', 'None'),
    ('sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/(105*x**4)', 'x', '-oo', 'oo', 'None'),
    ('sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/x**4', 'x', '-oo', '0', 'None'),
    ('sin(x)*sin(3*x)*sin(5*x)*sin(7*x)/x**4', 'x', '0', 'oo', 'None'),
    ('sin(x)/(x**2 + 1)', 'x', '0', 'oo', 'None'),
    ('sin(x)/sqrt(x)', 'x', '0', 'oo', 'None'),
    ('sin(x)/x**2', 'x', '0', 'oo', 'None'),
    ('sin(x)/x', 'x', '-oo', 'oo', 'None'),
    ('sin(x)/x', 'x', '0', 'oo', 'None'),
    ('sin(x)/x', 'x', '1', 'oo', 'None'),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_dirichlet_integral(i: int) -> None:
    text, variable, lower, upper, assumptions = CASES[i]
    x = Symbol(variable)
    f, a, b = parse(text), parse(lower), parse(upper)
    facts = [] if assumptions == 'None' else [relation(assumptions)]

    def python() -> object:
        found = dirichlet_integral(f, x, a, b, facts or None)
        return None if found is None else Tuple(found.value, found.condition)
    got = outcome(lambda: rc.logic_call('dirichlet', [f, x, a, b, Tuple(*facts)]))
    if got != 'SKIP':
        assert got == outcome(python), text
