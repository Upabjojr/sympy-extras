"""The Rust port of ``integrals/elliptic.py`` against the Python one on
the integrands of its tests (read with the core's parser; the numbers of
the dummies inside ``IndexedRoot`` are not compared)."""
from __future__ import annotations

import re
from typing import Callable

import pytest

from sympy_extras.core import Ge, Gt, Le, Lt, Symbol, Tuple
from sympy_extras.core.logic import Boolean, to_boolean
from sympy_extras.core.parsing import parse
from sympy_extras.integrals.elliptic import elliptic_integral, radicand, real_roots

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return 'None' if r is None or str(r) == 'NONE' else re.sub(r'\b([a-z]+)\d+\b', r'\1', str(r))


def relation(text: str) -> Boolean:
    for op, build in ((' >= ', Ge), (' <= ', Le), (' > ', Gt), (' < ', Lt)):
        if op in text:
            left, right = text.split(op)
            return to_boolean(build(parse(left), parse(right)))
    raise ValueError(text)


def facts(texts: list[str]) -> list[Boolean]:
    return [relation(part.strip().strip('()')) for text in texts for part in text.split(' & ')]


CASES: list[tuple[str, str, str, str, str, list[str]]] = [
    ('integral', '1/sqrt(1 - x**4)', 'x', '0', '1', []),
    ('integral', 'x**2/sqrt(1 - x**4)', 'x', '0', '1', []),
    ('integral', '1/sqrt((1 - x**2)*(-k**2*x**2 + 1))', 'x', '0', '1', ['k < 1']),
    ('integral', '1/sqrt(x*(1 - x)*(-k**2*x + 1))', 'x', '0', '1', ['k < 1']),
    ('integral', '1/sqrt(x**3 - x)', 'x', '1', 'oo', []),
    ('integral', '1/sqrt(x*(2 - x)*(3 - x)*(x - 1))', 'x', '1', '2', []),
    ('integral', '1/sqrt((-b + x)*(q**2 + (-p + x)**2)*(b + d - x))', 'x', 'b', 'b + d', []),
    ('integral', '1/sqrt((-a + x)*(q**2 + (-p + x)**2))', 'x', 'a', 'oo', []),
    ('integral', '1/sqrt((a - x)*(q**2 + (-p + x)**2))', 'x', '-oo', 'a', []),
    ('integral', '1/sqrt(x**4 + 1)', 'x', '0', 'oo', []),
    ('integral', '1/sqrt(x**3 + 1)', 'x', '0', 'oo', []),
    ('integral', '1/sqrt(x**3 + 1)', 'x', '-1', '0', []),
    ('integral', '1/sqrt(x**3 + 1)', 'x', '-1', 'oo', []),
    ('integral', '1/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2', '3', []),
    ('integral', '1/sqrt((x**2 + 1)*(x**2 + 4))', 'x', '-oo', 'oo', []),
    ('integral', '1/sqrt((x**2 + 1)*(x**2 + 4))', 'x', '0', '1', []),
    ('integral', '1/sqrt((x**2 - x + 2)*(x**2 + x + 1))', 'x', '-oo', 'oo', []),
    ('integral', '1/(sqrt((x**2 + 1)*(x**2 + 4))*(x**2 + 2))', 'x', '0', 'oo', []),
    ('integral', '1/sqrt(x**4 + 4*x**2 + 5)', 'x', '0', 'oo', []),
    ('radicand', 'x/sqrt((1 - x**2)*(4 - x**2))', 'x', '', '', []),
    ('radicand', '1/sqrt(x**5 - 1)', 'x', '', '', []),
    ('real_roots', 'x**3 - x', 'x', '', '', []),
    ('real_roots', '(1 - x**2)*(-k**2*x**2 + 1)', 'x', '', '', ['k < 1']),
    ('real_roots', 'x**3 + 1', 'x', '', '', []),
    ('real_roots', 'x**3 - 3*x + 1', 'x', '', '', []),
    ('integral', '1/sqrt(x*(1 - x)*(4 - x))', 'x', '0', '1/2', []),
    ('integral', 'x**2/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2', '3', []),
    ('integral', '1/(sqrt((3 - x)*(x + 2)*(x**2 + 1))*(x - 5))', 'x', '-2', '1', []),
    ('integral', '1/sqrt(x**3 + 1)', 'x', '-1', '2', []),
    ('integral', '1/sqrt((x**2 + 1)*(x**2 + 4))', 'x', '-3', '2', []),
    ('integral', 'sqrt(-k**2*sin(theta)**2 + 1)', 'theta', '0', 'pi/2', ['k < 1']),
    ('integral', '1/sqrt(-k**2*sin(theta)**2 + 1)', 'theta', '0', 'pi/3', ['k < 1']),
    ('integral', 'sin(theta)/sqrt(-k**2*sin(theta)**2 + 1)', 'theta', '0', 'pi/2', ['k < 1']),
    ('integral', '1/sqrt(x*(1 - x)*(4 - x))', 'x', '0', '2', []),
    ('integral', 'x/sqrt(x**3 + 1)', 'x', '0', 'oo', []),
    ('integral', '1/sqrt((-x**2 - 1)*(x**2 + 4))', 'x', '0', '1', []),
    ('integral', '1/sqrt(x**5 + 1)', 'x', '0', '1', []),
    ('integral', 'x/sqrt((a - x)*(-b + x)*(q**2 + (-p + x)**2))', 'x', 'b', 'a', ['b < a', 'q > 0']),
    ('radicand', 'sqrt(x)*sqrt(1 - x)/sqrt(4 - x)', 'x', '', '', []),
    ('radicand', 'x**(3/2)/sqrt((1 - x)*(4 - x))', 'x', '', '', []),
    ('radicand', '1/sqrt(1 - x**2)', 'x', '', '', []),
    ('radicand', '1/sqrt(sin(x))', 'x', '', '', []),
    ('radicand', '1/(x**3 - x)', 'x', '', '', []),
    ('real_roots', 'x**2*(x - 1)', 'x', '', '', []),
    ('real_roots', 'x*(-k + x)*(x - 1)', 'x', '', '', []),
    ('integral', '1/sqrt(x*(1 - x)*(4 - x))', 'x', '0', '1', []),
    ('integral', 'x/sqrt(x*(1 - x)*(-k**2*x + 1))', 'x', '0', '1', ['k < 1']),
    ('integral', '1/sqrt(-x**3 + x)', 'x', '-oo', '-1', []),
    ('integral', '1/sqrt(x*(1 - x)*(4 - x))', 'x', '1', '4', []),
    ('integral', '1/(sqrt(x*(1 - x)*(4 - x))*(x + 2))', 'x', '0', '1', []),
    ('integral', 'sqrt(x*(1 - x)*(4 - x))', 'x', '0', '1', []),
    ('integral', '1/sqrt((1 - x)*(x - 4)*(x - 3)*(x - 2))', 'x', '1', '2', []),
    ('integral', '1/sqrt(x*(x - 3)*(x - 2)*(x - 1))', 'x', '3', 'oo', []),
    ('integral', 'x**2/sqrt((1 - x**2)*(-k**2*x**2 + 1))', 'x', '0', '1', ['k < 1']),
    ('integral', 'sqrt(1 - x**4)', 'x', '0', '1', []),
    ('integral', 'x/sqrt(1 - x**4)', 'x', '0', '1', []),
    ('integral', '1/sqrt(-k**2*sin(theta)**2 + 1)', 'theta', '0', 'pi/2', ['k < 1']),
    ('integral', 'sin(theta)**2/sqrt(-k**2*sin(theta)**2 + 1)', 'theta', '0', 'pi/2', ['k < 1']),
    ('integral', '1/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2', '1', []),
    ('integral', '1/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '0', '3', []),
    ('integral', '1/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '0', '1', []),
    ('integral', '1/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-3/2', '-1/2', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2', '3', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '0', '1', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-3/2', '0', []),
    ('integral', '(x**2 - 3*x + 1)/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2', '3', []),
    ('integral', '1/sqrt((x - 3)*(x + 2)*(x**2 + 1))', 'x', '-2', '3', []),
    ('integral', '1/sqrt((a - x)*(-b + x)*(q**2 + (-p + x)**2))', 'x', 'b', 'a', ['(q > 0) & (b < a)']),
    ('integral', 'x/sqrt((a - x)*(-b + x)*(q**2 + (-p + x)**2))', 'x', 'b', 'a', ['(q > 0) & (b < a)']),
    ('integral', '1/sqrt(x**3 + 1)', 'x', '2', '5', []),
    ('integral', '1/sqrt(1 - x**3)', 'x', '-oo', '1', []),
    ('integral', '1/sqrt(1 - x**3)', 'x', '-3', '1', []),
    ('integral', '1/sqrt(1 - x**3)', 'x', '-oo', '-2', []),
    ('integral', 'sqrt(x**3 + 1)', 'x', '-1', '0', []),
    ('integral', 'sqrt(x**3 + 1)', 'x', '0', '2', []),
    ('integral', 'x/sqrt(x**3 + 1)', 'x', '-1', '1', []),
    ('integral', 'x**2/sqrt(x**3 + 1)', 'x', '-1', '3', []),
    ('integral', '1/sqrt(a**3 + x**3)', 'x', '0', 'oo', []),
    ('integral', '1/sqrt(a**3 + x**3)', 'x', '-a', '0', []),
    ('integral', '1/sqrt(x**4 + 1)', 'x', '0', '1', []),
    ('integral', '1/sqrt(x**4 + x**2 + 1)', 'x', '0', 'oo', []),
    ('integral', '1/sqrt((1 - x**2)*(k**2 + x**2))', 'x', '-1', '1', []),
    ('integral', 'sqrt(x**3 + 1)', 'x', '-1', 'oo', []),
    ('integral', '1/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '3', 'oo', []),
    ('integral', '1/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '-oo', '2', []),
    ('integral', '1/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '4', 'oo', []),
    ('integral', '1/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '3', '4', []),
    ('integral', '1/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '-oo', '0', []),
    ('integral', 'x/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '3', '5', []),
    ('integral', '1/(x*sqrt((x - 3)*(x - 2)*(x**2 + 1)))', 'x', '3', 'oo', []),
    ('integral', 'x/sqrt((x - 3)*(x - 2)*(x**2 + 1))', 'x', '4', 'oo', []),
    ('integral', '1/(sqrt((3 - x)*(x + 2)*(x**2 + 1))*(x - 5))', 'x', '-2', '3', []),
    ('integral', '1/(sqrt((3 - x)*(x + 2)*(x**2 + 1))*(x + 5))', 'x', '-2', '1', []),
    ('integral', '1/((x + 1)*sqrt(x**3 + 1))', 'x', '0', 'oo', []),
    ('integral', 'x/((x + 1)*sqrt(x**3 + 1))', 'x', '0', 'oo', []),
    ('integral', '1/((x + 2)*sqrt(x**3 + 1))', 'x', '0', 'oo', []),
    ('integral', '1/((x + 2)**2*sqrt(x**3 + 1))', 'x', '0', 'oo', []),
    ('integral', '1/(sqrt((3 - x)*(x + 2)*(x**2 + 1))*(x - 1))', 'x', '0', '2', []),
    ('integral', '1/(x**2*sqrt(x**3 + 1))', 'x', '-1', '1', []),
    ('integral', '1/(sqrt((3 - x)*(x + 2)*(x**2 + 1))*(x - 1)**2)', 'x', '-2', '0', []),
    ('integral', '1/(x**2*sqrt(x**3 + 1))', 'x', '1', 'oo', []),
    ('integral', '1/sqrt((1 - x**2)*(c + x**2))', 'x', '-1', '1', ['c > 0']),
    ('integral', '1/sqrt((1 - x**2)*(c + x**2))', 'x', '-1', '1', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '-2/5', '1/2', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '2/5', '12/5', []),
    ('integral', 'x/sqrt((3 - x)*(x + 2)*(x**2 + 1))', 'x', '11/10', '3', []),
    ('integral', 'x/sqrt((x**2 + 1)*(x**2 + 4))', 'x', '-3', '2', []),
    ('integral', 'x**2/sqrt((x**2 - x + 2)*(x**2 + x + 1))', 'x', '-1', '3', []),
    ('integral', '1/(sqrt((x**2 + 1)*(x**2 + 4))*(x - 5))', 'x', '-oo', '2', []),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_elliptic(i: int) -> None:
    kind, text, variable, lower, upper, given = CASES[i]
    x = Symbol(variable)
    f = parse(text)
    assumptions = facts(given)
    if kind == 'integral':
        a, b = parse(lower), parse(upper)

        def python() -> object:
            found = elliptic_integral(f, x, a, b, assumptions or None)
            return None if found is None else Tuple(found.value, found.condition)
        got = outcome(lambda: rc.logic_call('elliptic', [Symbol('integral'), f, x, a, b, Tuple(*assumptions)]))
    elif kind == 'real_roots':
        def python() -> object:
            found = real_roots(f, x, assumptions or None)
            return None if found is None else Tuple(*found)
        got = outcome(lambda: rc.logic_call('elliptic', [Symbol('real_roots'), f, x, Tuple(*assumptions)]))
    else:
        def python() -> object:
            found = radicand(f, x)
            return None if found is None else Tuple(*found)
        got = outcome(lambda: rc.logic_call('elliptic', [Symbol('radicand'), f, x]))
    if got != 'SKIP':
        assert got == outcome(python), text
