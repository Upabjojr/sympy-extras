"""The Rust port of ``simplify/hyperexpand.py`` against the Python one on
the functions of its tests (the numbers of the dummies are not compared)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from sympy_extras.core import Symbol
from sympy_extras.core.logic import false, true
from sympy_extras.core.parsing import parse
from sympy_extras.functions.special.hyper import hyper, meijerg
from sympy_extras.simplify.hyperexpand import hyperexpand

from ..conftest import load

rc = load("sympy_extras_rs.core")


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return re.sub(r'\b_?([a-zA-Z]+)\d+\b', r'\1', str(r))


def split(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current = ''
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return [p.strip() for p in parts]


def read(text: str) -> Any:
    text = text.strip()
    if text.startswith('hyper(') and text.endswith(')'):
        return hyper(*[read(p) for p in split(text[len('hyper('):-1])])
    if text.startswith('meijerg(') and text.endswith(')'):
        a, b, z = [read(p) for p in split(text[len('meijerg('):-1])]
        return meijerg(a[0], a[1], b[0], b[1], z)
    if text.startswith('(') and text.endswith(')'):
        return tuple(read(p) for p in split(text[1:-1]))
    return parse(text)


CASES: list[tuple[str, bool, str, str]] = [
    ('meijerg(((1,), ()), ((), ()), z)', False, 'default', 'None'),
    ('meijerg(((), (1,)), ((0,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((1/2,), (0,)), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0,), (1/2,)), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((a,), (-a,)), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((a,), (a + 1/2,)), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((1, 1), (0, 0)), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((a,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0, 1/2), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0, 1/2), ()), z**2/4)', False, 'default', 'None'),
    ('meijerg(((), ()), ((a, b), ()), z)', False, 'default', 'None'),
    ('meijerg(((), (1,)), ((a, b), ()), z)', False, 'default', 'None'),
    ('meijerg(((1 - a,), ()), ((0,), (-a,)), z)', False, 'default', 'None'),
    ('meijerg(((a,), ()), ((b,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), (a,)), ((b,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0, 0), ()), z)', False, 'default', 'None'),
    ('meijerg(((), (1/2,)), ((0,), ()), z)', False, 'default', 'None'),
    ('meijerg(((), ()), ((-1/2, 1/2), ()), z)', False, 'default', 'None'),
    ('meijerg(((a,), (a,)), ((b,), (b,)), z)', False, 'default', 'None'),
    ('hyper((nu/2 + 1/2, nu/2 + 1), (nu + 1,), -w)', False, 'default', 'None'),
    ('hyper((1/3,), (4/3,), z)', False, 'default', 'None'),
    ('hyper((4/3,), (7/3,), z)', False, 'default', 'None'),
    ('hyper((), (), -z)', False, 'default', 'None'),
    ('hyper((), (1,), -z)', False, 'default', 'None'),
    ('hyper((1,), (), -a**2/b**2)', False, 'default', 'None'),
    ('hyper((1,), (), -Abs(k)**2/a**2)', False, 'default', 'None'),
    ('hyper((1,), (), -a**2/Abs(k)**2)', False, 'default', 'None'),
    ('hyper((1,), (), -k**2/a**2)', False, 'default', 'None'),
    ('meijerg(((1/2,), (1/2, 1/2, 1/2)), ((0, 0, 1/2, 1/2), ()), a**2/4)', False, 'default', 'None'),
    ('meijerg(((1/2,), (1/2, 1)), ((0, 1/2, 1), ()), s)', False, 'default', 'None'),
    ('meijerg(((1/2,), (1/2, 1)), ((0, 1/2, 1), ()), (-a + s)/a)', False, 'default', 'None'),
    ('meijerg(((), ()), ((0, n), ()), a*s)', False, 'default', 'None'),
    ('meijerg(((0, 1/10), (3/10,)), ((1/3,), (-1, 2/7)), x)', False, 'default', '0'),
    ('meijerg(((0, 1/10), (3/10,)), ((1/3,), (-1, 2/7)), 2*x)', False, 'default', '0'),
    ('meijerg(((1/10, 1/2), (3/10,)), ((1/3,), (-1/2, 2/7)), x**2)', False, 'default', '0'),
    ('meijerg(((1/10, 1/3), (3/10,)), ((1/3,), (-2/3, 2/7)), x**(3/2))', False, 'default', '0'),
    ('meijerg(((-3, 1/10), (3/10,)), ((1/3,), (-4, 2/7)), x)', False, 'default', '0'),
    ('meijerg(((-1/14, 1/10), (3/10,)), ((1/3,), (-15/14, 2/7)), 4*x**(7/3))', False, 'default', '0'),
    ('meijerg(((0,), ()), ((0,), (-1, 0)), x**2/4)', False, 'default', '0'),
    ('meijerg(((0,), (1/2,)), ((0,), (-1, 0, 1/2)), x**2/4)', False, 'default', '0'),
    ('meijerg(((1/2,), ()), ((1/2,), (-1/2, -1/2)), x**2/4)', False, 'default', '0'),
    ('meijerg(((0, 1), (1, 1, 2)), ((1/2, 1), (0,)), x**(-2))', False, 'default', 'None'),
    ('meijerg(((1/2, 1), (1, 1, 1)), ((1/2, 1), (-1/2,)), x**(-2))', False, 'default', 'None'),
    ('meijerg(((1/2, 1), (1/2, 3/2, 3/2)), ((1/2, 1), (0,)), x**(-2))', False, 'default', 'None'),
    ('meijerg(((1/2,), ()), ((1/2,), (-1/2, -1/2)), x**4/4)', False, 'default', '0'),
    ('meijerg(((5/6, 1), ()), ((1/2,), (-1/6, 0)), x**6)', False, 'default', '0'),
    ('meijerg(((1/2,), ()), ((1/2,), (-1/2, 0)), x**2/4)', False, 'default', '0'),
    ('meijerg(((1/2, 1), ()), ((1/2,), (-1/2, 0, 0)), x**2/4)', False, 'default', '0'),
    ('meijerg(((3/4,), ()), ((1/2,), (-1/4, 0)), pi**2*x**4/16)', False, 'default', '0'),
    ('meijerg(((3/4,), ()), ((0,), (-1/4, 1/2)), pi**2*x**4/16)', False, 'default', '0'),
    ('meijerg(((-1, 3/2), ()), ((0,), (-2,)), 2*x/3)', False, 'default', '0'),
    ('hyper((-n,), (), -1)', False, 'default', 'None'),
    ('hyper((1, 1), (n + 2,), -1)', False, 'default', 'None'),
    ('hyper((1 - n,), (), -1)', False, 'default', 'None'),
    ('hyper((1, 1), (n + 1,), -1)', False, 'default', 'None'),
    ('hyper((), (), x)', False, 'default', 'None'),
    ('hyper((1,), (), 1/2)', False, 'default', 'None'),
    ('hyper((1, 1), (), 1)', False, 'default', 'None'),
    ('hyper((1, n + 2), (), 1)', False, 'default', 'None'),
    ('hyper((1, 1), (2,), -1)', False, 'default', 'None'),
    ('hyper((1/2, 1), (3/2,), 1)', False, 'default', 'None'),
    ('hyper((2, 1 - n), (1,), -1)', False, 'default', 'None'),
    ('hyper((1, 1, n + 2), (n + 1, n + 1), -1)', False, 'default', 'None'),
    ('hyper((-m, -m), (1,), 1)', False, 'default', 'None'),
    ('hyper((1/2, 1), (3/2,), -1)', False, 'default', 'None'),
    ('hyper((2,), (), x)', False, 'default', 'None'),
    ('hyper((), (1/2,), x**2/4)', False, 'default', 'None'),
    ('hyper((a,), (), x)', False, 'default', 'None'),
    ('hyper((-n, -n), (1,), 1)', False, 'default', 'None'),
    ('hyper((1, 1, 1), (n + 2, n + 2), 1)', False, 'default', 'None'),
    ('hyper((1,), (2,), 1)', False, 'default', 'None'),
    ('hyper((1,), (3,), x)', False, 'default', 'None'),
    ('hyper((1, 1), (2,), x)', False, 'default', 'None'),
    ('hyper((1/2,), (), 4*x)', False, 'default', 'None'),
    ('hyper((0, 1), (), -1/2)', False, 'default', 'None'),
    ('hyper((1,), (n + 2,), x)', False, 'default', 'None'),
    ('hyper((1,), (), 1/x)', False, 'default', 'None'),
    ('hyper((1,), (), x)', False, 'default', 'None'),
    ('hyper((), (1,), 1)', False, 'default', 'None'),
    ('hyper((-k,), (), -1)', False, 'default', 'None'),
    ('hyper((1 - k,), (), -1)', False, 'default', 'None'),
    ('hyper((2, 1 - k), (1,), -1)', False, 'default', 'None'),
    ('hyper((2, 2, 1 - k), (1, 1), -1)', False, 'default', 'None'),
    ('hyper((2, 2, 2, 1 - k), (1, 1, 1), -1)', False, 'default', 'None'),
]


@pytest.mark.parametrize('i', range(len(CASES)))
def test_hyperexpand(i: int) -> None:
    text, allow_hyper, rewrite, place = CASES[i]
    e: Any = read(text)
    where: Any = None if place == 'None' else parse(place)
    got = outcome(lambda: rc.logic_call('hyperexpand', [e, true if allow_hyper else false, Symbol(rewrite),
                                                       Symbol('NONE') if where is None else where]))
    if got != 'SKIP':
        assert got == outcome(lambda: hyperexpand(e, allow_hyper, None if rewrite == 'None' else rewrite, where)), text
