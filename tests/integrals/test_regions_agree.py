"""The Rust port of integrals/regions.py and integrals/axisymmetric.py
against the Python ones: the region integrals of the Python tests (the
decomposition, the polar, scaled and cylindrical routes, the bounds solved
for the last variable, polytopes, symbolic dimensions, the Hausdorff
measure), the cases read by ``eval`` in the core's namespace with the
integers exact, the fresh dummies numbered by their first appearance."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.integrals.regions import integrate_by_ranges

from ..conftest import load

rc = load("sympy_extras_rs.core")


class _Namespace(dict):  # type: ignore[type-arg]
    def __missing__(self, name: str) -> Symbol:
        return Symbol(name)


_NS = _Namespace()
for _mod in (core, functions):
    for _k in dir(_mod):
        if not _k.startswith('_'):
            _NS[_k] = getattr(_mod, _k)
_NS.update(Integer=Integer, Eq=core.eq, a=Symbol('a', positive=True), b=Symbol('b', positive=True),
           c=Symbol('c', positive=True), R=Symbol('R', positive=True), h=Symbol('h', positive=True))


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _renumbered(text: str) -> str:
    names: dict[str, str] = {}
    return re.sub(r'_[A-Za-z]+\d+_?', lambda g: names.setdefault(g.group(0), '_d%d' % len(names)), text)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return _renumbered(str(r))


#: (integrand, condition, options): variables, assumptions, measure,
#: dimension, parameters as texts
CASES: list[tuple[str, str, dict[str, str]]] = [
    ('x*y', '(0 < x) & (x < y) & (y < 1)', {}),
    ('x*y', '(x > 0) & (y > 0) & (x + y < 1)', {}),
    ('1', '(0 < x) & (x < y) & (y < x**2) & (x < 2)', {}),
    ('exp(x)', '(0 < x) & (x < 1) & (0 < y) & (y < 1)', {}),
    ('x', 'Eq(x**2 + y**2, 1)', {}),
    ('1', 'x**2 + y**2 < 0', {}),
    ('1', '(x > 1) & (x < 0) & (y > 0) & (y < 1)', {}),
    ('1', 'x**2 + y**2 < r**2', {'variables': '[x, y]'}),
    ('1', 'x**2 + y**2 < r**2', {'variables': '[x, y]', 'assumptions': '[r > 0]'}),
    ('1', 'x**2 + y**2 < r**2', {'variables': '[x, y]', 'assumptions': '[r < 0]'}),
    ('1', 'x**2 + y**2 < r**2', {'parameters': '[r]'}),
    ('1', '(x > 0) & (x < r)', {'parameters': '[r]'}),
    ('1', '(0 < x) & (x < y) & (y < x**2)', {}),
    ('1', 'exp(x) < y', {}),
    ('exp(-x**2 - y**2)', 'x**2 + y**2 < 1', {}),
    ('1/(1 + x**2 + y**2)', 'x**2 + y**2 < 3', {}),
    ('1', '(x**2 + y**2 > 1) & (x**2 + y**2 < 4)', {}),
    ('x**2', 'x**2 + y**2 < 1', {}),
    ('1', '(x > 0) & (x < 1) & (y > 0) & (y < exp(x))', {}),
    ('1', '(x > 1) & (x < 2) & (y > 0) & (y < log(x))', {}),
    ('exp(-y)', '(x > 0) & (x < 1) & (y > x)', {}),
    ('1', '(x > 0) & (x < 2) & (y > x) & (y < 1)', {}),
    ('1', '(x > 0) & (x < 1) & (y > 0) & (y*exp(x) < 1)', {}),
    ('1', '(x > 0) & (y > 0) & (y < exp(x))', {}),
    ('1', '(x > 1) & (x < 2) & (y > 0) & (exp(y) < x)', {}),
    ('1', '(x > 0) & (x < 1) & (y > 0) & (y**2 < x)', {}),
    ('1', '(x > 0) & (x < 1) & (y > 0) & (y**2 < exp(x))', {}),
    ('1', '(x > 0) & (x < 1) & (y > 0) & (y**3 < x)', {}),
    ('1', 'x**2/a**2 + y**2/b**2 < 1', {'variables': '[x, y]'}),
    ('x**2', 'x**2/a**2 + y**2/b**2 < 1', {'variables': '[x, y]'}),
    ('1', 'x**2/a**2 + y**2/b**2 + z**2/c**2 < 1', {'variables': '[x, y, z]'}),
    ('1', '4*x**2 + 9*y**2 < 36', {}),
    ('1', 'x**2 + y**2 - 2*x < 3', {}),
    ('1', '(x**2/a**2 + y**2/b**2 > 1) & (x**2/a**2 + y**2/b**2 < 4)', {'variables': '[x, y]'}),
    ('x*y', 'x**2/4 + y**2 < 1', {}),
    ('1', '(x**2 + y**2 < R**2) & (z > 0) & (z < h)', {'variables': '[x, y, z]'}),
    ('1', '(x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2)', {}),
    ('z', '(sqrt(x**2 + y**2) < z) & (z < 1)', {}),
    ('1', '(x**2 + y**2 < z) & (z < 4)', {}),
    ('x**2 + y**2', '(x**2 + y**2 < 1) & (z > 0) & (z < 2)', {}),
    ('exp(-x - y)', '(0 < x) & (0 < y) & (y < x)', {}),
    ('1', '(x > 0) & (y > 0) & (y < exp(-x))', {}),
    ('1', '(x > 1) & (0 < y) & (y < 1/x**2)', {}),
    ('exp(-x**2)', '(y > 0) & (y < 1)', {'variables': '[x, y]'}),
    ('1', '(x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0)', {'variables': '[y, x]'}),
    ('1', '(x**2 + y**2 + z**2 < 1) & (x**2 + y**2 < 1/4)', {}),
    ('1', '(x**2 + y**2 + z**2 < 1) & (x + y + z > 1)', {}),
    ('z', '(x**2 + y**2 + z**2 < 1) & (z > x**2 + y**2)', {}),
    ('1', 'x**2 + y**2 + z**2 + w**2 < 1', {}),
    ('1', 'r < 1', {'dimension': 'n'}),
    ('exp(-r**2)', 'true', {'dimension': '3', 'variables': '[r]'}),
    ('r**2', '(r > 1) & (r < 2)', {'dimension': '2'}),
    ('1', 'Eq(x**2 + y**2, 1)', {'measure': 'hausdorff'}),
    ('1', 'Eq(y, x**2) & (x > 0) & (x < 1)', {'measure': 'hausdorff'}),
    ('y', 'Eq(x**2 + y**2, 1) & (y > 0)', {'measure': 'hausdorff'}),
    ('1', 'Eq(y, 2*x) & (x > 0) & (x < 1)', {'measure': 'hausdorff'}),
    ('1', 'Eq(x, 1) & (y > 0) & (y < 2)', {'measure': 'hausdorff'}),
    ('x + y', 'Eq(x, 1) & (y > 0) & (y < 2)', {'measure': 'hausdorff'}),
    ('1', 'Eq(z, x**2 + y**2) & (z < 1)', {'measure': 'hausdorff'}),
    ('x**2', 'Eq(x**2, 4)', {'measure': 'hausdorff'}),
    ('1', 'x**2 + y**2 < 1', {'measure': 'hausdorff'}),
    ('1', 'Eq(x**2 + y**2 + z**2, 1)', {'measure': 'hausdorff'}),
    ('1', 'Eq(x**2 + y**2 + z**2, 1) & Eq(z, 0)', {'measure': 'hausdorff'}),
    ('1', 'Eq(z, x**2 + y**2) & Eq(z, 1)', {'measure': 'hausdorff'}),
    ('1', 'x', {}),
]


def _options(o: dict[str, str]) -> dict[str, Any]:
    return {k: _read(v) for k, v in o.items()}


def python(f: str, condition: str, o: dict[str, str]) -> object:
    opts = _options(o)
    return integrate_by_ranges(_read(f), _read(condition), opts.get('variables'), opts.get('assumptions'),
                               measure=o.get('measure', 'lebesgue'), dimension=opts.get('dimension'),
                               parameters=opts.get('parameters'))


def rust(f: str, condition: str, o: dict[str, str]) -> object:
    opts = _options(o)
    none = Symbol('NONE')
    args = [Symbol('regions'), _read(f), _read(condition),
            Tuple(*opts['variables']) if 'variables' in opts else none,
            Tuple(*opts.get('assumptions', [])), Symbol(o.get('measure', 'lebesgue')),
            opts.get('dimension', none), Tuple(*opts['parameters']) if 'parameters' in opts else none]
    return rc.logic_call('calculus', args)


@pytest.mark.parametrize('i', range(len(CASES)))
def test_regions(i: int) -> None:
    f, condition, o = CASES[i]
    expected = outcome(lambda: python(f, condition, o))
    got = outcome(lambda: rust(f, condition, o))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (f, condition, o)
