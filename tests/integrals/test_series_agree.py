"""The Rust port of :mod:`sympy_extras.integrals.series` against the
Python one on the calls of the Python tests of the module (recorded in
``series_cases.json``, read back by ``eval`` in the core's namespace with
the integers exact; the symbols are plain, their assumptions printed
away). A ``ConditionalValue`` is compared as its value and condition, an
``Expansion`` by its coefficient, basis, start, constant and condition;
the numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.integrals import series
from sympy_extras.integrals.conditions import ConditionalValue
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
_NS.update(Integer=Integer, Sum=Sum, Product=Product)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'series_cases.json')) as _fh:
    CASES: list[list[str]] = json.load(_fh)


def _renumbered(text: str) -> str:
    """The fresh dummies (``_k27``) numbered by their first appearance: the
    counters which name them run apart when other tests ran before."""
    names: dict[str, str] = {}
    return re.sub(r'_k\d+', lambda m: names.setdefault(m.group(0), '_k%d' % len(names)), text)


def _outcome(f: Any) -> str:
    try:
        value = f()
    except NotImplementedError as e:
        if 'not ported' in str(e):
            pytest.skip(str(e)[:100])
        return 'ERR NotImplementedError'
    except (TypeError, ValueError) as e:
        return 'ERR %s' % type(e).__name__
    if value is None:
        return 'NONE'
    if isinstance(value, ConditionalValue):
        return _norm(Tuple(value.value, value.condition))
    if isinstance(value, series.Expansion):
        return _norm(Tuple(value.coefficient, value.basis, Integer(value.start), value.constant, value.condition))
    return _norm(value)


def _python(name: str, f: Any, x: Any, a: Any, b: Any, assumptions: Any, extra: Any) -> Any:
    if name == 'sum_series':
        return series.sum_series(f, x, int(a), assumptions)
    if name == 'fourier_expansion':
        return series.fourier_expansion(f, x, a, b, assumptions, bool(extra))
    if name == 'fourier_integral':
        return series.fourier_integral(f, x, a, b, assumptions, extra)
    return series.series_integral(f, x, a, b, assumptions, extra)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_series(case: int) -> None:
    name, f_text, x_text, a_text, b_text, assumptions_text, extra_text = CASES[case]
    f, x, a, b = _read(f_text), _read(x_text), _read(a_text), _read(b_text)
    assumptions = None if assumptions_text == 'None' else _read(assumptions_text)
    extra = None if extra_text == 'None' else _read(extra_text)
    none = Symbol('NONE')
    args = [Symbol(name), f, x, a, b, Tuple() if assumptions is None else Tuple(assumptions),
            none if extra is None else Symbol('compute' if extra else 'nocompute') if isinstance(extra, bool) else extra]
    found = _renumbered(_outcome(lambda: rc.logic_call('series_integral', args)))
    expected = _renumbered(_outcome(lambda: _python(name, f, x, a, b, assumptions, extra)))
    assert found == expected
