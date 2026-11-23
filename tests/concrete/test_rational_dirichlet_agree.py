"""The Rust ports of :mod:`sympy_extras.concrete.rational` and
:mod:`sympy_extras.concrete.dirichlet` against the Python ones on the
calls of the Python tests of the modules (recorded in
``rational_dirichlet_cases.json``, read back by ``eval`` in the core's
namespace with the integers exact; the symbols are plain)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete import dirichlet, rational
from sympy_extras.core import Integer, Symbol, Tuple
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
_NS.update(Integer=Integer)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'rational_dirichlet_cases.json')) as _fh:
    CASES: list[list[Any]] = json.load(_fh)


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
    if isinstance(value, rational.RationalDecomposition):
        return _norm(Tuple(value.g, value.h))
    if isinstance(value, tuple):
        return _norm(Tuple(*value))
    return _norm(value)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_rational_dirichlet(case: int) -> None:
    name, f_text, spec_text, lower_text, special_values = CASES[case]
    f, spec, lower = _read(f_text), _read(spec_text), _read(lower_text)
    flag = Symbol('special' if special_values else 'nospecial')
    if name == 'rational_sum':
        k, a, b = spec
        found = _outcome(lambda: rc.logic_call('rational', [Symbol(name), f, k, a, b, flag]))
        expected = _outcome(lambda: rational.rational_sum(f, spec, special_values))
    elif name == 'abramov_decomposition':
        found = _outcome(lambda: rc.logic_call('rational', [Symbol(name), f, spec]))
        expected = _outcome(lambda: rational.abramov_decomposition(f, spec))
    else:
        found = _outcome(lambda: rc.logic_call('rational', [Symbol(name), f, spec, lower, flag]))
        expected = _outcome(lambda: dirichlet.dirichlet_series(f, spec, lower, special_values))
    assert found == expected
