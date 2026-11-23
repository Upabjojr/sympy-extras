"""The Rust port of :mod:`sympy_extras.concrete.zeilberger` against the
Python one on the calls of the Python tests of the module (recorded in
``zeilberger_cases.json``, read back by ``eval`` in the core's namespace
with the integers exact; the symbols are plain). A ``Telescoper`` is
compared by its coefficients and certificate; the numbers of the dummies
are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete import zeilberger as zb
from sympy_extras.concrete.expr_with_limits import Product, Sum
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
_NS.update(Integer=Integer, Sum=Sum, Product=Product)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'zeilberger_cases.json')) as _fh:
    CASES: list[list[str]] = json.load(_fh)


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
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, zb.Telescoper):
        return _norm(Tuple(Tuple(*value.coefficients), value.certificate))
    return _norm(value)


def _python(name: str, a1: Any, a2: Any, a3: Any, a4: Any, i5: int, i6: int) -> Any:
    if name == 'zeilberger':
        return zb.zeilberger(a1, a2, a3, max_order=i5, min_order=i6)
    if name == 'wz_certificate':
        return zb.wz_certificate(a1, a2, a3, a4)
    if name == 'wz_prove':
        return zb.wz_prove(a1, a2, a3, a4, i5)
    return zb.zeilberger_sum(a1, a2, a3, max_order=i5, special_values=bool(i6))


@pytest.mark.parametrize('case', range(len(CASES)))
def test_zeilberger(case: int) -> None:
    name, t1, t2, t3, t4, t5, t6 = CASES[case]
    a1, a2 = _read(t1), _read(t2)
    a3 = None if t3 == 'NONE' else _read(t3)
    a4 = None if t4 == 'NONE' else _read(t4)
    i5, i6 = int(t5), int(t6)
    none = Symbol('NONE')
    args = [Symbol(name), a1, Tuple(*a2) if isinstance(a2, tuple) else a2,
            none if a3 is None else a3, none if a4 is None else a4, Integer(i5), Integer(i6)]
    found = _outcome(lambda: rc.logic_call('zeilberger', args))
    expected = _outcome(lambda: _python(name, a1, a2, a3, a4, i5, i6))
    assert found == expected
