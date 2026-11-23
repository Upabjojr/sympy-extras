"""The Rust port of :mod:`sympy_extras.solvers.recurr` against the Python
one on the calls of the Python tests of the module (recorded in
``recurr_cases.json``, read back by ``eval`` in the core's namespace with
the integers exact; the symbols are plain, ``y`` and ``S`` undefined
functions). The Gosper–Petkovšek forms are compared as expressions."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Function, Integer, Symbol, Tuple
from sympy_extras.polys.polytools import Poly
from sympy_extras.solvers import recurr
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
_NS.update(Integer=Integer, Sum=Sum, Product=Product, y=Function('y'), S=Function('S'))


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'recurr_cases.json')) as _fh:
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
    if isinstance(value, (list, tuple)):
        return _norm(Tuple(*[v.as_expr() if isinstance(v, Poly) else v for v in value]))
    return _norm(value)


def _python(name: str, first: Any, second: Any, third: Any, rhs: Any, start: int, count: int) -> Any:
    if name == 'rsolve_linear':
        return recurr.rsolve_linear(first, second, third)
    if name == 'is_recurrence_solution':
        return recurr.is_recurrence_solution(first, second, third, rhs, start, count)
    if name == 'gosper_form':
        return recurr.gosper_form(first, second)
    return getattr(recurr, name)(first, second)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_recurr(case: int) -> None:
    name, first_text, second_text, third_text, rhs_text, start_text, count_text = CASES[case]
    first, second, rhs = _read(first_text), _read(second_text), _read(rhs_text)
    third = None if third_text == 'NONE' else _read(third_text)
    start, count = int(start_text), int(count_text)
    args: list[Any] = [Symbol(name), Tuple(*first) if isinstance(first, list) else first, second]
    if third is None:
        args.append(Symbol('NONE'))
    elif isinstance(third, dict):
        args.append(Tuple(*[Tuple(k, v) for k, v in third.items()]))
    else:
        args.append(third)
    args += [rhs, Integer(start), Integer(count)]
    found = _outcome(lambda: rc.logic_call('recurr', args))
    expected = _outcome(lambda: _python(name, first, second, third, rhs, start, count))
    assert found == expected
