"""The Rust port of :mod:`sympy_extras.concrete.creative` against the
Python one on the calls of the Python tests of the module (recorded in
``creative_cases.json``, read back by ``eval`` in the core's namespace
with the integers exact; the symbols are plain). A
``CreativeTelescoper`` is compared by its coefficients and certificate;
the numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete import creative
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


with open(os.path.join(os.path.dirname(__file__), 'creative_cases.json')) as _fh:
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
    if isinstance(value, creative.CreativeTelescoper):
        return _norm(Tuple(Tuple(*value.coefficients), value.certificate))
    return _norm(value)


def _python(name: str, f: Any, second: Any, third: Any, order: int, extensions: list[Any], special: bool) -> Any:
    if name == 'creative_telescoping':
        return creative.creative_telescoping(f, second, third, order, extensions)
    return creative.definite_sum(f, second, third, order, extensions, special)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_creative(case: int) -> None:
    name, f_text, second_text, third_text, order_text, extensions_text, special_text = CASES[case]
    f, second = _read(f_text), _read(second_text)
    third = None if third_text == 'NONE' else _read(third_text)
    extensions = list(_read(extensions_text))
    order, special = int(order_text), special_text == 'True'
    none = Symbol('NONE')
    args = [Symbol(name), f, Tuple(*second) if isinstance(second, tuple) else second,
            none if third is None else third, Integer(order), Tuple(*extensions),
            Symbol('special' if special else 'nospecial')]
    found = _outcome(lambda: rc.logic_call('creative', args))
    expected = _outcome(lambda: _python(name, f, second, third, order, extensions, special))
    assert found == expected
