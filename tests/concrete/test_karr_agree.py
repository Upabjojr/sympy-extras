"""The Rust port of :mod:`sympy_extras.concrete.karr` against the Python
one on the calls of the Python tests of the module (recorded as printed
core expressions in ``karr_cases.json``, read back by ``eval`` in the
core's namespace with the integers exact; the symbols are plain). The
numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete import karr
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Integer, Symbol
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


with open(os.path.join(os.path.dirname(__file__), 'karr_cases.json')) as _fh:
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
    return _norm('NONE' if value is None else value)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_karr(case: int) -> None:
    name, f_text, spec_text, auto, special_values = CASES[case]
    f = _read(f_text)
    spec = _read(spec_text)
    if isinstance(spec, tuple):
        k, a, b = spec
        bounds: list[Any] = [a, b]
    else:
        k, bounds = spec, [Symbol('NONE'), Symbol('NONE')]
    flags = [Symbol('auto' if auto else 'noauto'), Symbol('special' if special_values else 'nospecial')]
    found = _outcome(lambda: rc.logic_call('karr', [Symbol(name), f, k] + bounds + flags))
    if name == 'summation':
        expected = _outcome(lambda: karr.summation(f, spec, auto=auto, special_values=special_values))
    elif name == 'karr_term':
        expected = _outcome(lambda: karr.karr_term(f, spec, auto=auto, special_values=special_values))
    else:
        expected = _outcome(lambda: karr.karr_sum(f, spec, auto=auto, special_values=special_values))
    assert found == expected
