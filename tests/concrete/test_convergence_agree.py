"""The Rust port of :mod:`sympy_extras.concrete.convergence` against the
Python one on the calls of the Python tests of the module (recorded as
printed core expressions in ``convergence_cases.json`` and read back by
``eval`` in the core's namespace, the integers exact; the symbols are
plain, their assumptions printed away)."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete import convergence
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.relational import Eq, Ne
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
_NS.update(Eq=Eq, Ne=Ne, Integer=Integer)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'convergence_cases.json')) as _fh:
    CASES: list[list[Any]] = json.load(_fh)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_convergence(case: int) -> None:
    name, term_text, n_text, assumptions_text, lower_text = CASES[case]
    term, n, lower = _read(term_text), _read(n_text), _read(lower_text)
    read = _read(assumptions_text) if assumptions_text != 'None' else []
    assumptions = list(read) if isinstance(read, (list, tuple)) else [read]
    try:
        found = _norm(rc.logic_call('convergence', [Symbol(name), term, n, Tuple(*assumptions), lower]))
    except NotImplementedError as e:
        if 'not ported' in str(e):
            pytest.skip(str(e)[:100])
        raise
    function = getattr(convergence, name)
    value = function(term, n, assumptions or None, lower)
    expected = _norm('NONE' if value is None else value)
    assert found == expected
