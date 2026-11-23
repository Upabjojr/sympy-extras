"""The Rust port of :func:`sympy_extras.integrals.definite.definite_integral`
against the Python one on the calls of the Python tests (recorded as
printed core expressions in ``definite_cases.json`` and read back by
``eval`` in the core's namespace, the integers exact; the symbols are
plain, their assumptions printed away). A case whose Rust run reaches a
module not ported yet (residues, periodic, ...) is skipped before the
Python run. The numbers of the dummies are not compared.

Left out: ``exp(-s*t)*besselk(0, k*t)`` over ``(0, oo)`` under ``k > 0, s >
k``, where the native ``expand`` of Python does not descend into the
branches of a ``Piecewise`` (``factor`` leaves them as they are) and
SymPy's, which the Rust port follows, does, so that ``tidy`` takes
another form of the same value."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
import sympy_extras.sets as sets
from sympy_extras.assumptions.facts import element
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.integrals.definite import definite_integral
from ..conftest import load

rc = load("sympy_extras_rs.core")


class _Namespace(dict):  # type: ignore[type-arg]
    def __missing__(self, name: str) -> Symbol:
        return Symbol(name)


_NS = _Namespace()
for _mod in (core, functions, sets):
    for _k in dir(_mod):
        if not _k.startswith('_'):
            _NS[_k] = getattr(_mod, _k)
_NS.update(Eq=Eq, Ne=Ne, Contains=element, Integer=Integer)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'definite_cases.json')) as _fh:
    CASES: list[list[Any]] = json.load(_fh)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_definite_integral(case: int) -> None:
    f_text, limits_text, assumptions_text, flags = CASES[case]
    f = _read(f_text)
    x, a, b = _read(limits_text)
    read = _read(assumptions_text) if assumptions_text != 'None' else []
    assumptions = list(read) if isinstance(read, (list, tuple)) else [read]
    try:
        found = _norm(rc.logic_call('definite', [Symbol('integral'), f, x, a, b, Tuple(*assumptions),
                                                 Tuple(*[Symbol(s) for s in flags])]))
    except NotImplementedError as e:
        if 'not ported' in str(e):
            pytest.skip(str(e)[:100])
        raise
    kw = dict(conds='none' if 'none' in flags else 'piecewise', recognize='recognize' in flags,
              principal_value='principal_value' in flags, finite_part='finite_part' in flags,
              regularize='regularize' in flags, special_values='no_special_values' not in flags)
    expected = _norm(definite_integral(f, (x, a, b), assumptions or None, **kw))  # type: ignore[arg-type]
    assert found == expected
