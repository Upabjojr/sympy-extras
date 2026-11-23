"""The Rust ports of :mod:`sympy_extras.integrals.residues`,
:mod:`~sympy_extras.integrals.contours` and
:mod:`~sympy_extras.integrals.periodic` against the Python ones on the
calls of the Python tests of the modules (recorded as
printed core expressions in ``residues_cases.json`` and read back by
``eval`` in the core's namespace, the integers exact; the symbols are
plain, their assumptions printed away, so that some calls fail: the
errors are compared too). The numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
import sympy_extras.sets as sets
from sympy_extras.assumptions.facts import element
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.integrals import contours, periodic, residues
from sympy_extras.integrals.conditions import ConditionalValue
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


def _out(r: Optional[ConditionalValue]) -> object:
    return 'NONE' if r is None else Tuple(r.value, r.condition)


with open(os.path.join(os.path.dirname(__file__), 'residues_cases.json')) as _fh:
    CASES: list[list[Any]] = json.load(_fh)


def _python(name: str, f: Any, x: Any, a: Any, b: Any, assumptions: list[Any], half: bool) -> object:
    given = assumptions or None
    if name == 'to_laurent':
        found = periodic.to_laurent(f, x, a)
        return 'NONE' if found is None else found
    if name == 'laurent_coefficient':
        found = periodic.laurent_coefficient(f, x, a)
        return 'NONE' if found is None else found
    if name == 'mean_value_integral':
        return _out(periodic.mean_value_integral(f, x, a, b, given))
    if name == 'rectangular_integral':
        return _out(contours.rectangular_integral(f, x, given))
    if name == 'sector_integral':
        return _out(contours.sector_integral(f, x, given))
    if name == 'indented_integral':
        return _out(contours.indented_integral(f, x, a, b, given))
    if name == 'contour_integral':
        return _out(contours.contour_integral(f, x, a, b, given))
    if name == 'real_line_integral':
        return _out(residues.real_line_integral(f, x, given))
    if name == 'fourier_integral':
        return _out(residues.fourier_integral(f, x, given, half=half))
    if name == 'keyhole_integral':
        return _out(residues.keyhole_integral(f, x, given))
    if name == 'period_integral':
        return _out(residues.period_integral(f, x, a, b, given))
    return _out(residues.residue_integral(f, x, a, b, given))


@pytest.mark.parametrize('case', range(len(CASES)))
def test_residues(case: int) -> None:
    name, f_text, x_text, a_text, b_text, assumptions_text, half = CASES[case]
    f, x, a, b = _read(f_text), _read(x_text), _read(a_text), _read(b_text)
    read = _read(assumptions_text) if assumptions_text != 'None' else []
    assumptions = list(read) if isinstance(read, (list, tuple)) else [read]
    try:
        found = _norm(rc.logic_call('residues', [Symbol(name), f, x, a, b, Tuple(*assumptions),
                                                 Symbol('half' if half else 'full')]))
    except NotImplementedError as e:
        if 'not ported' in str(e):
            pytest.skip(str(e)[:100])
        raise
    except (TypeError, ValueError) as e:
        found = 'ERR %s' % type(e).__name__
    try:
        expected = _norm(_python(name, f, x, a, b, assumptions, half))
    except (TypeError, ValueError) as e:
        expected = 'ERR %s' % type(e).__name__
    assert found == expected
