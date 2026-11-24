"""The Rust ports of :mod:`sympy_extras.solvers.linear_ode`,
:mod:`~sympy_extras.solvers.kovacic` and :mod:`~sympy_extras.solvers.special`
(with :mod:`~sympy_extras.solvers.irregular`,
:mod:`~sympy_extras.solvers.factorization`, :mod:`~sympy_extras.solvers.dsolve`
and :mod:`~sympy_extras.solvers._special_values` behind them) against the
Python ones on the calls of their Python tests (recorded in
``ode_cases.json``, read back by ``eval`` in the core's namespace with the
integers exact; the symbols are plain, ``y`` and ``u`` undefined
functions). A ``KovacicSolution`` is compared by its case, ``omega``, its
polynomial and the solution; the numbers of the dummies are not
compared."""
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
from sympy_extras.solvers import kovacic, linear_ode, special
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
_NS.update(Integer=Integer, Sum=Sum, Product=Product, y=Function('y'), u=Function('u'))


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'ode_cases.json')) as _fh:
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
    if isinstance(value, kovacic.KovacicSolution):
        omega = Symbol('NONE') if value.omega is None else value.omega
        return _norm(Tuple(Integer(value.case), omega, value.omega_polynomial, value.solution))
    if isinstance(value, list):
        return _norm(Tuple(*value))
    return _norm(value)


def _python(name: str, a: Any, b: Any, flags: list[bool]) -> Any:
    if name == 'dsolve_linear':
        return linear_ode.dsolve_linear(a, b, flags[0], flags[1], flags[2])
    if name == 'dsolve_kovacic':
        return kovacic.dsolve_kovacic(a, b, flags[2])
    if name == 'liouvillian_solution':
        return kovacic.liouvillian_solution(a, b)
    return special.special_solutions(a, b)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_ode(case: int) -> None:
    name, a_text, b_text, *flag_texts = CASES[case]
    a, b = _read(a_text), _read(b_text)
    flags = [t == 'True' for t in flag_texts]
    args = [Symbol(name), a, b] + [Symbol('yes' if v else 'no') for v in flags]
    found = _outcome(lambda: rc.logic_call('ode', args))
    expected = _outcome(lambda: _python(name, a, b, flags))
    assert found == expected
