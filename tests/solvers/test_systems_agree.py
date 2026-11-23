"""The Rust ports of :mod:`sympy_extras.solvers.charpit`,
:mod:`~sympy_extras.solvers.dae`, :mod:`~sympy_extras.solvers.pde` and
:mod:`~sympy_extras.solvers.linear_systems` against the Python ones on the
calls of their Python tests (recorded in ``systems_cases.json``, read back
by ``eval`` in the core's namespace with the integers exact). Matrices are
compared as ``(rows, cols, entries)``, the result objects through tuples
of their fields; the numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.core import Function, Integer, Symbol, Tuple
from sympy_extras.core.logic import false, true
from sympy_extras.matrices import ExprMatrix
from sympy_extras.solvers import charpit, dae, linear_systems, pde
from ..conftest import load

rc = load("sympy_extras_rs.core")


class _Namespace(dict):  # type: ignore[type-arg]
    def __missing__(self, name: str) -> Any:
        if name in ('u', 'v', 'q', 'f', 'g') or re.fullmatch(r'[yzf]\d+', name):
            return Function(name)
        return Symbol(name)


_NS = _Namespace()
for _mod in (core, functions):
    for _k in dir(_mod):
        if not _k.startswith('_'):
            _NS[_k] = getattr(_mod, _k)
_NS.update(Integer=Integer, ExprMatrix=ExprMatrix.from_rows, u=Function('u'))


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'systems_cases.json')) as _fh:
    CASES: list[list[str]] = json.load(_fh)


def _matrix(m: ExprMatrix) -> Tuple:
    return Tuple(Integer(m.rows), Integer(m.cols), Tuple(*m.entries))


def _shown(value: Any) -> Any:
    if value is None:
        return Symbol('NONE')
    if isinstance(value, bool):
        return true if value else false
    if isinstance(value, int):
        return Integer(value)
    if isinstance(value, ExprMatrix):
        return _matrix(value)
    if isinstance(value, (list, tuple)):
        return Tuple(*[_shown(v) for v in value])
    if isinstance(value, dae.DAESolution):
        return Tuple(_matrix(value.solution), Tuple(*value.constants), Integer(value.index))
    if isinstance(value, linear_systems.LinearSystemSolution):
        return Tuple(_matrix(value.fundamental), _shown(value.particular), _shown(value.complete))
    if hasattr(value, 'coefficients') and hasattr(value, 'x'):
        return Tuple(*value.coefficients)
    if hasattr(value, 'xi') and hasattr(value, 'eta'):
        return Tuple(Tuple(*value.xi), Tuple(*value.eta))
    return value


def _outcome(f: Callable[[], Any]) -> str:
    try:
        value = f()
    except NotImplementedError as e:
        if 'not ported' in str(e):
            pytest.skip(str(e)[:100])
        return 'ERR NotImplementedError'
    except (TypeError, ValueError) as e:
        return 'ERR %s' % type(e).__name__
    return _norm(_shown(value))


def _python(name: str, t: list[str]) -> Any:
    r = [None if s == 'None' else (s == 'True' if s in ('True', 'False') else _read(s)) for s in t]
    if name in ('complete_integral', 'check_complete_integral'):
        return getattr(charpit, name)(*r)
    if name in ('matrix_exponential', 'core_nilpotent_decomposition', 'dae_index', 'dsolve_dae', 'check_dae',
                'dae_matrices'):
        return getattr(dae, name)(*r)
    if name == 'pde_symmetries':
        return pde.pde_symmetries(r[0], r[1], int(t[2]), superposition=r[3])
    if name == 'pdsolve_lie':
        return pde.pdsolve_lie(r[0], r[1], degree=int(t[2]), check=r[3])
    return getattr(linear_systems, name)(*r)


def _rust_argument(text: str) -> Any:
    if text in ('True', 'False'):
        return Symbol('yes' if text == 'True' else 'no')
    if text == 'None':
        return Symbol('NONE')
    value = _read(text)
    if isinstance(value, ExprMatrix):
        return _matrix(value)
    if isinstance(value, (list, tuple)):
        return Tuple(*value)
    return value


@pytest.mark.parametrize('case', range(len(CASES)))
def test_systems(case: int) -> None:
    name, *texts = CASES[case]
    args = [Symbol(name)] + [_rust_argument(t) for t in texts]
    found = _outcome(lambda: rc.logic_call('systems', args))
    expected = _outcome(lambda: _python(name, texts))
    assert found == expected
