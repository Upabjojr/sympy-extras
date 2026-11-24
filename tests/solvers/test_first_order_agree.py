"""The Rust ports of :mod:`sympy_extras.solvers.first_order`,
:mod:`~sympy_extras.solvers.abel`, :mod:`~sympy_extras.solvers.second_order`,
:mod:`~sympy_extras.solvers.ode` and :mod:`~sympy_extras.solvers.darboux`
against the Python ones on the calls of their Python tests (recorded in
``first_order_cases.json``, read back by ``eval`` in the core's namespace
with the integers exact; the symbols are plain, ``y`` an undefined
function). The result objects are compared through tuples of their
fields; the numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Function, Integer, Symbol, Tuple
from sympy_extras.core.logic import false, true
from sympy_extras.solvers import abel, darboux, first_order, ode, second_order
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
_NS.update(Integer=Integer, Sum=Sum, Product=Product, _applied_y=Function('y'), _applied_u=Function('u'))


def _read(text: str) -> Any:
    # y and u are functions where they are applied, symbols elsewhere (the
    # second order rules and the vector fields take y as a symbol)
    text = re.sub(r'(?<![\w.])([yu])\(', r'_applied_\1(', text)
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def _norm(r: object) -> str:
    return re.sub(r'\b_?([a-zA-Z]+)(_\d+)?\b', r'\1', str(r))


with open(os.path.join(os.path.dirname(__file__), 'first_order_cases.json')) as _fh:
    CASES: list[list[str]] = json.load(_fh)


def _shown(value: Any) -> Any:
    if value is None:
        return Symbol('NONE')
    if isinstance(value, bool):
        return true if value else false
    if isinstance(value, (list, tuple)):
        return Tuple(*[_shown(v) for v in value])
    if isinstance(value, abel.AbelInvariants):
        return Tuple(value.s3, value.s5, value.I1, value.I2)
    if isinstance(value, abel.AbelTransformation):
        return Tuple(value.xi, value.P, value.Q)
    if isinstance(value, second_order.FirstIntegral):
        return Tuple(value.factor, value.integral)
    if isinstance(value, second_order.Linearization):
        return Tuple(value.t, value.u, value.rhs)
    if isinstance(value, darboux.DarbouxPolynomial):
        return Tuple(value.polynomial, value.cofactor)
    if isinstance(value, darboux.ExponentialFactor):
        return Tuple(value.numerator, value.denominator, value.cofactor)
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


def _flag(text: str) -> bool:
    return text == 'True'


def _python(name: str, t: list[str]) -> Any:
    r = [_read(s) if s not in ('True', 'False', 'None') else None for s in t]
    if name == 'riccati_ode':
        return first_order.riccati_ode(r[0], r[1], _flag(t[2]))
    if name == 'dsolve_first_order':
        return first_order.dsolve_first_order(r[0], r[1], _flag(t[2]))
    if name in ('chini_ode', 'abel_ode', 'lagrange_ode'):
        return getattr(first_order, name)(r[0], r[1])
    if name == 'abel_by_invariants':
        return abel.abel_by_invariants(r[0], r[1])
    if name in ('abel_invariants', 'constant_invariant'):
        return getattr(abel, name)(tuple(r[0]), r[1])
    if name == 'particular_solution':
        return abel.particular_solution(tuple(r[0]), r[1], int(t[2]))
    if name == 'abel_equivalence':
        return abel.abel_equivalence(tuple(r[0]), r[1], tuple(r[2]), r[3])
    if name == 'air_solution':
        return abel.air_solution(tuple(r[0]), r[1], r[2])
    if name == 'second_order_rhs':
        return second_order.second_order_rhs(r[0], r[1])
    if name in ('integrating_factor_xy', 'integrating_factor_p', 'is_linearizable', 'linearize'):
        return getattr(second_order, name)(r[0], r[1], r[2], r[3])
    if name == 'transformed_rhs':
        return second_order.transformed_rhs(*r)
    if name == 'ode_symmetries':
        return ode.ode_symmetries(r[0], r[1], int(t[2]))
    if name == 'dsolve_second_order':
        return second_order.dsolve_second_order(r[0], r[1], int(t[2]), _flag(t[3]))
    if name == 'check_ode_solution':
        return ode.check_ode_solution(r[0], r[1], r[2], solve_for_func=_flag(t[3]))
    if name == 'dsolve_lie':
        return ode.dsolve_lie(r[0], r[1], int(t[2]), check=_flag(t[3]), special_values=_flag(t[4]))
    if name == 'solve_ode':
        return ode.solve_ode(r[0], r[1], int(t[2]), check=_flag(t[3]), special_values=_flag(t[4]))
    if name == 'prelle_singer':
        return darboux.prelle_singer(r[0], r[1], int(t[2]), r[3])
    if name == 'vector_field':
        return darboux.vector_field(r[0], r[1])
    # darboux_polynomials, exponential_factors, darboux_integrating_factor,
    # darboux_first_integral
    return getattr(darboux, name)(r[0], r[1], r[2], r[3], int(t[4]), r[5])


def _rust_argument(text: str) -> Any:
    if text in ('True', 'False'):
        return Symbol('yes' if text == 'True' else 'no')
    if text == 'None':
        return Symbol('NONE')
    value = _read(text)
    return Tuple(*value) if isinstance(value, tuple) else value


@pytest.mark.parametrize('case', range(len(CASES)))
def test_first_order(case: int) -> None:
    name, *texts = CASES[case]
    args = [Symbol(name)] + [_rust_argument(t) for t in texts]
    found = _outcome(lambda: rc.logic_call('first_order', args))
    expected = _outcome(lambda: _python(name, texts))
    assert found == expected
