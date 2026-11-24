"""The Rust ports of :mod:`sympy_extras.integrals.telescoping`,
:mod:`~sympy_extras.integrals.reduction` and
:mod:`~sympy_extras.integrals.dfinite` against the Python ones on the calls
of their Python tests (recorded in ``telescoping_cases.json``, read back by
``eval`` in the core's namespace with the integers exact; the symbols are
plain, their assumptions printed away). A telescoper is compared by its
coefficients and certificate, a ``ConditionalValue`` by its value and
condition; the numbers of the dummies are not compared."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.integrals import dfinite, reduction, telescoping
from sympy_extras.integrals.conditions import ConditionalValue
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


with open(os.path.join(os.path.dirname(__file__), 'telescoping_cases.json')) as _fh:
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
    if isinstance(value, ConditionalValue):
        return _norm(Tuple(value.value, value.condition))
    if isinstance(value, telescoping.DifferentialTelescoper):
        return _norm(Tuple(Tuple(*value.coefficients), value.certificate))
    if isinstance(value, dfinite.DFiniteTelescoper):
        return _norm(Tuple(Tuple(*value.coefficients), Tuple(*[Tuple(b, g) for b, g in value.certificate.items()])))
    if isinstance(value, tuple) and value and isinstance(value[0], dfinite.Annihilator):
        return _norm(Tuple(*[Tuple(*a.coefficients) for a in value]))
    if isinstance(value, tuple):
        return _norm(Tuple(*value))
    return _norm(value)


def _python(name: str, args: list[Any], o1: Any, o2: Any) -> Any:
    if name == 'almkvist_zeilberger':
        return telescoping.almkvist_zeilberger(args[0], args[1], args[2], int(o1), int(o2))
    if name == 'reduction_telescoper':
        return reduction.reduction_telescoper(args[0], args[1], args[2], int(o1))
    if name == 'chyzak':
        return dfinite.chyzak(args[0], args[1], args[2], int(o1), int(o2))
    if name == 'annihilators':
        return dfinite.annihilators(args[0], args[1], args[2])
    if name == 'hermite_reduce':
        return reduction.hermite_reduce(args[0], args[1], args[2], args[3])
    f, x, a, b, t = args[:5]
    module = {'holonomic_ode': telescoping, 'holonomic_integral': telescoping, 'reduction_integral': reduction,
              'dfinite_integral': dfinite}[name]
    return getattr(module, name)(f, x, a, b, t, o1)


@pytest.mark.parametrize('case', range(len(CASES)))
def test_telescoping(case: int) -> None:
    name, *texts, o1_text, o2_text = CASES[case]
    args = [None if text in ('None', 'NONE') else _read(text) for text in texts]
    integral = name in ('holonomic_ode', 'holonomic_integral', 'reduction_integral', 'dfinite_integral')
    o1: Any = (None if o1_text == 'None' else _read(o1_text)) if integral else o1_text
    none = Symbol('NONE')
    bridge = [none if v is None else v for v in args]
    if integral:
        bridge.append(Tuple() if o1 is None else Tuple(o1))
        bridge.append(Integer(0))
    else:
        bridge += [Integer(int(o1_text)), Integer(int(o2_text))]
    found = _outcome(lambda: rc.logic_call('telescoping', [Symbol(name)] + bridge))
    expected = _outcome(lambda: _python(name, args, o1, o2_text))
    assert found == expected
