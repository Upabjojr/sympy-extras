"""The Rust port of :mod:`sympy_extras.concrete.qhyper` against the Python
one on the calls of the Python tests of the module (the cases read by
``eval`` in the core's namespace, with ``QPochhammer`` and ``qbinomial``,
the integers exact)."""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

import sympy_extras.core as core
import sympy_extras.functions as functions
import sympy_extras.interop  # noqa: F401  (the cases of the parameters go through SymPy)
from sympy_extras.concrete import qhyper as qh
from sympy_extras.core import Integer, Symbol, Tuple
from sympy_extras.core.basic import Basic

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
_NS.update(Integer=Integer, QPochhammer=qh.QPochhammer, qbinomial=qh.qbinomial, qpochhammer=qh.qpochhammer)


def _read(text: str) -> Any:
    return eval(re.sub(r'(?<![\w.])(\d+)(?![\w.])', r'Integer(\1)', text), _NS)


def outcome(f: Callable[[], object]) -> str:
    try:
        r = f()
    except NotImplementedError as e:
        return 'SKIP' if 'not ported' in str(e) else 'NotImplementedError'
    except Exception as e:
        return type(e).__name__
    return str(r)


#: the symbol standing for q**k in the ratios (a Dummy on both sides)
QQ = Symbol('QQ')

#: (name, arguments as text)
CASES: list[tuple[str, list[str]]] = [
    ('qpochhammer', ['a', 'q', '0']),
    ('qpochhammer', ['a', 'q', '3']),
    ('qpochhammer', ['a', 'q', 'k']),
    ('qpochhammer', ['a', 'q', '-2']),
    ('qpochhammer', ['0', 'q', 'k']),
    ('qbinomial', ['4', '2', 'q']),
    ('qbinomial', ['5', '2', 'q']),
    ('qbinomial', ['n', 'k', 'q']),
    ('q_ratio', ['q**k', 'k', 'q']),
    ('q_ratio', ['q**(k*(k - 1)/2)', 'k', 'q']),
    ('q_ratio', ['qpochhammer(a, q, k)', 'k', 'q']),
    ('q_ratio', ['1/qpochhammer(q, q, n - k)', 'k', 'q']),
    ('q_ratio', ['k*q**k', 'k', 'q']),
    ('q_ratio', ['qpochhammer(a, q, k**2)', 'k', 'q']),
    ('q_ratio', ['qpochhammer(a, q, k)*q**k', 'k', 'q']),
    ('q_ratio', ['qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k', 'k', 'q']),
    ('normal_in', ['(q*q**n*x + q*q**n - q**(n + 1) - q**(n + 1)*x)/(q*q**n - 1)', 'n', 'q']),
    ('normal_in', ['q**(2*n + 1)/(q**n*q - q)', 'n', 'q']),
    ('qgosper_term', ['q**k', 'k', 'q', 'False']),
    ('qgosper_term', ['q**k', 'k', 'q', 'True']),
    ('qgosper_term', ['k*q**k', 'k', 'q', 'False']),
    ('qgosper_term', ['q**k*qpochhammer(a, q, k)', 'k', 'q', 'False']),
    ('qgosper_term', ['q**(2*k)', 'k', 'q', 'False']),
    ('qgosper_term', ['q**(k*(k - 1)/2)', 'k', 'q', 'False']),
    ('qgosper_term', ['q**k/((1 - q**k)*(1 - q**(k + 1)))', 'k', 'q', 'False']),
    ('qgosper_sum', ['q**k', 'k', '0', 'n', 'q', 'False']),
    ('qgosper_sum', ['q**k', 'k', '0', 'n', 'q', 'True']),
    ('qgosper_sum', ['q**(2*k)', 'k', '0', 'n', 'q', 'True']),
    ('qgosper_sum', ['q**k*qpochhammer(a, q, k)', 'k', '0', 'n', 'q', 'False']),
    ('qgosper_sum', ['q**k*qpochhammer(a, q, k)', 'k', '0', 'n', 'q', 'True']),
    ('qgosper_sum', ['qbinomial(k, 2, q)*q**k', 'k', '0', 'n', 'q', 'True']),
    ('qgosper_sum', ['q**k/((1 - q**k)*(1 - q**(k + 1)))', 'k', '1', 'n', 'q', 'False']),
    ('qgosper_sum', ['q**(k*(k - 1)/2)', 'k', '0', 'n', 'q', 'False']),
    ('qzeilberger', ['qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k', 'n', 'k', 'q', '3']),
    ('qzeilberger', ['qbinomial(n, k, q)*(-1)**k*q**(k*(k - 1)/2)', 'n', 'k', 'q', '1']),
    ('qzeilberger', ['qbinomial(n, k, q)', 'n', 'k', 'q', '2']),
    ('qzeilberger', ['k*q**k', 'n', 'k', 'q', '2']),
]


def _args(raw: list[str]) -> list[Any]:
    return [_read(a) for a in raw]


def python(name: str, raw: list[str]) -> object:
    a = _args(raw)
    if name == 'qpochhammer':
        return qh.qpochhammer(*a)
    if name == 'qbinomial':
        return qh.qbinomial(*a)
    if name == 'q_ratio':
        found = qh.q_ratio(*a)
        return Symbol('NONE') if found is None else found[0].subs(found[1], QQ)
    if name == 'normal_in':
        return qh.normal_in(*a)
    if name == 'qgosper_term':
        found = qh.qgosper_term(a[0], a[1], a[2], special_values=a[3] is core.true or str(a[3]) == 'True')
        return Symbol('NONE') if found is None else found
    if name == 'qgosper_sum':
        found = qh.qgosper_sum(a[0], (a[1], a[2], a[3]), a[4], special_values=str(a[5]) == 'True')
        return Symbol('NONE') if found is None else found
    Z = qh.qzeilberger(a[0], a[1], a[2], a[3], max_order=int(a[4]))
    return Symbol('NONE') if Z is None else Tuple(Tuple(*Z.coefficients), Z.certificate)


def rust(name: str, raw: list[str]) -> object:
    a: list[Basic] = [Symbol(r) if r in ('True', 'False') else _read(r) for r in raw]
    r = rc.logic_call('calculus', [Symbol('qhyper'), Symbol(name), *a])
    if name == 'q_ratio' and isinstance(r, Tuple):
        ratio, Q = r.args
        return ratio.subs(Q, QQ)
    return r


@pytest.mark.parametrize('i', range(len(CASES)))
def test_qhyper(i: int) -> None:
    name, raw = CASES[i]
    expected = outcome(lambda: python(name, raw))
    got = outcome(lambda: rust(name, raw))
    if got == 'SKIP':
        pytest.skip('not ported')
    assert got == expected, (name, raw)
