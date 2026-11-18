"""``sympy_extras.simplify.trigsimp`` (``trigsimp`` with ``matching`` and
``fu``, ``futrig``, ``exptrigsimp``) against its Rust port; the outputs
compared as strings."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('trigsimp', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _trig_cases() -> list[Any]:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import cos, cosh, cot, log, sec, sin, sinh, sqrt, tan, tanh
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    return [sin(x)**2 + cos(x)**2, 2*sin(x)*cos(x), sin(x)/cos(x), cosh(x)**2 - sinh(x)**2, 1 - cos(x)**2,
            sin(x)**4 - cos(x)**4, tan(x)**2 + 1, sin(x)*cos(y) + cos(x)*sin(y), a*sin(x)**2 + a*cos(x)**2,
            cos(2*x) + 2*sin(x)**2, sec(x)**2 - 1, (1 + tan(x)**2)*cos(x)**2, sin(x)**3*cos(x) + sin(x)*cos(x)**3,
            sinh(x)/cosh(x), log(2*sin(x)*cos(x)), sqrt(1 - sin(x)**2), cos(x)**2*tan(x), 3*sin(x)**2 + 3*cos(x)**2 + y,
            cos(x)*cos(y) - sin(x)*sin(y), tan(x)*cot(x), sin(x)*tan(x)/sec(x), tanh(x)*cosh(x), x*sin(y)**2 + x*cos(y)**2,
            (sin(x) + cos(x))**2, sin(2*x)/(2*cos(x)), Integer(2)*cos(x/2)**2 - 1, sinh(x)**2 + cosh(x)**2]


@pytest.mark.parametrize("index", range(len(_trig_cases())))
def test_trigsimp(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import false
    from sympy_extras.simplify.trigsimp import futrig, trigsimp
    e = _trig_cases()[index]
    for method in ('matching', 'fu', 'groebner'):
        got = _outcome(lambda: _call('trigsimp', e, Symbol(method)))
        expected = _outcome(lambda: trigsimp(e, method=method))
        assert got == expected, (e, method)
    assert _outcome(lambda: _call('futrig', e, false)) == _outcome(lambda: futrig(e, hyper=False)), e


def test_exptrigsimp(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.functions import cos, cosh, exp, sin, sinh
    from sympy_extras.simplify.trigsimp import exptrigsimp
    x, y = Symbol('x'), Symbol('y')
    cases = [exp(x) + exp(-x), (exp(x) - exp(-x))/2, exp(2*x)/(1 + exp(2*x)), cosh(x) + sinh(x), (1 - exp(-x))/(1 + exp(-x)),
             exp(x)*cos(y), cosh(x) - sinh(x), sin(x) + cos(x), exp(x)/(exp(x) - 1), (exp(2*x) - 1)/(exp(2*x) + 1)]
    different = []
    for e in cases:
        got, expected = _outcome(lambda: _call('exptrigsimp', e)), _outcome(lambda: exptrigsimp(e))
        if got != expected:
            different.append((e, got, expected))
    assert not different, different[:3]
