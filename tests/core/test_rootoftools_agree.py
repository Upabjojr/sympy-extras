"""``sympy_extras.polys.rootoftools`` (``RootSum``, ``root_sum``) against
its Rust port: the evaluation, ``doit``, the derivative, the round trip
of the node through the bridge and its numerical value; the outputs
compared as strings."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('root_sum', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _symbols() -> Any:
    from sympy_extras.core.symbol import Symbol
    return Symbol('t'), Symbol('x'), Symbol('a')


def _cases() -> list[tuple[Any, Any, bool, bool]]:
    from sympy_extras.core import Integer, Lambda, Rational, sqrt
    from sympy_extras.functions import exp, log
    t, x, a = _symbols()
    L = Lambda(t, t*log(x + t))
    out = []
    for f in [t - 2, t**3 - 1, 4*t**3 - 2*t - 2, t**3 - t - 1, t**2 + 1/(4*a**2), t**2 - 2, t**4 + 1, 2*t**2 + 2*t + 1,
              t**3 - sqrt(Integer(2)), t**2 - a*t + 1, (t - 1)**2*(t + 3), t**5 - t + 1]:
        for fun in [L, Lambda(t, 2*t*log(x + t) + 3), Lambda(t, x), Lambda(t, t), Lambda(t, t**2), Lambda(t, 1/(x + t)),
                    Lambda(t, exp(t)), Lambda(t, t/(t**2 + 1)), None]:
            for quadratic in (False, True):
                out.append((f, fun, quadratic, True))
        out.append((f, Lambda(t, 1/(x + t)), False, False))
    return out


@pytest.mark.parametrize("index", range(0, 240, 3))
def test_root_sum(rust: Any, index: int) -> None:
    from sympy_extras.core import Tuple
    from sympy_extras.core.logic import false, true
    from sympy_extras.polys.rootoftools import root_sum
    t, x, a = _symbols()
    for i in range(index, min(index + 3, len(_cases()))):
        f, fun, quadratic, auto = _cases()[i]
        var, body = (fun.variables[0], fun.expr) if fun is not None else (Tuple(), Tuple())
        got = _outcome(lambda: _call('root_sum', f, var, body, t, true if auto else false, true if quadratic else false))
        expected = _outcome(lambda: root_sum(f, fun, t, auto=auto, quadratic=quadratic))
        assert got == expected, (f, fun, quadratic, auto)


def test_node_operations(rust: Any) -> None:
    from sympy_extras.core import Lambda
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions import log
    from sympy_extras.polys.rootoftools import RootSum
    t, x, a = _symbols()
    rc = load("sympy_extras_rs.core")
    nodes = [RootSum(t**2 - 1, Lambda(t, t**2), t), RootSum(t**3 - t - 1, Lambda(t, t*log(x + t)), t),
             RootSum(t**3 - t - 1, Lambda(t, log(t*x)), t), RootSum(t**3 + t + 1, Lambda(t, t*log(x - t)), t),
             RootSum(t**4 + 1, Lambda(t, t*log(x - t)), t), RootSum(t**3 - 2, Lambda(t, 1/t), t)]
    for r in nodes:
        assert _outcome(lambda: _call('doit', r)) == _outcome(lambda: r.doit()), r
        assert _outcome(lambda: _call('diff', r, x)) == _outcome(lambda: r.diff(x)), r
        assert bool(_call('is_number', r)) == (not r.free_symbols), r
        # the round trip through the bridge
        back = rc.logic_call('root_sum', [__import__('sympy_extras.core', fromlist=['Symbol']).Symbol('doit'), _call('diff', r, a)])
        assert str(back) == str(r.diff(a).doit()), r
    for r in [nodes[3].subs(x, 2), RootSum(t**3 - t - 1, Lambda(t, t**2 + 1), t), RootSum(t**5 - t + 1, Lambda(t, log(3 - t)), t)]:
        expected = numeric_value(r, 80)
        got = rc.logic_call('reliable', [__import__('sympy_extras.core', fromlist=['Symbol']).Symbol('value'), r,
                                         __import__('sympy_extras.core', fromlist=['Integer']).Integer(20),
                                         __import__('sympy_extras.core', fromlist=['Tuple']).Tuple()])
        assert expected is not None and got.args, r
        value = complex(float(got.args[0]), float(got.args[1]))
        assert abs(complex(expected) - value) < 1e-12, (r, expected, got)
