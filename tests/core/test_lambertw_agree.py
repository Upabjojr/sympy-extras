"""``LambertW`` (``sympy_extras.functions.elementary.exponential``)
against its Rust port: the evaluation at the special points, the
derivative, the facts and the numerical values on several branches."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _rc() -> Any:
    return load("sympy_extras_rs.core")


def _outcome(f: Any) -> Any:
    try:
        return ('ok', str(f()))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _nodes() -> list[Any]:
    from sympy_extras.core import E, I, Integer, Rational, Symbol, oo, pi
    from sympy_extras.functions import exp, log
    from sympy_extras.functions.elementary.exponential import LambertW
    x = Symbol('x')
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    args = [Integer(0), E, -1/E, -log(Integer(2))/2, 2*log(Integer(2)), -pi/2, exp(1 + E), oo, x, p, n, Integer(3), Rational(-1, 5),
            Rational(-1, 2), I, -2*exp(Integer(-2)), Integer(-1), 1 + I, Rational(1, 10)]
    out = []
    for a in args:
        out.append(LambertW(a))
        for k in (Integer(0), Integer(-1), Integer(1), Integer(2), Symbol('k', integer=True)):
            out.append(LambertW(a, k))
    return out


@pytest.mark.parametrize("index", range(0, 114, 6))
def test_lambertw(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol, Tuple, Integer
    from sympy_extras.core.evalf import numeric_value
    rc = _rc()
    x = Symbol('x')
    for e in _nodes()[index:index + 6]:
        # the evaluation
        got = _outcome(lambda: rc.logic_call('root_sum', [Symbol('doit'), e]))
        assert got == _outcome(lambda: e.doit()), e
        evaluated = e.doit()
        # the facts
        for p in ('real', 'extended_real', 'zero', 'finite', 'algebraic', 'commutative'):
            r = rc.logic_call('is', [evaluated, Symbol(p)])
            value = None if str(r) == '()' else bool(r)
            assert value == getattr(evaluated, 'is_' + p), (evaluated, p)
        # the derivative in x of the node at x
        if evaluated.free_symbols:
            assert _outcome(lambda: rc.logic_call('root_sum', [Symbol('diff'), evaluated, x])) == \
                _outcome(lambda: evaluated.diff(x)), evaluated
        # the numerical value
        if not evaluated.free_symbols:
            expected = numeric_value(evaluated, 80)
            got_v = rc.logic_call('reliable', [Symbol('value'), evaluated, Integer(20), Tuple()])
            if expected is None:
                continue
            assert got_v.args, evaluated
            value = complex(float(got_v.args[0]), float(got_v.args[1]))
            assert abs(complex(expected) - value) < 1e-12*max(1.0, abs(complex(expected))), (evaluated, expected, value)
