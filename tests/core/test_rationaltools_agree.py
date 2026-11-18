"""``sympy_extras.integrals.risch.rationaltools`` against its Rust port:
``ratint`` (real and complex forms, parameters, algebraic and
transcendental constants, RootSum), ``ratint_ratpart``,
``ratint_logpart``, ``polynomial_part``, ``log_to_atan`` and
``log_to_real``; the outputs compared as strings."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load


def _call(op: str, *args: Any) -> Any:
    from sympy_extras.core import Symbol
    return load("sympy_extras_rs.core").logic_call('ratint', [Symbol(op), *args])


def _outcome(f: Any) -> Any:
    try:
        r = f()
        return ('ok', 'None' if r is None else str(r))
    except NotImplementedError:
        return ('SKIP', None)
    except Exception as exc:
        return ('ERROR ' + type(exc).__name__, None)


def _symbols() -> Any:
    from sympy_extras.core.symbol import Symbol
    return Symbol('x'), Symbol('a'), Symbol('b'), Symbol('t'), Symbol('p', positive=True)


def _integrands() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.functions import tan
    x, a, b, t, p = _symbols()
    return [Integer(0), Integer(7), x, 8*x**7 + 2*x + 1, 36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), 1/(x**2 + 1),
            (x**4 - 3*x**2 + 6)/(x**6 - 5*x**4 + 5*x**2 + 4), (x**7 - 24*x**4 - 4*x**2 + 8*x - 8)/(x**8 + 6*x**6 + 12*x**4 + 8*x**2),
            1/(x**2 + x + 1), 1/(x**3 + 1), 1/(x - 1)**2, (x**3 + 1)/(x*(x - 1)**3), 1/(x**2 + 1)**3, (x + 3)/(x**2 + 4*x + 13),
            1/(x**4 + 4), x**5 - 3*x + Rational(1, 2), (x**4 - 1)/(x - 1), 1/(x**4 + 1), x**3 - x, (x**3 - x)/(x - 1),
            x/((x - a)*(x + a)), 1/(x*(a + b*x)**3), 1/(x**2 + a**2), 1/(x**3 - x - 1), 1/(x + a), 1/(x**2 + a),
            x/(x**2 + a*x + b), 1/((x - a)*(x - b)), 1/(x**2 + p), 1/(x**2 + p**2), p/(x**2 + 1)**2, 1/(x**3 + a),
            (-x**2 + sqrt(Integer(2)))/(x**4 - sqrt(Integer(2))*x**2 + 1), 1/(x - sqrt(Integer(3))), 1/(x**2 + sqrt(Integer(2))),
            1/(x - I), 1/(x**2 + pi), 1/(x**2 + tan(Integer(1))), (x**2 + 1)/(x**3 - 2*x + 4), 1/(x**5 - 1)]


@pytest.mark.parametrize("index", range(len(_integrands())))
def test_ratint(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.logic import false, true
    from sympy_extras.integrals.risch.rationaltools import ratint
    x, a, b, t, p = _symbols()
    f = _integrands()[index]
    for real, flag in [(None, Symbol('None')), (True, true), (False, false)]:
        got = _outcome(lambda: _call('ratint', f, x, t, flag))
        expected = _outcome(lambda: ratint(f, x, real=real))
        assert got == expected, (f, real)


def test_parts(rust: Any) -> None:
    from sympy_extras.core import Integer, Rational, Symbol, sqrt, Tuple
    from sympy_extras.core.logic import false, true
    from sympy_extras.integrals.risch.rationaltools import (log_to_atan, log_to_real, polynomial_part, ratint,
                                                            ratint_logpart, ratint_ratpart)
    x, a, b, t, p = _symbols()
    y = Symbol('y')
    assert _outcome(lambda: _call('ratint_pair', x, x**2 - 1, x, t, Symbol('None'))) == _outcome(lambda: ratint((x, x**2 - 1), x))
    for f, g in [(Integer(36), x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), (Integer(1), x**2 + a**2), (x, (x - 1)**2*(x + 1)),
                 (Integer(1), (x**2 + sqrt(Integer(2)))**2)]:
        assert _outcome(lambda: tuple(str(v) for v in _call('ratpart', f, g, x).args)) == \
            _outcome(lambda: tuple(str(v) for v in ratint_ratpart(f, g, x))), (f, g)
    for f, g in [(x, x**2 - 9), (x**2, x**3 - 5), (Integer(1), x**2 + x + 1), (Integer(1), x**3 - x - 1), (a, x**2 + a)]:
        got = _outcome(lambda: [tuple(str(v) for v in pair.args) for pair in _call('logpart', f, g, x, t).args])
        assert got == _outcome(lambda: [tuple(str(v) for v in pair) for pair in ratint_logpart(f, g, x, t)]), (f, g)
    for f, g in [(x**3 + a, x + 1), (x**4 - 1, x**2 + 3)]:
        assert _outcome(lambda: tuple(str(v) for v in _call('polynomial_part', f, g, x).args)) == \
            _outcome(lambda: tuple(str(v) for v in polynomial_part(f, g, x))), (f, g)
    for f, g in [(x, Integer(1)), (x + Rational(1, 2), sqrt(Integer(3))/2), (sqrt(Integer(3))/2, x + Rational(1, 2)), (x**3 - 3*x, x**2 - 1)]:
        assert _outcome(lambda: _call('log_to_atan', f, g, x)) == _outcome(lambda: log_to_atan(f, g, x)), (f, g)
    for h, q, only in [(x + 3*y/2 + Rational(1, 2), 3*y**2 + 1, False), (x**2 - 1, 1 - 2*y, False), (x**2 - 1, 1 - 2*y, True),
                       (x - y, y**2 + 1, False), (x - y, y**3 - y - 1, False)]:
        got = _outcome(lambda: _call('log_to_real', h, q, x, y, true if only else false))
        got = ('ok', 'None') if got == ('ok', '()') else got
        assert got == _outcome(lambda: log_to_real(h, q, x, y, complex_only=only)), (h, q, only)
