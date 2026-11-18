"""The Rust port of the series machinery (``core/order.py``,
``core/series.py`` and the series methods of the core classes, ``exp``
and ``log``) against the Python modules: order terms, their arithmetic and
``contains``, leading terms, ``leadterm``, ``nseries`` and ``series``,
compared as strings; a part the Rust core does not have yet counts as
``'SKIP'``."""
from __future__ import annotations

from typing import Callable

import pytest

from sympy_extras.core import Integer, Rational, Symbol, Tuple, oo
from sympy_extras.core.basic import Basic
from sympy_extras.core.order import Order, order
from sympy_extras.core import pi
from sympy_extras.functions import cos, cosh, cot, coth, csc, csch, exp, log, sec, sech, sin, sinh, sqrt, tan, tanh

from ..conftest import load

rc = load("sympy_extras_rs.core")
x, y = Symbol('x'), Symbol('y')
xp = Symbol('xp', positive=True)


def call(op: str, *args: Basic) -> Basic:
    return rc.logic_call('series', [Symbol(op), *args])


def outcome(f: Callable[[], object]) -> str:
    try:
        return str(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as e:   # the kind of error must agree
        return type(e).__name__


def agree(python: Callable[[], object], rust: Callable[[], object]) -> None:
    r = outcome(rust)
    if r == 'SKIP':
        return
    assert r == outcome(python)


EXPRESSIONS: list[Basic] = [
    x, x**2 + x, 1/x + 1, (1 + x)/x, 1/(1 + x), x*y + x**2*y, exp(x), exp(x)/x, log(1 + x), log(x), x*log(x),
    exp(x**2) - 1, (1 + x)**Rational(1, 2), sqrt(x + x**2), 1/(1 - x), x**x, exp(1/x), log(1 + x)/x, (exp(x) - 1)/x,
    (1 + x)**y, exp(x)*log(1 + x), 1/(x + x**2), (1 - exp(x))**2, sqrt(1 + x) - 1, log(2 + x), exp(2 + x),
    x**3*exp(x), log(1 + x)**2, exp(x)**2, 1/(1 + exp(x)), log(exp(x) + 1), (x + 2)**-2, x/(1 - x - x**2),
    exp(-x)*(1 + x), log((1 + x)/(1 - x)), y*exp(y*x), (1 + 2*x)**Rational(3, 2), 1/sqrt(1 - x), x**y,
    exp(x*y) - exp(x), log(x + x**2), x**2*log(x)**2, 1/log(1 + x), sqrt(xp + xp**3),
    # the trigonometric and hyperbolic functions
    sin(x), cos(x), tan(x), cot(x), sec(x), csc(x), sinh(x), cosh(x), tanh(x), coth(x), sech(x), csch(x),
    sin(x)/x, (1 - cos(x))/x**2, tan(x) - x, x/sin(x), sin(2*x + 1), cos(pi/2 + x), tan(pi/2 + x), cot(pi + x),
    sec(pi/2 + x), csc(pi + x), sin(1/x), cos(1/x), tan(1/x), sin(x)**2 + cos(x)**2, exp(sin(x)), log(cos(x)),
    sinh(x)/x, cosh(x) - 1, tanh(2*x), coth(x) - 1/x, sech(x**2), csch(x)*x, sin(x)*sinh(x), cos(x + y),
    tan(x)**2, sin(sin(x)), cos(x)**Rational(1, 2), 1/(1 - cos(x)), sinh(1/x), tanh(1/x),
]


@pytest.mark.parametrize('i', range(len(EXPRESSIONS)))
def test_leading_terms(i: int) -> None:
    e = EXPRESSIONS[i]
    agree(lambda: e.as_leading_term(x), lambda: call('leading', e, Tuple(x)))
    agree(lambda: e.leadterm(x), lambda: tuple(call('leadterm', e, x).args))


@pytest.mark.parametrize('i', range(len(EXPRESSIONS)))
def test_nseries(i: int) -> None:
    e = EXPRESSIONS[i]
    for n in (0, 1, 3, 5):
        agree(lambda: e.nseries(x, 0, n), lambda: call('nseries', e, x, Integer(n)))


@pytest.mark.parametrize('i', range(len(EXPRESSIONS)))
def test_series(i: int) -> None:
    e = EXPRESSIONS[i]
    for x0, n, d in [(0, 4, '+'), (0, 3, '-'), (1, 3, '+'), (-2, 2, '+'), (0, 0, '+')]:
        agree(lambda: e.series(x, x0, n, d), lambda: call('series', e, x, Integer(x0), Integer(n), Symbol(d)))


ORDERS: list[tuple[Basic, tuple[tuple[Basic, Basic], ...]]] = [
    (x, ()), (x**2 + x, ()), (1 + x, ()), (x*y + x**2, ()), (x*(x + y), ((x, 0), (y, 0))), (Integer(3), ()),
    (2*x**3, ()), (x**2 + 1/x, ((x, 0),)), (x**2 + x, ((x, oo),)), (exp(x) - 1, ()), (x*log(x) + x, ()),
    (log(x)**2 + x, ()), (1/x + x**3, ((x, oo),)), (x - 1, ((x, 1),)), (y*x, ((x, 0),)), (x**Rational(1, 2) + x, ()),
    (sqrt(x)*x, ()), (x**y, ((x, 0),)), (exp(x), ((x, oo),)),
]


@pytest.mark.parametrize('i', range(len(ORDERS)))
def test_order(i: int) -> None:
    e, limits = ORDERS[i]
    agree(lambda: order(e, *[Tuple(s, p) for s, p in limits]),
          lambda: call('order', e, Tuple(*[Tuple(s, Integer(p) if isinstance(p, int) else p) for s, p in limits])))


CONTAINS: list[tuple[Basic, Basic]] = [
    (order(x), x**2), (order(x**2), x), (order(x), y), (order(x), Integer(1)), (order(1, x), x), (order(x), order(x**2)),
    (order(x**2), order(x)), (order(x*log(x)), x), (order(x), x*log(x)), (order(x**2, Tuple(x, oo)), x),
    (order(x), order(x, Tuple(x, oo))), (order(x*y), x**2*y), (order(x + x**2), x**3), (order(1/x), Integer(5)),
    (order(x**Rational(3, 2)), x**2), (order(log(x)), Integer(1)), (order(x), exp(x) - 1),
]


@pytest.mark.parametrize('i', range(len(CONTAINS)))
def test_contains(i: int) -> None:
    o, e = CONTAINS[i]
    agree(lambda: {True: 'True', False: 'False', None: '()'}[o.contains(e)], lambda: call('contains', o, e))


def test_order_arithmetic_and_parts() -> None:
    terms = [[x, order(x**2)], [x**2, order(x)], [Integer(1), order(1, x)], [Integer(3), order(x)],
             [x, x**3, order(x**2)], [order(x), order(x**2)], [order(x), order(y)], [exp(x), order(x)],
             [x**2, order(x**2, Tuple(x, oo))]]
    for t in terms:
        agree(lambda: sum(t, Integer(0)), lambda: call('add', Tuple(*t)))
    for f in [[x, order(x)], [order(x), order(x)], [order(x), order(y)], [Integer(2), order(x)], [1/x, order(x**3)],
              [order(x), order(x, Tuple(x, oo))]]:
        agree(lambda: f[0]*f[1], lambda: call('mul', Tuple(*f)))
    for b, e in [(order(x), Integer(2)), (order(x), Integer(-1)), (order(x), Rational(1, 2)), (order(x), y),
                 (order(x), Integer(0))]:
        agree(lambda: b**e, lambda: call('pow', b, e))
    for s in [x + x**2 + order(x**3), order(x**2), Integer(1) + order(1, x), x + order(x*log(x)),
              x + order(x**2*log(x)**2), x + y, order(x**Rational(5, 2))]:
        agree(lambda: s.getn() if s.getn() is not None else '()', lambda: call('getn', s))
    for s in [x + 1 + 1/x**5, 1 + x, x + x**2, x*y + x**2 + y**2, x + log(x)]:
        agree(lambda: s.extract_leading_order([x]), lambda: tuple(tuple(p.args) for p in call('extract', s, Tuple(x)).args))
    for o, old, new in [(order(x), x, y), (order(x**2), x, 2*x), (order(x), x, x - 1), (order(x), x, x**2),
                        (order(x*y), y, Integer(1)), (order(x, Tuple(x, oo)), x, 1/x), (order(x), x, x + 1)]:
        agree(lambda: o.subs(old, new), lambda: call('subs', o, old, new))
    assert str(call('order', x**2 + x, Tuple())) == str(Order(x))


def test_dir() -> None:
    from sympy_extras.core import I
    for e, cdir in [(x, Integer(1)), (x**2, Integer(-1)), (-x + x**2, Integer(1)), (I*x, Integer(1)),
                    (x**3 - x, I)]:
        agree(lambda: e.dir(x, cdir), lambda: call('dir', e, x, cdir))
