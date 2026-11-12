"""The exponential, trigonometric, logarithmic and Fresnel integrals of
the Rust core against ``sympy_extras.functions.special.error_functions``:
``expint``, ``Ei``, ``Si``, ``Ci``, ``Shi``, ``Chi``, ``li``, ``Li``,
``fresnels`` and ``fresnelc`` of random numbers, infinities and symbols,
compared by structure (or the error), facts, conjugates, derivatives and
values (the series of the Rust core against mpmath at 200 more bits;
``expint`` of a non-integer order has no value in the Rust core: there
the values, and what Python takes from them, are not compared)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    x, r, p = Symbol('x'), Symbol('r', real=True), Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(3), Integer(-4), Rational(1, 2),
            Rational(-3, 4), sqrt(2), I, 1 + I, Rational(1, 3) - 2*I, oo, -oo, x, r, p, -x, I*p, -p]


def orders() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, sqrt
    return [Integer(0), Integer(1), Integer(2), Integer(3), Integer(-1), Rational(1, 2), Rational(3, 2), sqrt(2),
            Symbol('n', integer=True, positive=True)]


ONE = ['Ei', 'Si', 'Ci', 'Shi', 'Chi', 'li', 'Li', 'fresnels', 'fresnelc']


@pytest.mark.parametrize("seed", range(40))
def test_integral_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Rational, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import error_functions as ef
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("integral-functions", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(ONE + ['expint', 'expint'])
        args = [rng.choice(orders()), rng.choice(arguments())] if name == 'expint' else [rng.choice(arguments())]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(ef, name)(*args)), (name, args)
        try:
            e = getattr(ef, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        order = args[0]
        partial = (name == 'expint' and not e.free_symbols
                   and not (isinstance(order, Rational) and order.q == 1 and order.p >= 0))
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
            g = outcome(lambda: rc.conjugate(r))
            if g != 'SKIP':
                assert g == outcome(lambda: conjugate(e)), e
        g = outcome(lambda: rc.diff(r, (x, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, x)), e
        if not e.free_symbols and not partial:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import error_functions as ef
    rc = load("sympy_extras_rs.core")
    points: list[Any] = [Rational(1, 7), sqrt(2), Integer(5), Integer(-3), 2 + 3*I, Rational(1, 2) - 4*I, 6*I,
                         Rational(-9, 2), Integer(12)]
    classes = [ef.EiNode, ef.SiNode, ef.CiNode, ef.ShiNode, ef.ChiNode, ef.LiNode, ef.OffsetLiNode, ef.FresnelS,
               ef.FresnelC]
    for z in points:
        nodes = [c(z) for c in classes] + [ef.Expint(Integer(n), z) for n in (0, 1, 2, 5)]
        for e in nodes:
            for prec in (30, 53, 100, 300):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)
