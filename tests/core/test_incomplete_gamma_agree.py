"""The incomplete gammas and multigamma of the Rust core against
``sympy_extras.functions.special.gamma_functions``: ``lowergamma``,
``uppergamma`` and ``multigamma`` of random orders and arguments
(numbers, infinities and symbols), compared by structure (or the error),
facts (not where Python's multigamma raises TypeError comparing a
non-real argument), conjugates, derivatives and values (the Rust series
against mpmath's ``gammainc`` at 200 more bits)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def orders() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    return [Integer(0), Integer(1), Integer(2), Integer(4), Integer(-1), Integer(-3), Rational(1, 2), Rational(3, 2),
            Rational(7, 2), Rational(-1, 2), Rational(-5, 2), Rational(1, 3), sqrt(2), 1 + I,
            Symbol('n', integer=True, positive=True), Symbol('a')]


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, nan, oo, sqrt
    x, p = Symbol('x'), Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(2), Integer(-3), Rational(1, 2), sqrt(3), I, 2 - I, oo, -oo, nan, x, p,
            -x, 2*x]


def multi_orders() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    return [Integer(1), Integer(2), Integer(3), Integer(0), Integer(-2), Rational(1, 2), Symbol('m'),
            Symbol('m', integer=True, positive=True)]


@pytest.mark.parametrize("seed", range(40))
def test_incomplete_gamma(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import gamma_functions as gf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("incomplete-gamma", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(['lowergamma', 'uppergamma', 'uppergamma', 'multigamma'])
        if name == 'multigamma':
            args = [rng.choice(arguments() + orders()), rng.choice(multi_orders())]
        else:
            args = [rng.choice(orders()), rng.choice(arguments())]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(gf, name)(*args)), (name, args)
        try:
            e = getattr(gf, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        try:
            expected = dict(facts_of(e))
        except TypeError:
            # Python's multigamma compares a non-real 2*x with p - 1
            # (SymPy, and the Rust core, leave the fact unknown)
            expected = None
        if expected is not None:
            assert rc.facts(r) == expected, e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        g = outcome(lambda: rc.diff(r, (x, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, x)), e
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import gamma_functions as gf
    rc = load("sympy_extras_rs.core")
    orders_: list[Any] = [Rational(1, 3), sqrt(2), Integer(-2), Integer(-5), Rational(-7, 3), 2 + 3*I, Rational(1, 2) - I,
                          Integer(6)]
    points: list[Any] = [Rational(1, 7), sqrt(2), Integer(5), Integer(-3), 2 + 3*I, 6*I, Integer(15)]
    for a in orders_:
        for z in points:
            for e in (gf.LowerGamma(a, z), gf.UpperGamma(a, z)):
                for prec in (30, 53, 100, 300):
                    ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                    assert (ve is None) == (vr is None), (e, prec)
                    if ve is not None:
                        assert close(vr, ve, prec), (e, prec)
