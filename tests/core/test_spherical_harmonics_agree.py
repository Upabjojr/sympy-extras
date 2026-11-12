"""The spherical harmonics of the Rust core against
``sympy_extras.functions.special.spherical_harmonics``: ``Ynm``, ``Znm``
and ``Ynm_c`` of random degrees, orders and angles (numbers, symbols,
negated ones), compared by structure (or the error), facts (not for a
number without a Rust value), conjugates,
derivatives and values (mpmath's ``spherharm`` at 200 more bits, for an
integer degree and order: Rust has no other value)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def degrees() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    return [Integer(0), Integer(1), Integer(2), Integer(3), Integer(5), Integer(-2), Rational(1, 2), Symbol('n'),
            Symbol('n', integer=True, nonnegative=True)]


def orders() -> list[Any]:
    from sympy_extras.core import Integer, Symbol
    m = Symbol('m')
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-3), Integer(4), m, -m,
            Symbol('k', positive=True), -Symbol('k', positive=True)]


def angles() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi
    t = Symbol('t')
    return [Integer(0), Integer(1), Rational(1, 3), pi/4, -pi/3, Integer(-2), 1 + I, t, -t, 2*t,
            Symbol('r', real=True)]


@pytest.mark.parametrize("seed", range(40))
def test_spherical_harmonics(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import spherical_harmonics as sh
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("spherical-harmonics", seed))
    t = Symbol('t')
    for _ in range(10):
        name = rng.choice(['Ynm', 'Ynm', 'Znm', 'Ynm_c'])
        args = [rng.choice(degrees()), rng.choice(orders()), rng.choice(angles()), rng.choice(angles())]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(sh, name)(*args)), (name, args)
        try:
            e = getattr(sh, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        # without a Rust value (a non-integer degree or order) Python's facts
        # of a number come from its value: not compared
        partial = not e.free_symbols and rc.numeric_value(r, 53) is None
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (t, Symbol('m'), Symbol('r', real=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                if vr is None:
                    continue
                ve = finite(numeric_value(e, prec + 200))
                assert ve is not None, (e, prec)
                assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special.spherical_harmonics import YnmNode
    rc = load("sympy_extras_rs.core")
    for n in (0, 1, 2, 3, 7, 12, -4):
        for m in (0, 1, -1, 2, -3, 5, 13):
            for theta in (Rational(1, 3), pi/5, sqrt(2), 2 + I, Integer(0), pi):
                for phi in (Rational(2, 7), -pi/3, 1 - I):
                    e = YnmNode(Integer(n), Integer(m), theta, phi)
                    for prec in (30, 53, 100, 300):
                        ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                        assert (ve is None) == (vr is None), (e, prec)
                        if ve is not None:
                            assert close(vr, ve, prec), (e, prec)
