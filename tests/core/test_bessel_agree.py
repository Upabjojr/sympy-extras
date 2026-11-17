"""The Bessel functions of the Rust core (part 1) against
``sympy_extras.functions.special.bessel``: ``besselj``, ``bessely``,
``besseli``, ``besselk``, ``hankel1``, ``hankel2`` and the helpers
``_besseli``, ``_besselk`` of random orders and arguments, compared by
structure (or the error), facts, conjugates, derivatives and values.
Rust has values for ``besselj``/``besseli`` by the series of ``0F1`` and
for the others at a non-integer order only: where Rust has a value it
must agree; a numerical node without a Rust value is not compared on
the facts Python takes from its value."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

NAMES = ['besselj', 'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'besselis', 'besselks']
CLASSES = ['BesselJ', 'BesselY', 'BesselI', 'BesselK', 'Hankel1', 'Hankel2', 'BesselIs', 'BesselKs']


def orders(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    n = Symbol('n', integer=True)
    return rng.choice([Integer(0), Integer(rng.randint(1, 4)), Integer(rng.randint(-4, -1)), Rational(1, 2),
                       Rational(-1, 2), Rational(7, 3), Rational(-5, 3), I, 1 + I, n, -n, Symbol('nu'),
                       Symbol('p', positive=True), Symbol('q', negative=True), Symbol('w', imaginary=True)])


def arguments(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    x = Symbol('x')
    return rng.choice([Integer(0), Integer(1), Integer(-2), Rational(1, 3), Rational(9, 2), sqrt(2), I, 2*I, 1 + I,
                       -1 + I/2, oo, -oo, I*oo, -I*oo, x, -x, I*x, Symbol('r', real=True), Symbol('s', positive=True),
                       -Symbol('s', positive=True)])


@pytest.mark.parametrize("seed", range(50))
def test_bessel(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import bessel as bs
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("bessel", seed))
    classes = tuple(getattr(bs, c) for c in CLASSES)
    for _ in range(10):
        name = rng.choice(NAMES)
        args = [orders(rng), arguments(rng)]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(bs, name)(*args)), (name, args)
        try:
            e = getattr(bs, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        numeric = not e.free_symbols and e.has(*classes)
        if not numeric or rc.numeric_value(r, 53) is not None:
            assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (Symbol('x'), Symbol('nu'), Symbol('n', integer=True), Symbol('r', real=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                if vr is not None:
                    ve = finite(numeric_value(e, prec + 200))
                    assert ve is not None and close(vr, ve, prec), (e, prec, vr, ve)


def test_values(rust: Any) -> None:
    """The values Rust has, against mpmath's."""
    from sympy_extras.core import I, Integer, Rational
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import bessel as bs
    rc = load("sympy_extras_rs.core")
    cases = [('besselj', 0, 3), ('besselj', 2, Rational(1, 3)), ('besselj', Rational(1, 3), 5 + I),
             ('besseli', 1, Rational(7, 2)), ('besseli', Rational(-1, 3), 2*I + 1), ('bessely', Rational(1, 2), 2),
             ('bessely', Rational(2, 3), 1 + I), ('besselk', Rational(1, 3), Rational(1, 2)),
             ('besselk', Rational(5, 2), 4), ('hankel1', Rational(1, 4), 3), ('hankel2', Rational(1, 4), 1 + 2*I),
             ('besselj', 3, -Integer(2) + I)]
    for name, nu, z in cases:
        e, r = getattr(bs, name)(nu, z), rc.logic_call(name, [Integer(0) + nu, Integer(0) + z])
        for prec in (53, 200):
            vr = rc.numeric_value(r, prec)
            assert vr is not None, (e, prec)
            assert close(vr, finite(numeric_value(e, prec + 200)), prec), (e, prec)
