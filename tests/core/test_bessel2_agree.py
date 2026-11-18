"""The Bessel functions of the Rust core (part 2) against
``sympy_extras.functions.special.bessel``: the spherical ``jn``, ``yn``,
``hn1``, ``hn2``, the Airy functions and ``marcumq`` of random arguments,
compared by structure (or the error), facts, conjugates, complex parts
(Airy), derivatives and values (where Rust has one it must agree; a
numerical node without a Rust value is not compared on the facts Python
takes from its value)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bessel_agree import arguments, orders
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

SPHERICAL = ['jn', 'yn', 'hn1', 'hn2']
AIRY = ['airyai', 'airybi', 'airyaiprime', 'airybiprime']
CLASSES = ['SphericalJ', 'SphericalY', 'SphericalHankel1', 'SphericalHankel2', 'AiryAi', 'AiryBi', 'AiryAiPrime',
           'AiryBiPrime', 'MarcumQ', 'BesselI']


def airy_arguments(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol, nan
    return rng.choice([arguments(rng), arguments(rng), Integer(-3), Rational(5, 2), Integer(4), 2 - I, nan,
                       Symbol('x', real=True)])


def marcumq_arguments(rng: random.Random) -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol
    a = Symbol('a')
    pick = lambda: rng.choice([Integer(0), Integer(1), Integer(2), Rational(1, 2), a, Symbol('b'), Symbol('m')])
    m, x, y = pick(), pick(), pick()
    if rng.random() < 0.3:
        y = x
    return [m, x, y]


def check(rc: Any, e: Any, r: Any, classes: tuple[Any, ...], airy: bool) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    numeric = not e.free_symbols and e.has(*classes)
    if not numeric or rc.numeric_value(r, 53) is not None:
        assert rc.facts(r) == dict(facts_of(e)), e
    g = outcome(lambda: rc.conjugate(r))
    if g != 'SKIP':
        assert g == outcome(lambda: conjugate(e)), e
    if airy:
        g = outcome(lambda: rc.as_real_imag(r))
        if g != 'SKIP':
            assert g == outcome(lambda: e.as_real_imag()), e
    for v in (Symbol('x'), Symbol('x', real=True), Symbol('nu'), Symbol('a'), Symbol('b'), Symbol('m')):
        g = outcome(lambda: rc.diff(r, (v, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, v)), (e, v)
    if not e.free_symbols:
        for prec in (53, 120):
            vr = rc.numeric_value(r, prec)
            if vr is not None:
                ve = finite(numeric_value(e, prec + 200))
                assert ve is not None and close(vr, ve, prec), (e, prec, vr, ve)


@pytest.mark.parametrize("seed", range(50))
def test_bessel_part2(rust: Any, seed: int) -> None:
    from sympy_extras.functions.special import bessel as bs
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("bessel2", seed))
    classes = tuple(getattr(bs, c) for c in CLASSES)
    for _ in range(10):
        kind = rng.random()
        if kind < 0.4:
            name = rng.choice(SPHERICAL)
            args = [orders(rng), arguments(rng)]
        elif kind < 0.8:
            name = rng.choice(AIRY)
            args = [airy_arguments(rng)]
        else:
            name = 'marcumq'
            args = marcumq_arguments(rng)
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(bs, name)(*args)), (name, args)
        try:
            e = getattr(bs, name)(*args)
        except Exception:
            continue
        check(rc, e, rc.logic_call(name, args), classes, name in AIRY)


def test_values(rust: Any) -> None:
    """The values Rust has, against mpmath's."""
    from sympy_extras.core import I, Integer, Rational
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import bessel as bs
    rc = load("sympy_extras_rs.core")
    cases = [('jn', [0, 3]), ('jn', [2, Rational(1, 3)]), ('yn', [1, 2 + I]), ('hn1', [1, 3]), ('hn2', [0, 1 + I]),
             ('airyai', [3]), ('airyai', [-5]), ('airybi', [Rational(1, 2)]), ('airyaiprime', [2 - I]),
             ('airybiprime', [-2]), ('airyai', [6])]
    for name, args in cases:
        args = [Integer(0) + a for a in args]
        e, r = getattr(bs, name)(*args), rc.logic_call(name, args)
        for prec in (53, 200):
            vr = rc.numeric_value(r, prec)
            assert vr is not None, (e, prec)
            assert close(vr, finite(numeric_value(e, prec + 200)), prec), (e, prec)
