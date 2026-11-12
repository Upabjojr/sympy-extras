"""The elliptic integrals of the Rust core against
``sympy_extras.functions.special.elliptic_integrals``: ``elliptic_k``,
``elliptic_f``, ``elliptic_e`` (complete and incomplete) and
``elliptic_pi`` (complete and incomplete) of random arguments (numbers,
infinities, multiples of ``pi/2``, symbols, negated ones), compared by
structure (or the error), facts, conjugates, derivatives and values
(mpmath at 200 more bits; ``elliptic_pi`` has no Rust value: there the
values and the facts Python takes from them are not compared)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

ARITY = {'elliptic_k': [1], 'elliptic_f': [2], 'elliptic_e': [1, 2], 'elliptic_pi': [2, 3]}


def values() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, zoo
    m, z = Symbol('m'), Symbol('z')
    return [Integer(0), Integer(1), Integer(-1), Rational(1, 2), Integer(2), Rational(-3, 4), Rational(1, 3), I,
            1 + I, oo, -oo, zoo, I*oo, pi/2, -pi, 3*pi/2, pi/5, m, z, -z, Symbol('r', real=True),
            Symbol('p', positive=True)]


@pytest.mark.parametrize("seed", range(40))
def test_elliptic_integrals(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import elliptic_integrals as ei
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("elliptic-integrals", seed))
    for _ in range(12):
        name = rng.choice(list(ARITY))
        args = [rng.choice(values()) for _ in range(rng.choice(ARITY[name]))]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(ei, name)(*args)), (name, args)
        try:
            e = getattr(ei, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        partial = not e.free_symbols and rc.numeric_value(r, 53) is None and e.has(ei.EllipticPi)
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (Symbol('z'), Symbol('m'), Symbol('r', real=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols and not partial:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import elliptic_integrals as ei
    rc = load("sympy_extras_rs.core")
    ms: list[Any] = [Rational(1, 3), Rational(-5, 2), Rational(9, 10), Integer(3), 2 + 3*I, Rational(1, 2) - I, -4*I,
                     Rational(1, 1000), Integer(-40), Rational(999, 1000)]
    zs: list[Any] = [Rational(1, 7), Integer(1), Integer(-2), Integer(5), 2 + I, Rational(1, 3) - 2*I, 7*pi/3, sqrt(2)]
    nodes: list[Any] = []
    for m in ms:
        nodes += [ei.EllipticK(m), ei.EllipticE(m)]
        nodes += [ei.EllipticF(z, m) for z in zs] + [ei.EllipticE(z, m) for z in zs]
    for e in nodes:
        for prec in (30, 53, 100, 300):
            ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
            assert (ve is None) == (vr is None), (e, prec)
            if ve is not None:
                assert close(vr, ve, prec), (e, prec, vr, ve)
