"""The Mathieu functions of the Rust core against
``sympy_extras.functions.special.mathieu_functions``: ``mathieus``,
``mathieuc``, ``mathieusprime`` and ``mathieucprime`` of random
parameters and arguments (numbers, a zero ``q``, symbols,
negated ones), compared by structure (or the error), facts, conjugates,
derivatives and values (none for the Mathieu nodes: only their
trigonometric limits have values)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome

NAMES = ['mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime']


def values() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi
    a, z = Symbol('a'), Symbol('z')
    return [Integer(0), Integer(1), Integer(4), Rational(-1, 2), I, 2 - I, pi/3, a, -a, z, -z,
            2*z, Symbol('r', real=True), -Symbol('p', positive=True)]


@pytest.mark.parametrize("seed", range(40))
def test_mathieu(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import mathieu_functions as mf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("mathieu", seed))
    for _ in range(12):
        name = rng.choice(NAMES)
        args = [rng.choice(values()) for _ in range(3)]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(mf, name)(*args)), (name, args)
        try:
            e = getattr(mf, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (Symbol('z'), Symbol('a'), Symbol('r', real=True)):
            g = outcome(lambda: rc.diff(r, (v, 1)))
            if g != 'SKIP':
                assert g == outcome(lambda: diff(e, v)), (e, v)
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)
