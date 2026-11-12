"""``SingularityFunction`` of the Rust core against
``sympy_extras.functions.special.singularity_functions``: random
arguments, offsets and exponents (numbers, infinities, nan and symbols
with assumptions), compared by structure (or the error), facts,
conjugates, derivatives and values (against mpmath at 200 more bits)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def points() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, nan, oo, sqrt
    x, r, p = Symbol('x'), Symbol('r', real=True), Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(-2), Rational(3, 2), sqrt(2), I, 1 + I, oo, -oo, nan, x, r, p, -p, r + 1]


def exponents() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, nan
    return [Integer(0), Integer(1), Integer(2), Integer(5), Integer(-1), Integer(-2), Integer(-3), Integer(-4),
            Integer(-5), Rational(1, 2), Rational(-1, 2), I, nan, Symbol('n'), Symbol('k', positive=True, integer=True)]


@pytest.mark.parametrize("seed", range(40))
def test_singularity_function(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special.singularity_functions import SingularityFunction
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("singularity", seed))
    x = Symbol('x')
    for _ in range(15):
        args = [rng.choice(points()), rng.choice(points()), rng.choice(exponents())]
        got = outcome(lambda: rc.logic_call('SingularityFunction', args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: SingularityFunction(*args)), args
        try:
            e = SingularityFunction(*args)
        except Exception:
            continue
        r = rc.logic_call('SingularityFunction', args)
        assert rc.facts(r) == dict(facts_of(e)), e
        g = outcome(lambda: rc.conjugate(r))
        if g != 'SKIP':
            assert g == outcome(lambda: conjugate(e)), e
        for v in (x, Symbol('r', real=True), Symbol('p', positive=True)):
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


def test_values(rust: Any) -> None:
    from sympy_extras.core import Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special.singularity_functions import SingularityFunctionNode
    rc = load("sympy_extras_rs.core")
    for x in (Integer(3), sqrt(5), Rational(-1, 3), pi, Integer(2)):
        for a in (Integer(2), Rational(1, 7)):
            for n in (Integer(0), Integer(3), Rational(5, 2), sqrt(2), Integer(-1), Integer(-3)):
                e = SingularityFunctionNode(x, a, n)
                for prec in (30, 53, 100, 300):
                    ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                    assert (ve is None) == (vr is None), (e, prec)
                    if ve is not None:
                        assert close(vr, ve, prec), (e, prec)
