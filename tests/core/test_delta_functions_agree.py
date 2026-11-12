"""``DiracDelta`` and ``Heaviside`` of the Rust core against
``sympy_extras.functions.special.delta_functions``: random arguments
(numbers, symbols with signs, sums, imaginary terms), orders and values
at zero, compared by structure (or the error), facts, values, derivatives
and conjugates; and the derivatives which use them (``Max``, ``Min``,
``sign``)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, pi, sqrt
    x, p, n, z = Symbol('x', real=True), Symbol('p', positive=True), Symbol('n', negative=True), Symbol('z')
    return [Integer(0), Integer(2), Integer(-3), Rational(1, 2), sqrt(2) - 1, pi - 3, x, -x, 2*x, -3*x, x - 1,
            1 - x, p, n, p + 1, -p, z, I, I*x, x + I, x**2, -x**3, x*p]


@pytest.mark.parametrize("seed", range(40))
def test_delta_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import delta_functions as d
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("delta", seed))
    x = Symbol('x', real=True)
    for _ in range(10):
        arg = rng.choice(arguments())
        if rng.random() < 0.5:
            second: Any = rng.choice([None, Integer(0), Integer(1), Integer(2), Integer(3), Integer(-1),
                                      Rational(1, 2)])
            expected = outcome(lambda: d.dirac_delta(arg) if second is None else d.dirac_delta(arg, second))
            got = outcome(lambda: rc.dirac_delta(arg, second))
            build = (lambda: d.dirac_delta(arg) if second is None else d.dirac_delta(arg, second))
        else:
            second = rng.choice([None, Integer(0), Integer(1), Rational(1, 2), Symbol('h')])
            expected = outcome(lambda: d.heaviside(arg) if second is None else d.heaviside(arg, second))
            got = outcome(lambda: rc.heaviside(arg, second))
            build = (lambda: d.heaviside(arg) if second is None else d.heaviside(arg, second))
        assert got == expected, (arg, second)
        if isinstance(expected, str):
            continue
        e = build()
        r = rc.roundtrip(e)
        assert repr(r) == repr(e)
        assert rc.facts(r) == dict(facts_of(e)), e
        if not e.free_symbols:
            ve, vr = numeric_value(e, 53), rc.numeric_value(r, 53)
            assert (ve is None) == (vr is None), e
            if ve is not None:
                assert close(vr, ve, 53), e
        assert outcome(lambda: rc.diff(r, (x, 1))) == outcome(lambda: diff(e, x)), e
        assert outcome(lambda: rc.conjugate(r)) == outcome(lambda: conjugate(e)), e


def test_derivatives_through_the_step(rust: Any) -> None:
    from sympy_extras.core import I, Symbol
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import sign
    from sympy_extras.functions.elementary.miscellaneous import max_, min_
    rc = load("sympy_extras_rs.core")
    x, y, p = Symbol('x', real=True), Symbol('y', real=True), Symbol('p', positive=True)
    for e in [max_(x, 1), min_(x, y), max_(x, y, 2), min_(x**2, x, 3), max_(x*p, y), min_(x, -x),
              sign(x), sign(x**2 - 1), sign(I*x), sign(Symbol('z'))]:
        for v in [x, y]:
            assert outcome(lambda: rc.diff(e, (v, 1))) == outcome(lambda: diff(e, v)), (e, v)
