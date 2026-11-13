"""The error functions of the Rust core against
``sympy_extras.functions.special.error_functions``: ``erf``, ``erfc``,
``erfi``, ``erfinv``, ``erfcinv``, ``erf2`` and ``erf2inv`` of random
numbers, infinities, nested inverses and symbols, compared by structure
(or the error), facts, conjugates, derivatives, real and imaginary parts
and values (the Taylor series of the Rust core and Newton's method for
the inverse against mpmath at 200 more bits)."""

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
    from sympy_extras.functions.special.error_functions import erf, erfcinv, erfinv
    x, r, p = Symbol('x'), Symbol('r', real=True), Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(3), Rational(1, 2), Rational(-3, 4), sqrt(2),
            I, 1 + I, Rational(1, 3) - 2*I, oo, -oo, I*oo, x, r, p, -x, x + I, -p, erf(r), erfinv(x), erfcinv(x),
            Rational(1, 2)*I]


ONE = ['erf', 'erfc', 'erfi', 'erfinv', 'erfcinv']
TWO = ['erf2', 'erf2inv']


@pytest.mark.parametrize("seed", range(40))
def test_error_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import error_functions as ef
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("error-functions", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(ONE + TWO)
        args = [rng.choice(arguments()) for _ in range(2 if name in TWO else 1)]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(ef, name)(*args)), (name, args)
        try:
            e = getattr(ef, name)(*args)
        except Exception:
            continue
        r = rc.logic_call(name, args)
        assert rc.facts(r) == dict(facts_of(e)), e
        for got_, expected_ in ((lambda: rc.conjugate(r), lambda: conjugate(e)),
                                (lambda: rc.diff(r, (x, 1)), lambda: diff(e, x)),
                                (lambda: rc.as_real_imag(r), lambda: e.as_real_imag())):
            g = outcome(got_)
            if g != 'SKIP':
                assert g == outcome(expected_), e
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
    from sympy_extras.functions.special.error_functions import Erf, Erf2, Erfc, Erfcinv, Erfi, Erfinv
    rc = load("sympy_extras_rs.core")
    points: list[Any] = [Rational(1, 7), sqrt(2), Integer(4), Integer(-5), 2 + 3*I, Rational(1, 2) - 4*I, 6*I,
                         Rational(-9, 2)]
    for z in points:
        for e in [Erf(z), Erfc(z), Erfi(z), Erf2(z, Integer(1))]:
            for prec in (30, 53, 100, 300):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)
    for v in [Rational(1, 3), Rational(-9, 10), Rational(99, 100), Rational(1, 10**6)]:
        for e in [Erfinv(v), Erfcinv(1 + v)]:
            for prec in (30, 53, 200):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert close(vr, ve, prec), (e, prec)
