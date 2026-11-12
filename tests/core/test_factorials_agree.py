"""The factorials of the Rust core against
``sympy_extras.functions.combinatorial.factorials``: ``factorial``,
``binomial``, ``rf``, ``ff``, ``subfactorial`` and ``factorial2`` of
random integers, rationals, radicals, imaginary numbers, infinities and
symbols with assumptions, compared by structure (or the error), facts
and values (the gamma function of the Rust core against mpmath's, at
several precisions)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt
    n, m, k = Symbol('n', integer=True, nonnegative=True), Symbol('m', integer=True), Symbol('k', integer=True)
    x, p = Symbol('x'), Symbol('p', positive=True)
    return [Integer(0), Integer(1), Integer(5), Integer(12), Integer(-3), Integer(-4), Rational(1, 2),
            Rational(-7, 3), Rational(5, 2), sqrt(2), pi, I, 1 + I, oo, -oo, n, m, k, x, p, n + 1, m - 2, 2*n,
            2*n + 1]


def finite(v: Any) -> Any:
    """mpmath's infinite value (a pole of gamma in the numerator) is no
    value for the Rust core."""
    import mpmath
    if v is not None and (mpmath.isinf(v.real) or mpmath.isinf(v.imag)):
        return None
    return v


NAMES = {'factorial': 1, 'binomial': 2, 'rf': 2, 'ff': 2, 'subfactorial': 1, 'factorial2': 1}


@pytest.mark.parametrize("seed", range(40))
def test_factorials(rust: Any, seed: int) -> None:
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.combinatorial import factorials
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("factorials", seed))
    for _ in range(12):
        name = rng.choice(list(NAMES))
        args = [rng.choice(arguments()) for _ in range(NAMES[name])]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        expected = outcome(lambda: getattr(factorials, name)(*args))
        assert got == expected, (name, args)
        if isinstance(expected, str):
            continue
        e = getattr(factorials, name)(*args)
        r = rc.logic_call(name, args)
        assert rc.facts(r) == dict(facts_of(e)), e
        if not e.free_symbols:
            for prec in (53, 120):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(r, prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_gamma_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.combinatorial.factorials import Factorial, FallingFactorial, RisingFactorial
    rc = load("sympy_extras_rs.core")
    points: list[Any] = [sqrt(2), Rational(1, 2), Rational(-1, 2), Rational(-7, 3), pi, I, 1 + I, -3 + 2*I,
                         Integer(30) + I/7, Rational(1, 1000), Integer(-5) + Rational(1, 10**6), 50*I]
    for z in points:
        for e in [Factorial(z), RisingFactorial(z, sqrt(3)), FallingFactorial(z, Rational(1, 3))]:
            for prec in (30, 53, 100, 300):
                # Python's own value near a pole is only as good as the
                # argument rounded to prec: the truth at a higher precision
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)
