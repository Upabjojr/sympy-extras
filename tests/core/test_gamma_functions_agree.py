"""The gamma functions of the Rust core against
``sympy_extras.functions.special.gamma_functions``: ``gamma``,
``loggamma``, ``polygamma``, ``digamma`` and ``trigamma`` of random
integers, half-integers, rationals of small denominators, radicals,
imaginary numbers, infinities and symbols with assumptions, compared by
structure (or the error), facts, conjugates and derivatives, and the
values (the Stirling series of the Rust core against mpmath's at 200
more bits); with them the derivatives of the factorials and the gamma
form of the binomial. ``polygamma(n, z)`` for ``n >= 1`` at the integers
and at 1/2 needs ``zeta``, not ported: skipped."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt, zoo
    n, k = Symbol('n', integer=True, positive=True), Symbol('k', integer=True)
    x, p, r = Symbol('x'), Symbol('p', positive=True), Symbol('r', noninteger=True, real=True)
    return [Integer(0), Integer(1), Integer(4), Integer(-2), Rational(1, 2), Rational(7, 2), Rational(-5, 2),
            Rational(1, 3), Rational(-5, 6), Rational(3, 4), Rational(2, 7), sqrt(2), pi, I, 2 + I, oo, -oo, zoo,
            n, k, x, p, r, p + 1, I*oo]


NAMES = {'gamma': 1, 'loggamma': 1, 'polygamma': 2, 'digamma': 1, 'trigamma': 1}


@pytest.mark.parametrize("seed", range(40))
def test_gamma_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import gamma_functions as g
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("gamma-functions", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(list(NAMES))
        args = [rng.choice(arguments()) for _ in range(NAMES[name])]
        if name == 'polygamma':
            args[0] = rng.choice([Integer(-1), Integer(0), Integer(1), Integer(2), Integer(3), Symbol('m')])
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        expected = outcome(lambda: getattr(g, name)(*args))
        assert got == expected, (name, args)
        if isinstance(expected, str):
            continue
        e = getattr(g, name)(*args)
        r = rc.logic_call(name, args)
        assert rc.facts(r) == dict(facts_of(e)), e
        for got_, expected_ in ((lambda: rc.conjugate(r), lambda: conjugate(e)),
                                (lambda: rc.diff(r, (x, 1)), lambda: diff(e, x))):
            g_ = outcome(got_)
            if g_ != 'SKIP':
                assert g_ == outcome(expected_), e
        if not e.free_symbols:
            for prec in (53, 120):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(r, prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special.gamma_functions import Gamma, LogGamma, PolyGamma
    rc = load("sympy_extras_rs.core")
    points: list[Any] = [sqrt(2), Rational(1, 3), Rational(-1, 3), Rational(-7, 3), pi, I, 1 + I, -3 + 2*I,
                         Integer(30) + I/7, Rational(1, 1000), -2 - I/5, 40*I, -Rational(11, 2) + I]
    for z in points:
        for e in [Gamma(z), LogGamma(z), PolyGamma(Integer(0), z), PolyGamma(Integer(1), z),
                  PolyGamma(Integer(3), z)]:
            for prec in (30, 53, 100, 300):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_factorials_through_gamma(rust: Any) -> None:
    from sympy_extras.core import Rational, Symbol, sqrt
    from sympy_extras.core.function import diff
    from sympy_extras.functions.combinatorial import factorials as f
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    for e in [f.factorial(x), f.factorial(x**2), f.binomial(x, y), f.binomial(x, 3*x), f.rf(x, y), f.ff(y, x),
              f.subfactorial(x), f.factorial2(x)]:
        for v in [x, y]:
            g_ = outcome(lambda: rc.diff(e, (v, 1)))
            if g_ != 'SKIP':
                assert g_ == outcome(lambda: diff(e, v)), (e, v)
    for n, k in [(Rational(5), Rational(1, 2)), (sqrt(2), Rational(1, 3)), (Rational(-1, 2), sqrt(3))]:
        assert outcome(lambda: rc.logic_call('binomial', [n, k])) == outcome(lambda: f.binomial(n, k)), (n, k)
