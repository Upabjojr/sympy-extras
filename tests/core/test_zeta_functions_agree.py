"""The zeta functions of the Rust core against
``sympy_extras.functions.special.zeta_functions``: ``zeta`` (Riemann and
Hurwitz), ``dirichlet_eta``, ``polylog``, ``lerchphi``, ``riemann_xi``
and ``stieltjes`` of random integers, rationals, imaginary numbers and
symbols, compared by structure (or the error), facts, conjugates and
derivatives, and the values (Euler-Maclaurin against mpmath at 200 more
bits). The values of ``polylog`` outside ``|z| < 3/4``, of ``lerchphi``
and of ``stieltjes`` are not ported: where the Rust core has none, the
values (and the facts Python takes from them) are not compared; with them
``polygamma`` at the integers and at 1/2."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def orders() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo
    return [Integer(n) for n in range(-5, 7)] + [Rational(1, 2), Rational(-3, 2), Rational(5, 3), 2 + I, I,
                                                oo, Symbol('s'), Symbol('n', integer=True), Symbol('k', positive=True)]


def points() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, sqrt
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(3), Integer(-2), Rational(1, 2), Rational(3, 2),
            Rational(-1, 2), Rational(1, 3), Rational(-1, 4) + I/5, sqrt(2), I, -I, 1 + I, (1 - I)/2, oo,
            Symbol('x'), Symbol('a', positive=True), (sqrt(5) - 1)/2]


PARTIAL = {'polylog', 'lerchphi', 'stieltjes'}


def mpmath_pole_value(name: str, args: list[Any]) -> bool:
    """A Hurwitz zeta at a nonpositive integer ``a`` with ``re(s) > 0``: a
    pole, where mpmath's complex path gives a number (``zeta(mpc(5/3),
    mpc(0))`` is 3.12..., its real path raises NoConvergence); the Rust
    core has no value there."""
    from sympy_extras.core import Integer, Rational
    from sympy_extras.functions.elementary.complexes import re
    if name not in ('zeta', 'dirichlet_eta') or len(args) < 2:
        return False
    s, a = args
    candidates = [a] if name == 'zeta' else [a, (a + 1)/Integer(2)]
    poles = [c for c in candidates if isinstance(c, Rational) and c.q == 1 and c.p <= 0]
    return bool(poles) and re(s).is_positive is True


@pytest.mark.parametrize("seed", range(40))
def test_zeta_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    from sympy_extras.functions.special import zeta_functions as zf
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("zeta-functions", seed))
    x = Symbol('x')
    for _ in range(12):
        name = rng.choice(['zeta', 'dirichlet_eta', 'polylog', 'lerchphi', 'riemann_xi', 'stieltjes'])
        if name in ('zeta', 'dirichlet_eta', 'stieltjes'):
            args = [rng.choice(orders())] + ([rng.choice(points())] if rng.random() < 0.6 else [])
        elif name == 'polylog':
            args = [rng.choice(orders()), rng.choice(points())]
        elif name == 'lerchphi':
            args = [rng.choice(points()), rng.choice(orders()), rng.choice(points())]
        else:
            args = [rng.choice(orders())]
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        expected = outcome(lambda: getattr(zf, name)(*args))
        assert got == expected, (name, args)
        if isinstance(expected, str):
            continue
        e = getattr(zf, name)(*args)
        r = rc.logic_call(name, args)
        vr = rc.numeric_value(r, 53) if not e.free_symbols else None
        partial = vr is None and (name in PARTIAL or mpmath_pole_value(name, args)) and not e.free_symbols
        if not partial:
            assert rc.facts(r) == dict(facts_of(e)), e
        checks: list[tuple[Any, Any]] = [(lambda: rc.diff(r, (x, 1)), lambda: diff(e, x))]
        if not partial:
            # the conjugate of a real value is itself: Python knows it from the value
            checks.append((lambda: rc.conjugate(r), lambda: conjugate(e)))
        for got_, expected_ in checks:
            g_ = outcome(got_)
            if g_ != 'SKIP':
                assert g_ == outcome(expected_), e
        if not e.free_symbols:
            for prec in (53, 120):
                vr = rc.numeric_value(r, prec)
                if vr is None and (name in PARTIAL or mpmath_pole_value(name, args)):
                    continue
                ve = finite(numeric_value(e, prec + 200))
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_values(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, pi, sqrt
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special.zeta_functions import DirichletEta, PolyLog, RiemannXi, Zeta
    rc = load("sympy_extras_rs.core")
    orders_: list[Any] = [Rational(1, 2), Integer(3), Rational(-7, 2), 2 + I, Rational(1, 2) + 14*I, sqrt(2), pi]
    for s in orders_:
        for e in [Zeta(s), Zeta(s, Rational(1, 3)), Zeta(s, 5 + I), Zeta(s, Rational(-7, 3)), DirichletEta(s),
                  DirichletEta(s, Rational(3, 2)), RiemannXi(s), PolyLog(s, Rational(1, 2)),
                  PolyLog(s, Rational(-2, 3) + I/5)]:
            for prec in (30, 53, 100, 300):
                ve, vr = finite(numeric_value(e, prec + 200)), rc.numeric_value(rc.roundtrip(e), prec)
                assert (ve is None) == (vr is None), (e, prec)
                if ve is not None:
                    assert close(vr, ve, prec), (e, prec)


def test_polygamma_through_zeta(rust: Any) -> None:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.functions.special.gamma_functions import polygamma, trigamma
    rc = load("sympy_extras_rs.core")
    for n in [Integer(1), Integer(2), Integer(3)]:
        for z in [Integer(1), Integer(2), Integer(5), Rational(1, 2)]:
            assert outcome(lambda: rc.logic_call('polygamma', [n, z])) == outcome(lambda: polygamma(n, z)), (n, z)
    assert outcome(lambda: rc.logic_call('trigamma', [Integer(2)])) == outcome(lambda: trigamma(Integer(2)))
