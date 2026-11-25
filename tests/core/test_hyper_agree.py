"""``hyper``, ``meijerg`` and ``appellf1`` of the Rust core against
``sympy_extras.functions.special.hyper``: random parameter lists (with
common parameters to cancel, nonpositive integers, symbols) and
arguments, compared by structure (or the error), facts, conjugates,
derivatives (``meijerg``'s in its parameters too), ``hyper``'s
``radius_of_convergence``, ``eta`` and ``convergence_statement``,
``meijerg``'s ``nu``, ``delta`` and ``get_period``, and values (the
ports of mpmath's ``hyper``, ``meijerg`` and ``appellf1``: where Rust
gives a value it must agree)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_bridge_agree import close
from .test_factorials_agree import finite
from .test_integers_agree import outcome


def close_in_modulus(rust_value: Any, python_value: Any, prec: int) -> bool:
    """``close`` with both parts measured against the modulus: the
    reference of a huge real value (``1F2`` at ``10**6``, ``e**2000``)
    carries an imaginary part of rounding noise, far below the precision
    of the modulus but not of itself."""
    from fractions import Fraction
    if rust_value is None or python_value is None:
        return rust_value is None and python_value is None
    re, im = rust_value
    approx: list[Fraction] = []
    for part in (python_value.real, python_value.imag):
        sign, man, exp, _ = part._mpf_
        value = Fraction(int(man) * 2**int(exp)) if exp >= 0 else Fraction(int(man), 2**int(-exp))
        approx.append(-value if sign else value)
    scale = max(Fraction(1), abs(approx[0]) + abs(approx[1]))
    tolerance = Fraction(1, 2**(prec - 12)) * scale
    return abs(re - approx[0]) <= tolerance and abs(im - approx[1]) <= tolerance


def parameter(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    a, b = Symbol('a'), Symbol('b')
    return rng.choice([Integer(1), Integer(2), Integer(0), Integer(-1), Integer(-2), Integer(-3), Rational(1, 2),
                       Rational(-1, 2), Rational(3, 2), Rational(1, 3), a, b, a + 1, 2*a, -a, Symbol('p', positive=True),
                       1 + I])


def parameters(rng: random.Random, k: int) -> list[Any]:
    return [parameter(rng) for _ in range(k)]


def argument(rng: random.Random) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    x, z = Symbol('x'), Symbol('z')
    return rng.choice([Integer(0), Integer(1), Integer(-1), Rational(1, 2), Rational(-1, 3), Integer(2), Integer(5),
                       I/2, Rational(1, 4) + I/3, x, z, x**2, 2*x, Symbol('r', real=True)])


def check(rc: Any, e: Any, r: Any, variables: tuple[Any, ...]) -> None:
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.core.function import diff
    from sympy_extras.functions.elementary.complexes import conjugate
    assert rc.facts(r) == dict(facts_of(e)), e
    g = outcome(lambda: rc.conjugate(r))
    if g != 'SKIP':
        assert g == outcome(lambda: conjugate(e)), e
    for v in variables:
        g = outcome(lambda: rc.diff(r, (v, 1)))
        if g != 'SKIP':
            assert g == outcome(lambda: diff(e, v)), (e, v)
    if not e.free_symbols:
        for prec in (53, 120):
            vr = rc.numeric_value(r, prec)
            if vr is not None:
                ve = finite(numeric_value(e, prec + 200))
                assert ve is not None and close(vr, ve, prec), (e, prec, vr, ve)


@pytest.mark.parametrize("seed", range(40))
def test_hyper(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("hyper", seed))
    variables = (Symbol('x'), Symbol('z'), Symbol('a'), Symbol('r', real=True))
    for _ in range(10):
        ap = parameters(rng, rng.randint(0, 3))
        bq = parameters(rng, rng.randint(0, 2))
        if ap and rng.random() < 0.3:
            bq.append(rng.choice(ap))  # a common parameter, cancelled
        args = [Tuple(*ap), Tuple(*bq), argument(rng)]
        got = outcome(lambda: rc.logic_call('hyper', args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: hy.hyper(*args)), args
        try:
            e = hy.hyper(*args)
        except Exception:
            continue
        r = rc.logic_call('hyper', args)
        check(rc, e, r, variables)
        assert outcome(lambda: rc.logic_call('hyper_radius', [r])) == outcome(lambda: e.radius_of_convergence), e
        assert outcome(lambda: rc.logic_call('hyper_eta', [r])) == outcome(lambda: e.eta), e
        assert str(rc.logic_call('hyper_convergence', [r])) == str(e.convergence_statement), e


@pytest.mark.parametrize("seed", range(40))
def test_meijerg(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("meijerg", seed))
    variables = (Symbol('x'), Symbol('z'), Symbol('a'), Symbol('b'))
    for _ in range(10):
        lists = [parameters(rng, rng.randint(0, 2)) for _ in range(4)]
        if rng.random() < 0.4 and lists[0]:
            lists[3].append(lists[0][0] - rng.randint(0, 2))  # pairs: derivatives in the parameters
        if rng.random() < 0.4 and lists[1]:
            lists[2].append(lists[1][0] + rng.randint(-2, 0))
        args = [Tuple(*p) for p in lists] + [argument(rng)]
        got = outcome(lambda: rc.logic_call('meijerg', args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: hy.meijerg(*args)), args
        try:
            e = hy.meijerg(*args)
        except Exception:
            continue
        r = rc.logic_call('meijerg', args)
        check(rc, e, r, variables)
        assert outcome(lambda: rc.logic_call('meijerg_nu', [r])) == outcome(lambda: e.nu), e
        assert outcome(lambda: rc.logic_call('meijerg_delta', [r])) == outcome(lambda: e.delta), e
        assert outcome(lambda: rc.logic_call('meijerg_period', [r])) == outcome(lambda: e.get_period()), e


@pytest.mark.parametrize("seed", range(20))
def test_appellf1(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("appellf1", seed))
    variables = (Symbol('x'), Symbol('z'), Symbol('a'))
    for _ in range(10):
        args = parameters(rng, 4) + [argument(rng), argument(rng)]
        got = outcome(lambda: rc.logic_call('appellf1', args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: hy.appellf1(*args)), args
        e = hy.appellf1(*args)
        r = rc.logic_call('appellf1', args)
        if e.free_symbols or not e.has(hy.AppellF1):
            check(rc, e, r, variables)


def test_values(rust: Any) -> None:
    """The values Rust has: entire functions, ``2F1`` everywhere (inside
    the disc by the series, outside by mpmath's transformations),
    terminating series."""
    from sympy_extras.core import I, Integer, Rational
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    t3 = ((Rational(1, 2), Rational(1, 3), Rational(1, 4)), (Rational(3, 2), 2))
    cases = [((), (), Integer(3)), ((Rational(1, 2),), (Rational(3, 2),), -Integer(4)), ((1, 2), (3,), Rational(1, 2)),
             ((Integer(-3), Rational(1, 3)), (Rational(5, 2),), Integer(7)), ((1 + I,), (2,), 2*I),
             ((), (Rational(1, 2), 2), -Integer(20)), ((Rational(1, 3), 1), (Rational(1, 2),), -Rational(2, 3)),
             # 2F1 beyond |z| <= 0.8 (mpmath's transformations): Gauss at 1, 1/z (the cut z > 1 too),
             # 1 - z, z/(z - 1), Gosper's recurrence on the unit circle
             ((1, 2), (5,), Integer(1)), ((Rational(1, 2), Rational(1, 3)), (Rational(5, 2),), -Integer(3)),
             ((1, 2), (3,), Integer(2)), ((1, 1), (2,), -Integer(2)), ((1, 2), (Rational(1, 2),), 5 + 2*I),
             ((Rational(1, 3), 1), (Rational(1, 2),), Rational(3, 5) + Rational(4, 5)*I),
             ((Rational(1, 2), Rational(1, 2)), (1,), Rational(9, 10)),
             ((Rational(1, 2), Rational(1, 3)), (2,), -Rational(9, 10)),
             # (q+1)Fq beyond the unit circle (mpmath's 1/z transformation; integer gaps of ap perturbed)
             (*t3, -Integer(2)),
             ((1, 2, Rational(1, 2)), (3, Rational(5, 2)), Integer(3)),
             ((Rational(1, 2), Rational(1, 3), 1), (2, Rational(7, 3)), 2 + I),
             ((Rational(1, 2), Rational(1, 3), Rational(1, 5), Rational(1, 7)), (2, 3, 4), -Integer(5)),
             ((1, 2, 3), (4, 5), -Integer(4)),
             # p <= q at a large argument (mpmath's asymptotic expansions of 0F1, 1F2, 2F2, 2F3)
             ((), (Rational(3, 2),), Integer(10)**4), ((), (Rational(1, 3),), -Integer(10)**4),
             ((Rational(1, 2), Rational(1, 3)), (Rational(5, 2), 2), Integer(50)),
             ((Rational(1, 2), Rational(1, 3)), (Rational(5, 2), 2), -Integer(50)),
             ((1, Rational(1, 3)), (Rational(3, 2), Rational(7, 4)), 20 + 30*I),
             ((Rational(1, 2),), (Rational(3, 2), Rational(5, 3)), -Integer(10)**6),
             ((Rational(1, 3),), (Rational(1, 2), 2), Integer(10)**6),
             ((Rational(1, 2), Rational(1, 3)), (Rational(3, 2), 2, Rational(5, 4)), -Integer(10)**6),
             ((Rational(1, 4), 1), (Rational(1, 2), Rational(3, 2), 2), Integer(10)**6),
             # (q+1)Fq near the unit circle (mpmath's _hypq1fq: the direct series up to hypsum's maxterms,
             # the 'r+s' acceleration of nsum, the alternate summation of 3F2 near 1)
             (*t3, Rational(9, 10)), (*t3, -Integer(1)),
             (*t3, Rational(3, 5) + 4*I/5), (*t3, Rational(51, 50)),
             (*t3, Rational(97, 100) + I/50),
             ((Rational(1, 2), Rational(1, 3), Rational(1, 4), Rational(1, 5)), (Rational(3, 2), 2, Rational(5, 2)),
              -Rational(21, 20)),
             ((1, 1, 1), (2, 2), -Integer(1)),
             # at 1 where the series converges (zeta(2))
             ((1, 1, 1), (2, 2), Integer(1))]
    for ap, bq, z in cases:
        args = [Tuple(*ap), Tuple(*bq), z]
        e, r = hy.hyper(*args), rc.logic_call('hyper', args)
        for prec in (53, 200):
            vr = rc.numeric_value(r, prec)
            assert vr is not None, (e, prec)
            assert close_in_modulus(vr, finite(numeric_value(e, prec + 200)), prec), (e, prec)


def test_meijerg_values(rust: Any) -> None:
    """The values of ``meijerg`` by Slater's theorem (mpmath's ``meijerg``):
    distinct ``b``, ``b`` differing by integers (the perturbation of the
    parameters), ``p = q`` on both sides of the unit circle, the second
    series for ``p > q``."""
    from sympy_extras.core import I, Integer, Rational
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    h = Rational(1, 2)
    cases = [((), (), (Integer(0),), (), Integer(2)),                              # exp(-z)
             ((), (), (h, Integer(0)), (), Rational(9, 4)),                         # distinct b
             ((), (-h, Integer(1)), (Integer(0), Integer(0), Integer(1)), (-h,), Rational(9, 16)),
             ((), (-h, Integer(1)), (Integer(0), Integer(0), Integer(0)), (-h,), Integer(3)),
             ((Integer(1),), (), (Integer(1),), (Integer(0),), Rational(1, 3)),     # p = q, |z| < 1
             ((Integer(1),), (), (Integer(1),), (Integer(0),), Integer(3)),         # p = q, |z| > 1
             ((Integer(1), Integer(1)), (), (Integer(1),), (), Integer(5)),  # p > q
             ((), (), (h, -h), (), 1 + I)]
    for an, ap, bm, bq, z in cases:
        args = [Tuple(*an), Tuple(*ap), Tuple(*bm), Tuple(*bq), z]
        e, r = hy.meijerg(*args), rc.logic_call('meijerg', args)
        for prec in (53, 120):
            vr = rc.numeric_value(r, prec)
            ve = finite(numeric_value(e, prec + 200))
            assert vr is not None and ve is not None and close(vr, ve, prec), (e, prec, vr, ve)


def test_appellf1_values(rust: Any) -> None:
    """The values of ``appellf1`` (mpmath's ``appellf1``): the double
    series through ``2F1`` with the smaller argument outside, the
    terminating cases, complex arguments, and the transformation to
    ``(x - y)/(x - 1)`` when both arguments are near or beyond the unit
    circle."""
    from sympy_extras.core import I, Integer, Rational
    from sympy_extras.core.evalf import numeric_value
    from sympy_extras.functions.special import hyper as hy
    rc = load("sympy_extras_rs.core")
    h = Rational(1, 2)
    cases = [(Integer(1), Integer(2), Integer(3), Integer(5), Rational(1, 4), h),
             (h, Rational(1, 3), Rational(1, 5), Integer(2), -Rational(1, 3), h),
             (Integer(1), Integer(1), Integer(1), Integer(2), h, Rational(1, 3)),
             (Integer(-3), h, Rational(1, 3), Integer(2), Integer(2), Integer(3)),
             (h, Integer(-2), Rational(1, 3), Rational(3, 2), Rational(1, 5), Integer(4)),
             (h, Rational(1, 3), Rational(1, 4), Rational(3, 2), h + I/4, -Rational(1, 3) + I/5),
             (h, Rational(1, 3), Rational(1, 4), Rational(3, 2), Integer(-1), -Rational(6, 5)),
             (Rational(1, 3), Integer(1), h, Integer(2), Rational(1, 2), Integer(-3))]
    for args in cases:
        e, r = hy.appellf1(*args), rc.logic_call('appellf1', list(args))
        for prec in (53, 120):
            vr = rc.numeric_value(r, prec)
            ve = finite(numeric_value(e, prec + 200))
            assert vr is not None and ve is not None and close(vr, ve, prec), (e, prec, vr, ve)
