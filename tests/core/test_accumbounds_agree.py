"""Accumulation bounds of the Rust core against
``sympy_extras.core.accumbounds``: random bounds (numbers, infinities,
constants, symbols with signs) and operands, compared by structure for
the evaluating functions of the core (sums and products with the bounds
as coefficients, powers, the operators ``/`` and ``-``), the functions
which give or take bounds (``sin``, ``cos``, ``tan``, ``cot`` of
infinities and bounds, ``atan``, ``atanh``, ``asech`` of ``zoo``,
``frac`` of the infinities, ``exp`` and ``log`` of bounds, and the
other functions of bounds), ``abs``, membership, union and intersection,
with the facts."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def ends() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, oo, pi, sqrt
    p = Symbol('p', positive=True)
    return [Integer(-2), Integer(-1), Integer(0), Integer(1), Integer(3), Rational(1, 2), sqrt(2), pi, -pi/2,
            -oo, oo, p, p + 1]


def operands() -> list[Any]:
    from sympy_extras.core import I, Integer, Rational, Symbol, oo, pi, sqrt, zoo
    x, p, n = Symbol('x', real=True), Symbol('p', positive=True), Symbol('n', negative=True)
    return [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-3), Rational(1, 2), Rational(-1, 3), sqrt(2),
            pi, oo, -oo, zoo, I, x, p, n, x + 1, Symbol('z')]


def random_bounds(rng: random.Random) -> Any:
    from sympy_extras.core.accumbounds import accum_bounds
    while True:
        a, b = rng.choice(ends()), rng.choice(ends())
        try:
            return accum_bounds(a, b)
        except ValueError:
            continue


@pytest.mark.parametrize("seed", range(40))
def test_accumbounds(rust: Any, seed: int) -> None:
    from sympy_extras.core import add, mul, pow
    from sympy_extras.core.accumbounds import AccumBounds, ab_abs, accum_bounds
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.functions.elementary.trigonometric import cos, cot, sin, tan
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("accumbounds", seed))
    for _ in range(8):
        a, b = rng.choice(ends()), rng.choice(ends())
        assert outcome(lambda: rc.logic_call('accum_bounds', [a, b])) == outcome(lambda: accum_bounds(a, b)), (a, b)
        s = random_bounds(rng)
        t = rng.choice([random_bounds(rng), rng.choice(operands())])
        u = rng.choice(operands())
        cases: list[tuple[Any, Any]] = [
            (lambda: rc.add(s, t), lambda: add(s, t)),
            (lambda: rc.add(s, t, u), lambda: add(s, t, u)),
            (lambda: rc.mul(s, t), lambda: mul(s, t)),
            (lambda: rc.mul(u, s), lambda: mul(u, s)),
            (lambda: rc.mul(s, t, u), lambda: mul(s, t, u)),
            (lambda: rc.sub(s, t), lambda: s - t),
            (lambda: rc.sub(u, s), lambda: u - s),
            (lambda: rc.div(s, t), lambda: s / t),
            (lambda: rc.div(u, s), lambda: u / s),
            (lambda: rc.neg(s), lambda: -s),
            (lambda: rc.pow(s, t), lambda: pow(s, t)),
            (lambda: rc.pow(u, s), lambda: pow(u, s)),
            (lambda: rc.logic_call('ab_abs', [s]), lambda: ab_abs(s) if isinstance(s, AccumBounds) else abs(s)),
        ]
        from sympy_extras.functions.elementary import complexes, exponential, hyperbolic, integers
        from sympy_extras.functions.elementary.trigonometric import atan
        more: list[tuple[str, Any]] = [('exp', exponential.exp), ('log', exponential.log),
                                       ('sinh', hyperbolic.sinh), ('cosh', hyperbolic.cosh), ('atan', atan),
                                       ('abs_', complexes.abs_), ('re', complexes.re), ('im', complexes.im),
                                       ('sign', complexes.sign), ('floor', integers.floor)]
        for name, f in [('sin', sin), ('cos', cos), ('tan', tan), ('cot', cot)] + more:
            cases.append((lambda name=name: getattr(rc, name)(s), lambda f=f: f(s)))
        for got, expected in cases:
            g = outcome(got)
            if g != 'SKIP':
                assert g == outcome(expected), (s, t, u)
        if isinstance(s, AccumBounds):
            assert rc.facts(s) == dict(facts_of(s)), s
            if isinstance(t, AccumBounds):
                assert outcome(lambda: rc.logic_call('ab_union', [s, t])) == outcome(lambda: s.union(t)), (s, t)
                assert outcome(lambda: rc.logic_call('ab_intersection', [s, t])) == \
                    outcome(lambda: s.intersection(t)), (s, t)
            assert outcome(lambda: rc.logic_call('ab_contains', [s, u])) == outcome(lambda: s.contains(u)), (s, u)


def test_functions_giving_bounds(rust: Any) -> None:
    from sympy_extras.core import Symbol, oo, zoo
    from sympy_extras.functions.elementary.integers import frac
    from sympy_extras.functions.elementary.hyperbolic import asech, atanh
    from sympy_extras.functions.elementary.trigonometric import atan, cos, cot, sin, tan
    rc = load("sympy_extras_rs.core")
    r = Symbol('r', extended_real=True, finite=False)
    table: list[tuple[str, Any, list[Any]]] = [('sin', sin, [oo, -oo]), ('cos', cos, [oo, -oo, r]), ('tan', tan, [oo, -oo]),
                          ('cot', cot, [oo, -oo]), ('atan', atan, [zoo]), ('atanh', atanh, [zoo]),
                          ('asech', asech, [zoo]), ('frac', frac, [oo, -oo, oo + 1])]
    for name, f, args in table:
        for a in args:
            assert outcome(lambda: getattr(rc, name)(a)) == outcome(lambda: f(a)), (name, a)
            assert repr(rc.roundtrip(f(a))) == repr(f(a))
