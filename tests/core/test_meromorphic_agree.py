"""``is_meromorphic`` against its Rust port: random expressions of sums,
products, powers (integer, rational, symbolic exponents), elementary and
special functions (incomplete gamma, Bessel functions), undefined
functions and piecewise expressions, at finite points, zeros of the
arguments and infinity."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Function, Integer, Rational, Symbol, pi
    from sympy_extras.functions import (Abs, Piecewise, acos, asin, atan, besseli, besselj, besselk, bessely, cos, cosh,
                                        exp, gamma, hankel1, jn, log, lowergamma, re, sin, tan, uppergamma)
    x, y = Symbol('x'), Symbol('y')
    n = Symbol('n', integer=True)
    if depth == 0:
        return rng.choice([x, y, Integer(2), Integer(-1), Rational(1, 2), pi, x - 1, x**2 + 1])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    f = Function('f')
    builders = [lambda: a + b, lambda: a*b, lambda: a**2, lambda: a**-1, lambda: a**Rational(1, 2), lambda: a**b,
                lambda: exp(a), lambda: log(a), lambda: sin(a), lambda: cos(a)/b, lambda: tan(a), lambda: asin(a),
                lambda: acos(a), lambda: atan(a), lambda: Abs(a), lambda: re(a), lambda: gamma(a), lambda: cosh(a),
                lambda: lowergamma(rng.choice([n, Integer(2), Rational(1, 2), x]), a),
                lambda: uppergamma(rng.choice([n, Integer(-1), y]), a), lambda: besselj(rng.choice([0, 1, n, x]), a),
                lambda: bessely(rng.choice([0, 1, Rational(1, 2)]), a), lambda: besseli(rng.choice([0, 2]), a),
                lambda: besselk(0, a), lambda: hankel1(1, a), lambda: jn(0, a), lambda: f(a), lambda: f(a, b),
                lambda: Piecewise((a, x > 0), (b, True))]
    while True:
        try:
            return rng.choice(builders)()
        except (ValueError, TypeError):
            continue


@pytest.mark.parametrize("seed", range(40))
def test_is_meromorphic(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol, oo, zoo
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("meromorphic", seed))
    x = Symbol('x')
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.core.basic import preorder_traversal
    from sympy_extras.core.expr import Expr
    for _ in range(5):
        e = expression(rng, rng.randint(1, 3))
        # the facts of some numeric special values come from their numeric
        # value in Rust only (jn(0, pi)): the rules then see other facts
        if any(isinstance(t, Expr) and not t.free_symbols and t.args and rc.facts(t) != dict(facts_of(t)) for t in preorder_traversal(e)):
            continue
        for point in (Integer(0), Integer(1), Integer(-1), Integer(2), oo, zoo):
            expected = outcome(lambda: e.is_meromorphic(x, point))
            got = outcome(lambda: rc.logic_call('is_meromorphic', [e, x, point]))
            # the bridge answers 0 for None
            assert str(got) == ('0' if str(expected) == 'None' else str(expected)), (e, point)
