"""``gammasimp`` and ``combsimp`` against their Rust port: the docstring
examples, the reflection, duplication and multiplication formulas, and
random products and quotients of gamma functions, factorials, binomials
and rising and falling factorials with shifted arguments."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def fixed() -> list[Any]:
    from sympy_extras.core import Rational, Symbol, pi
    from sympy_extras.functions import binomial, factorial, ff, gamma, rf, sin
    x, y = Symbol('x'), Symbol('y')
    n, k = Symbol('n', integer=True), Symbol('k', integer=True)
    return [gamma(x)/gamma(x - 3), gamma(x)*gamma(1 - x), factorial(n)/factorial(n - 3),
            binomial(n + 1, k + 1)/binomial(n, k), gamma(x)*gamma(x + Rational(1, 2))/gamma(2*x),
            gamma(x)*gamma(x + Rational(1, 3))*gamma(x + Rational(2, 3)), gamma(Rational(4, 3)),
            gamma(Rational(-5, 2))*x, rf(x, 3)/ff(x + 2, 3), sin(gamma(x + 1)/gamma(x)) + y,
            x*gamma(x)/gamma(x + 2), gamma(x + 1)*gamma(1 - x)/pi, binomial(n, k)*factorial(k)*factorial(n - k),
            factorial(n + 1)/(factorial(k)*factorial(n - k + 1)), gamma(y)*gamma(x - y)/gamma(x)]


def expression(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions import binomial, factorial, ff, gamma, rf
    x = Symbol('x')
    n, k = Symbol('n', integer=True), Symbol('k', integer=True)
    shift = lambda: Integer(rng.randint(-2, 2)) + rng.choice([0, 0, Rational(1, 2)])
    var = lambda: rng.choice([x, n, k, 2*x])
    builders = [lambda: gamma(var() + shift()), lambda: factorial(rng.choice([n, k]) + rng.randint(-1, 2)),
                lambda: binomial(n + rng.randint(0, 2), k + rng.randint(0, 1)), lambda: rf(var(), rng.randint(1, 3)),
                lambda: ff(var(), rng.randint(1, 3)), lambda: var() + shift()]
    e = Integer(1)
    for _ in range(rng.randint(2, 4)):
        f = rng.choice(builders)()
        e = e*f if rng.random() < 0.6 else e/f
    return e


@pytest.mark.parametrize("seed", range(30))
def test_gammasimp(rust: Any, seed: int) -> None:
    from sympy_extras.simplify.gammasimp import combsimp, gammasimp
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("gammasimp", seed))
    cases = [fixed()[seed % len(fixed())]] + [expression(rng) for _ in range(3)]
    for e in cases:
        assert outcome(lambda: rc.logic_call('gammasimp', [e])) == outcome(lambda: gammasimp(e)), e
        assert outcome(lambda: rc.logic_call('combsimp', [e])) == outcome(lambda: combsimp(e)), e
