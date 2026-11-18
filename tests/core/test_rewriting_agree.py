"""``rewrite`` of the core against its Rust port, for the ported targets
(``gamma``, ``factorial``, ``binomial``, ``RisingFactorial``,
``FallingFactorial``): random expressions of the combinatorial functions
and the beta function, with and without ``piecewise`` and patterns."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.functions import beta, binomial, catalan, factorial, factorial2, ff, gamma, rf, sin, subfactorial
    x, n, k = Symbol('x'), Symbol('n', integer=True), Symbol('k', integer=True, nonnegative=True)
    if depth == 0:
        return rng.choice([x, n, k, Integer(3), Rational(1, 2), x + 1, n - k])
    a, b = expression(rng, depth - 1), expression(rng, depth - 1)
    builders = [lambda: factorial(a), lambda: binomial(a, b), lambda: rf(a, b), lambda: ff(a, b), lambda: gamma(a),
                lambda: catalan(a), lambda: factorial2(a), lambda: subfactorial(a), lambda: beta(a, b),
                lambda: a*b, lambda: a + b, lambda: a/b, lambda: sin(a)]
    while True:
        try:
            return rng.choice(builders)()
        except (ValueError, TypeError):  # factorial2 of a non-integer
            continue


def targets() -> list[tuple[str, Any]]:
    from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial, FallingFactorial, RisingFactorial
    from sympy_extras.functions.special.gamma_functions import Gamma
    return [('gamma', Gamma), ('factorial', Factorial), ('binomial', Binomial), ('RisingFactorial', RisingFactorial),
            ('FallingFactorial', FallingFactorial)]


@pytest.mark.parametrize("seed", range(40))
def test_rewrite(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import binomial, factorial, ff, gamma, rf
    from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial, FallingFactorial, RisingFactorial
    from sympy_extras.functions.special.gamma_functions import Gamma
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("rewriting", seed))
    x = Symbol('x')
    samples = [(Factorial, factorial(x)), (Binomial, binomial(x, 2*x)), (RisingFactorial, rf(x, 2*x)),
               (FallingFactorial, ff(x, 2*x)), (Gamma, gamma(x))]
    for _ in range(4):
        e = expression(rng, rng.randint(1, 2))
        name, cls = rng.choice(targets())
        piecewise = rng.random() < 0.5
        chosen = rng.sample(samples, rng.randint(0, 2)) if rng.random() < 0.3 else []
        patterns = [c for c, _ in chosen]
        nodes = [s for _, s in chosen]
        got = outcome(lambda: rc.logic_call('rewrite', [e, Symbol(name), Integer(int(piecewise)), *nodes]))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: e.rewrite(*patterns, cls, piecewise=piecewise)), (e, name, piecewise, patterns)
