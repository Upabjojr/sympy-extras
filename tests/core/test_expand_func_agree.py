"""``expand(func=True)`` of the Rust core against
``sympy_extras.core.function.expand``: the gamma, polygamma, loggamma,
beta, binomial, harmonic, zeta, polylog, lerchphi, Bessel, spherical
Bessel, spherical harmonic, Airy and hypergeometric rules (with other hints),
compared by structure and printing; the rules which are not ported in
Rust (an error) are skipped."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_arithmetic_agree import shape

HINTS: list[dict[str, bool]] = [{}, {"deep": False}, {"mul": False, "multinomial": False, "log": False,
                                                       "power_exp": False, "power_base": False}, {"trig": True}]


def node(rng: random.Random) -> Any:
    from sympy_extras.core import GoldenRatio, I, Integer, Rational, Symbol, TribonacciConstant, pi
    from sympy_extras.functions import (Ynm, erf2, erfc, erfi, expint, airyai, airyaiprime, airybi, airybiprime, beta, exp, besseli, besselj, besselk, bessely, binomial, digamma, gamma, hankel1,
                                        harmonic, hn1, hn2, hyper, jn, lerchphi, loggamma, polygamma, polylog,
                                        trigamma, yn, zeta)
    x, y = Symbol('x'), Symbol('y')
    n = Symbol('n', integer=True)
    s = Symbol('s', real=True)
    shift = lambda: rng.choice([Integer(0), Integer(1), Integer(2), Integer(-1), Integer(-3), Rational(1, 2),
                                Rational(5, 2), Rational(-7, 3)])
    var = lambda: rng.choice([x, y, 2*x, n])
    builders = [lambda: gamma(var() + shift()), lambda: gamma(shift()),
                lambda: polygamma(rng.choice([0, 1, 2, -1, s, Rational(1, 2)]), rng.choice([var() + shift(), 3*x, shift()])),
                lambda: loggamma(rng.choice([Rational(1, 3), Rational(2, 5), x, Integer(3), Rational(7, 3), Rational(-5, 2)])),
                lambda: harmonic(rng.choice([Rational(1, 3), Rational(7, 3), Rational(5, 4)])),
                lambda: digamma(var() + rng.randint(-2, 2)), lambda: trigamma(rng.choice([x + 2, 2*x, Rational(1, 2)])),
                lambda: beta(var(), var() + 1), lambda: binomial(var() + rng.randint(0, 3), rng.choice([n, 2, x, n - 1])),
                lambda: binomial(rng.choice([Integer(5), Rational(1, 2)]), x),
                lambda: harmonic(var() + rng.randint(-2, 3)), lambda: harmonic(x + 2, s),
                lambda: zeta(s, rng.choice([Integer(3), Integer(0), Integer(-2), x])), lambda: zeta(x, n),
                lambda: polylog(rng.choice([1, 0, -1, -2, 2, s]), x),
                lambda: lerchphi(rng.choice([Integer(1), x, Integer(-1), I, -I, exp(pi*I/3)]), rng.choice([s, Integer(-1), Integer(0), Integer(-2)]),
                                 rng.choice([x, y + 1])),
                lambda: rng.choice([besselj, bessely, besseli, besselk, hankel1])(rng.choice([Rational(5, 2), 3, -3, Rational(-3, 2), s, 1]), x),
                lambda: rng.choice([jn, yn, hn1, hn2])(rng.choice([0, 1, 2, -1, -2, s]), x),
                lambda: Ynm(rng.choice([0, 1, 2]), rng.choice([0, 1, -1]), y, x),
                lambda: hyper([x, y], [s], 1), lambda: GoldenRatio, lambda: TribonacciConstant,
                lambda: rng.choice([erfc, erfi])(rng.choice([x, 2*x + 1, I*x, Rational(1, 2)])), lambda: erf2(x, y + 1),
                lambda: expint(rng.choice([Integer(1), Integer(2), Integer(3), s, Rational(1, 2)]), rng.choice([x, 2*x])),
                lambda: rng.choice([airyai, airybi, airyaiprime, airybiprime])(
                    rng.choice([(2*x**3)**Rational(1, 3), 3*(x**3)**Rational(2, 3), (x**6)**Rational(1, 3), x, (x**2)**Rational(1, 2),
                                (y*x**3)**Rational(1, 3), 2*x + 1]))]
    while True:
        try:
            return rng.choice(builders)()
        except (ValueError, TypeError, ZeroDivisionError):
            continue


def _undummied(e: Any) -> Any:
    from sympy_extras.core.symbol import Dummy, Symbol
    return e.xreplace({d: Symbol(d.name) for d in e.atoms(Dummy)})


@pytest.mark.parametrize("seed", range(40))
def test_expand_func(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.function import expand
    from sympy_extras.core.symbol import Dummy
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("expand func", seed))
    x = Symbol('x')
    compared = 0
    for _ in range(5):
        e = node(rng)
        if rng.random() < 0.3:
            e = e * node(rng) + x
        for hints in HINTS:
            try:
                expected = expand(e, func=True, **hints)
            except (ZeroDivisionError, ValueError, ArithmeticError, TypeError, NotImplementedError):
                continue
            try:
                got = rc.expand(e, func=True, **hints)
            except NotImplementedError:
                continue
            if expected.atoms(Dummy):
                # fresh dummies (the sums of loggamma and harmonic at rationals) are counted apart
                # on the two sides, and their counters order the terms: compared as named symbols
                got, expected = _undummied(got), _undummied(expected)
                assert shape(got) == shape(expected) and str(got) == str(expected), (e, hints)
                compared += 1
                continue
            assert shape(got) == shape(expected) and rc.sstr(got) == str(expected), (e, hints)
            compared += 1
    assert compared >= 8
