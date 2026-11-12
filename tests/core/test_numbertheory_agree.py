"""The number-theoretic functions of the Rust core against
``sympy_extras.functions.combinatorial.numbers``: ``totient``,
``reduced_totient``, ``divisor_sigma``, ``udivisor_sigma``, ``primenu``,
``primeomega``, ``mobius``, ``primepi``, the Legendre, Jacobi and
Kronecker symbols, ``tribonacci`` (and its polynomials), ``motzkin`` and
``andre`` of random integers, rationals, products of powers, infinities
and symbols with assumptions, compared by structure (or the error) and
facts."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_integers_agree import outcome


def arguments() -> list[Any]:
    from sympy_extras.core import Integer, Rational, Symbol, oo, pi
    n, k = Symbol('n', integer=True, positive=True), Symbol('k', integer=True)
    p = Symbol('p', prime=True)
    return [Integer(v) for v in (1, 2, 3, 4, 7, 12, 30, 36, 97, 360, 105, -5, -8, 0)] + \
        [Rational(1, 2), Rational(17, 2), pi, oo, -oo, n, k, p, Symbol('x'), Integer(3)**n, n**2,
         Integer(2)*Integer(3)**k, Symbol('q', positive=True)]


ONE = ['totient', 'reduced_totient', 'primenu', 'primeomega', 'mobius', 'primepi', 'motzkin', 'andre']
TWO = ['legendre_symbol', 'jacobi_symbol', 'kronecker_symbol']
OPTIONAL = ['divisor_sigma', 'udivisor_sigma', 'tribonacci']


@pytest.mark.parametrize("seed", range(40))
def test_number_theory(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.assumptions import facts_of
    from sympy_extras.functions.combinatorial import numbers as nb
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("numbertheory", seed))
    for _ in range(16):
        name = rng.choice(ONE + TWO + OPTIONAL)
        args = [rng.choice(arguments())]
        if name in TWO:
            args.append(rng.choice(arguments()))
        elif name in OPTIONAL and rng.random() < 0.5:
            args.append(rng.choice([Integer(0), Integer(1), Integer(2), Integer(-1), Rational_half(),
                                    Symbol('m', integer=True, positive=True), Symbol('x')]))
        got = outcome(lambda: rc.logic_call(name, args))
        if got == 'SKIP':
            continue
        assert got == outcome(lambda: getattr(nb, name)(*args)), (name, args)
        try:
            e = getattr(nb, name)(*args)
        except Exception:
            continue
        assert rc.facts(rc.logic_call(name, args)) == dict(facts_of(e)), e


def Rational_half() -> Any:
    from sympy_extras.core import Rational
    return Rational(1, 2)
