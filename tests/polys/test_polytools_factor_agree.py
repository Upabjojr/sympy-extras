"""``factor``, ``factor_list``, ``sqf``, ``sqf_list`` and their fraction
forms of the Rust port against ``sympy_extras.polys.polytools``, on
random products of powers of polynomials (integer, symbolic and
fractional exponents, number bases, rational functions), with and
without generators, compared through ``str`` and the class of the
error; the cases which the Python factors over ``ZZ_I``, ``RR`` or
through SymPy are ``NotImplementedError`` in the port and skipped."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_polytools_functions_agree import compare, random_polynomial


def random_product(rng: random.Random, variables: list[Any]) -> Any:
    from sympy_extras.core import Integer, Rational, Symbol, sqrt
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    n = Symbol('n')
    factors: list[Any] = [rng.choice([Integer(rng.randint(-6, 6)), Rational(rng.randint(-4, 4), rng.randint(1, 4))])]
    for _ in range(rng.randint(1, 3)):
        base = random_polynomial(rng, variables)
        exponent = rng.choice([Integer(1), Integer(1), Integer(2), Integer(3), Integer(-1), Integer(-2), n,
                               Rational(1, 2), 2*n])
        factors.append(pow(base, exponent))
    if rng.random() < 0.2:
        factors.append(pow(rng.choice([Integer(2), Integer(3)]), rng.choice([n, Rational(1, 2)])))
    if rng.random() < 0.1:
        factors.append(sqrt(2))
    return mul(*factors)


@pytest.mark.parametrize("seed", range(40))
def test_factor_functions(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys import polytools as pt
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polytools factor", seed))
    x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    compared = 0
    for _ in range(4):
        variables = rng.choice([[x], [x], [x, y], [x, a]])
        e = random_product(rng, variables)
        gens = rng.choice([[], [], variables])
        compared += compare(rc, "factor", lambda: pt.factor(e, *gens), [e], gens)
        compared += compare(rc, "factor_nofraction", lambda: pt.factor(e, *gens, fraction=False), [e], gens)
        compared += compare(rc, "factor_deep", lambda: pt.factor(e, *gens, deep=True), [e], gens)
        compared += compare(rc, "sqf", lambda: pt.sqf(e, *gens), [e], gens)
        compared += compare(rc, "factor_list", lambda: pt.factor_list(e, *gens), [e], gens)
        compared += compare(rc, "sqf_list", lambda: pt.sqf_list(e, *gens), [e], gens)
        compared += compare(rc, "factor_list_fraction", lambda: pt.factor_list_fraction(e, *gens), [e], gens)
        compared += compare(rc, "sqf_list_fraction", lambda: pt.sqf_list_fraction(e, *gens), [e], gens)
    assert compared >= 16


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Rational, Symbol, sqrt
    from sympy_extras.functions import exp, sin
    from sympy_extras.polys import polytools as pt
    rc = load("sympy_extras_rs.core")
    x, y, n = Symbol('x'), Symbol('y'), Symbol('n')
    p = Symbol('p', positive=True)
    cases = [Integer(0), Integer(12), Rational(-3, 4), x, -x, x**2 - 1, (x**2 - 1)**n, (x**2 - 1)**Rational(1, 2),
             2**n*(x + 1), (4*x**2 - 4)**n, exp(x)**2 - 1, sin(x)**2 + 2*sin(x) + 1, x**2*y - y,
             (x - y)/(y - x), 1/(x**2 - 1), (p*x - p)**n, (2 + I)*x, x + sqrt(2), (x**2 + 2*x + 1)/(x**2 - 1)]
    for e in cases:
        for gens in ([], [x]):
            for name, python in [("factor", lambda: pt.factor(e, *gens)), ("sqf", lambda: pt.sqf(e, *gens)),
                                 ("factor_list", lambda: pt.factor_list(e, *gens)),
                                 ("sqf_list", lambda: pt.sqf_list(e, *gens)),
                                 ("factor_deep", lambda: pt.factor(e, *gens, deep=True)),
                                 ("factor_list_fraction", lambda: pt.factor_list_fraction(e, *gens))]:
                compare(rc, name, python, [e], gens)
