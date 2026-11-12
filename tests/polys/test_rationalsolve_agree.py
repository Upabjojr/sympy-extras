"""``solveset_rational`` of the Rust port against
``sympy_extras.polys.rationalsolve``: random equations rational in the
unknown (products of polynomial factors, quotients with denominators
whose roots are excluded, symbolic coefficients, powers), and equations
outside the class (functions of the unknown, radicals), compared through
``repr`` and the class of the error."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_polytools_functions_agree import compare, random_polynomial


@pytest.mark.parametrize("seed", range(40))
def test_solveset_rational(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.polys.rationalsolve import solveset_rational
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("rationalsolve", seed))
    x, a = Symbol('x'), Symbol('a')
    compared = 0
    for _ in range(4):
        variables = rng.choice([[x], [x], [x, a]])
        e: Any = random_polynomial(rng, variables)
        kind = rng.random()
        if kind < 0.3:
            e = mul(e, random_polynomial(rng, variables))
        elif kind < 0.6:
            e = mul(e, pow(random_polynomial(rng, variables), Integer(-1)))
        elif kind < 0.75:
            e = add(e, mul(random_polynomial(rng, [x]), pow(random_polynomial(rng, [x]), Integer(-1))))
        elif kind < 0.85:
            e = pow(e, Integer(rng.choice([2, 3])))
        compared += compare(rc, "solveset_rational", lambda: solveset_rational(e, x), [e], [x])
    assert compared >= 2


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import I, Integer, Symbol, oo, sqrt, zoo
    from sympy_extras.functions import exp, sin
    from sympy_extras.polys.rationalsolve import solveset_rational
    rc = load("sympy_extras_rs.core")
    x, d, y = Symbol('x'), Symbol('d'), Symbol('y')
    cases = [x - d**2, x - 1/d, d**2 + 1, Integer(0), Integer(3), d, 1/d, (d - 1)/(d - 1), (d**2 - 1)/(d - 1),
             sin(d), exp(d) - 1, sqrt(d) - 2, d**3 - 2, d**4 + d + 1, y*d**3 + 1, oo*d, zoo*(d - 1),
             d**5 - d - 1, (d - I)**2, 1/d + 1/(d - 1), d**2 - y, (d**2 - 2)*(d**3 - 1), d**6 - 1]
    for e in cases:
        compare(rc, "solveset_rational", lambda: solveset_rational(e, d), [e], [d])
