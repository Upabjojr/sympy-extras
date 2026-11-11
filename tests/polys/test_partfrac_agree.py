"""``apart`` of the Rust port against ``sympy_extras.polys.partfrac``:
random rational functions in ``x`` with repeated linear and quadratic
factors in the denominator, rational and parametric coefficients, with
and without the generator given, compared through ``str`` and the class
of the error."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_polytools_functions_agree import compare, random_polynomial


@pytest.mark.parametrize("seed", range(40))
def test_apart(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    from sympy_extras.polys.partfrac import apart
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("apart", seed))
    x, a = Symbol('x'), Symbol('a')
    compared = 0
    for _ in range(3):
        variables = rng.choice([[x], [x], [x], [x, a]])
        denominator: Any = Integer(1)
        for _ in range(rng.randint(1, 3)):
            denominator = mul(denominator, pow(random_polynomial(rng, [x]), Integer(rng.randint(1, 3))))
        e = mul(random_polynomial(rng, variables), pow(denominator, Integer(-1)))
        given = rng.choice([[], [x]])
        compared += compare(rc, "apart", lambda: apart(e, given[0] if given else None), [e], given)
    assert compared >= 2


def test_edge_cases(rust: Any) -> None:
    from sympy_extras.core import Integer, Symbol, sqrt
    from sympy_extras.polys.partfrac import apart
    rc = load("sympy_extras_rs.core")
    x, y = Symbol('x'), Symbol('y')
    cases: list[tuple[Any, list[Any]]] = [
        (Integer(3), []), (x, []), (1/(x**2 - 1), []), (1/(x*(x + 1)**2), [x]), ((x**3 + 1)/(x**2 + 1), []),
        (1/(x**2 + 1)**2, []), (y/(x*(x + y)), [x]), (1/(x*y + y), []), (1/(x**2 - 2), []),
        (sqrt(2)/(x - 1), []), ((x**2 - 1)/(x - 1), []), (1/(2*x + 4), []),
    ]
    for e, given in cases:
        compare(rc, "apart", lambda: apart(e, given[0] if given else None), [e], given)
    compare(rc, "apart_full", lambda: apart(1/(x**2 - 1), full=True), [1/(x**2 - 1)], [])
