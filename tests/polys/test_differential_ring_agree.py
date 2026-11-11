"""``DifferentialRing`` of the Rust port (``terms``, ``integral``,
``leader``, ``from_terms``) against
``sympy_extras.polys.differential.ring``: the docstring examples, the
errors, and random differential polynomials with coefficients rational
in the variables and a parameter, under the orderly and an elimination
ranking."""

from __future__ import annotations

import random
from typing import Any, Optional

import pytest

from ..conftest import load, stable_seed


def python_terms(R: Any, eq: Any) -> list[tuple[Any, str]]:
    return [(m, repr(R.coefficient_expr(c))) for m, c in R.terms(eq).items()]


def python_integral(R: Any, eq: Any) -> list[tuple[Any, str]]:
    return [(m, repr(R.integer_expr(c))) for m, c in R.integral(R.terms(eq)).items()]


def rust_pairs(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, str]]:
    return [(m, repr(c)) for m, c in pairs]


def setting() -> tuple[Any, ...]:
    from sympy_extras.core import Function, Symbol
    x, y, p = Symbol('x'), Symbol('y'), Symbol('p')
    u, v = Function('u')(x, y), Function('v')(x)
    return x, y, p, u, v


def check(rc: Any, functions: list[Any], eq: Any, ranking: Optional[list[list[Any]]] = None,
          parameters: tuple[Any, ...] = ()) -> None:
    from sympy_extras.polys.differential.ring import DifferentialRing
    R = DifferentialRing(functions, ranking=ranking, parameters=parameters)
    kwargs = {"ranking": ranking, "parameters": parameters}
    try:
        expected = python_terms(R, eq)
    except ValueError as error:
        with pytest.raises(ValueError) as raised:
            rc.differential_terms(functions, eq, **kwargs)
        assert str(raised.value) == str(error), (eq, functions)
        return
    assert rust_pairs(rc.differential_terms(functions, eq, **kwargs)) == expected, (eq, functions, ranking)
    assert rust_pairs(rc.differential_integral(functions, eq, **kwargs)) == python_integral(R, eq), eq
    if any(m for m, _ in expected):
        leader, back = rc.differential_leader(functions, eq, **kwargs)
        assert repr(leader) == repr(R.leader(eq)), eq
        assert repr(back) == repr(R.from_terms(R.terms(eq))), eq


def test_examples_and_errors(rust: Any) -> None:
    from sympy_extras.core import Eq, Function, Symbol, sqrt
    rc = load("sympy_extras_rs.core")
    x, y, p, u, v = setting()
    w = Function('w')(x, y)
    z = Symbol('z')
    check(rc, [u, w], u.diff(x) + w.diff(y, 2))
    check(rc, [u, w], u.diff(x) + w.diff(y, 2), ranking=[[u], [w]])
    check(rc, [u], u.diff(x, y) + u.diff(y, 2) * u)
    check(rc, [u], Eq(u.diff(x)**2, x * u))
    for bad in [u.diff(x) * z, sqrt(x) * u, 1 / u, Function('q')(x), u.diff(z), x / (x - x + 1) + 2]:
        check(rc, [u], bad)


@pytest.mark.parametrize("seed", range(30))
def test_random_polynomials(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    rc = load("sympy_extras_rs.core")
    x, y, p, u, v = setting()
    rng = random.Random(stable_seed("differential ring", seed))
    derivatives = [u, v, u.diff(x), u.diff(y), u.diff(x, y), u.diff(y, 2), v.diff(x), v.diff(x, 3)]
    coefficients = [Integer(rng.randint(-4, 4)), Rational(rng.randint(-3, 3), rng.randint(1, 4)), x, y, p,
                    1 / (x + 1), x * y - p, 1 / (p * y)]
    for _ in range(8):
        terms = []
        for _ in range(rng.randint(1, 4)):
            factors = [rng.choice(coefficients)]
            for _ in range(rng.randint(0, 2)):
                factors.append(pow(rng.choice(derivatives), Integer(rng.randint(1, 2))))
            terms.append(mul(*factors))
        eq = add(*terms)
        ranking = rng.choice([None, [[u], [v]], [[v], [u]]])
        check(rc, [u, v], eq, ranking=ranking, parameters=(p,))
