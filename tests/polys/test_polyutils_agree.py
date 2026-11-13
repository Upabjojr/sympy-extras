"""``dict_from_expr`` and ``expr_from_dict`` of the Rust port against
``sympy_extras.polys.polyutils``, over ``ZZ``, ``QQ``, ``ZZ_I``, ``QQ_I``
and ``GF(7)``: random polynomial expressions (sums, products and
nonnegative integer powers of the generators and of coefficients, built
by the evaluating functions of the core) and expressions which are not
polynomials (the same ``ValueError``)."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed


def domains() -> list[tuple[Any, Any]]:
    from sympy_extras.polys.domains import QQ, ZZ
    from sympy_extras.polys.domains.finitefield import FiniteField
    from sympy_extras.polys.domains.gaussiandomains import QQ_I, ZZ_I
    return [(ZZ, "ZZ"), (QQ, "QQ"), (ZZ_I, "ZZ_I"), (QQ_I, "QQ_I"), (FiniteField(7), 7)]


def random_expression(rng: random.Random, depth: int) -> Any:
    from sympy_extras.core import I, Integer, Rational, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    leaves = [x, y, z, Integer(rng.randint(-5, 5)), Rational(rng.randint(-5, 5), rng.randint(1, 4)), I,
              Symbol('t'), pow(x, Integer(-1)), pow(Integer(2), Rational(1, 2))]
    if depth == 0:
        return rng.choice(leaves)
    kind = rng.choice(["add", "mul", "pow", "leaf"])
    if kind == "add":
        return add(*[random_expression(rng, depth - 1) for _ in range(rng.randint(2, 3))])
    if kind == "mul":
        return mul(*[random_expression(rng, depth - 1) for _ in range(rng.randint(2, 3))])
    if kind == "pow":
        return pow(random_expression(rng, depth - 1), Integer(rng.randint(0, 3)))
    return rng.choice(leaves)


@pytest.mark.parametrize("seed", range(30))
def test_dict_from_expr_agrees(rust: Any, seed: int) -> None:
    from sympy_extras.polys.polyutils import dict_from_expr, expr_from_dict
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("polyutils", seed))
    compared = 0
    for _ in range(20):
        e = random_expression(rng, 3)
        gens = rng.choice([['x'], ['x', 'y'], ['x', 'y', 'z'], ['y', 'x']])
        for K, name in domains():
            try:
                expected = dict_from_expr(e, gens, K)
            except ValueError:
                with pytest.raises(ValueError):
                    rc.dict_from_expr(e, gens, name)
                continue
            got = rc.dict_from_expr(e, gens, name)
            assert got == expected, (e, gens, name)
            back = rc.expr_from_dict(got, gens, name)
            assert repr(back) == repr(expr_from_dict(expected, gens, K)), (e, gens, name)
            compared += 1
    assert compared >= 10
