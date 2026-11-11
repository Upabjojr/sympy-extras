"""``DifferentialPolynomial`` of the Rust port against
``sympy_extras.polys.differential.polynomial``: the conversion and its
expression, leader, degree, initial, separant, derivatives, and Ritt's
pseudo-, partial and full remainders by small chains, on random
differential polynomials over two rankings."""

from __future__ import annotations

import random
from typing import Any, Optional

import pytest

from ..conftest import load, stable_seed


def python_ops(functions: list[Any], eq: Any, chain: list[Any], ranking: Optional[list[list[Any]]],
               parameters: tuple[Any, ...]) -> dict[str, Any]:
    from sympy_extras.polys.differential.polynomial import DifferentialPolynomial as P
    from sympy_extras.polys.differential.ring import DifferentialRing
    R = DifferentialRing(functions, ranking=ranking, parameters=parameters)
    p = P.from_expr(R, eq)
    members = [P.from_expr(R, c) for c in chain]
    result: dict[str, Any] = {"to_expr": p.to_expr()}
    if not p.is_ground:
        result.update(leader=p.leader_expr(), degree=p.degree(), initial=p.initial().to_expr(),
                      separant=p.separant().to_expr())
    for i in range(R.n):
        result["derivative_%d" % i] = p.derivative(i).to_expr()
    if members and not members[0].is_ground:
        result["pseudo_remainder"] = p.pseudo_remainder(members[0], members[0].leader()).to_expr()
    if members and all(not m.is_ground for m in members):
        result["partial_remainder"] = p.partial_remainder(members).to_expr()
        result["remainder"] = p.remainder(members).to_expr()
    return result


def same(got: dict[str, Any], expected: dict[str, Any]) -> bool:
    return {k: repr(v) for k, v in got.items()} == {k: repr(v) for k, v in expected.items()}


def test_docstring_example(rust: Any) -> None:
    from sympy_extras.core import Function, Symbol
    rc = load("sympy_extras_rs.core")
    x = Symbol('x')
    u = Function('u')(x)
    eq, chain = u.diff(x)**2 * 2 - 4 * u, [u.diff(x, 2) - u]
    assert same(rc.differential_polynomial_ops([u], eq, chain), python_ops([u], eq, chain, None, ()))


@pytest.mark.parametrize("seed", range(30))
def test_random_polynomials(rust: Any, seed: int) -> None:
    from sympy_extras.core import Function, Integer, Symbol
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    from sympy_extras.core.power import pow
    rc = load("sympy_extras_rs.core")
    x, y, p = Symbol('x'), Symbol('y'), Symbol('p')
    u, v = Function('u')(x, y), Function('v')(x)
    rng = random.Random(stable_seed("differential polynomial", seed))
    derivatives = [u, v, u.diff(x), u.diff(y), u.diff(x, y), u.diff(y, 2), v.diff(x), v.diff(x, 2)]
    coefficients = [Integer(rng.randint(-3, 3)) or Integer(1), x, y, p, x * y - p, x + 1]

    def polynomial(size: int) -> Any:
        terms = []
        for _ in range(size):
            factors = [rng.choice(coefficients)]
            for _ in range(rng.randint(0, 2)):
                factors.append(pow(rng.choice(derivatives), Integer(rng.randint(1, 2))))
            terms.append(mul(*factors))
        return add(*terms)

    for _ in range(4):
        eq = polynomial(rng.randint(1, 4))
        if eq == 0:
            continue
        ranking = rng.choice([None, [[u], [v]], [[v], [u]]])
        chain = [rng.choice([u.diff(x) - u * v, u.diff(y, 2) + x * u, u**2 - p * u.diff(x)]),
                 rng.choice([v.diff(x) - v**2, x * v.diff(x, 2) + v])]
        expected = python_ops([u, v], eq, chain, ranking, (p,))
        got = rc.differential_polynomial_ops([u, v], eq, chain, ranking=ranking, parameters=(p,))
        assert same(got, expected), (eq, chain, ranking)
