"""``solve_linear_unique``, ``linear_root``, ``solve_one`` and
``numeric_linear_solution`` of the Rust port against
``sympy_extras.solvers.linsolve``: the docstring examples and random
linear systems with rational function coefficients in other symbols
(unique, underdetermined, inconsistent, nonlinear, with coefficients
outside the field)."""

from __future__ import annotations

import random
from typing import Any, Optional

import pytest

from ..conftest import load, stable_seed


def same(got: Any, expected: Any) -> bool:
    if expected is None or got is None:
        return got is None and expected is None
    if isinstance(expected, dict):
        return {repr(k): repr(v) for k, v in got.items()} == {repr(k): repr(v) for k, v in expected.items()}
    if isinstance(expected, list):
        return [repr(g) for g in got] == [repr(x) for x in expected]
    return repr(got) == repr(expected)


def test_docstring_examples(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.solvers import linsolve as l
    rc = load("sympy_extras_rs.core")
    a, b, n, i, k, m = (Symbol(c) for c in 'abnikm')
    cases: list[tuple[str, tuple[Any, ...]]] = [
        ("solve_linear_unique", ([a + b - 1, a - b - n], [a, b])),
        ("solve_linear_unique", ([a + b - 1, 2*a + 2*b - 2], [a, b])),
        ("solve_linear_unique", ([a*n - 1, b - a], [a, b])),
        ("linear_root", (i - 2*k, k)), ("linear_root", (2*k - i - 1, k)), ("linear_root", (k**2 - i, k)),
        ("solve_one", (2*k + 1, m)), ("solve_one", (i - 2*m, m)), ("solve_one", (m**2 - i, m)),
        ("numeric_linear_solution", ([1 - i, 1 - n],)), ("numeric_linear_solution", ([i + n - 1],)),
    ]
    for name, args in cases:
        assert same(getattr(rc, name)(*args), getattr(l, name)(*args)), (name, args)


def random_system(rng: random.Random) -> tuple[list[Any], list[Any]]:
    from sympy_extras.core import Integer, Rational, Symbol, sqrt
    from sympy_extras.core.add import add
    from sympy_extras.core.mul import mul
    u, v, w, n, t = (Symbol(c) for c in 'uvwnt')
    unknowns = [u, v, w][:rng.randint(1, 3)]
    coefficients = [Integer(rng.randint(-3, 3)), Rational(rng.randint(-3, 3), rng.randint(1, 3)), n, 1 / (n + 1),
                    n * t, Integer(0), sqrt(Integer(2)) if rng.random() < 0.1 else Integer(1)]
    equations = []
    for _ in range(rng.randint(len(unknowns) - 1, len(unknowns) + 1)):
        terms = [mul(rng.choice(coefficients), x) for x in unknowns]
        terms.append(rng.choice(coefficients + [n**2, t - 1]))
        if rng.random() < 0.08:
            terms.append(unknowns[0]**2)
        equations.append(add(*terms))
    return equations, unknowns


@pytest.mark.parametrize("seed", range(40))
def test_random_systems(rust: Any, seed: int) -> None:
    from sympy_extras.solvers import linsolve as l
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("linsolve", seed))
    for _ in range(10):
        equations, unknowns = random_system(rng)
        expected: Optional[list[Any]] = l.solve_linear_unique(equations, unknowns)
        assert same(rc.solve_linear_unique(equations, unknowns), expected), (equations, unknowns)
        assert same(rc.numeric_linear_solution(equations), l.numeric_linear_solution(equations)), equations
        for e in equations:
            for x in unknowns:
                assert same(rc.solve_one(e, x), l.solve_one(e, x)), (e, x)
