"""Basic.simplify of the relations and formulas is native (the port of
SymPy's simplify of Booleans); checked against SymPy's simplify."""
from __future__ import annotations

import random

import sympy

from sympy_extras.interop import from_sympy, to_sympy


def test_simplify_of_relations_matches_sympy() -> None:
    x, y = sympy.symbols('x y')
    r = sympy.Symbol('r', real=True)
    rng = random.Random(5)
    sides = [x, y, r, 2 * x, x + y, x - 3 * y + 1, sympy.Integer(2), x**2, 2 * x + 4, sympy.Rational(1, 2) * y]
    relations = [sympy.Eq, sympy.Ne, sympy.Gt, sympy.Lt, sympy.Ge, sympy.Le]
    for _ in range(60):
        relation = rng.choice(relations)(rng.choice(sides), rng.choice(sides), evaluate=False)
        e: sympy.Basic = relation
        if rng.random() < 0.3:
            e = sympy.And(relation, rng.choice(relations)(rng.choice(sides), rng.choice(sides)))
        want = sympy.simplify(e)
        got = to_sympy(from_sympy(e).simplify())
        assert got == want, (e, got, want)
