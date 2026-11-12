"""Cylindrical descriptions of the Rust core (``polys/cad/cylindrical.rs``)
against ``sympy_extras.polys.cad.cylindrical``: the docstring examples and
random formulas of ``test_cad_qe_agree`` in two variables, compared by
structure for ``cylindrical_formula``, ``cylindrical_set`` and
``cylindrical_cases`` (partial and full decompositions). The cases where
the Rust core raises ``NotImplementedError`` are skipped."""

from __future__ import annotations

import random
from typing import Any

import pytest

from ..conftest import load, stable_seed
from .test_cad_qe_agree import outcome, random_formula, tree


def test_examples(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.relational import eq
    from sympy_extras.polys.cad import cylindrical as cy
    rc = load("sympy_extras_rs.core")
    a, x, y, z = [Symbol(n) for n in 'axyz']
    formulas: list[tuple[Any, list[Any], list[tuple[str, Any]]]] = [
        (x**2 + y**2 < 1, [x, y], []),
        ((x**2 + y**2 <= 1) & (y >= x), [x, y], []),
        (eq(z**2, x) & (z > y), [x, y], [('exists', z)]),
        ((y**3 - 3*y + x > 0) & (x > 1), [x, y], []),
    ]
    for formula, gens, prefix in formulas:
        for partial in [True, False]:
            assert tree(rc.cylindrical_call('cylindrical_formula', formula, gens, prefix, [], None, partial)) == \
                tree(cy.cylindrical_formula(formula, gens, prefix, partial=partial)), formula
    sets: list[Any] = [(x**2 + y**2 <= 1) & (x + y >= 1), (x**2 + y**2)*((x - 3)**2 + y**2 - 1) <= 0,
                       x**2 + y**2 < 0]
    for formula in sets:
        assert tree(rc.cylindrical_call('cylindrical_set', formula, [x, y])) == \
            tree(cy.cylindrical_set(formula, [x, y])), formula
    for formula in [eq(x**2, a), x**2 <= a, eq(a*x, 1)]:
        assert tree(rc.cylindrical_call('cylindrical_cases', formula, [a], [], [x])) == \
            tree(cy.cylindrical_cases(formula, [a], [x])), formula


@pytest.mark.parametrize("seed", range(40))
def test_cylindrical(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.cad import cylindrical as cy
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad-cylindrical", seed))
    x, y = Symbol('x'), Symbol('y')
    formula = random_formula(rng, [x, y], 2)
    partial = rng.random() < 0.7
    gens = [x, y] if rng.random() < 0.7 else [y, x]
    calls: list[tuple[Any, Any]] = [
        (lambda: rc.cylindrical_call('cylindrical_formula', formula, gens, [], [], None, partial),
         lambda: cy.cylindrical_formula(formula, gens, partial=partial)),
        (lambda: rc.cylindrical_call('cylindrical_set', formula, gens, [], [], None, partial),
         lambda: cy.cylindrical_set(formula, gens, partial=partial)),
        (lambda: rc.cylindrical_call('cylindrical_cases', formula, gens[:1], [], gens[1:], None, partial),
         lambda: cy.cylindrical_cases(formula, gens[:1], gens[1:], partial=partial)),
    ]
    for got_call, expected_call in calls:
        got = outcome(got_call)
        # what the Rust core does not take yet: skipped
        if got != 'SKIP':
            assert got == outcome(expected_call), (formula, gens, partial)
