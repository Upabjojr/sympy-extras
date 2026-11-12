"""Quantifier elimination of the Rust core (``polys/cad/qe.rs``) against
``sympy_extras.polys.cad.qe``: random Boolean combinations of polynomial
relations of low degree in two and three variables under random
quantifier prefixes, compared by structure for ``quantifier_elimination``
(partial and full decompositions), ``decide``, ``solution_set``,
``sample_points`` and ``truth_tables``. The last resort of
``quantifier_elimination`` (the root functions of ``cad.cylindrical``) is
not ported: those cases are skipped."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed


def tree(e: Any) -> Any:
    args = getattr(e, 'args', None)
    if isinstance(e, (list, tuple)):
        return [tree(a) for a in e]
    if isinstance(e, dict):
        return sorted((repr(k), tree(v)) for k, v in e.items())
    if not args:
        return repr(e)
    return (type(e).__name__, tuple(tree(a) for a in args))


def outcome(f: Callable[[], Any]) -> Any:
    try:
        return tree(f())
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def random_poly(rng: random.Random, variables: list[Any]) -> Any:
    """A sum of a constant and up to three monomials (of total degree at
    most 2 in three variables, at most 4 in two)."""
    from sympy_extras.core import Integer
    total: Any = Integer(rng.randint(-3, 3))
    top = 2 if len(variables) == 3 else 2
    for _ in range(rng.randint(1, 3)):
        monomial: Any = Integer(rng.choice([-2, -1, 1, 2]))
        chosen = rng.sample(variables, rng.randint(1, min(2, len(variables))))
        for v in chosen:
            monomial = monomial * v**rng.randint(1, top if len(variables) == 2 else 3 - len(chosen))
        total = total + monomial
    return total


def random_formula(rng: random.Random, variables: list[Any], depth: int) -> Any:
    from sympy_extras.core.logic import and_, implies, not_, or_
    from sympy_extras.core.relational import eq, ge, gt, le, lt, ne
    relations: list[Any] = [eq, ne, lt, le, gt, ge]
    if depth == 0 or rng.random() < 0.5:
        relation = rng.choice(relations)
        return relation(random_poly(rng, variables), 0)
    kind = rng.choice(['and', 'or', 'not', 'implies'])
    if kind == 'not':
        return not_(random_formula(rng, variables, depth - 1))
    a, b = random_formula(rng, variables, depth - 1), random_formula(rng, variables, depth - 1)
    connectives: dict[str, Any] = {'and': and_, 'or': or_, 'implies': implies}
    return connectives[kind](a, b)


@pytest.mark.parametrize("seed", range(30))
def test_qe(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.polys.cad import qe
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad-qe", seed))
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    variables = [x, y] if rng.random() < 0.6 else [x, y, z]
    formula = random_formula(rng, variables, 2 if len(variables) == 2 else 1)
    bound = rng.sample(variables, rng.randint(1, len(variables) - 1))
    quantifiers = [(rng.choice(['exists', 'forall']), v) for v in bound]
    partial = rng.random() < 0.7
    expected = outcome(lambda: qe.quantifier_elimination(formula, quantifiers, partial=partial))
    got = outcome(lambda: rc.qe_call('quantifier_elimination', formula, quantifiers, None, None, partial))
    if got != 'SKIP':
        assert got == expected, (formula, quantifiers, partial)
    every = [(rng.choice(['exists', 'forall']), v) for v in variables]
    assert outcome(lambda: rc.qe_call('decide', formula, every, None, None, partial)) == \
        outcome(lambda: qe.decide(formula, every, partial=partial)), (formula, every)
    free = [v for v in variables if v not in bound]
    if len(free) == 1:
        assert outcome(lambda: rc.qe_call('solution_set', formula, quantifiers, free, None, partial)) == \
            outcome(lambda: qe.solution_set(formula, free[0], quantifiers, partial=partial)), formula
    if len(variables) == 2:
        assert outcome(lambda: rc.qe_call('sample_points', formula, [], variables, None, partial)) == \
            outcome(lambda: qe.sample_points(formula, variables, partial=partial)), formula
        other = random_formula(rng, variables, 1)
        cells, tables = qe.truth_tables([formula, other], variables, partial=partial)
        assert rc.qe_call('truth_tables', [formula, other], [], variables, None, partial) == \
            ([repr(c) for c in cells], tables), (formula, other)


def test_examples(rust: Any) -> None:
    from sympy_extras.core import Symbol
    from sympy_extras.core.relational import eq
    from sympy_extras.polys.cad import qe
    rc = load("sympy_extras_rs.core")
    a, b, c, x, y, z = [Symbol(n) for n in 'abcxyz']
    cases: list[tuple[Any, list[tuple[str, Any]], Any]] = [
        (x**2 + b*x + c > 0, [('forall', x)], None),
        (eq(a*x**2 + b*x + c, 0), [('exists', x)], None),
        (eq(x**2 + y**2, 1), [('exists', y)], None),
        (x**2 + y**2 < 1, [('forall', x), ('exists', y)], None),
        (eq(y, x**2), [('forall', x), ('exists', y)], None),
        (eq(z**2, x) & (z > y), [('exists', z)], [x, y]),
        # the augmented projection (sympy-extras#9)
        (eq(z**2, x) & (z > y), [('exists', z)], [y, x]),
        (eq(z**4, x) & (z > y), [('exists', z)], [x, y]),
        (eq(z**3 - 3*z, x) & (z > y), [('exists', z)], [x, y]),
        ((x**2 + y**2 + z**2 < 1) & (z > x + y), [('exists', z)], [x, y]),
    ]
    for formula, quantifiers, free in cases:
        for method, partial in [(None, True), (None, False), ('hong', True)]:
            got = rc.qe_call('quantifier_elimination', formula, quantifiers, free, method, partial)
            expected = qe.quantifier_elimination(formula, quantifiers, free, method, partial)
            assert tree(got) == tree(expected), (formula, method, partial)


@pytest.mark.parametrize("seed", range(12))
def test_structured(rust: Any, seed: int) -> None:
    # an equation of degree two in the bound variable with a bound on it:
    # the kind whose solution set the projection factors do not describe
    # (the generator of test_solution_formulas_at_random_points)
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.core.logic import and_
    from sympy_extras.core.relational import eq, ge, gt, le, lt
    from sympy_extras.polys.cad import qe
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad-qe-structured", seed))
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

    def polynomial(degree: int, terms: int) -> Any:
        monomials: list[Any] = [Integer(1), x, y]
        if degree >= 2:
            monomials += [x*x, x*y, y*y]
        total: Any = Integer(0)
        for m in rng.sample(monomials, rng.randint(1, min(terms, len(monomials)))):
            total = total + rng.choice([-3, -2, -1, 1, 2, 3])*m
        return total

    p: Any = z**2 + polynomial(2, 2)
    if rng.random() < 0.5:
        p = p + polynomial(1, 2)*z
    bound = z - polynomial(rng.choice([1, 2]), 2)
    relation: Any = rng.choice([lt, gt, le, ge])
    formula = and_(eq(p, 0), relation(bound, 0))
    prefix = [(rng.choice(['exists', 'forall']), z)]
    got = outcome(lambda: rc.qe_call('quantifier_elimination', formula, prefix, [x, y]))
    if got != 'SKIP':
        assert got == outcome(lambda: qe.quantifier_elimination(formula, prefix, [x, y])), (formula, prefix)
