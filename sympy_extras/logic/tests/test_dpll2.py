"""The conjunctive normal forms of sympy_extras.logic.cnf and the SAT
solver of sympy_extras.logic.dpll2, checked against the truth tables of
the formulas (an independent oracle) and against hand computations."""
from __future__ import annotations

import random
from itertools import product
from typing import Optional

from sympy_extras.core import Q, Symbol
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanFalse, BooleanTrue, Equivalent, Implies, Not, Or,
    Xor, false, true)
from sympy_extras.logic.cnf import Clauses, conjunction, to_clauses
from sympy_extras.logic.dpll2 import SATSolver, is_satisfiable, propositional_models

a, b, c, d, e = Symbol('a'), Symbol('b'), Symbol('c'), Symbol('d'), Symbol('e')
_ATOMS: list[Symbol] = [a, b, c, d, e]


def _truth(f: Boolean, env: dict[Boolean, bool]) -> bool:
    """The value of ``f`` under ``env``, connective by connective."""
    found = env.get(f)
    if found is not None:
        return found
    if isinstance(f, BooleanTrue):
        return True
    if isinstance(f, BooleanFalse):
        return False
    if isinstance(f, Not):
        return not _truth(f.arg, env)
    if isinstance(f, And):
        return all(_truth(g, env) for g in f.args)
    if isinstance(f, Or):
        return any(_truth(g, env) for g in f.args)
    if isinstance(f, Xor):
        return sum(_truth(g, env) for g in f.args) % 2 == 1
    if isinstance(f, Implies):
        return not _truth(f.args[0], env) or _truth(f.args[1], env)
    if isinstance(f, Equivalent):
        return len({_truth(g, env) for g in f.args}) == 1
    if isinstance(f, ITE):
        return _truth(f.args[1], env) if _truth(f.args[0], env) else _truth(f.args[2], env)
    raise TypeError(f)


def _random_formula(rng: random.Random, depth: int) -> Boolean:
    """An unevaluated formula, so that tautologies and repeated atoms stay."""
    if depth == 0 or rng.random() < 0.3:
        atom: Boolean = rng.choice(_ATOMS[:rng.randint(2, 5)])
        return atom if rng.random() < 0.6 else Not(atom)
    kind = rng.randrange(7)
    args = [_random_formula(rng, depth - 1) for _ in range(rng.randint(2, 3))]
    if kind == 0:
        return And(*args)
    if kind == 1:
        return Or(*args)
    if kind == 2:
        return Not(args[0])
    if kind == 3:
        return Xor(*args)
    if kind == 4:
        return Implies(args[0], args[1])
    if kind == 5:
        return Equivalent(*args)
    return ITE(args[0], args[1], _random_formula(rng, depth - 1))


def test_models_are_the_rows_of_the_truth_table() -> None:
    rng = random.Random(3)
    for _ in range(300):
        f = _random_formula(rng, rng.randint(1, 4))
        atoms = sorted(f.free_symbols, key=str)
        models = list(propositional_models(to_clauses(f)))
        keys = [frozenset(m.items()) for m in models]
        # each model once, every atom of the formula assigned
        assert len(set(keys)) == len(keys)
        assert all(set(m) == set(atoms) for m in models)
        expected: set[frozenset[tuple[Boolean, bool]]] = set()
        for values in product([False, True], repeat=len(atoms)):
            env: dict[Boolean, bool] = dict(zip(atoms, values))
            if _truth(f, env):
                expected.add(frozenset(env.items()))
        assert set(keys) == expected, f
        assert is_satisfiable(to_clauses(f)) == bool(expected)


def test_order_of_the_models_is_deterministic() -> None:
    f = Or(And(a, b), Xor(c, d), Not(e))
    first = [list(m.items()) for m in propositional_models(to_clauses(f))]
    assert first == [list(m.items()) for m in propositional_models(to_clauses(f))]
    # the atoms are numbered as the clauses meet them, not by their hashes
    assert to_clauses(Or(b, a)).atoms == to_clauses(Or(b, a)).atoms


def test_solver_on_clauses() -> None:
    # the example of SymPy's dpll2 docstring: two models of five clauses
    solver = SATSolver([[2, -3], [1], [3, -3], [2, -2], [3, -2]], 3)
    assert list(solver.find_models()) == [{1: True, 2: False, 3: False}, {1: True, 2: True, 3: True}]
    assert list(SATSolver([[1], [-1]], 1).find_models()) == []
    # no clause: the negative literal of the last variable is decided first
    assert list(SATSolver([], 2).find_models()) == [{1: False, 2: False}, {1: True, 2: False},
                                                     {1: False, 2: True}, {1: True, 2: True}]


def test_pigeonhole_is_unsatisfiable() -> None:
    # three pigeons in two holes: p[i][j] for pigeon i in hole j
    p = [[Symbol('p%d%d' % (i, j)) for j in range(2)] for i in range(3)]
    clauses: list[Boolean] = [Or(p[i][0], p[i][1]) for i in range(3)]
    clauses += [Or(Not(p[i][j]), Not(p[k][j])) for j in range(2) for i in range(3) for k in range(i + 1, 3)]
    assert not is_satisfiable(to_clauses(And(*clauses)))
    assert is_satisfiable(to_clauses(And(*clauses[:2], *clauses[3:])))


def test_truth_values_and_tautologies() -> None:
    assert to_clauses(true).clauses == () and to_clauses(false).is_false
    assert list(propositional_models(to_clauses(true))) == [{}]
    assert list(propositional_models(to_clauses(false))) == []
    # a tautological clause keeps its atom a variable
    assert list(propositional_models(to_clauses(Or(a, Not(a))))) == [{a: False}, {a: True}]


def test_expansion_and_conjunction() -> None:
    x = Symbol('x')

    def expand(atom: Boolean) -> Optional[Boolean]:
        if atom == Q.real(x):
            return Or(Q.positive(x), Q.zero(x), Q.negative(x))
        return None

    cnf = to_clauses(And(Q.real(x), Not(Q.positive(x))), expand)
    assert set(cnf.atoms) == {Q.positive(x), Q.zero(x), Q.negative(x)}
    both = conjunction([cnf, to_clauses(And(Not(Q.zero(x)), Not(Q.negative(x))))])
    assert not is_satisfiable(both)
    assert isinstance(both, Clauses) and len(both.atoms) == 3


def test_same_models_in_the_same_order_as_sympy_dpll2() -> None:
    # the port makes the same decisions as SymPy's solver on the same
    # clauses (given as lists, so that SymPy keeps their order)
    from sympy.logic.algorithms.dpll2 import SATSolver as SymPySATSolver
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(1, 7)
        clauses: list[list[int]] = []
        for _ in range(rng.randint(0, 12)):
            literals = {rng.choice([1, -1])*rng.randint(1, n) for _ in range(rng.randint(1, 4))}
            clauses.append(sorted(literals, key=lambda lit: (abs(lit), lit)))
        theirs = SymPySATSolver([list(c) for c in clauses], set(range(1, n + 1)), set(), list(range(1, n + 1)))
        expected = [sorted(m.items()) for m in theirs._find_model()]
        assert [sorted(m.items()) for m in SATSolver(clauses, n).find_models()] == expected, clauses
