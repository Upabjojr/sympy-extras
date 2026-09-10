from __future__ import annotations

import itertools
import random

from sympy import (S, Eq, Ne, Mod, And, Or, Not, Implies, Matrix, Tuple, Lambda, FiniteSet, ImageSet,
    ConditionSet, Symbol, Integer, true, false)
from sympy.abc import x, y, z
from sympy.core.basic import Basic
from sympy.logic.boolalg import Boolean
from sympy.testing.pytest import raises

from sympy_extras.assumptions import resolve, solve, ForAll, Exists
from sympy_extras.solvers.integers import (hermite_normal_form_with_transform,
    linear_diophantine_system, hilbert_basis, minimal_nonnegative_solutions, cooper,
    presburger_quantifier_elimination, is_presburger)
from sympy_extras._testing import untyped


def test_hermite_normal_form() -> None:
    for A in ([[2, 4, 6], [3, 1, 5]], [[6, 10, 15]], [[1, 0], [0, 1]], [[0, 0], [0, 0]],
              [[4, 6], [6, 9]], [[2, 3, 5], [7, 11, 13], [1, 1, 1]]):
        H, U = hermite_normal_form_with_transform(A)
        assert Matrix(A)*Matrix(U) == Matrix(H)
        assert abs(Matrix(U).det()) == 1
        # echelon form with nonnegative entries left of positive pivots
        pc = 0
        for row in H:
            beyond = [j for j in range(pc, len(row)) if row[j] != 0]
            if beyond:
                assert beyond == [pc]
                assert row[pc] > 0
                assert all(0 <= v < row[pc] for v in row[:pc])
                pc += 1


def test_linear_diophantine_system() -> None:
    result = linear_diophantine_system([Eq(3*x + 5*y, 7)], [x, y])
    assert result is not None
    values, parameters = result
    assert len(parameters) == 1
    t = parameters[0]
    assert (3*values[0] + 5*values[1] - 7).expand() == 0
    assert all(v.subs(t, 3).is_integer for v in values)
    assert linear_diophantine_system([Eq(2*x + 4*y, 3)], [x, y]) is None
    assert linear_diophantine_system([x + y - 1, x + y - 2], [x, y]) is None
    result = linear_diophantine_system([x + y - 3, x - y - 1], [x, y])
    assert result is not None and result[0] == [2, 1] and result[1] == []
    result = linear_diophantine_system([6*x + 10*y + 15*z - 1], [x, y, z])
    assert result is not None
    values, parameters = result
    assert len(parameters) == 2
    assert (6*values[0] + 10*values[1] + 15*values[2] - 1).expand() == 0
    raises(ValueError, lambda: linear_diophantine_system([x*y - 1], [x, y]))
    raises(ValueError, lambda: linear_diophantine_system([x/2 - 1], [x]))


def test_hilbert_basis() -> None:
    assert hilbert_basis([[1, 1, -1]]) == [[0, 1, 1], [1, 0, 1]]
    assert hilbert_basis([[2, -3]]) == [[3, 2]]
    assert hilbert_basis([[1, 1]]) == []
    basis = hilbert_basis([[1, 1, -1, -1]])
    assert basis == [[0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]]
    # every nonnegative solution in a box is a combination of the basis
    A = [[2, 3, -5, 1], [1, -1, 0, -1]]
    basis = hilbert_basis(A)
    for v in itertools.product(range(4), repeat=4):
        if any(v) and all(sum(r[j]*v[j] for j in range(4)) == 0 for r in A):
            assert any(all(a >= b for a, b in zip(v, s)) for s in basis)
    raises(RuntimeError, lambda: hilbert_basis([[1, -1]], limit=1))


def test_minimal_nonnegative_solutions() -> None:
    assert minimal_nonnegative_solutions([[3, 5]], [22]) == ([[4, 2]], [])
    assert minimal_nonnegative_solutions([[1, 1]], [2]) == ([[0, 2], [1, 1], [2, 0]], [])
    assert minimal_nonnegative_solutions([[3, -5]], [1]) == ([[2, 1]], [[5, 3]])
    assert minimal_nonnegative_solutions([[2, 4]], [3]) == ([], [])


Point = dict[Basic, Basic]


def _brute_force(formula: Boolean, variable: Symbol, values: Point, bound: int = 12) -> bool:
    return any(formula.xreplace({**values, variable: Integer(v)}) is S.true for v in range(-bound, bound + 1))


def _check_cooper(formula: Boolean, free: list[Symbol], bound: int = 4) -> None:
    result = cooper(formula, x)
    assert not result.has(x)
    for point in itertools.product(range(-bound, bound + 1), repeat=len(free)):
        values: Point = {s: Integer(v) for s, v in zip(free, point)}
        expected = _brute_force(formula, x, values)
        actual = result.xreplace(values)
        assert actual in (S.true, S.false), (formula, values, actual)
        assert (actual is S.true) == expected, (formula, values, result)


def test_cooper() -> None:
    assert cooper(Eq(2*x, y), x) == Eq(Mod(y, 2), 0)
    assert cooper((x > y) & (x < y + 2), x) is S.true
    assert cooper((x > y) & (x < y + 1), x) is S.false
    assert cooper(x > y, x) is S.true
    assert cooper(Eq(x, y) & Eq(x, z), x) in (Eq(y, z), Eq(z, y))
    formulas = [
        (x > y) & (x < y + 2), (3*x > y) & (3*x < y + 3), Eq(2*x, y),
        (x > y) & (x < z), (2*x >= y) & (3*x <= z), Eq(2*x + 3*y, z),
        Ne(x, y) & (x > y - 1) & (x < y + 2), Eq(Mod(x + y, 3), 0) & (x > z) & (x < z + 2),
        Not(Eq(Mod(x, 2), 0)) & (x >= y) & (x <= y + 1), Or(Eq(x, y), Eq(x, z)) & (x > 0),
        Implies(x > y, x > z) & (x <= y + 1), Eq(Mod(2*x + y, 4), 0) & (x > z),
        (5*x > 2*y + 1) & (5*x < 2*y + 7), Eq(3*x, y) & Ne(x, z),
    ]
    for f in formulas:
        free = sorted([s for s in f.free_symbols if isinstance(s, Symbol) and s != x], key=lambda s: s.name)
        _check_cooper(f, free)


def test_cooper_random() -> None:
    rng = random.Random(7)
    kinds = ['lt', 'gt', 'le', 'ge', 'eq', 'ne', 'mod']
    for _ in range(40):
        atoms: list[Boolean] = []
        for _ in range(rng.randint(1, 3)):
            a = rng.choice([1, 2, 3, -1, -2])
            b = rng.randint(-2, 2)
            c = rng.randint(-3, 3)
            e = a*x + b*y + c
            kind = rng.choice(kinds)
            if kind == 'lt':
                atoms.append(e < 0)
            elif kind == 'gt':
                atoms.append(e > 0)
            elif kind == 'le':
                atoms.append(e <= 0)
            elif kind == 'ge':
                atoms.append(e >= 0)
            elif kind == 'eq':
                atoms.append(Eq(e, 0))
            elif kind == 'ne':
                atoms.append(Ne(e, 0))
            else:
                atoms.append(Eq(Mod(e, rng.choice([2, 3])), 0))
        connective = rng.choice([And, Or])
        f = connective(*atoms) if len(atoms) > 1 else atoms[0]
        if rng.random() < 0.3:
            f = Not(f)
        _check_cooper(f, [y], bound=5)


def test_presburger_quantifier_elimination() -> None:
    assert presburger_quantifier_elimination(x < y, [('forall', x), ('exists', y)]) is S.true
    assert presburger_quantifier_elimination(x < y, [('exists', y), ('forall', x)]) is S.false
    assert presburger_quantifier_elimination(
        Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2), [('forall', x), ('exists', y)]) is S.true
    assert presburger_quantifier_elimination(
        Eq(3*y, x) | Eq(3*y, x + 1), [('forall', x), ('exists', y)]) is S.false
    assert is_presburger((2*x > y) & Eq(Mod(x, 3), 0), {x, y})
    assert not is_presburger(x*y > 0, {x, y})
    assert not is_presburger(x > y, {x})
    raises(ValueError, lambda: cooper(x**2 > y, x))


def test_resolve_integers() -> None:
    assert resolve(Exists(x, Eq(2*x, y)), domain=S.Integers) == Eq(Mod(y, 2), 0)
    assert resolve(Exists(x, (x > y) & (x < y + 1)), domain=S.Integers) is S.false
    assert resolve(Exists(x, (x > y) & (x < y + 1))) == S.true
    assert resolve(ForAll(x, Exists(y, y > x)), domain=S.Integers) is S.true
    assert resolve(Exists(y, ForAll(x, y > x)), domain=S.Integers) is S.false
    assert resolve(Exists(x, Eq(3*x, y) & (x > 0)), domain=S.Integers) == And(y > 0, Eq(Mod(y, 3), 0))
    assert resolve(Exists(x, Eq(3*x, y) & (x > 0)), domain=S.Integers, assumptions=y > 5) == Eq(Mod(y, 3), 0)
    assert resolve(ForAll(x, (x > y) | (x < z)), domain=S.Integers) == (z > y)
    raises(ValueError, lambda: resolve(ForAll(x, x**2 > y), domain=S.Integers))
    raises(NotImplementedError, lambda: resolve(ForAll(x, x > 0), domain=S.Rationals))


def test_solve_integers() -> None:
    assert solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=S.Integers) == FiniteSet(Tuple(4, 2))
    assert solve(Eq(x + y, 2), [x, y], (x >= 0) & (y >= 0), domain=S.Integers) == FiniteSet(Tuple(0, 2), Tuple(1, 1), Tuple(2, 0))
    assert solve(Eq(2*x + 4*y, 3), [x, y], domain=S.Integers) is S.EmptySet
    assert solve([x + y - 3, x - y - 1], [x, y], domain=S.Integers) == FiniteSet(Tuple(2, 1))
    assert solve([x + y - 3, x - y - 1], [x, y], x > 5, domain=S.Integers) is S.EmptySet
    result = solve(Eq(3*x + 5*y, 22), [x, y], domain=S.Integers)
    assert isinstance(result, ImageSet)
    lam = result.lamda
    assert isinstance(lam, Lambda)
    t = lam.variables[0]
    values = lam.expr
    assert (3*values.args[0] + 5*values.args[1] - 22).expand() == 0
    assert values.subs(t, 1) in result
    result = solve([x + y + z - 6, x - y - 2], [x, y, z], x > 0, domain=S.Integers)
    assert isinstance(result, ConditionSet)
    # the solutions in nonnegative integers are infinitely many: parametrised
    result = solve(Eq(3*x - 5*y, 1), [x, y], (x >= 0) & (y >= 0), domain=S.Integers)
    assert isinstance(result, ConditionSet)
    # nonlinear systems still go to nonlinsolve
    assert solve([x**2 - 4, x - y], [x, y], x > 0, domain=S.Integers) == FiniteSet(Tuple(2, 2))


def test_wrong_types() -> None:
    raises(TypeError, lambda: untyped(cooper)(1, x))
    raises(ValueError, lambda: linear_diophantine_system([Eq(x, 1)], []))


def test_mutually_covering_disjuncts_are_both_kept() -> None:
    # sympy-extras#45 and #51: _drop_covered tested each disjunct against
    # the *original* list, so two disjuncts pinning the same point covered
    # each other and were both dropped, losing the point. This
    # disjunction is satisfiable at x = 0 and came back False.
    from sympy_extras.solvers.integers import _drop_covered
    x = Symbol('x', integer=True)
    kept = _drop_covered(Or(And(Eq(x, 0), Ne(x, 1)), And(Eq(x, 0), Ne(x, 2))))
    assert kept.subs(x, 0) is true


def test_cooper_keeps_the_solution_of_two_equalities() -> None:
    # sympy-extras#51: y = 0 forces -x + 2 = 0, so the answer is x = 2,
    # but a positive coefficient >= 2 on the eliminated variable made
    # cooper return False; the sign-flipped equation was fine.
    x, y = Symbol('x', integer=True), Symbol('y', integer=True)
    for sign in (1, -1):
        eliminated = cooper(And(Eq(y, 0), Eq(sign*(-x + 3*y + 2), 0)), y)
        assert eliminated.subs(x, 2) is true
        assert eliminated.subs(x, 3) is false


def test_resolve_does_not_depend_on_the_order_of_bound_variables() -> None:
    # sympy-extras#51: Exists([x, y], F) and Exists([y, x], F) gave
    # different answers, because one elimination order hit the lost
    # solution; (2, 0) is a witness so both must be True.
    x, y = Symbol('x', integer=True), Symbol('y', integer=True)
    F = And(Eq(y, 0), Eq(x - 3*y - 2, 0))
    assert resolve(Exists([x, y], F), domain=S.Integers) is true
    assert resolve(Exists([y, x], F), domain=S.Integers) is true


def test_resolve_does_not_call_a_false_universal_true() -> None:
    # sympy-extras#51 (universal direction): F is false at (1, 1), so the
    # universal statement is false, yet resolve said True in one variable
    # order -- the dangerous direction for a decision procedure.
    x, y = Symbol('x', integer=True), Symbol('y', integer=True)
    F = Or(Ne(1 - x, 0), (-x + 2*y + 1 <= 0),
           And(Ne(-x + y - 1, 0), Eq(-2*x - 3*y - 2, 0)))
    assert F.subs({x: 1, y: 1}) is false
    assert resolve(ForAll([x, y], F), domain=S.Integers) is false
    assert resolve(ForAll([y, x], F), domain=S.Integers) is false


def test_universal_with_two_equalities_in_the_conclusion() -> None:
    # sympy-extras#45: forall x, y: (x + y = 0 and x - y = 0) implies
    # (x = 0 and y = 0) is valid, and resolve returned False because
    # _drop_covered removed both disjuncts holding at x = 0.
    x, y = Symbol('x', integer=True), Symbol('y', integer=True)
    claim = Implies(And(Eq(x + y, 0), Eq(x - y, 0)), And(Eq(x, 0), Eq(y, 0)))
    assert resolve(ForAll([x, y], claim), domain=S.Integers) is true


def test_cooper_numerical_semigroup() -> None:
    # sympy-extras#27: {2n + 3k : n, k >= 0} is every nonnegative integer
    # except 1, but the second elimination returned (m >= 0) & (m even),
    # wrong at m = 3 (n = 0, k = 1) and every odd m >= 3.
    n, k, m = (Symbol(s, integer=True) for s in 'nkm')
    condition = resolve(Exists((n, k), And(Eq(2*n + 3*k, m), n >= 0, k >= 0)), S.Integers)
    for value in range(0, 12):
        expected = value != 1
        assert bool(condition.subs(m, value)) is expected, value


def test_a_residue_other_than_zero_is_presburger() -> None:
    # sympy-extras#44: Eq(Mod(x, 3), 1) was refused as not Presburger,
    # although it is 3 | x - 1; a residue outside [0, k) never holds.
    x = Symbol('x', integer=True)
    assert is_presburger(Eq(Mod(x, 3), 1), {x})
    condition = resolve(Exists([x], And(Eq(Mod(x, 3), 1), x > 0, x < 3)), domain=S.Integers)
    assert condition is true
    assert resolve(Exists([x], Eq(Mod(x, 3), 5)), domain=S.Integers) is false
    assert resolve(ForAll([x], Ne(Mod(x, 3), 7)), domain=S.Integers) is true


def test_nonlinear_universal_goals_are_decided_over_the_reals() -> None:
    # sympy-extras#46: every nonlinear formula was refused over the
    # integers; a universal statement true over R is true over Z, and an
    # existential one false over R is false over Z.
    a, b = Symbol('a', integer=True), Symbol('b', integer=True)
    assert resolve(ForAll([a], Implies(3 < a, 0 <= a**3 + a)), domain=S.Integers) is true
    assert resolve(ForAll([a, b], Implies(And(0 < a, 0 < b), 0 < a*b)), domain=S.Integers) is true
    assert resolve(Exists([a], And(a**2 < 0)), domain=S.Integers) is false
    # the other directions do not transfer, and are still refused
    raises(ValueError, lambda: resolve(Exists([a], Eq(a**2, 2)), domain=S.Integers))
