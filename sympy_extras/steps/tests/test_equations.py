"""Step-by-step equations: the solution set of every node of every tree
(the alternatives included) is the real solution set SymPy finds for
that node's own equation; the kinds of the steps, rejected roots,
alternatives and rendering."""
from __future__ import annotations

import random

import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import sin, sqrt
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.sets.sets import EmptySet, FiniteSet, Reals, finite_set
from sympy_extras.steps.core import Step, count_steps, format_steps
from sympy_extras.steps.equations import (AddToBothSidesRule, AlternativeRule, ClearDenominatorsRule, CompleteSquareRule,
                                          ContradictionRule, DivideBothSidesRule, DontKnowRule, EquationStep, IdentityRule,
                                          QuadraticFormulaRule, RootRule, SolvedRule, SquareBothSidesRule, ZeroProductRule,
                                          manual_solve, solve_steps)

x, a = Symbol('x'), Symbol('a')
sx = sympy.Symbol('x', real=True)


def _expected(lhs: Expr, rhs: Expr) -> sympy.Set:
    rule: dict[sympy.Basic | complex, sympy.Expr | complex] = {sympy.Symbol('x'): sx}
    equation = sympy.Eq(sympy_expr(to_sympy(lhs)).subs(rule), sympy_expr(to_sympy(rhs)).subs(rule), evaluate=False)
    return sympy.solveset(equation, sx, sympy.S.Reals)


def _agrees(step: EquationStep) -> bool:
    found = step.solutions()
    expected = _expected(step.lhs, step.rhs)
    if found is Reals:
        return bool(expected == sympy.S.Reals)
    if found is EmptySet:
        return bool(expected == sympy.S.EmptySet)
    assert isinstance(found, FiniteSet)
    ours = [sympy_expr(to_sympy(e)) for e in found.args]
    if not isinstance(expected, sympy.FiniteSet) or len(expected) != len(ours):
        return False
    return all(any(abs(sympy.N(e - o, 40)) < sympy.Float('1e-30') for o in ours) for e in expected)


def _check_tree(step: Step) -> None:
    assert isinstance(step, EquationStep)
    if not step.contains_dont_know():
        assert _agrees(step), step
    for child in step.substeps():
        _check_tree(child)


EQUATIONS: list[tuple[Expr, Expr]] = [
    (2*x + 3, Integer(7)), (3*x - 1, x + 5), (2*(x + 1), 3*x - 4), (x/2 + Rational(1, 3), Integer(2)), (x, Integer(4)),
    (Integer(5), 2*x - 1), (x + 1, x + 1), (x, x + 1), (x**2 + x, x**2 + 3),
    (x**2 - 5*x + 6, Integer(0)), (x**2, 4*x - 4), (x**2 - 2*x - 1, Integer(0)), (x**2 + 1, Integer(0)),
    (2*x**2 + 3*x, Integer(2)), (x**2, Integer(2)), ((x - 1)**2, Integer(9)), ((2*x + 1)**2, Integer(0)), (x*(x - 3), Integer(4)),
    (x**3 - x, Integer(0)), (x**3, Integer(8)), (x**3 - 2, Integer(0)), (x**4 - 5*x**2 + 4, Integer(0)), (x**4, Integer(-1)),
    (x**4 - 16, Integer(0)), (x**3 - 6*x**2 + 11*x, Integer(6)),
    (1/(x - 1), 2/(x**2 - 1)), (x/(x + 1), Integer(2)), (1/x + 1/(x + 1), Integer(1)), ((x**2 - 1)/(x - 1), Integer(3)),
    (1/x, 1/(2*x - 3)), (2/(x - 2), x/(x - 2)),
    (sqrt(x + 2), x), (sqrt(2*x + 3) + 1, x), (sqrt(x), Integer(3)), (sqrt(x - 1), Integer(-2)), (x, sqrt(6 - x)),
    (2*sqrt(x + 1), x - 2),
]


def test_every_node_against_solveset() -> None:
    for lhs, rhs in EQUATIONS:
        steps = solve_steps(lhs, rhs, x)
        assert not steps.contains_dont_know(), (lhs, rhs)
        _check_tree(steps)
        assert manual_solve(lhs, rhs, x) == steps.solutions()


def test_every_alternative_against_solveset() -> None:
    for lhs, rhs in EQUATIONS[::2]:
        steps = solve_steps(lhs, rhs, x, branch=True)
        assert not steps.contains_dont_know()
        _check_tree(steps)
        assert steps.solutions() == solve_steps(lhs, rhs, x).solutions()


def test_random_linear_and_quadratic_equations() -> None:
    rnd = random.Random(191)
    for _ in range(30):
        def side(degree: int) -> Expr:
            return sum((Rational(rnd.randint(-5, 5), rnd.choice([1, 1, 2]))*x**k for k in range(degree + 1)), Integer(0))
        degree = rnd.choice([1, 1, 2])
        lhs, rhs = side(degree), side(rnd.randint(0, degree))
        steps = solve_steps(lhs, rhs, x)
        assert not steps.contains_dont_know(), (lhs, rhs)
        _check_tree(steps)


def test_rule_trees_and_kinds() -> None:
    linear = solve_steps(2*x + 3, 7, x)
    assert isinstance(linear, AddToBothSidesRule) and linear.term == -3 and linear.kind == 'equivalence'
    assert isinstance(linear.substep, DivideBothSidesRule) and linear.substep.divisor == 2
    assert linear.substep.substep == SolvedRule(x, Integer(2), x, Integer(2))
    factored = solve_steps(x**2 - 5*x + 6, 0, x)
    assert isinstance(factored, ZeroProductRule) and factored.kind == 'cases' and set(factored.factors) == {x - 2, x - 3}
    formula = solve_steps(x**2 - 2*x - 1, 0, x)
    assert isinstance(formula, QuadraticFormulaRule) and (formula.a, formula.b, formula.c, formula.discriminant) == (1, -2, -1, 8)
    assert formula.solutions() == finite_set(1 - sqrt(Integer(2)), 1 + sqrt(Integer(2)))
    assert solve_steps(x**2 + 1, 0, x).solutions() is EmptySet
    assert isinstance(solve_steps(x + 1, x + 1, x), IdentityRule) and isinstance(solve_steps(x, x + 1, x), ContradictionRule)
    roots = solve_steps((x - 1)**2, 9, x)
    assert isinstance(roots, RootRule) and roots.exponent == 2 and len(roots.cases) == 2
    assert solve_steps(x**4, -1, x).solutions() is EmptySet and solve_steps(x**3, -8, x).solutions() == finite_set(-2)


def test_rejected_roots() -> None:
    # squaring: x + 2 = x**2 has the roots -1 and 2, and -1 does not satisfy sqrt(x + 2) = x
    radical = solve_steps(sqrt(x + 2), x, x)
    assert isinstance(radical, SquareBothSidesRule) and radical.kind == 'implication'
    assert radical.rejected == (-1,) and radical.unverified == () and radical.solutions() == finite_set(2)
    assert radical.substep.solutions() == finite_set(-1, 2)
    # a denominator: the root 1 of x + 1 = 2 is a pole of both sides (it was accepted before the check of the poles)
    rational = solve_steps(1/(x - 1), 2/(x**2 - 1), x)
    assert isinstance(rational, ClearDenominatorsRule) and rational.rejected == (1,) and rational.solutions() is EmptySet
    assert solve_steps(sqrt(x - 1), -2, x).solutions() is EmptySet
    # a common factor is not cancelled before the check: (x**2 - 1)/(x - 1) = 2 has no solution, x = 1 being excluded
    assert solve_steps((x**2 - 1)/(x - 1), 2, x).solutions() is EmptySet
    kept = solve_steps(x/(x + 1), 2, x)
    assert isinstance(kept, ClearDenominatorsRule) and kept.rejected == () and kept.solutions() == finite_set(-2)


def test_alternatives() -> None:
    ways = solve_steps(x**2 - 5*x + 6, 0, x, branch=True)
    assert isinstance(ways, AlternativeRule)
    assert [way.rule for way in ways.alternatives] == ['ZeroProductRule', 'QuadraticFormulaRule', 'CompleteSquareRule']
    assert all(way.solutions() == finite_set(2, 3) for way in ways.alternatives)
    completed = ways.alternatives[2]
    assert isinstance(completed, CompleteSquareRule) and completed.shift == Rational(-5, 2) and completed.value == Rational(1, 4)
    # an irreducible quadratic has no factoring; a linear equation may move the unknowns or the constants first
    assert [way.rule for way in solve_steps(x**2 - 2*x - 1, 0, x, branch=True).substeps()] == [
        'QuadraticFormulaRule', 'CompleteSquareRule']
    linear = solve_steps(3*x - 1, x + 5, x, branch=True)
    assert isinstance(linear, AlternativeRule) and len(linear.alternatives) == 2
    assert count_steps(linear) > count_steps(solve_steps(3*x - 1, x + 5, x))
    # the first alternative is the default derivation
    assert solve_steps(x**2 - 5*x + 6, 0, x) == ways.alternatives[0]


def test_parameters_and_unsolved() -> None:
    assert solve_steps(a*x + 1, 3, x).solutions() == finite_set(2/a)
    assert solve_steps(x**2, a**2, x, branch=False).solutions() == finite_set(-sqrt(a**2), sqrt(a**2))
    for lhs, rhs in ((x**3 + x + 1, Integer(0)), (sin(x), Rational(1, 2)), (x**5 - x - 1, Integer(0))):
        steps = solve_steps(lhs, rhs, x)
        assert steps.contains_dont_know() and manual_solve(lhs, rhs, x) is None
    assert isinstance(solve_steps(sin(x), 0, x), DontKnowRule)
    # a factor which is not solved leaves the tree incomplete, and says where
    partial = solve_steps((x - 1)*(x**3 + x + 1), 0, x)
    assert partial.contains_dont_know() and format_steps(partial).splitlines()[0].endswith('-> ?')


def test_rendering() -> None:
    assert format_steps(solve_steps(2*x + 3, 7, x)).splitlines() == [
        'AddToBothSidesRule: 2*x + 3 = 7 -> {2}', '  DivideBothSidesRule: 2*x = 4 -> {2}', '    SolvedRule: x = 2 -> {2}']
    assert format_steps(solve_steps(2*x + 3, 7, x), depth=0) == 'AddToBothSidesRule: 2*x + 3 = 7 -> {2}'
    # the expansion of the sides is a minor step
    hidden = format_steps(solve_steps((x + 1)**2, x**2 + 5, x)).splitlines()
    shown = format_steps(solve_steps((x + 1)**2, x**2 + 5, x), minor=True).splitlines()
    assert shown[0].startswith('RewriteRule: ') and not hidden[0].startswith('RewriteRule: ') and len(shown) == len(hidden) + 1
