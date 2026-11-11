"""Step-by-step systems: the solutions of every system node of every
tree (alternatives included) are compared with the real solutions SymPy
finds for that node's own system; rule trees, rejected solutions,
systems which are not solved."""
from __future__ import annotations

import random
from typing import Optional, Sequence

import sympy

from sympy_extras.core import I, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr, ExprLike
from sympy_extras.sets.sets import Reals, Set
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.functions import exp, sqrt
from sympy_extras.steps.core import Step, format_steps
from sympy_extras.steps.systems import (AlternativeRule, DontKnowRule, EliminationRule, InconsistentRule,
                                        SingleEquationRule, SubstitutionRule, SystemStep, manual_system as manual_system_in,
                                        system_steps as system_steps_in)


def system_steps(equations: Sequence[tuple[ExprLike, ExprLike]], variables: Sequence[Symbol], branch: bool = False) -> SystemStep:
    """The tests of this first part are about the real solutions."""
    return system_steps_in(equations, variables, Reals, branch)


def manual_system(equations: Sequence[tuple[ExprLike, ExprLike]], variables: Sequence[Symbol]
                  ) -> Optional[tuple[tuple[Expr, ...], ...]]:
    return manual_system_in(equations, variables, Reals)

x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

_System = tuple[list[tuple[Expr, Expr]], list[Symbol]]


def _sympy(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def _real_solutions(step: SystemStep) -> list[tuple[sympy.Expr, ...]]:
    """The real solutions of the system of the node, by SymPy."""
    symbols = [sympy.Symbol(v.name) for v in step.variables]
    equations = [_sympy(lhs) - _sympy(rhs) for lhs, rhs in step.equations]
    found = sympy.solve(equations, symbols, dict=True)
    solutions: list[tuple[sympy.Expr, ...]] = []
    for solution in found:
        assert set(solution) == set(symbols), (step.equations, solution)       # finitely many
        values = tuple(sympy.sympify(solution[s]) for s in symbols)
        if all(abs(sympy.im(sympy.N(value, 30))) < sympy.Float('1e-20') for value in values):
            solutions.append(values)
    return solutions


def _same(a: sympy.Expr, b: sympy.Expr) -> bool:
    return bool(abs(sympy.N(a - b, 40)) < sympy.Float('1e-30'))


def _check_tree(step: Step) -> None:
    if isinstance(step, SystemStep) and step.variables and not step.contains_dont_know():
        ours = [tuple(_sympy(value) for value in solution) for solution in step.solutions()]
        symbols = [sympy.Symbol(v.name) for v in step.variables]
        # every solution found satisfies the system of the node ...
        for candidate in ours:
            rule: dict[sympy.Basic | complex, sympy.Expr | complex] = dict(zip(symbols, candidate))
            for lhs, rhs in step.equations:
                assert _same(_sympy(lhs).subs(rule), _sympy(rhs).subs(rule)), (step.equations, candidate)
        # ... and none of SymPy's is missing (SymPy itself misses some: solve([y**2 + sqrt(y)], [y]) is empty,
        # although 0 is a solution, so the numbers are compared only in this direction)
        expected = _real_solutions(step)
        assert len(ours) >= len(expected) and len(set(ours)) == len(ours), (step.equations, ours, expected)
        for solution in expected:
            assert any(all(_same(u, v) for u, v in zip(candidate, solution)) for candidate in ours), (step.equations, solution)
    for child in step.substeps():
        _check_tree(child)


SYSTEMS: list[_System] = [
    ([(x + y, Integer(3)), (x - y, Integer(1))], [x, y]),
    ([(2*x + 3*y, Integer(7)), (4*x - y, Integer(7))], [x, y]),
    ([(3*x + 2*y, Integer(12)), (5*x - 3*y, Integer(1))], [x, y]),
    ([(x/2 + y/3, Integer(1)), (x - y, Rational(1, 2))], [x, y]),
    ([(x + y + z, Integer(6)), (x - y, Integer(-1)), (y + z, Integer(5))], [x, y, z]),
    ([(2*x - y + z, Integer(3)), (x + 2*y - z, Integer(1)), (3*x + y + 2*z, Integer(11))], [x, y, z]),
    ([(x + y, Integer(1)), (2*x + 2*y, Integer(3))], [x, y]),
    ([(x**2 + y**2, Integer(5)), (x + y, Integer(1))], [x, y]),
    ([(y, x**2), (y, x + 2)], [x, y]),
    ([(x*y, Integer(6)), (x + y, Integer(5))], [x, y]),
    ([(x**2 + y**2, Integer(1)), (y, x + 5)], [x, y]),
    ([(x**2, y), (y**2, x)], [x, y]),
    ([(x**2 - y, Integer(1)), (x - y, Integer(-1))], [x, y]),
    ([(x**2 + y**2, Integer(4)), (y, x)], [x, y]),
    ([(x, Integer(2)), (x, Integer(3))], [x]),
    ([(x, Integer(2)), (x**2, Integer(4))], [x]),
    ([(x + y, Integer(2)), (x - y, Integer(0)), (x*y, Integer(1))], [x, y]),
    ([(x + y, Integer(2)), (x - y, Integer(0)), (x*y, Integer(5))], [x, y]),
]


def test_every_node_against_sympy() -> None:
    for equations, variables in SYSTEMS:
        # the alternatives of the systems in three unknowns are many (every equation and unknown, recursively)
        for branch in ((False, True) if len(variables) < 3 else (False,)):
            steps = system_steps(equations, variables, branch=branch)
            assert not steps.contains_dont_know(), equations
            _check_tree(steps)
        assert manual_system(equations, variables) == system_steps(equations, variables).solutions()


def test_random_linear_systems() -> None:
    rnd = random.Random(77)
    for size, unknowns in ((2, [x, y]), (3, [x, y, z])):
        for _ in range(8):
            while True:
                rows = [[rnd.randint(-4, 4) for _ in range(size)] for _ in range(size)]
                if sympy.Matrix(rows).det() != 0:
                    break
            values = [Rational(rnd.randint(-5, 5), rnd.choice([1, 2])) for _ in range(size)]
            equations: list[tuple[Expr, Expr]] = []
            for row in rows:
                lhs: Expr = Integer(0)
                rhs: Expr = Integer(0)
                for coefficient, unknown, value in zip(row, unknowns, values):
                    lhs, rhs = lhs + coefficient*unknown, rhs + coefficient*value
                equations.append((lhs, rhs))
            assert manual_system(equations, unknowns) == (tuple(values),), equations


def test_rule_trees() -> None:
    steps = system_steps([(x + y, 3), (x - y, 1)], [x, y])
    assert isinstance(steps, SubstitutionRule) and (steps.index, steps.variable, steps.expressions) == (0, x, (3 - y,))
    assert steps.remaining == (y,) and steps.rejected == () and steps.solutions() == ((2, 1),)
    [case] = steps.cases
    assert isinstance(case, SingleEquationRule) and case.solutions() == ((1,),)
    # the unknown with the coefficient -1 is preferred to the first one
    preferred = system_steps([(2*x + 3*y, 7), (4*x - y, 7)], [x, y])
    assert isinstance(preferred, SubstitutionRule) and (preferred.index, preferred.variable) == (1, y)
    # two values of the eliminated unknown: two cases
    quadratic = system_steps([(x**2, y), (y**2, x)], [x, y])
    assert quadratic.solutions() == ((0, 0), (1, 1))
    # 2 = 3 after the substitution: no solution, although an unknown is left ("not solved" before)
    parallel = system_steps([(x + y, 1), (2*x + 2*y, 3)], [x, y])
    assert isinstance(parallel, SubstitutionRule) and isinstance(parallel.cases[0], InconsistentRule)
    assert parallel.solutions() == ()
    assert format_steps(steps, depth=1).splitlines() == [
        'SubstitutionRule: x + y = 3, x - y = 1 -> {(2, 1)}',
        '  AddToBothSidesRule: x + y = 3 -> {3 - y}',
        '  SingleEquationRule: 3 - 2*y = 1 -> {(1,)}']


def test_alternatives_and_elimination() -> None:
    ways = system_steps([(2*x + 3*y, 7), (4*x - y, 7)], [x, y], branch=True)
    assert isinstance(ways, AlternativeRule)
    names = [way.rule for way in ways.alternatives]
    assert names.count('SubstitutionRule') == 4 and names.count('EliminationRule') == 2
    assert all(way.solutions() == ((2, 1),) for way in ways.alternatives)
    elimination = next(way for way in ways.alternatives if isinstance(way, EliminationRule))
    # 4*(2 x + 3 y) - 2*(4 x - y) = 14 y, and 4*7 - 2*7 = 14
    assert (elimination.first, elimination.second, elimination.variable) == (0, 1, x)
    assert (elimination.first_multiplier, elimination.second_multiplier) == (4, -2)
    assert elimination.combined == (14*y, 14)


def test_rejected_and_irrational_solutions() -> None:
    # sqrt(x) = y, y = x - 2: x = y + 2 gives y = 2 and y = -1, and sqrt(1) is not -1
    radical = system_steps([(sqrt(x), y), (y, x - 2)], [x, y])
    assert not radical.contains_dont_know() and radical.solutions() == ((4, 2),)
    circle = manual_system([(x**2 + y**2, 4), (y, x)], [x, y])
    assert circle is not None and set(circle) == {(sqrt(Integer(2)), sqrt(Integer(2))), (-sqrt(Integer(2)), -sqrt(Integer(2)))}


def test_not_solved() -> None:
    # infinitely many solutions, a free unknown, a transcendental system
    for equations, variables in (([(x + y, 1), (2*x + 2*y, 2)], [x, y]), ([(2*x + y, 5)], [x, y]),
                                 ([(x*exp(x) + y, 1), (x + exp(y), 2)], [x, y])):
        steps = system_steps(equations, variables)
        assert steps.contains_dont_know() and manual_system(equations, variables) is None
    assert isinstance(system_steps([(2*x + y, 5)], [x, y]), (DontKnowRule, SubstitutionRule))
    try:
        system_steps([(x, 1)], [x, x])
    except ValueError:
        pass
    else:
        raise AssertionError('repeated unknowns accepted')


def test_complex_domain() -> None:
    # the default: x**2 + y**2 = 1, y = x + 5 has two complex solutions and no real one
    sx, sy = sympy.Symbol('x'), sympy.Symbol('y')
    for equations in ([(x**2 + y**2, Integer(1)), (y, x + 5)], [(x**2 + y, Integer(0)), (x + y, Integer(1))],
                      [(x*y, Integer(1)), (x + y, Integer(1))], [(x + y, Integer(3)), (x - y, Integer(1))]):
        found = manual_system_in(equations, [x, y])
        assert found is not None
        expected = sympy.solve([sympy_expr(to_sympy(lhs)) - sympy_expr(to_sympy(rhs)) for lhs, rhs in equations], [sx, sy], dict=True)
        assert len(found) == len(expected), (equations, found, expected)
        ours = [tuple(sympy_expr(to_sympy(value)) for value in solution) for solution in found]
        for solution in expected:
            assert any(_same(u, solution[sx]) and _same(v, solution[sy]) for u, v in ours), (equations, solution)
    assert manual_system_in([(x**2 + y**2, 1), (y, x + 5)], [x, y], Reals) == ()
    assert manual_system_in([(x**2, -1)], [x]) == ((-I,), (I,)) or manual_system_in([(x**2, -1)], [x]) == ((I,), (-I,))
    assert manual_system_in([(x**2, -1)], [x], Reals) == ()
    try:
        system_steps_in([(x, 1)], [x], sympy_free_domain())
    except ValueError:
        pass
    else:
        raise AssertionError('a domain which is neither Reals nor Complexes was accepted')


def sympy_free_domain() -> Set:
    from sympy_extras.sets.fancysets import Integers
    return Integers


def test_cancelling_unknown_and_partial_systems() -> None:
    from sympy_extras.functions import abs_
    # y + |x| = y + 2 says nothing about y: it was read as an equation in y without solution, and the
    # elimination branch reported that the system had none
    equations: list[tuple[Expr, Expr]] = [(abs_(x), y), (y, Integer(2))]
    for branch in (False, True):
        steps = system_steps(equations, [x, y], branch=branch)
        assert set(steps.solutions()) == {(Integer(-2), Integer(2)), (Integer(2), Integer(2))} and steps.complete()
    assert manual_system([(y + abs_(x), y + 2), (y, Integer(2))], [x, y]) == manual_system(equations, [x, y])
    # over the complex numbers the same solutions, as a part of them
    partial = system_steps_in(equations, [x, y])
    assert set(partial.solutions()) == {(Integer(-2), Integer(2)), (Integer(2), Integer(2))} and not partial.complete()
    assert format_steps(partial, depth=0).endswith('(not all)')
    assert system_steps_in([(x + y, 3), (x - y, 1)], [x, y]).complete()
