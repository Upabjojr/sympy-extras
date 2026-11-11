"""Step-by-step equations: the solution set of every node of every tree
(the alternatives included) is the real solution set SymPy finds for
that node's own equation; the kinds of the steps, rejected roots,
alternatives and rendering."""
from __future__ import annotations

import random

import sympy

from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import abs_, cos, exp, log, sin, sqrt, tan
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.sets.sets import EmptySet, FiniteSet, Reals, finite_set
from sympy_extras.steps.core import Step, count_steps, format_steps
from sympy_extras.steps.equations import (AbsoluteValueRule, CardanoRule, PeriodicRule, RealRootsRule, SubstitutionRule,
                                          TakeLogarithmRule, AddToBothSidesRule, AlternativeRule, ClearDenominatorsRule, CompleteSquareRule,
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
    for lhs, rhs in ((x*exp(x), Integer(1)), (sin(x), x), (x + log(x), Integer(0))):
        steps = solve_steps(lhs, rhs, x)
        assert steps.contains_dont_know() and manual_solve(lhs, rhs, x) is None
    assert isinstance(solve_steps(sin(x), x, x), DontKnowRule)
    # a factor which is not solved leaves the tree incomplete, and says where
    partial = solve_steps(abs_(x*exp(x) - 1), 2, x)
    assert partial.contains_dont_know() and format_steps(partial).splitlines()[0].endswith('-> ?')


def test_rendering() -> None:
    assert format_steps(solve_steps(2*x + 3, 7, x)).splitlines() == [
        'AddToBothSidesRule: 2*x + 3 = 7 -> {2}', '  DivideBothSidesRule: 2*x = 4 -> {2}', '    SolvedRule: x = 2 -> {2}']
    assert format_steps(solve_steps(2*x + 3, 7, x), depth=0) == 'AddToBothSidesRule: 2*x + 3 = 7 -> {2}'
    # the expansion of the sides is a minor step
    hidden = format_steps(solve_steps((x + 1)**2, x**2 + 5, x)).splitlines()
    shown = format_steps(solve_steps((x + 1)**2, x**2 + 5, x), minor=True).splitlines()
    assert shown[0].startswith('RewriteRule: ') and not hidden[0].startswith('RewriteRule: ') and len(shown) == len(hidden) + 1


# ---------------------------------------------------------------------
# radicals, absolute values, exponentials, logarithms, higher degrees

def _numerically_satisfied(lhs: Expr, rhs: Expr, value: sympy.Expr) -> bool:
    rule: dict[sympy.Basic | complex, sympy.Expr | complex] = {sympy.Symbol('x'): value}
    difference = sympy_expr(to_sympy(lhs)).subs(rule) - sympy_expr(to_sympy(rhs)).subs(rule)
    return bool(abs(sympy.N(difference, 40)) < sympy.Float('1e-25'))


def _real_sides(lhs: Expr, rhs: Expr, value: sympy.Expr) -> bool:
    rule: dict[sympy.Basic | complex, sympy.Expr | complex] = {sympy.Symbol('x'): value}
    sides = [sympy.N(sympy_expr(to_sympy(side)).subs(rule), 30) for side in (lhs, rhs)]
    return all(abs(sympy.im(side)) < sympy.Float('1e-20') for side in sides)


MORE: list[tuple[Expr, Expr]] = [
    (sqrt(x + 5) + sqrt(x), Integer(5)), (sqrt(2*x + 1) - sqrt(x), Integer(1)), (sqrt(x + 1) + sqrt(x - 1), Integer(1)),
    (sqrt(3*x + 1), sqrt(x) + 1),
    (abs_(x - 3), Integer(5)), (abs_(2*x + 1), x + 4), (abs_(x), Integer(-1)), (abs_(x - 1) + 2, Integer(5)),
    (abs_(x**2 - 4), Integer(3)), (abs_(x - 2), 2*x + 5), (3*abs_(x + 1), Integer(0)),
    (2**x, Integer(8)), (exp(2*x), Integer(5)), (3*2**(x + 1), Integer(48)), (exp(x), Integer(-1)),
    (2**(x + 1), 2**(3*x - 5)), (exp(2*x) - 3*exp(x) + 2, Integer(0)), (exp(x) + 2, Integer(7)), (exp(x**2), Integer(3)),
    (log(x), Integer(2)), (2*log(x - 1), Integer(4)), (log(x) + log(x - 3), log(Integer(10))), (log(x + 1), log(2*x - 3)),
    (log(x)**2 - 3*log(x) + 2, Integer(0)), (log(x - 5), log(1 - x)),
    (x**3 + x + 1, Integer(0)), (x**3 - 3*x + 1, Integer(0)), (x**4 - x - 1, Integer(0)), (x**5 - x - 1, Integer(0)),
    (2*x**3 - 4*x, Integer(-5)), (x**4 + x**2 + 1, Integer(0)),
]


def test_more_equations_against_solveset() -> None:
    for lhs, rhs in MORE:
        steps = solve_steps(lhs, rhs, x)
        assert not steps.contains_dont_know(), (lhs, rhs)
        expected = _expected(lhs, rhs)
        found = steps.solutions()
        if isinstance(expected, sympy.FiniteSet) or expected == sympy.S.EmptySet:
            # SymPy's solveset over the reals keeps -2 for log(x) + log(x - 3) = log(10), where the logarithms are
            # not real: its candidates are filtered by the real values of the two sides
            real = [e for e in expected if _real_sides(lhs, rhs, e)] if isinstance(expected, sympy.FiniteSet) else []
            ours = [sympy_expr(to_sympy(e)) for e in found.args] if isinstance(found, FiniteSet) else []
            assert found is not Reals and len(ours) == len(real), (lhs, rhs, found, expected)
            assert all(any(abs(sympy.N(e - o, 40)) < sympy.Float('1e-30') for o in ours) for e in real), (lhs, rhs)
        else:
            # SymPy does not give a finite set: the solutions found are at least solutions
            assert isinstance(found, FiniteSet)
            assert all(_numerically_satisfied(lhs, rhs, sympy_expr(to_sympy(e))) for e in found.args)
        if isinstance(found, FiniteSet):
            assert all(_numerically_satisfied(lhs, rhs, sympy_expr(to_sympy(e))) for e in found.args), (lhs, rhs)


def test_several_radicals_and_absolute_values() -> None:
    two = solve_steps(sqrt(x + 5) + sqrt(x), 5, x)
    assert two.solutions() == finite_set(4)
    # both candidates of sqrt(2 x + 1) - sqrt(x) = 1 are solutions; sqrt(x + 1) + sqrt(x - 1) = 1 has none
    assert solve_steps(sqrt(2*x + 1) - sqrt(x), 1, x).solutions() == finite_set(0, 4)
    assert solve_steps(sqrt(x + 1) + sqrt(x - 1), 1, x).solutions() is EmptySet
    absolute = solve_steps(abs_(2*x + 1), x + 4, x)
    assert isinstance(absolute, AbsoluteValueRule) and absolute.kind == 'implication' and len(absolute.cases) == 2
    assert absolute.solutions() == finite_set(Rational(-5, 3), 3) and absolute.rejected == ()
    # |x - 2| = 2 x + 5: the case x - 2 = 2 x + 5 gives -7, where the right side is negative
    rejecting = solve_steps(abs_(x - 2), 2*x + 5, x)
    assert isinstance(rejecting, AbsoluteValueRule) and rejecting.rejected == (-7,) and rejecting.solutions() == finite_set(-1)
    assert solve_steps(abs_(x), -1, x).solutions() is EmptySet
    assert solve_steps(abs_(x**2 - 4), 3, x).solutions() == finite_set(-sqrt(Integer(7)), -1, 1, sqrt(Integer(7)))


def test_exponentials_and_logarithms() -> None:
    power = solve_steps(2**x, 8, x)
    assert isinstance(power, TakeLogarithmRule) and power.base == 2 and power.solutions() == finite_set(3)
    assert solve_steps(3*2**(x + 1), 48, x).solutions() == finite_set(3)
    assert solve_steps(exp(2*x), 5, x).solutions() == finite_set(log(Integer(5))/2)
    assert solve_steps(exp(x), -1, x).solutions() is EmptySet
    substitution = solve_steps(exp(2*x) - 3*exp(x) + 2, 0, x)
    assert isinstance(substitution, SubstitutionRule) and substitution.expression == exp(x) and substitution.kind == 'cases'
    assert substitution.substep.solutions() == finite_set(1, 2) and substitution.solutions() == finite_set(0, log(Integer(2)))
    # the arguments of the logarithms must be positive: 5 is kept, -2 is rejected
    combined = solve_steps(log(x) + log(x - 3), log(Integer(10)), x)
    assert combined.solutions() == finite_set(5) and combined.kind == 'implication'
    # x = 3 makes both arguments negative: equal complex logarithms are not a real solution ({3} before)
    assert solve_steps(log(x - 5), log(1 - x), x).solutions() is EmptySet
    assert solve_steps(log(x), 2, x).solutions() == finite_set(exp(Integer(2)))


def test_cubics_and_higher_degrees() -> None:
    cardano = solve_steps(x**3 + x + 1, 0, x)
    assert isinstance(cardano, CardanoRule) and (cardano.shift, cardano.p, cardano.q) == (0, 1, 1)
    [root] = cardano.solutions().args
    assert _numerically_satisfied(x**3 + x + 1, Integer(0), sympy_expr(to_sympy(root)))
    # three real roots: the formula would go through complex numbers, the roots are isolated instead
    three = solve_steps(x**3 - 3*x + 1, 0, x)
    assert isinstance(three, RealRootsRule) and len(three.roots) == 3
    assert len(solve_steps(x**4 - x - 1, 0, x).solutions().args) == 2
    assert solve_steps(x**4 + x**2 + 1, 0, x).solutions() is EmptySet
    # with branch=True both the formula and the isolated root are kept
    ways = solve_steps(x**3 + x + 1, 0, x, branch=True)
    assert [way.rule for way in ways.substeps()] == ['CardanoRule', 'RealRootsRule']


def test_trigonometric_equations() -> None:
    cases: list[tuple[Expr, Expr, int]] = [
        (sin(x), Rational(1, 2), 2), (cos(2*x), Integer(0), 2), (tan(x), Integer(1), 1), (2*cos(x) + 1, Integer(0), 2),
        (sin(x), Integer(1), 1), (cos(x), Integer(-1), 1), (sin(3*x - 1), Rational(1, 3), 2), (sin(x), cos(x), 1),
        (2*sin(x)**2 - sin(x) - 1, Integer(0), 3), (cos(x)**2, Rational(1, 4), 4), (tan(x)**2, Integer(3), 2),
    ]
    for lhs, rhs, families in cases:
        steps = solve_steps(lhs, rhs, x)
        assert not steps.contains_dont_know(), (lhs, rhs)
        found = to_sympy(steps.solutions())
        parts = list(found.args) if isinstance(found, sympy.Union) else [found]
        assert len(parts) == families, (lhs, rhs, found)
        values: list[sympy.Expr] = []
        for part in parts:
            assert isinstance(part, sympy.ImageSet)
            for k in range(-6, 7):
                value = part.lamda(k)
                assert _numerically_satisfied(lhs, rhs, value), (lhs, rhs, value)
                values.append(value)
        # completeness: the solutions in (-9, 9) are the sign changes and double zeros found by a fine scan
        inside = sorted({round(float(v), 9) for v in values if -9 < float(v) < 9})
        function = sympy.lambdify(sympy.Symbol('x'), sympy_expr(to_sympy(lhs)) - sympy_expr(to_sympy(rhs)), 'math')
        grid = [-9 + 18*i/40000 for i in range(40001)]
        samples = []
        for point in grid:
            try:
                samples.append(function(point))
            except (ValueError, ZeroDivisionError, OverflowError):
                samples.append(float('nan'))
        located = 0
        for i in range(1, len(samples) - 1):
            before, here, after = samples[i - 1], samples[i], samples[i + 1]
            if before != before or here != here or after != after or abs(here) > 1e3 or abs(before) > 1e3:
                continue
            crossing = before*here < 0 and abs(before - here) < 1
            touching = abs(here) < 1e-6 and abs(here) <= abs(before) and abs(here) < abs(after)
            located += crossing or (touching and before*after > 0)
        assert located == len(inside), (lhs, rhs, located, len(inside))
    assert solve_steps(sin(x), 2, x).solutions() is EmptySet and solve_steps(cos(x), -3, x).solutions() is EmptySet
    periodic = solve_steps(sin(x), Rational(1, 2), x)
    assert isinstance(periodic, PeriodicRule) and periodic.function == 'sin' and str(periodic.parameter) == 'n'
    assert periodic.parameter.is_integer and len(periodic.cases) == 2
