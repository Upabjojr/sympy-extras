"""Step-by-step derivatives: every step of every tree is checked against
the derivative computed by the core (compared by SymPy), on the textbook
examples and on random expressions; the alternatives agree; the solver's
depth limit, memory and loop protection."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Generator, Optional, Sequence

import sympy

from sympy_extras.core import Function, Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions import atan, cos, exp, log, sin, sqrt, tan
from sympy_extras.interop import sympy_expr, to_sympy
from sympy_extras.steps.core import AlternativeStep, Context, Step, StepSolver, atomic, count_steps, format_steps
from sympy_extras.steps.derivatives import (AlternativeRule, ChainRule, ConstantRule, DerivativeStep, DontKnowRule, PowerRule,
                                            ProductRule, QuotientRule, RewriteRule, SumRule, derivative_steps,
                                            manual_derivative)

x, a = Symbol('x'), Symbol('a')


def _same(first: Expr, second: Expr) -> bool:
    """Equal as functions: the difference vanishes at two points, to 30
    digits (SymPy's simplify does not identify x**x with exp(x*log(x)))."""
    difference = sympy_expr(to_sympy(first)) - sympy_expr(to_sympy(second))
    if difference == 0:
        return True
    sx, sa, sf = sympy.Symbol('x'), sympy.Symbol('a'), sympy.Function('f')
    for px, pa in ((sympy.Rational(7, 5), sympy.Rational(11, 7)), (sympy.Rational(5, 9), sympy.Rational(3, 2))):
        value = difference.subs(sf(sx), sympy.exp(sx/3)).doit().subs({sx: px, sa: pa})
        if abs(sympy.N(value, 40)) > sympy.Float('1e-30'):
            return False
    return True


def _check_tree(step: Step) -> None:
    """Every node states the derivative of its own expression."""
    assert isinstance(step, DerivativeStep)
    if not step.contains_dont_know():
        assert _same(step.result(), step.expression.diff(step.variable)), step
    for child in step.substeps():
        _check_tree(child)


def test_textbook_examples() -> None:
    # Spivak, Chapter 10
    cases: list[tuple[Expr, Expr]] = [
        (x**5 - 3*x**2 + 7, 5*x**4 - 6*x),
        (x*sin(x), sin(x) + x*cos(x)),
        (sin(x)/x, (x*cos(x) - sin(x))/x**2),
        (sin(x**2), 2*x*cos(x**2)),
        (sin(cos(x)), -sin(x)*cos(cos(x))),
        (sqrt(1 + x**2), x/sqrt(1 + x**2)),
        (exp(-x**2), -2*x*exp(-x**2)),
        (log(log(x)), 1/(x*log(x))),
        (x**x, x**x*(log(x) + 1)),
        (2**x, 2**x*log(2)),
        (atan(1/x), -1/(1 + x**2)),
        (tan(x), 1 + tan(x)**2),
        (a*x**2 + a**2, 2*a*x),
    ]
    for f, expected in cases:
        steps = derivative_steps(f, x)
        assert not steps.contains_dont_know() and _same(steps.result(), expected), f
        assert manual_derivative(f, x) == steps.result()
        _check_tree(steps)


def test_rule_trees() -> None:
    assert derivative_steps(Integer(7), x) == ConstantRule(Integer(7), x)
    assert derivative_steps(x**3, x) == PowerRule(x**3, x, Integer(3))
    assert isinstance(derivative_steps(x**2 + sin(x), x), SumRule)
    product = derivative_steps(x*sin(x), x)
    assert isinstance(product, ProductRule) and product.first == x and product.second == sin(x)
    assert isinstance(derivative_steps(sin(x)/x, x), QuotientRule)
    chain = derivative_steps(sin(x**2), x)
    assert isinstance(chain, ChainRule) and chain.inner == x**2 and str(chain.placeholder) == 'u'
    assert chain.outer_step.variable == chain.placeholder and chain.inner_step == PowerRule(x**2, x, Integer(2))
    # the placeholder avoids the names in use
    u = Symbol('u')
    other = derivative_steps(sin(u*x), x)
    assert isinstance(other, ChainRule) and str(other.placeholder) == 'u1'
    power = derivative_steps(x**x, x)
    assert isinstance(power, RewriteRule) and power.rewritten == exp(x*log(x))
    assert [step.rule for step in product.substeps()] == ['VariableRule', 'FunctionRule']


def _random_expression(rnd: random.Random, depth: int) -> Expr:
    if depth == 0:
        return rnd.choice([x, x, Integer(2), a, x**2, Rational(1, 3)])
    kind = rnd.randint(0, 6)
    first, second = _random_expression(rnd, depth - 1), _random_expression(rnd, depth - 1)
    if kind == 0:
        return first + second
    if kind == 1:
        return first*second
    if kind == 2:
        return first/(second**2 + 1)
    if kind == 3:
        return rnd.choice([sin, cos, exp, atan])(first)
    if kind == 4:
        return (first**2 + 1)**Rational(rnd.randint(1, 5), 2)
    if kind == 5:
        return log(first**2 + 2)
    return (second**2 + 1)**first


def test_random_expressions() -> None:
    rnd = random.Random(181)
    for _ in range(40):
        f = _random_expression(rnd, rnd.randint(1, 3))
        steps = derivative_steps(f, x)
        assert not steps.contains_dont_know(), f
        assert _same(steps.result(), f.diff(x)), f
    for _ in range(8):
        _check_tree(derivative_steps(_random_expression(rnd, 2), x))


def test_alternatives() -> None:
    both = derivative_steps(sin(x)/x, x, branch=True)
    assert isinstance(both, AlternativeRule) and isinstance(both, AlternativeStep)
    assert [alternative.rule for alternative in both.alternatives] == ['QuotientRule', 'ProductRule']
    assert all(_same(alternative.result(), both.result()) for alternative in both.alternatives)
    _check_tree(both)
    # one way only: no alternative node; the default keeps the first way
    assert derivative_steps(x*sin(x), x, branch=True) == derivative_steps(x*sin(x), x)
    assert derivative_steps(sin(x)/x, x) == both.alternatives[0]
    # alternatives inside a tree: the quotient is an argument
    nested = derivative_steps(exp(x/(x + 1)), x, branch=True)
    assert isinstance(nested, ChainRule) and isinstance(nested.inner_step, AlternativeRule)
    assert count_steps(nested) > count_steps(derivative_steps(exp(x/(x + 1)), x))


def test_rendering_is_collapsible() -> None:
    steps = derivative_steps(x**2*sin(3*x), x)
    full = format_steps(steps).splitlines()
    assert full[0].startswith('ProductRule: d/dx(x**2*sin(3*x)) = ') and len(full) == count_steps(steps) == 6
    assert format_steps(steps, depth=0).splitlines() == full[:1]
    assert format_steps(steps, depth=1).splitlines() == full[:3]
    assert full[3] == '    FunctionRule: d/du(sin(u)) = cos(u)'
    # a rewriting is a minor step: hidden by default, its substep takes its place
    power = derivative_steps(x**x, x)
    shown = format_steps(power).splitlines()
    assert shown[0].startswith('ChainRule: ') and format_steps(power, minor=True).splitlines()[0].startswith('RewriteRule: ')
    assert len(format_steps(power, minor=True).splitlines()) == len(shown) + 1
    both = format_steps(derivative_steps(sin(x)/x, x, branch=True)).splitlines()
    assert both[1].startswith('  [1] QuotientRule: ') and any(line.startswith('  [2] ProductRule: ') for line in both)


def test_unknown_functions_and_depth() -> None:
    f = Function('f')
    unknown = derivative_steps(x*f(x), x)
    assert unknown.contains_dont_know() and isinstance(unknown, ProductRule)
    assert isinstance(unknown.second_step, DontKnowRule)
    assert _same(unknown.result(), (x*f(x)).diff(x))
    # the depth limit leaves the deeper subproblems undone, and the limited result is not remembered
    shallow = derivative_steps(sin(sin(sin(x))), x, max_depth=2)
    assert shallow.contains_dont_know() and not derivative_steps(sin(sin(sin(x))), x).contains_dont_know()
    assert derivative_steps(sin(x), x, max_depth=1).rule == 'FunctionRule'


class _Counter:
    def __init__(self) -> None:
        self.calls = 0


def test_solver_memory_and_loops() -> None:
    # a toy domain: the problems are integers, n needs n - 1 twice; 0 is atomic; -1 leads back to itself

    @dataclass(frozen=True)
    class Leaf(Step):
        value: int

        def statement(self) -> str:
            return str(self.value)

    @dataclass(frozen=True)
    class Node(Step):
        value: int
        left: Step
        right: Step

        def substeps(self) -> tuple[Step, ...]:
            return (self.left, self.right)

        def statement(self) -> str:
            return str(self.value)

    @dataclass(frozen=True)
    class Unknown(Step):
        value: int

        def contains_dont_know(self) -> bool:
            return True

        def statement(self) -> str:
            return '?'

    counter = _Counter()

    def leaf(problem: int) -> Optional[Step]:
        return Leaf(problem)

    def node(problem: int, context: Context) -> Generator[int, Step, Optional[Step]]:
        counter.calls += 1
        # the problem -1 asks for itself: a loop
        left = yield (problem if problem < 0 else problem - 1)
        right = yield (0 if problem < 0 else problem - 1)
        return Node(problem, left, right)

    def candidates(problem: int) -> Sequence[Callable[[int, Context], Generator[int, Step, Optional[Step]]]]:
        return [atomic(leaf)] if problem == 0 else [node]

    def first(problem: int, found: Sequence[Step]) -> Step:
        return found[0]

    solver: StepSolver[int, Step] = StepSolver(candidates, Unknown, first)
    tree = solver.solve(12)
    # 2**12 leaves in the tree, but each subproblem is solved once
    assert count_steps(tree) == 2**13 - 1 and counter.calls == 12
    again: StepSolver[int, Step] = StepSolver(candidates, Unknown, first)
    looping = again.solve(-1)
    assert looping.contains_dont_know() and isinstance(looping, Node) and isinstance(looping.left, Unknown)
