r"""Systems of equations step by step: substitution and elimination, as a
tree of rules.

:func:`system_steps` solves a system of equations in several unknowns
over the complex numbers (the default) or the real numbers
(``domain=Reals``) the way it is done by hand, on the engine of
:mod:`sympy_extras.steps.core` (generator rules which yield their
subproblems and return rule dataclasses). A problem is a tuple of
equations with its unknowns; the subproblems are smaller systems:

==========================  =================================================
``SingleEquationRule``      one equation in one unknown: its derivation
                            (:mod:`sympy_extras.steps.equations`)
``SubstitutionRule``        an equation is solved for an unknown, the
                            result is substituted in the others, and the
                            smaller system is solved
``EliminationRule``         multiples of two equations are added so that
                            an unknown disappears
``ConsistentRule``          no unknown is left and the equations hold
``InconsistentRule``        no unknown is left and an equation fails
``AlternativeRule``         several derivations
``DontKnowRule``            no rule applies
``InProgressRule``          the problem met again while being solved
==========================  =================================================

**Substitution.** The equation and the unknown are chosen so that the
equation is solved for it by :func:`~sympy_extras.steps.equations.solve_steps`
with finitely many solutions (one for an unknown which occurs linearly,
two for $y^2 = x$): for each, the smaller system is a case. The
solutions of the cases are completed with the value of the eliminated
unknown and *checked in every equation of the system*; the ones which
fail are the ``rejected`` field (an expression may have been obtained by
a step which is not an equivalence).

**Elimination.** For two equations linear in an unknown with constant
coefficients, the combination which cancels it replaces one of them.

By default the first way which works is kept, an unknown with the
coefficient $\pm 1$ being preferred for the substitution; with
``branch=True`` every choice of equation and unknown, and the
eliminations, are kept as alternatives.

``solutions()`` is the tuple of the solutions, each a tuple of values in
the order of the unknowns. Systems with infinitely many solutions are
not solved.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.systems import system_steps, manual_system
>>> x, y = Symbol('x'), Symbol('y')
>>> print(format_steps(system_steps([(x + y, 3), (x - y, 1)], [x, y]), depth=1))
SubstitutionRule: x + y = 3, x - y = 1 -> {(2, 1)}
  AddToBothSidesRule: x + y = 3 -> {3 - y}
  SingleEquationRule: 3 - 2*y = 1 -> {(1,)}
>>> manual_system([(x**2 + y**2, 5), (x + y, 1)], [x, y])
((2, -1), (-1, 2))
>>> manual_system([(x + y, 1), (2*x + 2*y, 3)], [x, y])
()

References
==========

.. [1] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
       (the architecture: generator rules returning rule dataclasses),
       SymPy master, 2026; SymPy issue 11088.
.. [2] I. M. Gelfand, A. Shen, *Algebra*, Birkhäuser, 1993 (systems of
       equations: substitution and elimination).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Optional, Sequence

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.fancysets import Complexes
from sympy_extras.sets.sets import EmptySetClass, FiniteSet, Reals, Set

from .core import AlternativeStep, Context, Step, StepSolver, atomic
from .equations import EquationStep, solve_steps

__all__ = ['SystemProblem', 'SystemStep', 'SingleEquationRule', 'SubstitutionRule', 'EliminationRule', 'ConsistentRule',
           'InconsistentRule', 'AlternativeRule', 'DontKnowRule', 'InProgressRule', 'system_steps', 'manual_system']

_Equation = tuple[Expr, Expr]
_Solution = tuple[Expr, ...]


@dataclass(frozen=True)
class SystemProblem:
    """The equations ``lhs = rhs`` in the unknowns ``variables``."""

    equations: tuple[_Equation, ...]
    variables: tuple[Symbol, ...]
    domain: Set = Complexes


@dataclass(frozen=True)
class SystemStep(Step):
    """A step of the solution of a system."""

    equations: tuple[_Equation, ...]
    variables: tuple[Symbol, ...]

    def solutions(self) -> tuple[_Solution, ...]:
        """The solutions: tuples of values in the order of the unknowns;
        ``ValueError`` when a part of the tree is not solved."""
        raise NotImplementedError

    def statement(self) -> str:
        system = ', '.join('%s = %s' % equation for equation in self.equations)
        if self.contains_dont_know():
            return '%s -> ?' % (system,)
        return '%s -> {%s}' % (system, ', '.join(str(solution) for solution in self.solutions()))


@dataclass(frozen=True)
class SingleEquationRule(SystemStep):
    """One equation in one unknown: ``step`` is its derivation."""

    step: EquationStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.step,)

    def solutions(self) -> tuple[_Solution, ...]:
        found = self.step.solutions()
        if isinstance(found, EmptySetClass):
            return ()
        if not isinstance(found, FiniteSet):
            raise ValueError('the equation has the solutions %s: not finitely many' % (found,))
        return tuple((value,) for value in found.args if isinstance(value, Expr))


@dataclass(frozen=True)
class ConsistentRule(SystemStep):
    """No unknown is left and every equation holds."""

    def solutions(self) -> tuple[_Solution, ...]:
        return ((),)


@dataclass(frozen=True)
class InconsistentRule(SystemStep):
    """No unknown is left and an equation does not hold."""

    def solutions(self) -> tuple[_Solution, ...]:
        return ()


@dataclass(frozen=True)
class SubstitutionRule(SystemStep):
    """The equation number ``index`` is solved for ``variable``
    (``isolation`` is that derivation): each of the ``expressions`` found
    is substituted in the other equations, and ``cases`` are the smaller
    systems, in the ``remaining`` unknowns. ``rejected`` are the
    solutions so obtained which do not satisfy the system."""

    index: int
    variable: Symbol
    isolation: EquationStep
    expressions: tuple[Expr, ...]
    remaining: tuple[Symbol, ...]
    cases: tuple[SystemStep, ...]
    rejected: tuple[_Solution, ...]

    def substeps(self) -> tuple[Step, ...]:
        return (self.isolation,) + self.cases

    def solutions(self) -> tuple[_Solution, ...]:
        found: list[_Solution] = []
        for expression, case in zip(self.expressions, self.cases):
            for partial in case.solutions():
                solution = _completed(self.variables, self.remaining, partial, self.variable, expression)
                if solution not in found and solution not in self.rejected:
                    found.append(solution)
        return tuple(found)


@dataclass(frozen=True)
class EliminationRule(SystemStep):
    """``first_multiplier`` times the equation ``first`` plus
    ``second_multiplier`` times the equation ``second`` has no
    ``variable``: it replaces the equation ``second`` (``substep`` solves
    the system so obtained, by a substitution of the equation which kept
    the unknown)."""

    first: int
    second: int
    first_multiplier: Expr
    second_multiplier: Expr
    variable: Symbol
    combined: _Equation
    substep: SystemStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def solutions(self) -> tuple[_Solution, ...]:
        return self.substep.solutions()


@dataclass(frozen=True)
class AlternativeRule(SystemStep, AlternativeStep):
    alternatives: tuple[SystemStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.alternatives

    def solutions(self) -> tuple[_Solution, ...]:
        return self.alternatives[0].solutions()


@dataclass(frozen=True)
class DontKnowRule(SystemStep):
    def contains_dont_know(self) -> bool:
        return True

    def solutions(self) -> tuple[_Solution, ...]:
        raise ValueError('the system is not solved')


@dataclass(frozen=True)
class InProgressRule(SystemStep):
    """The problem is met again while it is being solved (the derivation
    leads around in a loop): a placeholder for work in progress, where
    ``DontKnowRule`` says that no rule applies."""

    def contains_dont_know(self) -> bool:
        return True

    def solutions(self) -> tuple[_Solution, ...]:
        raise ValueError('the system is not solved')


# ---------------------------------------------------------------------
# helpers

_Run = Generator[SystemProblem, SystemStep, Optional[SystemStep]]
_Rule = Callable[[SystemProblem, Context], Generator[SystemProblem, SystemStep, Optional[SystemStep]]]


def _normal(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


def _completed(variables: Sequence[Symbol], remaining: Sequence[Symbol], partial: _Solution, variable: Symbol,
               expression: Expr) -> _Solution:
    """The solution of the whole system from one of the smaller system
    and the expression of the eliminated unknown."""
    value = expression
    for symbol, number in zip(remaining, partial):
        value = value.subs(symbol, number)
    values = dict(zip(remaining, partial))
    values[variable] = _normal(value)
    return tuple(values[v] for v in variables)


def _satisfies(equations: Sequence[_Equation], variables: Sequence[Symbol], solution: _Solution, real: bool
               ) -> Optional[bool]:
    """Whether the values satisfy every equation, when that is decided."""
    undecided = False
    for lhs, rhs in equations:
        left, right = lhs, rhs
        for symbol, value in zip(variables, solution):
            left, right = left.subs(symbol, value), right.subs(symbol, value)
        if left.is_finite is False or right.is_finite is False:
            return False
        if real and (left.is_real is False or right.is_real is False):
            return False
        difference = _normal(left - right)
        if difference == 0:
            continue
        if difference.is_zero is False or difference.is_positive is True or difference.is_negative is True:
            return False
        from sympy_extras.simplify.structure import is_zero
        try:
            verdict = is_zero(difference) if not difference.free_symbols else None
        except (ValueError, NotImplementedError, TypeError):
            verdict = None
        if verdict is False:
            return False
        if verdict is None:
            undecided = True
    return None if undecided else True


def _occurring(equations: Sequence[_Equation], variables: Sequence[Symbol]) -> list[Symbol]:
    return [v for v in variables if any(lhs.has(v) or rhs.has(v) for lhs, rhs in equations)]


def _linear_coefficient(equation: _Equation, variable: Symbol) -> Optional[Expr]:
    """The coefficient of the unknown when the equation is linear in it
    with a coefficient free of every symbol."""
    difference = expand(equation[0] - equation[1])
    if not difference.is_polynomial(variable):
        return None
    coefficient = difference.diff(variable)
    if coefficient.free_symbols or coefficient == 0:
        return None
    return coefficient


# ---------------------------------------------------------------------
# the rules

def _trivial(problem: SystemProblem) -> Optional[SystemStep]:
    """No unknown occurs any more (2 = 3 after a substitution: no
    solution, whatever unknowns are left)."""
    holds = _satisfies(problem.equations, (), (), problem.domain is Reals)
    if holds is True:
        # with unknowns left every value of them is a solution: infinitely many, not solved here
        return None if problem.variables else ConsistentRule(problem.equations, problem.variables)
    if holds is False:
        return InconsistentRule(problem.equations, problem.variables)
    return None


def _single(problem: SystemProblem) -> Optional[SystemStep]:
    (lhs, rhs), = problem.equations
    step = solve_steps(lhs, rhs, problem.variables[0], problem.domain)
    if step.contains_dont_know():
        return None
    found = step.solutions()
    if not isinstance(found, (FiniteSet, EmptySetClass)):
        return None
    return SingleEquationRule(problem.equations, problem.variables, step)


def _substitution(index: int, variable: Symbol) -> _Rule:
    def rule(problem: SystemProblem, context: Context) -> _Run:
        lhs, rhs = problem.equations[index]
        if not (lhs.has(variable) or rhs.has(variable)):
            return None
        isolation = solve_steps(lhs, rhs, variable, problem.domain)
        if isolation.contains_dont_know():
            return None
        found = isolation.solutions()
        if isinstance(found, EmptySetClass):
            # the equation has no solution in this unknown whatever the others are
            return InconsistentRule(problem.equations, problem.variables)
        if not isinstance(found, FiniteSet):
            return None
        expressions = tuple(e for e in found.args if isinstance(e, Expr))
        remaining = tuple(v for v in problem.variables if v != variable)
        others = [equation for position, equation in enumerate(problem.equations) if position != index]
        cases: list[SystemStep] = []
        for expression in expressions:
            substituted = tuple((expand(left.subs(variable, expression)), expand(right.subs(variable, expression)))
                                for left, right in others)
            cases.append((yield SystemProblem(substituted, remaining, problem.domain)))
        if any(case.contains_dont_know() for case in cases):
            return SubstitutionRule(problem.equations, problem.variables, index, variable, isolation, expressions, remaining,
                                    tuple(cases), ())
        rejected: list[_Solution] = []
        for expression, case in zip(expressions, cases):
            for partial in case.solutions():
                solution = _completed(problem.variables, remaining, partial, variable, expression)
                if _satisfies(problem.equations, problem.variables, solution, problem.domain is Reals) is False and solution not in rejected:
                    rejected.append(solution)
        return SubstitutionRule(problem.equations, problem.variables, index, variable, isolation, expressions, remaining,
                                tuple(cases), tuple(rejected))
    rule.__name__ = 'substitution_%d_%s' % (index, variable)
    return rule


def _elimination(first: int, second: int, variable: Symbol) -> _Rule:
    def rule(problem: SystemProblem, context: Context) -> _Run:
        a = _linear_coefficient(problem.equations[first], variable)
        b = _linear_coefficient(problem.equations[second], variable)
        if a is None or b is None:
            return None
        # b*(first) - a*(second) has no term in the unknown
        left = expand(b*problem.equations[first][0] - a*problem.equations[second][0])
        right = expand(b*problem.equations[first][1] - a*problem.equations[second][1])
        combined = (left, right)
        equations = tuple(combined if position == second else equation
                          for position, equation in enumerate(problem.equations))
        substep = yield SystemProblem(equations, problem.variables, problem.domain)
        return EliminationRule(problem.equations, problem.variables, first, second, b, -a, variable, combined, substep)
    rule.__name__ = 'elimination_%d_%d_%s' % (first, second, variable)
    return rule


def _candidates(problem: SystemProblem) -> Sequence[_Rule]:
    occurring = _occurring(problem.equations, problem.variables)
    if not occurring:
        return [atomic(_trivial)]
    if len(problem.variables) == 1 and len(problem.equations) == 1:
        return [atomic(_single)]
    if len(occurring) < len(problem.variables):
        return []                     # an unknown which occurs nowhere is free: infinitely many solutions or none
    choices: list[tuple[int, int, Symbol]] = []
    for index, equation in enumerate(problem.equations):
        for variable in problem.variables:
            if not (equation[0].has(variable) or equation[1].has(variable)):
                continue
            coefficient = _linear_coefficient(equation, variable)
            # an unknown with the coefficient 1 or -1 first, then the other linear ones, then the rest
            rank = 2 if coefficient is None else (0 if coefficient in (Integer(1), Integer(-1)) else 1)
            choices.append((rank, index, variable))
    choices.sort(key=lambda choice: (choice[0], choice[1], problem.variables.index(choice[2])))
    rules: list[_Rule] = [_substitution(index, variable) for _, index, variable in choices]
    for variable in problem.variables:
        linear = [index for index, equation in enumerate(problem.equations)
                  if _linear_coefficient(equation, variable) is not None]
        # an elimination only where the unknown would remain in one equation less
        if len(linear) >= 2 and len(problem.equations) >= 2:
            first, second = linear[0], linear[1]
            still = [index for index, equation in enumerate(problem.equations)
                     if index != second and (equation[0].has(variable) or equation[1].has(variable))]
            if len(still) == 1:
                rules.append(_elimination(first, second, variable))
    return rules


def _dont_know(problem: SystemProblem) -> SystemStep:
    return DontKnowRule(problem.equations, problem.variables)


def _in_progress(problem: SystemProblem) -> SystemStep:
    return InProgressRule(problem.equations, problem.variables)


def _alternative(problem: SystemProblem, alternatives: Sequence[SystemStep]) -> SystemStep:
    return AlternativeRule(problem.equations, problem.variables, tuple(alternatives))


def system_steps(equations: Sequence[tuple[ExprLike, ExprLike]], variables: Sequence[Symbol], domain: Set = Complexes,
                 branch: bool = False, max_depth: Optional[int] = 20) -> SystemStep:
    """The derivation of the solutions over the ``domain`` (``Complexes``,
    the default, or ``Reals``) of the system of equations ``(lhs, rhs)``
    in the unknowns ``variables``, as a tree of rules (see
    the module); the tree ``contains_dont_know()`` when the rules do not
    solve the system.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.systems import system_steps
    >>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    >>> steps = system_steps([(x + y + z, 6), (x - y, -1), (y + z, 5)], [x, y, z])
    >>> steps.rule, steps.variable, steps.expressions, steps.solutions()
    ('SubstitutionRule', x, (-y - z + 6,), ((1, 2, 3),))
    """
    if domain is not Reals and domain is not Complexes:
        raise ValueError('the domain is Reals or Complexes, not %s' % (domain,))
    problem = SystemProblem(tuple((to_expr(lhs), to_expr(rhs)) for lhs, rhs in equations), tuple(variables), domain)
    if len(set(problem.variables)) != len(problem.variables):
        raise ValueError('the unknowns are not distinct')
    solver: StepSolver[SystemProblem, SystemStep] = StepSolver(_candidates, _dont_know, _in_progress, _alternative, branch, max_depth)
    return solver.solve(problem)


def manual_system(equations: Sequence[tuple[ExprLike, ExprLike]], variables: Sequence[Symbol], domain: Set = Complexes
                  ) -> Optional[tuple[tuple[Expr, ...], ...]]:
    """The solutions found by the rules (tuples of values in the order
    of the unknowns, the empty tuple when there is none), or ``None``
    when the system is not solved.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.systems import manual_system
    >>> x, y = Symbol('x'), Symbol('y')
    >>> manual_system([(2*x + 3*y, 7), (4*x - y, 7)], [x, y]), manual_system([(y, x**2), (y, x + 2)], [x, y])
    (((2, 1),), ((-1, 1), (2, 4)))
    """
    steps = system_steps(equations, variables, domain)
    return None if steps.contains_dont_know() else steps.solutions()
