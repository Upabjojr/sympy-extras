r"""Inequalities step by step: the solution set of an inequality in one
unknown over the real numbers, as a tree of rules.

:func:`inequality_steps` works like
:func:`sympy_extras.steps.equations.solve_steps`, on the same engine
(generator rules which yield their subproblems and return rule
dataclasses):

==========================  =================================================
``SolvedRule``              $x < c$ (or $\le$, $>$, $\ge$): an interval
``TrueRule``                an inequality between numbers which holds
``FalseRule``               an inequality between numbers which does not
``AddToBothSidesRule``      the same term added to both sides
``DivideBothSidesRule``     both sides divided by a number, the inequality
                            reversed when it is negative (``flipped``)
``SignChartRule``           $N(x)/D(x) \gtrless 0$ from the signs between
                            the zeros and the poles
``AbsoluteValueRule``       $|A|$ replaced by $A$ where $A \ge 0$ and by
                            $-A$ where $A < 0$
``AlternativeRule``         several derivations
``DontKnowRule``            no rule applies
==========================  =================================================

**The sign chart.** A rational function with rational coefficients is
factored into irreducible factors. Its real zeros and poles are the real
roots of the factors; two different factors have no common root and an
irreducible factor has only simple roots, so the sign changes at a point
exactly when the factor of that point has an odd exponent (numerator and
denominator together). The sign to the right of all the points is that
of the leading coefficients, and the others follow from right to left:
no test point is evaluated. The points are ordered exactly, as real
algebraic numbers. The step records the factors, the points, which of
them are poles, the sign on each interval, and the derivations of the
roots of the factors (trees of :mod:`sympy_extras.steps.equations`),
which are its substeps.

The solution is a union of intervals; the zeros belong to it for
$\le$ and $\ge$, the poles never.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import abs_
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.inequalities import inequality_steps, manual_inequality
>>> x = Symbol('x')
>>> print(format_steps(inequality_steps(3 - 2*x < 7, x)))
AddToBothSidesRule: 3 - 2*x < 7 -> Interval.open(-2, oo)
  DivideBothSidesRule: -2*x < 4 -> Interval.open(-2, oo)
    SolvedRule: x > -2 -> Interval.open(-2, oo)
>>> chart = inequality_steps(x**2 - 5*x + 6 > 0, x)
>>> chart.rule, chart.points, chart.signs, chart.solutions()
('SignChartRule', (2, 3), (1, -1, 1), Union(Interval.open(-oo, 2), Interval.open(3, oo)))
>>> manual_inequality((x - 1)/(x + 2) <= 0, x), manual_inequality(abs_(x - 1) < 3, x)
(Interval.Lopen(-2, 1), Interval.open(-2, 4))

References
==========

.. [1] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
       (the architecture: generator rules returning rule dataclasses),
       SymPy master, 2026; SymPy issue 11088.
.. [2] I. M. Gelfand, A. Shen, *Algebra*, Birkhäuser, 1993 (inequalities
       by the method of intervals).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cmp_to_key
from typing import Callable, Generator, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean
from sympy_extras.core.numbers import Integer, Rational, oo
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import Ge, Gt, Le, Lt, Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.sets.sets import EmptySet, Reals, Set, finite_set, intersection, interval, union

from .core import AlternativeStep, Context, Step, StepSolver, atomic
from .equations import EquationStep, solve_steps

__all__ = ['InequalityProblem', 'InequalityStep', 'SolvedRule', 'TrueRule', 'FalseRule', 'AddToBothSidesRule',
           'DivideBothSidesRule', 'SignChartRule', 'AbsoluteValueRule', 'AlternativeRule', 'DontKnowRule',
           'inequality_steps', 'manual_inequality']

_REVERSED: dict[str, str] = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}


@dataclass(frozen=True)
class InequalityProblem:
    """``lhs relation rhs`` in ``variable``; the relation is ``'<'``,
    ``'<='``, ``'>'`` or ``'>='``."""

    lhs: Expr
    rhs: Expr
    relation: str
    variable: Symbol


@dataclass(frozen=True)
class InequalityStep(Step):
    """A step of the solution of ``lhs relation rhs`` in ``variable``."""

    lhs: Expr
    rhs: Expr
    relation: str
    variable: Symbol

    def solutions(self) -> Set:
        """The set of the real solutions; ``ValueError`` when a part of
        the tree is not solved."""
        raise NotImplementedError

    def statement(self) -> str:
        found = '?' if self.contains_dont_know() else str(self.solutions())
        return '%s %s %s -> %s' % (self.lhs, self.relation, self.rhs, found)


@dataclass(frozen=True)
class SolvedRule(InequalityStep):
    """``variable relation bound``."""

    bound: Expr

    def solutions(self) -> Set:
        if self.relation == '<':
            return interval(-oo, self.bound, True, True)
        if self.relation == '<=':
            return interval(-oo, self.bound, True, False)
        if self.relation == '>':
            return interval(self.bound, oo, True, True)
        return interval(self.bound, oo, False, True)


@dataclass(frozen=True)
class TrueRule(InequalityStep):
    def solutions(self) -> Set:
        return Reals


@dataclass(frozen=True)
class FalseRule(InequalityStep):
    def solutions(self) -> Set:
        return EmptySet


@dataclass(frozen=True)
class AddToBothSidesRule(InequalityStep):
    substep: InequalityStep
    term: Expr

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def solutions(self) -> Set:
        return self.substep.solutions()


@dataclass(frozen=True)
class DivideBothSidesRule(InequalityStep):
    """Both sides divided by ``divisor``; ``flipped`` when it is negative
    and the inequality is reversed."""

    substep: InequalityStep
    divisor: Expr
    flipped: bool

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def solutions(self) -> Set:
        return self.substep.solutions()


@dataclass(frozen=True)
class SignChartRule(InequalityStep):
    """The left side is a rational function compared with zero:
    ``numerator_factors`` and ``denominator_factors`` are its irreducible
    factors with their exponents, ``points`` its real zeros and poles in
    increasing order, ``poles`` those which are poles, and ``signs`` the
    sign on each of the ``len(points) + 1`` intervals they cut, from left
    to right. ``root_steps`` are the derivations of the roots of the
    factors."""

    numerator_factors: tuple[tuple[Expr, int], ...]
    denominator_factors: tuple[tuple[Expr, int], ...]
    points: tuple[Expr, ...]
    poles: tuple[Expr, ...]
    signs: tuple[int, ...]
    root_steps: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.root_steps

    def solutions(self) -> Set:
        wanted = 1 if self.relation in ('>', '>=') else -1
        if not self.points:
            return Reals if self.signs[0] == wanted else EmptySet
        pieces: list[Set] = []
        bounds: list[Expr] = [-oo] + list(self.points) + [oo]
        for position, sign in enumerate(self.signs):
            if sign == wanted:
                pieces.append(interval(bounds[position], bounds[position + 1], True, True))
        if self.relation in ('<=', '>='):
            zeros = [point for point in self.points if point not in self.poles]
            if zeros:
                pieces.append(finite_set(*zeros))
        return union(*pieces) if pieces else EmptySet


@dataclass(frozen=True)
class AbsoluteValueRule(InequalityStep):
    """``|inner|`` is ``inner`` where ``inner >= 0`` and ``-inner`` where
    ``inner < 0``: ``cases`` are the four inequalities ``inner >= 0``, the
    inequality with ``inner``, ``inner < 0``, the inequality with
    ``-inner``; the solutions are those of the first two together, or of
    the last two together."""

    inner: Expr
    cases: tuple[InequalityStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.cases

    def solutions(self) -> Set:
        where, first, elsewhere, second = (case.solutions() for case in self.cases)
        return union(intersection(where, first), intersection(elsewhere, second))


@dataclass(frozen=True)
class AlternativeRule(InequalityStep, AlternativeStep):
    alternatives: tuple[InequalityStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.alternatives

    def solutions(self) -> Set:
        return self.alternatives[0].solutions()


@dataclass(frozen=True)
class DontKnowRule(InequalityStep):
    def contains_dont_know(self) -> bool:
        return True

    def solutions(self) -> Set:
        raise ValueError('the inequality %s %s %s is not solved' % (self.lhs, self.relation, self.rhs))


# ---------------------------------------------------------------------
# the rules

_Run = Generator[InequalityProblem, InequalityStep, Optional[InequalityStep]]
_Rule = Callable[[InequalityProblem, Context], Generator[InequalityProblem, InequalityStep, Optional[InequalityStep]]]


def _normal(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


def _sign(e: Expr) -> Optional[int]:
    if e == 0:
        return 0
    if e.is_positive is True:
        return 1
    if e.is_negative is True:
        return -1
    return None


def _holds(sign: int, relation: str) -> bool:
    if relation == '<':
        return sign < 0
    if relation == '<=':
        return sign <= 0
    if relation == '>':
        return sign > 0
    return sign >= 0


def _trivial(problem: InequalityProblem) -> Optional[InequalityStep]:
    sign = _sign(_normal(problem.lhs - problem.rhs))
    if sign is None:
        return None
    if _holds(sign, problem.relation):
        return TrueRule(problem.lhs, problem.rhs, problem.relation, problem.variable)
    return FalseRule(problem.lhs, problem.rhs, problem.relation, problem.variable)


def _solved(problem: InequalityProblem) -> Optional[InequalityStep]:
    return SolvedRule(problem.lhs, problem.rhs, problem.relation, problem.variable, problem.rhs)


def _split(e: Expr, x: Symbol) -> tuple[Expr, Expr]:
    terms = [a for a in e.args if isinstance(a, Expr)] if isinstance(e, Add) else [e]
    return add(*[t for t in terms if t.has(x)]), add(*[t for t in terms if not t.has(x)])


def _added(problem: InequalityProblem, term: Expr) -> InequalityProblem:
    return InequalityProblem(expand(problem.lhs + term), expand(problem.rhs + term), problem.relation, problem.variable)


def _move_unknowns(problem: InequalityProblem, context: Context) -> _Run:
    moving, _ = _split(problem.rhs, problem.variable)
    if moving == 0:
        return None
    substep = yield _added(problem, -moving)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.relation, problem.variable, substep, -moving)


def _move_constants(problem: InequalityProblem, context: Context) -> _Run:
    _, moving = _split(problem.lhs, problem.variable)
    if moving == 0:
        return None
    substep = yield _added(problem, -moving)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.relation, problem.variable, substep, -moving)


def _divide(problem: InequalityProblem, context: Context) -> _Run:
    x = problem.variable
    if isinstance(problem.lhs, Add):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    sign = _sign(coefficient)
    if rest != x or coefficient == 1 or sign is None or sign == 0 or problem.rhs.has(x):
        return None
    relation = problem.relation if sign > 0 else _REVERSED[problem.relation]
    substep = yield InequalityProblem(x, _normal(problem.rhs/coefficient), relation, x)
    return DivideBothSidesRule(problem.lhs, problem.rhs, problem.relation, x, substep, coefficient, sign < 0)


def _expand_sides(problem: InequalityProblem, context: Context) -> _Run:
    lhs, rhs = expand(problem.lhs), expand(problem.rhs)
    if (lhs, rhs) == (problem.lhs, problem.rhs):
        return None
    substep = yield InequalityProblem(lhs, rhs, problem.relation, problem.variable)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.relation, problem.variable, substep, Integer(0))


def _standard_form(problem: InequalityProblem, context: Context) -> _Run:
    """Everything on the left (nothing is multiplied out: the sign of a
    denominator is not known)."""
    if problem.rhs == 0:
        return None
    substep = yield InequalityProblem(problem.lhs - problem.rhs, Integer(0), problem.relation, problem.variable)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.relation, problem.variable, substep, -problem.rhs)


def _rational_coefficients(e: Expr, x: Symbol) -> Optional[list[Fraction]]:
    from sympy_extras.matrices.eigen import _coefficients
    expanded = expand(e)
    if expanded == 0 or not expanded.is_polynomial(x):
        return None
    values: list[Fraction] = []
    for c in _coefficients(expanded, x):
        if not isinstance(c, Rational):
            return None
        values.append(Fraction(c.p, c.q))
    return values


def _factor_roots(coefficients: Sequence[Fraction], x: Symbol) -> list[tuple[Algebraic, Expr]]:
    """The real roots of an irreducible factor: the exact number and the
    expression which shows it."""
    from sympy_extras.core.numbers import number_from_fraction
    from sympy_extras.polys.cad.indexedroot import real_root
    from sympy_extras.polys.cad.samplepoints import real_roots_of
    algebraic = real_roots_of(list(coefficients))
    shown: list[Expr]
    if len(coefficients) == 2:
        shown = [number_from_fraction(-coefficients[1]/coefficients[0])]
    elif len(coefficients) == 3 and len(algebraic) == 2:
        a, b, c = (number_from_fraction(v) for v in coefficients)
        root = sqrt(b**2 - 4*a*c)
        pair = [(-b - root)/(2*a), (-b + root)/(2*a)]
        shown = pair if coefficients[0] > 0 else pair[::-1]
    else:
        shown = [real_root(r, x.name) for r in algebraic]
    return list(zip(algebraic, shown))


_Located = tuple[Algebraic, Expr, int, bool]            # (the number, how it is shown, the exponent, a pole)


def _by_value(first: _Located, second: _Located) -> int:
    from sympy_extras.polys.cad.samplepoints import compare_real
    return compare_real(first[0], second[0])


def _sign_chart(problem: InequalityProblem, context: Context) -> _Run:
    from sympy_extras.polys.polyerrors import PolynomialError
    from sympy_extras.polys.polytools import factor_list
    x = problem.variable
    if problem.rhs != 0:
        return None
    numerator, denominator = problem.lhs.together().as_numer_denom()
    numerator, denominator = expand(numerator), expand(denominator)
    if _rational_coefficients(numerator, x) is None or _rational_coefficients(denominator, x) is None:
        return None
    try:
        top_constant, top = factor_list(numerator, x)
        bottom_constant, bottom = factor_list(denominator, x)
    except (PolynomialError, NotImplementedError, ValueError):
        return None
    lead = _sign(top_constant)
    other = _sign(bottom_constant)
    if lead is None or other is None or lead == 0 or other == 0:
        return None
    sign = lead*other
    exponents: dict[Expr, int] = {}
    in_denominator: set[Expr] = set()
    numerator_factors: list[tuple[Expr, int]] = []
    denominator_factors: list[tuple[Expr, int]] = []
    for listed, target, below in ((top, numerator_factors, False), (bottom, denominator_factors, True)):
        for factor, multiplicity in listed:
            if not isinstance(multiplicity, Integer) or not factor.has(x):
                continue
            target.append((factor, multiplicity.p))
            exponents[factor] = exponents.get(factor, 0) + multiplicity.p
            if below:
                in_denominator.add(factor)
    located: list[_Located] = []
    root_steps: list[EquationStep] = []
    for factor, exponent in exponents.items():
        coefficients = _rational_coefficients(factor, x)
        if coefficients is None:
            return None
        if coefficients[0] < 0 and exponent % 2:
            sign = -sign
        roots = _factor_roots(coefficients, x)
        if roots:
            root_steps.append(solve_steps(factor, Integer(0), x))
        for number, shown in roots:
            located.append((number, shown, exponent, factor in in_denominator))
    located.sort(key=cmp_to_key(_by_value))
    # from the right, where the sign is that of the leading coefficients, to the left
    signs = [sign]
    for _, _, exponent, _ in reversed(located):
        signs.append(-signs[-1] if exponent % 2 else signs[-1])
    signs.reverse()
    points = tuple(shown for _, shown, _, _ in located)
    poles = tuple(shown for _, shown, _, pole in located if pole)
    return SignChartRule(problem.lhs, problem.rhs, problem.relation, x, tuple(numerator_factors), tuple(denominator_factors),
                         points, poles, tuple(signs), tuple(root_steps))
    yield problem               # never reached: the rule has no subproblem of its own kind


def _absolute_value(problem: InequalityProblem, context: Context) -> _Run:
    x = problem.variable
    nodes = sorted((node for e in (problem.lhs, problem.rhs) for node in e.atoms(Abs) if isinstance(node, Expr) and node.has(x)),
                   key=str)
    if not nodes:
        return None
    node = nodes[0]
    inner = node.args[0]
    if not isinstance(inner, Expr):
        return None
    cases: list[InequalityStep] = []
    for relation, replacement in (('>=', inner), ('<', -inner)):
        cases.append((yield InequalityProblem(inner, Integer(0), relation, x)))
        # subs, which evaluates: x + |x - 1| becomes 2*x - 1, not the unevaluated x + (x - 1)
        cases.append((yield InequalityProblem(problem.lhs.subs(node, replacement), problem.rhs.subs(node, replacement),
                                              problem.relation, x)))
    return AbsoluteValueRule(problem.lhs, problem.rhs, problem.relation, x, inner, tuple(cases))


def _candidates(problem: InequalityProblem) -> Sequence[_Rule]:
    lhs, rhs, x = problem.lhs, problem.rhs, problem.variable
    if not lhs.has(x) and not rhs.has(x):
        return [atomic(_trivial)]
    if any(node.has(x) for e in (lhs, rhs) for node in e.atoms(Abs)):
        return [_absolute_value]
    if lhs == x and not rhs.has(x):
        return [atomic(_solved)]
    difference = expand(lhs - rhs)
    if difference.is_polynomial(x):
        coefficients = _rational_coefficients(difference, x)
        if not difference.has(x):
            return [atomic(_trivial)]
        from sympy_extras.matrices.eigen import _coefficients
        degree = len(_coefficients(difference, x)) - 1 if difference != 0 else 0
        if degree == 1:
            return [_expand_sides, _move_unknowns, _move_constants, _divide]
        if coefficients is None:
            return []
    return [_standard_form, _sign_chart]


def _dont_know(problem: InequalityProblem) -> InequalityStep:
    return DontKnowRule(problem.lhs, problem.rhs, problem.relation, problem.variable)


def _alternative(problem: InequalityProblem, alternatives: Sequence[InequalityStep]) -> InequalityStep:
    return AlternativeRule(problem.lhs, problem.rhs, problem.relation, problem.variable, tuple(alternatives))


def _problem(inequality: Boolean, x: Symbol) -> InequalityProblem:
    relation: str
    if not isinstance(inequality, Relational):
        # x <= x is already True for Python's operators: build the node (Le(x, x)) to see its steps
        raise TypeError('%s is not an inequality (a comparison which was evaluated?)' % (inequality,))
    if isinstance(inequality, Lt):
        relation = '<'
    elif isinstance(inequality, Le):
        relation = '<='
    elif isinstance(inequality, Gt):
        relation = '>'
    elif isinstance(inequality, Ge):
        relation = '>='
    else:
        raise TypeError('%s is not an inequality' % (inequality,))
    lhs, rhs = inequality.lhs, inequality.rhs
    if not isinstance(lhs, Expr) or not isinstance(rhs, Expr):
        raise TypeError('%s is not an inequality between expressions' % (inequality,))
    return InequalityProblem(lhs, rhs, relation, x)


def inequality_steps(inequality: Boolean, x: Symbol, branch: bool = False, max_depth: Optional[int] = 30) -> InequalityStep:
    """The derivation of the real solutions of an inequality (``<``,
    ``<=``, ``>`` or ``>=`` between expressions) in ``x``, as a tree of
    rules (see the module); the tree ``contains_dont_know()`` when the
    rules do not solve it.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.core import format_steps
    >>> from sympy_extras.steps.inequalities import inequality_steps
    >>> x = Symbol('x')
    >>> print(format_steps(inequality_steps((x - 1)/(x + 2) <= 0, x)))
    SignChartRule: (x - 1)/(x + 2) <= 0 -> Interval.Lopen(-2, 1)
      AddToBothSidesRule: x - 1 = 0 -> {1}
        SolvedRule: x = 1 -> {1}
      AddToBothSidesRule: x + 2 = 0 -> {-2}
        SolvedRule: x = -2 -> {-2}
    """
    solver: StepSolver[InequalityProblem, InequalityStep] = StepSolver(_candidates, _dont_know, _alternative, branch,
                                                                       max_depth)
    return solver.solve(_problem(inequality, x))


def manual_inequality(inequality: Boolean, x: Symbol) -> Optional[Set]:
    """The set of the real solutions of the inequality found by the
    rules, or ``None``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.inequalities import manual_inequality
    >>> x = Symbol('x')
    >>> manual_inequality(x**2 < 4, x), manual_inequality(x**2 + 1 > 0, x), manual_inequality(x**3 - x >= 0, x)
    (Interval.open(-2, 2), Reals, Union(Interval(-1, 0), Interval(1, oo)))
    """
    steps = inequality_steps(inequality, x)
    return None if steps.contains_dont_know() else steps.solutions()
