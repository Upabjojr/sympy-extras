r"""Equations step by step: the solution of an equation in one unknown
over the real numbers as a tree of the operations done by hand.

:func:`solve_steps` returns the derivation as a tree of rule
dataclasses, built by generator rules which the solver of
:mod:`sympy_extras.steps.core` drives (the architecture of SymPy's
``manualintegrate`` [1]_, applied to equations). A problem is an
equation ``lhs = rhs`` in a variable; a rule turns it into one simpler
equation, or into several, and yields them as subproblems:

===========================  ================================================
``SolvedRule``               $x = c$
``IdentityRule``             an equation true for every $x$
``ContradictionRule``        an equation true for no $x$
``AddToBothSidesRule``       the same term added to both sides
``DivideBothSidesRule``      both sides divided by a nonzero constant
``RewriteRule``              the sides expanded and collected
``ZeroProductRule``          a product is zero: one equation per factor
``QuadraticFormulaRule``     $a x^2 + b x + c = 0$ by the formula
``CompleteSquareRule``       $a x^2 + b x + c = 0$ as $(x + p)^2 = q$
``RootRule``                 $u^n = q$: the real $n$-th roots
``ClearDenominatorsRule``    both sides multiplied by the denominators
``SquareBothSidesRule``      a square root isolated and both sides squared
``SwapSidesRule``            the sides exchanged
``AlternativeRule``          several derivations of the same solution
``DontKnowRule``             no rule applies
===========================  ================================================

**What a step does to the solutions.** Each rule has a ``kind``:

* ``'equivalence'``: the new equation has the same solutions (a term
  added to both sides, a division by a nonzero number);
* ``'cases'``: the solutions are the union of those of the substeps (a
  zero product, the two square roots);
* ``'implication'``: the new equation follows from the old one but may
  have more solutions (multiplying by a denominator, squaring). The step
  then *checks* every solution of the new equation in the original one;
  the ones which fail are its ``rejected`` field, and the ones whose
  check could not be decided its ``unverified`` field.

**Alternatives.** A quadratic equation is solved by factoring when it
factors over the rationals, by the formula, and by completing the
square: the first which applies by default, all of them with
``branch=True``. A linear equation may move its constants or its
unknowns first.

``solutions()`` gives the set of the real solutions of a step, assembled
from its substeps. With symbols other than the unknown the solution is
the generic one (a coefficient by which a side is divided is taken to be
nonzero).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sqrt
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.equations import solve_steps, manual_solve
>>> x = Symbol('x')
>>> print(format_steps(solve_steps(2*x + 3, 7, x)))
AddToBothSidesRule: 2*x + 3 = 7 -> {2}
  DivideBothSidesRule: 2*x = 4 -> {2}
    SolvedRule: x = 2 -> {2}
>>> print(format_steps(solve_steps(x**2 - 5*x + 6, 0, x)))
ZeroProductRule: x**2 - 5*x + 6 = 0 -> {2, 3}
  AddToBothSidesRule: x - 3 = 0 -> {3}
    SolvedRule: x = 3 -> {3}
  AddToBothSidesRule: x - 2 = 0 -> {2}
    SolvedRule: x = 2 -> {2}
>>> manual_solve(x**2 - 2*x - 1, 0, x), manual_solve(sqrt(x + 2), x, x)
({1 - sqrt(2), 1 + sqrt(2)}, {2})
>>> [way.rule for way in solve_steps(x**2 - 5*x + 6, 0, x, branch=True).substeps()]
['ZeroProductRule', 'QuadraticFormulaRule', 'CompleteSquareRule']

References
==========

.. [1] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
       (rules as generators returning rule dataclasses, driven by a
       solver), SymPy master, 2026; SymPy issue 11088, *Step by Step
       solutions*.
.. [2] I. M. Gelfand, A. Shen, *Algebra*, Birkhäuser, 1993 (linear and
       quadratic equations, completing the square, equations with
       radicals and their extraneous roots).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Generator, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational, nan, zoo
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.sets import EmptySet, FiniteSet, Reals, Set, finite_set, union

from .core import AlternativeStep, Context, Step, StepSolver, atomic

__all__ = ['EquationProblem', 'EquationStep', 'SolvedRule', 'IdentityRule', 'ContradictionRule', 'AddToBothSidesRule',
           'DivideBothSidesRule', 'RewriteRule', 'ZeroProductRule', 'QuadraticFormulaRule', 'CompleteSquareRule', 'RootRule',
           'ClearDenominatorsRule', 'SquareBothSidesRule', 'SwapSidesRule', 'AlternativeRule', 'DontKnowRule',
           'solve_steps', 'manual_solve']


@dataclass(frozen=True)
class EquationProblem:
    """The equation ``lhs = rhs`` in the unknown ``variable``."""

    lhs: Expr
    rhs: Expr
    variable: Symbol


@dataclass(frozen=True)
class EquationStep(Step):
    """A step of the solution of ``lhs = rhs`` in ``variable``."""

    #: what the step does to the solutions: 'equivalence', 'cases' or 'implication'
    kind: ClassVar[str] = 'equivalence'

    lhs: Expr
    rhs: Expr
    variable: Symbol

    def solutions(self) -> Set:
        """The set of the real solutions; ``ValueError`` when a part of
        the tree is not solved."""
        raise NotImplementedError

    def statement(self) -> str:
        found = '?' if self.contains_dont_know() else str(self.solutions())
        return '%s = %s -> %s' % (self.lhs, self.rhs, found)


@dataclass(frozen=True)
class SolvedRule(EquationStep):
    """``variable = value``."""

    value: Expr

    def solutions(self) -> Set:
        return finite_set(self.value)


@dataclass(frozen=True)
class IdentityRule(EquationStep):
    """The two sides are equal for every value of the unknown."""

    def solutions(self) -> Set:
        return Reals


@dataclass(frozen=True)
class ContradictionRule(EquationStep):
    """The two sides are equal for no value of the unknown."""

    def solutions(self) -> Set:
        return EmptySet


@dataclass(frozen=True)
class _OneStep(EquationStep):
    """A step with one substep and the same solutions."""

    substep: EquationStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def solutions(self) -> Set:
        return self.substep.solutions()


@dataclass(frozen=True)
class AddToBothSidesRule(_OneStep):
    """``term`` is added to both sides (a subtraction when it has a
    minus sign)."""

    term: Expr


@dataclass(frozen=True)
class DivideBothSidesRule(_OneStep):
    """Both sides are divided by ``divisor``, which is not zero."""

    divisor: Expr


@dataclass(frozen=True)
class RewriteRule(_OneStep):
    """The sides are expanded and collected (a minor step)."""

    minor: ClassVar[bool] = True


@dataclass(frozen=True)
class SwapSidesRule(_OneStep):
    """The sides are exchanged (a minor step)."""

    minor: ClassVar[bool] = True


@dataclass(frozen=True)
class ZeroProductRule(EquationStep):
    """The left side is ``constant`` times the product of ``factors``:
    it vanishes when one of them does."""

    kind: ClassVar[str] = 'cases'

    constant: Expr
    factors: tuple[Expr, ...]
    cases: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.cases

    def solutions(self) -> Set:
        return union(*[case.solutions() for case in self.cases])


@dataclass(frozen=True)
class QuadraticFormulaRule(EquationStep):
    """``a*x**2 + b*x + c = 0``: the roots ``(-b -+ sqrt(discriminant))/
    (2*a)``, none when the discriminant is a negative number."""

    a: Expr
    b: Expr
    c: Expr
    discriminant: Expr

    def solutions(self) -> Set:
        if self.discriminant.is_negative is True:
            return EmptySet
        if self.discriminant == 0:
            return finite_set(-self.b/(2*self.a))
        root = sqrt(self.discriminant)
        return finite_set((-self.b - root)/(2*self.a), (-self.b + root)/(2*self.a))


@dataclass(frozen=True)
class CompleteSquareRule(_OneStep):
    """The equation is ``(x + shift)**2 = value``, the substep."""

    shift: Expr
    value: Expr


@dataclass(frozen=True)
class RootRule(EquationStep):
    """``base**exponent = value`` with a positive integer exponent: the
    real roots of ``value``, one equation ``base = root`` for each."""

    kind: ClassVar[str] = 'cases'

    base: Expr
    exponent: int
    value: Expr
    cases: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.cases

    def solutions(self) -> Set:
        return union(*[case.solutions() for case in self.cases]) if self.cases else EmptySet


@dataclass(frozen=True)
class _Checked(EquationStep):
    """A step whose new equation may have more solutions: those of the
    substep are checked in the original equation."""

    kind: ClassVar[str] = 'implication'

    substep: EquationStep
    rejected: tuple[Expr, ...]
    unverified: tuple[Expr, ...]

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def solutions(self) -> Set:
        found = self.substep.solutions()
        if not isinstance(found, FiniteSet):
            return found
        kept = [e for e in found.args if isinstance(e, Expr) and e not in self.rejected]
        return finite_set(*kept) if kept else EmptySet


@dataclass(frozen=True)
class ClearDenominatorsRule(_Checked):
    """Both sides are multiplied by ``denominator``; the solutions at
    which it vanishes are rejected."""

    denominator: Expr


@dataclass(frozen=True)
class SquareBothSidesRule(_Checked):
    """Both sides are squared; the solutions which do not satisfy the
    original equation are rejected."""


@dataclass(frozen=True)
class AlternativeRule(EquationStep, AlternativeStep):
    """Several derivations; the solutions are those of the first."""

    alternatives: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.alternatives

    def solutions(self) -> Set:
        return self.alternatives[0].solutions()


@dataclass(frozen=True)
class DontKnowRule(EquationStep):
    """No rule applies to the equation."""

    def contains_dont_know(self) -> bool:
        return True

    def solutions(self) -> Set:
        raise ValueError('the equation %s = %s is not solved' % (self.lhs, self.rhs))


# ---------------------------------------------------------------------
# helpers

_Run = Generator[EquationProblem, EquationStep, Optional[EquationStep]]
_Rule = Callable[[EquationProblem, Context], Generator[EquationProblem, EquationStep, Optional[EquationStep]]]


def _normal(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


def _split(e: Expr, x: Symbol) -> tuple[Expr, Expr]:
    """``(with, without)``: the terms of a sum which contain ``x`` and
    the others."""
    terms = [a for a in e.args if isinstance(a, Expr)] if isinstance(e, Add) else [e]
    return add(*[t for t in terms if t.has(x)]), add(*[t for t in terms if not t.has(x)])


def _polynomial(e: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The coefficients (highest degree first) of ``e`` as a polynomial
    in ``x``, or ``None``."""
    expanded = expand(e)
    if expanded == 0:
        return [Integer(0)]
    if not expanded.is_polynomial(x):
        return None
    from sympy_extras.matrices.eigen import _coefficients
    try:
        return _coefficients(expanded, x)
    except ValueError:
        return None


def _denominator(e: Expr, x: Symbol) -> Expr:
    """The denominator of ``e`` as it is written (nothing is cancelled:
    a factor removed from it would hide a value at which ``e`` is not
    defined) when it contains ``x``, else one."""
    _, denominator = e.as_numer_denom()
    return expand(denominator) if denominator.has(x) else Integer(1)


def _radicals(e: Expr, x: Symbol) -> list[Expr]:
    """The square roots in ``e`` whose radicand contains ``x``."""
    return [node for node in e.atoms(Pow) if isinstance(node, Pow) and node.exp == Rational(1, 2) and node.base.has(x)]


def _satisfies(problem: EquationProblem, value: Expr) -> Optional[bool]:
    """Whether ``value`` is a solution, when that is decided exactly."""
    x = problem.variable
    left, right = problem.lhs.subs(x, value), problem.rhs.subs(x, value)
    if left.is_finite is False or right.is_finite is False or left.has(nan, zoo) or right.has(nan, zoo):
        # a pole of one of the sides: not a solution
        return False
    try:
        difference = _normal(left - right)
    except ZeroDivisionError:
        return False
    if difference == 0 or difference.is_zero is True:
        return True
    if difference.is_zero is False or (difference.is_number and difference.is_real is False):
        return False
    if difference.is_positive is True or difference.is_negative is True:
        return False
    return None


# ---------------------------------------------------------------------
# the rules

def _trivial(problem: EquationProblem) -> Optional[EquationStep]:
    difference = _normal(problem.lhs - problem.rhs)
    if difference == 0:
        return IdentityRule(problem.lhs, problem.rhs, problem.variable)
    if not difference.has(problem.variable):
        return ContradictionRule(problem.lhs, problem.rhs, problem.variable)
    return None


def _solved(problem: EquationProblem) -> Optional[EquationStep]:
    return SolvedRule(problem.lhs, problem.rhs, problem.variable, problem.rhs)


def _rewrite(problem: EquationProblem, context: Context) -> _Run:
    lhs, rhs = expand(problem.lhs), expand(problem.rhs)
    if (lhs, rhs) == (problem.lhs, problem.rhs):
        return None
    substep = yield EquationProblem(lhs, rhs, problem.variable)
    return RewriteRule(problem.lhs, problem.rhs, problem.variable, substep)


def _swap(problem: EquationProblem, context: Context) -> _Run:
    substep = yield EquationProblem(problem.rhs, problem.lhs, problem.variable)
    return SwapSidesRule(problem.lhs, problem.rhs, problem.variable, substep)


def _added(problem: EquationProblem, term: Expr) -> EquationProblem:
    return EquationProblem(expand(problem.lhs + term), expand(problem.rhs + term), problem.variable)


def _move_unknowns(problem: EquationProblem, context: Context) -> _Run:
    """The terms with the unknown leave the right side."""
    moving, _ = _split(problem.rhs, problem.variable)
    if moving == 0:
        return None
    substep = yield _added(problem, -moving)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.variable, substep, -moving)


def _move_constants(problem: EquationProblem, context: Context) -> _Run:
    """The terms without the unknown leave the left side."""
    _, moving = _split(problem.lhs, problem.variable)
    if moving == 0:
        return None
    substep = yield _added(problem, -moving)
    return AddToBothSidesRule(problem.lhs, problem.rhs, problem.variable, substep, -moving)


def _divide(problem: EquationProblem, context: Context) -> _Run:
    """``a*x = c``."""
    x = problem.variable
    if isinstance(problem.lhs, Add):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    if rest != x or coefficient == 1 or coefficient == 0 or problem.rhs.has(x):
        return None
    substep = yield EquationProblem(x, _normal(problem.rhs/coefficient), x)
    return DivideBothSidesRule(problem.lhs, problem.rhs, x, substep, coefficient)


def _standard_form(problem: EquationProblem, context: Context) -> _Run:
    """Everything on the left, expanded."""
    x = problem.variable
    lhs = expand(problem.lhs - problem.rhs)
    if problem.rhs == 0:
        if lhs == problem.lhs:
            return None
        substep = yield EquationProblem(lhs, Integer(0), x)
        return RewriteRule(problem.lhs, problem.rhs, x, substep)
    substep = yield EquationProblem(lhs, Integer(0), x)
    return AddToBothSidesRule(problem.lhs, problem.rhs, x, substep, -problem.rhs)


def _zero_product(problem: EquationProblem, context: Context) -> _Run:
    """A polynomial which factors over the rationals, equal to zero."""
    x = problem.variable
    if problem.rhs != 0:
        return None
    from sympy_extras.polys.polyerrors import PolynomialError
    from sympy_extras.polys.polytools import factor_list
    symbols = sorted((s for s in problem.lhs.free_symbols if isinstance(s, Symbol) and s != x), key=lambda s: s.name)
    try:
        constant, listed = factor_list(problem.lhs, x, *symbols)
    except (PolynomialError, NotImplementedError, ValueError):
        return None
    factors = [factor for factor, _ in listed if factor.has(x)]
    outside = mul(constant, *[pow(factor, multiplicity) for factor, multiplicity in listed if not factor.has(x)])
    if len(factors) < 2 and not any(multiplicity != 1 for factor, multiplicity in listed if factor.has(x)):
        return None
    cases: list[EquationStep] = []
    for factor in factors:
        cases.append((yield EquationProblem(factor, Integer(0), x)))
    return ZeroProductRule(problem.lhs, problem.rhs, x, outside, tuple(factors), tuple(cases))


def _quadratic_formula(problem: EquationProblem) -> Optional[EquationStep]:
    coefficients = _polynomial(problem.lhs, problem.variable)
    if problem.rhs != 0 or coefficients is None or len(coefficients) != 3:
        return None
    a, b, c = coefficients
    return QuadraticFormulaRule(problem.lhs, problem.rhs, problem.variable, a, b, c, _normal(b**2 - 4*a*c))


def _complete_square(problem: EquationProblem, context: Context) -> _Run:
    x = problem.variable
    coefficients = _polynomial(problem.lhs, x)
    if problem.rhs != 0 or coefficients is None or len(coefficients) != 3:
        return None
    a, b, c = coefficients
    shift = _normal(b/(2*a))
    value = _normal(shift**2 - c/a)
    substep = yield EquationProblem(pow(x + shift, Integer(2)), value, x)
    return CompleteSquareRule(problem.lhs, problem.rhs, x, substep, shift, value)


def _root(problem: EquationProblem, context: Context) -> _Run:
    """``u**n = q`` with ``q`` free of the unknown."""
    x = problem.variable
    lhs, value = problem.lhs, problem.rhs
    if not (isinstance(lhs, Pow) and isinstance(lhs.exp, Integer) and lhs.exp.p >= 2) or value.has(x):
        return None
    base, n = lhs.base, lhs.exp.p
    roots: list[Expr]
    if value == 0:
        roots = [Integer(0)]
    elif n % 2 == 1:
        roots = [pow(value, Rational(1, n)) if value.is_negative is not True else -pow(-value, Rational(1, n))]
    elif value.is_negative is True:
        roots = []
    else:
        principal = pow(value, Rational(1, n))
        roots = [principal, -principal]
    cases: list[EquationStep] = []
    for root in roots:
        cases.append((yield EquationProblem(base, root, x)))
    return RootRule(lhs, value, x, base, n, value, tuple(cases))


def _checked(problem: EquationProblem, substep: EquationStep) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    """``(rejected, unverified)`` among the solutions of the substep."""
    if substep.contains_dont_know():
        return (), ()
    found = substep.solutions()
    if not isinstance(found, FiniteSet):
        return (), ()
    rejected: list[Expr] = []
    unverified: list[Expr] = []
    for candidate in found.args:
        if not isinstance(candidate, Expr):
            continue
        verdict = _satisfies(problem, candidate)
        if verdict is False:
            rejected.append(candidate)
        elif verdict is None:
            unverified.append(candidate)
    return tuple(rejected), tuple(unverified)


def _clear_denominators(problem: EquationProblem, context: Context) -> _Run:
    x = problem.variable
    left, right = _denominator(problem.lhs, x), _denominator(problem.rhs, x)
    if left == 1 and right == 1:
        return None
    common = left if right == 1 or _normal(left/right) == 1 else (right if left == 1 else expand(left*right))
    if left != 1 and right != 1 and common != left:
        # the least common denominator: the product divided by what the two share
        quotient = _normal(left/right)
        numerator, denominator = quotient.as_numer_denom()
        common = expand(right*numerator)
    substep = yield EquationProblem(expand(_normal(problem.lhs*common)), expand(_normal(problem.rhs*common)), x)
    rejected, unverified = _checked(problem, substep)
    return ClearDenominatorsRule(problem.lhs, problem.rhs, x, substep, rejected, unverified, common)


def _isolate_radical(problem: EquationProblem, context: Context) -> _Run:
    """One square root alone on the left: the other terms go right."""
    x = problem.variable
    terms = [a for a in problem.lhs.args if isinstance(a, Expr)] if isinstance(problem.lhs, Add) else [problem.lhs]
    radical_terms = [t for t in terms if _radicals(t, x)]
    if len(radical_terms) != 1 or len(terms) == 1:
        return None
    moving = add(*[t for t in terms if t is not radical_terms[0]])
    substep = yield EquationProblem(radical_terms[0], expand(problem.rhs - moving), x)
    return AddToBothSidesRule(problem.lhs, problem.rhs, x, substep, -moving)


def _square(problem: EquationProblem, context: Context) -> _Run:
    """``c*sqrt(A) = B``: squared."""
    x = problem.variable
    if isinstance(problem.lhs, Add):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    if not (isinstance(rest, Pow) and rest.exp == Rational(1, 2)) or coefficient == 0:
        return None
    substep = yield EquationProblem(expand(coefficient**2*rest.base), expand(problem.rhs**2), x)
    rejected, unverified = _checked(problem, substep)
    return SquareBothSidesRule(problem.lhs, problem.rhs, x, substep, rejected, unverified)


def _candidates(problem: EquationProblem) -> Sequence[_Rule]:
    lhs, rhs, x = problem.lhs, problem.rhs, problem.variable
    if not lhs.has(x) and not rhs.has(x):
        return [atomic(_trivial)]
    if lhs == x and not rhs.has(x):
        return [atomic(_solved)]
    if _denominator(lhs, x) != 1 or _denominator(rhs, x) != 1:
        return [_clear_denominators]
    if _radicals(lhs, x) or _radicals(rhs, x):
        if not _radicals(lhs, x):
            return [_swap]
        return [_isolate_radical, _square]
    if isinstance(lhs, Pow) and isinstance(lhs.exp, Integer) and lhs.exp.p >= 2 and not rhs.has(x):
        coefficients = _polynomial(lhs.base, x)
        if coefficients is not None and len(coefficients) == 2:
            return [_root]
    coefficients = _polynomial(lhs - rhs, x)
    if coefficients is None:
        return []
    degree = len(coefficients) - 1
    if degree <= 0:
        return [atomic(_trivial)]
    if degree == 1:
        return [_rewrite, _move_unknowns, _move_constants, _divide]
    if rhs != 0 or expand(lhs) != lhs:
        return [_standard_form]
    if degree == 2:
        return [_zero_product, atomic(_quadratic_formula), _complete_square]
    binomial = all(c == 0 for c in coefficients[1:-1])
    if binomial:
        return [_zero_product, _binomial]
    return [_zero_product]


def _binomial(problem: EquationProblem, context: Context) -> _Run:
    """``a*x**n + c = 0`` as ``x**n = -c/a``."""
    x = problem.variable
    coefficients = _polynomial(problem.lhs, x)
    if problem.rhs != 0 or coefficients is None:
        return None
    n = len(coefficients) - 1
    value = _normal(-coefficients[-1]/coefficients[0])
    substep = yield EquationProblem(pow(x, Integer(n)), value, x)
    return DivideBothSidesRule(problem.lhs, problem.rhs, x, substep, coefficients[0])


def _dont_know(problem: EquationProblem) -> EquationStep:
    return DontKnowRule(problem.lhs, problem.rhs, problem.variable)


def _alternative(problem: EquationProblem, alternatives: Sequence[EquationStep]) -> EquationStep:
    return AlternativeRule(problem.lhs, problem.rhs, problem.variable, tuple(alternatives))


def solve_steps(lhs: ExprLike, rhs: ExprLike, x: Symbol, branch: bool = False, max_depth: Optional[int] = 30) -> EquationStep:
    """The derivation of the real solutions of ``lhs = rhs`` in ``x``,
    as a tree of rules (see the module); the tree
    ``contains_dont_know()`` when the rules do not solve the equation.
    With ``branch`` every workable way is kept at each level, in
    :class:`AlternativeRule` nodes; ``max_depth`` limits the nesting.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.core import format_steps
    >>> from sympy_extras.steps.equations import solve_steps
    >>> x = Symbol('x')
    >>> steps = solve_steps(1/(x - 1), 2/(x**2 - 1), x)
    >>> print(format_steps(steps))
    ClearDenominatorsRule: 1/(x - 1) = 2/(x**2 - 1) -> EmptySet
      AddToBothSidesRule: x + 1 = 2 -> {1}
        SolvedRule: x = 1 -> {1}
    >>> steps.kind, steps.rejected
    ('implication', (1,))
    """
    solver: StepSolver[EquationProblem, EquationStep] = StepSolver(_candidates, _dont_know, _alternative, branch, max_depth)
    return solver.solve(EquationProblem(to_expr(lhs), to_expr(rhs), x))


def manual_solve(lhs: ExprLike, rhs: ExprLike, x: Symbol) -> Optional[Set]:
    """The set of the real solutions found by the rules, or ``None``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.steps.equations import manual_solve
    >>> x = Symbol('x')
    >>> manual_solve(3*x - 1, x + 5, x), manual_solve(x**2 + 1, 0, x), manual_solve(x**3, 8, x), manual_solve(x + 1, x + 1, x)
    ({3}, EmptySet, {2}, Reals)
    """
    steps = solve_steps(lhs, rhs, x)
    return None if steps.contains_dont_know() else steps.solutions()


