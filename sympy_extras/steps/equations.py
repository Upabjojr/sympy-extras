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
``AbsoluteValueRule``        $|A| = B$: the cases $A = B$ and $A = -B$
``TakeLogarithmRule``        $a^f = c$: $f = \log c/\log a$
``EqualExponentsRule``       $a^f = a^g$: $f = g$
``ExponentiateRule``         $\log f = c$: $f = e^c$
``EqualArgumentsRule``       $\log f = \log g$: $f = g$, checked
``CombineLogarithmsRule``    a sum of logarithms as one, checked
``SubstitutionRule``         a polynomial in $e^x$, $\sin x$, $\log x$, ...
``PeriodicRule``             $\sin f = c$, $\cos f = c$, $\tan f = c$
``TangentRule``              $a \sin f = b \cos f$ as $\tan f = b/a$
``CardanoRule``              a cubic with one real root
``RealRootsRule``            the real roots of a polynomial, isolated
``AlternativeRule``          several derivations of the same solution
``DontKnowRule``             no rule applies
``InProgressRule``           the problem met again while being solved
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

Equations with several square roots are squared as many times as needed.
A trigonometric equation has the solutions of each of its cases for
every integer ``n``: ``solutions()`` is then a union of images of the
integers. A polynomial of degree three or more which does not factor has
its real roots isolated exactly (``IndexedRoot``: the polynomial and the
rank of the root); a cubic with a single real root is also solved by
Cardano's formula.

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
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational, nan, pi, zoo
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import acos, asin, atan, exp, log
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, Tan
from sympy_extras.sets.fancysets import Integers, image_set
from sympy_extras.sets.sets import EmptySet, FiniteSet, Reals, Set, finite_set, union

from .core import AlternativeStep, Context, Step, StepSolver, atomic

__all__ = ['EquationProblem', 'EquationStep', 'SolvedRule', 'IdentityRule', 'ContradictionRule', 'AddToBothSidesRule',
           'DivideBothSidesRule', 'RewriteRule', 'ZeroProductRule', 'QuadraticFormulaRule', 'CompleteSquareRule', 'RootRule',
           'ClearDenominatorsRule', 'SquareBothSidesRule', 'SwapSidesRule', 'AbsoluteValueRule', 'TakeLogarithmRule',
           'EqualExponentsRule', 'ExponentiateRule', 'EqualArgumentsRule', 'CombineLogarithmsRule', 'SubstitutionRule',
           'PeriodicRule', 'TangentRule', 'CardanoRule', 'RealRootsRule', 'AlternativeRule', 'DontKnowRule', 'InProgressRule',
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
class AbsoluteValueRule(EquationStep):
    """``|inner| = value``: the cases ``inner = value`` and ``inner =
    -value``, whose solutions are checked in the original equation (the
    value must not be negative there)."""

    kind: ClassVar[str] = 'implication'

    inner: Expr
    cases: tuple[EquationStep, ...]
    rejected: tuple[Expr, ...]
    unverified: tuple[Expr, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.cases

    def solutions(self) -> Set:
        found = union(*[case.solutions() for case in self.cases])
        if not isinstance(found, FiniteSet):
            return found
        kept = [e for e in found.args if isinstance(e, Expr) and e not in self.rejected]
        return finite_set(*kept) if kept else EmptySet


@dataclass(frozen=True)
class TakeLogarithmRule(_OneStep):
    """``base**f = value`` with positive ``base`` and ``value``: the
    logarithm of both sides."""

    base: Expr


@dataclass(frozen=True)
class EqualExponentsRule(_OneStep):
    """``base**f = base**g``: the exponents are equal."""

    base: Expr


@dataclass(frozen=True)
class ExponentiateRule(_OneStep):
    """``log(f) = value``: the exponential of both sides."""


@dataclass(frozen=True)
class EqualArgumentsRule(_Checked):
    """``log(f) = log(g)``: the arguments are equal, where both are
    positive (the solutions are checked)."""


@dataclass(frozen=True)
class CombineLogarithmsRule(_Checked):
    """A sum of logarithms is the logarithm of the product, where each
    argument is positive (the solutions are checked)."""


@dataclass(frozen=True)
class SubstitutionRule(EquationStep):
    """The equation is one in ``placeholder = expression``: ``substep``
    solves it for the placeholder, and ``cases`` solve ``expression =
    value`` for each value found."""

    kind: ClassVar[str] = 'cases'

    placeholder: Symbol
    expression: Expr
    substep: EquationStep
    cases: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,) + self.cases

    def solutions(self) -> Set:
        return union(*[case.solutions() for case in self.cases]) if self.cases else EmptySet


@dataclass(frozen=True)
class PeriodicRule(EquationStep):
    """``function(f) = value`` for a sine, cosine or tangent: the cases
    ``f = angle + period*parameter`` for every integer ``parameter``. The
    solutions are the images of the integers."""

    kind: ClassVar[str] = 'cases'

    function: str
    value: Expr
    parameter: Symbol
    cases: tuple[EquationStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.cases

    def solutions(self) -> Set:
        families: list[Set] = []
        for case in self.cases:
            found = case.solutions()
            if not isinstance(found, FiniteSet):
                raise ValueError('the solutions %s of a periodic case are not expressions in %s' % (found, self.parameter))
            for expression in found.args:
                families.append(image_set(Tuple(self.parameter), expression, Integers))
        return union(*families) if families else EmptySet


@dataclass(frozen=True)
class TangentRule(_OneStep):
    """``a*sin(f) = b*cos(f)`` divided by ``a*cos(f)``, which does not
    vanish at a solution: ``tan(f) = b/a``."""


@dataclass(frozen=True)
class CardanoRule(EquationStep):
    """A cubic with one real root: with ``x = t - shift`` it is ``t**3 +
    p*t + q = 0`` and ``t`` is the sum of the real cube roots of ``-q/2
    +- sqrt(q**2/4 + p**3/27)``."""

    shift: Expr
    p: Expr
    q: Expr

    def solutions(self) -> Set:
        inner = sqrt(self.q**2/4 + self.p**3/27)
        return finite_set(_real_cube_root(-self.q/2 + inner) + _real_cube_root(-self.q/2 - inner) - self.shift)


@dataclass(frozen=True)
class RealRootsRule(EquationStep):
    """A polynomial with rational coefficients which does not factor:
    its real roots, isolated exactly (each is given by the polynomial
    and its rank among the real roots)."""

    roots: tuple[Expr, ...]

    def solutions(self) -> Set:
        return finite_set(*self.roots) if self.roots else EmptySet


def _real_cube_root(value: Expr) -> Expr:
    if value.is_negative is True:
        return -pow(-value, Rational(1, 3))
    return pow(value, Rational(1, 3))


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


@dataclass(frozen=True)
class InProgressRule(EquationStep):
    """The problem is met again while it is being solved (the derivation
    leads around in a loop): a placeholder for work in progress, where
    ``DontKnowRule`` says that no rule applies."""

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
    if left.is_real is False or right.is_real is False:
        # a side is not a real number there (the logarithm of a negative number): not a solution over the reals,
        # even when the two complex values are equal
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
    if difference.free_symbols:
        return None
    from sympy_extras.simplify.structure import is_zero
    try:
        return is_zero(difference)
    except (ValueError, NotImplementedError, TypeError):
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
    """One square root alone on the left: the other terms (other square
    roots among them) go right."""
    x = problem.variable
    terms = [a for a in problem.lhs.args if isinstance(a, Expr)] if isinstance(problem.lhs, Add) else [problem.lhs]
    radical_terms = [t for t in terms if _radicals(t, x)]
    if not radical_terms or len(terms) == 1:
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


def _absolute_values(e: Expr, x: Symbol) -> list[Expr]:
    return [node for node in e.atoms(Abs) if isinstance(node, Expr) and node.has(x)]


def _isolate(kind: Callable[[Expr, Symbol], list[Expr]]) -> _Rule:
    """The first term of the left side with a node of the kind stays;
    the other terms go right."""
    def rule(problem: EquationProblem, context: Context) -> _Run:
        x = problem.variable
        terms = [a for a in problem.lhs.args if isinstance(a, Expr)] if isinstance(problem.lhs, Add) else [problem.lhs]
        chosen = [t for t in terms if kind(t, x)]
        if not chosen or len(terms) == 1:
            return None
        moving = add(*[t for t in terms if t is not chosen[0]])
        substep = yield EquationProblem(chosen[0], expand(problem.rhs - moving), x)
        return AddToBothSidesRule(problem.lhs, problem.rhs, x, substep, -moving)
    return rule


def _absolute_value(problem: EquationProblem, context: Context) -> _Run:
    """``c*|A| = B``."""
    x = problem.variable
    if isinstance(problem.lhs, Add):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    if not isinstance(rest, Abs) or coefficient == 0:
        return None
    inner = rest.args[0]
    if not isinstance(inner, Expr):
        return None
    value = _normal(problem.rhs/coefficient)
    if value.is_negative is True:
        return ContradictionRule(problem.lhs, problem.rhs, x)
    targets = [value] if value == 0 else [value, -value]
    cases: list[EquationStep] = []
    for target in targets:
        cases.append((yield EquationProblem(inner, expand(target), x)))
    rejected: list[Expr] = []
    unverified: list[Expr] = []
    for case in cases:
        if case.contains_dont_know():
            continue
        found = case.solutions()
        if not isinstance(found, FiniteSet):
            continue
        for candidate in found.args:
            if not isinstance(candidate, Expr):
                continue
            verdict = _satisfies(problem, candidate)
            if verdict is False and candidate not in rejected:
                rejected.append(candidate)
            elif verdict is None and candidate not in unverified:
                unverified.append(candidate)
    return AbsoluteValueRule(problem.lhs, problem.rhs, x, inner, tuple(cases), tuple(rejected), tuple(unverified))


def _exponential_parts(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(base, exponent)`` when ``e`` is ``exp(f)`` or ``a**f`` with a
    constant positive ``a`` and ``f`` containing the unknown."""
    if isinstance(e, Exp):
        argument = e.args[0]
        if isinstance(argument, Expr) and argument.has(x):
            return exp(Integer(1)), argument
    if isinstance(e, Pow) and not e.base.has(x) and e.exp.has(x) and e.base.is_positive is True and e.base != 1:
        return e.base, e.exp
    return None


def _take_logarithm(problem: EquationProblem, context: Context) -> _Run:
    """``c*a**f = v``."""
    x = problem.variable
    if isinstance(problem.lhs, Add) or problem.rhs.has(x):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    parts = _exponential_parts(rest, x)
    if parts is None or coefficient == 0:
        return None
    base, exponent = parts
    value = _normal(problem.rhs/coefficient)
    if value.is_positive is not True:
        if value.is_negative is True or value == 0:
            return ContradictionRule(problem.lhs, problem.rhs, x)
        return None
    # log(8)/log(2) is 3: the logarithms of rational numbers are split over the primes
    target = _normal(log(value) if isinstance(rest, Exp) else log(value)/log(base))
    substep = yield EquationProblem(exponent, target, x)
    return TakeLogarithmRule(problem.lhs, problem.rhs, x, substep, base)


def _equal_exponents(problem: EquationProblem, context: Context) -> _Run:
    x = problem.variable
    left, right = _exponential_parts(problem.lhs, x), _exponential_parts(problem.rhs, x)
    if left is None or right is None or left[0] != right[0]:
        return None
    substep = yield EquationProblem(left[1], right[1], x)
    return EqualExponentsRule(problem.lhs, problem.rhs, x, substep, left[0])


def _logarithm_argument(e: Expr, x: Symbol) -> Optional[Expr]:
    if isinstance(e, Log):
        argument = e.args[0]
        if isinstance(argument, Expr) and argument.has(x):
            return argument
    return None


def _exponentiate(problem: EquationProblem, context: Context) -> _Run:
    """``c*log(f) = v``."""
    x = problem.variable
    if isinstance(problem.lhs, Add) or problem.rhs.has(x):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    argument = _logarithm_argument(rest, x)
    if argument is None or coefficient == 0:
        return None
    substep = yield EquationProblem(argument, exp(_normal(problem.rhs/coefficient)), x)
    return ExponentiateRule(problem.lhs, problem.rhs, x, substep)


def _equal_arguments(problem: EquationProblem, context: Context) -> _Run:
    x = problem.variable
    left, right = _logarithm_argument(problem.lhs, x), _logarithm_argument(problem.rhs, x)
    if left is None or right is None:
        return None
    substep = yield EquationProblem(left, right, x)
    rejected, unverified = _checked(problem, substep)
    return EqualArgumentsRule(problem.lhs, problem.rhs, x, substep, rejected, unverified)


def _combined(e: Expr, x: Symbol) -> Optional[Expr]:
    """A sum of multiples of logarithms as one logarithm, with the terms
    which are not logarithms left over; ``None`` when there is nothing to
    combine."""
    terms = [a for a in e.args if isinstance(a, Expr)] if isinstance(e, Add) else [e]
    product: Expr = Integer(1)
    others: list[Expr] = []
    count = 0
    for term in terms:
        coefficient, rest = (Integer(1), term) if isinstance(term, Log) else term.as_independent(x)
        argument = _logarithm_argument(rest, x)
        if argument is not None and isinstance(coefficient, Integer):
            product = product*pow(argument, coefficient)
            count += 1
        else:
            others.append(term)
    if count < 2:
        return None
    return add(log(product), *others)


def _combine_logarithms(problem: EquationProblem, context: Context) -> _Run:
    x = problem.variable
    left, right = _combined(problem.lhs, x), _combined(problem.rhs, x)
    if left is None and right is None:
        return None
    substep = yield EquationProblem(problem.lhs if left is None else left, problem.rhs if right is None else right, x)
    rejected, unverified = _checked(problem, substep)
    return CombineLogarithmsRule(problem.lhs, problem.rhs, x, substep, rejected, unverified)


def _fresh(name: str, problem: EquationProblem, integer: bool = False) -> Symbol:
    used = {s.name for e in (problem.lhs, problem.rhs) for s in e.free_symbols if isinstance(s, Symbol)}
    used.add(problem.variable.name)
    candidate, k = name, 0
    while candidate in used:
        k += 1
        candidate = '%s%d' % (name, k)
    return Symbol(candidate, integer=True) if integer else Symbol(candidate)


def _primitive_root(n: int) -> tuple[int, int]:
    """``(r, k)`` with ``n == r**k`` and ``k`` as large as possible
    (``n >= 2``)."""
    for k in range(n.bit_length(), 1, -1):
        r = round(n**(1.0/k))
        for candidate in (r - 1, r, r + 1):
            if candidate >= 2 and candidate**k == n:
                return candidate, k
    return n, 1


def _common_base(difference: Expr, x: Symbol, u: Symbol) -> Optional[tuple[Expr, Expr]]:
    """Powers of integers with one exponent, the bases being powers of
    one integer: ``4**x - 2**x - 2`` is ``u**2 - u - 2`` with ``u = 2**x``."""
    powers: list[tuple[Pow, int]] = []
    for node in sorted(difference.atoms(Pow), key=str):
        if isinstance(node, Pow) and node.exp.has(x):
            base = node.base
            if isinstance(base, Integer) and base.p >= 2:
                powers.append((node, base.p))
    if len(powers) < 2 or len({node.exp for node, _ in powers}) != 1:
        return None
    roots = [_primitive_root(base) for _, base in powers]
    if len({root for root, _ in roots}) != 1:
        return None
    root = roots[0][0]
    rule: dict[Basic, Basic] = {node: pow(u, Integer(k)) for (node, _), (_, k) in zip(powers, roots)}
    return difference.xreplace(rule), pow(Integer(root), powers[0][0].exp)


def _substitution(problem: EquationProblem, context: Context) -> _Run:
    """The equation is a polynomial of degree two or more in one
    transcendental expression of the unknown."""
    from sympy_extras.simplify.normalform import _exponentials
    x = problem.variable
    difference = expand(problem.lhs - problem.rhs)
    u = _fresh('u', problem)
    candidates: list[tuple[Expr, Expr]] = []                 # (the equation in u, what u stands for)
    back: dict[Symbol, Expr] = {}
    [prepared] = _exponentials([difference], back)
    if len(back) == 1:
        [(placeholder, expression)] = list(back.items())
        candidates.append((prepared.subs(placeholder, u), expression))
    for node in sorted(difference.atoms(Sin, Cos, Tan, Log, Abs), key=str):
        if isinstance(node, Expr) and node.has(x):
            rule: dict[Basic, Basic] = {node: u}
            candidates.append((difference.xreplace(rule), node))
    common = _common_base(difference, x, u)
    if common is not None:
        candidates.append(common)
    for equation, expression in candidates:
        coefficients = None if equation.has(x) else _polynomial(equation, u)
        if coefficients is None or len(coefficients) < 3:
            continue
        substep = yield EquationProblem(equation, Integer(0), u)
        if substep.contains_dont_know():
            continue
        found = substep.solutions()
        if not isinstance(found, FiniteSet) and found is not EmptySet:
            continue
        cases: list[EquationStep] = []
        for value in (found.args if isinstance(found, FiniteSet) else ()):
            if isinstance(value, Expr):
                cases.append((yield EquationProblem(expression, value, x)))
        return SubstitutionRule(problem.lhs, problem.rhs, x, u, expression, substep, tuple(cases))
    return None


def _periodic(problem: EquationProblem, context: Context) -> _Run:
    """``c*sin(f) = v``, and the same for the cosine and the tangent."""
    x = problem.variable
    if isinstance(problem.lhs, Add) or problem.rhs.has(x):
        return None
    coefficient, rest = problem.lhs.as_independent(x)
    if not isinstance(rest, (Sin, Cos, Tan)) or coefficient == 0:
        return None
    argument = rest.args[0]
    if not isinstance(argument, Expr):
        return None
    value = _normal(problem.rhs/coefficient)
    n = _fresh('n', problem, integer=True)
    angles: list[Expr]
    name: str
    if isinstance(rest, Tan):
        name, angles = 'tan', [atan(value) + pi*n]
    else:
        if (value - 1).is_positive is True or (value + 1).is_negative is True:
            return ContradictionRule(problem.lhs, problem.rhs, x)
        if isinstance(rest, Sin):
            name, principal = 'sin', asin(value)
            angles = [principal + 2*pi*n] if value in (Integer(1), Integer(-1)) else [principal + 2*pi*n, pi - principal + 2*pi*n]
        else:
            name, principal = 'cos', acos(value)
            angles = [principal + 2*pi*n] if value in (Integer(1), Integer(-1)) else [principal + 2*pi*n, -principal + 2*pi*n]
    cases: list[EquationStep] = []
    for angle in angles:
        cases.append((yield EquationProblem(argument, expand(angle), x)))
    return PeriodicRule(problem.lhs, problem.rhs, x, name, value, n, tuple(cases))


def _tangent(problem: EquationProblem, context: Context) -> _Run:
    """``a*sin(f) = b*cos(f)``."""
    x = problem.variable
    if isinstance(problem.lhs, Add) or isinstance(problem.rhs, Add):
        return None
    a, left = problem.lhs.as_independent(x)
    b, right = problem.rhs.as_independent(x)
    if isinstance(left, Cos) and isinstance(right, Sin):
        a, left, b, right = b, right, a, left
    if not (isinstance(left, Sin) and isinstance(right, Cos)) or left.args != right.args or a == 0:
        return None
    argument = left.args[0]
    if not isinstance(argument, Expr):
        return None
    from sympy_extras.functions import tan
    substep = yield EquationProblem(tan(argument), _normal(b/a), x)
    return TangentRule(problem.lhs, problem.rhs, x, substep)


def _cardano(problem: EquationProblem) -> Optional[EquationStep]:
    coefficients = _polynomial(problem.lhs, problem.variable)
    if problem.rhs != 0 or coefficients is None or len(coefficients) != 4:
        return None
    if not all(isinstance(c, Rational) for c in coefficients):
        return None
    a, b, c, d = coefficients
    shift = b/(3*a)
    p = _normal(c/a - b**2/(3*a**2))
    q = _normal(2*b**3/(27*a**3) - b*c/(3*a**2) + d/a)
    if (q**2/4 + p**3/27).is_positive is not True:
        return None                                  # three real roots: the formula goes through complex numbers
    return CardanoRule(problem.lhs, problem.rhs, problem.variable, shift, p, q)


def _real_roots(problem: EquationProblem) -> Optional[EquationStep]:
    x = problem.variable
    coefficients = _polynomial(problem.lhs, x)
    if problem.rhs != 0 or coefficients is None or len(coefficients) < 4:
        return None
    from fractions import Fraction
    values: list[Fraction] = []
    for c in coefficients:
        if not isinstance(c, Rational):
            return None
        values.append(Fraction(c.p, c.q))
    from sympy_extras.polys.cad.indexedroot import real_root
    from sympy_extras.polys.cad.samplepoints import real_roots_of
    roots = tuple(real_root(r, x.name) for r in real_roots_of(values))
    return RealRootsRule(problem.lhs, problem.rhs, x, roots)


def _transcendental(e: Expr, x: Symbol) -> bool:
    """Whether ``e`` has an exponential, a logarithm, a sine, cosine or
    tangent of the unknown."""
    if any(node.has(x) for node in e.atoms(Exp, Log, Sin, Cos, Tan)):
        return True
    return any(isinstance(node, Pow) and not node.base.has(x) and node.exp.has(x) for node in e.atoms(Pow))


def _kernel_terms(e: Expr, x: Symbol) -> list[Expr]:
    return [e] if _transcendental(e, x) else []


def _candidates(problem: EquationProblem) -> Sequence[_Rule]:
    lhs, rhs, x = problem.lhs, problem.rhs, problem.variable
    if not lhs.has(x) and not rhs.has(x):
        return [atomic(_trivial)]
    if lhs == x and not rhs.has(x):
        return [atomic(_solved)]
    if _denominator(lhs, x) != 1 or _denominator(rhs, x) != 1:
        return [_clear_denominators]
    if _absolute_values(lhs, x) or _absolute_values(rhs, x):
        if not _absolute_values(lhs, x):
            return [_swap]
        return [_absolute_value, _isolate(_absolute_values), _substitution]
    if _radicals(lhs, x) or _radicals(rhs, x):
        if not _radicals(lhs, x):
            return [_swap]
        return [_isolate_radical, _square]
    if _transcendental(lhs, x) or _transcendental(rhs, x):
        if not _transcendental(lhs, x):
            return [_swap]
        return [_take_logarithm, _exponentiate, _periodic, _equal_exponents, _equal_arguments, _tangent,
                _combine_logarithms, _substitution, _isolate(_kernel_terms), _rewrite]
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
    rules: list[_Rule] = [_zero_product]
    if binomial:
        rules.append(_binomial)
    if degree == 3:
        rules.append(atomic(_cardano))
    rules.append(atomic(_real_roots))
    return rules


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


def _in_progress(problem: EquationProblem) -> EquationStep:
    return InProgressRule(problem.lhs, problem.rhs, problem.variable)


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
    solver: StepSolver[EquationProblem, EquationStep] = StepSolver(_candidates, _dont_know, _in_progress, _alternative, branch, max_depth)
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


