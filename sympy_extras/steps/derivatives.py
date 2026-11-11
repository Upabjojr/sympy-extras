r"""Derivatives step by step: the tree of the rules of differentiation.

:func:`derivative_steps` returns the derivation of $\frac{d}{dx} f$ as
a tree of rule dataclasses, built by generator rules which the solver of
:mod:`sympy_extras.steps.core` drives:

==========================  =================================================
``ConstantRule``            $c' = 0$
``VariableRule``            $x' = 1$
``PowerRule``               $(x^n)' = n x^{n-1}$
``ExponentialRule``         $(a^x)' = a^x \log a$
``FunctionRule``            the table: $\sin$, $\cos$, $\exp$, $\log$, ...
``ConstantMultipleRule``    $(c f)' = c f'$
``SumRule``                 $(f + g + \dots)' = f' + g' + \dots$
``ProductRule``             $(f g)' = f' g + f g'$
``QuotientRule``            $(f/g)' = (f' g - f g')/g^2$
``ChainRule``               $(f(u(x)))' = f'(u)\, u'(x)$
``RewriteRule``             the same function written otherwise
``AlternativeRule``         several derivations of the same derivative
``DontKnowRule``            the derivative is left unevaluated
==========================  =================================================

The chain rule solves two subproblems: the outer function of a
placeholder variable, and the inner function. A power $f^g$ with a
variable exponent is rewritten as $e^{g \log f}$. With ``branch=True`` a
quotient is differentiated both by the quotient rule and as the product
$f \cdot g^{-1}$, and both derivations are kept.

Every step has its ``result()``, assembled from those of its substeps
exactly as the rule says, without simplification: the answer a student
would write. :func:`manual_derivative` returns it.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin, exp
>>> from sympy_extras.steps.core import format_steps
>>> from sympy_extras.steps.derivatives import derivative_steps, manual_derivative
>>> x = Symbol('x')
>>> steps = derivative_steps(x**2*sin(3*x), x)
>>> print(format_steps(steps))
ProductRule: d/dx(x**2*sin(3*x)) = 3*x**2*cos(3*x) + 2*x*sin(3*x)
  PowerRule: d/dx(x**2) = 2*x
  ChainRule: d/dx(sin(3*x)) = 3*cos(3*x)
    FunctionRule: d/du(sin(u)) = cos(u)
    ConstantMultipleRule: d/dx(3*x) = 3
      VariableRule: d/dx(x) = 1
>>> print(format_steps(steps, depth=1))
ProductRule: d/dx(x**2*sin(3*x)) = 3*x**2*cos(3*x) + 2*x*sin(3*x)
  PowerRule: d/dx(x**2) = 2*x
  ChainRule: d/dx(sin(3*x)) = 3*cos(3*x)
>>> manual_derivative(exp(x**2), x)
2*x*exp(x**2)
>>> both = derivative_steps(sin(x)/x, x, branch=True)
>>> both.rule, [alternative.rule for alternative in both.substeps()]
('AlternativeRule', ['QuotientRule', 'ProductRule'])

References
==========

.. [1] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
       (rules as generators returning rule dataclasses), SymPy master,
       2026; SymPy issue 11088, *Step by Step solutions*.
.. [2] M. Spivak, *Calculus*, 4th ed., Publish or Perish, 2008, Chapter
       10 (differentiation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Generator, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Derivative
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import cos, cosh, exp, log, sin, sinh
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Acosh, Asinh, Atanh, Cosh, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Acos, Asin, Atan, Cos, Cot, Csc, Sec, Sin, Tan

from .core import AlternativeStep, Context, Step, StepSolver, atomic

__all__ = ['DerivativeProblem', 'DerivativeStep', 'ConstantRule', 'VariableRule', 'PowerRule', 'ExponentialRule',
           'FunctionRule', 'ConstantMultipleRule', 'SumRule', 'ProductRule', 'QuotientRule', 'ChainRule', 'RewriteRule',
           'AlternativeRule', 'DontKnowRule', 'derivative_steps', 'manual_derivative']


@dataclass(frozen=True)
class DerivativeProblem:
    """The derivative of ``expression`` with respect to ``variable``."""

    expression: Expr
    variable: Symbol


@dataclass(frozen=True)
class DerivativeStep(Step):
    """A step of a differentiation: the rule applied to ``expression``,
    a function of ``variable``."""

    expression: Expr
    variable: Symbol

    def result(self) -> Expr:
        """The derivative, assembled as the rule says."""
        raise NotImplementedError

    def statement(self) -> str:
        return 'd/d%s(%s) = %s' % (self.variable, self.expression, self.result())


@dataclass(frozen=True)
class ConstantRule(DerivativeStep):
    def result(self) -> Expr:
        return Integer(0)


@dataclass(frozen=True)
class VariableRule(DerivativeStep):
    def result(self) -> Expr:
        return Integer(1)


@dataclass(frozen=True)
class PowerRule(DerivativeStep):
    """``variable**exponent`` with a constant exponent."""

    exponent: Expr

    def result(self) -> Expr:
        return self.exponent*pow(self.variable, self.exponent - 1)


@dataclass(frozen=True)
class ExponentialRule(DerivativeStep):
    """``base**variable`` with a constant base."""

    base: Expr

    def result(self) -> Expr:
        return self.expression*log(self.base)


def _table(name: str, u: Expr) -> Optional[Expr]:
    """The derivative of the elementary function ``name`` at ``u``."""
    if name == 'sin':
        return cos(u)
    if name == 'cos':
        return -sin(u)
    if name == 'tan':
        return 1/cos(u)**2
    if name == 'cot':
        return -1/sin(u)**2
    if name == 'sec':
        return sin(u)/cos(u)**2
    if name == 'csc':
        return -cos(u)/sin(u)**2
    if name == 'exp':
        return exp(u)
    if name == 'log':
        return 1/u
    if name == 'sinh':
        return cosh(u)
    if name == 'cosh':
        return sinh(u)
    if name == 'tanh':
        return 1/cosh(u)**2
    if name == 'asin':
        return 1/sqrt(1 - u**2)
    if name == 'acos':
        return -1/sqrt(1 - u**2)
    if name == 'atan':
        return 1/(1 + u**2)
    if name == 'asinh':
        return 1/sqrt(u**2 + 1)
    if name == 'acosh':
        return 1/sqrt(u**2 - 1)
    if name == 'atanh':
        return 1/(1 - u**2)
    return None


_NAMES: dict[type, str] = {Sin: 'sin', Cos: 'cos', Tan: 'tan', Cot: 'cot', Sec: 'sec', Csc: 'csc', Exp: 'exp', Log: 'log',
                           Sinh: 'sinh', Cosh: 'cosh', Tanh: 'tanh', Asin: 'asin', Acos: 'acos', Atan: 'atan',
                           Asinh: 'asinh', Acosh: 'acosh', Atanh: 'atanh'}


@dataclass(frozen=True)
class FunctionRule(DerivativeStep):
    """An elementary function of the variable itself, from the table;
    ``function`` is its name."""

    function: str

    def result(self) -> Expr:
        value = _table(self.function, self.variable)
        if value is None:
            raise ValueError('no derivative in the table for %s' % (self.function,))
        return value


@dataclass(frozen=True)
class ConstantMultipleRule(DerivativeStep):
    constant: Expr
    rest: Expr
    substep: DerivativeStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def result(self) -> Expr:
        return self.constant*self.substep.result()


@dataclass(frozen=True)
class SumRule(DerivativeStep):
    terms: tuple[DerivativeStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.terms

    def result(self) -> Expr:
        return add(*[term.result() for term in self.terms])


@dataclass(frozen=True)
class ProductRule(DerivativeStep):
    """``first*second``."""

    first: Expr
    second: Expr
    first_step: DerivativeStep
    second_step: DerivativeStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.first_step, self.second_step)

    def result(self) -> Expr:
        return self.first_step.result()*self.second + self.first*self.second_step.result()


@dataclass(frozen=True)
class QuotientRule(DerivativeStep):
    """``numerator/denominator``."""

    numerator: Expr
    denominator: Expr
    numerator_step: DerivativeStep
    denominator_step: DerivativeStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.numerator_step, self.denominator_step)

    def result(self) -> Expr:
        return ((self.numerator_step.result()*self.denominator - self.numerator*self.denominator_step.result())
                / self.denominator**2)


@dataclass(frozen=True)
class ChainRule(DerivativeStep):
    """The expression is the function of ``placeholder`` differentiated
    by ``outer_step``, at ``placeholder = inner``."""

    inner: Expr
    placeholder: Symbol
    outer_step: DerivativeStep
    inner_step: DerivativeStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.outer_step, self.inner_step)

    def result(self) -> Expr:
        return self.outer_step.result().subs(self.placeholder, self.inner)*self.inner_step.result()


@dataclass(frozen=True)
class RewriteRule(DerivativeStep):
    """The expression equals ``rewritten``, which is differentiated (a
    minor step)."""

    minor: ClassVar[bool] = True

    rewritten: Expr
    substep: DerivativeStep

    def substeps(self) -> tuple[Step, ...]:
        return (self.substep,)

    def result(self) -> Expr:
        return self.substep.result()


@dataclass(frozen=True)
class AlternativeRule(DerivativeStep, AlternativeStep):
    """Several derivations; the result is that of the first."""

    alternatives: tuple[DerivativeStep, ...]

    def substeps(self) -> tuple[Step, ...]:
        return self.alternatives

    def result(self) -> Expr:
        return self.alternatives[0].result()


@dataclass(frozen=True)
class DontKnowRule(DerivativeStep):
    """No rule applies: the derivative is left as it is."""

    def contains_dont_know(self) -> bool:
        return True

    def result(self) -> Expr:
        return Derivative(self.expression, self.variable)


# ---------------------------------------------------------------------
# the rules: generators which yield the subproblems and return the step

_Rule = Callable[[DerivativeProblem, Context], Generator[DerivativeProblem, DerivativeStep, Optional[DerivativeStep]]]
_Run = Generator[DerivativeProblem, DerivativeStep, Optional[DerivativeStep]]


def _constant(problem: DerivativeProblem) -> Optional[DerivativeStep]:
    return ConstantRule(problem.expression, problem.variable)


def _variable(problem: DerivativeProblem) -> Optional[DerivativeStep]:
    return VariableRule(problem.expression, problem.variable)


def _power(problem: DerivativeProblem) -> Optional[DerivativeStep]:
    expression = problem.expression
    if isinstance(expression, Pow):
        return PowerRule(expression, problem.variable, expression.exp)
    return None


def _exponential(problem: DerivativeProblem) -> Optional[DerivativeStep]:
    expression = problem.expression
    if isinstance(expression, Pow):
        return ExponentialRule(expression, problem.variable, expression.base)
    return None


def _function(problem: DerivativeProblem) -> Optional[DerivativeStep]:
    name = _NAMES.get(type(problem.expression))
    return None if name is None else FunctionRule(problem.expression, problem.variable, name)


def _constant_multiple(problem: DerivativeProblem, context: Context) -> _Run:
    constant, rest = problem.expression.as_independent(problem.variable)
    substep = yield DerivativeProblem(rest, problem.variable)
    return ConstantMultipleRule(problem.expression, problem.variable, constant, rest, substep)


def _sum(problem: DerivativeProblem, context: Context) -> _Run:
    terms: list[DerivativeStep] = []
    for term in problem.expression.args:
        if not isinstance(term, Expr):
            return None
        terms.append((yield DerivativeProblem(term, problem.variable)))
    return SumRule(problem.expression, problem.variable, tuple(terms))


def _split_product(expression: Expr) -> tuple[Expr, Expr]:
    factors = [a for a in expression.args if isinstance(a, Expr)]
    return factors[0], mul(*factors[1:])


def _product(problem: DerivativeProblem, context: Context) -> _Run:
    first, second = _split_product(problem.expression)
    first_step = yield DerivativeProblem(first, problem.variable)
    second_step = yield DerivativeProblem(second, problem.variable)
    return ProductRule(problem.expression, problem.variable, first, second, first_step, second_step)


def _quotient(problem: DerivativeProblem, context: Context) -> _Run:
    numerator, denominator = problem.expression.as_numer_denom()
    numerator_step = yield DerivativeProblem(numerator, problem.variable)
    denominator_step = yield DerivativeProblem(denominator, problem.variable)
    return QuotientRule(problem.expression, problem.variable, numerator, denominator, numerator_step, denominator_step)


def _quotient_as_product(problem: DerivativeProblem, context: Context) -> _Run:
    numerator, denominator = problem.expression.as_numer_denom()
    reciprocal = pow(denominator, Integer(-1))
    first_step = yield DerivativeProblem(numerator, problem.variable)
    second_step = yield DerivativeProblem(reciprocal, problem.variable)
    return ProductRule(problem.expression, problem.variable, numerator, reciprocal, first_step, second_step)


def _placeholder(expression: Expr, variable: Symbol) -> Symbol:
    """A symbol for the inner function: ``u``, or ``u1``, ... when the
    name is taken (the same for the same problem, so that equal
    subproblems are recognised)."""
    used = {s.name for s in expression.free_symbols if isinstance(s, Symbol)} | {variable.name}
    name, k = 'u', 0
    while name in used:
        k += 1
        name = 'u%d' % k
    return Symbol(name)


def _chain(problem: DerivativeProblem, context: Context) -> _Run:
    expression, x = problem.expression, problem.variable
    u = _placeholder(expression, x)
    outer: Expr
    inner: Expr
    if isinstance(expression, Pow):
        if not expression.exp.has(x):
            inner, outer = expression.base, pow(u, expression.exp)
        elif not expression.base.has(x):
            inner, outer = expression.exp, pow(expression.base, u)
        else:
            return None
    elif type(expression) in _NAMES and len(expression.args) == 1 and isinstance(expression.args[0], Expr):
        inner = expression.args[0]
        outer = _apply(_NAMES[type(expression)], u)
    else:
        return None
    outer_step = yield DerivativeProblem(outer, u)
    inner_step = yield DerivativeProblem(inner, x)
    return ChainRule(expression, x, inner, u, outer_step, inner_step)


def _apply(name: str, u: Expr) -> Expr:
    """The elementary function ``name`` at ``u``, unevaluated."""
    for cls, known in _NAMES.items():
        if known == name:
            node = cls(u)
            if isinstance(node, Expr):
                return node
    raise ValueError('unknown function %s' % (name,))


def _general_power(problem: DerivativeProblem, context: Context) -> _Run:
    expression = problem.expression
    if not isinstance(expression, Pow):
        return None
    rewritten = exp(expression.exp*log(expression.base))
    substep = yield DerivativeProblem(rewritten, problem.variable)
    return RewriteRule(expression, problem.variable, rewritten, substep)


def _candidates(problem: DerivativeProblem) -> Sequence[_Rule]:
    expression, x = problem.expression, problem.variable
    if not expression.has(x):
        return [atomic(_constant)]
    if expression == x:
        return [atomic(_variable)]
    if isinstance(expression, Add):
        return [_sum]
    if isinstance(expression, Mul):
        constant, _ = expression.as_independent(x)
        if constant != 1:
            return [_constant_multiple]
        _, denominator = expression.as_numer_denom()
        if denominator != 1:
            return [_quotient, _quotient_as_product]
        return [_product]
    if isinstance(expression, Pow):
        if expression.base == x and not expression.exp.has(x):
            return [atomic(_power)]
        if expression.exp == x and not expression.base.has(x):
            return [atomic(_exponential)]
        if expression.base.has(x) and expression.exp.has(x):
            return [_general_power]
        return [_chain]
    if type(expression) in _NAMES and len(expression.args) == 1:
        return [atomic(_function)] if expression.args[0] == x else [_chain]
    return []


def _dont_know(problem: DerivativeProblem) -> DerivativeStep:
    return DontKnowRule(problem.expression, problem.variable)


def _alternative(problem: DerivativeProblem, alternatives: Sequence[DerivativeStep]) -> DerivativeStep:
    return AlternativeRule(problem.expression, problem.variable, tuple(alternatives))


def derivative_steps(f: ExprLike, x: Symbol, branch: bool = False, max_depth: Optional[int] = 100) -> DerivativeStep:
    """The derivation of the derivative of ``f`` with respect to ``x``,
    as a tree of rules (see the module). With ``branch`` every workable
    rule is kept at each level, in :class:`AlternativeRule` nodes;
    ``max_depth`` limits the depth of the tree.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.steps.derivatives import derivative_steps
    >>> x = Symbol('x')
    >>> derivative_steps(5*x**3, x)
    ConstantMultipleRule(expression=5*x**3, variable=x, constant=5, rest=x**3, substep=PowerRule(expression=x**3, variable=x, exponent=3))
    >>> derivative_steps(x**x, x).result()
    (log(x) + 1)*exp(x*log(x))
    """
    solver: StepSolver[DerivativeProblem, DerivativeStep] = StepSolver(_candidates, _dont_know, _alternative, branch, max_depth)
    return solver.solve(DerivativeProblem(to_expr(f), x))


def manual_derivative(f: ExprLike, x: Symbol) -> Expr:
    """The derivative found by the rules, as they give it (not
    simplified); a ``Derivative`` is left where no rule applies.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import sin, cos
    >>> from sympy_extras.steps.derivatives import manual_derivative
    >>> x = Symbol('x')
    >>> manual_derivative(sin(x)*cos(x), x), manual_derivative(x**2/(x + 1), x)
    (-sin(x)**2 + cos(x)**2, (-x**2 + 2*x*(x + 1))/(x + 1)**2)
    """
    return derivative_steps(f, x).result()


