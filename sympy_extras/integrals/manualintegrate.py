r"""Integration by the rules a student applies by hand, with the steps.

The integrand is matched against a table of rules (powers, exponentials,
the trigonometric and hyperbolic functions and their inverses, the
special functions whose derivatives are elementary) and, when none
applies, transformed by the general techniques of a calculus course:

- linearity (sums, constant factors);
- substitution `u = g(x)` for the subterms `g` whose derivative divides
  the integrand (the method of SIN [Moses71]_), and `u = (a + b x)^{1/n}`
  for the roots of a linear function;
- integration by parts with the LIATE choice of `u` (logarithmic,
  inverse trigonometric, algebraic, trigonometric, exponential), cyclic
  when the integral comes back (`e^x \sin x`);
- the trigonometric identities for powers and products of sines and
  cosines, tangents and secants, cotangents and cosecants, and the
  substitutions `x = \sqrt{a/b} \tan\theta`, `\sin\theta`, `\sec\theta`
  for `a + b x^2` under a root;
- the completed square for `1/\sqrt{a + b x + c x^2}`, the recursion for a
  polynomial over that root, and partial fractions, cancellation and
  expansion as rewritings.

The result of :func:`integral_steps` is the tree of the steps: every
rule is an immutable dataclass recording what it did (the substitution
of a :class:`URule`, the `u` and `dv` of a :class:`PartsRule`) and the
steps of its subproblems, and ``eval()`` builds the antiderivative from
it; :func:`manualintegrate` evaluates the tree. A tree with a
:class:`DontKnowRule` leaves an unevaluated
:class:`~sympy_extras.integrals.integral.Integral` in place of the part
it could not do.

This is a port of SymPy's ``sympy.integrals.manualintegrate`` (SymPy
1.14), written on the generator design of SymPy's development branch: a
rule is a generator function which *yields* the integrals it needs and
receives their steps back, and :class:`IntegralSolver` drives the rules,
breaks the cycles (an integral met again while it is being solved gets a
:class:`DontKnowRule`) and remembers the integrals it solved completely.
The patterns SymPy matches with wildcards (linear and quadratic
arguments, powers of sines and cosines) are read off the coefficients of
the polynomials instead, so that they do not depend on the matcher.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native ``apart``, ``cancel``, ``simplify`` and
``trigsimp``; the substitution of the trigonometric rule is inverted
directly instead of by ``solve``.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import sin, cos, exp, log, tan
>>> from sympy_extras.integrals.manualintegrate import manualintegrate, integral_steps
>>> x = symbols('x')
>>> manualintegrate(1/x, x)
log(x)
>>> manualintegrate(log(x), x)
x*log(x) - x
>>> manualintegrate(exp(x)/(1 + exp(2*x)), x)
atan(exp(x))
>>> manualintegrate(cos(x)**4*sin(x)**3, x)
cos(x)**7/7 - cos(x)**5/5
>>> manualintegrate(tan(x), x)
-log(cos(x))
>>> integral_steps(sin(x), x)
SinRule(integrand=sin(x), variable=x)

References
==========

.. [Moses71] J. Moses, *Symbolic Integration: The Stormy Decade*,
   Communications of the ACM 14 (1971), 548-560.
.. [SymPyManual] SymPy Development Team, ``sympy/integrals/manualintegrate.py``
   (SymPy 1.14, and the ``IntegrationSolver`` of the development branch).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; frozen dataclasses for the rules; generator rules
# driven by IntegralSolver (SymPy master's design) instead of global caches;
# the wildcard patterns read off polynomial coefficients; no solve.
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Final, Generator, Optional, Sequence

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Derivative, expand
from sympy_extras.core.logic import Boolean, BooleanFalse, BooleanTrue, and_, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import E, Integer, Number, Rational, pi
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq, eq, gt, lt, ne
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import abs_
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Cosh, Coth, Csch, Sech, Sinh, Tanh, asinh, cosh, sinh,
                                                          tanh)
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.elementary.trigonometric import (Acos, Acot, Acsc, Asec, Asin, Atan, Cos, Cot, Csc, Sec,
                                                             Sin, Tan, acos, asin, atan, cos, cot, csc, sec, sin,
                                                             tan)
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside, dirac_delta, heaviside
from sympy_extras.functions.special.elliptic_integrals import elliptic_e, elliptic_f
from sympy_extras.functions.special.error_functions import Chi, Ci, Ei, Shi, Si, erf, erfi, fresnelc, fresnels, li
from sympy_extras.functions.special.gamma_functions import uppergamma
from sympy_extras.functions.special.polynomials import (AssocLaguerre, ChebyshevT, ChebyshevU, Gegenbauer, Hermite,
                                                        Jacobi, Laguerre, Legendre, assoc_laguerre, chebyshevt,
                                                        gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy_extras.functions.special.zeta_functions import PolyLog, polylog
from sympy_extras.ntheory.factor_ import factorint
from sympy_extras.polys.polyerrors import BasePolynomialError

from .integral import Integral

__all__ = ['Rule', 'AtomicRule', 'ConstantRule', 'ConstantTimesRule', 'PowerRule', 'NestedPowRule', 'AddRule',
           'URule', 'PartsRule', 'CyclicPartsRule', 'SinRule', 'CosRule', 'SecTanRule', 'CscCotRule', 'Sec2Rule',
           'Csc2Rule', 'SinhRule', 'CoshRule', 'ExpRule', 'ReciprocalRule', 'ArcsinRule', 'ArcsinhRule',
           'ReciprocalSqrtQuadraticRule', 'SqrtQuadraticDenomRule', 'SqrtQuadraticRule', 'AlternativeRule',
           'DontKnowRule', 'DerivativeRule', 'RewriteRule', 'CompleteSquareRule', 'PiecewiseRule', 'HeavisideRule',
           'DiracDeltaRule', 'TrigSubstitutionRule', 'ArctanRule', 'JacobiRule', 'GegenbauerRule', 'ChebyshevTRule',
           'ChebyshevURule', 'LegendreRule', 'HermiteRule', 'LaguerreRule', 'AssocLaguerreRule', 'CiRule', 'ChiRule',
           'EiRule', 'SiRule', 'ShiRule', 'LiRule', 'ErfRule', 'FresnelCRule', 'FresnelSRule', 'PolylogRule',
           'UpperGammaRule', 'EllipticFRule', 'EllipticERule', 'IntegralInfo', 'IntegralSolver', 'integral_steps',
           'manualintegrate', 'manual_diff', 'manual_subs', 'find_substitutions']

_COUNTER: Final = itertools.count()


def _fresh(name: str) -> Dummy:
    """A dummy with a name of its own (the typed polynomials identify
    their symbols by name)."""
    return Dummy('%s_manual_%d' % (name, next(_COUNTER)))


def _is_true(b: Boolean) -> bool:
    return isinstance(b, BooleanTrue)


def _is_false(b: Boolean) -> bool:
    return isinstance(b, BooleanFalse)


def _zero(e: Expr) -> bool:
    """Whether ``e`` is zero once simplified (SymPy's ``simplify(e) == 0``),
    the numbers compared at once."""
    if isinstance(e, Number):
        return e == 0
    if e.is_zero is True:
        return True
    from sympy_extras.simplify.simplify import simplify
    return simplify(e) == 0


def _same(a: Expr, b: Expr) -> bool:
    """``a.equals(b)`` without the bridge: equal, or a difference which
    cancels to zero."""
    if a == b:
        return True
    from sympy_extras.polys.polytools import cancel
    try:
        return cancel(a - b) == 0
    except (NotImplementedError, ValueError, TypeError, ZeroDivisionError):
        return False


# ---------------------------------------------------------------------------
# the rules

# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:Rule
@dataclass(frozen=True)
class Rule(ABC):
    """A step of the integration of ``integrand`` in ``variable``: the
    class is the rule, the fields what it needs; ``eval()`` is the
    antiderivative."""
    integrand: Expr
    variable: Symbol

    @abstractmethod
    def eval(self) -> Expr:
        """The antiderivative this step gives."""

    @abstractmethod
    def contains_dont_know(self) -> bool:
        """Whether some part of the integral was left undone."""


@dataclass(frozen=True)
class AtomicRule(Rule, ABC):
    """A simple rule that does not depend on other rules."""

    def contains_dont_know(self) -> bool:
        return False


@dataclass(frozen=True)
class ConstantRule(AtomicRule):
    """integrate(a, x)  ->  a*x"""

    def eval(self) -> Expr:
        return self.integrand * self.variable


@dataclass(frozen=True)
class ConstantTimesRule(Rule):
    """integrate(a*f(x), x)  ->  a*integrate(f(x), x)"""
    constant: Expr
    other: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.constant * self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass(frozen=True)
class PowerRule(AtomicRule):
    """integrate(x**a, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return piecewise(((self.base**(self.exp + 1))/(self.exp + 1), ne(self.exp, -1)), (log(self.base), true))


@dataclass(frozen=True)
class NestedPowRule(AtomicRule):
    """integrate((x**a)**b, x)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        m = self.base * self.integrand
        return piecewise((m / (self.exp + 1), ne(self.exp, -1)), (m * log(self.base), true))


@dataclass(frozen=True)
class AddRule(Rule):
    """integrate(f(x) + g(x), x) -> integrate(f(x), x) + integrate(g(x), x)"""
    substeps: tuple[Rule, ...]

    def eval(self) -> Expr:
        return add(*(substep.eval() for substep in self.substeps))

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.substeps)


@dataclass(frozen=True)
class URule(Rule):
    """integrate(f(g(x))*g'(x), x) -> integrate(f(u), u), u = g(x)"""
    u_var: Symbol
    u_func: Expr
    substep: Rule

    def eval(self) -> Expr:
        result = self.substep.eval()
        if isinstance(self.u_func, Pow):
            base, exp_ = self.u_func.as_base_exp()
            if exp_ == -1:
                # avoid needless -log(1/x) from substitution
                result = result.subs(log(self.u_var), -log(base))
        return result.subs(self.u_var, self.u_func)

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass(frozen=True)
class PartsRule(Rule):
    """integrate(u(x)*v'(x), x) -> u(x)*v(x) - integrate(u'(x)*v(x), x)"""
    u: Expr
    dv: Expr
    v_step: Rule
    second_step: Optional[Rule]  # None when a substep of CyclicPartsRule

    def eval(self) -> Expr:
        if self.second_step is None:
            raise ValueError('a part of a cyclic integration by parts has no value of its own')
        v = self.v_step.eval()
        return self.u * v - self.second_step.eval()

    def contains_dont_know(self) -> bool:
        return self.v_step.contains_dont_know() or (
            self.second_step is not None and self.second_step.contains_dont_know())


@dataclass(frozen=True)
class CyclicPartsRule(Rule):
    """Apply PartsRule multiple times to integrate exp(x)*sin(x)"""
    parts_rules: tuple[PartsRule, ...]
    coefficient: Expr

    def eval(self) -> Expr:
        result: list[Expr] = []
        sign = 1
        for rule in self.parts_rules:
            result.append(sign * rule.u * rule.v_step.eval())
            sign *= -1
        return add(*result) / (1 - self.coefficient)

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.parts_rules)


@dataclass(frozen=True)
class TrigRule(AtomicRule, ABC):
    pass


@dataclass(frozen=True)
class SinRule(TrigRule):
    """integrate(sin(x), x) -> -cos(x)"""

    def eval(self) -> Expr:
        return -cos(self.variable)


@dataclass(frozen=True)
class CosRule(TrigRule):
    """integrate(cos(x), x) -> sin(x)"""

    def eval(self) -> Expr:
        return sin(self.variable)


@dataclass(frozen=True)
class SecTanRule(TrigRule):
    """integrate(sec(x)*tan(x), x) -> sec(x)"""

    def eval(self) -> Expr:
        return sec(self.variable)


@dataclass(frozen=True)
class CscCotRule(TrigRule):
    """integrate(csc(x)*cot(x), x) -> -csc(x)"""

    def eval(self) -> Expr:
        return -csc(self.variable)


@dataclass(frozen=True)
class Sec2Rule(TrigRule):
    """integrate(sec(x)**2, x) -> tan(x)"""

    def eval(self) -> Expr:
        return tan(self.variable)


@dataclass(frozen=True)
class Csc2Rule(TrigRule):
    """integrate(csc(x)**2, x) -> -cot(x)"""

    def eval(self) -> Expr:
        return -cot(self.variable)


@dataclass(frozen=True)
class HyperbolicRule(AtomicRule, ABC):
    pass


@dataclass(frozen=True)
class SinhRule(HyperbolicRule):
    """integrate(sinh(x), x) -> cosh(x)"""

    def eval(self) -> Expr:
        return cosh(self.variable)


@dataclass(frozen=True)
class CoshRule(HyperbolicRule):
    """integrate(cosh(x), x) -> sinh(x)"""

    def eval(self) -> Expr:
        return sinh(self.variable)


@dataclass(frozen=True)
class ExpRule(AtomicRule):
    """integrate(a**x, x) -> a**x/ln(a)"""
    base: Expr
    exp: Expr

    def eval(self) -> Expr:
        return self.integrand / log(self.base)


@dataclass(frozen=True)
class ReciprocalRule(AtomicRule):
    """integrate(1/x, x) -> ln(x)"""
    base: Expr

    def eval(self) -> Expr:
        return log(self.base)


@dataclass(frozen=True)
class ArcsinRule(AtomicRule):
    """integrate(1/sqrt(1-x**2), x) -> asin(x)"""

    def eval(self) -> Expr:
        return asin(self.variable)


@dataclass(frozen=True)
class ArcsinhRule(AtomicRule):
    """integrate(1/sqrt(1+x**2), x) -> asin(x)"""

    def eval(self) -> Expr:
        return asinh(self.variable)


@dataclass(frozen=True)
class ReciprocalSqrtQuadraticRule(AtomicRule):
    """integrate(1/sqrt(a+b*x+c*x**2), x) -> log(2*sqrt(c)*sqrt(a+b*x+c*x**2)+b+2*c*x)/sqrt(c)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return log(2*sqrt(c)*sqrt(a + b*x + c*x**2) + b + 2*c*x)/sqrt(c)


@dataclass(frozen=True)
class SqrtQuadraticDenomRule(AtomicRule):
    """integrate(poly(x)/sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr
    coeffs: tuple[Expr, ...]

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        # Integrate poly/sqrt(a+b*x+c*x**2) using recursion.
        # coeffs are coefficients of the polynomial.
        # Let I_n = x**n/sqrt(a+b*x+c*x**2), then
        # I_n = A * x**(n-1)*sqrt(a+b*x+c*x**2) - B * I_{n-1} - C * I_{n-2}
        # where A = 1/(n*c), B = (2*n-1)*b/(2*n*c), C = (n-1)*a/(n*c)
        # See https://github.com/sympy/sympy/pull/23608 for proof.
        result_coeffs: list[Expr] = []
        coeffs = list(self.coeffs)
        for i in range(len(coeffs) - 2):
            n = len(coeffs) - 1 - i
            coeff = coeffs[i]/(c*n)
            result_coeffs.append(coeff)
            coeffs[i + 1] -= (2*n - 1)*b/2*coeff
            coeffs[i + 2] -= (n - 1)*a*coeff
        d, e = coeffs[-1], coeffs[-2]
        s = sqrt(a + b*x + c*x**2)
        constant = d - b*e/(2*c)
        I0: Expr = S.Zero
        if constant != 0:
            step = _inverse_trig_generic(1/s, x)
            if step is None:
                raise ValueError('no inverse trigonometric rule for %s' % (1/s))
            I0 = constant*step.eval()
        return add(*(result_coeffs[i]*x**(len(coeffs) - 2 - i) for i in range(len(result_coeffs))), e/c)*s + I0


@dataclass(frozen=True)
class SqrtQuadraticRule(AtomicRule):
    """integrate(sqrt(a+b*x+c*x**2), x)"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        step = _sqrt_quadratic_generic(self.integrand, self.variable)
        if step is None:
            raise ValueError('no rule for %s' % self.integrand)
        return step.eval()


@dataclass(frozen=True)
class AlternativeRule(Rule):
    """Multiple ways to do integration."""
    alternatives: tuple[Rule, ...]

    def eval(self) -> Expr:
        return self.alternatives[0].eval()

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep in self.alternatives)


@dataclass(frozen=True)
class DontKnowRule(Rule):
    """Leave the integral as is."""

    def eval(self) -> Expr:
        return Integral(self.integrand, self.variable)

    def contains_dont_know(self) -> bool:
        return True


@dataclass(frozen=True)
class DerivativeRule(AtomicRule):
    """integrate(f'(x), x) -> f(x)"""

    def eval(self) -> Expr:
        if not isinstance(self.integrand, Derivative):
            raise TypeError('not a derivative: %s' % self.integrand)
        specs: list[tuple[Symbol, int]] = []
        done = False
        for var, count in self.integrand.variable_count:
            if var == self.variable and not done:
                count -= 1
                done = True
            if count:
                specs.append((var, count))
        # the node, as SymPy's Derivative(...): d/dy of sin(x) stays
        return Derivative(self.integrand.expr, *specs) if specs else self.integrand.expr


@dataclass(frozen=True)
class RewriteRule(Rule):
    """Rewrite integrand to another form that is easier to handle."""
    rewritten: Expr
    substep: Rule

    def eval(self) -> Expr:
        return self.substep.eval()

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass(frozen=True)
class CompleteSquareRule(RewriteRule):
    """Rewrite a+b*x+c*x**2 to a-b**2/(4*c) + c*(x+b/(2*c))**2"""
    pass


#: a piece of a :class:`PiecewiseRule`: the step and the condition under which it holds
Piece = tuple[Rule, Boolean]


@dataclass(frozen=True)
class PiecewiseRule(Rule):
    subfunctions: tuple[Piece, ...]

    def eval(self) -> Expr:
        return piecewise(*[(substep.eval(), cond) for substep, cond in self.subfunctions])

    def contains_dont_know(self) -> bool:
        return any(substep.contains_dont_know() for substep, _ in self.subfunctions)


@dataclass(frozen=True)
class HeavisideRule(Rule):
    harg: Expr
    ibnd: Expr
    substep: Rule

    def eval(self) -> Expr:
        # If we are integrating over x and the integrand has the form
        #       Heaviside(m*x+b)*g(x) == Heaviside(harg)*g(symbol)
        # then there needs to be continuity at -b/m == ibnd,
        # so we subtract the appropriate term.
        result = self.substep.eval()
        return heaviside(self.harg) * (result - result.subs(self.variable, self.ibnd))

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


@dataclass(frozen=True)
class DiracDeltaRule(AtomicRule):
    n: Expr
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        if n == 0:
            return heaviside(a + b*x)/b
        return dirac_delta(a + b*x, n - 1)/b


@dataclass(frozen=True)
class TrigSubstitutionRule(Rule):
    theta: Symbol
    func: Expr
    rewritten: Expr
    substep: Rule
    restriction: Boolean

    def eval(self) -> Expr:
        from sympy_extras.simplify.radsimp import fraction
        from sympy_extras.simplify.trigsimp import trigsimp
        theta, func, x = self.theta, self.func, self.variable
        func = func.subs(sec(theta), 1/cos(theta))
        func = func.subs(csc(theta), 1/sin(theta))
        func = func.subs(cot(theta), 1/tan(theta))
        trig_functions = [f for f in func.atoms(Sin, Cos, Tan)]
        if len(trig_functions) != 1:
            raise ValueError('not a trigonometric substitution: %s' % func)
        trig_function = as_expr(trig_functions[0])
        # SymPy's solve(x - func, trig_function): func is c*t or c/t
        constant, dependent = func.as_independent(theta, as_Add=False)
        if dependent == trig_function:
            relation = x/constant
        elif dependent == 1/trig_function:
            relation = constant/x
        else:
            raise ValueError('not a trigonometric substitution: %s' % func)
        numer, denom = fraction(relation)

        if isinstance(trig_function, Sin):
            opposite = numer
            hypotenuse = denom
            adjacent = sqrt(denom**2 - numer**2)
            inverse = asin(relation)
        elif isinstance(trig_function, Cos):
            adjacent = numer
            hypotenuse = denom
            opposite = sqrt(denom**2 - numer**2)
            inverse = acos(relation)
        else:  # tan
            opposite = numer
            adjacent = denom
            hypotenuse = sqrt(denom**2 + numer**2)
            inverse = atan(relation)

        substitution: list[tuple[Basic, Basic]] = [
            (sin(theta), opposite/hypotenuse),
            (cos(theta), adjacent/hypotenuse),
            (tan(theta), opposite/adjacent),
            (theta, inverse)
        ]
        # the multiple angles expanded first: SymPy's trigsimp writes
        # sin(2*atan(x)) as 2*x/(x**2 + 1) afterwards, the native one does
        # not, nor does the core evaluate sin(atan(x))
        result = expand(self.substep.eval(), trig=True).subs(substitution)
        return piecewise((trigsimp(_inverse_compositions(result)), self.restriction))

    def contains_dont_know(self) -> bool:
        return self.substep.contains_dont_know()


def _inverse_composition(node: Basic) -> Basic:
    """A trigonometric function of an inverse trigonometric function, as
    SymPy's ``sin(atan(z))`` evaluates (the core does not)."""
    if not isinstance(node, (Sin, Cos, Tan)) or not isinstance(node.args[0], (Asin, Acos, Atan)):
        return node
    inner = node.args[0]
    z = as_expr(inner.args[0])
    if isinstance(inner, Atan):
        values = (z/sqrt(1 + z**2), 1/sqrt(1 + z**2), z)
    elif isinstance(inner, Asin):
        values = (z, sqrt(1 - z**2), z/sqrt(1 - z**2))
    else:
        values = (sqrt(1 - z**2), z, sqrt(1 - z**2)/z)
    if isinstance(node, Sin):
        return values[0]
    if isinstance(node, Cos):
        return values[1]
    return values[2]


def _inverse_compositions(e: Expr) -> Expr:
    def composed(node: Basic) -> bool:
        return isinstance(node, (Sin, Cos, Tan)) and isinstance(node.args[0], (Asin, Acos, Atan))
    return as_expr(e.replace(composed, _inverse_composition))


@dataclass(frozen=True)
class ArctanRule(AtomicRule):
    """integrate(a/(b*x**2+c), x) -> a/b / sqrt(c/b) * atan(x/sqrt(c/b))"""
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return a/b / sqrt(c/b) * atan(x/sqrt(c/b))


@dataclass(frozen=True)
class OrthogonalPolyRule(AtomicRule, ABC):
    n: Expr


@dataclass(frozen=True)
class JacobiRule(OrthogonalPolyRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        n, a, b, x = self.n, self.a, self.b, self.variable
        return piecewise(
            (2*jacobi(n + 1, a - 1, b - 1, x)/(n + a + b), ne(n + a + b, 0)),
            (x, eq(n, 0)),
            ((a + b + 2)*x**2/4 + (a - b)*x/2, eq(n, 1)))


@dataclass(frozen=True)
class GegenbauerRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        n, a, x = self.n, self.a, self.variable
        return piecewise(
            (gegenbauer(n + 1, a - 1, x)/(2*(a - 1)), ne(a, 1)),
            (chebyshevt(n + 1, x)/(n + 1), ne(n, -1)),
            (S.Zero, true))


@dataclass(frozen=True)
class ChebyshevTRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return piecewise(
            ((chebyshevt(n + 1, x)/(n + 1) - chebyshevt(n - 1, x)/(n - 1))/2, ne(abs_(n), 1)),
            (x**2/2, true))


@dataclass(frozen=True)
class ChebyshevURule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return piecewise(
            (chebyshevt(n + 1, x)/(n + 1), ne(n, -1)),
            (S.Zero, true))


@dataclass(frozen=True)
class LegendreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return (legendre(n + 1, x) - legendre(n - 1, x))/(2*n + 1)


@dataclass(frozen=True)
class HermiteRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return hermite(n + 1, x)/(2*(n + 1))


@dataclass(frozen=True)
class LaguerreRule(OrthogonalPolyRule):
    def eval(self) -> Expr:
        n, x = self.n, self.variable
        return laguerre(n, x) - laguerre(n + 1, x)


@dataclass(frozen=True)
class AssocLaguerreRule(OrthogonalPolyRule):
    a: Expr

    def eval(self) -> Expr:
        return -assoc_laguerre(self.n + 1, self.a - 1, self.variable)


@dataclass(frozen=True)
class IRule(AtomicRule, ABC):
    a: Expr
    b: Expr


@dataclass(frozen=True)
class CiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cos(b)*Ci(a*x) - sin(b)*Si(a*x)


@dataclass(frozen=True)
class ChiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return cosh(b)*Chi(a*x) + sinh(b)*Shi(a*x)


@dataclass(frozen=True)
class EiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return exp(b)*Ei(a*x)


@dataclass(frozen=True)
class SiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sin(b)*Ci(a*x) + cos(b)*Si(a*x)


@dataclass(frozen=True)
class ShiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return sinh(b)*Chi(a*x) + cosh(b)*Shi(a*x)


@dataclass(frozen=True)
class LiRule(IRule):
    def eval(self) -> Expr:
        a, b, x = self.a, self.b, self.variable
        return li(a*x + b)/a


@dataclass(frozen=True)
class ErfRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        if a.is_extended_real:
            return piecewise(
                (sqrt(pi)/sqrt(-a)/2 * exp(c - b**2/(4*a)) * erf((-2*a*x - b)/(2*sqrt(-a))), lt(a, 0)),
                (sqrt(pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) * erfi((2*a*x + b)/(2*sqrt(a))), true))
        return sqrt(pi)/sqrt(a)/2 * exp(c - b**2/(4*a)) * erfi((2*a*x + b)/(2*sqrt(a)))


@dataclass(frozen=True)
class FresnelCRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*pi)) +
            sin(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*pi)))


@dataclass(frozen=True)
class FresnelSRule(AtomicRule):
    a: Expr
    b: Expr
    c: Expr

    def eval(self) -> Expr:
        a, b, c, x = self.a, self.b, self.c, self.variable
        return sqrt(pi)/sqrt(2*a) * (
            cos(b**2/(4*a) - c)*fresnels((2*a*x + b)/sqrt(2*a*pi)) -
            sin(b**2/(4*a) - c)*fresnelc((2*a*x + b)/sqrt(2*a*pi)))


@dataclass(frozen=True)
class PolylogRule(AtomicRule):
    a: Expr
    b: Expr

    def eval(self) -> Expr:
        return polylog(self.b + 1, self.a * self.variable)


@dataclass(frozen=True)
class UpperGammaRule(AtomicRule):
    a: Expr
    e: Expr

    def eval(self) -> Expr:
        a, e, x = self.a, self.e, self.variable
        return x**e * (-a*x)**(-e) * uppergamma(e + 1, -a*x)/a


@dataclass(frozen=True)
class EllipticFRule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_f(self.variable, self.d/self.a)/sqrt(self.a)


@dataclass(frozen=True)
class EllipticERule(AtomicRule):
    a: Expr
    d: Expr

    def eval(self) -> Expr:
        return elliptic_e(self.variable, self.d/self.a)*sqrt(self.a)


@dataclass(frozen=True)
class IntegralInfo:
    """An integral to find: the problem a rule yields to the solver."""
    integrand: Expr
    symbol: Symbol


# ---------------------------------------------------------------------------
# the engine

#: a rule: a generator which yields the integrals it needs, receives their
#: steps and returns its own step, or ``None`` when it does not apply
Steps = Generator[IntegralInfo, Rule, Optional[Rule]]
Strategy = Callable[[IntegralInfo, 'IntegralSolver'], Steps]


def _atomic(function: Callable[[IntegralInfo], Optional[Rule]]) -> Strategy:
    """A strategy without subproblems, from a plain function."""
    def strategy(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
        return function(integral)
        yield integral  # never reached: makes the function a generator
    return strategy


class IntegralSolver:
    """The state of one integration: the integrals being solved (an
    integral met again on its own path is cut, SymPy's ``_integral_cache``),
    the number of times each `u` of an integration by parts was used
    (``_parts_u_cache``), and the steps of the integrals solved
    completely, which do not depend on the path."""

    def __init__(self) -> None:
        self._cache_dummy: Final[Dummy] = _fresh('z')
        self._active: Final[set[Expr]] = set()
        self._parts_u: Final[dict[Expr, int]] = {}
        self._solved: Final[dict[IntegralInfo, Rule]] = {}

    def key(self, e: Expr, symbol: Symbol) -> Expr:
        return e.xreplace({symbol: self._cache_dummy})

    def use_parts_u(self, u: Expr, symbol: Symbol) -> bool:
        """Count a use of ``u`` in an integration by parts; ``False`` when
        it was used too often already."""
        cachekey = self.key(u, symbol)
        count = self._parts_u.get(cachekey, 0)
        if count > 2:
            return False
        self._parts_u[cachekey] = count + 1
        return True

    def solve(self, integral: IntegralInfo) -> Rule:
        """The steps of ``integral`` (SymPy's ``integral_steps``)."""
        known = self._solved.get(integral)
        if known is not None:
            return known
        cachekey = self.key(integral.integrand, integral.symbol)
        if cachekey in self._active:
            # Stop this attempt, because it leads around in a loop
            return DontKnowRule(integral.integrand, integral.symbol)
        self._active.add(cachekey)
        try:
            result = self.run(_integral_steps(integral, self))
        finally:
            self._active.discard(cachekey)
        if result is None:
            result = DontKnowRule(integral.integrand, integral.symbol)
        if not result.contains_dont_know():
            self._solved[integral] = result
        return result

    def run(self, steps: Steps) -> Optional[Rule]:
        """Drive a rule: solve the integrals it yields and send their steps
        back, until it returns."""
        try:
            request = next(steps)
            while True:
                request = steps.send(self.solve(request))
        except StopIteration as finished:
            value: Optional[Rule] = finished.value
            return value


def _steps(integrand: Expr, symbol: Symbol) -> Generator[IntegralInfo, Rule, Rule]:
    """The steps of a subproblem: ``rule = yield from _steps(f, x)`` in a
    strategy is SymPy's ``rule = integral_steps(f, x)``."""
    rule = yield IntegralInfo(integrand, symbol)
    return rule


# ---------------------------------------------------------------------------
# polynomial patterns (SymPy's wildcards a + b*x + c*x**2)

def _coefficients(e: Expr, x: Symbol) -> Optional[list[Expr]]:
    """The coefficients of ``e`` as a polynomial in ``x``, of the lowest
    degree first; ``None`` when it is not one."""
    if e.is_polynomial(x) is not True:
        return None
    coefficients: dict[int, Expr] = {}
    for term in Add.make_args(expand(e)):
        coeff, dependent = term.as_independent(x, as_Add=False)
        if dependent == 1:
            k = 0
        elif dependent == x:
            k = 1
        elif isinstance(dependent, Pow) and dependent.base == x and isinstance(dependent.exp, Integer) \
                and int(dependent.exp) > 0:
            k = int(dependent.exp)
        else:
            return None
        coefficients[k] = coefficients.get(k, S.Zero) + coeff
    if not coefficients:
        return [S.Zero]
    return [coefficients.get(k, S.Zero) for k in range(max(coefficients) + 1)]


def _linear(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(a, b)`` with ``e = a + b*x`` and ``b`` nonzero."""
    coefficients = _coefficients(e, x)
    if coefficients is None or len(coefficients) != 2 or coefficients[1].is_zero is True:
        return None
    return coefficients[0], coefficients[1]


def _quadratic(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(a, b, c)`` with ``e = a + b*x + c*x**2`` and ``c`` nonzero."""
    coefficients = _coefficients(e, x)
    if coefficients is None or len(coefficients) != 3 or coefficients[2].is_zero is True:
        return None
    return coefficients[0], coefficients[1], coefficients[2]


def _times(x: Symbol, e: Expr) -> Optional[Expr]:
    """``a`` with ``e = a*x`` (SymPy's pattern ``a*x``, ``a`` free of ``x``)."""
    coefficients = _coefficients(e, x)
    if coefficients is None or len(coefficients) != 2 or coefficients[0] != 0:
        return None
    return coefficients[1]


def _is_integer(e: Expr) -> bool:
    return isinstance(e, Integer)


# ---------------------------------------------------------------------------
# substitutions

# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:manual_diff
def manual_diff(f: Expr, symbol: Symbol) -> Expr:
    """Derivative of f in form expected by find_substitutions

    SymPy's derivatives for some trig functions (like cot) are not in a form
    that works well with finding substitutions; this replaces the
    derivatives for those particular forms with something that works better.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cot
    >>> from sympy_extras.integrals.manualintegrate import manual_diff
    >>> x = symbols('x')
    >>> manual_diff(cot(2*x), x)
    -2*csc(2*x)**2
    """
    if f.args:
        arg = as_expr(f.args[0])
        if isinstance(f, Tan):
            return arg.diff(symbol) * sec(arg)**2
        elif isinstance(f, Cot):
            return -arg.diff(symbol) * csc(arg)**2
        elif isinstance(f, Sec):
            return arg.diff(symbol) * sec(arg) * tan(arg)
        elif isinstance(f, Csc):
            return -arg.diff(symbol) * csc(arg) * cot(arg)
        elif isinstance(f, Add):
            return add(*(manual_diff(as_expr(a), symbol) for a in f.args))
        elif isinstance(f, Mul):
            if len(f.args) == 2 and isinstance(f.args[0], Number):
                return f.args[0] * manual_diff(as_expr(f.args[1]), symbol)
    return f.diff(symbol)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:manual_subs
def manual_subs(expr: Expr, sequence: Sequence[tuple[Expr, Expr]]) -> Expr:
    """``expr.subs(sequence)`` with additional logic for substitution of
    invertible functions: for ``log(x) = y`` the powers of ``x`` become
    exponentials ``exp(a*y)``, and ``x`` itself ``exp(y)``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.manualintegrate import manual_subs
    >>> x, u = symbols('x u')
    >>> manual_subs(x**2*log(x), [(log(x), u)])
    u*exp(2*u)
    """
    new_subs: list[tuple[Basic, Basic]] = []
    for old, new in sequence:
        if isinstance(old, Log):
            # If log(x) = y, then exp(a*log(x)) = exp(a*y)
            # that is, x**a = exp(a*y). Replace nontrivial powers of x
            # before subs turns them into `exp(y)**a`, but
            # do not replace x itself yet, to avoid `log(exp(y))`.
            x0 = as_expr(old.args[0])

            def is_power(e: Basic, x0: Expr = x0) -> bool:
                return isinstance(e, Pow) and e.base == x0

            def power(e: Basic, new: Expr = new) -> Basic:
                return exp(as_expr(e.args[1]) * new)

            expr = as_expr(expr.replace(is_power, power))
            new_subs.append((x0, exp(new)))
    pairs: list[tuple[Basic, Basic]] = [(old, new) for old, new in sequence]
    return expr.subs(pairs + new_subs)


#: the inverse trigonometric functions
_INVERSE_TRIG: Final[tuple[type[Basic], ...]] = (Atan, Asin, Acos, Acot, Acsc, Asec)
#: SymPy's TrigonometricFunction
_TRIG: Final[tuple[type[Basic], ...]] = (Sin, Cos, Tan, Cot, Sec, Csc)
#: SymPy's HyperbolicFunction
_HYPERBOLIC: Final[tuple[type[Basic], ...]] = (Sinh, Cosh, Tanh, Coth, Sech, Csch)
#: SymPy's OrthogonalPolynomial, with the index of the variable among the arguments
_ORTHOGONAL: Final[dict[type[Basic], int]] = {Jacobi: 3, Gegenbauer: 2, ChebyshevT: 1, ChebyshevU: 1, Legendre: 1,
                                              Hermite: 1, Laguerre: 1, AssocLaguerre: 2}

#: a substitution: u, the constant factor and the integrand in u
Substitution = tuple[Expr, Expr, Expr]


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:find_substitutions
def find_substitutions(integrand: Expr, symbol: Symbol, u_var: Symbol) -> list[Substitution]:
    """The substitutions ``u = g(x)`` which take ``integrand`` to a
    function of ``u_var`` times a constant: ``(g, constant, new
    integrand)`` (the method of SIN).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cos, sin
    >>> from sympy_extras.integrals.manualintegrate import find_substitutions
    >>> x, u = symbols('x u')
    >>> find_substitutions(sin(x)*cos(x)**2, x, u)
    [(cos(x), -1, u**2)]
    """
    from sympy_extras.polys.polytools import degree
    results: list[Substitution] = []

    def test_subterm(u: Expr, u_diff: Expr) -> Optional[tuple[Expr, Expr]]:
        if u_diff == 0:
            return None
        substituted = integrand / u_diff
        substituted = manual_subs(substituted, [(u, u_var)]).cancel()

        if substituted.has_free(symbol):
            return None
        # avoid increasing the degree of a rational function
        if integrand.is_rational_function(symbol) is True and substituted.is_rational_function(u_var) is True:
            deg_before = max(degree(t, symbol) for t in integrand.as_numer_denom())
            deg_after = max(degree(t, u_var) for t in substituted.as_numer_denom())
            if deg_after > deg_before:
                return None
        return substituted.as_independent(u_var, as_Add=False)

    def exp_subterms(term: Expr) -> list[Expr]:
        linear_coeffs: list[int] = []
        terms: list[Expr] = []
        for exp_ in term.atoms(Exp):
            arg = as_expr(exp_.args[0])
            if symbol not in arg.free_symbols:
                continue
            coefficient = _times(symbol, arg)
            if coefficient is not None and isinstance(coefficient, Integer):
                linear_coeffs.append(int(coefficient))
            else:
                terms.append(as_expr(exp_))
        if linear_coeffs:
            terms.append(exp(Integer(math.gcd(*linear_coeffs))*symbol))
        return terms

    def possible_subterms(term: Expr) -> list[Expr]:
        if isinstance(term, (*_TRIG, *_HYPERBOLIC, *_INVERSE_TRIG, Exp, Log, Heaviside)):
            return [as_expr(term.args[0])]
        for klass, index in _ORTHOGONAL.items():
            if isinstance(term, klass):
                return [as_expr(term.args[index])]
        if isinstance(term, Mul):
            r: list[Expr] = []
            for u in term.args:
                r.append(as_expr(u))
                r.extend(possible_subterms(as_expr(u)))
            return r
        if isinstance(term, Pow):
            r = [as_expr(arg) for arg in term.args if arg.has(symbol)]
            if isinstance(term.exp, Integer):
                e = abs(int(term.exp))
                r.extend([term.base**d for d in sorted(factorint(e)) if 1 < d < e])
                if isinstance(term.base, Add):
                    r.extend([t for t in possible_subterms(term.base) if isinstance(t, Pow)])
            return r
        if isinstance(term, Add):
            r = []
            for arg in term.args:
                r.append(as_expr(arg))
                r.extend(possible_subterms(as_expr(arg)))
            return r
        return []

    for u in list(dict.fromkeys(possible_subterms(integrand) + exp_subterms(integrand))):
        if u == symbol:
            continue
        u_diff = manual_diff(u, symbol)
        found = test_subterm(u, u_diff)
        if found is not None:
            constant, new_integrand = found
            if new_integrand == integrand.subs(symbol, u_var):
                continue
            substitution = (u, constant, new_integrand)
            if substitution not in results:
                results.append(substitution)

    return results


# ---------------------------------------------------------------------------
# strategies

#: the failures of the native operations a rule may meet (the polynomials
#: over domains not ported, a division by zero), which make it not apply
_FAILURES: Final[tuple[type[Exception], ...]] = (BasePolynomialError, NotImplementedError, ValueError, TypeError,
                                                  ZeroDivisionError, ArithmeticError)


def _guarded(rule: Strategy, integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    """``rule`` on ``integral``, ``None`` when it fails (SymPy's rules
    raise where the native polynomials do not take the coefficients)."""
    try:
        found = yield from rule(integral, solver)
    except _FAILURES:
        return None
    return found


def _first(*strategies: Strategy) -> Strategy:
    """SymPy's ``do_one(null_safe(...), ...)``: the first strategy which
    gives a step."""
    def strategy(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
        for rule in strategies:
            found = yield from _guarded(rule, integral, solver)
            if found is not None:
                return found
        return None
    return strategy


def _rewriter(condition: Callable[[Expr, Symbol], bool], rewrite: Callable[[Expr, Symbol], Expr]) -> Strategy:
    """Strategy that rewrites an integrand."""
    def strategy(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
        integrand, symbol = integral.integrand, integral.symbol
        if condition(integrand, symbol):
            rewritten = rewrite(integrand, symbol)
            if rewritten != integrand:
                substep = yield from _steps(rewritten, symbol)
                if not isinstance(substep, DontKnowRule):
                    return RewriteRule(integrand, symbol, rewritten, substep)
        return None
    return strategy


def _alternatives(*strategies: Strategy) -> Strategy:
    """Strategy that makes an AlternativeRule out of multiple possible results."""
    def strategy(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
        alts: list[Rule] = []
        for rule in strategies:
            result = yield from _guarded(rule, integral, solver)
            if result is not None and not isinstance(result, DontKnowRule) and result not in alts:
                alts.append(result)
        if len(alts) == 1:
            return alts[0]
        if alts:
            doable = [rule for rule in alts if not rule.contains_dont_know()]
            if doable:
                return AlternativeRule(integral.integrand, integral.symbol, tuple(doable))
            return AlternativeRule(integral.integrand, integral.symbol, tuple(alts))
        return None
    return strategy


def _condition(test: Callable[[IntegralInfo], bool], rule: Strategy) -> Strategy:
    def strategy(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
        if not test(integral):
            return None
        found = yield from rule(integral, solver)
        return found
    return strategy


def _constant_rule(integral: IntegralInfo) -> Optional[Rule]:
    return ConstantRule(integral.integrand, integral.symbol)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:power_rule
def _power_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    base, expt = integrand.as_base_exp()

    if symbol not in expt.free_symbols and isinstance(base, Symbol):
        if _zero(expt + 1):
            return ReciprocalRule(integrand, symbol, base)
        return PowerRule(integrand, symbol, base, expt)
    elif symbol not in base.free_symbols and isinstance(expt, Symbol):
        rule = ExpRule(integrand, symbol, base, expt)
        zero = log(base).is_zero
        if zero is False:
            return rule
        elif zero is True:
            return ConstantRule(S.One, symbol)
        return PiecewiseRule(integrand, symbol, ((rule, ne(log(base), 0)), (ConstantRule(S.One, symbol), true)))
    return None


def _exp_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    argument = as_expr(integrand.args[0])
    if isinstance(argument, Symbol):
        return ExpRule(integrand, symbol, E, argument)
    return None


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:orthogonal_poly_rule
def _orthogonal_poly_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    for klass, var_index in _ORTHOGONAL.items():
        if not isinstance(integrand, klass):
            continue
        args = [as_expr(a) for a in integrand.args]
        if args[var_index] != symbol or any(v.has(symbol) for v in args[:var_index]):
            return None
        if isinstance(integrand, Jacobi):
            return JacobiRule(integrand, symbol, args[0], args[1], args[2])
        if isinstance(integrand, Gegenbauer):
            return GegenbauerRule(integrand, symbol, args[0], args[1])
        if isinstance(integrand, ChebyshevT):
            return ChebyshevTRule(integrand, symbol, args[0])
        if isinstance(integrand, ChebyshevU):
            return ChebyshevURule(integrand, symbol, args[0])
        if isinstance(integrand, Legendre):
            return LegendreRule(integrand, symbol, args[0])
        if isinstance(integrand, Hermite):
            return HermiteRule(integrand, symbol, args[0])
        if isinstance(integrand, Laguerre):
            return LaguerreRule(integrand, symbol, args[0])
        return AssocLaguerreRule(integrand, symbol, args[0], args[1])
    return None


def _over_symbol(integrand: Expr, symbol: Symbol) -> Optional[Expr]:
    """``g`` with ``integrand = g/x`` exactly (SymPy's pattern ``g/x`` with
    no other factor), ``None`` otherwise."""
    if not isinstance(integrand, Mul):
        return None
    factors = list(Mul.make_args(integrand))
    inverse = 1/symbol
    if inverse not in factors:
        return None
    factors.remove(inverse)
    if len(factors) != 1:
        return None
    return factors[0]


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:special_function_rule
def _special_function_rule(integral: IntegralInfo) -> Optional[Rule]:
    """The integrals of the special functions with elementary derivatives:
    ``exp(a*x + b)/x`` (Ei), ``cos``, ``cosh``, ``sin``, ``sinh`` of ``a*x + b``
    over ``x`` (Ci, Chi, Si, Shi), ``1/log(a*x + b)`` (li), ``exp``, ``sin``
    and ``cos`` of a quadratic (erf, Fresnel), ``x**e*exp(a*x)`` (upper
    gamma), ``polylog(b, a*x)/x`` and ``sqrt(a - d*sin(x)**2)`` and its
    reciprocal (elliptic integrals)."""
    integrand, x = integral.integrand, integral.symbol
    if isinstance(integrand, Mul):
        g = _over_symbol(integrand, x)
        if g is not None:
            if isinstance(g, (Exp, Cos, Cosh, Sin, Sinh)):
                linear = _linear(as_expr(g.args[0]), x)
                if linear is not None:
                    b, a = linear
                    if isinstance(g, Exp):
                        return EiRule(integrand, x, a, b)
                    if isinstance(g, Cos):
                        return CiRule(integrand, x, a, b)
                    if isinstance(g, Cosh):
                        return ChiRule(integrand, x, a, b)
                    if isinstance(g, Sin):
                        return SiRule(integrand, x, a, b)
                    return ShiRule(integrand, x, a, b)
        factors = list(Mul.make_args(integrand))
        if len(factors) == 2:
            for power, exponential in (factors, factors[::-1]):
                if not isinstance(exponential, Exp):
                    continue
                rate = _times(x, as_expr(exponential.args[0]))
                if rate is None or rate.is_zero is not False:
                    continue
                base, e = power.as_base_exp()
                if base != x or e.has(x) or (e.is_nonnegative is True and e.is_integer is True):
                    continue
                return UpperGammaRule(integrand, x, rate, e)
        if g is not None and isinstance(g, PolyLog):
            order = as_expr(g.args[0])
            rate = _times(x, as_expr(g.args[1]))
            if rate is not None and rate.is_zero is False and not order.has(x):
                return PolylogRule(integrand, x, rate, order)
        return None
    if isinstance(integrand, Pow):
        base, e = integrand.as_base_exp()
        if e == -1 and isinstance(base, Log):
            linear = _linear(as_expr(base.args[0]), x)
            if linear is not None:
                return LiRule(integrand, x, linear[1], linear[0])
        if e in (S.Half, -S.Half):
            coefficients = _coefficients_in_sine_square(base, x)
            if coefficients is not None:
                a, d = coefficients
                if a != d:
                    if e == S.Half:
                        return EllipticERule(integrand, x, a, d)
                    return EllipticFRule(integrand, x, a, d)
        return None
    if isinstance(integrand, (Exp, Sin, Cos)):
        quadratic = _quadratic(as_expr(integrand.args[0]), x)
        if quadratic is not None:
            c, b, a = quadratic
            if isinstance(integrand, Exp):
                return ErfRule(integrand, x, a, b, c)
            if isinstance(integrand, Sin):
                return FresnelSRule(integrand, x, a, b, c)
            return FresnelCRule(integrand, x, a, b, c)
    return None


def _coefficients_in_sine_square(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(a, d)`` with ``e = a - d*sin(x)**2``, ``a`` and ``d`` nonzero and
    free of ``x``."""
    t = _fresh('s')
    replaced = as_expr(e._xreplace_evaluated({sin(x): t}))
    if replaced.has(x):
        return None
    coefficients = _coefficients(replaced, t)
    if coefficients is None or len(coefficients) != 3 or coefficients[1] != 0:
        return None
    a, d = coefficients[0], -coefficients[2]
    if a.is_zero is not False or d.is_zero is not False:
        return None
    return a, d


def _add_degenerate_step(generic_cond: Boolean, generic_step: Rule, degenerate_step: Optional[Rule]) -> Rule:
    if degenerate_step is None:
        return generic_step
    subfunctions: list[Piece]
    if isinstance(generic_step, PiecewiseRule):
        subfunctions = [(substep, and_(cond, generic_cond)) for substep, cond in generic_step.subfunctions]
    else:
        subfunctions = [(generic_step, generic_cond)]
    if isinstance(degenerate_step, PiecewiseRule):
        subfunctions += list(degenerate_step.subfunctions)
    else:
        subfunctions.append((degenerate_step, true))
    return PiecewiseRule(generic_step.integrand, generic_step.variable, tuple(subfunctions))


class _NoMatch(Exception):
    pass


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:nested_pow_rule
def _nested_pow_rule(integral: IntegralInfo) -> Optional[Rule]:
    # nested (c*(a+b*x)**d)**e
    integrand, x = integral.integrand, integral.symbol
    conditions: list[Boolean] = []

    def get_base_exp(expr: Expr) -> tuple[Expr, Expr]:
        if not expr.has_free(x):
            return S.One, S.Zero
        if isinstance(expr, Mul):
            _, terms = expr.as_coeff_mul()
            if not terms:
                return S.One, S.Zero
            results = [get_base_exp(term) for term in terms]
            bases = {b for b, _ in results}
            bases.discard(S.One)
            if len(bases) == 1:
                return bases.pop(), add(*(e for _, e in results))
            raise _NoMatch
        if isinstance(expr, Pow):
            b, e = expr.base, expr.exp
            if e.has_free(x):
                raise _NoMatch
            base_, sub_exp = get_base_exp(b)
            return base_, sub_exp * e
        linear = _linear(expr, x)
        if linear is not None:
            a, b = linear
            conditions.append(ne(b, 0))
            return x + a/b, S.One
        raise _NoMatch

    try:
        base, exp_ = get_base_exp(integrand)
    except _NoMatch:
        return None
    generic_cond = conditions[-1] if conditions else true
    degenerate_step: Optional[Rule] = None
    if not _is_true(generic_cond):
        # equivalent with subs(b, 0) but no need to find b
        degenerate_step = ConstantRule(integrand.subs(x, 0), x)
    generic_step = NestedPowRule(integrand, x, base, exp_)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


def _make_inverse_trig(integrand: Expr, symbol: Symbol, arcsin: bool, a: Expr, sign_a: int, c: Expr, sign_c: int,
                       h: Expr) -> Rule:
    u_var = _fresh('u')
    rewritten = 1/sqrt(sign_a*a + sign_c*c*(symbol - h)**2)  # a>0, c>0
    quadratic_base: Expr = sqrt(c/a)*(symbol - h)
    constant = 1/sqrt(c)
    u_func: Optional[Expr] = None
    variable: Symbol = symbol
    if quadratic_base != symbol:
        u_func = quadratic_base
        variable = u_var
    standard_form = 1/sqrt(sign_a + sign_c*variable**2)
    substep: Rule = ArcsinRule(standard_form, variable) if arcsin else ArcsinhRule(standard_form, variable)
    if constant != 1:
        substep = ConstantTimesRule(constant*standard_form, symbol, constant, standard_form, substep)
    if u_func is not None:
        substep = URule(rewritten, symbol, u_var, u_func, substep)
    if h != 0:
        substep = CompleteSquareRule(integrand, symbol, rewritten, substep)
    return substep


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:inverse_trig_rule
def _inverse_trig(integral: IntegralInfo, degenerate: bool) -> Generator[IntegralInfo, Rule, Optional[Rule]]:
    """SymPy's ``inverse_trig_rule``; ``degenerate=False`` on the recursive
    calls where the coefficient of the quadratic term is known nonzero."""
    integrand, symbol = integral.integrand, integral.symbol
    base, exp_ = integrand.as_base_exp()
    match = _quadratic(base, symbol)
    if match is None:
        return None
    a, b, c = match
    generic_cond = ne(c, 0)
    degenerate_step: Optional[Rule] = None
    if degenerate and not _is_true(generic_cond):
        if b.is_zero is True:
            degenerate_step = ConstantRule(a ** exp_, symbol)
        else:
            degenerate_step = yield from _sqrt_linear(IntegralInfo((a + b * symbol) ** exp_, symbol))

    if _zero(2*exp_ + 1):
        h, k = -b/(2*c), a - b**2/(4*c)  # rewrite base to k + c*(symbol-h)**2
        non_square_cond = ne(k, 0)
        square_step: Optional[Rule] = None
        if not _is_true(non_square_cond):
            square_step = NestedPowRule(1/sqrt(c*(symbol - h)**2), symbol, symbol - h, S.NegativeOne)
        if _is_false(non_square_cond):
            return square_step
        generic_step: Rule = ReciprocalSqrtQuadraticRule(integrand, symbol, a, b, c)
        step = _add_degenerate_step(non_square_cond, generic_step, square_step)
        if k.is_real and c.is_real:
            # list of ((rule, base_exp, a, sign_a, b, sign_b), condition)
            rules: list[Piece] = []
            for arcsin, sign_c, cond in (  # don't apply ArccoshRule to x**2-1
                    (True, -1, and_(gt(k, 0), lt(c, 0))),  # 1-x**2
                    (False, 1, and_(gt(k, 0), gt(c, 0)))):  # 1+x**2
                c_ = -c if arcsin else c
                if _is_true(cond):
                    return _make_inverse_trig(integrand, symbol, arcsin, k, 1, c_, sign_c, h)
                if not _is_false(cond):
                    rules.append((_make_inverse_trig(integrand, symbol, arcsin, k, 1, c_, sign_c, h), cond))
            if rules:
                if not k.is_positive:  # conditions are not thorough, need fall back rule
                    rules.append((generic_step, true))
                step = PiecewiseRule(integrand, symbol, tuple(rules))
            else:
                step = generic_step
        return _add_degenerate_step(generic_cond, step, degenerate_step)
    if exp_ == S.Half:
        sqrt_step = SqrtQuadraticRule(integrand, symbol, a, b, c)
        return _add_degenerate_step(generic_cond, sqrt_step, degenerate_step)
    return None


def _inverse_trig_generic(integrand: Expr, symbol: Symbol) -> Optional[Rule]:
    """``inverse_trig_rule(..., degenerate=False)``, which asks for no
    subproblem."""
    steps = _inverse_trig(IntegralInfo(integrand, symbol), False)
    try:
        next(steps)
    except StopIteration as finished:
        value: Optional[Rule] = finished.value
        return value
    raise ValueError('the generic inverse trigonometric rule asked for a subproblem')


def _inverse_trig_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    found = yield from _inverse_trig(integral, True)
    return found


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:add_rule
def _add_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    results: list[Rule] = []
    for g in integrand.as_ordered_terms():
        step = yield from _steps(g, symbol)
        results.append(step)
    return AddRule(integrand, symbol, tuple(results))


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:mul_rule
def _mul_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    # Constant times function case
    coeff, f = integrand.as_independent(symbol)
    if coeff != 1:
        next_step = yield from _steps(f, symbol)
        return ConstantTimesRule(integrand, symbol, coeff, f, next_step)
    return None


@dataclass(frozen=True)
class _Parts:
    """One integration by parts: ``u``, ``dv``, ``v``, ``du`` and the step
    which gives ``v``."""
    u: Expr
    dv: Expr
    v: Expr
    du: Expr
    v_step: Rule


#: a choice of ``(u, dv)``
_Choice = Optional[tuple[Expr, Expr]]


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:_parts_rule
def _parts_rule(integrand: Expr, symbol: Symbol) -> Generator[IntegralInfo, Rule, Optional[_Parts]]:
    from sympy_extras.simplify.simplify import simplify

    # LIATE rule:
    # log, inverse trig, algebraic, trigonometric, exponential
    def pull_out_algebraic(integrand: Expr) -> _Choice:
        integrand = integrand.cancel().together()
        # iterating over Piecewise args would not work here
        algebraic = ([] if isinstance(integrand, Piecewise) or not isinstance(integrand, Mul)
                     else [as_expr(arg) for arg in integrand.args if as_expr(arg).is_algebraic_expr(symbol)])
        if algebraic:
            u = mul(*algebraic)
            dv = (integrand / u).cancel()
            return u, dv
        return None

    def pull_out_u(*functions: type[Basic]) -> Callable[[Expr], _Choice]:
        def pull_out_u_rl(integrand: Expr) -> _Choice:
            if any(integrand.has(f) for f in functions):
                args = [as_expr(arg) for arg in integrand.args if isinstance(arg, functions)]
                if args:
                    u = mul(*args)
                    dv = integrand / u
                    return u, dv
            return None
        return pull_out_u_rl

    liate_rules: list[Callable[[Expr], _Choice]] = [
        pull_out_u(Log), pull_out_u(*_INVERSE_TRIG), pull_out_algebraic, pull_out_u(Sin, Cos), pull_out_u(Exp)]

    dummy = _fresh('temporary')
    # we can integrate log(x) and atan(x) by setting dv = 1
    if isinstance(integrand, (Log, *_INVERSE_TRIG)):
        integrand = mul(dummy, integrand)

    for index, rule in enumerate(liate_rules):
        result = rule(integrand)
        if result is None:
            continue
        u, dv = result

        # Don't pick u to be a constant if possible
        if symbol not in u.free_symbols and not u.has(dummy):
            return None

        u = u.subs(dummy, 1)
        dv = dv.subs(dummy, 1)

        # Don't pick a non-polynomial algebraic to be differentiated
        if rule is pull_out_algebraic and u.is_polynomial(symbol) is not True:
            return None
        # Don't trade one logarithm for another
        if isinstance(u, Log):
            rec_dv = 1/dv
            if rec_dv.is_polynomial(symbol) is True:
                from sympy_extras.polys.polytools import degree
                if degree(rec_dv, symbol) == 1:
                    return None

        # Can integrate a polynomial times OrthogonalPolynomial
        if rule is pull_out_algebraic:
            if isinstance(dv, Derivative) or dv.has(*_TRIG) or isinstance(dv, tuple(_ORTHOGONAL)):
                v_step = yield from _steps(dv, symbol)
                if v_step.contains_dont_know():
                    return None
                du = u.diff(symbol)
                v = v_step.eval()
                return _Parts(u, dv, v, du, v_step)

        # make sure dv is amenable to integration
        accept = False
        if index < 2:  # log and inverse trig are usually worth trying
            accept = True
        elif (rule is pull_out_algebraic and dv.args and all(isinstance(a, (Sin, Cos, Exp)) for a in dv.args)):
            accept = True
        else:
            for lrule in liate_rules[index + 1:]:
                r = lrule(integrand)
                if r is not None and _same(r[0].subs(dummy, 1), dv):
                    accept = True
                    break

        if accept:
            du = u.diff(symbol)
            v_step = yield from _steps(simplify(dv), symbol)
            if not v_step.contains_dont_know():
                v = v_step.eval()
                return _Parts(u, dv, v, du, v_step)
    return None


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:parts_rule
def _parts(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    constant, integrand = integrand.as_coeff_Mul()

    result = yield from _parts_rule(integrand, symbol)

    steps: list[_Parts] = []
    if result is not None:
        u, v, du = result.u, result.v, result.du
        steps.append(result)

        if isinstance(v, Integral):
            return None

        # Set a limit on the number of times u can be used
        if isinstance(u, (Sin, Cos, Exp, Sinh, Cosh)):
            if not solver.use_parts_u(u, symbol):
                return None

        # Try cyclic integration by parts a few times
        for _ in range(4):
            coefficient = ((v * du) / integrand).cancel()
            if coefficient == 1:
                break
            if symbol not in coefficient.free_symbols:
                rule: Rule = CyclicPartsRule(
                    integrand, symbol,
                    tuple(PartsRule(p.u*p.dv, symbol, p.u, p.dv, p.v_step, None) for p in steps),
                    (-1) ** len(steps) * coefficient)
                if constant != 1:
                    rule = ConstantTimesRule(constant * integrand, symbol, constant, integrand, rule)
                return rule

            # _parts_rule is sensitive to constants, factor it out
            next_constant, next_integrand = (v * du).as_coeff_Mul()
            result = yield from _parts_rule(next_integrand, symbol)

            if result is not None:
                u = result.u * next_constant
                du = result.du * next_constant
                v = result.v
                steps.append(_Parts(u, result.dv, v, du, result.v_step))
            else:
                break

    def make_second_step(steps: list[_Parts], integrand: Expr) -> Generator[IntegralInfo, Rule, Rule]:
        if steps:
            p = steps[0]
            second = yield from make_second_step(steps[1:], p.v * p.du)
            return PartsRule(integrand, symbol, p.u, p.dv, p.v_step, second)
        rule = yield from _steps(integrand, symbol)
        return rule

    if steps:
        p = steps[0]
        second = yield from make_second_step(steps[1:], p.v * p.du)
        parts: Rule = PartsRule(integrand, symbol, p.u, p.dv, p.v_step, second)
        if constant != 1:
            parts = ConstantTimesRule(constant * integrand, symbol, constant, integrand, parts)
        return parts
    return None


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_rule
def _trig_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    if integrand == sin(symbol):
        return SinRule(integrand, symbol)
    if integrand == cos(symbol):
        return CosRule(integrand, symbol)
    if integrand == sec(symbol)**2:
        return Sec2Rule(integrand, symbol)
    if integrand == csc(symbol)**2:
        return Csc2Rule(integrand, symbol)

    if isinstance(integrand, Tan):
        arg = as_expr(integrand.args[0])
        rewritten = sin(arg) / cos(arg)
    elif isinstance(integrand, Cot):
        arg = as_expr(integrand.args[0])
        rewritten = cos(arg) / sin(arg)
    elif isinstance(integrand, Sec):
        arg = as_expr(integrand.args[0])
        rewritten = ((sec(arg)**2 + tan(arg) * sec(arg)) / (sec(arg) + tan(arg)))
    elif isinstance(integrand, Csc):
        arg = as_expr(integrand.args[0])
        rewritten = ((csc(arg)**2 + cot(arg) * csc(arg)) / (csc(arg) + cot(arg)))
    else:
        return None

    substep = yield from _steps(rewritten, symbol)
    return RewriteRule(integrand, symbol, rewritten, substep)


def _trig_product_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    if integrand == sec(symbol) * tan(symbol):
        return SecTanRule(integrand, symbol)
    if integrand == csc(symbol) * cot(symbol):
        return CscCotRule(integrand, symbol)
    return None


def _rational_parts(integrand: Expr, symbol: Symbol) -> Optional[tuple[list[Expr], list[Expr]]]:
    """The coefficients of the numerator and the denominator of
    ``integrand`` as polynomials in ``symbol`` (lowest degree first), with
    the factors free of ``symbol`` kept in the numerator."""
    if not isinstance(integrand, (Mul, Pow)):
        return None
    numerator, denominator = integrand.as_numer_denom()
    num = _coefficients(numerator, symbol)
    den = _coefficients(denominator, symbol)
    if num is None or den is None:
        return None
    return num, den


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:quadratic_denom_rule
def _quadratic_denom_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    parts = _rational_parts(integrand, symbol)
    if parts is None:
        return None
    num, den = parts
    if len(num) > 2 or len(den) != 3:
        return None
    if len(num) == 1 and den[1] == 0:
        # a/(b*x**2 + c)
        a, b, c = num[0], den[2], den[0]
        # (SymPy's constant is 1/b: its a is always 1 here, the constant
        # factors being taken out before)
        power: Rule = PowerRule(symbol**-2, symbol, symbol, Integer(-2))
        if a/b != 1:
            power = ConstantTimesRule(integrand, symbol, a/b, symbol**-2, power)
        if c == 0:
            # a/(b*x**2), which SymPy's pattern meets only unexpanded (the
            # coefficients read here took 1/(4*u + 2*(u - 1)**2 - 2) for an
            # arctangent of zero width, and its value for nan)
            return power
        general_rule = ArctanRule(integrand, symbol, a, b, c)
        if b.is_extended_real and c.is_extended_real:
            positive_cond = gt(c/b, 0)
            if _is_true(positive_cond):
                return general_rule
            coeff = a/(2*sqrt(-c)*sqrt(b))
            constant = sqrt(-c/b)
            r1 = 1/(symbol - constant)
            r2 = 1/(symbol + constant)
            log_steps: tuple[Rule, ...] = (
                ReciprocalRule(r1, symbol, symbol - constant),
                ConstantTimesRule(-r2, symbol, S.NegativeOne, r2, ReciprocalRule(r2, symbol, symbol + constant)))
            rewritten = sub = r1 - r2
            negative_step: Rule = AddRule(sub, symbol, log_steps)
            if coeff != 1:
                rewritten = Mul(coeff, sub)
                negative_step = ConstantTimesRule(rewritten, symbol, coeff, sub, negative_step)
            negative_step = RewriteRule(integrand, symbol, rewritten, negative_step)
            if _is_false(positive_cond):
                return negative_step
            return PiecewiseRule(integrand, symbol, ((general_rule, positive_cond), (negative_step, true)))
        return PiecewiseRule(integrand, symbol, ((general_rule, ne(c, 0)), (power, true)))

    if len(num) == 1:
        # a/(b*x**2 + c*x + d)
        b, c = den[2], den[1]
        if b.is_zero is True:
            return None
        u = _fresh('u')
        u_func = symbol + c/(2*b)
        integrand2 = integrand.subs(symbol, u - c/(2*b))
        next_step = yield from _steps(integrand2, u)
        return URule(integrand2, symbol, u, u_func, next_step)

    # (a*x + b)/(c*x**2 + d*x + e)
    a, b = num[1], num[0]
    c, d, e = den[2], den[1], den[0]
    if c.is_zero is True:
        return None
    denominator = c * symbol**2 + d * symbol + e
    const = a/(2*c)
    numer1 = (2*c*symbol + d)
    numer2 = - const*d + b
    u = _fresh('u')
    reciprocal = yield from _steps(u**(-1), u)
    step1: Rule = URule(integrand, symbol, u, denominator, reciprocal)
    if const != 1:
        step1 = ConstantTimesRule(const*numer1/denominator, symbol, const, numer1/denominator, step1)
    if numer2.is_zero is True:
        return step1
    step2 = yield from _steps(numer2/denominator, symbol)
    substeps = AddRule(integrand, symbol, (step1, step2))
    rewriten = const*numer1/denominator + numer2/denominator
    return RewriteRule(integrand, symbol, rewriten, substeps)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:sqrt_linear_rule
def _sqrt_linear(integral: IntegralInfo) -> Generator[IntegralInfo, Rule, Optional[Rule]]:
    """Substitute common (a+b*x)**(1/n)"""
    integrand, x = integral.integrand, integral.symbol
    a0: Expr = S.Zero
    b0: Expr = S.Zero
    bases: list[Expr] = []
    qs: list[int] = []
    bs: list[Expr] = []
    for pow_ in integrand.atoms(Pow):  # collect all (a+b*x)**(p/q)
        if not isinstance(pow_, Pow):
            continue
        base, exp_ = pow_.base, pow_.exp
        if isinstance(exp_, Integer) or x not in base.free_symbols:  # skip 1/x and sqrt(2)
            continue
        if not isinstance(exp_, Rational):  # exclude x**pi
            return None
        match = _linear(base, x)
        if match is None:  # skip non-linear
            continue  # for sqrt(x+sqrt(x)), although base is non-linear, we can still substitute sqrt(x)
        a1, b1 = match
        if a0*b1 != a1*b0 or (b0/b1).is_nonnegative is not True:  # cannot transform sqrt(x) to sqrt(x+1) or sqrt(-x)
            return None
        if b0 == 0 or _is_true(gt(b0/b1, 1)):  # choose the latter of sqrt(2*x) and sqrt(x) as representative
            a0, b0 = a1, b1
        bases.append(base)
        bs.append(b1)
        qs.append(int(exp_.q))
    if b0 == 0:  # no such pattern found
        return None
    q0 = math.lcm(*qs)
    u_x = (a0 + b0*x)**Rational(1, q0)
    u = _fresh('u')
    replacements: list[tuple[Basic, Basic]] = [(base**Rational(1, q), (b/b0)**Rational(1, q)*u**Rational(q0, q))
                                               for base, b, q in zip(bases, bs, qs)]
    substituted = integrand.subs(replacements).subs(x, (u**q0 - a0)/b0)
    substep = yield from _steps(substituted*u**(q0 - 1)*q0/b0, u)
    if not substep.contains_dont_know():
        step: Rule = URule(integrand, x, u, u_x, substep)
        generic_cond = ne(b0, 0)
        if not _is_true(generic_cond):  # possible degenerate case
            simplified = integrand.subs({b: 0 for b in bs})
            degenerate_step = yield from _steps(simplified, x)
            step = PiecewiseRule(integrand, x, ((step, generic_cond), (degenerate_step, true)))
        return step
    return None


def _sqrt_linear_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    found = yield from _sqrt_linear(integral)
    return found


@dataclass(frozen=True)
class _SqrtQuadratic:
    """A match of ``f*sqrt(a + b*x + c*x**2)**n``: ``f`` a polynomial
    (its coefficients, lowest degree first), ``n`` odd."""
    a: Expr
    b: Expr
    c: Expr
    f: Expr
    coefficients: list[Expr] = field(hash=False)
    n: int


def _match_sqrt_quadratic(integrand: Expr, x: Symbol) -> Optional[_SqrtQuadratic]:
    for factor in Mul.make_args(integrand):
        base, e = factor.as_base_exp()
        if not isinstance(e, Rational) or isinstance(e, Integer) or e.q != 2:
            continue
        quadratic = _quadratic(base, x)
        if quadratic is None:
            continue
        f = integrand / factor
        coefficients = _coefficients(f, x)
        if coefficients is None:
            continue
        a, b, c = quadratic
        return _SqrtQuadratic(a, b, c, f, coefficients, int(e.p))
    return None


def _polynomial(coefficients: Sequence[Expr], x: Symbol) -> Expr:
    return add(*(c*x**k for k, c in enumerate(coefficients)))


def _sqrt_quadratic_denom(a: Expr, b: Expr, c: Expr, x: Symbol, numer: list[Expr], integrand: Expr) -> Optional[Rule]:
    """``(d + e*x + ...)/sqrt(a+b*x+c*x**2)`` (SymPy's
    ``sqrt_quadratic_denom_rule``); ``numer`` the coefficients of the
    numerator, lowest degree first."""
    denom = sqrt(a + b*x + c*x**2)
    while len(numer) > 1 and numer[-1] == 0:
        numer = numer[:-1]
    deg = len(numer) - 1
    if deg <= 1:
        # integrand == (d+e*x)/sqrt(a+b*x+c*x**2)
        e, d = (numer[1], numer[0]) if deg == 1 else (S.Zero, numer[0])
        # rewrite numerator to A*(2*c*x+b) + B
        A = e/(2*c)
        B = d - A*b
        pre_substitute = (2*c*x + b)/denom
        constant_step: Optional[Rule] = None
        linear_step: Optional[Rule] = None
        if A != 0:
            u = _fresh('u')
            pow_rule = PowerRule(1/sqrt(u), u, u, -S.Half)
            linear_step = URule(pre_substitute, x, u, a + b*x + c*x**2, pow_rule)
            if A != 1:
                linear_step = ConstantTimesRule(A*pre_substitute, x, A, pre_substitute, linear_step)
        if B != 0:
            constant_step = _inverse_trig_generic(1/denom, x)
            if constant_step is not None and B != 1:
                constant_step = ConstantTimesRule(B/denom, x, B, 1/denom, constant_step)
        if linear_step is not None and constant_step is not None:
            both = Add(A*pre_substitute, B/denom)
            return RewriteRule(integrand, x, both, AddRule(both, x, (linear_step, constant_step)))
        return linear_step if linear_step is not None else constant_step
    return SqrtQuadraticDenomRule(integrand, x, a, b, c, tuple(reversed(numer)))


def _sqrt_quadratic_generic_step(match: _SqrtQuadratic, integrand: Expr, x: Symbol) -> Optional[Rule]:
    a, b, c, n = match.a, match.b, match.c, match.n
    if n > 0:  # rewrite poly * sqrt(s)**(2*k-1) to poly*s**k / sqrt(s)
        numer = _coefficients(_polynomial(match.coefficients, x) * (a + b*x + c*x**2)**((n + 1)//2), x)
        if numer is None:
            return None
        rewritten = _polynomial(numer, x)/sqrt(a + b*x + c*x**2)
        substep = _sqrt_quadratic_denom(a, b, c, x, numer, rewritten)
        if substep is None:
            return None
        return RewriteRule(integrand, x, rewritten, substep)
    if n == -1:
        return _sqrt_quadratic_denom(a, b, c, x, match.coefficients, integrand)
    return None  # todo: handle n < -1 case


def _sqrt_quadratic_generic(integrand: Expr, x: Symbol) -> Optional[Rule]:
    """``sqrt_quadratic_rule(..., degenerate=False)``."""
    match = _match_sqrt_quadratic(integrand, x)
    if match is None:
        return None
    return _sqrt_quadratic_generic_step(match, integrand, x)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:sqrt_quadratic_rule
def _sqrt_quadratic_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, x = integral.integrand, integral.symbol
    match = _match_sqrt_quadratic(integrand, x)
    if match is None:
        return None
    a, b, c, f, n = match.a, match.b, match.c, match.f, match.n
    generic_cond = ne(c, 0)
    degenerate_step: Optional[Rule] = None
    if not _is_true(generic_cond):
        if b.is_zero is True:
            degenerate_step = yield from _steps(f*sqrt(a)**n, x)
        else:
            degenerate_step = yield from _sqrt_linear(IntegralInfo(f*sqrt(a + b*x)**n, x))
    generic_step = _sqrt_quadratic_generic_step(match, integrand, x)
    if generic_step is None:
        return None
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:hyperbolic_rule
def _hyperbolic_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    if not isinstance(integrand, _HYPERBOLIC) or integrand.args[0] != symbol:
        return None
    if isinstance(integrand, Sinh):
        return SinhRule(integrand, symbol)
    if isinstance(integrand, Cosh):
        return CoshRule(integrand, symbol)
    u = _fresh('u')
    if isinstance(integrand, Tanh):
        rewritten = sinh(symbol)/cosh(symbol)
        return RewriteRule(integrand, symbol, rewritten,
                           URule(rewritten, symbol, u, cosh(symbol), ReciprocalRule(1/u, u, u)))
    if isinstance(integrand, Coth):
        rewritten = cosh(symbol)/sinh(symbol)
        return RewriteRule(integrand, symbol, rewritten,
                           URule(rewritten, symbol, u, sinh(symbol), ReciprocalRule(1/u, u, u)))
    half = tanh(symbol/2)
    if isinstance(integrand, Sech):
        # SymPy's sech(x).rewrite(tanh)
        rewritten = (1 - half**2)/(1 + half**2)
        return RewriteRule(integrand, symbol, rewritten,
                           URule(rewritten, symbol, u, half, ArctanRule(2/(u**2 + 1), u, Integer(2), S.One, S.One)))
    # csch(x).rewrite(tanh)
    rewritten = (1 - half**2)/(2*half)
    return RewriteRule(integrand, symbol, rewritten, URule(rewritten, symbol, u, half, ReciprocalRule(1/u, u, u)))


@dataclass(frozen=True)
class _TrigPowers:
    """A match of ``f(a*x)**m * g(b*x)**n`` (SymPy's ``sincos_pattern`` and
    the others): the integers ``m``, ``n`` and the coefficients, zero for
    an absent factor."""
    a: Expr
    b: Expr
    m: Expr
    n: Expr


def _match_trig_powers(integrand: Expr, symbol: Symbol, first: type[Basic],
                       second: type[Basic]) -> Optional[_TrigPowers]:
    a: Expr = S.Zero
    b: Expr = S.Zero
    m: Expr = S.Zero
    n: Expr = S.Zero
    seen_first = seen_second = False
    for factor in Mul.make_args(integrand):
        base, e = factor.as_base_exp()
        if not isinstance(e, Integer):
            return None
        if not isinstance(base, (first, second)):
            return None
        coefficient = _times(symbol, as_expr(base.args[0]))
        if coefficient is None:
            return None
        if isinstance(base, first):
            if seen_first:
                return None
            seen_first = True
            a, m = coefficient, e
        else:
            if seen_second:
                return None
            seen_second = True
            b, n = coefficient, e
    return _TrigPowers(a, b, m, n)


def _even(e: Expr) -> bool:
    return isinstance(e, Integer) and int(e) % 2 == 0


def _odd(e: Expr) -> bool:
    return isinstance(e, Integer) and int(e) % 2 == 1


def _at_least(e: Expr, k: int) -> bool:
    return isinstance(e, Integer) and int(e) >= k


def _trig_rewrite(integrand: Expr, symbol: Symbol, rewritten: Expr) -> Steps:
    if rewritten != integrand:
        substep = yield from _steps(rewritten, symbol)
        return RewriteRule(integrand, symbol, rewritten, substep)
    return None


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_sincos_rule
def _trig_sincos_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    if not integrand.has(Sin, Cos):
        return None
    match = _match_trig_powers(integrand, symbol, Sin, Cos)
    if match is None:
        return None
    a, b, m, n = match.a, match.b, match.m, match.n
    rewritten: Optional[Expr] = None
    if _even(m) and _even(n) and _at_least(m, 0) and _at_least(n, 0):
        rewritten = (((1 - cos(2*a*symbol)) / 2) ** (m / 2)) * (((1 + cos(2*b*symbol)) / 2) ** (n / 2))
    elif _odd(m) and _at_least(m, 3):
        rewritten = (1 - cos(a*symbol)**2)**((m - 1) / 2) * sin(a*symbol) * cos(b*symbol) ** n
    elif _odd(n) and _at_least(n, 3):
        rewritten = (1 - sin(b*symbol)**2)**((n - 1) / 2) * cos(b*symbol) * sin(a*symbol) ** m
    if rewritten is None:
        return None
    found = yield from _trig_rewrite(integrand, symbol, rewritten)
    return found


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_tansec_rule
def _trig_tansec_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    integrand = integrand.subs({1 / cos(symbol): sec(symbol)})
    if not integrand.has(Tan, Sec):
        return None
    match = _match_trig_powers(integrand, symbol, Tan, Sec)
    if match is None:
        return None
    a, b, m, n = match.a, match.b, match.m, match.n
    rewritten: Optional[Expr] = None
    if _odd(m):
        rewritten = (sec(a*symbol)**2 - 1) ** ((m - 1) / 2) * tan(a*symbol) * sec(b*symbol) ** n
    elif _even(n) and _at_least(n, 4):
        rewritten = (1 + tan(b*symbol)**2) ** (n/2 - 1) * sec(b*symbol)**2 * tan(a*symbol) ** m
    elif m == 2 and n == 0:
        rewritten = sec(a*symbol)**2 - 1
    if rewritten is None:
        return None
    found = yield from _trig_rewrite(integrand, symbol, rewritten)
    return found


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_cotcsc_rule
def _trig_cotcsc_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    integrand = integrand.subs([(1 / sin(symbol), csc(symbol)), (1 / tan(symbol), cot(symbol)),
                                (cos(symbol) / tan(symbol), cot(symbol))])
    if not integrand.has(Cot, Csc):
        return None
    match = _match_trig_powers(integrand, symbol, Cot, Csc)
    if match is None:
        return None
    a, b, m, n = match.a, match.b, match.m, match.n
    rewritten: Optional[Expr] = None
    if _odd(m):
        rewritten = (csc(a*symbol)**2 - 1) ** ((m - 1) / 2) * cot(a*symbol) * csc(b*symbol) ** n
    elif _even(n) and _at_least(n, 4):
        rewritten = (1 + cot(b*symbol)**2) ** (n/2 - 1) * csc(b*symbol)**2 * cot(a*symbol) ** m
    if rewritten is None:
        return None
    found = yield from _trig_rewrite(integrand, symbol, rewritten)
    return found


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_sindouble_rule
def _trig_sindouble_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    double = sin(2*symbol)
    factors = list(Mul.make_args(integrand))
    if double not in factors:
        return None
    factors.remove(double)
    if mul(*factors).has(double):
        return None
    sin_double = 2*sin(symbol)*cos(symbol)/double
    found = yield from _steps(integrand * sin_double, symbol)
    return found


_trig_powers_products_rule: Final[Strategy] = _first(_trig_sincos_rule, _trig_tansec_rule, _trig_cotcsc_rule,
                                                     _trig_sindouble_rule)


def _known_sign(e: Expr) -> tuple[bool, bool]:
    """``(positive, negative)`` as SymPy's ``(a.is_number and a > 0) or
    a.is_positive`` and its counterpart."""
    positive = e.is_positive is True or (e.is_number and _is_true(gt(e, 0)))
    negative = e.is_negative is True or (e.is_number and _is_true(lt(e, 0)))
    return positive, negative


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:trig_substitution_rule
def _trig_substitution_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    from sympy_extras.simplify.trigsimp import trigsimp
    integrand, symbol = integral.integrand, integral.symbol
    theta = _fresh('theta')

    matches: list[tuple[Expr, Expr]] = []
    for node in preorder_traversal(integrand):
        if not isinstance(node, Add):
            continue
        coefficients = _coefficients(node, symbol)
        if coefficients is None or len(coefficients) != 3 or coefficients[1] != 0:
            continue
        a, b = coefficients[0], coefficients[2]
        if a.is_zero is True or b.is_zero is True or (a, b) in matches:
            continue
        matches.append((a, b))
    for a, b in matches:
        a_positive, a_negative = _known_sign(a)
        b_positive, b_negative = _known_sign(b)
        x_func: Optional[Expr] = None
        restriction: Boolean = true
        if a_positive and b_positive:
            # a**2 + b*x**2. Assume sec(theta) > 0, -pi/2 < theta < pi/2
            x_func = (sqrt(a)/sqrt(b)) * tan(theta)
            # Do not restrict the domain: tan(theta) takes on any real
            # value on the interval -pi/2 < theta < pi/2 so x takes on
            # any value
            restriction = true
        elif a_positive and b_negative:
            # a**2 - b*x**2. Assume cos(theta) > 0, -pi/2 < theta < pi/2
            constant = sqrt(a)/sqrt(-b)
            x_func = constant * sin(theta)
            restriction = and_(gt(symbol, -constant), lt(symbol, constant))
        elif a_negative and b_positive:
            # b*x**2 - a**2. Assume tan(theta) > 0, 0 < theta < pi/2
            constant = sqrt(-a)/sqrt(b)
            x_func = constant * sec(theta)
            # x > constant: sqrt(tan(theta)**2) = tan(theta) needs 0 <
            # theta < pi/2 (SymPy's -constant < x < constant is wrong for
            # x < 0: the derivative of its antiderivative of
            # sqrt(16*x**2 - 9)/x misses the integrand by 16.1*I at x = -1/2)
            restriction = gt(symbol, constant)
        if x_func is None:
            continue
        # Manually simplify sqrt(trig(theta)**2) to trig(theta)
        # Valid due to assumed domain restriction
        substitutions: list[tuple[Expr, Expr]] = []
        for f in (sin, cos, tan, sec, csc, cot):
            substitutions.append((sqrt(f(theta)**2), f(theta)))
            substitutions.append((sqrt(f(theta)**(-2)), 1/f(theta)))

        replaced = trigsimp(integrand.subs(symbol, x_func))
        replaced = manual_subs(replaced, substitutions)
        if replaced.has(symbol):
            continue
        replaced = trigsimp(replaced * manual_diff(x_func, theta))
        replaced = as_expr(replaced._xreplace_evaluated({1/cos(theta): sec(theta)}))
        substep = yield from _steps(replaced, theta)
        if not substep.contains_dont_know():
            return TrigSubstitutionRule(integrand, symbol, theta, x_func, replaced, substep, restriction)
    return None


def _heaviside_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    factors = list(Mul.make_args(integrand))
    for factor in factors:
        if not isinstance(factor, Heaviside):
            continue
        linear = _linear(as_expr(factor.args[0]), symbol)
        if linear is None:
            continue
        b, m = linear
        rest = list(factors)
        rest.remove(factor)
        g = mul(*rest)
        if g == 0:
            return None
        # f = Heaviside(m*x + b)*g
        substep = yield from _steps(g, symbol)
        return HeavisideRule(integrand, symbol, m*symbol + b, -b/m, substep)
    return None


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:dirac_delta_rule
def _dirac_delta_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, x = integral.integrand, integral.symbol
    n = S.Zero if len(integrand.args) == 1 else as_expr(integrand.args[1])
    if not isinstance(n, Integer) or int(n) < 0:
        return None
    match = _linear(as_expr(integrand.args[0]), x)
    if match is None:
        return None
    a, b = match
    generic_cond = ne(b, 0)
    degenerate_step: Optional[Rule] = None
    if not _is_true(generic_cond):
        degenerate_step = ConstantRule(dirac_delta(a, n), x)
    generic_step = DiracDeltaRule(integrand, x, n, a, b)
    return _add_degenerate_step(generic_cond, generic_step, degenerate_step)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:substitution_rule
def _substitution_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol

    u_var = _fresh('u')
    substitutions = find_substitutions(integrand, symbol, u_var)
    ways: list[Rule] = []
    for u_func, c, substituted in substitutions:
        subrule = yield from _steps(substituted, u_var)
        if subrule.contains_dont_know():
            continue

        if not _zero(c - 1):
            _, denom = c.as_numer_denom()
            subrule = ConstantTimesRule(c * substituted, u_var, c, substituted, subrule)

            if denom.free_symbols:
                pieces: list[Piece] = []
                could_be_zero = list(Mul.make_args(denom))
                for expr in could_be_zero:
                    if expr.is_zero is not False:
                        substep = yield from _steps(manual_subs(integrand, [(expr, S.Zero)]), symbol)
                        pieces.append((substep, eq(expr, 0)))
                pieces.append((subrule, true))
                subrule = PiecewiseRule(substituted, symbol, tuple(pieces))

        ways.append(URule(integrand, symbol, u_var, u_func, subrule))

    if len(ways) > 1:
        return AlternativeRule(integrand, symbol, tuple(ways))
    if ways:
        return ways[0]
    return None


def _apart(integrand: Expr, symbol: Symbol) -> Expr:
    from sympy_extras.polys.partfrac import apart
    try:
        return apart(integrand, symbol)
    except _FAILURES:
        # the native apart takes no parameters in the leading coefficient
        # of the denominator (x**2/(a**2*x**2 + 1)): its polynomial part
        # at least, by the division over the expressions
        return _polynomial_part(integrand, symbol)


def _polynomial_part(integrand: Expr, symbol: Symbol) -> Expr:
    """``q + r/d`` for ``integrand = n/d``, by the long division of the
    coefficients of ``n`` by those of ``d`` (``integrand`` itself when
    the numerator is of lower degree, or not a polynomial)."""
    from sympy_extras.polys.polytools import cancel
    numerator, denominator = integrand.as_numer_denom()
    num = _coefficients(numerator, symbol)
    den = _coefficients(denominator, symbol)
    if num is None or den is None or len(den) < 2 or len(num) < len(den) or den[-1] == 0:
        return integrand
    remainder = list(num)
    quotient: list[Expr] = [S.Zero] * (len(num) - len(den) + 1)
    for k in range(len(quotient) - 1, -1, -1):
        c = cancel(remainder[k + len(den) - 1] / den[-1])
        quotient[k] = c
        for j, d in enumerate(den):
            remainder[k + j] = cancel(remainder[k + j] - c*d)
    rest = _polynomial(remainder[:len(den) - 1], symbol)
    return _polynomial(quotient, symbol) + rest/denominator


def _is_rational(integrand: Expr, symbol: Symbol) -> bool:
    return integrand.is_rational_function() is True


def _cancel(integrand: Expr, symbol: Symbol) -> Expr:
    try:
        return integrand.cancel()
    except _FAILURES:
        return integrand


def _always(integrand: Expr, symbol: Symbol) -> bool:
    return True


def _expandable(integrand: Expr, symbol: Symbol) -> bool:
    return isinstance(integrand, (Pow, Mul)) or all(
        isinstance(arg, Pow) or as_expr(arg).is_polynomial(symbol) is True for arg in integrand.args)


def _expanded(integrand: Expr, symbol: Symbol) -> Expr:
    return integrand.expand()


def _different_trig_arguments(integrand: Expr, symbol: Symbol) -> bool:
    return len({a.args[0] for a in integrand.atoms(*_TRIG)}) > 1


def _trig_expanded(integrand: Expr, symbol: Symbol) -> Expr:
    return expand(integrand, trig=True)


_partial_fractions_rule: Final[Strategy] = _rewriter(_is_rational, _apart)
_cancel_rule: Final[Strategy] = _rewriter(_always, _cancel)
_distribute_expand_rule: Final[Strategy] = _rewriter(_expandable, _expanded)
# If there are trig functions with different arguments, expand them
_trig_expand_rule: Final[Strategy] = _rewriter(_different_trig_arguments, _trig_expanded)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:derivative_rule
def _derivative_rule(integral: IntegralInfo) -> Optional[Rule]:
    integrand, symbol = integral.integrand, integral.symbol
    if not isinstance(integrand, Derivative):
        return None
    diff_variables = integrand.variables
    undifferentiated_function = integrand.expr
    integrand_variables = undifferentiated_function.free_symbols

    if symbol in integrand_variables:
        if symbol in diff_variables:
            return DerivativeRule(integrand, symbol)
        return DontKnowRule(integrand, symbol)
    return ConstantRule(integrand, symbol)


def _rewrites_rule(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    integrand, symbol = integral.integrand, integral.symbol
    if integrand == 1/cos(symbol):
        rewritten = integrand.subs(1/cos(symbol), sec(symbol))
        substep = yield from _steps(rewritten, symbol)
        return RewriteRule(integrand, symbol, rewritten, substep)
    return None


def _key(integrand: Expr, symbol: Symbol) -> str:
    """SymPy's ``key``: the class the integrand is dispatched on."""
    if symbol not in integrand.free_symbols:
        return 'Number'
    if isinstance(integrand, Symbol):
        return 'Symbol'
    if isinstance(integrand, _TRIG):
        return 'TrigonometricFunction'
    if isinstance(integrand, tuple(_ORTHOGONAL)):
        return 'OrthogonalPolynomial'
    for name, klass in (('Pow', Pow), ('Exp', Exp), ('Add', Add), ('Mul', Mul), ('Derivative', Derivative),
                        ('Heaviside', Heaviside), ('DiracDelta', DiracDelta), ('Log', Log)):
        if isinstance(integrand, klass):
            return name
    if isinstance(integrand, _INVERSE_TRIG):
        return 'InverseTrigonometricFunction'
    return type(integrand).__name__


def _is_mul_or_pow(integral: IntegralInfo) -> bool:
    return _key(integral.integrand, integral.symbol) in ('Mul', 'Pow')


def _is_mul_log_or_inverse(integral: IntegralInfo) -> bool:
    return _key(integral.integrand, integral.symbol) in ('Mul', 'Log', 'InverseTrigonometricFunction')


_SWITCH: Final[dict[str, Strategy]] = {
    'Pow': _first(_atomic(_power_rule), _inverse_trig_rule, _sqrt_linear_rule, _quadratic_denom_rule),
    'Symbol': _atomic(_power_rule),
    'Exp': _atomic(_exp_rule),
    'Add': _add_rule,
    'Mul': _first(_mul_rule, _atomic(_trig_product_rule), _heaviside_rule, _quadratic_denom_rule,
                  _sqrt_linear_rule, _sqrt_quadratic_rule),
    'Derivative': _atomic(_derivative_rule),
    'TrigonometricFunction': _trig_rule,
    'Heaviside': _heaviside_rule,
    'DiracDelta': _atomic(_dirac_delta_rule),
    'OrthogonalPolynomial': _atomic(_orthogonal_poly_rule),
    'Number': _atomic(_constant_rule),
}

_GENERAL: Final[Strategy] = _first(
    _trig_rule,
    _atomic(_hyperbolic_rule),
    _alternatives(
        _rewrites_rule,
        _substitution_rule,
        _condition(_is_mul_or_pow, _partial_fractions_rule),
        _condition(_is_mul_or_pow, _cancel_rule),
        _condition(_is_mul_log_or_inverse, _parts),
        _condition(_is_mul_or_pow, _distribute_expand_rule),
        _trig_powers_products_rule,
        _trig_expand_rule),
    _condition(_is_mul_or_pow, _atomic(_nested_pow_rule)),
    _trig_substitution_rule)


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:integral_steps
def _integral_steps(integral: IntegralInfo, solver: IntegralSolver) -> Steps:
    found = yield from _guarded(_atomic(_special_function_rule), integral, solver)
    if found is not None:
        return found
    switch = _SWITCH.get(_key(integral.integrand, integral.symbol))
    if switch is not None:
        found = yield from _guarded(switch, integral, solver)
        if found is not None:
            return found
    found = yield from _GENERAL(integral, solver)
    if found is not None:
        return found
    return DontKnowRule(integral.integrand, integral.symbol)


def integral_steps(integrand: ExprLike, symbol: Symbol) -> Rule:
    """Returns the steps needed to compute an integral.

    This function attempts to mirror what a student would do by hand as
    closely as possible: the result is the first rule of the tree, whose
    substeps are rules again; ``eval()`` gives the antiderivative.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, sin
    >>> from sympy_extras.integrals.manualintegrate import integral_steps
    >>> x = symbols('x')
    >>> integral_steps((x**2 + 3)**2, x)  # doctest: +NORMALIZE_WHITESPACE
    RewriteRule(integrand=(x**2 + 3)**2, variable=x, rewritten=x**4 + 6*x**2 + 9,
    substep=AddRule(integrand=x**4 + 6*x**2 + 9, variable=x,
    substeps=(PowerRule(integrand=x**4, variable=x, base=x, exp=4),
    ConstantTimesRule(integrand=6*x**2, variable=x, constant=6, other=x**2,
    substep=PowerRule(integrand=x**2, variable=x, base=x, exp=2)),
    ConstantRule(integrand=9, variable=x))))
    >>> integral_steps(exp(x)/(1 + exp(2*x)), x).contains_dont_know()
    False
    """
    return IntegralSolver().solve(IntegralInfo(to_expr(integrand), symbol))


# PORTED FROM SYMPY: sympy/integrals/manualintegrate.py:manualintegrate
def manualintegrate(f: ExprLike, var: Symbol) -> Expr:
    """The indefinite integral of ``f`` in ``var`` by an algorithm that
    resembles what a student would do by hand (the evaluation of
    :func:`integral_steps`); an unevaluated
    :class:`~sympy_extras.integrals.integral.Integral` is left for the
    parts it cannot do.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, cos, exp
    >>> from sympy_extras.integrals.manualintegrate import manualintegrate
    >>> x = symbols('x')
    >>> manualintegrate(cos(x)**4*sin(x), x)
    -cos(x)**5/5
    >>> manualintegrate(exp(x)*sin(x), x)
    exp(x)*sin(x)/2 - exp(x)*cos(x)/2
    """
    result = integral_steps(f, var).eval()
    # If we got Piecewise with two parts, put generic first
    if isinstance(result, Piecewise) and len(result.args) == 2:
        first, second = result.args
        cond = first.cond
        if isinstance(cond, Eq) and isinstance(second.cond, BooleanTrue):
            result = piecewise((second.expr, ne(as_expr(cond.args[0]), as_expr(cond.args[1]))), (first.expr, true))
    return result
