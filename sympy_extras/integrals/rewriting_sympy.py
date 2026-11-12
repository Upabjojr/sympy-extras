"""The SymPy face of the canonical forms and substitutions before
integration (:mod:`sympy_extras.integrals.rewriting`), for the
integrators which are still written on SymPy (:mod:`.indefinite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (:class:`Substitution` with
SymPy attributes). The new variables are the dummies of the native module,
each with a name of its own (``_t3``).

Examples
========

>>> from sympy import symbols, exp, sinh, sqrt
>>> from sympy_extras.integrals.rewriting_sympy import rewritten_forms, power_substitutions
>>> x, z, t = symbols('x z t')
>>> a, c = symbols('a c', positive=True)
>>> rewritten_forms(a**(c*z)*exp(z**2), z)
[exp(z**2)*exp(c*z*log(a)), exp(c*z*log(a) + z**2)]
>>> rewritten_forms(x/sinh(x + 2), x)
[x/(-exp(-x - 2)/2 + exp(x + 2)/2)]
>>> s = power_substitutions(sqrt(x)/(1 + x), x)[0]
>>> s.integrand.subs(s.variable, t), s.back
(2*t**2/(t**2 + 1), sqrt(x))
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import (Assumptions, _core_symbol, _sympy_symbol, core_assumptions, core_boolean,
                                                sympy_boolean)

from . import rewriting as native

__all__ = ['Substitution', 'rewritten_forms', 'power_substitutions', 'substitute_back', 'implied_assumptions']


class Substitution:
    """:class:`sympy_extras.integrals.rewriting.Substitution` on SymPy
    objects: a change of variable ``x -> t``, ``integrand`` is ``g(t)`` with
    ``f(x) dx = g(t) dt``, ``variable`` the new symbol ``t``, ``back`` the
    expression of ``t`` in ``x``, ``forward`` that of ``x`` in ``t`` and
    ``facts`` the facts about ``x`` under which it holds."""

    def __init__(self, integrand: Expr, variable: Symbol, back: Expr, forward: Expr,
                 facts: Optional[list[Boolean]] = None) -> None:
        self.integrand: Final[Expr] = integrand
        self.variable: Final[Symbol] = variable
        self.back: Final[Expr] = back
        self.forward: Final[Expr] = forward
        self.facts: Final[list[Boolean]] = list(facts or [])

    def __repr__(self) -> str:
        return "Substitution(%s, %s, back=%s)" % (self.integrand, self.variable, self.back)

    @staticmethod
    def from_core(s: native.Substitution) -> Substitution:
        return Substitution(sympy_expr(s.integrand), _sympy_symbol(s.variable), sympy_expr(s.back),
                            sympy_expr(s.forward), [sympy_boolean(fact) for fact in s.facts])

    def to_core(self) -> native.Substitution:
        return native.Substitution(core_expr(self.integrand), _core_symbol(self.variable), core_expr(self.back),
                                   core_expr(self.forward), [core_boolean(fact) for fact in self.facts])


def implied_assumptions(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Boolean]:
    """:func:`sympy_extras.integrals.rewriting.implied_assumptions` on SymPy
    objects: the facts the canonical forms of ``f`` assume beyond the given
    assumptions (``a > 0`` for each base ``a`` written as an exponential).

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.rewriting_sympy import implied_assumptions
    >>> c, d, z = symbols('c d z')
    >>> implied_assumptions(c**(d*z), z)
    [c > 0]
    """
    found = native.implied_assumptions(core_expr(sympify(f)), _core_symbol(x), core_assumptions(assumptions))
    return [sympy_boolean(fact) for fact in found]


def rewritten_forms(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> list[Expr]:
    """:func:`sympy_extras.integrals.rewriting.rewritten_forms` on SymPy
    objects: canonical forms of ``f`` for integration in ``x``, each equal
    to ``f`` wherever ``f`` is real, in order of preference, at most five.

    >>> from sympy import symbols, exp, cosh, asinh, sqrt, Abs
    >>> from sympy_extras.integrals.rewriting_sympy import rewritten_forms
    >>> x = symbols('x')
    >>> v = symbols('v', real=True)
    >>> rewritten_forms(exp(x**2)**v*exp(3*x), x)
    [exp(3*x)*exp(v*x**2), exp(v*x**2 + 3*x)]
    >>> rewritten_forms(2**x*cosh(x), x)
    [(exp(x)/2 + exp(-x)/2)*exp(x*log(2)), exp(-x + x*log(2))/2 + exp(x*log(2) + x)/2]
    >>> rewritten_forms(x*Abs(x), x, [x > 0])
    [x**2]
    >>> rewritten_forms(asinh(x)/sqrt(x**2 + 1), x)
    [log(x + sqrt(x**2 + 1))/sqrt(x**2 + 1)]
    """
    found = native.rewritten_forms(core_expr(sympify(f)), _core_symbol(x), core_assumptions(assumptions))
    return [sympy_expr(form) for form in found]


def power_substitutions(f: ExprLike, x: Symbol) -> list[Substitution]:
    """:func:`sympy_extras.integrals.rewriting.power_substitutions` on SymPy
    objects: the substitutions ``x = t**k``, ``u = exp(c*x)``, ``x =
    exp(t)``, ``s = x**2``, Chebyshev's, the Möbius roots by region and
    those of nested radicals; most promising first, at most three.

    >>> from sympy import symbols, sqrt, exp, log, S
    >>> from sympy_extras.integrals.rewriting_sympy import power_substitutions
    >>> x, t = symbols('x t')
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(1/(sqrt(x) + x**(S(1)/3)), x)]
    [6*t**3/(t + 1)]
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(exp(x)/(exp(2*x) - 1), x)]
    [1/(t**2 - 1)]
    >>> [s.integrand.subs(s.variable, t) for s in power_substitutions(1/(x*log(x)**2), x)]
    [t**(-2)]
    """
    found = native.power_substitutions(core_expr(sympify(f)), _core_symbol(x))
    return [Substitution.from_core(s) for s in found]


def substitute_back(F: ExprLike, t: Symbol, back: ExprLike) -> Expr:
    """:func:`sympy_extras.integrals.rewriting.substitute_back` on SymPy
    objects: the antiderivative ``F(t)`` returned to ``x``, ``F(back(x))``.

    >>> from sympy import symbols, sqrt, atan
    >>> from sympy_extras.integrals.rewriting_sympy import substitute_back
    >>> x, t = symbols('x t')
    >>> substitute_back(2*t - 2*atan(t), t, sqrt(x))
    2*sqrt(x) - 2*atan(sqrt(x))
    """
    return sympy_expr(native.substitute_back(core_expr(sympify(F)), _core_symbol(t), core_expr(sympify(back))))
