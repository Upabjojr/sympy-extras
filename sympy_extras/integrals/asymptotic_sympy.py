"""The SymPy face of the asymptotic expansions of parametric integrals
(:mod:`sympy_extras.integrals.asymptotic`): Watson's lemma, Laplace's
method, the stationary phase, steepest descent and the uniform
expansions.

The functions take and return SymPy objects (expressions, symbols, and
assumptions written with SymPy's relations and the memberships of
:mod:`sympy_extras.assumptions.facts_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions. Nothing of the
module stays on SymPy.

Examples
========

>>> from sympy import symbols, exp, sqrt, oo, log, cos, sin, pi
>>> from sympy_extras.integrals.asymptotic_sympy import asymptotic_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> asymptotic_integral(exp(-t*x)/(1 + x), x, 0, oo, t)
2/t**3 - 1/t**2 + 1/t + O(t**(-4), (t, oo))
>>> asymptotic_integral(exp(-t*x)*sqrt(x), x, 0, oo, t)
sqrt(pi)/(2*t**(3/2))
>>> asymptotic_integral(exp(t*(log(x) - x)), x, 0, oo, t, order=2)
sqrt(2)*sqrt(pi)*(t + 1/12)*exp(-t)/t**(3/2) + O(exp(-t)/t**(5/2), (t, oo))
>>> asymptotic_integral(cos(t*sin(x))/pi, x, 0, pi, t)
sqrt(2)*sin(t + pi/4)/(sqrt(pi)*sqrt(t)) + O(t**(-3/2), (t, oo))
"""
from __future__ import annotations

from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr

from . import asymptotic as native

__all__ = ['asymptotic_integral', 'watson_lemma', 'laplace_method', 'stationary_phase',
           'steepest_descent', 'uniform_expansion', 'exponential_part', 'oscillatory_part']


def _sympy(e: Optional[CoreExpr]) -> Optional[Expr]:
    """A native expansion as a SymPy expression: the terms and the order
    converted apart and added, the sum left unevaluated when SymPy cannot
    compare the terms with the order (its ``gruntz`` stops at an
    ``exp(I t c)`` or an ``erfc`` of a complex argument)."""
    if e is None:
        return None
    tail = e.getO()
    if tail is None:
        return sympy_expr(e)
    leading, order = sympy_expr(e.removeO()), sympy_expr(tail)
    try:
        return as_expr(leading + order)
    except (NotImplementedError, ValueError, TypeError):
        return Add(leading, order, evaluate=False)


def exponential_part(f: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(phi, h)`` with ``f == phi(x) * exp(t*h(x))``
    (:func:`sympy_extras.integrals.asymptotic.exponential_part` on SymPy
    objects).

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.asymptotic_sympy import exponential_part
    >>> x, t = symbols('x t')
    >>> exponential_part(x**t*exp(-t*x)/(1 + x), x, t)
    (1/(x + 1), -x + log(x))
    """
    found = native.exponential_part(core_expr(f), _core_symbol(x), _core_symbol(t))
    return None if found is None else (sympy_expr(found[0]), sympy_expr(found[1]))


def oscillatory_part(f: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Expr, Expr, str]]:
    """``(phi, h, kind)`` with ``f == phi(x) * K(t*h(x))``
    (:func:`sympy_extras.integrals.asymptotic.oscillatory_part` on SymPy
    objects).

    >>> from sympy import symbols, cos, sin, pi
    >>> from sympy_extras.integrals.asymptotic_sympy import oscillatory_part
    >>> x, t = symbols('x t')
    >>> oscillatory_part(cos(t*sin(x))/pi, x, t)
    (1/pi, sin(x), 'cos')
    """
    found = native.oscillatory_part(core_expr(f), _core_symbol(x), _core_symbol(t))
    return None if found is None else (sympy_expr(found[0]), sympy_expr(found[1]), found[2])


def watson_lemma(phi: ExprLike, x: Symbol, t: Symbol, order: int = 3) -> Optional[Expr]:
    """The expansion of ``Integral(exp(-t*x)*phi(x), (x, 0, oo))`` for
    ``t -> oo`` by Watson's lemma
    (:func:`sympy_extras.integrals.asymptotic.watson_lemma` on SymPy
    objects).

    >>> from sympy import symbols, sin
    >>> from sympy_extras.integrals.asymptotic_sympy import watson_lemma
    >>> x, t = symbols('x t', positive=True)
    >>> watson_lemma(sin(x), x, t, 2)
    -1/t**4 + t**(-2) + O(t**(-6), (t, oo))
    """
    return _sympy(native.watson_lemma(core_expr(phi), _core_symbol(x), _core_symbol(t), order))


def laplace_method(phi: ExprLike, h: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol, order: int = 2,
                   assumptions: Assumptions = None) -> Optional[Expr]:
    """The expansion of ``Integral(phi*exp(t*h), (x, a, b))`` for ``t -> oo``
    by Laplace's method
    (:func:`sympy_extras.integrals.asymptotic.laplace_method` on SymPy
    objects).

    >>> from sympy import symbols, cosh, oo, cos, pi
    >>> from sympy_extras.integrals.asymptotic_sympy import laplace_method
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> laplace_method(1, -cosh(x), x, -oo, oo, t, 2)
    sqrt(2)*sqrt(pi)*(t - 1/8)*exp(-t)/t**(3/2) + O(exp(-t)/t**(5/2), (t, oo))
    >>> laplace_method(1, -x**4, x, -oo, oo, t, 1)
    gamma(1/4)/(2*t**(1/4)) + O(t**(-3/4), (t, oo))
    >>> laplace_method(1, cos(x), x, 0, 2*pi, t, 1)
    sqrt(2)*sqrt(pi)*exp(t)/sqrt(t) + O(exp(t)/t, (t, oo))
    """
    return _sympy(native.laplace_method(core_expr(phi), core_expr(h), _core_symbol(x), core_expr(a), core_expr(b),
                                        _core_symbol(t), order, core_assumptions(assumptions)))


def steepest_descent(phi: ExprLike, h: ExprLike, x: Symbol, t: Symbol, assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(t*h), (x, -oo, oo))`` for
    ``t -> oo`` from the saddle points of ``h`` in the complex plane
    (:func:`sympy_extras.integrals.asymptotic.steepest_descent` on SymPy
    objects).

    >>> from sympy import symbols, I
    >>> from sympy_extras.integrals.asymptotic_sympy import steepest_descent
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> steepest_descent(1, -x**2 + I*x, x, t)
    sqrt(pi)*exp(-t/4)/sqrt(t) + O(exp(-t/4)/t**(3/2), (t, oo))
    """
    return _sympy(native.steepest_descent(core_expr(phi), core_expr(h), _core_symbol(x), _core_symbol(t),
                                          core_assumptions(assumptions)))


def stationary_phase(phi: ExprLike, h: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol, kind: str = 'exp',
                     assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(I*t*h), (x, a, b))`` (or with
    ``cos(t*h)``, ``sin(t*h)``) for ``t -> oo``
    (:func:`sympy_extras.integrals.asymptotic.stationary_phase` on SymPy
    objects).

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.asymptotic_sympy import stationary_phase
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> stationary_phase(1, x**2, x, -1, 1, t)
    sqrt(pi)*exp(I*pi/4)/sqrt(t) + O(t**(-3/2), (t, oo))
    """
    return _sympy(native.stationary_phase(core_expr(phi), core_expr(h), _core_symbol(x), core_expr(a), core_expr(b),
                                          _core_symbol(t), kind, core_assumptions(assumptions)))


def uniform_expansion(phi: ExprLike, h: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol, kind: str = 'exp',
                      assumptions: Assumptions = None) -> Optional[Expr]:
    """The leading term of ``Integral(phi*exp(I*t*h), (x, a, b))`` uniform
    in the parameters of ``h`` as its stationary points coalesce
    (:func:`sympy_extras.integrals.asymptotic.uniform_expansion` on SymPy
    objects).

    >>> from sympy import symbols, oo
    >>> from sympy_extras.integrals.asymptotic_sympy import uniform_expansion
    >>> x = symbols('x')
    >>> a, t = symbols('a t', positive=True)
    >>> uniform_expansion(1, x**3/3 - a*x, x, -oo, oo, t)
    2*pi*airyai(-a*t**(2/3))/t**(1/3) + O(t**(-4/3), (t, oo))
    """
    return _sympy(native.uniform_expansion(core_expr(phi), core_expr(h), _core_symbol(x), core_expr(a), core_expr(b),
                                           _core_symbol(t), kind, core_assumptions(assumptions)))


def asymptotic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, parameter: Symbol,
                        order: int = 3, assumptions: Assumptions = None, uniform: bool = False) -> Optional[Expr]:
    """The first ``order`` terms of the expansion of ``Integral(f, (x, a, b))``
    as ``parameter -> oo``, with an ``Order`` term, or ``None`` when no
    method applies (:func:`sympy_extras.integrals.asymptotic.asymptotic_integral`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import symbols, exp, oo, sqrt, I
    >>> from sympy_extras.integrals.asymptotic_sympy import asymptotic_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> asymptotic_integral(exp(-t*x)*sqrt(1 + x), x, 0, oo, t, order=2)
    1/(2*t**2) + 1/t + O(t**(-3), (t, oo))
    >>> asymptotic_integral(exp(-t*(x**2 - 1)**2), x, -oo, oo, t, order=1)
    sqrt(pi)/sqrt(t) + O(t**(-3/2), (t, oo))
    >>> asymptotic_integral(exp(-t*x**2 + I*t*x), x, -oo, oo, t)
    sqrt(pi)*exp(-t/4)/sqrt(t) + O(exp(-t/4)/t**(3/2), (t, oo))
    """
    return _sympy(native.asymptotic_integral(core_expr(f), _core_symbol(x), core_expr(a), core_expr(b),
                                             _core_symbol(parameter), order, core_assumptions(assumptions), uniform))
