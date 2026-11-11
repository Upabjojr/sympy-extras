"""The SymPy face of the convergence tests of series and products
(:mod:`sympy_extras.concrete.convergence`), and the steps which are not
ported.

:func:`sum_convergence`, :func:`product_convergence` and
:func:`is_convergent` take and return SymPy objects (expressions,
symbols, and assumptions written with SymPy's relations and the
memberships of :mod:`sympy_extras.assumptions.facts_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions.

The steps of the native functions which stay written on SymPy are called
from them at run time (the functions ending in ``_of_core``, which take
and return objects of the core), both in the last resort for terms
without parameters: SymPy's ``integrate`` of the integral test
(:func:`improper_integral_of_core`) and SymPy's ``Sum.is_convergent``
(:func:`is_convergent_of_core`).

Examples
========

>>> from sympy import sin
>>> from sympy.abc import n, x, p
>>> from sympy_extras.concrete.convergence_sympy import sum_convergence
>>> sum_convergence(x**n/n, n)
(x >= -1) & (x < 1)
>>> sum_convergence(1/n**p, n)
p > 1
>>> sum_convergence(sin(n)/n, n)
True
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean

from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions, sympy_boolean
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr

from . import convergence as native

__all__ = ['sum_convergence', 'product_convergence', 'is_convergent', 'Condition', 'improper_integral_of_core',
           'is_convergent_of_core']

#: a condition on the parameters (``None`` when undecided)
Condition = Optional[Boolean]


# ---------------------------------------------------------------------------
# the steps of the native functions which stay on SymPy

def improper_integral_of_core(term: CoreExpr, n: CoreSymbol, lower: CoreExpr) -> Optional[CoreExpr]:
    """SymPy's ``integrate(term, (n, lower, oo))`` under the time limit of
    the settings: ``None`` when it does not finish or leaves an
    ``Integral``."""
    from sympy.core.numbers import oo
    from sympy.integrals.integrals import Integral, integrate
    integrand, variable, start = sympy_expr(term), _sympy_symbol(n), sympy_expr(lower)
    value = attempt(lambda: integrate(integrand, (variable, start, oo)), settings.timeout)
    if value is None or not isinstance(value, Expr) or value.has(Integral):
        return None
    return core_expr(value)


def is_convergent_of_core(term: CoreExpr, n: CoreSymbol, lower: CoreExpr) -> Optional[bool]:
    """SymPy's ``Sum(term, (n, lower, oo)).is_convergent()`` under the time
    limit of the settings (``None`` when it does not finish). Its
    ``False`` means "not shown" (sympy-extras#41)."""
    from sympy.concrete.summations import Sum
    from sympy.core.numbers import oo
    summand, variable, start = sympy_expr(term), _sympy_symbol(n), sympy_expr(lower)
    value = attempt(lambda: Sum(summand, (variable, start, oo)).is_convergent(), settings.timeout)
    if value is None:
        return None
    return bool(value)


# ---------------------------------------------------------------------------
# the face

def _condition(condition: native.Condition) -> Condition:
    return None if condition is None else sympy_boolean(condition)


def sum_convergence(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                    lower: Union[Expr, int] = 1) -> Condition:
    """The condition on the parameters under which ``Sum(term, (n, lower,
    oo))`` converges: ``S.true``, ``S.false``, a Boolean in the
    parameters, or ``None`` when the tests do not decide
    (:func:`sympy_extras.concrete.convergence.sum_convergence` on SymPy
    objects).

    Parameters
    ==========

    term : Expr
        The summand, a function of ``n`` and of parameters.
    n : Symbol
        The summation variable (an integer tending to infinity).
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters, as statements.
    lower : Expr, optional
        The lower limit of the sum (it does not affect convergence, but
        the terms must be defined from it on).

    Examples
    ========

    >>> from sympy import factorial, binomial, log
    >>> from sympy.abc import n, x, p
    >>> from sympy_extras.concrete.convergence_sympy import sum_convergence
    >>> sum_convergence(x**n, n)
    (x > -1) & (x < 1)
    >>> sum_convergence(x**n/n, n)
    (x >= -1) & (x < 1)
    >>> sum_convergence(1/n**p, n)
    p > 1
    >>> sum_convergence(1/n**p, n, p > 2)
    True
    >>> sum_convergence(factorial(n)/n**n, n)
    True
    >>> sum_convergence(binomial(2*n, n)/4**n, n)
    False
    >>> sum_convergence((-1)**n/log(n), n, lower=2)
    True
    >>> sum_convergence(1/(n*log(n)), n, lower=2)
    False
    """
    return _condition(native.sum_convergence(core_expr(sympify(term)), _core_symbol(n), core_assumptions(assumptions),
                                             core_expr(sympify(lower))))


def is_convergent(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                  lower: Union[Expr, int] = 1) -> Optional[bool]:
    """Whether ``Sum(term, (n, lower, oo))`` converges under the
    assumptions: ``True``, ``False`` or ``None``
    (:func:`sympy_extras.concrete.convergence.is_convergent` on SymPy
    objects).

    Examples
    ========

    >>> from sympy.abc import n, x
    >>> from sympy_extras.concrete.convergence_sympy import is_convergent
    >>> is_convergent(x**n, n, (x > 0) & (x < 1))
    True
    >>> is_convergent(x**n, n, x > 1)
    False
    >>> is_convergent(x**n, n) is None
    True
    """
    return native.is_convergent(core_expr(sympify(term)), _core_symbol(n), core_assumptions(assumptions),
                                core_expr(sympify(lower)))


def product_convergence(term: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
                        lower: Union[Expr, int] = 1) -> Condition:
    """The condition under which the infinite product ``Product(term,
    (n, lower, oo))`` converges to a nonzero limit
    (:func:`sympy_extras.concrete.convergence.product_convergence` on SymPy
    objects).

    Examples
    ========

    >>> from sympy.abc import n, x
    >>> from sympy_extras.concrete.convergence_sympy import product_convergence
    >>> product_convergence(1 + 1/n**2, n)
    True
    >>> product_convergence(1 + x/n**2, n)
    True
    >>> product_convergence(1 + 1/n, n)
    False
    >>> product_convergence(1 + x**n, n)
    (x > -1) & (x < 1)
    """
    return _condition(native.product_convergence(core_expr(sympify(term)), _core_symbol(n),
                                                 core_assumptions(assumptions), core_expr(sympify(lower))))
