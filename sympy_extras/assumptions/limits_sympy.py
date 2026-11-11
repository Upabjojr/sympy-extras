"""The SymPy face of limits and series expansions under statement
assumptions (:mod:`sympy_extras.assumptions.limits`), and the steps which
are not ported.

:func:`limit`, :func:`limit_seq` and :func:`series` take and return SymPy
objects (expressions, symbols, and assumptions written with SymPy's
relations and the memberships of
:mod:`sympy_extras.assumptions.facts_sympy`), converted by
:mod:`sympy_extras.interop` around the native functions.

The steps of the native functions which stay written on SymPy are called
from them at run time (the functions ending in ``_of_core``, which take
and return objects of the core): SymPy's ``limit_seq``
(:func:`sympy.series.limitseq.limit_seq`, not ported).

Examples
========

>>> from sympy import exp, oo, symbols
>>> from sympy_extras.assumptions.limits_sympy import limit, limit_seq
>>> a, n, x = symbols('a n x')
>>> limit(exp(a*x), x, oo, assumptions=a < 0)
0
>>> limit_seq(a**n, n, assumptions=a > 0)
Piecewise((oo, a > 1), (1, Eq(a, 1)), (0, a < 1))
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.sets.sets import Set

from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import limits as native
from .ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions, core_domain
from .intervals_sympy import core_expr, sympy_expr

__all__ = ['limit', 'limit_seq', 'series', 'limit_seq_of_core']


# ---------------------------------------------------------------------------
# the steps of the native functions which stay on SymPy

def limit_seq_of_core(e: CoreExpr, n: CoreSymbol) -> Optional[CoreExpr]:
    """SymPy's :func:`~sympy.series.limitseq.limit_seq` of a sequence in
    ``n`` (``None`` when it gives up; its ``NotImplementedError`` when a
    result depends on a sign is passed on)."""
    from sympy.series.limitseq import limit_seq as _limit_seq
    value = _limit_seq(sympy_expr(e), _sympy_symbol(n))
    return None if value is None else core_expr(sympify(value))


# ---------------------------------------------------------------------------
# the face

def limit(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int], direction: str = '+',
          assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The limit of ``expr`` as ``x`` tends to ``x0`` under assumptions on
    the parameters (:func:`sympy_extras.assumptions.limits.limit` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import exp, oo, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions.limits_sympy import limit
    >>> limit(exp(a*x), x, oo, assumptions=a < 0)
    0
    >>> limit(exp(a*x), x, oo)
    Piecewise((oo, a > 0), (1, Eq(a, 0)), (0, a < 0))
    >>> limit(x**a, x, oo, assumptions=a > 0)
    oo
    >>> limit((1 + a/x)**x, x, oo)
    exp(a)
    >>> limit(x**a*log(x), x, 0, assumptions=a > 0)
    0
    """
    return sympy_expr(native.limit(core_expr(sympify(expr)), _core_symbol(x), core_expr(sympify(x0)), direction,
                                   core_assumptions(assumptions), core_domain(domain)))


def limit_seq(expr: Union[Expr, int], n: Symbol, assumptions: Assumptions = None,
              domain: Optional[Set] = None) -> Expr:
    """The limit of the sequence ``expr`` as the integer ``n`` tends to
    infinity under assumptions on the parameters
    (:func:`sympy_extras.assumptions.limits.limit_seq` on SymPy objects).

    Examples
    ========

    >>> from sympy import factorial, binomial
    >>> from sympy.abc import a, n, x
    >>> from sympy_extras.assumptions.limits_sympy import limit_seq
    >>> limit_seq(x**n/factorial(n), n)
    0
    >>> limit_seq(a**n, n, assumptions=(a > 0) & (a < 1))
    0
    >>> limit_seq(a**n, n, assumptions=a > 0)
    Piecewise((oo, a > 1), (1, Eq(a, 1)), (0, a < 1))
    >>> limit_seq(a**n, n, assumptions=(a > -1) & (a < 0))
    0
    >>> limit_seq((1 + x/n)**n, n)
    exp(x)
    >>> limit_seq(binomial(2*n, n)/4**n, n)
    0
    """
    return sympy_expr(native.limit_seq(core_expr(sympify(expr)), _core_symbol(n), core_assumptions(assumptions),
                                       core_domain(domain)))


def series(expr: Union[Expr, int], x: Symbol, x0: Union[Expr, int] = 0, n: int = 6,
           assumptions: Assumptions = None, domain: Optional[Set] = None) -> Expr:
    """The series expansion of ``expr`` about ``x0`` with the parameters
    carrying the assumptions (:func:`sympy_extras.assumptions.limits.series`
    on SymPy objects).

    Examples
    ========

    >>> from sympy import sqrt, log
    >>> from sympy.abc import a, x
    >>> from sympy_extras.assumptions.limits_sympy import series
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a > 0)
    x/(2*a) + a + O(x**2)
    >>> series(sqrt(a**2 + x), x, 0, 2, assumptions=a < 0)
    -x/(2*a) - a + O(x**2)
    >>> series(log(a*x), x, 1, 2, assumptions=a > 0)
    -1 + log(a) + x + O((x - 1)**2, (x, 1))
    """
    return sympy_expr(native.series(core_expr(sympify(expr)), _core_symbol(x), core_expr(sympify(x0)), n,
                                    core_assumptions(assumptions), core_domain(domain)))
