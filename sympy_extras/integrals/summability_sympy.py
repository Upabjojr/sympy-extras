"""The SymPy face of the summability methods for divergent oscillatory
integrals (:mod:`sympy_extras.integrals.summability`), for the integrators
which are still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The regularised integrals of the native module
are computed by :func:`~sympy_extras.integrals.definite.conditional_integral`
through :func:`conditional_integral_of_core` (of :mod:`.laplace_sympy`).

Examples
========

>>> from sympy import symbols, sin, cos, exp, oo
>>> from sympy_extras.integrals.summability_sympy import summable_integral
>>> x = symbols('x')
>>> summable_integral(sin(x), x, 0, oo, 'abel')
ConditionalValue(1)
>>> summable_integral(cos(x), x, 0, oo, 'cesaro')
ConditionalValue(0)
>>> summable_integral(x*cos(x), x, 0, oo, 'abel')
ConditionalValue(-1)
>>> summable_integral(sin(x)**2, x, 0, oo, 'abel') is None
True
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_assumptions

from . import conditions as native_conditions
from . import summability as native
from .conditions_sympy import ConditionalValue
from .laplace_sympy import conditional_integral_of_core

__all__ = ['summable_integral', 'oscillatory_tail', 'METHODS', 'conditional_integral_of_core']

#: the summability methods (:data:`sympy_extras.integrals.summability.METHODS`)
METHODS = native.METHODS


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def summable_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, method: str,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.summability.summable_integral` on
    SymPy objects: the ``method`` mean (``'abel'``, ``'cesaro'`` or
    ``'gaussian'``) of ``Integral(f, (x, a, b))`` with the condition on
    the parameters under which it holds, or ``None`` when the mean does
    not exist or cannot be found.

    Examples
    ========

    >>> from sympy import symbols, sin, cos, exp, I, oo
    >>> from sympy_extras.integrals.summability_sympy import summable_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> summable_integral(sin(a*x), x, 0, oo, 'abel')
    ConditionalValue(1/a)
    >>> summable_integral(exp(I*x), x, 0, oo, 'gaussian')
    ConditionalValue(I)
    >>> summable_integral(cos(x), x, -oo, oo, 'abel')
    ConditionalValue(0)
    >>> summable_integral(x, x, 0, oo, 'abel') is None
    True
    """
    return _conditional(native.summable_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                                 core_expr(sympify(b)), method, core_assumptions(assumptions)))


def oscillatory_tail(f: ExprLike, x: Symbol, a: ExprLike, method: str,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.summability.oscillatory_tail` on SymPy
    objects: the ``method`` mean of the tail ``Integral(f, (x, a, oo))``,
    or ``None``.

    Examples
    ========

    >>> from sympy import symbols, sin, oo, pi
    >>> from sympy_extras.integrals.summability_sympy import oscillatory_tail
    >>> x = symbols('x')
    >>> oscillatory_tail(sin(x), x, pi, 'abel')
    ConditionalValue(-1)
    >>> oscillatory_tail(x*sin(x), x, 0, 'cesaro')
    ConditionalValue(0)
    """
    return _conditional(native.oscillatory_tail(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                                method, core_assumptions(assumptions)))


def _cesaro_mean(g: Expr, t: Symbol, order: int, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.summability._cesaro_mean` on SymPy
    objects: the ``(C, order)`` mean of ``Integral(g, (t, 0, oo))``."""
    return _conditional(native._cesaro_mean(core_expr(g), _core_symbol(t), order, core_assumptions(assumptions)))
