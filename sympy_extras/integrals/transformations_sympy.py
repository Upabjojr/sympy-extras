"""The SymPy face of Frullani's theorem, the Cauchy–Schlömilch
transformation and Glasser's master theorem
(:mod:`sympy_extras.integrals.transformations`), for the integrators which
are still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`). The transformed integrals of the native module
are computed by :func:`~sympy_extras.integrals.definite.conditional_integral`
through :func:`conditional_integral_of_core` (of :mod:`.laplace_sympy`).

Examples
========

>>> from sympy import symbols, exp, cos, atan, oo
>>> from sympy_extras.integrals.transformations_sympy import frullani, glasser
>>> x = symbols('x')
>>> a, b = symbols('a b', positive=True)
>>> frullani((exp(-a*x) - exp(-b*x))/x, x, a > 0)
ConditionalValue(log(b/a))
>>> glasser(exp(-(x - 1/x)**2), x, 0, oo)
ConditionalValue(sqrt(pi)/2)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, core_assumptions

from . import conditions as native_conditions
from . import transformations as native
from .conditions_sympy import ConditionalValue
from .laplace_sympy import conditional_integral_of_core

__all__ = ['frullani', 'glasser', 'transformation_integral', 'conditional_integral_of_core']


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def frullani(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.transformations.frullani` on SymPy
    objects: ``Integral(f, (x, 0, oo))`` by Frullani's theorem for ``f =
    c*(g(a*x) - g(b*x))/x``; ``None`` when ``f`` is not of this shape or
    the limits of ``g`` at ``0`` and ``oo`` are not finite. The condition
    ``a > 0`` and ``b > 0`` is reported.

    >>> from sympy import symbols, atan
    >>> from sympy_extras.integrals.transformations_sympy import frullani
    >>> x = symbols('x')
    >>> a, b = symbols('a b', positive=True)
    >>> frullani((atan(a*x) - atan(b*x))/x, x)
    ConditionalValue(-pi*log(b/a)/2)
    """
    return _conditional(native.frullani(core_expr(sympify(f)), _core_symbol(x), core_assumptions(assumptions)))


def glasser(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.transformations.glasser` on SymPy
    objects: ``Integral(f, (x, a, b))`` by Glasser's master theorem, for
    ``f = F(u(x))`` with ``u = x - sum(a_i/(x - b_i))``, ``a_i > 0``, over
    the real line, or over ``(0, oo)`` when ``F`` is even; ``None`` when
    the shape is not recognised.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.transformations_sympy import glasser
    >>> x = symbols('x')
    >>> glasser(1/(1 + (x - 1/x)**2), x, 0, oo)
    ConditionalValue(pi/2)
    """
    return _conditional(native.glasser(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                       core_expr(sympify(b)), core_assumptions(assumptions)))


def transformation_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                            assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.transformations.transformation_integral`
    on SymPy objects: the first of the identities of the module which
    applies.

    >>> from sympy import symbols, exp, S
    >>> from sympy_extras.integrals.transformations_sympy import transformation_integral
    >>> x = symbols('x')
    >>> transformation_integral(exp(-x), x, S.Zero, S.One) is None
    True
    """
    return _conditional(native.transformation_integral(core_expr(sympify(f)), _core_symbol(x),
                                                       core_expr(sympify(a)), core_expr(sympify(b)),
                                                       core_assumptions(assumptions)))
