"""``refine``: an expression simplified under assumptions.

The simplification is SymPy's ``refine`` (its handlers for ``Abs``,
``Pow``, ``sign``, ``re``, ``im``, ``atan2``, ``Piecewise`` and the
relations live in the packages of SymPy which define those objects),
applied through the conversion with the assumptions given and those of
``global_assumptions``, written with the ``Q`` of this core [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: refine*,
       https://docs.sympy.org/latest/modules/assumptions/refine.html
"""
from __future__ import annotations

import sympy

from .assume import global_assumptions
from .basic import Basic
from .logic import BooleanLike, and_, to_boolean

__all__ = ['refine']


def refine(expr: Basic, assumptions: BooleanLike = True) -> Basic:
    """``expr`` simplified under the assumptions (and the global ones).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Q, refine, sqrt, Eq
    >>> x = Symbol('x')
    >>> refine(sqrt(x**2), Q.positive(x)), refine(sqrt(x**2), Q.real(x)), refine(sqrt(x**2))
    (x, Abs(x), sqrt(x**2))
    >>> refine((-1)**(x + 1), Q.even(x)), refine(Eq(x, 0), Q.positive(x)), refine(x > 0, Q.positive(x))
    (-1, False, True)
    """
    from .interop import through_sympy, to_sympy
    everything = and_(to_boolean(assumptions), *global_assumptions)
    return through_sympy(expr, lambda s: sympy.refine(s, to_sympy(everything)))
