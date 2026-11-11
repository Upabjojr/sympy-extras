"""The SymPy face of the Thue equation solver
(:mod:`sympy_extras.solvers.thue_equation`).

:func:`thue` takes a SymPy form and SymPy symbols, converted by
:mod:`sympy_extras.interop` around the native function; the errors of
the native polynomials come out as SymPy's ``PolynomialError``. The
other functions (:func:`units_of_order`, :func:`elements_of_norm`,
:func:`short_vectors`, :func:`lll`, :class:`ThueEquation`) work on
integers and integer vectors and are the native ones themselves.
Nothing of the solver stays on SymPy.

Examples
========

>>> from sympy.abc import x, y
>>> from sympy_extras.solvers.thue_equation_sympy import thue
>>> thue(x**3 - 2*y**3, 1, x, y)
[(-1, -1), (1, 0)]
"""
from __future__ import annotations

from sympy.core.expr import Expr

from sympy_extras._typing import as_expr
from sympy_extras.assumptions.intervals_sympy import core_expr
from sympy_extras.polys.cad.projection_sympy import sympy_errors

from . import thue_equation as native
from .thue_equation import (OrderElement, ThueEquation, at_precision, elements_of_norm, lll, short_vectors,
    units_of_order)

__all__ = ['thue', 'ThueEquation', 'units_of_order', 'elements_of_norm', 'lll', 'short_vectors', 'OrderElement',
           'at_precision']


def thue(F: Expr, m: int, x: Expr, y: Expr, bound: int = 64) -> list[tuple[int, int]]:
    """All the integer solutions of the Thue equation ``F(x, y) = m`` for
    an irreducible binary form of degree at least 3
    (:func:`sympy_extras.solvers.thue_equation.thue` on SymPy objects).
    ``x`` and ``y`` are the generators, symbols or any other expressions.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.thue_equation_sympy import thue
    >>> thue(x**3 - 2*y**3, 1, x, y)
    [(-1, -1), (1, 0)]
    >>> thue(x**3 - 3*x*y**2 - y**3, 1, x, y)
    [(-3, 2), (-1, 1), (0, -1), (1, -3), (1, 0), (2, 1)]
    """
    form = core_expr(as_expr(F))
    with sympy_errors():
        return native.thue(form, int(m), core_expr(as_expr(x)), core_expr(as_expr(y)), bound)
