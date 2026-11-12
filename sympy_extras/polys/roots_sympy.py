"""The SymPy face of the roots in radicals (:mod:`sympy_extras.polys.roots`).

:func:`radical_form` and :func:`in_radicals` take and return SymPy
objects (``CRootOf`` and sets of SymPy), converted by
:mod:`sympy_extras.interop` around the native functions.

Examples
========

>>> from sympy import CRootOf, FiniteSet
>>> from sympy.abc import x
>>> from sympy_extras.polys.roots_sympy import radical_form, in_radicals
>>> radical_form(CRootOf(x**6 - 2*x**3 - 1, 1))
(1 + sqrt(2))**(1/3)
>>> in_radicals(FiniteSet(CRootOf(x**2 - 2, 0), CRootOf(x**5 - x - 1, 0)))
{-sqrt(2), CRootOf(x**5 - x - 1, 0)}
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.polys.rootoftools import ComplexRootOf
from sympy.sets.sets import Set

from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr, sympy_set
from sympy_extras.interop import from_sympy
from sympy_extras.sets.sets import Set as CoreSet

from . import roots as native

__all__ = ['radical_form', 'in_radicals']


def radical_form(root: ComplexRootOf) -> Optional[Expr]:
    """The root written in radicals, or ``None`` when it is left as a root
    object (:func:`sympy_extras.polys.roots.radical_form`).

    Examples
    ========

    >>> from sympy import CRootOf
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.roots_sympy import radical_form
    >>> radical_form(CRootOf(x**2 - 2, 1))
    sqrt(2)
    >>> radical_form(CRootOf(x**6 - 2*x**3 - 1, 1))
    (1 + sqrt(2))**(1/3)
    >>> radical_form(CRootOf(x**5 - x - 1, 0)) is None
    True
    """
    found = native.radical_form(core_expr(root))
    return None if found is None else sympy_expr(found)


def in_radicals(result: Set) -> Set:
    """The set with its root objects written in radicals when
    :func:`radical_form` allows (:func:`sympy_extras.polys.roots.in_radicals`).

    Examples
    ========

    >>> from sympy import CRootOf, FiniteSet
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.roots_sympy import in_radicals
    >>> in_radicals(FiniteSet(CRootOf(x**2 - 2, 0), CRootOf(x**5 - x - 1, 0)))
    {-sqrt(2), CRootOf(x**5 - x - 1, 0)}
    """
    converted = from_sympy(result)
    if not isinstance(converted, CoreSet):
        raise TypeError("expected a set, got %s" % (result,))
    found = native.in_radicals(converted)
    return result if found is converted else sympy_set(found)
