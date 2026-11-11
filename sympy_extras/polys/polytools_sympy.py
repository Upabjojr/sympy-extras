"""The steps of :mod:`sympy_extras.polys.polytools` which stay on SymPy:
the coefficient domains which the typed layer does not have.

:mod:`sympy_extras.polys.polytools` computes over ``ZZ``, ``QQ``,
``ZZ[a, ...]``, ``QQ[a, ...]``, the Gaussian domains ``ZZ_I`` and ``QQ_I``
(without generators given) and ``RR``; the functions here are what it
calls (importing this module where it needs them) when the expression
needs another domain:

* :func:`invert_of_core`: SymPy's ``invert`` of two numbers which are not
  both rational (a domain ``ZZ_I``, ``QQ_I``, ``RR``, ``CC`` or ``EX``);
* :func:`cancel_of_core`: SymPy's ``cancel`` over ``CC`` and ``EX``, and
  with generators over a coefficient domain other than ``ZZ[...]`` and
  ``QQ[...]`` or with a generator which is not a factor of the expression
  (``x**2`` in an expression in ``x``);
* :func:`factor_of_core`: SymPy's ``factor`` over ``EX``, and over
  ``ZZ_I``, ``QQ_I`` and ``RR`` with generators given.

Each takes and returns core expressions (converted by
:mod:`sympy_extras.interop`).

Examples
========

>>> from sympy_extras.core import Float, I, Integer, Symbol
>>> from sympy_extras.polys.polytools_sympy import cancel_of_core, factor_of_core, invert_of_core
>>> x = Symbol('x')
>>> invert_of_core(I, Integer(3))
-I
>>> cancel_of_core((Float(1.5) + I)*x/(x**2 + x), ())
(1.5 + 1.0*I)/(1.0*x + 1.0)
>>> factor_of_core(x**2 + 2*I*x - 1, (x,), True)
(x + I)**2
"""
from __future__ import annotations

from typing import Sequence

import sympy

from sympy_extras.core.expr import Expr
from sympy_extras.interop import expr_from_sympy, expr_through_sympy, to_sympy

__all__ = ['invert_of_core', 'cancel_of_core', 'factor_of_core']


def invert_of_core(a: Expr, b: Expr) -> Expr:
    """SymPy's ``invert(a, b)``: the inverse of ``a`` modulo ``b``."""
    return expr_from_sympy(sympy.invert(to_sympy(a), to_sympy(b)))


def cancel_of_core(e: Expr, gens: Sequence[Expr]) -> Expr:
    """SymPy's ``cancel(e, *gens)``."""
    sympy_gens = [to_sympy(g) for g in gens]
    return expr_through_sympy(e, lambda s: sympy.cancel(s, *sympy_gens))


def factor_of_core(e: Expr, gens: Sequence[Expr], fraction: bool) -> Expr:
    """SymPy's ``factor(e, *gens, fraction=fraction)``."""
    sympy_gens = [to_sympy(g) for g in gens]
    return expr_through_sympy(e, lambda s: sympy.factor(s, *sympy_gens, fraction=fraction))
