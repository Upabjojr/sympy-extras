"""``together``: a sum of rational expressions over a common denominator,
without expanding or cancelling (``1/x + 1/y`` is ``(x + y)/(x*y)``).

Ported from SymPy's ``sympy.polys.rationaltools``: the sums are put
together bottom up by :func:`~sympy_extras.core.exprtools.gcd_terms`
(which takes the common factors of the terms out), the arguments of
functions only with ``deep=True``.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp
>>> x, y = Symbol('x'), Symbol('y')
>>> together(1/x + 1/y), together(1/(x*y) + 1/y**2), together(exp(1/x + 1/y)), together(exp(1/x + 1/y), deep=True)
((x + y)/(x*y), (x + y)/(x*y**2), exp(1/y + 1/x), exp((x + y)/(x*y)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rationaltools.py (together) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; lists of expressions are not accepted (map over them).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Final

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.exprtools import gcd_terms
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.power import Pow, pow

__all__ = ['together']


class _Together:
    __slots__ = ('deep', 'fraction')

    def __init__(self, deep: bool, fraction: bool) -> None:
        self.deep: Final[bool] = deep
        self.fraction: Final[bool] = fraction

    def __call__(self, e: Basic) -> Basic:
        if not e.args or (e.is_Function and not self.deep):
            return e
        if isinstance(e, Add):
            return gcd_terms([_expr(self(a)) for a in e.args], fraction=self.fraction)
        if isinstance(e, Pow):
            base = _expr(self(e.base))
            exponent = _expr(self(e.exp)) if self.deep else e.exp
            return pow(base, exponent)
        return e._evaluated(tuple(self(a) for a in e.args))


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


# PORTED FROM SYMPY: sympy/polys/rationaltools.py:together
def together(expr: ExprLike, deep: bool = False, fraction: bool = True) -> Expr:
    """The sums of ``expr`` over common denominators, bottom up, with the
    common factors of their terms taken out (``gcd_terms``), not expanded;
    the arguments of functions too with ``deep``; with ``fraction=False``
    the denominators of the terms are kept apart."""
    return _expr(_Together(deep, fraction)(to_expr(expr)))
