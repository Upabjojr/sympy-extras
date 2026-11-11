"""The step of :mod:`sympy_extras.solvers.inequalities` which stays on
SymPy.

:func:`solve_inequality_of_core` (called from the native module at run
time, on objects of the core) is SymPy's ``_solve_inequality`` of a
relational which is not rational in the variable with rational
coefficients (``solve_univariate_inequality`` is not ported).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp
>>> from sympy_extras.solvers.inequalities_sympy import solve_inequality_of_core
>>> x = Symbol('x')
>>> solve_inequality_of_core(exp(x) > 2, x)
log(2) < x
"""
from __future__ import annotations

from sympy_extras.core.logic import Boolean
from sympy_extras.core.relational import Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.interop import from_sympy, to_sympy

__all__ = ['solve_inequality_of_core']


def solve_inequality_of_core(ie: Relational, s: Symbol) -> Boolean:
    """SymPy's ``_solve_inequality`` of one relational (its exceptions go
    on)."""
    from sympy.solvers.inequalities import _solve_inequality
    r = from_sympy(_solve_inequality(to_sympy(ie), to_sympy(s)))
    if not isinstance(r, Boolean):
        raise TypeError('a condition is expected, not %s' % (r,))
    return r
