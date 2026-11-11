"""The typed entry point of the transcendental Risch algorithm on SymPy
objects (:mod:`sympy_extras.integrals.risch`, ported from SymPy with Aaron
Meurer's extensions): ``risch_antiderivative`` and ``is_nonelementary``,
``risch_integrate`` under the time limit.

Examples
========

>>> from sympy import symbols, tan
>>> from sympy_extras.integrals.risch.risch_sympy import risch_antiderivative
>>> x = symbols('x')
>>> risch_antiderivative(tan(x)**2, x)
-x + tan(x)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral

from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.settings import settings
from .risch import NonElementaryIntegral, risch_integrate

__all__ = ['risch_antiderivative', 'is_nonelementary', 'NonElementaryIntegral']


def _run(f: Expr, x: Symbol) -> Optional[Expr]:
    """``risch_integrate`` under the time limit; ``None`` when it gives up
    (``NotImplementedError``, ``ValueError``, the time)."""
    return attempt(lambda: as_expr(risch_integrate(f, x)), settings.timeout)


def risch_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An elementary antiderivative of ``f`` with respect to ``x`` by the
    transcendental Risch algorithm, or ``None`` when ``f`` has none (the
    algorithm proves it) or when the algorithm does not apply (an
    algebraic function, an unimplemented case, the time limit).

    Examples
    ========

    >>> from sympy import symbols, tan, exp, log, sin
    >>> from sympy_extras.integrals.risch.risch_sympy import risch_antiderivative
    >>> x = symbols('x')
    >>> risch_antiderivative(tan(x), x)
    log(tan(x)**2 + 1)/2
    >>> risch_antiderivative(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> risch_antiderivative(sin(x)/x, x) is None
    True
    """
    f_ = as_expr(f)
    found = _run(f_, x)
    if found is None or found.has(Integral):
        return None
    return found


def is_nonelementary(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the Risch algorithm proves that ``f`` has no elementary
    antiderivative (``True``), finds one (``False``), or cannot tell
    (``None``: an unimplemented case, a non-transcendental tower, the
    time limit).

    Examples
    ========

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.risch.risch_sympy import is_nonelementary
    >>> x = symbols('x')
    >>> is_nonelementary(exp(x)/x, x)
    True
    >>> is_nonelementary(exp(x)*x, x)
    False
    """
    found = _run(as_expr(f), x)
    if found is None:
        return None
    if found.has(NonElementaryIntegral):
        return True
    if found.has(Integral):
        return None
    return False
