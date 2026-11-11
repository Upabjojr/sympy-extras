"""The SymPy face of the heuristic Risch integrator
(:mod:`sympy_extras.integrals.heurisch`), for the integrators which are
still written on SymPy.

The functions take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native ones; nothing of the method
stays on SymPy.

Examples
========

>>> from sympy import symbols, exp, cos
>>> from sympy_extras.integrals.heurisch_sympy import heurisch_antiderivative, heurisch_cases
>>> x, y, n = symbols('x y n')
>>> heurisch_antiderivative(y*x*cos(x**2), x)
y*sin(x**2)/2
>>> heurisch_antiderivative(exp(x)/x, x)
Ei(x)
>>> heurisch_cases(cos(n*x), x)
Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy

from . import heurisch as native

__all__ = ['components', 'heurisch_antiderivative', 'heurisch_cases']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def components(f: ExprLike, x: Symbol) -> set[Expr]:
    """:func:`sympy_extras.integrals.heurisch.components` on SymPy
    objects: the functional components of ``f`` in ``x``.

    >>> from sympy import symbols, sin, cos
    >>> from sympy_extras.integrals.heurisch_sympy import components
    >>> x = symbols('x')
    >>> sorted(components(sin(x)*cos(x)**2, x), key=str)
    [cos(x), sin(x), x]
    """
    return {sympy_expr(g) for g in native.components(core_expr(as_expr(f)), _core_symbol(x))}


def _real_forms(F: Expr) -> Expr:
    """:func:`sympy_extras.integrals.heurisch._real_forms` on SymPy
    objects.

    >>> from sympy import symbols, I, log
    >>> from sympy_extras.integrals.heurisch_sympy import _real_forms
    >>> x = symbols('x')
    >>> _real_forms(-I*log(x - I)/2 + I*log(x + I)/2)
    atan(x)
    """
    return sympy_expr(native._real_forms(core_expr(F)))


def _verified(F: Expr, f: Expr, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.heurisch._verified` on SymPy objects:
    ``F`` when ``F' == f``, symbolically or numerically where ``f`` is
    real; ``None`` otherwise."""
    found = native._verified(core_expr(F), core_expr(f), _core_symbol(x))
    return None if found is None else sympy_expr(found)


def heurisch_antiderivative(f: ExprLike, x: Symbol, hints: Optional[Sequence[ExprLike]] = None,
                            degree_offset: int = 0, unnecessary_permutations: Optional[list[tuple[Expr, Symbol]]] = None,
                            retries: int = 2) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.heurisch.heurisch_antiderivative` on
    SymPy objects: an antiderivative of ``f`` in ``x`` by the heuristic
    Risch method, verified by differentiation, or ``None``.

    >>> from sympy import symbols, tan, sqrt
    >>> from sympy_extras.integrals.heurisch_sympy import heurisch_antiderivative
    >>> x, y = symbols('x y')
    >>> heurisch_antiderivative(y*tan(x), x)
    y*log(tan(x)**2 + 1)/2
    >>> heurisch_antiderivative(1/sqrt(x**2 + 1), x)
    asinh(x)
    """
    hint_terms = [core_expr(as_expr(h)) for h in hints] if hints else None
    permutations = None if unnecessary_permutations is None else \
        [(core_expr(t), _core_symbol(v)) for t, v in unnecessary_permutations]
    found = native.heurisch_antiderivative(core_expr(as_expr(f)), _core_symbol(x), hint_terms, degree_offset,
                                           permutations, retries)
    return None if found is None else sympy_expr(found)


def heurisch_cases(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.heurisch.heurisch_cases` on SymPy
    objects: the antiderivative of ``f`` with the cases of the parameters
    where a denominator of the generic antiderivative vanishes.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.heurisch_sympy import heurisch_cases
    >>> a, x = symbols('a x')
    >>> heurisch_cases(exp(a*x), x)
    Piecewise((exp(a*x)/a, Ne(a, 0)), (x, True))
    """
    found = native.heurisch_cases(core_expr(as_expr(f)), _core_symbol(x))
    return None if found is None else sympy_expr(found)
