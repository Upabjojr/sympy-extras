"""The SymPy face of the rational integrator
(:mod:`sympy_extras.integrals.risch.rationaltools`): ``ratint``,
``ratint_ratpart``, ``ratint_logpart``, ``log_to_atan`` and
``log_to_real`` on SymPy expressions and ``Poly`` objects, for the
users of SymPy objects (the faces of :mod:`..trigonometric` and
:mod:`..radicals`).

The arguments are converted by :mod:`sympy_extras.interop` (a ``Poly`` by
its expression), the native functions run on the core, and the results
are converted back: a ``RootSum`` of the core becomes SymPy's
``RootSum``, the pairs of :func:`ratint_logpart` SymPy's ``Poly`` in
``x`` and in ``t``.

Examples
========

>>> from sympy import Poly, symbols
>>> from sympy_extras.integrals.risch.rationaltools_sympy import ratint, ratint_logpart
>>> x, t = symbols('x t')
>>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)
(12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)
>>> ratint_logpart(Poly(1, x), Poly(x**2 + x + 1, x), x, t)
[(Poly(x + 3*t/2 + 1/2, x, domain='QQ[t]'), Poly(3*t**2 + 1, t, domain='ZZ'))]
"""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras.core.symbol import Dummy as CoreDummy, Symbol as CoreSymbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.interop import from_sympy, to_sympy

from . import rationaltools as native

__all__ = ['ratint', 'ratint_ratpart', 'ratint_logpart', 'log_to_atan', 'log_to_real']


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (converted,))
    return converted


def _core(e: Union[Poly, Expr, int]) -> CoreExpr:
    if isinstance(e, Poly):
        return core_expr(e.as_expr())
    return core_expr(e)


def ratint(f: Union[Expr, tuple[Expr, Expr]], x: Symbol, *, symbol: Union[str, Symbol] = 't',
           real: Optional[bool] = None) -> Expr:
    """:func:`sympy_extras.integrals.risch.rationaltools.ratint` on SymPy
    objects: an antiderivative of the rational function ``f`` (or of
    ``p/q`` given as the pair ``(p, q)``).

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.risch.rationaltools_sympy import ratint
    >>> x = symbols('x')
    >>> ratint(1/(x**2 + 1), x), ratint(1/(x**2 + 1), x, real=False)
    (atan(x), -I*log(x - I)/2 + I*log(x + I)/2)
    """
    integrand: Union[CoreExpr, tuple[CoreExpr, CoreExpr]]
    if isinstance(f, tuple):
        integrand = (_core(f[0]), _core(f[1]))
    else:
        integrand = _core(f)
    name: Union[str, CoreSymbol] = symbol if isinstance(symbol, str) else _core_symbol(symbol)
    return sympy_expr(native.ratint(integrand, _core_symbol(x), symbol=name, real=real))


def ratint_ratpart(f: Union[Poly, Expr], g: Union[Poly, Expr], x: Symbol) -> tuple[Expr, Expr]:
    """:func:`sympy_extras.integrals.risch.rationaltools.ratint_ratpart` on
    SymPy objects (Horowitz–Ostrogradsky).

    >>> from sympy import Poly, symbols
    >>> from sympy_extras.integrals.risch.rationaltools_sympy import ratint_ratpart
    >>> x = symbols('x')
    >>> ratint_ratpart(Poly(36, x), Poly(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x), x)
    ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))
    """
    A, B = native.ratint_ratpart(_core(f), _core(g), _core_symbol(x))
    return sympy_expr(A), sympy_expr(B)


def ratint_logpart(f: Union[Poly, Expr], g: Union[Poly, Expr], x: Symbol,
                   t: Optional[Symbol] = None) -> list[tuple[Poly, Poly]]:
    """:func:`sympy_extras.integrals.risch.rationaltools.ratint_logpart` on
    SymPy objects (Lazard–Rioboo–Trager): the pairs as SymPy's ``Poly`` in
    ``x`` and in ``t``.

    >>> from sympy import Poly, symbols
    >>> from sympy_extras.integrals.risch.rationaltools_sympy import ratint_logpart
    >>> x, t = symbols('x t')
    >>> ratint_logpart(Poly(12, x), Poly(x**2 - x - 2, x), x, t)
    [(Poly(x - 3*t/8 - 1/2, x, domain='QQ[t]'), Poly(-t**2 + 16, t, domain='ZZ'))]
    """
    variable: CoreSymbol = CoreDummy('t') if t is None else _core_symbol(t)
    pairs = native.ratint_logpart(_core(f), _core(g), _core_symbol(x), variable)
    t_ = _sympy_symbol(variable)
    return [(Poly(sympy_expr(h), x), Poly(sympy_expr(q), t_, composite=False)) for h, q in pairs]


def log_to_atan(f: Poly, g: Poly) -> Expr:
    """:func:`sympy_extras.integrals.risch.rationaltools.log_to_atan` on
    SymPy's ``Poly`` (in the same generator).

    >>> from sympy import Poly, symbols
    >>> from sympy_extras.integrals.risch.rationaltools_sympy import log_to_atan
    >>> x = symbols('x')
    >>> log_to_atan(Poly(x, x), Poly(1, x))
    2*atan(x)
    """
    gen = f.gen
    return sympy_expr(native.log_to_atan(_core(f), _core(g), _core_symbol(gen)))


def log_to_real(h: Poly, q: Poly, x: Symbol, t: Symbol, complex_only: bool = False) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.risch.rationaltools.log_to_real` on
    SymPy objects; ``None`` when the roots of ``q`` are not all found (or,
    with ``complex_only``, none is complex).

    >>> from sympy import Poly, symbols, S
    >>> from sympy_extras.integrals.risch.rationaltools_sympy import log_to_real
    >>> x, y = symbols('x y')
    >>> log_to_real(Poly(x + 3*y/2 + S(1)/2, x), Poly(3*y**2 + 1, y), x, y)
    2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
    """
    found = native.log_to_real(_core(h), _core(q), _core_symbol(x), _core_symbol(t), complex_only=complex_only)
    return None if found is None else sympy_expr(found)
