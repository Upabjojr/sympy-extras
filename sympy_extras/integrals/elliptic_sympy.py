"""The SymPy face of the reduction to Legendre's elliptic integrals
(:mod:`sympy_extras.integrals.elliptic`), for the integrators which are
still written on SymPy (:mod:`sympy_extras.integrals.definite`).

The functions take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native ones: the elliptic
functions become SymPy's ``elliptic_k``, ``elliptic_e``, ``elliptic_f``
and ``elliptic_pi``, the indexed roots of a numeric radicand SymPy's
``CRootOf`` (a value in them is converted without evaluation, as the old
module built it).

Examples
========

>>> from sympy import symbols, sqrt, oo, sin, pi
>>> from sympy_extras.integrals.elliptic_sympy import elliptic_integral
>>> x, theta = symbols('x theta')
>>> k = symbols('k', positive=True)
>>> elliptic_integral(1/sqrt(x*(1 - x)*(1 - k**2*x)), x, 0, 1, k < 1)
ConditionalValue(2*elliptic_k(k**2))
>>> elliptic_integral(1/sqrt(x*(x - 1)*(2 - x)*(3 - x)), x, 1, 2)
ConditionalValue(elliptic_k(3/4))
>>> elliptic_integral(sqrt(1 - k**2*sin(theta)**2), theta, 0, pi/2, k < 1)
ConditionalValue(elliptic_e(k**2))
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, core_values, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.polys.cad.indexedroot import IndexedRoot

from . import conditions as native_conditions
from . import elliptic as native
from .conditions_sympy import ConditionalValue

__all__ = ['elliptic_integral', 'radicand', 'real_roots', 'legendre_reduction', 'Reduction']


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


def _sympy_value(e: CoreExpr) -> Expr:
    """A value of the native module on SymPy objects: a value in indexed
    roots (``CRootOf``) without evaluation, as the old module built it."""
    if e.has(IndexedRoot):
        return as_expr(to_sympy(e, evaluate=False))
    return sympy_expr(e)


def _sympy_conditional(v: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    if v is None:
        return None
    return ConditionalValue(_sympy_value(v.value), sympy_boolean(v.condition))


class Reduction:
    """The Legendre form of an elliptic integral on SymPy objects,
    ``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))``
    (:class:`sympy_extras.integrals.elliptic.Reduction` on the core).

    Attributes
    ==========

    Q : Expr
        A rational function of ``s``.
    m : Expr
        The parameter (``k**2``).
    s : Symbol
    lower, upper : Expr
        The range of ``s`` inside ``[0, 1]``.
    """

    __slots__ = ('Q', 'm', 's', 'lower', 'upper')

    def __init__(self, Q: ExprLike, m: ExprLike, s: Symbol, lower: ExprLike, upper: ExprLike) -> None:
        self.Q: Final[Expr] = as_expr(Q)
        self.m: Final[Expr] = as_expr(m)
        self.s: Final[Symbol] = s
        self.lower: Final[Expr] = as_expr(lower)
        self.upper: Final[Expr] = as_expr(upper)

    def __repr__(self) -> str:
        return "Reduction(%s, m=%s, (%s, %s, %s))" % (self.Q, self.m, self.s, self.lower, self.upper)

    def to_core(self) -> native.Reduction:
        """The same reduction on the objects of the core."""
        return native.Reduction(core_expr(self.Q), core_expr(self.m), _core_symbol(self.s),
                                core_expr(self.lower), core_expr(self.upper))

    @staticmethod
    def from_core(r: native.Reduction) -> Reduction:
        """A reduction of the core on SymPy objects."""
        return Reduction(sympy_expr(r.Q), sympy_expr(r.m), _sympy_symbol(r.s), sympy_expr(r.lower),
                         sympy_expr(r.upper))


def radicand(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """:func:`sympy_extras.integrals.elliptic.radicand` on SymPy objects:
    ``(R, P, e)`` with ``f == R * P**e``, ``e`` equal to ``+-1/2`` and
    ``P`` a cubic or a quartic in ``x``, or ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.elliptic_sympy import radicand
    >>> x = symbols('x')
    >>> radicand(x/sqrt((1 - x**2)*(4 - x**2)), x)
    (x, (1 - x**2)*(4 - x**2), -1/2)
    """
    found = native.radicand(core_expr(as_expr(f)), _core_symbol(x))
    if found is None:
        return None
    R, P, e = found
    return (sympy_expr(R), sympy_expr(P), sympy_expr(e))


def real_roots(P: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """:func:`sympy_extras.integrals.elliptic.real_roots` on SymPy
    objects: the roots of ``P`` sorted increasingly when they are all real
    and distinct (numeric ones as ``CRootOf`` or radicals), ``None``
    otherwise.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.elliptic_sympy import real_roots
    >>> x = symbols('x')
    >>> k = symbols('k', positive=True)
    >>> real_roots((1 - x**2)*(1 - k**2*x**2), x, k < 1)
    [-1/k, -1, 1, 1/k]
    """
    found = native.real_roots(core_expr(as_expr(P)), _core_symbol(x), core_assumptions(assumptions))
    return None if found is None else [sympy_expr(r) for r in found]


def legendre_reduction(reduction: Reduction, assumptions: Assumptions = None) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.elliptic.legendre_reduction` on SymPy
    objects: ``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))``
    in Legendre's integrals, ``None`` for a divergent integral.

    >>> from sympy import symbols, S
    >>> from sympy_extras.integrals.elliptic_sympy import Reduction, legendre_reduction
    >>> s = symbols('s', positive=True)
    >>> m = symbols('m', positive=True)
    >>> legendre_reduction(Reduction(S.One, m, s, S.Zero, S.One))
    2*elliptic_k(m)
    >>> legendre_reduction(Reduction(s, m, s, S.Zero, S.One))
    2*(-elliptic_e(m) + elliptic_k(m))/m
    """
    if not isinstance(reduction, Reduction):
        raise TypeError("%s is not a Reduction" % (reduction,))
    found = native.legendre_reduction(reduction.to_core(), core_assumptions(assumptions))
    return None if found is None else _sympy_value(found)


def elliptic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.elliptic.elliptic_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` for the square root of a cubic or
    a quartic times a rational function, in Legendre's elliptic
    integrals; ``None`` when the integral is not of this kind.

    >>> from sympy import symbols, sqrt, oo, S
    >>> from sympy_extras.integrals.elliptic_sympy import elliptic_integral
    >>> x = symbols('x')
    >>> elliptic_integral(1/sqrt(x*(1 - x)*(4 - x)), x, 0, 1)
    ConditionalValue(elliptic_k(1/4))
    >>> elliptic_integral(x/sqrt(x*(1 - x)*(4 - x)), x, 0, S.Half)
    ConditionalValue(-4*elliptic_e(pi/4, 1/4) + 4*elliptic_f(pi/4, 1/4))
    >>> elliptic_integral(1/sqrt((x**2 + 1)*(x + 2)*(3 - x)), x, -2, 3)
    ConditionalValue(2**(3/4)*sqrt(5)*elliptic_k(sqrt(2)/4 + 1/2)/5)
    >>> elliptic_integral(1/sqrt((x**2 + 1)*(x**2 + 4)), x, -oo, oo)
    ConditionalValue(elliptic_k(3/4))
    """
    found = native.elliptic_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                     core_expr(as_expr(b)), core_assumptions(assumptions))
    return _sympy_conditional(found)


def _decide_numerically(query: Boolean, values: dict[Symbol, Expr]) -> Optional[bool]:
    """:func:`sympy_extras.integrals.elliptic._decide_numerically` on SymPy
    objects: a relation on dummies standing for algebraic numbers decided
    numerically at sixty digits (``None`` within the margin)."""
    return native._decide_numerically(core_boolean(query), core_values(values))


def _restore(value: Expr, radicals: dict[Symbol, Expr], facts: Sequence[Boolean]) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.elliptic._restore` on SymPy objects:
    the value with the radicals and the dummies of algebraic numbers
    replaced by their expressions."""
    restored = native._restore(core_expr(value), core_values(radicals), [core_boolean(f) for f in facts])
    return None if restored is None else _sympy_value(restored)
