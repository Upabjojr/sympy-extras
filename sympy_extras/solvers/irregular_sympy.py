"""The SymPy face of the exponential parts at the irregular finite
singular points (:mod:`sympy_extras.solvers.irregular`).

:class:`Place` takes a SymPy ``Poly`` and writes the root ``alpha`` of a
place of degree more than one as SymPy's ``CRootOf`` (the native module
uses a symbol, the generator of its number field); the functions take the
SymPy :class:`~sympy_extras.solvers.linear_ode_sympy.LinearOperator` and
return SymPy objects, converted by :mod:`sympy_extras.interop`.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
>>> from sympy_extras.solvers.irregular_sympy import finite_exponential_parts, irregular_places
>>> y = Function('y')(x)
>>> L = LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y)
>>> irregular_places(L)
[Poly(x, x, domain='QQ')]
>>> finite_exponential_parts(L)
([0, 1/x, -1/x], True)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf

from sympy_extras._typing import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.interop import from_sympy

from . import irregular as native
from .linear_ode_sympy import LinearOperator, core_upoly, sympy_poly

__all__ = ['Place', 'irregular_places', 'local_exponential_parts', 'finite_exponential_parts', 'conjugate_sum']


class Place:
    """A finite singular point: the irreducible monic ``q`` whose roots are
    the conjugate points, one root ``alpha`` (a rational number, or a
    ``CRootOf``), and the field ``K(alpha)``.

    Examples
    ========

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.irregular_sympy import Place
    >>> Place(Poly(x**2 - 2, x), x).alpha
    CRootOf(x**2 - 2, 0)
    """

    def __init__(self, q: Poly, x: Symbol) -> None:
        self.native = native.Place(core_upoly(q), _core_symbol(x))
        self.q: Poly = q.monic()
        self.x: Symbol = x
        self.degree: int = q.degree()
        if self.degree == 1:
            self.alpha: Expr = sympy_expr(self.native.alpha)
        else:
            self.alpha = CRootOf(self.q.as_expr(), 0)

    def to_core(self, e: Expr) -> CoreExpr:
        """An expression in ``alpha`` as an expression of the native
        place."""
        if self.degree == 1:
            return core_expr(as_expr(e))
        d = Dummy('alpha')
        value = core_expr(as_expr(e.xreplace({self.alpha: d})))
        key = from_sympy(d)
        rule: dict[CoreBasic, CoreBasic] = {key: self.native.alpha}
        replaced = value._xreplace_evaluated(rule)
        if not isinstance(replaced, CoreExpr):
            raise TypeError("%s is not an expression" % (replaced,))
        return replaced

    def of_core(self, e: CoreExpr) -> Expr:
        """An expression of the native place in ``alpha`` as a SymPy
        expression."""
        if self.degree == 1:
            return sympy_expr(e)
        return as_expr(sympy_expr(e).xreplace({sympy_expr(self.native.alpha): self.alpha}))

    def value(self, p: Poly) -> Expr:
        """``p(alpha)`` as an expression in ``alpha`` of degree less than
        that of ``q``."""
        return self.of_core(self.native.value(core_upoly(p)))


def conjugate_sum(place: Place, r: Expr, m: int) -> Expr:
    """``sum_sigma sigma(r)/(x - sigma(alpha))**m`` over the conjugates of
    ``alpha`` (:func:`sympy_extras.solvers.irregular.conjugate_sum`).

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.irregular_sympy import Place, conjugate_sum
    >>> P = Place(Poly(x**2 - 2, x), x)
    >>> conjugate_sum(P, P.alpha, 2)
    8*x/(x**2 - 2)**2
    """
    return sympy_expr(native.conjugate_sum(place.native, place.to_core(r), m))


def local_exponential_parts(L: LinearOperator, place: Place, below: Optional[int] = None, depth: int = 0
                            ) -> tuple[list[Expr], bool]:
    """The candidate integrals ``F`` of the exponential parts at the place
    (:func:`sympy_extras.solvers.irregular.local_exponential_parts`).

    >>> from sympy import Function, Poly
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.irregular_sympy import Place, local_exponential_parts
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**3*y.diff(x) - (2 + x)*y, y)
    >>> local_exponential_parts(L, Place(Poly(x, x), x))
    ([0, -1/x**2, -1/x - 1/x**2], True)
    """
    found, complete = native.local_exponential_parts(L.core(), place.native, below, depth)
    return [sympy_expr(e) for e in found], complete


def irregular_places(L: LinearOperator) -> list[Poly]:
    """The irreducible factors of the leading coefficient at whose roots
    the operator is not regular singular, as monic polynomials
    (:func:`sympy_extras.solvers.irregular.irregular_places`).

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.irregular_sympy import irregular_places
    >>> y = Function('y')(x)
    >>> irregular_places(LinearOperator.from_equation(x**3*(x - 1)*y.diff(x, 2) + y, y))
    [Poly(x, x, domain='QQ')]
    >>> irregular_places(LinearOperator.from_equation(x*y.diff(x, 2) + y, y))
    []
    """
    return [sympy_poly(q).monic() for q in native.irregular_places(L.core())]


def finite_exponential_parts(L: LinearOperator, max_combinations: int = 200) -> tuple[list[Expr], bool]:
    """The candidate integrals ``F`` of the exponential parts at all the
    irregular finite singular points of ``L``
    (:func:`sympy_extras.solvers.irregular.finite_exponential_parts`).

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode_sympy import LinearOperator
    >>> from sympy_extras.solvers.irregular_sympy import finite_exponential_parts
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x) + y, y)
    >>> finite_exponential_parts(L)
    ([0, 1/x], True)
    """
    found, complete = native.finite_exponential_parts(L.core(), max_combinations)
    return [sympy_expr(e) for e in found], complete
