"""The SymPy face of Trager's algorithm on a general curve
(:mod:`sympy_extras.integrals.curves`), for the code which is still
written on SymPy (:mod:`sympy_extras.integrals.trager_sympy`, which takes
from it the residues of the fields without radicals as ``CRootOf``).

The functions and classes take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native ones; nothing of the
algorithm stays on SymPy. What the face adds: the curve collected in the
primitive element as SymPy's ``minimal_polynomial`` wrote it, the
function field of :mod:`sympy_extras.polys.functionfields_sympy` for
:class:`AlgebraicIntegrand`, and the number field of the residues as
SymPy's (:func:`sympy_extras.polys.puiseux_sympy.sympy_field`): a field
whose generator is neither real nor in radicals is written with SymPy's
``CRootOf``, which the core does not port.

Examples
========

>>> from sympy import symbols, sqrt
>>> from sympy_extras.integrals.curves_sympy import curve_antiderivative, is_nonelementary_on_curve
>>> x = symbols('x')
>>> curve_antiderivative(1/(sqrt(x + 1) + sqrt(x - 1)), x)
-x*sqrt(x - 1)/3 + x*sqrt(x + 1)/3 + sqrt(x - 1)/3 + sqrt(x + 1)/3
>>> is_nonelementary_on_curve(x/sqrt(1 - x**3), x)
True
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.simplify.radsimp import collect

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Dummy as CoreDummy, Symbol as CoreSymbol
from sympy_extras.interop import from_sympy
from sympy_extras.polys.functionfields import Element as NativeElement
from sympy_extras.polys.functionfields_sympy import Element, FunctionField
from sympy_extras.polys.puiseux import Number
from sympy_extras.polys.puiseux_sympy import sympy_field

from . import curves as native

__all__ = ['curve_antiderivative', 'curve_reduce', 'is_nonelementary_on_curve', 'integrate_on_curve',
           'AlgebraicIntegrand', 'CurveIntegral']


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


class Generator:
    """A radical `g = \\text{base}^{1/q}` of the integrand on SymPy
    objects (the native :class:`sympy_extras.integrals.curves._Generator`)."""

    def __init__(self, symbol: Symbol, base: Expr, q: int, original: Expr) -> None:
        self.symbol = symbol
        self.base = base
        self.q = q
        self.original = original


class AlgebraicIntegrand:
    """:class:`sympy_extras.integrals.curves.AlgebraicIntegrand` on SymPy
    objects: the integrand as an element of the function field
    (:mod:`sympy_extras.polys.functionfields_sympy`) of a primitive
    element of its radicals.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves_sympy import AlgebraicIntegrand
    >>> x = symbols('x')
    >>> A = AlgebraicIntegrand(1/(sqrt(x + 1) + sqrt(x - 1)), x)
    >>> y = symbols('y')
    >>> A.curve.subs(A.y, y)
    -4*x*y**2 + y**4 + 4
    >>> A.element.as_expr().subs(A.y, y)
    x*y - y**3/4
    """

    def __init__(self, f: ExprLike, x: Symbol, integrand: Optional[native.AlgebraicIntegrand] = None) -> None:
        self.native = native.AlgebraicIntegrand(core_expr(f), _core_symbol(x)) if integrand is None else integrand
        self.x = x
        self.original = as_expr(f)
        self.y = as_symbol(sympy_expr(self.native.y))
        self.generators = [Generator(as_symbol(sympy_expr(g.symbol)), sympy_expr(g.base), g.q, sympy_expr(g.original))
                           for g in self.native.generators]
        #: the algebra of the radicals (native: ``dimension``, ``monomials``)
        self.algebra = self.native.algebra
        self.weights = list(self.native.weights)
        self.theta = sympy_expr(self.native.theta)
        self.curve = as_expr(collect(sympy_expr(self.native.curve), self.y))
        self.field = FunctionField(self.curve, x, self.y)
        self.element: Element = self.field.element(sympy_expr(self.native.element_expr()))

    def to_field(self, expr: ExprLike) -> Element:
        """A rational function of `x` and the generator symbols as an
        element of the field of the primitive element."""
        e = self.native.to_field(core_expr(expr))
        return self.field.element(sympy_expr(native._element_expr(e, self.native.x, self.native.y)))

    def polynomial_to_expr(self, p: ExprLike) -> Expr:
        """A rational expression in `x` and the primitive element `y`
        written in the original radicals through the monomial basis."""
        return sympy_expr(self.native.polynomial_to_expr(core_expr(p)))


class CurveIntegral:
    """:class:`sympy_extras.integrals.curves.CurveIntegral` on SymPy
    objects: the verdict ``nonelementary``, the elementary part and the
    remaining integrand as SymPy expressions, the field of the residues
    as SymPy's number field."""

    def __init__(self, result: native.CurveIntegral, x: Symbol) -> None:
        self.native = result
        self.x = x
        #: ``True`` when the integral is proved not elementary, ``False``
        #: when found, ``None`` when undecided
        self.nonelementary: Optional[bool] = result.nonelementary
        self._integrand: Optional[AlgebraicIntegrand] = None
        self._placeholder = CoreDummy('theta')

    @property
    def integrand(self) -> AlgebraicIntegrand:
        if self._integrand is None:
            self._integrand = AlgebraicIntegrand(sympy_expr(self.native.integrand.original), self.x,
                                                 self.native.integrand)
        return self._integrand

    @property
    def residue_field(self) -> Domain:
        """The number field of the residues (SymPy's ``QQ`` or an
        ``AlgebraicField``)."""
        return sympy_field(self.native.residue_field)

    @property
    def remainder(self) -> Optional[NativeElement[Number]]:
        """The integrand left in `z` (the native element), ``None`` when
        nothing is left."""
        return self.native.remainder

    def _theta(self) -> Optional[CoreExpr]:
        K = self.native.residue_field
        if K.degree > 1 and isinstance(K.alpha_expr, CoreSymbol):
            return self._placeholder
        return None

    def _sympy(self, e: CoreExpr, theta: Optional[CoreExpr]) -> Expr:
        converted = sympy_expr(e)
        K = self.residue_field
        if theta is None or not isinstance(K, AlgebraicField):
            return converted
        ext = as_expr(K.ext.as_expr())
        return as_expr(converted.xreplace({sympy_expr(theta): ext}))

    def elementary_part(self) -> Expr:
        """The elementary part found, in `x` and the original radicals."""
        theta = self._theta()
        return self._sympy(self.native.elementary_part(theta), theta)

    def remainder_expr(self) -> Expr:
        """The remaining integrand in `x`."""
        theta = self._theta()
        return self._sympy(self.native.remainder_expr(theta), theta)


def integrate_on_curve(f: ExprLike, x: Symbol, logarithms: bool = True) -> CurveIntegral:
    """:func:`sympy_extras.integrals.curves.integrate_on_curve` on SymPy
    objects; :class:`ValueError` when ``f`` is not an algebraic function
    of `x` of the kind the parser understands.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves_sympy import integrate_on_curve
    >>> x = symbols('x')
    >>> result = integrate_on_curve(x/sqrt(1 - x**3), x)
    >>> result.nonelementary, result.elementary_part()
    (True, -2*sqrt(1 - x**3)/x)
    """
    return CurveIntegral(native.integrate_on_curve(core_expr(f), _core_symbol(x), logarithms), x)


def curve_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.curves.curve_reduce` on SymPy
    objects: ``(F, h)`` with ``f == F' + h``, or ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves_sympy import curve_reduce
    >>> x = symbols('x')
    >>> curve_reduce(1/(x*sqrt(1 - x**3)), x)
    (-log((x**3 - 2*sqrt(1 - x**3) - 2)/(2*x**3))/3, 0)
    >>> curve_reduce(x/sqrt(1 - x**3), x)
    (-2*sqrt(1 - x**3)/x, 2*sqrt(1 - x**3)/(x**5 - x**2))
    """
    try:
        result = integrate_on_curve(f, x)
        return result.elementary_part(), result.remainder_expr()
    except ValueError:
        return None


def curve_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.curves.curve_antiderivative` on SymPy
    objects: an antiderivative of the algebraic function ``f``, or
    ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves_sympy import curve_antiderivative
    >>> x = symbols('x')
    >>> curve_antiderivative(1/(sqrt(x + 1) + sqrt(x - 1)), x)
    -x*sqrt(x - 1)/3 + x*sqrt(x + 1)/3 + sqrt(x - 1)/3 + sqrt(x + 1)/3
    """
    try:
        result = integrate_on_curve(f, x)
        if result.nonelementary is not False or result.remainder is not None:
            return None
        return result.elementary_part()
    except ValueError:
        return None


def is_nonelementary_on_curve(f: ExprLike, x: Symbol) -> Optional[bool]:
    """:func:`sympy_extras.integrals.curves.is_nonelementary_on_curve` on
    SymPy objects.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.curves_sympy import is_nonelementary_on_curve
    >>> x = symbols('x')
    >>> is_nonelementary_on_curve(x/sqrt(1 - x**3), x)
    True
    """
    try:
        result = integrate_on_curve(f, x)
    except ValueError:
        return None
    return result.nonelementary
