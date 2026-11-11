"""The SymPy face of the Darboux polynomials and the Prelle–Singer
procedure (:mod:`sympy_extras.solvers.darboux`).

:func:`darboux_polynomials`, :func:`exponential_factors`,
:func:`darboux_integrating_factor`, :func:`darboux_first_integral`,
:func:`prelle_singer` and :func:`vector_field` take and return SymPy
objects, converted by :mod:`sympy_extras.interop` around the native
functions; :class:`DarbouxPolynomial`, :class:`ExponentialFactor` and
:class:`DarbouxIntegral` carry SymPy expressions. The modules still
written on SymPy (:mod:`~sympy_extras.solvers.first_order`) use them.

The quadratures of the integrating factor go through
:func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x, y
>>> from sympy_extras.solvers.darboux_sympy import darboux_polynomials, prelle_singer
>>> darboux_polynomials(y*(x - 1), x*(1 - y), x, y)
[DarbouxPolynomial(y, x - 1), DarbouxPolynomial(x, 1 - y)]
>>> f = Function('y')(x)
>>> prelle_singer(f.diff(x) - (x + f)/(x - f), f)
Eq(-log(x**2 + y(x)**2)/2 - atan(x/y(x)), C1)
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp

from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.interop import from_sympy, to_sympy

from . import darboux as native

__all__ = ['DarbouxPolynomial', 'ExponentialFactor', 'DarbouxIntegral', 'vector_field',
           'darboux_polynomials', 'exponential_factors', 'darboux_integrating_factor',
           'darboux_first_integral', 'prelle_singer']


# ---------------------------------------------------------------------------
# conversions

def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _core(e: ExprLike) -> CoreExpr:
    return core_expr(as_expr(e))


def _optional(e: Optional[CoreExpr]) -> Optional[Expr]:
    return None if e is None else sympy_expr(e)


# ---------------------------------------------------------------------------
# the results

class DarbouxPolynomial:
    """:class:`sympy_extras.solvers.darboux.DarbouxPolynomial` with SymPy
    expressions: a Darboux polynomial ``f`` of the field ``D = Q d/dx + P
    d/dy``, irreducible over the coefficient field, and its cofactor
    ``g``, ``D f = g f``.

    Attributes
    ==========

    polynomial : Expr
        ``f``, monic in the lex order of ``x, y``.
    cofactor : Expr
        ``g``, of degree at most ``d - 1``.
    """

    def __init__(self, polynomial: Expr, cofactor: Expr) -> None:
        self.polynomial = polynomial
        self.cofactor = cofactor

    @staticmethod
    def of_core(found: native.DarbouxPolynomial) -> DarbouxPolynomial:
        return DarbouxPolynomial(sympy_expr(found.polynomial), sympy_expr(found.cofactor))

    def __repr__(self) -> str:
        return "DarbouxPolynomial(%s, %s)" % (self.polynomial, self.cofactor)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DarbouxPolynomial):
            return NotImplemented
        return bool(self.polynomial == other.polynomial and self.cofactor == other.cofactor)

    def __hash__(self) -> int:
        return hash((self.polynomial, self.cofactor))


class ExponentialFactor:
    """:class:`sympy_extras.solvers.darboux.ExponentialFactor` with SymPy
    expressions: an exponential factor ``exp(A/B)`` of the field, ``D(A/B)
    = L`` a polynomial of degree at most ``d - 1``, its cofactor.

    Attributes
    ==========

    numerator, denominator : Expr
        ``A`` and ``B``; ``B`` is ``1`` or a product of Darboux
        polynomials.
    cofactor : Expr
        ``L``.
    """

    def __init__(self, numerator: Expr, denominator: Expr, cofactor: Expr) -> None:
        self.numerator = numerator
        self.denominator = denominator
        self.cofactor = cofactor

    @staticmethod
    def of_core(found: native.ExponentialFactor) -> ExponentialFactor:
        return ExponentialFactor(sympy_expr(found.numerator), sympy_expr(found.denominator),
                                 sympy_expr(found.cofactor))

    @property
    def argument(self) -> Expr:
        """``A/B``."""
        return as_expr(self.numerator/self.denominator)

    @property
    def factor(self) -> Expr:
        """``exp(A/B)``."""
        return as_expr(exp(self.argument))

    def __repr__(self) -> str:
        return "ExponentialFactor(%s, %s, %s)" % (self.numerator, self.denominator, self.cofactor)


class DarbouxIntegral:
    """:class:`sympy_extras.solvers.darboux.DarbouxIntegral` with SymPy
    expressions: the Darboux integrating factor and the first integral of
    a field, when found.

    Attributes
    ==========

    polynomials : list[DarbouxPolynomial]
        The Darboux polynomials found within the degree bound.
    exponential_factors : list[ExponentialFactor]
        The exponential factors found.
    integrating_factor : Expr or None
        ``R`` with ``R (Q dy - P dx)`` exact.
    first_integral : Expr or None
        ``I(x, y)`` with ``D I = 0``, not constant.
    """

    def __init__(self, polynomials: list[DarbouxPolynomial], factors: list[ExponentialFactor],
                 integrating_factor: Optional[Expr], first_integral: Optional[Expr]) -> None:
        self.polynomials = polynomials
        self.exponential_factors = factors
        self.integrating_factor = integrating_factor
        self.first_integral = first_integral

    @staticmethod
    def of_core(found: native.DarbouxIntegral) -> DarbouxIntegral:
        return DarbouxIntegral([DarbouxPolynomial.of_core(d) for d in found.polynomials],
                               [ExponentialFactor.of_core(e) for e in found.exponential_factors],
                               _optional(found.integrating_factor), _optional(found.first_integral))

    def __repr__(self) -> str:
        return "DarbouxIntegral(integrating_factor=%s, first_integral=%s)" % (
            self.integrating_factor, self.first_integral)


# ---------------------------------------------------------------------------
# the face

def _field_of(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, extension: Optional[Expr] = None) -> native._Context:
    """The native field ``Q d/dx + P d/dy`` of SymPy polynomials
    (:func:`sympy_extras.solvers.darboux._field_of`)."""
    return native._field_of(_core(P), _core(Q), _core_symbol(x), _core_symbol(y),
                            None if extension is None else _core(extension))


def _first_integral_over(field: native._Context, degree: int) -> Optional[Expr]:
    """The first integral over the coefficient field of ``field`` alone,
    as a SymPy expression."""
    return _optional(field.first_integral_over(degree))


def _quadratic_extension(field: native._Context) -> Optional[Expr]:
    """The square root of the discriminant of the first irreducible
    quadratic factor of ``x P_d - y Q_d``, as a SymPy expression."""
    return _optional(field.quadratic_extension())


def vector_field(equation: Basic, f: AppliedUndef) -> Optional[tuple[Expr, Expr, Symbol, Symbol]]:
    """``(P, Q, x, y)`` of the first order equation ``y' = P(x, y)/Q(x, y)``
    with ``P`` and ``Q`` coprime polynomials in ``x`` and ``y`` (``y`` a
    fresh symbol for ``f``), or ``None`` when the equation is not of this
    form (:func:`sympy_extras.solvers.darboux.vector_field` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.darboux_sympy import vector_field
    >>> y = Function('y')(x)
    >>> vector_field((x - y)*y.diff(x) - x - y, y)
    (x + y, x - y, x, y)
    """
    found = native.vector_field(_core_basic(equation), _core_function(f))
    if found is None:
        return None
    P, Q, x, y = found
    return sympy_expr(P), sympy_expr(Q), _sympy_symbol(x), _sympy_symbol(y)


def darboux_polynomials(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[DarbouxPolynomial]:
    """The irreducible Darboux polynomials of total degree at most
    ``degree`` of the field ``Q d/dx + P d/dy`` with their cofactors
    (:func:`sympy_extras.solvers.darboux.darboux_polynomials` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import I
    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux_sympy import darboux_polynomials
    >>> for f in darboux_polynomials(y*(x - 1), x*(1 - y), x, y):      # Lotka–Volterra
    ...     print(f)
    DarbouxPolynomial(y, x - 1)
    DarbouxPolynomial(x, 1 - y)
    >>> darboux_polynomials(x + y, x - y, x, y, degree=2)
    [DarbouxPolynomial(x**2 + y**2, 2)]
    >>> darboux_polynomials(x + y, x - y, x, y, degree=1, extension=I)
    [DarbouxPolynomial(x - I*y, 1 - I), DarbouxPolynomial(x + I*y, 1 + I)]
    """
    return [DarbouxPolynomial.of_core(d) for d in _field_of(P, Q, x, y, extension).polynomials(degree)]


def exponential_factors(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                        extension: Optional[Expr] = None) -> list[ExponentialFactor]:
    """The exponential factors ``exp(A/B)`` of the field ``Q d/dx + P d/dy``
    (:func:`sympy_extras.solvers.darboux.exponential_factors` on SymPy
    objects).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux_sympy import exponential_factors
    >>> exponential_factors(x*y + 1, 1, x, y, degree=1)         # y' = x y + 1
    [ExponentialFactor(x, 1, 1), ExponentialFactor(x**2/2, 1, x)]
    """
    return [ExponentialFactor.of_core(e) for e in _field_of(P, Q, x, y, extension).exponential_factors(degree)]


def darboux_integrating_factor(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                               extension: Optional[Expr] = None) -> Optional[Expr]:
    """A Darboux integrating factor ``R`` of ``Q dy - P dx``
    (:func:`sympy_extras.solvers.darboux.darboux_integrating_factor` on
    SymPy objects).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux_sympy import darboux_integrating_factor
    >>> darboux_integrating_factor(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    1/(x*y)
    >>> darboux_integrating_factor(x*y + 1, 1, x, y, degree=1)      # y' = x y + 1
    exp(-x**2/2)
    """
    return _optional(_field_of(P, Q, x, y, extension).integrating_factor(degree))


def darboux_first_integral(P: ExprLike, Q: ExprLike, x: Symbol, y: Symbol, degree: int = 4,
                           extension: Optional[Expr] = None) -> Optional[Expr]:
    """A first integral ``I(x, y)`` of ``y' = P/Q`` by the Prelle–Singer
    procedure (:func:`sympy_extras.solvers.darboux.darboux_first_integral`
    on SymPy objects).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.darboux_sympy import darboux_first_integral
    >>> darboux_first_integral(y*(x - 1), x*(1 - y), x, y)      # Lotka–Volterra
    -x - y + log(x) + log(y)
    >>> darboux_first_integral(x + y, x - y, x, y)
    -log(x**2 + y**2)/2 - atan(x/y)
    >>> darboux_first_integral(x*y**2 - y, x**2*y - x, x, y, degree=2)
    x/y
    """
    return _optional(_field_of(P, Q, x, y, extension).first_integral(degree))


def prelle_singer(equation: Basic, f: AppliedUndef, degree: int = 4, extension: Optional[Expr] = None) -> Optional[Eq]:
    """The implicit solution ``Eq(I(x, y), C1)`` of a first order equation
    ``y' = P(x, y)/Q(x, y)`` by the Prelle–Singer procedure, or ``None``
    (:func:`sympy_extras.solvers.darboux.prelle_singer` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.darboux_sympy import prelle_singer
    >>> y = Function('y')(x)
    >>> prelle_singer(y.diff(x) - (x + y)/(x - y), y)
    Eq(-log(x**2 + y(x)**2)/2 - atan(x/y(x)), C1)
    >>> prelle_singer(x*(1 - y)*y.diff(x) - y*(x - 1), y)
    Eq(-x - y(x) + log(x) + log(y(x)), C1)
    >>> prelle_singer(x**2*y.diff(x) - x*y + 1, y)
    Eq(y(x), C1*x + 1/(2*x))
    """
    found = native.prelle_singer(_core_basic(equation), _core_function(f), degree,
                                 None if extension is None else _core(extension))
    if found is None:
        return None
    converted = to_sympy(found)
    if not isinstance(converted, Eq):
        raise TypeError("%s is not an equation" % (converted,))
    return converted
