"""The SymPy face of the residue integrator (:mod:`sympy_extras.integrals.residues`),
for the integrators which are still written on SymPy (:mod:`.definite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`; the rational functions with SymPy's ``Poly``,
the poles as SymPy expressions, ``CRootOf`` for the roots of numerical
denominators whose radicals are unwieldy). Two steps of the native
module stay on SymPy and are here: :func:`complex_roots_of_core`, the
``CRootOf`` roots of such a denominator, and
:func:`verify_numerically_of_core`, the quadratures of :mod:`.definite`
for a value placed by sampling which the native quadratures do not
settle.

Examples
========

>>> from sympy import symbols, cos, oo, pi, sqrt
>>> from sympy_extras.integrals.residues_sympy import residue_integral
>>> x = symbols('x')
>>> a, b = symbols('a b', positive=True)
>>> residue_integral(1/(x**4 + 1), x, -oo, oo)
ConditionalValue(sqrt(2)*pi/2)
>>> residue_integral(cos(b*x)/(x**2 + a**2), x, -oo, oo)
ConditionalValue(pi*exp(-a*b)/a)
>>> residue_integral(1/(2 + cos(x)), x, 0, 2*pi)
ConditionalValue(2*sqrt(3)*pi/3)
>>> residue_integral(sqrt(x)/(x**2 + 1), x, 0, oo)
ConditionalValue(sqrt(2)*pi/2)
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions as CoreAssumptions
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.logic import to_boolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import residues as native
from .conditions import ConditionalValue as CoreConditionalValue
from .conditions_sympy import ConditionalValue

__all__ = ['residue_integral', 'rational_function', 'poles', 'real_line_integral',
           'fourier_integral', 'keyhole_integral', 'period_integral', 'complex_roots_of_core',
           'verify_numerically_of_core']


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


def _conditional(v: Optional[CoreConditionalValue]) -> Optional[ConditionalValue]:
    return None if v is None else ConditionalValue.from_core(v)


def _sympy_poly(p: native._Polynomial) -> Poly:
    return Poly(sympy_expr(p.as_expr()), _sympy_symbol(p.x))


def _core_polynomial(q: Poly) -> native._Polynomial:
    if len(q.gens) != 1:
        raise TypeError("%s is not univariate" % (q,))
    return native._Polynomial([core_expr(as_expr(c)) for c in q.all_coeffs()], _core_symbol(q.gens[0]))


class _Fraction:
    """:class:`sympy_extras.integrals.residues._Fraction` on SymPy
    objects: a rational function ``numerator/denominator`` of ``x``, as
    SymPy's polynomials with coefficients free of ``x``."""

    __slots__ = ('numerator', 'denominator', 'x')

    def __init__(self, numerator: Poly, denominator: Poly, x: Symbol) -> None:
        self.numerator: Final[Poly] = numerator
        self.denominator: Final[Poly] = denominator
        self.x: Final[Symbol] = x

    def as_expr(self) -> Expr:
        return as_expr(self.numerator.as_expr() / self.denominator.as_expr())

    @property
    def excess(self) -> int:
        """``deg Q - deg P``."""
        return int(self.denominator.degree()) - int(self.numerator.degree())

    @staticmethod
    def from_core(f: native._Fraction) -> _Fraction:
        """A fraction of the core on SymPy objects."""
        return _Fraction(_sympy_poly(f.numerator), _sympy_poly(f.denominator), _sympy_symbol(f.x))


class _Pole:
    """:class:`sympy_extras.integrals.residues._Pole` on SymPy objects: a
    pole with its multiplicity."""

    __slots__ = ('point', 'multiplicity')

    def __init__(self, point: Expr, multiplicity: int) -> None:
        self.point: Final[Expr] = point
        self.multiplicity: Final[int] = multiplicity

    def __repr__(self) -> str:
        return "_Pole(%s, %s)" % (self.point, self.multiplicity)


def rational_function(f: ExprLike, x: Symbol) -> Optional[_Fraction]:
    """:func:`sympy_extras.integrals.residues.rational_function` on SymPy
    objects: ``f`` as a quotient of polynomials in ``x`` with
    coefficients free of ``x`` and the common factors cancelled, or
    ``None``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.residues_sympy import rational_function
    >>> x = symbols('x')
    >>> fraction = rational_function((x + 1)/((x + 1)*(x**2 + 1)), x)
    >>> fraction.numerator, fraction.denominator, fraction.excess
    (Poly(1, x, domain='ZZ'), Poly(x**2 + 1, x, domain='ZZ'), 2)
    """
    found = native.rational_function(core_expr(as_expr(f)), _core_symbol(x))
    return None if found is None else _Fraction.from_core(found)


def poles(q: Poly) -> Optional[list[_Pole]]:
    """:func:`sympy_extras.integrals.residues.poles` on SymPy objects: the
    roots of ``q`` with multiplicities, or ``None`` when they cannot all
    be found (radicals for parametric coefficients, ``CRootOf`` for
    numerical ones whose radicals are unwieldy).

    >>> from sympy import symbols, Poly
    >>> from sympy_extras.integrals.residues_sympy import poles
    >>> x = symbols('x')
    >>> sorted((str(p.point), p.multiplicity) for p in poles(Poly((x**2 + 1)**2*(x - 3), x)))
    [('-I', 2), ('3', 1), ('I', 2)]
    """
    if not isinstance(q, Poly):
        raise TypeError("%s is not a Poly" % (q,))
    found = native.poles(_core_polynomial(q))
    if found is None:
        return None
    return [_Pole(sympy_expr(p.point), p.multiplicity) for p in found]


def _numerically(q: Boolean) -> Optional[bool]:
    """:func:`sympy_extras.integrals.residues._numerically` on SymPy
    objects: a relation between numbers decided by evaluation, ``None``
    when the difference of the sides is not clearly of one sign."""
    return native._numerically(core_boolean(q))


def real_line_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.residues.real_line_integral` on SymPy
    objects: ``Integral(R(x), (x, -oo, oo))`` for a rational function with
    ``deg Q >= deg P + 2`` and no real poles.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.residues_sympy import real_line_integral
    >>> x = symbols('x')
    >>> real_line_integral(1/(x**2 + 1)**2, x)
    ConditionalValue(pi/2)
    """
    return _conditional(native.real_line_integral(core_expr(as_expr(f)), _core_symbol(x),
                                                  core_assumptions(assumptions)))


def fourier_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                     half: bool = False) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.residues.fourier_integral` on SymPy
    objects: ``Integral(R(x)*exp(I*k*x), (x, -oo, oo))``, and with
    ``cos(k*x)`` or ``sin(k*x)``, by Jordan's lemma; with ``half=True``
    the integral over ``(0, oo)`` of an even integrand.

    >>> from sympy import symbols, cos, sin
    >>> from sympy_extras.integrals.residues_sympy import fourier_integral
    >>> x, k = symbols('x k')
    >>> fourier_integral(cos(2*x)/(x**2 + 1), x)
    ConditionalValue(pi*exp(-2))
    >>> fourier_integral(x*sin(k*x)/(x**2 + 4), x)
    ConditionalValue(pi*exp(-2*k), k > 0)
    """
    return _conditional(native.fourier_integral(core_expr(as_expr(f)), _core_symbol(x),
                                                core_assumptions(assumptions), half))


def keyhole_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.residues.keyhole_integral` on SymPy
    objects: ``Integral(x**alpha * log(x)**n * R(x), (x, 0, oo))`` by the
    keyhole contour.

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.residues_sympy import keyhole_integral
    >>> x, a = symbols('x a')
    >>> keyhole_integral(x**a/(x + 1), x)
    ConditionalValue(-pi/sin(pi*a), (a > -1) & (a < 0))
    >>> keyhole_integral(1/(x**3 + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    >>> keyhole_integral(log(x)/(x**2 + 1), x)
    ConditionalValue(0)
    """
    return _conditional(native.keyhole_integral(core_expr(as_expr(f)), _core_symbol(x),
                                                core_assumptions(assumptions)))


def period_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                    assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.residues.period_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` over ``k`` periods for a rational
    function of ``sin(x)`` and ``cos(x)``.

    >>> from sympy import symbols, cos, pi
    >>> from sympy_extras.integrals.residues_sympy import period_integral
    >>> x = symbols('x')
    >>> a = symbols('a', positive=True)
    >>> period_integral(1/(1 - 2*a*cos(x) + a**2), x, 0, 2*pi, a < 1)
    ConditionalValue(2*pi/(1 - a**2))
    """
    return _conditional(native.period_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                               core_expr(as_expr(b)), core_assumptions(assumptions)))


def residue_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.residues.residue_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` by the residue theorem when ``f``
    is one of the families of the module documentation; ``None``
    otherwise.

    >>> from sympy import symbols, sin, oo, pi, log
    >>> from sympy_extras.integrals.residues_sympy import residue_integral
    >>> x = symbols('x')
    >>> residue_integral(x*sin(x)/(x**2 + 1), x, -oo, oo)
    ConditionalValue(pi*exp(-1))
    >>> residue_integral(log(x)**2/(x**2 + 1), x, 0, oo)
    ConditionalValue(pi**3/8)
    >>> residue_integral(1/(5 + 3*sin(x)), x, 0, 2*pi)
    ConditionalValue(pi/2)
    >>> residue_integral(1/(x**2 + 1), x, 0, 1) is None
    True
    """
    return _conditional(native.residue_integral(core_expr(as_expr(f)), _core_symbol(x), core_expr(as_expr(a)),
                                                core_expr(as_expr(b)), core_assumptions(assumptions)))


# ---------------------------------------------------------------------------
# The steps of the native module which stay on SymPy

def complex_roots_of_core(g: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[list[CoreExpr]]:
    """The roots of the irreducible polynomial ``g`` in ``x`` with rational
    coefficients as SymPy's ``CRootOf`` (``all_roots``), on core
    expressions (the step of :mod:`sympy_extras.integrals.residues` which
    stays on SymPy: the core has no node for a complex algebraic number);
    ``None`` when they are not found within ``seconds``."""
    variable = _sympy_symbol(x)
    try:
        poly = Poly(sympy_expr(g), variable)
    except PolynomialError:
        return None
    found = attempt(lambda: [as_expr(r) for r in poly.all_roots()], seconds)
    if found is None:
        return None
    return [core_expr(r) for r in found]


def _sympy_assumptions(assumptions: CoreAssumptions) -> Assumptions:
    if assumptions is None:
        return None
    if isinstance(assumptions, (CoreBoolean, bool)):
        return as_boolean(to_sympy(to_boolean(assumptions)))
    return [as_boolean(to_sympy(to_boolean(a))) for a in assumptions]


def verify_numerically_of_core(value: CoreExpr, f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                               assumptions: CoreAssumptions) -> Optional[bool]:
    """:func:`sympy_extras.integrals.definite.verify_numerically` of a value
    of the native module (the step of
    :mod:`sympy_extras.integrals.residues` which stays on SymPy, taken
    when the native quadratures settle nothing): ``False`` when the value
    disagrees with the quadratures at samples of the parameters, ``True``
    when it agrees, ``None`` when nothing could be checked."""
    from .definite import verify_numerically
    return verify_numerically(sympy_expr(value), sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b),
                              _sympy_assumptions(assumptions))
