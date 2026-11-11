r"""The SymPy face of :mod:`sympy_extras.polys.rur`: the rational
univariate representation of a zero-dimensional ideal (Rouillier) on SymPy
objects, and the step which stays on SymPy.

:func:`rur` takes an ideal of :mod:`sympy_extras.polys.ideals_sympy` (over
``QQ``) or a list of SymPy expressions or ``Poly`` objects, converts it to
the native ideal (:func:`~sympy_extras.polys.ideals_sympy.native_ideal`),
computes the representation natively and returns a
:class:`RationalUnivariateRepresentation` of SymPy objects, as before the
native port: the form a SymPy expression, the polynomials ``Poly`` objects
in ``t`` over ``QQ``, the solutions ``CRootOf`` objects or radicals.

What stays here, on SymPy: the exact solutions, whose coordinates are
SymPy's root objects ``CRootOf`` (not ported); the native
:meth:`RationalUnivariateRepresentation.solutions
<sympy_extras.polys.rur.RationalUnivariateRepresentation.solutions>` calls
:func:`solutions_of_core` at run time.

Examples
========

>>> from sympy.abc import x, y, t
>>> from sympy_extras.polys.rur_sympy import rur
>>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
>>> R.polynomial
Poly(t**2 - 1/2, t, domain='QQ')
>>> R.polynomials()
[Poly(t, t, domain='QQ'), Poly(t, t, domain='QQ')]
>>> R.solutions()
[{x: -sqrt(2)/2, y: -sqrt(2)/2}, {x: sqrt(2)/2, y: sqrt(2)/2}]
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence, Union

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly, invert

from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import rur as native
from .domains.rationalfield import QQ as CoreQQ
from .ideals_sympy import Ideal, native_ideal
from .polyclasses import DensePoly

__all__ = ['rur', 'RationalUnivariateRepresentation', 'System', 'solutions_of_core']

#: a system: an ideal, or the polynomials which generate it
System = Union[Ideal, Sequence[Union[Expr, Poly]]]


class RationalUnivariateRepresentation:
    """The rational univariate representation of a zero-dimensional ideal:
    the separating form ``form`` in the variables ``symbols``, the
    squarefree polynomial ``polynomial`` in ``t`` whose roots are the
    values of the form at the zeros, the characteristic polynomial
    ``characteristic`` of the form (its roots with the multiplicities of
    the zeros), and the coordinates ``numerators[i]/denominator`` of the
    zero at a root.

    Examples
    ========

    >>> from sympy.abc import x, y, t
    >>> from sympy_extras.polys.rur_sympy import rur
    >>> R = rur([x**2 - 2, y**2 - 3], x, y, t=t)
    >>> R.form, R.polynomial
    (x + y, Poly(t**4 - 10*t**2 + 1, t, domain='QQ'))
    >>> R.denominator, R.numerators
    (Poly(4*t**3 - 20*t, t, domain='QQ'), [Poly(8*t**2 + 8, t, domain='QQ'), Poly(12*t**2 - 12, t, domain='QQ')])
    >>> R.coordinates(2)
    [-5, -9/2]
    """

    def __init__(self, symbols: Sequence[Symbol], t: Symbol, form: Expr, polynomial: Poly, characteristic: Poly,
                 denominator: Poly, numerators: Sequence[Poly]) -> None:
        self.symbols: list[Symbol] = list(symbols)
        self.t: Symbol = t
        self.form: Expr = form
        self.polynomial: Poly = polynomial
        self.characteristic: Poly = characteristic
        self.denominator: Poly = denominator
        self.numerators: list[Poly] = list(numerators)

    def __repr__(self) -> str:
        return 'RationalUnivariateRepresentation(%s, %s, %s)' % (self.form, self.polynomial.as_expr(),
                                                                 [g.as_expr() for g in self.numerators])

    def coordinates(self, root: Union[Expr, int]) -> list[Expr]:
        """The coordinates of the zero at a root of ``polynomial`` (the
        rational functions evaluated at the root).

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur_sympy import rur
        >>> rur([x**2 - 2, y - x - 1], x, y, t=t).coordinates(sqrt(2))
        [sqrt(2), 1 + sqrt(2)]
        """
        value = as_expr(root)
        d = as_expr(self.denominator.as_expr().subs(self.t, value))
        return [as_expr(as_expr((g.as_expr().subs(self.t, value)/d).radsimp()).expand()) for g in self.numerators]

    def polynomials(self) -> list[Poly]:
        """The coordinates as polynomials in ``t`` of degree less than that
        of ``polynomial``: ``numerators[i]`` times the inverse of
        ``denominator`` modulo ``polynomial`` (the form of a lexicographic
        Gröbner basis in shape position).

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur_sympy import rur
        >>> rur([x**2 - 2, y**2 - 3], x, y, t=t).polynomials()
        [Poly(1/2*t**3 - 9/2*t, t, domain='QQ'), Poly(-1/2*t**3 + 11/2*t, t, domain='QQ')]
        """
        if self.polynomial.degree() <= 0:
            return [Poly(0, self.t, domain=QQ) for _ in self.numerators]
        inverse = invert(self.denominator, self.polynomial)
        return [(g*inverse).rem(self.polynomial) for g in self.numerators]

    def solutions(self, real: bool = False) -> list[dict[Symbol, Expr]]:
        """The zeros of the ideal, each as a dictionary of the variables:
        the roots of ``polynomial`` (:class:`~sympy.CRootOf` objects, or
        radicals when SymPy finds them) substituted in :meth:`polynomials`;
        with ``real=True`` the real zeros only.

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur_sympy import rur
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).solutions()
        [{x: -sqrt(2), y: -2*sqrt(2)}, {x: sqrt(2), y: 2*sqrt(2)}]
        >>> rur([x**2 + 1, y - x], x, y, t=t).solutions(real=True)
        []
        """
        if self.polynomial.degree() <= 0:
            return []
        polys = self.polynomials()
        roots = self.polynomial.real_roots() if real else self.polynomial.all_roots()
        result = []
        for r in roots:
            root = as_expr(r)
            result.append({s: as_expr(as_expr(g.as_expr().subs(self.t, root)).expand()) for s, g in zip(self.symbols, polys)})
        return result

    def numerical_solutions(self, digits: int = 15) -> list[dict[Symbol, Expr]]:
        """The zeros numerically: the roots of ``polynomial`` to
        ``digits`` digits substituted in :meth:`polynomials` (much faster
        than evaluating :meth:`solutions`, whose ``CRootOf`` objects are
        refined at every evaluation).

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur_sympy import rur
        >>> rur([x**2 - 2, y - x**3], x, y, t=t).numerical_solutions(6)
        [{x: -1.41421, y: -2.82843}, {x: 1.41421, y: 2.82843}]
        """
        if self.polynomial.degree() <= 0:
            return []
        polys = self.polynomials()
        # the coordinate polynomials have large coefficients: the roots are
        # found with as many extra digits as the coefficients have
        height = max((abs(int(c.numerator)) for g in polys for c in g.coeffs()), default=1)
        extra = len(str(height)) + self.polynomial.degree()
        result = []
        for root in self.polynomial.nroots(n=digits + extra):
            value = as_expr(root)
            result.append({s: as_expr(as_expr(g.as_expr().subs(self.t, value)).evalf(digits)) for s, g in zip(self.symbols, polys)})
        return result

    def multiplicity(self, root: Union[Expr, int]) -> int:
        """The multiplicity of the zero at a root of ``polynomial``: the
        multiplicity of the root in ``characteristic``.

        Examples
        ========

        >>> from sympy.abc import x, y, t
        >>> from sympy_extras.polys.rur_sympy import rur
        >>> rur([(x - 1)**2, y - x], x, y, t=t).multiplicity(1)
        2
        """
        value = as_expr(root)
        count = 0
        p = self.characteristic
        linear = Poly(self.t - value, self.t)
        while p.degree() > 0:
            q, r = p.div(linear)
            if not r.is_zero:
                break
            p, count = q, count + 1
        return count


def rur(system: System, *gens: Symbol, t: Optional[Symbol] = None) -> RationalUnivariateRepresentation:
    """:func:`sympy_extras.polys.rur.rur` on SymPy objects: the rational
    univariate representation of a zero-dimensional system of polynomial
    equations with rational coefficients (Rouillier's algorithm), with the
    first separating form among the candidates of
    :func:`~sympy_extras.polys.ideals.separating_forms`.

    Parameters
    ==========

    system : Ideal or list of Expr or Poly
        The ideal, or the polynomials (equated to zero).
    gens : Symbols
        The variables, when ``system`` is a list of polynomials (all its
        free symbols by default).
    t : Symbol
        The variable of the representation (``t`` by default).

    Raises
    ======

    NotImplementedError
        If the system has infinitely many complex solutions or its
        coefficients are not rational.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t
    >>> from sympy_extras.polys.rur_sympy import rur
    >>> R = rur([x**2 + y**2 - 1, x - y], x, y, t=t)
    >>> R.form, R.polynomial, R.denominator, R.numerators
    (x, Poly(t**2 - 1/2, t, domain='QQ'), Poly(2*t, t, domain='QQ'), [Poly(1, t, domain='QQ'), Poly(1, t, domain='QQ')])

    Neither ``x`` nor ``x + y`` separates the zeros of the next system
    (two of them have ``x = 0``, two have ``x + y = 1``), ``x + 2y`` does:

    >>> R = rur([x**2 + y**2 - 1, x*y], x, y, t=t)
    >>> R.form, R.polynomial
    (x + 2*y, Poly(t**4 - 5*t**2 + 4, t, domain='QQ'))
    >>> R.polynomials()
    [Poly(-1/3*t**3 + 4/3*t, t, domain='QQ'), Poly(1/6*t**3 - 1/6*t, t, domain='QQ')]
    >>> R.solutions()
    [{x: 0, y: -1}, {x: -1, y: 0}, {x: 1, y: 0}, {x: 0, y: 1}]

    A system with a zero of multiplicity three:

    >>> R = rur([x**3, y - x - 1, z - y**2], x, y, z, t=t)
    >>> R.polynomial, R.characteristic, R.multiplicity(0)
    (Poly(t, t, domain='QQ'), Poly(t**3, t, domain='QQ'), 3)
    >>> R.solutions()
    [{x: 0, y: 1, z: 1}]
    """
    T = Symbol('t') if t is None else as_symbol(t)
    ideal = system if isinstance(system, Ideal) else Ideal(list(system), *gens)
    if ideal.domain != QQ:
        raise NotImplementedError("real zeros are counted for ideals over QQ, not %s" % ideal.domain)
    found = native.rur(native_ideal(ideal, CoreQQ), t=_core_symbol(T))
    return RationalUnivariateRepresentation(list(ideal.symbols), T, _sympy_expr(found.form), _poly(found.polynomial, T),
                                            _poly(found.characteristic, T), _poly(found.denominator, T),
                                            [_poly(g, T) for g in found.numerators])


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_expr(e: CoreExpr) -> Expr:
    return as_expr(to_sympy(e))


def _core_expr(e: Expr) -> CoreExpr:
    converted = from_sympy(e)
    if not isinstance(converted, CoreExpr):
        raise TypeError("%s is not an expression" % (e,))
    return converted


def _poly(p: DensePoly[Fraction], t: Symbol) -> Poly:
    """A native polynomial over ``QQ`` as a SymPy ``Poly`` in ``t``."""
    poly: Poly = Poly.from_list([QQ(c.numerator, c.denominator) for c in p.rep], t, domain=QQ)
    return poly


def solutions_of_core(polynomial: DensePoly[Fraction], polys: Sequence[DensePoly[Fraction]],
                      symbols: Sequence[CoreSymbol], real: bool) -> list[native.Point]:
    """The exact zeros of a native representation: the roots of
    ``polynomial`` (SymPy's ``CRootOf``, radicals when SymPy finds them;
    the real ones only with ``real``) substituted in the coordinate
    polynomials ``polys``, as core expressions (the root objects wrapped
    by :mod:`sympy_extras.interop`)."""
    t = as_symbol(to_sympy(polynomial.gen))
    sympy_polys = [_poly(g, t) for g in polys]
    f = _poly(polynomial, t)
    roots = f.real_roots() if real else f.all_roots()
    result: list[native.Point] = []
    for r in roots:
        root = as_expr(r)
        result.append({s: _core_expr(as_expr(as_expr(g.as_expr().subs(t, root)).expand()))
                       for s, g in zip(symbols, sympy_polys)})
    return result
