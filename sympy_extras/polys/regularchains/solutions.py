"""The exact solutions of a regular chain without free variables.

The chain ``T_1(v_1), T_2(v_1, v_2), ...`` is solved from the bottom, like
:meth:`~sympy_extras.polys.regularchains.regularchain.RegularChain.numerical_solutions`,
with algebraic numbers:

* a polynomial whose coefficients at the point found so far are rational
  numbers has its roots as root objects in SymPy's order
  (:func:`~sympy_extras.polys.complexrootof.polynomial_roots`), in
  radicals where :func:`~sympy_extras.polys.roots.radical_form` writes
  them so, the real ones otherwise as
  :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` and the others
  as :class:`~sympy_extras.polys.complexrootof.ComplexRootOf`;
* a polynomial of degree one in its main variable is solved by a
  division: the coordinate is a rational function of the ones below (every
  coordinate but the first one, for a system in shape position), which is
  written as a root object of its own polynomial when it involves root
  objects, by the numerical matching below (kept as it is when the match
  is not clear);
* otherwise the coordinate is a root of a polynomial with rational
  coefficients, obtained from the chain by iterated resultants (from a
  Gröbner basis of the saturated ideal when a resultant vanishes): its
  roots are root objects, and those which belong to the point are told from
  the others numerically. The chain being squarefree they are simple roots
  of the polynomial at the point, as many as its degree, so that the
  numerical roots of the latter (mpmath's ``polyroots``) are matched with
  the values of the root objects, which are exact (their isolating regions
  refined). This step is numerical: it is refused (``NotImplementedError``)
  when the match is not clear at the working precision.

The numerical values of the coordinates are kept along (mpmath numbers at
the working precision): the coefficients at a point are evaluated from
them, never by the evaluation of an expression with a hidden zero.

The module is native: it imports no SymPy (the SymPy face has its own,
``regularchains_sympy.solutions``, which matches SymPy's ``CRootOf`` with
``Poly.nroots`` through their isolating intervals). The numerical roots
are those of SymPy's ``Poly.nroots``, ported.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.polys.regularchains.solutions import exact_solutions
>>> x, y = symbols('x y')
>>> exact_solutions([y**2 - 2, x*y - 1], [y, x])
[{y: -sqrt(2), x: -sqrt(2)/2}, {y: sqrt(2), x: sqrt(2)/2}]
>>> exact_solutions([y**3 - y - 1, x - y**2], [y, x])[0]
{y: IndexedRoot(y**3 - y - 1, y, 0), x: IndexedRoot(x**3 - 2*x**2 + x - 1, x, 0)}
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polytools.py (Poly.nroots: the numerical roots and their order)
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on mpmath numbers.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Final, Optional, Sequence

import mpmath

from sympy_extras._timeout import attempt
from sympy_extras.core.basic import preorder_traversal
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import pow as power
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.cad.indexedroot import IndexedRoot
from sympy_extras.polys.complexrootof import ComplexRootOf, algebraic_number, polynomial_roots, real_roots_indexed
from sympy_extras.polys.complexrootof import _complex_value
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.polytools import Poly, resultant
from sympy_extras.polys.sqfreetools import dup_sqf_part

__all__ = ['exact_solutions']

#: the digits of the numerical matching, over those the coefficients need
_DIGITS = 40

#: a point: the coordinates found so far
Point = dict[Symbol, Expr]


class _Root:
    """A root of a polynomial with rational coefficients: its exact form,
    its numerical value and whether it is real."""

    __slots__ = ('exact', 'numeric', 'real')

    def __init__(self, exact: Expr, numeric: mpmath.mpc, real: bool) -> None:
        self.exact: Final[Expr] = exact
        self.numeric: Final[mpmath.mpc] = numeric
        self.real: Final[bool] = real


class _Point:
    """The coordinates found so far, exactly and numerically."""

    __slots__ = ('exact', 'numeric', 'real')

    def __init__(self, exact: Point, numeric: dict[Symbol, mpmath.mpc], real: bool) -> None:
        self.exact: Final[Point] = exact
        self.numeric: Final[dict[Symbol, mpmath.mpc]] = numeric
        self.real: Final[bool] = real

    def extended(self, v: Symbol, exact: Expr, numeric: mpmath.mpc, real: bool) -> _Point:
        first, second = dict(self.exact), dict(self.numeric)
        first[v] = exact
        second[v] = numeric
        return _Point(first, second, self.real and real)


class _Polynomial:
    """A polynomial of the chain as a polynomial in its main variable whose
    coefficients are polynomials with rational coefficients in the
    variables below (as dictionaries of monomials)."""

    __slots__ = ('coefficients', 'exprs', 'below')

    def __init__(self, p: Expr, v: Symbol, below: Sequence[Symbol]) -> None:
        poly = Poly(p, v, *below) if below else Poly(p, v)
        n = poly.degree(0)
        terms: list[dict[tuple[int, ...], Fraction]] = [{} for _ in range(n + 1)]
        for monomial, c in poly.terms():
            if not isinstance(c, Rational):
                raise NotImplementedError("the chain has a coefficient which is not rational: %s" % (c,))
            terms[n - monomial[0]][monomial[1:]] = c.as_fraction()
        self.coefficients: Final[list[dict[tuple[int, ...], Fraction]]] = terms
        self.below: Final[tuple[Symbol, ...]] = tuple(below)
        self.exprs: Final[list[Expr]] = [_expr(t, self.below) for t in terms]

    def numeric(self, point: dict[Symbol, mpmath.mpc]) -> list[mpmath.mpc]:
        """The coefficients at the numerical point (at the working precision)."""
        values = [point[s] for s in self.below]
        found: list[mpmath.mpc] = []
        for terms in self.coefficients:
            total = mpmath.mpc(0)
            for monomial, c in terms.items():
                term = mpmath.mpc(mpmath.mpf(c.numerator)/c.denominator)
                for value, e in zip(values, monomial):
                    if e:
                        term *= value**e
                total += term
            found.append(total)
        return found


def _expr(terms: dict[tuple[int, ...], Fraction], gens: tuple[Symbol, ...]) -> Expr:
    if not gens:
        return number_from_fraction(terms.get((), Fraction(0)))
    return Poly.from_dict(dict(terms), *gens).as_expr()


def _has_root_objects(e: Expr) -> bool:
    return any(isinstance(node, (ComplexRootOf, IndexedRoot)) for node in preorder_traversal(e))


def _value(e: Expr, prec: int) -> Optional[tuple[mpmath.mpc, bool]]:
    """The value of a rational number or of a root object (or of its
    radical form) and whether it is real; ``None`` otherwise."""
    a = algebraic_number(e)
    if a is None:
        return None
    return _complex_value(a, prec), a.is_real


def _roots(coefficients: list[Fraction], v: Symbol, prec: int) -> list[_Root]:
    """The roots of a square-free polynomial with rational coefficients."""
    found: list[_Root] = []
    for root in polynomial_roots(coefficients, v, radicals=False):
        nodes = [node for node in preorder_traversal(root) if isinstance(node, ComplexRootOf)]
        if not nodes:
            known = _value(root, prec)
            if known is None:
                raise NotImplementedError("no value for the root %s" % (root,))
            found.append(_Root(root, known[0], known[1]))
            continue
        # a root of a factor, possibly times an integer basis
        node = nodes[0]
        a = node.algebraic()
        scale = Integer(1) if root == node else mul(root, power(node, Integer(-1)))
        if not isinstance(scale, Integer):
            raise NotImplementedError("unexpected form of the root %s" % (root,))
        with mpmath.workprec(prec):
            numeric = _complex_value(a, prec)*scale.p
        from sympy_extras.polys.roots import radical_form
        form = attempt(lambda: radical_form(node), 5, native=True)
        exact = real_roots_indexed(node) if form is None else form
        found.append(_Root(exact if scale.p == 1 else mul(scale, exact), numeric, a.is_real))
    return found


def _nearest(candidates: list[_Root], value: mpmath.mpc, digits: int) -> Optional[_Root]:
    """The candidate whose numerical value is ``value``, when it is
    clearly the only one."""
    near, far = mpmath.mpf(10)**(-(digits // 2)), mpmath.mpf(10)**(-(digits // 4))
    distances = sorted(((abs(value - candidate.numeric), index) for index, candidate in enumerate(candidates)),
                       key=lambda pair: pair[0])
    if not distances or not distances[0][0] < near or (len(distances) > 1 and not distances[1][0] > far):
        return None
    return candidates[distances[0][1]]


def _identified(candidates: list[_Root], numeric: list[mpmath.mpc], digits: int) -> Optional[list[_Root]]:
    """The candidates which are the roots of the numerical polynomial with
    the coefficients ``numeric``, in the order of :func:`_numerical_roots`."""
    values = _numerical_roots(numeric)
    if values is None:
        return None
    chosen: list[_Root] = []
    for value in values:
        candidate = _nearest(candidates, value, digits)
        if candidate is None or any(candidate is other for other in chosen):
            return None
        chosen.append(candidate)
    return chosen


# PORTED FROM SYMPY: sympy/polys/polytools.py:Poly.nroots
def _numerical_roots(numeric: list[mpmath.mpc]) -> Optional[list[mpmath.mpc]]:
    """The roots of the numerical polynomial as SymPy's ``Poly.nroots``
    (``maxsteps=200``) finds and sorts them: mpmath's ``polyroots`` with
    ten times the degree as extra precision (fifteen times when it does
    not converge), the real roots first, then by real part, by the size of
    the imaginary part and its sign."""
    degree = len(numeric) - 1
    for extra in (10, 15):
        try:
            found = mpmath.polyroots(numeric, maxsteps=200, cleanup=True, error=False, extraprec=degree*extra)
        except mpmath.libmp.libhyper.NoConvergence:
            continue
        except (ZeroDivisionError, ValueError):
            return None
        roots = [mpmath.mpc(r) for r in found]
        return sorted(roots, key=lambda r: (1 if r.imag else 0, r.real, abs(r.imag), _sign(r.imag)))
    return None


def _sign(value: mpmath.mpf) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _eliminant(polys: Sequence[Expr], variables: Sequence[Symbol], k: int) -> Optional[list[Fraction]]:
    """A polynomial with rational coefficients in ``variables[k]`` which
    vanishes at that coordinate of every point of the chain (its
    coefficients, square-free)."""
    r = polys[k]
    for j in range(k - 1, -1, -1):
        r = resultant(r, polys[j], variables[j])
    if r == 0:
        return None
    return _rational_coefficients(r, variables[k])


def _rational_coefficients(e: Expr, v: Symbol) -> list[Fraction]:
    """The coefficients of the polynomial ``e`` in ``v`` alone,
    square-free."""
    coefficients: list[Fraction] = []
    for c in Poly(e, v).all_coeffs():
        if not isinstance(c, Rational):
            raise NotImplementedError("a polynomial in %s alone is expected, got %s" % (v, e))
        coefficients.append(c.as_fraction())
    return dup_sqf_part(coefficients, QQ)


def _digits(polys: Sequence[Expr], variables: Sequence[Symbol]) -> int:
    """The working digits: as many as the coefficients of the chain have,
    their denominators cleared, over :data:`_DIGITS`."""
    total = _DIGITS
    for p in polys:
        coefficients: list[Fraction] = []
        for _, term in Poly(p, *variables).terms():
            if isinstance(term, Rational):
                coefficients.append(term.as_fraction())
        denominator = 1
        for q in coefficients:
            denominator = denominator*q.denominator//_gcd(denominator, q.denominator)
        total += len(str(max((abs(q.numerator*(denominator//q.denominator)) for q in coefficients), default=0)))
    return total


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def exact_solutions(polys: Sequence[Expr], variables: Sequence[Symbol], real: bool = False,
                    fallback: Optional[Callable[[Symbol], Expr]] = None) -> list[Point]:
    """The points of the squarefree regular chain ``polys`` with the main
    ``variables`` (by increasing main variable, no free variable), the
    real ones only with ``real``; ``fallback(v)`` is a polynomial in ``v``
    alone which vanishes on the chain, asked for when a resultant vanishes.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.regularchains.solutions import exact_solutions
    >>> x, y = symbols('x y')
    >>> exact_solutions([y**2 + 1, x - y], [y, x], real=True)
    []
    >>> exact_solutions([y**3 - 2, x - y], [y, x], real=True)
    [{y: 2**(1/3), x: 2**(1/3)}]
    """
    digits = _digits(polys, variables)
    prec = mpmath.libmp.dps_to_prec(digits)
    with mpmath.workdps(digits):
        return _solve(polys, variables, real, fallback, digits, prec)


def _solve(polys: Sequence[Expr], variables: Sequence[Symbol], real: bool,
           fallback: Optional[Callable[[Symbol], Expr]], digits: int, prec: int) -> list[Point]:
    from sympy_extras.simplify.radsimp import radsimp
    points = [_Point({}, {}, True)]
    for k, (p, v) in enumerate(zip(polys, variables)):
        polynomial = _Polynomial(p, v, list(variables[:k]))
        candidates: Optional[list[_Root]] = None
        searched = False
        following: list[_Point] = []
        for point in points:
            if real and not point.real:
                continue
            exact = [as_expr(c.subs(point.exact)) if point.exact else c for c in polynomial.exprs]
            numeric = polynomial.numeric(point.numeric)
            if len(exact) == 2:
                value = mul(Integer(-1), exact[1], power(exact[0], Integer(-1)))
                approximate = -numeric[1]/numeric[0]
                if not _has_root_objects(value):
                    simpler = attempt(lambda: radsimp(value), 5, native=True)
                    value = value if simpler is None else simpler
                else:
                    # a rational function of root objects is written as a
                    # root object of its own polynomial
                    if candidates is None and not searched:
                        searched = True
                        eliminant = _eliminant(polys, variables, k)
                        if eliminant is not None:
                            candidates = _roots(eliminant, v, prec)
                    root = None if candidates is None else _nearest(candidates, approximate, digits)
                    value = value if root is None else root.exact
                following.append(point.extended(v, value, approximate, True))
                continue
            if all(isinstance(c, Rational) for c in exact):
                found: Optional[list[_Root]] = _roots([_fraction(c) for c in exact], v, prec)
            else:
                if candidates is None:
                    searched = True
                    eliminant = _eliminant(polys, variables, k)
                    if eliminant is None and fallback is not None:
                        eliminant = _rational_coefficients(fallback(v), v)
                    if eliminant is None:
                        raise NotImplementedError("no polynomial in %s alone was found" % v)
                    candidates = _roots(eliminant, v, prec)
                found = _identified(candidates, numeric, digits)
            if found is None:
                raise NotImplementedError("the roots of %s at %s were not told apart with %s digits"
                    % (p, point.exact, digits))
            following.extend(point.extended(v, root.exact, root.numeric, root.real) for root in found)
        points = following
    return [point.exact for point in points if point.real or not real]


def _fraction(c: Expr) -> Fraction:
    if not isinstance(c, Rational):
        raise TypeError("a rational number is expected, got %s" % (c,))
    return c.as_fraction()
