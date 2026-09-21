"""The exact solutions of a regular chain without free variables.

The chain ``T_1(v_1), T_2(v_1, v_2), ...`` is solved from the bottom, like
:meth:`~sympy_extras.polys.regularchains.RegularChain.numerical_solutions`,
with algebraic numbers:

* a polynomial whose coefficients at the point found so far are rational
  numbers has its roots as root objects (``CRootOf``), in radicals where
  :func:`~sympy_extras.polys.roots.radical_form` writes them so;
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
  numerical roots of the latter are matched with the numerical roots of the
  former, which are matched with the root objects by their isolating
  intervals. This step is numerical: it is refused (``NotImplementedError``)
  when the match is not clear at the working precision.

The residuals of the candidates are not evaluated with SymPy's ``evalf``:
the residual of a true root is exactly zero, for which ``evalf`` raises the
precision again and again, refining the complex root object each time (a
minute for one residual of ``x**5 - x - 1 - sqrt(2)`` at a root of degree
ten). The numerical roots come from ``Poly.nroots``.
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.polys.polytools import Poly, resultant
from sympy.polys.rootoftools import ComplexRootOf
from sympy.simplify.radsimp import radsimp

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.polys.roots import radical_form

__all__ = ['exact_solutions']

#: the digits of the numerical matching, over those the coefficients need
_DIGITS = 40


class _Root:
    """A root of a polynomial with rational coefficients: its exact form,
    its numerical value and whether it is real."""

    __slots__ = ('exact', 'numeric', 'real')

    def __init__(self, exact: Expr, numeric: Expr, real: bool) -> None:
        self.exact = exact
        self.numeric = numeric
        self.real = real


class _Point:
    """The coordinates found so far."""

    __slots__ = ('exact', 'numeric', 'real')

    def __init__(self, exact: dict[Symbol, Expr], numeric: dict[Symbol, Expr], real: bool) -> None:
        self.exact = exact
        self.numeric = numeric
        self.real = real

    def extended(self, v: Symbol, exact: Expr, numeric: Expr, real: bool) -> _Point:
        first, second = dict(self.exact), dict(self.numeric)
        first[v] = exact
        second[v] = numeric
        return _Point(first, second, self.real and real)


def _distance(first: Expr, second: Expr) -> Expr:
    return as_expr(Abs(first - second))


def _inside(value: Expr, root: ComplexRootOf, tolerance: Expr) -> bool:
    """Whether the numerical ``value`` lies in the isolating interval of
    the root object (a rectangle for a complex root), within the
    ``tolerance``: the interval of a root object which has been evaluated
    is refined, to less than the error of the numerical value (the bug: the
    real solutions of a system asked for after the complex ones were
    refused, a rectangle being 1e-95 wide by then)."""
    interval = root._get_interval()
    x, y = as_expr(re(value)), as_expr(im(value))
    if root.is_real:
        low, high = Rational(str(interval.a)), Rational(str(interval.b))
        return bool(Abs(y) < tolerance and low - tolerance <= x <= high + tolerance)
    return bool(Rational(str(interval.ax)) - tolerance <= x <= Rational(str(interval.bx)) + tolerance
        and Rational(str(interval.ay)) - tolerance <= y <= Rational(str(interval.by)) + tolerance)


def _roots(poly: Poly, digits: int) -> Optional[list[_Root]]:
    """The roots of a squarefree polynomial with rational coefficients."""
    tolerance = Rational(1, 10)**(digits // 2)
    numerical = [as_expr(value) for value in poly.nroots(n=digits, maxsteps=200)]
    found: list[_Root] = []
    for root in poly.all_roots(radicals=False):
        exact = as_expr(root)
        if not isinstance(exact, ComplexRootOf):
            found.append(_Root(exact, as_expr(exact.evalf(digits)), True))
            continue
        matches = [value for value in numerical if _inside(value, exact, tolerance)]
        if len(matches) != 1:
            return None
        form = attempt(lambda: radical_form(exact), 5)
        found.append(_Root(exact if form is None else form, matches[0], bool(exact.is_real)))
    return found


def _nearest(candidates: list[_Root], value: Expr, digits: int) -> Optional[_Root]:
    """The candidate whose numerical value is ``value``, when it is
    clearly the only one."""
    near, far = Rational(1, 10)**(digits // 2), Rational(1, 10)**(digits // 4)
    distances = sorted(((_distance(value, candidate.numeric), index)
        for index, candidate in enumerate(candidates)), key=lambda pair: float(pair[0]))
    if not distances or not distances[0][0] < near or (len(distances) > 1 and not distances[1][0] > far):
        return None
    return candidates[distances[0][1]]


def _identified(candidates: list[_Root], poly: Poly, digits: int) -> Optional[list[_Root]]:
    """The candidates which are the roots of the numerical ``poly``."""
    chosen: list[_Root] = []
    for value in poly.nroots(n=digits, maxsteps=200):
        candidate = _nearest(candidates, as_expr(value), digits)
        if candidate is None or any(candidate is other for other in chosen):
            return None
        chosen.append(candidate)
    return chosen


def _eliminant(polys: list[Expr], variables: list[Symbol], k: int) -> Optional[Poly]:
    """A polynomial with rational coefficients in ``variables[k]`` which
    vanishes at that coordinate of every point of the chain."""
    r = polys[k]
    for j in range(k - 1, -1, -1):
        r = as_expr(resultant(r, polys[j], variables[j]))
    if r == 0:
        return None
    poly: Poly = Poly(r, variables[k], domain='QQ')
    return poly


def exact_solutions(polys: list[Expr], variables: list[Symbol], real: bool = False,
        fallback: Optional[Callable[[Symbol], Poly]] = None) -> list[dict[Symbol, Expr]]:
    """The points of the squarefree regular chain ``polys`` with the main
    ``variables`` (by increasing main variable, no free variable), the
    real ones only with ``real``; ``fallback(v)`` is a polynomial in ``v``
    alone which vanishes on the chain, asked for when a resultant vanishes.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.regularchains.solutions import exact_solutions
    >>> exact_solutions([y**2 - 2, x*y - 1], [y, x])
    [{y: -sqrt(2), x: -sqrt(2)/2}, {y: sqrt(2), x: sqrt(2)/2}]
    >>> exact_solutions([y**2 + 1, x - y], [y, x], real=True)
    []
    """
    digits = _DIGITS + sum(len(str(Poly(p, *variables).clear_denoms()[1].max_norm())) for p in polys)
    points = [_Point({}, {}, True)]
    for k, (p, v) in enumerate(zip(polys, variables)):
        # the coefficients are substituted one by one: a root object has the
        # symbol of its polynomial inside, which Poly would take for ``v``
        coefficients = [as_expr(c) for c in Poly(p, v).all_coeffs()]
        candidates: Optional[list[_Root]] = None
        searched = False
        following: list[_Point] = []
        for point in points:
            if real and not point.real:
                continue
            exact = [as_expr(c.xreplace(point.exact)) for c in coefficients]
            numeric = [as_expr(as_expr(c.xreplace(point.numeric)).evalf(digits)) for c in coefficients]
            if len(exact) == 2:
                value = as_expr(-exact[1] / exact[0])
                approximate = as_expr(-numeric[1] / numeric[0])
                if not value.has(ComplexRootOf):
                    simpler = attempt(lambda: as_expr(radsimp(value)), 5)
                    value = value if simpler is None else simpler
                else:
                    # a rational function of root objects is written as a
                    # root object of its own polynomial: FiniteSet sorts its
                    # elements, and the sort key of a sum evaluates its terms
                    # (45 s for the twelve points of three equations, 0.2 s
                    # for each evaluation of a complex root object)
                    if candidates is None and not searched:
                        searched = True
                        eliminant = _eliminant(polys, variables, k)
                        if eliminant is not None:
                            candidates = _roots(Poly(eliminant.sqf_part().as_expr(), v, domain='QQ'), digits)
                    root = None if candidates is None else _nearest(candidates, approximate, digits)
                    value = value if root is None else root.exact
                following.append(point.extended(v, value, approximate, True))
                continue
            if all(c.is_Rational for c in exact):
                found = _roots(Poly(exact, v, domain='QQ'), digits)
            else:
                if candidates is None:
                    searched = True
                    eliminant = _eliminant(polys, variables, k)
                    if eliminant is None and fallback is not None:
                        eliminant = fallback(v)
                    if eliminant is None:
                        raise NotImplementedError("no polynomial in %s alone was found" % v)
                    candidates = _roots(Poly(eliminant.sqf_part().as_expr(), v, domain='QQ'), digits)
                found = None if candidates is None else _identified(candidates, Poly(numeric, v), digits)
            if found is None:
                raise NotImplementedError("the roots of %s at %s were not told apart with %s digits"
                    % (p, point.exact, digits))
            following.extend(point.extended(v, root.exact, root.numeric, root.real) for root in found)
        points = following
    return [point.exact for point in points if point.real or not real]

