r"""Exponential parts of the solutions of a linear ODE at its irregular
finite singular points, from the local Newton polygon.

A hyperexponential solution $y = \exp(\int \rho)$ of $L(y) = \sum_i a_i
y^{(i)} = 0$ with $\rho \in \overline{K}(x)$ has, at a root $\alpha$ of an
irreducible factor $q$ of $a_n$, a principal part

$$\rho = \frac{r}{(x - \alpha)^m} + \cdots + \frac{e}{x - \alpha} + \cdots$$

The terms of order $m \ge 2$ are the *exponential part* at $\alpha$ (the
point is irregular singular when there is one; $e$ is the local exponent,
handled by the Fuchsian ansatz of :mod:`sympy_extras.solvers.linear_ode`).
With $y^{(i)}/y = R_i(\rho) = \rho^i + \cdots$ (the Riccati polynomials)
and $v_i$ the order of $a_i$ at $\alpha$, the term $a_i R_i(\rho)$ has
order $v_i - m i$ when $m \ge 2$, and $L(y)/y = 0$ forces the minimum of
$v_i - m i$ to be attained twice: $m$ is the slope of an edge of the lower
Newton polygon of the points $(i, v_i)$, and $r$ a root of the
characteristic equation $\sum_{i \in \text{edge}} \operatorname{lc}_\alpha(a_i)
r^i = 0$ over $K(\alpha)$, $\operatorname{lc}_\alpha(a_i)$ the coefficient
of $(x - \alpha)^{v_i}$ in $a_i$. The next term is found the same way
after dividing the solutions by $\exp(\int r/(x - \alpha)^m)$ (Bronstein;
the Newton polygon of Malgrange and Ramis), down to the order $2$.

For a solution with $\rho \in K(x)$ the principal parts at the conjugate
roots of $q$ are conjugate, and their sum $\sum_\sigma \sigma(r)/(x -
\sigma\alpha)^m$ is in $K(x)$: it is computed as a trace over
$K(\alpha) = K[c]/(q(c))$, and so is its integral $\sum_\sigma
-\sigma(r)/((m - 1)(x - \sigma\alpha)^{m-1})$. The characteristic
equation is factored over $K(\alpha)$ (the native number field
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField`, Trager's
algorithm), and only its roots in $K(\alpha)$ are followed when $q$ is not
linear; a root outside $K(\alpha)$ (a solution whose $\rho$ is defined
over a larger field) makes the search incomplete, as do slopes which are
not integers (ramified exponential parts: no hyperexponential solution
has them, so they are dropped silently) and a recursion deeper than the
order of the pole.

The module works on the objects of the core and imports no SymPy (its
face on SymPy objects is :mod:`sympy_extras.solvers.irregular_sympy`); the
root ``alpha`` of a place of degree more than one is a symbol, the
generator of the number field (the face writes it as SymPy's ``CRootOf``).

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.linear_ode import LinearOperator
>>> from sympy_extras.solvers.irregular import finite_exponential_parts, irregular_places
>>> x = Symbol('x')
>>> y = Function('y')(x)
>>> L = LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y)
>>> irregular_places(L)
[UPoly(x, x)]
>>> finite_exponential_parts(L)
([0, 1/x, -1/x], True)

The solutions are ``exp(1/x)`` and ``exp(-1/x)``, with ``rho = -1/x**2``
and ``1/x**2`` whose integrals are the parts listed (``0`` is always a
candidate).

References
==========

.. [1] M. Bronstein, *Linear ordinary differential equations: breaking
       through the order 2 barrier*, ISSAC '92, 42-48 (the exponential
       parts at a singularity from the Newton polygon).
.. [2] M. van der Put, M. F. Singer, *Galois Theory of Linear Differential
       Equations*, Springer, 2003, Section 3.3 (the Newton polygon of a
       differential operator).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product as cartesian
from typing import Final, Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField

from .linear_ode import (LinearOperator, UPoly, _FAILURES, _normal, _order_at, _places, _replaced,
                         _root_multiplicities, _sum)

__all__ = ['Place', 'irregular_places', 'local_exponential_parts', 'finite_exponential_parts', 'conjugate_sum']

#: how many terms of an exponential part are followed at one point
MAX_DEPTH = 6

_ZERO: Final[Expr] = Integer(0)


class Place:
    """A finite singular point: the irreducible monic ``q`` whose roots are
    the conjugate points, one root ``alpha`` (a rational number, or a
    symbol standing for a root of ``q``, the generator of ``K(alpha)``),
    and the field ``K(alpha)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.irregular import Place
    >>> from sympy_extras.solvers.linear_ode import UPoly
    >>> x = Symbol('x')
    >>> Place(UPoly.of(2*x - 1, x), x).alpha
    1/2
    """

    def __init__(self, q: UPoly, x: Symbol) -> None:
        self.q: Final[UPoly] = q.monic()
        self.x: Final[Symbol] = x
        self.degree: Final[int] = q.degree()
        self.alpha: Final[Expr] = -self.q.coeff(0) if self.degree == 1 else Dummy('alpha')

    def value(self, p: UPoly) -> Expr:
        """``p(alpha)`` as an expression in ``alpha`` of degree less than
        that of ``q`` (the polynomial ``p`` in ``x``)."""
        if self.degree == 1:
            return _normal(_replaced(p.as_expr(), {self.x: self.alpha}))
        reduced = p.rem(self.q)
        return _replaced(reduced.as_expr(), {self.x: self.alpha})

    def field(self) -> Optional[AlgebraicField]:
        """``K(alpha)`` as a number field, ``None`` when ``q`` does not
        have rational coefficients."""
        try:
            minpoly = [QQ.from_expr(c) for c in self.q.all_coeffs()]
        except _FAILURES:
            return None
        return AlgebraicField(minpoly, self.alpha)

    def __repr__(self) -> str:
        return "Place(%s, %s)" % (self.q.as_expr(), self.x)


def _leading_value(place: Place, p: UPoly, order: int) -> Expr:
    """The coefficient of ``(x - alpha)**order`` in ``q**order p`` at
    ``alpha``: ``p(alpha) q'(alpha)**order``."""
    derivative = place.q.diff()
    value = place.value(p)
    if order:
        value = value*place.value(derivative)**order
    if place.degree == 1:
        return value
    return place.value(UPoly.of(_replaced(value, {place.alpha: place.x}), place.x))


def _lower_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The lower convex hull of points with increasing abscissae."""
    hull: list[tuple[int, int]] = []
    for point in points:
        while len(hull) >= 2:
            (i1, v1), (i2, v2) = hull[-2], hull[-1]
            i3, v3 = point
            # drop the middle point when it lies above the chord
            if (v2 - v1)*(i3 - i1) >= (v3 - v1)*(i2 - i1):
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def _roots_in_field(characteristic: UPoly, place: Place) -> tuple[list[Expr], bool]:
    """The roots of the characteristic polynomial: all of them (closed
    forms) at a rational point, those in ``K(alpha)`` otherwise; and
    whether no root was missed."""
    if place.degree == 1:
        found = _root_multiplicities(characteristic)
        complete = sum(found.values()) == characteristic.degree()
        return [r for r in found if r != _ZERO], complete
    field = place.field()
    if field is None:
        return [], False
    try:
        dense = [field.from_expr(expand(c)) for c in characteristic.all_coeffs()]
    except _FAILURES:
        return [], False
    _, factors = field.factor_dense(dense)
    result: list[Expr] = []
    complete = True
    for f, _ in factors:
        if len(f) != 2:
            complete = False
            continue
        a, b = f
        r = field.to_expr(field.neg(field.exquo(b, a)))
        if r != _ZERO:
            result.append(r)
    return result, complete


def conjugate_sum(place: Place, r: Expr, m: int) -> Expr:
    r"""$\sum_\sigma \sigma(r)/(x - \sigma\alpha)^m$ over the conjugates of
    ``alpha``, for ``r`` in ``K(alpha)`` (an expression in ``alpha``), as a
    rational function in ``K(x)``: the trace of $r\, \tilde q_\alpha(x)^m$
    over $K(\alpha)$ divided by $q(x)^m$, where $\tilde q_\alpha(x) = q(x)/(x
    - \alpha)$.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.irregular import Place, conjugate_sum
    >>> from sympy_extras.solvers.linear_ode import UPoly
    >>> x = Symbol('x')
    >>> P = Place(UPoly.of(x**2 - 2, x), x)
    >>> conjugate_sum(P, P.alpha, 2)
    8*x/(x**2 - 2)**2

    Indeed ``sqrt(2)/(x - sqrt(2))**2 - sqrt(2)/(x + sqrt(2))**2`` is that
    function.
    """
    x = place.x
    if place.degree == 1:
        return r/(x - place.alpha)**m
    c = Dummy('c')
    q_c = UPoly.of(_replaced(place.q.as_expr(), {x: c}), c)
    # q(x)/(x - c) as a polynomial in x with coefficients in K[c]
    coefficients = place.q.all_coeffs()
    cofactor: list[Expr] = []
    carry: Expr = _ZERO
    for a in coefficients[:-1]:
        carry = carry*c + a
        cofactor.append(carry)
    cofactor_expr = _sum([a*x**(len(cofactor) - 1 - i) for i, a in enumerate(cofactor)])
    r_c = _replaced(r, {place.alpha: c})
    numerator = UPoly.of(expand(cofactor_expr**m), x)
    # the power sums of the roots of q, for the traces of the powers of c
    # (the exponents are below the degree of q after the reduction)
    power_sums = _power_sums(place.q, place.degree - 1)
    traced: dict[int, Expr] = {}
    for k, a in numerator.terms_.items():
        term = UPoly.of(expand(a*r_c), c).rem(q_c)
        traced[k] = _sum([value*power_sums[j] for j, value in term.terms_.items()])
    return UPoly(traced, x).as_expr()/place.q.as_expr()**m


def _power_sums(q: UPoly, count: int) -> list[Expr]:
    """``p_0, ..., p_count`` with ``p_k`` the sum of the ``k``-th powers of
    the roots of the monic ``q`` (Newton's identities)."""
    coefficients = q.all_coeffs()
    n = q.degree()
    sums: list[Expr] = [Integer(n)]
    for k in range(1, count + 1):
        total: Expr = _ZERO
        for i in range(1, min(k, n) + 1):
            total = total - coefficients[i]*sums[k - i]
        if k <= n:
            total = total - k*coefficients[k]
        sums.append(total)
    return sums


class _Term:
    """A candidate leading term of an exponential part, ``r/(x -
    alpha)**m`` summed over the conjugates, by its integral ``F``."""

    def __init__(self, F: Expr, m: int) -> None:
        self.F: Final[Expr] = F
        self.m: Final[int] = m


def _leading_terms(L: LinearOperator, place: Place, below: Optional[int]) -> tuple[list[_Term], bool]:
    """The candidate leading terms of the exponential part at the place:
    for the edges of the lower Newton polygon of slope ``m >= 2`` (and
    ``m < below``), ``r`` a root of the characteristic polynomial of the
    edge; and whether every candidate was found."""
    x = L.x
    data: list[tuple[int, int, UPoly]] = []
    for i, c in enumerate(L.coefficients):
        if c != _ZERO:
            order, cofactor = _order_at(UPoly.of(c, x), place.q)
            data.append((i, order, cofactor))
    if len(data) < 2:
        return [], True
    hull = _lower_hull([(i, order) for i, order, _ in data])
    candidates: list[_Term] = []
    complete = True
    t = Dummy('t')
    for (i1, v1), (i2, v2) in zip(hull, hull[1:]):
        # v_i - m i constant on the edge: m = (v2 - v1)/(i2 - i1)
        slope = Fraction(v2 - v1, i2 - i1)
        if slope < 2 or slope.denominator != 1:
            continue
        m = int(slope)
        if below is not None and m >= below:
            continue
        on_edge = [(i, _leading_value(place, cofactor, order)) for i, order, cofactor in data
                   if order - v1 == slope*(i - i1)]
        characteristic = UPoly.of(_sum([value*t**i for i, value in on_edge]), t)
        found, found_complete = _roots_in_field(characteristic, place)
        complete = complete and found_complete
        for r in found:
            # the integral of r/(x - alpha)**m is -r/((m - 1) (x - alpha)**(m - 1))
            integral = conjugate_sum(place, -r/(m - 1), m - 1)
            candidates.append(_Term(_normal(integral), m))
    return candidates, complete


def local_exponential_parts(L: LinearOperator, place: Place, below: Optional[int] = None, depth: int = 0
                            ) -> tuple[list[Expr], bool]:
    """The candidate integrals ``F`` of the exponential parts at the place
    (solutions ``exp(F) * ...``), term by term from the Newton polygon,
    each term of lower order than the one before; ``0`` is always a
    candidate. The flag says whether the candidates are all the possible
    parts.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, UPoly
    >>> from sympy_extras.solvers.irregular import Place, local_exponential_parts
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**3*y.diff(x) - (2 + x)*y, y)
    >>> local_exponential_parts(L, Place(UPoly.of(x, x), x))
    ([0, -1/x**2, -1/x - 1/x**2], True)

    ``rho = 2/x**3 + 1/x**2``: the first term alone is listed too (each
    level of the recursion keeps ``0``), the solution is
    ``exp(-1/x**2 - 1/x)``.
    """
    parts: list[Expr] = [_ZERO]
    leading, complete = _leading_terms(L, place, below)
    if depth >= MAX_DEPTH:
        return parts, complete and not leading
    for term in leading:
        M = L.transformed(exp(term.F))
        lower, lower_complete = local_exponential_parts(M, place, term.m, depth + 1)
        complete = complete and lower_complete
        for part in lower:
            candidate = term.F + part
            if candidate not in parts:
                parts.append(candidate)
    return parts, complete


def irregular_places(L: LinearOperator) -> list[UPoly]:
    """The irreducible factors of the leading coefficient at whose roots
    the operator is not regular singular (``ord(a_i) < ord(a_n) - (n -
    i)`` for some ``i``), as monic polynomials.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.irregular import irregular_places
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> irregular_places(LinearOperator.from_equation(x**3*(x - 1)*y.diff(x, 2) + y, y))
    [UPoly(x, x)]
    >>> irregular_places(LinearOperator.from_equation(x*y.diff(x, 2) + y, y))
    []
    """
    x = L.x
    n = L.order
    result: list[UPoly] = []
    for q, multiplicity in _places(L.coefficients[-1], x):
        for i, c in enumerate(L.coefficients[:-1]):
            if c == _ZERO:
                continue
            order, _ = _order_at(UPoly.of(c, x), q)
            if order < multiplicity - (n - i):
                result.append(q.monic())
                break
    return result


def finite_exponential_parts(L: LinearOperator, max_combinations: int = 200) -> tuple[list[Expr], bool]:
    """The candidate integrals ``F`` of the exponential parts at all the
    irregular finite singular points of ``L`` (with polynomial
    coefficients), one combination of the local candidates per entry
    (``0`` is always one), and whether the list is complete.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.irregular import finite_exponential_parts
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x) + y, y)
    >>> finite_exponential_parts(L)
    ([0, 1/x], True)
    """
    places = [Place(q, L.x) for q in irregular_places(L)]
    complete = True
    local: list[list[Expr]] = []
    for place in places:
        parts, parts_complete = local_exponential_parts(L, place)
        complete = complete and parts_complete
        local.append(parts)
    combinations = list(cartesian(*local)) if local else [()]
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]
        complete = False
    result: list[Expr] = []
    for combination in combinations:
        total: Expr = _ZERO
        for part in combination:
            total = total + part
        if total not in result:
            result.append(total)
    return result, complete
