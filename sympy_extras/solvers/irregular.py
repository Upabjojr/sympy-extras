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
$K(\alpha) = K[c]/(q(c))$. The characteristic equation is factored over
$K(\alpha)$ (SymPy's algebraic fields on a ``CRootOf``), and only its
roots in $K(\alpha)$ are followed when $q$ is not linear; a root outside
$K(\alpha)$ (a solution whose $\rho$ is defined over a larger field) makes
the search incomplete, as do slopes which are not integers (ramified
exponential parts: no hyperexponential solution has them, so they are
dropped silently) and a recursion deeper than the order of the pole.

Examples
========

>>> from sympy import Function
>>> from sympy.abc import x
>>> from sympy_extras.solvers.linear_ode import LinearOperator
>>> from sympy_extras.solvers.irregular import finite_exponential_parts, irregular_places
>>> y = Function('y')(x)
>>> L = LinearOperator.from_equation(x**4*y.diff(x, 2) + 2*x**3*y.diff(x) - y, y)
>>> irregular_places(L)
[Poly(x, x, domain='QQ')]
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

from itertools import product as cartesian
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.rationaltools import ratint_ratpart
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import CoercionFailed, NotAlgebraic, PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor_list
from sympy.polys.rootoftools import CRootOf

from sympy_extras._typing import as_expr

from .linear_ode import LinearOperator

__all__ = ['irregular_places', 'local_exponential_parts', 'finite_exponential_parts', 'conjugate_sum']

#: how many terms of an exponential part are followed at one point
MAX_DEPTH = 6


class Place:
    """A finite singular point: the irreducible monic ``q`` whose roots are
    the conjugate points, one root ``alpha`` (a rational number, or a
    ``CRootOf``), and the field ``K(alpha)``."""

    def __init__(self, q: Poly, x: Symbol) -> None:
        self.q: Poly = q.monic()
        self.x: Symbol = x
        self.degree: int = q.degree()
        if self.degree == 1:
            self.alpha: Expr = as_expr(-self.q.all_coeffs()[1])
        else:
            self.alpha = CRootOf(self.q.as_expr(), 0)

    def value(self, p: Poly) -> Expr:
        """``p(alpha)`` as an expression in ``alpha`` of degree less than
        that of ``q`` (the polynomial ``p`` in ``x``)."""
        if self.degree == 1:
            return as_expr(p.as_expr().subs(self.x, self.alpha))
        reduced = p.rem(self.q)
        return as_expr(reduced.as_expr().subs(self.x, self.alpha))


def _order_at(p: Poly, q: Poly) -> tuple[int, Poly]:
    """The order of ``p`` at the roots of ``q`` and the cofactor
    ``p/q**order``."""
    order = 0
    while not p.is_zero:
        quotient, remainder = p.div(q)
        if not remainder.is_zero:
            break
        p = quotient
        order += 1
    return order, p


def _leading_value(place: Place, p: Poly, order: int) -> Expr:
    """The coefficient of ``(x - alpha)**order`` in ``q**order p`` at
    ``alpha``: ``p(alpha) q'(alpha)**order``."""
    derivative = place.q.diff(place.x)
    value = place.value(p)
    if order:
        value = as_expr(value*place.value(derivative)**order)
    if place.degree == 1:
        return value
    return place.value(Poly(as_expr(value).subs(place.alpha, place.x), place.x))


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


def _roots_in_field(characteristic: Poly, place: Place) -> tuple[list[Expr], bool]:
    """The roots of the characteristic polynomial: all of them (closed
    forms) at a rational point, those in ``K(alpha)`` otherwise; and
    whether no root was missed."""
    if place.degree == 1:
        found = roots(characteristic)
        complete = sum(found.values()) == characteristic.degree()
        return [as_expr(r) for r in found if r != 0], complete
    field = QQ.algebraic_field(place.alpha)
    try:
        over_field = Poly(characteristic.as_expr(), characteristic.gen, domain=field)
    except (CoercionFailed, NotAlgebraic, PolynomialError):
        return [], False
    _, factors = over_field.factor_list()
    result: list[Expr] = []
    complete = True
    for f, _ in factors:
        if f.degree() != 1:
            complete = False
            continue
        a, b = f.all_coeffs()
        r = as_expr(-b/a)
        if r != 0:
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

    >>> from sympy import Poly, sqrt
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.irregular import Place, conjugate_sum
    >>> P = Place(Poly(x**2 - 2, x), x)
    >>> conjugate_sum(P, P.alpha, 2)
    8*x/(x**2 - 2)**2

    Indeed ``sqrt(2)/(x - sqrt(2))**2 - sqrt(2)/(x + sqrt(2))**2`` is that
    function.
    """
    x = place.x
    if place.degree == 1:
        return as_expr(r/(x - place.alpha)**m)
    c = Dummy('c')
    q_c = Poly(place.q.as_expr().subs(x, c), c)
    # q(x)/(x - c) as a polynomial in x with coefficients in K[c]
    coefficients = [as_expr(a) for a in place.q.all_coeffs()]
    cofactor: list[Expr] = []
    carry: Expr = S.Zero
    for a in coefficients[:-1]:
        carry = as_expr(carry*c + a)
        cofactor.append(carry)
    cofactor_poly = Poly(Add(*[a*x**(len(cofactor) - 1 - i) for i, a in enumerate(cofactor)]), x)
    r_c = Poly(as_expr(r).subs(place.alpha, c), c)
    numerator = cofactor_poly**m
    # the power sums of the roots of q, for the traces of the powers of c
    # (the exponents are below the degree of q after the reduction)
    power_sums = _power_sums(place.q, place.degree - 1)
    traced: list[Expr] = []
    for a in numerator.all_coeffs():
        term = Poly(a, c).mul(r_c).rem(q_c)
        total: Expr = S.Zero
        for (k,), value in term.terms():
            total = as_expr(total + value*power_sums[k])
        traced.append(total)
    numerator_x = Poly(Add(*[a*x**(len(traced) - 1 - i) for i, a in enumerate(traced)]), x)
    return as_expr(numerator_x.as_expr()/place.q.as_expr()**m)


def _power_sums(q: Poly, count: int) -> list[Expr]:
    """``p_0, ..., p_count`` with ``p_k`` the sum of the ``k``-th powers of
    the roots of the monic ``q`` (Newton's identities)."""
    coefficients = [as_expr(a) for a in q.all_coeffs()]
    n = q.degree()
    sums: list[Expr] = [as_expr(S(n))]
    for k in range(1, count + 1):
        total: Expr = S.Zero
        for i in range(1, min(k, n) + 1):
            total = as_expr(total - coefficients[i]*sums[k - i])
        if k <= n:
            total = as_expr(total - k*coefficients[k])
        sums.append(as_expr(total))
    return sums


def _leading_terms(L: LinearOperator, place: Place, below: Optional[int]) -> tuple[list[tuple[Expr, int]], bool]:
    """The candidate leading terms ``(E, m)`` of the exponential part at
    the place: ``E`` the sum over the conjugates of ``r/(x - alpha)**m``
    for the edges of the lower Newton polygon of slope ``m >= 2`` (and
    ``m < below``), ``r`` a root of the characteristic polynomial of the
    edge; and whether every candidate was found."""
    x = L.x
    data: list[tuple[int, int, Poly]] = []
    for i, c in enumerate(L.coefficients):
        if c != 0:
            order, cofactor = _order_at(Poly(c, x), place.q)
            data.append((i, order, cofactor))
    if len(data) < 2:
        return [], True
    hull = _lower_hull([(i, order) for i, order, _ in data])
    candidates: list[tuple[Expr, int]] = []
    complete = True
    t = Dummy('t')
    for (i1, v1), (i2, v2) in zip(hull, hull[1:]):
        # v_i - m i constant on the edge: m = (v2 - v1)/(i2 - i1)
        slope = Rational(v2 - v1, i2 - i1)
        if slope < 2 or not slope.is_integer:
            continue
        m = int(slope)
        if below is not None and m >= below:
            continue
        on_edge = [(i, _leading_value(place, cofactor, order)) for i, order, cofactor in data
                   if order - v1 == slope*(i - i1)]
        characteristic = Poly(as_expr(Add(*[value*t**i for i, value in on_edge])), t)
        found, found_complete = _roots_in_field(characteristic, place)
        complete = complete and found_complete
        for r in found:
            candidates.append((conjugate_sum(place, r, m), m))
    return candidates, complete


def _rational_integral(E: Expr, x: Symbol) -> Expr:
    """The integral of a rational function whose poles have no residue (a
    rational function)."""
    numerator, denominator = as_expr(E).as_numer_denom()
    try:
        rational_part, log_part = ratint_ratpart(Poly(numerator, x), Poly(denominator, x), x)
    except (PolynomialError, CoercionFailed):
        from sympy.integrals.integrals import integrate
        return as_expr(integrate(E, x))
    if as_expr(log_part) != 0:
        raise AssertionError('an exponential part with a residue')
    return as_expr(rational_part)


def local_exponential_parts(L: LinearOperator, place: Place, below: Optional[int] = None, depth: int = 0
                            ) -> tuple[list[Expr], bool]:
    """The candidate integrals ``F`` of the exponential parts at the place
    (solutions ``exp(F) * ...``), term by term from the Newton polygon,
    each term of lower order than the one before; ``0`` is always a
    candidate. The flag says whether the candidates are all the possible
    parts.

    Examples
    ========

    >>> from sympy import Function, Poly
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.irregular import Place, local_exponential_parts
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**3*y.diff(x) - (2 + x)*y, y)
    >>> local_exponential_parts(L, Place(Poly(x, x), x))
    ([0, -1/x**2, -1/x - 1/x**2], True)

    ``rho = 2/x**3 + 1/x**2``: the first term alone is listed too (each
    level of the recursion keeps ``0``), the solution is
    ``exp(-1/x**2 - 1/x)``.
    """
    x = L.x
    parts: list[Expr] = [S.Zero]
    leading, complete = _leading_terms(L, place, below)
    if depth >= MAX_DEPTH:
        return parts, complete and not leading
    for E, m in leading:
        F = _rational_integral(E, x)
        M = L.transformed(exp(F))
        lower, lower_complete = local_exponential_parts(M, place, m, depth + 1)
        complete = complete and lower_complete
        for part in lower:
            candidate = as_expr(F + part)
            if candidate not in parts:
                parts.append(candidate)
    return parts, complete


def irregular_places(L: LinearOperator) -> list[Poly]:
    """The irreducible factors of the leading coefficient at whose roots
    the operator is not regular singular (``ord(a_i) < ord(a_n) - (n -
    i)`` for some ``i``), as monic polynomials.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.irregular import irregular_places
    >>> y = Function('y')(x)
    >>> irregular_places(LinearOperator.from_equation(x**3*(x - 1)*y.diff(x, 2) + y, y))
    [Poly(x, x, domain='QQ')]
    >>> irregular_places(LinearOperator.from_equation(x*y.diff(x, 2) + y, y))
    []
    """
    x = L.x
    n = L.order
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    result: list[Poly] = []
    for f, multiplicity in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        for i, c in enumerate(L.coefficients[:-1]):
            if c == 0:
                continue
            order, _ = _order_at(Poly(c, x), q)
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

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> from sympy_extras.solvers.irregular import finite_exponential_parts
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
        total: Expr = S.Zero
        for part in combination:
            total = as_expr(total + part)
        if total not in result:
            result.append(total)
    return result, complete
