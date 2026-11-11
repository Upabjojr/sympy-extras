r"""Classical differential geometry: parametrised curves (speed,
curvature, torsion, the Frenet frame, arc length element) and
parametrised surfaces in space (the two fundamental forms, the Gaussian
and the mean curvature, the principal curvatures, the unit normal).

**Curves.** For a regular curve $\mathbf{r}(t)$ in the plane or in
space [1]_,

.. math:: \kappa = \frac{|\mathbf{r}' \times \mathbf{r}''|}{|\mathbf{r}'|^3},
          \qquad
          \tau = \frac{(\mathbf{r}' \times \mathbf{r}'') \cdot \mathbf{r}'''}{|\mathbf{r}' \times \mathbf{r}''|^2},

and the Frenet frame is $\mathbf{T} = \mathbf{r}'/|\mathbf{r}'|$,
$\mathbf{B} = \mathbf{r}' \times \mathbf{r}''/|\mathbf{r}' \times
\mathbf{r}''|$, $\mathbf{N} = \mathbf{B} \times \mathbf{T}$. A plane
curve has the signed curvature $(x' y'' - y' x'')/(x'^2 + y'^2)^{3/2}$.

**Surfaces.** For $\mathbf{r}(u, v)$ with the unit normal $\mathbf{n} =
\mathbf{r}_u \times \mathbf{r}_v / |\mathbf{r}_u \times \mathbf{r}_v|$,
the coefficients of the first and second fundamental forms are

.. math:: E = \mathbf{r}_u \cdot \mathbf{r}_u,\ F = \mathbf{r}_u \cdot \mathbf{r}_v,\
          G = \mathbf{r}_v \cdot \mathbf{r}_v, \qquad
          L = \mathbf{r}_{uu} \cdot \mathbf{n},\ M = \mathbf{r}_{uv} \cdot \mathbf{n},\
          N = \mathbf{r}_{vv} \cdot \mathbf{n},

and the Gaussian and mean curvatures are [1]_

.. math:: K = \frac{L N - M^2}{E G - F^2}, \qquad
          H = \frac{E N - 2 F M + G L}{2 (E G - F^2)}.

$K$ is computed without the square root of the normal: with $W = E G -
F^2$ and the determinants $l = [\mathbf{r}_{uu}, \mathbf{r}_u,
\mathbf{r}_v]$, $m$, $n$, it is $(l n - m^2)/W^2$, a rational expression
in the derivatives. By Gauss' Theorema Egregium it only depends on the
first fundamental form: it is half the scalar curvature of the metric
$\begin{pmatrix} E & F \\ F & G \end{pmatrix}$
(:class:`sympy_extras.diffgeom.Metric`), which the tests use as an
independent computation.

The symbols stand for real numbers. Every result is reduced by
:func:`sympy_extras.simplify.normalform.normal_form`; in the square root
of a quotient of polynomials the squares are taken out, as absolute
values unless the factor is positive (a symbol declared positive, a sum
of even powers).

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import cos, sin
>>> from sympy_extras.diffgeom.classical import Curve, Surface
>>> t, u, v = Symbol('t'), Symbol('u'), Symbol('v')
>>> a, b, R = Symbol('a', positive=True), Symbol('b', positive=True), Symbol('R', positive=True)
>>> helix = Curve([a*cos(t), a*sin(t), b*t], t)
>>> helix.curvature(), helix.torsion()
(a/(a**2 + b**2), b/(a**2 + b**2))
>>> sphere = Surface([R*sin(u)*cos(v), R*sin(u)*sin(v), R*cos(u)], u, v)
>>> sphere.first_fundamental_form(), sphere.gaussian_curvature()
((R**2, 0, R**2*sin(u)**2), R**(-2))
>>> Surface([u*cos(v), u*sin(v), v], u, v).mean_curvature()          # the helicoid is a minimal surface
0

References
==========

.. [1] M. P. do Carmo, *Differential Geometry of Curves and Surfaces*,
       Prentice-Hall, 1976, Sections 1.5 (curvature and torsion), 2.5 (the
       first fundamental form), 3.2 and 3.3 (the second fundamental form,
       Gaussian and mean curvature) and 4.3 (the Theorema Egregium).
"""
from __future__ import annotations

from typing import Final, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import abs_
from sympy_extras.matrices.linalg import _collect_kernels
from sympy_extras.simplify.normalform import _numer_denom, normal_form

__all__ = ['Curve', 'Surface']

_Vector = list[Expr]


def _dot(a: Sequence[Expr], b: Sequence[Expr]) -> Expr:
    return add(*[p*q for p, q in zip(a, b)])


def _cross(a: Sequence[Expr], b: Sequence[Expr]) -> _Vector:
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def _triple(a: Sequence[Expr], b: Sequence[Expr], c: Sequence[Expr]) -> Expr:
    return _dot(a, _cross(b, c))


def _positive_polynomial(f: Expr, symbols: Sequence[Symbol]) -> bool:
    """A sum of even powers with positive coefficients, or an expression
    the core knows to be nonnegative."""
    if f.is_nonnegative is True:
        return True
    from sympy_extras.polys.polytools import Poly
    terms = Poly(f, *symbols).terms()
    return all(all(k % 2 == 0 for k in monomial) and coefficient.is_positive is True for monomial, coefficient in terms)


def _polynomial_in_symbols(e: Expr) -> bool:
    found: dict[Expr, int] = {}
    _collect_kernels(e, found)
    return all(isinstance(k, Symbol) for k in found)


def _root_of_factors(e: Expr) -> tuple[Expr, Expr]:
    """``(outside, inside)`` with ``sqrt(e) = outside*sqrt(inside)`` for a
    polynomial in symbols, the squares taken out."""
    symbols = sorted((s for s in e.free_symbols if isinstance(s, Symbol)), key=lambda s: s.name)
    if not symbols:
        return Integer(1), e
    from sympy_extras.polys.polytools import factor_list
    content, factors = factor_list(e, *symbols)
    outside: list[Expr] = []
    inside: list[Expr] = [content]
    for factor, exponent in factors:
        if not isinstance(exponent, Integer):
            return Integer(1), e
        k = exponent.p
        if k // 2:
            base = factor if _positive_polynomial(factor, symbols) else abs_(factor)
            outside.append(pow(base, Integer(k // 2)))
        if k % 2:
            inside.append(factor)
    return mul(*outside), mul(*inside)


def _split_sqrt(e: Expr) -> tuple[Expr, Expr, Expr]:
    """``(outside, top, bottom)`` with ``sqrt(e) = outside*sqrt(top)/
    sqrt(bottom)`` for a nonnegative quantity: the squares of the factors
    of the numerator and of the denominator are taken out when these are
    polynomials in symbols."""
    reduced = normal_form(e)
    if reduced == 0:
        return Integer(0), Integer(1), Integer(1)
    numerator, denominator = _numer_denom(reduced)
    if not (_polynomial_in_symbols(numerator) and _polynomial_in_symbols(denominator)):
        return Integer(1), numerator, denominator
    top_out, top_in = _root_of_factors(numerator)
    bottom_out, bottom_in = _root_of_factors(denominator)
    return top_out/bottom_out, top_in, bottom_in


def _sqrt(e: Expr) -> Expr:
    outside, top, bottom = _split_sqrt(e)
    return outside*sqrt(top)/sqrt(bottom)


def _over_sqrt(e: Expr, radicand: Expr, power: int = 1) -> Expr:
    """``e/sqrt(radicand)**power`` (``power`` odd), the rational part
    reduced and the square roots left as factors."""
    outside, top, bottom = _split_sqrt(radicand)
    half = power // 2
    rational = normal_form(e*bottom**half/(outside**power*top**half))
    return rational*sqrt(bottom)/sqrt(top)


class Curve:
    """The parametrised curve with the coordinates ``components`` (two or
    three expressions) as functions of ``parameter``."""

    def __init__(self, components: Sequence[ExprLike], parameter: Symbol) -> None:
        if len(components) not in (2, 3):
            raise ValueError('a curve in the plane or in space has two or three coordinates')
        self.components: Final[tuple[Expr, ...]] = tuple(to_expr(c) for c in components)
        self.parameter: Final[Symbol] = parameter
        self.dimension: Final[int] = len(components)

    def __repr__(self) -> str:
        return 'Curve(%s, %s)' % (list(self.components), self.parameter)

    def derivative(self, order: int = 1) -> list[Expr]:
        """The derivative of the given order of the position vector."""
        return [normal_form(c.diff(self.parameter, order)) if order else c for c in self.components]

    def _space(self, order: int) -> _Vector:
        vector = self.derivative(order)
        zero: Expr = Integer(0)
        return vector + [zero]*(3 - len(vector))

    def speed(self) -> Expr:
        r"""$|\mathbf{r}'|$: the arc length is its integral.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos, sin
        >>> from sympy_extras.diffgeom.classical import Curve
        >>> t = Symbol('t')
        >>> Curve([cos(t), sin(t), t], t).speed(), Curve([t, t**2], t).speed()
        (sqrt(2), sqrt(4*t**2 + 1))
        """
        velocity = self.derivative(1)
        return _sqrt(_dot(velocity, velocity))

    def tangent(self) -> list[Expr]:
        """The unit tangent vector."""
        velocity = self.derivative(1)
        speed2 = _dot(velocity, velocity)
        return [_over_sqrt(c, speed2) for c in velocity]

    def curvature(self) -> Expr:
        r"""The curvature $\kappa \ge 0$.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom.classical import Curve
        >>> t = Symbol('t')
        >>> Curve([t, t**2], t).curvature()                     # the parabola
        2/(4*t**2 + 1)**(3/2)
        """
        first, second = self._space(1), self._space(2)
        product = _cross(first, second)
        return _sqrt(_dot(product, product)/_dot(first, first)**3)

    def signed_curvature(self) -> Expr:
        """The signed curvature of a plane curve: positive when the
        curve turns to the left."""
        if self.dimension != 2:
            raise ValueError('the signed curvature is that of a plane curve')
        (x1, y1), (x2, y2) = self.derivative(1), self.derivative(2)
        return _over_sqrt(x1*y2 - y1*x2, x1**2 + y1**2, 3)

    def torsion(self) -> Expr:
        r"""The torsion $\tau$ of a space curve (zero for a plane curve);
        ``ValueError`` where the curvature vanishes identically.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom.classical import Curve
        >>> t = Symbol('t')
        >>> Curve([t, t**2, t**3], t).torsion()                 # the twisted cubic
        3/(9*t**4 + 9*t**2 + 1)
        """
        first, second, third = self._space(1), self._space(2), self._space(3)
        product = _cross(first, second)
        norm2 = normal_form(_dot(product, product))
        if norm2 == 0:
            raise ValueError('the curvature vanishes: the torsion is not defined')
        return normal_form(_dot(product, third)/norm2)

    def frenet_frame(self) -> tuple[list[Expr], list[Expr], list[Expr]]:
        """The unit tangent, principal normal and binormal vectors of a
        space curve (three coordinates each)."""
        first, second = self._space(1), self._space(2)
        product = _cross(first, second)
        norm2, speed2 = normal_form(_dot(product, product)), _dot(first, first)
        if norm2 == 0:
            raise ValueError('the curvature vanishes: the Frenet frame is not defined')
        tangent = [_over_sqrt(c, speed2) for c in first]
        binormal = [_over_sqrt(c, norm2) for c in product]
        # N = B x T = (r' x r'') x r' / (|r' x r''| |r'|)
        normal = [_over_sqrt(c, norm2*speed2) for c in _cross(product, first)]
        return tangent, normal, binormal


class Surface:
    """The parametrised surface in space with the coordinates
    ``components`` as functions of the parameters ``u`` and ``v``."""

    def __init__(self, components: Sequence[ExprLike], u: Symbol, v: Symbol) -> None:
        if len(components) != 3:
            raise ValueError('a surface in space has three coordinates')
        if u == v:
            raise ValueError('the parameters are not distinct')
        self.components: Final[tuple[Expr, ...]] = tuple(to_expr(c) for c in components)
        self.u: Final[Symbol] = u
        self.v: Final[Symbol] = v
        self._ru: Final[_Vector] = [c.diff(u) for c in self.components]
        self._rv: Final[_Vector] = [c.diff(v) for c in self.components]

    def __repr__(self) -> str:
        return 'Surface(%s, %s, %s)' % (list(self.components), self.u, self.v)

    def first_fundamental_form(self) -> tuple[Expr, Expr, Expr]:
        """The coefficients ``(E, F, G)``."""
        return (normal_form(_dot(self._ru, self._ru)), normal_form(_dot(self._ru, self._rv)),
                normal_form(_dot(self._rv, self._rv)))

    def metric_determinant(self) -> Expr:
        r"""$W = E G - F^2 = |\mathbf{r}_u \times \mathbf{r}_v|^2$; the
        element of area is its square root."""
        E, F, G = self.first_fundamental_form()
        return normal_form(E*G - F**2)

    def area_element(self) -> Expr:
        """The element of area: the integral over the parameters is the
        area.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom.classical import Surface
        >>> u, v = Symbol('u'), Symbol('v')
        >>> Surface([u, v, u*v], u, v).area_element()
        sqrt(u**2 + v**2 + 1)
        """
        return _sqrt(self.metric_determinant())

    def normal(self) -> list[Expr]:
        """The unit normal vector, along ``r_u x r_v``."""
        W = self.metric_determinant()
        if W == 0:
            raise ValueError('the parametrisation is not regular')
        return [_over_sqrt(c, W) for c in _cross(self._ru, self._rv)]

    def _second_determinants(self) -> tuple[Expr, Expr, Expr]:
        """``(l, m, n)``: the second fundamental form times ``sqrt(W)``."""
        u, v = self.u, self.v
        ruu = [c.diff(u) for c in self._ru]
        ruv = [c.diff(v) for c in self._ru]
        rvv = [c.diff(v) for c in self._rv]
        return (normal_form(_triple(ruu, self._ru, self._rv)), normal_form(_triple(ruv, self._ru, self._rv)),
                normal_form(_triple(rvv, self._ru, self._rv)))

    def second_fundamental_form(self) -> tuple[Expr, Expr, Expr]:
        """The coefficients ``(L, M, N)``."""
        W = self.metric_determinant()
        if W == 0:
            raise ValueError('the parametrisation is not regular')
        l, m, n = self._second_determinants()
        return _over_sqrt(l, W), _over_sqrt(m, W), _over_sqrt(n, W)

    def gaussian_curvature(self) -> Expr:
        """The Gaussian curvature ``K``.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom.classical import Surface
        >>> u, v = Symbol('u'), Symbol('v')
        >>> Surface([u, v, u*v], u, v).gaussian_curvature()             # a hyperbolic paraboloid
        -1/(u**4 + 2*u**2*v**2 + 2*u**2 + v**4 + 2*v**2 + 1)
        """
        W = self.metric_determinant()
        if W == 0:
            raise ValueError('the parametrisation is not regular')
        l, m, n = self._second_determinants()
        return normal_form((l*n - m**2)/W**2)

    def mean_curvature(self) -> Expr:
        """The mean curvature ``H`` (its sign is that of the normal
        ``r_u x r_v``)."""
        E, F, G = self.first_fundamental_form()
        W = self.metric_determinant()
        if W == 0:
            raise ValueError('the parametrisation is not regular')
        l, m, n = self._second_determinants()
        return _over_sqrt((E*n - 2*F*m + G*l)/2, W, 3)

    def principal_curvatures(self) -> tuple[Expr, Expr]:
        r"""The principal curvatures $H \mp \sqrt{H^2 - K}$.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos, sin
        >>> from sympy_extras.diffgeom.classical import Surface
        >>> u, v = Symbol('u'), Symbol('v')
        >>> Surface([cos(u), sin(u), v], u, v).principal_curvatures()   # a cylinder
        (-1, 0)
        """
        H, K = self.mean_curvature(), self.gaussian_curvature()
        root = _sqrt(normal_form(H**2 - K))
        return normal_form(H - root), normal_form(H + root)
