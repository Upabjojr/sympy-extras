r"""Gradient, divergence, curl and Laplacian in orthogonal curvilinear
coordinates.

Coordinates $q_1, q_2, q_3$ are *orthogonal* when the tangent vectors
$\partial \mathbf{x} / \partial q_i$ are pairwise perpendicular; their
lengths $h_i$ are the scale factors (Lamé coefficients), and a vector
field is given by its *physical* components along the unit vectors
$\mathbf{e}_i = h_i^{-1}\, \partial \mathbf{x} / \partial q_i$. In these
terms [1]_ [2]_

.. math::

   \nabla f &= \sum_i \frac{1}{h_i} \frac{\partial f}{\partial q_i}\, \mathbf{e}_i, \\
   \nabla \cdot \mathbf{F} &= \frac{1}{h_1 h_2 h_3} \sum_i \frac{\partial}{\partial q_i}
        \Bigl(\frac{h_1 h_2 h_3}{h_i} F_i\Bigr), \\
   (\nabla \times \mathbf{F})_i &= \frac{1}{h_j h_k}
        \Bigl(\frac{\partial (h_k F_k)}{\partial q_j} - \frac{\partial (h_j F_j)}{\partial q_k}\Bigr)
        \quad (i, j, k \text{ cyclic}), \\
   \Delta f &= \nabla \cdot \nabla f.

The gradient, the divergence and the Laplacian hold in any dimension;
the curl is that of three dimensions. :func:`cartesian`,
:func:`cylindrical` and :func:`spherical` are the usual systems;
:meth:`OrthogonalCoordinates.from_transformation` computes the scale
factors of any other from the Cartesian coordinates as functions of the
new ones, checking the orthogonality.

The results are reduced modulo $\sin^2 + \cos^2 = 1$ and the relations
of radicals (:func:`sympy_extras.simplify.normalform.normal_form`).

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.vector import spherical, cylindrical
>>> r, theta, phi, rho, z = (Symbol(s) for s in ['r', 'theta', 'phi', 'rho', 'z'])
>>> S = spherical(r, theta, phi)
>>> S.scale_factors
(1, r, r*sin(theta))
>>> S.gradient(1/r), S.laplacian(1/r), S.divergence([r, 0, 0])
([-1/r**2, 0, 0], 0, 3)
>>> S.curl([0, 0, r*sin(theta)])                 # a rigid rotation about the axis
[2*cos(theta), -2*sin(theta), 0]
>>> cylindrical(rho, phi, z).laplacian(Function('f')(rho))
(rho*Derivative(f(rho), (rho, 2)) + Derivative(f(rho), rho))/rho

References
==========

.. [1] P. M. Morse, H. Feshbach, *Methods of Theoretical Physics*, vol.
       1, McGraw-Hill, 1953, Section 1.3 (curvilinear coordinates) and
       the table at the end of Chapter 1.
.. [2] G. B. Arfken, H. J. Weber, F. E. Harris, *Mathematical Methods for
       Physicists*, 7th ed., Academic Press, 2013, Section 3.10.
"""
from __future__ import annotations

from typing import Callable, Final, Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions import sin

__all__ = ['OrthogonalCoordinates', 'cartesian', 'cylindrical', 'spherical']


def _default_simplifier(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


class OrthogonalCoordinates:
    """Orthogonal coordinates ``coordinates`` with the scale factors
    ``scale_factors`` (the lengths of the tangent vectors of the
    coordinate lines). Vector fields are sequences of physical
    components, along the unit tangent vectors."""

    def __init__(self, coordinates: Sequence[Symbol], scale_factors: Sequence[ExprLike],
                 simplifier: Optional[Callable[[Expr], Expr]] = None) -> None:
        if len(coordinates) != len(scale_factors):
            raise ValueError('%d scale factors for %d coordinates' % (len(scale_factors), len(coordinates)))
        if len(set(coordinates)) != len(coordinates):
            raise ValueError('the coordinates are not distinct')
        self.coordinates: Final[tuple[Symbol, ...]] = tuple(coordinates)
        self.scale_factors: Final[tuple[Expr, ...]] = tuple(to_expr(h) for h in scale_factors)
        self.dimension: Final[int] = len(coordinates)
        self.simplifier: Final[Callable[[Expr], Expr]] = _default_simplifier if simplifier is None else simplifier

    @classmethod
    def from_transformation(cls, coordinates: Sequence[Symbol], cartesian_coordinates: Sequence[ExprLike],
                            simplifier: Optional[Callable[[Expr], Expr]] = None) -> OrthogonalCoordinates:
        """The system in which the Cartesian coordinates are the given
        functions of ``coordinates``; ``ValueError`` when the coordinate
        lines are not orthogonal. The scale factors are the square roots
        of the squared lengths of the tangent vectors, as they come (the
        square root of a square is not taken).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.vector import OrthogonalCoordinates
        >>> u, v = Symbol('u'), Symbol('v')
        >>> parabolic = OrthogonalCoordinates.from_transformation([u, v], [(u**2 - v**2)/2, u*v])
        >>> parabolic.scale_factors
        (sqrt(u**2 + v**2), sqrt(u**2 + v**2))
        >>> parabolic.laplacian(u**2 - v**2)
        0
        """
        tidy = _default_simplifier if simplifier is None else simplifier
        functions = [to_expr(f) for f in cartesian_coordinates]
        tangents = [[f.diff(q) for f in functions] for q in coordinates]
        n = len(coordinates)
        for i in range(n):
            for j in range(i):
                if tidy(add(*[a*b for a, b in zip(tangents[i], tangents[j])])) != 0:
                    raise ValueError('the coordinates %s and %s are not orthogonal' % (coordinates[j], coordinates[i]))
        return cls(coordinates, [sqrt(tidy(add(*[a*a for a in tangent]))) for tangent in tangents], simplifier)

    def __repr__(self) -> str:
        return 'OrthogonalCoordinates(%s, %s)' % (list(self.coordinates), list(self.scale_factors))

    def _vector(self, components: Sequence[ExprLike]) -> list[Expr]:
        if len(components) != self.dimension:
            raise ValueError('%d components in dimension %d' % (len(components), self.dimension))
        return [to_expr(c) for c in components]

    def volume_element(self) -> Expr:
        """The product of the scale factors: the volume of the cell
        spanned by unit increments of the coordinates."""
        return self.simplifier(mul(*self.scale_factors)) if self.scale_factors else Integer(1)

    def gradient(self, f: ExprLike) -> list[Expr]:
        """The physical components of the gradient of a scalar."""
        scalar = to_expr(f)
        return [self.simplifier(scalar.diff(q)/h) for q, h in zip(self.coordinates, self.scale_factors)]

    def divergence(self, field: Sequence[ExprLike]) -> Expr:
        """The divergence of a vector field."""
        F = self._vector(field)
        volume = mul(*self.scale_factors)
        terms = [(volume/h*component).diff(q) for q, h, component in zip(self.coordinates, self.scale_factors, F)]
        return self.simplifier(add(*terms)/volume)

    def curl(self, field: Sequence[ExprLike]) -> list[Expr]:
        """The curl of a vector field, in three dimensions."""
        if self.dimension != 3:
            raise ValueError('the curl is defined in dimension three')
        F = self._vector(field)
        q, h = self.coordinates, self.scale_factors
        result: list[Expr] = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            result.append(self.simplifier(((h[k]*F[k]).diff(q[j]) - (h[j]*F[j]).diff(q[k]))/(h[j]*h[k])))
        return result

    def laplacian(self, f: ExprLike) -> Expr:
        """The Laplacian of a scalar."""
        scalar = to_expr(f)
        volume = mul(*self.scale_factors)
        terms = [(volume/h**2*scalar.diff(q)).diff(q) for q, h in zip(self.coordinates, self.scale_factors)]
        return self.simplifier(add(*terms)/volume)

    def vector_laplacian(self, field: Sequence[ExprLike]) -> list[Expr]:
        r"""The Laplacian of a vector field in three dimensions,
        $\nabla (\nabla \cdot \mathbf{F}) - \nabla \times (\nabla \times
        \mathbf{F})$."""
        first = self.gradient(self.divergence(field))
        second = self.curl(self.curl(field))
        return [self.simplifier(a - b) for a, b in zip(first, second)]


def cartesian(x: Symbol, y: Symbol, z: Symbol) -> OrthogonalCoordinates:
    """Cartesian coordinates: all the scale factors are one."""
    return OrthogonalCoordinates([x, y, z], [1, 1, 1])


def cylindrical(rho: Symbol, phi: Symbol, z: Symbol) -> OrthogonalCoordinates:
    r"""Cylindrical coordinates $(\rho \cos\varphi, \rho \sin\varphi,
    z)$, with the scale factors $(1, \rho, 1)$."""
    return OrthogonalCoordinates([rho, phi, z], [1, rho, 1])


def spherical(r: Symbol, theta: Symbol, phi: Symbol) -> OrthogonalCoordinates:
    r"""Spherical coordinates $(r \sin\theta \cos\varphi, r \sin\theta
    \sin\varphi, r \cos\theta)$ ($\theta$ the polar angle), with the scale
    factors $(1, r, r \sin\theta)$."""
    return OrthogonalCoordinates([r, theta, phi], [1, r, r*sin(theta)])


