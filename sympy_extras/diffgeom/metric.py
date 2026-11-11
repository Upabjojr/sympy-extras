r"""The curvature of a (pseudo-)Riemannian metric from its components:
Christoffel symbols, the Riemann, Ricci, Einstein and Weyl tensors, the
scalar curvature, geodesics, covariant derivatives and Killing vectors.

A :class:`Metric` is the matrix $g_{ij}$ of a metric in coordinates
$x^1, \dots, x^n$. With $g^{ij}$ its inverse, the definitions are those
of Misner, Thorne and Wheeler [1]_ (also Wald [2]_ and Carroll [3]_):

.. math::

   \Gamma^i_{jk} &= \tfrac12\, g^{il} \bigl(\partial_j g_{lk} + \partial_k g_{lj} - \partial_l g_{jk}\bigr), \\
   R^i{}_{jkl} &= \partial_k \Gamma^i_{lj} - \partial_l \Gamma^i_{kj}
                  + \Gamma^i_{km} \Gamma^m_{lj} - \Gamma^i_{lm} \Gamma^m_{kj}, \\
   R_{jl} &= R^i{}_{jil}, \qquad R = g^{jl} R_{jl}, \qquad
   G_{jl} = R_{jl} - \tfrac12 R\, g_{jl},

so that the sphere of radius $a$ has the scalar curvature $2/a^2$. The
Weyl tensor (for $n \ge 3$) is

.. math:: C_{ijkl} = R_{ijkl} - \frac{1}{n - 2} \bigl(g_{ik} R_{jl} - g_{il} R_{jk} - g_{jk} R_{il} + g_{jl} R_{ik}\bigr)
          + \frac{R}{(n - 1)(n - 2)} \bigl(g_{ik} g_{jl} - g_{il} g_{jk}\bigr).

Only the components which the symmetries leave independent are computed
($\Gamma^i_{jk}$ for $j \le k$, $R^i{}_{jkl}$ for $k < l$), each is
simplified once and kept. The simplifier is
:func:`sympy_extras.simplify.normalform.normal_form` unless another one
is given: a quotient of polynomials in lowest terms, reduced modulo
$\sin^2 + \cos^2 = 1$ and the relations of the radicals, which is zero
exactly when the component vanishes by these relations. The entries of the
metric may contain undefined functions of the coordinates (``a(t)``):
their derivatives appear in the results.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin
>>> from sympy_extras.diffgeom import Metric
>>> a, theta, phi = Symbol('a'), Symbol('theta'), Symbol('phi')
>>> sphere = Metric([[a**2, 0], [0, a**2*sin(theta)**2]], [theta, phi])
>>> sphere.christoffel(0, 1, 1), sphere.christoffel(1, 0, 1)
(-sin(theta)*cos(theta), cos(theta)/sin(theta))
>>> sphere.ricci(), sphere.ricci_scalar()
(ExprMatrix([[1, 0], [0, sin(theta)**2]]), 2/a**2)
>>> t, r, m = Symbol('t'), Symbol('r'), Symbol('m')
>>> schwarzschild = Metric([[-(1 - 2*m/r), 0, 0, 0], [0, 1/(1 - 2*m/r), 0, 0], [0, 0, r**2, 0],
...                         [0, 0, 0, r**2*sin(theta)**2]], [t, r, theta, phi])
>>> schwarzschild.ricci() == schwarzschild.ricci().zeros(4, 4), schwarzschild.kretschmann()
(True, 48*m**2/r**6)

References
==========

.. [1] C. W. Misner, K. S. Thorne, J. A. Wheeler, *Gravitation*, Freeman,
       1973, Chapters 8 (connection), 11 (curvature) and 13 (the metric
       connection, the symmetries of the Riemann tensor, the Weyl
       tensor).
.. [2] R. M. Wald, *General Relativity*, University of Chicago Press,
       1984, Chapter 3 and Appendix C (Killing vectors).
.. [3] S. M. Carroll, *Spacetime and Geometry*, Addison-Wesley, 2004,
       Chapter 3.
"""
from __future__ import annotations

from typing import Callable, Final, Optional, Sequence, Union

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Function, expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.symbol import Symbol
from sympy_extras.matrices.dense import ExprMatrix

__all__ = ['Metric']

_Rank3 = list[list[list[Expr]]]
_Rank4 = list[list[list[list[Expr]]]]

_ZERO: Final[Expr] = Integer(0)


def _default_simplifier(e: Expr) -> Expr:
    from sympy_extras.simplify.normalform import normal_form
    return normal_form(e)


class Metric:
    """The metric with the components ``components`` (a symmetric,
    invertible matrix: an :class:`~sympy_extras.matrices.ExprMatrix` or
    its rows) in the coordinates ``coordinates``. Indices count from
    zero, in the order of the coordinates."""

    def __init__(self, components: Union[ExprMatrix, Sequence[Sequence[ExprLike]]], coordinates: Sequence[Symbol],
                 simplifier: Optional[Callable[[Expr], Expr]] = None) -> None:
        g = components if isinstance(components, ExprMatrix) else ExprMatrix.from_rows(components)
        n = len(coordinates)
        if g.shape != (n, n):
            raise ValueError('a %d x %d matrix for %d coordinates' % (g.rows, g.cols, n))
        if len(set(coordinates)) != n:
            raise ValueError('the coordinates are not distinct')
        self.simplifier: Final[Callable[[Expr], Expr]] = _default_simplifier if simplifier is None else simplifier
        if any(self.simplifier(g[i, j] - g[j, i]) != 0 for i in range(n) for j in range(i)):
            raise ValueError('the matrix of a metric is symmetric')
        self.g: Final[ExprMatrix] = g
        self.coordinates: Final[tuple[Symbol, ...]] = tuple(coordinates)
        self.dimension: Final[int] = n
        self._inverse: Optional[ExprMatrix] = None
        self._christoffel: Optional[_Rank3] = None
        self._riemann: Optional[_Rank4] = None
        self._ricci: Optional[ExprMatrix] = None

    @classmethod
    def from_line_element(cls, ds2: ExprLike, coordinates: Sequence[Symbol], differentials: Sequence[Symbol],
                          simplifier: Optional[Callable[[Expr], Expr]] = None) -> Metric:
        """The metric of the line element ``ds2``, a quadratic form in the
        symbols ``differentials`` standing for the differentials of the
        coordinates.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> r, phi, dr, dphi = Symbol('r'), Symbol('phi'), Symbol('dr'), Symbol('dphi')
        >>> Metric.from_line_element(dr**2 + r**2*dphi**2, [r, phi], [dr, dphi]).g
        ExprMatrix([[1, 0], [0, r**2]])
        """
        n = len(coordinates)
        if len(differentials) != n:
            raise ValueError('%d differentials for %d coordinates' % (len(differentials), n))
        form = expand(to_expr(ds2))
        rows: list[list[Expr]] = []
        total: Expr = Integer(0)
        for i in range(n):
            line: list[Expr] = []
            for j in range(n):
                if i == j:
                    entry = form.coeff(differentials[i], 2)
                else:
                    entry = form.coeff(differentials[i], 1).coeff(differentials[j], 1)/2
                line.append(entry)
                total = total + entry*differentials[i]*differentials[j]
            rows.append(line)
        if any(row[j].has(d) for row in rows for j in range(n) for d in differentials) or expand(form - total) != 0:
            raise ValueError('%s is not a quadratic form in the differentials' % (ds2,))
        return cls(rows, coordinates, simplifier)

    def __repr__(self) -> str:
        return 'Metric(%r, %s)' % (self.g, list(self.coordinates))

    # -- the metric

    @property
    def inverse(self) -> ExprMatrix:
        """The contravariant components ``g^{ij}``."""
        if self._inverse is None:
            n = self.dimension
            if all(self.g[i, j] == 0 for i in range(n) for j in range(n) if i != j):
                if any(self.g[i, i] == 0 for i in range(n)):
                    raise ValueError('the metric is degenerate')
                self._inverse = ExprMatrix(n, n, [self.simplifier(1/self.g[i, i]) if i == j else _ZERO
                                                  for i in range(n) for j in range(n)])
            else:
                from sympy_extras.matrices.linalg import inverse
                try:
                    self._inverse = inverse(self.g).applyfunc(self.simplifier)
                except ValueError:
                    raise ValueError('the metric is degenerate') from None
        return self._inverse

    def determinant(self) -> Expr:
        """The determinant of ``g_{ij}`` (the volume element is the
        square root of its absolute value).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin
        >>> from sympy_extras.diffgeom import Metric
        >>> r, theta, phi = Symbol('r'), Symbol('theta'), Symbol('phi')
        >>> Metric([[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]], [r, theta, phi]).determinant()
        r**4*sin(theta)**2
        """
        from sympy_extras.matrices.linalg import det
        return self.simplifier(det(self.g))

    def lower(self, vector: Sequence[ExprLike]) -> list[Expr]:
        """The covariant components ``g_{ij} V^j`` of a vector."""
        V = self._vector(vector)
        n = self.dimension
        return [self.simplifier(add(*[self.g[i, j]*V[j] for j in range(n)])) for i in range(n)]

    def raise_index(self, covector: Sequence[ExprLike]) -> list[Expr]:
        """The contravariant components ``g^{ij} w_j`` of a covector."""
        w = self._vector(covector)
        n = self.dimension
        inverse = self.inverse
        return [self.simplifier(add(*[inverse[i, j]*w[j] for j in range(n)])) for i in range(n)]

    def _vector(self, components: Sequence[ExprLike]) -> list[Expr]:
        if len(components) != self.dimension:
            raise ValueError('%d components in dimension %d' % (len(components), self.dimension))
        return [to_expr(c) for c in components]

    # -- the connection

    def _connection(self) -> _Rank3:
        if self._christoffel is None:
            n = self.dimension
            x = self.coordinates
            g, inverse = self.g, self.inverse
            # dg[l][j][k] = d_l g_jk
            dg = [[[g[j, k].diff(x[l]) for k in range(n)] for j in range(n)] for l in range(n)]
            table: _Rank3 = [[[_ZERO]*n for _ in range(n)] for _ in range(n)]
            for j in range(n):
                for k in range(j, n):
                    first = [dg[j][l][k] + dg[k][l][j] - dg[l][j][k] for l in range(n)]
                    for i in range(n):
                        terms = [inverse[i, l]*first[l] for l in range(n) if inverse[i, l] != 0 and first[l] != 0]
                        value = self.simplifier(add(*terms)/2) if terms else _ZERO
                        table[i][j][k] = table[i][k][j] = value
            self._christoffel = table
        return self._christoffel

    def christoffel(self, i: int, j: int, k: int) -> Expr:
        r"""The Christoffel symbol of the second kind $\Gamma^i_{jk}$.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> r, phi = Symbol('r'), Symbol('phi')
        >>> polar = Metric([[1, 0], [0, r**2]], [r, phi])
        >>> polar.christoffel(0, 1, 1), polar.christoffel(1, 0, 1)
        (-r, 1/r)
        """
        return self._connection()[i][j][k]

    def christoffel_symbols(self) -> dict[tuple[int, int, int], Expr]:
        r"""The nonzero $\Gamma^i_{jk}$ with $j \le k$, by ``(i, j, k)``.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> r, phi = Symbol('r'), Symbol('phi')
        >>> Metric([[1, 0], [0, r**2]], [r, phi]).christoffel_symbols()
        {(0, 1, 1): -r, (1, 0, 1): 1/r}
        """
        n = self.dimension
        table = self._connection()
        return {(i, j, k): table[i][j][k] for i in range(n) for j in range(n) for k in range(j, n) if table[i][j][k] != 0}

    def christoffel_first_kind(self, i: int, j: int, k: int) -> Expr:
        r"""$\Gamma_{ijk} = \tfrac12 (\partial_j g_{ik} + \partial_k g_{ij} - \partial_i g_{jk})$."""
        x, g = self.coordinates, self.g
        return self.simplifier((g[i, k].diff(x[j]) + g[i, j].diff(x[k]) - g[j, k].diff(x[i]))/2)

    # -- curvature

    def _curvature(self) -> _Rank4:
        if self._riemann is None:
            n = self.dimension
            x = self.coordinates
            gamma = self._connection()
            table: _Rank4 = [[[[_ZERO]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(k + 1, n):
                            terms: list[Expr] = [gamma[i][l][j].diff(x[k]), -gamma[i][k][j].diff(x[l])]
                            for m in range(n):
                                if gamma[i][k][m] != 0 and gamma[m][l][j] != 0:
                                    terms.append(gamma[i][k][m]*gamma[m][l][j])
                                if gamma[i][l][m] != 0 and gamma[m][k][j] != 0:
                                    terms.append(-gamma[i][l][m]*gamma[m][k][j])
                            value = self.simplifier(add(*terms))
                            table[i][j][k][l] = value
                            table[i][j][l][k] = self.simplifier(-value) if value != 0 else _ZERO
            self._riemann = table
        return self._riemann

    def riemann(self, i: int, j: int, k: int, l: int) -> Expr:
        r"""The component $R^i{}_{jkl}$ of the Riemann tensor.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin
        >>> from sympy_extras.diffgeom import Metric
        >>> theta, phi = Symbol('theta'), Symbol('phi')
        >>> Metric([[1, 0], [0, sin(theta)**2]], [theta, phi]).riemann(0, 1, 0, 1)
        sin(theta)**2
        """
        return self._curvature()[i][j][k][l]

    def riemann_lower(self, i: int, j: int, k: int, l: int) -> Expr:
        r"""The covariant component $R_{ijkl} = g_{im} R^m{}_{jkl}$."""
        table = self._curvature()
        return self.simplifier(add(*[self.g[i, m]*table[m][j][k][l] for m in range(self.dimension)]))

    def riemann_components(self) -> dict[tuple[int, int, int, int], Expr]:
        r"""The nonzero $R^i{}_{jkl}$ with $k < l$, by ``(i, j, k, l)``."""
        n = self.dimension
        table = self._curvature()
        return {(i, j, k, l): table[i][j][k][l] for i in range(n) for j in range(n) for k in range(n)
                for l in range(k + 1, n) if table[i][j][k][l] != 0}

    def ricci(self) -> ExprMatrix:
        r"""The Ricci tensor $R_{jl} = R^i{}_{jil}$.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> x, y = Symbol('x'), Symbol('y')
        >>> Metric([[1/y**2, 0], [0, 1/y**2]], [x, y]).ricci()          # the hyperbolic plane
        ExprMatrix([[-1/y**2, 0], [0, -1/y**2]])
        """
        if self._ricci is None:
            n = self.dimension
            table = self._curvature()
            self._ricci = ExprMatrix(n, n, [self.simplifier(add(*[table[i][j][i][l] for i in range(n)]))
                                            for j in range(n) for l in range(n)])
        return self._ricci

    def ricci_scalar(self) -> Expr:
        r"""The scalar curvature $R = g^{jl} R_{jl}$ (twice the Gaussian
        curvature of a surface).

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> x, y = Symbol('x'), Symbol('y')
        >>> Metric([[1/y**2, 0], [0, 1/y**2]], [x, y]).ricci_scalar()
        -2
        """
        n = self.dimension
        ricci, inverse = self.ricci(), self.inverse
        return self.simplifier(add(*[inverse[j, l]*ricci[j, l] for j in range(n) for l in range(n)]))

    def einstein(self) -> ExprMatrix:
        r"""The Einstein tensor $G_{jl} = R_{jl} - \tfrac12 R\, g_{jl}$."""
        n = self.dimension
        ricci, scalar = self.ricci(), self.ricci_scalar()
        return ExprMatrix(n, n, [self.simplifier(ricci[j, l] - scalar*self.g[j, l]/2) for j in range(n) for l in range(n)])

    def _lowered(self) -> _Rank4:
        n = self.dimension
        table = self._curvature()
        return [[[[self.simplifier(add(*[self.g[i, m]*table[m][j][k][l] for m in range(n)])) for l in range(n)]
                  for k in range(n)] for j in range(n)] for i in range(n)]

    def kretschmann(self) -> Expr:
        r"""The Kretschmann scalar $R_{ijkl} R^{ijkl}$."""
        n = self.dimension
        inverse = self.inverse
        lowered = self._lowered()
        # raise the four indices one at a time
        raised = lowered
        for position in range(4):
            updated: _Rank4 = [[[[_ZERO]*n for _ in range(n)] for _ in range(n)] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            index = [i, j, k, l]
                            terms: list[Expr] = []
                            for m in range(n):
                                factor = inverse[index[position], m]
                                if factor == 0:
                                    continue
                                source = list(index)
                                source[position] = m
                                component = raised[source[0]][source[1]][source[2]][source[3]]
                                if component != 0:
                                    terms.append(factor*component)
                            if terms:
                                updated[i][j][k][l] = add(*terms)
            raised = updated
        return self.simplifier(add(*[lowered[i][j][k][l]*raised[i][j][k][l] for i in range(n) for j in range(n)
                                     for k in range(n) for l in range(n) if lowered[i][j][k][l] != 0]))

    def weyl(self, i: int, j: int, k: int, l: int) -> Expr:
        r"""The component $C_{ijkl}$ of the Weyl (conformal curvature)
        tensor; zero in dimension three, ``ValueError`` below."""
        n = self.dimension
        if n < 3:
            raise ValueError('the Weyl tensor is defined from the dimension 3 on')
        g, ricci, scalar = self.g, self.ricci(), self.ricci_scalar()
        value = (self.riemann_lower(i, j, k, l)
                 - (g[i, k]*ricci[j, l] - g[i, l]*ricci[j, k] - g[j, k]*ricci[i, l] + g[j, l]*ricci[i, k])/(n - 2)
                 + scalar*(g[i, k]*g[j, l] - g[i, l]*g[j, k])/((n - 1)*(n - 2)))
        return self.simplifier(value)

    def is_flat(self) -> bool:
        """Whether the Riemann tensor vanishes (after simplification)."""
        return not self.riemann_components()

    # -- geodesics and derivatives

    def geodesic_equations(self, parameter: Symbol) -> list[Expr]:
        r"""The left-hand sides of the geodesic equations

        .. math:: \ddot x^i + \Gamma^i_{jk}\, \dot x^j \dot x^k = 0

        with each coordinate replaced by the function of ``parameter`` of
        the same name.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> r, phi, s = Symbol('r'), Symbol('phi'), Symbol('s')
        >>> Metric([[1, 0], [0, r**2]], [r, phi]).geodesic_equations(s)
        [-r(s)*Derivative(phi(s), s)**2 + Derivative(r(s), (s, 2)), Derivative(phi(s), (s, 2)) + 2*Derivative(phi(s), s)*Derivative(r(s), s)/r(s)]
        """
        n = self.dimension
        gamma = self._connection()
        paths: list[Expr] = [Function(coordinate.name)(parameter) for coordinate in self.coordinates]
        rule: dict[Basic, Basic] = {coordinate: path for coordinate, path in zip(self.coordinates, paths)}
        velocity = [path.diff(parameter) for path in paths]
        equations: list[Expr] = []
        for i in range(n):
            terms: list[Expr] = [paths[i].diff(parameter, 2)]
            for j in range(n):
                for k in range(j, n):
                    if gamma[i][j][k] != 0:
                        weight = 1 if j == k else 2
                        terms.append(weight*gamma[i][j][k].xreplace(rule)*velocity[j]*velocity[k])
            equations.append(add(*terms))
        return equations

    def covariant_derivative(self, vector: Sequence[ExprLike]) -> ExprMatrix:
        r"""The matrix $\nabla_j V^i = \partial_j V^i + \Gamma^i_{jk} V^k$
        (row $i$, column $j$) of the covariant derivative of a vector
        field given by its contravariant components."""
        V = self._vector(vector)
        n = self.dimension
        x = self.coordinates
        gamma = self._connection()
        return ExprMatrix(n, n, [self.simplifier(V[i].diff(x[j]) + add(*[gamma[i][j][k]*V[k] for k in range(n)]))
                                 for i in range(n) for j in range(n)])

    def divergence(self, vector: Sequence[ExprLike]) -> Expr:
        r"""The divergence $\nabla_i V^i$ of a vector field."""
        derivative = self.covariant_derivative(vector)
        return self.simplifier(add(*[derivative[i, i] for i in range(self.dimension)]))

    def gradient(self, f: ExprLike) -> list[Expr]:
        r"""The contravariant components $g^{ij} \partial_j f$ of the
        gradient of a scalar."""
        scalar = to_expr(f)
        return self.raise_index([scalar.diff(x) for x in self.coordinates])

    def laplacian(self, f: ExprLike) -> Expr:
        r"""The Laplace–Beltrami operator $g^{ij} (\partial_i \partial_j
        f - \Gamma^k_{ij} \partial_k f)$ on a scalar.

        >>> from sympy_extras.core import Function, Symbol
        >>> from sympy_extras.diffgeom import Metric
        >>> r, phi = Symbol('r'), Symbol('phi')
        >>> f = Function('f')(r)
        >>> Metric([[1, 0], [0, r**2]], [r, phi]).laplacian(f)
        (r*Derivative(f(r), (r, 2)) + Derivative(f(r), r))/r
        """
        scalar = to_expr(f)
        n = self.dimension
        x = self.coordinates
        gamma, inverse = self._connection(), self.inverse
        first = [scalar.diff(x[k]) for k in range(n)]
        terms: list[Expr] = []
        for i in range(n):
            for j in range(n):
                if inverse[i, j] == 0:
                    continue
                second = first[j].diff(x[i]) - add(*[gamma[k][i][j]*first[k] for k in range(n)])
                terms.append(inverse[i, j]*second)
        return self.simplifier(add(*terms))

    def killing_equations(self, vector: Sequence[ExprLike]) -> ExprMatrix:
        r"""The symmetric matrix $\nabla_i \xi_j + \nabla_j \xi_i$ for the
        vector field with the contravariant components ``vector``: zero
        exactly for a Killing vector (an infinitesimal isometry)."""
        n = self.dimension
        x = self.coordinates
        gamma = self._connection()
        xi = self.lower(vector)
        covariant = [[xi[j].diff(x[i]) - add(*[gamma[k][i][j]*xi[k] for k in range(n)]) for j in range(n)] for i in range(n)]
        return ExprMatrix(n, n, [self.simplifier(covariant[i][j] + covariant[j][i]) for i in range(n) for j in range(n)])

    def is_killing(self, vector: Sequence[ExprLike]) -> bool:
        """Whether the vector field is a Killing vector.

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin
        >>> from sympy_extras.diffgeom import Metric
        >>> theta, phi = Symbol('theta'), Symbol('phi')
        >>> sphere = Metric([[1, 0], [0, sin(theta)**2]], [theta, phi])
        >>> sphere.is_killing([0, 1]), sphere.is_killing([1, 0])
        (True, False)
        """
        equations = self.killing_equations(vector)
        return all(e == 0 for e in equations.entries)


