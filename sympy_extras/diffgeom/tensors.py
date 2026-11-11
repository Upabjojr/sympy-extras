r"""Tensors by components on a metric: index positions, raising and
lowering, contraction, products and the covariant derivative.

A :class:`Tensor` holds the components of a tensor field in the
coordinates of a :class:`~sympy_extras.diffgeom.metric.Metric`, with the
position of each index: ``'u'`` (contravariant, up) or ``'d'``
(covariant, down). The operations are those of the Ricci calculus [1]_:

* raising and lowering with the metric, $T^i{}_j = g^{ik} T_{kj}$;
* contraction of an upper with a lower index, and the product $S
  \otimes T$;
* the covariant derivative, which adds a lower index (the last one):

  .. math:: \nabla_c T^{a\dots}{}_{b\dots} = \partial_c T^{a\dots}{}_{b\dots}
            + \Gamma^a_{cm} T^{m\dots}{}_{b\dots} + \dots
            - \Gamma^m_{cb} T^{a\dots}{}_{m\dots} - \dots

* symmetric and antisymmetric parts in two indices, the trace.

Every component is simplified by the simplifier of the metric
(:func:`sympy_extras.simplify.normalform.normal_form` by default), so
that a tensor which vanishes has zero components: the metric is
covariantly constant, $\nabla_c g_{ab} = 0$, and the Einstein tensor has
no divergence, $\nabla^a G_{ab} = 0$ [2]_.

:func:`metric_tensor`, :func:`riemann_tensor`, :func:`ricci_tensor` and
:func:`einstein_tensor` give the tensors of a metric.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sin
>>> from sympy_extras.diffgeom import Metric
>>> from sympy_extras.diffgeom.tensors import Tensor, metric_tensor, riemann_tensor, einstein_tensor
>>> theta, phi = Symbol('theta'), Symbol('phi')
>>> sphere = Metric([[1, 0], [0, sin(theta)**2]], [theta, phi])
>>> g = metric_tensor(sphere)
>>> g.covariant_derivative().is_zero, g.raise_index(0).components
(True, (1, 0, 0, 1))
>>> R = riemann_tensor(sphere)
>>> R.positions, R[0, 1, 0, 1], R.contract(0, 2).components
('uddd', sin(theta)**2, (1, 0, 0, sin(theta)**2))
>>> V = Tensor(sphere, [0, 1], 'u')                       # the Killing field d/dphi
>>> V.lower_index(0).covariant_derivative().symmetrize(0, 1).is_zero
True

References
==========

.. [1] J. A. Schouten, *Ricci-Calculus*, 2nd ed., Springer, 1954,
       Chapters I and III.
.. [2] C. W. Misner, K. S. Thorne, J. A. Wheeler, *Gravitation*, Freeman,
       1973, Chapters 3 (index gymnastics), 8 and 13 (covariant
       derivatives, the Bianchi identities).
"""
from __future__ import annotations

from itertools import product
from typing import Final, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Integer

from .metric import Metric

__all__ = ['Tensor', 'metric_tensor', 'riemann_tensor', 'ricci_tensor', 'einstein_tensor']

_Index = tuple[int, ...]


class Tensor:
    """The tensor on ``metric`` with the given ``components`` (in
    row-major order: the last index varies fastest; ``n**rank`` of them)
    and index ``positions``, a string of ``'u'`` and ``'d'`` (empty for a
    scalar)."""

    __slots__ = ('metric', 'components', 'positions', 'rank')

    def __init__(self, metric: Metric, components: Sequence[ExprLike], positions: str) -> None:
        if any(p not in 'ud' for p in positions):
            raise ValueError("the positions of the indices are 'u' or 'd': %r" % (positions,))
        rank = len(positions)
        if len(components) != metric.dimension**rank:
            raise ValueError('%d components for a tensor of rank %d in dimension %d'
                             % (len(components), rank, metric.dimension))
        self.metric: Final[Metric] = metric
        self.components: Final[tuple[Expr, ...]] = tuple(to_expr(c) for c in components)
        self.positions: Final[str] = positions
        self.rank: Final[int] = rank

    # -- access

    def _offset(self, indices: Sequence[int]) -> int:
        n = self.metric.dimension
        if len(indices) != self.rank or any(not 0 <= i < n for i in indices):
            raise IndexError('the indices %s of a tensor of rank %d in dimension %d' % (tuple(indices), self.rank, n))
        offset = 0
        for i in indices:
            offset = offset*n + i
        return offset

    def __getitem__(self, indices: Sequence[int]) -> Expr:
        return self.components[self._offset(indices)]

    def _indices(self) -> list[_Index]:
        return list(product(range(self.metric.dimension), repeat=self.rank))

    def _build(self, positions: str, values: dict[_Index, Expr]) -> Tensor:
        n = self.metric.dimension
        zero: Expr = Integer(0)
        simplify = self.metric.simplifier
        components = [simplify(values[index]) if index in values else zero
                      for index in product(range(n), repeat=len(positions))]
        return Tensor(self.metric, components, positions)

    @property
    def is_zero(self) -> bool:
        """Every component is zero (after simplification)."""
        return all(self.metric.simplifier(c) == 0 for c in self.components)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Tensor) and other.metric is self.metric and other.positions == self.positions
                and other.components == self.components)

    def __hash__(self) -> int:
        return hash((self.positions, self.components))

    def __repr__(self) -> str:
        return 'Tensor(%r, %s)' % (self.positions, list(self.components))

    # -- algebra

    def _same_kind(self, other: Tensor) -> None:
        if other.metric is not self.metric or other.positions != self.positions:
            raise ValueError('tensors of different kinds: %r and %r' % (self.positions, other.positions))

    def __add__(self, other: Tensor) -> Tensor:
        self._same_kind(other)
        simplify = self.metric.simplifier
        return Tensor(self.metric, [simplify(a + b) for a, b in zip(self.components, other.components)], self.positions)

    def __sub__(self, other: Tensor) -> Tensor:
        self._same_kind(other)
        simplify = self.metric.simplifier
        return Tensor(self.metric, [simplify(a - b) for a, b in zip(self.components, other.components)], self.positions)

    def __neg__(self) -> Tensor:
        return Tensor(self.metric, [-c for c in self.components], self.positions)

    def scale(self, factor: ExprLike) -> Tensor:
        """The tensor times a scalar."""
        c = to_expr(factor)
        simplify = self.metric.simplifier
        return Tensor(self.metric, [simplify(c*a) for a in self.components], self.positions)

    def __mul__(self, other: Tensor) -> Tensor:
        """The tensor product: the indices of ``other`` follow those of
        ``self``."""
        if other.metric is not self.metric:
            raise ValueError('tensors on different metrics')
        simplify = self.metric.simplifier
        components = [simplify(a*b) if a != 0 and b != 0 else Integer(0) for a in self.components for b in other.components]
        return Tensor(self.metric, components, self.positions + other.positions)

    def transpose(self, i: int, j: int) -> Tensor:
        """The tensor with the indices ``i`` and ``j`` exchanged."""
        positions = list(self.positions)
        positions[i], positions[j] = positions[j], positions[i]
        values: dict[_Index, Expr] = {}
        for index in self._indices():
            swapped = list(index)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            values[index] = self[swapped]
        return self._build(''.join(positions), values)

    def symmetrize(self, i: int, j: int) -> Tensor:
        """The symmetric part in two indices of the same position."""
        if self.positions[i] != self.positions[j]:
            raise ValueError('the indices %d and %d are in different positions' % (i, j))
        return (self + self.transpose(i, j)).scale(Integer(1)/2)

    def antisymmetrize(self, i: int, j: int) -> Tensor:
        """The antisymmetric part in two indices of the same position."""
        if self.positions[i] != self.positions[j]:
            raise ValueError('the indices %d and %d are in different positions' % (i, j))
        return (self - self.transpose(i, j)).scale(Integer(1)/2)

    # -- the metric

    def _moved(self, k: int, up: bool) -> Tensor:
        n = self.metric.dimension
        if not 0 <= k < self.rank:
            raise IndexError('no index %d in a tensor of rank %d' % (k, self.rank))
        if (self.positions[k] == 'u') == up:
            return self
        matrix = self.metric.inverse if up else self.metric.g
        values: dict[_Index, Expr] = {}
        for index in self._indices():
            terms: list[Expr] = []
            for m in range(n):
                factor = matrix[index[k], m]
                if factor == 0:
                    continue
                source = list(index)
                source[k] = m
                component = self[source]
                if component != 0:
                    terms.append(factor*component)
            if terms:
                values[index] = add(*terms)
        positions = self.positions[:k] + ('u' if up else 'd') + self.positions[k + 1:]
        return self._build(positions, values)

    def raise_index(self, k: int) -> Tensor:
        """The tensor with the index ``k`` up."""
        return self._moved(k, True)

    def lower_index(self, k: int) -> Tensor:
        """The tensor with the index ``k`` down."""
        return self._moved(k, False)

    def with_positions(self, positions: str) -> Tensor:
        """The same tensor with its indices in the given positions."""
        if len(positions) != self.rank:
            raise ValueError('%d positions for a tensor of rank %d' % (len(positions), self.rank))
        result = self
        for k, position in enumerate(positions):
            result = result._moved(k, position == 'u')
        return result

    def contract(self, i: int, j: int) -> Tensor:
        """The contraction of an upper with a lower index; the other
        indices keep their order."""
        if i == j or {self.positions[i], self.positions[j]} != {'u', 'd'}:
            raise ValueError('a contraction takes one upper and one lower index')
        n = self.metric.dimension
        kept = [k for k in range(self.rank) if k not in (i, j)]
        values: dict[_Index, Expr] = {}
        for index in product(range(n), repeat=len(kept)):
            terms: list[Expr] = []
            for m in range(n):
                full = [0]*self.rank
                for position, value in zip(kept, index):
                    full[position] = value
                full[i] = full[j] = m
                component = self[full]
                if component != 0:
                    terms.append(component)
            if terms:
                values[index] = add(*terms)
        return self._build(''.join(self.positions[k] for k in kept), values)

    def trace(self) -> Expr:
        """The trace of a tensor of rank two (an index is raised when
        both are in the same position)."""
        if self.rank != 2:
            raise ValueError('the trace of a tensor of rank two')
        mixed = self if self.positions in ('ud', 'du') else self.raise_index(0) if self.positions == 'dd' else self.lower_index(0)
        return mixed.contract(0, 1).components[0]

    def scalar(self) -> Expr:
        """The value of a tensor of rank zero."""
        if self.rank:
            raise ValueError('not a scalar: rank %d' % self.rank)
        return self.components[0]

    # -- derivatives

    def covariant_derivative(self) -> Tensor:
        """The covariant derivative: a tensor with one more lower index,
        the last, which is the index of the derivative."""
        n = self.metric.dimension
        x = self.metric.coordinates
        values: dict[_Index, Expr] = {}
        for index in self._indices():
            for c in range(n):
                terms: list[Expr] = [self[index].diff(x[c])]
                for k, position in enumerate(self.positions):
                    for m in range(n):
                        source = list(index)
                        source[k] = m
                        component = self[source]
                        if component == 0:
                            continue
                        if position == 'u':
                            gamma = self.metric.christoffel(index[k], c, m)
                            if gamma != 0:
                                terms.append(gamma*component)
                        else:
                            gamma = self.metric.christoffel(m, c, index[k])
                            if gamma != 0:
                                terms.append(-gamma*component)
                values[index + (c,)] = add(*terms)
        return self._build(self.positions + 'd', values)

    def divergence(self, k: int = 0) -> Tensor:
        """The divergence on the index ``k``: the covariant derivative
        contracted with it (the index is raised first when it is down)."""
        raised = self.raise_index(k)
        return raised.covariant_derivative().contract(k, self.rank)


def metric_tensor(metric: Metric) -> Tensor:
    """The metric ``g_ab`` as a tensor (positions ``'dd'``)."""
    n = metric.dimension
    return Tensor(metric, [metric.g[i, j] for i in range(n) for j in range(n)], 'dd')


def riemann_tensor(metric: Metric) -> Tensor:
    """The Riemann tensor ``R^a_bcd`` (positions ``'uddd'``)."""
    n = metric.dimension
    return Tensor(metric, [metric.riemann(i, j, k, l) for i in range(n) for j in range(n) for k in range(n)
                           for l in range(n)], 'uddd')


def ricci_tensor(metric: Metric) -> Tensor:
    """The Ricci tensor ``R_ab`` (positions ``'dd'``)."""
    n = metric.dimension
    ricci = metric.ricci()
    return Tensor(metric, [ricci[i, j] for i in range(n) for j in range(n)], 'dd')


def einstein_tensor(metric: Metric) -> Tensor:
    """The Einstein tensor ``G_ab`` (positions ``'dd'``)."""
    n = metric.dimension
    einstein = metric.einstein()
    return Tensor(metric, [einstein[i, j] for i in range(n) for j in range(n)], 'dd')
