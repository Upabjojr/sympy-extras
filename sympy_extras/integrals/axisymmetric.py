r"""Region integrals with a rotational symmetry.

When a group of `k \ge 2` integration variables `v_1, \ldots, v_k` enters
every relation of the condition and the integrand only through
`v_1^2 + \cdots + v_k^2`, the region is a body of revolution about the
subspace of the other variables, and the integral over it reduces to an
integral over its *profile* in the half-space `\rho > 0`:

.. math::

    \int_{\Omega} f \, dV
    = \frac{2\pi^{k/2}}{\Gamma(k/2)} \int_{\Omega'} f(\rho, u)\, \rho^{k-1}
      \, d\rho \, du,
    \qquad \Omega' = \{(\rho, u) : (\rho, 0, \ldots, 0, u) \in \Omega,\ \rho > 0\},

`2\pi^{k/2}/\Gamma(k/2)` being the area of the unit sphere `S^{k-1}`:
`2\pi` for a body of revolution about an axis in space, `4\pi` for the
rotation of three variables in four. The same factor multiplies the
measure of the profile for a hypersurface of revolution given by an
equation among the relations, so areas of caps, of paraboloids and
lengths of circles of intersection come out of the same reduction. The
profile has one variable fewer for each group, is described by the same
polynomials with the group replaced by `(\rho, 0, \ldots, 0)`, and goes
back to :func:`~sympy_extras.integrals.regions.integrate_by_ranges`,
whose decomposition then works in one dimension less; two spheres, a
sphere cut by a paraboloid or a cylinder and their higher-dimensional
versions become plane regions bounded by conics.

Rotation invariance of a polynomial `P` in the group is decided exactly:
`P` is a function of `v_1^2 + \cdots + v_k^2` if and only if it is
annihilated by every infinitesimal rotation,
`v_i \partial P/\partial v_j - v_j \partial P/\partial v_i = 0`.
The largest invariant group is used, the pairs first when there are
several, and the reduced integral may be reduced again (two groups in
four variables).

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native ``cancel``, ``expand``, typed polynomials and
numerical comparison of :mod:`.conditions`. The profile integral is
computed by :func:`~sympy_extras.integrals.regions.integrate_by_ranges`,
which is still written on SymPy, through
:func:`~sympy_extras.integrals.axisymmetric_sympy.integrate_by_ranges_of_core`:
SymPy is loaded on that path only. The orthonormal frame of
:func:`aligned_frame` is the Gram–Schmidt process of SymPy's
``Matrix.orthogonalize``, ported. The functions on SymPy objects are in
:mod:`.axisymmetric_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, Integer, Rational, eq
>>> from sympy_extras.integrals.axisymmetric import axisymmetric_integral
>>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
>>> axisymmetric_integral(Integer(1), (x**2 + y**2 + z**2 < 1) & ((x - 1)**2 + y**2 + z**2 < 1), [x, y, z])
5*pi/12
>>> axisymmetric_integral(Integer(1), eq(x**2 + y**2 + z**2, 1) & (z > Rational(1, 2)), [x, y, z],
...                       measure='hausdorff')
pi

References
==========

.. [Apostol] T. M. Apostol, *Calculus*, vol. II, 2nd ed., Wiley, 1969,
   section 11.28 (solids of revolution and cylindrical coordinates).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/matrices/decompositions.py (_QRdecomposition_optional),
# sympy/matrices/subspaces.py (_orthogonalize) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on lists of core expressions (_orthogonalize only); the
# rest of the module is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import random
from itertools import combinations
from typing import Final, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import expand, expand_mul
from sympy_extras.core.logic import Boolean, BooleanTrue, and_, to_boolean
from sympy_extras.core.numbers import Integer, Rational, pi
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import Eq, Relational
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import abs_, conjugate
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.polys.polytools import cancel
from sympy_extras.settings import settings

from .conditions import numerically_equal

__all__ = ['axisymmetric_integral', 'rotation_group', 'rotation_invariant', 'aligned_frame', 'Frame']

_COUNTER: Final = itertools.count()


def _fresh(name: str) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so the radius of a group and the radius of the group
    of the reduced integral would be taken for one symbol."""
    return Dummy('%s%d' % (name, next(_COUNTER)))


def _evaluated(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e.xreplace(rule)`` with the rebuilt nodes evaluated."""
    return as_expr(e._xreplace_evaluated(rule))


def _relations(formula: Boolean) -> list[Relational]:
    """The relations of ``formula``, in a fixed order."""
    found = [atom for atom in formula.atoms(Relational) if isinstance(atom, Relational)]
    return sorted(found, key=lambda r: r.sort_key())


def rotation_invariant(e: Expr, group: Sequence[Symbol], samples: int = 3) -> bool:
    """Whether ``e`` takes the same value at random rational points and at
    their images under rotations in the coordinate planes of ``group``,
    the rotations exact (``cos, sin = (1 - t**2)/(1 + t**2), 2*t/(1 +
    t**2)`` with ``t`` rational: the angles 37, 90, 127 and 143 degrees,
    so that a quadrant indicator cannot keep every sample in its
    quadrant). A finite test: the derivative test of
    :func:`_invariant` is blind to a function constant off a set of
    measure zero (``Heaviside(x)``, ``Piecewise((1, x*y > 0), (0,
    True))``), whose derivative vanishes at every sample.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import heaviside
    >>> from sympy_extras.integrals.axisymmetric import rotation_invariant
    >>> x, y = Symbol('x'), Symbol('y')
    >>> rotation_invariant(x**2 + y**2, [x, y]), rotation_invariant(heaviside(x), [x, y])
    (True, False)
    """
    rng = random.Random(str((e, tuple(group))))
    others = sorted(e.free_symbols - set(group), key=lambda s: s.sort_key())
    for _ in range(samples):
        point: dict[Basic, Basic] = {v: Rational(rng.randint(-300, 300), 100) for v in list(group) + others}
        base = _evaluated(e, point).evalf(30)
        if not base.is_number:
            return False
        for u, v in combinations(group, 2):
            for t in (Rational(1, 3), S.One, Integer(2), Integer(3)):
                c, s = (1 - t**2) / (1 + t**2), 2 * t / (1 + t**2)
                pu, pv = as_expr(point[u]), as_expr(point[v])
                rotated = dict(point)
                rotated[u], rotated[v] = c * pu - s * pv, s * pu + c * pv
                value = _evaluated(e, rotated).evalf(30)
                if not value.is_number:
                    return False
                gap = abs_(base - value).evalf(30)
                scale = (1 + abs_(base) + abs_(value)).evalf(30)
                if not gap.is_comparable or not scale.is_comparable or not isinstance(gap <= scale / 10**12,
                                                                                       BooleanTrue):
                    return False
    return True


def _invariant(e: Expr, group: Sequence[Symbol]) -> bool:
    """Whether ``e`` depends on the variables of ``group`` through the sum
    of their squares only: annihilated by every infinitesimal rotation
    ``u d/dv - v d/du``, and unchanged by finite rotations at random
    points (the derivative test alone accepted ``Heaviside(x)`` and
    ``Piecewise((1, x*y > 0), (0, True))``, constant off a set of
    measure zero)."""
    if not rotation_invariant(e, group):
        return False
    for u, v in combinations(group, 2):
        generator = u * e.diff(v) - v * e.diff(u)
        if generator == 0:
            continue
        reduced = attempt(lambda: cancel(generator), settings.timeout, native=True)
        if reduced is None:
            return False
        if reduced != 0 and not numerically_equal(reduced, Integer(0)):
            return False
    return True


def rotation_group(f: ExprLike, formula: Boolean, names: Sequence[Symbol]) -> Optional[tuple[Symbol, ...]]:
    """The largest group of at least two variables through whose sum of
    squares alone the integrand and every relation of the condition
    depend on them; ``None`` when there is none.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.integrals.axisymmetric import rotation_group
    >>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    >>> rotation_group(z, (x**2 + y**2 + z**2 < 1) & (z > x**2 + y**2), [x, y, z])
    (x, y)
    >>> rotation_group(1, (x**2 + y**2 + z**2 < 1) & (x > 0), [x, y, z])
    (y, z)
    >>> rotation_group(x*y, x**2 + y**2 + z**2 < 1, [x, y, z]) is None
    True
    """
    expressions: list[Expr] = [to_expr(f)]
    for atom in _relations(formula):
        expressions.append(atom.lhs - atom.rhs)
    for size in range(len(names), 1, -1):
        for group in combinations(names, size):
            if all(_invariant(e, group) for e in expressions):
                return tuple(group)
    return None


def _sphere_area(k: int) -> Expr:
    """The area of the unit sphere ``S**(k-1)``, ``2 pi**(k/2)/gamma(k/2)``."""
    return 2 * pi**(Integer(k) / 2) / gamma(Integer(k) / 2)


def _normal(p: Expr, names: Sequence[Symbol]) -> Optional[list[Expr]]:
    """The coefficients of a polynomial of degree one in ``names``, ``None``
    otherwise (the coefficients may hold other symbols and radicals)."""
    if not p.has(*names):
        return None
    coefficients: list[Expr] = [Integer(0)] * len(names)
    for term in Add.make_args(expand(p)):
        coefficient, dependent = term.as_independent(*names, as_Add=False)
        if dependent == 1:
            continue
        positions = [i for i, v in enumerate(names) if v == dependent]
        if not positions:
            return None
        i = positions[0]
        coefficients[i] = coefficients[i] + coefficient
    if all(c == 0 for c in coefficients):
        return None
    return coefficients


def _proportional(u: list[Expr], v: list[Expr]) -> bool:
    """Whether the nonzero vectors ``u`` and ``v`` are proportional (the
    matrix of the two has rank one): every minor of order two cancels."""
    for i, j in combinations(range(len(u)), 2):
        if cancel(expand(u[i] * v[j] - u[j] * v[i])) != 0:
            return False
    return True


# PORTED FROM SYMPY: sympy/matrices/decompositions.py:_QRdecomposition_optional
def _orthogonalize(vectors: list[list[Expr]]) -> list[list[Expr]]:
    """SymPy's ``Matrix.orthogonalize(*vectors, normalize=True,
    rankcheck=False)``: the Gram–Schmidt process with the projections
    expanded, the vectors which become zero dropped, then the others
    normalised (the dot product conjugates its first argument, the norm
    is ``sqrt(sum(Abs(v_i)**2))``)."""
    def dot(u: list[Expr], v: list[Expr]) -> Expr:
        total: Expr = S.Zero
        for a, b in zip(u, v):
            total = total + conjugate(a) * b
        return total

    def is_zero(v: list[Expr]) -> Optional[bool]:
        verdicts = [a.is_zero for a in v]
        if all(t is True for t in verdicts):
            return True
        if any(t is False for t in verdicts):
            return False
        return None

    Q = [list(v) for v in vectors]
    ranked: list[int] = []
    for j in range(len(Q)):
        for i in range(j):
            if is_zero(Q[i]) is True:
                continue
            r = expand_mul(dot(Q[i], Q[j]) / dot(Q[i], Q[i]))
            Q[j] = [a - b * r for a, b in zip(Q[j], Q[i])]
        Q[j] = [expand_mul(a) for a in Q[j]]
        if is_zero(Q[j]) is not True:
            ranked.append(j)
    basis: list[list[Expr]] = []
    for j in ranked:
        total: Expr = S.Zero
        for a in Q[j]:
            total = total + abs_(a)**2
        norm = sqrt(total)
        basis.append([a / norm for a in Q[j]])
    return basis


class Frame:
    """The transformed problem of :func:`aligned_frame`: the integrand,
    the condition and the variables in the new frame; the similarity has
    ratio ``1/ratio``."""

    __slots__ = ('integrand', 'condition', 'variables', 'ratio')

    def __init__(self, integrand: Expr, condition: Boolean, variables: list[Symbol], ratio: Expr) -> None:
        self.integrand: Final[Expr] = integrand
        self.condition: Final[Boolean] = condition
        self.variables: Final[list[Symbol]] = variables
        self.ratio: Final[Expr] = ratio


def aligned_frame(f: ExprLike, formula: Boolean, names: Sequence[Symbol]) -> Optional[Frame]:
    """The integrand, the condition and the variables in the frame whose
    first axis is the common normal ``a`` of the relations of degree one,
    when every other relation is invariant under all rotations (a ball, a
    sphere), scaled by ``|a|``: ``x = sum(e_j Y_j)/|a|`` with ``e_1 = a/|a|``,
    a similarity of ratio ``1/|a|``, so that ``a.x = Y_1`` and
    ``|x|**2 = sum(Y_j**2)/|a|**2`` keep rational coefficients and the
    region is a body of revolution about the ``Y_1`` axis. The integral is
    ``|a|**(-d)`` times the integral in the new frame, ``d`` the dimension
    of the measure; ``None`` when there is no such frame.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.integrals.axisymmetric import aligned_frame
    >>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    >>> frame = aligned_frame(1, (x**2 + y**2 + z**2 < 1) & (x + y + z > 1), [x, y, z])
    >>> Y1, Y2, Y3 = frame.variables
    >>> frame.condition == ((Y1 - 1 > 0) & (Y1**2/3 + Y2**2/3 + Y3**2/3 - 1 < 0)), frame.ratio
    (True, sqrt(3))
    """
    normal: Optional[list[Expr]] = None
    relations = _relations(formula)
    for atom in relations:
        p = atom.lhs - atom.rhs
        found = _normal(p, names)
        if found is not None:
            if normal is None:
                normal = found
            elif not _proportional(normal, found):
                return None
            continue
        if not _invariant(p, names):
            return None
    if normal is None:
        return None
    n = len(names)
    total: Expr = S.Zero
    for c in normal:
        total = total + c * c
    length = sqrt(total)
    unit: list[list[Expr]] = [[Integer(1) if i == j else Integer(0) for i in range(n)] for j in range(n)]
    basis = _orthogonalize([normal] + unit)
    if len(basis) != n:
        return None
    ys: list[Symbol] = [_fresh('Y%d_' % j) for j in range(1, n + 1)]
    image: list[Expr] = [Integer(0)] * n
    for e, yj in zip(basis, ys):
        image = [image[i] + e[i] * yj / length for i in range(n)]
    replacement: dict[Basic, Basic] = {v: cancel(image[i]) for i, v in enumerate(names)}
    new_atoms: dict[Basic, Basic] = {}
    for atom in relations:
        difference = atom.lhs - atom.rhs
        rewritten = cancel(expand(_evaluated(difference, replacement)))
        new_atoms[atom] = atom._evaluated((rewritten, Integer(0)))
    condition = to_boolean(formula._xreplace_evaluated(new_atoms))
    integrand = cancel(expand(_evaluated(to_expr(f), replacement)))
    return Frame(integrand, condition, ys, length)


def axisymmetric_integral(f: ExprLike, formula: Boolean, names: Sequence[Symbol],
                          assumptions: Sequence[Boolean] = (), measure: str = 'lebesgue') -> Optional[Expr]:
    """The integral of ``f`` over the region ``formula`` of the variables
    ``names`` through the profile of its rotational symmetry, see the
    module documentation; ``None`` when there is no group of variables to
    rotate or the profile integral is not evaluated. ``measure`` is
    ``'lebesgue'`` or ``'hausdorff'`` as in
    :func:`~sympy_extras.integrals.regions.integrate_by_ranges`, which
    computes the profile integral on SymPy's side.

    >>> from sympy_extras.core import Symbol, Integer, Rational
    >>> from sympy_extras.integrals.axisymmetric import axisymmetric_integral
    >>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    >>> axisymmetric_integral(Integer(1), (x**2 + y**2 + z**2 < 1) & (x**2 + y**2 < Rational(1, 4)), [x, y, z])
    pi*(16 - 9*sqrt(3))/12 + sqrt(3)*pi/4
    """
    from .axisymmetric_sympy import integrate_by_ranges_of_core
    f_ = to_expr(f)
    group = rotation_group(f_, formula, names)
    if group is None:
        # a ball cut by tilted planes: the frame along their normal
        frame = aligned_frame(f_, formula, names)
        if frame is None or rotation_group(frame.integrand, frame.condition, frame.variables) is None:
            return None
        value = axisymmetric_integral(frame.integrand, frame.condition, frame.variables, assumptions, measure)
        if value is None:
            return None
        # a similarity of ratio 1/|a| scales a measure of dimension d by |a|**(-d)
        equations = len([atom for atom in formula.atoms(Relational) if isinstance(atom, Eq)])
        dimension = len(names) - (equations if measure == 'hausdorff' else 0)
        return value / frame.ratio**dimension
    k = len(group)
    # a plain dummy: with positive=True the relation rho > 0 evaluates to
    # True and vanishes, the profile is even in rho, and the odd weight
    # rho**(k-1) integrates to 0 over both signs
    rho = _fresh('rho')
    replacement: dict[Basic, Basic] = {group[0]: rho}
    for v in group[1:]:
        replacement[v] = Integer(0)
    profile = to_boolean(formula._xreplace_evaluated(replacement))
    rest = [v for v in names if v not in group]
    weight = _sphere_area(k) * rho**(k - 1)
    reduced = _evaluated(f_, replacement) * weight
    condition = and_(profile, rho > 0)
    # the radius last first: the profile is solved for it (a cap is
    # rho = sqrt(1 - z**2)), and the other order is the fallback
    for order in (rest + [rho], [rho] + rest):
        value = integrate_by_ranges_of_core(reduced, condition, order, list(assumptions), measure, settings.timeout)
        if value is not None and rho not in value.free_symbols:
            return value
    return None
