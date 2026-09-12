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

Examples
========

>>> from sympy import symbols, Eq, S
>>> from sympy_extras.integrals.axisymmetric import axisymmetric_integral
>>> x, y, z = symbols('x y z')
>>> axisymmetric_integral(1, (x**2 + y**2 + z**2 < 1) & ((x - 1)**2 + y**2 + z**2 < 1), [x, y, z])
5*pi/12
>>> axisymmetric_integral(1, Eq(x**2 + y**2 + z**2, 1) & (z > S(1)/2), [x, y, z], measure='hausdorff')
pi

References
==========

.. [Apostol] T. M. Apostol, *Calculus*, vol. II, 2nd ed., Wiley, 1969,
   section 11.28 (solids of revolution and cylindrical coordinates).
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.numbers import pi
from sympy.core.relational import Eq, Relational
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import expand
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import And, Boolean
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy_extras.settings import settings
from .conditions import numerically_equal

__all__ = ['axisymmetric_integral', 'rotation_group']


def _invariant(e: Expr, group: Sequence[Symbol]) -> bool:
    """Whether ``e`` depends on the variables of ``group`` through the sum
    of their squares only: annihilated by every infinitesimal rotation
    ``u d/dv - v d/du``."""
    for u, v in combinations(group, 2):
        generator = as_expr(u * e.diff(v) - v * e.diff(u))
        if generator == 0:
            continue
        try:
            reduced = attempt(lambda: as_expr(cancel(generator)), settings.timeout)
        except PolynomialError:
            reduced = None
        if reduced is None:
            return False
        if reduced != 0 and not numerically_equal(reduced, as_expr(0)):
            return False
    return True


def rotation_group(f: ExprLike, formula: Boolean, names: Sequence[Symbol]) -> Optional[tuple[Symbol, ...]]:
    """The largest group of at least two variables through whose sum of
    squares alone the integrand and every relation of the condition
    depend on them; ``None`` when there is none.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.axisymmetric import rotation_group
    >>> x, y, z = symbols('x y z')
    >>> rotation_group(z, (x**2 + y**2 + z**2 < 1) & (z > x**2 + y**2), [x, y, z])
    (x, y)
    >>> rotation_group(1, (x**2 + y**2 + z**2 < 1) & (x > 0), [x, y, z])
    (y, z)
    >>> rotation_group(x*y, x**2 + y**2 + z**2 < 1, [x, y, z]) is None
    True
    """
    expressions: list[Expr] = [as_expr(f)]
    for atom in formula.atoms(Relational):
        expressions.append(as_expr(atom.lhs) - as_expr(atom.rhs))
    for size in range(len(names), 1, -1):
        for group in combinations(names, size):
            if all(_invariant(e, group) for e in expressions):
                return tuple(group)
    return None


def _sphere_area(k: int) -> Expr:
    """The area of the unit sphere ``S**(k-1)``, ``2 pi**(k/2)/gamma(k/2)``."""
    return as_expr(2 * pi**(as_expr(k) / 2) / gamma(as_expr(k) / 2))


def _normal(p: Expr, names: Sequence[Symbol]) -> Optional[list[Expr]]:
    """The coefficients of a polynomial of degree one in ``names``, ``None``
    otherwise."""
    if not p.has(*names):
        return None
    try:
        poly = Poly(p, *names)
    except PolynomialError:
        return None
    if poly.total_degree() != 1:
        return None
    coefficients = [as_expr(poly.coeff_monomial(v)) for v in names]
    return None if any(c.has(*names) for c in coefficients) else coefficients


def aligned_frame(f: Expr, formula: Boolean, names: Sequence[Symbol]) -> Optional[_Frame]:
    """The integrand, the condition and the variables in the frame whose
    first axis is the common normal ``a`` of the relations of degree one,
    when every other relation is invariant under all rotations (a ball, a
    sphere), scaled by ``|a|``: ``x = sum(e_j Y_j)/|a|`` with ``e_1 = a/|a|``,
    a similarity of ratio ``1/|a|``, so that ``a.x = Y_1`` and
    ``|x|**2 = sum(Y_j**2)/|a|**2`` keep rational coefficients and the
    region is a body of revolution about the ``Y_1`` axis. The integral is
    ``|a|**(-d)`` times the integral in the new frame, ``d`` the dimension
    of the measure; ``None`` when there is no such frame.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.axisymmetric import aligned_frame
    >>> x, y, z = symbols('x y z')
    >>> frame = aligned_frame(1, (x**2 + y**2 + z**2 < 1) & (x + y + z > 1), [x, y, z])
    >>> frame.condition, frame.ratio
    ((_Y1 - 1 > 0) & (_Y1**2/3 + _Y2**2/3 + _Y3**2/3 - 1 < 0), sqrt(3))
    """
    normal: Optional[list[Expr]] = None
    for atom in formula.atoms(Relational):
        p = as_expr(atom.lhs) - as_expr(atom.rhs)
        found = _normal(p, names)
        if found is not None:
            if normal is None:
                normal = found
            elif Matrix([normal, found]).rank() != 1:
                return None
            continue
        if not _invariant(p, names):
            return None
    if normal is None:
        return None
    n = len(names)
    a = Matrix(normal)
    length = as_expr(sqrt(as_expr((a.T * a)[0])))
    basis = Matrix.orthogonalize(a, *[Matrix.eye(n)[:, j] for j in range(n)], normalize=True, rankcheck=False)
    if len(basis) != n:
        return None
    ys: list[Symbol] = [Dummy('Y%d' % j) for j in range(1, n + 1)]
    image = Matrix.zeros(n, 1)
    for e, yj in zip(basis, ys):
        image = image + e * yj / length
    replacement = {v: as_expr(cancel(image[i])) for i, v in enumerate(names)}
    new_atoms: dict[Boolean, Boolean] = {}
    for atom in formula.atoms(Relational):
        difference = as_expr(atom.lhs) - as_expr(atom.rhs)
        rewritten = as_expr(cancel(expand(difference.xreplace(replacement))))
        new_atoms[as_boolean(atom)] = as_boolean(atom.func(rewritten, 0))
    condition = as_boolean(formula.xreplace(new_atoms))
    integrand = as_expr(cancel(expand(as_expr(f).xreplace(replacement))))
    return _Frame(integrand, condition, ys, length)


class _Frame:
    """The transformed problem of :func:`aligned_frame`: the similarity
    has ratio ``1/ratio``."""

    def __init__(self, integrand: Expr, condition: Boolean, variables: list[Symbol], ratio: Expr) -> None:
        self.integrand = integrand
        self.condition = condition
        self.variables = variables
        self.ratio = ratio


def axisymmetric_integral(f: Expr, formula: Boolean, names: Sequence[Symbol],
                          assumptions: Sequence[Boolean] = (), measure: str = 'lebesgue') -> Optional[Expr]:
    """The integral of ``f`` over the region ``formula`` of the variables
    ``names`` through the profile of its rotational symmetry, see the
    module documentation; ``None`` when there is no group of variables to
    rotate or the profile integral is not evaluated. ``measure`` is
    ``'lebesgue'`` or ``'hausdorff'`` as in
    :func:`~sympy_extras.integrals.regions.integrate_by_ranges`.

    >>> from sympy import symbols, sqrt, Rational
    >>> from sympy_extras.integrals.axisymmetric import axisymmetric_integral
    >>> x, y, z = symbols('x y z')
    >>> axisymmetric_integral(1, (x**2 + y**2 + z**2 < 1) & (x**2 + y**2 < Rational(1, 4)), [x, y, z])
    pi*(16 - 9*sqrt(3))/12 + sqrt(3)*pi/4
    """
    from .regions import IntegralByRanges, integrate_by_ranges
    f_ = as_expr(f)
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
        return as_expr(value / frame.ratio**dimension)
    k = len(group)
    # a plain dummy: with positive=True the relation rho > 0 evaluates to
    # True and vanishes, the profile is even in rho, and the odd weight
    # rho**(k-1) integrates to 0 over both signs
    rho = Dummy('rho')
    replacement: dict[Expr, Expr] = {group[0]: rho}
    for v in group[1:]:
        replacement[v] = as_expr(0)
    profile = as_boolean(formula.xreplace(replacement))
    rest = [v for v in names if v not in group]
    weight = as_expr(_sphere_area(k) * rho**(k - 1))
    reduced = as_expr(f_.xreplace(replacement) * weight)
    condition = as_boolean(And(profile, rho > 0))
    outer = list(assumptions)
    # the radius last first: the profile is solved for it (a cap is
    # rho = sqrt(1 - z**2)), and the other order is the fallback
    for order in (rest + [rho], [rho] + rest):
        value = attempt(lambda: integrate_by_ranges(reduced, condition, order, outer, measure=measure),
                        settings.timeout)
        if value is not None and not value.has(IntegralByRanges) and not free_symbols(value) & {rho}:
            return value
    return None
