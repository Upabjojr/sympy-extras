"""Integrals over regions given by inequalities, through cylindrical
algebraic decomposition.

``IntegralByRanges(f, condition)`` is the integral of ``f`` over the
semialgebraic set `\\{(x_1, \\ldots, x_n) : \\text{condition}\\}`, the
counterpart of Mathematica's ``Integrate[f, {x, y} ∈ region]`` and of
``Integrate[Boole[condition] f, ...]``. The region is decomposed into
cylindrical cells (Collins' cylindrical algebraic decomposition,
:mod:`sympy_extras.polys.cad`), on each of which the integral is an
iterated integral with explicit bounds:

1. the polynomials of the atoms of the condition are decomposed with
   respect to the variable order *parameters first, then the integration
   variables in the given order*, so that every cell of the decomposition
   is cylindrical over a cell of the parameter space;
2. the condition is evaluated on every cell from the signs of the
   polynomials; the cells of full dimension on which it holds cover the
   region up to a set of measure zero (the sections, which are skipped);
3. a cell of full dimension is a *stack*: over the cell of the previous
   level the variable `x_k` ranges between two consecutive real roots
   of the projection polynomials of level `k` (or `\\pm\\infty`). Each root
   is a continuous algebraic function of `x_1, \\ldots, x_{k-1}` over the
   cell (delineability); it is written explicitly by solving the
   polynomial for `x_k` and taking the branch which agrees with the root
   at the sample point of the cell (Mathematica uses parametric ``Root``
   objects here; when SymPy finds no explicit branch the integral is
   left unevaluated);
4. the iterated integral is computed innermost variable first with
   :func:`~sympy_extras.integrals.definite_integral`, the outer bounds
   being passed as assumptions, and the cells are summed;
5. the cells of the parameter space give a case distinction: the result
   is a ``Piecewise`` whose conditions are the sign conditions of the
   projection polynomials on each parameter cell.

Four routes are tried around the decomposition [Apostol]_:

* **polar coordinates** first: a disc or an annulus (`r_1^2 < x^2 + y^2 <
  R^2`, or the ball in three variables) with an integrand depending on the
  point only through `x^2 + y^2` is `2\\pi \\int_{r_1}^R g(\\rho)\\, \\rho\\, d\\rho`
  (`4\\pi \\int g(\\rho)\\, \\rho^2\\, d\\rho` for the ball);
* **an affine change of variables** to the polar case: an ellipse or an
  ellipsoid, and any region `\\sum c_i x_i^2 + \\sum b_i x_i < v` with
  coefficients of one sign (a diagonal quadratic form, the square
  completed), is the image of a disc or ball under `x_i = u_i/\\sqrt{c_i} -
  b_i/(2 c_i)`, with the Jacobian `\\prod c_i^{-1/2}`; the polar route is
  applied to the transformed integrand when it depends on `\\sum u_i^2`
  only (the area `\\pi a b` of the ellipse `x^2/a^2 + y^2/b^2 < 1`);
* **cylindrical coordinates** in three variables: a disc or annulus
  condition on `x^2 + y^2` (or none) together with bounds on `z` which are
  linear in `z` and depend on `x, y` only through `x^2 + y^2`, and an
  integrand `g(x^2 + y^2, z)`, give `2\\pi \\int \\int g(\\rho^2, z)\\, \\rho\\, dz\\, d\\rho`,
  the range of `\\rho` being where the bounds on `z` are in the right order
  (the paraboloid `0 < z < x^2 + y^2 < 1`, the cone `\\sqrt{x^2 + y^2} < z < 1`);
* **bounds solved for the last variable** when the decomposition does not
  apply (a condition which is not polynomial, `y < \\exp(x)`) or fails: a
  conjunction of relations linear in the last variable `y` describes, over
  each point of the other variables, the interval `\\max(\\text{lower}) < y <
  \\min(\\text{upper})`; the range of the other variables is cut where the
  two bounds cross (with :func:`sympy_extras.assumptions.solve`), the
  inner integral is computed with :func:`~sympy_extras.integrals.definite_integral`
  and the outer integral by the same function on the remaining variables
  (Fubini's theorem); an outer variable without bounds ranges over the
  whole line, and the outer integral over an infinite range is again a
  matter for :func:`~sympy_extras.integrals.definite_integral`.

Integrals over polytopes are computed through the decomposition as any
polynomial region; the formulas of Lasserre and Brion for polynomial
integrands over polytopes [Lasserre]_ are not used.

The bounds solved for the last variable need not be linear in it: a
relation which :func:`sympy_extras.assumptions.solve` turns into an
interval of `y` with explicit endpoints (`y^2 < e^x` with `y > 0` into
`y < e^{x/2}`, `e^y < x` into `y < \\log x`, `y^3 < x` into `y <
x^{1/3}`) contributes its endpoints to the bounds.

With ``measure='hausdorff'`` the integral is taken with respect to the
`(n-k)`-dimensional Hausdorff measure on the variety described by the
`k` *equations* of the condition, cut by its inequalities: the length
of a curve and the area of a surface, with the integrand weighted along
them (Mathematica integrates over ``ImplicitRegion`` this way). The
cells of the decomposition on which the condition holds are then the
sections of dimension `n - 1`, on each of which one variable `x_k` is an
explicit branch `x_k = \\varphi(x_1, \\ldots, x_{k-1})` of a root of the
equation (:func:`_explicit_root`), and the integral is the integral of
`f(\\ldots, \\varphi, \\ldots) \\sqrt{1 + \\sum_j (\\partial \\varphi / \\partial x_j)^2}`
over the cell of the other variables [Apostol]_ (chapter 12): the arc
length `\\int \\sqrt{1 + \\varphi'(x)^2}\\, dx` of a graph `y = \\varphi(x)`, the
area `\\int\\int \\sqrt{1 + \\varphi_x^2 + \\varphi_y^2}\\, dx\\, dy` of a graph
`z = \\varphi(x, y)`; the outer integral over the base cell is again an
integral over a region, so that a surface over a disc is done in polar
coordinates. With `k` equations (a curve in space as the intersection of
two surfaces) the cells are sections at `k` levels, each section
variable an explicit branch with the earlier ones substituted, and the
weight is the Gram determinant `\\sqrt{\\det(I + G^T G)}` of the graph
parametrisation, `G` the Jacobian of the branches with respect to the
free variables (`\\sqrt{1 + x'(t)^2 + y'(t)^2}` for a curve `(x(t), y(t),
t)`). When a root has no explicit branch, or the condition has no
equation, the integral is left unevaluated.

A bound which is a root of a cubic in the last variable is written in
Viète's trigonometric or hyperbolic form (:func:`_trigonometric_roots`),
real where Cardano's formula needs complex radicals; a bound without an
explicit form in the last variable is retried with the variables in
another order, where it may be explicit (`x^3 + x y < 1` is `y < (1 -
x^3)/x`), and a numeric bound is an algebraic number (``CRootOf``) in
the value.

Examples
========

>>> from sympy import symbols, pi
>>> from sympy_extras.integrals.regions import IntegralByRanges, integrate_by_ranges
>>> x, y, z, r = symbols('x y z r')
>>> IntegralByRanges(1, x**2 + y**2 < 1)
IntegralByRanges(1, x**2 + y**2 < 1, (x, y))
>>> IntegralByRanges(1, x**2 + y**2 < 1).doit()
pi
>>> integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1))
1/24
>>> integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r > 0)
pi*r**2
>>> from sympy import exp
>>> integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1)
-pi*exp(-1) + pi
>>> integrate_by_ranges(1, (x > 0) & (x < 1) & (y > 0) & (y < exp(x)))
-1 + E
>>> a, b = symbols('a b', positive=True)
>>> integrate_by_ranges(1, x**2/a**2 + y**2/b**2 < 1, [x, y])
pi*a*b
>>> integrate_by_ranges(1, (x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2))
pi/2
>>> integrate_by_ranges(1, (x > 0) & (y > 0) & (y < exp(-x)))
1

The length of the unit circle and the area of the unit sphere, with the
Hausdorff measure:

>>> from sympy import Eq
>>> integrate_by_ranges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(1, Eq(x**2 + y**2 + z**2, 1), measure='hausdorff')
4*pi

The length of the circle `z = 1` on the paraboloid `z = x^2 + y^2`, a
curve in space given by two equations, and the area under the cubic
`y = x^3 - x` between its root and `x = 0`, with the variables ordered
so that the boundary root is in the last variable:

>>> integrate_by_ranges(1, Eq(z, x**2 + y**2) & Eq(z, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(1, (x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0), [y, x])
-CRootOf(x**3 - x - 1, 0)**4/4 - 1/4 + CRootOf(x**3 - x - 1, 0)**2/2 + CRootOf(x**3 - x - 1, 0)

References
==========

.. [Apostol] T. M. Apostol, *Calculus*, vol. II, 2nd ed., Wiley, 1969,
   sections 11.11 (regions between two graphs, Fubini's theorem) and
   11.27–11.29 (polar coordinates); chapter 12 (the area of a surface
   given as a graph, surface integrals).
.. [Lasserre] J. B. Lasserre, *Integration on a convex polytope*,
   Proceedings of the AMS 126 (1998), pp. 2433-2441; M. Brion, *Points
   entiers dans les polyèdres convexes*, Annales scientifiques de l'ENS
   21 (1988), pp. 653-663.
.. [Collins] G. E. Collins, *Quantifier elimination for real closed
   fields by cylindrical algebraic decomposition*, Automata Theory and
   Formal Languages, Lecture Notes in Computer Science 33, Springer,
   1975, pp. 134-183.
.. [Strzebonski] A. Strzeboński, *Cylindrical algebraic decomposition
   using validated numerics*, Journal of Symbolic Computation 41 (2006),
   pp. 1021-1038 (the decomposition behind Mathematica's ``Integrate``
   over regions and ``CylindricalDecomposition``).
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, nan, oo, zoo
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Str, Symbol
from sympy.functions.elementary.complexes import Abs, im, re, sign
from sympy.functions.elementary.hyperbolic import acosh, asinh, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import acos, cos
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.numbers import pi
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, Or, true
from sympy.matrices.dense import Matrix, eye
from sympy.sets.sets import EmptySet, FiniteSet, Interval, Set, Union
from sympy.simplify.simplify import simplify
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, factor
from sympy.solvers.solvers import solve as sympy_solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, as_symbol, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import normalize
from sympy_extras.assumptions.solve import solve
from sympy_extras.assumptions.refine import refine
from sympy_extras.polys.cad.lifting import CAD, CADCell, cylindrical_algebraic_decomposition
from sympy_extras.polys.cad.qe import _compile
from sympy_extras.polys.cad.samplepoints import RealAlgebraic, compare_real
from sympy_extras.settings import settings
from .conditions import numerically_equal
from .definite import definite_integral

__all__ = ['IntegralByRanges', 'integrate_by_ranges']

#: the numerical tolerance for matching a branch to a root at the sample point
_TOLERANCE = 1e-20

#: the measures of integration: the Lebesgue measure of `\\mathbb{R}^n` and
#: the `(n-1)`-dimensional Hausdorff measure on the hypersurface of the
#: equation of the condition
_MEASURES = ('lebesgue', 'hausdorff')


class IntegralByRanges(Expr):
    """The integral of ``integrand`` over the region described by
    ``condition``, a Boolean combination of polynomial inequalities and
    equations in the integration ``variables`` (and parameters).

    The constructor does not evaluate: :meth:`doit` (or
    :func:`integrate_by_ranges`) computes the integral.

    ``measure='hausdorff'`` integrates with respect to the
    `(n-1)`-dimensional Hausdorff measure on the hypersurface given by
    the equation of the condition (arc length, surface area), see the
    module documentation.

    Examples
    ========

    >>> from sympy import symbols, Eq
    >>> from sympy_extras.integrals.regions import IntegralByRanges
    >>> x, y = symbols('x y')
    >>> region = IntegralByRanges(x*y, (x > 0) & (x < y) & (y < 1))
    >>> region.integrand, region.condition, region.variables
    (x*y, (x > 0) & (x < y) & (y < 1), (x, y))
    >>> region.doit()
    1/8
    >>> circle = IntegralByRanges(1, Eq(x**2 + y**2, 1), measure='hausdorff')
    >>> circle.measure == 'hausdorff'
    True
    >>> circle.doit()
    2*pi
    """

    def __new__(cls, integrand: ExprLike, condition: object,
                variables: Optional[Sequence[Symbol]] = None,
                measure: str = 'lebesgue') -> IntegralByRanges:
        integrand_ = as_expr(integrand)
        condition_ = as_boolean(condition)
        if variables is None:
            names = sorted_symbols(free_symbols(condition_))
        else:
            names = [as_symbol(v) for v in variables]
        if not names:
            raise ValueError("no integration variable: the condition has no symbols")
        if measure not in _MEASURES:
            raise ValueError("measure must be one of %s, got %r" % (", ".join(_MEASURES), measure))
        args: list[Basic] = [integrand_, condition_, Tuple(*names)]
        if measure != _MEASURES[0]:
            args.append(Str(measure))
        obj = Expr.__new__(cls, *args)
        if not isinstance(obj, IntegralByRanges):
            raise TypeError("unexpected construction of IntegralByRanges")
        return obj

    @property
    def integrand(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def condition(self) -> Boolean:
        return as_boolean(self.args[1])

    @property
    def variables(self) -> tuple[Symbol, ...]:
        names = self.args[2]
        if not isinstance(names, Tuple):
            raise TypeError("the variables of IntegralByRanges must be a Tuple")
        return tuple(as_symbol(v) for v in names.args)

    @property
    def measure(self) -> str:
        """``'lebesgue'`` or ``'hausdorff'``."""
        if len(self.args) < 4:
            return _MEASURES[0]
        name = self.args[3]
        if not isinstance(name, Str):
            raise TypeError("the measure of IntegralByRanges must be a Str")
        return str(name)

    def doit(self, assumptions: Assumptions = None, **hints: object) -> Expr:
        """The value, or the integral unchanged when it cannot be computed."""
        return integrate_by_ranges(self.integrand, self.condition, self.variables, assumptions,
                                   self.measure)


class _Bound:
    """A root of a polynomial in ``x_k`` over a cell: the ``index``-th real
    root of ``poly`` at the sample point, and its explicit form."""

    def __init__(self, poly: Poly, index: int, value: RealAlgebraic) -> None:
        self.poly = poly
        self.index = index
        self.value = value


class _Stack:
    """The bounds of one cell of full dimension, level by level, as
    explicit expressions (``-oo``/``oo`` for an unbounded sector)."""

    def __init__(self, cell: CADCell, bounds: list[tuple[Expr, Expr]]) -> None:
        self.cell = cell
        self.bounds = bounds


def _root_sequence(parent: CADCell, polys: Sequence[Poly], gens: Sequence[Symbol]) -> list[_Bound]:
    """The sorted distinct real roots of the polynomials of a level over
    the sample point of the parent cell, each attributed to the first
    polynomial having it."""
    found: list[_Bound] = []
    for poly in polys:
        roots = parent.sample.real_roots(poly, gens)
        if roots is None:
            continue
        for index, root in enumerate(roots):
            if not any(compare_real(root, b.value) == 0 for b in found):
                found.append(_Bound(poly, index, root))
    found.sort(key=lambda b: _Key(b.value))
    return found


class _Key:
    __slots__ = ('value',)

    def __init__(self, value: RealAlgebraic) -> None:
        self.value = value

    def __lt__(self, other: _Key) -> bool:
        return compare_real(self.value, other.value) < 0


def _explicit_root(bound: _Bound, parent: CADCell, gens: Sequence[Symbol]) -> Optional[Expr]:
    """The root as an expression in the earlier variables: the solution of
    the polynomial for the last generator which takes the value of the
    root at the sample point of the parent cell."""
    x = gens[-1]
    equation = as_expr(bound.poly.as_expr())
    point = {g: p for g, p in zip(gens[:-1], parent.point)}
    if not equation.has(*gens[:-1]) if len(gens) > 1 else True:
        return as_expr(bound.value)
    target = as_expr(bound.value).evalf(30)
    trigonometric = _trigonometric_roots(bound.poly, x)
    for candidates in (trigonometric, attempt(lambda: sympy_solve(equation, x), settings.timeout)):
        if not isinstance(candidates, list):
            continue
        matches: list[Expr] = []
        for candidate in candidates:
            c = as_expr(candidate)
            value = as_expr(c.xreplace(point)).evalf(30)
            if not value.is_number or value.has(nan, zoo, oo):
                continue                                    # a form for the other sign of p
            real_part, imaginary_part = as_expr(re(value)), as_expr(im(value))
            if not (imaginary_part.is_number and imaginary_part.is_comparable
                    and abs(imaginary_part) < _TOLERANCE):
                continue                                    # Cardano's form of a real root has a tiny imaginary part
            difference = as_expr(abs(real_part - target))
            if difference.is_number and difference.is_comparable and difference < _TOLERANCE:
                matches.append(c)
        if len(matches) == 1:
            return matches[0]
    return None


def _trigonometric_roots(poly: Poly, x: Symbol) -> Optional[list[Expr]]:
    """The roots of a cubic in ``x`` in Viète's trigonometric and
    hyperbolic forms for the depressed cubic ``t**3 + p*t + q``: the three
    real roots ``2*sqrt(-p/3)*cos(acos(3*q*sqrt(-3/p)/(2*p))/3 - 2*pi*k/3)``
    where Cardano's formula needs complex radicals (the *casus
    irreducibilis*), the single real root ``-2*sqrt(p/3)*sinh(asinh(3*q*
    sqrt(3/p)/(2*p))/3)`` for ``p > 0`` and ``-2*sign(q)*sqrt(-p/3)*cosh(
    acosh(-3*Abs(q)*sqrt(-3/p)/(2*p))/3)`` for ``p < 0``, all shifted by
    ``-b/(3*a)``; the candidates which are real at the sample point are
    selected by the caller. ``None`` for another degree.

    >>> from sympy import symbols, Poly
    >>> from sympy_extras.integrals.regions import _trigonometric_roots
    >>> x = symbols('x')
    >>> [r.evalf(6) for r in _trigonometric_roots(Poly(x**3 - 3*x + 1, x), x)[:3]]
    [1.53209, 0.347296, -1.87939]
    >>> _trigonometric_roots(Poly(x**3 + x - 1, x), x)[3].evalf(6)
    0.682328
    """
    if poly.degree(x) != 3:
        return None
    coefficients = [as_expr(c) for c in Poly(poly.as_expr(), x).all_coeffs()]
    a, b, c, d = coefficients
    p = as_expr((3 * a * c - b**2) / (3 * a**2))
    q = as_expr((2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3))
    if p == 0:
        return None
    shift = -b / (3 * a)
    radius = 2 * sqrt(-p / 3)
    angle = acos(3 * q * sqrt(-3 / p) / (2 * p)) / 3
    found = [as_expr(radius * cos(angle - 2 * pi * k / 3) + shift) for k in range(3)]
    found.append(as_expr(-2 * sqrt(p / 3) * sinh(asinh(3 * q * sqrt(3 / p) / (2 * p)) / 3) + shift))
    found.append(as_expr(-2 * sign(q) * sqrt(-p / 3) * cosh(acosh(-3 * Abs(q) * sqrt(-3 / p) / (2 * p)) / 3) + shift))
    return found


def _stack(cell: CADCell, cad: CAD, first: int, skip: Sequence[int] = (),
           replace: Optional[dict[Symbol, Expr]] = None) -> Optional[_Stack]:
    """The explicit bounds of a cell in the levels from ``first``
    (1-based) upwards, all of them sectors but the levels ``skip``
    (sections, left out), whose variables are replaced in the bounds of
    the later levels by ``replace``."""
    ancestors: list[CADCell] = []
    current: Optional[CADCell] = cell
    while current is not None and current.level > 0:
        ancestors.append(current)
        current = current.parent
    ancestors.reverse()
    bounds: list[tuple[Expr, Expr]] = []
    for level in range(first, len(cad.gens) + 1):
        if level in skip:
            continue
        node = ancestors[level - 1]
        parent = node.parent
        if parent is None:
            return None
        if node.is_section:
            return None
        gens = cad.gens[:level]
        roots = _root_sequence(parent, cad.projection[level - 1], gens)
        position = (node.index[-1] - 1) // 2          # the sector lies between root position-1 and position
        lower: Expr = S.NegativeInfinity
        upper: Expr = S.Infinity
        if position >= 1:
            found = _explicit_root(roots[position - 1], parent, gens)
            if found is None:
                return None
            lower = found
        if position < len(roots):
            found = _explicit_root(roots[position], parent, gens)
            if found is None:
                return None
            upper = found
        if replace:
            lower = as_expr(lower.xreplace(replace))
            upper = as_expr(upper.xreplace(replace))
        bounds.append((lower, upper))
    return _Stack(cell, bounds)


def _ask(query: Boolean, assumptions: list[Boolean]) -> Optional[bool]:
    """``ask`` which treats contradictory assumptions (an empty cell of
    the decomposition under its parameter condition) as undecided."""
    try:
        return ask(query, assumptions)
    except ValueError:
        return None


def _split_roots(expr: Expr, assumptions: list[Boolean]) -> Expr:
    """Square roots (and other rational powers) of products with factors
    known positive under the assumptions split into powers of the
    factors, ``sqrt((1 - x**2)*(1 - u**2))`` into
    ``sqrt(1 - x**2)*sqrt(1 - u**2)`` when ``-1 < x < 1``, so that the
    integrand of an inner integral has the parameters outside the
    radicals."""
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        exponent = as_expr(node.exp)
        if not isinstance(exponent, Rational) or exponent.q == 1:
            return node
        base = as_expr(factor(node.base))
        factors = list(base.args) if isinstance(base, Mul) else [base]
        positive: list[tuple[Expr, Expr]] = []             # (a factor known positive, its exponent)
        rest: list[Expr] = []
        sign = 1
        for f in factors:
            f_ = as_expr(f)
            power: Expr = S.One
            if isinstance(f, Pow) and as_expr(f.exp).is_integer:
                # a reciprocal (x - 1)**-1 is split by the sign of x - 1
                f_, power = as_expr(f.base), as_expr(f.exp)
            if f_.is_positive or (not f_.is_number and _ask(as_boolean(f_ > 0), assumptions) is True):
                positive.append((f_, power))
            elif f_.is_negative or (not f_.is_number and _ask(as_boolean(f_ < 0), assumptions) is True):
                positive.append((-f_, power))
                if as_expr(power).is_odd:
                    sign = -sign
            else:
                rest.append(f_**power)
        if not positive:
            return node
        result: Expr = S.One
        for f_, power in positive:
            result = result * f_**(power * exponent)
        remaining = as_expr(sign * Mul(*rest))
        return result * remaining**exponent

    if not expr.has(Pow):
        return expr
    return as_expr(expr.replace(lambda n: isinstance(n, Pow), rewrite))


def _iterated(integrand: Expr, variables: Sequence[Symbol], stack: _Stack,
              assumptions: list[Boolean]) -> Optional[Expr]:
    """The iterated integral over the stack, innermost variable first.

    A sector between opposite bounds ``(-h, h)`` (the two roots of an even
    quadratic, as in a disc) is rescaled to ``(-1, 1)`` by ``x = h*u``
    so that the bound leaves the integrand: the radicals are then split
    by :func:`_split_roots` and the parameters of the inner integral come
    out of them."""
    value = integrand
    for k in range(len(variables) - 1, -1, -1):
        x = variables[k]
        lower, upper = stack.bounds[k]
        outer: list[Boolean] = list(assumptions)
        for j in range(k):
            lo, hi = stack.bounds[j]
            if lo != -oo:
                outer.append(as_boolean(variables[j] > lo))
            if hi != oo:
                outer.append(as_boolean(variables[j] < hi))
        if not value.has(x):
            if lower == -oo or upper == oo:
                return None
            value = as_expr(value * (upper - lower))
            continue
        inner = list(outer)
        if lower != -oo:
            inner.append(as_boolean(x > lower))
        if upper != oo:
            inner.append(as_boolean(x < upper))
        if any(part is S.false for part in inner):
            return S.Zero                                   # an empty cell under its parameter condition
        piece = _split_roots(value, inner)
        try:
            result = definite_integral(piece, (x, lower, upper), outer)
        except ValueError:
            return S.Zero                                   # contradictory bounds: an empty cell
        if result.has(Integral, IntegralByRanges) and lower == -upper and upper != oo \
                and _ask(as_boolean(upper > 0), outer) is True:
            u = Dummy('u')
            rescaled = _split_roots(as_expr(value.subs(x, upper * u) * upper),
                                    outer + [as_boolean(upper > 0), as_boolean(u > -1), as_boolean(u < 1)])
            result = definite_integral(rescaled, (u, S.NegativeOne, S.One), outer)
        if result.has(Integral, IntegralByRanges):
            return None
        value = result
    return value


def _parameter_condition(cell: CADCell, cad: CAD, levels: int) -> Boolean:
    """The sign conditions of the projection polynomials of the first
    ``levels`` levels at the ancestor of ``cell`` of that level."""
    return _cell_condition(cell, cad, 1, levels)


def _cell_condition(cell: CADCell, cad: CAD, first: int, last: int) -> Boolean:
    """The sign conditions of the projection polynomials of the levels
    ``first`` to ``last`` (1-based) at the ancestor of ``cell`` of level
    ``last``."""
    levels = last
    parts: list[Boolean] = []
    ancestor: Optional[CADCell] = cell
    while ancestor is not None and ancestor.level > levels:
        ancestor = ancestor.parent
    if ancestor is None:
        return true
    for level in range(first, levels + 1):
        gens = cad.gens[:level]
        for poly in cad.projection[level - 1]:
            sign = ancestor.sample.sign(poly, gens) if len(ancestor.sample) == level else \
                _ancestor_at(ancestor, level).sample.sign(poly, gens)
            expression = as_expr(poly.as_expr())
            if sign > 0:
                parts.append(as_boolean(Gt(expression, 0)))
            elif sign < 0:
                parts.append(as_boolean(Lt(expression, 0)))
            else:
                parts.append(as_boolean(Eq(expression, 0)))
    return as_boolean(And(*parts))


def _ancestor_at(cell: CADCell, level: int) -> CADCell:
    current = cell
    while current.level > level:
        parent = current.parent
        if parent is None:
            break
        current = parent
    return current



def _atoms(formula: Boolean) -> Optional[list[Relational]]:
    """The relations of a conjunction of strict or weak inequalities;
    ``None`` for any other formula (equations, disjunctions)."""
    parts = list(formula.args) if isinstance(formula, And) else [formula]
    atoms: list[Relational] = []
    for part in parts:
        if not isinstance(part, (Lt, Le, Gt, Ge)):
            return None
        atoms.append(part)
    return atoms


def _radial_atom(atom: Relational, names: Sequence[Symbol]) -> Optional[tuple[bool, Expr]]:
    """``(upper, value)`` when the relation bounds ``x**2 + y**2 (+ z**2)``:
    ``upper`` for a bound from above (``rho**2 < value``), else from
    below; ``None`` when the relation is not of that form."""
    difference = as_expr(atom.lhs) - as_expr(atom.rhs)
    radius = as_expr(sum(v**2 for v in names))
    try:
        poly = Poly(difference, *names)
    except PolynomialError:
        return None
    coefficient = as_expr(poly.coeff_monomial(names[0]**2))
    if coefficient == 0 or coefficient.has(*names):
        return None
    rest = as_expr((difference - coefficient * radius).expand())
    if rest.has(*names):
        return None
    # coefficient * rho**2 + rest  <  0  (or <=, >, >=)
    bound = as_expr(-rest / coefficient)
    less = isinstance(atom, (Lt, Le))
    positive = coefficient.is_positive is True
    negative = coefficient.is_negative is True
    if not positive and not negative:
        return None
    return (less == positive, bound)


def _radial_integrand(f: Expr, names: Sequence[Symbol], rho: Symbol) -> Optional[Expr]:
    """``g(rho)`` with ``f == g(sqrt(x**2 + y**2 + ...))``, or ``None``
    when ``f`` is not a function of the distance to the origin."""
    radius = sqrt(as_expr(sum(v**2 for v in names)))
    on_axis: dict[Symbol, Expr] = {names[0]: radius}
    for v in names[1:]:
        on_axis[v] = S.Zero
    candidate = as_expr(f.xreplace(on_axis))
    if candidate.has(S.NaN, S.ComplexInfinity):
        return None
    difference = as_expr(f - candidate)
    simpler = attempt(lambda: as_expr(simplify(difference)), settings.timeout)
    if simpler is None or simpler != 0:
        if not numerically_equal(f, candidate, [as_boolean(v > 0) for v in names]):
            return None
    axis: dict[Symbol, Expr] = {names[0]: rho}
    for v in names[1:]:
        axis[v] = S.Zero
    return as_expr(f.xreplace(axis))


def _radial(f: Expr, formula: Boolean, names: Sequence[Symbol], assumptions: list[Boolean]) -> Optional[Expr]:
    """The integral over a disc, an annulus or a ball of a function of the
    distance to the origin, in polar (spherical) coordinates."""
    if len(names) not in (2, 3):
        return None
    atoms = _atoms(formula)
    if atoms is None or not atoms:
        return None
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    for atom in atoms:
        found = _radial_atom(atom, names)
        if found is None:
            return None
        is_upper, value = found
        if is_upper:
            if upper is not None:
                return None
            upper = value
        else:
            if lower is not None:
                return None
            lower = value
    if upper is None:
        return None
    rho = Dummy('rho', positive=True)
    g = _radial_integrand(f, names, rho)
    if g is None:
        return None
    positive = ask(as_boolean(upper > 0), assumptions)
    if positive is False:
        return S.Zero
    inner: list[Boolean] = list(assumptions)
    if positive is None:
        inner.append(as_boolean(upper > 0))
    if lower is None:
        start: Expr = S.Zero
    else:
        if ask(as_boolean(lower >= 0), inner) is not True or ask(as_boolean(lower < upper), inner) is not True:
            return None
        start = as_expr(sqrt(lower))
    dimension = len(names)
    measure = 2 * pi * rho if dimension == 2 else 4 * pi * rho**2
    radius = as_expr(sqrt(upper))
    refined = attempt(lambda: as_expr(refine(radius, inner)), settings.timeout)
    if refined is not None:
        radius = refined
    value = definite_integral(as_expr(g * measure), (rho, start, radius), inner)
    if value.has(Integral, IntegralByRanges):
        return None
    tidy = attempt(lambda: as_expr(refine(value, inner)), settings.timeout)
    if tidy is not None:
        value = tidy
    if positive is None:
        return as_expr(Piecewise((value, as_boolean(upper > 0)), (S.Zero, True)))
    return value


def _form_atom(atom: Relational, names: Sequence[Symbol],
               assumptions: list[Boolean]) -> Optional[tuple[bool, list[Expr], list[Expr], Expr]]:
    """``(upper, coefficients, shifts, value)`` when the relation bounds a
    diagonal quadratic form with the square completed,
    ``sum(c_i*(x_i + s_i)**2) < value`` (``upper``) or ``> value``, with
    all ``c_i`` positive under the assumptions; ``None`` otherwise (a
    cross term, a coefficient of unknown sign, a missing square)."""
    difference = as_expr(atom.lhs) - as_expr(atom.rhs)
    try:
        poly = Poly(difference, *names)
    except PolynomialError:
        return None
    if poly.total_degree() != 2:
        return None
    coefficients: list[Expr] = []
    linear: list[Expr] = []
    for i, v in enumerate(names):
        square = [0] * len(names)
        square[i] = 2
        c = as_expr(poly.coeff_monomial(v**2))
        b = as_expr(poly.coeff_monomial(v))
        if c == 0:
            return None
        coefficients.append(c)
        linear.append(b)
    for monomial, _ in poly.terms():
        if sum(monomial) == 2 and max(monomial) == 1:
            return None                                     # a cross term
    sign_: Optional[bool] = None
    for c in coefficients:
        positive = c.is_positive is True or ask(as_boolean(c > 0), assumptions) is True
        negative = c.is_negative is True or ask(as_boolean(c < 0), assumptions) is True
        if not positive and not negative:
            return None
        if sign_ is None:
            sign_ = positive
        elif sign_ != positive:
            return None
    assert sign_ is not None
    if not sign_:
        coefficients = [-c for c in coefficients]
        linear = [-b for b in linear]
        difference = -difference
    less = isinstance(atom, (Lt, Le))
    shifts = [as_expr(b / (2 * c)) for b, c in zip(linear, coefficients)]
    completed = as_expr(sum(c * (v + t)**2 for c, v, t in zip(coefficients, names, shifts)))
    rest = as_expr((difference - completed).expand())
    if rest.has(*names):
        return None
    return (less == sign_, coefficients, shifts, as_expr(-rest))


def _scaled_radial(f: Expr, formula: Boolean, names: Sequence[Symbol],
                   assumptions: list[Boolean]) -> Optional[Expr]:
    """The integral over an ellipse or ellipsoid (a diagonal quadratic
    form bounded from above, and from below by a proportional one) of a
    function which is radial after the affine change of variables
    ``x_i = u_i/sqrt(c_i) - s_i``, through :func:`_radial`."""
    if len(names) not in (2, 3):
        return None
    atoms = _atoms(formula)
    if atoms is None or not atoms:
        return None
    upper: Optional[tuple[list[Expr], list[Expr], Expr]] = None
    lower: Optional[tuple[list[Expr], list[Expr], Expr]] = None
    for atom in atoms:
        found = _form_atom(atom, names, assumptions)
        if found is None:
            return None
        is_upper, coefficients, shifts, value = found
        if is_upper:
            if upper is not None:
                return None
            upper = (coefficients, shifts, value)
        else:
            if lower is not None:
                return None
            lower = (coefficients, shifts, value)
    if upper is None:
        return None
    coefficients, shifts, value = upper
    if all(c == 1 for c in coefficients) and all(t == 0 for t in shifts):
        return None                                         # the polar route already looked at it
    us = [Dummy(v.name) for v in names]
    substitution: dict[Expr, Expr] = {}
    jacobian: Expr = S.One
    for v, u_, c, t in zip(names, us, coefficients, shifts):
        substitution[v] = as_expr(u_ / sqrt(c) - t)
        jacobian = jacobian / sqrt(c)
    radius = as_expr(sum(u_**2 for u_ in us))
    parts: list[Boolean] = [as_boolean(radius < value)]
    if lower is not None:
        low_coefficients, low_shifts, low_value = lower
        ratio = as_expr(low_coefficients[0] / coefficients[0])
        for c, d in zip(coefficients, low_coefficients):
            if as_expr(d - ratio * c).simplify() != 0:
                return None
        if any(as_expr(t - t_).simplify() != 0 for t, t_ in zip(shifts, low_shifts)):
            return None
        if ask(as_boolean(ratio > 0), assumptions) is not True:
            return None
        parts.append(as_boolean(radius > low_value / ratio))
    g = as_expr(f.xreplace(substitution) * jacobian)
    return _radial(g, as_boolean(And(*parts)), us, assumptions)


def _cylindrical(f: Expr, formula: Boolean, names: Sequence[Symbol],
                 assumptions: list[Boolean]) -> Optional[Expr]:
    """The integral in cylindrical coordinates: a disc or annulus condition
    on ``x**2 + y**2`` (or none), bounds on ``z`` linear in ``z`` which
    depend on ``x, y`` through ``x**2 + y**2`` only, and an integrand
    ``g(x**2 + y**2, z)``, see the module documentation."""
    if len(names) != 3:
        return None
    atoms = _atoms(formula)
    if atoms is None or not atoms:
        return None
    x, y, z = names
    plane = (x, y)
    lower_radius: Optional[Expr] = None
    upper_radius: Optional[Expr] = None
    lowers: list[Expr] = []
    uppers: list[Expr] = []
    rho = Dummy('rho', positive=True)
    for atom in atoms:
        if not atom.has(z):
            found = _radial_atom(atom, plane)
            if found is None:
                return None
            is_upper, value = found
            if is_upper:
                if upper_radius is not None:
                    return None
                upper_radius = value
            else:
                if lower_radius is not None:
                    return None
                lower_radius = value
            continue
        bound = _linear_bound(atom, z, assumptions)
        if bound is None:
            return None
        is_upper, expression = bound
        radial = _radial_integrand(expression, plane, rho) if expression.has(x, y) else expression
        if radial is None:
            return None
        (uppers if is_upper else lowers).append(radial)
    g = _radial_integrand(f, plane, rho) if f.has(x, y) else f
    if g is None:
        return None
    inner_assumptions: list[Boolean] = list(assumptions)
    lower_z: Expr = S.NegativeInfinity
    upper_z: Expr = S.Infinity
    if lowers:
        found_lower = _extreme(lowers, True, inner_assumptions)
        if found_lower is None:
            return None
        lower_z = found_lower
    if uppers:
        found_upper = _extreme(uppers, False, inner_assumptions)
        if found_upper is None:
            return None
        upper_z = found_upper
    # the range of rho: inside the annulus, where the bounds on z are ordered
    conditions: list[Boolean] = []
    if upper_radius is not None:
        if ask(as_boolean(upper_radius > 0), assumptions) is not True:
            return None
        conditions.append(as_boolean(rho < sqrt(upper_radius)))
    if lower_radius is not None:
        if ask(as_boolean(lower_radius >= 0), assumptions) is not True:
            return None
        conditions.append(as_boolean(rho > sqrt(lower_radius)))
    if lower_z != -oo and upper_z != oo:
        conditions.append(as_boolean(lower_z < upper_z))
    if not conditions:
        return None
    where = attempt(lambda: solve(as_boolean(And(*conditions)), rho, assumptions + [as_boolean(rho > 0)],
                                  domain=S.Reals), settings.timeout)
    if where is None:
        return None
    pieces = _interval_conditions(as_set_(where), rho)
    if pieces is None:
        return None
    total: Expr = S.Zero
    for piece in pieces:
        limits = _piece_limits(piece, rho)
        if limits is None:
            return None
        start, stop = limits
        rho_assumptions = assumptions + [as_boolean(rho > start), as_boolean(rho < stop)] \
            if stop != oo else assumptions + [as_boolean(rho > start)]
        inner = definite_integral(as_expr(g * rho), (z, lower_z, upper_z), rho_assumptions)
        if inner.has(Integral, IntegralByRanges):
            return None
        value = definite_integral(as_expr(2 * pi * inner), (rho, start, stop), assumptions)
        if value.has(Integral, IntegralByRanges):
            return None
        total = total + value
    return as_expr(total)


def _piece_limits(piece: Boolean, rho: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(start, stop)`` of a conjunction ``rho > start & rho < stop``
    (``0`` and ``oo`` when a side is missing)."""
    start: Expr = S.Zero
    stop: Expr = S.Infinity
    parts = list(piece.args) if isinstance(piece, And) else ([] if piece is true else [piece])
    for part in parts:
        if isinstance(part, (Gt, Ge)) and part.lhs == rho:
            start = as_expr(part.rhs)
        elif isinstance(part, (Lt, Le)) and part.lhs == rho:
            stop = as_expr(part.rhs)
        elif isinstance(part, (Lt, Le)) and part.rhs == rho:
            start = as_expr(part.lhs)
        elif isinstance(part, (Gt, Ge)) and part.rhs == rho:
            stop = as_expr(part.lhs)
        else:
            return None
    return (start, stop)


def _linear_bound(atom: Relational, y: Symbol, assumptions: list[Boolean]) -> Optional[tuple[bool, Expr]]:
    """``(upper, bound)`` for a relation linear in ``y``: ``y < bound``
    (``upper``) or ``y > bound``; ``None`` when the relation is not linear
    in ``y`` or the sign of the coefficient of ``y`` is not known."""
    difference = as_expr(atom.lhs) - as_expr(atom.rhs)
    try:
        poly = Poly(difference, y)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    coefficients = [as_expr(c) for c in poly.all_coeffs()]
    p, q = coefficients[0], coefficients[1]
    # p*y + q < 0  (Lt, Le) or > 0 (Gt, Ge)
    less = isinstance(atom, (Lt, Le))
    positive = p.is_positive is True or ask(as_boolean(p > 0), assumptions) is True
    negative = p.is_negative is True or ask(as_boolean(p < 0), assumptions) is True
    if not positive and not negative:
        return None
    bound = as_expr(-q / p)
    return (less == positive, bound)


def _solved_bound(atom: Relational, y: Symbol, assumptions: list[Boolean]) -> Optional[list[tuple[bool, Expr]]]:
    """The bounds ``(upper, bound)`` on ``y`` from a relation: the one of
    :func:`_linear_bound` when the relation is linear in ``y``, else the
    finite endpoints of the interval of ``y`` which
    :func:`sympy_extras.assumptions.solve` finds under the assumptions
    (``y**2 < x`` with ``x > 0`` gives ``-sqrt(x) < y < sqrt(x)``,
    ``exp(y) < x`` gives ``y < log(x)``); ``None`` when the solution is
    not one interval."""
    linear = _linear_bound(atom, y, assumptions)
    if linear is not None:
        return [linear]
    where = attempt(lambda: solve(atom, y, assumptions, domain=S.Reals), settings.timeout)
    if not isinstance(where, Interval):
        return None
    found: list[tuple[bool, Expr]] = []
    left, right = as_expr(where.left), as_expr(where.right)
    if left != -oo:
        found.append((False, left))
    if right != oo:
        found.append((True, right))
    return found


def _extreme(bounds: list[Expr], largest: bool, assumptions: list[Boolean]) -> Optional[Expr]:
    """The largest (smallest) of the bounds, when the assumptions order
    them; ``None`` otherwise."""
    best = bounds[0]
    for other in bounds[1:]:
        if ask(as_boolean(other >= best if largest else other <= best), assumptions) is True:
            best = other
        elif ask(as_boolean(other <= best if largest else other >= best), assumptions) is not True:
            return None
    return best


def _interval_conditions(where: Set, x: Symbol) -> Optional[list[Boolean]]:
    """A set of real numbers as a list of conjunctions ``a < x < b``, one
    per interval (points are dropped: measure zero); ``None`` for a set
    which is not a union of intervals."""
    if where is S.Reals:
        return [true]
    if isinstance(where, EmptySet) or where is S.EmptySet:
        return []
    if isinstance(where, FiniteSet):
        return []
    if isinstance(where, Interval):
        parts: list[Boolean] = []
        if where.left != -oo:
            parts.append(as_boolean(x > as_expr(where.left)))
        if where.right != oo:
            parts.append(as_boolean(x < as_expr(where.right)))
        return [as_boolean(And(*parts))]
    if isinstance(where, Union):
        pieces: list[Boolean] = []
        for member in where.args:
            found = _interval_conditions(as_set_(member), x)
            if found is None:
                return None
            pieces.extend(found)
        return pieces
    return None


def as_set_(value: Basic) -> Set:
    if not isinstance(value, Set):
        raise TypeError("a set is expected, got %s" % (value,))
    return value


def _solved_bounds(f: Expr, formula: Boolean, names: Sequence[Symbol],
                   assumptions: list[Boolean]) -> Optional[Expr]:
    """The integral through the bounds of the last variable solved from
    the relations (Fubini), see the module documentation."""
    atoms = _atoms(formula)
    if atoms is None or not atoms:
        return None
    y = names[-1]
    outer_names = list(names[:-1])
    outer_atoms: list[Boolean] = []
    lowers: list[Expr] = []
    uppers: list[Expr] = []
    outer_assumptions: list[Boolean] = list(assumptions)
    for atom in atoms:
        if not atom.has(y):
            outer_atoms.append(atom)
            outer_assumptions.append(atom)
    for atom in atoms:
        if not atom.has(y):
            continue
        found = _solved_bound(atom, y, outer_assumptions)
        if found is None:
            return None
        for is_upper, bound in found:
            (uppers if is_upper else lowers).append(bound)
    lower: Expr = S.NegativeInfinity
    upper: Expr = S.Infinity
    if lowers:
        found_lower = _extreme(lowers, True, outer_assumptions)
        if found_lower is None:
            return None
        lower = found_lower
    if uppers:
        found_upper = _extreme(uppers, False, outer_assumptions)
        if found_upper is None:
            return None
        upper = found_upper
    if not outer_names:
        if outer_atoms:
            return None
        value = definite_integral(f, (y, lower, upper), assumptions)
        return None if value.has(Integral, IntegralByRanges) else value
    # where the two bounds are in the right order
    pieces: list[Boolean] = [true]
    if lower != -oo and upper != oo:
        ordered = ask(as_boolean(lower < upper), outer_assumptions)
        if ordered is False:
            return S.Zero
        if ordered is None:
            if len(outer_names) != 1:
                return None
            x = outer_names[0]
            where = attempt(lambda: solve(as_boolean(lower < upper), x, outer_assumptions, domain=S.Reals),
                            settings.timeout)
            if where is None:
                return None
            found_pieces = _interval_conditions(as_set_(where), x)
            if found_pieces is None:
                return None
            pieces = found_pieces
    inner_assumptions = outer_assumptions + [as_boolean(y > lower)] * (lower != -oo) \
        + [as_boolean(y < upper)] * (upper != oo)
    inner = definite_integral(f, (y, lower, upper), inner_assumptions)
    if inner.has(Integral, IntegralByRanges):
        return None
    total: Expr = S.Zero
    for piece in pieces:
        condition = as_boolean(And(*outer_atoms, piece))
        if condition is true:
            # the other variables are unbounded: the whole line each,
            # innermost first
            value = inner
            for v in reversed(outer_names):
                value = definite_integral(value, (v, -oo, oo), assumptions)
                if value.has(Integral, IntegralByRanges):
                    return None
        else:
            value = integrate_by_ranges(inner, condition, outer_names, assumptions)
            if value.has(Integral, IntegralByRanges):
                return None
        total = total + value
    return as_expr(total)


def integrate_by_ranges(integrand: ExprLike, condition: object,
                        variables: Optional[Sequence[Symbol]] = None,
                        assumptions: Assumptions = None,
                        measure: str = 'lebesgue') -> Expr:
    """The integral of ``integrand`` over the region ``condition``.

    Parameters
    ==========

    integrand : Expr
    condition : Boolean
        A Boolean combination of polynomial relations (``<``, ``<=``,
        ``>``, ``>=``, ``Eq``, ``Ne``) with rational coefficients, for
        the decomposition; or a conjunction of inequalities linear in the
        last variable (``y < exp(x)``), solved for it; discs, ellipses,
        balls and cylinders with radial integrands take the shortcuts of
        the module documentation.
    variables : list of Symbol, optional
        The integration variables, in the order of the decomposition
        (the last one is integrated first); by default every symbol of
        the condition. The other symbols are parameters.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters; the cases of the parameter space
        they refute are dropped.
    measure : ``'lebesgue'`` (default) or ``'hausdorff'``
        With the Hausdorff measure the condition must contain exactly one
        equation, and the integral is taken on the hypersurface it
        describes (the length of a curve, the area of a surface), see the
        module documentation.

    Returns
    =======

    The value, a ``Piecewise`` over the cases of the parameters, or the
    unevaluated :class:`IntegralByRanges` when a bound has no explicit
    form or an inner integral is not computed.

    Examples
    ========

    >>> from sympy import symbols, pi, exp, log
    >>> from sympy_extras.integrals.regions import integrate_by_ranges
    >>> x, y, z, r = symbols('x y z r')
    >>> integrate_by_ranges(x**2, x**2 + y**2 < 1)
    pi/4
    >>> integrate_by_ranges(1/(1 + x**2 + y**2), x**2 + y**2 < 3)
    2*pi*log(2)
    >>> integrate_by_ranges(1, (x > 1) & (x < 2) & (y > 0) & (y < log(x)))
    -1 + log(4)
    >>> integrate_by_ranges(1, x**2 + y**2 + z**2 < 1)
    4*pi/3
    >>> integrate_by_ranges(1, (0 < x) & (x < y) & (y < x**2) & (x < 2))
    5/6
    >>> integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y], r > 0)
    pi*r**2

    Without the assumption the answer is a ``Piecewise`` over the three
    cells ``r < 0``, ``r = 0`` and ``r > 0`` of the parameter space:

    >>> integrate_by_ranges(1, x**2 + y**2 < r**2, [x, y]).subs(r, -2)
    4*pi

    The length of the arc of the parabola ``y = x**2`` over ``0 < x < 1``
    and the area of the paraboloid ``z = x**2 + y**2`` below ``z = 1``:

    >>> from sympy import Eq
    >>> integrate_by_ranges(1, Eq(y, x**2) & (x > 0) & (x < 1), measure='hausdorff')
    asinh(2)/4 + sqrt(5)/2
    >>> integrate_by_ranges(1, Eq(z, x**2 + y**2) & (z < 1), measure='hausdorff')
    pi*(-1 + 5*sqrt(5))/6
    """
    node = IntegralByRanges(integrand, condition, variables, measure)
    f, formula, names = node.integrand, normalize(node.condition), node.variables
    extra: list[Boolean] = []
    if assumptions is not None:
        extra = [as_boolean(a) for a in ([assumptions] if isinstance(assumptions, (Boolean, bool))
                                         else assumptions)]
    if measure == 'hausdorff':
        surface = _hausdorff(f, formula, names, assumptions, extra)
        return node if surface is None else surface
    radial = _radial(f, formula, names, extra)
    if radial is not None:
        return radial
    scaled = _scaled_radial(f, formula, names, extra)
    if scaled is not None:
        return scaled
    cylindrical = _cylindrical(f, formula, names, extra)
    if cylindrical is not None:
        return cylindrical
    found = _decomposed(node, f, formula, names, assumptions, extra)
    if found is not None:
        return found
    solved = _solved_bounds(f, formula, names, extra)
    if solved is not None:
        return solved
    for order in _other_orders(names):
        # a bound with no explicit form in the last variable (a root of
        # x**3 + x*y - 1 in x) may be explicit in another (y = (1 - x**3)/x)
        found = attempt(lambda: _decomposed(node, f, formula, order, assumptions, extra), settings.timeout)
        if found is not None:
            return found
    return node


def _other_orders(names: Sequence[Symbol]) -> list[list[Symbol]]:
    """The other orders of the variables to try: every permutation of up
    to three variables, the reversed order beyond."""
    if len(names) < 2:
        return []
    if len(names) > 3:
        return [list(reversed(names))]
    return [list(order) for order in permutations(names) if list(order) != list(names)]


def _decomposed(node: IntegralByRanges, f: Expr, formula: Boolean, names: Sequence[Symbol],
                assumptions: Assumptions, extra: list[Boolean]) -> Optional[Expr]:
    """The integral through the cylindrical algebraic decomposition, or
    ``None`` when the condition is not polynomial, a bound has no explicit
    form or an inner integral is not computed."""
    parameters = sorted_symbols(free_symbols(formula) - set(names))
    gens: list[Symbol] = parameters + list(names)
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    try:
        compiled = _compile(formula, gens, polys, index)
        cad = cylindrical_algebraic_decomposition(polys, gens)
    except (ValueError, PolynomialError, TypeError):
        return None
    m = len(parameters)
    values: dict[CADCell, Expr] = {}
    for cell in cad.cells:
        if not compiled(cell.signs):
            continue
        if any(i % 2 == 0 for i in cell.index[m:]):
            continue                                        # a section: measure zero
        base = _ancestor_at(cell, m) if m else cell
        parameter_condition = _parameter_condition(cell, cad, m) if m else true
        if m and ask(parameter_condition, assumptions) is False:
            continue                                        # a refuted case: its bounds need no explicit form
        stack = _stack(cell, cad, m + 1)
        if stack is None:
            return None
        found = _iterated(f, names, stack, extra + ([parameter_condition] if m else []))
        if found is None:
            return None
        value: Expr = found
        if m:
            refined = attempt(lambda: as_expr(refine(value, extra + [parameter_condition])), settings.timeout)
            if refined is not None:
                value = refined
        values[base] = as_expr(values.get(base, S.Zero) + value)
    return _cases(values, cad, m, assumptions)


def _cases(values: dict[CADCell, Expr], cad: CAD, m: int, assumptions: Assumptions) -> Expr:
    """The sum of the values of the cells over the parameter cells, as a
    ``Piecewise`` over the sign conditions of the parameter cells (the
    plain sum without parameters, ``m == 0``)."""
    if m == 0:
        total: Expr = S.Zero
        for value in values.values():
            total = total + value
        return as_expr(total)
    cases: list[tuple[Expr, Boolean]] = []
    for base in cad.cells_at(m):
        cond = _parameter_condition(base, cad, m)
        if ask(cond, assumptions) is False:
            continue
        cases.append((values.get(base, S.Zero), cond))
    if not cases:
        return S.Zero
    if len(cases) == 1:
        return cases[0][0]
    grouped: dict[Expr, list[Boolean]] = {}
    order: list[Expr] = []
    for value, cond in cases:
        if value not in grouped:
            grouped[value] = []
            order.append(value)
        grouped[value].append(cond)
    order.sort(key=lambda v: -len(grouped[v]))
    branches: list[tuple[Expr, Boolean]] = []
    for i, value in enumerate(order):
        cond = as_boolean(Or(*grouped[value])) if i + 1 < len(order) else true
        branches.append((value, cond))
    return as_expr(Piecewise(*branches))



def _pruned(condition: Boolean) -> Boolean:
    """The conjunction without the atoms implied by the others (the sign
    conditions of the projection polynomials of a cell repeat what the
    higher levels say: ``x**2 < 1`` under ``x**2 + y**2 < 1``)."""
    parts = [as_boolean(a) for a in (condition.args if isinstance(condition, And) else [condition])]
    kept = list(parts)
    for part in parts:
        rest = [other for other in kept if other is not part]
        if rest and _ask(part, rest) is True:
            kept = rest
    return as_boolean(And(*kept))


def _section_value(cell: CADCell, cad: CAD, m: int, f: Expr, names: Sequence[Symbol],
                   assumptions: list[Boolean]) -> Optional[Expr]:
    """The integral of ``f`` with the Hausdorff measure over a cell of
    dimension `n - k` which is a section at `k` levels: each section
    variable is the explicit branch `\\varphi_i` of its root, the earlier
    section variables substituted, and the integral of `f \\sqrt{\\det(I +
    G^T G)}` (`G` the Jacobian of the branches with respect to the free
    variables, the Gram determinant of the graph parametrisation; `1 +
    |\\nabla \\varphi|^2` for one equation) is taken over the free variables
    of the cell; ``None`` when a branch is not explicit or the integral
    is not computed."""
    n = len(cad.gens)
    sections = [level for level in range(m + 1, n + 1) if cell.index[level - 1] % 2 == 0]
    if not sections:
        return None
    branches: dict[Symbol, Expr] = {}
    for k in sections:
        node = _ancestor_at(cell, k)
        parent = node.parent
        gens = cad.gens[:k]
        roots = _root_sequence(parent, cad.projection[k - 1], gens) if parent is not None else []
        position = node.index[-1] // 2 - 1
        if not 0 <= position < len(roots) or parent is None:
            return None
        phi = _explicit_root(roots[position], parent, gens)
        if phi is None:
            return None
        branches[names[k - m - 1]] = as_expr(phi.xreplace(branches))
    solved = [names[k - m - 1] for k in sections]
    rest = [v for v in names if v not in solved]
    jacobian = Matrix([[as_expr(branches[v].diff(u)) for u in rest] for v in solved])
    gram = as_expr((eye(len(rest)) + jacobian.T * jacobian).det()) if rest else S.One
    simpler = attempt(lambda: as_expr(simplify(gram)), settings.timeout)
    if simpler is not None:
        gram = simpler
    g = as_expr(f.xreplace(branches) * sqrt(gram))
    if not rest:
        return g                                            # a point: the counting measure
    stack = _stack(cell, cad, m + 1, sections, branches)
    if stack is None:
        return None
    facts = list(assumptions)
    for v, (lower, upper) in zip(rest, stack.bounds):
        facts.extend([as_boolean(v > lower)] * (lower != -oo) + [as_boolean(v < upper)] * (upper != oo))
    g = _split_roots(g, facts)                              # sqrt(-1/(x**2 - 1)) sqrt(1 - x**2) = 1 on (-1, 1)
    if sections == [n] and len(rest) >= 2:
        # the base cell as a region (a disc under a surface goes to polar coordinates)
        condition = _pruned(_cell_condition(cell, cad, m + 1, n - 1))
        found = integrate_by_ranges(g, condition, rest, assumptions)
        if not found.has(Integral, IntegralByRanges):
            return found
    return _iterated(g, rest, stack, assumptions)


def _hausdorff(f: Expr, formula: Boolean, names: Sequence[Symbol],
               assumptions: Assumptions, extra: list[Boolean]) -> Optional[Expr]:
    """The integral with the `(n-k)`-dimensional Hausdorff measure on the
    variety of the `k` equations of the condition, through the
    decomposition: the sum of :func:`_section_value` over the cells of
    dimension `n - k` on which the condition holds; ``None`` when the
    condition has no equation, a branch is not explicit or an integral
    is not computed."""
    parts = list(formula.args) if isinstance(formula, And) else [formula]
    equations = sum(1 for part in parts if isinstance(part, Eq))
    if equations == 0:
        return None
    if not all(isinstance(part, (Eq, Lt, Le, Gt, Ge)) for part in parts):
        return None
    parameters = sorted_symbols(free_symbols(formula) - set(names))
    gens: list[Symbol] = parameters + list(names)
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    try:
        compiled = _compile(formula, gens, polys, index)
        cad = cylindrical_algebraic_decomposition(polys, gens)
    except (ValueError, PolynomialError, TypeError):
        return None
    m = len(parameters)
    values: dict[CADCell, Expr] = {}
    for cell in cad.cells:
        if not compiled(cell.signs):
            continue
        if sum(1 for i in cell.index[m:] if i % 2 == 0) != equations:
            continue                                        # not of dimension n - k: measure zero
        base = _ancestor_at(cell, m) if m else cell
        parameter_condition = _parameter_condition(cell, cad, m) if m else true
        if m and ask(parameter_condition, assumptions) is False:
            continue
        found = _section_value(cell, cad, m, f, names, extra + ([parameter_condition] if m else []))
        if found is None:
            return None
        value: Expr = found
        if m:
            refined = attempt(lambda: as_expr(refine(value, extra + [parameter_condition])), settings.timeout)
            if refined is not None:
                value = refined
        values[base] = as_expr(values.get(base, S.Zero) + value)
    return _cases(values, cad, m, assumptions)
