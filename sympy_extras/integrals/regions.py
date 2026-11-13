r"""Integrals over regions given by inequalities, through cylindrical
algebraic decomposition.

``IntegralByRanges(f, condition)`` is the integral of ``f`` over the
semialgebraic set `\{(x_1, \ldots, x_n) : \text{condition}\}`, the
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
   of the projection polynomials of level `k` (or `\pm\infty`). Each root
   is a continuous algebraic function of `x_1, \ldots, x_{k-1}` over the
   cell (delineability); it is written explicitly by solving the
   polynomial for `x_k` and taking the branch which agrees with the root
   at the sample point of the cell (Mathematica uses parametric ``Root``
   objects here; when no explicit branch is found the integral is left
   unevaluated);
4. the iterated integral is computed innermost variable first by the
   one-dimensional integrator (by default
   :func:`~sympy_extras.integrals.definite.definite_integral`), the outer
   bounds being passed as assumptions, and the cells are summed;
5. the cells of the parameter space give a case distinction: the result
   is a ``Piecewise`` whose conditions are the sign conditions of the
   projection polynomials on each parameter cell.

Four routes are tried around the decomposition [Apostol]_:

* **polar coordinates** first: a disc or an annulus (`r_1^2 < x^2 + y^2 <
  R^2`, or the ball in three variables) with an integrand depending on the
  point only through `x^2 + y^2` is `2\pi \int_{r_1}^R g(\rho)\, \rho\, d\rho`
  (`4\pi \int g(\rho)\, \rho^2\, d\rho` for the ball);
* **an affine change of variables** to the polar case: an ellipse or an
  ellipsoid, and any region `\sum c_i x_i^2 + \sum b_i x_i < v` with
  coefficients of one sign (a diagonal quadratic form, the square
  completed), is the image of a disc or ball under `x_i = u_i/\sqrt{c_i} -
  b_i/(2 c_i)`, with the Jacobian `\prod c_i^{-1/2}`; the polar route is
  applied to the transformed integrand when it depends on `\sum u_i^2`
  only (the area `\pi a b` of the ellipse `x^2/a^2 + y^2/b^2 < 1`);
* **cylindrical coordinates** in three variables: a disc or annulus
  condition on `x^2 + y^2` (or none) together with bounds on `z` which are
  linear in `z` and depend on `x, y` only through `x^2 + y^2`, and an
  integrand `g(x^2 + y^2, z)`, give `2\pi \int \int g(\rho^2, z)\, \rho\, dz\, d\rho`,
  the range of `\rho` being where the bounds on `z` are in the right order
  (the paraboloid `0 < z < x^2 + y^2 < 1`, the cone `\sqrt{x^2 + y^2} < z < 1`);
* **bounds solved for the last variable** when the decomposition does not
  apply (a condition which is not polynomial, `y < \exp(x)`) or fails: a
  conjunction of relations linear in the last variable `y` describes, over
  each point of the other variables, the interval `\max(\text{lower}) < y <
  \min(\text{upper})`; the range of the other variables is cut where the
  two bounds cross (with :func:`sympy_extras.assumptions.solve.solve`), the
  inner integral is computed by the integrator and the outer integral by
  the same function on the remaining variables (Fubini's theorem); an
  outer variable without bounds ranges over the whole line, and the outer
  integral over an infinite range is again a matter for the integrator.

A polynomial over a bounded polytope (a conjunction of inequalities
linear in the variables, with parameters allowed in the constant terms)
is integrated exactly without the decomposition [Lasserre]_
[Baldoni]_: the vertices are enumerated from the facets, the polytope
is triangulated by the pulling triangulation from a vertex (cones from
the vertex over the pulling triangulations of the facets not containing
it), each simplex is mapped affinely onto the standard simplex and the
monomials are integrated by Dirichlet's formula `\int_\Delta u^a\, du =
\prod a_i! / (|a| + d)!`, times the Jacobian of the map. The
exponential integrals of Brion's theorem [Lasserre]_ are the generating
function of the same sums; the simplex formula of [Baldoni]_ for powers
of linear forms is the case of a single monomial after the map. The
route is taken before the decomposition and agrees with it (the tests
compare the two).

With ``dimension=n``, ``n`` an integer or a symbol, a radial integrand
`f(r)` over a ball, an annulus or the whole space of `\mathbb{R}^n`,
described by a condition on the radial variable `r` alone (`r < R`,
`(a < r) \wedge (r < R)`, ``True``), is
`\frac{2\pi^{n/2}}{\Gamma(n/2)} \int f(r)\, r^{n-1}\, dr`, the area of the
unit sphere times the radial integral; the decomposition is skipped, so
the dimension may be symbolic (the volume `\pi^{n/2}/\Gamma(n/2 + 1)` of
the unit ball). With an integer dimension and the condition written in
the variables, the polar route below applies in any dimension.

The bounds solved for the last variable need not be linear in it: a
relation which :func:`sympy_extras.assumptions.solve.solve` turns into an
interval of `y` with explicit endpoints (`y^2 < e^x` with `y > 0` into
`y < e^{x/2}`, `e^y < x` into `y < \log x`, `y^3 < x` into `y <
x^{1/3}`) contributes its endpoints to the bounds.

With ``measure='hausdorff'`` the integral is taken with respect to the
`(n-k)`-dimensional Hausdorff measure on the variety described by the
`k` *equations* of the condition, cut by its inequalities: the length
of a curve and the area of a surface, with the integrand weighted along
them (Mathematica integrates over ``ImplicitRegion`` this way). The
cells of the decomposition on which the condition holds are then the
sections of dimension `n - 1`, on each of which one variable `x_k` is an
explicit branch `x_k = \varphi(x_1, \ldots, x_{k-1})` of a root of the
equation (:func:`_explicit_root`), and the integral is the integral of
`f(\ldots, \varphi, \ldots) \sqrt{1 + \sum_j (\partial \varphi / \partial x_j)^2}`
over the cell of the other variables [Apostol]_ (chapter 12): the arc
length `\int \sqrt{1 + \varphi'(x)^2}\, dx` of a graph `y = \varphi(x)`, the
area `\int\int \sqrt{1 + \varphi_x^2 + \varphi_y^2}\, dx\, dy` of a graph
`z = \varphi(x, y)`; the outer integral over the base cell is again an
integral over a region, so that a surface over a disc is done in polar
coordinates. With `k` equations (a curve in space as the intersection of
two surfaces) the cells are sections at `k` levels, each section
variable an explicit branch with the earlier ones substituted, and the
weight is the Gram determinant `\sqrt{\det(I + G^T G)}` of the graph
parametrisation, `G` the Jacobian of the branches with respect to the
free variables (`\sqrt{1 + x'(t)^2 + y'(t)^2}` for a curve `(x(t), y(t),
t)`). When a root has no explicit branch, or the condition has no
equation, the integral is left unevaluated.

A bound which is a root of a cubic in the last variable is written in
Viète's trigonometric or hyperbolic form (:func:`_trigonometric_roots`),
real where Cardano's formula needs complex radicals; a bound without an
explicit form in the last variable is retried with the variables in
another order, where it may be explicit (`x^3 + x y < 1` is `y < (1 -
x^3)/x`), and a numeric bound is an algebraic number (an
:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`, SymPy's
``CRootOf`` on the face) in the value.

The module works on the objects of :mod:`sympy_extras.core` and imports
no SymPy: the native decomposition (:mod:`sympy_extras.polys.cad`), the
native ``ask``, ``refine``, ``solve`` and ``simplify``, the roots of
:mod:`sympy_extras.polys.polyroots` for the explicit branches, the
native simplex method for the boundedness of a polytope, and the region
integrals with a rotational symmetry of :mod:`.axisymmetric`. The
one-dimensional integrals are the business of a typed
:data:`~sympy_extras.integrals.axisymmetric.Integrator`, a parameter of
:func:`integrate_by_ranges`: by default the native
:func:`~sympy_extras.integrals.definite.definite_integral` (which loads
SymPy only for SymPy's ``integrate``, its last method); an integrator
written on the core (polynomials by their antiderivatives) runs the
decomposition and the iterated integrals without SymPy. The solutions of a polynomial equation
for the branches follow the polynomial case of SymPy's ``solve``
(``sympy.solvers.solvers._solve``), ported. The functions on SymPy
objects are in :mod:`.regions_sympy`.

Examples
========

>>> from sympy_extras.core import Symbol, Integer, eq
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.regions import IntegralByRanges, integrate_by_ranges
>>> x, y, z, r = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('r')
>>> IntegralByRanges(Integer(1), x**2 + y**2 < 1)
IntegralByRanges(1, x**2 + y**2 < 1, (x, y))
>>> IntegralByRanges(Integer(1), x**2 + y**2 < 1).doit()
pi
>>> integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1))
1/24
>>> integrate_by_ranges(Integer(1), x**2 + y**2 < r**2, [x, y], r > 0)
pi*r**2
>>> integrate_by_ranges(exp(-x**2 - y**2), x**2 + y**2 < 1)
-pi*exp(-1) + pi
>>> integrate_by_ranges(Integer(1), (x > 0) & (x < 1) & (y > 0) & (y < exp(x)))
-1 + E
>>> a, b = Symbol('a', positive=True), Symbol('b', positive=True)
>>> integrate_by_ranges(Integer(1), x**2/a**2 + y**2/b**2 < 1, [x, y])
pi*a*b
>>> integrate_by_ranges(Integer(1), (x**2 + y**2 < 1) & (z > 0) & (z < x**2 + y**2))
pi/2

The length of the unit circle, with the Hausdorff measure, and the area
under the cubic `y = x^3 - x` between its root and `x = 0`, with the
variables ordered so that the boundary root is in the last variable:

>>> integrate_by_ranges(Integer(1), eq(x**2 + y**2, 1), measure='hausdorff')
2*pi
>>> integrate_by_ranges(Integer(1), (x > 0) & (y > 0) & (y < 1) & (x**3 - x - y < 0), [y, x])
-IndexedRoot(x**3 - x - 1, x, 0)**4/4 - 1/4 + IndexedRoot(x**3 - x - 1, x, 0)**2/2 + IndexedRoot(x**3 - x - 1, x, 0)

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
.. [Baldoni] V. Baldoni, N. Berline, J. A. De Loera, M. Köppe, M. Vergne,
   *How to integrate a polynomial over a simplex*, Mathematics of
   Computation 80 (2011), pp. 297-325.
.. [Collins] G. E. Collins, *Quantifier elimination for real closed
   fields by cylindrical algebraic decomposition*, Automata Theory and
   Formal Languages, Lecture Notes in Computer Science 33, Springer,
   1975, pp. 134-183.
.. [Strzebonski] A. Strzeboński, *Cylindrical algebraic decomposition
   using validated numerics*, Journal of Symbolic Computation 41 (2006),
   pp. 1021-1038 (the decomposition behind Mathematica's ``Integrate``
   over regions and ``CylindricalDecomposition``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/solvers.py (_solve, the polynomial case) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on core expressions, the roots of polys.polyroots
# (_solutions only); the rest of the module is original code.
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from itertools import combinations, permutations
from typing import Final, Optional, Sequence, Union

import mpmath

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element, normalize
from sympy_extras.assumptions.refine import refine
from sympy_extras.assumptions.solve import solve
from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Application, expand
from sympy_extras.core.logic import And, Boolean, BooleanFalse, BooleanLike, BooleanTrue, and_, false, or_, to_boolean, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import I, Integer, Rational, number_from_fraction, oo, pi
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq, gt, lt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs, Sign, abs_, sign
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Acosh, Asinh, Atanh, Cosh, Sinh, Tanh, acosh, asinh, cosh, sinh
from sympy_extras.functions.elementary.integers import Ceiling, Floor
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.elementary.trigonometric import Acos, Asin, Atan, Cos, Sin, Tan, acos, cos
from sympy_extras.functions.special.gamma_functions import gamma
from sympy_extras.polys.cad.indexedroot import real_root
from sympy_extras.polys.cad.lifting import CAD, CADCell, cylindrical_algebraic_decomposition
from sympy_extras.polys.cad.qe import _compile, _Compiled
from sympy_extras.polys.cad.samplepoints import compare_real
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, factor
from sympy_extras.polys.roots import radical_form
from sympy_extras.sets.sets import EmptySetClass, FiniteSet, Interval, Reals, Set, Union as SetUnion, intersection, \
    interval
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers._special_values import determinant
from sympy_extras.solvers.simplex import linear_program

from .axisymmetric import Integrator, axisymmetric_integral, rotation_invariant
from .conditions import numerically_equal

__all__ = ['IntegralByRanges', 'integrate_by_ranges', 'Integrator']

#: the numerical tolerance for matching a branch to a root at the sample point
_TOLERANCE = mpmath.mpf('1e-20')

#: the measures of integration: the Lebesgue measure of `\\mathbb{R}^n` and
#: the `(n-1)`-dimensional Hausdorff measure on the hypersurface of the
#: equation of the condition
_MEASURES = ('lebesgue', 'hausdorff')

#: the working precision (bits) of the numerical matching of the branches
_PREC = 120

_COUNTER: Final = itertools.count()

#: a polynomial in the integration variables with coefficients free of
#: them: the exponents of each term and its coefficient
Coefficients = dict[tuple[int, ...], Expr]


def _fresh(name: str, positive: bool = False) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so two dummies of one name would be one symbol."""
    if positive:
        return Dummy('%s%d' % (name, next(_COUNTER)), positive=True)
    return Dummy('%s%d' % (name, next(_COUNTER)))


def _evaluated(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e.xreplace(rule)`` with the rebuilt nodes evaluated (the core's
    ``xreplace`` does not evaluate)."""
    return as_expr(e._xreplace_evaluated(rule))


def _symbol(v: Basic) -> Symbol:
    if not isinstance(v, Symbol):
        raise TypeError("expected a symbol, got %s" % (v,))
    return v


def _ops(e: Expr) -> int:
    """The number of operations of ``e`` as a Python integer."""
    count = e.count_ops()
    if not isinstance(count, Integer):
        raise TypeError("count_ops gave %s" % (count,))
    return count.p


def _definite(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: list[Boolean]) -> Optional[Expr]:
    """The default integrator: :func:`~sympy_extras.integrals.definite.definite_integral`
    without the cases of the special values of the parameters; ``None``
    when the integral is not computed (it keeps an ``Integral``). The
    ``ValueError`` of bounds which contradict the assumptions goes on to
    the caller (an empty cell)."""
    from .definite import definite_integral
    from .integral import is_unevaluated
    value = definite_integral(f, (x, a, b), list(assumptions), special_values=False)
    if is_unevaluated(value) or value.has(IntegralByRanges):
        return None
    return value


class IntegralByRanges(Expr):
    """The integral of ``integrand`` over the region described by
    ``condition``, a Boolean combination of polynomial inequalities and
    equations in the integration ``variables`` (and parameters).

    The constructor checks and stores its arguments and does not
    evaluate: :meth:`doit` (or :func:`integrate_by_ranges`) computes the
    integral.

    ``measure='hausdorff'`` integrates with respect to the
    `(n-1)`-dimensional Hausdorff measure on the hypersurface given by
    the equation of the condition (arc length, surface area), see the
    module documentation.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, eq, Integer
    >>> from sympy_extras.integrals.regions import IntegralByRanges
    >>> x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    >>> region = IntegralByRanges(x*y, (x > 0) & (x < y) & (y < 1))
    >>> region.integrand, region.condition, region.variables
    (x*y, (x > 0) & (x < y) & (y < 1), (x, y))
    >>> region.doit()
    1/8
    >>> IntegralByRanges(Integer(1), x**2 + y**2 < a**2, parameters=[a]).variables
    (x, y)
    >>> circle = IntegralByRanges(Integer(1), eq(x**2 + y**2, 1), measure='hausdorff')
    >>> circle.measure, circle.free_symbols
    ('hausdorff', set())
    """

    __slots__ = ('_integrand', '_condition', '_variables', '_measure', '_dimension')

    def __init__(self, integrand: ExprLike, condition: BooleanLike,
                 variables: Union[None, Sequence[Symbol], Tuple] = None,
                 measure: Union[str, Symbol] = 'lebesgue',
                 dimension: Optional[ExprLike] = None,
                 parameters: Optional[Sequence[Symbol]] = None) -> None:
        # rebuilt from its arguments (func(*args)): the variables a Tuple,
        # the measure the name of a symbol
        if isinstance(measure, Symbol):
            measure = measure.name
        integrand_ = to_expr(integrand)
        condition_ = to_boolean(condition)
        excluded = [_symbol(p) for p in parameters] if parameters is not None else []
        if variables is None:
            names = [v for v in sorted(condition_.free_symbols, key=lambda s: s.name) if v not in excluded]
            if dimension is not None:
                names = [_radial_symbol(names)]
        else:
            names = [_symbol(v) for v in (variables.args if isinstance(variables, Tuple) else variables)]
            clash = [v for v in names if v in excluded]
            if clash:
                raise ValueError("%s cannot be both an integration variable and a parameter" % (clash[0],))
        if not names:
            raise ValueError("no integration variable: the condition has no symbols")
        if measure not in _MEASURES:
            raise ValueError("measure must be one of %s, got %r" % (", ".join(_MEASURES), measure))
        if dimension is not None and len(names) != 1:
            raise ValueError("with a dimension the integral is radial, in one variable")
        self._integrand: Final[Expr] = integrand_
        self._condition: Final[Boolean] = condition_
        self._variables: Final[tuple[Symbol, ...]] = tuple(names)
        self._measure: Final[str] = measure
        self._dimension: Final[Optional[Expr]] = None if dimension is None else to_expr(dimension)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        args: list[Basic] = [self._integrand, self._condition, Tuple(*self._variables)]
        if self._measure != _MEASURES[0] or self._dimension is not None:
            args.append(Symbol(self._measure))
        if self._dimension is not None:
            args.append(self._dimension)
        return tuple(args)

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        if len(args) < 3:
            raise TypeError("invalid arguments of an IntegralByRanges: %s" % (args,))
        f, condition, names = args[0], args[1], args[2]
        if not (isinstance(f, Expr) and isinstance(condition, Boolean) and isinstance(names, Tuple)):
            raise TypeError("invalid arguments of an IntegralByRanges: %s" % (args,))
        measure = _MEASURES[0]
        if len(args) > 3:
            label = args[3]
            if not isinstance(label, Symbol):
                raise TypeError("the measure of an IntegralByRanges is a name, got %s" % (label,))
            measure = label.name
        dimension: Optional[Expr] = None
        if len(args) > 4:
            dimension = as_expr(args[4])
        return IntegralByRanges(f, condition, [_symbol(v) for v in names.args], measure, dimension)

    @property
    def integrand(self) -> Expr:
        return self._integrand

    @property
    def condition(self) -> Boolean:
        return self._condition

    @property
    def variables(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def measure(self) -> str:
        """``'lebesgue'`` or ``'hausdorff'``."""
        return self._measure

    @property
    def dimension(self) -> Optional[Expr]:
        """The dimension of the space of a radial integral, ``None`` for
        an integral in the variables themselves."""
        return self._dimension

    @property
    def free_symbols(self) -> set[Symbol]:
        """The symbols of the integrand and the condition other than the
        integration variables, which are bound."""
        symbols = set(self._integrand.free_symbols) | set(self._condition.free_symbols)
        if self._dimension is not None:
            symbols |= set(self._dimension.free_symbols)
        return symbols - set(self._variables)

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return self._variables

    def _is_number_node(self) -> Optional[bool]:
        return not self.free_symbols

    def _eval_facts(self) -> dict[str, bool]:
        return {}

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # a bound variable is not substituted
        if self == old:
            return new
        if old in self._variables:
            return self
        return Expr._subs(self, old, new)

    def doit(self) -> Expr:
        """The value, or the integral unchanged when it cannot be computed."""
        return integrate_by_ranges(self._integrand, self._condition, self._variables, None,
                                   self._measure, self._dimension)


def _radial_symbol(names: Sequence[Symbol]) -> Symbol:
    """The radial variable among the symbols of a condition: the one
    named ``r``, or the only one."""
    for name in names:
        if name.name == 'r':
            return name
    if len(names) == 1:
        return names[0]
    if not names:
        return Symbol('r')
    raise ValueError("the radial variable of a condition in %s must be given" % (list(names),))


# ---------------------------------------------------------------------------
# polynomials in the integration variables with any coefficients

def _coefficients(e: Expr, names: Sequence[Symbol]) -> Optional[Coefficients]:
    """``e`` as a polynomial in ``names`` whose coefficients are free of
    them (SymPy's ``Poly(e, *names)`` with any coefficient domain): the
    exponents of each term and its coefficient, the zero terms dropped;
    ``None`` when ``e`` is not a polynomial in ``names``."""
    terms: Coefficients = {}
    for term in Add.make_args(expand(e)):
        coefficient, dependent = term.as_independent(*names, as_Add=False)
        exponents = [0] * len(names)
        for part in Mul.make_args(dependent):
            if part == 1:
                continue
            base, exponent = part.as_base_exp()
            if base not in names or not isinstance(exponent, Integer) or exponent.p < 0:
                return None
            exponents[list(names).index(_symbol(base))] += exponent.p
        key = tuple(exponents)
        terms[key] = terms[key] + coefficient if key in terms else coefficient
    return {k: v for k, v in terms.items() if v != 0}


def _coefficient(terms: Coefficients, key: tuple[int, ...]) -> Expr:
    return terms.get(key, S.Zero)


def _total_degree(terms: Coefficients) -> int:
    return max((sum(k) for k in terms), default=-1)


def _unit(n: int, i: int, e: int) -> tuple[int, ...]:
    """The exponents of ``x_i**e`` among ``n`` variables."""
    return tuple(e if j == i else 0 for j in range(n))


class _Bound:
    """A root of a polynomial in ``x_k`` over a cell: the ``index``-th real
    root of ``poly`` at the sample point, and its explicit form."""

    __slots__ = ('poly', 'index', 'value')

    def __init__(self, poly: Poly, index: int, value: Algebraic) -> None:
        self.poly: Final[Poly] = poly
        self.index: Final[int] = index
        self.value: Final[Algebraic] = value


class _Stack:
    """The bounds of one cell of full dimension, level by level, as
    explicit expressions (``-oo``/``oo`` for an unbounded sector)."""

    __slots__ = ('cell', 'bounds')

    def __init__(self, cell: CADCell, bounds: list[tuple[Expr, Expr]]) -> None:
        self.cell: Final[CADCell] = cell
        self.bounds: Final[list[tuple[Expr, Expr]]] = bounds


class _Key:
    __slots__ = ('value',)

    def __init__(self, value: Algebraic) -> None:
        self.value: Final[Algebraic] = value

    def __lt__(self, other: _Key) -> bool:
        return compare_real(self.value, other.value) < 0


def _root_sequence(parent: CADCell, polys: Sequence[Poly], gens: Sequence[Expr]) -> list[_Bound]:
    """The sorted distinct real roots of the polynomials of a level over
    the sample point of the parent cell, each attributed to the first
    polynomial having it."""
    found: list[_Bound] = []
    for poly in polys:
        roots_ = parent.sample.real_roots(poly, gens)
        if roots_ is None:
            continue
        for index, root in enumerate(roots_):
            if not any(compare_real(root, b.value) == 0 for b in found):
                found.append(_Bound(poly, index, root))
    found.sort(key=lambda b: _Key(b.value))
    return found


def _number(a: Algebraic, name: str) -> Expr:
    """A real algebraic number as an expression: a rational number, else
    the root object of its minimal polynomial in the variable ``name``
    (SymPy's ``CRootOf`` on the face)."""
    return real_root(a, name)


def _explicit_root(bound: _Bound, parent: CADCell, gens: Sequence[Expr]) -> Optional[Expr]:
    """The root as an expression in the earlier variables: the solution of
    the polynomial for the last generator which takes the value of the
    root at the sample point of the parent cell."""
    x = _symbol(gens[-1])
    equation = bound.poly.as_expr()
    if len(gens) == 1 or not equation.has(*gens[:-1]):
        return _radical_form(_number(bound.value, x.name))
    point: dict[Symbol, Expr] = {_symbol(g): _number(p, _symbol(g).name) for g, p in zip(gens[:-1], parent.point)}
    target = numeric_value(_number(bound.value, x.name), _PREC)
    if target is None:
        return None
    trigonometric = _trigonometric_roots(bound.poly, x)
    for candidates in (trigonometric, attempt(lambda: _solutions(equation, x), settings.timeout, native=True)):
        if candidates is None:
            continue
        matches: list[Expr] = []
        for c in candidates:
            value = reliable_value(c, 30, point)
            if value is None:
                # a form for the other sign of p, or a value whose branch a
                # rounding error chooses (see reliable_form)
                continue
            z = numeric_value(value, _PREC)
            if z is None or abs(z.imag) >= _TOLERANCE:
                continue                                    # Cardano's form of a real root has a tiny imaginary part
            if abs(z.real - target.real) < _TOLERANCE:
                matches.append(c)
        if len(matches) == 1:
            return matches[0]
    return None


# PORTED FROM SYMPY: sympy/solvers/solvers.py:_solve
def _solutions(equation: Expr, x: Symbol) -> list[Expr]:
    """The solutions of a polynomial equation in ``x`` as SymPy's
    ``solve`` gives them: the explicit roots, simplified when there are
    at most two of them."""
    found = roots(equation, x)
    if not isinstance(found, dict):
        return []
    candidates = list(found)
    if len(candidates) <= 2:
        candidates = [simplify(c) for c in candidates]
    return candidates


def _radical_form(root: Expr) -> Expr:
    """A numeric root object of degree at most four written in radicals
    when the radicals are real and short, else the root object itself
    (the bug: the bound ``CRootOf(4*x**2 - 3, 1)`` of a cell stayed a
    ``CRootOf`` where ``sqrt(3)/2`` was meant, and ``definite_integral``
    left the integral up to it unevaluated).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.cad.indexedroot import indexed_root
    >>> from sympy_extras.integrals.regions import _radical_form
    >>> x = Symbol('x')
    >>> _radical_form(indexed_root(4*x**2 - 3, x, 1))
    sqrt(3)/2
    >>> _radical_form(indexed_root(x**5 - x - 1, x, 0))
    IndexedRoot(x**5 - x - 1, x, 0)
    """
    from sympy_extras.polys.cad.indexedroot import IndexedRoot
    if not isinstance(root, IndexedRoot):
        return root
    terms = _coefficients(root.expr, [root.variable])
    if terms is None or _total_degree(terms) > 4:
        return root
    candidate = attempt(lambda: radical_form(root), settings.timeout, native=True)
    if candidate is None or isinstance(candidate, IndexedRoot) or candidate.has(I) or len(str(candidate)) > 200:
        return root
    return candidate


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

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.polytools import Poly
    >>> from sympy_extras.integrals.regions import _trigonometric_roots
    >>> x = Symbol('x')
    >>> [r.evalf(6) for r in _trigonometric_roots(Poly(x**3 - 3*x + 1, x), x)[:3]]
    [1.53209, 0.347296, -1.87939]
    >>> _trigonometric_roots(Poly(x**3 + x - 1, x), x)[3].evalf(6)
    0.682328
    """
    terms = _coefficients(poly.as_expr(), [x])
    if terms is None or _total_degree(terms) != 3:
        return None
    a, b, c, d = (_coefficient(terms, (k,)) for k in (3, 2, 1, 0))
    p = (3 * a * c - b**2) / (3 * a**2)
    q = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (27 * a**3)
    if p == 0:
        return None
    shift = -b / (3 * a)
    radius = 2 * sqrt(-p / 3)
    angle = acos(3 * q * sqrt(-3 / p) / (2 * p)) / 3
    found = [radius * cos(angle - 2 * pi * k / 3) + shift for k in range(3)]
    found.append(-2 * sqrt(p / 3) * sinh(asinh(3 * q * sqrt(3 / p) / (2 * p)) / 3) + shift)
    found.append(-2 * sign(q) * sqrt(-p / 3) * cosh(acosh(-3 * abs_(q) * sqrt(-3 / p) / (2 * p)) / 3) + shift)
    return found


def _stack(cell: CADCell, cad: CAD, first: int, skip: Sequence[int] = (),
           replace: Optional[dict[Basic, Basic]] = None) -> Optional[_Stack]:
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
        roots_ = _root_sequence(parent, cad.projection[level - 1], gens)
        position = (node.index[-1] - 1) // 2          # the sector lies between root position-1 and position
        lower: Expr = S.NegativeInfinity
        upper: Expr = S.Infinity
        if position >= 1:
            found = _explicit_root(roots_[position - 1], parent, gens)
            if found is None:
                return None
            lower = found
        if position < len(roots_):
            found = _explicit_root(roots_[position], parent, gens)
            if found is None:
                return None
            upper = found
        if replace:
            lower = _evaluated(lower, replace)
            upper = _evaluated(upper, replace)
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
    ``sqrt(1 - x)*sqrt(1 + x)*sqrt(1 - u**2)`` when ``-1 < x < 1``, so that the
    integrand of an inner integral has the parameters outside the
    radicals.

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.integrals.regions import _split_roots
    >>> x, u = Symbol('x'), Symbol('u')
    >>> _split_roots(sqrt((1 - x**2)*(1 - u**2)), [x > -1, x < 1])
    sqrt(-(u - 1)*(u + 1))*sqrt(1 - x)*sqrt(x + 1)
    """
    def rewrite(node: Basic) -> Basic:
        if not isinstance(node, Pow):
            return node
        exponent = node.exp
        if not isinstance(exponent, Rational) or exponent.q == 1:
            return node
        found = attempt(lambda: factor(node.base), None, native=True)
        if found is None:
            return node
        base = found
        factors = list(base.args) if isinstance(base, Mul) else [base]
        positive: list[tuple[Expr, Expr]] = []             # (a factor known positive, its exponent)
        rest: list[Expr] = []
        sign_ = 1
        for f in factors:
            f_ = as_expr(f)
            power: Expr = S.One
            if isinstance(f, Pow) and f.exp.is_integer:
                # a reciprocal (x - 1)**-1 is split by the sign of x - 1
                f_, power = f.base, f.exp
            if f_.is_positive or (not f_.is_number and _ask(f_ > 0, assumptions) is True):
                positive.append((f_, power))
            elif f_.is_negative or (not f_.is_number and _ask(f_ < 0, assumptions) is True):
                positive.append((-f_, power))
                if power.is_odd:
                    sign_ = -sign_
            else:
                rest.append(f_**power)
        if not positive:
            return node
        result: Expr = S.One
        for f_, power in positive:
            result = result * f_**(power * exponent)
        product: Expr = S.One
        for f_ in rest:
            product = product * f_
        remaining = Integer(sign_) * product
        return result * remaining**exponent

    if not expr.has(Pow):
        return expr
    return as_expr(expr.replace(lambda n: isinstance(n, Pow), rewrite))


def _iterated(integrand: Expr, variables: Sequence[Symbol], stack: _Stack,
              assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
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
                outer.append(variables[j] > lo)
            if hi != oo:
                outer.append(variables[j] < hi)
        if not value.has(x):
            if lower == -oo or upper == oo:
                return None
            value = value * (upper - lower)
            continue
        inner = list(outer)
        if lower != -oo:
            inner.append(x > lower)
        if upper != oo:
            inner.append(x < upper)
        if any(isinstance(part, BooleanFalse) for part in inner):
            return S.Zero                                   # an empty cell under its parameter condition
        piece = _split_roots(value, inner)
        try:
            result = integrator(piece, x, lower, upper, outer)
            if piece != value and (result is None or result.has(I) or _unwieldy(result)):
                # the split roots sqrt(-a - x)*sqrt(-a + x) under a < 0 gave
                # a complex form of pi*a**2/2 where the product sqrt((a - x)*(a + x)) gives it plainly
                unsplit = integrator(value, x, lower, upper, outer)
                if unsplit is not None and (result is None or _ops(unsplit) < _ops(result)):
                    result = unsplit
        except ValueError:
            return S.Zero                                   # contradictory bounds: an empty cell
        if result is None and lower == -upper and upper != oo and _ask(upper > 0, outer) is True:
            u = _fresh('u')
            rescaled = _split_roots(value.subs(x, upper * u) * upper, outer + [upper > 0, u > -1, u < 1])
            result = integrator(rescaled, u, S.NegativeOne, S.One, outer)
        if result is None:
            return None
        value = result
    return value


def _unwieldy(value: Expr) -> bool:
    """Whether ``value`` has logarithms of radicals or more than a few
    dozen operations, the marks of an antiderivative taken along a
    complex branch."""
    return _ops(value) > 40 or any(node.args[0].has(Pow) for node in value.atoms(Log))


def _parameter_condition(cell: CADCell, cad: CAD, levels: int, explicit: bool = False) -> Boolean:
    """The condition of the parameter cell of ``cell``: the sign
    conditions of the projection polynomials of the first ``levels``
    levels, which the integrals of the cell take as assumptions; with
    ``explicit``, the cell bounded by its roots in explicit form (``a >
    sqrt(2)``) where they have one, for the labels of the ``Piecewise``:
    the signs alone do not tell the cells apart (``a**2 - 2 > 0`` holds
    on ``a < -sqrt(2)`` and on ``a > sqrt(2)``, and the value of one was
    attached to the other), and the explicit bounds are poor assumptions
    (the integral over ``(-sqrt(2), a)`` under ``a < sqrt(2)`` was not
    computed)."""
    return _cell_condition(cell, cad, 1, levels, explicit=explicit)


def _cell_condition(cell: CADCell, cad: CAD, first: int, last: int, explicit: bool = False) -> Boolean:
    """The sign conditions of the projection polynomials of the levels
    ``first`` to ``last`` (1-based) at the ancestor of ``cell`` of level
    ``last``; with ``explicit``, each level's variable between its roots
    in explicit form instead, where the roots have one."""
    levels = last
    parts: list[Boolean] = []
    ancestor: Optional[CADCell] = cell
    while ancestor is not None and ancestor.level > levels:
        ancestor = ancestor.parent
    if ancestor is None:
        return true
    for level in range(first, levels + 1):
        node = _ancestor_at(ancestor, level)
        if explicit:
            found = _explicit_cell(node, cad, level)
            if found is not None:
                parts.extend(found)
                continue
        gens = cad.gens[:level]
        for poly in cad.projection[level - 1]:
            sign_ = node.sample.sign(poly, gens)
            expression = poly.as_expr()
            if sign_ > 0:
                parts.append(gt(expression, 0))
            elif sign_ < 0:
                parts.append(lt(expression, 0))
            else:
                parts.append(eq(expression, 0))
    return and_(*parts)


def _explicit_cell(node: CADCell, cad: CAD, level: int) -> Optional[list[Boolean]]:
    """The relations placing the variable of ``level`` in the cell
    ``node`` of that level: between its neighbouring roots for a sector,
    equal to its root for a section, the roots as expressions in the
    earlier variables; ``None`` when a root has no explicit form."""
    parent = node.parent
    if parent is None:
        return None
    gens = cad.gens[:level]
    x = gens[-1]
    roots_ = _root_sequence(parent, cad.projection[level - 1], gens)
    parts: list[Boolean] = []
    if node.is_section:
        position = node.index[-1] // 2 - 1
        if not 0 <= position < len(roots_):
            return None
        root = _explicit_root(roots_[position], parent, gens)
        if root is None:
            return None
        parts.append(eq(x, root))
        return parts
    position = (node.index[-1] - 1) // 2
    if position >= 1:
        lower = _explicit_root(roots_[position - 1], parent, gens)
        if lower is None:
            return None
        parts.append(x > lower)
    if position < len(roots_):
        upper = _explicit_root(roots_[position], parent, gens)
        if upper is None:
            return None
        parts.append(x < upper)
    return parts


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


def _radius(names: Sequence[Symbol]) -> Expr:
    total: Expr = S.Zero
    for v in names:
        total = total + v**2
    return total


def _radial_atom(atom: Relational, names: Sequence[Symbol]) -> Optional[tuple[bool, Expr]]:
    """``(upper, value)`` when the relation bounds ``x**2 + y**2 (+ z**2)``:
    ``upper`` for a bound from above (``rho**2 < value``), else from
    below; ``None`` when the relation is not of that form."""
    difference = atom.lhs - atom.rhs
    terms = _coefficients(difference, names)
    if terms is None:
        return None
    coefficient = _coefficient(terms, _unit(len(names), 0, 2))
    if coefficient == 0:
        return None
    rest = expand(difference - coefficient * _radius(names))
    if rest.has(*names):
        return None
    # coefficient * rho**2 + rest  <  0  (or <=, >, >=)
    bound = -rest / coefficient
    less = isinstance(atom, (Lt, Le))
    positive = coefficient.is_positive is True
    negative = coefficient.is_negative is True
    if not positive and not negative:
        return None
    return (less == positive, bound)


def _radial_integrand(f: Expr, names: Sequence[Symbol], rho: Symbol) -> Optional[Expr]:
    """``g(rho)`` with ``f == g(sqrt(x**2 + y**2 + ...))``, or ``None``
    when ``f`` is not a function of the distance to the origin.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import exp, heaviside
    >>> from sympy_extras.integrals.regions import _radial_integrand
    >>> x, y, rho = Symbol('x'), Symbol('y'), Symbol('rho', positive=True)
    >>> _radial_integrand(exp(-x**2 - y**2), [x, y], rho), _radial_integrand(heaviside(x), [x, y], rho)
    (exp(-rho**2), None)
    """
    radius = sqrt(_radius(names))
    on_axis: dict[Basic, Basic] = {names[0]: radius}
    for v in names[1:]:
        on_axis[v] = S.Zero
    candidate = _evaluated(f, on_axis)
    if candidate.has(S.NaN, S.ComplexInfinity):
        return None
    difference = f - candidate
    simpler = attempt(lambda: simplify(difference), settings.timeout, native=True)
    if simpler is None or simpler != 0:
        # at points of every orthant, and under rotations: Heaviside(x)
        # agrees with 1 = Heaviside(rho) on the positive orthant alone
        if not numerically_equal(f, candidate) or not rotation_invariant(f, names):
            return None
    axis: dict[Basic, Basic] = {names[0]: rho}
    for v in names[1:]:
        axis[v] = S.Zero
    return _evaluated(f, axis)


def _radial(f: Expr, formula: Boolean, names: Sequence[Symbol], assumptions: list[Boolean],
            integrator: Integrator) -> Optional[Expr]:
    """The integral over a disc, an annulus or a ball (in any dimension)
    of a function of the distance to the origin, in polar (spherical)
    coordinates."""
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
    rho = _fresh('rho', positive=True)
    g = _radial_integrand(f, names, rho)
    if g is None:
        return None
    positive = ask(upper > 0, assumptions)
    if positive is False:
        return S.Zero
    inner: list[Boolean] = list(assumptions)
    if positive is None:
        inner.append(upper > 0)
    if lower is None:
        start: Expr = S.Zero
    else:
        if ask(lower >= 0, inner) is not True or ask(lower < upper, inner) is not True:
            return None
        start = sqrt(lower)
    measure = _sphere_area(Integer(len(names))) * rho**(len(names) - 1)
    radius = sqrt(upper)
    refined = attempt(lambda: as_expr(refine(radius, inner)), settings.timeout, native=True)
    if refined is not None:
        radius = refined
    found_value = integrator(g * measure, rho, start, radius, inner)
    if found_value is None:
        return None
    value = found_value
    tidy = attempt(lambda: as_expr(refine(value, inner)), settings.timeout, native=True)
    if tidy is not None:
        value = tidy
    if positive is None:
        return _conditional(value, upper, assumptions)
    return value


def _conditional(value: Expr, bound: Expr, assumptions: list[Boolean]) -> Expr:
    """``value`` where ``bound > 0`` and ``0`` elsewhere (the region is
    empty): the plain ``value`` when ``bound`` is never negative and
    ``value`` vanishes wherever ``bound`` does (the disc of radius
    ``Abs(a)``: ``pi*a**2`` for every ``a``, ``0`` at ``a = 0`` included),
    the ``Piecewise`` otherwise.

    >>> from sympy_extras.core import Symbol, pi
    >>> from sympy_extras.integrals.regions import _conditional
    >>> a = Symbol('a')
    >>> _conditional(pi*a**2, a**2, []), _conditional(a, a, [])
    (pi*a**2, Piecewise((a, a > 0), (0, True)))
    """
    parameters = sorted(bound.free_symbols, key=lambda s: s.name)
    # the parameters are real (a**2 >= 0 is undecided for a complex a)
    if ask(bound >= 0, assumptions + [element(p, Reals) for p in parameters]) is True:
        zeros = attempt(lambda: _zeros(bound, parameters), settings.timeout, native=True) if parameters else []
        if zeros is not None and all(simplify(_evaluated(value, zero)) == 0 for zero in zeros):
            return value
    return piecewise((value, bound > 0), (S.Zero, true))


def _zeros(bound: Expr, parameters: list[Symbol]) -> Optional[list[dict[Basic, Basic]]]:
    """The points of the parameters where ``bound`` vanishes, solved for
    the first parameter of which it is a polynomial with all its roots
    explicit (as SymPy's ``solve(bound, parameters, dict=True)``: ``a =
    I*b`` and ``a = -I*b`` for ``a**2 + b**2``), or for a single parameter
    the finite solution set of :func:`~sympy_extras.assumptions.solve.solve`;
    ``None`` otherwise."""
    for p in parameters:
        terms = _coefficients(bound, [p])
        if terms is None or _total_degree(terms) < 1:
            continue
        found = roots(bound, p)
        if isinstance(found, dict) and sum(found.values()) == _total_degree(terms):
            return [{p: r} for r in found]
    if len(parameters) != 1:
        return None
    p = parameters[0]
    where = solve(eq(bound, 0), p)
    if not isinstance(where, (FiniteSet, EmptySetClass)):
        return None
    return [{p: v} for v in where.args]


def _form_atom(atom: Relational, names: Sequence[Symbol],
               assumptions: list[Boolean]) -> Optional[tuple[bool, list[Expr], list[Expr], Expr]]:
    """``(upper, coefficients, shifts, value)`` when the relation bounds a
    diagonal quadratic form with the square completed,
    ``sum(c_i*(x_i + s_i)**2) < value`` (``upper``) or ``> value``, with
    all ``c_i`` positive under the assumptions; ``None`` otherwise (a
    cross term, a coefficient of unknown sign, a missing square)."""
    difference = atom.lhs - atom.rhs
    terms = _coefficients(difference, names)
    if terms is None or _total_degree(terms) != 2:
        return None
    n = len(names)
    coefficients: list[Expr] = []
    linear: list[Expr] = []
    for i in range(n):
        c = _coefficient(terms, _unit(n, i, 2))
        b = _coefficient(terms, _unit(n, i, 1))
        if c == 0:
            return None
        coefficients.append(c)
        linear.append(b)
    for monomial in terms:
        if sum(monomial) == 2 and max(monomial) == 1:
            return None                                     # a cross term
    sign_: Optional[bool] = None
    for c in coefficients:
        positive = c.is_positive is True or ask(c > 0, assumptions) is True
        negative = c.is_negative is True or ask(c < 0, assumptions) is True
        if not positive and not negative:
            return None
        if sign_ is None:
            sign_ = positive
        elif sign_ != positive:
            return None
    if sign_ is None:
        return None
    if not sign_:
        coefficients = [-c for c in coefficients]
        linear = [-b for b in linear]
        difference = -difference
    less = isinstance(atom, (Lt, Le))
    shifts = [b / (2 * c) for b, c in zip(linear, coefficients)]
    completed: Expr = S.Zero
    for c, v, t in zip(coefficients, names, shifts):
        completed = completed + c * (v + t)**2
    rest = expand(difference - completed)
    if rest.has(*names):
        return None
    return (less == sign_, coefficients, shifts, -rest)


def _scaled_radial(f: Expr, formula: Boolean, names: Sequence[Symbol],
                   assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
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
    us: list[Symbol] = [_fresh(v.name) for v in names]
    substitution: dict[Basic, Basic] = {}
    jacobian: Expr = S.One
    for v, u_, c, t in zip(names, us, coefficients, shifts):
        substitution[v] = u_ / sqrt(c) - t
        jacobian = jacobian / sqrt(c)
    radius = _radius(us)
    parts: list[Boolean] = [radius < value]
    if lower is not None:
        low_coefficients, low_shifts, low_value = lower
        ratio = low_coefficients[0] / coefficients[0]
        for c, d in zip(coefficients, low_coefficients):
            if simplify(d - ratio * c) != 0:
                return None
        if any(simplify(t - t_) != 0 for t, t_ in zip(shifts, low_shifts)):
            return None
        if ask(ratio > 0, assumptions) is not True:
            return None
        parts.append(radius > low_value / ratio)
    g = _evaluated(f, substitution) * jacobian
    return _radial(g, and_(*parts), us, assumptions, integrator)


def _cylindrical(f: Expr, formula: Boolean, names: Sequence[Symbol],
                 assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
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
    rho = _fresh('rho', positive=True)
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
        if ask(upper_radius > 0, assumptions) is not True:
            return None
        conditions.append(rho < sqrt(upper_radius))
    if lower_radius is not None:
        if ask(lower_radius >= 0, assumptions) is not True:
            return None
        conditions.append(rho > sqrt(lower_radius))
    if lower_z != -oo and upper_z != oo:
        conditions.append(lower_z < upper_z)
    if not conditions:
        return None
    where = attempt(lambda: solve(and_(*conditions), rho, assumptions + [rho > 0], domain=Reals),
                    settings.timeout, native=True)
    if where is None:
        return None
    # rho > 0 is not applied by solve to rho**2 < a with a symbolic:
    # the interval (-sqrt(a), sqrt(a)) made the odd integrand vanish
    pieces = _interval_conditions(intersection(where, interval(S.Zero, oo, True, True)), rho)
    if pieces is None:
        return None
    total: Expr = S.Zero
    for piece in pieces:
        limits = _piece_limits(piece, rho)
        if limits is None:
            return None
        start, stop = limits
        rho_assumptions = assumptions + [rho > start, rho < stop] if stop != oo else assumptions + [rho > start]
        inner = integrator(g * rho, z, lower_z, upper_z, rho_assumptions)
        if inner is None:
            return None
        outer = integrator(2 * pi * inner, rho, start, stop, assumptions)
        if outer is None:
            return None
        total = total + outer
    return total


def _piece_limits(piece: Boolean, rho: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(start, stop)`` of a conjunction ``rho > start & rho < stop``
    (``0`` and ``oo`` when a side is missing)."""
    start: Expr = S.Zero
    stop: Expr = S.Infinity
    parts = list(piece.args) if isinstance(piece, And) else ([] if isinstance(piece, BooleanTrue) else [piece])
    for part in parts:
        if isinstance(part, (Gt, Ge)) and part.lhs == rho:
            start = part.rhs
        elif isinstance(part, (Lt, Le)) and part.lhs == rho:
            stop = part.rhs
        elif isinstance(part, (Lt, Le)) and part.rhs == rho:
            start = part.lhs
        elif isinstance(part, (Gt, Ge)) and part.rhs == rho:
            stop = part.lhs
        else:
            return None
    return (start, stop)


def _linear_bound(atom: Relational, y: Symbol, assumptions: list[Boolean]) -> Optional[tuple[bool, Expr]]:
    """``(upper, bound)`` for a relation linear in ``y``: ``y < bound``
    (``upper``) or ``y > bound``; ``None`` when the relation is not linear
    in ``y`` or the sign of the coefficient of ``y`` is not known."""
    difference = atom.lhs - atom.rhs
    terms = _coefficients(difference, [y])
    if terms is None or _total_degree(terms) != 1:
        return None
    p, q = _coefficient(terms, (1,)), _coefficient(terms, (0,))
    # p*y + q < 0  (Lt, Le) or > 0 (Gt, Ge)
    less = isinstance(atom, (Lt, Le))
    positive = p.is_positive is True or ask(p > 0, assumptions) is True
    negative = p.is_negative is True or ask(p < 0, assumptions) is True
    if not positive and not negative:
        return None
    return (less == positive, -q / p)


def _solved_bound(atom: Relational, y: Symbol, assumptions: list[Boolean]) -> Optional[list[tuple[bool, Expr]]]:
    """The bounds ``(upper, bound)`` on ``y`` from a relation: the one of
    :func:`_linear_bound` when the relation is linear in ``y``, else the
    finite endpoints of the interval of ``y`` which
    :func:`sympy_extras.assumptions.solve.solve` finds under the assumptions
    (``y**2 < x`` with ``x > 0`` gives ``-sqrt(x) < y < sqrt(x)``,
    ``exp(y) < x`` gives ``y < log(x)``); ``None`` when the solution is
    not one interval."""
    linear = _linear_bound(atom, y, assumptions)
    if linear is not None:
        return [linear]
    where = attempt(lambda: solve(atom, y, assumptions, domain=Reals), settings.timeout, native=True)
    if not isinstance(where, Interval):
        return None
    found: list[tuple[bool, Expr]] = []
    if where.left != -oo:
        found.append((False, where.left))
    if where.right != oo:
        found.append((True, where.right))
    return found


def _extreme(bounds: list[Expr], largest: bool, assumptions: list[Boolean]) -> Optional[Expr]:
    """The largest (smallest) of the bounds, when the assumptions order
    them; ``None`` otherwise."""
    best = bounds[0]
    for other in bounds[1:]:
        if ask(other >= best if largest else other <= best, assumptions) is True:
            best = other
        elif ask(other <= best if largest else other >= best, assumptions) is not True:
            return None
    return best


def _interval_conditions(where: Set, x: Symbol) -> Optional[list[Boolean]]:
    """A set of real numbers as a list of conjunctions ``a < x < b``, one
    per interval (points are dropped: measure zero); ``None`` for a set
    which is not a union of intervals."""
    if where == Reals:
        return [true]
    if isinstance(where, (EmptySetClass, FiniteSet)):
        return []
    if isinstance(where, Interval):
        parts: list[Boolean] = []
        if where.left != -oo:
            parts.append(x > where.left)
        if where.right != oo:
            parts.append(x < where.right)
        return [and_(*parts)]
    if isinstance(where, SetUnion):
        pieces: list[Boolean] = []
        for member in where.args:
            if not isinstance(member, Set):
                return None
            found = _interval_conditions(member, x)
            if found is None:
                return None
            pieces.extend(found)
        return pieces
    return None


def _solved_bounds(f: Expr, formula: Boolean, names: Sequence[Symbol],
                   assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
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
    # the region lies where the bounds are real: y < log(x) restricts x
    # to x > 0 (the bug: the inner integral was taken over -1 < x < 1)
    for bound in lowers + uppers:
        conditions = _reality_conditions(bound)
        if conditions is None:
            return None
        for condition in conditions:
            if condition is not true and condition not in outer_atoms:
                outer_atoms.append(condition)
                outer_assumptions.append(condition)
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
        return integrator(f, y, lower, upper, assumptions)
    # where the two bounds are in the right order
    pieces: list[Boolean] = [true]
    if lower != -oo and upper != oo:
        ordered = ask(lower < upper, outer_assumptions)
        if ordered is False:
            return S.Zero
        if ordered is None:
            if len(outer_names) != 1:
                return None
            x = outer_names[0]
            where = attempt(lambda: solve(lower < upper, x, outer_assumptions, domain=Reals),
                            settings.timeout, native=True)
            if where is None:
                return None
            found_pieces = _interval_conditions(where, x)
            if found_pieces is None:
                return None
            pieces = found_pieces
    inner_assumptions = outer_assumptions + ([y > lower] if lower != -oo else []) \
        + ([y < upper] if upper != oo else [])
    inner = integrator(f, y, lower, upper, inner_assumptions)
    if inner is None:
        return None
    total: Expr = S.Zero
    for piece in pieces:
        condition = and_(*outer_atoms, piece)
        value: Optional[Expr]
        if isinstance(condition, BooleanTrue):
            # the other variables are unbounded: the whole line each,
            # innermost first
            value = inner
            for v in reversed(outer_names):
                value = integrator(value, v, S.NegativeInfinity, S.Infinity, assumptions)
                if value is None:
                    return None
        else:
            found_value = integrate_by_ranges(inner, condition, outer_names, assumptions, integrator=integrator)
            if found_value.has(IntegralByRanges):
                return None
            value = found_value
        total = total + value
    return total


# ---------------------------------------------------------------------------
# Radial integrals in a symbolic dimension

#: functions real at every real argument
_REAL_FUNCTIONS = (Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh, Atan, Asinh, Abs, Sign, Floor, Ceiling)


def _reality_conditions(e: Expr) -> Optional[list[Boolean]]:
    """The conditions under which ``e`` is real: ``u > 0`` for ``log(u)``,
    ``u >= 0`` for a rational power ``u**(p/q)`` (the principal root
    is not real for ``u < 0``, whatever ``q``), ``-1 <= u <= 1`` for
    ``asin``, ``acos``, and so on; ``None`` for a function whose reality
    is not known.

    >>> from sympy_extras.core import Symbol, Rational, sqrt
    >>> from sympy_extras.functions import log, exp
    >>> from sympy_extras.integrals.regions import _reality_conditions
    >>> x = Symbol('x')
    >>> _reality_conditions(log(x + 1) - 1), _reality_conditions(exp(x))
    ([x + 1 > 0], [])
    >>> sorted(_reality_conditions(x**Rational(1, 3) + sqrt(x - 1)), key=str)
    [x - 1 >= 0, x >= 0]
    """
    if e.has(Piecewise):
        return None                                         # SymPy's Piecewise is a function of unknown reality
    conditions: list[Boolean] = []
    for node in sorted(e.atoms(Application), key=lambda n: n.sort_key()):
        if not node.args:
            continue
        argument = as_expr(node.args[0])
        if isinstance(node, Log):
            conditions.append(argument > 0)
        elif isinstance(node, (Asin, Acos)):
            conditions.extend([argument >= -1, argument <= 1])
        elif isinstance(node, Acosh):
            conditions.append(argument >= 1)
        elif isinstance(node, Atanh):
            conditions.extend([argument > -1, argument < 1])
        elif not isinstance(node, _REAL_FUNCTIONS):
            return None
    for power in sorted(e.atoms(Pow), key=str):
        if not isinstance(power, Pow):
            continue
        base, exponent = power.base, power.exp
        if isinstance(exponent, Rational) and not exponent.is_integer and base.free_symbols:
            conditions.append(base >= 0 if exponent.p > 0 else base > 0)
        elif exponent.free_symbols and base.is_positive is not True:
            return None
    return conditions


def _sphere_area(n: Expr) -> Expr:
    """The area `2 pi^{n/2} / Gamma(n/2)` of the unit sphere of
    `\\mathbb{R}^n`."""
    return 2 * pi**(n / 2) / gamma(n / 2)


def _symbolic_radial(f: Expr, formula: Boolean, r: Symbol, n: Expr,
                     assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
    """``Integral(f(r), R^n)`` over the ball, annulus or whole space that
    ``formula`` describes through bounds on the radial variable ``r``:
    the area of the unit sphere times ``Integral(f * r**(n - 1), (r, a, R))``."""
    if isinstance(formula, BooleanTrue):
        atoms: list[Relational] = []
    else:
        found = _atoms(formula)
        if found is None:
            return None
        atoms = found
    lower: Optional[Expr] = None
    upper: Optional[Expr] = None
    for atom in atoms:
        bound = _linear_bound(atom, r, assumptions)
        if bound is None:
            return None
        is_upper, value = bound
        if is_upper:
            if upper is not None:
                return None
            upper = value
        else:
            if lower is not None:
                return None
            lower = value
    inner: list[Boolean] = list(assumptions) + [n > 0]
    start: Expr = S.Zero if lower is None else lower
    if lower is not None and ask(lower >= 0, inner) is not True:
        return None
    end: Expr = oo if upper is None else upper
    if upper is not None and ask(start < upper, inner) is not True:
        return None
    if f.has(r) and ask(r > 0, inner) is None:
        rho = _fresh('rho', positive=True)
        rule: dict[Basic, Basic] = {r: rho}
        f = _evaluated(f, rule)
        start, end, r = _evaluated(start, rule), _evaluated(end, rule), rho
    radial = integrator(f * r**(n - 1), r, start, end, inner)
    if radial is None:
        return None
    total = _sphere_area(n) * radial
    tidy = attempt(lambda: simplify(total), settings.timeout, native=True)
    return total if tidy is None else tidy


# ---------------------------------------------------------------------------
# Polynomials over polytopes

class _Polytope:
    """A bounded polytope ``A x <= b`` with rational ``A``, ``b`` free of
    the variables, and its vertices."""

    __slots__ = ('rows', 'rhs', 'vertices')

    def __init__(self, rows: list[list[Fraction]], rhs: list[Expr], vertices: list[list[Expr]]) -> None:
        self.rows: Final[list[list[Fraction]]] = rows
        self.rhs: Final[list[Expr]] = rhs
        self.vertices: Final[list[list[Expr]]] = vertices

    def on_facet(self, vertex: int, facet: int) -> bool:
        """Whether the vertex lies on the hyperplane of the facet."""
        return expand(_dot(self.rows[facet], self.vertices[vertex]) - self.rhs[facet]) == 0


def _dot(row: Sequence[Fraction], point: Sequence[Expr]) -> Expr:
    total: Expr = S.Zero
    for a, v in zip(row, point):
        total = total + number_from_fraction(a) * v
    return total


def _linear_rows(atoms: Sequence[Relational], names: Sequence[Symbol]) \
        -> Optional[tuple[list[list[Fraction]], list[Expr]]]:
    """The inequalities as rows ``a . x <= b``: ``None`` when one is not
    linear in the variables with rational coefficients (parameters are
    allowed in the constant terms only)."""
    rows: list[list[Fraction]] = []
    rhs: list[Expr] = []
    n = len(names)
    for atom in atoms:
        difference = atom.lhs - atom.rhs
        if isinstance(atom, (Gt, Ge)):
            difference = -difference
        terms = _coefficients(difference, names)
        if terms is None or _total_degree(terms) > 1:
            return None
        row: list[Fraction] = []
        for i in range(n):
            coefficient = _coefficient(terms, _unit(n, i, 1))
            if not isinstance(coefficient, Rational):
                return None
            row.append(Fraction(coefficient.p, coefficient.q))
        if all(c == 0 for c in row):
            return None
        rows.append(row)
        rhs.append(-_coefficient(terms, tuple(0 for _ in range(n))))
    return rows, rhs


def _bounded(rows: list[list[Fraction]]) -> Optional[bool]:
    """Whether ``A x <= b`` is bounded: no direction ``d != 0`` with
    ``A d <= 0`` (the extreme value of each coordinate over the recession
    cone cut to the unit cube is 0), by linear programming."""
    d = len(rows[0])
    box = [[Fraction(1 if j == i else 0) for j in range(d)] for i in range(d)] \
        + [[Fraction(-1 if j == i else 0) for j in range(d)] for i in range(d)]
    A = rows + box
    b = [Fraction(0)] * len(rows) + [Fraction(1)] * (2 * d)
    for j in range(d):
        for orientation in (1, -1):
            c = [Fraction(orientation if i == j else 0) for i in range(d)]
            found = linear_program(c, A, b)
            if found.status != 'optimal' or found.value is None:
                return None
            if found.value > 0:
                return False
    return True


def _solve_rational(matrix: list[list[Fraction]], rhs: list[Expr]) -> Optional[list[Expr]]:
    """The solution of the square system ``matrix . x = rhs`` with a
    rational matrix, the coordinates expanded; ``None`` when the matrix
    is singular."""
    n = len(matrix)
    inverse = [[Fraction(1 if j == i else 0) for j in range(n)] for i in range(n)]
    rows = [list(row) for row in matrix]
    for col in range(n):
        pivot = next((i for i in range(col, n) if rows[i][col] != 0), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        inverse[col], inverse[pivot] = inverse[pivot], inverse[col]
        scale = rows[col][col]
        rows[col] = [a / scale for a in rows[col]]
        inverse[col] = [a / scale for a in inverse[col]]
        for i in range(n):
            if i != col and rows[i][col] != 0:
                factor_ = rows[i][col]
                rows[i] = [a - factor_ * b for a, b in zip(rows[i], rows[col])]
                inverse[i] = [a - factor_ * b for a, b in zip(inverse[i], inverse[col])]
    return [expand(_dot(inverse[k], rhs)) for k in range(n)]


def _vertices(rows: list[list[Fraction]], rhs: list[Expr],
              assumptions: list[Boolean]) -> Optional[list[list[Expr]]]:
    """The vertices of ``A x <= b``: the intersections of ``d`` facet
    hyperplanes which satisfy the other inequalities, decided under the
    assumptions on the parameters (``None`` when undecided)."""
    d = len(rows[0])
    vertices: list[list[Expr]] = []
    for subset in combinations(range(len(rows)), d):
        coordinates = _solve_rational([rows[i] for i in subset], [rhs[i] for i in subset])
        if coordinates is None:
            continue
        feasible = True
        for i in range(len(rows)):
            if i in subset:
                continue
            slack = expand(_dot(rows[i], coordinates) - rhs[i])
            if slack.is_number:
                if slack.is_positive:
                    feasible = False
                    break
                continue
            verdict = ask(slack <= 0, assumptions)
            if verdict is None:
                return None
            if verdict is False:
                feasible = False
                break
        if feasible and coordinates not in vertices:
            vertices.append(coordinates)
    return vertices


def _rank(matrix: list[list[Expr]]) -> int:
    """The rank of a matrix of polynomials in the parameters (over their
    field of rational functions): the largest order of a minor which does
    not expand to zero."""
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    for k in range(min(rows, cols), 0, -1):
        for chosen in combinations(range(rows), k):
            for columns in combinations(range(cols), k):
                minor = determinant([[matrix[i][j] for j in columns] for i in chosen])
                if expand(minor) != 0:
                    return k
    return 0


def _affine_rank(points: Sequence[Sequence[Expr]]) -> int:
    """The dimension of the affine hull of the points."""
    if len(points) < 2:
        return 0
    base = points[0]
    return _rank([[p[k] - base[k] for k in range(len(base))] for p in points[1:]])


def _pulling(polytope: _Polytope, face: list[int], k: int) -> list[list[int]]:
    """The pulling triangulation of a face of dimension ``k`` given by
    its vertices: cones from the first vertex over the triangulations of
    the facets of the face not containing it."""
    if k == 0:
        return [[face[0]]]
    apex = face[0]
    simplices: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    for facet in range(len(polytope.rows)):
        if polytope.on_facet(apex, facet):
            continue
        on_facet = [v for v in face if polytope.on_facet(v, facet)]
        key = frozenset(on_facet)
        if len(on_facet) < k or key in seen:
            continue
        if _affine_rank([polytope.vertices[v] for v in on_facet]) != k - 1:
            continue
        seen.add(key)
        for simplex in _pulling(polytope, on_facet, k - 1):
            simplices.append([apex] + simplex)
    return simplices


def _simplex_integral(f: Expr, names: Sequence[Symbol], vertices: Sequence[Sequence[Expr]],
                      assumptions: list[Boolean]) -> Optional[Expr]:
    """``Integral(f, simplex)`` by the affine map ``x = s0 + sum (s_i - s0)
    u_i`` from the standard simplex and Dirichlet's formula
    ``Integral(u**a) = prod(a_i!) / (|a| + d)!``; ``None`` when the sign
    of the Jacobian is undecided."""
    d = len(names)
    base = vertices[0]
    us: list[Symbol] = [_fresh('u%d_' % i) for i in range(d)]
    substitution: dict[Basic, Basic] = {}
    for k, v in enumerate(names):
        image: Expr = base[k]
        for i in range(d):
            image = image + (vertices[i + 1][k] - base[k]) * us[i]
        substitution[v] = image
    jacobian = expand(determinant([[vertices[i + 1][k] - base[k] for k in range(d)] for i in range(d)]))
    if jacobian == 0:
        return S.Zero
    if jacobian.is_number:
        if not jacobian.is_extended_real:
            return None
        volume = abs_(jacobian)
    elif ask(jacobian > 0, assumptions) is True:
        volume = jacobian
    elif ask(jacobian < 0, assumptions) is True:
        volume = -jacobian
    else:
        return None
    pulled = _coefficients(_evaluated(f, substitution), us)
    if pulled is None:
        return None
    total: Expr = S.Zero
    for monomial, coefficient in pulled.items():
        numerator = 1
        for a in monomial:
            numerator *= math.factorial(a)
        weight = Rational(numerator, math.factorial(sum(monomial) + d))
        total = total + coefficient * weight
    return volume * total


def _polytope(f: Expr, formula: Boolean, names: Sequence[Symbol], assumptions: list[Boolean]) -> Optional[Expr]:
    """A polynomial over a bounded polytope, summed over the simplices of
    the pulling triangulation; ``None`` when the region is not a bounded
    polytope with the vertices decided, or the integrand not a polynomial.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.assumptions.facts import normalize
    >>> from sympy_extras.integrals.regions import _polytope
    >>> x, y = Symbol('x'), Symbol('y')
    >>> _polytope(x*y, normalize((x > 0) & (y > 0) & (x + y < 1)), [x, y], [])
    1/24
    """
    atoms = _atoms(formula)
    if atoms is None or not atoms:
        return None
    if any(not atom.has(*names) for atom in atoms):
        return None
    linear = _linear_rows(atoms, names)
    if linear is None:
        return None
    rows, rhs = linear
    if _coefficients(f, names) is None:
        return None
    if _bounded(rows) is not True:
        return None
    vertices = _vertices(rows, rhs, assumptions)
    if vertices is None:
        return None
    if not vertices:
        return S.Zero
    d = len(names)
    if _affine_rank(vertices) < d:
        return S.Zero
    polytope = _Polytope(rows, rhs, vertices)
    total: Expr = S.Zero
    for simplex in _pulling(polytope, list(range(len(vertices))), d):
        if len(simplex) != d + 1:
            return None
        value = _simplex_integral(f, names, [vertices[i] for i in simplex], assumptions)
        if value is None:
            return None
        total = total + value
    tidy = attempt(lambda: factor(total), settings.timeout, native=True)
    return expand(total) if tidy is None else tidy


def integrate_by_ranges(integrand: ExprLike, condition: BooleanLike,
                        variables: Optional[Sequence[Symbol]] = None,
                        assumptions: Assumptions = None,
                        measure: str = 'lebesgue',
                        dimension: Optional[ExprLike] = None,
                        parameters: Optional[Sequence[Symbol]] = None,
                        integrator: Optional[Integrator] = None) -> Expr:
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
        the condition other than the ``parameters``. The other symbols
        are parameters.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters; the cases of the parameter space
        they refute are dropped.
    measure : ``'lebesgue'`` (default) or ``'hausdorff'``
        With the Hausdorff measure the condition must contain exactly one
        equation, and the integral is taken on the hypersurface it
        describes (the length of a curve, the area of a surface), see the
        module documentation.
    dimension : int, Integer or Symbol, optional
        A radial integral in `\\mathbb{R}^n`: the condition bounds the
        radial variable alone (``r < R``, ``(a < r) & (r < R)``, ``True``
        for the whole space), which is the variable given, the symbol
        named ``r`` of the condition or its only symbol, and the
        integrand is a function of it; the dimension may be symbolic.
    parameters : list of Symbol, optional
        The symbols which are not integrated over, when ``variables`` is
        not given: ``x**2 + y**2 < a**2`` with ``parameters=[a]`` is the
        disc of radius ``Abs(a)``, where without either the region is the
        unbounded solid in ``(a, x, y)``.
    integrator : Integrator, optional
        The one-dimensional integrals ``integrator(f, x, a, b,
        assumptions)``: the value, ``None`` when it is not computed, a
        ``ValueError`` for bounds which contradict the assumptions; by
        default :func:`~sympy_extras.integrals.definite.definite_integral`
        on SymPy's side.

    Returns
    =======

    The value, a ``Piecewise`` over the cases of the parameters, or the
    unevaluated :class:`IntegralByRanges` when a bound has no explicit
    form or an inner integral is not computed.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, Rational, eq
    >>> from sympy_extras.integrals.regions import integrate_by_ranges
    >>> x, y, z, r = Symbol('x'), Symbol('y'), Symbol('z'), Symbol('r')
    >>> integrate_by_ranges(x**2, x**2 + y**2 < 1)
    pi/4
    >>> integrate_by_ranges(Integer(1), (0 < x) & (x < y) & (y < x**2) & (x < 2))
    5/6
    >>> integrate_by_ranges(Integer(1), x**2 + y**2 < r**2, parameters=[r])
    pi*r**2
    >>> integrate_by_ranges(Integer(1), (x > 0) & (x < r), parameters=[r])
    Piecewise((0, Eq(r, 0) | (r < 0)), (r, True))
    >>> integrate_by_ranges(Integer(1), eq(z, x**2 + y**2) & (z < 1), measure='hausdorff')
    pi*(-1 + 5*sqrt(5))/6

    An integrator written on the core: the polynomials by their
    antiderivatives, so that nothing goes to SymPy.

    >>> from sympy_extras.core import oo, expand
    >>> from sympy_extras.integrals.risch.rationaltools import ratint
    >>> def polynomial(f, x, a, b, assumptions):
    ...     if a == -oo or b == oo:
    ...         return None
    ...     F = ratint(f, x)
    ...     return expand(F.subs(x, b) - F.subs(x, a))
    >>> integrate_by_ranges(x*y, (x > 0) & (y > 0) & (x + y < 1), integrator=polynomial)
    1/24
    """
    node = IntegralByRanges(integrand, condition, variables, measure, dimension, parameters)
    inner = _definite if integrator is None else integrator
    f, formula, names = node.integrand, normalize(node.condition), node.variables
    if node.measure == _MEASURES[0] and dimension is None:
        formula = _full_measure(formula, names)
        if isinstance(formula, BooleanFalse):
            return S.Zero
    extra: list[Boolean] = []
    if assumptions is not None:
        if isinstance(assumptions, (Boolean, bool)):
            extra = [to_boolean(assumptions)]
        else:
            extra = [to_boolean(a) for a in assumptions]
    if node.measure == 'hausdorff':
        revolved = axisymmetric_integral(f, formula, names, extra, measure=node.measure, integrator=inner)
        if revolved is not None:
            return revolved
        surface = _hausdorff(f, formula, names, extra, inner)
        return node if surface is None else surface
    if node.dimension is not None:
        symbolic = _symbolic_radial(f, formula, names[0], node.dimension, extra, inner)
        return node if symbolic is None else symbolic
    radial = _radial(f, formula, names, extra, inner)
    if radial is not None:
        return radial
    scaled = _scaled_radial(f, formula, names, extra, inner)
    if scaled is not None:
        return scaled
    cylindrical = _cylindrical(f, formula, names, extra, inner)
    if cylindrical is not None:
        return cylindrical
    polytope = _polytope(f, formula, names, extra)
    if polytope is not None:
        return polytope
    revolved = axisymmetric_integral(f, formula, names, extra, integrator=inner)
    if revolved is not None:
        return revolved
    found = _decomposed(f, formula, names, extra, inner)
    if found is not None:
        return found
    solved = _solved_bounds(f, formula, names, extra, inner)
    if solved is not None:
        return solved
    for order in _other_orders(names):
        # a bound with no explicit form in the last variable (a root of
        # x**3 + x*y - 1 in x) may be explicit in another (y = (1 - x**3)/x)
        found = attempt(lambda: _decomposed(f, formula, order, extra, inner), settings.timeout, native=True)
        if found is not None:
            return found
    return node


def _full_measure(formula: Boolean, names: Sequence[Symbol]) -> Boolean:
    """The condition without its equations and non-equations in the
    integration variables (``Eq(p, 0)`` false, ``Ne(p, 0)`` true): they
    change the set on a union of hypersurfaces, of Lebesgue measure zero
    (the bug: a box cut along the curve of a ``Ne`` was decomposed along
    it, with algebraic bounds whose logarithms ran out of memory).
    Relations in the parameters alone are kept.

    >>> from sympy_extras.core import Symbol, eq, ne
    >>> from sympy_extras.integrals.regions import _full_measure
    >>> x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
    >>> _full_measure((x > 0) & ne(x*y - 1, 0) & eq(a, 1), [x, y])
    Eq(a, 1) & (x > 0)
    """
    replacements: dict[Basic, Basic] = {}
    for atom in formula.atoms(Eq, Ne):
        if not isinstance(atom, Relational):
            continue
        if (atom.lhs - atom.rhs).free_symbols & set(names):
            replacements[atom] = false if isinstance(atom, Eq) else true
    if not replacements:
        return formula
    return to_boolean(formula._xreplace_evaluated(replacements))


def _other_orders(names: Sequence[Symbol]) -> list[list[Symbol]]:
    """The other orders of the variables to try: every permutation of up
    to three variables, the reversed order beyond."""
    if len(names) < 2:
        return []
    if len(names) > 3:
        return [list(reversed(names))]
    return [list(order) for order in permutations(names) if list(order) != list(names)]


def _decomposition(formula: Boolean, gens: list[Expr]) -> Optional[tuple[_Compiled, CAD]]:
    """The compiled condition and the decomposition sign-invariant for
    its polynomials; ``None`` when the condition is not polynomial."""
    polys: list[Poly] = []
    index: dict[Poly, int] = {}
    try:
        compiled = _compile(formula, gens, polys, index)
        cad = cylindrical_algebraic_decomposition(polys, gens)
    except (ValueError, TypeError, BasePolynomialError):
        return None
    return compiled, cad


def _decomposed(f: Expr, formula: Boolean, names: Sequence[Symbol], assumptions: list[Boolean],
                integrator: Integrator) -> Optional[Expr]:
    """The integral through the cylindrical algebraic decomposition, or
    ``None`` when the condition is not polynomial, a bound has no explicit
    form or an inner integral is not computed."""
    parameters = sorted(formula.free_symbols - set(names), key=lambda s: s.name)
    gens: list[Expr] = [*parameters, *names]
    decomposition = _decomposition(formula, gens)
    if decomposition is None:
        return None
    compiled, cad = decomposition
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
        found = _iterated(f, names, stack, assumptions + ([parameter_condition] if m else []), integrator)
        if found is None:
            return None
        value: Expr = found
        if m:
            refined = attempt(lambda: as_expr(refine(value, assumptions + [parameter_condition])),
                              settings.timeout, native=True)
            if refined is not None:
                value = refined
        values[base] = values[base] + value if base in values else value
    return _cases(values, cad, m, assumptions)


def _cases(values: dict[CADCell, Expr], cad: CAD, m: int, assumptions: list[Boolean]) -> Expr:
    """The sum of the values of the cells over the parameter cells, as a
    ``Piecewise`` over the sign conditions of the parameter cells (the
    plain sum without parameters, ``m == 0``)."""
    if m == 0:
        total: Expr = S.Zero
        for value in values.values():
            total = total + value
        return total
    cases: list[tuple[Expr, Boolean, CADCell]] = []
    for base in cad.cells_at(m):
        cond = _parameter_condition(base, cad, m, explicit=True)
        if ask(cond, assumptions) is False:
            continue
        cases.append((values.get(base, S.Zero), cond, base))
    if not cases:
        return S.Zero
    if len(cases) == 1:
        return cases[0][0]
    whole = _one_expression(cases)
    if whole is not None:
        return whole
    grouped: dict[Expr, list[Boolean]] = {}
    order: list[Expr] = []
    for value, cond, _ in cases:
        if value not in grouped:
            grouped[value] = []
            order.append(value)
        grouped[value].append(cond)
    order.sort(key=lambda v: -len(grouped[v]))
    branches: list[tuple[Expr, Boolean]] = []
    for i, value in enumerate(order):
        cond = or_(*grouped[value]) if i + 1 < len(order) else true
        branches.append((value, cond))
    return piecewise(*branches)


def _one_expression(cases: list[tuple[Expr, Boolean, CADCell]]) -> Optional[Expr]:
    """The common value of the cases when the full-dimensional parameter
    cells all have the same value and each lower-dimensional cell's
    value is that expression at the cell (``pi*a**2`` on ``a < 0`` and on
    ``a > 0``, ``0 = pi*0**2`` at ``a = 0``: the disc of radius ``Abs(a)``
    is ``pi*a**2`` for every ``a``); ``None`` when the cases differ or a
    lower-dimensional cell has no explicit equations."""
    sectors = [value for value, _, base in cases if all(i % 2 for i in base.index)]
    if not sectors or any(value != sectors[0] for value in sectors[1:]):
        return None
    common = sectors[0]
    for value, condition, base in cases:
        if all(i % 2 for i in base.index):
            continue
        substitutions: dict[Basic, Basic] = {}
        for atom in (condition.args if isinstance(condition, And) else (condition,)):
            if isinstance(atom, Eq) and isinstance(atom.lhs, Symbol):
                substitutions[atom.lhs] = atom.rhs
        if not substitutions:
            return None
        rule = substitutions
        restricted = attempt(lambda: simplify(_evaluated(common, rule) - _evaluated(value, rule)),
                             settings.timeout, native=True)
        if restricted is None or restricted != 0:
            return None
    return common


def _pruned(condition: Boolean) -> Boolean:
    """The conjunction without the atoms implied by the others (the sign
    conditions of the projection polynomials of a cell repeat what the
    higher levels say: ``x**2 < 1`` under ``x**2 + y**2 < 1``)."""
    parts = [to_boolean(a) for a in (condition.args if isinstance(condition, And) else (condition,))]
    kept = list(parts)
    for part in parts:
        rest = [other for other in kept if other is not part]
        if rest and _ask(part, rest) is True:
            kept = rest
    return and_(*kept)


def _section_value(cell: CADCell, cad: CAD, m: int, f: Expr, names: Sequence[Symbol],
                   assumptions: list[Boolean], integrator: Integrator) -> Optional[Expr]:
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
    branches: dict[Basic, Basic] = {}
    for k in sections:
        node = _ancestor_at(cell, k)
        parent = node.parent
        gens = cad.gens[:k]
        roots_ = _root_sequence(parent, cad.projection[k - 1], gens) if parent is not None else []
        position = node.index[-1] // 2 - 1
        if not 0 <= position < len(roots_) or parent is None:
            return None
        phi = _explicit_root(roots_[position], parent, gens)
        if phi is None:
            return None
        branches[names[k - m - 1]] = _evaluated(phi, branches)
    solved = [names[k - m - 1] for k in sections]
    rest = [v for v in names if v not in solved]
    jacobian = [[as_expr(branches[v]).diff(u) for u in rest] for v in solved]
    if rest:
        gram_matrix: list[list[Expr]] = []
        for i in range(len(rest)):
            row: list[Expr] = []
            for j in range(len(rest)):
                entry: Expr = S.One if i == j else S.Zero
                product: Expr = S.Zero
                for k_ in range(len(solved)):
                    product = product + jacobian[k_][i] * jacobian[k_][j]
                row.append(entry + product)
            gram_matrix.append(row)
        gram = determinant(gram_matrix)
    else:
        gram = S.One
    gram_ = gram
    simpler = attempt(lambda: simplify(gram_), settings.timeout, native=True)
    if simpler is not None:
        gram = simpler
    g = _evaluated(f, branches) * sqrt(gram)
    if not rest:
        return g                                            # a point: the counting measure
    stack = _stack(cell, cad, m + 1, sections, branches)
    if stack is None:
        return None
    facts = list(assumptions)
    for v, (lower, upper) in zip(rest, stack.bounds):
        facts.extend(([v > lower] if lower != -oo else []) + ([v < upper] if upper != oo else []))
    g = _split_roots(g, facts)                              # sqrt(-1/(x**2 - 1)) sqrt(1 - x**2) = 1 on (-1, 1)
    if sections == [n] and len(rest) >= 2:
        # the base cell as a region (a disc under a surface goes to polar coordinates)
        condition = _pruned(_cell_condition(cell, cad, m + 1, n - 1))
        found = integrate_by_ranges(g, condition, rest, assumptions, integrator=integrator)
        if not found.has(IntegralByRanges):
            return found
    return _iterated(g, rest, stack, assumptions, integrator)


def _hausdorff(f: Expr, formula: Boolean, names: Sequence[Symbol], assumptions: list[Boolean],
               integrator: Integrator) -> Optional[Expr]:
    """The integral with the `(n-k)`-dimensional Hausdorff measure on the
    variety of the equations of the condition, through the
    decomposition: the sum of :func:`_section_value` over the cells of
    the largest dimension `n - k` on which the condition holds (`k` the
    number of equations when they are independent; ``Eq(y, x) & Eq(x -
    y, 0)`` is a curve, not a point: the bug gave it length 0); ``None``
    when the condition has no equation, a branch is not explicit or an
    integral is not computed."""
    parts = list(formula.args) if isinstance(formula, And) else [formula]
    equations = sum(1 for part in parts if isinstance(part, Eq))
    if equations == 0:
        return None
    if not all(isinstance(part, (Eq, Lt, Le, Gt, Ge)) for part in parts):
        return None
    parameters = sorted(formula.free_symbols - set(names), key=lambda s: s.name)
    gens: list[Expr] = [*parameters, *names]
    decomposition = _decomposition(formula, gens)
    if decomposition is None:
        return None
    compiled, cad = decomposition
    m = len(parameters)
    values: dict[CADCell, Expr] = {}
    holding = [cell for cell in cad.cells if compiled(cell.signs)]
    if not holding:
        return S.Zero
    codimension = min(sum(1 for i in cell.index[m:] if i % 2 == 0) for cell in holding)
    if codimension == 0:
        return None
    for cell in holding:
        if sum(1 for i in cell.index[m:] if i % 2 == 0) != codimension:
            continue                                        # not of dimension n - k: measure zero
        base = _ancestor_at(cell, m) if m else cell
        parameter_condition = _parameter_condition(cell, cad, m) if m else true
        if m and ask(parameter_condition, assumptions) is False:
            continue
        found = _section_value(cell, cad, m, f, names, assumptions + ([parameter_condition] if m else []),
                               integrator)
        if found is None:
            return None
        value: Expr = found
        if m:
            refined = attempt(lambda: as_expr(refine(value, assumptions + [parameter_condition])),
                              settings.timeout, native=True)
            if refined is not None:
                value = refined
        values[base] = values[base] + value if base in values else value
    return _cases(values, cad, m, assumptions)
