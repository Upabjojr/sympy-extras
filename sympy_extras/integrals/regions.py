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

Two routes are tried around the decomposition [Apostol]_:

* **polar coordinates** first: a disc or an annulus (`r_1^2 < x^2 + y^2 <
  R^2`, or the ball in three variables) with an integrand depending on the
  point only through `x^2 + y^2` is `2\\pi \\int_{r_1}^R g(\\rho)\\, \\rho\\, d\\rho`
  (`4\\pi \\int g(\\rho)\\, \\rho^2\\, d\\rho` for the ball);
* **bounds solved for the last variable** when the decomposition does not
  apply (a condition which is not polynomial, `y < \\exp(x)`) or fails: a
  conjunction of relations linear in the last variable `y` describes, over
  each point of the other variables, the interval `\\max(\\text{lower}) < y <
  \\min(\\text{upper})`; the range of the other variables is cut where the
  two bounds cross (with :func:`sympy_extras.assumptions.solve`), the
  inner integral is computed with :func:`~sympy_extras.integrals.definite_integral`
  and the outer integral by the same function on the remaining variables
  (Fubini's theorem).

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

References
==========

.. [Apostol] T. M. Apostol, *Calculus*, vol. II, 2nd ed., Wiley, 1969,
   sections 11.11 (regions between two graphs, Fubini's theorem) and
   11.27–11.29 (polar coordinates).
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

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.core.numbers import pi
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, Or, true
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


class IntegralByRanges(Expr):
    """The integral of ``integrand`` over the region described by
    ``condition``, a Boolean combination of polynomial inequalities and
    equations in the integration ``variables`` (and parameters).

    The constructor does not evaluate: :meth:`doit` (or
    :func:`integrate_by_ranges`) computes the integral.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.regions import IntegralByRanges
    >>> x, y = symbols('x y')
    >>> region = IntegralByRanges(x*y, (x > 0) & (x < y) & (y < 1))
    >>> region.integrand, region.condition, region.variables
    (x*y, (x > 0) & (x < y) & (y < 1), (x, y))
    >>> region.doit()
    1/8
    """

    def __new__(cls, integrand: ExprLike, condition: object,
                variables: Optional[Sequence[Symbol]] = None) -> IntegralByRanges:
        integrand_ = as_expr(integrand)
        condition_ = as_boolean(condition)
        if variables is None:
            names = sorted_symbols(free_symbols(condition_))
        else:
            names = [as_symbol(v) for v in variables]
        if not names:
            raise ValueError("no integration variable: the condition has no symbols")
        obj = Expr.__new__(cls, integrand_, condition_, Tuple(*names))
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

    def doit(self, assumptions: Assumptions = None, **hints: object) -> Expr:
        """The value, or the integral unchanged when it cannot be computed."""
        return integrate_by_ranges(self.integrand, self.condition, self.variables, assumptions)


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
    candidates = attempt(lambda: sympy_solve(equation, x), settings.timeout)
    if candidates is None or not isinstance(candidates, list):
        return None
    target = as_expr(bound.value).evalf(30)
    matches: list[Expr] = []
    for candidate in candidates:
        c = as_expr(candidate)
        value = as_expr(c.xreplace(point)).evalf(30)
        if value.is_real is False or not value.is_number:
            continue
        difference = as_expr(abs(value - target))
        if difference.is_number and difference < _TOLERANCE:
            matches.append(c)
    if len(matches) != 1:
        return None
    return matches[0]


def _stack(cell: CADCell, cad: CAD, first: int) -> Optional[_Stack]:
    """The explicit bounds of a cell of full dimension in the levels from
    ``first`` (1-based) upwards."""
    ancestors: list[CADCell] = []
    current: Optional[CADCell] = cell
    while current is not None and current.level > 0:
        ancestors.append(current)
        current = current.parent
    ancestors.reverse()
    bounds: list[tuple[Expr, Expr]] = []
    for level in range(first, len(cad.gens) + 1):
        node = ancestors[level - 1]
        parent = node.parent
        if parent is None:
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
        bounds.append((lower, upper))
    return _Stack(cell, bounds)


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
        positive: list[Expr] = []
        rest: list[Expr] = []
        sign = 1
        for f in factors:
            f_ = as_expr(f)
            if f_.is_positive or (not f_.is_number and ask(as_boolean(f_ > 0), assumptions) is True):
                positive.append(f_)
            elif f_.is_negative or (not f_.is_number and ask(as_boolean(f_ < 0), assumptions) is True):
                positive.append(-f_)
                sign = -sign
            else:
                rest.append(f_)
        if not positive:
            return node
        result: Expr = S.One
        for f_ in positive:
            result = result * f_**exponent
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
        piece = _split_roots(value, inner)
        result = definite_integral(piece, (x, lower, upper), outer)
        if result.has(Integral, IntegralByRanges) and lower == -upper and upper != oo \
                and ask(as_boolean(upper > 0), outer) is True:
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
    parts: list[Boolean] = []
    ancestor: Optional[CADCell] = cell
    while ancestor is not None and ancestor.level > levels:
        ancestor = ancestor.parent
    if ancestor is None:
        return true
    for level in range(1, levels + 1):
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
    the relations linear in it (Fubini), see the module documentation."""
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
        found = _linear_bound(atom, y, outer_assumptions)
        if found is None:
            return None
        is_upper, bound = found
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
            return None                                     # the other variables are unbounded
        value = integrate_by_ranges(inner, condition, outer_names, assumptions)
        if value.has(Integral, IntegralByRanges):
            return None
        total = total + value
    return as_expr(total)


def integrate_by_ranges(integrand: ExprLike, condition: object,
                        variables: Optional[Sequence[Symbol]] = None,
                        assumptions: Assumptions = None) -> Expr:
    """The integral of ``integrand`` over the region ``condition``.

    Parameters
    ==========

    integrand : Expr
    condition : Boolean
        A Boolean combination of polynomial relations (``<``, ``<=``,
        ``>``, ``>=``, ``Eq``, ``Ne``) with rational coefficients, for
        the decomposition; or a conjunction of inequalities linear in the
        last variable (``y < exp(x)``), solved for it.
    variables : list of Symbol, optional
        The integration variables, in the order of the decomposition
        (the last one is integrated first); by default every symbol of
        the condition. The other symbols are parameters.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the parameters; the cases of the parameter space
        they refute are dropped.

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
    """
    node = IntegralByRanges(integrand, condition, variables)
    f, formula, names = node.integrand, normalize(node.condition), node.variables
    extra: list[Boolean] = []
    if assumptions is not None:
        extra = [as_boolean(a) for a in ([assumptions] if isinstance(assumptions, (Boolean, bool))
                                         else assumptions)]
    radial = _radial(f, formula, names, extra)
    if radial is not None:
        return radial
    found = _decomposed(node, f, formula, names, assumptions, extra)
    if found is not None:
        return found
    solved = _solved_bounds(f, formula, names, extra)
    return node if solved is None else solved


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
        stack = _stack(cell, cad, m + 1)
        if stack is None:
            return None
        base = _ancestor_at(cell, m) if m else cell
        parameter_condition = _parameter_condition(cell, cad, m) if m else true
        if m and ask(parameter_condition, assumptions) is False:
            continue
        found = _iterated(f, names, stack, extra + ([parameter_condition] if m else []))
        if found is None:
            return None
        value: Expr = found
        if m:
            refined = attempt(lambda: as_expr(refine(value, extra + [parameter_condition])), settings.timeout)
            if refined is not None:
                value = refined
        values[base] = as_expr(values.get(base, S.Zero) + value)
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

