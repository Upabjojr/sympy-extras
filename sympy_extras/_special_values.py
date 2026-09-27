"""The isolated values of the parameters at which a closed form fails.

A closed form computed for generic values of the parameters may be
undefined at isolated values of them where the problem itself is not:
the integral of ``y**x`` over ``(0, 2)`` is ``(y**2 - 1)/log(y)``, which
is ``0/0`` at ``y = 1``, where the integral is ``2``; the integral of
``exp(k*x)`` over ``(0, 1)`` is ``(exp(k) - 1)/k``, undefined at ``k =
0``, where the integral is ``1``. The results of the package carry such
values as cases of their own, before the generic one, as SymPy's
``integrate`` does for ``x**n`` at ``n = -1``:
``Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True))``.

The mechanism is independent of the problem (:func:`with_special_values`):

1. **The candidates** are the zeros, in the parameters, of what makes the
   generic formula undefined: the bases of powers with a negative or a
   symbolic exponent (the denominators, ``0**0``), the arguments of
   logarithms, ``s - 1`` for ``zeta(s)`` (:func:`special_points`). Each
   is factored, and a factor gives a set of points which is described
   by equations ``Eq(p, v)`` with one parameter ``p`` solved for: a
   factor linear in a parameter with a numerical coefficient (``a - b``
   gives ``a = b``, a hyperplane), the real roots of a polynomial in one
   parameter (explicit ones: of degree two, or rational), the finite
   real solution sets of a transcendental factor in one parameter
   (``log(y)`` gives ``y = 1``, ``exp(k) - 1`` gives ``k = 0``). A factor
   whose zeros are not of this form (``a*b - 1``, ``a**2 + b**2`` over
   the reals, which vanishes at one point only, ``sin(a)``, zero at
   infinitely many points) gives none: those values are not listed.
   Only real values are listed (the conditions of the package are
   written for real parameters), and only those the assumptions and the
   flags of the symbols allow.

2. **A candidate is kept** when the branch of the value taken there is
   undefined at it: its conditions are evaluated at the point in turn
   (a branch whose condition is false there is skipped, so that ``n =
   -1`` is no special value of ``Piecewise((1/(n + 1), n > -1), ...)``
   when another branch covers it), and the value of the first which may
   hold is ``nan``, ``zoo`` or infinite there, or is the unevaluated
   fallback of the caller (the point lies between the cases, as ``c =
   0`` between ``c < 0`` and ``0 < c < 1`` for ``log(Abs(x - c))`` over
   ``(0, 1)``), and the problem itself is defined there. A point of the
   fallback, or of an undefined branch whose condition the point leaves
   undecided, is listed only when the problem there has a finite closed
   form: otherwise it lies on the boundary of a region of divergence
   (``1/x`` over ``(p, q)`` at ``p = 0``) rather than between cases.

3. **The problem is solved again** at the point, by the caller's
   ``compute`` with the point substituted: the value there, which may
   differ from the limit of the generic formula (the limit is only a
   fallback of the callers, checked numerically). The recursion is the
   caller's: a public function which applies this mechanism to its own
   results applies it to the problem at the point as well, whose
   parameters are fewer by one, so that ``sin(a*x)*sin(b*x)`` gets the
   case ``a = b`` and, inside it, ``b = 0``. The cases are flattened
   into one ``Piecewise``: the conditions ``Eq(a, b) & Eq(b, 0)`` become
   ``Eq(a, 0) & Eq(b, 0)``, and the cases with more equations come first.

The costs are bounded: at most ``_MAX_POINTS`` points per value, the
candidates found under a short time limit, and the problems at the
points solved under the time limit of the caller (a point whose problem
is not solved in time is not listed).

Examples
========

>>> from sympy import symbols, log, Integer
>>> from sympy_extras._special_values import special_points, with_special_values
>>> y, k, x = symbols('y k x')
>>> special_points((y**2 - 1)/log(y), [y])
[{y: 0}, {y: 1}]
>>> with_special_values((y**2 - 1)/log(y), [y], None, lambda point, assumptions: Integer(2))
Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True))
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import nan, oo, zoo
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, Or, false, true
from sympy.polys.polytools import Poly, factor
from sympy.polys.polyroots import roots
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.sat import satisfiable
from sympy_extras.settings import settings

__all__ = ['Point', 'special_points', 'with_special_values', 'point_condition', 'condition_point', 'point_assumptions',
           'undefined_at', 'equality_points']

#: the values of some parameters, ``{a: b, b: 0}``
Point = dict[Symbol, Expr]

#: a problem solved at a point, under the assumptions at the point: its
#: value, or ``None`` when it cannot be computed
Compute = Callable[[Point, Assumptions], Optional[Expr]]

#: the most points listed for one value
_MAX_POINTS = 4
#: the seconds given to the factorisation or the solution of one candidate
_CANDIDATE_SECONDS = 2.0
#: the largest degree of a polynomial in one parameter whose irrational
#: real roots are listed (explicit square roots)
_MAX_RADICAL_DEGREE = 2


def special_points(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions = None) -> list[Point]:
    """The candidate special values of the ``parameters`` for ``value``:
    the points where a denominator, the argument of a logarithm or
    ``s - 1`` for ``zeta(s)`` vanishes, when they are described by
    equations in one parameter each (see the module documentation);
    real values only, those the assumptions and the flags of the
    symbols refute left out.

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras._special_values import special_points
    >>> a, b, k = symbols('a b k')
    >>> special_points((exp(k) - 1)/k, [k])
    [{k: 0}]
    >>> special_points(1/(a**2 - b**2), [a, b])
    [{a: -b}, {a: b}]
    >>> special_points(1/(a**2 + 1), [a]), special_points(1/sin(a), [a])
    ([], [])
    """
    wanted = set(parameters)
    found: list[Point] = []
    for part in _undefined_parts(value, wanted):
        for point in _zeros(part, wanted):
            if point not in found and point_assumptions(point, assumptions) is not None:
                found.append(point)
    return sorted(found, key=lambda point: str(sorted(point.items(), key=str)))


def _undefined_parts(e: Expr, parameters: set[Symbol]) -> list[Expr]:
    """The expressions of ``e`` whose zeros may leave it undefined: the
    bases of powers with a negative or a symbolic exponent, the arguments
    of logarithms, ``s - 1`` for ``zeta(s)``; the unevaluated integrals,
    sums and products are not looked into."""
    parts: list[Expr] = []

    def add(u: Expr) -> None:
        if free_symbols(u) & parameters and u not in parts:
            parts.append(u)

    def walk(node: Basic) -> None:
        if isinstance(node, (Integral, Sum, Product)):
            return
        if isinstance(node, Pow):
            exponent = as_expr(node.exp)
            if exponent.is_negative or (free_symbols(exponent) & parameters and exponent.is_nonnegative is not True):
                add(as_expr(node.base))
        elif isinstance(node, log):
            add(as_expr(node.args[0]))
        elif isinstance(node, zeta) and len(node.args) == 1:
            add(as_expr(node.args[0]) - 1)
        for argument in node.args:
            walk(argument)

    walk(e)
    return parts


def _zeros(part: Expr, parameters: set[Symbol]) -> list[Point]:
    """The zeros of ``part`` in the parameters which are points of the
    form of the module documentation, factor by factor; a factor with
    another symbol (the variable of integration) gives none."""
    factored = attempt(lambda: as_expr(factor(part)), _CANDIDATE_SECONDS)
    if factored is None:
        factored = part
    points: list[Point] = []
    for factor_ in Mul.make_args(factored):
        g = as_expr(factor_)
        if isinstance(g, Pow) and as_expr(g.exp).is_positive:
            g = as_expr(g.base)
        symbols = sorted_symbols(free_symbols(g))
        if not symbols or set(symbols) - parameters:
            continue
        for point in _factor_zeros(g, symbols):
            if _real_for_real_parameters(list(point.values())[0]) and point not in points:
                points.append(point)
    return points


def _factor_zeros(g: Expr, symbols: list[Symbol]) -> list[Point]:
    """The zeros of an irreducible factor: ``p = v`` for the first
    parameter ``p`` in which it is linear (with a numerical coefficient
    first, then with any), the real roots of a polynomial in one
    parameter, or the finite real solution set in the first parameter
    which has one."""
    if g.is_polynomial(*symbols):
        polynomials = [(p, Poly(g, p)) for p in symbols]
        for constant in (True, False):
            for p, polynomial in polynomials:
                if polynomial.degree() == 1 and (not constant or not free_symbols(as_expr(polynomial.LC()))):
                    return [{p: as_expr(-polynomial.nth(0) / polynomial.LC())}]
        if len(symbols) != 1:
            return []
        p, polynomial = polynomials[0]
        found = attempt(lambda: roots(polynomial, filter='R'), _CANDIDATE_SECONDS)
        if not found:
            return []
        return [{p: as_expr(r)} for r in found
                if polynomial.degree() <= _MAX_RADICAL_DEGREE or as_expr(r).is_rational]
    for p in symbols:
        solutions = attempt(lambda: solveset(g, p, S.Reals), _CANDIDATE_SECONDS)
        if isinstance(solutions, FiniteSet) and 0 < len(solutions) <= _MAX_POINTS:
            return [{p: as_expr(v)} for v in solutions]
    return []


def _real_for_real_parameters(v: Expr) -> bool:
    """Whether ``v`` is real for real nonzero values of its symbols (``1/b``
    is, ``sqrt(b)`` is not known to be)."""
    replacement = {s: Dummy(s.name, real=True, nonzero=True) for s in sorted_symbols(free_symbols(v))}
    return as_expr(v.xreplace(replacement)).is_extended_real is True


def point_condition(point: Point) -> Boolean:
    """``Eq(p, v)`` for every parameter of the point.

    >>> from sympy import symbols
    >>> from sympy_extras._special_values import point_condition
    >>> a, b = symbols('a b')
    >>> point_condition({a: 0, b: 0})
    Eq(a, 0) & Eq(b, 0)
    """
    return as_boolean(And(*[Eq(p, v) for p, v in sorted(point.items(), key=lambda item: item[0].name)]))


def point_assumptions(point: Point, assumptions: Assumptions) -> Optional[list[Boolean]]:
    """The assumptions with the point substituted, those which become true
    dropped; ``None`` when one becomes false, or when the flags of a
    symbol refute the point (``Eq(y, 0)`` for a positive ``y``).

    >>> from sympy import symbols
    >>> from sympy_extras._special_values import point_assumptions
    >>> a, b = symbols('a b')
    >>> point_assumptions({a: b}, [a > 0, b < 2])
    [b > 0, b < 2]
    >>> point_assumptions({a: 0}, a > 0) is None
    True
    """
    if any(as_boolean(Eq(p, v)) is false for p, v in point.items()):
        return None
    items: list[Boolean] = []
    if isinstance(assumptions, (Basic, bool)):
        items.append(as_boolean(assumptions))
    elif assumptions is not None:
        items.extend(as_boolean(a) for a in assumptions)
    substituted: list[Boolean] = []
    for item in items:
        try:
            at = as_boolean(item.xreplace(point))
        except TypeError:
            # a comparison SymPy cannot make at the point: kept unsubstituted
            substituted.append(item)
            continue
        if at is false:
            return None
        if at is not true:
            substituted.append(at)
    if substituted and attempt(lambda: satisfiable(as_boolean(And(*substituted))), _CANDIDATE_SECONDS) is False:
        # (s > s under k = s, for s > k: the relations of plain symbols do
        # not evaluate)
        return None
    return substituted


def undefined_at(e: Expr, point: Point) -> bool:
    """Whether ``e`` is undefined at the point: ``nan``, ``zoo`` or an
    infinity it did not have appears when the point is substituted.

    >>> from sympy import symbols, sin
    >>> from sympy_extras._special_values import undefined_at
    >>> a = symbols('a')
    >>> undefined_at(sin(a)/a, {a: 0}), undefined_at(sin(a)/a, {a: 1})
    (True, False)
    """
    try:
        at = as_expr(e.xreplace(point))
    except (TypeError, ValueError, ZeroDivisionError, RecursionError):
        return True
    if at.has(nan, zoo):
        return True
    return at.has(oo, -oo) and not e.has(oo, -oo)


def _branches(value: Expr) -> list[tuple[Expr, Boolean]]:
    if isinstance(value, Piecewise):
        return [(as_expr(pair.args[0]), as_boolean(pair.args[1])) for pair in value.args]
    return [(value, true)]


def _unevaluated(e: Expr) -> bool:
    return e.has(Integral, Sum, Product)


def _taken_undefined(branches: Sequence[tuple[Expr, Boolean]], point: Point) -> Optional[str]:
    """How the branch taken at the point fails there: ``'undefined'``
    when its value is undefined at the point and its condition holds
    there, ``'fallback'`` when it is the unevaluated fallback of the
    caller or an undefined branch whose condition the point leaves
    undecided, ``None`` when it is defined.
    The conditions are evaluated at the point in turn, a branch whose
    condition is false is skipped, and the first whose condition may
    hold decides (one whose condition is undecided and whose value is
    defined leaves the decision to the next)."""
    for e, condition in branches:
        try:
            at = as_boolean(condition.xreplace(point))
        except TypeError:
            at = condition
        if at is false:
            continue
        failing = 'fallback' if _unevaluated(e) else 'undefined' if undefined_at(e, point) else None
        if failing == 'undefined' and at is not true:
            # a branch which may not be taken there: the point may lie on
            # the boundary of its region (1/(x**2 - 1) over (p, q) at p = 1)
            failing = 'fallback'
        if failing is not None or at is true:
            return failing
    return None


def _flattened(special: Expr, point: Point) -> list[tuple[Expr, Boolean, int]]:
    """The value at the point as cases ``(value, condition, equations)``:
    the branches of a ``Piecewise`` under the point's equations and their
    own conditions, an inner point ``Eq(b, 0)`` composed with the outer
    ``Eq(a, b)`` into ``Eq(a, 0) & Eq(b, 0)``."""
    cases: list[tuple[Expr, Boolean, int]] = []
    for e, condition in _branches(special):
        inners = [condition_point(as_boolean(c)) for c in (condition.args if isinstance(condition, Or) else [condition])]
        if all(inner is not None for inner in inners):
            for inner in inners:
                assert inner is not None
                composed: Point = {p: as_expr(v.xreplace(inner)) for p, v in point.items()}
                composed.update(inner)
                cases.append((e, point_condition(composed), len(composed)))
        else:
            cases.append((e, as_boolean(And(point_condition(point), condition)), len(point)))
    return cases


def equality_points(value: Expr, parameters: Sequence[Symbol]) -> list[Point]:
    """The points of the branches of a ``Piecewise`` value whose
    conditions are equations (or disjunctions of conjunctions of
    equations) in the ``parameters``: the isolated values, which the
    numerical checks sampling the regions of the branches never reach.

    >>> from sympy import symbols, Piecewise, Eq, log
    >>> from sympy_extras._special_values import equality_points
    >>> y, c = symbols('y c')
    >>> equality_points(Piecewise((2, Eq(y, 1)), ((y**2 - 1)/log(y), True)), [y])
    [{y: 1}]
    >>> equality_points(Piecewise((-1, Eq(c, 0) | Eq(c, 1)), (c, c > 1)), [c])
    [{c: 0}, {c: 1}]
    """
    points: list[Point] = []
    wanted = set(parameters)
    for _, condition in _branches(value):
        for disjunct in (condition.args if isinstance(condition, Or) else [condition]):
            point = condition_point(as_boolean(disjunct))
            if point and set(point) <= wanted and point not in points:
                points.append(point)
    return points


def condition_point(condition: Boolean) -> Optional[Point]:
    """The point of a condition which is ``true`` or a conjunction of
    equations ``Eq(p, v)`` with a symbol ``p`` not in ``v``, the
    equations solved; ``None`` for another condition.

    >>> from sympy import symbols, Eq
    >>> from sympy_extras._special_values import condition_point
    >>> a, b = symbols('a b')
    >>> condition_point(Eq(a, b) & Eq(b, 0)), condition_point(a > 0)
    ({a: 0, b: 0}, None)
    """
    if condition is true:
        return {}
    parts = condition.args if isinstance(condition, And) else (condition,)
    point: Point = {}
    for part in parts:
        if not isinstance(part, Eq):
            return None
        lhs, rhs = as_expr(part.lhs), as_expr(part.rhs)
        if isinstance(lhs, Symbol) and not rhs.has(lhs):
            point[lhs] = rhs
        elif isinstance(rhs, Symbol) and not lhs.has(rhs):
            point[rhs] = lhs
        else:
            return None
    # the equations solved: Eq(a, b) & Eq(b, 0) is a = 0, b = 0
    for _ in range(len(point)):
        point = {p: as_expr(v.xreplace(point)) for p, v in point.items()}
    if any(free_symbols(v) & set(point) for v in point.values()):
        return None
    return point


def with_special_values(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions,
                        compute: Compute, defined: Optional[Callable[[Point], bool]] = None,
                        sympy_style: bool = False, budget: Optional[float] = None) -> Expr:
    """``value`` with a case for each isolated value of the ``parameters``
    at which the branch it takes there is undefined (see the module
    documentation), computed by ``compute(point, assumptions at the
    point)``; ``defined(point)`` tells whether the problem itself is
    defined at the point (points where it is not are left out, and so
    are those where ``compute`` gives ``nan`` or ``zoo``).

    The cases come first, as ``(value at the point, Eq(p, v))``, the
    branches of ``value`` after them; with ``sympy_style`` a single case
    for a value which is not a ``Piecewise`` is written the way SymPy's
    ``integrate`` writes it, ``Piecewise((generic, Ne(p, v)), (special,
    True))``. ``budget`` bounds the seconds spent on the problems at the
    points (the time limit of the settings when ``None``); a point whose
    problem takes longer, or which ``compute`` cannot solve, is left out.

    >>> from sympy import symbols, exp, Integer
    >>> from sympy_extras._special_values import with_special_values
    >>> k = symbols('k')
    >>> with_special_values((exp(k) - 1)/k, [k], None, lambda point, assumptions: Integer(1))
    Piecewise((1, Eq(k, 0)), ((exp(k) - 1)/k, True))
    >>> with_special_values(exp(k)/k, [k], None, lambda point, assumptions: Integer(1), sympy_style=True)
    Piecewise((exp(k)/k, Ne(k, 0)), (1, True))
    """
    if not parameters:
        return value
    branches = _branches(value)
    points: list[Point] = []
    for e, _ in branches:
        if _unevaluated(e):
            continue
        for point in special_points(e, parameters, assumptions):
            if point not in points:
                points.append(point)
    limit = settings.timeout if budget is None else budget
    started = time.monotonic()
    cases: list[tuple[Expr, Boolean, int]] = []
    listed = 0
    for point in points:
        if listed >= _MAX_POINTS:
            break
        at = point_assumptions(point, assumptions)
        failing = None if at is None else _taken_undefined(branches, point)
        if failing is None:
            continue
        if defined is not None and not defined(point):
            continue
        remaining: Optional[float] = None
        if limit is not None:
            remaining = limit - (time.monotonic() - started) / settings.time_scale
            if remaining <= 0:
                break
        special = attempt(lambda: compute(point, at), remaining)
        if special is None or special.has(nan, zoo):
            # (the problem at the point has no value either: the sum of
            # 1/((k + a)*(k + a + 1)) from k = 0 has the term 1/0 at a = 0)
            continue
        if failing == 'fallback' and (_unevaluated(special) or special.has(oo, -oo)):
            # a point where the value was left to the fallback, and which
            # the problem at the point does not settle either, or where it
            # diverges: on the boundary of a region of divergence (1/x over
            # (p, q) at p = 0) rather than isolated
            continue
        listed += 1
        for case in _flattened(special, point):
            if all(case[1] != other[1] for other in cases):
                cases.append(case)
    if not cases:
        return value
    cases.sort(key=lambda case: -case[2])
    single = condition_point(cases[0][1]) if sympy_style and len(cases) == 1 and not isinstance(value, Piecewise) else None
    if single:
        generic = as_boolean(Or(*[Ne(p, v) for p, v in sorted(single.items(), key=lambda item: item[0].name)]))
        return as_expr(Piecewise((value, generic), (cases[0][0], True)))
    return as_expr(Piecewise(*[(e, condition) for e, condition, _ in cases], *branches))

