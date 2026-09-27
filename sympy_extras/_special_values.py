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

4. **A point where the problem is not defined gets no case**: the
   caller's ``defined`` tells, and :func:`defined_problem` is the rule of
   the integrals: the integrand at the point must be a function of the
   variable on a set of positive measure, so a point which leaves in it
   ``nan``, ``zoo``, a power ``0**e`` which SymPy leaves unevaluated (the
   sign of ``e`` unknown: ``a**(b*z)`` at ``a = 0`` is ``0`` or ``zoo``
   according to the sign of ``b*z``, ``e**(1/x)`` at ``e = 0`` is not
   defined for ``x < 0``) or a ``DiracDelta`` of a constant
   (``DiracDelta(a*u)`` at ``a = 0``) is left out, and so is the problem
   which would be solved again there. A caller may also refuse the cases
   whose value at the point is unevaluated (``keep_unevaluated``): the
   antiderivatives do, an unevaluated ``Integral`` in a case making the
   whole answer unevaluated for a value of the parameters of no interest
   to most users, while a definite integral keeps it, the honest value of
   a problem which is defined there.

The costs are bounded, the cases being an addition to a result already
found: at most ``_MAX_POINTS`` points per value, and the whole work (the
candidates, their factorisations and solutions, the problems at the
points) done within a budget of its own, ``_BUDGET_SHARE`` of the time
limit of the settings and at most ``_REMAINING_SHARE`` of what is left of
an enclosing time limit (:func:`~sympy_extras._timeout.remaining_time`),
so that the cases never make a solved problem miss its time limit; the
points not reached in time, and those whose problem is not solved in
time, are not listed. A candidate which is a parameter itself (``1/k``)
or linear in one is read off without factorisation.

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
from sympy.core.traversal import preorder_traversal
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
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Boolean, Or, false, true
from sympy.polys.polytools import Poly, factor
from sympy.polys.polyroots import roots
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset

from sympy_extras._timeout import attempt, remaining_time
from sympy_extras._typing import as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.sat import satisfiable
from sympy_extras.settings import settings

__all__ = ['Point', 'special_points', 'with_special_values', 'point_condition', 'condition_point', 'point_assumptions',
           'undefined_at', 'equality_points', 'defined_problem']

#: the values of some parameters, ``{a: b, b: 0}``
Point = dict[Symbol, Expr]

#: a problem solved at a point, under the assumptions at the point: its
#: value, or ``None`` when it cannot be computed
Compute = Callable[[Point, Assumptions], Optional[Expr]]

#: the most points listed for one value
_MAX_POINTS = 4
#: the seconds given to the factorisation or the solution of one candidate
_CANDIDATE_SECONDS = 2.0
#: the share of ``settings.timeout`` given to the special values of one
#: value (the candidates and the problems at the points together)
_BUDGET_SHARE = 0.1
#: the largest share of what is left of an enclosing time limit they take
_REMAINING_SHARE = 0.5
#: the largest degree of a polynomial in one parameter whose irrational
#: real roots are listed (explicit square roots)
_MAX_RADICAL_DEGREE = 2


class _Deadline:
    """The end of a budget of seconds (in the units of the settings, like
    the time limits), or no end for ``None``."""

    def __init__(self, seconds: Optional[float]) -> None:
        self.end: Optional[float] = None if seconds is None else time.monotonic() + seconds * settings.time_scale

    def left(self, cap: Optional[float] = None) -> Optional[float]:
        """The seconds left, at most ``cap``; ``None`` for no limit at all
        (no end and no cap). A value which is not positive means that the
        time is up (a non-positive limit means no limit to
        :func:`~sympy_extras._timeout.attempt`, so the callers stop
        instead)."""
        if self.end is None:
            return cap
        left = (self.end - time.monotonic()) / settings.time_scale
        return left if cap is None else min(cap, left)

    def up(self) -> bool:
        left = self.left()
        return left is not None and left <= 0


def special_points(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions = None,
                   seconds: Optional[float] = None) -> list[Point]:
    """The candidate special values of the ``parameters`` for ``value``:
    the points where a denominator, the argument of a logarithm or
    ``s - 1`` for ``zeta(s)`` vanishes, when they are described by
    equations in one parameter each (see the module documentation);
    real values only, those the assumptions and the flags of the
    symbols refute left out. With ``seconds``, the candidates found
    within that many seconds (those not reached are left out).

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
    return _special_points(value, parameters, assumptions, _Deadline(seconds))


def _special_points(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions,
                    deadline: _Deadline) -> list[Point]:
    wanted = set(parameters)
    found: list[Point] = []
    for part in _undefined_parts(value, wanted):
        if deadline.up():
            break
        for point in _zeros(part, wanted, deadline):
            if point not in found and _point_assumptions(point, assumptions, deadline) is not None:
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


def _zeros(part: Expr, parameters: set[Symbol], deadline: _Deadline) -> list[Point]:
    """The zeros of ``part`` in the parameters which are points of the
    form of the module documentation, factor by factor; a factor with
    another symbol (the variable of integration) gives none. A part
    which is a parameter, or is linear in the parameters, is not
    factored (the factorisation of every candidate made the cases cost
    more than the integrals of the census)."""
    symbols = free_symbols(part)
    if not symbols & parameters:
        return []
    if isinstance(part, Symbol):
        return [{part: S.Zero}]
    factors: list[Basic] = []
    if symbols <= parameters and part.is_polynomial(*sorted_symbols(symbols)) \
            and Poly(part, *sorted_symbols(symbols)).total_degree() == 1:
        factors = [part]
    else:
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        factored = attempt(lambda: as_expr(factor(part)), seconds)
        factors = list(Mul.make_args(part if factored is None else factored))
    points: list[Point] = []
    for factor_ in factors:
        g = as_expr(factor_)
        if isinstance(g, Pow) and as_expr(g.exp).is_positive:
            g = as_expr(g.base)
        g_symbols = sorted_symbols(free_symbols(g))
        if not g_symbols or set(g_symbols) - parameters:
            continue
        for point in _factor_zeros(g, g_symbols, deadline):
            if _real_for_real_parameters(list(point.values())[0]) and point not in points:
                points.append(point)
    return points


def _factor_zeros(g: Expr, symbols: list[Symbol], deadline: _Deadline) -> list[Point]:
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
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        found = attempt(lambda: roots(polynomial, filter='R'), seconds)
        if not found:
            return []
        return [{p: as_expr(r)} for r in found
                if polynomial.degree() <= _MAX_RADICAL_DEGREE or as_expr(r).is_rational]
    for p in symbols:
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        solutions = attempt(lambda: solveset(g, p, S.Reals), seconds)
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
    return _point_assumptions(point, assumptions, _Deadline(None))


def _point_assumptions(point: Point, assumptions: Assumptions, deadline: _Deadline) -> Optional[list[Boolean]]:
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
    seconds = deadline.left(_CANDIDATE_SECONDS)
    if substituted and (seconds is None or seconds > 0) \
            and attempt(lambda: satisfiable(as_boolean(And(*substituted))), seconds) is False:
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


def defined_problem(integrand: Expr, variables: Sequence[Symbol]) -> bool:
    """Whether ``integrand``, a problem with a point of the parameters
    substituted, is still defined as a function of the ``variables`` (the
    variables of integration) on a set of positive measure (the rule of
    the module documentation): not when ``nan`` or ``zoo`` appears
    (``log(0)``, a division by zero), when a power ``0**e`` stays
    unevaluated (SymPy evaluates it for an exponent of known sign, so
    that its value, ``0``, ``1`` or ``zoo``, depends on the sign of an
    ``e`` which is not known: ``a**(b*z)`` at ``a = 0``), or when a
    ``DiracDelta`` of an expression free of the variables remains (the
    delta of a constant, ``DiracDelta(a*u)`` at ``a = 0``, is no
    function).

    >>> from sympy import symbols, DiracDelta, Integer
    >>> from sympy_extras._special_values import defined_problem
    >>> a, b, z = symbols('a b z')
    >>> defined_problem((a**(b*z)/z).subs(a, 0), [z]), defined_problem((a**(b*z)/z).subs(a, 1), [z])
    (False, True)
    >>> defined_problem(DiracDelta(a*z).subs(a, 0), [z]), defined_problem(Integer(0)**2*z, [z])
    (False, True)
    """
    if integrand.has(nan, zoo):
        return False
    wanted = set(variables)
    for node in preorder_traversal(integrand):
        if isinstance(node, Pow) and as_expr(node.base).is_zero:
            return False
        if isinstance(node, DiracDelta) and not free_symbols(as_expr(node.args[0])) & wanted:
            return False
    return True


def with_special_values(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions,
                        compute: Compute, defined: Optional[Callable[[Point], bool]] = None,
                        sympy_style: bool = False, budget: Optional[float] = None,
                        keep_unevaluated: bool = True) -> Expr:
    """``value`` with a case for each isolated value of the ``parameters``
    at which the branch it takes there is undefined (see the module
    documentation), computed by ``compute(point, assumptions at the
    point)``; ``defined(point)`` tells whether the problem itself is
    defined at the point (points where it is not are left out, and so
    are those where ``compute`` gives ``nan`` or ``zoo``, and, unless
    ``keep_unevaluated``, those where it leaves an unevaluated
    ``Integral``, ``Sum`` or ``Product``).

    The cases come first, as ``(value at the point, Eq(p, v))``, the
    branches of ``value`` after them; with ``sympy_style`` a single case
    for a value which is not a ``Piecewise`` is written the way SymPy's
    ``integrate`` writes it, ``Piecewise((generic, Ne(p, v)), (special,
    True))``. ``budget`` bounds the seconds spent on the whole (the
    candidates and the problems at the points); when ``None``,
    ``_BUDGET_SHARE`` of the time limit of the settings, and never more
    than ``_REMAINING_SHARE`` of what is left of an enclosing time limit.
    The points not reached in time, those whose problem takes longer and
    those which ``compute`` cannot solve are left out.

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
    limit = budget
    if limit is None and settings.timeout is not None:
        limit = _BUDGET_SHARE * settings.timeout
    enclosing = remaining_time()
    if enclosing is not None:
        limit = _REMAINING_SHARE * enclosing if limit is None else min(limit, _REMAINING_SHARE * enclosing)
    deadline = _Deadline(limit)
    branches = _branches(value)
    points: list[Point] = []
    for e, _ in branches:
        if _unevaluated(e):
            continue
        for point in _special_points(e, parameters, assumptions, deadline):
            if point not in points:
                points.append(point)
    cases: list[tuple[Expr, Boolean, int]] = []
    listed = 0
    for point in points:
        if listed >= _MAX_POINTS or deadline.up():
            break
        at = _point_assumptions(point, assumptions, deadline)
        failing = None if at is None else _taken_undefined(branches, point)
        if failing is None:
            continue
        if defined is not None and not defined(point):
            continue
        remaining = deadline.left()
        if remaining is not None and remaining <= 0:
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
        if not keep_unevaluated and _unevaluated(special):
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
