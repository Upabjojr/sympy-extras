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
   flags of the symbols allow. A caller may add candidates where the
   formula is defined but degenerate (``degenerate``): the zeros of the
   Wronskian of the fundamental system in a general solution of an ODE,
   ``C1*exp(a*x) + C2*exp(b*x)`` at ``a = b``
   (:mod:`sympy_extras.solvers._special_values`); they are kept when the
   branch taken there is the degenerate one.

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

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras._special_values_sympy` is its face on SymPy objects
(the integrators and the sums written on SymPy use it). The candidates
are found by the native factorisation, roots and transcendental solver
(:func:`~sympy_extras.polys.polytools.factor`,
:func:`~sympy_extras.polys.polyroots.roots`,
:func:`~sympy_extras.solvers.transcendental.solve_transcendental`) and
the assumptions at a point decided by the native
:func:`~sympy_extras.assumptions.sat.satisfiable`; the real roots of a
polynomial whose coefficients are not rational (``a**2 - pi``) by the
formulas over ``EX`` of :func:`~sympy_extras.polys.polyroots.ex_roots`.
One step goes through SymPy at run time when the native code does not
take it: the solutions of a transcendental factor which the native solver
does not reduce
(:func:`~sympy_extras.solvers.transcendental_sympy.solveset_of_core`).

Examples
========

>>> from sympy_extras.core import symbols, Integer
>>> from sympy_extras.functions import log
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

from sympy_extras._timeout import attempt, remaining_time
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.concrete.expr_with_limits import Product, Sum
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import And, Boolean, BooleanFalse, BooleanTrue, Or, and_, or_, to_boolean, true
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, NegativeInfinity, nan, oo, zoo
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq, eq, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Log
from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
from sympy_extras.functions.special.delta_functions import DiracDelta
from sympy_extras.functions.special.zeta_functions import Zeta
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import factor
from sympy_extras.sets.sets import FiniteSet, Set
from sympy_extras.settings import settings

__all__ = ['Point', 'Compute', 'special_points', 'with_special_values', 'point_condition', 'condition_point',
           'point_assumptions', 'undefined_at', 'equality_points', 'defined_problem']

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
#: the names of the unevaluated integrals, sums and products of SymPy
#: (wrapped in nodes of the core), which are not looked into
_UNEVALUATED_FOREIGN = ('Integral', 'Sum', 'Product')


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


# ---------------------------------------------------------------------------
# helpers on the objects of the core

def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def _boolean(b: Basic) -> Boolean:
    if not isinstance(b, Boolean):
        raise TypeError("a Boolean is expected, got %s" % (b,))
    return b


def _replaced(e: Expr, point: Point) -> Expr:
    """``e`` with the point substituted, evaluated (SymPy's ``xreplace``)."""
    rule: dict[Basic, Basic] = {p: v for p, v in point.items()}
    return _expr(e._xreplace_evaluated(rule))


def _replaced_boolean(b: Boolean, point: Point) -> Boolean:
    rule: dict[Basic, Basic] = {p: v for p, v in point.items()}
    return _boolean(b._xreplace_evaluated(rule))


def _symbols(e: Basic) -> set[Symbol]:
    """The free symbols of ``e`` which are plain symbols."""
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _sorted(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _infinite(e: Expr) -> bool:
    return e.has(oo, NegativeInfinity())


def _coefficients(e: Expr, p: Symbol) -> Optional[dict[int, Expr]]:
    """The nonzero coefficients of ``e`` as a polynomial in ``p``, by
    degree (SymPy's ``Poly(e, p)``, whose coefficients may be any
    expression free of ``p``), ``None`` when it is not one."""
    gathered: dict[int, list[Expr]] = {}
    for t in Add.make_args(expand(e)):
        c, k = t.as_coeff_exponent(p)
        if c.has(p) or not isinstance(k, Integer) or k.p < 0:
            return None
        gathered.setdefault(int(k.p), []).append(c)
    result: dict[int, Expr] = {}
    for degree, cs in gathered.items():
        coefficient = add(*cs)
        if coefficient != 0:
            result[degree] = coefficient
    return result


def _total_degree(e: Expr, symbols: Sequence[Symbol]) -> Optional[int]:
    """The total degree of ``e`` in the ``symbols`` (SymPy's
    ``Poly(e, *symbols).total_degree()``), ``None`` when it is not a
    polynomial in them."""
    degree = 0
    for t in Add.make_args(expand(e)):
        rest = t
        d = 0
        for s in symbols:
            c, k = rest.as_coeff_exponent(s)
            if c.has(s) or not isinstance(k, Integer) or k.p < 0:
                return None
            d += int(k.p)
            rest = c
        if rest != 0:
            degree = max(degree, d)
    return degree


def _unevaluated(e: Expr) -> bool:
    """Whether ``e`` has an unevaluated sum or product, or an unevaluated
    integral (of SymPy, wrapped in a node of the core)."""
    for node in preorder_traversal(e):
        if isinstance(node, (Sum, Product)):
            return True
        if node._is_foreign_node() and node._compare_name() in _UNEVALUATED_FOREIGN:
            return True
    return False


def _not_looked_into(node: Basic) -> bool:
    if isinstance(node, (Sum, Product)):
        return True
    return node._is_foreign_node() and node._compare_name() in _UNEVALUATED_FOREIGN


# ---------------------------------------------------------------------------
# the candidates

def special_points(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions = None,
                   seconds: Optional[float] = None) -> list[Point]:
    """The candidate special values of the ``parameters`` for ``value``:
    the points where a denominator, the argument of a logarithm or
    ``s - 1`` for ``zeta(s)`` vanishes, when they are described by
    equations in one parameter each (see the module documentation);
    real values only, those the assumptions and the flags of the
    symbols refute left out. With ``seconds``, the candidates found
    within that many seconds (those not reached are left out).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp, sin
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
    return _points_of(_undefined_parts(value, set(parameters)), parameters, assumptions, deadline)


def _points_of(parts: Sequence[Expr], parameters: Sequence[Symbol], assumptions: Assumptions,
               deadline: _Deadline) -> list[Point]:
    """The zeros of the ``parts`` in the ``parameters`` which are points
    (see :func:`_zeros`), those the assumptions refute left out."""
    wanted = set(parameters)
    found: list[Point] = []
    for part in parts:
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
        if _symbols(u) & parameters and u not in parts:
            parts.append(u)

    def walk(node: Basic) -> None:
        if _not_looked_into(node):
            return
        if isinstance(node, Pow):
            exponent = node.exp
            if exponent.is_negative or (_symbols(exponent) & parameters and exponent.is_nonnegative is not True):
                add(node.base)
        elif isinstance(node, Log):
            add(_expr(node.args[0]))
        elif isinstance(node, Zeta) and len(node.args) == 1:
            add(_expr(node.args[0]) - 1)
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
    symbols = _symbols(part)
    if not symbols & parameters:
        return []
    if isinstance(part, Symbol):
        return [{part: Integer(0)}]
    factors: list[Expr] = []
    if symbols <= parameters and part.is_polynomial(*_sorted(symbols)) is True \
            and _total_degree(part, _sorted(symbols)) == 1:
        factors = [part]
    else:
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        factored = attempt(lambda: factor(part), seconds, native=True)
        factors = list(Mul.make_args(part if factored is None else factored))
    points: list[Point] = []
    for factor_ in factors:
        g = factor_
        while isinstance(g, Pow) and g.exp.is_positive:
            g = g.base
        g_symbols = _sorted(_symbols(g))
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
    if g.is_polynomial(*symbols) is True:
        polynomials: list[tuple[Symbol, dict[int, Expr]]] = []
        for p in symbols:
            coefficients = _coefficients(g, p)
            if coefficients is None:
                return []
            polynomials.append((p, coefficients))
        for constant in (True, False):
            for p, coefficients in polynomials:
                if max(coefficients, default=0) == 1 and (not constant or not _symbols(coefficients[1])):
                    return [{p: _expr(-coefficients.get(0, Integer(0)) / coefficients[1])}]
        if len(symbols) != 1:
            return []
        p, coefficients = polynomials[0]
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        found = attempt(lambda: _real_roots(g, p), seconds, native=True)
        if not found:
            return []
        degree = max(coefficients, default=0)
        return [{p: r} for r in found if degree <= _MAX_RADICAL_DEGREE or r.is_rational]
    for p in symbols:
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if seconds is not None and seconds <= 0:
            return []
        variable = p
        solutions = attempt(lambda: _real_solutions(g, variable), seconds, native=True)
        if isinstance(solutions, FiniteSet) and 0 < len(solutions) <= _MAX_POINTS:
            return [{p: v} for v in solutions]
    return []


def _real_roots(g: Expr, p: Symbol) -> list[Expr]:
    """The real roots of a polynomial in ``p`` with constant coefficients
    (SymPy's ``roots(g, filter='R')``): by the native roots, for
    coefficients the native domains do not take (``pi``) by the formulas
    over ``EX`` (none when they are not ported: a cubic or a quartic)."""
    try:
        found = roots(g, p, filter='R')
    except BasePolynomialError:
        from sympy_extras.polys.polyroots import ex_roots
        try:
            listed = ex_roots(g, p)
        except NotImplementedError:
            return []
        return [r for r in listed if all(a.is_real for a in r.as_numer_denom())]
    return [r for r in found if isinstance(r, Expr)]


def _real_solutions(g: Expr, p: Symbol) -> Optional[Set]:
    """The real solutions of ``g = 0`` in ``p`` (SymPy's ``solveset(g, p,
    S.Reals)``): by the native transcendental solver, by SymPy's solveset
    when it does not reduce the equation."""
    from sympy_extras.solvers.transcendental import solve_transcendental
    found = solve_transcendental(Eq(g, Integer(0)), p)
    if found is not None:
        return found
    from sympy_extras.solvers.transcendental_sympy import solveset_of_core
    return solveset_of_core(Eq(g, Integer(0)), p)


def _real_for_real_parameters(v: Expr) -> bool:
    """Whether ``v`` is real for real nonzero values of its symbols (``1/b``
    is, ``sqrt(b)`` is not known to be)."""
    replacement: Point = {s: Dummy(s.name, real=True, nonzero=True) for s in _sorted(_symbols(v))}
    return _replaced(v, replacement).is_extended_real is True


# ---------------------------------------------------------------------------
# points and conditions

def point_condition(point: Point) -> Boolean:
    """``Eq(p, v)`` for every parameter of the point.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras._special_values import point_condition
    >>> a, b = symbols('a b')
    >>> point_condition({a: Integer(0), b: Integer(0)})
    Eq(a, 0) & Eq(b, 0)
    """
    return and_(*[eq(p, v) for p, v in sorted(point.items(), key=lambda item: item[0].name)])


def _assumption_items(assumptions: Assumptions) -> list[Boolean]:
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(a) for a in assumptions]


def point_assumptions(point: Point, assumptions: Assumptions) -> Optional[list[Boolean]]:
    """The assumptions with the point substituted, those which become true
    dropped; ``None`` when one becomes false, or when the flags of a
    symbol refute the point (``Eq(y, 0)`` for a positive ``y``).

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras._special_values import point_assumptions
    >>> a, b = symbols('a b')
    >>> point_assumptions({a: b}, [a > 0, b < 2])
    [b > 0, b < 2]
    >>> point_assumptions({a: Integer(0)}, a > 0) is None
    True
    """
    return _point_assumptions(point, assumptions, _Deadline(None))


def _point_assumptions(point: Point, assumptions: Assumptions, deadline: _Deadline) -> Optional[list[Boolean]]:
    if any(isinstance(eq(p, v), BooleanFalse) for p, v in point.items()):
        return None
    substituted: list[Boolean] = []
    for item in _assumption_items(assumptions):
        try:
            at = _replaced_boolean(item, point)
        except TypeError:
            # a comparison which cannot be made at the point: kept unsubstituted
            substituted.append(item)
            continue
        if isinstance(at, BooleanFalse):
            return None
        if not isinstance(at, BooleanTrue):
            substituted.append(at)
    seconds = deadline.left(_CANDIDATE_SECONDS)
    if substituted and (seconds is None or seconds > 0):
        from sympy_extras.assumptions.sat import satisfiable
        if attempt(lambda: satisfiable(and_(*substituted)), seconds, native=True) is False:
            # (s > s under k = s, for s > k: the relations of plain symbols do
            # not evaluate)
            return None
    return substituted


def undefined_at(e: Expr, point: Point) -> bool:
    """Whether ``e`` is undefined at the point: ``nan``, ``zoo`` or an
    infinity it did not have appears when the point is substituted.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras._special_values import undefined_at
    >>> a = symbols('a')
    >>> undefined_at(sin(a)/a, {a: Integer(0)}), undefined_at(sin(a)/a, {a: Integer(1)})
    (True, False)
    """
    try:
        at = _replaced(e, point)
    except (TypeError, ValueError, ZeroDivisionError, RecursionError):
        return True
    if at.has(nan, zoo):
        return True
    return _infinite(at) and not _infinite(e)


def _branches(value: Expr) -> list[tuple[Expr, Boolean]]:
    if isinstance(value, Piecewise):
        return [(pair.expr, pair.cond) for pair in value.args]
    return [(value, true)]


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
            at = _replaced_boolean(condition, point)
        except TypeError:
            at = condition
        if isinstance(at, BooleanFalse):
            continue
        failing = 'fallback' if _unevaluated(e) else 'undefined' if undefined_at(e, point) else None
        if failing == 'undefined' and not isinstance(at, BooleanTrue):
            # a branch which may not be taken there: the point may lie on
            # the boundary of its region (1/(x**2 - 1) over (p, q) at p = 1)
            failing = 'fallback'
        if failing is not None or isinstance(at, BooleanTrue):
            return failing
    return None


def _taken_branch(branches: Sequence[tuple[Expr, Boolean]], point: Point) -> Optional[int]:
    """The index of the branch taken at the point: the first whose
    condition is not false there, when it is true there; ``None`` when
    the point leaves it undecided."""
    for index, (_, condition) in enumerate(branches):
        try:
            at = _replaced_boolean(condition, point)
        except TypeError:
            return None
        if isinstance(at, BooleanFalse):
            continue
        return index if isinstance(at, BooleanTrue) else None
    return None


def _disjuncts(condition: Boolean) -> list[Boolean]:
    if isinstance(condition, Or):
        return [_boolean(c) for c in condition.args]
    return [condition]


def _flattened(special: Expr, point: Point) -> list[tuple[Expr, Boolean, int]]:
    """The value at the point as cases ``(value, condition, equations)``:
    the branches of a ``Piecewise`` under the point's equations and their
    own conditions, an inner point ``Eq(b, 0)`` composed with the outer
    ``Eq(a, b)`` into ``Eq(a, 0) & Eq(b, 0)``."""
    cases: list[tuple[Expr, Boolean, int]] = []
    for e, condition in _branches(special):
        inners = [condition_point(c) for c in _disjuncts(condition)]
        found = [inner for inner in inners if inner is not None]
        if len(found) == len(inners):
            for inner in found:
                composed: Point = {p: _replaced(v, inner) for p, v in point.items()}
                composed.update(inner)
                cases.append((e, point_condition(composed), len(composed)))
        else:
            cases.append((e, and_(point_condition(point), condition), len(point)))
    return cases


def equality_points(value: Expr, parameters: Sequence[Symbol]) -> list[Point]:
    """The points of the branches of a ``Piecewise`` value whose
    conditions are equations (or disjunctions of conjunctions of
    equations) in the ``parameters``: the isolated values, which the
    numerical checks sampling the regions of the branches never reach.

    >>> from sympy_extras.core import symbols, Eq, Integer
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.functions.elementary.piecewise import piecewise
    >>> from sympy_extras._special_values import equality_points
    >>> y, c = symbols('y c')
    >>> equality_points(piecewise((Integer(2), Eq(y, 1)), ((y**2 - 1)/log(y), True)), [y])
    [{y: 1}]
    >>> equality_points(piecewise((Integer(-1), Eq(c, 0) | Eq(c, 1)), (c, c > 1)), [c])
    [{c: 0}, {c: 1}]
    """
    points: list[Point] = []
    wanted = set(parameters)
    for _, condition in _branches(value):
        for disjunct in _disjuncts(condition):
            point = condition_point(disjunct)
            if point and set(point) <= wanted and point not in points:
                points.append(point)
    return points


def condition_point(condition: Boolean) -> Optional[Point]:
    """The point of a condition which is ``true`` or a conjunction of
    equations ``Eq(p, v)`` with a symbol ``p`` not in ``v``, the
    equations solved; ``None`` for another condition.

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras._special_values import condition_point
    >>> a, b = symbols('a b')
    >>> condition_point(Eq(a, b) & Eq(b, 0)), condition_point(a > 0)
    ({a: 0, b: 0}, None)
    """
    if isinstance(condition, BooleanTrue):
        return {}
    parts = condition.args if isinstance(condition, And) else (condition,)
    point: Point = {}
    for part in parts:
        if not isinstance(part, Eq):
            return None
        lhs, rhs = part.lhs, part.rhs
        if isinstance(lhs, Symbol) and not rhs.has(lhs):
            point[lhs] = rhs
        elif isinstance(rhs, Symbol) and not lhs.has(rhs):
            point[rhs] = lhs
        else:
            return None
    # the equations solved: Eq(a, b) & Eq(b, 0) is a = 0, b = 0
    for _ in range(len(point)):
        solved = point
        point = {p: _replaced(v, solved) for p, v in solved.items()}
    if any(_symbols(v) & set(point) for v in point.values()):
        return None
    return point


def defined_problem(integrand: Expr, variables: Sequence[Symbol]) -> bool:
    """Whether ``integrand``, a problem with a point of the parameters
    substituted, is still defined as a function of the ``variables`` (the
    variables of integration) on a set of positive measure (the rule of
    the module documentation): not when ``nan`` or ``zoo`` appears
    (``log(0)``, a division by zero), when a power ``0**e`` stays
    unevaluated (a power of zero is evaluated for an exponent of known
    sign, so that its value, ``0``, ``1`` or ``zoo``, depends on the sign
    of an ``e`` which is not known: ``a**(b*z)`` at ``a = 0``), or when a
    ``DiracDelta`` of an expression free of the variables remains (the
    delta of a constant, ``DiracDelta(a*u)`` at ``a = 0``, is no
    function).

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.functions import DiracDelta
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
        if isinstance(node, Pow) and node.base.is_zero:
            return False
        if isinstance(node, DiracDelta) and not _symbols(node.args[0]) & wanted:
            return False
    return True


# ---------------------------------------------------------------------------
# the cases

def with_special_values(value: Expr, parameters: Sequence[Symbol], assumptions: Assumptions,
                        compute: Compute, defined: Optional[Callable[[Point], bool]] = None,
                        sympy_style: bool = False, budget: Optional[float] = None,
                        keep_unevaluated: bool = True,
                        degenerate: Optional[Callable[[Expr], list[Expr]]] = None) -> Expr:
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

    ``degenerate(e)``, for a branch ``e`` of ``value``, gives expressions
    whose zeros in the parameters are points where ``e`` is defined but
    is not the value of the problem (the Wronskian of the fundamental
    system of a general solution of an ODE, which vanishes where two
    characteristic roots coincide): those points are candidates too, and
    are kept when the branch taken there is ``e``.

    >>> from sympy_extras.core import symbols, Integer
    >>> from sympy_extras.functions import exp
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
    # the points where a branch is degenerate, with the index of the branch
    degenerate_at: list[tuple[Point, int]] = []
    for index, (e, _) in enumerate(branches):
        if _unevaluated(e):
            continue
        for point in _special_points(e, parameters, assumptions, deadline):
            if point not in points:
                points.append(point)
        seconds = deadline.left(_CANDIDATE_SECONDS)
        if degenerate is None or (seconds is not None and seconds <= 0):
            continue
        branch = e
        callback = degenerate
        parts = attempt(lambda: callback(branch), seconds, native=True)
        for point in _points_of(parts or [], parameters, assumptions, deadline):
            degenerate_at.append((point, index))
            if point not in points:
                points.append(point)
    cases: list[tuple[Expr, Boolean, int]] = []
    listed = 0
    for point in points:
        if listed >= _MAX_POINTS or deadline.up():
            break
        at = _point_assumptions(point, assumptions, deadline)
        failing = None if at is None else _taken_undefined(branches, point)
        if failing is None and at is not None and (point, _taken_branch(branches, point)) in degenerate_at:
            failing = 'undefined'
        if failing is None:
            continue
        if defined is not None and not defined(point):
            continue
        remaining = deadline.left()
        if remaining is not None and remaining <= 0:
            break
        at_point = point
        special = attempt(lambda: compute(at_point, at), remaining, native=True)
        if special is None or special.has(nan, zoo):
            # (the problem at the point has no value either: the sum of
            # 1/((k + a)*(k + a + 1)) from k = 0 has the term 1/0 at a = 0)
            continue
        if failing == 'fallback' and (_unevaluated(special) or _infinite(special)):
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
        generic = or_(*[ne(p, v) for p, v in sorted(single.items(), key=lambda item: item[0].name)])
        return piecewise((value, generic), (cases[0][0], true))
    return piecewise(*[(e, condition) for e, condition, _ in cases], *branches)
