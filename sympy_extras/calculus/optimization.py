r"""Exact global optimisation of polynomials and rational functions under
polynomial constraints.

The set of values taken by $f(x_1, \dots, x_n)$ on the set $C$ defined
by a Boolean combination of polynomial equations and inequalities is

.. math:: \{\, t \in \mathbb{R} : \exists x\ (C(x) \wedge f(x) = t) \,\},

a formula of the first order theory of the real numbers with one free
variable. Quantifier elimination by cylindrical algebraic decomposition
(:func:`sympy_extras.polys.cad.qe.solution_set`) turns it into a finite
union of points and intervals whose endpoints are rational numbers or
real roots of polynomials [1]_ [2]_. Everything about the *global*
optimum is read from that set:

* the infimum and the supremum are its ends ($\pm\infty$ when $f$ is
  unbounded on $C$);
* they are a minimum and a maximum exactly when the ends belong to the
  set, which decides the cases a search for critical points cannot:
  $(x y - 1)^2 + x^2$ is positive everywhere and has the infimum zero;
* the points where an optimum is attained come from the same
  decomposition, with the variable $t$ lowest: the cells of the
  decomposition of $\{(t, x) : C(x) \wedge f(x) = t\}$ above the
  optimal value have sample points with exact coordinates. There is one
  point for each cell: all the optimal points when they are finitely
  many, and a sample of each piece otherwise.

:func:`minimize_parametric` and :func:`maximize_parametric` give the
optimum as a function of parameters, case by case: the set of the lower
bounds $\{m : \forall x\ (C(x) \Rightarrow f(x) \ge m)\}$ is described
cylindrically over the space of the parameters
(:func:`sympy_extras.polys.cad.cylindrical.cylindrical_cases`), its
upper end being the infimum, and the parameters for which $f$ is
unbounded come from one more elimination.

No assumption of convexity, compactness or regularity is made, and no
numerical step is involved. The price is that of the decomposition,
doubly exponential in the number of variables: a few variables of
moderate degree. The coefficients are rational numbers; a rational
function $p/q$ is handled through $p = t q \wedge q \ne 0$.

Examples
========

>>> from sympy_extras.core import Symbol, Eq
>>> from sympy_extras.calculus.optimization import minimize, maximize, value_range
>>> x, y = Symbol('x'), Symbol('y')
>>> minimize((1 - x)**2 + 100*(y - x**2)**2)             # the Rosenbrock function
Optimum(0, attained at [{x: 1, y: 1}])
>>> maximize(x*y, Eq(x**2 + y**2, 1)).value, value_range(x/(1 + x**2))
(1/2, Interval(-1/2, 1/2))
>>> minimize((x*y - 1)**2 + x**2)
Optimum(0, not attained)
>>> minimize(x**4 - 3*x**2 + x)
Optimum(IndexedRoot(256*t**3 + 1152*t**2 + 864*t - 81, t, 0), attained at [{x: IndexedRoot(4*x**3 - 6*x + 1, x, 0)}])
>>> maximize(x + y, (x + 2*y <= 4) & (3*x + y <= 6) & (x >= 0) & (y >= 0))        # a linear program
Optimum(14/5, attained at [{x: 8/5, y: 6/5}])

References
==========

.. [1] G. E. Collins, *Quantifier elimination for real closed fields by
       cylindrical algebraic decomposition*, Lecture Notes in Computer
       Science 33, Springer, 1975, 134-183.
.. [2] A. Strzeboński, *Solving systems of strict polynomial inequalities*
       and the description of ``Minimize`` in the implementation notes of
       Mathematica: global optimisation by cylindrical algebraic
       decomposition, with the objective as the lowest variable.
.. [3] J. Nie, J. Demmel, B. Sturmfels, *Minimizing polynomials via sum
       of squares over the gradient ideal*, Mathematical Programming 106
       (2006), 587-606 (the difficulty of infima which are not attained).
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import And, Boolean, BooleanLike, Implies, to_boolean, true
from sympy_extras.core.numbers import Infinity, NegativeInfinity, Rational, oo
from sympy_extras.core.relational import Eq, Ne
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.sets import EmptySetClass, FiniteSet, Interval, Set, Union

__all__ = ['Optimum', 'value_range', 'minimize', 'maximize', 'minimize_parametric', 'maximize_parametric']


class Optimum:
    """The infimum or supremum ``value`` of a function on a set
    (``-oo`` or ``oo`` when it is unbounded), whether it is ``attained``,
    and then the ``points`` (assignments of the variables) found where it
    is: one for each cell of the decomposition."""

    __slots__ = ('value', 'attained', 'points')

    def __init__(self, value: Expr, attained: bool, points: Sequence[dict[Symbol, Expr]]) -> None:
        self.value: Final[Expr] = value
        self.attained: Final[bool] = attained
        self.points: Final[tuple[dict[Symbol, Expr], ...]] = tuple(points)

    @property
    def is_finite(self) -> bool:
        return not isinstance(self.value, (Infinity, NegativeInfinity))

    def __repr__(self) -> str:
        if not self.attained:
            return 'Optimum(%s, not attained)' % (self.value,)
        return 'Optimum(%s, attained at %s)' % (self.value, list(self.points))


def _fresh(used: set[str]) -> Symbol:
    name, k = 't', 0
    while name in used:
        name, k = 't%d' % k, k + 1
    return Symbol(name)


class _Problem:
    """The formula ``C(x) and f(x) = t`` with its variables."""

    def __init__(self, f: ExprLike, constraints: BooleanLike, variables: Optional[Sequence[Symbol]]) -> None:
        from sympy_extras.matrices.linalg import normal
        from sympy_extras.simplify.radsimp import fraction
        objective = to_expr(f)
        condition = to_boolean(constraints)
        symbols = {s for s in objective.free_symbols if isinstance(s, Symbol)}
        symbols |= {s for s in condition.free_symbols if isinstance(s, Symbol)}
        if variables is None:
            chosen = sorted(symbols, key=default_sort_key)
        else:
            chosen = list(variables)
            if symbols - set(chosen):
                raise ValueError('the parameters %s are not allowed: every symbol is a variable of the optimisation'
                                 % (sorted(symbols - set(chosen), key=default_sort_key),))
        self.variables: Final[list[Symbol]] = chosen
        self.value: Final[Symbol] = _fresh({s.name for s in symbols} | {s.name for s in chosen})
        numerator, denominator = fraction(normal(objective))
        parts: list[Boolean] = [Eq(numerator, self.value*denominator)]
        if denominator.free_symbols:
            parts.append(Ne(denominator, 0))
        if condition is not true:
            parts.append(condition)
        self.formula: Final[Boolean] = parts[0] if len(parts) == 1 else And(*parts)

    def range(self) -> Set:
        from sympy_extras.polys.cad.qe import solution_set
        return solution_set(self.formula, self.value, [('exists', v) for v in self.variables])

    def points(self, optimum: Expr) -> list[dict[Symbol, Expr]]:
        from sympy_extras.assumptions.sat import _compare
        from sympy_extras.polys.cad.qe import sample_points
        found: list[dict[Symbol, Expr]] = []
        for point in sample_points(self.formula, [self.value] + self.variables):
            if _compare(point[self.value], optimum) == 0:
                found.append({v: point[v] for v in self.variables})
        return found


_Piece = tuple[Expr, bool, Expr, bool]          # (low, low belongs, high, high belongs)


def _pieces(s: Set) -> list[_Piece]:
    if isinstance(s, EmptySetClass):
        return []
    if isinstance(s, Interval):
        return [(s.start, not s.left_open, s.end, not s.right_open)]
    if isinstance(s, FiniteSet):
        result: list[_Piece] = []
        for element in s.args:
            if not isinstance(element, Expr):
                raise TypeError('%s is not a number' % (element,))
            result.append((element, True, element, True))
        return result
    if isinstance(s, Union):
        collected: list[_Piece] = []
        for part in s.args:
            if not isinstance(part, Set):
                raise TypeError('%s is not a set' % (part,))
            collected.extend(_pieces(part))
        return collected
    raise NotImplementedError('the set of values %s is not a union of intervals and points' % (s,))


def _less(a: Expr, b: Expr) -> bool:
    """``a < b`` for real algebraic numbers and the infinities."""
    if a == b:
        return False
    if isinstance(a, NegativeInfinity) or isinstance(b, Infinity):
        return True
    if isinstance(a, Infinity) or isinstance(b, NegativeInfinity):
        return False
    from sympy_extras.assumptions.sat import _compare
    return _compare(a, b) < 0


def _extreme(s: Set, lowest: bool) -> tuple[Expr, bool]:
    """The infimum (or supremum) of a nonempty set and whether it
    belongs to it."""
    best: Optional[tuple[Expr, bool]] = None
    for low, low_in, high, high_in in _pieces(s):
        end, inside = (low, low_in) if lowest else (high, high_in)
        if best is None or (_less(end, best[0]) if lowest else _less(best[0], end)):
            best = (end, inside)
        elif end == best[0] and inside:
            best = (end, True)
    if best is None:
        raise ValueError('the constraints have no solution')
    if isinstance(best[0], (Infinity, NegativeInfinity)):
        return best[0], False
    return best


def value_range(f: ExprLike, constraints: BooleanLike = True, variables: Optional[Sequence[Symbol]] = None) -> Set:
    """The set of the values of ``f`` on the points satisfying
    ``constraints``: a union of intervals and points with exact ends
    (empty when the constraints have no solution).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.optimization import value_range
    >>> x, y = Symbol('x'), Symbol('y')
    >>> value_range(x**2 + y**2, x*y > 1), value_range(x/y, (x > 0) & (y > 0) & (x + y < 1))
    (Interval.open(2, oo), Interval.open(0, oo))
    >>> value_range(1/x)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    """
    return _Problem(f, constraints, variables).range()


def _optimize(f: ExprLike, constraints: BooleanLike, variables: Optional[Sequence[Symbol]], lowest: bool) -> Optimum:
    problem = _Problem(f, constraints, variables)
    value, attained = _extreme(problem.range(), lowest)
    return Optimum(value, attained, problem.points(value) if attained else [])


def minimize(f: ExprLike, constraints: BooleanLike = True, variables: Optional[Sequence[Symbol]] = None) -> Optimum:
    """The global infimum of the polynomial or rational function ``f``
    (rational coefficients) over the real points satisfying
    ``constraints``, a Boolean combination of polynomial equations and
    inequalities: its exact value, whether it is a minimum, and points
    where it is attained (see :class:`Optimum`). ``ValueError`` when the
    constraints have no solution. ``variables`` (by default all the
    symbols) fixes the order of the variables in the decomposition.

    >>> from sympy_extras.core import Symbol, Eq
    >>> from sympy_extras.calculus.optimization import minimize
    >>> x, y = Symbol('x'), Symbol('y')
    >>> minimize(x**2 + y**2, Eq(x + y, 1))
    Optimum(1/2, attained at [{x: 1/2, y: 1/2}])
    >>> minimize(x**3 - x), minimize(x + y, (x*y >= 1) & (x > 0))
    (Optimum(-oo, not attained), Optimum(2, attained at [{x: 1, y: 1}]))
    >>> minimize(1/(1 + x**2))
    Optimum(0, not attained)
    """
    return _optimize(f, constraints, variables, True)


def maximize(f: ExprLike, constraints: BooleanLike = True, variables: Optional[Sequence[Symbol]] = None) -> Optimum:
    """The global supremum of ``f`` over the points satisfying
    ``constraints`` (see :func:`minimize`).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.optimization import maximize
    >>> x, y = Symbol('x'), Symbol('y')
    >>> maximize(x*(1 - x)), maximize(x*y, (x >= 0) & (y >= 0) & (x + y <= 2))
    (Optimum(1/4, attained at [{x: 1/2}]), Optimum(1, attained at [{x: 1, y: 1}]))
    """
    return _optimize(f, constraints, variables, False)




def _parametric(f: ExprLike, constraints: BooleanLike, variables: Optional[Sequence[Symbol]], parameters: Sequence[Symbol],
                lowest: bool) -> list[tuple[Boolean, Expr]]:
    from sympy_extras.core.logic import false
    from sympy_extras.core.relational import Ge, Gt, Le, Lt
    from sympy_extras.matrices.linalg import normal
    from sympy_extras.polys.cad.cylindrical import cylindrical_cases, cylindrical_formula
    from sympy_extras.simplify.radsimp import fraction
    objective = to_expr(f)
    condition = to_boolean(constraints)
    names = list(parameters)
    if not names:
        raise ValueError('no parameter: use minimize or maximize')
    symbols = {s for s in objective.free_symbols if isinstance(s, Symbol)}
    symbols |= {s for s in condition.free_symbols if isinstance(s, Symbol)}
    chosen = sorted(symbols - set(names), key=default_sort_key) if variables is None else list(variables)
    if set(chosen) & set(names) or symbols - set(chosen) - set(names):
        raise ValueError('every symbol is a variable or a parameter, and none is both')
    numerator, denominator = fraction(normal(objective))
    if not isinstance(denominator, Rational):
        raise NotImplementedError('a parametric optimum of a rational function: the objective must be a polynomial')
    bound = _fresh({s.name for s in symbols} | {s.name for s in chosen} | {s.name for s in names})
    inner: list[tuple[str, Symbol]] = [('forall', v) for v in chosen]
    beyond: list[tuple[str, Symbol]] = [('forall', bound)] + [('exists', v) for v in chosen]
    within: Boolean = Ge(objective, bound) if lowest else Le(objective, bound)
    outside: Boolean = Lt(objective, bound) if lowest else Gt(objective, bound)
    if condition is not true:
        within = Implies(condition, within)
        outside = And(condition, outside)
    results: list[tuple[Boolean, Expr]] = []
    for case, bounds in cylindrical_cases(within, names, [bound], quantifiers=inner):
        if not isinstance(bounds, Interval):
            raise NotImplementedError('the set of the bounds %s is not an interval' % (bounds,))
        end = bounds.end if lowest else bounds.start
        other = bounds.start if lowest else bounds.end
        if isinstance(end, (Infinity, NegativeInfinity)) or not isinstance(other, (Infinity, NegativeInfinity)):
            # every number is a bound: the constraints have no solution for these parameters
            continue
        results.append((case, end))
    unbounded = cylindrical_formula(outside, names, beyond)
    if unbounded is not false:
        extreme: Expr = -oo if lowest else oo
        results.append((unbounded, extreme))
    return results


def minimize_parametric(f: ExprLike, constraints: BooleanLike = True, variables: Optional[Sequence[Symbol]] = None,
                        parameters: Sequence[Symbol] = ()) -> list[tuple[Boolean, Expr]]:
    """The global infimum of the polynomial ``f`` over the points
    satisfying ``constraints``, as a function of ``parameters``: pairs of
    a condition on the parameters and of the infimum under it (``-oo``
    where the function is unbounded below). The values of the parameters
    for which the constraints have no solution are in no pair. The
    conditions are cylindrical: the first parameter between numbers, the
    second between functions of the first.

    >>> from sympy_extras.core import Eq, Symbol
    >>> from sympy_extras.calculus.optimization import minimize_parametric
    >>> x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')
    >>> minimize_parametric(a*x**2 + x, parameters=[a])
    [(a > 0, -1/(4*a)), (a <= 0, -oo)]
    >>> minimize_parametric(x**2 + y**2, Eq(x + y, a), parameters=[a])
    [(True, a**2/2)]
    >>> minimize_parametric(x, Eq(x**2, a), parameters=[a])                   # no solution for a < 0
    [(a >= 0, -sqrt(a))]
    >>> minimize_parametric(a*x - x**2, (x >= 0) & (x <= 1), parameters=[a])
    [(a <= 1, a - 1), (a > 1, 0)]
    """
    return _parametric(f, constraints, variables, parameters, True)


def maximize_parametric(f: ExprLike, constraints: BooleanLike = True, variables: Optional[Sequence[Symbol]] = None,
                        parameters: Sequence[Symbol] = ()) -> list[tuple[Boolean, Expr]]:
    """The global supremum of ``f`` as a function of ``parameters`` (see
    :func:`minimize_parametric`).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.calculus.optimization import maximize_parametric
    >>> x, a = Symbol('x'), Symbol('a')
    >>> maximize_parametric(x*(a - x), parameters=[a]), maximize_parametric(a*x, (x >= 0) & (x <= 1), parameters=[a])
    ([(True, a**2/4)], [(a < 0, 0), (a >= 0, a)])
    """
    return _parametric(f, constraints, variables, parameters, False)
