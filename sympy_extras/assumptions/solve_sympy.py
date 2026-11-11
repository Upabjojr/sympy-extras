"""The SymPy face of solving under assumptions
(:mod:`sympy_extras.assumptions.solve`), and the steps of ``solve``
which are not ported.

:func:`solve` takes and returns SymPy objects (the assumptions as
statements, SymPy's sets), converted by :mod:`sympy_extras.interop`
around the native function: the result is the native set rebuilt by
SymPy's constructors.

The steps of the native ``solve`` which stay written on SymPy are called
from it at run time (the functions ending in ``_of_core``, which take and
return objects of the core): SymPy's ``solveset`` for the equations which
are not rational and for the inequalities, ``nonlinsolve``,
``periodicity`` and the intersections of sets which the core does not
evaluate (with ``Range``, ``ImageSet``, ...).
The transcendental equations (:mod:`sympy_extras.solvers.transcendental`),
:func:`sympy_extras._numeric.reliable_value`, the cylindrical
algebraic decomposition (:mod:`sympy_extras.polys.cad`),
the radical forms of the roots (:mod:`sympy_extras.polys.roots`), the
isolation of real roots (:mod:`sympy_extras.solvers.isolation`) and the
linear Diophantine systems (:mod:`sympy_extras.solvers.integers`), the
triangular decompositions (:mod:`sympy_extras.polys.regularchains`, whose
exact points go through :mod:`sympy_extras.polys.regularchains_sympy`) and
the cases of the parameters (:mod:`sympy_extras.solvers.parametric`) are
native.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import FiniteSet, Set

from sympy_extras._typing import as_set
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import PointSet as CorePointSet
from sympy_extras.sets.sets import Set as CoreSet

from . import solve as native
from .ask_sympy import Assumptions, _sympy_symbol, core_assumptions, core_boolean, core_domain, sympy_boolean
from .intervals_sympy import core_expr, core_symbol, sympy_expr, sympy_set

__all__ = ['solve', 'Statement', 'evaluated_set_of_core', 'solveset_of_core',
    'periodicity_of_core', 'nonlinsolve_of_core']

#: an equation or inequality to solve: an expression (equated to zero), a
#: relation or a Boolean combination of relations
Statement = Union[Expr, Boolean, bool]


# ---------------------------------------------------------------------------
# the conversions at the boundary

def _core_set(s: Basic) -> CoreSet:
    """A SymPy set as a set of the core (a wrapper for the sets not
    ported)."""
    converted = from_sympy(as_set(s))
    if not isinstance(converted, CoreSet):
        raise TypeError("expected a set, got %s" % (s,))
    return converted


def _core_symbols(symbols: Sequence[CoreSymbol]) -> list[Symbol]:
    return [_sympy_symbol(s) for s in symbols]


# ---------------------------------------------------------------------------
# the steps of the native solve which stay on SymPy

def evaluated_set_of_core(s: CoreSet) -> CoreSet:
    """The set rebuilt by SymPy's constructors, which evaluate the
    intersections the core leaves alone (``Interval`` and ``Integers``,
    ``ImageSet`` and ``Reals``, ...)."""
    return _core_set(to_sympy(s))


def solveset_of_core(relation: CoreBoolean, x: CoreSymbol, domain: CoreSet) -> CoreSet:
    """:func:`sympy.solveset` of one relation (its exceptions go on)."""
    from sympy.solvers.solveset import solveset
    return _core_set(solveset(sympy_boolean(relation), _sympy_symbol(x), sympy_set(domain)))




def periodicity_of_core(f: CoreExpr, x: CoreSymbol) -> Optional[CoreExpr]:
    """SymPy's ``periodicity`` (``None`` when ``f`` is not periodic)."""
    from sympy.calculus.util import periodicity
    value = periodicity(sympy_expr(f), _sympy_symbol(x))
    return core_expr(value) if isinstance(value, Expr) else None


def nonlinsolve_of_core(equations: list[CoreExpr], symbols: list[CoreSymbol]) -> Optional[CoreSet]:
    """SymPy's ``nonlinsolve`` (its exceptions go on); ``None`` for a
    finite set of points with sets among their coordinates (which no
    substitution checks)."""
    from sympy.solvers.solveset import nonlinsolve
    found = as_set(nonlinsolve([sympy_expr(e) for e in equations], _core_symbols(symbols)))
    if isinstance(found, FiniteSet) and any(not isinstance(value, Expr) for point in found.args
                                            for value in point.args):
        # nonlinsolve puts sets among the coordinates: (-sqrt(...), -sqrt(3*z/2
        # - 5/4), Interval.open(-oo, 5/6)) for two equations in x, y, z
        return None
    return _core_set(found)


# ---------------------------------------------------------------------------
# the face

def _as_statement(item: object) -> native.Statement:
    e = sympify(item)
    if isinstance(e, Expr):
        # a Symbol is also a Boolean in SymPy: an expression is equated to zero
        return core_expr(e)
    if isinstance(e, Boolean):
        return core_boolean(e)
    raise TypeError("an equation or a Boolean is expected, got %s" % (e,))


def _clearly_not_real(e: Basic) -> bool:
    """Whether the imaginary part of a number is clearly nonzero, by
    numerical evaluation (:func:`sympy_extras.assumptions.solve._clearly_not_real`)."""
    return native._clearly_not_real(from_sympy(e))


def _refuted(solutions: FiniteSet, equations: Sequence[Expr], symbols: Sequence[Symbol]) -> bool:
    """Whether a point of ``solutions`` clearly does not satisfy the
    equations (:func:`sympy_extras.assumptions.solve._refuted`); points with
    sets among their coordinates are refuted."""
    if any(not isinstance(value, Expr) for point in solutions.args for value in point.args):
        return True
    points = from_sympy(solutions)
    if not isinstance(points, CorePointSet):
        raise TypeError("expected a finite set of points, got %s" % (solutions,))
    return native._refuted(points, [core_expr(e) for e in equations], [core_symbol(s) for s in symbols])


def solve(equations: Union[Statement, Sequence[Statement]],
          symbols: Union[None, Symbol, Sequence[Symbol]] = None,
          assumptions: Assumptions = None, domain: Optional[Set] = None, cases: bool = False) -> Set:
    """Solve equations and inequalities under assumptions
    (:func:`sympy_extras.assumptions.solve.solve` on SymPy objects), the
    counterpart of Mathematica's ``Solve[eqns, vars, dom]`` with
    ``Assumptions``.

    Parameters
    ==========

    equations : expression, relation, Boolean combination, or a list of them
        An expression is equated to zero. Inequalities and Boolean
        combinations are accepted for one unknown.
    symbols : Symbol or list of Symbols, optional
        The unknowns; by default all the free symbols.
    assumptions : Boolean or list of Booleans, optional
        Assumptions on the unknowns and on the parameters, written as
        statements (see :mod:`sympy_extras.assumptions`); the global
        assumptions are used as well.
    domain : Set, optional
        The set the unknowns belong to (``S.Reals``, ``S.Integers``, ...).
        Without it the unknowns are real when the assumptions say so and
        complex otherwise.
    cases : bool
        Whether the values of the parameters are discussed, as
        Mathematica's ``Reduce`` does, for a system of polynomial
        relations with rational coefficients: the result is a union of
        sets ``ConditionSet(unknowns, condition on the parameters,
        solutions)``, one for each case (the values of the parameters with
        no solution are in none). Over the complex numbers, for equations
        and inequations, the cases are those of a triangular decomposition
        (:func:`~sympy_extras.solvers.parametric.parametric_cases`); over
        the reals (``domain=S.Reals``, the parameters are then real
        numbers too), with inequalities as well, those of a cylindrical
        decomposition with the parameters first
        (:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_cases`). Without it
        the solutions are the generic ones: ``{b/a}`` for ``a*x = b``.

    Returns
    =======

    A SymPy set. For one unknown with polynomial equations and
    inequalities over the reals the solution set is computed exactly by
    cylindrical algebraic decomposition, as a union of intervals and
    points with algebraic endpoints. Otherwise the equations are solved
    (the rational ones over the complex numbers by the native port of
    :func:`sympy.solveset`, the others by :func:`sympy.solveset` itself,
    or :func:`sympy.nonlinsolve` for systems) and the solutions are kept
    when the assumptions hold at them, dropped when they fail and
    collected in a :class:`~sympy.sets.conditionset.ConditionSet` when
    this cannot be decided. The points of ``nonlinsolve`` are substituted
    back in the equations: a polynomial system with rational coefficients
    which they do not satisfy (``nonlinsolve`` leaves free an unknown it
    cannot solve for) is solved by a triangular decomposition into regular
    chains (:mod:`sympy_extras.polys.regularchains`), and another one is
    returned as a ``ConditionSet`` of its equations.

    Over the reals (``domain=S.Reals``, or real unknowns), a system of
    polynomial inequalities in several unknowns, with or without
    equations, and a system of equations with infinitely many solutions,
    are described cylindrically
    (:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_set`): the points with
    algebraic coordinates as a finite set, and the rest as a
    ``ConditionSet`` whose condition bounds the first unknown by numbers,
    the second one by functions of the first, and so on (explicit roots of
    polynomials of degree at most two, :class:`~sympy_extras.polys.cad.indexedroot_sympy.IndexedRoot`
    beyond). The parameters of the system are bounded before the unknowns,
    within what the assumptions say of them.

    Examples
    ========

    >>> from sympy import S, Eq, sqrt, sin, exp
    >>> from sympy.abc import x, y, a
    >>> from sympy_extras.assumptions.solve_sympy import solve
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> solve(x**2 - 2, x, x > 0)
    {sqrt(2)}
    >>> solve(x**2 - 2, x)
    {-sqrt(2), sqrt(2)}
    >>> solve((x**2 - 2 > 0) & (x < 3), x, domain=S.Reals)
    Union(Interval.open(-oo, -sqrt(2)), Interval.open(sqrt(2), 3))
    >>> solve(x**2 < 10, x, domain=S.Integers)
    Range(-3, 4, 1)
    >>> solve(x**2 - a, x, (x > 0) & (a > 0))
    {sqrt(a)}
    >>> solve(x**2 - a, x, x > 0)
    ConditionSet(x, x > 0, {-sqrt(a), sqrt(a)})
    >>> solve(exp(x) - 2, x, x > 1)
    EmptySet
    >>> solve([x**2 + y**2 - 1, x - y], [x, y], x > 0)
    {(sqrt(2)/2, sqrt(2)/2)}
    >>> solve([x**2 + y**2 <= 1, x + y >= 1], [x, y], domain=S.Reals)
    ConditionSet((x, y), (x >= 0) & (x <= 1) & (y >= 1 - x) & (y <= sqrt(1 - x**2)), ProductSet(Reals, Reals))
    >>> solve([Eq(x**2 + y**2, a), x > y], [x, y], a > 0, domain=S.Reals)
    ConditionSet((x, y), ..., ProductSet(Reals, Reals))
    >>> solve(a*x - 1, x, cases=True)
    ConditionSet(x, Ne(a, 0), {1/a})
    >>> solve(Eq(x**2, a), x, domain=S.Reals, cases=True)
    ConditionSet(x, a >= 0, {-sqrt(a), sqrt(a)})
    >>> solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=S.Integers)
    {(4, 2)}
    >>> solve(Eq(3*x + 5*y, 22), [x, y], domain=S.Integers)
    ImageSet(Lambda(t0, (44 - 5*t0, 3*t0 - 22)), Integers)
    """
    items = [equations] if isinstance(equations, (Basic, bool)) else list(equations)
    statements = [_as_statement(item) for item in items]
    unknowns: Union[None, CoreSymbol, list[CoreSymbol]]
    if symbols is None:
        unknowns = None
    elif isinstance(symbols, Symbol):
        unknowns = core_symbol(symbols)
    else:
        unknowns = [core_symbol(s) for s in symbols]
    found = native.solve(statements, unknowns, core_assumptions(assumptions), core_domain(domain), cases)
    return as_set(to_sympy(found))
