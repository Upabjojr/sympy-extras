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
evaluate (with ``Range``, ``ImageSet``, ...); the cylindrical algebraic
decomposition (:mod:`sympy_extras.polys.cad`) and the radical forms of
its roots (:func:`sympy_extras.polys.roots.in_radicals`), the triangular
decompositions (:func:`sympy_extras.polys.regularchains.triangularize`,
:func:`sympy_extras.solvers.parametric.parametric_cases`), all written on
SymPy's polynomials; the transcendental equations
(:func:`sympy_extras.solvers.transcendental.solve_transcendental`) and
the isolation of real roots (:mod:`sympy_extras.solvers.isolation`), on
SymPy's ``solveset``; the linear Diophantine systems
(:mod:`sympy_extras.solvers.integers`) and
:func:`sympy_extras._numeric.reliable_value`.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import FiniteSet, Set

from sympy_extras._typing import Truth, as_set
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import PointSet as CorePointSet
from sympy_extras.sets.sets import Set as CoreSet
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from . import solve as native
from .ask_sympy import Assumptions, _sympy_symbol, core_assumptions, core_boolean, core_domain, sympy_boolean
from .intervals_sympy import core_expr, core_symbol, sympy_expr, sympy_set

__all__ = ['solve', 'Statement', 'evaluated_set_of_core', 'solveset_of_core', 'reliable_value_of_core',
    'solution_set_of_core', 'in_radicals_of_core', 'solve_transcendental_of_core', 'periodicity_of_core',
    'isolate_real_roots_of_core', 'point_between_of_core', 'compare_points_of_core', 'nonlinsolve_of_core',
    'cylindrical_set_of_core', 'cylindrical_formula_of_core', 'cylindrical_cases_of_core',
    'triangularize_of_core', 'linear_system_of_core', 'linear_diophantine_system_of_core',
    'minimal_nonnegative_solutions_of_core', 'parametric_cases_of_core']

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


def reliable_value_of_core(e: CoreExpr, digits: int) -> Optional[CoreExpr]:
    """:func:`sympy_extras._numeric.reliable_value` of a constant."""
    from sympy_extras._numeric import reliable_value
    value = reliable_value(sympy_expr(e), digits)
    return None if value is None else core_expr(value)


def solution_set_of_core(formula: CoreBoolean, x: CoreSymbol) -> CoreSet:
    """The real solutions of a polynomial formula by the CAD
    (:func:`sympy_extras.polys.cad.solution_set`)."""
    from sympy_extras.polys.cad.qe import solution_set
    return _core_set(solution_set(sympy_boolean(formula), _sympy_symbol(x)))


def in_radicals_of_core(s: CoreSet) -> CoreSet:
    """:func:`sympy_extras.polys.roots.in_radicals`: the roots of
    polynomials written in radicals when that is reasonable."""
    from sympy_extras.polys.roots import in_radicals
    return _core_set(in_radicals(sympy_set(s)))


def solve_transcendental_of_core(formula: CoreBoolean, x: CoreSymbol,
                                 solver: Callable[[CoreBoolean, CoreSymbol], CoreSet],
                                 decide: Callable[[CoreBoolean], Truth]) -> Optional[CoreSet]:
    """:func:`sympy_extras.solvers.transcendental.solve_transcendental`,
    with the polynomial formulas solved and the conditions decided by the
    native functions given."""
    from sympy_extras.solvers.transcendental import solve_transcendental

    def sympy_solver(f: Boolean, v: Symbol) -> Optional[Set]:
        return sympy_set(solver(core_boolean(f), core_symbol(v)))

    def sympy_decide(c: Boolean) -> Truth:
        return decide(core_boolean(c))

    found = solve_transcendental(sympy_boolean(formula), _sympy_symbol(x), solver=sympy_solver, decide=sympy_decide)
    return None if found is None else _core_set(found)


def periodicity_of_core(f: CoreExpr, x: CoreSymbol) -> Optional[CoreExpr]:
    """SymPy's ``periodicity`` (``None`` when ``f`` is not periodic)."""
    from sympy.calculus.util import periodicity
    value = periodicity(sympy_expr(f), _sympy_symbol(x))
    return core_expr(value) if isinstance(value, Expr) else None


def isolate_real_roots_of_core(f: CoreExpr, x: CoreSymbol, region: CoreSet) -> Optional[list[CoreExpr]]:
    """:func:`sympy_extras.solvers.isolation.isolate_real_roots`."""
    from sympy_extras.solvers.isolation import isolate_real_roots
    roots = isolate_real_roots(sympy_expr(f), _sympy_symbol(x), sympy_set(region))
    return None if roots is None else [core_expr(r) for r in roots]


def point_between_of_core(p: CoreExpr, q: CoreExpr) -> Optional[CoreExpr]:
    """:func:`sympy_extras.solvers.isolation.point_between`."""
    from sympy_extras.solvers.isolation import point_between
    found = point_between(sympy_expr(p), sympy_expr(q))
    return None if found is None else core_expr(found)


def compare_points_of_core(p: CoreExpr, q: CoreExpr) -> Optional[int]:
    """:func:`sympy_extras.solvers.isolation.compare_points`."""
    from sympy_extras.solvers.isolation import compare_points
    return compare_points(sympy_expr(p), sympy_expr(q))


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


def cylindrical_set_of_core(formula: CoreBoolean, symbols: list[CoreSymbol]) -> CoreSet:
    """:func:`sympy_extras.polys.cad.cylindrical_set`."""
    from sympy_extras.polys.cad.cylindrical import cylindrical_set
    return _core_set(cylindrical_set(sympy_boolean(formula), _core_symbols(symbols)))


def cylindrical_formula_of_core(formula: CoreBoolean, gens: list[CoreSymbol]) -> CoreBoolean:
    """:func:`sympy_extras.polys.cad.cylindrical_formula`."""
    from sympy_extras.polys.cad.cylindrical import cylindrical_formula
    return core_boolean(cylindrical_formula(sympy_boolean(formula), _core_symbols(gens)))


def cylindrical_cases_of_core(formula: CoreBoolean, parameters: list[CoreSymbol],
                              unknowns: list[CoreSymbol]) -> list[tuple[CoreBoolean, CoreSet]]:
    """:func:`sympy_extras.polys.cad.cylindrical_cases`."""
    from sympy_extras.polys.cad.cylindrical import cylindrical_cases
    found = cylindrical_cases(sympy_boolean(formula), _core_symbols(parameters), _core_symbols(unknowns))
    return [(core_boolean(condition), _core_set(solutions)) for condition, solutions in found]


def triangularize_of_core(equations: list[CoreExpr], symbols: list[CoreSymbol]
                          ) -> Optional[list[native.TriangularChain]]:
    """The regular chains of :func:`sympy_extras.polys.regularchains.triangularize`
    (``None`` when it fails or does not end in time)."""
    from sympy_extras.polys.regularchains import RegularChain, triangularize
    sympy_equations, sympy_symbols = [sympy_expr(e) for e in equations], _core_symbols(symbols)
    try:
        chains = attempt(lambda: triangularize(sympy_equations, *sympy_symbols), settings.timeout)
    except ZeroDivisionError:
        chains = None
    if chains is None:
        return None

    def points(chain: RegularChain) -> Callable[[bool], Optional[native.ChainPoints]]:
        def found(real: bool) -> Optional[native.ChainPoints]:
            return [{core_symbol(s): core_expr(v) for s, v in solution.items()} for solution in chain.solutions(real)]
        return found

    result: list[native.TriangularChain] = []
    for chain in chains:
        polys = [core_expr(p) for p in chain.polys] if chain.dimension else []
        variables = [core_symbol(v) for v in chain.main_variables] if chain.dimension else []
        result.append(native.TriangularChain(chain.dimension, polys, variables, points(chain)))
    return result


def linear_system_of_core(equations: Sequence[CoreExpr], symbols: Sequence[CoreSymbol]
                          ) -> Optional[tuple[list[list[int]], list[int]]]:
    """``A``, ``b`` with the equations ``A x = b`` with integer
    coefficients (``None`` when the system is not of this kind)."""
    from sympy_extras.solvers.integers import _linear_system
    return _linear_system([sympy_expr(e) for e in equations], _core_symbols(list(symbols)))


def linear_diophantine_system_of_core(equations: Sequence[CoreExpr], symbols: Sequence[CoreSymbol]
                                      ) -> Optional[tuple[list[CoreExpr], list[CoreSymbol]]]:
    """:func:`sympy_extras.solvers.integers.linear_diophantine_system`."""
    from sympy_extras.solvers.integers import linear_diophantine_system
    found = linear_diophantine_system([sympy_expr(e) for e in equations], _core_symbols(list(symbols)))
    if found is None:
        return None
    values, parameters = found
    return [core_expr(v) for v in values], [core_symbol(p) for p in parameters]


def minimal_nonnegative_solutions_of_core(A: list[list[int]], b: list[int]) -> tuple[list[list[int]], list[list[int]]]:
    """:func:`sympy_extras.solvers.integers.minimal_nonnegative_solutions`."""
    from sympy_extras.solvers.integers import minimal_nonnegative_solutions
    return minimal_nonnegative_solutions(A, b)


def parametric_cases_of_core(equations: list[CoreExpr], unknowns: list[CoreSymbol], parameters: list[CoreSymbol],
                             inequations: list[CoreExpr]) -> list[native.ParametricCase]:
    """:func:`sympy_extras.solvers.parametric.parametric_cases`."""
    from sympy_extras.solvers.parametric import parametric_cases
    found = parametric_cases([sympy_expr(e) for e in equations], _core_symbols(unknowns),
                             _core_symbols(parameters), [sympy_expr(e) for e in inequations])
    return [native.ParametricCase(core_boolean(case.condition),
                                  None if case.solutions is None else _core_set(case.solutions),
                                  [core_expr(e) for e in case.equations]) for case in found]


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
        (:func:`~sympy_extras.polys.cad.cylindrical_cases`). Without it
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
    (:func:`~sympy_extras.polys.cad.cylindrical_set`): the points with
    algebraic coordinates as a finite set, and the rest as a
    ``ConditionSet`` whose condition bounds the first unknown by numbers,
    the second one by functions of the first, and so on (explicit roots of
    polynomials of degree at most two, :class:`~sympy_extras.polys.cad.IndexedRoot`
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
