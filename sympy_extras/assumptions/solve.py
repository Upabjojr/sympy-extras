"""Solving equations and inequalities under assumptions."""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.polys.rootoftools import CRootOf
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or,
    Not, true)
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import Reals, Integers, Naturals, Rationals, Complexes
from sympy.sets.sets import (Set, FiniteSet, Intersection, Complement,
    Union as SetUnion, EmptySet)
from sympy.solvers.solveset import solveset, nonlinsolve

from sympy_extras._typing import Truth, as_boolean, as_expr, as_set, free_symbols, sorted_symbols
from sympy_extras.polys.cad import solution_set
from sympy_extras.polys.cad.samplepoints import _root_poly

from .ask import Assumptions, _facts, _evaluate
from .facts import Facts, normalize, conjuncts, to_polynomial
from .refine import _Refiner

__all__ = ['solve']

#: an equation or inequality to solve: an expression (equated to zero), a
#: relation or a Boolean combination of relations
Statement = Union[Expr, Boolean, bool]


def _as_statement(item: object) -> Boolean:
    e = sympify(item)
    if isinstance(e, Expr):
        # a Symbol is also a Boolean in SymPy: an expression is equated to zero
        return Eq(e, 0)
    if isinstance(e, Boolean):
        return e
    raise TypeError("an equation or a Boolean is expected, got %s" % (e,))


def _domain_set(facts: Facts, symbols: Sequence[Symbol], domain: Optional[Set]) -> Set:
    """The set the solutions are searched in by SymPy's solvers."""
    if domain is not None:
        domain_ = as_set(domain)
        if isinstance(domain_, (Reals, Integers, Naturals, Rationals, Complexes)) or domain_.is_subset(S.Reals):
            return domain_
    if all(s in facts.real for s in symbols):
        return S.Reals
    return S.Complexes


def _solveset(formula: Boolean, x: Symbol, domain: Set) -> Set:
    """Solve a Boolean combination of relations in one variable with
    :func:`sympy.solveset`, atom by atom."""
    if isinstance(formula, BooleanTrue):
        return domain
    if isinstance(formula, BooleanFalse):
        return S.EmptySet
    if isinstance(formula, And):
        return Intersection(*[_solveset(as_boolean(a), x, domain) for a in formula.args])
    if isinstance(formula, Or):
        return SetUnion(*[_solveset(as_boolean(a), x, domain) for a in formula.args])
    if isinstance(formula, Not):
        return Complement(domain, _solveset(as_boolean(formula.args[0]), x, domain))
    if isinstance(formula, Contains):
        if formula.args[0] == x:
            return Intersection(domain, as_set(formula.args[1]))
        return ConditionSet(x, formula, domain)
    if isinstance(formula, Relational):
        if x not in formula.free_symbols:
            return ConditionSet(x, formula, domain)
        try:
            return as_set(solveset(formula, x, domain))
        except (NotImplementedError, ValueError, TypeError):
            return ConditionSet(x, formula, domain)
    return ConditionSet(x, formula, domain)


def _holds_at(formula: Boolean, values: dict[Basic, Basic], facts: Facts) -> Truth:
    """Whether the formula holds at the values, under the facts about the
    remaining symbols."""
    instance = normalize(as_boolean(formula.xreplace(values)))
    return _evaluate(instance, facts)


def _filter_finite(elements: Sequence[Basic], x: Symbol, condition: Boolean, facts: Facts) -> Set:
    """Keep the elements at which the condition holds, and put the
    undecided ones in a :class:`~sympy.sets.conditionset.ConditionSet`."""
    kept: list[Basic] = []
    undecided: list[Basic] = []
    for e in elements:
        value = _holds_at(condition, {x: e}, facts)
        if value is True:
            kept.append(e)
        elif value is None:
            undecided.append(e)
    result: Set = FiniteSet(*kept) if kept else S.EmptySet
    if undecided:
        result = SetUnion(result, ConditionSet(x, condition, FiniteSet(*undecided)))
    return result


def _restrict(result: Set, x: Symbol, condition: Boolean, facts: Facts) -> Set:
    """Restrict a solution set to the values satisfying the condition."""
    if condition is true:
        return result
    if isinstance(result, FiniteSet):
        return _filter_finite(list(result.args), x, condition, facts)
    if isinstance(result, (SetUnion, Intersection)):
        return as_set(result.func(*[_restrict(as_set(a), x, condition, facts) for a in result.args]))
    if isinstance(result, EmptySet):
        return result
    if free_symbols(condition) <= {x} and x in facts.real:
        polynomial = to_polynomial(condition, facts.real)
        if polynomial is not None:
            return Intersection(result, _radicals(solution_set(polynomial, x)))
    return ConditionSet(x, condition, result)


def _radicals(result: Set) -> Set:
    """Write the algebraic numbers of degree at most two in radicals."""
    replacements: dict[Basic, Basic] = {}
    for r in result.atoms(CRootOf):
        if _root_poly(r).degree() <= 2:
            replacements[r] = CRootOf(r.expr, int(as_expr(r.args[1])), radicals=True)
    return as_set(result.xreplace(replacements)) if replacements else result


def _univariate(formula: Boolean, x: Symbol, facts: Facts, domain: Optional[Set]) -> Set:
    dom = _domain_set(facts, [x], domain)
    # the membership in the domain is handled by the solvers themselves
    on_x = And(*[c for c in facts.conjuncts if x in c.free_symbols
                 and not (isinstance(c, Contains) and c.args[0] == x and c.args[1] == dom)])
    others = And(*[c for c in facts.conjuncts if x not in c.free_symbols])
    if others is not true:
        # the facts about the parameters are used when deciding the solutions
        facts = Facts([others]) if others.free_symbols else facts
    parameters = free_symbols(formula) - {x}
    polynomial_on_x = to_polynomial(as_boolean(on_x), facts.real) if free_symbols(on_x) <= {x} else None
    polynomial_formula = to_polynomial(formula, facts.real) if not parameters and x in facts.real else None
    if polynomial_on_x is not None and polynomial_formula is not None:
        real = _radicals(solution_set(And(polynomial_formula, polynomial_on_x), x))
        if isinstance(dom, Reals):
            return real
        return Intersection(real, dom)
    # the parameters carry their assumptions while SymPy solves; with
    # parameters the equations are solved over the complex numbers and the
    # membership in the domain is decided afterwards
    refiner = _Refiner(facts)
    forward: dict[Basic, Basic] = {}
    back_symbols: dict[Basic, Basic] = {}
    for parameter in sorted(parameters, key=lambda s: s.name):
        dummy = refiner.dummy_for(parameter)
        if dummy is not None:
            forward[parameter] = dummy
            back_symbols[dummy] = parameter
    abstracted = formula.xreplace(forward)
    condition = as_boolean(on_x)
    solving_domain = dom
    if parameters and not isinstance(dom, Complexes):
        solving_domain = S.Complexes
        implies_real = isinstance(dom, Reals) and any(isinstance(c, (Gt, Lt, Ge, Le)) for c in conjuncts(condition))
        if not implies_real:
            condition = And(Contains(x, dom), condition)
    result = _solveset(as_boolean(abstracted), x, solving_domain)
    result = as_set(result.xreplace(back_symbols))
    return _restrict(result, x, condition, facts)


def _multivariate(statements: Sequence[Boolean], symbols: Sequence[Symbol], facts: Facts,
                  domain: Optional[Set]) -> Set:
    equations: list[Expr] = []
    conditions: list[Boolean] = []
    for s in statements:
        for c in conjuncts(normalize(s)):
            if isinstance(c, Eq):
                equations.append(as_expr(c.lhs - c.rhs))
            else:
                conditions.append(c)
    conditions.extend(c for c in facts.conjuncts if free_symbols(c) & set(symbols))
    condition = And(*conditions)
    if not equations:
        raise NotImplementedError("systems of inequalities in several variables are not solved; "
                                  "use resolve for a description of the solution set")
    try:
        solutions = nonlinsolve(equations, list(symbols))
    except (NotImplementedError, ValueError, TypeError):
        raise NotImplementedError("the system cannot be solved")
    dom = _domain_set(facts, symbols, domain)
    if not isinstance(solutions, FiniteSet):
        return ConditionSet(tuple(symbols), condition, solutions) if condition is not true else solutions
    kept: list[Basic] = []
    undecided: list[Basic] = []
    for point in solutions.args:
        values: dict[Basic, Basic] = {s: v for s, v in zip(symbols, point.args)}
        membership = And(*[Contains(v, dom) for v in point.args]) if not isinstance(dom, Complexes) else true
        value = _holds_at(And(condition, membership), values, facts)
        if value is True:
            kept.append(point)
        elif value is None:
            undecided.append(point)
    result: Set = FiniteSet(*kept) if kept else S.EmptySet
    if undecided:
        result = SetUnion(result, ConditionSet(tuple(symbols), And(condition, membership), FiniteSet(*undecided)))
    return result


def solve(equations: Union[Statement, Sequence[Statement]],
          symbols: Union[None, Symbol, Sequence[Symbol]] = None,
          assumptions: Assumptions = None, domain: Optional[Set] = None) -> Set:
    """Solve equations and inequalities under assumptions, the counterpart
    of Mathematica's ``Solve[eqns, vars, dom]`` with ``Assumptions``.

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

    Returns
    =======

    A SymPy set. For one unknown with polynomial equations and
    inequalities over the reals the solution set is computed exactly by
    cylindrical algebraic decomposition, as a union of intervals and
    points with algebraic endpoints. Otherwise :func:`sympy.solveset` (or
    :func:`sympy.nonlinsolve` for systems) solves the equations and the
    solutions are kept when the assumptions hold at them, dropped when
    they fail and collected in a
    :class:`~sympy.sets.conditionset.ConditionSet` when this cannot be
    decided.

    Examples
    ========

    >>> from sympy import S, Eq, sqrt, sin, exp
    >>> from sympy.abc import x, y, a
    >>> from sympy_extras.assumptions import solve, element
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
    """
    items = [equations] if isinstance(equations, (Basic, bool)) else list(equations)
    statements = [_as_statement(item) for item in items]
    all_symbols = set().union(*[free_symbols(s) for s in statements]) if statements else set()
    if symbols is None:
        unknowns = sorted_symbols(all_symbols)
    elif isinstance(symbols, Symbol):
        unknowns = [symbols]
    else:
        unknowns = list(symbols)
    if not unknowns:
        raise ValueError("no unknown to solve for")
    try:
        facts = _facts(assumptions, domain, all_symbols | set(unknowns))
    except ValueError:
        return S.EmptySet
    if len(unknowns) == 1:
        x = unknowns[0]
        formula = normalize(And(*statements))
        return _univariate(formula, x, facts, domain)
    return _multivariate(statements, unknowns, facts, domain)
