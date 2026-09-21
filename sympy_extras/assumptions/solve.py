"""Solving equations and inequalities under assumptions."""
from __future__ import annotations

import random
from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.numbers import Rational, Integer
from sympy.core.mod import Mod
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.core.containers import Tuple
from sympy.core.function import Lambda
from sympy.core.evalf import N
from sympy.functions.elementary.complexes import im
from sympy.core.sympify import sympify
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, false,
    Not, true)
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import Reals, Integers, Naturals, Rationals, Complexes, ImageSet
from sympy.sets.sets import (Set, FiniteSet, Intersection, Complement, Interval,
    Union as SetUnion, EmptySet, ProductSet, imageset)
from sympy.solvers.solveset import solveset, nonlinsolve

from sympy_extras._typing import Truth, as_boolean, as_expr, as_set, free_symbols, sorted_symbols
from sympy_extras.polys.cad import cylindrical_cases, cylindrical_formula, cylindrical_set, solution_set
from sympy_extras.polys.roots import in_radicals
from sympy_extras.solvers.transcendental import solve_transcendental
from sympy.polys.numberfields.minpoly import minimal_polynomial
from sympy.polys.polytools import Poly, cancel
from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from .ask import Assumptions, _facts, _evaluate
from .facts import Facts, normalize, conjuncts, to_polynomial
from .refine import _Refiner
from .analysis import domain_of, certified_sign, _pieces

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


def _clearly_not_real(e: Basic) -> bool:
    """Whether the imaginary part of a number is clearly nonzero, by
    numerical evaluation (when the settings allow it)."""
    if not settings.numerical_checks:
        return False
    digits = settings.precision
    # a value whose branch a rounding error chooses is not "clearly"
    # anything (the bug: log(-2*sqrt(2) - 2*sqrt(z)) - I*pi with z a sum
    # which is exactly zero, a real number, was found not real)
    value = reliable_value(e, digits) if isinstance(e, Expr) else None
    if value is None:
        return False
    imaginary = N(im(value), digits)
    return isinstance(imaginary, Expr) and imaginary.is_number and bool(abs(imaginary) > Rational(1, 10)**(digits*2//3))


def _same_number(first: Basic, second: Basic) -> bool:
    """Whether two constant expressions are the same number: structurally,
    by simplification, or -- for algebraic numbers written in different
    radicals -- by a high precision evaluation confirmed by a common
    minimal polynomial."""
    if first == second:
        return True
    if not (isinstance(first, Expr) and isinstance(second, Expr)):
        return False
    if first.free_symbols or second.free_symbols:
        return False
    difference = attempt(lambda: as_expr(simplify(first - second)), settings.timeout)
    if difference is not None and difference == 0:
        return True
    # two conjugates have the same minimal polynomial, and are told apart
    # by their values only: not by values whose branch a rounding error
    # chooses (see reliable_form)
    apart, size = reliable_value(as_expr(first - second), 60), reliable_value(first, 60)
    if apart is None or size is None:
        return False
    try:
        gap, scale = abs(complex(apart)), max(1.0, abs(complex(size)))
    except (TypeError, ValueError, OverflowError):
        return False
    if gap > 1e-50*scale:
        return False
    if not (first.is_algebraic and second.is_algebraic):
        return False
    t = Dummy('t')
    polynomials = attempt(lambda: (minimal_polynomial(first, t), minimal_polynomial(second, t)),
                          settings.timeout)
    return polynomials is not None and polynomials[0] == polynomials[1]


def _filter_finite(elements: Sequence[Basic], x: Symbol, condition: Boolean, facts: Facts,
                   formula: Boolean = true, real: bool = False) -> Set:
    """Keep the elements at which the condition holds, and put the
    undecided ones in a :class:`~sympy.sets.conditionset.ConditionSet`.
    Elements at which the solved formula itself provably fails (SymPy's
    solvers return extraneous roots of radical equations), and elements
    which are clearly not real when ``real`` solutions are wanted, are
    dropped."""
    kept: list[Basic] = []
    undecided: list[Basic] = []
    for e in elements:
        if real and _clearly_not_real(e):
            continue
        if formula is not true:
            at_point = normalize(as_boolean(formula.xreplace({x: e})))
            if _evaluate(at_point, facts) is False or _numeric_root(at_point) is False:
                continue
        value = _holds_at(condition, {x: e}, facts)
        if value is True:
            # the same number in two radical forms is one solution, and a
            # FiniteSet only merges structurally equal ones
            # (sympy-extras#50)
            if not any(_same_number(e, k) for k in kept):
                kept.append(e)
        elif value is None:
            undecided.append(e)
    result: Set = FiniteSet(*kept) if kept else S.EmptySet
    if undecided:
        result = SetUnion(result, ConditionSet(x, condition, FiniteSet(*undecided)))
    return result


def _numeric_root(instance: Boolean) -> Truth:
    """Whether the equations of a formula without free symbols hold, by
    evaluation with 30 digits: ``False`` when a residual is clearly not
    zero, ``True`` when all vanish, ``None`` when this cannot be told."""
    if not settings.numerical_checks:
        return None
    equations = [c for c in conjuncts(instance) if isinstance(c, Eq)]
    if not equations or any(free_symbols(c) for c in equations):
        return None
    digits = settings.precision
    for c in equations:
        residual = reliable_value(as_expr(c.lhs - c.rhs), digits)
        if residual is None:
            return None
        if abs(residual) > Rational(1, 10)**(digits*2//3):
            return False
    return True


def _restrict(result: Set, x: Symbol, condition: Boolean, facts: Facts, formula: Boolean = true,
              real: bool = False) -> Set:
    """Restrict a solution set to the values satisfying the condition (and
    check the finite candidates against the solved formula)."""
    if isinstance(result, FiniteSet):
        return _filter_finite(list(result.args), x, condition, facts, formula, real)
    if isinstance(result, (SetUnion, Intersection)):
        return as_set(result.func(*[_restrict(as_set(a), x, condition, facts, formula, real) for a in result.args]))
    if condition is true:
        return result
    if isinstance(result, EmptySet):
        return result
    if free_symbols(condition) <= {x} and x in facts.real:
        polynomial = to_polynomial(condition, facts.real)
        if polynomial is not None:
            return Intersection(result, _radicals(solution_set(polynomial, x)))
    return ConditionSet(x, condition, result)


def _radicals(result: Set) -> Set:
    """Write the algebraic numbers in radicals when that is reasonable
    (see :func:`sympy_extras.polys.roots.in_radicals`)."""
    return in_radicals(result)


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
    if isinstance(dom, Reals) and polynomial_on_x is not None:
        # transcendental formulas reduced to polynomial ones through
        # their kernels (exp, log, sin and cos, roots); the parameters
        # carry their assumptions into the polynomial solving and the
        # inversion of the kernels
        facts_ = facts
        transcendental = attempt(
            lambda: solve_transcendental(formula, x,
                                         solver=lambda f, v: _univariate(f, v, facts_, S.Reals),
                                         decide=lambda c: _evaluate(normalize(c), facts_)),
            settings.timeout)
        if transcendental is not None:
            if polynomial_on_x is not true:
                transcendental = Intersection(transcendental, _radicals(solution_set(polynomial_on_x, x)))
            return _restrict(as_set(transcendental), x, true, facts, formula, real=True)
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
        ordered_condition = any(isinstance(c, (Gt, Lt, Ge, Le)) for c in conjuncts(condition))
        ordered_formula = any(isinstance(a, (Gt, Lt, Ge, Le)) for a in formula.atoms(Relational))
        if isinstance(dom, Reals) and ordered_formula:
            # an inequality only makes sense between real numbers
            solving_domain = S.Reals
        else:
            solving_domain = S.Complexes
            if not (isinstance(dom, Reals) and ordered_condition):
                condition = And(Contains(x, dom), condition)
    if not parameters and dom.is_subset(S.Reals) is True:
        # a periodic inequality: SymPy answers over one period only
        periodic = attempt(lambda: _periodic(formula, x, facts, dom), settings.timeout)
        if periodic is not None:
            # the conditions on the unknown which are not a region of the
            # real line (an integrality) are applied as for any answer
            return _restrict(periodic, x, condition, facts, formula, real=True)
    result = _solveset(as_boolean(abstracted), x, solving_domain)
    result = as_set(result.xreplace(back_symbols))
    restricted = _restrict(result, x, condition, facts, formula, real=dom.is_subset(S.Reals) is True)
    if not parameters and restricted.has(ConditionSet) and dom.is_subset(S.Reals) is True:
        # no closed form: the real roots are isolated instead
        isolated = attempt(lambda: _isolated(formula, x, facts, dom), settings.timeout)
        if isolated is not None:
            return isolated
    return restricted


def _period(formula: Boolean, x: Symbol) -> Optional[Expr]:
    """The common period of the relations of a formula in one unknown, or
    ``None`` when one of them is not periodic."""
    from sympy.calculus.util import periodicity
    period: Optional[Expr] = None
    for atom in formula.atoms(Relational):
        difference = as_expr(atom.lhs - atom.rhs)
        if x not in free_symbols(difference):
            continue
        value = periodicity(difference, x)
        if not isinstance(value, Expr) or value.is_positive is not True or value.is_finite is not True:
            return None
        if free_symbols(value):
            return None
        period = value if period is None else _common_period(period, value)
        if period is None:
            return None
    return period


def _common_period(first: Expr, second: Expr) -> Optional[Expr]:
    """The least common multiple of two periods (``None`` when their
    ratio is not rational, so that no common period exists)."""
    ratio = as_expr(simplify(first/second))
    if not isinstance(ratio, Rational):
        return None
    return as_expr(first*ratio.q)


def _periodic(formula: Boolean, x: Symbol, facts: Facts, dom: Set) -> Optional[Set]:
    """A periodic inequality solved over one period and tiled over the
    region (``None`` when it is not periodic).

    SymPy's ``solveset`` answers a periodic inequality over a single
    period only -- ``solveset(sin(x) > 0, x, S.Reals)`` is
    ``Interval.open(0, pi)``, which leaves out every other period -- so
    the solutions over one period are tiled here instead: explicitly when
    the region meets a few periods, and through the residue ``Mod(x, T)``
    when it meets infinitely many.  Equations are left alone: for them
    ``solveset`` returns the periodic family as an ``ImageSet``.
    """
    if free_symbols(formula) != {x} or not formula.atoms(Gt, Lt, Ge, Le):
        return None
    if formula.atoms(Eq, Ne):
        return None
    period = _period(formula, x)
    if period is None:
        return None
    window = Interval.Ropen(S.Zero, period)
    base = _base_over_period(formula, x, period)
    if base is None:
        return None
    # the region the solutions are looked for in; when the assumptions do
    # not describe a region of the real line (an integrality among them)
    # the relations alone describe one, and the rest is applied afterwards
    described = domain_of(x, facts.with_reals([x]))
    if described is None:
        relations = [c for c in facts.conjuncts
                     if isinstance(c, Relational) and free_symbols(c) <= {x}]
        described = domain_of(x, Facts(relations, S.Reals, [x]).with_reals([x]))
    region = as_set(dom if described is None else Intersection(described, dom))
    if isinstance(base, EmptySet) or isinstance(region, EmptySet):
        return S.EmptySet
    if base == window:
        return region
    tiles = _tiles(_pieces(region), period)
    if tiles is None:
        return ConditionSet(x, Contains(Mod(x, period), base), region)
    shift = Dummy('shift')
    copies = [as_set(imageset(Lambda(shift, shift + k*period), base)) for k in tiles]
    return as_set(Intersection(SetUnion(*copies), region))


#: the largest number of periods tiled explicitly (beyond it the solutions
#: are described by their residue)
_TILES = 64


def _tiles(pieces: Optional[Sequence[Interval]], period: Expr) -> Optional[range]:
    """The periods meeting the region, or ``None`` when there are too
    many of them (or infinitely many)."""
    if not pieces:
        return None
    low = as_expr(min((as_expr(piece.start) for piece in pieces), key=_ordering))
    high = as_expr(max((as_expr(piece.end) for piece in pieces), key=_ordering))
    if low.is_finite is not True or high.is_finite is not True:
        return None
    first, last = floor(low/period), floor(high/period)
    if not (isinstance(first, Integer) and isinstance(last, Integer)):
        return None
    if last - first + 1 > _TILES:
        return None
    return range(int(first), int(last) + 1)


def _ordering(point: Expr) -> float:
    """A key ordering the endpoints of the pieces of a region."""
    value = N(point, 20)
    return float(value) if isinstance(value, Expr) and value.is_real and value.is_finite \
        else (float('-inf') if point is S.NegativeInfinity else float('inf'))


def _base_over_period(formula: Boolean, x: Symbol, period: Expr) -> Optional[Set]:
    """The solutions over one period ``[0, T)``, isolated from the sign of
    the relation between its roots when that works (the answer does not
    then depend on SymPy's solving of the inequality), and left to
    ``solveset`` over the bounded window otherwise."""
    window = Interval.Ropen(S.Zero, period)
    facts = Facts([], S.Reals, [x])
    isolated = attempt(lambda: _isolated(formula, x, facts, Interval(S.Zero, period)), settings.timeout)
    if isolated is not None:
        return as_set(Intersection(isolated, window))
    solved = _solveset(formula, x, window)
    if solved.has(ConditionSet):
        return None
    return solved


def _isolated(formula: Boolean, x: Symbol, facts: Facts, dom: Set) -> Optional[Set]:
    """An equation or inequality in one real unknown without parameters
    solved through the isolated real roots of the difference of its sides
    (:mod:`sympy_extras.solvers.isolation`): exact roots where SymPy finds
    them, :class:`~sympy_extras.solvers.isolation.TranscendentalRoot`
    objects otherwise; ``None`` when the roots cannot be isolated."""
    # imported here: the isolation module uses the analysis of this package
    from sympy_extras.solvers.isolation import isolate_real_roots, point_between, compare_points
    if not isinstance(formula, Relational) or free_symbols(formula) != {x}:
        return None
    # the region the solutions are looked for in; when the assumptions do
    # not describe a region of the real line (an integrality among them)
    # the relations alone describe one, and the rest is applied afterwards
    described = domain_of(x, facts.with_reals([x]))
    if described is None:
        relations = [c for c in facts.conjuncts
                     if isinstance(c, Relational) and free_symbols(c) <= {x}]
        described = domain_of(x, Facts(relations, S.Reals, [x]).with_reals([x]))
    region = as_set(dom if described is None else Intersection(described, dom))
    pieces = _pieces(region)
    if pieces is None:
        return None
    f = as_expr(formula.lhs - formula.rhs)
    roots = isolate_real_roots(f, x, region)
    if roots is None:
        return None
    if isinstance(formula, Eq):
        return FiniteSet(*roots) if roots else S.EmptySet
    if isinstance(formula, Ne):
        return as_set(Complement(region, FiniteSet(*roots))) if roots else region
    strict = isinstance(formula, (Gt, Lt))
    wanted = 1 if isinstance(formula, (Gt, Ge)) else -1
    parts: list[Set] = []
    for piece in pieces:
        a, b = as_expr(piece.start), as_expr(piece.end)
        inner = [r for r in roots if compare_points(r, a) == 1 and compare_points(b, r) == 1]
        points: list[Expr] = [a] + inner + [b]
        for p, q in zip(points, points[1:]):
            sample = point_between(p, q)
            if sample is None:
                return None
            s = certified_sign(as_expr(f.subs(x, sample)))
            if s is None:
                return None
            if s == wanted:
                parts.append(Interval.open(p, q))
        if not strict:
            parts.extend(FiniteSet(r) for r in inner)
        for closed, point in ((not piece.left_open, a), (not piece.right_open, b)):
            if closed and point.is_finite:
                s = certified_sign(as_expr(f.subs(x, point)))
                if s == wanted or (s == 0 and not strict):
                    parts.append(FiniteSet(point))
    return as_set(SetUnion(*parts)) if parts else S.EmptySet


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
    dom = _domain_set(facts, symbols, domain)
    real = dom.is_subset(S.Reals) is True
    if not equations:
        region = _cylindrical(equations, conditions, symbols, facts) if real and dom == S.Reals else None
        if region is None:
            raise NotImplementedError("systems of inequalities in several unknowns are solved over the reals "
                                      "(domain=S.Reals, or real unknowns), for polynomial relations with "
                                      "rational coefficients, within the time limit")
        return region
    if isinstance(domain, Integers) or set(symbols) <= facts.integer:
        integer = _integer_linear_system(equations, conditions, symbols, facts)
        if integer is not None:
            return integer
    # a polynomial system with finitely many solutions: the exact points of
    # its regular chains, all of them by construction. nonlinsolve leaves an
    # unknown free when it cannot solve for it: {(x, -sqrt(2)), (x, sqrt(2))}
    # for [x**5 - x - 1 - y, y**2 - 2], and x, y polynomials in a free z for
    # three equations with twelve solutions, after 37 s where the chains
    # take two
    solutions = _by_regular_chains(equations, symbols, real, True)
    if solutions is None and real and dom == S.Reals:
        # real solutions in positive dimension: the cylindrical description
        # of the set, which has the conditions for the solutions to be real
        # and the inequalities in it (nonlinsolve gives families over the
        # complex numbers, x = sqrt(1 - y**2) for every y, under a condition
        # which is not evaluated)
        region = _cylindrical(equations, conditions, symbols, facts)
        if region is not None:
            return region
    unsolved = False
    unchecked = solutions is None
    if solutions is None:
        try:
            solutions = nonlinsolve(equations, list(symbols))
        except (NotImplementedError, ValueError, TypeError):
            unsolved = True
    # the points of the chains are not substituted back: they are exact, and
    # evalf spends the whole time limit on a residual which is exactly zero
    # at a root object (the bug: 34 s for two quadratic equations)
    if solutions is None or (unchecked and isinstance(solutions, FiniteSet) and _refuted(solutions, equations, symbols)):
        # the points of nonlinsolve are substituted back, and a system they
        # do not satisfy goes through the chains, with their families
        solutions = _by_regular_chains(equations, symbols, real, False)
        if solutions is None:
            if unsolved:
                raise NotImplementedError("the system cannot be solved")
            # a refuted answer is not returned: the system as it is
            return ConditionSet(Tuple(*symbols), And(condition, *[Eq(e, 0) for e in equations]),
                                ProductSet(*[dom] * len(symbols)))
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


def _parametric(statements: Sequence[Boolean], symbols: Sequence[Symbol], facts: Facts,
                domain: Optional[Set]) -> Optional[Set]:
    """The solutions with the cases of the parameters (``cases=True``), or
    ``None`` when the system has no parameter, is not a conjunction of
    polynomial relations of the kind its domain allows, or the
    decomposition does not end in time."""
    relations: list[Boolean] = []
    for s in statements:
        relations.extend(conjuncts(normalize(s)))
    parameters = sorted_symbols(set().union(*[free_symbols(c) for c in relations]) - set(symbols)) if relations else []
    if not parameters:
        return None
    unknown: Basic = symbols[0] if len(symbols) == 1 else Tuple(*symbols)
    dom = _domain_set(facts, symbols, domain)

    def informative(c: Boolean) -> bool:
        return not (isinstance(c, Contains) and c.args[1] == S.Reals)

    hypotheses = [c for c in facts.conjuncts if informative(c)]
    pieces: list[Set] = []
    if dom == S.Reals:
        system = And(*relations, *[c for c in hypotheses if free_symbols(c) <= set(parameters) | set(symbols)])
        discussed = attempt(lambda: cylindrical_cases(system, parameters, list(symbols)), settings.timeout)
        if discussed is None:
            return None
        for condition, found in discussed:
            pieces.append(_under(unknown, condition, found, facts))
        return as_set(SetUnion(*pieces)) if pieces else S.EmptySet
    if not isinstance(dom, Complexes) or not all(isinstance(c, (Eq, Ne)) for c in relations):
        return None
    equations = [as_expr(c.lhs - c.rhs) for c in relations if isinstance(c, Eq)]
    inequations = [as_expr(c.lhs - c.rhs) for c in relations if isinstance(c, Ne)]
    if not equations:
        return None
    from sympy_extras.solvers.parametric import parametric_cases
    found_cases = attempt(lambda: parametric_cases(equations, list(symbols), parameters, inequations), settings.timeout)
    if found_cases is None:
        return None
    space: Set = S.Complexes if len(symbols) == 1 else ProductSet(*[S.Complexes] * len(symbols))
    for case in found_cases:
        condition = case.condition
        if case.solutions is None:
            solutions: Set = ConditionSet(unknown, And(*[Eq(e, 0) for e in case.equations]), space)
        elif len(symbols) == 1:
            values = [point.args[0] for point in case.solutions.args]
            solutions = space if symbols[0] in values else FiniteSet(*values)
        else:
            solutions = case.solutions
        pieces.append(_under(unknown, And(condition, *[c for c in hypotheses if free_symbols(c) & set(symbols)]),
                             solutions, facts))
    return as_set(SetUnion(*pieces)) if pieces else S.EmptySet


def _under(unknown: Basic, condition: Boolean, solutions: Set, facts: Facts) -> Set:
    """The solutions of a case: nothing when the facts refute its
    condition, the solutions themselves when they prove it."""
    holds = _evaluate(normalize(as_boolean(condition)), facts) if free_symbols(condition) else None
    if condition == true or holds is True:
        return solutions
    if condition == false or holds is False:
        return S.EmptySet
    return ConditionSet(unknown, condition, solutions)


def _cylindrical(equations: Sequence[Expr], conditions: Sequence[Boolean], symbols: Sequence[Symbol],
                 facts: Facts) -> Optional[Set]:
    """The real solutions of a system of polynomial equations and
    inequalities with rational coefficients, from a cylindrical algebraic
    decomposition (:func:`~sympy_extras.polys.cad.cylindrical_set`): the
    isolated points, and a condition set which bounds the first unknown by
    numbers, the second by functions of the first, and so on. The
    parameters come before the unknowns, with what the assumptions say of
    them, and the condition bounds them first. ``None`` when the system is
    not of this kind or the decomposition does not end in time."""
    # every variable of the decomposition is a real number: the memberships
    # in the reals say nothing more
    def informative(c: Boolean) -> bool:
        return not (isinstance(c, Contains) and c.args[1] == S.Reals)

    system = And(*[c for c in conditions if informative(c)], *[Eq(e, 0) for e in equations])
    parameters = sorted_symbols(free_symbols(system) - set(symbols))
    hypotheses = [c for c in facts.conjuncts
                  if informative(c) and free_symbols(c) and free_symbols(c) <= set(parameters)]

    def decompose() -> Set:
        if not parameters:
            return cylindrical_set(system, list(symbols))
        formula = cylindrical_formula(And(system, *hypotheses), parameters + list(symbols))
        if formula == false:
            return S.EmptySet
        return ConditionSet(Tuple(*symbols), formula, ProductSet(*[S.Reals] * len(symbols)))

    return attempt(decompose, settings.timeout)


def _refuted(solutions: FiniteSet, equations: Sequence[Expr], symbols: Sequence[Symbol]) -> bool:
    """Whether a point of ``solutions`` clearly does not satisfy the
    equations: the residuals are evaluated numerically, at random rational
    values of the symbols left in the point."""
    rng = random.Random(0)
    if any(not isinstance(value, Expr) for point in solutions.args for value in point.args):
        # nonlinsolve puts sets among the coordinates: (-sqrt(...), -sqrt(3*z/2
        # - 5/4), Interval.open(-oo, 5/6)) for two equations in x, y, z
        return True

    def check() -> bool:
        for point in solutions.args:
            values: dict[Basic, Basic] = {s: v for s, v in zip(symbols, point.args)}
            for e in equations:
                residual = as_expr(e.xreplace(values))
                free = sorted_symbols(free_symbols(residual))
                for _ in range(2 if free else 1):
                    sample: dict[Basic, Basic] = {s: Rational(rng.randint(2, 40), rng.randint(2, 9)) for s in free}
                    terms = [as_expr(as_expr(term.xreplace(values)).xreplace(sample)) for term in e.expand().as_ordered_terms()]
                    scale = sum(abs(complex(term.evalf(15))) for term in terms)
                    if abs(complex(as_expr(residual.xreplace(sample)).evalf(15))) > 1e-8 * (1 + scale):
                        return True
        return False

    try:
        return attempt(check, settings.timeout) is True
    except ZeroDivisionError:
        return False


def _by_regular_chains(equations: Sequence[Expr], symbols: Sequence[Symbol], real: bool,
                       isolated: bool) -> Optional[Set]:
    """The solutions of a polynomial system with rational coefficients and
    no parameter, from its triangular decomposition: the points of the
    chains without free variables, exactly, and for a chain with free
    variables whose polynomials have degree at most two in their main
    variables the families it parametrizes, the free variables standing for
    themselves as in ``nonlinsolve`` (not with ``isolated``: the systems
    with finitely many solutions only); ``None`` otherwise."""
    from sympy_extras.polys.regularchains import triangularize
    try:
        chains = attempt(lambda: triangularize(list(equations), *symbols), settings.timeout)
    except ZeroDivisionError:
        chains = None
    if chains is None or (isolated and any(chain.dimension for chain in chains)):
        return None
    points: list[Tuple] = []
    for chain in chains:
        if chain.dimension == 0:
            found = attempt(lambda: chain.solutions(real), settings.timeout)
            if found is None:
                return None
            points.extend(Tuple(*[solution[s] for s in symbols]) for solution in found)
            continue
        families: list[dict[Basic, Basic]] = [{}]
        for p, v in zip(chain.polys, chain.main_variables):
            poly = Poly(p, v)
            if poly.degree() > 2:
                return None
            extended: list[dict[Basic, Basic]] = []
            for family in families:
                coefficients = [as_expr(as_expr(c).xreplace(family)) for c in poly.all_coeffs()]
                if len(coefficients) == 2:
                    values = [as_expr(cancel(-coefficients[1] / coefficients[0]))]
                else:
                    a, b, c = coefficients
                    root = sqrt(as_expr(cancel(b**2 - 4 * a * c)))
                    values = [as_expr((-b - root) / (2 * a)), as_expr((-b + root) / (2 * a))]
                for value in values:
                    longer = dict(family)
                    longer[v] = value
                    extended.append(longer)
            families = extended
        found_ = [Tuple(*[family.get(s, s) for s in symbols]) for family in families]
        if _refuted(FiniteSet(*found_), equations, symbols):
            return None
        points.extend(found_)
    return FiniteSet(*points)


def _integer_linear_system(equations: Sequence[Expr], conditions: Sequence[Boolean],
                           symbols: Sequence[Symbol], facts: Facts) -> Optional[Set]:
    """The integer solutions of a linear system with integer coefficients
    (by the Hermite normal form), as a parametrised set; the minimal
    nonnegative solutions (Contejean–Devie) when the unknowns are known to
    be nonnegative and the solutions are finitely many. ``None`` when the
    system is not linear."""
    from sympy_extras.solvers.integers import (_linear_system, linear_diophantine_system,
        minimal_nonnegative_solutions)
    system = _linear_system(equations, symbols)
    if system is None:
        return None
    general = linear_diophantine_system(equations, symbols)
    if general is None:
        return S.EmptySet
    values, parameters = general
    # the memberships in the integers are satisfied by construction
    condition = And(*[c for c in conditions
                      if not (isinstance(c, Contains) and c.args[1] == S.Integers and c.args[0] in symbols)])
    if not parameters:
        point = Tuple(*values)
        holds = _holds_at(condition, {s: v for s, v in zip(symbols, values)}, facts)
        if holds is False:
            return S.EmptySet
        if holds is True:
            return FiniteSet(point)
        return ConditionSet(Tuple(*symbols), condition, FiniteSet(point))
    nonnegative = all(_evaluate(s >= 0, facts) is True for s in symbols)
    if nonnegative:
        A, b = system
        try:
            particular, homogeneous = minimal_nonnegative_solutions(A, b)
        except RuntimeError:
            particular, homogeneous = [], [[0]]
        if not homogeneous:
            points = [Tuple(*[Integer(c) for c in p]) for p in particular]
            kept = [p for p in points if _holds_at(condition, {s: v for s, v in zip(symbols, p.args)}, facts) is not False]
            return FiniteSet(*kept) if kept else S.EmptySet
    solutions: Set = ImageSet(Lambda(Tuple(*parameters), Tuple(*values)),
                              ProductSet(*[S.Integers]*len(parameters)) if len(parameters) > 1 else S.Integers)
    if condition is true:
        return solutions
    return ConditionSet(Tuple(*symbols), condition, solutions)


def solve(equations: Union[Statement, Sequence[Statement]],
          symbols: Union[None, Symbol, Sequence[Symbol]] = None,
          assumptions: Assumptions = None, domain: Optional[Set] = None, cases: bool = False) -> Set:
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
    points with algebraic endpoints. Otherwise :func:`sympy.solveset` (or
    :func:`sympy.nonlinsolve` for systems) solves the equations and the
    solutions are kept when the assumptions hold at them, dropped when
    they fail and collected in a
    :class:`~sympy.sets.conditionset.ConditionSet` when this cannot be
    decided. The points of ``nonlinsolve`` are substituted back in the
    equations: a polynomial system with rational coefficients which they
    do not satisfy (``nonlinsolve`` leaves free an unknown it cannot solve
    for) is solved by a triangular decomposition into regular chains
    (:mod:`sympy_extras.polys.regularchains`), and another one is returned
    as a ``ConditionSet`` of its equations.

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
    if cases:
        discussed = _parametric(statements, unknowns, facts, domain)
        if discussed is not None:
            return discussed
    if len(unknowns) == 1:
        x = unknowns[0]
        formula = normalize(And(*statements))
        return _univariate(formula, x, facts, domain)
    return _multivariate(statements, unknowns, facts, domain)
