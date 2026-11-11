"""Solving equations and inequalities under assumptions.

The equations, the unknowns, the assumptions and the solutions are
objects of the core (:mod:`sympy_extras.core`, :mod:`sympy_extras.sets`):
the solutions are sets of the core, condition sets
(:class:`~sympy_extras.sets.conditionset.ConditionSet`), images of the
integers (:class:`~sympy_extras.sets.fancysets.ImageSet`) and finite sets
of points (:class:`~sympy_extras.sets.sets.PointSet`) among them.

The module imports no SymPy: :mod:`sympy_extras.assumptions.solve_sympy`
is its face on SymPy objects. The assumptions are decided by the native
:mod:`~sympy_extras.assumptions.ask` (with :mod:`~sympy_extras.assumptions.facts`,
:mod:`~sympy_extras.assumptions.refine` and
:mod:`~sympy_extras.assumptions.analysis`), the rational equations over
the complex numbers are solved by the native
:func:`~sympy_extras.polys.rationalsolve.solveset_rational`, the families
of a regular chain with the native polynomials, and the candidates are
checked natively. The steps which are not ported go through SymPy at run
time (the face holds them, the functions ending in ``_of_core``): SymPy's
``solveset`` (the other equations and the inequalities in one unknown),
``nonlinsolve``, ``periodicity`` and the evaluation of an intersection of
sets which is not ported (with ``Range``, ...), the triangular decompositions (:mod:`sympy_extras.polys.regularchains`,
:func:`sympy_extras.solvers.parametric.parametric_cases`), the
transcendental equations (:mod:`sympy_extras.solvers.transcendental`) and
the reliable numerical values of :mod:`sympy_extras._numeric`. The
cylindrical algebraic decomposition (:mod:`sympy_extras.polys.cad.qe`:
``solution_set``; :mod:`sympy_extras.polys.cad.cylindrical`:
``cylindrical_set``, ``cylindrical_formula``, ``cylindrical_cases``), the
radical forms of the roots of the CAD (:mod:`sympy_extras.polys.roots`),
the isolation of real roots (:mod:`sympy_extras.solvers.isolation`) and
the linear Diophantine systems (:mod:`sympy_extras.solvers.integers`) are
native.
"""
from __future__ import annotations

import random
from typing import Callable, Final, Optional, Sequence, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.evalf import N
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanTrue, Not, Or, and_, false, to_boolean,
    true)
from sympy_extras.core.mod import Mod
from sympy_extras.core.numbers import Integer, NegativeInfinity, Rational
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational, eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import abs_, im
from sympy_extras.functions.elementary.integers import floor
from sympy_extras.sets.conditionset import ConditionSet, condition_set
from sympy_extras.sets.fancysets import (Complexes, ComplexesClass, Integers, IntegersClass,
    Naturals0Class, NaturalsClass, RationalsClass, image_set)
from sympy_extras.sets.sets import (Complement, Contains, EmptySet, EmptySetClass, FiniteSet, Intersection,
    Interval, PointSet, Reals, RealsClass, Set, Union as SetUnion, complement, finite_set, intersection, interval,
    point_set, product_set, union)
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from .analysis import _pieces, certified_sign, domain_of
from .ask import Assumptions, Truth, _evaluate, _facts
from .facts import Facts, conjuncts, normalize, to_polynomial
from .refine import _Refiner

__all__ = ['solve', 'Statement', 'TriangularChain', 'ParametricCase']

#: an equation or inequality to solve: an expression (equated to zero), a
#: relation or a Boolean combination of relations
Statement = Union[Expr, Boolean, bool]

#: the points of a zero-dimensional chain, as values of the unknowns
ChainPoints = list[dict[Symbol, Expr]]


class TriangularChain:
    """A regular chain of a triangular decomposition, as the solver uses
    it (built by the face from :class:`sympy_extras.polys.regularchains.RegularChain`).

    Attributes
    ==========

    dimension : int
        The number of free variables.
    polys : list of Expr
        The polynomials of the chain.
    main_variables : list of Symbol
        Their main variables.
    solutions : callable
        ``solutions(real)``: the points of a chain of dimension zero (the
        real ones only with ``real``), ``None`` when they are not found.
    """

    def __init__(self, dimension: int, polys: list[Expr], main_variables: list[Symbol],
                 solutions: Callable[[bool], Optional[ChainPoints]]) -> None:
        self.dimension: Final[int] = dimension
        self.polys: Final[list[Expr]] = polys
        self.main_variables: Final[list[Symbol]] = main_variables
        self.solutions: Final[Callable[[bool], Optional[ChainPoints]]] = solutions


class ParametricCase:
    """A case of a system with parameters
    (:class:`sympy_extras.solvers.parametric.ParametricCase` on the core):
    the condition on the parameters, the solutions (a finite set of
    points, or ``None``) and the equations of the chain."""

    def __init__(self, condition: Boolean, solutions: Optional[Set], equations: list[Expr]) -> None:
        self.condition: Final[Boolean] = condition
        self.solutions: Final[Optional[Set]] = solutions
        self.equations: Final[list[Expr]] = equations


def _sorted_symbols(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _as_statement(item: Statement) -> Boolean:
    if isinstance(item, bool):
        return to_boolean(item)
    if isinstance(item, Expr):
        return eq(item, 0)
    if isinstance(item, Boolean):
        return item
    raise TypeError("an equation or a Boolean is expected, got %s" % (item,))


def _membership(x: Expr, s: Set) -> Boolean:
    """SymPy's ``Contains(x, s)``: the truth value when the membership is
    decided, the unevaluated membership otherwise."""
    found = s.contains(x)
    if isinstance(found, (BooleanTrue, BooleanFalse)):
        return found
    return Contains(x, s)


def _as_set(b: Basic) -> Set:
    if not isinstance(b, Set):
        raise TypeError("expected a set, got %s" % (b,))
    return b


def _unevaluated(s: Basic) -> bool:
    """Whether a set has an intersection, a union or a complement which
    the sets of the core do not evaluate."""
    if isinstance(s, (Intersection, Complement)):
        return True
    return any(_unevaluated(a) for a in s.args if isinstance(a, Set))


def _evaluated_set(s: Set) -> Set:
    """The set evaluated by SymPy when the core leaves an operation on it
    unevaluated (an intersection with the integers, ...)."""
    if not _unevaluated(s):
        return s
    from .solve_sympy import evaluated_set_of_core
    return evaluated_set_of_core(s)


def _intersect(*sets: Set) -> Set:
    """SymPy's ``Intersection``."""
    return _evaluated_set(intersection(*sets))


def _domain_set(facts: Facts, symbols: Sequence[Symbol], domain: Optional[Set]) -> Set:
    """The set the solutions are searched in by the solvers."""
    if domain is not None:
        if isinstance(domain, (RealsClass, IntegersClass, NaturalsClass, Naturals0Class, RationalsClass,
                               ComplexesClass)) or domain.is_subset(Reals):
            return domain
    if all(s in facts.real for s in symbols):
        return Reals
    return Complexes


def _solveset(formula: Boolean, x: Symbol, domain: Set) -> Set:
    """Solve a Boolean combination of relations in one variable, atom by
    atom: a rational equation over the complex numbers natively, the
    others with SymPy's ``solveset``."""
    if isinstance(formula, BooleanTrue):
        return domain
    if isinstance(formula, BooleanFalse):
        return EmptySet
    if isinstance(formula, And):
        return _intersect(*[_solveset(a, x, domain) for a in formula.args if isinstance(a, Boolean)])
    if isinstance(formula, Or):
        return union(*[_solveset(a, x, domain) for a in formula.args if isinstance(a, Boolean)])
    if isinstance(formula, Not):
        negated = formula.args[0]
        if isinstance(negated, Boolean):
            return _evaluated_set(complement(domain, _solveset(negated, x, domain)))
    if isinstance(formula, Contains):
        if formula.args[0] == x:
            return _intersect(domain, _as_set(formula.args[1]))
        return condition_set(x, formula, domain)
    if isinstance(formula, Relational):
        if x not in formula.free_symbols:
            return condition_set(x, formula, domain)
        rational = _rational_solutions(formula, x, domain)
        if rational is not None:
            return rational
        from .solve_sympy import solveset_of_core
        try:
            return solveset_of_core(formula, x, domain)
        except (NotImplementedError, ValueError, TypeError):
            return condition_set(x, formula, domain)
    return condition_set(x, formula, domain)


def _rational_solutions(formula: Relational, x: Symbol, domain: Set) -> Optional[Set]:
    """The solutions of a rational equation over the complex numbers by the
    native :func:`~sympy_extras.polys.rationalsolve.solveset_rational`
    (SymPy's answer), ``None`` for the other equations."""
    if not isinstance(formula, Eq) or not isinstance(domain, ComplexesClass) or x.given_assumptions:
        return None
    from sympy_extras.polys.rationalsolve import solveset_rational
    f = as_expr(formula.lhs - formula.rhs)
    if not _low_degree(f, x):
        return None
    found = attempt(lambda: solveset_rational(f, x), settings.timeout, native=True)
    if found is None:
        return None
    solutions, excluded = found
    result = finite_set(*solutions)
    if excluded:
        result = complement(result, finite_set(*excluded))
        if not isinstance(result, (FiniteSet, EmptySetClass)):
            return None
    return result


#: the largest degree of the numerator solved natively: the roots of the
#: quartics, which SymPy writes by Ferrari's formulas over its domain EX,
#: come out of the port in another (trigonometric) form
_NATIVE_DEGREE = 3


def _low_degree(f: Expr, x: Symbol) -> bool:
    """Whether the numerator of ``f`` is a polynomial in ``x`` of degree at
    most :data:`_NATIVE_DEGREE`."""
    from sympy_extras.polys.polyerrors import BasePolynomialError
    from sympy_extras.polys.polytools import degree
    numerator = f.as_numer_denom()[0]
    try:
        return numerator.is_polynomial(x) is True and degree(numerator, x) <= _NATIVE_DEGREE
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return False


def _holds_at(formula: Boolean, values: dict[Basic, Basic], facts: Facts) -> Truth:
    """Whether the formula holds at the values, under the facts about the
    remaining symbols."""
    instance = normalize(to_boolean(formula._xreplace_evaluated(values)))
    return _evaluate(instance, facts)


def _reliable_value(e: Expr, digits: int) -> Optional[Expr]:
    from .solve_sympy import reliable_value_of_core
    return reliable_value_of_core(e, digits)


def _clearly_not_real(e: Basic) -> bool:
    """Whether the imaginary part of a number is clearly nonzero, by
    numerical evaluation (when the settings allow it)."""
    if not settings.numerical_checks:
        return False
    digits = settings.precision
    # a value whose branch a rounding error chooses is not "clearly"
    # anything (the bug: log(-2*sqrt(2) - 2*sqrt(z)) - I*pi with z a sum
    # which is exactly zero, a real number, was found not real)
    value = _reliable_value(e, digits) if isinstance(e, Expr) else None
    if value is None:
        return False
    imaginary = N(im(value), digits)
    return imaginary.is_number and (abs_(imaginary) > Rational(1, 10)**(digits*2//3)) is true


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
    from sympy_extras.simplify.simplify import simplify
    difference = attempt(lambda: simplify(first - second), settings.timeout, native=True)
    if difference is not None and difference == 0:
        return True
    if difference is not None and difference.is_number and difference.is_zero is False:
        # a number the core knows not to be zero: the evaluation below
        # would find it apart
        return False
    # two conjugates have the same minimal polynomial, and are told apart
    # by their values only: not by values whose branch a rounding error
    # chooses (see reliable_form)
    apart, size = _reliable_value(first - second, 60), _reliable_value(first, 60)
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
    from sympy_extras.polys.numberfields.minpoly import minimal_polynomial
    t = Dummy('t')
    polynomials = attempt(lambda: (minimal_polynomial(first, t), minimal_polynomial(second, t)),
                          settings.timeout, native=True)
    return polynomials is not None and polynomials[0] == polynomials[1]


def _filter_finite(elements: Sequence[Expr], x: Symbol, condition: Boolean, facts: Facts,
                   formula: Boolean = true, real: bool = False) -> Set:
    """Keep the elements at which the condition holds, and put the
    undecided ones in a :class:`~sympy_extras.sets.conditionset.ConditionSet`.
    Elements at which the solved formula itself provably fails (SymPy's
    solvers return extraneous roots of radical equations), and elements
    which are clearly not real when ``real`` solutions are wanted, are
    dropped."""
    kept: list[Expr] = []
    undecided: list[Expr] = []
    for e in elements:
        if real and _clearly_not_real(e):
            continue
        if formula is not true:
            at_point = normalize(to_boolean(formula._xreplace_evaluated({x: e})))
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
    result: Set = finite_set(*kept) if kept else EmptySet
    if undecided:
        result = union(result, condition_set(x, condition, finite_set(*undecided)))
    return result


def _numeric_root(instance: Boolean) -> Truth:
    """Whether the equations of a formula without free symbols hold, by
    evaluation with 30 digits: ``False`` when a residual is clearly not
    zero, ``True`` when all vanish, ``None`` when this cannot be told."""
    if not settings.numerical_checks:
        return None
    equations = [c for c in conjuncts(instance) if isinstance(c, Eq)]
    if not equations or any(c.free_symbols for c in equations):
        return None
    digits = settings.precision
    for c in equations:
        residual = _reliable_value(as_expr(c.lhs - c.rhs), digits)
        if residual is None:
            return None
        if (abs_(residual) > Rational(1, 10)**(digits*2//3)) is true:
            return False
    return True


def _restrict(result: Set, x: Symbol, condition: Boolean, facts: Facts, formula: Boolean = true,
              real: bool = False) -> Set:
    """Restrict a solution set to the values satisfying the condition (and
    check the finite candidates against the solved formula)."""
    if isinstance(result, FiniteSet):
        return _filter_finite(list(result.elements), x, condition, facts, formula, real)
    if isinstance(result, SetUnion):
        return union(*[_restrict(a, x, condition, facts, formula, real) for a in result.sets])
    if isinstance(result, Intersection):
        return _intersect(*[_restrict(a, x, condition, facts, formula, real) for a in result.sets])
    if condition is true:
        return result
    if isinstance(result, EmptySetClass):
        return result
    if condition.free_symbols <= {x} and x in facts.real:
        polynomial = to_polynomial(condition, facts.real)
        if polynomial is not None:
            return _intersect(result, _radicals(_solution_set(polynomial, x)))
    return condition_set(x, condition, result)


def _solution_set(formula: Boolean, x: Symbol) -> Set:
    """The real solutions of a polynomial formula in one unknown by the
    CAD (:func:`sympy_extras.polys.cad.qe.solution_set`)."""
    from sympy_extras.polys.cad.qe import solution_set
    return solution_set(formula, x)


def _radicals(result: Set) -> Set:
    """Write the algebraic numbers in radicals when that is reasonable
    (see :func:`sympy_extras.polys.roots.in_radicals`)."""
    from sympy_extras.polys.roots import in_radicals
    return in_radicals(result)


def _univariate(formula: Boolean, x: Symbol, facts: Facts, domain: Optional[Set]) -> Set:
    dom = _domain_set(facts, [x], domain)
    # the membership in the domain is handled by the solvers themselves
    on_x = and_(*[c for c in facts.conjuncts if x in c.free_symbols
                  and not (isinstance(c, Contains) and c.args[0] == x and c.args[1] == dom)])
    others = and_(*[c for c in facts.conjuncts if x not in c.free_symbols])
    if others is not true:
        # the facts about the parameters are used when deciding the solutions
        facts = Facts([others]) if others.free_symbols else facts
    parameters = formula.free_symbols - {x}
    polynomial_on_x = to_polynomial(on_x, facts.real) if on_x.free_symbols <= {x} else None
    polynomial_formula = to_polynomial(formula, facts.real) if not parameters and x in facts.real else None
    if polynomial_on_x is not None and polynomial_formula is not None:
        real = _radicals(_solution_set(and_(polynomial_formula, polynomial_on_x), x))
        if isinstance(dom, RealsClass):
            return real
        return _intersect(real, dom)
    if isinstance(dom, RealsClass) and polynomial_on_x is not None:
        # transcendental formulas reduced to polynomial ones through
        # their kernels (exp, log, sin and cos, roots); the parameters
        # carry their assumptions into the polynomial solving and the
        # inversion of the kernels
        facts_ = facts
        from .solve_sympy import solve_transcendental_of_core
        transcendental = attempt(
            lambda: solve_transcendental_of_core(formula, x,
                                                 lambda f, v: _univariate(f, v, facts_, Reals),
                                                 lambda c: _evaluate(normalize(c), facts_)),
            settings.timeout)
        if transcendental is not None:
            if polynomial_on_x is not true:
                transcendental = _intersect(transcendental, _radicals(_solution_set(polynomial_on_x, x)))
            return _restrict(transcendental, x, true, facts, formula, real=True)
    # the parameters carry their assumptions while the equations are
    # solved; with parameters they are solved over the complex numbers and
    # the membership in the domain is decided afterwards
    refiner = _Refiner(facts)
    forward: dict[Basic, Basic] = {}
    back_symbols: dict[Basic, Basic] = {}
    for parameter in _sorted_symbols(parameters):
        dummy = refiner.dummy_for(parameter)
        if dummy is not None:
            forward[parameter] = dummy
            back_symbols[dummy] = parameter
    abstracted = to_boolean(formula._xreplace_evaluated(forward))
    condition = on_x
    solving_domain = dom
    if parameters and not isinstance(dom, ComplexesClass):
        ordered_condition = any(isinstance(c, (Gt, Lt, Ge, Le)) for c in conjuncts(condition))
        ordered_formula = any(isinstance(a, (Gt, Lt, Ge, Le)) for a in formula.atoms(Relational))
        if isinstance(dom, RealsClass) and ordered_formula:
            # an inequality only makes sense between real numbers
            solving_domain = Reals
        else:
            solving_domain = Complexes
            if not (isinstance(dom, RealsClass) and ordered_condition):
                condition = and_(_membership(x, dom), condition)
    if not parameters and dom.is_subset(Reals) is True:
        # a periodic inequality: SymPy answers over one period only
        periodic = attempt(lambda: _periodic(formula, x, facts, dom), settings.timeout)
        if periodic is not None:
            # the conditions on the unknown which are not a region of the
            # real line (an integrality) are applied as for any answer
            return _restrict(periodic, x, condition, facts, formula, real=True)
    result = _solveset(abstracted, x, solving_domain)
    result = _as_set(result._xreplace_evaluated(back_symbols))
    restricted = _restrict(result, x, condition, facts, formula, real=dom.is_subset(Reals) is True)
    if not parameters and restricted.has(ConditionSet) and dom.is_subset(Reals) is True:
        # no closed form: the real roots are isolated instead
        isolated = attempt(lambda: _isolated(formula, x, facts, dom), settings.timeout)
        if isolated is not None:
            return isolated
    return restricted


def _period(formula: Boolean, x: Symbol) -> Optional[Expr]:
    """The common period of the relations of a formula in one unknown, or
    ``None`` when one of them is not periodic."""
    from .solve_sympy import periodicity_of_core
    period: Optional[Expr] = None
    for atom in formula.atoms(Relational):
        if not isinstance(atom, Relational):
            continue
        difference = as_expr(atom.lhs - atom.rhs)
        if x not in difference.free_symbols:
            continue
        value = periodicity_of_core(difference, x)
        if value is None or value.is_positive is not True or value.is_finite is not True:
            return None
        if value.free_symbols:
            return None
        period = value if period is None else _common_period(period, value)
        if period is None:
            return None
    return period


def _common_period(first: Expr, second: Expr) -> Optional[Expr]:
    """The least common multiple of two periods (``None`` when their
    ratio is not rational, so that no common period exists)."""
    from sympy_extras.simplify.simplify import simplify
    ratio = simplify(first/second)
    if not isinstance(ratio, Rational):
        return None
    return first*ratio.q


def _region(x: Symbol, facts: Facts, dom: Set) -> Set:
    """The region the solutions are looked for in; when the assumptions do
    not describe a region of the real line (an integrality among them)
    the relations alone describe one, and the rest is applied afterwards."""
    described = domain_of(x, facts.with_reals([x]))
    if described is None:
        relations: list[Boolean] = [c for c in facts.conjuncts
                                    if isinstance(c, Relational) and c.free_symbols <= {x}]
        described = domain_of(x, Facts(relations, Reals, [x]).with_reals([x]))
    return dom if described is None else _intersect(described, dom)


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
    if formula.free_symbols != {x} or not formula.atoms(Gt, Lt, Ge, Le):
        return None
    if formula.atoms(Eq, Ne):
        return None
    period = _period(formula, x)
    if period is None:
        return None
    window = interval(Integer(0), period, False, True)
    base = _base_over_period(formula, x, period)
    if base is None:
        return None
    region = _region(x, facts, dom)
    if isinstance(base, EmptySetClass) or isinstance(region, EmptySetClass):
        return EmptySet
    if base == window:
        return region
    tiles = _tiles(_pieces(region), period)
    if tiles is None:
        return condition_set(x, _membership(Mod(x, period), base), region)
    copies = [_translated(base, k*period) for k in tiles]
    return _intersect(union(*copies), region)


def _translated(s: Set, offset: Expr) -> Set:
    """The set moved by ``offset`` (SymPy's ``imageset(Lambda(t, t +
    offset), s)``)."""
    if offset == 0:
        return s
    if isinstance(s, Interval):
        return interval(s.start + offset, s.end + offset, s.left_open, s.right_open)
    if isinstance(s, FiniteSet):
        return finite_set(*[e + offset for e in s.elements])
    if isinstance(s, SetUnion):
        return union(*[_translated(a, offset) for a in s.sets])
    if isinstance(s, EmptySetClass):
        return s
    shift = Dummy('shift')
    return image_set(Tuple(shift), shift + offset, s)


#: the largest number of periods tiled explicitly (beyond it the solutions
#: are described by their residue)
_TILES = 64


def _tiles(pieces: Optional[Sequence[Interval]], period: Expr) -> Optional[range]:
    """The periods meeting the region, or ``None`` when there are too
    many of them (or infinitely many)."""
    if not pieces:
        return None
    low = min((piece.start for piece in pieces), key=_ordering)
    high = max((piece.end for piece in pieces), key=_ordering)
    if low.is_finite is not True or high.is_finite is not True:
        return None
    first, last = floor(low/period), floor(high/period)
    if not (isinstance(first, Integer) and isinstance(last, Integer)):
        return None
    if int(last) - int(first) + 1 > _TILES:
        return None
    return range(int(first), int(last) + 1)


def _ordering(point: Expr) -> float:
    """A key ordering the endpoints of the pieces of a region."""
    value = N(point, 20)
    if value.is_real and value.is_finite:
        return float(value)
    return float('-inf') if isinstance(point, NegativeInfinity) else float('inf')


def _base_over_period(formula: Boolean, x: Symbol, period: Expr) -> Optional[Set]:
    """The solutions over one period ``[0, T)``, isolated from the sign of
    the relation between its roots when that works (the answer does not
    then depend on SymPy's solving of the inequality), and left to
    ``solveset`` over the bounded window otherwise."""
    window = interval(Integer(0), period, False, True)
    facts = Facts([], Reals, [x])
    isolated = attempt(lambda: _isolated(formula, x, facts, interval(Integer(0), period)), settings.timeout)
    if isolated is not None:
        return _intersect(isolated, window)
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
    from sympy_extras.solvers.isolation import compare_points, isolate_real_roots, point_between
    if not isinstance(formula, Relational) or formula.free_symbols != {x}:
        return None
    region = _region(x, facts, dom)
    pieces = _pieces(region)
    if pieces is None:
        return None
    f = as_expr(formula.lhs - formula.rhs)
    roots = isolate_real_roots(f, x, region)
    if roots is None:
        return None
    if isinstance(formula, Eq):
        return finite_set(*roots) if roots else EmptySet
    if isinstance(formula, Ne):
        return _evaluated_set(complement(region, finite_set(*roots))) if roots else region
    strict = isinstance(formula, (Gt, Lt))
    wanted = 1 if isinstance(formula, (Gt, Ge)) else -1
    parts: list[Set] = []
    for piece in pieces:
        a, b = piece.start, piece.end
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
                parts.append(interval(p, q, True, True))
        if not strict:
            parts.extend(finite_set(r) for r in inner)
        for closed, point in ((not piece.left_open, a), (not piece.right_open, b)):
            if closed and point.is_finite:
                s = certified_sign(as_expr(f.subs(x, point)))
                if s == wanted or (s == 0 and not strict):
                    parts.append(finite_set(point))
    return union(*parts) if parts else EmptySet


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
    conditions.extend(c for c in facts.conjuncts if c.free_symbols & set(symbols))
    condition = and_(*conditions)
    dom = _domain_set(facts, symbols, domain)
    real = dom.is_subset(Reals) is True
    if not equations:
        region = _cylindrical(equations, conditions, symbols, facts) if real and dom == Reals else None
        if region is None:
            raise NotImplementedError("systems of inequalities in several unknowns are solved over the reals "
                                      "(domain=S.Reals, or real unknowns), for polynomial relations with "
                                      "rational coefficients, within the time limit")
        return region
    if isinstance(domain, IntegersClass) or set(symbols) <= facts.integer:
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
    if solutions is None and real and dom == Reals:
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
    refuted = False
    if solutions is None:
        from .solve_sympy import nonlinsolve_of_core
        try:
            solutions = nonlinsolve_of_core(equations, list(symbols))
            # None: points with sets among their coordinates
            refuted = solutions is None
        except (NotImplementedError, ValueError, TypeError):
            unsolved = True
    # the points of the chains are not substituted back: they are exact, and
    # evalf spends the whole time limit on a residual which is exactly zero
    # at a root object (the bug: 34 s for two quadratic equations)
    if solutions is None or (unchecked and isinstance(solutions, PointSet)
                             and _refuted(solutions, equations, symbols)):
        # the points of nonlinsolve are substituted back, and a system they
        # do not satisfy goes through the chains, with their families
        solutions = _by_regular_chains(equations, symbols, real, False)
        if solutions is None:
            if unsolved and not refuted:
                raise NotImplementedError("the system cannot be solved")
            # a refuted answer is not returned: the system as it is
            return condition_set(Tuple(*symbols), and_(condition, *[eq(e, 0) for e in equations]),
                                 product_set(*[dom] * len(symbols)))
    if not isinstance(solutions, PointSet):
        return condition_set(Tuple(*symbols), condition, solutions) if condition is not true else solutions
    kept: list[Tuple] = []
    undecided: list[Tuple] = []
    for point in solutions.points:
        values: dict[Basic, Basic] = {s: v for s, v in zip(symbols, point.args)}
        membership = and_(*[_membership(as_expr(v), dom) for v in point.args]) \
            if not isinstance(dom, ComplexesClass) else true
        value = _holds_at(and_(condition, membership), values, facts)
        if value is True:
            kept.append(point)
        elif value is None:
            undecided.append(point)
    result: Set = point_set(*kept) if kept else EmptySet
    if undecided:
        # the memberships in the domain are those of the unknowns, which the
        # condition set gives the coordinates of each point in turn (the
        # bug: the memberships of the coordinates of the last point, left
        # over from the loop, were the condition of every point, and the
        # real branch of x**3 = 1 - y**3 - y**2 was dropped because a
        # complex one is not real)
        inside = and_(*[_membership(s, dom) for s in symbols]) if not isinstance(dom, ComplexesClass) else true
        result = union(result, condition_set(Tuple(*symbols), and_(condition, inside), point_set(*undecided)))
    return result


def _informative(c: Boolean) -> bool:
    """Whether an assumption says more than the membership of a variable in
    the reals (every variable of a decomposition is real)."""
    return not (isinstance(c, Contains) and c.args[1] == Reals)


def _parametric(statements: Sequence[Boolean], symbols: Sequence[Symbol], facts: Facts,
                domain: Optional[Set]) -> Optional[Set]:
    """The solutions with the cases of the parameters (``cases=True``), or
    ``None`` when the system has no parameter, is not a conjunction of
    polynomial relations of the kind its domain allows, or the
    decomposition does not end in time."""
    relations: list[Boolean] = []
    for s in statements:
        relations.extend(conjuncts(normalize(s)))
    found_symbols: set[Symbol] = set()
    for c in relations:
        found_symbols |= c.free_symbols
    parameters = _sorted_symbols(found_symbols - set(symbols))
    if not parameters:
        return None
    unknown: Basic = symbols[0] if len(symbols) == 1 else Tuple(*symbols)
    dom = _domain_set(facts, symbols, domain)
    hypotheses = [c for c in facts.conjuncts if _informative(c)]
    pieces: list[Set] = []
    if dom == Reals:
        system = and_(*relations, *[c for c in hypotheses if c.free_symbols <= set(parameters) | set(symbols)])
        from sympy_extras.polys.cad.cylindrical import cylindrical_cases
        discussed = attempt(lambda: cylindrical_cases(system, parameters, list(symbols)), settings.timeout,
                            native=True)
        if discussed is None:
            return None
        for condition, found in discussed:
            pieces.append(_under(unknown, condition, found, facts))
        return union(*pieces) if pieces else EmptySet
    if not isinstance(dom, ComplexesClass) or not all(isinstance(c, (Eq, Ne)) for c in relations):
        return None
    equations = [as_expr(c.lhs - c.rhs) for c in relations if isinstance(c, Eq)]
    inequations = [as_expr(c.lhs - c.rhs) for c in relations if isinstance(c, Ne)]
    if not equations:
        return None
    from .solve_sympy import parametric_cases_of_core
    found_cases = attempt(lambda: parametric_cases_of_core(equations, list(symbols), parameters, inequations),
                          settings.timeout)
    if found_cases is None:
        return None
    space: Set = Complexes if len(symbols) == 1 else product_set(*[Complexes] * len(symbols))
    for case in found_cases:
        if case.solutions is None:
            solutions: Set = condition_set(unknown, and_(*[eq(e, 0) for e in case.equations]), space)
        elif len(symbols) == 1:
            values = [as_expr(point.args[0]) for point in case.solutions.args if isinstance(point, Tuple)]
            solutions = space if symbols[0] in values else finite_set(*values)
        else:
            solutions = case.solutions
        pieces.append(_under(unknown, and_(case.condition, *[c for c in hypotheses if c.free_symbols & set(symbols)]),
                             solutions, facts))
    return union(*pieces) if pieces else EmptySet


def _under(unknown: Basic, condition: Boolean, solutions: Set, facts: Facts) -> Set:
    """The solutions of a case: nothing when the facts refute its
    condition, the solutions themselves when they prove it."""
    holds = _evaluate(normalize(condition), facts) if condition.free_symbols else None
    if condition == true or holds is True:
        return solutions
    if condition == false or holds is False:
        return EmptySet
    return condition_set(unknown, condition, solutions)


def _cylindrical(equations: Sequence[Expr], conditions: Sequence[Boolean], symbols: Sequence[Symbol],
                 facts: Facts) -> Optional[Set]:
    """The real solutions of a system of polynomial equations and
    inequalities with rational coefficients, from a cylindrical algebraic
    decomposition (:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_set`): the
    isolated points, and a condition set which bounds the first unknown by
    numbers, the second by functions of the first, and so on. The
    parameters come before the unknowns, with what the assumptions say of
    them, and the condition bounds them first. ``None`` when the system is
    not of this kind or the decomposition does not end in time."""
    from sympy_extras.polys.cad.cylindrical import cylindrical_formula, cylindrical_set
    system = and_(*[c for c in conditions if _informative(c)], *[eq(e, 0) for e in equations])
    parameters = _sorted_symbols(system.free_symbols - set(symbols))
    hypotheses = [c for c in facts.conjuncts
                  if _informative(c) and c.free_symbols and c.free_symbols <= set(parameters)]

    def decompose() -> Set:
        if not parameters:
            return cylindrical_set(system, list(symbols))
        formula = cylindrical_formula(and_(system, *hypotheses), parameters + list(symbols))
        if formula == false:
            return EmptySet
        return condition_set(Tuple(*symbols), formula, product_set(*[Reals] * len(symbols)))

    return attempt(decompose, settings.timeout, native=True)


def _refuted(solutions: PointSet, equations: Sequence[Expr], symbols: Sequence[Symbol]) -> bool:
    """Whether a point of ``solutions`` clearly does not satisfy the
    equations: the residuals are evaluated numerically, at random rational
    values of the symbols left in the point."""
    rng = random.Random(0)

    def check() -> bool:
        for point in solutions.points:
            values: dict[Basic, Basic] = {s: v for s, v in zip(symbols, point.args)}
            for e in equations:
                residual = as_expr(e._xreplace_evaluated(values))
                free = _sorted_symbols(residual.free_symbols)
                for _ in range(2 if free else 1):
                    sample: dict[Basic, Basic] = {s: Rational(rng.randint(2, 40), rng.randint(2, 9)) for s in free}
                    terms = [as_expr(as_expr(term._xreplace_evaluated(values))._xreplace_evaluated(sample))
                             for term in e.expand().as_ordered_terms()]
                    scale = sum(abs(complex(term.evalf(15))) for term in terms)
                    if abs(complex(as_expr(residual._xreplace_evaluated(sample)).evalf(15))) > 1e-8 * (1 + scale):
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
    from sympy_extras.polys.polytools import Poly, cancel
    from .solve_sympy import triangularize_of_core
    chains = triangularize_of_core(list(equations), list(symbols))
    if chains is None or (isolated and any(chain.dimension for chain in chains)):
        return None
    points: list[Tuple] = []
    for chain in chains:
        if chain.dimension == 0:
            solutions = chain.solutions
            found = attempt(lambda: solutions(real), settings.timeout)
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
                coefficients = [as_expr(c._xreplace_evaluated(family)) for c in poly.all_coeffs()]
                if len(coefficients) == 2:
                    values = [cancel(-coefficients[1] / coefficients[0])]
                else:
                    a, b, c = coefficients
                    root = sqrt(cancel(b**2 - 4 * a * c))
                    values = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
                for value in values:
                    longer = dict(family)
                    longer[v] = value
                    extended.append(longer)
            families = extended
        found_ = [Tuple(*[family.get(s, s) for s in symbols]) for family in families]
        candidates = point_set(*found_)
        if isinstance(candidates, PointSet) and _refuted(candidates, equations, symbols):
            return None
        points.extend(found_)
    return point_set(*points)


def _integer_linear_system(equations: Sequence[Expr], conditions: Sequence[Boolean],
                           symbols: Sequence[Symbol], facts: Facts) -> Optional[Set]:
    """The integer solutions of a linear system with integer coefficients
    (by the Hermite normal form), as a parametrised set; the minimal
    nonnegative solutions (Contejean–Devie) when the unknowns are known to
    be nonnegative and the solutions are finitely many. ``None`` when the
    system is not linear."""
    from sympy_extras.solvers.integers import _linear_system, linear_diophantine_system, minimal_nonnegative_solutions
    system = _linear_system(equations, symbols)
    if system is None:
        return None
    general = linear_diophantine_system(equations, symbols)
    if general is None:
        return EmptySet
    values, parameters = general
    # the memberships in the integers are satisfied by construction
    condition = and_(*[c for c in conditions
                       if not (isinstance(c, Contains) and c.args[1] == Integers and c.args[0] in symbols)])
    if not parameters:
        point = Tuple(*values)
        holds = _holds_at(condition, {s: v for s, v in zip(symbols, values)}, facts)
        if holds is False:
            return EmptySet
        if holds is True:
            return point_set(point)
        return condition_set(Tuple(*symbols), condition, point_set(point))
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
            return point_set(*kept) if kept else EmptySet
    solutions = image_set(Tuple(*parameters), Tuple(*values),
                          product_set(*[Integers] * len(parameters)) if len(parameters) > 1 else Integers)
    if condition is true:
        return solutions
    return condition_set(Tuple(*symbols), condition, solutions)


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
        statements (see :mod:`sympy_extras.assumptions.facts`); the global
        assumptions of :mod:`sympy_extras.assumptions.context` are used as
        well.
    domain : Set, optional
        The set the unknowns belong to (``Reals``, ``Integers``, ...).
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
        the reals (``domain=Reals``, the parameters are then real numbers
        too), with inequalities as well, those of a cylindrical
        decomposition with the parameters first
        (:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_cases`). Without it
        the solutions are the generic ones: ``{b/a}`` for ``a*x = b``.

    Returns
    =======

    A set of :mod:`sympy_extras.sets`. For one unknown with polynomial
    equations and inequalities over the reals the solution set is computed
    exactly by cylindrical algebraic decomposition, as a union of
    intervals and points with algebraic endpoints. Otherwise the equations
    are solved (the rational ones over the complex numbers natively, the
    others by SymPy's ``solveset``, or ``nonlinsolve`` for systems) and the
    solutions are kept when the assumptions hold at them, dropped when
    they fail and collected in a
    :class:`~sympy_extras.sets.conditionset.ConditionSet` when this cannot
    be decided. The points of ``nonlinsolve`` are substituted back in the
    equations: a polynomial system with rational coefficients which they
    do not satisfy (``nonlinsolve`` leaves free an unknown it cannot solve
    for) is solved by a triangular decomposition into regular chains
    (:mod:`sympy_extras.polys.regularchains`), and another one is returned
    as a ``ConditionSet`` of its equations.

    Over the reals (``domain=Reals``, or real unknowns), a system of
    polynomial inequalities in several unknowns, with or without
    equations, and a system of equations with infinitely many solutions,
    are described cylindrically
    (:func:`~sympy_extras.polys.cad.cylindrical.cylindrical_set`): the points with
    algebraic coordinates as a finite set of points, and the rest as a
    ``ConditionSet`` whose condition bounds the first unknown by numbers,
    the second one by functions of the first, and so on. The parameters
    of the system are bounded before the unknowns, within what the
    assumptions say of them.

    Examples
    ========

    >>> from sympy_extras.core import Eq, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.sets import Integers, Reals
    >>> from sympy_extras.assumptions.solve import solve
    >>> x, y, a = symbols('x y a')
    >>> solve(x**2 - 2, x), solve(x**2 + 1, x)
    ({-sqrt(2), sqrt(2)}, {-I, I})
    >>> solve(x**2 - 2, x, x > 0)
    {sqrt(2)}
    >>> solve((x**2 - 2 > 0) & (x < 3), x, domain=Reals)
    Union(Interval.open(-oo, -sqrt(2)), Interval.open(sqrt(2), 3))
    >>> solve(x**2 - a, x, (x > 0) & (a > 0)), solve(a*x - 1, x, a > 0)
    ({sqrt(a)}, {1/a})
    >>> solve(exp(x) - 2, x, x > 1)
    EmptySet
    >>> solve([x**2 + y**2 - 1, x - y], [x, y], x > 0)
    {(sqrt(2)/2, sqrt(2)/2)}
    >>> solve(a*x - 1, x, cases=True)
    ConditionSet(x, Ne(a, 0), {1/a})
    >>> solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=Integers)
    {(4, 2)}
    """
    items: list[Statement] = [equations] if isinstance(equations, (Basic, bool)) else list(equations)
    statements = [_as_statement(item) for item in items]
    all_symbols: set[Symbol] = set()
    for s in statements:
        all_symbols |= s.free_symbols
    if symbols is None:
        unknowns = _sorted_symbols(all_symbols)
    elif isinstance(symbols, Symbol):
        unknowns = [symbols]
    else:
        unknowns = list(symbols)
    if not unknowns:
        raise ValueError("no unknown to solve for")
    try:
        facts = _facts(assumptions, domain, all_symbols | set(unknowns))
    except ValueError:
        return EmptySet
    if cases:
        discussed = _parametric(statements, unknowns, facts, domain)
        if discussed is not None:
            return discussed
    if len(unknowns) == 1:
        x = unknowns[0]
        formula = normalize(and_(*statements))
        return _univariate(formula, x, facts, domain)
    return _multivariate(statements, unknowns, facts, domain)
