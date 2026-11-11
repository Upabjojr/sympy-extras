"""Satisfiability of formulas mixing Boolean variables, polynomial relations
and memberships.

The formulas are Booleans of the core (:mod:`sympy_extras.core.logic`):
Boolean symbols, relations, memberships in the sets of
:mod:`sympy_extras.sets`, predicates of the core's ``Q`` and the
quantified formulas of :mod:`sympy_extras.assumptions.quantifiers`.

The Boolean structure of a formula is handled by a SAT solver on a
propositional abstraction in which every relation or membership is
replaced by a Boolean variable. Each candidate model is then checked
against the theory: the predicates implied by the chosen literals must be
consistent with the known facts about the predicates, and the polynomial
relations between real variables must have a common real solution, found
by cylindrical algebraic decomposition, which also provides a witness.
This is the lazy approach of SMT solvers. When the equations among the
relations have finitely many complex solutions, the absence of a real
solution is proved first by counting them with Hermite's quadratic form
(:mod:`sympy_extras.polys.hermite`), without a decomposition.

As in Mathematica's ``Reduce`` and ``FindInstance``, inequalities are
statements about real numbers: a variable appearing in an inequality of the
formula is taken to be real, also when the inequality comes from the
negation of another one (``Not(x >= 0)`` is ``x < 0``). Equations and
``Ne`` do not make their variables real, so a real solution of an
equation is a witness but the absence of real solutions is only
conclusive when the variables are known to be real.

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.sat_sympy` is its face on SymPy objects.
The abstraction, the theory check, the search for integer points and the
evaluation of the candidates are native; the steps which are not ported
go through SymPy at run time (the face holds them, the functions ending
in ``_of_core``): SymPy's SAT solver
(:func:`sympy.logic.inference.satisfiable`) on the propositional
abstraction, the consistency of predicates with SymPy's known facts
(:func:`~sympy_extras.assumptions.facts_sympy.predicates_consistent`),
the count of real solutions by Hermite's form
(:mod:`sympy_extras.polys.hermite`) and the floor and the comparison of
SymPy's ``CRootOf`` (not ported). The sample points and the solution sets
of the cylindrical algebraic decomposition are native
(:mod:`sympy_extras.polys.cad.qe`), and so are the floor and the
comparison of the real algebraic numbers they give.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Literal, Optional, Sequence, Union, overload

from sympy_extras.core.assume import AppliedPredicate
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import (Boolean, BooleanFalse, BooleanLike, BooleanTrue, Not, and_, false, not_,
    to_boolean, true)
from sympy_extras.core.numbers import Integer, NegativeInfinity, Infinity, Rational
from sympy_extras.core.relational import Eq, Ne, Relational
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.sets.sets import Contains, FiniteSet, Interval, Reals, RealsClass, Set, Union as SetUnion
from sympy_extras._timeout import attempt

from .ask import Assumptions, _facts
from .facts import Facts, _REAL_SETS, _consequences, _is_atom, normalize, to_polynomial, to_predicates
from .quantifiers import Quantifier

__all__ = ['satisfiable', 'tautology', 'find_instance', 'Model', 'Witness']


#: a model: Boolean variables to truth values, real variables to values
Model = dict[Basic, Union[bool, Expr]]
#: a witness point
Witness = dict[Symbol, Expr]

#: the seconds given to the count of real solutions of a zero-dimensional
#: system before the decomposition of the space is used instead
_HERMITE_TIME_LIMIT = 2.0


def _symbols(e: Basic) -> set[Symbol]:
    """The free symbols which are plain symbols."""
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _sorted_symbols(symbols: set[Symbol]) -> list[Symbol]:
    return sorted(symbols, key=lambda s: s.name)


def _abstract(formula: Boolean) -> tuple[Boolean, dict[Basic, Boolean]]:
    """Replace the non-propositional atoms of a normalized formula by
    Boolean dummies. Returns the abstraction and the map from dummies to
    atoms."""
    atoms: dict[Basic, Boolean] = {}
    mapping: dict[Basic, Basic] = {}
    for atom in formula.atoms(Relational, Contains, AppliedPredicate):
        if not isinstance(atom, Boolean):
            continue
        d = Dummy('p%d' % len(atoms))
        atoms[d] = atom
        mapping[atom] = d
    return to_boolean(formula._xreplace_evaluated(mapping)), atoms


def _literals(model: dict[Basic, bool], atoms: dict[Basic, Boolean]) -> list[Boolean]:
    """The theory literals selected by a propositional model."""
    literals: list[Boolean] = []
    for symbol, value in model.items():
        if not isinstance(symbol, Dummy):
            continue
        atom = atoms.get(symbol)
        if atom is None:
            continue
        literals.append(atom if value else not_(atom))
    return literals


def _floor(a: Expr) -> int:
    """The floor of a real algebraic number."""
    if isinstance(a, Rational):
        return a.p // a.q
    from sympy_extras.polys.cad.indexedroot import algebraic_of
    from sympy_extras.polys.cad.samplepoints import floor_scaled
    number = algebraic_of(a)
    if number is not None:
        return floor_scaled(number, 0)
    from .sat_sympy import floor_of_core
    return floor_of_core(a)


def _compare(a: Expr, b: Expr) -> int:
    """The sign of ``a - b`` for real algebraic numbers."""
    if isinstance(a, Rational) and isinstance(b, Rational):
        d = Fraction(a.p, a.q) - Fraction(b.p, b.q)
        return (d > 0) - (d < 0)
    from sympy_extras.polys.cad.indexedroot import algebraic_of
    from sympy_extras.polys.cad.samplepoints import compare_real
    first, second = algebraic_of(a), algebraic_of(b)
    if first is not None and second is not None:
        return compare_real(first, second)
    from .sat_sympy import compare_real_of_core
    return compare_real_of_core(a, b)


def _integer_in(sets: Set) -> Optional[Expr]:
    """An integer in a union of intervals and points with real algebraic
    endpoints, or ``None`` if there is none."""
    parts = list(sets.args) if isinstance(sets, SetUnion) else [sets]
    for part in parts:
        if isinstance(part, FiniteSet):
            for v in part.elements:
                if v.is_integer:
                    return v
            continue
        if not isinstance(part, Interval):
            continue
        a, b = part.start, part.end
        if isinstance(a, NegativeInfinity):
            if isinstance(b, Infinity):
                return Integer(0)
            c = _floor(b)
            if part.right_open and Integer(c) == b:
                c -= 1
            return Integer(c)
        c = _floor(a)
        if Integer(c) != a or part.left_open:
            c += 1
        if isinstance(b, Infinity) or _compare(Integer(c), b) < 0 or (Integer(c) == b and not part.right_open):
            return Integer(c)
    return None


def _integer_witness(formula: Boolean, points: Sequence[Witness], integer: set[Symbol],
                     gens: Sequence[Symbol]) -> Union[Witness, bool, None]:
    """A point with integer values for the variables in ``integer`` (and
    real values for the others) satisfying the polynomial ``formula``.

    Returns the point, ``False`` if there is none or ``None`` if it could
    not be decided. With a single integer variable the answer is exact:
    the formula is projected on that variable and the resulting union of
    intervals is searched for an integer. With several integer variables
    only the neighbourhood of the real sample points is searched.
    """
    from sympy_extras.polys.cad.qe import sample_points, solution_set
    integer_gens = [g for g in gens if g in integer]
    if len(integer_gens) == 1:
        [n] = integer_gens
        others = [g for g in gens if g != n]
        sets: Optional[Set]
        try:
            sets = solution_set(formula, n, [('exists', others)] if others else [])
        except NotImplementedError:
            sets = None
        if sets is not None:
            k = _integer_in(sets)
            if k is None:
                return False
            witness: Witness = {n: k}
            if others:
                points_ = sample_points(to_boolean(formula.subs(n, k)), others)
                if not points_:
                    return None
                witness.update(points_[0])
            return witness
    for point in points:
        choices: list[list[Expr]] = []
        for g in gens:
            v = point[g]
            if g in integer:
                lo = _floor(v)
                choices.append([Integer(i) for i in sorted({lo, lo + 1, lo - 1, lo + 2})])
            else:
                choices.append([v])
        for combo in product(*choices):
            candidate: Witness = dict(zip(gens, combo))
            value = formula.subs(list(candidate.items()))
            if isinstance(value, BooleanTrue):
                return candidate
    return None


def _without_real_solution(formula: Boolean, gens: Sequence[Symbol]) -> bool:
    """Whether a conjunction of polynomial relations whose equations have
    finitely many complex solutions is proved to have no real solution by
    counting them with Hermite's quadratic form, which is much cheaper than
    a decomposition of the space when there is none (every cell must then
    be visited); ``False`` when this does not apply or takes too long, and
    the decomposition decides."""
    if len(gens) < 2:
        return False
    from .sat_sympy import decide_zero_dimensional_of_core
    decided = attempt(lambda: decide_zero_dimensional_of_core(formula, list(gens)), _HERMITE_TIME_LIMIT)
    return decided is False


def _is_negated(lit: Boolean, cls: type[Basic]) -> bool:
    """Whether ``lit`` is the negation of a ``cls``."""
    return isinstance(lit, Not) and isinstance(lit.arg, cls)


def _theory_check(literals: Sequence[Boolean], facts: Facts) -> Union[Witness, bool, None]:
    """Check the conjunction of theory literals for consistency.

    Returns a witness dict (possibly empty) if the literals are
    satisfiable, ``False`` if they are not and ``None`` if it cannot be
    decided.
    """
    from sympy_extras.polys.cad.qe import sample_points
    from .sat_sympy import predicates_consistent_of_core
    predicates: list[Boolean] = []
    for lit in literals:
        p = to_predicates(lit)
        if p is not None:
            predicates.append(p)
        if _is_atom(lit):
            predicates.extend(_consequences(lit))
    predicates.append(facts.predicates)
    if not predicates_consistent_of_core(and_(*predicates)):
        return False

    # variables which are real by an inequality literal or a membership
    real = set(facts.real)
    integer = set(facts.integer)
    for lit in literals:
        if isinstance(lit, Relational) and not isinstance(lit, (Eq, Ne)):
            real.update(_symbols(lit))
        if isinstance(lit, Contains):
            x, s = lit.args
            if isinstance(x, Symbol):
                if isinstance(s, _REAL_SETS):
                    real.add(x)
                if not isinstance(s, RealsClass):
                    integer.add(x)

    # a real solution of the polynomial atoms is a solution whatever the
    # domain of the variables, so every variable is taken as real to look
    # for a witness; the absence of a real solution only proves that there
    # is no solution if the variables are known to be real
    candidates = set(real)
    for lit in literals:
        atom = lit.arg if isinstance(lit, Not) else lit
        if isinstance(atom, Relational):
            candidates.update(_symbols(atom))
    polynomial: list[Boolean] = []
    unknown = False
    for lit in literals:
        p = to_polynomial(lit, candidates)
        if p is None:
            if (isinstance(lit, Contains) or _is_negated(lit, Contains)) and to_predicates(lit) is not None:
                continue
            if isinstance(lit, AppliedPredicate) or _is_negated(lit, AppliedPredicate):
                continue
            unknown = True
        elif not isinstance(p, BooleanTrue):
            polynomial.append(p)
    premise = facts.polynomial
    if not isinstance(premise, BooleanTrue):
        polynomial.append(premise)

    witness: Witness = {}
    if polynomial:
        formula = and_(*polynomial)
        gens = _sorted_symbols(_symbols(formula))
        if set(gens) <= real and _without_real_solution(formula, gens):
            return False
        points = sample_points(formula, gens)
        if not points:
            return False if set(gens) <= real else None
        integer_gens = [g for g in gens if g in integer]
        if integer_gens:
            exact = [p for p in points if all(p[g].is_integer for g in integer_gens)]
            if exact:
                witness = exact[0]
            else:
                found = _integer_witness(formula, points, integer, gens)
                if found is None or found is False or found is True:
                    return None if found is None else bool(found)
                witness = found
        else:
            witness = points[0]
    if unknown:
        return None
    return witness


@overload
def satisfiable(formula: BooleanLike, assumptions: Assumptions = None, domain: Optional[Set] = None,
                all_models: Literal[False] = False) -> Union[Model, bool, None]: ...


@overload
def satisfiable(formula: BooleanLike, assumptions: Assumptions = None, domain: Optional[Set] = None, *,
                all_models: Literal[True]) -> Optional[list[Model]]: ...


def satisfiable(formula: BooleanLike, assumptions: Assumptions = None, domain: Optional[Set] = None,
                all_models: bool = False) -> Union[Model, bool, None, list[Model]]:
    """Whether a formula can be true, the counterpart of Mathematica's
    ``SatisfiableQ`` (together with :func:`find_instance`).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of Boolean variables, relations between
        expressions, memberships in sets and predicates.
    assumptions : Boolean or list of Booleans, optional
        Assumptions which must hold as well. The global assumptions of
        :mod:`sympy_extras.assumptions.context` are used too.
    domain : Set, optional
        A named set (``Reals``, ``Integers``, ...) all the variables are
        assumed to belong to.
    all_models : bool
        Return the list of all models of the propositional abstraction
        which are consistent with the theory instead of the first one.

    Returns
    =======

    A model, a dict mapping the Boolean variables to ``True``/``False`` and
    the real variables of the polynomial relations to a witness point, if
    the formula is satisfiable; ``False`` if it is not; ``None`` if it
    could not be decided (for instance if a relation is not polynomial).
    With ``all_models=True`` a list of models is returned (empty if the
    formula is unsatisfiable, ``None`` if undecided).

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.sets import Integers, Reals
    >>> from sympy_extras.assumptions.sat import satisfiable
    >>> p, q, x, y = symbols('p q x y')
    >>> satisfiable(p & ~q)
    {p: True, q: False}
    >>> satisfiable(p & ~p)
    False
    >>> satisfiable((x**2 + y**2 < 1) & (x + y > 1))
    {x: 1/2, y: 2/3}
    >>> satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0))
    {p: False, x: 2}
    >>> satisfiable(Eq(x**2, 2), domain=Integers), satisfiable((x > 1) & (x < 3), domain=Integers)
    (False, {x: 2})
    >>> satisfiable(Eq(x**2, -1), domain=Reals), satisfiable(Eq(x**2, -1)) is None
    (False, True)
    """
    formula_ = to_boolean(formula)
    if formula_.has(Quantifier):
        from .resolve import resolve
        formula_ = resolve(formula_, assumptions=assumptions, domain=domain)
    facts = _facts(assumptions, domain, formula_.free_symbols)
    formula_ = normalize(formula_)
    abstract, atoms = _abstract(formula_)
    if isinstance(abstract, BooleanFalse):
        return [] if all_models else False
    if isinstance(abstract, BooleanTrue):
        return [{}] if all_models else {}
    from .sat_sympy import propositional_models_of_core
    results: list[Model] = []
    unknown = False
    for model in propositional_models_of_core(abstract):
        literals = _literals(model, atoms)
        result = _theory_check(literals, facts)
        if result is None:
            unknown = True
            continue
        if result is False:
            continue
        full: Model = {s: v for s, v in model.items() if s not in atoms}
        if isinstance(result, dict):
            for key, val in result.items():
                full[key] = val
        if not all_models:
            return full
        results.append(full)
    if all_models:
        return None if (unknown and not results) else results
    return None if unknown else False


def tautology(formula: BooleanLike, assumptions: Assumptions = None, domain: Optional[Set] = None
              ) -> Optional[bool]:
    """Whether a formula is always true, the counterpart of Mathematica's
    ``TautologyQ``: ``True``, ``False`` or ``None`` if undecided.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.sets import Reals
    >>> from sympy_extras.assumptions.sat import tautology
    >>> p, q, x, y = symbols('p q x y')
    >>> tautology(p | ~p), tautology(p | q)
    (True, False)
    >>> tautology(x**2 + y**2 >= 2*x*y, domain=Reals), tautology(x**2 > 0, domain=Reals)
    (True, False)
    """
    result = satisfiable(not_(formula), assumptions, domain)
    if result is None:
        return None
    return result is False


def find_instance(formula: BooleanLike, variables: Union[Symbol, Sequence[Symbol]], domain: Optional[Set] = Reals,
                  assumptions: Assumptions = None, count: int = 1) -> Optional[list[Witness]]:
    """Values of the variables satisfying a formula, the counterpart of
    Mathematica's ``FindInstance[expr, vars, dom, n]``.

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of relations, memberships and predicates.
    variables : Symbol or list of Symbols
        The variables to find values for.
    domain : Set
        ``Reals`` (default) or ``Integers``: the set the variables belong
        to (``None``: no domain, a model of :func:`satisfiable`).
    assumptions : Boolean or list of Booleans, optional
        Assumptions which must hold as well.
    count : int
        The number of instances wanted.

    Returns
    =======

    A list of dicts mapping the variables to exact values, empty if there
    is no instance, or ``None`` if it could not be decided. Over the reals
    the instances are the sample points of the cells of a cylindrical
    algebraic decomposition satisfying the formula, so ``count`` large
    enough returns one point per cell.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.sets import Integers
    >>> from sympy_extras.assumptions.sat import find_instance
    >>> x, y = symbols('x y')
    >>> find_instance((x**2 + y**2 < 1) & (x > y), [x, y])
    [{x: 0, y: -1/2}]
    >>> find_instance(Eq(x**2, 4), [x], count=2), find_instance(Eq(x**2, 2), [x], Integers)
    ([{x: -2}, {x: 2}], [])
    >>> find_instance((x**2 > 5) & (x < 0), [x], Integers), find_instance(x**2 < 0, [x])
    ([{x: -3}], [])
    """
    formula_ = to_boolean(formula)
    variables_: list[Symbol] = []
    for v in ([variables] if isinstance(variables, Symbol) else variables):
        if not isinstance(v, Symbol):
            raise TypeError("expected a symbol, got %s" % (v,))
        variables_.append(v)
    if domain is not None and not isinstance(domain, Set):
        raise TypeError("domain must be a set, got %s" % (domain,))
    if formula_.has(Quantifier):
        from .resolve import resolve
        formula_ = resolve(formula_, assumptions=assumptions, domain=Reals)
    facts = _facts(assumptions, None, set(variables_))
    formula_ = normalize(and_(formula_, facts.formula))
    if isinstance(domain, RealsClass):
        real = facts.real | set(variables_)
        poly = to_polynomial(formula_, real)
        if poly is not None and poly.free_symbols <= set(variables_):
            if isinstance(poly, BooleanTrue):
                return [{v: Integer(0) for v in variables_}]
            if isinstance(poly, BooleanFalse):
                return []
            from sympy_extras.polys.cad.qe import sample_points
            points = sample_points(poly, variables_)
            return points[:count]
    results: list[Witness] = []
    models = satisfiable(formula_, domain=domain, all_models=True)
    if models is None:
        return None
    for model in models:
        assignment: dict[Basic, Basic] = {}
        for s, value in model.items():
            assignment[s] = (true if value else false) if isinstance(value, bool) else value
        instance: Witness = {}
        for v in variables_:
            found = assignment.get(v)
            instance[v] = found if isinstance(found, Expr) else Integer(0)
        for key, val in instance.items():
            assignment[key] = val
        # the model was checked against the theory; only an evaluation to
        # False rejects it
        if isinstance(formula_.subs(list(assignment.items())), BooleanFalse):
            continue
        if instance not in results:
            results.append(instance)
        if len(results) >= count:
            break
    return results
