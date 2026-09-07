"""Satisfiability of formulas mixing Boolean variables, polynomial relations
and memberships.

The Boolean structure of a formula is handled by SymPy's SAT solver
(:func:`sympy.logic.inference.satisfiable`) on a propositional abstraction
in which every relation or membership is replaced by a Boolean variable.
Each candidate model is then checked against the theory: the predicates
implied by the chosen literals must be consistent (checked with the known
facts of :mod:`sympy.assumptions`) and the polynomial relations between
real variables must have a common real solution, found by cylindrical
algebraic decomposition, which also provides a witness. This is the lazy
approach of SMT solvers.

As in Mathematica's ``Reduce`` and ``FindInstance``, inequalities are
statements about real numbers: a variable appearing in an inequality of the
formula is taken to be real, also when the inequality comes from the
negation of another one (SymPy rewrites ``Not(x >= 0)`` as ``x < 0``).
Equations and ``Ne`` do not make their variables real, so a real solution
of an equation is a witness but the absence of real solutions is only
conclusive when the variables are known to be real.
"""
from __future__ import annotations

from itertools import product
from typing import Optional, Sequence, Union

from sympy.assumptions.assume import AppliedPredicate
from sympy.core.numbers import Integer
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, And, Not, true, false
from sympy.logic.inference import satisfiable as _sympy_satisfiable
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Reals
from sympy.sets.sets import Interval, FiniteSet, Set, Union as SetUnion

from sympy_extras._typing import Truth, as_boolean, as_expr, as_symbol, free_symbols, sorted_symbols

from sympy_extras.polys.cad import (sample_points as _cad_sample_points,
    solution_set as _cad_solution_set)
from sympy_extras.polys.cad.samplepoints import compare_real, _floor_scaled

from typing import Literal, overload

from .ask import Assumptions, _facts
from .facts import Facts
from .facts import (normalize, to_predicates, to_polynomial,
    predicates_consistent, _consequences, _is_atom, _REAL_SETS)
from .quantifiers import Quantifier

__all__ = ['satisfiable', 'tautology', 'find_instance']


#: a model: Boolean variables to truth values, real variables to values
Model = dict[Basic, Union[bool, Expr]]
#: a witness point
Witness = dict[Symbol, Expr]


def _abstract(formula: Boolean) -> tuple[Boolean, dict[Dummy, Boolean]]:
    """Replace the non-propositional atoms of a normalized formula by
    Boolean dummies. Returns the abstraction and the map from dummies to
    atoms."""
    atoms: dict[Dummy, Boolean] = {}
    mapping: dict[Basic, Dummy] = {}
    for atom in formula.atoms(Relational, Contains, AppliedPredicate):
        d = Dummy('p%d' % len(atoms))
        atoms[d] = atom
        mapping[atom] = d
    return formula.xreplace(mapping), atoms


def _literals(model: dict[Basic, bool], atoms: dict[Dummy, Boolean]) -> list[Boolean]:
    """The theory literals selected by a propositional model."""
    literals: list[Boolean] = []
    for symbol, value in model.items():
        if not isinstance(symbol, Dummy):
            continue
        atom = atoms.get(symbol)
        if atom is None:
            continue
        literals.append(atom if value else Not(atom))
    return literals


def _integer_in(sets: Set) -> Optional[Expr]:
    """An integer in a union of intervals and points with real algebraic
    endpoints, or ``None`` if there is none."""
    parts = sets.args if isinstance(sets, SetUnion) else [sets]
    for part in parts:
        if isinstance(part, FiniteSet):
            for v in part.args:
                if v.is_integer:
                    return as_expr(v)
            continue
        if not isinstance(part, Interval):
            continue
        a, b = part.start, part.end
        if a is S.NegativeInfinity:
            if b is S.Infinity:
                return S.Zero
            c = Integer(_floor_scaled(b, 0))
            if part.right_open and c == b:
                c -= 1
            return c
        c = Integer(_floor_scaled(a, 0))
        if c != a or part.left_open:
            c += 1
        if b is S.Infinity or compare_real(c, b) < 0 or (c == b and not part.right_open):
            return c
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
    integer_gens = [g for g in gens if g in integer]
    if len(integer_gens) == 1:
        [n] = integer_gens
        others = [g for g in gens if g != n]
        quantifiers = [('exists', others)] if others else []
        try:
            sets = _cad_solution_set(formula, n, quantifiers)
        except NotImplementedError:
            sets = None
        if sets is not None:
            k = _integer_in(sets)
            if k is None:
                return False
            witness: Witness = {n: k}
            if others:
                points_ = _cad_sample_points(as_boolean(formula.subs(n, k)), others)
                if not points_:
                    return None
                witness.update(points_[0])
            return witness
    for point in points:
        choices: list[list[Expr]] = []
        for g in gens:
            v = point[g]
            if g in integer:
                lo, hi = Integer(_floor_scaled(v, 0)), Integer(_floor_scaled(v, 0)) + 1
                choices.append(sorted({lo, hi, lo - 1, hi + 1}))
            else:
                choices.append([v])
        for combo in product(*choices):
            candidate: Witness = dict(zip(gens, combo))
            value = formula.subs(list(candidate.items()))
            if value is true:
                return candidate
    return None


def _theory_check(literals: Sequence[Boolean], facts: Facts) -> Union[Witness, bool, None]:
    """Check the conjunction of theory literals for consistency.

    Returns a witness dict (possibly empty) if the literals are
    satisfiable, ``False`` if they are not and ``None`` if it cannot be
    decided.
    """
    predicates: list[Boolean] = []
    for lit in literals:
        p = to_predicates(lit)
        if p is not None:
            predicates.append(p)
        if _is_atom(lit):
            predicates.extend(_consequences(lit))
    predicates.append(facts.predicates)
    if not predicates_consistent(And(*predicates)):
        return False

    # variables which are real by an inequality literal or a membership
    real = set(facts.real)
    integer = set(facts.integer)
    for lit in literals:
        if isinstance(lit, Relational) and not isinstance(lit, (Eq, Ne)):
            real.update(s for s in lit.free_symbols if isinstance(s, Symbol))
        if isinstance(lit, Contains) and isinstance(lit.args[0], Symbol):
            if isinstance(lit.args[1], _REAL_SETS):
                real.add(lit.args[0])
            if not isinstance(lit.args[1], Reals):
                integer.add(lit.args[0])

    # a real solution of the polynomial atoms is a solution whatever the
    # domain of the variables, so every variable is taken as real to look
    # for a witness; the absence of a real solution only proves that there
    # is no solution if the variables are known to be real
    candidates = set(real)
    for lit in literals:
        atom = lit.args[0] if isinstance(lit, Not) else lit
        if isinstance(atom, Relational):
            candidates.update(s for s in atom.free_symbols if isinstance(s, Symbol))
    polynomial: list[Boolean] = []
    unknown = False
    for lit in literals:
        p = to_polynomial(lit, candidates)
        if p is None:
            if isinstance(lit, Contains) and to_predicates(lit) is not None:
                continue
            if isinstance(lit, Not) and isinstance(lit.args[0], Contains) and \
                    to_predicates(lit) is not None:
                continue
            if isinstance(lit, AppliedPredicate) or (
                    isinstance(lit, Not) and isinstance(lit.args[0], AppliedPredicate)):
                continue
            unknown = True
        elif p is not true:
            polynomial.append(p)
    premise = facts.polynomial
    if premise is not true:
        polynomial.append(premise)

    witness: Witness = {}
    if polynomial:
        formula = And(*polynomial)
        gens = sorted_symbols(free_symbols(formula))
        points = _cad_sample_points(formula, gens)
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
def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, all_models: Literal[False] = False
                ) -> Union[Model, bool, None]: ...


@overload
def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, *, all_models: Literal[True]
                ) -> Optional[list[Model]]: ...


def satisfiable(formula: Union[Boolean, bool], assumptions: Assumptions = None,
                domain: Optional[Set] = None, all_models: bool = False
                ) -> Union[Model, bool, None, list[Model]]:
    """Whether a formula can be true, the counterpart of Mathematica's
    ``SatisfiableQ`` (together with :func:`find_instance`).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of Boolean variables, relations between
        expressions, memberships in sets and predicates.
    assumptions : Boolean or list of Booleans, optional
        Assumptions which must hold as well. The global assumptions are
        used too.
    domain : Set, optional
        A named SymPy set all the variables are assumed to belong to.
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

    >>> from sympy import symbols, S, Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import satisfiable, element
    >>> p, q = symbols('p q')
    >>> satisfiable(p & ~q)
    {p: True, q: False}
    >>> satisfiable(p & ~p)
    False
    >>> satisfiable((x**2 + y**2 < 1) & (x + y > 1))
    {x: 1/2, y: 2/3}
    >>> satisfiable((x**2 + y**2 < 1) & (x + y > 2))
    False
    >>> satisfiable((p | (x > 1)) & (~p | (x < 0)) & (x > 0))
    {p: False, x: 2}
    >>> satisfiable(Eq(x**2, 2), domain=S.Integers)
    False
    >>> satisfiable((x > 1) & (x < 3), domain=S.Integers)
    {x: 2}
    >>> satisfiable(Eq(x**2, -1), domain=S.Reals)
    False
    >>> satisfiable(Eq(x**2, -1)) is None
    True
    """
    formula_ = as_boolean(formula)
    if formula_.has(Quantifier):
        from .resolve import resolve
        formula_ = resolve(formula_, assumptions=assumptions, domain=domain)
    facts = _facts(assumptions, domain, formula_.free_symbols)
    formula_ = normalize(formula_)
    abstract, atoms = _abstract(formula_)
    if abstract is false:
        return [] if all_models else False
    if abstract is true:
        return [{}] if all_models else {}
    models = _sympy_satisfiable(abstract, all_models=True)
    if models is False:
        return [] if all_models else False
    results: list[Model] = []
    unknown = False
    for model in models:
        if model is False:
            break
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


def tautology(formula: Union[Boolean, bool], assumptions: Assumptions = None,
              domain: Optional[Set] = None) -> Truth:
    """Whether a formula is always true, the counterpart of Mathematica's
    ``TautologyQ``: ``True``, ``False`` or ``None`` if undecided.

    Examples
    ========

    >>> from sympy import symbols, S
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import tautology
    >>> p, q = symbols('p q')
    >>> tautology(p | ~p)
    True
    >>> tautology(p | q)
    False
    >>> tautology(x**2 + y**2 >= 2*x*y, domain=S.Reals)
    True
    >>> tautology((x > 0) | (x <= 0), domain=S.Reals)
    True
    >>> tautology(x**2 > 0, domain=S.Reals)
    False
    """
    result = satisfiable(Not(sympify(formula)), assumptions, domain)
    if result is None:
        return None
    return result is False


def find_instance(formula: Union[Boolean, bool], variables: Union[Symbol, Sequence[Symbol]],
                  domain: Set = S.Reals, assumptions: Assumptions = None,
                  count: int = 1) -> Optional[list[Witness]]:
    """Values of the variables satisfying a formula, the counterpart of
    Mathematica's ``FindInstance[expr, vars, dom, n]``.

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of relations, memberships and predicates.
    variables : Symbol or list of Symbols
        The variables to find values for.
    domain : Set
        ``S.Reals`` (default) or ``S.Integers``: the set the variables
        belong to.
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

    >>> from sympy import S, Eq
    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import find_instance
    >>> find_instance((x**2 + y**2 < 1) & (x > y), [x, y])
    [{x: 0, y: -1/2}]
    >>> find_instance((x**2 + y**2 < 1) & (x > y), [x, y], count=3)
    [{x: 0, y: -1/2}, {x: CRootOf(2*x**2 - 1, 1), y: 0}, {x: 3/4, y: 0}]
    >>> find_instance(Eq(x**2, 2), [x])
    [{x: CRootOf(x**2 - 2, 0)}]
    >>> find_instance(Eq(x**2, 2), [x], S.Integers)
    []
    >>> find_instance((x**2 > 5) & (x < 0), [x], S.Integers)
    [{x: -3}]
    >>> find_instance(x**2 < 0, [x])
    []
    """
    formula_ = as_boolean(formula)
    variables = [variables] if isinstance(variables, Symbol) else [as_symbol(v) for v in variables]
    domain = sympify(domain)
    if formula_.has(Quantifier):
        from .resolve import resolve
        formula_ = resolve(formula_, assumptions=assumptions, domain=S.Reals)
    facts = _facts(assumptions, None, set(variables))
    formula_ = normalize(And(formula_, facts.formula))
    if isinstance(domain, Reals):
        real = facts.real | set(variables)
        poly = to_polynomial(formula_, real)
        if poly is not None and poly.free_symbols <= set(variables):
            if poly is true:
                return [{v: S.Zero for v in variables}]
            if poly is false:
                return []
            points = _cad_sample_points(poly, variables)
            return points[:count]
    results: list[Witness] = []
    models = satisfiable(formula_, domain=domain, all_models=True)
    if models is None:
        return None
    for model in models:
        assignment: dict[Basic, Basic] = {s: (true if v is True else false if v is False else v)
                                          for s, v in model.items()}
        instance: Witness = {}
        for v in variables:
            value = assignment.get(v, S.Zero)
            instance[v] = as_expr(value) if isinstance(value, Expr) else S.Zero
        for key, val in instance.items():
            assignment[key] = val
        # the model was checked against the theory; only an evaluation to
        # False rejects it
        if formula_.subs(list(assignment.items())) is false:
            continue
        if instance not in results:
            results.append(instance)
        if len(results) >= count:
            break
    return results
