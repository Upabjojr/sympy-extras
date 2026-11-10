"""Quantifier elimination over the real numbers.

The formulas are Booleans of the core (:mod:`sympy_extras.core.logic`)
with the quantifiers of :mod:`sympy_extras.assumptions.quantifiers`.

The module works on the objects of the core and imports no SymPy:
:mod:`sympy_extras.assumptions.resolve_sympy` is its face on SymPy
objects. The eliminations themselves are not ported and go through SymPy
at run time (the face holds them, the functions ending in ``_of_core``):
the cylindrical algebraic decomposition
(:func:`sympy_extras.polys.cad.quantifier_elimination`), virtual
substitution (:mod:`sympy_extras.polys.virtual_substitution`), Cooper's
algorithm (:mod:`sympy_extras.solvers.integers`) and comprehensive
Gröbner systems (:mod:`sympy_extras.polys.comprehensive`), all written on
SymPy's polynomials. The prenex form, the assumptions and the
simplification of the result under them are native.
"""
from __future__ import annotations

from typing import Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.logic import (And, Boolean, BooleanAtom, BooleanFalse, BooleanLike, BooleanTrue, and_, false,
    to_boolean, true)
from sympy_extras.core.mod import Mod
from sympy_extras.core.relational import Relational
from sympy_extras.core.symbol import Symbol
from sympy_extras.sets.fancysets import Complexes, ComplexesClass, Integers, IntegersClass
from sympy_extras.sets.sets import Contains, Reals, RealsClass, Set
from sympy_extras.settings import settings
from sympy_extras._timeout import attempt

from .ask import Assumptions, _facts
from .facts import Facts, normalize, to_polynomial
from .quantifiers import Prefix, prenex

__all__ = ['resolve']


def _symbols(e: Basic) -> set[Symbol]:
    return {s for s in e.free_symbols if isinstance(s, Symbol)}


def _is_membership_in(c: Boolean, s: Set) -> bool:
    return isinstance(c, Contains) and c.args[1] == s


def resolve(formula: BooleanLike, domain: Optional[Set] = Reals, assumptions: Assumptions = None,
            method: Optional[str] = None) -> Boolean:
    """Eliminate the quantifiers of a formula over the real numbers, the
    counterpart of Mathematica's ``Resolve[expr, Reals]`` (and of
    ``Reduce`` for quantifier-free formulas).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of polynomial relations with rational
        coefficients, memberships in ``Reals`` and sign predicates, with
        quantifiers :class:`~sympy_extras.assumptions.quantifiers.ForAll`
        and :class:`~sympy_extras.assumptions.quantifiers.Exists` anywhere
        (the formula is put in prenex form first).
    domain : Set
        ``Reals`` (the default, also for ``None``): every variable, free
        or bound, is a real number. ``Integers``: every variable is an
        integer, and the formula must be one of Presburger arithmetic (a
        Boolean combination of linear relations with integer coefficients
        and divisibilities ``Eq(Mod(e, k), 0)``), whose quantifiers are
        eliminated by Cooper's algorithm. ``Complexes``: every variable
        is a complex number, the formula is a Boolean combination of
        polynomial equations and inequations, and the quantifiers are
        eliminated with comprehensive Gröbner systems (see
        :mod:`sympy_extras.polys.comprehensive`).
    assumptions : Boolean or list of Booleans, optional
        Polynomial assumptions on the free variables. The result is
        equivalent to the input under the assumptions and does not repeat
        them.
    method : ``'mccallum'``, ``'hong'`` or None
        The projection operator of the cylindrical algebraic decomposition.

    Returns
    =======

    ``true`` or ``false`` for a closed formula, otherwise a
    quantifier-free formula in the free variables equivalent to the input
    over the reals. With one free variable the result describes a union of
    intervals with exact endpoints; with more free variables it is written
    with sign conditions on the projection factors of the decomposition,
    augmented with the derivatives of the factors when those are not
    enough (Hong's solution formula construction, see
    :func:`~sympy_extras.polys.cad.quantifier_elimination`).
    Quantified variables in which the formula is linear or quadratic are
    eliminated first by virtual substitution (Loos–Weispfenning,
    Weispfenning), which needs no decomposition; with more than two free
    variables the result of that step is returned as it is.

    Examples
    ========

    >>> from sympy_extras.core import Eq, symbols
    >>> from sympy_extras.sets import Complexes, Integers
    >>> from sympy_extras.assumptions.resolve import resolve
    >>> from sympy_extras.assumptions.quantifiers import ForAll, Exists
    >>> a, b, c, x, y = symbols('a b c x y')
    >>> resolve(ForAll(x, x**2 + b*x + c > 0)), resolve(Exists(y, Eq(x**2 + y**2, 1)))
    (b**2 < 4*c, (x >= -1) & (x <= 1))
    >>> resolve(ForAll(x, Exists(y, y > x))), resolve(Exists(y, ForAll(x, y > x)))
    (True, False)
    >>> resolve(Exists(x, Eq(2*x, y)), domain=Integers), resolve(Exists(x, Eq(a*x, 1)), domain=Complexes)
    (Eq(Mod(y, 2), 0), Ne(a, 0))
    """
    formula_ = to_boolean(formula)
    domain_ = Reals if domain is None else domain
    if isinstance(domain_, IntegersClass):
        return _resolve_integers(formula_, assumptions)
    if isinstance(domain_, ComplexesClass):
        return _resolve_complexes(formula_, assumptions)
    if not isinstance(domain_, RealsClass):
        raise NotImplementedError(
            "quantifier elimination is only implemented over the real numbers, "
            "the complex numbers and, for linear formulas, the integers")
    from .resolve_sympy import quantifier_elimination_of_core, virtual_substitution_elimination_of_core, \
        with_sides_of_core
    prefix, matrix = prenex(formula_)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, Reals, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    real = facts.real | set(bound) | _symbols(matrix)
    poly = to_polynomial(matrix, real)
    if poly is None:
        raise ValueError(
            "the formula is not a Boolean combination of polynomial relations "
            "with rational coefficients: %s" % (matrix,))
    premise = to_polynomial(facts.formula, real)
    if premise is not None and not isinstance(premise, BooleanTrue):
        poly = and_(premise, poly)
    free = sorted(_symbols(poly) - set(bound), key=lambda s: s.name)
    if isinstance(poly, BooleanAtom):
        return poly
    if not prefix and not free:
        if isinstance(poly, Relational):
            raise TypeError("cannot determine the truth value of %s" % (poly,))
        return true
    # variables occurring linearly are eliminated by virtual substitution,
    # the others by cylindrical algebraic decomposition
    poly, remaining = virtual_substitution_elimination_of_core(poly, prefix)
    if isinstance(poly, BooleanAtom):
        return poly
    eliminated = len(remaining) < len(prefix)
    if not remaining and len(free) > 2:
        result = poly
    else:
        # the decomposition also simplifies a quantifier-free formula
        result = quantifier_elimination_of_core(poly, remaining, free, method)
    if eliminated:
        result = with_sides_of_core(result)
    return _refine_boolean_(result, facts, premise)


def _resolve_integers(formula: Boolean, assumptions: Assumptions) -> Boolean:
    """Quantifier elimination over the integers (Presburger arithmetic)."""
    from .resolve_sympy import is_presburger_of_core, presburger_quantifier_elimination_of_core
    prefix, matrix = prenex(formula)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, Integers, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    integers = facts.integer | set(bound) | _symbols(matrix)
    premise = and_(*[c for c in facts.conjuncts if _symbols(c) - set(bound)
                     and not _is_membership_in(c, Integers)])
    if not isinstance(premise, BooleanTrue):
        matrix = and_(premise, matrix)
    if not is_presburger_of_core(matrix, integers):
        # Z is a subset of R: a universal statement true over the reals is
        # true over the integers, an existential one false over the reals
        # is false over them. Neither reverses, so the relaxation can only
        # confirm; it decides most of the nonlinear goals of the Lean and
        # Rocq test suites at once (sympy-extras#46).
        relaxed = _real_relaxation(formula, prefix, assumptions)
        if relaxed is not None:
            return relaxed
        raise ValueError(
            "over the integers the formula must be a Boolean combination of linear "
            "relations with integer coefficients and divisibilities: %s" % (matrix,))
    result = presburger_quantifier_elimination_of_core(matrix, prefix)
    if not isinstance(premise, BooleanTrue) and not isinstance(result, BooleanAtom):
        from .refine import _refine_boolean
        result = _refine_boolean(result, facts)
    return result


def _real_relaxation(formula: Boolean, prefix: Prefix, assumptions: Assumptions) -> Optional[Boolean]:
    """The truth value of a closed formula with a prefix of one kind that
    transfers from the reals to the integers, or ``None``."""
    kinds = {kind for kind, _ in prefix}
    if len(kinds) != 1 or formula.has(Mod) or _symbols(formula) - {v for _, v in prefix}:
        return None
    try:
        over_reals = attempt(lambda: resolve(formula, assumptions=assumptions, domain=Reals), settings.timeout)
    except (NotImplementedError, ValueError, TypeError):
        return None
    if isinstance(over_reals, BooleanTrue) and kinds == {'forall'}:
        return true
    if isinstance(over_reals, BooleanFalse) and kinds == {'exists'}:
        return false
    return None


def _resolve_complexes(formula: Boolean, assumptions: Assumptions) -> Boolean:
    """Quantifier elimination over the complex numbers (equations and
    inequations only)."""
    from .resolve_sympy import complex_quantifier_elimination_of_core, reduce_complex_of_core
    prefix, matrix = prenex(formula)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, Complexes, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    conditions = [c for c in facts.conjuncts if _symbols(c) - set(bound) and not _is_membership_in(c, Complexes)]
    if not conditions:
        return complex_quantifier_elimination_of_core(matrix, prefix)
    premise = reduce_complex_of_core(and_(*conditions))
    result = complex_quantifier_elimination_of_core(and_(premise, matrix), prefix)
    # the result does not repeat the assumptions
    known: dict[Basic, Basic] = {c: true for c in (premise.args if isinstance(premise, And) else (premise,))}
    return reduce_complex_of_core(to_boolean(result._xreplace_evaluated(known)))


def _refine_boolean_(result: Boolean, facts: Facts, premise: Optional[Boolean]) -> Boolean:
    """Drop the conditions of the result which are implied by the
    assumptions."""
    if premise is not None and not isinstance(premise, BooleanTrue) and not isinstance(result, BooleanAtom):
        from .refine import _refine_boolean
        from .resolve_sympy import with_sides_of_core
        result = with_sides_of_core(_refine_boolean(result, facts))
    return result
