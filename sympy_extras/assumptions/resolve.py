"""Quantifier elimination over the real numbers."""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse, true, false
from sympy.sets.fancysets import Reals, Integers, Complexes
from sympy.sets.sets import Set

from sympy_extras.polys.cad import quantifier_elimination
from sympy_extras.polys.virtual_substitution import linear_quantifier_elimination, with_sides

from .facts import Facts, normalize, to_polynomial
from sympy_extras._typing import as_boolean, free_symbols, sorted_symbols

from .quantifiers import prenex
from .ask import Assumptions, _facts

__all__ = ['resolve']


def resolve(formula: Union[Boolean, bool], domain: Optional[Set] = S.Reals,
            assumptions: 'Assumptions' = None, method: Optional[str] = None) -> Boolean:
    """Eliminate the quantifiers of a formula over the real numbers, the
    counterpart of Mathematica's ``Resolve[expr, Reals]`` (and of
    ``Reduce`` for quantifier-free formulas).

    Parameters
    ==========

    formula : Boolean
        A Boolean combination of polynomial relations with rational
        coefficients, memberships in ``S.Reals`` and sign predicates, with
        quantifiers :class:`~sympy_extras.assumptions.ForAll` and
        :class:`~sympy_extras.assumptions.Exists` anywhere (the formula is
        put in prenex form first).
    domain : Set
        ``S.Reals`` (the default, also for ``None``): every variable, free
        or bound, is a real number. ``S.Integers``: every variable is an
        integer, and the formula must be one of Presburger arithmetic (a
        Boolean combination of linear relations with integer coefficients
        and divisibilities ``Eq(Mod(e, k), 0)``), whose quantifiers are
        eliminated by Cooper's algorithm. ``S.Complexes``: every variable
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

    ``S.true`` or ``S.false`` for a closed formula, otherwise a
    quantifier-free formula in the free variables equivalent to the input
    over the reals. With one free variable the result describes a union of
    intervals with exact endpoints; with more free variables it is written
    with sign conditions on the projection factors of the decomposition,
    and ``NotImplementedError`` is raised when those are not enough.
    Quantified variables in which the formula is linear are eliminated
    first by virtual substitution (Loos–Weispfenning), which needs no
    decomposition; with more than two free variables the result of that
    step is returned as it is.

    Examples
    ========

    >>> from sympy import S, Eq
    >>> from sympy.abc import a, b, c, x, y
    >>> from sympy_extras.assumptions import resolve, ForAll, Exists
    >>> resolve(ForAll(x, x**2 + b*x + c > 0))
    b**2 - 4*c < 0
    >>> resolve(Exists(x, Eq(a*x**2 + b*x + c, 0) & (a > 0)))
    (a > 0) & (4*a*c - b**2 <= 0)
    >>> resolve(ForAll(x, Exists(y, y > x)))
    True
    >>> resolve(Exists(y, ForAll(x, y > x)))
    False
    >>> resolve(Exists(y, Eq(x**2 + y**2, 1)))
    (x >= -1) & (x <= 1)
    >>> resolve(~ForAll(x, x**2 > 0))
    True
    >>> resolve(x**2 > 2)
    (x > CRootOf(x**2 - 2, 1)) | (x < CRootOf(x**2 - 2, 0))
    >>> resolve(Exists(x, Eq(2*x, y)), domain=S.Integers)
    Eq(Mod(y, 2), 0)
    >>> resolve(Exists(x, Eq(x**2 + 1, 0)), domain=S.Complexes)
    True
    >>> resolve(Exists(x, Eq(a*x, 1)), domain=S.Complexes)
    Ne(a, 0)
    >>> resolve(ForAll(x, Exists(y, Eq(3*y, x) | Eq(3*y, x + 1) | Eq(3*y, x + 2))), domain=S.Integers)
    True
    """
    formula_ = as_boolean(formula)
    domain = S.Reals if domain is None else sympify(domain)
    if isinstance(domain, Integers):
        return _resolve_integers(formula_, assumptions)
    if isinstance(domain, Complexes):
        return _resolve_complexes(formula_, assumptions)
    if not isinstance(domain, Reals):
        raise NotImplementedError(
            "quantifier elimination is only implemented over the real numbers, "
            "the complex numbers and, for linear formulas, the integers")
    prefix, matrix = prenex(formula_)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, S.Reals, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    real = facts.real | set(bound) | free_symbols(matrix)
    poly = to_polynomial(matrix, real)
    if poly is None:
        raise ValueError(
            "the formula is not a Boolean combination of polynomial relations "
            "with rational coefficients: %s" % (matrix,))
    premise = to_polynomial(facts.formula, real)
    if premise is not None and premise is not true:
        from sympy.logic.boolalg import And
        poly = And(premise, poly)
    free = sorted_symbols(free_symbols(poly) - set(bound))
    if isinstance(poly, (BooleanTrue, BooleanFalse)):
        return poly
    if not prefix and not free:
        return true if bool(poly) else false
    # variables occurring linearly are eliminated by virtual substitution,
    # the others by cylindrical algebraic decomposition
    poly, remaining = linear_quantifier_elimination(poly, prefix)
    if isinstance(poly, (BooleanTrue, BooleanFalse)):
        return poly
    eliminated = len(remaining) < len(prefix)
    if not remaining and len(free) > 2:
        result = poly
    else:
        # the decomposition also simplifies a quantifier-free formula
        result = quantifier_elimination(poly, remaining, free=free, method=method)
    if eliminated:
        result = with_sides(result)
    return _refine_boolean_(result, facts, premise)


def _resolve_integers(formula: Boolean, assumptions: 'Assumptions') -> Boolean:
    """Quantifier elimination over the integers (Presburger arithmetic)."""
    from sympy_extras.solvers.integers import is_presburger, presburger_quantifier_elimination
    prefix, matrix = prenex(formula)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, S.Integers, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    integers = facts.integer | set(bound) | free_symbols(matrix)
    from sympy.logic.boolalg import And
    from sympy.sets.contains import Contains
    premise = And(*[c for c in facts.conjuncts if free_symbols(c) - set(bound)
                    and not (isinstance(c, Contains) and c.args[1] == S.Integers)])
    if premise is not true:
        matrix = And(premise, matrix)
    if not is_presburger(matrix, integers):
        raise ValueError(
            "over the integers the formula must be a Boolean combination of linear "
            "relations with integer coefficients and divisibilities: %s" % (matrix,))
    result = presburger_quantifier_elimination(matrix, prefix)
    if premise is not true and result not in (true, false):
        from .refine import _refine_boolean
        result = _refine_boolean(result, facts)
    return result


def _resolve_complexes(formula: Boolean, assumptions: 'Assumptions') -> Boolean:
    """Quantifier elimination over the complex numbers (equations and
    inequations only)."""
    from sympy.logic.boolalg import And
    from sympy.sets.contains import Contains
    from sympy_extras.polys.comprehensive import complex_quantifier_elimination, reduce_complex
    prefix, matrix = prenex(formula)
    bound = [v for _, v in prefix]
    facts = _facts(assumptions, S.Complexes, matrix.free_symbols | set(bound))
    matrix = normalize(matrix)
    conditions = [c for c in facts.conjuncts if free_symbols(c) - set(bound)
                  and not (isinstance(c, Contains) and c.args[1] == S.Complexes)]
    if not conditions:
        return complex_quantifier_elimination(matrix, prefix)
    premise = reduce_complex(And(*conditions))
    result = complex_quantifier_elimination(And(premise, matrix), prefix)
    # the result does not repeat the assumptions
    known = {c: true for c in (premise.args if isinstance(premise, And) else [premise])}
    return reduce_complex(as_boolean(result.xreplace(known)))


def _refine_boolean_(result: Boolean, facts: Facts, premise: Optional[Boolean]) -> Boolean:
    """Drop the conditions of the result which are implied by the
    assumptions."""
    if premise is not None and premise is not true and result not in (true, false):
        from .refine import _refine_boolean
        result = with_sides(_refine_boolean(result, facts))
    return result
