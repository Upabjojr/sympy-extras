"""Quantifier elimination over the real numbers."""
from __future__ import annotations

from typing import Optional, Union

from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse, true, false
from sympy.sets.fancysets import Reals
from sympy.sets.sets import Set

from sympy_extras.polys.cad import quantifier_elimination

from .facts import normalize, to_polynomial
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
        Only ``S.Reals`` (the default, also for ``None``) is supported:
        every variable, free or bound, is a real number.
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

    Examples
    ========

    >>> from sympy import Eq
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
    """
    formula_ = as_boolean(formula)
    domain = S.Reals if domain is None else sympify(domain)
    if not isinstance(domain, Reals):
        raise NotImplementedError(
            "quantifier elimination is only implemented over the real numbers")
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
    result = quantifier_elimination(poly, prefix, free=free, method=method)
    if premise is not None and premise is not true and result not in (true, false):
        # drop the conditions which are implied by the assumptions
        from .refine import _refine_boolean
        result = _refine_boolean(result, facts)
    return result
