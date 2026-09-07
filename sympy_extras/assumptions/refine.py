"""Refining and simplifying expressions under assumptions."""
from __future__ import annotations

from typing import Callable, Optional, Union

from sympy.assumptions import refine as _sympy_refine
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Relational, Eq
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, sign, conjugate
from sympy.functions.elementary.integers import floor, ceiling
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Boolean, Not, true, false
from sympy.sets.contains import Contains
from sympy.sets.sets import Set
from sympy.simplify.simplify import simplify as _sympy_simplify

from .ask import Assumptions, _facts, _evaluate, _evaluate_atom
from sympy_extras._typing import as_boolean, as_expr

from .facts import Facts, normalize
from .quantifiers import Quantifier

__all__ = ['refine', 'simplify']


def _refine_boolean(formula: Boolean, facts: Facts) -> Boolean:
    """Replace the atoms of a Boolean formula which are decided by the
    facts."""
    formula = normalize(formula)
    value = _evaluate(formula, facts)
    if value is not None:
        return true if value else false
    decided: dict[Basic, Boolean] = {}
    for atom in formula.atoms(Relational, Contains, AppliedPredicate):
        value = _evaluate_atom(atom, facts)
        if value is not None:
            decided[atom] = true if value else false
    return formula.xreplace(decided) if decided else formula


def _refine_abs(e: Expr, facts: Facts) -> Expr:
    f = as_expr(e.args[0])
    if _evaluate(normalize(f >= 0), facts):
        return f
    if _evaluate(normalize(f <= 0), facts):
        return -f
    return e


def _refine_sign(e: Expr, facts: Facts) -> Expr:
    f = as_expr(e.args[0])
    if _evaluate(normalize(f > 0), facts):
        return S.One
    if _evaluate(normalize(f < 0), facts):
        return S.NegativeOne
    if _evaluate(normalize(Eq(f, 0)), facts):
        return S.Zero
    return e


def _refine_minmax(e: Expr, facts: Facts) -> Expr:
    args = [as_expr(a) for a in e.args]
    dominated: set[int] = set()
    for i, a in enumerate(args):
        for j, b in enumerate(args):
            if i == j or j in dominated:
                continue
            better = (b >= a) if isinstance(e, Max) else (b <= a)
            if _evaluate(normalize(better), facts):
                dominated.add(i)
                break
    if not dominated:
        return e
    remaining = [a for i, a in enumerate(args) if i not in dominated]
    return as_expr(e.func(*remaining))


def _refine_floor(e: Expr, facts: Facts) -> Expr:
    f = as_expr(e.args[0])
    if _evaluate(normalize(Contains(f, S.Integers)), facts):
        return f
    return e


def _refine_conjugate(e: Expr, facts: Facts) -> Expr:
    f = as_expr(e.args[0])
    if _evaluate(normalize(Contains(f, S.Reals)), facts):
        return f
    return e


def _refine_piecewise(e: Expr, facts: Facts) -> Expr:
    """The conditions are decided with the facts; the expression of a
    branch is refined with the facts and its own condition (and the
    negations of the conditions of the previous branches)."""
    pairs: list[tuple[Expr, Boolean]] = []
    excluded: list[Boolean] = []
    for branch in e.args:
        expr, cond = as_expr(branch.args[0]), as_boolean(branch.args[1])
        value = _evaluate(normalize(cond), facts)
        if value is False:
            excluded.append(cond)
            continue
        extra: list[Boolean] = [Not(c) for c in excluded] + ([cond] if value is None else [])
        try:
            local = Facts(facts.conjuncts + extra)
        except (ValueError, TypeError):
            local = facts
        expr = as_expr(_refine_expr(expr, local))
        excluded.append(cond)
        pairs.append((expr, true if value is True else cond))
        if value is True:
            break
    if not pairs:
        return S.NaN
    return Piecewise(*pairs)


def _refine_relational(e: Boolean, facts: Facts) -> Boolean:
    value = _evaluate(normalize(e), facts)
    if value is None:
        return e
    return true if value else false


def _expr_handler(handler: Callable[[Expr, Facts], Expr]) -> Callable[[Basic, Facts], Basic]:
    def wrapped(e: Basic, facts: Facts) -> Basic:
        return handler(as_expr(e), facts)
    return wrapped


def _boolean_handler(e: Basic, facts: Facts) -> Basic:
    return _refine_relational(as_boolean(e), facts)


_HANDLERS: list[tuple[Union[type, tuple[type, ...]], Callable[[Basic, Facts], Basic]]] = [
    (Relational, _boolean_handler),
    (Abs, _expr_handler(_refine_abs)),
    (sign, _expr_handler(_refine_sign)),
    ((Max, Min), _expr_handler(_refine_minmax)),
    ((floor, ceiling), _expr_handler(_refine_floor)),
    (conjugate, _expr_handler(_refine_conjugate)),
    (Piecewise, _expr_handler(_refine_piecewise)),
]


def _refine_pass(expr: Basic, facts: Facts) -> Basic:
    for cls, handler in _HANDLERS:
        expr = expr.replace(lambda e: isinstance(e, cls),
                            lambda e: handler(e, facts))
    return expr


def _refine_expr(expr: Basic, facts: Facts) -> Basic:
    """Refine a non-Boolean expression with the given facts."""
    try:
        expr = _sympy_refine(expr, facts.predicates)
    except (ValueError, TypeError, NotImplementedError):
        pass
    for _ in range(4):
        new = _refine_pass(expr, facts)
        if new == expr:
            break
        expr = new
    return expr


def refine(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
           domain: Optional[Set] = None) -> Basic:
    """Refine an expression using assumptions, the counterpart of
    Mathematica's ``Refine[expr, assum]``.

    Parameters
    ==========

    expr : Expr or Boolean
        The expression to refine.
    assumptions : Boolean or list of Booleans, optional
        Assumptions written as relations, memberships in sets or predicates,
        see :mod:`sympy_extras.assumptions`. The global assumptions are
        used as well.
    domain : Set, optional
        A named SymPy set all the variables are assumed to belong to.

    :func:`sympy.refine` is applied with the predicates implied by the
    assumptions, then relations, absolute values, signs, maxima and
    minima, floors and ceilings and the conditions of ``Piecewise``
    expressions are decided with :func:`~sympy_extras.assumptions.ask`,
    which uses cylindrical algebraic decomposition for polynomial
    inequalities.

    Examples
    ========

    >>> from sympy import Abs, sqrt, sign, Max, floor, Piecewise, S, re, im
    >>> from sympy.abc import x, y, n
    >>> from sympy_extras.assumptions import refine, element
    >>> refine(Abs(x - 1) + sqrt(x**2), x > 2)
    2*x - 1
    >>> refine(sign(x*y), (x > 0) & (y < 0))
    -1
    >>> refine(Max(x, x**2), (x > 0) & (x < 1))
    x
    >>> refine(floor(n + 1), element(n, S.Integers))
    n + 1
    >>> refine(Piecewise((1, x**2 + y**2 < 1), (2, x > 0), (3, True)), (x > 1) & (y > 0))
    2
    >>> refine(re(x) + im(y), element(x, S.Reals) & element(y, S.Reals))
    x
    >>> refine((x > 0) & (y > 0), x > 1)
    y > 0
    >>> refine(x*y > 0, (x > 0) & (y > 0))
    True
    """
    if expr is True or expr is False:
        return true if expr else false
    expr_ = sympify(expr)
    facts = _facts(assumptions, domain, expr_.free_symbols)
    if isinstance(expr_, Boolean):
        formula = expr_
        if formula.has(Quantifier):
            from .resolve import resolve
            formula = resolve(formula, assumptions=assumptions, domain=domain)
            if isinstance(formula, (type(true), type(false))):
                return formula
        return _refine_boolean(formula, facts)
    return _refine_expr(expr_, facts)


def simplify(expr: Union[Expr, Boolean, bool], assumptions: Assumptions = None,
             domain: Optional[Set] = None, **kwargs: object) -> Basic:
    """Simplify an expression using assumptions, the counterpart of
    Mathematica's ``Simplify[expr, assum]``.

    The expression is refined with :func:`refine`, simplified with
    :func:`sympy.simplify` (which receives the extra keyword arguments)
    and refined again.

    Examples
    ========

    >>> from sympy import Abs, sqrt, cos, sin
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions import simplify
    >>> simplify(sqrt(x**2) + Abs(x)*sin(x)**2 + Abs(x)*cos(x)**2, x < 0)
    -2*x
    >>> simplify((x**2 - 1)/(x - 1), x > 1)
    x + 1
    """
    refined = refine(expr, assumptions, domain)
    simplified = sympify(_sympy_simplify(refined, **kwargs))
    if isinstance(simplified, (Expr, Boolean)):
        return refine(simplified, assumptions, domain)
    return simplified
