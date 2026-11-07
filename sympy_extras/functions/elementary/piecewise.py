"""``Piecewise`` on the typed core: the unevaluated node (its pairs kept
as given) and the evaluating ``piecewise``.

Ported from SymPy's ``Piecewise`` and ``ExprCondPair``: ``piecewise``
drops the false conditions and the pairs after a true one, returns the
expression of a single true pair (``nan`` when no pair is left), makes
the relations canonical with their leading coefficient divided out
(``-2*x + 1 < 0`` is ``x > 1/2``), drops repeated conditions and the
redundant parts of a conjunction, collapses a nested ``Piecewise`` with
the same condition, and merges successive pairs with the same expression
(their conditions joined by ``Or``). Substitution re-evaluates. The
derivative and the conjugate apply to the expressions; the facts are
those all the expressions agree on. Not ported: ``piecewise_fold``,
``piecewise_simplify``, the integration and the interval methods.

Examples
========

>>> from sympy_extras.core import Symbol, Integer, lt, gt, true, false
>>> from sympy_extras.functions.elementary.piecewise import Piecewise, piecewise
>>> x = Symbol('x')
>>> f = piecewise((x**2, lt(x, 0)), (x, true))
>>> f, f.subs(x, -2), f.subs(x, 3), f.diff(x)
(Piecewise((x**2, x < 0), (x, True)), 4, 3, Piecewise((2*x, x < 0), (1, True)))
>>> piecewise((1, lt(-x, -1)), (4, lt(-x, -1)), (3, false), (2, true), (5, lt(x, 1)))
Piecewise((1, x > 1), (2, True))
>>> piecewise((x, true)), piecewise((1, lt(x, 0)), (1, gt(x, 2)), (0, true)), Piecewise((x, true))
(x, Piecewise((1, (x > 2) | (x < 0)), (0, True)), Piecewise((x, True)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py (Piecewise, ExprCondPair,
# _piecewise_collapse_arguments), sympy/core/relational.py (_canonical_coeff),
# sympy/logic/boolalg.py (distribute_and_over_or) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating class with an evaluating function, strict
# typing; the folding, simplification and integration not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Final, Optional, Union

from sympy_extras.core.add import sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import (And, Boolean, BooleanFalse, BooleanLike, BooleanTrue, Or, and_, or_,
                                     false, to_boolean, true)
from sympy_extras.core.mul import div
from sympy_extras.core.numbers import Rational, nan
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.relational import Ge, Gt, Le, Lt, Relational, _FUNCTIONS, _REVERSED
from sympy_extras.core.sorting import ClassKey

__all__ = ['ExprCondPair', 'Piecewise', 'piecewise']

PairLike = Union['ExprCondPair', tuple[ExprLike, BooleanLike]]

#: the facts SymPy's Piecewise gives when all its expressions agree
_AGREEING = ('finite', 'complex', 'even', 'imaginary', 'integer', 'irrational', 'negative', 'nonnegative',
             'nonpositive', 'nonzero', 'odd', 'polar', 'positive', 'extended_real', 'extended_positive',
             'extended_negative', 'extended_nonzero', 'extended_nonpositive', 'extended_nonnegative', 'real', 'zero',
             'commutative')


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:ExprCondPair
class ExprCondPair(Basic):
    """An expression with the condition under which it applies."""

    __slots__ = ('_expr', '_cond')

    def __init__(self, expr: ExprLike, cond: BooleanLike) -> None:
        self._expr: Final[Expr] = to_expr(expr)
        self._cond: Final[Boolean] = to_boolean(cond)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._expr, self._cond)

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def cond(self) -> Boolean:
        return self._cond

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 1, 'ExprCondPair')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        expr, cond = args
        if not isinstance(expr, Expr) or not isinstance(cond, Boolean):
            raise TypeError("ExprCondPair takes an expression and a condition")
        return ExprCondPair(expr, cond)

    def _sympystr(self, printer: StrPrinter) -> str:
        return '(%s, %s)' % (printer.doprint(self._expr), printer.doprint(self._cond))


def _pair(p: PairLike) -> ExprCondPair:
    return p if isinstance(p, ExprCondPair) else ExprCondPair(p[0], p[1])


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise
class Piecewise(Expr):
    """The unevaluated piecewise expression: the first pair whose condition
    holds gives the value (``piecewise`` evaluates)."""

    __slots__ = ('_args',)

    def __init__(self, *pairs: PairLike) -> None:
        if not pairs:
            raise TypeError("At least one (expr, cond) pair expected.")
        self._args: Final[tuple[ExprCondPair, ...]] = tuple(_pair(p) for p in pairs)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[ExprCondPair, ...]:
        return self._args

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 10000, 'Piecewise')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        pairs: list[ExprCondPair] = []
        for a in args:
            if not isinstance(a, ExprCondPair):
                raise TypeError("Piecewise takes (expr, cond) pairs")
            pairs.append(a)
        return _evaluate_pairs(pairs)

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'Piecewise(%s)' % ', '.join(printer.doprint(p) for p in self._args)

    def _eval_facts(self) -> dict[str, bool]:
        sub_facts = [facts_of(p.expr) for p in self._args]
        facts: dict[str, bool] = {}
        for name in _AGREEING:
            values = [s.get(name) for s in sub_facts]
            if all(v is True for v in values):
                facts[name] = True
            elif all(v is False for v in values):
                facts[name] = False
        return facts

    def map_expressions(self, f: Callable[[Expr], Expr]) -> Expr:
        """``f`` applied to every expression, the conditions kept
        (SymPy's ``_eval_derivative``, ``_eval_conjugate``, ...), evaluated."""
        return piecewise(*[(f(p.expr), p.cond) for p in self._args])



def piecewise(*pairs: PairLike) -> Expr:
    """The piecewise expression of the pairs, evaluated (SymPy's
    ``Piecewise(...)``)."""
    from sympy_extras.core.parameters import global_parameters
    given = [_pair(p) for p in pairs]
    if not given:
        raise TypeError("At least one (expr, cond) pair expected.")
    if not global_parameters.evaluate:
        return Piecewise(*given)
    return _evaluate_pairs(given)


def _evaluate_pairs(given: list[ExprCondPair]) -> Expr:
    # SymPy's __new__: the false conditions dropped, nothing after a true one
    kept: list[ExprCondPair] = []
    for p in given:
        if isinstance(p.cond, BooleanFalse):
            continue
        kept.append(p)
        if isinstance(p.cond, BooleanTrue):
            break
    value = _eval(kept)
    if value is not None:
        return value
    return Piecewise(*kept)


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise.eval
def _eval(args: list[ExprCondPair]) -> Optional[Expr]:
    if not args:
        return nan
    if len(args) == 1 and isinstance(args[0].cond, BooleanTrue):
        return args[0].expr
    collapsed = _collapse_arguments(args)
    if not collapsed:
        raise ValueError("There are no conditions (or none that are not trivially false) to define an expression.")
    if len(collapsed) != len(args) or any(a != b for a, b in zip(collapsed, args)):
        return _evaluate_pairs(collapsed)
    return None


def _coeff_add_rational(e: Expr) -> tuple[Expr, Expr]:
    c, rest = e.as_coeff_Add()
    if isinstance(c, Rational):
        return c, rest
    return Rational(0), e


# PORTED FROM SYMPY: sympy/core/relational.py:_canonical_coeff
def _canonical_coeff(rel: Relational) -> Boolean:
    """``-2*x + 1 < 0`` as ``x > 1/2``: the canonical relation with the
    rational constant moved right and the rational coefficient divided
    out, evaluated."""
    rel = rel.canonical
    b, rest = _coeff_add_rational(rel.lhs)
    m, lhs = rest.as_coeff_Mul(rational=True)
    rhs = div(sub(rel.rhs, b), m)
    cls = type(rel)
    if m.is_negative:
        cls = _REVERSED[cls]
    return _FUNCTIONS[cls](lhs, rhs)


def _canonical_conditions(c: Boolean) -> Boolean:
    """Every relation of ``c`` replaced by its :func:`_canonical_coeff`."""
    relations: dict[Basic, Basic] = {}
    stack: list[Basic] = [c]
    while stack:
        node = stack.pop()
        if isinstance(node, Relational):
            relations[node] = _canonical_coeff(node)
        elif isinstance(node, Boolean):
            stack.extend(node.args)
    if not relations:
        return c
    if isinstance(c, Relational):
        replaced = relations[c]
    else:
        replaced = c.xreplace(relations)
    if not isinstance(replaced, Boolean):
        raise TypeError("a condition is expected")
    return replaced


# PORTED FROM SYMPY: sympy/logic/boolalg.py:_distribute (distribute_and_over_or)
def _distribute_and_over_or(e: Boolean) -> Boolean:
    """The conjunctive normal form by distribution of ``Or`` over ``And``."""
    if isinstance(e, Or):
        for arg in e.args:
            if isinstance(arg, And):
                others = [a for a in e.args if a is not arg]
                rest = or_(*[to_boolean(a) for a in others])
                return and_(*[_distribute_and_over_or(or_(to_boolean(c), rest)) for c in arg.args])
        return e
    if isinstance(e, And):
        return and_(*[_distribute_and_over_or(to_boolean(a)) for a in e.args])
    return e


def _weak(r: Relational) -> Optional[Relational]:
    if isinstance(r, Lt):
        return Le(r.lhs, r.rhs)
    if isinstance(r, Gt):
        return Ge(r.lhs, r.rhs)
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:_piecewise_collapse_arguments
def _collapse_arguments(args: list[ExprCondPair]) -> list[ExprCondPair]:
    newargs: list[ExprCondPair] = []
    current: set[Boolean] = set()
    for pair in args:
        expr, cond = pair.expr, _canonical_conditions(pair.cond)
        if isinstance(expr, Piecewise):
            unmatching: list[ExprCondPair] = []
            for inner in expr.args:
                if inner.cond in current:
                    continue
                if inner.cond == cond:
                    if not isinstance(inner.cond, BooleanTrue):
                        expr = _evaluate_pairs(unmatching + [inner]) if unmatching else inner.expr
                    break
                unmatching.append(inner)
        candidates: list[Basic] = [cond]
        if isinstance(cond, And):
            candidates.extend(cond.args)
        if any(isinstance(c, Boolean) and c in current for c in candidates):
            continue
        if isinstance(cond, And):
            nonredundant: list[Boolean] = []
            contradicted = False
            for part in cond.args:
                c = to_boolean(part)
                if isinstance(c, Relational):
                    if c.negated.canonical in current:
                        continue
                    weak = _weak(c)
                    if weak is not None and weak in current:
                        contradicted = True
                        break
                nonredundant.append(c)
            cond = false if contradicted else and_(*nonredundant)
        elif isinstance(cond, Relational):
            if cond.negated.canonical in current:
                cond = true
        current.add(cond)
        if newargs:
            if newargs[-1].expr == expr:
                joined = or_(cond, newargs[-1].cond)
                if isinstance(joined, (And, Or)):
                    joined = _distribute_and_over_or(joined)
                newargs[-1] = ExprCondPair(expr, joined)
                continue
            if newargs[-1].cond == cond:
                continue
        newargs.append(ExprCondPair(expr, cond))
    return newargs

