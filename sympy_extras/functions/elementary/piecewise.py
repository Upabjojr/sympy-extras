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
those all the expressions agree on. :func:`piecewise_fold` brings the
``Piecewise`` of a sum, product or function to the top. Not ported:
``piecewise_exclusive`` and the integration; ``piecewise_simplify``
simplifies the conditions of several variables through SymPy; ``piecewise_fold`` keeps an ``ITE`` condition as it is where
SymPy rewrites it in conjunctive normal form.

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
from sympy_extras.core.symbol import Symbol

__all__ = ['ExprCondPair', 'Piecewise', 'piecewise', 'piecewise_fold']

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
    #: SymPy's class is a Function
    is_Function = True

    def __init__(self, *pairs: PairLike) -> None:
        if not pairs:
            raise TypeError("At least one (expr, cond) pair expected.")
        self._args: Final[tuple[ExprCondPair, ...]] = tuple(_pair(p) for p in pairs)
        Basic.__init__(self)

    # PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        """The leading term of the first piece whose condition holds at 0
        (None in SymPy when none does: then ``as_leading_term`` raises)."""
        from sympy_extras.core.numbers import Integer as _Integer
        for pair in self.args:
            c = pair.cond
            if isinstance(c, BooleanTrue) or isinstance(c.subs(x, _Integer(0)), BooleanTrue):
                return pair.expr.as_leading_term(x)
        raise NotImplementedError('as_leading_term(%s, %s)' % (self, x))

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

    # PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._eval_power
    def _eval_power_node(self, exponent: Expr) -> Optional[Expr]:
        """Each piece raised: ``Piecewise((x, c), (y, True))**2`` is
        ``Piecewise((x**2, c), (y**2, True))``."""
        from sympy_extras.core.power import pow
        return piecewise(*[(pow(p.expr, exponent), p.cond) for p in self._args])

    # PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._eval_transpose, _eval_adjoint
    def _eval_transpose_node(self) -> Optional[Expr]:
        return self.map_expressions(lambda e: e.transpose())

    def _eval_adjoint_node(self) -> Optional[Expr]:
        return self.map_expressions(lambda e: e.adjoint())

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




def _has_piecewise(e: Basic) -> bool:
    stack: list[Basic] = [e]
    while stack:
        node = stack.pop()
        if isinstance(node, Piecewise):
            return True
        stack.extend(node.args)
    return False


def _common_prefix(lists: list[list[Boolean]]) -> list[Boolean]:
    prefix: list[Boolean] = []
    for items in zip(*lists):
        if all(item == items[0] for item in items):
            prefix.append(items[0])
        else:
            break
    return prefix


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:piecewise_fold
def piecewise_fold(expr: Expr, evaluate: Optional[bool] = True) -> Expr:
    """The ``Piecewise`` parts of ``expr`` brought to the top: ``x +
    Piecewise((1, x > 0), (2, True))`` is ``Piecewise((x + 1, x > 0),
    (x + 2, True))``; Piecewise sums and products with the same conditions
    are combined pairwise. ``evaluate=None`` drops the repeated conditions
    and unwraps a single true pair, as SymPy's.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, gt, true
    >>> from sympy_extras.functions.elementary.piecewise import piecewise, piecewise_fold
    >>> x = Symbol('x')
    >>> piecewise_fold(x*piecewise((1, gt(x, 0)), (2, true)) + 3)
    Piecewise((x + 3, x > 0), (2*x + 3, True))
    """
    from itertools import product
    from sympy_extras.core.add import Add
    from sympy_extras.core.mul import Mul
    from sympy_extras.core.sorting import default_sort_key
    if not _has_piecewise(expr):
        return expr
    new_args: list[tuple[Expr, Boolean]] = []
    if isinstance(expr, Piecewise):
        for pair in expr.args:
            e, c = pair.expr, pair.cond
            if not isinstance(e, Piecewise):
                e = piecewise_fold(e)
            if isinstance(e, Piecewise):
                new_args.extend([(piecewise_fold(inner.expr), and_(inner.cond, c)) for inner in e.args])
            else:
                new_args.append((e, c))
    else:
        args: list[Expr]
        if isinstance(expr, Add) or (isinstance(expr, Mul) and expr.is_commutative):
            pieces = [a for a in expr.args if isinstance(a, Piecewise)]
            args = [a for a in expr.args if not isinstance(a, Piecewise)]
            groups: dict[tuple[Boolean, ...], list[Piecewise]] = {}
            for piece in pieces:
                if isinstance(piece, Piecewise):
                    groups.setdefault(tuple(pair.cond for pair in piece.args), []).append(piece)
            for key in sorted(groups, key=lambda cs: tuple(default_sort_key(c) for c in cs)):
                group = groups[key]
                if len(group) > 1:
                    pargs = [list(g.args) for g in group]
                    common = _common_prefix([[pair.cond for pair in a] for a in pargs])
                    n = len(common)
                    collected: list[tuple[Expr, BooleanLike]] = []
                    for i in range(n):
                        collected.append((_as_expression(expr._evaluated(tuple(a[i].expr for a in pargs))), common[i]))
                    remains: list[Expr] = []
                    for a in pargs:
                        if n == len(a):
                            continue
                        if isinstance(a[n].cond, BooleanTrue):
                            remains.append(a[n].expr)
                        else:
                            remains.append(Piecewise(*a[n:]))
                    if remains:
                        collected.append((_as_expression(expr._evaluated(tuple(remains))), true))
                    args.append(Piecewise(*collected))
                    continue
                args.extend(group)
        else:
            args = [a for a in expr.args if isinstance(a, Expr)]
        # the arguments which are not expressions (the limits of a Sum) are
        # kept as they are: dropped, the node could not be rebuilt
        parts: list[Basic] = list(args) if len(args) == len(expr.args) or isinstance(expr, (Add, Mul)) \
            else list(expr.args)
        options: list[list[tuple[Basic, Boolean]]] = []
        for part in parts:
            f = piecewise_fold(part) if isinstance(part, Expr) else part
            options.append([(pair.expr, pair.cond) for pair in f.args] if isinstance(f, Piecewise) else [(f, true)])
        for combination in product(*options):
            expressions = tuple(e for e, _ in combination)
            conditions: list[Boolean] = [c for _, c in combination]
            rebuilt = expr._evaluated(expressions)
            if not isinstance(rebuilt, Expr):
                raise TypeError("an expression is expected")
            new_args.append((rebuilt, and_(*conditions)))
    if evaluate is None:
        seen: dict[Boolean, Expr] = {}
        for e, c in reversed(new_args):
            seen[c] = e
        new_args = list(reversed([(e, c) for c, e in seen.items()]))
    result = piecewise(*new_args) if evaluate is not False else Piecewise(*new_args)
    if evaluate is None and isinstance(result, Piecewise) and len(result.args) == 1 \
            and isinstance(result.args[0].cond, BooleanTrue):
        return result.args[0].expr
    if isinstance(result, Piecewise) and any(_has_piecewise(pair.expr) for pair in result.args):
        return piecewise_fold(result)
    return result


def _as_expression(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("an expression is expected")
    return e


# ---------------------------------------------------------------------
# the intervals of a Piecewise and piecewise_simplify


def _lts_gts(r: Relational) -> tuple[Expr, Expr]:
    """SymPy's ``r.lts`` and ``r.gts``: the smaller and the greater side."""
    if isinstance(r, (Lt, Le)):
        return r.lhs, r.rhs
    return r.rhs, r.lhs


def _to_nnf(b: Boolean) -> Boolean:
    """The negations pushed to the relations (SymPy's
    ``eliminate_implications`` on the connectives of a condition)."""
    from sympy_extras.core.logic import Not, not_
    if isinstance(b, Not):
        inner = b.args[0]
        assert isinstance(inner, Boolean)
        if isinstance(inner, And):
            return or_(*[_to_nnf(not_(to_boolean(a))) for a in inner.args])
        if isinstance(inner, Or):
            return and_(*[_to_nnf(not_(to_boolean(a))) for a in inner.args])
        return b
    if isinstance(b, And):
        return and_(*[_to_nnf(to_boolean(a)) for a in b.args])
    if isinstance(b, Or):
        return or_(*[_to_nnf(to_boolean(a)) for a in b.args])
    return b


def _to_cnf(b: Boolean) -> Boolean:
    """SymPy's ``to_cnf`` (without simplification)."""
    return _distribute_and_over_or(_to_nnf(b))


def _distribute_or_over_and(e: Boolean) -> Boolean:
    """The disjunctive normal form by distribution of ``And`` over ``Or``
    (SymPy's ``distribute_or_over_and``)."""
    if isinstance(e, And):
        for arg in e.args:
            if isinstance(arg, Or):
                others = [a for a in e.args if a is not arg]
                rest = and_(*[to_boolean(a) for a in others])
                return or_(*[_distribute_or_over_and(and_(to_boolean(c), rest)) for c in arg.args])
        return e
    if isinstance(e, Or):
        return or_(*[_distribute_or_over_and(to_boolean(a)) for a in e.args])
    return e


IntervalPiece = tuple[Expr, Expr, Expr, int]


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:Piecewise._intervals
def piecewise_intervals(pw: Piecewise, sym: Symbol,
                        err_on_Eq: bool = False) -> tuple[bool, Union[str, list[IntervalPiece]]]:
    """The intervals of ``sym`` on which the pieces apply: ``(True, [(lower,
    upper, expr, index), ...])``, or ``(False, reason)`` when a condition
    cannot be written with intervals of ``sym``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions.elementary.piecewise import piecewise, piecewise_intervals
    >>> x = Symbol('x')
    >>> piecewise_intervals(piecewise((1, x < 0), (x, x < 2), (2, True)), x)
    (True, [(-oo, 0, 1, 0), (-oo, 2, x, 1), (-oo, oo, 2, 2)])
    """
    from sympy_extras.core.logic import false as false_, true as true_
    from sympy_extras.core.numbers import NegativeInfinity, oo
    from sympy_extras.core.relational import Eq, Ne
    from sympy_extras.core.sorting import ordered
    from sympy_extras.functions.elementary.miscellaneous import max_, min_
    from sympy_extras.solvers.inequalities import solve_inequality

    def nonsymfail(cond: Boolean) -> tuple[bool, str]:
        return False, 'A condition not involving %s appeared: %s' % (sym, cond)

    def solve_relational(r: Relational) -> tuple[bool, Union[str, Boolean]]:
        if sym not in r.free_symbols:
            return nonsymfail(r)
        try:
            rv = solve_inequality(r, sym)
        except NotImplementedError:
            return False, 'Unable to solve relational %s for %s.' % (r, sym)
        if isinstance(rv, Relational):
            if rv.lhs != sym or sym in rv.rhs.free_symbols:
                return False, 'Unable to solve relational %s for %s.' % (r, sym)
            if rv.rel_op == '==':
                rv = false_
            elif rv.rel_op == '!=':
                try:
                    rv = or_(sym < rv.rhs, sym > rv.rhs)
                except TypeError:
                    rv = true_
        elif rv == and_(NegativeInfinity() < sym, sym < oo):
            rv = true_
        return True, rv

    reps: dict[Basic, Basic] = {}
    for r in ordered(list(pw.atoms(Relational))):
        assert isinstance(r, Relational)
        ok, solved = solve_relational(r)
        if not ok:
            assert isinstance(solved, str)
            return False, solved
        assert isinstance(solved, Boolean)
        reps[r] = solved
    args: list[tuple[Expr, Boolean]] = []
    for pair in pw.args:
        c = pair.cond.xreplace(reps)
        assert isinstance(c, Boolean)
        args.append((pair.expr, c))
    expr_cond: list[tuple[int, Expr, Boolean]] = []
    default: Optional[Expr] = None
    idefault = -1
    for i, (expr, cond) in enumerate(args):
        if cond is false_:
            continue
        if cond is true_:
            default, idefault = expr, i
            break
        if isinstance(cond, Eq):
            if err_on_Eq:
                return False, 'encountered Eq condition: %s' % cond
            continue
        cond = _to_cnf(cond)
        if isinstance(cond, And):
            cond = _distribute_or_over_and(cond)
        if isinstance(cond, Or):
            expr_cond.extend((i, expr, to_boolean(o)) for o in cond.args if not isinstance(o, Eq))
        elif cond is not false_:
            expr_cond.append((i, expr, cond))
    int_expr: list[IntervalPiece] = []
    k = 0
    while k < len(expr_cond):   # SymPy extends the list it iterates over
        iarg, expr, cond = expr_cond[k]
        k += 1
        lower: Expr
        upper: Expr
        if isinstance(cond, And):
            lower, upper = NegativeInfinity(), oo
            exclude: list[Expr] = []
            for cond2 in cond.args:
                if not isinstance(cond2, Relational):
                    return False, 'expecting only Relationals'
                if isinstance(cond2, Eq):
                    lower = upper
                    if err_on_Eq:
                        return False, 'encountered secondary Eq condition'
                    break
                if isinstance(cond2, Ne):
                    l_, r_ = cond2.lhs, cond2.rhs
                    if l_ == sym:
                        exclude.append(r_)
                    elif r_ == sym:
                        exclude.append(l_)
                    else:
                        return nonsymfail(cond2)
                    continue
                lts, gts = _lts_gts(cond2)
                if lts == sym:
                    upper = min_(gts, upper)
                elif gts == sym:
                    lower = max_(lts, lower)
                else:
                    return nonsymfail(cond2)
            if exclude:
                # SymPy's `e < lower == True or e > upper == True` is a chained
                # comparison which never holds: no excluded point is skipped
                points = [e for e in ordered(exclude) if isinstance(e, Expr)]
                bounds: list[tuple[Expr, Expr]] = []
                previous = lower
                for e in points:
                    bounds.append((previous, e))
                    previous = e
                bounds.append((previous, upper))
                expr_cond.extend((iarg, expr, and_(lo < sym, sym < hi)) for lo, hi in bounds)
                continue
        elif isinstance(cond, Relational) and cond.rel_op != '!=':
            lower, upper = _lts_gts(cond)
            if lower == sym:
                lower = NegativeInfinity()
            elif upper == sym:
                upper = oo
            else:
                return nonsymfail(cond)
        else:
            return False, 'unrecognized condition: %s' % cond
        upper = max_(lower, upper)
        if err_on_Eq and lower == upper:
            return False, 'encountered Eq condition'
        if (lower >= upper) is not true_:
            int_expr.append((lower, upper, expr, iarg))
    if default is not None:
        int_expr.append((NegativeInfinity(), oo, default, idefault))
    unique: list[IntervalPiece] = []
    for piece in int_expr:
        if piece not in unique:
            unique.append(piece)
    return True, unique


def _simplify_condition(c: Boolean) -> Boolean:
    """SymPy's ``simplify`` of a condition: a relation by the native port
    of ``Relational._eval_simplify``
    (:func:`~sympy_extras.assumptions.refine.simplify_relational`), a
    compound condition through SymPy (``BooleanFunction._eval_simplify``,
    the patterns of the relations of ``And`` and ``Or``, is not ported);
    ``True`` and ``False`` are left as they are, as SymPy does."""
    if isinstance(c, (BooleanTrue, BooleanFalse)):
        return c
    if isinstance(c, Relational):
        from sympy_extras.assumptions.refine import simplify_relational
        return simplify_relational(c, 1.7, False, False, True)
    import sympy
    from sympy_extras.interop import from_sympy, to_sympy
    r = from_sympy(sympy.simplify(to_sympy(c)))
    return r if isinstance(r, Boolean) else c


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:piecewise_simplify_arguments
def piecewise_simplify_arguments(pw: Piecewise, doit: bool = True) -> Expr:
    """The pieces simplified: for one variable, the conditions rewritten as
    disjoint intervals of it (``nan`` where no piece applies); otherwise
    each condition simplified; the expressions simplified."""
    from sympy_extras.core.logic import true as true_
    from sympy_extras.core.numbers import Infinity, NegativeInfinity
    from sympy_extras.core.relational import Eq
    from sympy_extras.sets import EmptySet, Reals, Set, finite_set, interval, union
    from sympy_extras.simplify.simplify import simplify
    first = pw.args[0].cond.free_symbols
    new_args: Optional[list[tuple[Expr, Boolean]]] = None
    if len(first) == 1 and not pw.atoms(Eq):
        x = next(iter(first))
        assert isinstance(x, Symbol)
        ok, abe = piecewise_intervals(pw, x, err_on_Eq=True)

        def include(c: Boolean, a: Expr) -> bool:
            try:
                return c.subs(x, a) is true_
            except TypeError:
                return False

        def member(s: Set, a: Expr) -> bool:
            # SymPy's `a in covered`: a TypeError when it is not decided
            c = s.contains(a)
            if c is true_:
                return True
            if c is false:
                return False
            raise TypeError('did not evaluate to a bool: %r' % (c,))

        if ok:
            assert isinstance(abe, list)
            new_args = []
            covered: Set = EmptySet
            for a, b, e, i in abe:
                c = pw.args[i].cond
                incl_a = include(c, a)
                incl_b = include(c, b)
                iv = interval(a, b, not incl_a, not incl_b)
                cset = iv - covered
                if cset is EmptySet:
                    continue
                try:
                    a = cset.inf
                except NotImplementedError:
                    pass
                else:
                    incl_a = include(c, a)
                if incl_a and incl_b:
                    if a.is_infinite and b.is_infinite:
                        c = true_
                    elif b.is_infinite:
                        c = (x > a) if member(covered, a) else (x >= a)
                    elif a.is_infinite:
                        c = x <= b
                    elif member(covered, a):
                        c = and_(a < x, x <= b)
                    else:
                        c = and_(a <= x, x <= b)
                elif incl_a:
                    if a.is_infinite:
                        c = x < b
                    elif member(covered, a):
                        c = and_(a < x, x < b)
                    else:
                        c = and_(a <= x, x < b)
                elif incl_b:
                    if b.is_infinite:
                        c = x > a
                    else:
                        c = and_(a < x, x <= b)
                else:
                    if member(covered, a):
                        c = x < b
                    else:
                        c = and_(a < x, x < b)
                covered = union(covered, iv)
                if isinstance(a, NegativeInfinity) and incl_a:
                    covered = union(covered, finite_set(NegativeInfinity()))
                if isinstance(b, Infinity) and incl_b:
                    covered = union(covered, finite_set(b))
                new_args.append((e, c))
            if Reals.is_subset(covered) is not True:
                new_args.append((nan, true_))
    if new_args is None:
        new_args = [(p.expr, _simplify_condition(p.cond)) for p in pw.args]
    final: list[PairLike] = []
    for e, c in new_args:
        newe = simplify(e, doit=False)
        final.append((newe if newe != e else e, c))
    return piecewise(*final)


def _diff_wrt(e: Expr) -> bool:
    """SymPy's ``_diff_wrt``: a symbol, an undefined function, a derivative."""
    from sympy_extras.core.function import AppliedUndef, Derivative
    return isinstance(e, (Symbol, AppliedUndef, Derivative))


def _blessed(e: Relational) -> bool:
    from sympy_extras.core.numbers import NumberSymbol
    return _diff_wrt(e.lhs) and (_diff_wrt(e.rhs) or isinstance(e.rhs, (Rational, NumberSymbol)))


def _eqs_and_others(cond: Boolean) -> tuple[list[Relational], list[Boolean]]:
    from sympy_extras.core.relational import Eq
    if isinstance(cond, And):
        eqs: list[Relational] = [a for a in cond.args if isinstance(a, Eq)]
        others = [to_boolean(a) for a in cond.args if not isinstance(a, Eq)]
        return eqs, others
    if isinstance(cond, Eq):
        single: list[Relational] = [cond]
        return single, []
    return [], []


def _subs_pair(e: Basic, eq: Relational) -> Basic:
    return e.subs(eq.lhs, eq.rhs)


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:_piecewise_simplify_eq_and
def _piecewise_simplify_eq_and(args: list[tuple[Expr, Boolean]]) -> list[tuple[Expr, Boolean]]:
    """The expression of a piece with the equations of its condition
    substituted (``(x*y, Eq(y, 0) & ...)`` becomes ``(0, ...)``)."""
    from sympy_extras.core.sorting import ordered
    for i, (expr, cond) in enumerate(args):
        eqs, other = _eqs_and_others(cond)
        if eqs:
            eqs = [e for e in ordered(eqs) if isinstance(e, Relational)]
            for j, e in enumerate(list(eqs)):
                e = eqs[j]
                if _blessed(e):
                    expr = _as_expression(_subs_pair(expr, e))
                    eqs[j + 1:] = [r for r in (_subs_pair(ei, e) for ei in eqs[j + 1:]) if isinstance(r, Relational)]
                    other = [to_boolean(_subs_pair(ei, e)) for ei in other]
            cond = and_(*(list(eqs) + other))
            args[i] = (expr, cond)
    return args


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:_piecewise_simplify_equal_to_next_segment
def _piecewise_simplify_equal_to_next_segment(args: list[tuple[Expr, Boolean]]) -> list[tuple[Expr, Boolean]]:
    """A piece whose expression equals the next one under its equations
    takes the next expression."""
    from sympy_extras.core.sorting import ordered
    prevexpr: Optional[Expr] = None
    for i, (expr, cond) in reversed(list(enumerate(args))):
        if prevexpr is not None:
            eqs, other = _eqs_and_others(cond)
            _prevexpr, _expr = prevexpr, expr
            if eqs and not other:
                for e in ordered(eqs):
                    assert isinstance(e, Relational)
                    if len(args) == 2 or _blessed(e):
                        _prevexpr = _as_expression(_subs_pair(_prevexpr, e))
                        _expr = _as_expression(_subs_pair(_expr, e))
            if _prevexpr == _expr:
                args[i] = (args[i + 1][0], cond)
            else:
                prevexpr = expr
        else:
            prevexpr = expr
    return args


# PORTED FROM SYMPY: sympy/functions/elementary/piecewise.py:piecewise_simplify
def piecewise_simplify(expr: Expr, doit: bool = True) -> Expr:
    """A ``Piecewise`` simplified: its pieces (:func:`piecewise_simplify_arguments`),
    the equations of the conditions substituted, the pieces equal to the
    next one merged.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions.elementary.piecewise import piecewise, piecewise_simplify
    >>> x = Symbol('x')
    >>> piecewise_simplify(piecewise((1, x < 0), (1, x < 1), (2, True)))
    Piecewise((1, (x <= 0) | (x < 1)), (2, True))
    >>> piecewise_simplify(piecewise((x**2, (x > 0) & (x < 2)), (x + x, x <= 0), (0, True)))
    Piecewise((x**2, (x > 0) & (x < 2)), (2*x, x <= 0), (0, x >= 2))
    """
    if not isinstance(expr, Piecewise):
        return expr
    simplified = piecewise_simplify_arguments(expr, doit=doit)
    if not isinstance(simplified, Piecewise):
        return simplified
    args = [(p.expr, p.cond) for p in simplified.args]
    args = _piecewise_simplify_eq_and(args)
    args = _piecewise_simplify_equal_to_next_segment(args)
    return piecewise(*args)
