"""``signsimp``: sums with a removable minus sign rewritten as ``-1``
times the opposite sum, so that ``2*(x - y)/(y - x)`` becomes ``-2``.

Ported from SymPy's ``sympy.simplify.simplify.signsimp`` and the
``sub_pre``/``sub_post`` of ``sympy.simplify.cse_opts`` (``(x - y)/(y - x)``
is ``-1``): a sum which
can extract a minus sign is replaced by the unevaluated product ``-1 *
(-sum)``, the rebuilt products evaluated (which cancels opposite sums),
the remaining marks removed; a product whose double negation differs
from it (``-(-x)`` is not ``x`` for an unevaluated product) is replaced
by that double negation.

>>> from sympy_extras.core import Symbol
>>> x, y = Symbol('x'), Symbol('y')
>>> signsimp((x - y)/(y - x)), signsimp(1/(1 - x)), signsimp(x - y)
(-1, -1/(x - 1), x - y)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/simplify.py (signsimp),
# sympy/simplify/cse_opts.py (sub_pre, sub_post) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the evaluate=False form not ported (always evaluated).
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer

__all__ = ['signsimp', 'sub_pre', 'sub_post']


def _negated(e: Expr) -> Expr:
    return mul(Integer(-1), e)


def _is_minus_one(e: Basic) -> bool:
    return isinstance(e, Integer) and e.p == -1


def _is_one(e: Basic) -> bool:
    return isinstance(e, Integer) and e.p == 1


# PORTED FROM SYMPY: sympy/simplify/cse_opts.py:sub_pre
def sub_pre(e: Basic) -> Basic:
    """The sums which can extract a minus sign as ``-1*(-sum)``, then the
    remaining ones as ``1*(-1)*(-sum)`` (marks for :func:`sub_post`)."""
    adds = [a for a in e.atoms(Add) if isinstance(a, Add) and a.could_extract_minus_sign()]
    reps: dict[Basic, Basic] = {}
    ignore: set[Basic] = set()
    for a in adds:
        na = _negated(a)
        if isinstance(na, Mul):
            ignore.add(a)
            continue
        reps[a] = Mul(Integer(-1), na)
    e = e._xreplace_evaluated(reps)
    negs: dict[Basic, Basic] = {}
    remaining = [a for a in e.atoms(Add) if isinstance(a, Add)]
    for a in sorted(remaining, key=lambda b: b.sort_key()):
        if a in ignore:
            continue
        if a in reps:
            negs[a] = reps[a]
        elif a.could_extract_minus_sign():
            negs[a] = Mul(Integer(1), Integer(-1), _negated(a))
    return e._xreplace_evaluated(negs)


# PORTED FROM SYMPY: sympy/simplify/cse_opts.py:sub_post
def sub_post(e: Basic) -> Basic:
    """The marks ``1*(-1)*rest`` of :func:`sub_pre` replaced by ``-rest``."""
    replacements: list[tuple[Basic, Basic]] = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and len(node.args) > 2 and _is_one(node.args[0]) and _is_minus_one(node.args[1]):
            rest = [a for a in node.args[2:] if isinstance(a, Expr)]
            replacements.append((node, _negated(Mul(*rest) if len(rest) != 1 else rest[0])))
    for node, replacement in replacements:
        e = e._xreplace_evaluated({node: replacement})
    return e


def _double_negation(e: Basic) -> bool:
    return isinstance(e, Mul) and _negated(_negated(e)) != e


def _undo(e: Basic) -> Basic:
    return _negated(_negated(e)) if isinstance(e, Expr) else e


# PORTED FROM SYMPY: sympy/simplify/simplify.py:signsimp
def signsimp(expr: ExprLike) -> Expr:
    """``expr`` with the sums which can extract a minus sign written with
    it outside, which cancels opposite sums (SymPy's ``signsimp`` with
    ``evaluate=True``)."""
    e0 = to_expr(expr)
    if not e0.args:
        return e0
    e: Basic = e0.replace(_double_negation, _undo)
    e = sub_post(sub_pre(e))
    if not isinstance(e, Expr) or not e.args:
        return to_expr(e) if isinstance(e, Expr) else e0
    if isinstance(e, Add):
        return add(*[signsimp(a) for a in e.args if isinstance(a, Expr)])
    result = e.replace(_double_negation, _undo)
    return result if isinstance(result, Expr) else e0
