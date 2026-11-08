"""``trigsimp``, ``futrig`` and ``exptrigsimp``: trigonometric
simplification.

Ported from SymPy's ``sympy.simplify.trigsimp``: ``futrig`` (the default
``method='matching'``) applies, bottom up, a greedy tree of Fu's rules
minimising the number of trigonometric functions, then of operations, of
nodes, of arguments; hyperbolic functions go through Osborne's rule.
``exptrigsimp`` chooses between exponential and trigonometric forms and
recognises ``cosh``, ``sinh`` and ``tanh`` in products of
``1 +- exp(x)``.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import cos, sin, tan, cosh, sinh, exp
>>> x = Symbol('x')
>>> trigsimp(sin(x)**2 + cos(x)**2), trigsimp(2*sin(x)*cos(x)), trigsimp(sin(x)/cos(x))
(1, sin(2*x), tan(x))
>>> trigsimp(cosh(x)**2 - sinh(x)**2), exptrigsimp(exp(x) + exp(-x))
(1, 2*cosh(x))

Not ported yet: the methods ``groebner``, ``combined`` and ``old``.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/trigsimp.py (trigsimp, futrig, _futrig, _is_Expr,
# _eapply, exptrigsimp) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; only the methods 'matching' and 'fu'.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Literal, Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic, _node_count
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.expr import Expr, ExprLike, factors_of, to_expr
from sympy_extras.core.function import Derivative, count_ops, expand_mul
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import E, I, Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.traversal import bottom_up

from .fu import (HYPER, TR1, TR2, TR2i, TR3, TR5, TR6, TR8, TR10, TR10i, TR11, TR12, TR14, TR15, TR16, TR22, TR111, TRIG,
                 Chain, L, MeasureKey, Minimize, TRmorrie, _TR11, fu, greedy, hyper_as_trig)
from .sqrtdenest import _mexpand

__all__ = ['trigsimp', 'futrig', 'exptrigsimp']

Rule = Callable[[Expr], Expr]
Method = Literal['matching', 'fu', 'groebner', 'combined', 'old']


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:_is_Expr
def _is_Expr(e: Basic) -> bool:
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    if not isinstance(e, Expr):
        return False
    return all(_is_Expr(i) for i in e.args)


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:_eapply
def _eapply(func: Rule, e: Expr, cond: Optional[Callable[[Expr], bool]] = None) -> Expr:
    """``func`` applied to ``e``, or to the arguments satisfying ``cond``
    when ``e`` has non-expression arguments."""
    if _is_Expr(e) or not e.args:
        return func(e)
    return _expr(e._evaluated(tuple(
        _eapply(func, _expr(ei)) if isinstance(ei, Expr) and (cond is None or cond(ei)) else ei for ei in e.args)))


def _has_trig(x: Expr) -> bool:
    return x.has(*TRIG)


def _identity(x: Expr) -> Expr:
    return x


def _lops(x: Expr) -> MeasureKey:
    ops = count_ops(x)
    return (L(x), ops.p if isinstance(ops, Integer) else 0, _node_count(x), len(x.args), 1 if isinstance(x, Add) else 0)


def _factor(x: Expr) -> Expr:
    from sympy_extras.polys.polytools import factor
    return factor(x)


def _build_tree() -> Minimize:
    trigs = _has_trig
    return Minimize(_identity, Chain(
        TR3, TR1, TR12,
        lambda x: _eapply(_factor, x, trigs),
        TR2,
        Minimize(_identity, lambda x: _eapply(_mexpand, x, trigs)),
        TR2i,
        lambda x: _eapply(lambda i: _factor(i.normal()), x, trigs),
        TR14, TR5, TR10, TR11, _TR11, TR6,
        lambda x: _eapply(_factor, x, trigs),
        TR14,
        Minimize(_identity, lambda x: _eapply(_mexpand, x, trigs)),
        TR10i, TRmorrie,
        Minimize(_identity, TR8),
        Minimize(_identity, lambda x: TR2i(TR2(x))),
        Minimize(lambda x: _eapply(expand_mul, TR5(x), trigs), lambda x: _eapply(expand_mul, TR15(x), trigs)),
        Minimize(lambda x: _eapply(expand_mul, TR6(x), trigs), lambda x: _eapply(expand_mul, TR16(x), trigs)),
        TR111,
        Minimize(_identity, TR2i),
        Minimize(_identity, lambda x: _eapply(expand_mul, TR22(x), trigs)),
        TR1, TR2, TR2i,
        Minimize(_identity, lambda x: _eapply(factor_terms, TR12(x), trigs))))


_TREE: Optional[Rule] = None


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:_futrig
def _futrig(e: Expr) -> Expr:
    global _TREE
    if not e.has(*TRIG):
        return e
    coeff: Optional[Expr] = None
    if isinstance(e, Mul):
        independent = [f for f in factors_of(e) if not f.has(*TRIG)]
        dependent = [f for f in factors_of(e) if f.has(*TRIG)]
        coeff, e = mul(*independent), mul(*dependent)
    if _TREE is None:
        _TREE = greedy(_build_tree(), _lops)
    e = _TREE(e)
    if coeff is not None:
        e = mul(coeff, e)
    return e


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:futrig
def futrig(e: ExprLike, hyper: bool = True) -> Expr:
    """``_futrig`` bottom up (and on the hyperbolic functions through their
    trigonometric forms)."""
    expr = to_expr(e)
    if not expr.args:
        return expr
    old = expr
    expr = _expr(bottom_up(expr, lambda b: _futrig(b) if isinstance(b, Expr) else b))
    if hyper and expr.has(*HYPER):
        trig_form, back = hyper_as_trig(expr)
        expr = back(_expr(bottom_up(trig_form, lambda b: _futrig(b) if isinstance(b, Expr) else b)))
    if expr != old and isinstance(expr, Mul) and isinstance(expr.args[0], Rational):
        c, rest = expr.as_coeff_Mul()
        expr = mul(c, rest)
    return expr


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:trigsimp
def trigsimp(expr: ExprLike, method: Method = 'matching') -> Expr:
    """The expression simplified with trigonometric identities
    (``method='matching'``: :func:`futrig`; ``'fu'``: :func:`~.fu.fu`)."""
    e = to_expr(expr)
    if method == 'matching':
        return futrig(e)
    if method == 'fu':
        return fu(e)
    raise NotImplementedError('trigsimp with method=%r is not ported yet' % method)


def _exp_powers(rv: Expr) -> dict[Expr, Expr]:
    return dict(rv.as_powers_dict())


# PORTED FROM SYMPY: sympy/simplify/trigsimp.py:exptrigsimp
def exptrigsimp(expr: ExprLike) -> Expr:
    """Exponential and trigonometric forms chosen by the number of
    operations; ``cosh``/``sinh``/``tanh`` recognised in products with
    ``1 +- exp(x)``."""
    from sympy_extras.functions import cosh, sinh, tanh
    from sympy_extras.functions.elementary.exponential import Exp
    from sympy_extras.functions.elementary.trigonometric import Cos
    e0 = to_expr(expr)
    exp_trig_functions = TRIG + HYPER

    def ops(x: Expr) -> int:
        value = count_ops(x)
        return value.p if isinstance(value, Integer) else 0

    def exp_trig(b: Basic) -> Basic:
        if not isinstance(b, Expr):
            return b
        choices: list[Expr] = [b]
        if b.has(*exp_trig_functions):
            choices.append(_expr(b.rewrite(Exp)))
        choices.append(_expr(b.rewrite(Cos)))
        return min(choices, key=ops)

    newexpr = _expr(bottom_up(e0, exp_trig))

    def signlog(x: Expr, sign: Expr = Integer(1)) -> tuple[Optional[Expr], Optional[Expr]]:
        if x == E:
            return sign, Integer(1)
        if isinstance(x, Exp):
            return sign, _expr(x.args[0])
        if isinstance(x, Pow) and x.base == E:
            return sign, x.exp
        if sign == Integer(1):
            return signlog(mul(Integer(-1), x), Integer(-1))
        return None, None

    def f(b: Basic) -> Basic:
        if not isinstance(b, Mul):
            return b
        commutative = [g for g in factors_of(b) if g.is_commutative is not False]
        noncommutative = [g for g in factors_of(b) if g.is_commutative is False]
        if len(noncommutative) > 1:
            return mul(_expr(f(mul(*commutative))), *noncommutative)
        rvd = _exp_powers(b)
        newd = dict(rvd)
        zero = Integer(0)
        ee = rvd.get(E, zero)
        for k in list(rvd):
            if isinstance(k, Add) and len(k.args) == 2:
                c = _expr(k.args[0])
                sign, x = signlog(mul(_expr(k.args[1]), pow(c, Integer(-1))))
                if x is None or sign is None or (isinstance(x, Integer) and x.p == 0):
                    continue
                m = rvd[k]
                newd[k] = newd.get(k, zero) - m
                if ee == mul(Integer(-1), x, m, Rational(1, 2)):
                    newd[E] = newd.get(E, zero) - ee
                    ee = zero
                    key = mul(Integer(2), c, cosh(mul(x, Rational(1, 2)))) if sign == Integer(1) else \
                        mul(Integer(-2), c, sinh(mul(x, Rational(1, 2))))
                    newd[key] = newd.get(key, zero) + m
                elif newd.get(1 - mul(sign, pow(E, x)), zero) == -m:
                    del newd[1 - mul(sign, pow(E, x))]
                    key = mul(Integer(-1), c, pow(tanh(mul(x, Rational(1, 2))), Integer(-1))) if sign == Integer(1) \
                        else mul(Integer(-1), c, tanh(mul(x, Rational(1, 2))))
                    newd[key] = newd.get(key, zero) + m
                else:
                    key = 1 + mul(sign, pow(E, x))
                    newd[key] = newd.get(key, zero) + m
                    newd[c] = newd.get(c, zero) + m
        return mul(*[pow(k, v) for k, v in newd.items()])

    newexpr = _expr(bottom_up(newexpr, f))
    if newexpr.has(*HYPER):
        trig_form, back = hyper_as_trig(newexpr)
        newexpr = back(TR2i(trig_form))
    if newexpr.has(*TRIG):
        newexpr = TR2i(newexpr)
    if not (newexpr.has(I) and not e0.has(I)):
        return newexpr
    return e0
