"""The ``rewrite`` rules of the elementary functions in terms of ``exp``
and ``cos`` (SymPy's ``_eval_rewrite_as_exp`` and
``_eval_rewrite_as_cos``), as a typed table for
:func:`sympy_extras.core.rewriting.rewrite`.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py,
# sympy/functions/elementary/hyperbolic.py, sympy/functions/elementary/exponential.py,
# sympy/core/power.py (the _eval_rewrite_as_exp and _eval_rewrite_as_cos methods) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target) in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.rewriting import RewriteHints
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import E, I, Integer, Rational, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from . import hyperbolic as h, trigonometric as t
from .exponential import Exp, exp, log

#: a rule: the node and its (rewritten) arguments to the rewritten node,
#: ``None`` when the rule does not apply
Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]

HALF_PI = mul(pi, Rational(1, 2))


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _neg(e: Expr) -> Expr:
    from sympy_extras.core.mul import neg
    return neg(e)


def _trig_or_hyper(e: Expr) -> bool:
    return isinstance(e, (t.Sin, t.Cos, t.Tan, t.Cot, t.Sec, t.Csc, h.Sinh, h.Cosh, h.Tanh, h.Coth, h.Sech, h.Csch))


def _exp_arg(a: Expr) -> Expr:
    """SymPy's re-rewrite of a trigonometric or hyperbolic argument."""
    if _trig_or_hyper(a):
        from sympy_extras.core.rewriting import rewrite
        rebuilt = a._evaluated(a.args)
        if isinstance(rebuilt, Expr):
            r = rewrite(rebuilt, (), Exp, RewriteHints())
            if isinstance(r, Expr):
                return r
    return a


# trigonometric, as exp

def _sin_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    return mul(add(exp(mul(a, I)), _neg(exp(mul(_neg(a), I)))), _inv(mul(Integer(2), I)))


def _cos_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    return mul(add(exp(mul(a, I)), exp(mul(_neg(a), I))), Rational(1, 2))


def _tan_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    neg_exp, pos_exp = exp(mul(_neg(a), I)), exp(mul(a, I))
    return mul(I, add(neg_exp, _neg(pos_exp)), _inv(add(neg_exp, pos_exp)))


def _cot_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    neg_exp, pos_exp = exp(mul(_neg(a), I)), exp(mul(a, I))
    return mul(I, add(pos_exp, neg_exp), _inv(add(pos_exp, _neg(neg_exp))))


def _reciprocal(rule: Rule, of: Callable[[Expr], Expr]) -> Rule:
    """SymPy's ``_rewrite_reciprocal``: ``1/t`` with ``t`` the rule of the
    reciprocal function, unless it leaves that function unchanged."""
    def rewritten(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        r = rule(node, args, hints)
        if r is not None and r != of(args[0]):
            return _inv(r)
        return None
    return rewritten


# trigonometric, as cos

def _sin_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return t.Cos(add(args[0], _neg(HALF_PI)))


def _tan_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(t.Cos(add(args[0], _neg(HALF_PI))), _inv(t.cos(args[0])))


def _cot_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(t.cos(args[0]), _inv(t.Cos(add(args[0], _neg(HALF_PI)))))


def _sec_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _inv(t.cos(args[0]))


def _csc_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.rewriting import rewrite
    s = t.sin(args[0])
    r = rewrite(s, (), t.Cos, hints)
    return _inv(r) if isinstance(r, Expr) else None


# hyperbolic

def _sinh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(add(exp(args[0]), _neg(exp(_neg(args[0])))), Rational(1, 2))


def _cosh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(add(exp(args[0]), exp(_neg(args[0]))), Rational(1, 2))


def _tanh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    neg_exp, pos_exp = exp(_neg(args[0])), exp(args[0])
    return mul(add(pos_exp, _neg(neg_exp)), _inv(add(pos_exp, neg_exp)))


def _coth_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    neg_exp, pos_exp = exp(_neg(args[0])), exp(args[0])
    return mul(add(pos_exp, neg_exp), _inv(add(pos_exp, _neg(neg_exp))))


def _cosh_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return t.Cos(mul(I, args[0]))


def _sech_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _inv(t.Cos(mul(I, args[0])))


# exp and powers

def _exp_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = mul(I, args[0])
    return add(t.cos(a), mul(I, t.cos(add(a, HALF_PI))))


def _pow_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import abs_, arg
    base, expo = args
    if base.is_zero is True or base.has(Exp) or expo.has(Exp) or base == E:
        return pow(base, expo)
    if base.has(Symbol):
        # unevaluated without symbols in the exponent: exp(x*log(5)) is x**5
        e = mul(log(base), expo)
        return exp(e) if expo.has(Symbol) else Exp(e)
    return exp(mul(add(log(abs_(base)), mul(I, arg(base))), expo))


def _pow_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    if isinstance(node, Pow) and node.base == E:
        return _exp_cos(node, (node.exp,), hints)
    return None


def rules() -> dict[tuple[type, type], Rule]:
    """The rules keyed by (the class of the node, the target class)."""
    sec_exp_r = _reciprocal(_cos_exp, t.cos)
    csc_exp_r = _reciprocal(_sin_exp, t.sin)
    sech_exp_r = _reciprocal(_cosh_exp, h.cosh)
    csch_exp_r = _reciprocal(_sinh_exp, h.sinh)
    table: dict[tuple[type, type], Rule] = {
        (t.Sin, Exp): _sin_exp, (t.Cos, Exp): _cos_exp, (t.Tan, Exp): _tan_exp, (t.Cot, Exp): _cot_exp,
        (t.Sec, Exp): sec_exp_r, (t.Csc, Exp): csc_exp_r,
        (t.Sin, t.Cos): _sin_cos, (t.Tan, t.Cos): _tan_cos, (t.Cot, t.Cos): _cot_cos, (t.Sec, t.Cos): _sec_cos,
        (t.Csc, t.Cos): _csc_cos,
        (h.Sinh, Exp): _sinh_exp, (h.Cosh, Exp): _cosh_exp, (h.Tanh, Exp): _tanh_exp, (h.Coth, Exp): _coth_exp,
        (h.Sech, Exp): sech_exp_r, (h.Csch, Exp): csch_exp_r,
        (h.Cosh, t.Cos): _cosh_cos, (h.Sech, t.Cos): _sech_cos,
        (Exp, t.Cos): _exp_cos, (Pow, Exp): _pow_exp, (Pow, t.Cos): _pow_cos}
    return table


__all__ = ['rules', 'Rule']
