"""``logcombine`` and ``posify`` of SymPy's ``simplify`` module.

Ported from SymPy's ``sympy.simplify.simplify``: ``logcombine`` joins
logarithms, ``a*log(x) + log(y)`` into ``log(x**a*y)``, when the
coefficient ``a`` is real and the arguments positive (or with ``force``);
``posify`` replaces the symbols of unknown sign by positive dummies and
returns the substitution back.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import log
>>> x, y = Symbol('x', positive=True), Symbol('y', positive=True)
>>> a = Symbol('a', real=True)
>>> logcombine(a*log(x) + log(y) - log(2)), logcombine(log(x) - log(y))
(log(x**a*y/2), log(x/y))
>>> z = Symbol('z')
>>> e, back = posify(1/z + z)
>>> e.free_symbols == {list(back)[0]}, e.subs(back) == 1/z + z
(True, True)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/simplify.py (logcombine, posify) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; posify of a single expression (map it over a list).
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.core.traversal import bottom_up

__all__ = ['logcombine', 'posify']


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _is_log(e: Basic) -> bool:
    from sympy_extras.functions.elementary.exponential import Log
    return isinstance(e, Log)


def _arg(e: Expr) -> Expr:
    return _expr(e.args[0])


LogTerm = tuple[list[Expr], Expr]


# PORTED FROM SYMPY: sympy/simplify/simplify.py:logcombine
def logcombine(expr: ExprLike, force: bool = False) -> Expr:
    """Logarithms combined (see the module), bottom up."""
    from sympy_extras.functions.elementary.exponential import Log, log

    def gooda(a: Expr) -> bool:
        return not (isinstance(a, Integer) and a.p == -1) and (
            a.is_extended_real is True or (force and a.is_extended_real is not False))

    def goodlog(l: Expr) -> bool:
        a = _arg(l)
        return a.is_positive is True or (force and a.is_nonpositive is not False)

    def f(node: Basic) -> Basic:
        if not isinstance(node, (Add, Mul)):
            return node
        other: list[Expr] = []
        logs: list[tuple[list[Expr], list[Expr], list[Expr]]] = []
        log1: dict[tuple[Expr, ...], list[LogTerm]] = {}
        for a in (node.args if isinstance(node, Add) else (node,)):
            a = _expr(a)
            if _is_log(a) and goodlog(a):
                log1.setdefault((), []).append(([], a))
            elif not isinstance(a, Mul):
                other.append(a)
            else:
                ot: list[Expr] = []
                co: list[Expr] = []
                lo: list[Expr] = []
                for ai0 in a.args:
                    ai = _expr(ai0)
                    if isinstance(ai, Rational) and ai.p < 0:
                        ot.append(Integer(-1))
                        co.append(mul(Integer(-1), ai))
                    elif _is_log(ai) and goodlog(ai):
                        lo.append(ai)
                    elif gooda(ai):
                        co.append(ai)
                    else:
                        ot.append(ai)
                if len(lo) > 1:
                    logs.append((ot, co, lo))
                elif lo:
                    log1.setdefault(tuple(ot), []).append((co, lo[0]))
                else:
                    other.append(a)
        if len(other) == 1 and _is_log(other[0]):
            log1.setdefault((), []).append(([], other.pop()))
        if not logs and not log1:
            return node
        for o, e, l0 in logs:
            l = [_expr(v) for v in ordered(l0)]
            combined = log(pow(_arg(l.pop(0)), mul(*e)))
            while l:
                li = l.pop(0)
                combined = log(pow(_arg(li), combined))
            c = mul(*o)
            if _is_log(combined):
                log1.setdefault((c,), []).append(([], combined))
            else:
                other.append(mul(c, combined))
        joined: dict[Expr, Expr] = {}
        for key in list(log1):
            terms = log1.pop(key)
            inner = logcombine(mul(*[pow(_arg(lg), mul(*cs)) for cs, lg in terms]), force=force)
            joined[mul(*key)] = Log(inner)
        for k0 in ordered(list(joined)):
            k = _expr(k0)
            if k not in joined:
                continue
            minus_k = mul(Integer(-1), k)
            if minus_k in joined:
                num, den = k, minus_k
                if _ops(num) > _ops(den):
                    num, den = den, num
                numerator, denominator = joined.pop(num), joined.pop(den)
                other.append(mul(num, Log(mul(_arg(numerator), pow(_arg(denominator), Integer(-1))))))
            else:
                other.append(mul(k, joined.pop(k)))
        return add(*other)

    return _expr(bottom_up(to_expr(expr), f))


def _ops(e: Expr) -> int:
    from sympy_extras.core.function import count_ops
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0


# PORTED FROM SYMPY: sympy/simplify/simplify.py:posify
def posify(eq: ExprLike) -> tuple[Expr, dict[Basic, Basic]]:
    """``eq`` with the symbols of unknown sign replaced by positive
    dummies of the same name, and the mapping back."""
    e = to_expr(eq)
    reps: dict[Basic, Basic] = {}
    for s in e.free_symbols:
        if isinstance(s, Symbol) and s.is_positive is None:
            assumptions = dict(s.assumptions0)
            assumptions['positive'] = True
            reps[s] = Dummy(s.name, **assumptions)
    replaced = _expr(e.subs(reps))
    return replaced, {r: s for s, r in reps.items()}
