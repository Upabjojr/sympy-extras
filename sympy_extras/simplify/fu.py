"""The trigonometric transformations of Fu, Zhong and Zeng (the ``TR``
rules of ``fu``), part 1: ``TR0`` to ``TR9``.

Ported from SymPy's ``sympy.simplify.fu``: ``TR1`` writes ``sec``/``csc``
as reciprocals, ``TR2`` ``tan``/``cot`` as ratios and ``TR2i`` back,
``TR3`` reduces arguments (``sin(-x)``, ``cos(pi/2 - x)``), ``TR5``/``TR6``
replace even powers of ``sin``/``cos`` through ``sin**2 = 1 - cos**2``,
``TR7`` lowers ``cos**2``, ``TR8`` turns products into sums, ``TR9``
sums into products; ``hyper_as_trig`` maps hyperbolic functions to
trigonometric ones (Osborne's rule) and back. Each rule works bottom up.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import cos, sin, tan, sec
>>> x, y = Symbol('x'), Symbol('y')
>>> TR1(2*sec(x)), TR2(tan(x)), TR2i(sin(x)/cos(x)), TR5(sin(x)**2)
(2/cos(x), sin(x)/cos(x), tan(x), 1 - cos(x)**2)
>>> TR8(cos(x)*sin(y)), TR9(cos(x) + cos(y))
(-sin(x - y)/2 + sin(x + y)/2, 2*cos(x/2 - y/2)*cos(x/2 + y/2))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/fu.py (TR0, TR1, TR2, TR2i, TR3, _TR56, TR5,
# TR6, TR7, TR8, TR9, process_common_addends, trig_split, as_f_sign_1,
# _osborne, _osbornei, hyper_as_trig, _ROOT2, _ROOT3, _invROOT3) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; Factors with any exponents is ExprFactors.
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Hashable
from typing import Callable, Optional, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.exprtools import ExprFactors, factor_terms, gcd_terms
from sympy_extras.core.expr import Expr, ExprLike, factors_of, to_expr
from sympy_extras.core.function import expand_mul
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number, Rational, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy
from sympy_extras.core.traversal import bottom_up
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh, cosh, coth, csch, sech, sinh, tanh
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, cos, cot, csc, sec, sin, tan

__all__ = ['TR0', 'TR1', 'TR2', 'TR2i', 'TR3', 'TR4', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'TR10', 'TR10i', 'TR11',
           'TR12', 'TR12i', 'TR13', 'TR14', 'TR15', 'TR16', 'TR111', 'TR22', 'TRmorrie', 'TRpower', 'L', 'fu',
           'sincos_to_sum', 'trig_split', 'as_f_sign_1', 'process_common_addends', 'hyper_as_trig', 'greedy',
           'Chain', 'Minimize']

Rule = Callable[[Expr], Expr]
TRIG = (Sin, Cos, Tan, Cot, Sec, Csc)
HYPER = (Sinh, Cosh, Tanh, Coth, Sech, Csch)


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _arg(e: Expr) -> Expr:
    return _expr(e.args[0])


def _bottom(rv: Expr, f: Rule) -> Expr:
    return _expr(bottom_up(rv, lambda b: f(b) if isinstance(b, Expr) else b))


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _neg(e: Expr) -> Expr:
    return mul(Integer(-1), e)


def _half(e: Expr) -> Expr:
    return mul(e, Rational(1, 2))


def _true(b: Basic) -> bool:
    from sympy_extras.core.logic import true
    return b is true


def _ROOT2() -> Expr:
    from sympy_extras.functions import sqrt
    return sqrt(Integer(2))


def _ROOT3() -> Expr:
    from sympy_extras.functions import sqrt
    return sqrt(Integer(3))


def _invROOT3() -> Expr:
    return _inv(_ROOT3())


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR0
def TR0(rv: ExprLike) -> Expr:
    """``rv`` normalised, factored and expanded."""
    from sympy_extras.polys.polytools import factor
    return factor(to_expr(rv).normal()).expand()


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR1
def TR1(rv: ExprLike) -> Expr:
    """``sec`` and ``csc`` as ``1/cos`` and ``1/sin``."""
    def f(e: Expr) -> Expr:
        if isinstance(e, Sec):
            return _inv(cos(_arg(e)))
        if isinstance(e, Csc):
            return _inv(sin(_arg(e)))
        return e
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR2
def TR2(rv: ExprLike) -> Expr:
    """``tan`` and ``cot`` as ratios of ``sin`` and ``cos``."""
    def f(e: Expr) -> Expr:
        if isinstance(e, Tan):
            return mul(sin(_arg(e)), _inv(cos(_arg(e))))
        if isinstance(e, Cot):
            return mul(cos(_arg(e)), _inv(sin(_arg(e))))
        return e
    return _bottom(to_expr(rv), f)


Powers = dict[Expr, Optional[Expr]]


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR2i
def TR2i(rv: ExprLike, half: bool = False) -> Expr:
    """Ratios of ``sin`` and ``cos`` as ``tan`` (``half``: with
    ``1 + cos`` in the denominator, as ``tan`` of the half angle)."""
    from sympy_extras.polys.polytools import factor

    def ok(k: Expr, e: Expr) -> bool:
        return (e.is_integer is True or k.is_positive is True) and (
            isinstance(k, (Sin, Cos)) or (half and isinstance(k, Add) and len(k.args) >= 2 and any(
                any(isinstance(ai, Cos) or (isinstance(ai, Pow) and isinstance(ai.base, Cos))
                    for ai in factors_of(_expr(a))) for a in k.args)))

    def factorize(d: dict[Expr, Expr], ddone: list[tuple[Expr, Expr]]) -> None:
        newk: list[tuple[Expr, Expr]] = []
        for k in d:
            if isinstance(k, Add) and len(k.args) > 1:
                knew = factor(k) if half else factor_terms(k)
                if knew != k:
                    newk.append((k, knew))
        if newk:
            for k, _ in newk:
                del d[k]
            powers = mul(*[knew for _, knew in newk]).as_powers_dict()
            for k, value in powers.items():
                v = add(d[k], value) if k in d else value
                if ok(k, v):
                    d[k] = v
                else:
                    ddone.append((k, v))

    def f(e: Expr) -> Expr:
        if not isinstance(e, Mul):
            return e
        n0, d0 = e.as_numer_denom()
        if not n0.args or not d0.args:
            return e
        n = dict(n0.as_powers_dict())
        ndone = [(k, n.pop(k)) for k in list(n) if not ok(k, n[k])]
        if not n:
            return e
        d = dict(d0.as_powers_dict())
        ddone = [(k, d.pop(k)) for k in list(d) if not ok(k, d[k])]
        if not d:
            return e
        factorize(n, ndone)
        factorize(d, ddone)
        nn: Powers = dict(n)
        dd: Powers = dict(d)
        t: list[Expr] = []
        for k in list(nn):
            nk = nn[k]
            if nk is None:
                continue
            if isinstance(k, Sin):
                a: Expr = Cos(_arg(k))
                if a in dd and dd[a] == nk:
                    t.append(pow(tan(_arg(k)), nk))
                    nn[k] = dd[a] = None
                elif half:
                    a1 = add(Integer(1), a)
                    if a1 in dd and dd[a1] == nk:
                        t.append(pow(tan(_half(_arg(k))), nk))
                        nn[k] = dd[a1] = None
            elif isinstance(k, Cos):
                a = Sin(_arg(k))
                if a in dd and dd[a] == nk:
                    t.append(pow(tan(_arg(k)), _neg(nk)))
                    nn[k] = dd[a] = None
            elif half and isinstance(k, Add) and k.args[0] == Integer(1) and isinstance(k.args[1], Cos):
                a = Sin(_arg(_expr(k.args[1])))
                da = dd.get(a)
                if a in dd and da == nk and da is not None and (da.is_integer is True or a.is_positive is True):
                    t.append(pow(tan(_half(_arg(a))), _neg(nk)))
                    nn[k] = dd[a] = None
        if t:
            numerator = mul(*(t + [pow(b, ex) for b, ex in nn.items() if ex is not None and ex != 0]))
            denominator = mul(*[pow(b, ex) for b, ex in dd.items() if ex is not None and ex != 0])
            result = mul(numerator, _inv(denominator))
            return mul(result, mul(*[pow(b, ex) for b, ex in ndone]), _inv(mul(*[pow(b, ex) for b, ex in ddone])))
        return e

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR3
def TR3(rv: ExprLike) -> Expr:
    """Arguments reduced: signs out (``cos(-x)`` is ``cos(x)``) and
    ``pi/4 < x < pi/2`` mapped to the co-function of ``pi/2 - x``."""
    from .signsimp import signsimp
    fmap: dict[type, Callable[[Expr], Expr]] = {Cos: sin, Sin: cos, Tan: cot, Cot: tan, Sec: csc, Csc: sec}
    rebuild: dict[type, Callable[[Expr], Expr]] = {Cos: cos, Sin: sin, Tan: tan, Cot: cot, Sec: sec, Csc: csc}

    def f(e: Expr) -> Expr:
        if not isinstance(e, TRIG):
            return e
        e = rebuild[type(e)](signsimp(_arg(e)))
        if not isinstance(e, TRIG):
            return e
        a = _arg(e)
        first = add(a, _neg(mul(pi, Rational(1, 4)))).is_positive
        second = add(mul(pi, Rational(1, 2)), _neg(a)).is_positive
        if first is True and second is True:
            return fmap[type(e)](add(mul(pi, Rational(1, 2)), _neg(a)))
        return e

    e0 = to_expr(rv)
    e0 = _expr(e0.replace(lambda x: isinstance(x, TRIG),
                          lambda x: _expr(x).replace(lambda n: isinstance(n, Mul) and _expr(n).is_number,
                                                     lambda n: mul(*[_expr(a) for a in n.args]))))
    return _bottom(e0, f)


def _TR56(rv: Expr, f: type, g: Callable[[Expr], Expr], h: Callable[[Expr], Expr], max_: int, pow_: bool) -> Expr:
    from sympy_extras.ntheory.factor_ import perfect_power
    rebuild: dict[type, Callable[[Expr], Expr]] = {Sin: sin, Cos: cos, Tan: tan, Cot: cot}

    def _f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, f)):
            return e
        ex = e.exp
        if ex.is_real is not True:
            return e
        from sympy_extras.core.relational import gt, lt
        if _true(lt(ex, Integer(0))):
            return e
        if _true(gt(ex, Integer(max_))):
            return e
        if ex == Integer(1):
            return e
        arg = _arg(e.base)
        if ex == Integer(2):
            return h(pow(g(arg), Integer(2)))
        if not isinstance(ex, Integer):
            return e
        if ex.p % 2 == 1:
            return mul(rebuild[f](arg), pow(h(pow(g(arg), Integer(2))), Integer(ex.p // 2)))
        if ex.p == 4:
            k = 2
        elif not pow_:
            k = ex.p // 2
        else:
            if perfect_power(ex.p) is None:
                return e
            k = ex.p // 2
        return pow(h(pow(g(arg), Integer(2))), Integer(k))

    return _bottom(rv, _f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR5
def TR5(rv: ExprLike, max: int = 4, pow: bool = False) -> Expr:
    """``sin(x)**n`` for even ``n <= max`` through ``1 - cos(x)**2``."""
    return _TR56(to_expr(rv), Sin, cos, lambda x: add(Integer(1), _neg(x)), max, pow)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR6
def TR6(rv: ExprLike, max: int = 4, pow: bool = False) -> Expr:
    """``cos(x)**n`` for even ``n <= max`` through ``1 - sin(x)**2``."""
    return _TR56(to_expr(rv), Cos, sin, lambda x: add(Integer(1), _neg(x)), max, pow)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR7
def TR7(rv: ExprLike) -> Expr:
    """``cos(x)**2`` as ``(1 + cos(2*x))/2``."""
    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, Cos) and e.exp == Integer(2)):
            return e
        return _half(add(Integer(1), cos(mul(Integer(2), _arg(e.base)))))
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR8
def TR8(rv: ExprLike, first: bool = True) -> Expr:
    """Products and powers of ``sin`` and ``cos`` as sums."""
    def f(e: Expr) -> Expr:
        if not (isinstance(e, Mul) or (isinstance(e, Pow) and isinstance(e.base, (Cos, Sin)) and (
                e.exp.is_integer is True or e.base.is_positive is True))):
            return e
        if first:
            n, d = [expand_mul(i) for i in e.as_numer_denom()]
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            if newn != n or newd != d:
                e = gcd_terms(mul(newn, _inv(newd)))
                if isinstance(e, Mul) and isinstance(e.args[0], Rational) and len(e.args) == 2 \
                        and isinstance(e.args[1], Add):
                    c, rest = e.as_coeff_Mul()
                    e = mul(c, rest)
            return e
        cs: list[Expr] = []
        ss: list[Expr] = []
        others: list[Expr] = []
        for a in factors_of(e):
            if isinstance(a, Cos):
                cs.append(_arg(a))
            elif isinstance(a, Sin):
                ss.append(_arg(a))
            elif isinstance(a, Pow) and isinstance(a.exp, Integer) and a.exp.p > 0 and isinstance(a.base, (Cos, Sin)):
                (cs if isinstance(a.base, Cos) else ss).extend([_arg(a.base)] * a.exp.p)
            else:
                others.append(a)
        if not ((cs and ss) or len(cs) > 1 or len(ss) > 1):
            return e
        args = others
        for _ in range(min(len(cs), len(ss))):
            a1, a2 = ss.pop(), cs.pop()
            args.append(_half(add(sin(add(a1, a2)), sin(add(a1, _neg(a2))))))
        while len(cs) > 1:
            a1, a2 = cs.pop(), cs.pop()
            args.append(_half(add(cos(add(a1, a2)), cos(add(a1, _neg(a2))))))
        if cs:
            args.append(cos(cs.pop()))
        while len(ss) > 1:
            a1, a2 = ss.pop(), ss.pop()
            args.append(_half(add(_neg(cos(add(a1, a2))), cos(add(a1, _neg(a2))))))
        if ss:
            args.append(sin(ss.pop()))
        return TR8(expand_mul(mul(*args)))

    return _bottom(to_expr(rv), f)


Split = tuple[Expr, int, int, Expr, Expr, bool]


# PORTED FROM SYMPY: sympy/simplify/fu.py:process_common_addends
def process_common_addends(rv: Expr, do: Rule, key2: Optional[Callable[[Expr], Hashable]] = None,
                           key1: bool = True) -> Expr:
    """``do`` applied to the groups of terms with the same absolute
    coefficient (and the same ``key2``)."""
    absc: dict[tuple[Expr, Hashable], list[Expr]] = {}
    if key1:
        for a0 in rv.args:
            number, a = _expr(a0).as_coeff_Mul()
            c: Expr = number
            if c.is_negative is True:
                c, a = _neg(c), _neg(a)
            absc.setdefault((c, key2(a) if key2 else Integer(1)), []).append(a)
    elif key2:
        for a0 in rv.args:
            absc.setdefault((Integer(1), key2(_expr(a0))), []).append(_expr(a0))
    else:
        raise ValueError('must have at least one key')
    args: list[Expr] = []
    hit = False
    for (coefficient, _), v in absc.items():
        if len(v) > 1:
            group: Expr = Add(*v)
            new = do(group)
            if new != group:
                group = new
                hit = True
            args.append(mul(coefficient, group))
        else:
            args.append(mul(coefficient, v[0]))
    if hit:
        return add(*args)
    return rv


CosSin = tuple[Optional[Expr], Optional[Expr], Optional[Expr]]


def _pow_cos_sin(a: Expr, two: bool) -> Optional[CosSin]:
    c: Optional[Expr] = None
    s: Optional[Expr] = None
    co: Expr = Integer(1)
    number: Expr
    if isinstance(a, Mul):
        number, a = a.as_coeff_Mul()
        co = number
        if len(a.args) > 2 or not two:
            return None
        args = [_expr(t) for t in a.args] if isinstance(a, Mul) else [a]
        first = args.pop(0)
        if isinstance(first, Cos):
            c = first
        elif isinstance(first, Sin):
            s = first
        elif isinstance(first, Pow) and first.exp == Rational(1, 2):
            co = mul(co, first)
        else:
            return None
        if args:
            b = args[0]
            if isinstance(b, Cos):
                if c is not None:
                    s = b
                else:
                    c = b
            elif isinstance(b, Sin):
                if s is not None:
                    c = b
                else:
                    s = b
            elif isinstance(b, Pow) and b.exp == Rational(1, 2):
                co = mul(co, b)
            else:
                return None
        return (co if co != Integer(1) else None), c, s
    if isinstance(a, Cos):
        c = a
    elif isinstance(a, Sin):
        s = a
    if c is None and s is None:
        return None
    return (co if co != Integer(1) else None), c, s


# PORTED FROM SYMPY: sympy/simplify/fu.py:trig_split
def trig_split(a0: Expr, b0: Expr, two: bool = False) -> Optional[Split]:
    """``(gcd, n1, n2, a, b, iscos)`` when ``a0 + b0`` is
    ``gcd*(n1*f(a) + n2*f(b))`` with ``f`` both ``cos`` (``iscos``) or both
    ``sin`` (with ``two``, also products of a ``cos`` and a ``sin``, and
    the radical coefficients of ``sin(x + pi/4)`` and its kin)."""
    fa, fb = ExprFactors.of(a0), ExprFactors.of(b0)
    ua, ub = fa.normal(fb)
    gcd = fa.gcd(fb).as_expr()
    n1 = n2 = 1
    if Integer(-1) in ua.factors:
        ua = ua.quo(ExprFactors({Integer(-1): Integer(1)}))
        n1 = -n1
    elif Integer(-1) in ub.factors:
        ub = ub.quo(ExprFactors({Integer(-1): Integer(1)}))
        n2 = -n2
    a, b = ua.as_expr(), ub.as_expr()
    m = _pow_cos_sin(a, two)
    if m is None:
        return None
    coa, ca, sa = m
    m = _pow_cos_sin(b, two)
    if m is None:
        return None
    cob, cb, sb = m
    if (ca is None and cb is not None) or (ca is not None and isinstance(ca, Sin)):
        coa, ca, sa, cob, cb, sb = cob, cb, sb, coa, ca, sa
        n1, n2 = n2, n1
    if not two:
        c = ca if ca is not None else sa
        s = cb if cb is not None else sb
        if c is None or s is None or type(c) is not type(s):
            return None
        return gcd, n1, n2, _arg(c), _arg(s), isinstance(c, Cos)
    if coa is None and cob is None:
        if ca is not None and cb is not None and sa is not None and sb is not None:
            if (type(ca) is type(sa)) is not (type(cb) is type(sb)):
                return None
            args = {ca.args, sa.args}
            if not all(i.args in args for i in (cb, sb)):
                return None
            return gcd, n1, n2, _arg(ca), _arg(sa), type(ca) is type(sa)
    if (ca is not None and sa is not None) or (cb is not None and sb is not None) or (
            two and ((ca is None and sa is None) or (cb is None and sb is None))):
        return None
    c = ca if ca is not None else sa
    s = cb if cb is not None else sb
    if c is None or s is None or c.args != s.args:
        return None
    coa = coa if coa is not None else Integer(1)
    cob = cob if cob is not None else Integer(1)
    if coa == cob:
        return mul(gcd, _ROOT2()), n1, n2, _arg(c), mul(pi, Rational(1, 4)), False
    if mul(coa, _inv(cob)) == _ROOT3():
        return mul(gcd, Integer(2), cob), n1, n2, _arg(c), mul(pi, Rational(1, 3)), False
    if mul(coa, _inv(cob)) == _invROOT3():
        return mul(gcd, Integer(2), coa), n1, n2, _arg(c), mul(pi, Rational(1, 6)), False
    return None


# PORTED FROM SYMPY: sympy/simplify/fu.py:as_f_sign_1
def as_f_sign_1(e: Expr) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(g, f, s)`` with ``e = g*(f + s)`` and ``s`` equal to ``1`` or
    ``-1``, for a sum of two terms."""
    if not isinstance(e, Add) or len(e.args) != 2:
        return None
    a, b = _expr(e.args[0]), _expr(e.args[1])
    if a in (Integer(-1), Integer(1)):
        g: Expr = Integer(1)
        if isinstance(b, Mul) and isinstance(b.args[0], Number) and b.args[0].is_negative is True:
            a, b = _neg(a), _neg(b)
            g = Integer(-1)
        return g, b, a
    fa, fb = ExprFactors.of(a), ExprFactors.of(b)
    ua, ub = fa.normal(fb)
    gcd = fa.gcd(fb).as_expr()
    if Integer(-1) in ua.factors:
        ua = ua.quo(ExprFactors({Integer(-1): Integer(1)}))
        n1, n2 = -1, 1
    elif Integer(-1) in ub.factors:
        ub = ub.quo(ExprFactors({Integer(-1): Integer(1)}))
        n1, n2 = 1, -1
    else:
        n1 = n2 = 1
    a, b = ua.as_expr(), ub.as_expr()
    if a == Integer(1):
        a, b = b, a
        n1, n2 = n2, n1
    if n1 == -1:
        gcd = _neg(gcd)
        n2 = -n2
    if b == Integer(1):
        return gcd, a, Integer(n2)
    return None


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR9
def TR9(rv: ExprLike) -> Expr:
    """Sums of ``cos`` or of ``sin`` as products
    (``cos(a) + cos(b) = 2*cos((a + b)/2)*cos((a - b)/2)``)."""
    def do(e: Expr, first: bool = True) -> Expr:
        if not isinstance(e, Add):
            return e
        args: list[Optional[Expr]] = [_expr(a) for a in ordered(e.args)]
        if len(args) != 2:
            hit = False
            for i in range(len(args)):
                ai = args[i]
                if ai is None:
                    continue
                for j in range(i + 1, len(args)):
                    aj = args[j]
                    if aj is None:
                        continue
                    was = add(ai, aj)
                    new = do(was)
                    if new != was:
                        args[i] = new
                        args[j] = None
                        hit = True
                        break
            if hit:
                e = add(*[t for t in args if t is not None])
                if isinstance(e, Add):
                    e = do(e)
            return e
        first_arg, second_arg = args[0], args[1]
        if first_arg is None or second_arg is None:
            return e
        split = trig_split(first_arg, second_arg)
        if not split:
            return e
        gcd, n1, n2, a, b, iscos = split
        s_ab, d_ab = _half(add(a, b)), _half(add(a, _neg(b)))
        if iscos:
            if n1 == n2:
                return mul(gcd, Integer(n1), Integer(2), cos(s_ab), cos(d_ab))
            if n1 < 0:
                a, b = b, a
                s_ab, d_ab = _half(add(a, b)), _half(add(a, _neg(b)))
            return mul(Integer(-2), gcd, sin(s_ab), sin(d_ab))
        if n1 == n2:
            return mul(gcd, Integer(n1), Integer(2), sin(s_ab), cos(d_ab))
        if n1 < 0:
            a, b = b, a
            s_ab, d_ab = _half(add(a, b)), _half(add(a, _neg(b)))
        return mul(Integer(2), gcd, cos(s_ab), sin(d_ab))

    def f(e: Expr) -> Expr:
        if not isinstance(e, Add):
            return e
        return process_common_addends(e, do)

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:_osborne
def _osborne(e: Expr, d: Expr) -> Expr:
    def f(rv: Expr) -> Expr:
        if not isinstance(rv, HYPER):
            return rv
        a = _arg(rv)
        a = mul(a, d) if not isinstance(a, Add) else Add(*[mul(_expr(t), d) for t in a.args])
        if isinstance(rv, Sinh):
            return mul(I, sin(a))
        if isinstance(rv, Cosh):
            return cos(a)
        if isinstance(rv, Tanh):
            return mul(I, tan(a))
        if isinstance(rv, Coth):
            return mul(cot(a), _inv(I))
        if isinstance(rv, Sech):
            return sec(a)
        return mul(csc(a), _inv(I))
    return _bottom(e, f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:_osbornei
def _osbornei(e: Expr, d: Expr) -> Expr:
    def f(rv: Expr) -> Expr:
        if not isinstance(rv, TRIG):
            return rv
        const, x = _arg(rv).as_independent(d, as_Add=True)
        a = add(_expr(x.xreplace({d: Integer(1)})), mul(const, I))
        if isinstance(rv, Sin):
            return mul(sinh(a), _inv(I))
        if isinstance(rv, Cos):
            return cosh(a)
        if isinstance(rv, Tan):
            return mul(tanh(a), _inv(I))
        if isinstance(rv, Cot):
            return mul(coth(a), I)
        if isinstance(rv, Sec):
            return sech(a)
        return mul(csch(a), I)
    return _bottom(e, f)


Inverse = Callable[[Expr], Expr]


# PORTED FROM SYMPY: sympy/simplify/fu.py:hyper_as_trig
def hyper_as_trig(rv: ExprLike) -> tuple[Expr, Inverse]:
    """``rv`` with the hyperbolic functions written as trigonometric ones
    (the trigonometric functions masked), and the inverse map."""
    from .radsimp import collect
    from .signsimp import signsimp
    e = to_expr(rv)
    trigs = [t for t in e.atoms(*TRIG)]
    reps: dict[Basic, Basic] = {t: Dummy() for t in trigs}
    masked = _expr(e.xreplace(reps))
    back: dict[Basic, Basic] = {v: k for k, v in reps.items()}
    d = Dummy()

    def inverse(x: Expr) -> Expr:
        return collect(signsimp(_expr(_osbornei(x, d)._xreplace_evaluated(back))), I)

    return _osborne(masked, d), inverse


# ---------------------------------------------------------------------
# part 2

_REBUILD: dict[type, Callable[[Expr], Expr]] = {Sin: sin, Cos: cos, Tan: tan, Cot: cot, Sec: sec, Csc: csc}


def _ordered_exprs(args: tuple[Basic, ...]) -> list[Expr]:
    return [_expr(a) for a in ordered(list(args))]


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR4
def TR4(rv: ExprLike) -> Expr:
    """The values at the special angles (multiples of ``pi/6`` and
    ``pi/4``) evaluated."""
    def query(x: Basic) -> bool:
        if not isinstance(x, TRIG):
            return False
        r = mul(_arg(x), _inv(pi))
        return isinstance(r, Rational) and r.q in (1, 2, 3, 4, 6)

    def value(x: Basic) -> Basic:
        e = _expr(x)
        arg = _arg(e)
        return _REBUILD[type(e)](_expr(arg._evaluated(arg.args)) if arg.args else arg)

    return _expr(to_expr(rv).replace(query, value))


def _split_last(arg: Add, first: bool) -> tuple[Expr, Expr]:
    args = _ordered_exprs(arg.args) if first else [_expr(a) for a in arg.args]
    a = args.pop()
    b = Add(*args) if len(args) > 1 else args[0]
    return a, b


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR10
def TR10(rv: ExprLike, first: bool = True) -> Expr:
    """``sin`` and ``cos`` of sums expanded (addition formulas)."""
    def f(e: Expr) -> Expr:
        if not isinstance(e, (Cos, Sin)):
            return e
        arg = _arg(e)
        if not isinstance(arg, Add):
            return e
        a, b = _split_last(arg, first)
        if isinstance(b, Add):
            if isinstance(e, Sin):
                return add(mul(sin(a), TR10(cos(b), first=False)), mul(cos(a), TR10(sin(b), first=False)))
            return add(mul(cos(a), TR10(cos(b), first=False)), _neg(mul(sin(a), TR10(sin(b), first=False))))
        if isinstance(e, Sin):
            return add(mul(sin(a), cos(b)), mul(cos(a), sin(b)))
        return add(mul(cos(a), cos(b)), _neg(mul(sin(a), sin(b))))
    return _bottom(to_expr(rv), f)


def _pair_rule(do: Callable[[Expr], Expr]) -> Callable[[Expr], Expr]:
    """The pairwise search of TR9/TR10i over the terms of a sum."""
    def search(e: Expr) -> Expr:
        if not isinstance(e, Add):
            return e
        args: list[Optional[Expr]] = list(_ordered_exprs(e.args))
        hit = False
        for i in range(len(args)):
            ai = args[i]
            if ai is None:
                continue
            for j in range(i + 1, len(args)):
                aj = args[j]
                if aj is None:
                    continue
                was = add(ai, aj)
                new = do(was)
                if new != was:
                    args[i] = new
                    args[j] = None
                    hit = True
                    break
        if hit:
            e = add(*[t for t in args if t is not None])
            if isinstance(e, Add):
                e = do(e)
        return e
    return search


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR10i
def TR10i(rv: ExprLike) -> Expr:
    """Sums of products of ``sin`` and ``cos`` as ``sin``/``cos`` of sums
    (the inverse of TR10)."""
    def do(e: Expr, first: bool = True) -> Expr:
        if not isinstance(e, Add):
            return e
        if len(e.args) != 2:
            return _pair_rule(do)(e)
        a0, b0 = _ordered_exprs(e.args)
        split = trig_split(a0, b0, two=True)
        if not split:
            return e
        gcd, n1, n2, a, b, same = split
        gcd = mul(Integer(n1), gcd)
        if same:
            if n1 == n2:
                return mul(gcd, cos(add(a, _neg(b))))
            return mul(gcd, cos(add(a, b)))
        if n1 == n2:
            return mul(gcd, sin(add(a, b)))
        return mul(gcd, sin(add(b, _neg(a))))

    def f(e: Expr) -> Expr:
        if not isinstance(e, Add):
            return e
        e = process_common_addends(e, do, lambda x: tuple(ordered(list(x.free_symbols))))
        while isinstance(e, Add):
            byrad: dict[Expr, list[Optional[Expr]]] = {}
            for a0 in e.args:
                a = _expr(a0)
                hit = False
                if isinstance(a, Mul):
                    for ai in a.args:
                        if isinstance(ai, Pow) and ai.exp == Rational(1, 2) and isinstance(ai.base, Integer):
                            byrad.setdefault(ai, []).append(a)
                            hit = True
                            break
                if not hit:
                    byrad.setdefault(Integer(1), []).append(a)
            args: list[Expr] = []
            for a in list(byrad):
                for b in (mul(_ROOT3(), a), _invROOT3()):
                    if b in byrad:
                        for i in range(len(byrad[a])):
                            item = byrad[a][i]
                            if item is None:
                                continue
                            for j in range(len(byrad[b])):
                                other_item = byrad[b][j]
                                if other_item is None:
                                    continue
                                was = add(item, other_item)
                                new = do(was)
                                if new != was:
                                    args.append(new)
                                    byrad[a][i] = None
                                    byrad[b][j] = None
                                    break
            if args:
                e = add(*(args + [add(*[t for t in v if t is not None]) for v in byrad.values()]))
            else:
                e = do(e)
                break
        return e

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR11
def TR11(rv: ExprLike, base: Optional[Expr] = None) -> Expr:
    """Double angles reduced (``sin(2*x)`` is ``2*sin(x)*cos(x)``); with
    ``base``, only the functions of ``2*base``."""
    def f(e: Expr) -> Expr:
        if not isinstance(e, (Cos, Sin)):
            return e
        if base is not None:
            func = _REBUILD[type(e)]
            t = func(mul(base, Integer(2)))
            co: Expr = Integer(1)
            if isinstance(t, Mul):
                number, t = t.as_coeff_Mul()
                co = number
            if not isinstance(t, (Cos, Sin)):
                return e
            if _arg(e) == _arg(t):
                c, s_ = cos(base), sin(base)
                if isinstance(e, Cos):
                    return mul(add(pow(c, Integer(2)), _neg(pow(s_, Integer(2)))), _inv(co))
                return mul(Integer(2), c, s_, _inv(co))
            return e
        if not isinstance(_arg(e), Number):
            c, m = _arg(e).as_coeff_Mul(rational=True)
            if isinstance(c, Rational) and c.p % 2 == 0:
                arg = mul(Integer(c.p // 2), m, Rational(1, c.q) if c.q != 1 else Integer(1))
                cc, ss = TR11(cos(arg)), TR11(sin(arg))
                if isinstance(e, Sin):
                    return mul(Integer(2), ss, cc)
                return add(pow(cc, Integer(2)), _neg(pow(ss, Integer(2))))
        return e
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:_TR11
def _TR11(rv: ExprLike) -> Expr:
    """TR11 on the half angles which appear in the other part of a
    fraction."""
    def sincos_args(flat: Expr) -> dict[type, set[Expr]]:
        found: dict[type, set[Expr]] = {Cos: set(), Sin: set()}
        for fi in factors_of(flat):
            b, ex = fi.as_base_exp()
            if isinstance(ex, Integer) and ex.p > 0 and isinstance(b, (Cos, Sin)):
                found[type(b)].add(_arg(b))
        return found

    def f(e: Expr) -> Expr:
        n, d = e.as_numer_denom()
        num_args, den_args = sincos_args(n), sincos_args(d)

        def handle(e: Expr, num: dict[type, set[Expr]], den: dict[type, set[Expr]]) -> Expr:
            for narg in list(num[Sin]):
                half_arg = _half(narg)
                if half_arg in den[Cos]:
                    kind: type = Cos
                elif half_arg in den[Sin]:
                    kind = Sin
                else:
                    continue
                e = TR11(e, half_arg)
                den[kind].remove(half_arg)
            return e

        e = handle(e, num_args, den_args)
        return handle(e, den_args, num_args)

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR12
def TR12(rv: ExprLike, first: bool = True) -> Expr:
    """``tan`` of sums expanded."""
    def f(e: Expr) -> Expr:
        if not isinstance(e, Tan):
            return e
        arg = _arg(e)
        if not isinstance(arg, Add):
            return e
        a, b = _split_last(arg, first)
        tb = TR12(tan(b), first=False) if isinstance(b, Add) else tan(b)
        return mul(add(tan(a), tb), _inv(add(Integer(1), _neg(mul(tan(a), tb)))))
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR12i
def TR12i(rv: ExprLike) -> Expr:
    """``(tan(a) + tan(b))/(1 - tan(a)*tan(b))`` as ``tan(a + b)``."""
    from sympy_extras.polys.polytools import factor

    def ok_d(di: Expr) -> Optional[tuple[Expr, Expr]]:
        m = as_f_sign_1(di)
        if m:
            g, f_, s_ = m
            if s_ == Integer(-1) and isinstance(f_, Mul) and len(f_.args) == 2 and all(
                    isinstance(fi, Tan) for fi in f_.args):
                return g, f_
        return None

    def ok_n(ni: Expr) -> Optional[tuple[Expr, Expr]]:
        if isinstance(ni, Add) and len(ni.args) == 2:
            a, b = _expr(ni.args[0]), _expr(ni.args[1])
            if isinstance(a, Tan) and isinstance(b, Tan):
                return a, b
        return None

    def f(e: Expr) -> Expr:
        if not isinstance(e, (Add, Mul, Pow)):
            return e
        n, d = e.as_numer_denom()
        if not d.args or not n.args:
            return e
        dok: dict[Expr, Expr] = {}
        d_args: list[Expr] = list(factors_of(d))
        i = 0
        while i < len(d_args):
            di = d_args[i]
            m = ok_d(di)
            if m:
                g, t = m
                dok[add(*[_arg(_expr(x)) for x in t.args])] = Integer(1)
                d_args[i] = g
            elif isinstance(di, Add):
                fd = factor(di)
                if isinstance(fd, Mul):
                    d_args.extend(_expr(a) for a in fd.args)
                    d_args[i] = Integer(1)
            elif isinstance(di, Pow) and (di.exp.is_integer is True or di.base.is_positive is True):
                m = ok_d(di.base)
                if m:
                    g, t = m
                    dok[add(*[_arg(_expr(x)) for x in t.args])] = di.exp
                    d_args[i] = pow(g, di.exp)
                else:
                    fd = factor(di)
                    if isinstance(fd, Mul):
                        d_args.extend(_expr(a) for a in fd.args)
                        d_args[i] = Integer(1)
            i += 1
        if not dok:
            return e
        n_args: list[Expr] = list(factors_of(factor_terms(n)))
        hit = False
        i = 0
        while i < len(n_args):
            ni = n_args[i]
            m = ok_n(ni)
            if not m:
                m = ok_n(_neg(ni))
                if m:
                    n_args[i] = Integer(-1)
                else:
                    if isinstance(ni, Add):
                        fn = factor(ni)
                        if isinstance(fn, Mul):
                            n_args.extend(_expr(a) for a in fn.args)
                            n_args[i] = Integer(1)
                        i += 1
                        continue
                    if isinstance(ni, Pow) and (ni.exp.is_integer is True or ni.base.is_positive is True):
                        m = ok_n(ni.base)
                        if m:
                            n_args[i] = Integer(1)
                        else:
                            fn = factor(ni)
                            if isinstance(fn, Mul):
                                n_args.extend(_expr(a) for a in fn.args)
                                n_args[i] = Integer(1)
                            i += 1
                            continue
                    else:
                        i += 1
                        continue
            else:
                n_args[i] = Integer(1)
            hit = True
            s_ = add(*[_arg(t) for t in m])
            ed = dok[s_]
            newed = ed.extract_additively(Integer(1))
            if newed is not None:
                if not (isinstance(newed, Number) and newed == 0):
                    dok[s_] = newed
                else:
                    dok.pop(s_)
            n_args[i] = mul(n_args[i], _neg(tan(s_)))
            i += 1
        if hit:
            denominator = mul(*[pow(add(add(*[tan(_expr(a)) for a in (k.args if isinstance(k, Add) else (k,))]),
                                        Integer(-1)), ex) for k, ex in dok.items()])
            return mul(mul(*n_args), _inv(mul(*d_args)), _inv(denominator))
        return e

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR13
def TR13(rv: ExprLike) -> Expr:
    """Products of ``tan`` (``cot``) as sums through ``tan(a + b)``."""
    def f(e: Expr) -> Expr:
        if not isinstance(e, Mul):
            return e
        ts: list[Expr] = []
        cs: list[Expr] = []
        args: list[Expr] = []
        for a in factors_of(e):
            if isinstance(a, Tan):
                ts.append(_arg(a))
            elif isinstance(a, Cot):
                cs.append(_arg(a))
            else:
                args.append(a)
        if len(ts) < 2 and len(cs) < 2:
            return e
        while len(ts) > 1:
            t1, t2 = ts.pop(), ts.pop()
            s12 = tan(add(t1, t2))
            args.append(add(Integer(1), _neg(add(mul(tan(t1), _inv(s12)), mul(tan(t2), _inv(s12))))))
        if ts:
            args.append(tan(ts.pop()))
        while len(cs) > 1:
            t1, t2 = cs.pop(), cs.pop()
            c12 = cot(add(t1, t2))
            args.append(add(Integer(1), mul(cot(t1), c12), mul(cot(t2), c12)))
        if cs:
            args.append(cot(cs.pop()))
        return mul(*args)
    return _bottom(to_expr(rv), f)


def _value(n: Expr) -> float:
    from sympy_extras.core.numbers import Float
    if isinstance(n, (Rational, Float)):
        return float(n.as_fraction())
    value = n.evalf(15)
    return float(value.as_fraction()) if isinstance(value, (Rational, Float)) else 0.0


# PORTED FROM SYMPY: sympy/simplify/fu.py:TRmorrie
def TRmorrie(rv: ExprLike) -> Expr:
    """Morrie's law: ``cos(x)*cos(2*x)*...*cos(2**(k-1)*x)`` is
    ``sin(2**k*x)/(2**k*sin(x))``."""
    def f(e: Expr, first: bool = True) -> Expr:
        if not isinstance(e, Mul):
            return e
        if first:
            n, d = e.as_numer_denom()
            return mul(f(n, False), _inv(f(d, False)))
        args: dict[Expr, list[Expr]] = {}
        coss: dict[Expr, int] = {}
        other: list[Expr] = []
        for c0 in e.args:
            c = _expr(c0)
            b, ex = c.as_base_exp()
            if isinstance(ex, Integer) and isinstance(b, Cos):
                co, a = _arg(b).as_coeff_Mul()
                args.setdefault(a, []).append(co)
                coss[b] = ex.p
            else:
                other.append(c)
        new: list[Expr] = []
        for a in args:
            cl = args[a]
            cl.sort(key=_value)
            while cl:
                k = 0
                cc = ci = cl[0]
                while cc in cl:
                    k += 1
                    cc = mul(cc, Integer(2))
                if k > 1:
                    newarg = mul(sin(mul(Integer(2**k), ci, a)), Rational(1, 2**k), _inv(sin(mul(ci, a))))
                    take: Optional[int] = None
                    ccs: list[Expr] = []
                    for _ in range(k):
                        cc = mul(cc, Rational(1, 2))
                        key: Expr = Cos(mul(a, cc))
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    for _ in range(k):
                        cc = ccs.pop()
                        key = Cos(mul(a, cc))
                        coss[key] -= take or 0
                        if not coss[key]:
                            cl.remove(cc)
                    new.append(pow(newarg, Integer(take or 0)))
                else:
                    b = cos(mul(cl.pop(0), a))
                    other.append(pow(b, Integer(coss[b])))
        if new:
            return mul(*(new + other + [Cos(mul(k, a)) for a in args for k in args[a]]))
        return e

    return _bottom(to_expr(rv), lambda e: f(e))


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR14
def TR14(rv: ExprLike, first: bool = True) -> Expr:
    """Factored powers of ``sin``/``cos`` identities:
    ``(1 - cos(x))*(1 + cos(x))`` is ``sin(x)**2``."""
    from sympy_extras.core.logic import false, true

    def f(e: Expr) -> Expr:
        if not isinstance(e, Mul):
            return e
        if first:
            n, d = e.as_numer_denom()
            if d != Integer(1):
                newn, newd = TR14(n, first=False), TR14(d, first=False)
                if newn != n or newd != d:
                    e = mul(newn, _inv(newd))
                return e
        other: list[Expr] = []
        process: list[tuple[Expr, bool, Expr, Expr, Expr, Expr]] = []
        for a0 in e.args:
            a = _expr(a0)
            if isinstance(a, Pow):
                b, ex = a.as_base_exp()
                if not (ex.is_integer is True or b.is_positive is True):
                    other.append(a)
                    continue
                a = b
            else:
                ex = Integer(1)
            m = as_f_sign_1(a)
            if not m or not isinstance(m[1], (Cos, Sin)):
                other.append(a if ex == Integer(1) else pow(a, ex))
                continue
            g, fn, si = m
            process.append((g, isinstance(ex, Number), ex, fn, si, a))
        wrapped = {Tuple(g, true if isn else false, ex, fn, si, a): _Item(g, ex, fn, si, a)
                   for g, isn, ex, fn, si, a in process}
        queue = [wrapped[_tuple(t)] for t in ordered(list(wrapped))]
        nother = len(other)
        while queue:
            A = queue.pop(0)
            if queue:
                B = queue[0]
                if isinstance(A.e, Number) and isinstance(B.e, Number):
                    if A.f == B.f and A.si != B.si:
                        B = queue.pop(0)
                        take = A.e if _value(A.e) <= _value(B.e) else B.e
                        if B.e != take:
                            queue.insert(0, _Item(B.g, add(B.e, _neg(take)), B.f, B.si, B.a))
                        elif A.e != take:
                            queue.insert(0, _Item(A.g, add(A.e, _neg(take)), A.f, A.si, A.a))
                        t = sin if isinstance(A.f, Cos) else cos
                        other.append(pow(mul(_neg(A.g), B.g, pow(t(_arg(A.f)), Integer(2))), take))
                        continue
                elif A.e == B.e:
                    if A.f == B.f and A.si != B.si:
                        B = queue.pop(0)
                        t = sin if isinstance(A.f, Cos) else cos
                        other.append(pow(mul(_neg(A.g), B.g, pow(t(_arg(A.f)), Integer(2))), A.e))
                        continue
            other.append(pow(A.a, A.e))
        if len(other) != nother:
            return mul(*other)
        return e

    return _bottom(to_expr(rv), f)


class _Item:
    """A factor ``g*(f + si)**e`` of TR14 (the base ``a``)."""

    __slots__ = ('g', 'e', 'f', 'si', 'a')

    def __init__(self, g: Expr, e: Expr, f: Expr, si: Expr, a: Expr) -> None:
        self.g, self.e, self.f, self.si, self.a = g, e, f, si, a


def _tuple(b: Basic) -> Tuple:
    if not isinstance(b, Tuple):
        raise TypeError('a Tuple is expected')
    return b


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR15
def TR15(rv: ExprLike, max: int = 4, pow: bool = False) -> Expr:
    """Negative even powers of ``sin`` through ``1 + cot**2``."""
    from sympy_extras.core.power import pow as power

    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, Sin)):
            return e
        ex = e.exp
        if isinstance(ex, Integer) and ex.p % 2 == 1:
            return mul(TR15(power(e.base, Integer(ex.p + 1))), _inv(e.base))
        ia = _inv(e)
        a = _TR56(ia, Sin, cot, lambda x: add(Integer(1), x), max, pow)
        return a if a != ia else e
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR16
def TR16(rv: ExprLike, max: int = 4, pow: bool = False) -> Expr:
    """Negative even powers of ``cos`` through ``1 + tan**2``."""
    from sympy_extras.core.power import pow as power

    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, Cos)):
            return e
        ex = e.exp
        if isinstance(ex, Integer) and ex.p % 2 == 1:
            return mul(TR15(power(e.base, Integer(ex.p + 1))), _inv(e.base))
        ia = _inv(e)
        a = _TR56(ia, Cos, tan, lambda x: add(Integer(1), x), max, pow)
        return a if a != ia else e
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR111
def TR111(rv: ExprLike) -> Expr:
    """Negative powers of ``tan``, ``sin``, ``cos`` as ``cot``, ``csc``,
    ``sec``."""
    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and (e.base.is_positive is True or (
                e.exp.is_integer is True and e.exp.is_negative is True))):
            return e
        if isinstance(e.base, Tan):
            return pow(cot(_arg(e.base)), _neg(e.exp))
        if isinstance(e.base, Sin):
            return pow(csc(_arg(e.base)), _neg(e.exp))
        if isinstance(e.base, Cos):
            return pow(sec(_arg(e.base)), _neg(e.exp))
        return e
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TR22
def TR22(rv: ExprLike, max: int = 4, pow: bool = False) -> Expr:
    """Even powers of ``tan``/``cot`` through ``sec**2 - 1``/``csc**2 - 1``."""
    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, (Cot, Tan))):
            return e
        e = _TR56(e, Tan, sec, lambda x: add(x, Integer(-1)), max, pow)
        return _TR56(e, Cot, csc, lambda x: add(x, Integer(-1)), max, pow)
    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:TRpower
def TRpower(rv: ExprLike) -> Expr:
    """Positive integer powers of ``sin``/``cos`` as sums of multiple
    angles."""
    from sympy_extras.functions import binomial

    def f(e: Expr) -> Expr:
        if not (isinstance(e, Pow) and isinstance(e.base, (Sin, Cos))):
            return e
        b, n0 = e.as_base_exp()
        if not (isinstance(n0, Integer) and n0.p > 0):
            return e
        n = n0.p
        x = _arg(b)
        scale = pow(Integer(2), Integer(1 - n))
        if n % 2 == 1 and isinstance(b, Cos):
            e = mul(scale, add(*[mul(binomial(n, k), cos(mul(Integer(n - 2 * k), x))) for k in range((n + 1) // 2)]))
        elif n % 2 == 1:
            e = mul(scale, Integer((-1) ** ((n - 1) // 2)), add(*[mul(binomial(n, k), Integer((-1) ** k),
                                                                    sin(mul(Integer(n - 2 * k), x)))
                                                                for k in range((n + 1) // 2)]))
        elif isinstance(b, Cos):
            e = mul(scale, add(*[mul(binomial(n, k), cos(mul(Integer(n - 2 * k), x))) for k in range(n // 2)]))
        else:
            e = mul(scale, Integer((-1) ** (n // 2)), add(*[mul(binomial(n, k), Integer((-1) ** k),
                                                              cos(mul(Integer(n - 2 * k), x))) for k in range(n // 2)]))
        if n % 2 == 0:
            e = add(e, mul(pow(Integer(2), Integer(-n)), binomial(n, n // 2)))
        return e

    return _bottom(to_expr(rv), f)


# PORTED FROM SYMPY: sympy/simplify/fu.py:L
def L(rv: ExprLike) -> int:
    """The number of trigonometric functions in ``rv``."""
    from sympy_extras.core.basic import preorder_traversal
    return sum(1 for node in preorder_traversal(to_expr(rv)) if isinstance(node, TRIG))


# PORTED FROM SYMPY: sympy/simplify/fu.py:sincos_to_sum
def sincos_to_sum(expr: ExprLike) -> Expr:
    e = to_expr(expr)
    if not e.has(Sin, Cos):
        return e
    return TR8(expand_mul(TRpower(e)))


# ---------------------------------------------------------------------
# the strategies

class Chain:
    """Rules applied one after the other (SymPy's tuple in a tree)."""

    __slots__ = ('rules',)

    def __init__(self, *rules: 'Tree') -> None:
        self.rules: tuple[Tree, ...] = rules


class Minimize:
    """The best result of the alternatives (SymPy's list in a tree)."""

    __slots__ = ('rules',)

    def __init__(self, *rules: 'Tree') -> None:
        self.rules: tuple[Tree, ...] = rules


Tree = Union[Rule, Chain, Minimize]
MeasureKey = tuple[int, ...]
Measure = Callable[[Expr], MeasureKey]


# PORTED FROM SYMPY: sympy/strategies/tree.py:greedy (treeapply with minimize and chain)
def greedy(tree: Tree, objective: Measure) -> Rule:
    """The rule of a tree: a chain composes, a minimisation keeps the
    result of least ``objective``."""
    if isinstance(tree, Chain):
        rules = [greedy(t, objective) for t in tree.rules]

        def chained(e: Expr) -> Expr:
            for r in rules:
                e = r(e)
            return e
        return chained
    if isinstance(tree, Minimize):
        alternatives = [greedy(t, objective) for t in tree.rules]

        def minimized(e: Expr) -> Expr:
            return min((r(e) for r in alternatives), key=objective)
        return minimized
    return tree


def _identity(e: Expr) -> Expr:
    return e


def _rule(r: Callable[[Expr], Expr]) -> Rule:
    return r


CTR1 = Minimize(Chain(_rule(TR5), _rule(TR0)), Chain(_rule(TR6), _rule(TR0)), _identity)
CTR2 = Chain(_rule(TR11), Minimize(Chain(_rule(TR5), _rule(TR0)), Chain(_rule(TR6), _rule(TR0)), _rule(TR0)))
CTR3 = Minimize(Chain(_rule(TRmorrie), _rule(TR8), _rule(TR0)), Chain(_rule(TRmorrie), _rule(TR8), _rule(TR10i),
                                                                       _rule(TR0)), _identity)
CTR4 = Minimize(Chain(_rule(TR4), _rule(TR10i)), _identity)
RL1 = Chain(_rule(TR4), _rule(TR3), _rule(TR4), _rule(TR12), _rule(TR4), _rule(TR13), _rule(TR4), _rule(TR0))
RL2 = Minimize(
    Chain(_rule(TR4), _rule(TR3), _rule(TR10), _rule(TR4), _rule(TR3), _rule(TR11)),
    Chain(_rule(TR5), _rule(TR7), _rule(TR11), _rule(TR4)),
    Chain(CTR3, CTR1, _rule(TR9), CTR2, _rule(TR4), _rule(TR9), _rule(TR9), CTR4),
    _identity)


def _fu_measure(e: Expr) -> MeasureKey:
    from sympy_extras.core.function import count_ops
    ops = count_ops(e)
    return (L(e), ops.p if isinstance(ops, Integer) else 0)


# PORTED FROM SYMPY: sympy/simplify/fu.py:fu
def fu(rv: ExprLike, measure: Measure = _fu_measure) -> Expr:
    """The trigonometric simplification of Fu, Zhong and Zeng: the rule
    lists RL1 (tangents) and RL2 (sines and cosines) applied greedily for
    the fewest trigonometric functions, then the fewest operations.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import cos, sin, tan
    >>> x = Symbol('x')
    >>> fu(sin(x)**2 + cos(x)**2), fu(sin(x)/cos(x)), fu(1 - 2*sin(x)**2)
    (1, tan(x), cos(2*x))
    """
    fRL1 = greedy(RL1, measure)
    fRL2 = greedy(RL2, measure)
    was = to_expr(rv)
    e = TR1(was)
    if e.has(Tan, Cot):
        e1 = fRL1(e)
        if measure(e1) < measure(e):
            e = e1
        if e.has(Tan, Cot):
            e = TR2(e)
    if e.has(Sin, Cos):
        e1 = fRL2(e)
        e2 = TR8(TRmorrie(e1))
        e = min([was, e, e1, e2], key=measure)
    return min(TR2i(e), e, key=measure)
