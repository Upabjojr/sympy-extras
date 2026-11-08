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

from typing import Callable, Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
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

__all__ = ['TR0', 'TR1', 'TR2', 'TR2i', 'TR3', 'TR5', 'TR6', 'TR7', 'TR8', 'TR9', 'trig_split', 'as_f_sign_1',
           'process_common_addends', 'hyper_as_trig']

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
    rebuild: dict[type, Callable[[Expr], Expr]] = {Sin: sin, Cos: cos}

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
def process_common_addends(rv: Expr, do: Rule, key2: Optional[Callable[[Expr], Expr]] = None,
                           key1: bool = True) -> Expr:
    """``do`` applied to the groups of terms with the same absolute
    coefficient (and the same ``key2``)."""
    absc: dict[tuple[Expr, Expr], list[Expr]] = {}
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

