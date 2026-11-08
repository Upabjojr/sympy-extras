"""``sqrtdenest``: square roots of sums of square roots denested where
possible (``sqrt(5 + 2*sqrt(6))`` is ``sqrt(2) + sqrt(3)``).

Ported from SymPy's ``sympy.simplify.sqrtdenest`` (Borodin, Fagin,
Hopcroft and Tompa; Landau's algorithm for the recursive case;
biquadratic and symbolic denestings): ``sqrt(a + b*sqrt(r))`` is
``sqrt((a + d)/2) + sqrt((a - d)/2)`` when ``d = sqrt(a**2 - b**2 r)``
is simpler; the general denester works on nested lists of radicands.

>>> from sympy_extras.core import Symbol, Integer
>>> from sympy_extras.functions import sqrt
>>> sqrtdenest(sqrt(5 + 2*sqrt(6))), sqrtdenest(sqrt(3 + 2*sqrt(2))), sqrtdenest(sqrt(Integer(2)))
(sqrt(2) + sqrt(3), 1 + sqrt(2), sqrt(2))
>>> x = Symbol('x', positive=True)
>>> sqrtdenest(sqrt(x + 2*sqrt(x - 1)))
sqrt(x + 2*sqrt(x - 1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the polynomial of _sqrt_symbolic_denest from the
# coefficients in the dummy (SymPy's EX domain); comparisons through the
# core's relational functions (TypeError when undecided, as in SymPy).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import count_ops, expand_mul, expand_multinomial
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy

__all__ = ['sqrtdenest', 'sqrt_depth', 'is_algebraic', 'is_sqrt']


class SqrtdenestStopIteration(StopIteration):
    pass


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _mexpand(e: Expr) -> Expr:
    """SymPy's ``_mexpand``: ``expand_mul(expand_multinomial(e))``."""
    return expand_mul(expand_multinomial(e))


def _ops(e: Expr) -> int:
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0


def _sqrt(e: Expr) -> Expr:
    from sympy_extras.functions import sqrt
    return sqrt(e)


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _neg(e: Expr) -> Expr:
    return mul(Integer(-1), e)


def _truth(b: Basic) -> bool:
    from sympy_extras.core.logic import false, true
    if b is true:
        return True
    if b is false:
        return False
    raise TypeError('cannot determine truth value of %s' % (b,))


def _lt(a: Expr, b: Expr) -> bool:
    from sympy_extras.core.relational import lt
    return _truth(lt(a, b))


def _le(a: Expr, b: Expr) -> bool:
    from sympy_extras.core.relational import le
    return _truth(le(a, b))


def _is_rational(e: Expr) -> bool:
    return isinstance(e, Rational)


def _is_one(e: Expr) -> bool:
    return isinstance(e, Integer) and e.p == 1


def _is_zero(e: Expr) -> bool:
    return isinstance(e, Integer) and e.p == 0


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:is_sqrt
def is_sqrt(e: Expr) -> bool:
    return isinstance(e, Pow) and isinstance(e.exp, Rational) and e.exp.q == 2 and abs(e.exp.p) == 1


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:sqrt_depth
def sqrt_depth(p: Expr) -> int:
    """The depth of nesting of square roots (``I`` counts as one)."""
    if p == I:
        return 1
    if not p.args:
        return 0
    if isinstance(p, (Add, Mul)):
        return max(sqrt_depth(_expr(x)) for x in p.args)
    if is_sqrt(p) and isinstance(p, Pow):
        return sqrt_depth(p.base) + 1
    return 0


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:is_algebraic
def is_algebraic(p: Expr) -> bool:
    if isinstance(p, Rational):
        return True
    if not p.args:
        return False
    if isinstance(p, Pow) and (is_sqrt(p) or isinstance(p.exp, Integer)):
        return is_algebraic(p.base)
    if isinstance(p, (Add, Mul)):
        return all(is_algebraic(_expr(x)) for x in p.args)
    return False


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_subsets
def _subsets(n: int) -> list[list[int]]:
    if n == 1:
        return [[1]]
    if n == 2:
        return [[1, 0], [0, 1], [1, 1]]
    if n == 3:
        return [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    b = _subsets(n - 1)
    return [x + [0] for x in b] + [[0] * (n - 1) + [1]] + [x + [1] for x in b]


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:sqrtdenest
def sqrtdenest(expr: ExprLike, max_iter: int = 3) -> Expr:
    """The square roots of ``expr`` denested where possible (at most
    ``max_iter`` passes)."""
    e = expand_mul(to_expr(expr))
    for _ in range(max_iter):
        z = _sqrtdenest0(e)
        if e == z:
            return e
        e = z
    return e


Match = list[Expr]


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrt_match
def _sqrt_match(p0: Expr) -> Match:
    """``[a, b, r]`` with ``p = a + b*sqrt(r)``, ``r`` the deepest radicand;
    ``[]`` when there is none."""
    from .radsimp import split_surds
    p = _mexpand(p0)
    if isinstance(p, Number):
        return [p, Integer(0), Integer(0)]
    if isinstance(p, Add):
        pargs = sorted((_expr(a) for a in p.args), key=lambda t: t.sort_key())
        if all(_is_rational(pow(x, Integer(2))) and pow(x, Integer(2)).is_positive is True for x in pargs):
            rs, bs, as_ = split_surds(p)
            return [as_, bs, rs]
        v = [(sqrt_depth(x), x, i) for i, x in enumerate(pargs)]
        nmax = max(v, key=lambda t: (t[0], t[1].sort_key(), t[2]))
        if nmax[0] == 0:
            return []
        depth, _, i = nmax
        r = pargs.pop(i)
        v.pop(i)
        b: Expr = Integer(1)
        if isinstance(r, Mul):
            bv = [_expr(x) for x in r.args if sqrt_depth(_expr(x)) < depth]
            rv = [_expr(x) for x in r.args if sqrt_depth(_expr(x)) >= depth]
            b = Mul(*bv) if len(bv) > 1 else (bv[0] if bv else Integer(1))
            r = Mul(*rv) if len(rv) > 1 else (rv[0] if rv else Integer(1))
        a1: list[Expr] = []
        b1: list[Expr] = [b]
        for xd, x1, _ in v:
            if xd < depth:
                a1.append(x1)
            elif x1 == r:
                b1.append(Integer(1))
            elif isinstance(x1, Mul):
                x1args = [_expr(t) for t in x1.args]
                if r in x1args:
                    x1args.remove(r)
                    b1.append(mul(*x1args))
                else:
                    a1.append(x1)
            else:
                a1.append(x1)
        return [add(*a1), add(*b1), pow(r, Integer(2))]
    bc, rr = p.as_coeff_Mul()
    if is_sqrt(rr):
        return [Integer(0), bc, pow(rr, Integer(2))]
    return []


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrtdenest0
def _sqrtdenest0(e: Expr) -> Expr:
    if is_sqrt(e):
        n, d = e.as_numer_denom()
        if _is_one(d):
            if isinstance(n, Pow) and isinstance(n.base, Add):
                args = sorted((_expr(a) for a in n.base.args), key=lambda t: t.sort_key())
                if len(args) > 2 and all(isinstance(pow(x, Integer(2)), Integer) for x in args):
                    try:
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                e = _sqrt(_mexpand(add(*[_sqrtdenest0(x) for x in args])))
            return _sqrtdenest1(e)
        return mul(_sqrtdenest0(n), _inv(_sqrtdenest0(d)))
    if isinstance(e, Add):
        cs: list[Expr] = []
        args2: list[Expr] = []
        for arg in e.args:
            c, a = _expr(arg).as_coeff_Mul()
            cs.append(c)
            args2.append(a)
        if all(_is_rational(c) for c in cs) and all(is_sqrt(a) for a in args2):
            return _sqrt_ratcomb(cs, args2)
    if e.args:
        return _expr(e._evaluated(tuple(_sqrtdenest0(_expr(a)) if isinstance(a, Expr) else a for a in e.args)))
    return e


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrtdenest_rec
def _sqrtdenest_rec(e: Expr) -> Expr:
    from .radsimp import rad_rationalize, radsimp, split_surds
    if not isinstance(e, Pow):
        return sqrtdenest(e)
    if _lt(e.base, Integer(0)):
        return mul(_sqrt(Integer(-1)), _sqrtdenest_rec(_sqrt(_neg(e.base))))
    g, a, b = split_surds(e.base)
    a = mul(a, _sqrt(g))
    if _lt(a, b):
        a, b = b, a
    c2 = _mexpand(add(pow(a, Integer(2)), _neg(pow(b, Integer(2)))))
    if len(c2.args) > 2:
        g, a1, b1 = split_surds(c2)
        a1 = mul(a1, _sqrt(g))
        if _lt(a1, b1):
            a1, b1 = b1, a1
        c2_1 = _mexpand(add(pow(a1, Integer(2)), _neg(pow(b1, Integer(2)))))
        c_1 = _sqrtdenest_rec(_sqrt(c2_1))
        d_1 = _sqrtdenest_rec(_sqrt(add(a1, c_1)))
        num, den = rad_rationalize(b1, d_1)
        c = _mexpand(add(mul(d_1, _inv(_sqrt(Integer(2)))), mul(num, _inv(mul(den, _sqrt(Integer(2)))))))
    else:
        c = _sqrtdenest1(_sqrt(c2))
    if sqrt_depth(c) > 1:
        raise SqrtdenestStopIteration
    ac = add(a, c)
    if len(ac.args) >= len(e.args):
        if _ops(ac) >= _ops(e.base):
            raise SqrtdenestStopIteration
    d = sqrtdenest(_sqrt(ac))
    if sqrt_depth(d) > 1:
        raise SqrtdenestStopIteration
    num, den = rad_rationalize(b, d)
    r = add(mul(d, _inv(_sqrt(Integer(2)))), mul(num, _inv(mul(den, _sqrt(Integer(2))))))
    return _mexpand(radsimp(r))


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrtdenest1
def _sqrtdenest1(e: Expr, denester: bool = True) -> Expr:
    from sympy_extras.functions import root
    from .radsimp import radsimp
    if not is_sqrt(e) or not isinstance(e, Pow):
        return e
    a0 = e.base
    if not a0.args:
        return e
    val = _sqrt_match(a0)
    if not val:
        return e
    a, b, r = val
    d2 = _mexpand(add(pow(a, Integer(2)), _neg(mul(pow(b, Integer(2)), r))))
    if isinstance(d2, Rational):
        if d2.is_positive is True:
            z = _sqrt_numeric_denest(a, b, r, d2)
            if z is not None:
                return z
        else:
            dr2 = _mexpand(_neg(mul(d2, r)))
            dr = _sqrt(dr2)
            if isinstance(dr, Rational):
                z = _sqrt_numeric_denest(_mexpand(mul(b, r)), a, r, dr2)
                if z is not None:
                    return mul(z, _inv(root(r, Integer(4))))
    else:
        z = _sqrt_symbolic_denest(a, b, r)
        if z is not None:
            return z
    if not denester or not is_algebraic(e):
        return e
    res = sqrt_biquadratic_denest(e, a, b, r, d2)
    if res is not None:
        return res
    av0: list[Optional[Expr]] = [a, b, r, d2]
    z2 = _denester([radsimp(pow(e, Integer(2)))], av0, 0, sqrt_depth(e))[0]
    if av0[1] is None:
        return e
    if z2 is not None:
        if sqrt_depth(z2) == sqrt_depth(e) and _ops(z2) > _ops(e):
            return e
        return z2
    return e


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrt_symbolic_denest
def _sqrt_symbolic_denest(a: Expr, b: Expr, r: Expr) -> Optional[Expr]:
    from sympy_extras.polys.polytools import _decompose_with_gens
    from sympy_extras.polys.polyerrors import PolynomialError
    rval = _sqrt_match(r)
    if not rval:
        return None
    ra, rb, rr = rval
    if _is_zero(rb):
        return None
    y = Dummy('y', positive=True)
    substituted = _expr(a.subs(_sqrt(rr), mul(add(pow(y, Integer(2)), _neg(ra)), _inv(rb))))
    try:
        terms = _decompose_with_gens(_mexpand(substituted), [y])
    except PolynomialError:
        return None
    coefficients = {m[0]: add(*cs) for m, cs in terms.items()}
    coefficients = {k: c for k, c in coefficients.items() if not _is_zero(c)}
    if not coefficients or max(coefficients) != 2:
        return None
    ca = coefficients.get(2, Integer(0))
    cb = add(coefficients.get(1, Integer(0)), b)
    cc = coefficients.get(0, Integer(0))
    if _mexpand(add(pow(cb, Integer(2)), mul(Integer(-4), ca, cc))).equals(Integer(0)) is True:
        z = _sqrt(mul(ca, pow(add(_sqrt(r), mul(cb, _inv(mul(Integer(2), ca)))), Integer(2))))
        if z.is_number:
            c, p = z.as_content_primitive()
            z = _mexpand(Mul(c, p) if not _is_one(c) else p)
        return z
    return None


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrt_numeric_denest
def _sqrt_numeric_denest(a: Expr, b: Expr, r: Expr, d2: Expr) -> Optional[Expr]:
    from sympy_extras.functions import sign
    d = _sqrt(d2)
    s = add(a, d)
    if sqrt_depth(s) < sqrt_depth(r) + 1 or _is_rational(pow(s, Integer(2))):
        s1, s2 = sign(s), sign(b)
        if s1 == Integer(-1) and s2 == Integer(-1):
            s1 = s2 = Integer(1)
        res = mul(add(mul(s1, _sqrt(add(a, d))), mul(s2, _sqrt(add(a, _neg(d))))), _sqrt(Integer(2)), Rational(1, 2))
        return res.expand()
    return None


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:sqrt_biquadratic_denest
def sqrt_biquadratic_denest(e: Expr, a: Expr, b: Expr, r: Expr, d2: Expr) -> Optional[Expr]:
    from .radsimp import rad_rationalize, radsimp
    if not isinstance(e, Pow):
        return None
    if _le(r, Integer(0)) or _lt(d2, Integer(0)) or _is_zero(b) or sqrt_depth(e.base) < 2:
        return None
    for x in (a, b, r):
        for yv in x.args:
            y2 = pow(_expr(yv), Integer(2))
            if not isinstance(y2, Integer) or y2.p <= 0:
                return None
    sqd = _mexpand(sqrtdenest(_sqrt(radsimp(d2))))
    if sqrt_depth(sqd) > 1:
        return None
    half = Rational(1, 2)
    for x in (add(mul(a, half), mul(sqd, half)), add(mul(a, half), _neg(mul(sqd, half)))):
        A = sqrtdenest(_sqrt(x))
        if sqrt_depth(A) > 1:
            continue
        Bn, Bd = rad_rationalize(b, _mexpand(mul(Integer(2), A)))
        B = mul(Bn, _inv(Bd))
        z = add(A, mul(B, _sqrt(r)))
        if _lt(z, Integer(0)):
            z = _neg(z)
        return _mexpand(z)
    return None


Denested = tuple[Optional[Expr], Optional[list[int]]]


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_denester
def _denester(nested: list[Expr], av0: list[Optional[Expr]], h: int, max_depth_level: int) -> Denested:
    from sympy_extras.functions import root
    from .radsimp import radsimp
    if h > max_depth_level:
        return None, None
    if av0[1] is None:
        return None, None
    if av0[0] is None and all(isinstance(n, Number) for n in nested):
        for subset in _subsets(len(nested)):
            p = _mexpand(mul(*[nested[i] for i in range(len(subset)) if subset[i]]))
            if subset.count(1) > 1 and subset[-1]:
                p = _neg(p)
            sqp = _sqrt(p)
            if isinstance(sqp, Rational):
                return sqp, subset
        return _sqrt(nested[-1]), [0] * len(nested)
    R: Optional[Expr] = None
    values: list[Match]
    if av0[0] is not None:
        first, second, third, fourth = av0
        if first is None or second is None or third is None or fourth is None:
            return None, None
        values = [[first, second]]
        R = third
        nested2 = [fourth, R]
        av0[0] = None
    else:
        values = [v for v in (_sqrt_match(x) for x in nested) if v]
        for v in values:
            if not _is_zero(v[2]):
                if R is not None:
                    if R != v[2]:
                        av0[1] = None
                        return None, None
                else:
                    R = v[2]
        if R is None:
            return _sqrt(nested[-1]), [0] * len(nested)
        nested2 = [add(_mexpand(pow(v[0], Integer(2))), _neg(_mexpand(mul(R, pow(v[1], Integer(2)))))) for v in values] + [R]
    d, f = _denester(nested2, av0, h + 1, max_depth_level)
    if not f or d is None:
        return None, None
    if not any(f[i] for i in range(len(nested))):
        v = values[-1]
        return _sqrt(add(v[0], _mexpand(mul(v[1], d)))), f
    p = mul(*[nested[i] for i in range(len(nested)) if f[i]])
    vm = _sqrt_match(p)
    if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
        vm[0] = _neg(vm[0])
        vm[1] = _neg(vm[1])
    if not f[len(nested)]:
        vad = _mexpand(add(vm[0], d))
        if _le(vad, Integer(0)):
            return _sqrt(nested[-1]), [0] * len(nested)
        if not (sqrt_depth(vad) <= sqrt_depth(R) + 1 or isinstance(pow(vad, Integer(2)), Number)):
            av0[1] = None
            return None, None
        sqvad = _sqrtdenest1(_sqrt(vad), denester=False)
        if not sqrt_depth(sqvad) <= sqrt_depth(R) + 1:
            av0[1] = None
            return None, None
        sqvad1 = radsimp(_inv(sqvad))
        res = _mexpand(add(mul(sqvad, _inv(_sqrt(Integer(2)))),
                           mul(vm[1], _sqrt(R), sqvad1, _inv(_sqrt(Integer(2))))))
        return res, f
    s2 = add(_mexpand(mul(vm[1], R)), d)
    if _le(s2, Integer(0)):
        return _sqrt(nested[-1]), [0] * len(nested)
    FR, s = root(_mexpand(R), Integer(4)), _sqrt(s2)
    return _mexpand(add(mul(s, _inv(mul(_sqrt(Integer(2)), FR))), mul(vm[0], FR, _inv(mul(_sqrt(Integer(2)), s))))), f


# PORTED FROM SYMPY: sympy/simplify/sqrtdenest.py:_sqrt_ratcomb
def _sqrt_ratcomb(cs: list[Expr], args: list[Expr]) -> Expr:
    from .radsimp import radsimp

    def find(a: list[Expr]) -> Optional[tuple[Expr, int, int]]:
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                ai, aj = a[i], a[j]
                if not (isinstance(ai, Pow) and isinstance(aj, Pow)):
                    continue
                p = _mexpand(mul(ai.base, aj.base))
                s = sqrtdenest(_sqrt(p))
                if s != _sqrt(p):
                    return s, i, j
        return None

    found = find(args)
    if found is None:
        return add(*[mul(c, a) for c, a in zip(cs, args)])
    s, i1, i2 = found
    c2 = cs.pop(i2)
    args.pop(i2)
    a1 = args[i1]
    if isinstance(a1, Pow):
        cs[i1] = add(cs[i1], radsimp(mul(c2, s, _inv(a1.base))))
    return _sqrt_ratcomb(cs, args)
