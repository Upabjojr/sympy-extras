"""``powsimp``: powers with common bases or exponents combined
(``x**y*x**z`` is ``x**(y + z)``, ``x**y*z**y`` is ``(x*z)**y`` when
allowed).

Ported from SymPy's ``sympy.simplify.powsimp`` (``powsimp`` and
``_denest_pow``): with ``combine='exp'`` the exponents of equal bases
are added (radicals joined: ``sqrt(x)*sqrt(y)*sqrt(x*y)`` is ``x*y``),
with ``combine='base'`` the bases with equal exponents are multiplied
when the exponent is an integer or the bases are nonnegative (or
``force``), ``'all'`` does both; ``deep`` works in every node.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import exp, sqrt
>>> x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
>>> n = Symbol('n', integer=True)
>>> powsimp(x**y*x**z*y**z), powsimp(x**y*x**z*y**z, combine='exp'), powsimp(x**n*y**n)
(x**(y + z)*y**z, x**(y + z)*y**z, (x*y)**n)
>>> powsimp(exp(x)*exp(y)), powsimp(sqrt(x)*sqrt(y)), powsimp(sqrt(x)*sqrt(y), force=True)
(exp(x + y), sqrt(x)*sqrt(y), sqrt(x*y))

Not ported yet: ``powdenest`` (next, with ``logcombine`` and
``posify``); ``logcombine`` inside ``_denest_pow`` goes through SymPy.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/powsimp.py (powsimp, _denest_pow),
# sympy/ntheory/factor_.py (multiplicity of rationals) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the radical bookkeeping on integer exponents (a
# symbolic one is not combined); no MatrixSymbol.
# ---------------------------------------------------------------------
from __future__ import annotations

from math import lcm, prod
from typing import Callable, Literal, Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.expr import Expr, ExprLike, factors_of, to_expr
from sympy_extras.core.function import count_ops, expand_log, expand_mul
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import E, Float, Integer, Number, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['powsimp', 'multiplicity']

Combine = Literal['all', 'exp', 'base']
Measure = Callable[[Expr], int]
_y = Dummy('y')


def _count_ops(e: Expr) -> int:
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _is_exp(e: Expr) -> bool:
    from sympy_extras.functions.elementary.exponential import Exp
    return isinstance(e, Exp)


def _is_log(e: Expr) -> bool:
    from sympy_extras.functions.elementary.exponential import Log
    return isinstance(e, Log)


def _is_one(e: Expr) -> bool:
    return isinstance(e, Rational) and e.p == 1 and e.q == 1


def _int_remove(n: int, p: int) -> int:
    m = 0
    while n % p == 0:
        n //= p
        m += 1
    return m


# PORTED FROM SYMPY: sympy/ntheory/factor_.py:multiplicity
def multiplicity(p: Rational, n: Rational) -> int:
    """The greatest ``m`` with ``p**m`` dividing ``n`` (rationals: the
    multiplicities of numerators and denominators combined)."""
    if p.q == 1 and n.q == 1:
        if n.p == 0:
            raise ValueError('no such integer exists: multiplicity of 0 is not defined')
        return _int_remove(abs(n.p), abs(p.p)) if abs(p.p) > 1 else 0
    if p.q == 1:
        if n.p == 1:
            return -_int_remove(n.q, p.p)
        return _int_remove(abs(n.p), p.p) - _int_remove(n.q, p.p)
    if p.p == 1:
        return _int_remove(n.q, p.q)
    like = min(_int_remove(abs(n.p), p.p), _int_remove(n.q, p.q))
    cross = min(_int_remove(abs(n.p), p.q), _int_remove(n.q, p.p))
    return like - cross


# PORTED FROM SYMPY: sympy/simplify/powsimp.py:_denest_pow
def _denest_pow(eq: Expr) -> Expr:
    """``(b**e1)**e2`` denested where valid (SymPy's ``_eval_power`` of
    the base, ``exp`` of a product with logarithms, integer bases written
    as powers of their prime)."""
    from sympy_extras.functions import exp, log
    b, e = eq.as_base_exp()
    if (isinstance(b, Pow) or _is_exp(b)) and not _is_one(e):
        new = b._eval_power_node(e) if not isinstance(b, Pow) else None
        if isinstance(b, Pow):
            candidate = pow(b, e)
            new = candidate if not (isinstance(candidate, Pow) and candidate.base == b) else None
        if new is not None:
            eq = new
            b, e = new.as_base_exp()
    if b == E and isinstance(e, Mul):
        logs: list[Expr] = []
        other: list[Expr] = []
        for ei in e.args:
            if not isinstance(ei, Expr):
                continue
            if any(_is_log(ai) for ai in (ei.args if isinstance(ei, Add) else (ei,)) if isinstance(ai, Expr)):
                logs.append(ei)
            else:
                other.append(ei)
        from sympy_extras.core.interop import expr_through_sympy
        import sympy
        combined = expr_through_sympy(mul(*logs), sympy.logcombine)
        return pow(exp(combined), mul(*other))
    _, be = b.as_base_exp()
    if _is_one(be) and not (isinstance(b, Mul) or (isinstance(b, Rational) and b.q != 1) or b.is_positive is True):
        return eq
    polars = [bb.as_base_exp() for bb in factors_of(b) if bb.is_polar is True]
    nonpolars = [bb for bb in factors_of(b) if bb.is_polar is not True]
    if len(polars) == 1 and not isinstance(polars[0][0], Mul):
        return mul(pow(polars[0][0], mul(polars[0][1], e)), pow(mul(*nonpolars), e))
    if polars:
        return mul(*[pow(bb, mul(ee, e)) for bb, ee in polars], pow(mul(*nonpolars), e))
    if isinstance(b, Integer):
        logb = expand_log(log(b))
        if isinstance(logb, Mul) and len(logb.args) == 2:
            c, logarithm = logb.args
            if isinstance(c, Expr) and isinstance(logarithm, Expr) and logarithm.args:
                base = logarithm.args[0]
                if isinstance(base, Expr):
                    return pow(base, mul(e, c))
    if not isinstance(b, Mul) or any(not s.args for s in factors_of(b)):
        return eq
    return eq


# PORTED FROM SYMPY: sympy/simplify/powsimp.py:powsimp
def powsimp(expr: ExprLike, deep: bool = False, combine: Combine = 'all', force: bool = False,
            measure: Measure = _count_ops) -> Expr:
    """Powers with equal bases or exponents combined (see the module)."""
    from sympy_extras.functions import exp_polar
    e = to_expr(expr)

    def recurse(arg: Expr, deep_: Optional[bool] = None, combine_: Optional[Combine] = None) -> Expr:
        return powsimp(arg, deep if deep_ is None else deep_, combine if combine_ is None else combine_, force,
                       measure)

    if not e.args or e in (exp_polar(Integer(0)), exp_polar(Integer(1))):
        return e
    if deep or isinstance(e, Add) or (isinstance(e, Mul) and _y not in e.args):
        e = _expr(e._evaluated(tuple(recurse(_expr(w)) for w in e.args)))
    if isinstance(e, Pow):
        return mul(recurse(mul(e, _y), deep_=False), pow(_y, Integer(-1)))
    if not isinstance(e, Mul):
        return e
    if combine in ('exp', 'all'):
        combined = _combine_exponents(e, deep, combine, recurse, measure)
        return combined
    return _combine_bases(e, deep, force, recurse)


Key = tuple[Expr, int]


def _bkey(b: Expr, e: Optional[Expr] = None) -> tuple[Key, Expr]:
    """``((b**s, q), p)`` for ``e = p/q * s``."""
    if e is None:
        return _bkey(*b.as_base_exp())
    if isinstance(e, Integer):
        return (b, 1), e
    if isinstance(e, Rational):
        return (b, e.q), Integer(e.p)
    c, m = e.as_coeff_Mul(rational=True)
    if not _is_one(c) and isinstance(c, Rational):
        if m.is_integer is True:
            return (b, c.q), mul(m, Integer(c.p))
        return (pow(b, m), c.q), Integer(c.p)
    return (pow(b, e), 1), Integer(1)


def _int(e: Expr) -> Optional[int]:
    return e.p if isinstance(e, Integer) else None


def _combine_exponents(e: Mul, deep: bool, combine: Combine, recurse: Callable[..., Expr], measure: Measure) -> Expr:
    c_lists: dict[Expr, list[Expr]] = {}
    nc_part: list[Expr] = []
    coeff: Expr = Integer(1)
    for term0 in e.args:
        term = _expr(term0)
        if isinstance(term, Rational):
            coeff = mul(coeff, term)
            continue
        if isinstance(term, Pow):
            term = _denest_pow(term)
        if term.is_commutative is not False:
            b, ex = term.as_base_exp()
            if deep:
                b, ex = recurse(b), recurse(ex)
            if isinstance(b, Pow) or _is_exp(b):
                b, ex = pow(b, ex), Integer(1)
            c_lists.setdefault(b, []).append(ex)
        else:
            if nc_part:
                b1, e1 = nc_part[-1].as_base_exp()
                b2, e2 = term.as_base_exp()
                if b1 == b2 and e1.is_commutative is not False and e2.is_commutative is not False:
                    nc_part[-1] = pow(b1, add(e1, e2))
                    continue
            nc_part.append(term)
    c_powers: dict[Expr, Expr] = {}
    for b, exps in [(k, c_lists[k]) for k in (_expr(o) for o in ordered(list(c_lists)))]:
        if (isinstance(b, Rational) and b.p != 0 and not all(isinstance(ei, Number) for ei in exps)
                and not _is_one(coeff) and b not in (Integer(1), Integer(-1)) and isinstance(coeff, Rational)):
            m = multiplicity(_abs_rational(b), _abs_rational(coeff))
            if m:
                exps = exps + [Integer(m)]
                coeff = mul(coeff, pow(b, Integer(-m)))
        c_powers[b] = add(*exps)
    if not _is_one(coeff):
        c_powers[coeff] = add(c_powers[coeff], Integer(1)) if coeff in c_powers else Integer(1)
    skip: set[Expr] = set()
    for b, ex in list(c_powers.items()):
        if b in skip or b not in c_powers:
            continue
        if b.is_positive is True or b.is_polar is True:
            binv = pow(b, Integer(-1))
            if isinstance(b, Float) and b == 1:
                c_powers[b] = Integer(1)
                continue
            if b != binv and binv in c_powers:
                if _is_one(b.as_numer_denom()[0]):
                    c_powers.pop(b)
                    c_powers[binv] = add(c_powers[binv], mul(Integer(-1), ex))
                else:
                    skip.add(binv)
                    removed = c_powers.pop(binv)
                    c_powers[b] = add(c_powers[b], mul(Integer(-1), removed))
    minus_one = Integer(-1)
    for b, ex in list(c_powers.items()):
        nb = mul(Integer(-1), b)
        if (isinstance(b, (Symbol, Add))) and nb in c_powers and b in c_powers:
            if b.is_positive is not None or ex.is_integer is True:
                if ex.is_integer is True or b.is_negative is True:
                    c_powers[nb] = add(c_powers[nb], c_powers.pop(b))
                    moved = ex
                else:
                    moved = c_powers.pop(nb)
                    c_powers[b] = add(c_powers[b], moved)
                c_powers[minus_one] = add(c_powers[minus_one], moved) if minus_one in c_powers else moved
    pairs = [(b, ex) for b, ex in c_powers.items() if not (isinstance(ex, Number) and ex == 0)]

    common_b: dict[Key, Expr] = {}
    bases: list[Key] = []
    done: list[tuple[Expr, Expr]] = []
    for b0, e0 in pairs:
        key, ex = _bkey(b0, e0)
        common_b[key] = add(common_b[key], ex) if key in common_b else ex
        if key[1] != 1 and isinstance(key[0], Mul):
            bases.append(key)

    def update(k: Key) -> None:
        value = _int(common_b[k])
        if value is None:
            return
        newe, r = divmod(value, k[1])
        if not r:
            common_b.pop(k)
            if newe:
                for m in factors_of(pow(k[0], Integer(newe))):
                    kk, ee = _bkey(m)
                    common_b[kk] = add(common_b.get(kk, Integer(0)), ee)
                    if kk[1] != 1:
                        bases.append(kk)

    def ratq(x: Expr) -> int:
        return _bkey(x)[0][1]

    bases.sort(key=lambda k: k[0].sort_key())
    bases.sort(key=lambda k: measure(k[0]), reverse=True)
    for base in bases:
        if base not in common_b:
            continue
        b, exponent_q = base
        last = False
        qlcm = 1
        while True:
            bstart, qstart = b, qlcm
            bb: list[Key] = []
            ee: list[tuple[int, int]] = []
            for bi in factors_of(b):
                bib, bie = _bkey(bi)
                current = _int(common_b[bib]) if bib in common_b else None
                bie_int = _int(bie)
                if current is None or bie_int is None or current < bie_int:
                    ee, bb = [], []
                    break
                ee.append((bie_int, current))
                bb.append(bib)
            if ee:
                min1 = ee[0][1] // ee[0][0]
                ok = True
                for i in range(1, len(ee)):
                    rat = ee[i][1] // ee[i][0]
                    if rat < 1:
                        ok = False
                        break
                    min1 = min(min1, rat)
                if ok:
                    for i in range(len(bb)):
                        common_b[bb[i]] = add(common_b[bb[i]], Integer(-min1 * ee[i][0]))
                        update(bb[i])
                    common_b[base] = add(common_b[base], Integer(min1 * qstart * exponent_q))
            if last or len(common_b) == 1 or all(k[1] == 1 for k in common_b):
                break
            qlcm = lcm(*[ratq(bi) for bi in factors_of(bstart)])
            if qlcm == 1:
                break
            b = pow(bstart, Integer(qlcm))
            qlcm *= qstart
            if all(ratq(bi) == 1 for bi in factors_of(b)):
                last = True
        b, q = base
        done.append((b, mul(common_b.pop(base), Rational(1, q) if q != 1 else Integer(1))))
    from sympy_extras.functions import root
    for (b, q), ex in common_b.items():
        if (isinstance(b, Pow) or _is_exp(b)) and q != 1 and not isinstance(b.as_base_exp()[1], Rational):
            bb_, be_ = b.as_base_exp()
            b = pow(bb_, mul(be_, Rational(1, q)))
        else:
            b = root(b, Integer(q))
        done.append((b, ex))
    final = dict(done)
    newexpr = mul(*[pow(b, ex) for b, ex in final.items()])
    if combine == 'exp':
        return mul(newexpr, mul(*nc_part))
    return mul(recurse(mul(*nc_part), combine_='base'), recurse(newexpr, combine_='base'))


def _abs_rational(r: Expr) -> Rational:
    if not isinstance(r, Rational):
        raise TypeError('a rational is expected')
    return r if r.p >= 0 else Rational(-r.p, r.q)


def _terms(e: Expr) -> int:
    if isinstance(e, Add):
        return sum(_terms(_expr(a)) for a in e.args)
    if isinstance(e, Mul):
        return prod(_terms(_expr(m)) for m in e.args)
    return 1


def _coeff_isneg(e: Expr) -> bool:
    if isinstance(e, Number):
        return e.is_negative is True
    if isinstance(e, Mul) and e.args and isinstance(e.args[0], Number):
        return e.args[0].is_negative is True
    return False


def _combine_bases(e: Mul, deep: bool, force: bool, recurse: Callable[..., Expr]) -> Expr:
    c_powers: list[tuple[Expr, Expr]] = []
    nc_part: list[Expr] = []
    for term0 in e.args:
        term = _expr(term0)
        if term.is_commutative is not False:
            c_powers.append(term.as_base_exp())
        else:
            nc_part.append(term)
    for i in range(len(c_powers)):
        b, ex = c_powers[i]
        if not (all(x.is_nonnegative is True for x in b.as_numer_denom()) or ex.is_integer is True or force
                or b.is_polar is True):
            continue
        exp_c, exp_t = ex.as_coeff_Mul(rational=True)
        if not _is_one(exp_c) and not _is_one(exp_t):
            c_powers[i] = (pow(b, exp_c), exp_t)
    c_exp: dict[Expr, list[Expr]] = {}
    for b, ex in c_powers:
        if deep:
            ex = recurse(ex)
        if isinstance(ex, Add) and (b.is_positive is True or ex.is_integer is True):
            ex = factor_terms(ex)
            if _coeff_isneg(ex):
                ex = mul(Integer(-1), ex)
                b = pow(b, Integer(-1))
        c_exp.setdefault(ex, []).append(b)
    result: dict[Expr, list[Expr]] = {}
    for ex, found in c_exp.items():
        if len(found) == 1:
            new_base = found[0]
        elif ex.is_integer is True or force:
            new_base = mul(*found)
        else:
            unk: list[Expr] = []
            nonneg: list[Expr] = []
            neg: list[Expr] = []
            for bi in found:
                if bi.is_negative is True:
                    neg.append(bi)
                elif bi.is_nonnegative is True or bi.is_polar is True:
                    nonneg.append(bi)
                else:
                    unk.append(bi)
            if (len(unk) == 1 and not neg) or (len(neg) == 1 and not unk):
                nonneg.extend(unk + neg)
                unk, neg = [], []
            elif neg:
                israt = isinstance(ex, Rational)
                if not israt:
                    p, d = ex.as_numer_denom()
                    israt = p.is_integer is True and d.is_integer is True
                if israt:
                    neg = [mul(Integer(-1), w) for w in neg]
                    unk.extend([Integer(-1)] * len(neg))
                else:
                    unk.extend(neg)
                    neg = []
            for b in unk:
                result.setdefault(b, []).append(ex)
            new_base = mul(*(nonneg + neg))
            expanded = expand_mul(new_base, deep=False)
            if len(expanded.args if isinstance(expanded, Add) else (expanded,)) < _terms(new_base):
                new_base = factor_terms(expanded)
        result.setdefault(new_base, []).append(ex)
    c_part = [pow(b, ei) for b, es in result.items() for ei in es]
    return mul(*(c_part + nc_part))
