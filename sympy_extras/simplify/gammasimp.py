"""``gammasimp`` and ``combsimp``: simplification of gamma functions and
of combinatorial expressions.

Ported from SymPy's ``sympy.simplify.gammasimp`` and
``sympy.simplify.combsimp``. Gamma functions with arguments differing by
integers are merged through the rising factorial; the reflection
formula, the duplication formula and Gauss's multiplication theorem are
applied; ``combsimp`` keeps integer arguments integer and rewrites the
result as factorials and binomials.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import binomial, factorial, gamma
>>> x = Symbol('x')
>>> n, k = Symbol('n', integer=True), Symbol('k', integer=True)
>>> gammasimp(gamma(x)/gamma(x - 3)), gammasimp(gamma(x)*gamma(1 - x))
((x - 3)*(x - 2)*(x - 1), pi/sin(pi*x))
>>> combsimp(factorial(n)/factorial(n - 3)), combsimp(binomial(n + 1, k + 1)/binomial(n, k))
(n*(n - 2)*(n - 1), (n + 1)/(k + 1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/simplify/gammasimp.py (gammasimp, _gammasimp, _rf) and
# sympy/simplify/combsimp.py (combsimp, _gamma_as_comb) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the private function _rf is an evaluating Python
# function with the final rewrite as a parameter.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr, ExprLike, factors_of, to_expr
from sympy_extras.core.function import Application, count_ops
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.sorting import default_sort_key, ordered
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.core.traversal import bottom_up

__all__ = ['gammasimp', 'combsimp']

Leaf = Callable[[Expr, Expr], Expr]

ONE = Integer(1)


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _int(e: Expr) -> Optional[int]:
    return e.p if isinstance(e, Integer) else None


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_rf
def _rf(a: Expr, b: Expr, leaf: Leaf) -> Expr:
    """The rising factorial ``rf(a, b)`` with the integer parts of ``a``
    and ``b`` taken out as products; the rest is ``leaf(a, b)``."""
    n = _int(b)
    if n is not None:
        if n == 0:
            return ONE
        if n > 0:
            return mul(*[add(a, Integer(i)) for i in range(n)])
        return _inv(mul(*[add(a, Integer(-i)) for i in range(1, -n + 1)]))
    if isinstance(b, Add):
        c, b0 = b.as_coeff_Add()
        ci = _int(c)
        if ci is not None and ci > 0:
            return mul(_rf(a, b0, leaf), _rf(add(a, b0), c, leaf))
        if ci is not None and ci < 0:
            return mul(_rf(a, b0, leaf), _inv(_rf(add(a, b0, c), Integer(-ci), leaf)))
    if isinstance(a, Add):
        c, a0 = a.as_coeff_Add()
        ci = _int(c)
        if ci is not None and ci > 0:
            return mul(_rf(a0, b, leaf), _rf(add(a0, b), c, leaf), _inv(_rf(a0, c, leaf)))
        if ci is not None and ci < 0:
            return mul(_rf(a0, b, leaf), _rf(add(a0, c), Integer(-ci), leaf),
                       _inv(_rf(add(a0, b, c), Integer(-ci), leaf)))
    return leaf(a, b)


def _gamma_leaf(a: Expr, b: Expr) -> Expr:
    from sympy_extras.functions import gamma
    return mul(gamma(add(a, b)), _inv(gamma(a)))


def _comb_leaf(a: Expr, b: Expr) -> Expr:
    from sympy_extras.functions import gamma
    return gamma(add(b, ONE))


def _is_gamma(b: Basic) -> bool:
    from sympy_extras.functions.special.gamma_functions import Gamma
    return isinstance(b, Gamma)


def _through_rf(e: Expr, leaf: Leaf) -> Expr:
    """Every ``gamma(n)`` as ``_rf(1, n - 1)`` with the given leaf."""
    from sympy_extras.core.function import expand

    def value(b: Basic) -> Basic:
        return _rf(ONE, expand(add(_expr(b.args[0]), Integer(-1))), leaf)
    return _expr(e.replace(_is_gamma, value))


def _gamma_rational(r: Rational) -> Expr:
    """``expand_func(gamma(r))``: the integer part of a rational argument
    taken out (``gamma(4/3)`` is ``gamma(1/3)/3``)."""
    from sympy_extras.functions import gamma, rf
    if abs(r.p) > r.q:
        n = r.p // r.q
        rest = number_from_fraction(Fraction(r.p - n * r.q, r.q))
        return mul(gamma(rest), rf(rest, Integer(n)))
    return gamma(r)


def _gamma_factor(x: Expr) -> bool:
    if _is_gamma(x):
        return True
    if isinstance(x, (Add, Mul)):
        return any(_gamma_factor(_expr(xi)) for xi in x.args)
    if isinstance(x, Pow) and (x.exp.is_integer is True or x.base.is_positive is True):
        return _gamma_factor(x.base)
    return False


def _gamma_count(x: Expr) -> int:
    return sum(1 for node in preorder_traversal(x) if _is_gamma(node))


def _gamma_rat(x: Expr) -> Expr:
    xx = _through_rf(x, _gamma_leaf)
    return xx if _gamma_count(xx) < _gamma_count(x) else x


def _uniq(values: list[Fraction]) -> list[Fraction]:
    out: list[Fraction] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp._run
def _run(coeffs: list[Fraction]) -> Optional[tuple[int, Fraction, list[Fraction]]]:
    """A run ``u, u + 1/n, ..., u + (n-1)/n`` (modulo integers) among the
    coefficients, removed from them."""
    u = _uniq(coeffs)
    for i in range(len(u)):
        dj = [((u[j] - u[i]) % 1, j) for j in range(i + 1, len(u))]
        for one, _ in dj:
            if one.numerator == 1 and one.denominator != 1:
                n = one.denominator
                got = [i]
                get = list(range(1, n))
                complete = False
                for d, j in dj:
                    m = n * d
                    if m.denominator == 1 and m.numerator in get:
                        get.remove(m.numerator)
                        got.append(j)
                        if not get:
                            complete = True
                            break
                if not complete:
                    continue
                values = [u[j] for j in got]
                for c in values:
                    coeffs.remove(c)
                return n, values[0], values[1:]
    return None


def _number(f: Fraction) -> Expr:
    return number_from_fraction(f)


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp._mult_thm
def _mult_thm(gammas: list[Expr], numer: list[Expr]) -> None:
    """Gauss's multiplication theorem on the runs of arguments."""
    rats: dict[Expr, list[Fraction]] = {}
    for g in gammas:
        c, resid = g.as_coeff_Add()
        if isinstance(c, Rational):
            rats.setdefault(resid, []).append(c.as_fraction())
        else:
            rats.setdefault(g, []).append(Fraction(0))
    keys = sorted(rats, key=default_sort_key)
    result: dict[Expr, list[Expr]] = {}
    for resid in keys:
        coeffs = sorted(rats[resid])
        new: list[Expr] = []
        while True:
            run = _run(coeffs)
            if run is None:
                break
            n, ui, other = run
            for u in other:
                con = add(resid, _number(u), Integer(-1))
                for k in range(int(u - ui)):
                    numer.append(add(con, Integer(-k)))
            con = mul(Integer(n), add(resid, _number(ui)))
            numer.append(mul(pow(mul(Integer(2), pi), number_from_fraction(Fraction(n - 1, 2))),
                             pow(Integer(n), add(Rational(1, 2), mul(Integer(-1), con)))))
            new.append(con)
        result[resid] = [add(resid, _number(c)) for c in coeffs] + new
    gammas[:] = [g for resid in keys for g in result[resid]]


def _explicate(p: Expr) -> tuple[Optional[bool], list[Expr]]:
    if isinstance(p, Integer) and p.p == 1:
        return None, []
    b, e = p.as_base_exp()
    if isinstance(e, Integer):
        if _is_gamma(b):
            return True, [_expr(b.args[0])] * e.p
        return False, [b] * e.p
    return False, [p]


def _reflect(gammas: list[Expr], numer: list[Expr], denom: list[Expr]) -> None:
    """The reflection formula on pairs of arguments summing to an
    integer."""
    from sympy_extras.functions import sin
    new: list[Expr] = []
    while gammas:
        g1 = gammas.pop()
        if g1.is_integer is True:
            new.append(g1)
            continue
        for i, g2 in enumerate(gammas):
            n = _int(add(g1, g2, Integer(-1)))
            if n is None:
                continue
            numer.append(pi)
            denom.append(sin(mul(pi, g1)))
            gammas.pop(i)
            if n > 0:
                numer.extend(add(ONE, mul(Integer(-1), g1), Integer(k)) for k in range(n))
            elif n < 0:
                denom.extend(add(mul(Integer(-1), g1), Integer(-k)) for k in range(-n))
            break
        else:
            new.append(g1)
    gammas[:] = new


def _duplicate(ng: list[Expr], dg: list[Expr], no: list[Expr], do: list[Expr]) -> None:
    """The duplication formula on ``gamma(x)/gamma(y)`` with ``x - 2*y`` an
    integer."""
    from sympy_extras.functions import sqrt
    while True:
        found: Optional[tuple[Expr, Expr, int]] = None
        for x in ng:
            for y in dg:
                n = _int(add(x, mul(Integer(-2), y)))
                if n is not None:
                    found = (x, y, n)
                    break
            if found is not None:
                break
        if found is None:
            return
        x, y, n = found
        ng.remove(x)
        dg.remove(y)
        if n > 0:
            no.extend(add(mul(Integer(2), y), Integer(k)) for k in range(n))
        elif n < 0:
            do.extend(add(mul(Integer(2), y), Integer(-1 - k)) for k in range(-n))
        ng.append(add(y, Rational(1, 2)))
        no.append(pow(Integer(2), add(mul(Integer(2), y), Integer(-1))))
        do.append(sqrt(pi))


def _fuzzy(numer_gammas: list[Expr], denom_gammas: list[Expr], numer_others: list[Expr],
           denom_others: list[Expr]) -> None:
    """Factors ``x`` (``x - 1`` in the other part) absorbed into ``gamma(x)``
    when their quotient is a number."""
    from sympy_extras.polys.polytools import cancel
    Signature = tuple[frozenset[Symbol], frozenset[Basic]]
    inv: dict[Expr, Signature] = {}

    def compute_st(e: Expr) -> Signature:
        if e in inv:
            return inv[e]
        exps: set[Basic] = {p.exp for p in e.atoms(Pow) if isinstance(p, Pow)}
        return frozenset(e.free_symbols), frozenset(e.atoms(Application) | exps)

    def update_st(e: Expr) -> None:
        inv[e] = compute_st(e)

    def find_fuzzy(lst: list[Expr], x: Expr) -> Optional[Expr]:
        if not lst:
            return None
        s1, t1 = compute_st(x)
        for y in lst:
            s2, t2 = inv[y]
            if t1 != t2 or (not s1.intersection(s2) and (s1 or s2)):
                continue
            a = len(cancel(mul(x, _inv(y))).free_symbols)
            if a == 0 and (len(x.free_symbols) > 0 or len(y.free_symbols) > 0):
                return y
        return None

    for e in numer_gammas + denom_gammas + numer_others + denom_others:
        update_st(e)
    for gammas, numer, denom in [(numer_gammas, numer_others, denom_others),
                                 (denom_gammas, denom_others, numer_others)]:
        new: list[Expr] = []
        while gammas:
            g = gammas.pop()
            cont = True
            while cont:
                cont = False
                y = find_fuzzy(numer, g)
                if y is not None:
                    numer.remove(y)
                    if y != g:
                        q = mul(y, _inv(g))
                        numer.append(q)
                        update_st(q)
                    g = add(g, ONE)
                    cont = True
                y = find_fuzzy(denom, add(g, Integer(-1)))
                if y is not None:
                    denom.remove(y)
                    if y != add(g, Integer(-1)):
                        q = mul(add(g, Integer(-1)), _inv(y))
                        numer.append(q)
                        update_st(q)
                    g = add(g, Integer(-1))
                    cont = True
            new.append(g)
        gammas[:] = new


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp.rule_gamma
def _rule_gamma(expr: Expr, as_comb: bool, level: int = 0) -> Expr:
    from sympy_extras.functions import gamma
    if not expr.args:
        return expr
    if level == 0:
        expr = _expr(expr._evaluated(tuple(_rule_gamma(x, as_comb, 1) if isinstance(x, Expr) else x
                                           for x in expr.args)))
        level = 1
    if not isinstance(expr, Mul):
        return expr
    if level == 1:
        args = [f for f in factors_of(expr) if f.is_commutative is not False]
        nc = [f for f in factors_of(expr) if f.is_commutative is False]
        if not args:
            return expr
        if nc:
            return mul(_rule_gamma(mul(*args), as_comb, 2), mul(*nc))
        level = 2
    if level == 2:
        found = list(factors_of(expr))
        gamma_ind = mul(*[f for f in found if not _gamma_factor(f)])
        d = mul(*[f for f in found if _gamma_factor(f)])
        nd, dd = d.as_numer_denom()
        for ipass in range(2):
            factors = [_expr(f) for f in ordered(list(factors_of(nd)))]
            for i, ni in enumerate(factors):
                if isinstance(ni, Add):
                    ni, dd = add(*[_rule_gamma(_gamma_rat(mul(_expr(a), _inv(dd))), as_comb, 3)
                                   for a in ni.args]).as_numer_denom()
                    factors[i] = ni
                    if not dd.has(*_gamma_types()):
                        break
            nd = mul(*factors)
            if ipass == 0 and not _gamma_factor(nd):
                break
            nd, dd = dd, nd
        expr = mul(gamma_ind, nd, _inv(dd))
        if not (isinstance(expr, Mul) and (_gamma_factor(dd) or _gamma_factor(nd))):
            return expr
        level = 3
    if level == 3:
        while True:
            was = expr
            expr = _rule_gamma(expr, as_comb, 4)
            if expr == was:
                return expr

    numer_gammas: list[Expr] = []
    denom_gammas: list[Expr] = []
    numer_others: list[Expr] = []
    denom_others: list[Expr] = []
    newargs = [_expr(a) for a in ordered(list(expr.args))]
    while newargs:
        n, d = newargs.pop().as_numer_denom()
        isg, lst = _explicate(n)
        if isg:
            numer_gammas.extend(lst)
        elif isg is False:
            numer_others.extend(lst)
        isg, lst = _explicate(d)
        if isg:
            denom_gammas.extend(lst)
        elif isg is False:
            denom_others.extend(lst)

    if not as_comb:
        _reflect(numer_gammas, numer_others, denom_others)
        _reflect(denom_gammas, denom_others, numer_others)
        _duplicate(numer_gammas, denom_gammas, numer_others, denom_others)
        _duplicate(denom_gammas, numer_gammas, denom_others, numer_others)
        _mult_thm(numer_gammas, numer_others)
        _mult_thm(denom_gammas, denom_others)

    _fuzzy(numer_gammas, denom_gammas, numer_others, denom_others)
    return mul(mul(*[gamma(g) for g in numer_gammas]), _inv(mul(*[gamma(g) for g in denom_gammas])),
               mul(*numer_others), _inv(mul(*denom_others)))


def _gamma_types() -> tuple[type, ...]:
    from sympy_extras.functions.special.gamma_functions import Gamma
    return (Gamma,)


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:_gammasimp
def _gammasimp(expr: Expr, as_comb: bool) -> Expr:
    from sympy_extras.polys.polytools import factor
    expr = _through_rf(expr, _comb_leaf if as_comb else _gamma_leaf)
    was = factor(expr)
    expr = _rule_gamma(was, as_comb)
    if expr != was:
        expr = factor(expr)

    def value(b: Basic) -> Basic:
        arg = b.args[0]
        return _gamma_rational(arg) if isinstance(arg, Rational) else b
    return _expr(expr.replace(_is_gamma, value))


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:gammasimp
def gammasimp(expr: ExprLike) -> Expr:
    """The expression with its gamma functions (and the functions
    rewritten as gamma) simplified."""
    from sympy_extras.functions.special.gamma_functions import Gamma
    e = _expr(to_expr(expr).rewrite(Gamma))
    functions = {f for f in e.atoms(Application)}
    gammas = {f for f in functions if isinstance(f, Gamma)}
    if not gammas:
        return e
    others = (functions - gammas) & e.as_dummy().atoms(Application)
    if others:
        fun = [_expr(f) for f in ordered(list(others))]
        dum = [Dummy() for _ in fun]
        simp = [_expr(f._evaluated(tuple(_gammasimp(_expr(a), False) if isinstance(a, Expr) else a
                                         for a in f.args))) for f in fun]
        d = e.xreplace(dict(zip(fun, dum)))
        back: dict[Basic, Basic] = dict(zip(dum, simp))
        return _gammasimp(d, False).xreplace(back)
    return _gammasimp(e, False)


def _combinatorial_as_gamma(e: Expr) -> Expr:
    """Factorials, binomials and rising and falling factorials as gamma
    functions (SymPy's ``rewrite(gamma, piecewise=False)``)."""
    from sympy_extras.functions import gamma
    from sympy_extras.functions.combinatorial.factorials import (Binomial, Factorial, FallingFactorial,
                                                                  RisingFactorial)
    from sympy_extras.core.relational import le, lt
    from sympy_extras.core.logic import true

    def query(b: Basic) -> bool:
        return isinstance(b, (Factorial, Binomial, RisingFactorial, FallingFactorial))

    def value(b: Basic) -> Basic:
        args = [_expr(a) for a in b.args]
        if isinstance(b, Factorial):
            return gamma(add(args[0], ONE))
        if isinstance(b, Binomial):
            n, k = args
            return mul(gamma(add(n, ONE)), _inv(mul(gamma(add(k, ONE)), gamma(add(n, mul(Integer(-1), k), ONE)))))
        x, k = args
        minus_one_k = pow(Integer(-1), k)
        if isinstance(b, RisingFactorial):
            if le(x, Integer(0)) == true:
                return mul(minus_one_k, gamma(add(ONE, mul(Integer(-1), x))),
                           _inv(gamma(add(mul(Integer(-1), k), mul(Integer(-1), x), ONE))))
            return mul(gamma(add(x, k)), _inv(gamma(x)))
        if lt(x, Integer(0)) == true:
            return mul(minus_one_k, gamma(add(k, mul(Integer(-1), x))), _inv(gamma(mul(Integer(-1), x))))
        return mul(gamma(add(x, ONE)), _inv(gamma(add(x, mul(Integer(-1), k), ONE))))

    return _expr(e.replace(query, value))


# PORTED FROM SYMPY: sympy/simplify/combsimp.py:_gamma_as_comb
def _gamma_as_comb(expr: Expr) -> Expr:
    """Gamma functions as factorials, and ``(a + b)!/(a!*b!)`` as
    binomials."""
    from sympy_extras.functions import binomial, factorial
    from sympy_extras.functions.combinatorial.factorials import Factorial

    def to_factorial(b: Basic) -> Basic:
        return factorial(add(_expr(b.args[0]), Integer(-1)))
    expr = _expr(expr.replace(_is_gamma, to_factorial))

    def ops(x: Expr) -> int:
        value = count_ops(x)
        return value.p if isinstance(value, Integer) else 0

    def f(b: Basic) -> Basic:
        if not isinstance(b, Mul):
            return b
        rvd: dict[Expr, Expr] = dict(b.as_powers_dict())
        numer: list[Expr] = []
        denom: list[Expr] = []
        for k in list(rvd):
            m = rvd[k]
            if isinstance(k, Factorial) and isinstance(m, Integer):
                if m.p > 0:
                    numer.extend([_expr(k.args[0])] * m.p)
                else:
                    denom.extend([_expr(k.args[0])] * -m.p)
                rvd[k] = Integer(0)
        if not numer or not denom:
            return b
        nd: list[list[Expr]] = [numer, denom]
        hit = False
        for m_ in range(2):
            i = 0
            while i < len(nd[m_]):
                ai = nd[m_][i]
                for j in range(i + 1, len(nd[m_])):
                    aj = nd[m_][j]
                    total = add(ai, aj)
                    if total in nd[1 - m_]:
                        hit = True
                        nd[1 - m_].remove(total)
                        del nd[m_][j]
                        del nd[m_][i]
                        key = binomial(total, ai if ops(ai) < ops(aj) else aj)
                        rvd[key] = add(rvd.get(key, Integer(0)), Integer(-1 if m_ == 0 else 1))
                        break
                else:
                    i += 1
        if hit:
            return mul(mul(*([pow(k, v) for k, v in rvd.items()] + [factorial(k) for k in nd[0]])),
                       _inv(mul(*[factorial(k) for k in nd[1]])))
        return b

    return _expr(bottom_up(expr, f))


# PORTED FROM SYMPY: sympy/simplify/combsimp.py:combsimp
def combsimp(expr: ExprLike) -> Expr:
    """The combinatorial expression (factorials, binomials, rising and
    falling factorials) simplified, as factorials and binomials; through
    :func:`gammasimp` when a gamma argument is not an integer."""
    e = _combinatorial_as_gamma(to_expr(expr))
    e = _expr(e.rewrite(_gamma_types()[0])) if e.has(*_rewritable()) else e
    if any(_is_gamma(node) and _expr(node.args[0]).is_integer is not True for node in preorder_traversal(e)):
        return gammasimp(e)
    return _gamma_as_comb(_gammasimp(e, as_comb=True))


def _rewritable() -> tuple[type, ...]:
    from sympy_extras.functions.combinatorial.factorials import Factorial2, Subfactorial
    from sympy_extras.functions.special.beta_functions import Beta
    return (Factorial2, Subfactorial, Beta)
