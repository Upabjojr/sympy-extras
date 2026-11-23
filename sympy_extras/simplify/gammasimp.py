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
def _rf(a: Expr, b: Expr, leaf: Leaf, fractional: bool = False) -> Expr:
    """The rising factorial ``rf(a, b)`` with the integer parts of ``a``
    and ``b`` taken out as products; the rest is ``leaf(a, b)``. With
    ``fractional``, the integer part of a fractional shift of ``b`` too."""
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
            return mul(_rf(a, b0, leaf, fractional), _rf(add(a, b0), c, leaf, fractional))
        if ci is not None and ci < 0:
            return mul(_rf(a, b0, leaf, fractional), _inv(_rf(add(a, b0, c), Integer(-ci), leaf, fractional)))
        whole = c.p // c.q if fractional and isinstance(c, Rational) and ci is None else 0
        if isinstance(c, Rational) and whole != 0:
            # not SymPy's: the integer part of a fractional shift taken out
            # too, so that gamma(n + 3/2)/gamma(n + 1/2) is n + 1/2 (SymPy
            # leaves it, issue #25); the rest keeps a shift in [0, 1)
            rest = add(b0, number_from_fraction(Fraction(c.p - whole * c.q, c.q)))
            if whole > 0:
                return mul(_rf(a, rest, leaf, fractional), _rf(add(a, rest), Integer(whole), leaf, fractional))
            return mul(_rf(a, rest, leaf, fractional), _inv(_rf(add(a, rest, Integer(whole)), Integer(-whole), leaf, fractional)))
    if isinstance(a, Add):
        c, a0 = a.as_coeff_Add()
        ci = _int(c)
        if ci is not None and ci > 0:
            return mul(_rf(a0, b, leaf, fractional), _rf(add(a0, b), c, leaf, fractional),
                       _inv(_rf(a0, c, leaf, fractional)))
        if ci is not None and ci < 0:
            return mul(_rf(a0, b, leaf, fractional), _rf(add(a0, c), Integer(-ci), leaf, fractional),
                       _inv(_rf(add(a0, b, c), Integer(-ci), leaf, fractional)))
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


def _through_rf(e: Expr, leaf: Leaf, fractional: bool = False) -> Expr:
    """Every ``gamma(n)`` as ``_rf(1, n - 1)`` with the given leaf."""
    from sympy_extras.core.function import expand

    def value(b: Basic) -> Basic:
        return _rf(ONE, expand(add(_expr(b.args[0]), Integer(-1))), leaf, fractional)
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
    leaf = _comb_leaf if as_comb else _gamma_leaf
    integral = _through_rf(expr, leaf)
    # not SymPy's: gamma functions whose arguments differ by an integer
    # but have a fractional shift (gamma(n + 3/2)/gamma(n + 1/2), left by
    # SymPy, issue #25) merged too, when that leaves fewer of them
    fractional = _through_rf(expr, leaf, True)
    expr = fractional if _gamma_count(fractional) < _gamma_count(integral) else integral
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
        forward: dict[Basic, Basic] = dict(zip(fun, dum))
        d = _expr(e._xreplace_evaluated(forward))
        back: dict[Basic, Basic] = dict(zip(dum, simp))
        return _expr(_gammasimp(d, False)._xreplace_evaluated(back))
    return _gammasimp(e, False)


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
    given = to_expr(expr)
    # binomial(n, k) is 0 for a negative integer k, whatever n: for an upper
    # index which may be a negative integer the quotient of gamma functions,
    # symmetric in k and n - k, is not both binomial(n, k) and binomial(n, n - k)
    # (binomial(-3, 1) is -3, binomial(-3, -4) is 0). Binomials mirrored
    # in the input are kept as they are, and one which comes back mirrored
    # is turned again (SymPy: combsimp(binomial(n, n - k)) is binomial(n, k)
    # for integers of any sign, sympy/sympy#31275 V225, sympy/sympy#14612)
    risky = _negative_upper_binomials(given)
    # (binomial(2*n, n) and binomial(2*n + 2, n + 1) are mirrored and aligned: one family)
    conflicts = [b for b in risky if any(_mirrored(b, c) and not _aligned(b, c) for c in risky)]
    kept: dict[Basic, Basic] = {b: Dummy('binomial') for b in conflicts}
    back: dict[Basic, Basic] = {d: b for b, d in kept.items()}
    e = _expr(given.xreplace(kept)).rewrite(_gamma_types()[0], piecewise=False)
    if any(_is_gamma(node) and _expr(node.args[0]).is_integer is not True for node in preorder_traversal(e)):
        result = gammasimp(e)
    else:
        result = _gamma_as_comb(_gammasimp(e, as_comb=True))
    turned: dict[Basic, Basic] = {}
    for b in _negative_upper_binomials(result):
        if any(_aligned(b, c) for c in risky) or not any(_mirrored(b, c) for c in risky):
            continue
        upper, lower = _expr(b.args[0]), _expr(b.args[1])
        mirror = [c for c in risky if c.args[0] == upper and add(_expr(c.args[1]), lower) == upper]
        if not mirror:
            # mirrored up to a shift: no form of the result is known to hold
            return given
        turned[b] = mirror[0]
    found = _expr(_expr(result.xreplace(turned)).xreplace(back))
    if found == given:
        return given
    # the quotient of gamma functions is not the binomial where k <= n < 0
    # (binomial(-1, -1) is 0, the gamma functions have the limit 1):
    # combsimp(binomial(2*n + 1, n)/binomial(2*n + 1, n + 1)) was 1, and the quotient is 0 at n = -1
    checked = [b for b in risky if b not in conflicts]
    checked.extend(b for b in _negative_upper_binomials(found) if b not in risky)
    if not checked:
        return found
    poles, factorials = _undefined_factors(given)
    points: list[tuple[Symbol, int]] = []
    sound = True
    for b in checked:
        uncovered = _uncovered_points(b, poles, factorials)
        if uncovered is None:
            sound = False
            break
        points.extend(point for point in uncovered if point not in points)
    if sound and not points:
        return found
    if sound and len({symbol for symbol, _ in points}) == 1:
        from sympy_extras.core.logic import Boolean
        from sympy_extras.core.relational import eq
        from sympy_extras.functions.elementary.piecewise import piecewise
        cases: list[tuple[Expr, Boolean]] = []
        for symbol, value in sorted(points, key=lambda point: point[1]):
            at_point = _expr(given.subs(symbol, Integer(value)))
            if at_point.is_finite is True and at_point != _expr(found.subs(symbol, Integer(value))):
                cases.append((at_point, eq(symbol, Integer(value))))
        if not cases:
            return found
        return piecewise(*cases, (found, True))
    if isinstance(given, Add):
        # term by term: a binomial which stays as it is needs no proof
        return add(*[combsimp(term) for term in given.args if isinstance(term, Expr)])
    return given


def _undefined_factors(e: Expr) -> tuple[list[Expr], list[Expr]]:
    """The binomials which are factors of the denominator of a term of
    ``e`` and the arguments of the factorials which are factors of the
    numerator of a term: where such a binomial is zero, or such an
    argument a negative integer, the term (and ``e``) is not finite."""
    from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial
    poles: list[Expr] = []
    factorials: list[Expr] = []
    for term in Add.make_args(e):
        for factor in Mul.make_args(term):
            base, exponent = factor.as_base_exp()
            if isinstance(base, Binomial) and exponent.is_negative is True:
                poles.append(base)
            elif isinstance(base, Factorial) and exponent.is_positive is True:
                factorials.append(_expr(base.args[0]))
    return poles, factorials


def _linear(e: Expr, symbol: Symbol) -> Optional[tuple[Fraction, Fraction]]:
    """``(a, b)`` of ``e = a*symbol + b`` with rational numbers."""
    constant = _expr(e.subs(symbol, Integer(0)))
    slope = add(_expr(e.subs(symbol, Integer(1))), mul(Integer(-1), constant))
    if not isinstance(constant, Rational) or not isinstance(slope, Rational):
        return None
    if add(e, mul(Integer(-1), add(mul(slope, symbol), constant))).expand() != 0:
        return None
    return Fraction(slope.p, slope.q), Fraction(constant.p, constant.q)


def _at_least(a: Fraction, b: Fraction, strict: bool,
              bounds: tuple[Optional[int], Optional[int]]) -> Optional[tuple[Optional[int], Optional[int]]]:
    """The integers of ``bounds`` (``None`` is no bound) with ``a*s + b >=
    0`` (``> 0`` when ``strict``); ``None`` when there is none."""
    low, high = bounds
    if a == 0:
        return bounds if (b > 0 or (b == 0 and not strict)) else None
    root = -b/a
    if a > 0:
        least = root.numerator//root.denominator + 1 if strict else -((-root.numerator)//root.denominator)
        low = least if low is None else max(low, least)
    else:
        most = -((-root.numerator)//root.denominator) - 1 if strict else root.numerator//root.denominator
        high = most if high is None else min(high, most)
    if low is not None and high is not None and low > high:
        return None
    return low, high


def _uncovered_points(b: Expr, poles: list[Expr], factorials: list[Expr]) -> Optional[list[tuple[Symbol, int]]]:
    """The points of the set ``k <= n < 0`` of the binomial ``b = binomial(n,
    k)``, where it is not the quotient of gamma functions, at which the
    expression with the ``poles`` and the ``factorials`` of
    :func:`_undefined_factors` may be finite, as pairs of a symbol and
    its value; ``None`` when the set is not known to be covered up to
    finitely many points."""
    upper, lower = _expr(b.args[0]), _expr(b.args[1])
    if upper.is_nonnegative is True or add(upper, mul(Integer(-1), lower)).is_negative is True:
        return []
    # binomial(n', k') is 0 for a negative integer k', and k <= n < 0 is in k' < 0 for k' <= k or k' <= n
    for pole in poles:
        k_pole = _expr(pole.args[1])
        if k_pole.is_integer is True and (add(lower, mul(Integer(-1), k_pole)).is_nonnegative is True
                                          or add(upper, mul(Integer(-1), k_pole)).is_nonnegative is True):
            return []
    for m in factorials:
        if m.is_integer is True and (add(lower, mul(Integer(-1), m)).is_nonnegative is True
                                     or add(upper, mul(Integer(-1), m)).is_nonnegative is True):
            return []
    symbols = [x for x in (upper.free_symbols | lower.free_symbols) if isinstance(x, Symbol)]
    if len(symbols) != 1 or symbols[0].is_integer is not True:
        return None
    symbol = symbols[0]
    n_form, k_form = _linear(upper, symbol), _linear(lower, symbol)
    if n_form is None or k_form is None:
        return None
    start: tuple[Optional[int], Optional[int]] = (None, None)
    if symbol.is_positive is True:
        start = (1, None)
    elif symbol.is_nonnegative is True:
        start = (0, None)
    elif symbol.is_negative is True:
        start = (None, -1)
    elif symbol.is_nonpositive is True:
        start = (None, 0)
    # n - k >= 0 and -n > 0
    first = _at_least(n_form[0] - k_form[0], n_form[1] - k_form[1], False, start)
    if first is None:
        return []
    second = _at_least(-n_form[0], -n_form[1], True, first)
    if second is None:
        return []
    low, high = second
    if low is None or high is None or high - low > 16:
        return None
    uncovered: list[tuple[Symbol, int]] = []
    for value in range(low, high + 1):
        at = Integer(value)
        if any(_expr(_expr(pole.args[1]).subs(symbol, at)).is_negative is True
               and _expr(_expr(pole.args[1]).subs(symbol, at)).is_integer is True for pole in poles):
            continue
        if any(_expr(m.subs(symbol, at)).is_negative is True and _expr(m.subs(symbol, at)).is_integer is True
               for m in factorials):
            continue
        uncovered.append((symbol, value))
    return uncovered


def _negative_upper_binomials(e: Expr) -> list[Expr]:
    """The binomials of ``e`` whose upper index is an integer which may be
    negative."""
    from sympy_extras.functions.combinatorial.factorials import Binomial
    found: list[Expr] = []
    for node in preorder_traversal(e):
        if isinstance(node, Binomial) and node not in found:
            upper = _expr(node.args[0])
            if upper.is_integer is True and upper.is_nonnegative is not True:
                found.append(node)
    return found


def _constant_difference(a: Expr, b: Expr) -> bool:
    return isinstance(add(a, mul(Integer(-1), b)), Integer)


def _aligned(b: Expr, c: Expr) -> bool:
    """The upper and the lower indices of the binomials differ by integers."""
    return _constant_difference(_expr(b.args[0]), _expr(c.args[0])) \
        and _constant_difference(_expr(b.args[1]), _expr(c.args[1]))


def _mirrored(b: Expr, c: Expr) -> bool:
    """The lower index of one binomial is, up to integers, the upper
    index minus the lower index of the other."""
    return _constant_difference(_expr(b.args[0]), _expr(c.args[0])) \
        and _constant_difference(add(_expr(b.args[1]), _expr(c.args[1])), _expr(b.args[0]))
