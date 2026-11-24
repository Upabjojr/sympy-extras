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
from sympy_extras.core.numbers import Integer, Rational, nan, number_from_fraction, oo, pi, zoo
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
def gammasimp(expr: ExprLike, generic: bool = False) -> Expr:
    """The expression with its gamma functions (and the functions
    rewritten as gamma) simplified.

    The result has the values of the expression at the integer points
    which the assumptions of the symbols allow (see :func:`combsimp`);
    with ``generic`` it is the simplified quotient of gamma functions,
    an identity of meromorphic functions, as the algorithms on
    hypergeometric terms need it."""
    given = to_expr(expr)
    found = _gammasimp_generic(given)
    if found == given:
        return given
    if generic:
        return found
    # a binomial with an upper index which may be a negative integer is not the quotient of gamma functions
    # at every integer point (gammasimp(binomial(n, k)*factorial(k)) was gamma(n + 1)/gamma(n - k + 1), nan
    # for a negative integer n, where the product is n*(n - 1)*...*(n - k + 1))
    return _checked_at_integer_points(given, found, _negative_upper_binomials(given), gammasimp, True)


# PORTED FROM SYMPY: sympy/simplify/gammasimp.py:gammasimp
def _gammasimp_generic(given: Expr) -> Expr:
    """:func:`gammasimp` for generic values of the symbols (the integer
    points where a binomial is not its quotient of gamma functions are
    not looked at)."""
    from sympy_extras.functions.special.gamma_functions import Gamma
    e = _expr(given.rewrite(Gamma))
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
def combsimp(expr: ExprLike, generic: bool = False) -> Expr:
    """The combinatorial expression (factorials, binomials, rising and
    falling factorials) simplified, as factorials and binomials; through
    :func:`gammasimp` when a gamma argument is not an integer.

    The result has the value of the expression at every integer point
    which the assumptions of the symbols allow and where the expression
    is finite: where that is not known for a binomial whose upper index
    may be a negative integer, the result has cases for finitely many
    values of a symbol, or the expression is returned as it is. With
    ``generic`` the points ``n < 0 <= k`` of a binomial are not looked
    at (the result is then an identity of meromorphic functions which
    may be ``0/0`` or ``zoo/zoo`` at such points), as the algorithms on
    hypergeometric terms need it.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import binomial, factorial
    >>> n, k = Symbol('n', integer=True), Symbol('k', integer=True)
    >>> combsimp(binomial(n, k)*factorial(k)), combsimp(binomial(n, k)*factorial(k), generic=True)
    (binomial(n, k)*factorial(k), factorial(n)/factorial(-k + n))
    >>> p, q = Symbol('p', integer=True, nonnegative=True), Symbol('q', integer=True, nonnegative=True)
    >>> combsimp(binomial(p, q)*factorial(q)), combsimp(binomial(p, q)/binomial(p - 1, q))
    (factorial(p)/factorial(p - q), Piecewise((binomial(0, q)/binomial(-1, q), Eq(p, 0)), (p/(p - q), True)))
    """
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
        result = _gammasimp_generic(e)
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
    checked = [b for b in risky if b not in conflicts]
    checked.extend(b for b in _negative_upper_binomials(found) if b not in risky)
    return _checked_at_integer_points(given, found, checked, lambda term: combsimp(term, generic), not generic)


def _checked_at_integer_points(given: Expr, found: Expr, checked: list[Expr], again: Callable[[Expr], Expr],
                               above: bool) -> Expr:
    """``found``, the simplified form of ``given`` through gamma functions,
    where it holds at the integer points of the binomials ``checked``
    (those whose upper index may be a negative integer); with cases for
    finitely many values of one symbol where it does not, and ``given``
    (a sum term by term, with ``again``) where that is not known.

    The quotient of gamma functions is not the binomial where ``k <= n <
    0`` (``binomial(-1, -1)`` is 0, the gamma functions have the limit 1:
    ``combsimp(binomial(2*n + 1, n)/binomial(2*n + 1, n + 1))`` was 1, and
    the quotient is 0 at ``n = -1``). Where ``n < 0 <= k`` the binomial is
    the limit in ``n`` of a quotient of two gamma functions at their
    poles, which the simplified form need not have: ``binomial(n,
    k)*factorial(k)`` became ``factorial(n)/factorial(n - k)``, ``nan``
    for every negative ``n``, and ``binomial(n, k)/binomial(n - 1, k)``
    became ``n/(n - k)``, ``0/0`` at ``n = k = 0``, where it is 1. The
    second set is looked at with ``above`` only."""
    poles, factorials = _undefined_factors(given)
    if above and not _numerators_finite(found, poles, factorials):
        if isinstance(given, Add):
            return add(*[again(term) for term in given.args if isinstance(term, Expr)])
        return given
    if not checked:
        return found
    points: list[tuple[Symbol, int]] = []
    sound = True
    for b in checked:
        uncovered = _uncovered_points(b, poles, factorials, found if above else None)
        if uncovered is None:
            sound = False
            break
        points.extend(point for point in uncovered if point not in points)
    if not sound and above:
        on_lines = _line_points(given, found, checked)
        if on_lines is not None:
            sound, points = True, on_lines
    if sound and not points:
        return found
    if sound and len({symbol for symbol, _ in points}) == 1:
        from sympy_extras.core.logic import Boolean
        from sympy_extras.core.relational import eq
        from sympy_extras.functions.elementary.piecewise import piecewise
        cases: list[tuple[Expr, Boolean]] = []
        for symbol, value in sorted(points, key=lambda point: point[1]):
            at_point = _expr(given.subs(symbol, Integer(value)))
            if at_point == _expr(found.subs(symbol, Integer(value))):
                continue
            # a number which is not finite is no value of the expression; with other symbols the case is the
            # expression at the point, as it stands
            if at_point.is_finite is True or (at_point.free_symbols and at_point.is_finite is not False
                                              and not at_point.has(zoo, nan, oo)):
                cases.append((at_point, eq(symbol, Integer(value))))
        if not cases:
            return found
        return piecewise(*cases, (found, True))
    if isinstance(given, Add):
        # term by term: a binomial which stays as it is needs no proof
        return add(*[again(term) for term in given.args if isinstance(term, Expr)])
    return given


def _numerators_finite(found: Expr, poles: list[Expr], factorials: list[Expr]) -> bool:
    """Whether every factorial and gamma function of a numerator of
    ``found`` is finite wherever the expression with the ``poles`` and
    the ``factorials`` of :func:`_undefined_factors` may be: its argument
    is not a negative integer (it is not an integer, or not negative), or
    the expression is known not to be finite where it is one.

    ``combsimp(binomial(0, n - k)/binomial(0, k - 2))`` was ``(k -
    2)*factorial(-k)*factorial(k - 1)/(factorial(n - k)*factorial(k -
    n))``, ``nan`` at ``k = 2``, where the quotient is ``binomial(0, n -
    2)`` (the product of the factorials of ``-k`` and of ``k - 1`` has a
    limit, which the evaluation does not take)."""
    from sympy_extras.functions.combinatorial.factorials import Factorial
    for term in Add.make_args(found):
        for factor in Mul.make_args(term):
            base, exponent = factor.as_base_exp()
            if not isinstance(base, (Factorial, _gamma_types()[0])) or exponent.is_negative is True:
                continue
            argument = _expr(base.args[0])
            m = argument if isinstance(base, Factorial) else add(argument, Integer(-1))
            if m.is_integer is not True or m.is_nonnegative is True:
                continue
            if not _covered([m], poles, factorials):
                return False
    return True


def _line_points(given: Expr, found: Expr, checked: list[Expr]) -> Optional[list[tuple[Symbol, int]]]:
    """The values of a symbol, as pairs, outside which the rational
    function ``found`` is known to be ``given`` at every integer point
    where ``given`` is finite; ``None`` when this argument does not apply.

    When a symbol ``s`` is in the upper index of every binomial of
    ``checked`` and in no lower index of them, every factor of ``given``
    is, on a line where the other symbols are integers, a meromorphic
    function of ``s`` which has the value of the factor wherever that is
    finite (``binomial(u(s), k)`` is a polynomial in ``u`` for an integer
    ``k >= 0`` and 0 for ``k < 0``; a factorial and a gamma function are
    themselves), and it is on that line the quotient of gamma functions
    which was simplified. A rational function equal to it has its value
    wherever its denominator is not 0: the roots of the denominator are
    the values returned (``binomial(n + 1, k + 1)/binomial(n, k)`` is ``(n +
    1)/(k + 1)`` outside ``k = -1``, where it is not finite); a factor of
    the denominator in several symbols (``n/(n - k)`` for ``binomial(n,
    k)/binomial(n - 1, k)``, which is 1 at ``n = k = 0``) is not handled."""
    from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial
    from sympy_extras.polys.polytools import cancel, factor_list
    from sympy_extras.simplify.radsimp import fraction
    if any(not isinstance(f, (Binomial, Factorial, _gamma_types()[0])) for f in given.atoms(Application)):
        return None
    if found.atoms(Application) or found.is_rational_function() is not True:
        return None
    lines: Optional[set[Symbol]] = None
    for b in checked:
        upper, lower = _expr(b.args[0]), _expr(b.args[1])
        own = {x for x in upper.free_symbols - lower.free_symbols if upper.is_polynomial(x) is True}
        lines = own if lines is None else lines & own
    if not lines:
        return None
    denominator = fraction(cancel(found))[1]
    points: list[tuple[Symbol, int]] = []
    for base, _ in factor_list(denominator)[1]:
        symbols = [x for x in base.free_symbols if isinstance(x, Symbol)]
        if len(symbols) != 1:
            return None
        form = _linear(base, symbols[0])
        if form is None or form[0] == 0:
            return None
        root = -form[1]/form[0]
        low, high = _symbol_bounds(symbols[0])
        if root.denominator != 1 or symbols[0].is_integer is False or (low is not None and root < low) \
                or (high is not None and root > high):
            continue
        if (symbols[0], root.numerator) not in points:
            points.append((symbols[0], root.numerator))
    return points


def _undefined_factors(e: Expr) -> tuple[list[Expr], list[Expr]]:
    """The binomials which are factors of the denominator of a term of
    ``e`` and the arguments of the factorials which are factors of the
    numerator of a term: where such a binomial is zero, or such an
    argument a negative integer, the term (and ``e``) is not finite. A
    gamma function of a numerator counts as the factorial it is."""
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
            elif isinstance(base, _gamma_types()[0]) and exponent.is_positive is True:
                # gamma(m + 1) is factorial(m)
                factorials.append(add(_expr(base.args[0]), Integer(-1)))
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


def _covered(shifts: list[Expr], poles: list[Expr], factorials: list[Expr]) -> bool:
    """Whether the expression is known not to be finite on a set of
    integer points where every one of ``shifts`` is negative: a binomial
    of the denominator with a lower index ``k'`` (it is 0 for a negative
    integer ``k'``) or a factorial of the numerator of ``m`` with ``k' <=
    shift`` or ``m <= shift`` for one of them, or two of these indices
    with a sum which is at most a shift (one of them is negative then:
    ``binomial(n, k)*factorial(k)*factorial(n - k)`` is not finite for
    ``n < 0``)."""
    indices = [k for k in [_expr(pole.args[1]) for pole in poles] + factorials if k.is_integer is True]
    for shift in shifts:
        for i, first in enumerate(indices):
            rest = add(shift, mul(Integer(-1), first))
            if rest.is_nonnegative is True:
                return True
            if any(add(rest, mul(Integer(-1), second)).is_nonnegative is True for second in indices[i + 1:]):
                return True
    return False


def _symbol_bounds(symbol: Symbol) -> tuple[Optional[int], Optional[int]]:
    if symbol.is_positive is True:
        return (1, None)
    if symbol.is_nonnegative is True:
        return (0, None)
    if symbol.is_negative is True:
        return (None, -1)
    if symbol.is_nonpositive is True:
        return (None, 0)
    return (None, None)


Interval = tuple[Optional[int], Optional[int]]


def _where_finite(interval: Interval, symbol: Symbol, poles: list[Expr], factorials: list[Expr]) -> Optional[Interval]:
    """The integers of ``interval`` (values of ``symbol``) where no
    binomial of ``poles`` has a negative lower index and no argument of
    ``factorials`` is negative, among those which are ``a*symbol + b``
    with rational numbers: elsewhere the expression is not finite.
    ``None`` when there is none."""
    found: Optional[Interval] = interval
    for index in [_expr(pole.args[1]) for pole in poles] + factorials:
        if found is None:
            break
        if index.is_integer is not True or index.free_symbols != {symbol}:
            continue
        form = _linear(index, symbol)
        if form is not None:
            found = _at_least(form[0], form[1], False, found)
    return found


def _irregular_values(found: Expr, symbol: Symbol, interval: Interval) -> Optional[list[int]]:
    """The integers of ``interval``, values of ``symbol``, at which the
    evaluation of ``found`` may not be the limit of ``found``: where a
    factorial or a gamma function of a numerator has a pole (``zoo``, or
    ``zoo/zoo``) or a polynomial of a denominator a root. Elsewhere every
    factor of every term is continuous (the inverse of a factorial is 0 at
    its poles, a binomial has its own points). ``None`` when they are not
    known to be finitely many (a factor with another symbol, a binomial in
    a denominator, a function which is not known)."""
    from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial
    if found.has(zoo, nan, oo):
        return None
    values: list[int] = []
    for term in Add.make_args(found):
        for factor in Mul.make_args(term):
            base, exponent = factor.as_base_exp()
            if symbol not in base.free_symbols:
                if symbol in exponent.free_symbols and (base.free_symbols or not isinstance(base, Rational) or base.p == 0):
                    return None
                continue
            if symbol in exponent.free_symbols or not isinstance(exponent, Rational) or base.free_symbols != {symbol}:
                return None
            where: Optional[Interval] = None
            if isinstance(base, (Factorial, _gamma_types()[0])):
                if exponent.p < 0:
                    continue
                argument = _expr(base.args[0])
                form = _linear(argument if isinstance(base, Factorial) else add(argument, Integer(-1)), symbol)
                if form is None:
                    return None
                where = _at_least(-form[0], -form[1], True, interval)
            elif isinstance(base, Binomial):
                if exponent.p < 0:
                    return None
                continue
            elif base.is_polynomial(symbol) is True:
                if exponent.p > 0 and exponent.q == 1:
                    continue
                form = _linear(base, symbol)
                if form is None or form[0] == 0:
                    return None
                root = -form[1]/form[0]
                if root.denominator == 1 and (interval[0] is None or interval[0] <= root) \
                        and (interval[1] is None or root <= interval[1]):
                    where = (root.numerator, root.numerator)
            else:
                return None
            if where is None:
                continue
            low, high = where
            if low is None or high is None or high - low > 16:
                return None
            values.extend(value for value in range(low, high + 1) if value not in values)
    return values


def _uncovered_points(b: Expr, poles: list[Expr], factorials: list[Expr],
                      found: Optional[Expr]) -> Optional[list[tuple[Symbol, int]]]:
    """The points with a negative upper index of the binomial ``b =
    binomial(n, k)`` at which the expression with the ``poles`` and the
    ``factorials`` of :func:`_undefined_factors` may be finite and not
    equal to its simplified form, as pairs of a symbol and its value
    (every point with that value of the symbol is meant); ``None`` when
    they are not known to be finitely many values of one symbol.

    Where ``k <= n < 0`` the binomial is 0 and not the quotient of gamma
    functions; these points alone are looked at when ``found``, the
    simplified form, is not given. Where ``n < 0 <= k`` it is the limit
    in ``n`` alone of the quotient, of gamma functions at their poles,
    which a simplified form need not have. Where ``n < k < 0`` the
    binomial is 0, as every limit of the quotient is, and the simplified
    form has that limit where its evaluation is continuous
    (:func:`_irregular_values`, for a binomial in one symbol; every point
    otherwise). A lower index which is a number has no such point: the
    binomial is a polynomial in ``n``, or 0."""
    upper, lower = _expr(b.args[0]), _expr(b.args[1])
    everywhere = found is not None
    if upper.is_nonnegative is True:
        return []
    if lower.is_integer is not True:
        # a lower index which is not an integer (a symbol of no assumption is generic, as elsewhere)
        return []
    if everywhere and isinstance(lower, Integer):
        return []
    difference = add(upper, mul(Integer(-1), lower))
    # k < 0 (and k <= n when not everywhere): binomial(n', k') is 0 for a negative integer k', and the set is in
    # k' < 0 for k' <= k or k' <= n
    below = not (lower.is_nonnegative is True or (not everywhere and difference.is_negative is True)
                 or _covered([lower, upper], poles, factorials))
    # n < 0 <= k
    above = everywhere and not (lower.is_negative is True or _covered([upper, difference], poles, factorials))
    if not below and not above:
        return []
    symbols = [x for x in upper.free_symbols if isinstance(x, Symbol)]
    if len(symbols) != 1 or symbols[0].is_integer is not True:
        return None
    symbol = symbols[0]
    n_form = _linear(upper, symbol)
    if n_form is None:
        return None
    # -n > 0
    negative = _at_least(-n_form[0], -n_form[1], True, _symbol_bounds(symbol))
    if negative is None:
        return []
    #: the intervals of values of the symbol, and whether every point of them is meant (or the irregular ones)
    intervals: list[tuple[Interval, bool]] = []
    if lower.free_symbols - {symbol}:
        if not everywhere:
            # (k <= n < 0 with another symbol in the lower index: not looked into)
            return None
        intervals.append((negative, True))
    else:
        k_form = _linear(lower, symbol)
        if k_form is None:
            return None
        if below:
            # -k > 0 with n - k >= 0, and with k - n > 0
            first = _at_least(-k_form[0], -k_form[1], True, negative)
            if first is not None:
                inside = _at_least(n_form[0] - k_form[0], n_form[1] - k_form[1], False, first)
                if inside is not None:
                    intervals.append((inside, True))
                outside = _at_least(k_form[0] - n_form[0], k_form[1] - n_form[1], True, first)
                if outside is not None and everywhere:
                    intervals.append((outside, False))
        if above:
            # k >= 0: the limit of the gamma functions with both indices moving is not the binomial (it has the
            # factor (a - c)/a for the indices a*s + b and c*s + d: binomial(s - 5, 2*s) is 10 at s = 1, the
            # limit -10)
            second = _at_least(k_form[0], k_form[1], False, negative)
            if second is not None:
                intervals.append((second, True))
    values: list[int] = []
    for interval, whole in intervals:
        finite = _where_finite(interval, symbol, poles, factorials)
        if finite is None:
            continue
        if not whole and found is not None:
            irregular = _irregular_values(found, symbol, finite)
            if irregular is not None:
                values.extend(value for value in irregular if value not in values)
                continue
        low, high = finite
        if low is None or high is None or high - low > 16:
            return None
        values.extend(value for value in range(low, high + 1) if value not in values)
    uncovered: list[tuple[Symbol, int]] = []
    for value in sorted(values):
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
