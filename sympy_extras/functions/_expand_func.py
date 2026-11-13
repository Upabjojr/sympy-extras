"""The ``expand(func=True)`` rules of the special functions (SymPy's
``_eval_expand_func``), as a typed table for
:func:`sympy_extras.core.function.expand`; a rule returns ``None`` where
SymPy's rule does not apply (the node is left as it is).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: the _eval_expand_func methods of sympy/functions/special/gamma_functions.py
# (gamma, polygamma, loggamma, digamma, trigamma), beta_functions.py (beta), zeta_functions.py (zeta,
# polylog), error_functions.py (erfc, erfi, erf2, expint), sympy/functions/combinatorial/factorials.py
# (binomial), numbers.py (harmonic), sympy/functions/special/hyper.py (hyper, meijerg) and
# sympy/core/numbers.py (GoldenRatio, TribonacciConstant) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by class in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import ExpandHints
from sympy_extras.core.mul import Mul, mul, neg
from sympy_extras.core.numbers import (EulerGamma, Integer, Number, Rational, nan, number_from_fraction, pi)
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Dummy

__all__ = ['rules']

Rule = Callable[[Expr, ExpandHints], Optional[Expr]]

ONE = Integer(1)
MINUS_ONE = Integer(-1)


def _args(node: Expr) -> list[Expr]:
    return [a for a in node.args if isinstance(a, Expr)]


def _inv(e: Expr) -> Expr:
    return pow(e, MINUS_ONE)


def _q(p: int, q: int) -> Expr:
    return number_from_fraction(Fraction(p, q))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_expand_func
def _gamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import floor, gamma, rf
    [a] = _args(node)
    if isinstance(a, Rational) and abs(a.p) > a.q:
        n = a.p // a.q
        r = _q(a.p - n * a.q, a.q)
        return mul(gamma(r), rf(r, Integer(n)))
    if isinstance(a, Add):
        coeff, tail = a.as_coeff_add()
        tail_terms = list(tail)
        if isinstance(coeff, Rational) and coeff.p != 0 and coeff.q != 1:
            intpart = floor(coeff)
            tail_terms = [add(coeff, neg(intpart))] + tail_terms
            coeff = intpart if isinstance(intpart, Number) else coeff
        rest = tail_terms[0] if len(tail_terms) == 1 else Add(*tail_terms)
        return mul(gamma(rest), rf(rest, coeff))
    return gamma(a)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma._eval_expand_func
def _polygamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import cos, cot, factorial, floor, log, loggamma, polygamma, sin
    n, z = _args(node)
    if isinstance(n, Integer) and n.p >= 0:
        if isinstance(z, Add):
            coeff = _args(z)[0]
            if isinstance(coeff, Integer):
                e = neg(add(n, ONE))
                if coeff.p > 0:
                    tail = add(*[pow(add(z, Integer(-i)), e) for i in range(1, coeff.p + 1)])
                else:
                    tail = neg(add(*[pow(add(z, Integer(i)), e) for i in range(-coeff.p)]))
                return add(polygamma(n, add(z, neg(coeff))), mul(pow(MINUS_ONE, n), factorial(n), tail))
        elif isinstance(z, Mul):
            factors = _args(z)
            c = factors[0]
            rest = factors[1] if len(factors) == 2 else Mul(*factors[1:])
            if isinstance(c, Integer) and c.p > 0:
                terms = [polygamma(n, add(rest, _q(i, c.p))) for i in range(c.p)]
                if n.p == 0:
                    return add(mul(add(*terms), _inv(c)), log(c))
                return mul(add(*terms), _inv(pow(c, Integer(n.p + 1))))
    if isinstance(n, Integer) and n.p == 0 and isinstance(z, Rational):
        p, q = z.p, z.q
        part_1 = add(neg(EulerGamma), neg(mul(pi, cot(mul(Integer(p), pi, _q(1, q))), Rational(1, 2))),
                     neg(log(Integer(q))),
                     add(*[mul(cos(mul(Integer(2 * k), pi, Integer(p), _q(1, q))),
                               log(mul(Integer(2), sin(mul(Integer(k), pi, _q(1, q))))))
                           for k in range(1, q)]))
        if z.p > 0:
            m = floor(z)
            if isinstance(m, Integer):
                z0 = add(z, neg(m))
                return add(part_1, add(*[_inv(add(z0, Integer(k))) for k in range(m.p)]))
        elif z.p < 0:
            m = floor(add(ONE, neg(z)))
            if isinstance(m, Integer):
                z0 = add(z, m)
                return add(part_1, neg(add(*[_inv(add(z0, Integer(-1 - k))) for k in range(m.p)])))
    if isinstance(n, Integer) and n.p == -1:
        return add(loggamma(z), neg(mul(log(mul(Integer(2), pi)), Rational(1, 2))))
    if n.is_integer is False or n.is_nonnegative is False:
        # a derivative of zeta in its first argument
        from sympy_extras.functions import digamma, gamma, zeta
        s = Dummy('s')
        dzt = zeta(s, z).diff(s).subs(s, add(n, ONE))
        assert isinstance(dzt, Expr)
        return mul(add(dzt, mul(add(EulerGamma, digamma(neg(n))), zeta(add(n, ONE), z))), _inv(gamma(neg(n))))
    return polygamma(n, z)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_expand_func
def _loggamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import loggamma
    [z] = _args(node)
    if isinstance(z, Rational):
        n = z.p // z.q
        p = z.p - n * z.q
        if 0 < p < z.q:
            if n == 0:
                return loggamma(_q(p, z.q))
            from sympy_extras.concrete.expr_with_limits import Sum
            from sympy_extras.core.numbers import I
            from sympy_extras.core.symbol import Dummy
            from sympy_extras.functions import log
            k = Dummy('k')
            q = Integer(z.q)
            base = loggamma(_q(p, z.q)) - n * log(q)
            if n > 0:
                return base + Sum(log((k - 1) * q + p), (k, ONE, Integer(n)))
            return base + pi * I * n - Sum(log(k * q - p), (k, ONE, Integer(-n)))
    return node


def _digamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.core.function import expand
    from sympy_extras.functions import polygamma
    return expand(polygamma(Integer(0), _args(node)[0]), func=True)


def _trigamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.core.function import expand
    from sympy_extras.functions import polygamma
    return expand(polygamma(Integer(1), _args(node)[0]), func=True)


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta._eval_expand_func
def _beta(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import gamma
    x, y = _args(node)
    return mul(gamma(x), gamma(y), _inv(gamma(add(x, y))))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_expand_func
def _binomial(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from math import factorial
    from sympy_extras.functions import binomial
    n, k = _args(node)
    if isinstance(n, Number):
        return binomial(n, k)
    difference = add(n, neg(k))
    if isinstance(difference, Integer):
        k = difference
    if isinstance(k, Integer):
        if k.p == 0:
            return ONE
        if k.p < 0:
            return Integer(0)
        result: Expr = ONE
        for i in range(1, k.p + 1):
            result = mul(result, add(n, neg(k), Integer(i)))
        return mul(result, _inv(Integer(factorial(k.p))))
    return binomial(n, k)


# PORTED FROM SYMPY: sympy/functions/combinatorial/numbers.py:harmonic._eval_expand_func
def _harmonic(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import harmonic
    args = _args(node)
    n = args[0]
    m = args[1] if len(args) == 2 else ONE
    if m == ONE:
        if isinstance(n, Add):
            off = _args(n)[0]
            nnew = add(n, neg(off))
            if isinstance(off, Integer) and off.p > 0:
                return add(*([_inv(add(nnew, Integer(i))) for i in range(off.p, 0, -1)] + [harmonic(nnew)]))
            if isinstance(off, Integer) and off.p < 0:
                return add(*([neg(_inv(add(nnew, Integer(i)))) for i in range(0, off.p, -1)] + [harmonic(nnew)]))
        if isinstance(n, Rational):
            p, q = n.p, n.q
            u = p // q
            p = p - u * q
            if u >= 0 and 0 < p < q:
                from sympy_extras.concrete.expr_with_limits import sum_
                from sympy_extras.functions import cos, cot, floor, log, sin
                k = Dummy('k')
                qq, pp = Integer(q), Integer(p)
                t1 = mul(qq, sum_(_inv(add(mul(qq, k), pp)), (k, Integer(0), Integer(u))))
                t2 = mul(Integer(2), sum_(mul(cos(mul(Integer(2), pi, pp, k, _inv(qq))), log(sin(mul(pi, k, _inv(qq))))),
                                          (k, ONE, floor(mul(add(qq, MINUS_ONE), Rational(1, 2))))))
                t3 = add(mul(mul(pi, Rational(1, 2)), cot(mul(pi, pp, _inv(qq)))), log(mul(Integer(2), qq)))
                return add(t1, t2, neg(t3))
    return node


def _to_erf(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions.special.error_functions import Erf
    return node.rewrite(Erf)


def _expint(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions.special.error_functions import EiNode, Expint
    return node.rewrite(EiNode).rewrite(Expint)


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:zeta._eval_expand_func
def _zeta(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import harmonic, zeta
    args = _args(node)
    s = args[0]
    a = args[1] if len(args) > 1 else ONE
    if a.is_integer is True:
        if a.is_positive is True:
            return add(zeta(s), neg(harmonic(add(a, MINUS_ONE), s)))
        if a.is_nonpositive is True and (s.is_integer is False or s.is_nonpositive is False):
            return nan
    return node


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:polylog._eval_expand_func
def _polylog(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.core.function import diff, expand_mul
    from sympy_extras.functions import log, polylog
    s, z = _args(node)
    if s == ONE:
        return neg(log(add(ONE, neg(z))))
    if isinstance(s, Integer) and s.p <= 0:
        u = Dummy('u')
        start: Expr = mul(u, _inv(add(ONE, neg(u))))
        for _ in range(-s.p):
            start = mul(u, diff(start, u))
        rule: dict[Basic, Basic] = {u: z}
        result = expand_mul(start).subs(rule)
        if not isinstance(result, Expr):
            raise TypeError('an expression is expected')
        return result
    return polylog(s, z)


def _golden_ratio(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import sqrt
    return add(Rational(1, 2), mul(Rational(1, 2), sqrt(Integer(5))))


def _tribonacci(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import cbrt, sqrt
    r33 = sqrt(Integer(33))
    return mul(add(ONE, cbrt(add(Integer(19), neg(mul(Integer(3), r33)))), cbrt(add(Integer(19), mul(Integer(3), r33)))),
               Rational(1, 3))


# PORTED FROM SYMPY: sympy/polys/orthopolys.py:dup_spherical_bessel_fn and dup_spherical_bessel_fn_minus
def _spherical_fn_coefficients(n: int) -> list[int]:
    """The coefficients (highest degree first) of SymPy's
    ``spherical_bessel_fn(n, x)``, a polynomial in ``1/x``."""
    def lshift(f: list[int]) -> list[int]:
        return f + [0] if f else []

    def sub(f: list[int], g: list[int]) -> list[int]:
        size = max(len(f), len(g))
        f, g = [0] * (size - len(f)) + f, [0] * (size - len(g)) + g
        out = [a - b for a, b in zip(f, g)]
        while out and out[0] == 0:
            out.pop(0)
        return out

    if n >= 0:
        if n < 1:
            return [1, 0]
        m2, m1 = [1], [1, 0]
        for i in range(2, n + 1):
            m2, m1 = m1, sub([c * (2 * i - 1) for c in lshift(m1)], m2)
        return lshift(m1)
    m = -n
    m2, m1 = [1, 0], [0]
    m1 = []  # SymPy's [K.zero] is the zero polynomial
    for i in range(2, m + 1):
        m2, m1 = m1, sub([c * (3 - 2 * i) for c in lshift(m1)], m2)
    return m1


def _spherical_fn(n: int, z: Expr) -> Expr:
    """SymPy's ``spherical_bessel_fn(n, z)`` as an expression."""
    coefficients = _spherical_fn_coefficients(n)
    degree = len(coefficients) - 1
    return add(*[c * pow(z, Integer(-(degree - i))) for i, c in enumerate(coefficients) if c])


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:_jn and _yn
def _jn(n: int, z: Expr) -> Expr:
    from sympy_extras.functions import cos, sin
    return _spherical_fn(n, z) * sin(z) + MINUS_ONE**(n + 1) * _spherical_fn(-n - 1, z) * cos(z)


def _yn(n: int, z: Expr) -> Expr:
    from sympy_extras.functions import cos, sin
    return MINUS_ONE**(n + 1) * _spherical_fn(-n - 1, z) * sin(z) - _spherical_fn(n, z) * cos(z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalBesselBase._eval_expand_func with the _expand of
# jn, yn and SphericalHankelBase
def _spherical(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    """The closed forms in ``sin`` and ``cos`` for an integer order (the
    node itself otherwise)."""
    from sympy_extras.core.numbers import I
    nu, z = _args(node)
    if not isinstance(nu, Integer):
        return node
    n = nu.p
    name = type(node).__name__
    if name == 'SphericalJ':
        return _jn(n, z)
    if name == 'SphericalY':
        return _yn(n, z)
    sign = 1 if name == 'SphericalHankel1' else -1
    return (_jn(n, z) + sign * I * _yn(n, z)).expand()


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:lerchphi._eval_expand_func
def _lerchphi(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    """``zeta(s, a)`` at ``z = 1``; a rational function of ``z`` for an
    integer ``s <= 0``; for a rational ``a`` (after a shift into ``(0, 1]``)
    a sum of polylogarithms at the ``q``-th roots of ``z``; for ``z`` a root
    of unity a sum of Hurwitz zeta functions (Roach, *Hypergeometric
    function representations*, ISSAC 1997, section 18)."""
    from sympy_extras.core.function import diff
    from sympy_extras.core.numbers import I
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions import exp, exp_polar, floor, polylog, unpolarify, zeta
    from sympy_extras.functions.combinatorial.factorials import binomial
    from sympy_extras.functions.elementary.exponential import Exp
    from sympy_extras.functions.special.zeta_functions import PolyLog
    z, s, a = _args(node)
    if z == 1:
        return zeta(s, a)
    if isinstance(s, Integer) and s.p <= 0:
        t = Dummy('t')
        m = -s.p
        start: Expr = 1 / (1 - t)
        res: Expr = Integer(0)
        # the coefficients of (t + a)**m, lowest degree first
        for k in range(m + 1):
            res = res + binomial(Integer(m), Integer(k)) * pow(a, Integer(m - k)) * start
            start = t * diff(start, t)
        return res.subs(t, z)
    if isinstance(a, Rational):
        shift: Expr = Integer(0)
        factor: Expr = ONE
        if a > 1:
            n = floor(a)
            if n == a:
                n = n - 1
            assert isinstance(n, Integer)
            a = a - n
            factor = pow(z, -n)
            shift = add(*[-pow(z, Integer(k) - n) / pow(a + k, s) for k in range(int(n))])
        elif a <= 0:
            n = floor(-a) + 1
            assert isinstance(n, Integer)
            a = a + n
            factor = pow(z, n)
            shift = add(*[pow(z, n - 1 - k) / pow(a - k - 1, s) for k in range(int(n))])
        assert isinstance(a, Rational)
        p, q = a.p, a.q
        zet = exp_polar(2 * pi * I / q)
        root = pow(z, Rational(1, q))
        up_zet = unpolarify(zet)
        terms: list[Expr] = []
        for k in range(q):
            term = polylog(s, pow(zet, Integer(k)) * root)
            if isinstance(term, PolyLog):
                expanded = _polylog(term, hints)
                if expanded is not None:
                    term = expanded
            terms.append(term / pow(pow(up_zet, Integer(k)) * root, Integer(p)))
        return shift + factor * pow(Integer(q), s - 1) * add(*terms)
    if (isinstance(z, Exp) and isinstance(z.args[0] / (pi * I), Rational)) or z in (MINUS_ONE, I, -I):
        if z == MINUS_ONE:
            pp, qq = 1, 2
        elif z == I:
            pp, qq = 1, 4
        elif z == -I:
            pp, qq = -1, 4
        else:
            ratio = _args(z)[0] / (2 * pi * I)
            assert isinstance(ratio, Rational)
            pp, qq = ratio.p, ratio.q
        return add(*[exp(2 * pi * I * k * pp / qq) / pow(Integer(qq), s) * zeta(s, (k + a) / qq) for k in range(qq)])
    return node


#: SymPy's ``_a`` and ``_b`` of the Bessel classes (the signs of their recurrences)
_BESSEL_SIGNS: dict[str, tuple[int, int]] = {'BesselJ': (1, 1), 'BesselY': (1, 1), 'BesselI': (-1, 1),
                                             'BesselK': (1, -1), 'Hankel1': (1, 1), 'Hankel2': (1, 1)}


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase._eval_expand_func
def _bessel(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    """The recurrence down to an order in ``[-1, 1]``: ``f(nu)`` from
    ``f(nu - 1)`` and ``f(nu - 2)`` (or up from ``f(nu + 1)``,
    ``f(nu + 2)``), with the signs of the class; the node itself when the
    order is not real or already small."""
    nu, z = _args(node)
    a, b = _BESSEL_SIGNS[type(node).__name__]

    def at(order: Expr) -> Expr:
        g = node._evaluated((order, z))
        assert isinstance(g, Expr)
        if type(g) is type(node):
            expanded = _bessel(g, hints)
            return g if expanded is None else expanded
        return g

    if nu.is_real is True:
        if (nu - 1).is_positive is True:
            return -a * b * at(nu - 2) + 2 * a * (nu - 1) * at(nu - 1) / z
        if (nu + 1).is_negative is True:
            return 2 * b * (nu + 1) * at(nu + 1) / z - a * b * at(nu + 2)
    return node


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper._eval_expand_func
def _hyper(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions.special.gamma_functions import gamma
    from sympy_extras.functions.special.hyper import Hyper
    from sympy_extras.simplify.hyperexpand import hyperexpand
    assert isinstance(node, Hyper)
    if len(node.ap) == 2 and len(node.bq) == 1 and node.argument == ONE:
        a, b = node.ap
        c = node.bq[0]
        return mul(gamma(c), gamma(add(c, neg(a), neg(b))), _inv(gamma(add(c, neg(a)))), _inv(gamma(add(c, neg(b)))))
    return hyperexpand(node)


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:meijerg._eval_expand_func
def _meijerg(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.simplify.hyperexpand import hyperexpand
    return hyperexpand(node)


def rules() -> dict[type, Rule]:
    """The rules keyed by the class of the node."""
    from sympy_extras.core.numbers import GoldenRatioType, TribonacciConstantType
    from sympy_extras.functions.combinatorial.factorials import Binomial
    from sympy_extras.functions.combinatorial.numbers import Harmonic
    from sympy_extras.functions.special import error_functions as er, gamma_functions as g
    from sympy_extras.functions.special import bessel as bs
    from sympy_extras.functions.special.beta_functions import Beta
    from sympy_extras.functions.special.hyper import Hyper, MeijerG
    from sympy_extras.functions.special.zeta_functions import PolyLog, Zeta
    bessel: dict[type, Rule] = {c: _bessel for c in (bs.BesselJ, bs.BesselY, bs.BesselI, bs.BesselK, bs.Hankel1, bs.Hankel2)}
    bessel.update({c: _spherical for c in (bs.SphericalJ, bs.SphericalY, bs.SphericalHankel1, bs.SphericalHankel2)})
    from sympy_extras.functions.special.zeta_functions import LerchPhi
    bessel[LerchPhi] = _lerchphi
    from sympy_extras.functions.special._expand_func_bessel import rules as airy_ynm_rules
    bessel.update(airy_ynm_rules())
    return bessel | {g.Gamma: _gamma, g.PolyGamma: _polygamma, g.LogGamma: _loggamma, g.DiGamma: _digamma,
            g.TriGamma: _trigamma, Beta: _beta, Binomial: _binomial, Harmonic: _harmonic, er.Erfc: _to_erf,
            er.Erfi: _to_erf, er.Erf2: _to_erf, er.Expint: _expint, Zeta: _zeta, PolyLog: _polylog,
            GoldenRatioType: _golden_ratio, TribonacciConstantType: _tribonacci, Hyper: _hyper, MeijerG: _meijerg}
