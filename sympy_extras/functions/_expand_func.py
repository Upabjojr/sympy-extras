"""The ``expand(func=True)`` rules of the special functions (SymPy's
``_eval_expand_func``), as a typed table for
:func:`sympy_extras.core.function.expand`; a rule returns ``None`` where
the port stops (a ``Sum`` or a derivative of ``zeta`` would be needed):
SymPy expands that node.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: the _eval_expand_func methods of sympy/functions/special/gamma_functions.py
# (gamma, polygamma, loggamma, digamma, trigamma), beta_functions.py (beta), zeta_functions.py (zeta,
# polylog), error_functions.py (erfc, erfi, erf2, expint), sympy/functions/combinatorial/factorials.py
# (binomial), numbers.py (harmonic) and sympy/core/numbers.py (GoldenRatio, TribonacciConstant)
# (SymPy 1.14)
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
        return None  # a derivative of zeta: SymPy's
    return polygamma(n, z)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_expand_func
def _loggamma(node: Expr, hints: ExpandHints) -> Optional[Expr]:
    from sympy_extras.functions import loggamma
    [z] = _args(node)
    if isinstance(z, Rational):
        n = z.p // z.q
        p = z.p - n * z.q
        if 0 < p < z.q:
            if n != 0:
                return None  # a Sum: SymPy's
            return loggamma(_q(p, z.q))
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
                return None  # Sums: SymPy's
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


def rules() -> dict[type, Rule]:
    """The rules keyed by the class of the node."""
    from sympy_extras.core.numbers import GoldenRatioType, TribonacciConstantType
    from sympy_extras.functions.combinatorial.factorials import Binomial
    from sympy_extras.functions.combinatorial.numbers import Harmonic
    from sympy_extras.functions.special import error_functions as er, gamma_functions as g
    from sympy_extras.functions.special.beta_functions import Beta
    from sympy_extras.functions.special.zeta_functions import PolyLog, Zeta
    return {g.Gamma: _gamma, g.PolyGamma: _polygamma, g.LogGamma: _loggamma, g.DiGamma: _digamma,
            g.TriGamma: _trigamma, Beta: _beta, Binomial: _binomial, Harmonic: _harmonic, er.Erfc: _to_erf,
            er.Erfi: _to_erf, er.Erf2: _to_erf, er.Expint: _expint, Zeta: _zeta, PolyLog: _polylog,
            GoldenRatioType: _golden_ratio, TribonacciConstantType: _tribonacci}
