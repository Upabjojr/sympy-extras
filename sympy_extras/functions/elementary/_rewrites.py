"""The ``rewrite`` rules of the elementary functions in terms of ``exp``,
``cos`` and ``sin`` (SymPy's ``_eval_rewrite_as_exp``,
``_eval_rewrite_as_cos`` and ``_eval_rewrite_as_sin``), of the
trigonometric, hyperbolic and inverse functions in terms of each other, of
square roots (``'sqrt'``: the cosines of rational multiples of ``pi`` with
Fermat prime denominators), powers and Bessel functions, as a typed table
for :func:`sympy_extras.core.rewriting.rewrite`. Ported from SymPy's
``sympy.functions.elementary.trigonometric``, ``hyperbolic`` and
``exponential`` modules.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py,
# sympy/functions/elementary/hyperbolic.py, sympy/functions/elementary/exponential.py,
# sympy/functions/elementary/complexes.py (sign._eval_rewrite_as_Abs),
# sympy/core/power.py (the _eval_rewrite_as_exp, _eval_rewrite_as_cos and _eval_rewrite_as_sin
# methods; the rules of the trigonometric, hyperbolic and inverse functions between themselves and
# into tan, cot, sec, csc, sincos, sinc, besselj, Pow, pow, sqrt; exp, Pow and E as tanh, sqrt,
# Pow, sin, cos), sympy/core/numbers.py (Exp1), sympy/functions/elementary/
# _trigonometric_special.py (cos_table, migcdex, ipartfrac, fermat_coords) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target) in place of methods.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.rewriting import RewriteHints, RewriteTarget, RuleKey
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import E, I, Integer, Rational, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from . import hyperbolic as h, trigonometric as t
from .exponential import Exp, exp, log

#: a rule: the node and its (rewritten) arguments to the rewritten node,
#: ``None`` when the rule does not apply
Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]

HALF_PI = mul(pi, Rational(1, 2))


def _inv(e: Expr) -> Expr:
    return pow(e, Integer(-1))


def _neg(e: Expr) -> Expr:
    from sympy_extras.core.mul import neg
    return neg(e)


def _trig_or_hyper(e: Expr) -> bool:
    return isinstance(e, (t.Sin, t.Cos, t.Tan, t.Cot, t.Sec, t.Csc, h.Sinh, h.Cosh, h.Tanh, h.Coth, h.Sech, h.Csch))


def _exp_arg(a: Expr) -> Expr:
    """SymPy's re-rewrite of a trigonometric or hyperbolic argument."""
    if _trig_or_hyper(a):
        from sympy_extras.core.rewriting import rewrite
        rebuilt = a._evaluated(a.args)
        if isinstance(rebuilt, Expr):
            r = rewrite(rebuilt, (), Exp, RewriteHints())
            if isinstance(r, Expr):
                return r
    return a


# trigonometric, as exp

def _sin_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    return mul(add(exp(mul(a, I)), _neg(exp(mul(_neg(a), I)))), _inv(mul(Integer(2), I)))


def _cos_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    return mul(add(exp(mul(a, I)), exp(mul(_neg(a), I))), Rational(1, 2))


def _tan_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    neg_exp, pos_exp = exp(mul(_neg(a), I)), exp(mul(a, I))
    return mul(I, add(neg_exp, _neg(pos_exp)), _inv(add(neg_exp, pos_exp)))


def _cot_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = _exp_arg(args[0])
    neg_exp, pos_exp = exp(mul(_neg(a), I)), exp(mul(a, I))
    return mul(I, add(pos_exp, neg_exp), _inv(add(pos_exp, _neg(neg_exp))))


def _reciprocal(rule: Rule, of: Callable[[Expr], Expr]) -> Rule:
    """SymPy's ``_rewrite_reciprocal``: ``1/t`` with ``t`` the rule of the
    reciprocal function, unless it leaves that function unchanged."""
    def rewritten(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        r = rule(node, args, hints)
        if r is not None and r != of(args[0]):
            return _inv(r)
        return None
    return rewritten


# trigonometric, as cos

def _sin_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return t.Cos(add(args[0], _neg(HALF_PI)))


def _tan_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(t.Cos(add(args[0], _neg(HALF_PI))), _inv(t.cos(args[0])))


def _cot_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(t.cos(args[0]), _inv(t.Cos(add(args[0], _neg(HALF_PI)))))


def _sec_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _inv(t.cos(args[0]))


def _csc_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.rewriting import rewrite
    s = t.sin(args[0])
    r = rewrite(s, (), t.Cos, hints)
    return _inv(r) if isinstance(r, Expr) else None


# hyperbolic

def _sinh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(add(exp(args[0]), _neg(exp(_neg(args[0])))), Rational(1, 2))


def _cosh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return mul(add(exp(args[0]), exp(_neg(args[0]))), Rational(1, 2))


def _tanh_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    neg_exp, pos_exp = exp(_neg(args[0])), exp(args[0])
    return mul(add(pos_exp, _neg(neg_exp)), _inv(add(pos_exp, neg_exp)))


def _coth_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    neg_exp, pos_exp = exp(_neg(args[0])), exp(args[0])
    return mul(add(pos_exp, neg_exp), _inv(add(pos_exp, _neg(neg_exp))))


def _cosh_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return t.Cos(mul(I, args[0]))


def _sech_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _inv(t.Cos(mul(I, args[0])))


# exp and powers

def _exp_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    a = mul(I, args[0])
    return add(t.cos(a), mul(I, t.cos(add(a, HALF_PI))))


def _pow_exp(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import abs_, arg
    base, expo = args
    if base.is_zero is True or base.has(Exp) or expo.has(Exp) or base == E:
        return pow(base, expo)
    if base.has(Symbol):
        # unevaluated without symbols in the exponent: exp(x*log(5)) is x**5
        e = mul(log(base), expo)
        return exp(e) if expo.has(Symbol) else Exp(e)
    return exp(mul(add(log(abs_(base)), mul(I, arg(base))), expo))


def _pow_cos(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    if isinstance(node, Pow) and node.base == E:
        return _exp_cos(node, (node.exp,), hints)
    return None


# PORTED FROM SYMPY: the _eval_rewrite_as_sin methods of cos, tan, cot, sec, csc
# (sympy/functions/elementary/trigonometric.py), sinh, csch (hyperbolic.py), exp
# (exponential.py) and Pow (sympy/core/power.py)
def _sin_rules() -> dict[RuleKey, Rule]:
    """The trigonometric and hyperbolic functions and the exponentials in
    terms of ``sin``."""
    two = Integer(2)

    def cos_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.Sin(add(args[0], HALF_PI))

    def tan_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        x = args[0]
        return mul(two, pow(t.sin(x), two), _inv(t.sin(mul(two, x))))

    def cot_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        x = args[0]
        return mul(t.sin(mul(two, x)), _inv(mul(two, pow(t.sin(x), two))))

    def sec_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        from sympy_extras.core.rewriting import rewrite
        r = rewrite(t.cos(args[0]), (), t.Sin, hints)
        return _inv(r) if isinstance(r, Expr) else None

    def csc_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return _inv(t.sin(args[0]))

    def exp_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = mul(I, args[0])
        return add(t.sin(add(a, HALF_PI)), _neg(mul(I, t.sin(a))))

    def pow_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        if isinstance(e, Pow) and e.base == E:
            return exp_sin(e, (e.exp,), hints)
        return None

    def sinh_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return _neg(mul(I, t.sin(mul(I, args[0]))))

    def csch_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return mul(I, _inv(t.Sin(mul(I, args[0]))))

    return {(t.Cos, t.Sin): cos_sin, (t.Tan, t.Sin): tan_sin, (t.Cot, t.Sin): cot_sin, (t.Sec, t.Sin): sec_sin,
            (t.Csc, t.Sin): csc_sin, (Exp, t.Sin): exp_sin, (Pow, t.Sin): pow_sin, (h.Sinh, t.Sin): sinh_sin,
            (h.Csch, t.Sin): csch_sin}


# PORTED FROM SYMPY: the _eval_rewrite_as_log methods of asin, acos, atan, acot, asec, acsc
# (sympy/functions/elementary/trigonometric.py), asinh, acosh, atanh, acoth, asech, acsch
# (sympy/functions/elementary/hyperbolic.py), and cot._eval_rewrite_as_tan
def _log_rules() -> dict[RuleKey, Rule]:
    """The inverse trigonometric and hyperbolic functions as logarithms."""
    from sympy_extras.core.power import sqrt
    from .exponential import Log
    one = Integer(1)

    def asin(x: Expr) -> Expr:
        return -I * log(I * x + sqrt(1 - x**2))

    def acos(x: Expr) -> Expr:
        return HALF_PI + I * log(I * x + sqrt(1 - x**2))

    def atan(x: Expr) -> Expr:
        return I / 2 * (log(one - I * x) - log(one + I * x))

    def acot(x: Expr) -> Expr:
        return I / 2 * (log(1 - I / x) - log(1 + I / x))

    def asec(x: Expr) -> Expr:
        return HALF_PI + I * log(I / x + sqrt(1 - 1 / x**2))

    def acsc(x: Expr) -> Expr:
        return -I * log(I / x + sqrt(1 - 1 / x**2))

    def asinh(x: Expr) -> Expr:
        return log(x + sqrt(x**2 + 1))

    def acosh(x: Expr) -> Expr:
        return log(x + sqrt(x + 1) * sqrt(x - 1))

    def atanh(x: Expr) -> Expr:
        return (log(1 + x) - log(1 - x)) / 2

    def acoth(x: Expr) -> Expr:
        return (log(1 + 1 / x) - log(1 - 1 / x)) / 2

    def asech(x: Expr) -> Expr:
        return log(1 / x + sqrt(1 / x - 1) * sqrt(1 / x + 1))

    def acsch(x: Expr) -> Expr:
        return log(1 / x + sqrt(1 / x**2 + 1))

    def of(f: Callable[[Expr], Expr]) -> Rule:
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            return f(args[0])
        return rule

    def cot_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.tan(args[0])

    return {(t.Asin, Log): of(asin), (t.Acos, Log): of(acos), (t.Atan, Log): of(atan), (t.Acot, Log): of(acot),
            (t.Asec, Log): of(asec), (t.Acsc, Log): of(acsc), (h.Asinh, Log): of(asinh), (h.Acosh, Log): of(acosh),
            (h.Atanh, Log): of(atanh), (h.Acoth, Log): of(acoth), (h.Asech, Log): of(asech),
            (h.Acsch, Log): of(acsch), (t.Cot, t.Tan): cot_tan}


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name)."""
    sec_exp_r = _reciprocal(_cos_exp, t.cos)
    csc_exp_r = _reciprocal(_sin_exp, t.sin)
    sech_exp_r = _reciprocal(_cosh_exp, h.cosh)
    csch_exp_r = _reciprocal(_sinh_exp, h.sinh)
    table: dict[RuleKey, Rule] = {
        (t.Sin, Exp): _sin_exp, (t.Cos, Exp): _cos_exp, (t.Tan, Exp): _tan_exp, (t.Cot, Exp): _cot_exp,
        (t.Sec, Exp): sec_exp_r, (t.Csc, Exp): csc_exp_r,
        (t.Sin, t.Cos): _sin_cos, (t.Tan, t.Cos): _tan_cos, (t.Cot, t.Cos): _cot_cos, (t.Sec, t.Cos): _sec_cos,
        (t.Csc, t.Cos): _csc_cos,
        (h.Sinh, Exp): _sinh_exp, (h.Cosh, Exp): _cosh_exp, (h.Tanh, Exp): _tanh_exp, (h.Coth, Exp): _coth_exp,
        (h.Sech, Exp): sech_exp_r, (h.Csch, Exp): csch_exp_r,
        (h.Cosh, t.Cos): _cosh_cos, (h.Sech, t.Cos): _sech_cos,
        (Exp, t.Cos): _exp_cos, (Pow, Exp): _pow_exp, (Pow, t.Cos): _pow_cos}
    table.update(_log_rules())
    table.update(_sin_rules())
    table.update(_trig_rules())
    table.update(_hyperbolic_rules())
    table.update(_inverse_trig_rules())
    table.update(_inverse_hyperbolic_rules())
    table.update(_exp_rules())
    from .complexes import Abs, Sign
    table[(Sign, Abs)] = _sign_abs
    return table


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_rewrite_as_Abs
def _sign_abs(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq
    from .complexes import abs_
    from .piecewise import piecewise
    arg = args[0]
    return piecewise((Integer(0), eq(arg, Integer(0))), (mul(arg, _inv(abs_(arg))), True))


# ---------------------------------------------------------------------
# the trigonometric and hyperbolic functions in terms of each other, of
# square roots, powers and Bessel functions

def _rw(e: Expr, target: RewriteTarget, hints: RewriteHints) -> Expr:
    """The native ``e.rewrite(target)`` (SymPy's ``self.rewrite`` inside a
    rule)."""
    from sympy_extras.core.rewriting import rewrite
    r = rewrite(e, (), target, hints)
    assert isinstance(r, Expr)
    return r


def _cos_3() -> Expr:
    return Rational(1, 2)


def _cos_5() -> Expr:
    from sympy_extras.core.power import sqrt
    return (sqrt(Integer(5)) + 1) / 4


def _cos_17() -> Expr:
    from sympy_extras.core.power import sqrt
    s17, s2 = sqrt(Integer(17)), sqrt(Integer(2))
    return sqrt((15 + s17) / 32 + s2 * (sqrt(17 - s17) + sqrt(s2 * (-8 * sqrt(17 + s17) - (1 - s17)
                                                                    * sqrt(17 - s17)) + 6 * s17 + 34)) / 32)


def _cos_257() -> Expr:
    from sympy_extras.core.power import sqrt

    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        return (a + sqrt(a**2 + b)) / 2, (a - sqrt(a**2 + b)) / 2

    def f2(a: Expr, b: Expr) -> Expr:
        return (a - sqrt(a**2 + b)) / 2

    t1, t2 = f1(Integer(-1), Integer(256))
    z1, z3 = f1(t1, Integer(64))
    z2, z4 = f1(t2, Integer(64))
    y1, y5 = f1(z1, 4 * (5 + t1 + 2 * z1))
    y6, y2 = f1(z2, 4 * (5 + t2 + 2 * z2))
    y3, y7 = f1(z3, 4 * (5 + t1 + 2 * z3))
    y8, y4 = f1(z4, 4 * (5 + t2 + 2 * z4))
    x1, x9 = f1(y1, -4 * (t1 + y1 + y3 + 2 * y6))
    x2, x10 = f1(y2, -4 * (t2 + y2 + y4 + 2 * y7))
    x3, x11 = f1(y3, -4 * (t1 + y3 + y5 + 2 * y8))
    x4, x12 = f1(y4, -4 * (t2 + y4 + y6 + 2 * y1))
    x5, x13 = f1(y5, -4 * (t1 + y5 + y7 + 2 * y2))
    x6, x14 = f1(y6, -4 * (t2 + y6 + y8 + 2 * y3))
    x15, x7 = f1(y7, -4 * (t1 + y7 + y1 + 2 * y4))
    x8, x16 = f1(y8, -4 * (t2 + y8 + y2 + 2 * y5))
    v1 = f2(x1, -4 * (x1 + x2 + x3 + x6))
    v2 = f2(x2, -4 * (x2 + x3 + x4 + x7))
    v3 = f2(x8, -4 * (x8 + x9 + x10 + x13))
    v4 = f2(x9, -4 * (x9 + x10 + x11 + x14))
    v5 = f2(x10, -4 * (x10 + x11 + x12 + x15))
    v6 = f2(x16, -4 * (x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4 * (v2 + v3))
    u2 = -f2(-v4, -4 * (v5 + v6))
    w1 = -2 * f2(-u1, -4 * u2)
    return sqrt(sqrt(Integer(2)) * sqrt(w1 + 4) / 8 + Rational(1, 2))


_COS_VALUES: dict[int, Expr] = {}


# PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:cos_table
def _cos_value(q: int) -> Expr:
    """``cos(pi/q)`` in square roots for the Fermat primes 3, 5, 17, 257."""
    if q not in _COS_VALUES:
        table: dict[int, Callable[[], Expr]] = {3: _cos_3, 5: _cos_5, 17: _cos_17, 257: _cos_257}
        _COS_VALUES[q] = table[q]()
    return _COS_VALUES[q]


# PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:migcdex
def _migcdex(xs: list[int]) -> tuple[list[int], int]:
    from sympy_extras.core.intfunc import igcdex
    if not xs:
        return [], 0
    if len(xs) == 1:
        return [1], xs[0]
    if len(xs) == 2:
        u, v, h2 = igcdex(xs[0], xs[1])
        return [u, v], h2
    y, g = _migcdex(xs[1:])
    u, v, h2 = igcdex(xs[0], g)
    return [u] + [v * i for i in y], h2


# PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:ipartfrac
def _ipartfrac(denoms: list[int]) -> list[int]:
    if not denoms:
        return []
    denom = 1
    for d in denoms:
        denom *= d
    return _migcdex([denom // d for d in denoms])[0]


# PORTED FROM SYMPY: sympy/functions/elementary/_trigonometric_special.py:fermat_coords
def _fermat_coords(n: int) -> Optional[list[int]]:
    primes: list[int] = []
    for p in [3, 5, 17, 257, 65537]:
        quotient, remainder = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos._eval_rewrite_as_sqrt
def _cos_sqrt(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.basic import Basic, SubsValue
    from sympy_extras.core.function import expand, expand_trig
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.special.polynomials import chebyshevt
    from sympy_extras.ntheory.factor_ import factorint
    c = t.pi_coeff(args[0])
    if c is None or isinstance(c, Integer) or not isinstance(c, Rational):
        return None
    if c.q in (3, 5, 17, 257):
        rv = chebyshevt(Integer(c.p), _cos_value(c.q))
        return expand(rv) if c.q < 257 else rv
    if c.q % 2 == 0:
        # the factors of 2 removed recursively
        pico2 = c * 2
        nval = _rw(t.cos(pico2 * pi), 'sqrt', hints)
        half = (pico2 + 1) / 2
        assert isinstance(half, Rational)
        sign_cos = -1 if (half.p // half.q) % 2 else 1
        return sign_cos * sqrt((1 + nval) / 2)
    fc = _fermat_coords(c.q)
    denoms = fc if fc else [b**k for b, k in factorint(c.q).items()]
    apart = _ipartfrac(denoms)
    zs = [Symbol('z%d' % i) for i in range(len(denoms))]
    pairs: list[tuple[Basic, SubsValue]] = [(z, Rational(c.p * n, d) * pi) for z, n, d in zip(zs, apart, denoms)]
    expanded = expand_trig(t.Cos(add(*zs)), deep=False).subs(pairs)
    assert isinstance(expanded, Expr)
    if not fc or len(fc) == 1:
        return expanded
    return _rw(expanded, 'sqrt', hints)


def _of_log(arg: Expr) -> Optional[Expr]:
    """``x`` when ``arg`` is ``log(x)`` (SymPy's ``isinstance(arg, log)``)."""
    from .exponential import Log
    if isinstance(arg, Log) and len(arg.args) == 1:
        a = arg.args[0]
        assert isinstance(a, Expr)
        return a
    return None


# PORTED FROM SYMPY: the _eval_rewrite_as_Pow methods of sin, cos, tan, cot
# (sympy/functions/elementary/trigonometric.py)
def _sin_pow_log(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    x = _of_log(args[0])
    return None if x is None else I * pow(x, -I) / 2 - I * pow(x, I) / 2


def _cos_pow_log(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    x = _of_log(args[0])
    return None if x is None else pow(x, I) / 2 + pow(x, -I) / 2


def _tan_pow_log(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    x = _of_log(args[0])
    return None if x is None else I * (pow(x, -I) - pow(x, I)) / (pow(x, -I) + pow(x, I))


def _cot_pow_log(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    x = _of_log(args[0])
    return None if x is None else -I * (pow(x, -I) + pow(x, I)) / (pow(x, -I) - pow(x, I))


# PORTED FROM SYMPY: the _eval_rewrite_as_pow and _eval_rewrite_as_sqrt methods of sin, tan, cot
# (sympy/functions/elementary/trigonometric.py)
def _through_cos(target: str, check: bool) -> Rule:
    """SymPy's ``self.rewrite(cos).rewrite(target)`` (``None`` when a
    cosine remains, with ``check``: the rules of tan and cot)."""
    def rule(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        y = _rw(_rw(node, t.Cos, hints), target, hints)
        if check and y.has(t.Cos):
            return None
        return y
    return rule


# PORTED FROM SYMPY: ReciprocalTrigonometricFunction._rewrite_reciprocal,
# ReciprocalHyperbolicFunction._rewrite_reciprocal
def _reciprocal_call(rule: Rule, of: Callable[[Expr], Expr], cls: type) -> Rule:
    """SymPy's ``_rewrite_reciprocal`` calling the rule of the reciprocal
    function on ``of(self.args[0])`` (``None`` when that node is not of
    the reciprocal class, where SymPy's method is not found)."""
    def rewritten(node: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a0 = node.args[0]
        assert isinstance(a0, Expr)
        o = of(a0)
        if not isinstance(o, cls):
            return None
        r = rule(o, args, hints)
        if r is not None and r != of(args[0]):
            return _inv(r)
        return None
    return rewritten


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py: the _eval_rewrite_as_tan,
# _eval_rewrite_as_sincos, _eval_rewrite_as_cot, _eval_rewrite_as_csc, _eval_rewrite_as_sec,
# _eval_rewrite_as_sinc, _eval_rewrite_as_besselj methods of sin, cos, tan, cot, sec, csc and the
# _eval_rewrite_as_jn, _eval_rewrite_as_sin methods of sinc
def _trig_rules() -> dict[RuleKey, Rule]:
    from sympy_extras.core.logic import and_, true
    from sympy_extras.core.mod import mod
    from sympy_extras.core.power import sqrt
    from sympy_extras.core.relational import eq, ne
    from sympy_extras.functions.special.bessel import BesselJ, SphericalJ
    from .complexes import im
    from .piecewise import piecewise
    half = Rational(1, 2)
    zero, one = Integer(0), Integer(1)

    def besselj(nu: Expr, z: Expr) -> Expr:
        from sympy_extras.functions.special.bessel import besselj as bj
        return bj(nu, z)

    def sin_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        th = t.tan(half * args[0])
        return 2 * th / (1 + th**2)

    def sin_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return t.sin(a) * t.cos(a) / t.cos(a)

    def sin_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        ch = t.cot(half * a)
        return piecewise((zero, and_(eq(im(a), zero), eq(mod(a, pi), zero))), (2 * ch / (1 + ch**2), true))

    def sin_csc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.csc(args[0])

    def sin_sec(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.Sec(args[0] - HALF_PI)

    def sin_sinc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return args[0] * t.sinc(args[0])

    def sin_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return sqrt(pi * a / 2) * besselj(half, a)

    def cos_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        th = t.tan(half * args[0])**2
        return (1 - th) / (1 + th)

    def cos_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return t.sin(a) * t.cos(a) / t.sin(a)

    def cos_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        ch = t.cot(half * a)**2
        return piecewise((one, and_(eq(im(a), zero), eq(mod(a, 2 * pi), zero))), ((ch - 1) / (ch + 1), true))

    def cos_sec(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.sec(args[0])

    def cos_csc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / _rw(t.sec(args[0]), t.Csc, hints)

    def cos_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return piecewise((sqrt(pi * a / 2) * besselj(-half, a), ne(a, zero)), (one, true))

    def cos_pow(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        # cos._eval_rewrite_as_pow is its _eval_rewrite_as_sqrt
        return _cos_sqrt(e, args, hints)

    def tan_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.sin(args[0]) / t.cos(args[0])

    def tan_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.cot(args[0])

    def sin_and_cos_in(target: type) -> Rule:
        # tan._eval_rewrite_as_sec, _eval_rewrite_as_csc and those of cot
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            a = args[0]
            s, c = _rw(t.sin(a), target, hints), _rw(t.cos(a), target, hints)
            return s / c if isinstance(e, t.Tan) else c / s
        return rule

    def tan_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return besselj(half, a) / besselj(-half, a)

    def cot_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.cos(args[0]) / t.sin(args[0])

    def cot_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return besselj(-half, a) / besselj(half, a)

    def sec_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        ch = t.cot(args[0] / 2)**2
        return (ch + 1) / (ch - 1)

    def sec_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return t.sin(a) / (t.cos(a) * t.sin(a))

    def sec_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / _rw(t.cos(args[0]), t.Tan, hints)

    def sec_csc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.Csc(HALF_PI - args[0])

    def sec_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return piecewise((1 / (sqrt(pi * a) / sqrt(Integer(2)) * besselj(-half, a)), ne(a, zero)), (one, true))

    def csc_sincos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return t.cos(a) / (t.sin(a) * t.cos(a))

    def csc_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        ch = t.cot(args[0] / 2)
        return (1 + ch**2) / (2 * ch)

    def csc_sec(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.Sec(HALF_PI - args[0])

    def csc_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / _rw(t.sin(args[0]), t.Tan, hints)

    def csc_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return sqrt(2 / pi) * (1 / (sqrt(a) * besselj(half, a)))

    def sinc_jn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        from sympy_extras.functions.special.bessel import jn
        return jn(zero, args[0])

    def sinc_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return piecewise((t.sin(a) / a, ne(a, zero)), (one, true))

    rec = _reciprocal_call
    return {
        (t.Sin, 'Pow'): _sin_pow_log, (t.Sin, t.Tan): sin_tan, (t.Sin, 'sincos'): sin_sincos,
        (t.Sin, t.Cot): sin_cot, (t.Sin, 'pow'): _through_cos('pow', False),
        (t.Sin, 'sqrt'): _through_cos('sqrt', False), (t.Sin, t.Csc): sin_csc, (t.Sin, t.Sec): sin_sec,
        (t.Sin, t.Sinc): sin_sinc, (t.Sin, BesselJ): sin_besselj,
        (t.Cos, 'Pow'): _cos_pow_log, (t.Cos, t.Tan): cos_tan, (t.Cos, 'sincos'): cos_sincos,
        (t.Cos, t.Cot): cos_cot, (t.Cos, 'pow'): cos_pow, (t.Cos, 'sqrt'): _cos_sqrt, (t.Cos, t.Sec): cos_sec,
        (t.Cos, t.Csc): cos_csc, (t.Cos, BesselJ): cos_besselj,
        (t.Tan, 'Pow'): _tan_pow_log, (t.Tan, 'sincos'): tan_sincos, (t.Tan, t.Cot): tan_cot,
        (t.Tan, t.Sec): sin_and_cos_in(t.Sec), (t.Tan, t.Csc): sin_and_cos_in(t.Csc),
        (t.Tan, 'pow'): _through_cos('pow', True), (t.Tan, 'sqrt'): _through_cos('sqrt', True),
        (t.Tan, BesselJ): tan_besselj,
        (t.Cot, 'Pow'): _cot_pow_log, (t.Cot, 'sincos'): cot_sincos, (t.Cot, t.Sec): sin_and_cos_in(t.Sec),
        (t.Cot, t.Csc): sin_and_cos_in(t.Csc), (t.Cot, 'pow'): _through_cos('pow', True),
        (t.Cot, 'sqrt'): _through_cos('sqrt', True), (t.Cot, BesselJ): cot_besselj,
        # ReciprocalTrigonometricFunction: Pow, pow and sqrt by the rules of the reciprocal function
        (t.Sec, 'Pow'): rec(_cos_pow_log, t.cos, t.Cos), (t.Sec, 'pow'): rec(cos_pow, t.cos, t.Cos),
        (t.Sec, 'sqrt'): rec(_cos_sqrt, t.cos, t.Cos),
        (t.Csc, 'Pow'): rec(_sin_pow_log, t.sin, t.Sin), (t.Csc, 'pow'): rec(_through_cos('pow', False), t.sin, t.Sin),
        (t.Csc, 'sqrt'): rec(_through_cos('sqrt', False), t.sin, t.Sin),
        (t.Sec, t.Cot): sec_cot, (t.Sec, 'sincos'): sec_sincos, (t.Sec, t.Tan): sec_tan, (t.Sec, t.Csc): sec_csc,
        (t.Sec, BesselJ): sec_besselj,
        (t.Csc, 'sincos'): csc_sincos, (t.Csc, t.Cot): csc_cot, (t.Csc, t.Sec): csc_sec, (t.Csc, t.Tan): csc_tan,
        (t.Csc, BesselJ): csc_besselj,
        (t.Sinc, SphericalJ): sinc_jn, (t.Sinc, t.Sin): sinc_sin}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py: the _eval_rewrite_as_<name> methods
# of sinh, cosh, tanh, coth, csch, sech between themselves and into csc, sec, tan, cot
def _hyperbolic_rules() -> dict[RuleKey, Rule]:
    half = Rational(1, 2)
    ipi2 = I * pi / 2

    def sinh_cosh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return -I * h.cosh(args[0] + ipi2)

    def sinh_tanh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        th = h.tanh(half * args[0])
        return 2 * th / (1 - th**2)

    def sinh_coth(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        ch = h.coth(half * args[0])
        return 2 * ch / (ch**2 - 1)

    def sinh_csch(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.csch(args[0])

    def sinh_csc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return -I / t.csc(I * args[0])

    def cosh_sec(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / t.Sec(I * args[0])

    def cosh_sinh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return -I * h.Sinh(args[0] + ipi2)

    def cosh_tanh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        th = h.tanh(half * args[0])**2
        return (1 + th) / (1 - th)

    def cosh_coth(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        ch = h.coth(half * args[0])**2
        return (ch + 1) / (ch - 1)

    def cosh_sech(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.sech(args[0])

    def tanh_tan(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return -I * t.Tan(I * args[0])

    def tanh_cot(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return -I / t.Cot(I * args[0])

    def tanh_sinh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return I * h.sinh(a) / h.Sinh(ipi2 - a)

    def tanh_cosh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return I * h.Cosh(ipi2 - a) / h.cosh(a)

    def tanh_coth(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.coth(args[0])

    def coth_sinh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return -I * h.Sinh(ipi2 - a) / h.sinh(a)

    def coth_cosh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        return -I * h.cosh(a) / h.Cosh(ipi2 - a)

    def coth_tanh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.tanh(args[0])

    def csch_csc(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return I * t.Csc(I * args[0])

    def csch_cosh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return I / h.Cosh(args[0] + ipi2)

    def csch_sinh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.sinh(args[0])

    def sech_sec(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.Sec(I * args[0])

    def sech_sinh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return I / h.Sinh(args[0] + ipi2)

    def sech_cosh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return 1 / h.cosh(args[0])

    rec = _reciprocal_call
    return {
        (h.Sinh, h.Cosh): sinh_cosh, (h.Sinh, h.Tanh): sinh_tanh, (h.Sinh, h.Coth): sinh_coth,
        (h.Sinh, h.Csch): sinh_csch, (h.Sinh, t.Csc): sinh_csc,
        (h.Cosh, t.Sec): cosh_sec, (h.Cosh, h.Sinh): cosh_sinh, (h.Cosh, h.Tanh): cosh_tanh,
        (h.Cosh, h.Coth): cosh_coth, (h.Cosh, h.Sech): cosh_sech,
        (h.Tanh, t.Tan): tanh_tan, (h.Tanh, t.Cot): tanh_cot, (h.Tanh, h.Sinh): tanh_sinh,
        (h.Tanh, h.Cosh): tanh_cosh, (h.Tanh, h.Coth): tanh_coth,
        (h.Coth, h.Sinh): coth_sinh, (h.Coth, h.Cosh): coth_cosh, (h.Coth, h.Tanh): coth_tanh,
        # ReciprocalHyperbolicFunction: tanh and coth by the rules of the reciprocal function
        (h.Csch, h.Tanh): rec(sinh_tanh, h.sinh, h.Sinh), (h.Csch, h.Coth): rec(sinh_coth, h.sinh, h.Sinh),
        (h.Sech, h.Tanh): rec(cosh_tanh, h.cosh, h.Cosh), (h.Sech, h.Coth): rec(cosh_coth, h.cosh, h.Cosh),
        (h.Csch, t.Csc): csch_csc, (h.Csch, h.Cosh): csch_cosh, (h.Csch, h.Sinh): csch_sinh,
        (h.Sech, t.Sec): sech_sec, (h.Sech, h.Sinh): sech_sinh, (h.Sech, h.Cosh): sech_cosh}


def _of(f: Callable[[Expr], Expr]) -> Rule:
    def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return f(args[0])
    return rule


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py: the _eval_rewrite_as_asin,
# _eval_rewrite_as_acos, _eval_rewrite_as_atan, _eval_rewrite_as_acot, _eval_rewrite_as_asec and
# _eval_rewrite_as_acsc methods of asin, acos, atan, acot, asec, acsc
def _inverse_trig_rules() -> dict[RuleKey, Rule]:
    from sympy_extras.core.power import sqrt
    asin, acos, atan, acot, asec, acsc = t.asin, t.acos, t.atan, t.acot, t.asec, t.acsc
    p2 = HALF_PI

    def sx2x(x: Expr) -> Expr:
        return sqrt(x**2) / x

    def xs1x2(x: Expr) -> Expr:
        return x * sqrt(1 / x**2)

    return {
        (t.Asin, t.Acos): _of(lambda x: p2 - acos(x)),
        (t.Asin, t.Atan): _of(lambda x: 2 * atan(x / (1 + sqrt(1 - x**2)))),
        (t.Asin, t.Acot): _of(lambda x: 2 * acot((1 + sqrt(1 - x**2)) / x)),
        (t.Asin, t.Asec): _of(lambda x: p2 - asec(1 / x)),
        (t.Asin, t.Acsc): _of(lambda x: acsc(1 / x)),
        (t.Acos, t.Asin): _of(lambda x: p2 - asin(x)),
        (t.Acos, t.Atan): _of(lambda x: atan(sqrt(1 - x**2) / x) + p2 * (1 - x * sqrt(1 / x**2))),
        (t.Acos, t.Acot): _of(lambda x: p2 - 2 * acot((1 + sqrt(1 - x**2)) / x)),
        (t.Acos, t.Asec): _of(lambda x: asec(1 / x)),
        (t.Acos, t.Acsc): _of(lambda x: p2 - acsc(1 / x)),
        (t.Atan, t.Asin): _of(lambda x: sx2x(x) * (p2 - asin(1 / sqrt(1 + x**2)))),
        (t.Atan, t.Acos): _of(lambda x: sx2x(x) * acos(1 / sqrt(1 + x**2))),
        (t.Atan, t.Acot): _of(lambda x: acot(1 / x)),
        (t.Atan, t.Asec): _of(lambda x: sx2x(x) * asec(sqrt(1 + x**2))),
        (t.Atan, t.Acsc): _of(lambda x: sx2x(x) * (p2 - acsc(sqrt(1 + x**2)))),
        (t.Acot, t.Asin): _of(lambda x: xs1x2(x) * (p2 - asin(sqrt(-x**2) / sqrt(-x**2 - 1)))),
        (t.Acot, t.Acos): _of(lambda x: xs1x2(x) * acos(sqrt(-x**2) / sqrt(-x**2 - 1))),
        (t.Acot, t.Atan): _of(lambda x: atan(1 / x)),
        (t.Acot, t.Asec): _of(lambda x: xs1x2(x) * asec(sqrt((1 + x**2) / x**2))),
        (t.Acot, t.Acsc): _of(lambda x: xs1x2(x) * (p2 - acsc(sqrt((1 + x**2) / x**2)))),
        (t.Asec, t.Asin): _of(lambda x: p2 - asin(1 / x)),
        (t.Asec, t.Acos): _of(lambda x: acos(1 / x)),
        (t.Asec, t.Atan): _of(lambda x: p2 * (1 - sx2x(x)) + sx2x(x) * atan(sqrt(x**2 - 1))),
        (t.Asec, t.Acot): _of(lambda x: p2 * (1 - sx2x(x)) + sx2x(x) * acot(1 / sqrt(x**2 - 1))),
        (t.Asec, t.Acsc): _of(lambda x: p2 - acsc(x)),
        (t.Acsc, t.Asin): _of(lambda x: asin(1 / x)),
        (t.Acsc, t.Acos): _of(lambda x: p2 - acos(1 / x)),
        (t.Acsc, t.Atan): _of(lambda x: sx2x(x) * (p2 - atan(sqrt(x**2 - 1)))),
        (t.Acsc, t.Acot): _of(lambda x: sx2x(x) * (p2 - acot(1 / sqrt(x**2 - 1)))),
        (t.Acsc, t.Asec): _of(lambda x: p2 - asec(x))}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py: the _eval_rewrite_as_<name> methods
# of asinh, acosh, atanh, acoth, asech, acsch other than log and tractable
def _inverse_hyperbolic_rules() -> dict[RuleKey, Rule]:
    from sympy_extras.core.power import sqrt
    p2 = HALF_PI

    def asinh_acosh(x: Expr) -> Expr:
        ix = I * x
        return I * (sqrt(1 - ix) / sqrt(ix - 1) * h.acosh(ix) - p2)

    def acosh_factor(x: Expr) -> Expr:
        return sqrt(x - 1) / sqrt(1 - x)

    def acosh_atanh(x: Expr) -> Expr:
        sxm1, s1mx, sx2m1 = sqrt(x - 1), sqrt(1 - x), sqrt(x**2 - 1)
        return p2 * sxm1 / s1mx * (1 - x * sqrt(1 / x**2)) + sxm1 * sqrt(x + 1) / sx2m1 * h.atanh(sx2m1 / x)

    def atanh_asinh(x: Expr) -> Expr:
        f = sqrt(1 / (x**2 - 1))
        return pi * x / (2 * sqrt(-x**2)) - sqrt(-x) * sqrt(1 - x**2) / sqrt(x) * f * h.asinh(f)

    def acoth_asinh(x: Expr) -> Expr:
        return (pi * I / 2 * (sqrt((x - 1) / x) * sqrt(x / (x - 1)) - sqrt(1 + 1 / x) * sqrt(x / (x + 1)))
                + x * sqrt(1 / x**2) * h.asinh(sqrt(1 / (x**2 - 1))))

    def asech_factor(x: Expr) -> Expr:
        return sqrt(1 / x - 1) / sqrt(1 - 1 / x)

    def asech_atanh(x: Expr) -> Expr:
        return (I * pi * (1 - sqrt(x) * sqrt(1 / x) - I / 2 * sqrt(-x) / sqrt(x) - I / 2 * sqrt(x**2) / sqrt(-x**2))
                + sqrt(1 / (x + 1)) * sqrt(x + 1) * h.atanh(sqrt(1 - x**2)))

    def acsch_atanh(x: Expr) -> Expr:
        x2p1 = x**2 + 1
        return sqrt(-x**2) / x * (p2 - sqrt(-x2p1**2) / x2p1 * h.atanh(sqrt(x2p1)))

    return {
        (h.Asinh, h.Atanh): _of(lambda x: h.atanh(x / sqrt(1 + x**2))),
        (h.Asinh, h.Acosh): _of(asinh_acosh),
        (h.Asinh, t.Asin): _of(lambda x: -I * t.Asin(I * x)),
        (h.Asinh, t.Acos): _of(lambda x: I * t.Acos(I * x) - I * pi / 2),
        (h.Acosh, t.Acos): _of(lambda x: acosh_factor(x) * t.acos(x)),
        (h.Acosh, t.Asin): _of(lambda x: acosh_factor(x) * (p2 - t.asin(x))),
        (h.Acosh, h.Asinh): _of(lambda x: acosh_factor(x) * (p2 + I * h.Asinh(I * x))),
        (h.Acosh, h.Atanh): _of(acosh_atanh),
        (h.Atanh, h.Asinh): _of(atanh_asinh),
        (h.Acoth, h.Atanh): _of(lambda x: h.atanh(1 / x)),
        (h.Acoth, h.Asinh): _of(acoth_asinh),
        (h.Asech, h.Acosh): _of(lambda x: h.acosh(1 / x)),
        (h.Asech, h.Asinh): _of(lambda x: asech_factor(x) * (I * h.Asinh(I / x) + p2)),
        (h.Asech, h.Atanh): _of(asech_atanh),
        (h.Asech, h.Acsch): _of(lambda x: asech_factor(x) * (p2 - I * h.Acsch(I * x))),
        (h.Acsch, h.Asinh): _of(lambda x: h.asinh(1 / x)),
        (h.Acsch, h.Acosh): _of(lambda x: I * (sqrt(1 - I / x) / sqrt(I / x - 1) * h.Acosh(I / x) - p2)),
        (h.Acsch, h.Atanh): _of(acsch_atanh)}


def _exp_sqrt_of(e: Expr) -> Optional[Expr]:
    """SymPy's ``exp._eval_rewrite_as_sqrt`` on the exponent: ``exp(I*pi*c)``
    as ``cos(pi*c) + I*sin(pi*c)`` when both evaluate."""
    from sympy_extras.core.mul import Mul
    if not isinstance(e, Mul) or pi not in e.args or I not in e.args:
        return None
    # SymPy's e.coeff(pi*I) (the core's coeff does not find the product
    # pi*I among the factors)
    factors = list(e.args)
    factors.remove(pi)
    factors.remove(I)
    coeff = mul(*[f for f in factors if isinstance(f, Expr)])
    if coeff != 0 and coeff.is_number:
        cosine, sine = t.cos(pi * coeff), t.sin(pi * coeff)
        if not isinstance(cosine, t.Cos) and not isinstance(sine, t.Sin):
            return cosine + I * sine
    return None


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:exp._eval_rewrite_as_tanh,
# _eval_rewrite_as_sqrt, _eval_rewrite_as_Pow; sympy/core/power.py:Pow._eval_rewrite_as_tanh,
# Pow._eval_rewrite_as_sqrt; sympy/core/numbers.py:Exp1._eval_rewrite_as_sin, _eval_rewrite_as_cos
def _exp_rules() -> dict[RuleKey, Rule]:
    from .exponential import Log
    from sympy_extras.core.numbers import Exp1

    def exp_tanh_of(a: Expr) -> Expr:
        return (1 + h.tanh(a / 2)) / (1 - h.tanh(a / 2))

    def exp_tanh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return exp_tanh_of(args[0])

    def exp_sqrt(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return _exp_sqrt_of(args[0])

    def exp_pow(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = args[0]
        if a.is_Mul:
            logs = [f for f in a.args if isinstance(f, Log) and len(f.args) == 1]
            if logs:
                base = logs[0].args[0]
                assert isinstance(base, Expr)
                return pow(base, a.coeff(logs[0]))
        return None

    def pow_tanh(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        if isinstance(e, Pow) and e.base == E:
            return exp_tanh_of(e.exp)
        return None

    def pow_sqrt(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        if args[0] != E:
            return None
        return _exp_sqrt_of(args[1])

    def e_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.sin(I + HALF_PI) - I * t.sin(I)

    def e_cos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return t.cos(I) + I * t.cos(I + HALF_PI)

    return {(Exp, h.Tanh): exp_tanh, (Exp, 'sqrt'): exp_sqrt, (Exp, 'Pow'): exp_pow,
            (Pow, h.Tanh): pow_tanh, (Pow, 'sqrt'): pow_sqrt, (Exp1, t.Sin): e_sin, (Exp1, t.Cos): e_cos}


__all__ = ['rules', 'Rule']
