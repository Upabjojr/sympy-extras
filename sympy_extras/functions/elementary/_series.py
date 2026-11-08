"""The series of the trigonometric and hyperbolic functions.

The Taylor terms (``taylor_term``), the series (``_eval_nseries``) and the
leading terms (``_eval_as_leading_term``) of ``sin``, ``cos``, ``tan``,
``cot``, ``sec``, ``csc`` and of their hyperbolic counterparts, as the
methods of the classes call them. A leading term which SymPy gives as an
``AccumBounds`` (``sin(1/x)``) is SymPy's, through the conversion, until
accumulation bounds are ported.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py and
# sympy/functions/elementary/hyperbolic.py (the series methods) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; module functions shared by the classes; the previous
# terms of taylor_term are not used (SymPy's series pass only one).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.coreerrors import PoleError
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Application
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, nan, pi, zoo
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol

__all__ = ['sin_taylor_term', 'cos_taylor_term', 'tan_taylor_term', 'cot_taylor_term', 'sec_taylor_term',
           'csc_taylor_term', 'sinh_taylor_term', 'cosh_taylor_term', 'tanh_taylor_term', 'coth_taylor_term',
           'sech_taylor_term', 'csch_taylor_term']


def _factorial(n: int) -> Expr:
    from sympy_extras.functions.combinatorial.factorials import factorial
    return factorial(Integer(n))


def _bernoulli(n: int) -> Expr:
    from sympy_extras.functions.combinatorial.numbers import bernoulli
    return bernoulli(Integer(n))


def _euler(n: int) -> Expr:
    from sympy_extras.functions.combinatorial.numbers import euler
    return euler(Integer(n))


def _div(a: Expr, b: Expr) -> Expr:
    return mul(a, pow(b, Integer(-1)))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin.taylor_term
def sin_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return _div(mul(Integer((-1) ** (n // 2)), pow(x, Integer(n))), _factorial(n))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cos.taylor_term
def cos_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 1:
        return Integer(0)
    return _div(mul(Integer((-1) ** (n // 2)), pow(x, Integer(n))), _factorial(n))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:tan.taylor_term
def tan_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    a, b = (n - 1) // 2, 2 ** (n + 1)
    return mul(Integer((-1) ** a * b * (b - 1)), _div(_bernoulli(n + 1), _factorial(n + 1)), pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cot.taylor_term
def cot_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return pow(x, Integer(-1))
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return mul(Integer((-1) ** ((n + 1) // 2) * 2 ** (n + 1)), _div(_bernoulli(n + 1), _factorial(n + 1)),
               pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sec.taylor_term
def sec_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 1:
        return Integer(0)
    k = n // 2
    return mul(Integer((-1) ** k), _div(_euler(2 * k), _factorial(2 * k)), pow(x, Integer(2 * k)))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:csc.taylor_term
def csc_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return pow(x, Integer(-1))
    if n < 0 or n % 2 == 0:
        return Integer(0)
    k = n // 2 + 1
    return _div(mul(Integer((-1) ** (k - 1) * 2 * (2 ** (2 * k - 1) - 1)), _bernoulli(2 * k),
                    pow(x, Integer(2 * k - 1))), _factorial(2 * k))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sinh.taylor_term
def sinh_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return _div(pow(x, Integer(n)), _factorial(n))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:cosh.taylor_term
def cosh_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 1:
        return Integer(0)
    return _div(pow(x, Integer(n)), _factorial(n))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh.taylor_term
def tanh_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    a = 2 ** (n + 1)
    return mul(Integer(a * (a - 1)), _div(_bernoulli(n + 1), _factorial(n + 1)), pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:coth.taylor_term
def coth_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return pow(x, Integer(-1))
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return mul(Integer(2 ** (n + 1)), _div(_bernoulli(n + 1), _factorial(n + 1)), pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sech.taylor_term
def sech_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 1:
        return Integer(0)
    return mul(_div(_euler(n), _factorial(n)), pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:csch.taylor_term
def csch_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return pow(x, Integer(-1))
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return mul(Integer(2 * (1 - 2 ** n)), _div(_bernoulli(n + 1), _factorial(n + 1)), pow(x, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin._eval_nseries (and cos)
def sincos_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import log
    arg = node.args[0]
    if logx is not None:
        arg = arg.subs(log(x), logx)
    if arg.subs(x, Integer(0)).has(nan, zoo):
        raise PoleError("Cannot expand %s around 0" % (node,))
    return Application._eval_nseries(node, x, n, logx, cdir)


def _limit_dir(cdir: Expr) -> str:
    from sympy_extras.functions.elementary.complexes import re
    return '-' if re(cdir).is_negative is True else '+'


def _accum_bounds(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    """The leading term which SymPy gives as accumulation bounds (not
    ported): SymPy's."""
    from sympy_extras.core.interop import sympy_leading_term
    return sympy_leading_term(node, x, logx, cdir)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:{sin,cos,tan,cot,sec,csc}._eval_as_leading_term
def trig_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr,
                      multiple: Callable[[Expr], Expr],
                      at_multiple: Callable[[Expr, Expr], Expr],
                      shift: Callable[[Expr], Expr]) -> Expr:
    """The leading term of a trigonometric function: ``multiple(x0)`` is
    the multiple of the period at the value ``x0`` of the argument (an
    integer there), ``at_multiple(n, lt)`` the leading term from the one
    ``lt`` of ``arg - shift(n)``."""
    arg = node.args[0]
    x0 = arg.subs(x, Integer(0)).cancel()
    k = multiple(x0)
    if k.is_integer is True:
        lt = add(arg, mul(Integer(-1), shift(k))).as_leading_term(x)
        return at_multiple(k, lt)
    if isinstance(x0, ComplexInfinity):
        x0 = arg.limit(x, Integer(0), dir=_limit_dir(cdir))
    if isinstance(x0, (Infinity, NegativeInfinity)):
        return _accum_bounds(node, x, logx, cdir)
    return node._rebuilt([x0]) if x0.is_finite is True else node


def _half_pi() -> Expr:
    return mul(pi, pow(Integer(2), Integer(-1)))


def sin_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(x0, pi),
                             lambda k, lt: mul(pow(Integer(-1), k), lt), lambda k: mul(k, pi))


def cos_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(add(x0, _half_pi()), pi),
                             lambda k, lt: mul(pow(Integer(-1), k), lt),
                             lambda k: add(mul(k, pi), mul(Integer(-1), _half_pi())))


def tan_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(mul(Integer(2), x0), pi),
                             lambda k, lt: lt if k.is_even is True else mul(Integer(-1), pow(lt, Integer(-1))),
                             lambda k: mul(k, _half_pi()))


def cot_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(mul(Integer(2), x0), pi),
                             lambda k, lt: pow(lt, Integer(-1)) if k.is_even is True else mul(Integer(-1), lt),
                             lambda k: mul(k, _half_pi()))


def sec_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(add(x0, _half_pi()), pi),
                             lambda k, lt: _div(pow(Integer(-1), k), lt),
                             lambda k: add(mul(k, pi), mul(Integer(-1), _half_pi())))


def csc_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return trig_leading_term(node, x, logx, cdir, lambda x0: _div(x0, pi),
                             lambda k, lt: _div(pow(Integer(-1), k), lt), lambda k: mul(k, pi))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:tan._eval_nseries
def tan_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import Cos
    i = _div(mul(node.args[0].limit(x, Integer(0)), Integer(2)), pi)
    if not (i.is_Number and i == 0) and isinstance(i, Integer):
        return node.rewrite(Cos)._eval_nseries(x, n, logx, Integer(0))
    return Application._eval_nseries(node, x, n, logx, Integer(0))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:cot._eval_nseries
def cot_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import Cos, Tan
    i = _div(node.args[0].limit(x, Integer(0)), pi)
    if not (i.is_Number and i == 0) and isinstance(i, Integer):
        return node.rewrite(Cos)._eval_nseries(x, n, logx, Integer(0))
    return node.rewrite(Tan)._eval_nseries(x, n, logx, Integer(0))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:ReciprocalTrigonometricFunction._eval_nseries
def reciprocal_nseries(reciprocal: Expr, x: Symbol, n: int, logx: Optional[Expr]) -> Expr:
    return pow(reciprocal, Integer(-1))._eval_nseries(x, n, logx, Integer(0))


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:ReciprocalHyperbolicFunction._eval_as_leading_term
def reciprocal_leading_term(reciprocal: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return pow(reciprocal, Integer(-1))._eval_as_leading_term(x, logx, cdir)


def _hyperbolic_arg0(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> tuple[Expr, Expr]:
    arg = node.args[0].as_leading_term(x, logx=logx, cdir=cdir)
    arg0 = arg.subs(x, Integer(0))
    if isinstance(arg0, NaN):
        arg0 = arg.limit(x, Integer(0), dir='-' if cdir.is_negative is True else '+')
    return arg, arg0


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sinh._eval_as_leading_term
def sinh_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    arg, arg0 = _hyperbolic_arg0(node, x, logx, cdir)
    if arg0.is_zero is True:
        return arg
    if arg0.is_finite is True:
        return node._rebuilt([arg0])
    return node


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:cosh._eval_as_leading_term
def cosh_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    _, arg0 = _hyperbolic_arg0(node, x, logx, cdir)
    if arg0.is_zero is True:
        return Integer(1)
    if arg0.is_finite is True:
        return node._rebuilt([arg0])
    return node


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh._eval_as_leading_term (and coth)
def tanh_coth_leading_term(node: Application, x: Symbol, inverted: bool) -> Expr:
    from sympy_extras.core.order import Order, order
    arg = node.args[0].as_leading_term(x)
    o = order(Integer(1), x)
    if x in arg.free_symbols and isinstance(o, Order) and o.contains(arg):
        return pow(arg, Integer(-1)) if inverted else arg
    return node._rebuilt([arg])


# ---------------------------------------------------------------------
# the inverse trigonometric functions

def _sqrt(e: Expr) -> Expr:
    from sympy_extras.core.numbers import Rational
    return pow(e, Rational(1, 2))


def _rewrite_log(node: Expr) -> Expr:
    from sympy_extras.functions.elementary.exponential import Log
    return node.rewrite(Log)


def _ndir(arg: Expr, x: Symbol, cdir: Expr) -> Expr:
    return arg.dir(x, Integer(1) if cdir.is_Number and cdir == 0 else cdir)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:asin.taylor_term (and acos)
def asin_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    from sympy_extras.core.numbers import Rational
    from sympy_extras.functions.combinatorial.factorials import rf
    k = (n - 1) // 2
    return _div(mul(rf(Rational(1, 2), Integer(k)), pow(x, Integer(n))), mul(_factorial(k), Integer(n)))


def acos_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return _half_pi()
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return mul(Integer(-1), asin_taylor_term(n, x))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan.taylor_term
def atan_taylor_term(n: int, x: Expr) -> Expr:
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return _div(mul(Integer((-1) ** ((n - 1) // 2)), pow(x, Integer(n))), Integer(n))


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:acot.taylor_term
def acot_taylor_term(n: int, x: Expr) -> Expr:
    if n == 0:
        return _half_pi()
    if n < 0 or n % 2 == 0:
        return Integer(0)
    return _div(mul(Integer((-1) ** ((n + 1) // 2)), pow(x, Integer(n))), Integer(n))


def _is_number(e: Expr, value: int) -> bool:
    return e.is_Number and e == value


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:asin._eval_as_leading_term (and acos)
def asin_acos_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr, is_acos: bool) -> Expr:
    from sympy_extras.functions.elementary.complexes import im
    arg = node.args[0]
    x0 = arg.subs(x, Integer(0)).cancel()
    if isinstance(x0, NaN):
        return node._rebuilt([arg.as_leading_term(x)])
    if not is_acos and x0.is_zero is True:
        return arg.as_leading_term(x)
    # the branch points
    if is_acos and _is_number(x0, 1):
        return mul(_sqrt(Integer(2)), _sqrt(add(Integer(1), mul(Integer(-1), arg)).as_leading_term(x)))
    branch_points = (Integer(-1), zoo) if is_acos else (Integer(-1), Integer(1), zoo)
    if x0 in branch_points:
        lt = _rewrite_log(node)._eval_as_leading_term(x, logx, cdir)
        return lt if is_acos else lt.expand()
    # the points on the branch cuts (-oo, -1) U (1, oo)
    if add(Integer(1), mul(Integer(-1), pow(x0, Integer(2)))).is_negative is True:
        ndir = _ndir(arg, x, cdir)
        fx0 = node._rebuilt([x0])
        if im(ndir).is_negative is True:
            if x0.is_negative is True:
                return add(mul(Integer(2), pi), mul(Integer(-1), fx0)) if is_acos \
                    else add(mul(Integer(-1), pi), mul(Integer(-1), fx0))
        elif im(ndir).is_positive is True:
            if x0.is_positive is True:
                return mul(Integer(-1), fx0) if is_acos else add(pi, mul(Integer(-1), fx0))
        else:
            return _rewrite_log(node)._eval_as_leading_term(x, logx, cdir).expand()
    return node._rebuilt([x0])


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:asin._eval_nseries (and acos)
def asin_acos_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr, is_acos: bool) -> Expr:
    from sympy_extras.core.order import order
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions.elementary.complexes import im
    arg0 = node.args[0].subs(x, Integer(0))
    for point in (1, -1):
        if not _is_number(arg0, point):
            continue
        # the branch points
        t = Dummy('t', positive=True)
        at = add(Integer(point), mul(Integer(-point), pow(t, Integer(2))))
        ser = _rewrite_log(node._rebuilt([at])).nseries(t, 0, 2 * n)
        arg1 = add(Integer(1), mul(Integer(-point), node.args[0]))
        f = arg1.as_leading_term(x)
        g = _div(add(arg1, mul(Integer(-1), f)), f)
        if not g.is_meromorphic(x, 0):
            # cannot be expanded
            if n == 0:
                return order(Integer(1))
            if is_acos:
                value: Expr = Integer(0) if point == 1 else pi
            else:
                value = mul(Integer(point), _half_pi())
            return add(value, order(_sqrt(x)))
        res1 = _sqrt(add(Integer(1), g))._eval_nseries(x, n, logx, Integer(0))
        res = mul(res1.removeO(), _sqrt(f)).expand()
        return add(ser.removeO().subs(t, res).expand().powsimp(), order(pow(x, Integer(n)), x))
    res = Application._eval_nseries(node, x, n, logx, Integer(0))
    if isinstance(arg0, ComplexInfinity):
        return res
    # the points on the branch cuts (-oo, -1) U (1, oo)
    if add(Integer(1), mul(Integer(-1), pow(arg0, Integer(2)))).is_negative is True:
        ndir = _ndir(node.args[0], x, cdir)
        if im(ndir).is_negative is True:
            if arg0.is_negative is True:
                return add(mul(Integer(2), pi), mul(Integer(-1), res)) if is_acos \
                    else add(mul(Integer(-1), pi), mul(Integer(-1), res))
        elif im(ndir).is_positive is True:
            if arg0.is_positive is True:
                return mul(Integer(-1), res) if is_acos else add(pi, mul(Integer(-1), res))
        else:
            return _rewrite_log(node)._eval_nseries(x, n, logx, cdir)
    return res


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan._eval_as_leading_term
def atan_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.numbers import I
    from sympy_extras.functions.elementary.complexes import im, re
    arg = node.args[0]
    x0 = arg.subs(x, Integer(0)).cancel()
    if isinstance(x0, NaN):
        return node._rebuilt([arg.as_leading_term(x)])
    if x0.is_zero is True:
        return arg.as_leading_term(x)
    # the branch points
    if x0 in (mul(Integer(-1), I), I, zoo):
        return _rewrite_log(node)._eval_as_leading_term(x, logx, cdir).expand()
    # the points on the branch cuts (-I*oo, -I) U (I, I*oo)
    if add(Integer(1), pow(x0, Integer(2))).is_negative is True:
        ndir = _ndir(arg, x, cdir)
        fx0 = node._rebuilt([x0])
        if re(ndir).is_negative is True:
            if im(x0).is_positive is True:
                return add(fx0, mul(Integer(-1), pi))
        elif re(ndir).is_positive is True:
            if im(x0).is_negative is True:
                return add(fx0, pi)
        else:
            return _rewrite_log(node)._eval_as_leading_term(x, logx, cdir).expand()
    return node._rebuilt([x0])


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan._eval_nseries
def atan_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.numbers import I
    from sympy_extras.core.series import _less
    from sympy_extras.functions.elementary.complexes import im, re
    arg0 = node.args[0].subs(x, Integer(0))
    # the branch points
    if arg0 in (I, mul(Integer(-1), I)):
        return _rewrite_log(node)._eval_nseries(x, n, logx, cdir)
    res = Application._eval_nseries(node, x, n, logx, Integer(0))
    ndir = _ndir(node.args[0], x, cdir)
    if isinstance(arg0, ComplexInfinity):
        if _less(Integer(0), re(ndir)):
            return add(res, mul(Integer(-1), pi))
        return res
    # the points on the branch cuts (-I*oo, -I) U (I, I*oo)
    if add(Integer(1), pow(arg0, Integer(2))).is_negative is True:
        if re(ndir).is_negative is True:
            if im(arg0).is_positive is True:
                return add(res, mul(Integer(-1), pi))
        elif re(ndir).is_positive is True:
            if im(arg0).is_negative is True:
                return add(res, pi)
        else:
            return _rewrite_log(node)._eval_nseries(x, n, logx, cdir)
    return res


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:acot._eval_as_leading_term
def acot_leading_term(node: Application, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.numbers import I
    from sympy_extras.functions.elementary.complexes import im, re
    arg = node.args[0]
    x0 = arg.subs(x, Integer(0)).cancel()
    if isinstance(x0, NaN):
        return node._rebuilt([arg.as_leading_term(x)])
    if isinstance(x0, ComplexInfinity):
        return pow(arg, Integer(-1)).as_leading_term(x)
    # the branch points
    if x0 in (mul(Integer(-1), I), I, Integer(0)):
        return _rewrite_log(node)._eval_as_leading_term(x, logx, cdir).expand()
    # the points on the branch cut [-I, I]
    if x0.is_imaginary is True and add(Integer(1), pow(x0, Integer(2))).is_positive is True:
        ndir = _ndir(arg, x, cdir)
        fx0 = node._rebuilt([x0])
        if re(ndir).is_positive is True:
            if im(x0).is_positive is True:
                return add(fx0, pi)
        elif re(ndir).is_negative is True:
            if im(x0).is_negative is True:
                return add(fx0, mul(Integer(-1), pi))
        else:
            return _rewrite_log(node)._eval_as_leading_term(x, logx, cdir).expand()
    return node._rebuilt([x0])


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:acot._eval_nseries
def acot_nseries(node: Application, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.numbers import I
    from sympy_extras.core.series import _less
    from sympy_extras.functions.elementary.complexes import im, re
    arg0 = node.args[0].subs(x, Integer(0))
    # the branch points
    if arg0 in (I, mul(Integer(-1), I)):
        return _rewrite_log(node)._eval_nseries(x, n, logx, cdir)
    res = Application._eval_nseries(node, x, n, logx, Integer(0))
    if isinstance(arg0, ComplexInfinity):
        return res
    ndir = _ndir(node.args[0], x, cdir)
    if arg0.is_zero is True:
        if _less(re(ndir), Integer(0)):
            return add(res, mul(Integer(-1), pi))
        return res
    # the points on the branch cut [-I, I]
    if arg0.is_imaginary is True and add(Integer(1), pow(arg0, Integer(2))).is_positive is True:
        if re(ndir).is_positive is True:
            if im(arg0).is_positive is True:
                return add(res, pi)
        elif re(ndir).is_negative is True:
            if im(arg0).is_negative is True:
                return add(res, mul(Integer(-1), pi))
        else:
            return _rewrite_log(node)._eval_nseries(x, n, logx, cdir)
    return res


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:atan._eval_aseries (and acot)
def atan_acot_aseries(node: Application, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr],
                      is_acot: bool) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import atan
    if args0[0] in (Infinity(), NegativeInfinity()):
        inverse = atan(pow(node.args[0], Integer(-1)))
        e = inverse if is_acot else add(_half_pi(), mul(Integer(-1), inverse))
        return e._eval_nseries(x, n, logx, Integer(0))
    return Application._eval_aseries(node, n, args0, x, logx)
