"""The series expansions of the special functions with an ``_eval_nseries``
of their own in SymPy, as a table by class name for
:meth:`~sympy_extras.core.function.DefinedFunction._eval_nseries`.

A rule returns ``None`` where SymPy calls the generic expansion of
functions and that one is not available here (an argument going to an
infinity): the caller then falls back.

Two rules differ from SymPy on purpose (SymPy bugs, see issue #25):
``sinc`` expands in ``x`` (SymPy expands in the argument, so
``sinc(2*x)`` does not expand), and ``SingularityFunction`` expands to the
order asked (SymPy's order is the exponent of the function).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: the _eval_nseries methods of LambertW (exponential.py),
# Abs, sign, arg (complexes.py), sinc (trigonometric.py), gamma, loggamma
# (gamma_functions.py), expint, Ei, TrigonometricIntegral, li, Li
# (error_functions.py), SingularityFunction (singularity_functions.py),
# elliptic_k, elliptic_e (elliptic_integrals.py) of sympy/functions
# (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table of functions by class name; the rewrites are
# written out; the sinc and SingularityFunction fixes above.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import EulerGamma, I, Integer, Rational, pi
from sympy_extras.core.symbol import Symbol

__all__ = ['rules']

Rule = Callable[[Expr, Symbol, int, Optional[Expr], Expr], Optional[Expr]]
ZERO = Integer(0)


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _arg(f: Expr, i: int = 0) -> Expr:
    return _e(f.args[i])


def _generic(f: Expr, x: Symbol, n: int, logx: Optional[Expr]) -> Optional[Expr]:
    """SymPy's ``super()._eval_nseries``: the generic expansion of functions."""
    from sympy_extras.core.function import DefinedFunction, _generic_function_nseries
    assert isinstance(f, DefinedFunction)
    return _generic_function_nseries(f, x, n, logx)


def _ns(e: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr = ZERO) -> Expr:
    return e._eval_nseries(x, n, logx, cdir)


def _limit0(e: Expr, x: Symbol) -> Expr:
    from sympy_extras.core.limits import limit
    return limit(e, x, ZERO)


def _is_zero(e: Expr) -> bool:
    return e.is_zero is True


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:LambertW._eval_nseries
def _lambertw(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.core.add import add
    from sympy_extras.core.function import expand_multinomial
    from sympy_extras.core.order import order
    from sympy_extras.core.power import Pow
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.integers import ceiling
    if len(f.args) != 1:
        return _generic(f, x, n, logx)
    arg = _arg(f).nseries(x, n=n, logx=logx)
    lt = arg.as_leading_term(x, logx=logx)
    lte: Expr = lt.exp if isinstance(lt, Pow) else Integer(1)
    top = ceiling(Integer(n) / lte)
    if not isinstance(top, Integer):
        raise TypeError('cannot determine truth value of %s >= 1' % (top,))
    s: Expr = ZERO
    if top >= 1:
        s = expand_multinomial(add(*[Integer(-1)**(k - 1) * Integer(k)**(k - 2) / factorial(Integer(k - 1)) * arg**k
                                     for k in range(1, int(top))]))
    return s + order(x**n, x)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:Abs._eval_nseries
def _abs(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.core.function import expand
    from sympy_extras.functions.elementary.complexes import sign
    from sympy_extras.functions.elementary.exponential import log
    direction = _arg(f).leadterm(x)[0]
    if logx is not None and direction.has(log(x)):
        direction = _e(direction.subs(log(x), logx))
    s = _ns(_arg(f), x, n, logx)
    return expand(sign(direction) * s)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:sign._eval_nseries
def _sign(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import re
    arg0 = _arg(f)
    x0 = _e(arg0.subs(x, ZERO))
    if x0 != 0:
        return _e(f._evaluated((x0,)))
    d = arg0.dir(x, cdir) if cdir != 0 else cdir
    return Integer(-1) if re(d).is_negative is True else Integer(1)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:arg._eval_nseries
def _arg_rule(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.core.order import order
    if n <= 0:
        return order(Integer(1))
    return f._eval_as_leading_term(x, logx, cdir)


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sinc._eval_nseries
def _sinc(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.elementary.trigonometric import sin
    a = _arg(f)
    # SymPy rebinds x to the argument and expands in it: fixed, in x
    return _ns(sin(a) / a, x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_nseries
def _gamma(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import rf
    from sympy_extras.functions.special.gamma_functions import gamma
    a = _arg(f)
    x0 = _limit0(a, x)
    if not (isinstance(x0, Integer) and x0 <= 0):
        return _generic(f, x, n, logx)
    t = a - x0
    return _ns(gamma(t + 1) / rf(a, -x0 + 1), x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_nseries
def _loggamma(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.elementary.exponential import log
    from sympy_extras.functions.special.gamma_functions import gamma
    if _is_zero(_limit0(_arg(f), x)):
        return _ns(log(gamma(_arg(f))), x, n, logx)
    return _generic(f, x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:expint._eval_nseries
def _expint(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.core.add import add
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.complexes import unpolarify
    from sympy_extras.functions.elementary.exponential import exp, exp_polar
    from sympy_extras.functions.special.error_functions import Chi, Ei, Shi
    nu, z = _arg(f), _arg(f, 1)
    if not nu.has(x):
        if nu == 1:
            # expint._eval_rewrite_as_Si
            return _ns(Shi(z) - Chi(z), x, n, logx)
        if isinstance(nu, Integer) and nu > 1:
            # expint._eval_rewrite_as_Ei, DLMF 8.19.7
            k_nu = int(nu)
            w = -unpolarify(z)
            e1 = -Ei(z * exp_polar(-I * pi)) - I * pi
            g = (w**(nu - 1) / factorial(nu - 1) * e1
                 + exp(w) / factorial(nu - 1) * add(*[factorial(Integer(k_nu - k - 2)) * w**k for k in range(k_nu - 1)]))
            return _ns(g, x, n, logx)
    return _generic(f, x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_nseries
def _ei(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.special.error_functions import Chi, Shi
    z = _arg(f)
    if _is_zero(_limit0(z, x)):
        # Ei._eval_rewrite_as_Si
        g = Shi(z) + Chi(z) - I * pi if z.is_negative is True else Shi(z) + Chi(z)
        return _ns(g, x, n, logx)
    return _generic(f, x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:TrigonometricIntegral._eval_nseries
def _trig_integral(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr,
                   trigfunc: Callable[[Expr], Expr]) -> Optional[Expr]:
    from sympy_extras.core.power import Pow
    from sympy_extras.functions.elementary.exponential import log
    if _e(_arg(f).subs(x, ZERO)) != 0:
        return _generic(f, x, n, logx)
    baseseries = _ns(trigfunc(x), x, n, logx)
    at_zero = trigfunc(ZERO) != 0
    if at_zero:
        baseseries -= 1

    def integrate_power(t: Basic, k: Basic) -> Basic:
        return _e(t)**_e(k) / _e(k)

    baseseries = _e(baseseries.replace(Pow, integrate_power, simultaneous=False))
    if at_zero:
        baseseries += EulerGamma + log(x)
    return _ns(_e(baseseries.subs(x, _arg(f))), x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:li._eval_nseries
def _li(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.core.add import add
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.exponential import log
    z = _arg(f)
    s = [log(z)**k / (factorial(Integer(k)) * k) for k in range(1, n)]
    return EulerGamma + log(log(z)) + add(*s)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Li._eval_nseries
def _offset_li(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    from sympy_extras.functions.special.error_functions import li
    return _ns(li(_arg(f)) - li(Integer(2)), x, n, logx)


# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py:SingularityFunction._eval_nseries
def _singularity(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
    z, a, p = _arg(f), _arg(f, 1), _arg(f, 2)
    shift = _e((z - a).subs(x, ZERO))
    if p.is_negative is True:
        return ZERO
    if p.is_zero is True and shift.is_zero is True:
        return ZERO if cdir == -1 else Integer(1)
    if shift.is_positive is True:
        # SymPy passes the exponent as the order (its n is rebound): fixed
        return _ns((z - a)**p, x, n, logx, cdir)
    return ZERO


# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py:elliptic_k._eval_nseries and elliptic_e._eval_nseries
def _elliptic(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr, first: Rational) -> Optional[Expr]:
    from sympy_extras.functions.special.hyper import hyper
    from sympy_extras.simplify.hyperexpand import hyperexpand
    if len(f.args) != 1:
        return _generic(f, x, n, logx)
    m = _arg(f)
    g = pi / 2 * hyper([first, Rational(1, 2)], [Integer(1)], m)
    return hyperexpand(_ns(g, x, n, logx))


def _trig(name: str) -> Callable[[Expr], Expr]:
    from sympy_extras.functions.elementary.hyperbolic import cosh, sinh
    from sympy_extras.functions.elementary.trigonometric import cos, sin
    table: dict[str, Callable[[Expr], Expr]] = {'SiNode': sin, 'CiNode': cos, 'ShiNode': sinh, 'ChiNode': cosh}
    return table[name]


def rules() -> dict[str, Rule]:
    """The series rules by class name."""
    found: dict[str, Rule] = {
        'LambertW': _lambertw, 'Abs': _abs, 'Sign': _sign, 'Arg': _arg_rule, 'Sinc': _sinc,
        'Gamma': _gamma, 'LogGamma': _loggamma, 'Expint': _expint, 'EiNode': _ei,
        'LiNode': _li, 'OffsetLiNode': _offset_li, 'SingularityFunctionNode': _singularity,
        'EllipticK': lambda f, x, n, logx, cdir: _elliptic(f, x, n, logx, cdir, Rational(1, 2)),
        'EllipticE': lambda f, x, n, logx, cdir: _elliptic(f, x, n, logx, cdir, Rational(-1, 2)),
    }
    for name in ('SiNode', 'CiNode', 'ShiNode', 'ChiNode'):
        found[name] = _trig_integral_rule(_trig(name))
    return found


def _trig_integral_rule(trigfunc: Callable[[Expr], Expr]) -> Rule:
    def rule(f: Expr, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Optional[Expr]:
        return _trig_integral(f, x, n, logx, cdir, trigfunc)
    return rule
