"""The leading terms of the special functions with a rule of their own in
SymPy (``_eval_as_leading_term``), as a table by class name for
:meth:`~sympy_extras.core.function.DefinedFunction._eval_as_leading_term`.

Ported from SymPy's ``sympy.functions`` (the ``_eval_as_leading_term``
methods of LambertW, factorial, binomial, gamma, polygamma, digamma,
trigamma, zeta, erf, erfc, erfi, Ei, Si, Ci, Shi, Chi, fresnels, fresnelc,
arg, floor, ceiling, frac, besselj, bessely, besseli, besselk and
SingularityFunction).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: the _eval_as_leading_term methods of
# sympy/functions/special/error_functions.py, gamma_functions.py,
# zeta_functions.py, bessel.py, singularity_functions.py,
# sympy/functions/combinatorial/factorials.py and
# sympy/functions/elementary/exponential.py (LambertW), complexes.py (arg),
# integers.py (floor, ceiling, frac) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table of functions by class name.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import ComplexInfinity, EulerGamma, I, Infinity, Integer, NaN, NegativeInfinity, pi
from sympy_extras.core.symbol import Symbol

__all__ = ['rules']

Rule = Callable[[Expr, Symbol, Optional[Expr], Expr], Expr]


def _e(b: Basic) -> Expr:
    assert isinstance(b, Expr)
    return b


def _arg(f: Expr, i: int = 0) -> Expr:
    return _e(f.args[i])


def _same(f: Expr, *args: Expr) -> Expr:
    """SymPy's ``self.func(*args)``: the function of the new arguments."""
    return _e(f._evaluated(tuple(args)))


def _lead(e: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return e.as_leading_term(x, logx=logx, cdir=cdir)


def _limit_dir(arg: Expr, x: Symbol, negative: bool) -> Expr:
    from sympy_extras.core.limits import limit
    return limit(arg, x, Integer(0), '-' if negative else '+')


def _log_term(arg: Expr, x: Symbol, logx: Optional[Expr]) -> Expr:
    """``log(c) + e*log(x) + EulerGamma`` for the leading term ``c*x**e``."""
    from sympy_extras.functions.elementary.exponential import log
    c, ex = arg.as_coeff_exponent(x)
    lx = log(x) if logx is None else logx
    return log(c) + ex * lx + EulerGamma


def _re_negative(cdir: Expr) -> bool:
    from sympy_extras.functions.elementary.complexes import re
    return re(cdir).is_negative is True


# PORTED FROM SYMPY: sympy/functions/elementary/exponential.py:LambertW._eval_as_leading_term
def _lambertw(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.polys.polytools import cancel
    if len(f.args) == 1:
        arg = _arg(f)
        arg0 = cancel(_e(arg.subs(x, Integer(0))))
        if arg0.is_zero is not True:
            return _same(f, arg0)
        return arg.as_leading_term(x)
    # SymPy's method returns None here: Expr.as_leading_term raises
    raise NotImplementedError('as_leading_term(%s, %s)' % (f, x))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:factorial._eval_as_leading_term
def _factorial(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.coreerrors import PoleError
    arg = _arg(f).as_leading_term(x)
    arg0 = _e(arg.subs(x, Integer(0)))
    if arg0.is_zero is True:
        return Integer(1)
    if arg0.is_infinite is not True:
        return _same(f, arg)
    raise PoleError('Cannot expand %s around 0' % (f,))


# PORTED FROM SYMPY: sympy/functions/combinatorial/factorials.py:binomial._eval_as_leading_term
def _binomial(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.special.gamma_functions import Gamma
    rewritten = f.rewrite(Gamma)
    assert isinstance(rewritten, Expr)
    return rewritten._eval_as_leading_term(x, logx, cdir)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma._eval_as_leading_term
def _gamma(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.coreerrors import PoleError
    arg = _arg(f)
    x0 = _e(arg.subs(x, Integer(0)))
    if x0.is_integer is True and x0.is_nonpositive is True:
        n = -x0
        res = Integer(-1)**n / _same(f, n + 1)
        return res / (arg + n).as_leading_term(x)
    if x0.is_infinite is not True:
        return _same(f, x0)
    raise PoleError()


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma._eval_as_leading_term
def _polygamma(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.order import Order, order
    from sympy_extras.functions.elementary.exponential import log
    n, z = [a.as_leading_term(x) for a in f.args if isinstance(a, Expr)]
    o = order(z, x)
    assert isinstance(o, Order)
    if n == 0 and o.contains(1 / x):
        lx = log(x) if logx is None else logx
        n_o = o.getn()
        assert n_o is not None
        return n_o * lx
    return _same(f, n, z)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:digamma._eval_as_leading_term
def _digamma(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.special.gamma_functions import polygamma
    return polygamma(Integer(0), _arg(f)).as_leading_term(x)


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:trigamma._eval_as_leading_term
def _trigamma(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.functions.special.gamma_functions import polygamma
    return polygamma(Integer(1), _arg(f)).as_leading_term(x)


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:zeta._eval_as_leading_term
def _zeta(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.function import _generic_leading_term
    s = _arg(f)
    a = _arg(f, 1) if len(f.args) == 2 else Integer(1)
    try:
        _, ex = a.leadterm(x)
    except NotImplementedError:
        return f
    if ex.is_negative is True and s.is_positive is not True:
        raise NotImplementedError
    from sympy_extras.core.function import DefinedFunction
    assert isinstance(f, DefinedFunction)
    return _generic_leading_term(f, x, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_as_leading_term
def _erf(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.power import sqrt
    arg = _lead(_arg(f), x, logx, cdir)
    arg0 = _e(arg.subs(x, Integer(0)))
    if isinstance(arg0, ComplexInfinity):
        arg0 = _limit_dir(arg, x, cdir == -1)
    if x in arg.free_symbols and arg0.is_zero is True:
        return 2 * arg / sqrt(pi)
    return _same(f, arg0)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfc._eval_as_leading_term
def _erfc(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    arg = _lead(_arg(f), x, logx, cdir)
    arg0 = _e(arg.subs(x, Integer(0)))
    if isinstance(arg0, ComplexInfinity):
        arg0 = _limit_dir(arg, x, cdir == -1)
    if arg0.is_zero is True:
        return Integer(1)
    return _same(f, arg0)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfi._eval_as_leading_term
def _erfi(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.power import sqrt
    arg = _lead(_arg(f), x, logx, cdir)
    arg0 = _e(arg.subs(x, Integer(0)))
    if x in arg.free_symbols and arg0.is_zero is True:
        return 2 * arg / sqrt(pi)
    if arg0.is_finite is True:
        return _same(f, arg0)
    return _same(f, arg)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_as_leading_term
def _ei(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.function import DefinedFunction, _generic_leading_term
    from sympy_extras.core.limits import limit
    x0 = limit(_arg(f), x, Integer(0))
    arg = _arg(f).as_leading_term(x, cdir=cdir)
    direction = arg.dir(x, cdir)
    if x0.is_zero is True:
        term = _log_term(arg, x, logx)
        return term - (I * pi if _re_negative(direction) else Integer(0))
    assert isinstance(f, DefinedFunction)
    return _generic_leading_term(f, x, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Si._eval_as_leading_term and Shi._eval_as_leading_term
def _odd_integral(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr, own_arg: Expr) -> Expr:
    """Si and Shi: the argument for a vanishing one."""
    arg = own_arg
    arg0 = _e(arg.subs(x, Integer(0)))
    if isinstance(arg0, NaN):
        arg0 = _limit_dir(arg, x, _re_negative(cdir))
    if arg0.is_zero is True:
        return arg
    if arg0.is_infinite is not True:
        return _same(f, arg0)
    return f


def _si(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return _odd_integral(f, x, logx, cdir, _lead(_arg(f), x, logx, cdir))


def _shi(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    return _odd_integral(f, x, logx, cdir, _arg(f).as_leading_term(x))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ci._eval_as_leading_term and Chi._eval_as_leading_term
def _even_integral(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    """Ci and Chi: the logarithm for a vanishing argument."""
    arg = _lead(_arg(f), x, logx, cdir)
    arg0 = _e(arg.subs(x, Integer(0)))
    if isinstance(arg0, NaN):
        arg0 = _limit_dir(arg, x, _re_negative(cdir))
    if arg0.is_zero is True:
        return _log_term(arg, x, logx)
    if arg0.is_finite is True:
        return _same(f, arg0)
    return f


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:fresnels._eval_as_leading_term and fresnelc._eval_as_leading_term
def _fresnel(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr, sine: bool) -> Expr:
    from sympy_extras.core.numbers import Rational
    from sympy_extras.core.order import order
    arg = _lead(_arg(f), x, logx, cdir)
    arg0 = _e(arg.subs(x, Integer(0)))
    if isinstance(arg0, ComplexInfinity):
        arg0 = _limit_dir(arg, x, _re_negative(cdir))
    if arg0.is_zero is True:
        return pi * arg**3 / 6 if sine else arg
    if isinstance(arg0, (Infinity, NegativeInfinity)):
        s = 1 if isinstance(arg0, Infinity) else -1
        return s * Rational(1, 2) + order(x, x)
    return _same(f, arg0)


# PORTED FROM SYMPY: sympy/functions/elementary/complexes.py:arg._eval_as_leading_term
def _arg_rule(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.coreerrors import PoleError
    from sympy_extras.core.symbol import Dummy
    t = Dummy('t', positive=True)
    d = Integer(1) if cdir == 0 else cdir
    z = _e(_arg(f).subs(x, d * t))
    if z.is_positive is True:
        return Integer(0)
    if z.is_negative is True:
        return pi
    raise PoleError('Cannot expand %s around 0' % (f,))


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:floor._eval_as_leading_term and ceiling._eval_as_leading_term
def _round(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr, up: bool) -> Expr:
    from sympy_extras.core.accumbounds import AccumBounds
    arg = _arg(f)
    arg0 = _e(arg.subs(x, Integer(0)))
    r = _e(f.subs(x, Integer(0)))
    if isinstance(arg0, (NaN, AccumBounds)):
        arg0 = _limit_dir(arg, x, _re_negative(cdir))
        r = _same(f, arg0)
    if arg0.is_finite is True:
        if arg0 == r:
            ndir = arg.dir(x, cdir if cdir != 0 else Integer(1))
            if ndir.is_negative is True:
                return r if up else r - 1
            if ndir.is_positive is True:
                return r + 1 if up else r
            raise NotImplementedError('Not sure of sign of %s' % ndir)
        return r
    return _lead(arg, x, logx, cdir)


# PORTED FROM SYMPY: sympy/functions/elementary/integers.py:frac._eval_as_leading_term
def _frac(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.accumbounds import AccumBounds
    arg = _arg(f)
    arg0 = _e(arg.subs(x, Integer(0)))
    r = _e(f.subs(x, Integer(0)))
    if arg0.is_finite is True:
        if r.is_zero is True:
            ndir = arg.dir(x, cdir)
            if ndir.is_negative is True:
                return Integer(1)
            return _lead(arg - arg0, x, logx, cdir)
        return r
    if isinstance(arg0, (ComplexInfinity, Infinity, NegativeInfinity)):
        return AccumBounds(Integer(0), Integer(1))
    return _lead(arg, x, logx, cdir)


def _bessel_parts(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr, Expr]]:
    """``nu, z``, the leading term of ``z`` and its ``c*x**e``; None when
    the leading term of ``z`` is not known (SymPy then returns the function)."""
    nu, z = _arg(f), _arg(f, 1)
    try:
        lead = z.as_leading_term(x)
    except NotImplementedError:
        return None
    c, ex = lead.as_coeff_exponent(x)
    return nu, z, lead, c, ex


def _generic(f: Expr, x: Symbol, logx: Optional[Expr]) -> Expr:
    from sympy_extras.core.function import DefinedFunction, _generic_leading_term
    assert isinstance(f, DefinedFunction)
    return _generic_leading_term(f, x, logx)


def _large_positive(c: Expr, ex: Expr, cdir: Expr) -> bool:
    """SymPy's ``not (c*cdir**e).is_negative`` with ``cdir`` 0 taken as 1."""
    d = Integer(1) if cdir == 0 else cdir
    return (c * d**ex).is_negative is not True


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselj._eval_as_leading_term and besseli._eval_as_leading_term
def _bessel_ji(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr, modified: bool) -> Expr:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.trigonometric import cos
    from sympy_extras.functions.special.gamma_functions import gamma
    parts = _bessel_parts(f, x)
    if parts is None:
        return f
    nu, z, lead, c, ex = parts
    if ex.is_positive is True:
        return lead**nu / (2**nu * gamma(nu + 1))
    if ex.is_negative is True:
        if _large_positive(c, ex, cdir):
            # Abramowitz and Stegun 1965, p. 364 (J) and p. 377 (I)
            if modified:
                return exp(z) / sqrt(2 * pi * z)
            return sqrt(2) * cos(z - pi * (2 * nu + 1) / 4) / sqrt(pi * z)
        return f
    return _generic(f, x, logx)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:bessely._eval_as_leading_term
def _bessel_y(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.add import add
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.exponential import log
    from sympy_extras.functions.elementary.trigonometric import cos, sin
    from sympy_extras.functions.special.bessel import besselj
    from sympy_extras.functions.special.gamma_functions import digamma
    parts = _bessel_parts(f, x)
    if parts is None:
        return f
    nu, z, lead, c, ex = parts
    if ex.is_positive is True:
        term_one = (2 / pi) * log(z / 2) * besselj(nu, z)
        term_two = -(z / 2)**(-nu) * factorial(nu - 1) / pi if nu.is_positive is True else Integer(0)
        term_three = -(z / 2)**nu / (pi * factorial(nu)) * (digamma(nu + 1) - EulerGamma)
        return add(term_one, term_two, term_three).as_leading_term(x, logx=logx)
    if ex.is_negative is True:
        if _large_positive(c, ex, cdir):
            # Abramowitz and Stegun 1965, p. 364
            return (sqrt(2) * (-sin(pi * nu / 2 - z + pi / 4) + 3 * cos(pi * nu / 2 - z + pi / 4) / (8 * z))
                    * sqrt(1 / z) / sqrt(pi))
        return f
    return _generic(f, x, logx)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_as_leading_term
def _bessel_k(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.complexes import abs_
    from sympy_extras.functions.elementary.exponential import exp, log
    from sympy_extras.functions.special.gamma_functions import gamma
    parts = _bessel_parts(f, x)
    if parts is None:
        return f
    nu, z, lead, _, ex = parts
    if ex.is_positive is True:
        if nu.is_zero is True:
            # Abramowitz and Stegun (10th ed, 1972), 9.6.8
            term = -log(z) - EulerGamma + log(Integer(2))
        elif nu.is_nonzero is True:
            # Abramowitz and Stegun (10th ed, 1972), 9.6.9
            term = gamma(abs_(nu)) * (z / 2)**(-abs_(nu)) / 2
        else:
            raise NotImplementedError(f'Cannot proceed without knowing if {nu} is zero or not.')
        return term.as_leading_term(x, logx=logx)
    if ex.is_negative is True:
        # Abramowitz and Stegun (10th ed, 1972), 9.7.2
        return sqrt(pi) * exp(-lead) / sqrt(2 * lead)
    return _same(f, nu, lead)


# PORTED FROM SYMPY: sympy/functions/special/singularity_functions.py:SingularityFunction._eval_as_leading_term
def _singularity(f: Expr, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
    z, a, n = _arg(f), _arg(f, 1), _arg(f, 2)
    shift = _e((z - a).subs(x, Integer(0)))
    if n.is_negative is True:
        return Integer(0)
    if n.is_zero is True and shift.is_zero is True:
        return Integer(0) if cdir == -1 else Integer(1)
    if shift.is_positive is True:
        return shift**n
    return Integer(0)


def rules() -> dict[str, Rule]:
    """The leading-term rules by class name."""
    return {
        'LambertW': _lambertw, 'Factorial': _factorial, 'Binomial': _binomial, 'Gamma': _gamma,
        'PolyGamma': _polygamma, 'DiGamma': _digamma, 'TriGamma': _trigamma, 'Zeta': _zeta,
        'Erf': _erf, 'Erfc': _erfc, 'Erfi': _erfi, 'EiNode': _ei, 'SiNode': _si, 'ShiNode': _shi,
        'CiNode': _even_integral, 'ChiNode': _even_integral,
        'FresnelS': lambda f, x, logx, cdir: _fresnel(f, x, logx, cdir, True),
        'FresnelC': lambda f, x, logx, cdir: _fresnel(f, x, logx, cdir, False),
        'Arg': _arg_rule, 'Frac': _frac,
        'Floor': lambda f, x, logx, cdir: _round(f, x, logx, cdir, False),
        'Ceiling': lambda f, x, logx, cdir: _round(f, x, logx, cdir, True),
        'BesselJ': lambda f, x, logx, cdir: _bessel_ji(f, x, logx, cdir, False),
        'BesselI': lambda f, x, logx, cdir: _bessel_ji(f, x, logx, cdir, True),
        'BesselY': _bessel_y, 'BesselK': _bessel_k, 'SingularityFunctionNode': _singularity,
    }
