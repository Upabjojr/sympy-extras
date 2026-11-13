"""The ``rewrite`` rules of the special functions between themselves
(SymPy's ``_eval_rewrite_as_<name>`` of the Bessel and error functions),
as a typed table for :func:`sympy_extras.core.rewriting.rewrite`. A rule
returns ``None`` where SymPy's leaves the node as it is.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/bessel.py (besselj.
# _eval_rewrite_as_jn, besselj._eval_rewrite_as_besseli, besseli.
# _eval_rewrite_as_besselj, bessely._eval_rewrite_as_besselj, besselk.
# _eval_rewrite_as_besseli and _eval_rewrite_as_besselj), sympy/functions/
# special/error_functions.py (erfc, erfi, erf2 ._eval_rewrite_as_erf,
# expint._eval_rewrite_as_Ei, Ei._eval_rewrite_as_expint, Li._eval_rewrite_as_li), sympy/functions/special/
# zeta_functions.py (dirichlet_eta._eval_rewrite_as_zeta); and the other _eval_rewrite_as_<name>
# methods of erf, erfc, erfi, erf2, erfinv, erfcinv, Ei, expint, li, TrigonometricIntegral, Si, Ci, Shi,
# Chi, fresnels, fresnelc (error_functions.py), lowergamma, uppergamma, polygamma, digamma, trigamma
# (gamma_functions.py), lerchphi, polylog, zeta, dirichlet_eta, riemann_xi (zeta_functions.py), beta,
# betainc, betainc_regularized (beta_functions.py), elliptic_k, elliptic_f, elliptic_e, elliptic_pi
# (elliptic_integrals.py), hyper, HyperRep (hyper.py), marcumq (bessel.py) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target class or name); erfi._eval_rewrite_as_expint and
# li._eval_rewrite_as_Shi (Chi) corrected (wrong in SymPy 1.14).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional, Union

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.rewriting import NodeRule, RewriteHints, Rule, RuleKey
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['rules', 'node_rules']


def _besselj_jn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.power import sqrt
    from .bessel import jn
    nu, z = args
    return sqrt(2 * z / pi) * jn(nu - Rational(1, 2), z)


def _besselj_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import exp
    from .bessel import besseli
    nu, z = args
    return exp(I * pi * nu / 2) * besseli(nu, polar_lift(-I) * z)


def _besseli_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import exp
    from .bessel import besselj
    nu, z = args
    return exp(-I * pi * nu / 2) * besselj(nu, polar_lift(I) * z)


def _bessely_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.trigonometric import cos, csc
    from .bessel import besselj
    nu, z = args
    if nu.is_integer is False:
        return csc(pi * nu) * (cos(pi * nu) * besselj(nu, z) - besselj(-nu, z))
    return None


def _besselk_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.trigonometric import csc
    from .bessel import besseli
    nu, z = args
    if nu.is_integer is False:
        return pi * csc(pi * nu) * (besseli(-nu, z) - besseli(nu, z)) / 2
    return None


def _besselk_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.rewriting import rewrite
    from .bessel import BesselJ
    ai = _besselk_besseli(e, args, hints)
    if ai is None:
        return None
    rewritten = rewrite(ai, (), BesselJ, hints)
    assert isinstance(rewritten, Expr)
    return rewritten


def _erfc_erf(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .error_functions import erf
    return Integer(1) - erf(args[0])


def _erfi_erf(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .error_functions import erf
    return -I * erf(I * args[0])


def _erf2_erf(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .error_functions import erf
    x, y = args
    return erf(y) - erf(x)


def _expint_ei(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.elementary.complexes import unpolarify
    from sympy_extras.functions.elementary.exponential import exp, exp_polar
    from .error_functions import Ei
    nu, z = args
    e1 = -Ei(z * exp_polar(-I * pi)) - I * pi
    if nu == 1:
        return e1
    if isinstance(nu, Integer) and nu.p > 1:
        # DLMF 8.19.7
        x = -unpolarify(z)
        m = nu.p
        return (x**(nu - 1) / factorial(nu - 1) * e1
                + exp(x) / factorial(nu - 1) * add(*[factorial(Integer(m - k - 2)) * x**k for k in range(m - 1)]))
    return None


def _ei_expint(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import polar_lift
    from .error_functions import expint
    return -expint(Integer(1), polar_lift(Integer(-1)) * args[0]) - I * pi


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Li._eval_rewrite_as_li
def _offset_li_li(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .error_functions import li
    return li(args[0]) - li(Integer(2))


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:dirichlet_eta._eval_rewrite_as_zeta
def _eta_zeta(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.elementary.exponential import log
    from sympy_extras.functions.elementary.piecewise import piecewise
    from .gamma_functions import digamma
    from .zeta_functions import zeta
    s = args[0]
    a = args[1] if len(args) > 1 else Integer(1)
    two = Integer(2)
    if a == Integer(1):
        return piecewise((log(two), eq(s, Integer(1))), ((1 - two**(1 - s)) * zeta(s), True))
    return piecewise((log(two) - digamma(a) + digamma((a + 1) / 2), eq(s, Integer(1))),
                     (zeta(s, a) - two**(1 - s) * zeta(s, (a + 1) / 2), True))


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name)."""
    from . import bessel as b, error_functions as er, zeta_functions as z
    table: dict[RuleKey, Rule] = {(b.BesselJ, b.SphericalJ): _besselj_jn, (b.BesselJ, b.BesselI): _besselj_besseli,
            (b.BesselI, b.BesselJ): _besseli_besselj, (b.BesselY, b.BesselJ): _bessely_besselj,
            (b.BesselK, b.BesselI): _besselk_besseli, (b.BesselK, b.BesselJ): _besselk_besselj,
            (er.Erfc, er.Erf): _erfc_erf, (er.Erfi, er.Erf): _erfi_erf, (er.Erf2, er.Erf): _erf2_erf,
            (er.Expint, er.EiNode): _expint_ei, (er.EiNode, er.Expint): _ei_expint,
            (er.OffsetLiNode, er.LiNode): _offset_li_li,
            (z.DirichletEta, z.Zeta): _eta_zeta}
    table.update(_error_rules())
    table.update(_integral_function_rules())
    table.update(_gamma_zeta_rules())
    table.update(_beta_elliptic_rules())
    from .hyperrep import HyperRep
    table[(HyperRep, 'nonrep')] = _hyperrep(False)
    table[(HyperRep, 'nonrepsmall')] = _hyperrep(True)
    return table


HALF = Rational(1, 2)


def _rule(f: Callable[[tuple[Expr, ...]], Optional[Expr]]) -> Rule:
    """A rule of the (rewritten) arguments only."""
    def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return f(args)
    return rule


def _rewritten(e: Expr, target: Union[type, str], hints: RewriteHints) -> Expr:
    """SymPy's ``e.rewrite(target)`` inside a rule."""
    from sympy_extras.core.rewriting import rewrite
    r = rewrite(e, (), target, hints)
    assert isinstance(r, Expr)
    return r


def _dummy(name: str, args: tuple[Basic, ...]) -> Dummy:
    """SymPy's ``Dummy(uniquely_named_symbol(name, args).name)``."""
    from sympy_extras.core.containers import Tuple
    from sympy_extras.core.function import _uniquely_named_symbol
    return Dummy(_uniquely_named_symbol(name, Tuple(*args)).name)


def _integral(f: Expr, t: Symbol, a: Expr, b: Expr) -> Expr:
    """SymPy's ``Integral(f, (t, a, b))``, unevaluated."""
    from sympy_extras.integrals.integral import Integral
    return Integral(f, t, a, b)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_rewrite_as_<name>,
# erfc._eval_rewrite_as_<name>, erfi._eval_rewrite_as_<name>, erf2._eval_rewrite_as_<name>,
# erfinv._eval_rewrite_as_erfcinv, erfcinv._eval_rewrite_as_erfinv, fresnels._eval_rewrite_as_<name>,
# fresnelc._eval_rewrite_as_<name>
def _error_rules() -> dict[RuleKey, Rule]:
    """The error functions and the Fresnel integrals between themselves and
    as hypergeometric, Meijer G and incomplete gamma functions."""
    from sympy_extras.core.power import sqrt
    from sympy_extras.functions.elementary.miscellaneous import root
    from sympy_extras.functions.elementary.trigonometric import cos, sin
    from sympy_extras.functions.special.gamma_functions import UpperGamma, uppergamma
    from sympy_extras.functions.special.hyper import hyper, meijerg
    from . import error_functions as er
    from .error_functions import erf, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels
    one, zero = Integer(1), Integer(0)
    sqrt_pi = sqrt(pi)

    def fresnel_form(z: Expr) -> Expr:
        arg = (one - I) * z / sqrt_pi
        return (one + I) * (fresnelc(arg) - I * fresnels(arg))

    def erfi_fresnel_form(z: Expr) -> Expr:
        arg = (one + I) * z / sqrt_pi
        return (one - I) * (fresnelc(arg) - I * fresnels(arg))

    def erf_meijerg(z: Expr) -> Expr:
        return z / sqrt_pi * meijerg([HALF], [], [zero], [Rational(-1, 2)], z**2)

    def erf_hyper(z: Expr) -> Expr:
        return 2 * z / sqrt_pi * hyper([HALF], [3 * HALF], -z**2)

    def erf_uppergamma(z: Expr) -> Expr:
        return sqrt(z**2) / z * (one - uppergamma(HALF, z**2) / sqrt_pi)

    def erf_expint(z: Expr) -> Expr:
        return sqrt(z**2) / z - z * expint(HALF, z**2) / sqrt_pi

    def erf2_rule(target: Union[type, str]) -> Rule:
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            x, y = args
            return _rewritten(erf(y), target, hints) - _rewritten(erf(x), target, hints)
        return rule

    def erf2_uppergamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        x, y = args
        return erf_uppergamma(y) - erf_uppergamma(x)

    def erf2_erfc(args: tuple[Expr, ...]) -> Optional[Expr]:
        x, y = args
        return erfc(x) - erfc(y)

    def erf2_erfi(args: tuple[Expr, ...]) -> Optional[Expr]:
        x, y = args
        return I * (erfi(I * x) - erfi(I * y))

    def fresnels_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return (pi * z**Rational(9, 4) / (sqrt(Integer(2)) * (z**2)**Rational(3, 4) * (-z)**Rational(3, 4))
                * meijerg([], [one], [Rational(3, 4)], [Rational(1, 4), zero], -pi**2 * z**4 / 16))

    def fresnelc_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return (pi * z**Rational(3, 4) / (sqrt(Integer(2)) * root(z**2, Integer(4)) * root(-z, Integer(4)))
                * meijerg([], [one], [Rational(1, 4)], [Rational(3, 4), zero], -pi**2 * z**4 / 16))

    def fresnel_integral(trig: Callable[[Expr], Expr]) -> Rule:
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            z = args[0]
            t = _dummy('t', (z,))
            return _integral(trig(pi * t**2 / 2), t, zero, z)
        return rule

    plus, minus = (one + I) / 2 * sqrt_pi, (one - I) / 2 * sqrt_pi

    def fresnels_erf(args: tuple[Expr, ...]) -> Optional[Expr]:
        return (one + I) / 4 * (erf(plus * args[0]) - I * erf(minus * args[0]))

    def fresnelc_erf(args: tuple[Expr, ...]) -> Optional[Expr]:
        return (one - I) / 4 * (erf(plus * args[0]) + I * erf(minus * args[0]))

    def fresnels_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return pi * z**3 / 6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2 * z**4 / 16)

    def fresnelc_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return z * hyper([Rational(1, 4)], [HALF, Rational(5, 4)], -pi**2 * z**4 / 16)

    def erfc_expint(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return one - sqrt(z**2) / z + z * expint(HALF, z**2) / sqrt_pi

    def erfi_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return z / sqrt_pi * meijerg([HALF], [], [zero], [Rational(-1, 2)], -z**2)

    def erfi_uppergamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return sqrt(-z**2) / z * (uppergamma(HALF, -z**2) / sqrt_pi - one)

    def erfi_expint(args: tuple[Expr, ...]) -> Optional[Expr]:
        # SymPy's sqrt(-z**2)/z - z*expint(1/2, -z**2)/sqrt(pi) is off by
        # 2*sqrt(-z**2)/z (2*I at z = 7/10): -I*erf(I*z) with erf's rule
        z = args[0]
        return -sqrt(-z**2) / z - z * expint(HALF, -z**2) / sqrt_pi

    return {
        (er.Erf, er.Erfc): _rule(lambda a: one - erfc(a[0])),
        (er.Erf, er.Erfi): _rule(lambda a: -I * erfi(I * a[0])),
        (er.Erf, er.FresnelS): _rule(lambda a: fresnel_form(a[0])),
        (er.Erf, er.FresnelC): _rule(lambda a: fresnel_form(a[0])),
        (er.Erf, 'meijerg'): _rule(lambda a: erf_meijerg(a[0])),
        (er.Erf, 'hyper'): _rule(lambda a: erf_hyper(a[0])),
        (er.Erf, UpperGamma): _rule(lambda a: erf_uppergamma(a[0])),
        (er.Erf, er.Expint): _rule(lambda a: erf_expint(a[0])),
        (er.Erfc, er.Erfi): _rule(lambda a: one + I * erfi(I * a[0])),
        (er.Erfc, er.FresnelS): _rule(lambda a: one - fresnel_form(a[0])),
        (er.Erfc, er.FresnelC): _rule(lambda a: one - fresnel_form(a[0])),
        (er.Erfc, 'meijerg'): _rule(lambda a: one - erf_meijerg(a[0])),
        (er.Erfc, 'hyper'): _rule(lambda a: one - erf_hyper(a[0])),
        (er.Erfc, UpperGamma): _rule(lambda a: one - erf_uppergamma(a[0])),
        (er.Erfc, er.Expint): _rule(erfc_expint),
        (er.Erfi, er.Erfc): _rule(lambda a: I * erfc(I * a[0]) - I),
        (er.Erfi, er.FresnelS): _rule(lambda a: erfi_fresnel_form(a[0])),
        (er.Erfi, er.FresnelC): _rule(lambda a: erfi_fresnel_form(a[0])),
        (er.Erfi, 'meijerg'): _rule(erfi_meijerg),
        (er.Erfi, 'hyper'): _rule(lambda a: 2 * a[0] / sqrt_pi * hyper([HALF], [3 * HALF], a[0]**2)),
        (er.Erfi, UpperGamma): _rule(erfi_uppergamma),
        (er.Erfi, er.Expint): _rule(erfi_expint),
        (er.Erf2, er.Erfc): _rule(erf2_erfc), (er.Erf2, er.Erfi): _rule(erf2_erfi),
        (er.Erf2, er.FresnelS): erf2_rule(er.FresnelS), (er.Erf2, er.FresnelC): erf2_rule(er.FresnelC),
        (er.Erf2, 'meijerg'): erf2_rule('meijerg'), (er.Erf2, 'hyper'): erf2_rule('hyper'),
        (er.Erf2, UpperGamma): _rule(erf2_uppergamma), (er.Erf2, er.Expint): erf2_rule(er.Expint),
        (er.Erfinv, er.Erfcinv): _rule(lambda a: erfcinv(one - a[0])),
        (er.Erfcinv, er.Erfinv): _rule(lambda a: erfinv(one - a[0])),
        (er.FresnelS, er.Erf): _rule(fresnels_erf), (er.FresnelC, er.Erf): _rule(fresnelc_erf),
        (er.FresnelS, 'hyper'): _rule(fresnels_hyper), (er.FresnelC, 'hyper'): _rule(fresnelc_hyper),
        (er.FresnelS, 'meijerg'): _rule(fresnels_meijerg), (er.FresnelC, 'meijerg'): _rule(fresnelc_meijerg),
        (er.FresnelS, 'Integral'): fresnel_integral(sin), (er.FresnelC, 'Integral'): fresnel_integral(cos),
    }


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_rewrite_as_<name>,
# expint._eval_rewrite_as_<name>, li._eval_rewrite_as_<name>, TrigonometricIntegral._eval_rewrite_as_Ei,
# TrigonometricIntegral._eval_rewrite_as_uppergamma, Si._eval_rewrite_as_<name>, Ci._eval_rewrite_as_<name>,
# Shi._eval_rewrite_as_expint, Chi._eval_rewrite_as_expint
def _integral_function_rules() -> dict[RuleKey, Rule]:
    """The exponential, logarithmic and trigonometric integrals between
    themselves and as integrals, incomplete gamma, hypergeometric and
    Meijer G-functions."""
    from sympy_extras.core.numbers import E, EulerGamma, oo
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import Log, exp, exp_polar, log
    from sympy_extras.functions.elementary.trigonometric import Sinc, cos, sinc
    from sympy_extras.functions.special.gamma_functions import UpperGamma, uppergamma
    from sympy_extras.functions.special.hyper import hyper, meijerg
    from . import error_functions as er
    from .error_functions import E1, Chi, Ci, Ei, Li, Shi, Si, li
    one, zero = Integer(1), Integer(0)

    def half_log(z: Expr) -> Expr:
        return HALF * (log(log(z)) - log(one / log(z)))

    def ei_si(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        if z.is_negative:
            return Shi(z) + Chi(z) - I * pi
        return Shi(z) + Chi(z)

    def ei_li(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        if isinstance(z, Log):
            inner = z.args[0]
            assert isinstance(inner, Expr)
            return li(inner)
        return li(exp(z))

    def ei_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        t = _dummy('t', (z,))
        return _integral(E**t / t, t, -oo, z)

    def ei_uppergamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        return -uppergamma(zero, polar_lift(Integer(-1)) * args[0]) - I * pi

    def expint_si(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        nu, z = args
        if nu != 1:
            return e
        return Shi(z) - Chi(z)

    def expint_integral(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        n, x = [a for a in e.args if isinstance(a, Expr)]
        t = _dummy('t', args)
        return _integral(t**-n * exp(-t * x), t, one, oo)

    def expint_uppergamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        nu, z = args
        return z**(nu - 1) * uppergamma(one - nu, z)

    def li_si(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return Ci(I * log(z)) - I * Si(I * log(z)) - HALF * (log(one / log(z)) - log(log(z))) - log(I * log(z))

    def li_shi(args: tuple[Expr, ...]) -> Optional[Expr]:
        # SymPy's Chi(log(z)) - Shi(log(z)) - (log(1/log(z)) - log(log(z)))/2
        # is not li(z) (li(27/10) - 2.105...): Ei(log(z)) with Ei = Chi + Shi
        # and Ei's logarithm, -(log(L) + log(1/L))/2 for log(L) in Chi
        z = args[0]
        return Chi(log(z)) + Shi(log(z)) - HALF * (log(one / log(z)) + log(log(z)))

    def li_uppergamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return -uppergamma(zero, -log(z)) + half_log(z) - log(-log(z))

    def li_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return log(z) * hyper([one, one], [Integer(2), Integer(2)], log(z)) + half_log(z) + EulerGamma

    def li_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        return (-log(-log(z)) - HALF * (log(one / log(z)) - log(log(z)))
                - meijerg([], [one], [zero, zero], [], -log(z)))

    def expint_form(e: Expr, z: Expr) -> Expr:
        """The ``_eval_rewrite_as_expint`` of the trigonometric integrals."""
        if isinstance(e, er.SiNode):
            return pi / 2 + (E1(polar_lift(I) * z) - E1(polar_lift(-I) * z)) / 2 / I
        if isinstance(e, er.CiNode):
            return -(E1(polar_lift(I) * z) + E1(polar_lift(-I) * z)) / 2
        if isinstance(e, er.ShiNode):
            return (E1(z) - E1(exp_polar(I * pi) * z)) / 2 - I * pi / 2
        return -I * pi / 2 - (E1(z) + E1(exp_polar(I * pi) * z)) / 2

    def trig_expint(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return expint_form(e, args[0])

    def trig_via_expint(target: type) -> Rule:
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            return _rewritten(expint_form(e, args[0]), target, hints)
        return rule

    def si_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        t = _dummy('t', (z,))
        return _integral(sinc(t), t, zero, z)

    def ci_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        z = args[0]
        t = _dummy('t', (z,))
        return EulerGamma + log(z) - _integral((one - cos(t)) / t, t, zero, z)

    table: dict[RuleKey, Rule] = {
        (er.EiNode, UpperGamma): _rule(ei_uppergamma),
        (er.EiNode, er.LiNode): _rule(ei_li), (er.EiNode, 'Integral'): _rule(ei_integral),
        (er.Expint, UpperGamma): _rule(expint_uppergamma),
        (er.Expint, 'Integral'): expint_integral,
        (er.LiNode, er.OffsetLiNode): _rule(lambda a: Li(a[0]) + li(Integer(2))),
        (er.LiNode, er.EiNode): _rule(lambda a: Ei(log(a[0]))),
        (er.LiNode, UpperGamma): _rule(li_uppergamma),
        (er.LiNode, er.SiNode): _rule(li_si), (er.LiNode, er.CiNode): _rule(li_si),
        (er.LiNode, er.ShiNode): _rule(li_shi), (er.LiNode, er.ChiNode): _rule(li_shi),
        (er.LiNode, 'hyper'): _rule(li_hyper), (er.LiNode, 'meijerg'): _rule(li_meijerg),
        (er.TrigonometricIntegral, er.EiNode): trig_via_expint(er.EiNode),
        (er.TrigonometricIntegral, UpperGamma): trig_via_expint(UpperGamma),
        (er.SiNode, 'Integral'): _rule(si_integral), (er.SiNode, Sinc): _rule(si_integral),
        (er.CiNode, 'Integral'): _rule(ci_integral)}
    for target in (er.SiNode, er.CiNode, er.ChiNode, er.ShiNode):
        table[(er.EiNode, target)] = _rule(ei_si)
        table[(er.Expint, target)] = expint_si
    for cls in (er.SiNode, er.CiNode, er.ShiNode, er.ChiNode):
        table[(cls, er.Expint)] = trig_expint
    return table


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:lowergamma._eval_rewrite_as_<name>,
# uppergamma._eval_rewrite_as_<name>, polygamma._eval_rewrite_as_<name>, digamma._eval_rewrite_as_<name>,
# trigamma._eval_rewrite_as_<name>; sympy/functions/special/zeta_functions.py:lerchphi._eval_rewrite_as_<name>,
# polylog._eval_rewrite_as_lerchphi, zeta._eval_rewrite_as_<name>, dirichlet_eta._eval_rewrite_as_genocchi,
# riemann_xi._eval_rewrite_as_zeta
def _gamma_zeta_rules() -> dict[RuleKey, Rule]:
    """The incomplete gamma and polygamma functions, the zeta functions and
    the polylogarithm between themselves and in harmonic, Bernoulli and
    Genocchi numbers."""
    from sympy_extras.core.numbers import EulerGamma
    from sympy_extras.core.relational import eq
    from sympy_extras.functions.combinatorial.factorials import factorial
    from sympy_extras.functions.combinatorial.numbers import (Bernoulli, Genocchi, Harmonic, bernoulli, genocchi,
                                                              harmonic)
    from sympy_extras.functions.elementary.exponential import log
    from sympy_extras.functions.elementary.piecewise import piecewise
    from . import gamma_functions as g, zeta_functions as zf
    from .error_functions import Expint, expint
    from .gamma_functions import digamma, gamma, lowergamma, polygamma, uppergamma
    from .zeta_functions import dirichlet_eta, lerchphi, zeta
    one, minus_one, two = Integer(1), Integer(-1), Integer(2)

    def lower_expint(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        s = args[0]
        if s.is_integer and s.is_nonpositive:
            return e
        return _rewritten(_rewritten(e, g.UpperGamma, hints), Expint, hints)

    def polygamma_zeta(args: tuple[Expr, ...]) -> Optional[Expr]:
        n, z = args
        if n.is_integer and n.is_positive:
            return minus_one**(n + 1) * factorial(n) * zeta(n + 1, z)
        return None

    def polygamma_harmonic(args: tuple[Expr, ...]) -> Optional[Expr]:
        n, z = args
        if n.is_integer:
            if n.is_zero:
                return harmonic(z - 1) - EulerGamma
            return minus_one**(n + 1) * factorial(n) * (zeta(n + 1) - harmonic(z - 1, n + 1))
        return None

    def lerchphi_helper(target: type) -> Rule:
        def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
            from sympy_extras.core.function import ExpandHints
            from sympy_extras.functions._expand_func import _lerchphi
            res = _lerchphi(e, ExpandHints(True, True, True, True, True, False))
            if res is not None and res.has(target):
                return res
            return e
        return rule

    def s_a(args: tuple[Expr, ...]) -> tuple[Expr, Expr]:
        return args[0], args[1] if len(args) > 1 else one

    def zeta_bernoulli(args: tuple[Expr, ...]) -> Optional[Expr]:
        s, a = s_a(args)
        if a == 1 and s.is_integer and s.is_nonnegative and s.is_even:
            return -(2 * pi * I)**s * bernoulli(s) / (2 * factorial(s))
        return bernoulli(one - s, a) / (s - 1)

    def zeta_eta(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        a = s_a(args)[1]
        if a != 1:
            return e
        s = e.args[0]
        assert isinstance(s, Expr)
        return dirichlet_eta(s) / (1 - two**(1 - s))

    def zeta_lerchphi(args: tuple[Expr, ...]) -> Optional[Expr]:
        s, a = s_a(args)
        return lerchphi(one, s, a)

    def eta_genocchi(args: tuple[Expr, ...]) -> Optional[Expr]:
        s, a = s_a(args)
        return piecewise((log(two) - digamma(a) + digamma((a + 1) / 2), eq(s, one)),
                         (genocchi(one - s, a) / (2 * (s - 1)), True))

    def xi_zeta(args: tuple[Expr, ...]) -> Optional[Expr]:
        s = args[0]
        return s * (s - 1) * gamma(s / 2) * zeta(s) / (2 * pi**(s / 2))

    def harmonic_trigamma(args: tuple[Expr, ...]) -> Optional[Expr]:
        return -harmonic(args[0] - 1, two) + pi**2 / 6

    return {
        (g.LowerGamma, g.UpperGamma): _rule(lambda a: gamma(a[0]) - uppergamma(a[0], a[1])),
        (g.LowerGamma, Expint): lower_expint,
        (g.UpperGamma, g.LowerGamma): _rule(lambda a: gamma(a[0]) - lowergamma(a[0], a[1])),
        (g.UpperGamma, Expint): _rule(lambda a: expint(one - a[0], a[1]) * a[1]**a[0]),
        (g.PolyGamma, zf.Zeta): _rule(polygamma_zeta), (g.PolyGamma, Harmonic): _rule(polygamma_harmonic),
        (g.DiGamma, Harmonic): _rule(lambda a: harmonic(a[0] - 1) - EulerGamma),
        (g.DiGamma, g.PolyGamma): _rule(lambda a: polygamma(Integer(0), a[0])),
        (g.TriGamma, zf.Zeta): _rule(lambda a: zeta(two, a[0])),
        (g.TriGamma, g.PolyGamma): _rule(lambda a: polygamma(one, a[0])),
        (g.TriGamma, Harmonic): _rule(harmonic_trigamma),
        (zf.LerchPhi, zf.Zeta): lerchphi_helper(zf.Zeta), (zf.LerchPhi, zf.PolyLog): lerchphi_helper(zf.PolyLog),
        (zf.PolyLog, zf.LerchPhi): _rule(lambda a: a[1] * lerchphi(a[1], a[0], one)),
        (zf.Zeta, Bernoulli): _rule(zeta_bernoulli), (zf.Zeta, zf.DirichletEta): zeta_eta,
        (zf.Zeta, zf.LerchPhi): _rule(zeta_lerchphi),
        (zf.DirichletEta, Genocchi): _rule(eta_genocchi),
        (zf.RiemannXi, zf.Zeta): _rule(xi_zeta)}


# PORTED FROM SYMPY: sympy/functions/special/beta_functions.py:beta._eval_rewrite_as_Integral,
# betainc._eval_rewrite_as_<name>, betainc_regularized._eval_rewrite_as_<name>;
# sympy/functions/special/elliptic_integrals.py:elliptic_k._eval_rewrite_as_<name>,
# elliptic_f._eval_rewrite_as_Integral, elliptic_e._eval_rewrite_as_<name>, elliptic_pi._eval_rewrite_as_Integral;
# sympy/functions/special/bessel.py:marcumq._eval_rewrite_as_<name>
def _beta_elliptic_rules() -> dict[RuleKey, Rule]:
    """The beta functions and the elliptic integrals as integrals and
    hypergeometric functions; Marcum's Q-function."""
    from sympy_extras.core.numbers import oo
    from sympy_extras.core.power import sqrt
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.functions.elementary.exponential import exp
    from sympy_extras.functions.elementary.trigonometric import sin
    from sympy_extras.functions.special.bessel import BesselI, MarcumQ, besseli
    from sympy_extras.functions.special.hyper import hyper, meijerg
    from . import beta_functions as bf, elliptic_integrals as el
    from .beta_functions import beta
    one, zero = Integer(1), Integer(0)

    def node_args(e: Expr) -> list[Expr]:
        return [a for a in e.args if isinstance(a, Expr)]

    def beta_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        x = args[0]
        y = args[1] if len(args) > 1 else x
        t = _dummy('t', (x, y))
        return _integral(t**(x - 1) * (1 - t)**(y - 1), t, zero, one)

    def betainc_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        a, b, x1, x2 = args
        t = _dummy('t', args)
        return _integral(t**(a - 1) * (1 - t)**(b - 1), t, x1, x2)

    def betainc_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        a, b, x1, x2 = args
        return (x2**a * hyper([a, 1 - b], [a + 1], x2) - x1**a * hyper([a, 1 - b], [a + 1], x1)) / a

    def regularized_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        a, b, x1, x2 = args
        t = _dummy('t', args)
        integrand = t**(a - 1) * (1 - t)**(b - 1)
        return _integral(integrand, t, x1, x2) / _integral(integrand, t, zero, one)

    def regularized_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        a, b, x1, x2 = args
        expr = (x2**a * hyper([a, 1 - b], [a + 1], x2) - x1**a * hyper([a, 1 - b], [a + 1], x1)) / a
        return expr / beta(a, b)

    def k_integral(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        t = _dummy('t', args)
        m = node_args(e)[0]
        return _integral(1 / sqrt(1 - m * sin(t)**2), t, zero, pi / 2)

    def f_integral(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        t = _dummy('t', args)
        z, m = node_args(e)
        return _integral(1 / sqrt(1 - m * sin(t)**2), t, zero, z)

    def k_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        return pi * HALF * hyper([HALF, HALF], [one], args[0])

    def k_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        return meijerg([HALF, HALF], [], [zero], [zero], -args[0]) / 2

    def e_hyper(args: tuple[Expr, ...]) -> Optional[Expr]:
        if len(args) == 1:
            return (pi / 2) * hyper([Rational(-1, 2), HALF], [one], args[0])
        return None

    def e_meijerg(args: tuple[Expr, ...]) -> Optional[Expr]:
        if len(args) == 1:
            return -meijerg([HALF, Rational(3, 2)], [], [zero], [zero], -args[0]) / 4
        return None

    def e_integral(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        own = node_args(e)
        z, m = (pi / 2, own[0]) if len(own) == 1 else (own[0], own[1])
        t = _dummy('t', args)
        return _integral(sqrt(1 - m * sin(t)**2), t, zero, z)

    def pi_integral(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        own = node_args(e)
        if len(own) == 2:
            n, m, z = own[0], own[1], pi / 2
        else:
            n, z, m = own
        t = _dummy('t', args)
        return _integral(1 / ((1 - n * sin(t)**2) * sqrt(1 - m * sin(t)**2)), t, zero, z)

    def marcumq_integral(args: tuple[Expr, ...]) -> Optional[Expr]:
        m, a, b = args
        x = Dummy('x')
        return a**(1 - m) * _integral(x**m * exp(-(x**2 + a**2) / 2) * besseli(m - 1, a * x), x, b, oo)

    def marcumq_sum(args: tuple[Expr, ...]) -> Optional[Expr]:
        m, a, b = args
        k = Dummy('k')
        return exp(-(a**2 + b**2) / 2) * sum_((a / b)**k * besseli(k, a * b), (k, 1 - m, oo))

    def marcumq_besseli(args: tuple[Expr, ...]) -> Optional[Expr]:
        m, a, b = args
        if a == b:
            if m == 1:
                return (1 + exp(-a**2) * besseli(zero, a**2)) / 2
            if isinstance(m, Integer) and m.p >= 2:
                s = add(*[besseli(Integer(i), a**2) for i in range(1, m.p)])
                return HALF + exp(-a**2) * besseli(zero, a**2) / 2 + exp(-a**2) * s
        return None

    return {
        (bf.Beta, 'Integral'): _rule(beta_integral),
        (bf.BetaInc, 'Integral'): _rule(betainc_integral), (bf.BetaInc, 'hyper'): _rule(betainc_hyper),
        (bf.BetaIncRegularized, 'Integral'): _rule(regularized_integral),
        (bf.BetaIncRegularized, 'hyper'): _rule(regularized_hyper),
        (el.EllipticK, 'hyper'): _rule(k_hyper), (el.EllipticK, 'meijerg'): _rule(k_meijerg),
        (el.EllipticK, 'Integral'): k_integral, (el.EllipticF, 'Integral'): f_integral,
        (el.EllipticE, 'hyper'): _rule(e_hyper), (el.EllipticE, 'meijerg'): _rule(e_meijerg),
        (el.EllipticE, 'Integral'): e_integral, (el.EllipticPi, 'Integral'): pi_integral,
        (MarcumQ, 'Integral'): _rule(marcumq_integral), (MarcumQ, 'Sum'): _rule(marcumq_sum),
        (MarcumQ, BesselI): _rule(marcumq_besseli)}


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:hyper._eval_rewrite_as_Sum
def _hyper_sum(e: Basic, args: tuple[Basic, ...], hints: RewriteHints) -> Optional[Basic]:
    """The defining series where it converges, the node elsewhere."""
    from sympy_extras.core.mul import mul
    from sympy_extras.core.numbers import oo
    from sympy_extras.concrete.expr_with_limits import sum_
    from sympy_extras.functions.combinatorial.factorials import factorial, rf
    from sympy_extras.functions.elementary.piecewise import piecewise
    from .hyper import Hyper
    if not isinstance(e, Hyper):
        return None
    rebuilt = e._evaluated(args)
    if not isinstance(rebuilt, Hyper):
        return None
    n = Dummy('n', integer=True)
    coeff = mul(*[rf(a, n) for a in rebuilt.ap]) / mul(*[rf(b, n) for b in rebuilt.bq])
    return piecewise((sum_(coeff * rebuilt.argument**n / factorial(n), (n, Integer(0), oo)), e.convergence_statement),
                     (e, True))


# PORTED FROM SYMPY: sympy/functions/special/hyper.py:HyperRep._eval_rewrite_as_nonrep,
# HyperRep._eval_rewrite_as_nonrepsmall
def _hyperrep(small: bool) -> Rule:
    """The closed forms of the ``HyperRep`` functions (of the node itself,
    as SymPy's)."""
    def rule(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        from .hyperrep import HyperRep
        if not isinstance(e, HyperRep):
            return None
        return e.nonrepsmall() if small else e.nonrep()
    return rule


def node_rules() -> dict[RuleKey, NodeRule]:
    """The rules of the nodes whose arguments are not all expressions
    (``hyper``), keyed by (the class of the node, the target)."""
    from .hyper import Hyper
    return {(Hyper, 'Sum'): _hyper_sum}
