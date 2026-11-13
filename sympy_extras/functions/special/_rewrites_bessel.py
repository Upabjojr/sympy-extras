"""The ``rewrite`` rules of the Bessel, spherical Bessel and Airy functions
and of the spherical harmonics between themselves (SymPy's
``_eval_rewrite_as_<name>`` methods of ``bessel.py`` and
``spherical_harmonics.py``), as a typed table for
:func:`sympy_extras.core.rewriting.rewrite`. Ported from SymPy's
``sympy.functions.special.bessel`` and
``sympy.functions.special.spherical_harmonics``. A rule returns ``None``
where SymPy's leaves the node as it is.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/bessel.py, sympy/functions/special/spherical_harmonics.py
# (the _eval_rewrite_as_<name> methods of besselj, bessely, besseli, besselk, jn, yn, hn1, hn2,
# airyai, airybi, airyaiprime, airybiprime and Ynm) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target) in place of methods; hn1, hn2 as hankel1, hankel2 of
# order nu + 1/2 and the order -2/3 in airyaiprime as besseli off the right half plane (SymPy's are bugs).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Optional

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.rewriting import Rule, RuleKey, RewriteHints, RewriteTarget, rewrite

__all__ = ['rules']

HALF = Rational(1, 2)
THIRD = Rational(1, 3)
TWO_THIRDS = Rational(2, 3)
MINUS_ONE = Integer(-1)


def _rewritten(e: Optional[Expr], target: RewriteTarget, hints: RewriteHints) -> Optional[Expr]:
    """SymPy's ``e.rewrite(target)`` of the intermediate form of a rule,
    ``None`` kept."""
    if e is None:
        return None
    r = rewrite(e, (), target, hints)
    assert isinstance(r, Expr)
    return r


def _order_arg(node: Expr) -> tuple[Expr, Expr]:
    """The order and the argument of the node itself (SymPy's
    ``self.args``, not the rewritten ones)."""
    nu, z = node.args
    assert isinstance(nu, Expr) and isinstance(z, Expr)
    return nu, z


# -- the Bessel functions

# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselj._eval_rewrite_as_bessely
def _besselj_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.trigonometric import cot, csc
    from .bessel import bessely
    nu, z = args
    if nu.is_integer is False:
        return csc(pi * nu) * bessely(-nu, z) - cot(pi * nu) * bessely(nu, z)
    return None


def _bessely_as_besselj(nu: Expr, z: Expr) -> Optional[Expr]:
    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:bessely._eval_rewrite_as_besselj
    from sympy_extras.functions.elementary.trigonometric import cos, csc
    from .bessel import besselj
    if nu.is_integer is False:
        return csc(pi * nu) * (cos(pi * nu) * besselj(nu, z) - besselj(-nu, z))
    return None


def _besseli_as_besselj(nu: Expr, z: Expr) -> Expr:
    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_rewrite_as_besselj
    from sympy_extras.functions.elementary.complexes import polar_lift
    from sympy_extras.functions.elementary.exponential import exp
    from .bessel import besselj
    return exp(-I * pi * nu / 2) * besselj(nu, polar_lift(I) * z)


def _besselk_as_besseli(nu: Expr, z: Expr) -> Optional[Expr]:
    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_rewrite_as_besseli
    from sympy_extras.functions.elementary.trigonometric import csc
    from .bessel import besseli
    if nu.is_integer is False:
        return pi * csc(pi * nu) * (besseli(-nu, z) - besseli(nu, z)) / 2
    return None


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:bessely._eval_rewrite_as_besseli
def _bessely_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import BesselI
    return _rewritten(_bessely_as_besselj(*_order_arg(e)), BesselI, hints)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:bessely._eval_rewrite_as_yn
def _bessely_yn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import yn
    nu, z = args
    return sqrt(2 * z / pi) * yn(nu - HALF, _order_arg(e)[1])


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_rewrite_as_bessely
def _besseli_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import BesselY
    return _rewritten(_besseli_as_besselj(*_order_arg(e)), BesselY, hints)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli._eval_rewrite_as_jn
def _besseli_jn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import SphericalJ
    return _rewritten(_besseli_as_besselj(*_order_arg(e)), SphericalJ, hints)


def _besselk_as_bessely(e: Expr, hints: RewriteHints) -> Optional[Expr]:
    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_rewrite_as_besselj and
    # besselk._eval_rewrite_as_bessely
    from .bessel import BesselJ, BesselY
    return _rewritten(_rewritten(_besselk_as_besseli(*_order_arg(e)), BesselJ, hints), BesselY, hints)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_rewrite_as_bessely
def _besselk_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _besselk_as_bessely(e, hints)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk._eval_rewrite_as_yn
def _besselk_yn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import SphericalY
    return _rewritten(_besselk_as_bessely(e, hints), SphericalY, hints)


# -- the spherical Bessel functions

def _integer_order(rule: Rule) -> Rule:
    """SymPy's ``assume_integer_order``: the rule only for an order known
    to be an integer."""
    def ruled(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        return rule(e, args, hints) if args[0].is_integer is True else None
    return ruled


def _half_pi_over(z: Expr) -> Expr:
    return sqrt(pi / (2 * z))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:jn._eval_rewrite_as_besselj
def _jn_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    nu, z = args
    return _half_pi_over(z) * besselj(nu + HALF, z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:jn._eval_rewrite_as_bessely
def _jn_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import bessely
    nu, z = args
    return pow(MINUS_ONE, nu) * _half_pi_over(z) * bessely(-nu - HALF, z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:jn._eval_rewrite_as_yn
def _jn_yn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import yn
    nu, z = args
    return pow(MINUS_ONE, nu) * yn(-nu - 1, z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:yn._eval_rewrite_as_besselj
def _yn_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    nu, z = args
    return pow(MINUS_ONE, nu + 1) * _half_pi_over(z) * besselj(-nu - HALF, z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:yn._eval_rewrite_as_bessely
def _yn_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import bessely
    nu, z = args
    return _half_pi_over(z) * bessely(nu + HALF, z)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:yn._eval_rewrite_as_jn
def _yn_jn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import jn
    nu, z = args
    return pow(MINUS_ONE, nu + 1) * jn(-nu - 1, z)


def _hankel_rules(cls: type, sign: Expr, hankel: type) -> dict[RuleKey, Rule]:
    """The rules of ``hn1`` (``sign`` 1) or ``hn2`` (``sign`` -1)."""
    from .bessel import SphericalJ, SphericalY, besselj, bessely, jn, yn

    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalHankelBase._eval_rewrite_as_besselj
    def as_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        nu, z = args
        return _half_pi_over(z) * (besselj(nu + HALF, z) + sign * I * pow(MINUS_ONE, nu + 1) * besselj(-nu - HALF, z))

    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalHankelBase._eval_rewrite_as_bessely
    def as_bessely(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        nu, z = args
        return _half_pi_over(z) * (pow(MINUS_ONE, nu) * bessely(-nu - HALF, z) + sign * I * bessely(nu + HALF, z))

    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalHankelBase._eval_rewrite_as_yn
    def as_yn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        nu, z = args
        j = _rewritten(jn(nu, z), SphericalY, hints)
        assert j is not None
        return j + sign * I * yn(nu, z)

    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalHankelBase._eval_rewrite_as_jn
    def as_jn(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        nu, z = args
        y = _rewritten(yn(nu, z), SphericalJ, hints)
        assert y is not None
        return jn(nu, z) + sign * I * y

    # PORTED FROM SYMPY: sympy/functions/special/bessel.py:hn1._eval_rewrite_as_hankel1,
    # hn2._eval_rewrite_as_hankel2 (of order nu + 1/2: SymPy's order nu is wrong, hn1(0, z) is
    # -I*exp(I*z)/z, not sqrt(pi/(2*z))*hankel1(0, z))
    def as_hankel(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
        from .bessel import hankel1, hankel2
        nu, z = args
        h = hankel1 if sign == 1 else hankel2
        return _half_pi_over(z) * h(nu + HALF, z)

    from .bessel import BesselJ, BesselY
    return {(cls, BesselJ): _integer_order(as_besselj), (cls, BesselY): _integer_order(as_bessely),
            (cls, SphericalY): as_yn, (cls, SphericalJ): as_jn, (cls, hankel): _integer_order(as_hankel)}


# -- the Airy functions

def _negative_re(z: Expr) -> bool:
    from sympy_extras.functions.elementary.complexes import re
    return re(z).is_negative is True


def _positive_re(z: Expr) -> bool:
    from sympy_extras.functions.elementary.complexes import re
    return re(z).is_positive is True


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai._eval_rewrite_as_besselj
def _airyai_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    z = args[0]
    a = pow(-z, Rational(3, 2))
    if _negative_re(z):
        return THIRD * sqrt(-z) * (besselj(-THIRD, TWO_THIRDS * a) + besselj(THIRD, TWO_THIRDS * a))
    return None


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai._eval_rewrite_as_besseli
def _airyai_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besseli
    z = args[0]
    a = pow(z, Rational(3, 2))
    if _positive_re(z):
        return THIRD * sqrt(z) * (besseli(-THIRD, TWO_THIRDS * a) - besseli(THIRD, TWO_THIRDS * a))
    return THIRD * (pow(a, THIRD) * besseli(-THIRD, TWO_THIRDS * a) - z * pow(a, -THIRD) * besseli(THIRD, TWO_THIRDS * a))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai._eval_rewrite_as_hyper
def _airyai_hyper(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.miscellaneous import root
    from .gamma_functions import gamma
    from .hyper import hyper
    z = args[0]
    pf1 = 1 / (pow(Integer(3), TWO_THIRDS) * gamma(TWO_THIRDS))
    pf2 = z / (root(Integer(3), Integer(3)) * gamma(THIRD))
    return pf1 * hyper([], [TWO_THIRDS], z**3 / 9) - pf2 * hyper([], [Rational(4, 3)], z**3 / 9)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybi._eval_rewrite_as_besselj
def _airybi_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    z = args[0]
    a = pow(-z, Rational(3, 2))
    if _negative_re(z):
        return sqrt(-z / 3) * (besselj(-THIRD, TWO_THIRDS * a) - besselj(THIRD, TWO_THIRDS * a))
    return None


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybi._eval_rewrite_as_besseli
def _airybi_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besseli
    z = args[0]
    a = pow(z, Rational(3, 2))
    if _positive_re(z):
        return sqrt(z) / sqrt(Integer(3)) * (besseli(-THIRD, TWO_THIRDS * a) + besseli(THIRD, TWO_THIRDS * a))
    b = pow(a, THIRD)
    c = pow(a, -THIRD)
    return sqrt(THIRD) * (b * besseli(-THIRD, TWO_THIRDS * a) + z * c * besseli(THIRD, TWO_THIRDS * a))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybi._eval_rewrite_as_hyper
def _airybi_hyper(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.miscellaneous import root
    from .gamma_functions import gamma
    from .hyper import hyper
    z = args[0]
    pf1 = 1 / (root(Integer(3), Integer(6)) * gamma(TWO_THIRDS))
    pf2 = z * root(Integer(3), Integer(6)) / gamma(THIRD)
    return pf1 * hyper([], [TWO_THIRDS], z**3 / 9) + pf2 * hyper([], [Rational(4, 3)], z**3 / 9)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyaiprime._eval_rewrite_as_besselj
def _airyaiprime_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    z = args[0]
    a = pow(-z, Rational(3, 2))
    if _negative_re(z):
        return z / 3 * (besselj(-TWO_THIRDS, TWO_THIRDS * a) - besselj(TWO_THIRDS, TWO_THIRDS * a))
    return None


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyaiprime._eval_rewrite_as_besseli
def _airyaiprime_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besseli
    z = args[0]
    if _positive_re(z):
        a = TWO_THIRDS * pow(z, Rational(3, 2))
        return z / 3 * (besseli(TWO_THIRDS, a) - besseli(-TWO_THIRDS, a))
    p = pow(z, Rational(3, 2))
    b = pow(p, TWO_THIRDS)
    c = pow(p, -TWO_THIRDS)
    # SymPy has besseli(-1/3, ...) in the second term, wrong (airyaiprime(-7/10)
    # is -0.1446..., SymPy's form gives -0.1577 + 0.1401*I): the order is -2/3
    return THIRD * (z**2 * c * besseli(TWO_THIRDS, TWO_THIRDS * p) - b * besseli(-TWO_THIRDS, TWO_THIRDS * p))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyaiprime._eval_rewrite_as_hyper
def _airyaiprime_hyper(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.miscellaneous import root
    from .gamma_functions import gamma
    from .hyper import hyper
    z = args[0]
    pf1 = z**2 / (2 * pow(Integer(3), TWO_THIRDS) * gamma(TWO_THIRDS))
    pf2 = 1 / (root(Integer(3), Integer(3)) * gamma(THIRD))
    return pf1 * hyper([], [Rational(5, 3)], z**3 / 9) - pf2 * hyper([], [THIRD], z**3 / 9)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybiprime._eval_rewrite_as_besselj
def _airybiprime_besselj(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besselj
    z = args[0]
    a = TWO_THIRDS * pow(-z, Rational(3, 2))
    if _negative_re(z):
        return -z / sqrt(Integer(3)) * (besselj(-TWO_THIRDS, a) + besselj(TWO_THIRDS, a))
    return None


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybiprime._eval_rewrite_as_besseli
def _airybiprime_besseli(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from .bessel import besseli
    z = args[0]
    if _positive_re(z):
        a = TWO_THIRDS * pow(z, Rational(3, 2))
        return z / sqrt(Integer(3)) * (besseli(-TWO_THIRDS, a) + besseli(TWO_THIRDS, a))
    p = pow(z, Rational(3, 2))
    b = pow(p, TWO_THIRDS)
    c = pow(p, -TWO_THIRDS)
    return sqrt(THIRD) * (b * besseli(-TWO_THIRDS, TWO_THIRDS * p) + z**2 * c * besseli(TWO_THIRDS, TWO_THIRDS * p))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybiprime._eval_rewrite_as_hyper
def _airybiprime_hyper(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.miscellaneous import root
    from .gamma_functions import gamma
    from .hyper import hyper
    z = args[0]
    pf1 = z**2 / (2 * root(Integer(3), Integer(6)) * gamma(TWO_THIRDS))
    pf2 = root(Integer(3), Integer(6)) / gamma(THIRD)
    return pf1 * hyper([], [Rational(5, 3)], z**3 / 9) + pf2 * hyper([], [THIRD], z**3 / 9)


# -- the spherical harmonics

def _ynm_expanded(e: Expr) -> Expr:
    """SymPy's ``self.expand(func=True)`` of ``Ynm``."""
    from sympy_extras.core.function import expand
    from ._expand_func_bessel import ynm_expand_func
    return expand(ynm_expand_func(e), func=True)


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm._eval_rewrite_as_polynomial
def _ynm_polynomial(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _ynm_expanded(e)


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm._eval_rewrite_as_cos
def _ynm_cos(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    from sympy_extras.functions.elementary.complexes import Abs
    from sympy_extras.functions.elementary.trigonometric import sin
    from sympy_extras.simplify.simplify import simplify
    from sympy_extras.simplify.trigsimp import trigsimp
    theta = e.args[2]
    assert isinstance(theta, Expr)
    term = simplify(_ynm_expanded(e))
    cases: dict[Expr, Expr] = {Abs(sin(theta)): sin(theta)}
    replaced = term.xreplace(cases)
    assert isinstance(replaced, Expr)
    return simplify(trigsimp(replaced))


# PORTED FROM SYMPY: sympy/functions/special/spherical_harmonics.py:Ynm._eval_rewrite_as_sin
def _ynm_sin(e: Expr, args: tuple[Expr, ...], hints: RewriteHints) -> Optional[Expr]:
    return _ynm_cos(e, args, hints)


def rules() -> dict[RuleKey, Rule]:
    """The rules keyed by (the class of the node, the target class or
    SymPy name)."""
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin
    from . import bessel as b
    from .spherical_harmonics import YnmNode
    table: dict[RuleKey, Rule] = {
        (b.BesselJ, b.BesselY): _besselj_bessely, (b.BesselY, b.BesselI): _bessely_besseli,
        (b.BesselY, b.SphericalY): _bessely_yn, (b.BesselI, b.BesselY): _besseli_bessely,
        (b.BesselI, b.SphericalJ): _besseli_jn, (b.BesselK, b.BesselY): _besselk_bessely,
        (b.BesselK, b.SphericalY): _besselk_yn,
        (b.SphericalJ, b.BesselJ): _jn_besselj, (b.SphericalJ, b.BesselY): _jn_bessely,
        (b.SphericalJ, b.SphericalY): _jn_yn,
        (b.SphericalY, b.BesselJ): _integer_order(_yn_besselj), (b.SphericalY, b.BesselY): _integer_order(_yn_bessely),
        (b.SphericalY, b.SphericalJ): _yn_jn,
        (b.AiryAi, b.BesselJ): _airyai_besselj, (b.AiryAi, b.BesselI): _airyai_besseli, (b.AiryAi, 'hyper'): _airyai_hyper,
        (b.AiryBi, b.BesselJ): _airybi_besselj, (b.AiryBi, b.BesselI): _airybi_besseli, (b.AiryBi, 'hyper'): _airybi_hyper,
        (b.AiryAiPrime, b.BesselJ): _airyaiprime_besselj, (b.AiryAiPrime, b.BesselI): _airyaiprime_besseli,
        (b.AiryAiPrime, 'hyper'): _airyaiprime_hyper,
        (b.AiryBiPrime, b.BesselJ): _airybiprime_besselj, (b.AiryBiPrime, b.BesselI): _airybiprime_besseli,
        (b.AiryBiPrime, 'hyper'): _airybiprime_hyper,
        (YnmNode, 'polynomial'): _ynm_polynomial, (YnmNode, Cos): _ynm_cos, (YnmNode, Sin): _ynm_sin}
    table.update(_hankel_rules(b.SphericalHankel1, Integer(1), b.Hankel1))
    table.update(_hankel_rules(b.SphericalHankel2, MINUS_ONE, b.Hankel2))
    return table
