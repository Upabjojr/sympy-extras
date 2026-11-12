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
# expint._eval_rewrite_as_Ei, Ei._eval_rewrite_as_expint), sympy/functions/special/
# zeta_functions.py (dirichlet_eta._eval_rewrite_as_zeta) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; a table keyed by (class, target class).
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Optional

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import I, Integer, Rational, pi
from sympy_extras.core.rewriting import RewriteHints

__all__ = ['rules']

Rule = Callable[[Expr, tuple[Expr, ...], RewriteHints], Optional[Expr]]


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


def rules() -> dict[tuple[type, type], Rule]:
    """The rules keyed by (the class of the node, the target class)."""
    from . import bessel as b, error_functions as er, zeta_functions as z
    return {(b.BesselJ, b.SphericalJ): _besselj_jn, (b.BesselJ, b.BesselI): _besselj_besseli,
            (b.BesselI, b.BesselJ): _besseli_besselj, (b.BesselY, b.BesselJ): _bessely_besselj,
            (b.BesselK, b.BesselI): _besselk_besseli, (b.BesselK, b.BesselJ): _besselk_besselj,
            (er.Erfc, er.Erf): _erfc_erf, (er.Erfi, er.Erf): _erfi_erf, (er.Erf2, er.Erf): _erf2_erf,
            (er.Expint, er.EiNode): _expint_ei, (er.EiNode, er.Expint): _ei_expint,
            (z.DirichletEta, z.Zeta): _eta_zeta}
