"""The Bessel functions on the typed core: ``BesselJ``, ``BesselY``,
``BesselI``, ``BesselK`` (unevaluated nodes) and ``besselj``,
``bessely``, ``besseli``, ``besselk`` (evaluating).

Ported from SymPy's ``sympy.functions.special.bessel``: the values at
``z = 0`` (by the order), at the infinities (and the imaginary ones for
``bessely``, ``besseli``, ``besselk``), a minus sign taken out of the
argument (``besselj(nu, -z) = z**nu*(-z)**(-nu)*besselj(nu, z)``) or of
an integer order (``besselj(-n, z) = (-1)**n*besselj(n, z)``), ``I``
taken out of the argument for an integer order (``besselj(n, I*z) =
I**n*besseli(n, z)``); the derivative in the argument by the recurrence,
the conjugate off the negative axis, the extended-real facts and mpmath
values. Not ported: the branch factors of polar arguments, the Hankel,
spherical (``jn``, ``yn``) and Airy functions, the rewrites, expansions
and series.

Examples
========

>>> from sympy_extras.core import Symbol, I, oo
>>> from sympy_extras.functions.special.bessel import besselj, bessely, besseli, besselk, BesselJ
>>> x, n = Symbol('x'), Symbol('n', integer=True)
>>> besselj(0, 0), besselj(2, 0), bessely(0, 0), besseli(0, 0), besselk(0, 0), besselj(1, oo)
(1, 0, -oo, 1, oo, 0)
>>> besselj(-2, x), besselj(n, I*x), besseli(-n, x), besselj(n, x).diff(x), BesselJ(0, 0)
(besselj(2, x), I**n*besseli(n, x), besseli(n, x), besselj(n - 1, x)/2 - besselj(n + 1, x)/2, besselj(0, 0))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/bessel.py (BesselBase, besselj, bessely,
# besseli, besselk) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the polar branch factors, Hankel, spherical and Airy functions, rewrites
# and series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import (I, Infinity, Integer, NegativeInfinity, Rational, nan, oo, pi, zoo)
from sympy_extras.core.power import pow

__all__ = ['BesselBase', 'BesselJ', 'BesselY', 'BesselI', 'BesselK', 'besselj', 'bessely', 'besseli', 'besselk']

ONE = Integer(1)
MINUS_ONE = Integer(-1)
HALF = Rational(1, 2)


def _at_zero_regular(nu: Expr, one: Expr) -> Optional[Expr]:
    """SymPy's values of ``besselj`` and ``besseli`` at ``z = 0``."""
    from sympy_extras.functions.elementary.complexes import re
    if nu.is_zero:
        return one
    if (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
        return Integer(0)
    if re(nu).is_negative and nu.is_integer is not True:
        return zoo
    if nu.is_imaginary:
        return nan
    return None


def _at_zero_singular(nu: Expr, at_zero_order: Expr) -> Optional[Expr]:
    """SymPy's values of ``bessely`` and ``besselk`` at ``z = 0``."""
    from sympy_extras.functions.elementary.complexes import re
    if nu.is_zero:
        return at_zero_order
    if re(nu).is_zero is False:
        return zoo
    if re(nu).is_zero:
        return nan
    return None


def _rotated(z: Expr) -> Optional[Expr]:
    """``z/I`` when ``I`` can be taken out of ``z`` and the rest is not
    zero (SymPy's ``if newz:``)."""
    rotated = z.extract_multiplicatively(I)
    if rotated is None or rotated.is_zero:
        return None
    return rotated


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:BesselBase
class BesselBase(DefinedFunction):
    """The common methods of the Bessel functions ``f(nu, z)``: the
    derivative ``b/2*f(nu - 1, z) - a/2*f(nu + 1, z)`` and the conjugate."""

    nargs = frozenset({2})
    call_nargs = frozenset({2})
    #: the coefficients of the recurrence of the derivative (SymPy's _a, _b)
    a_coefficient: int = 1
    b_coefficient: int = 1

    @property
    def order(self) -> Expr:
        return self.args[0]

    @property
    def argument(self) -> Expr:
        return self.args[1]

    def fdiff(self, argindex: int) -> Expr:
        if argindex != 2:
            return DefinedFunction.fdiff(self, argindex)
        nu, z = self.args
        cls = type(self)
        return sub(mul(Rational(self.b_coefficient, 2), cls.apply(sub(nu, ONE), z)),
                   mul(Rational(self.a_coefficient, 2), cls.apply(add(nu, ONE), z)))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        nu, z = self.args
        if z.is_extended_negative is False:
            return type(self).apply(conjugate(nu), conjugate(z))
        return None

    #: whether the value is real for an integer order and a real argument
    #: (besselj, besseli) rather than a positive one (bessely, besselk)
    real_on_real_axis: bool = True

    def _eval_facts(self) -> dict[str, bool]:
        nu, z = self.args
        facts: dict[str, bool] = {'commutative': True}
        argument = facts_of(z)
        if facts_of(nu).get('integer') and argument.get('extended_real' if self.real_on_real_axis else 'positive'):
            facts['extended_real'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselj
class BesselJ(BesselBase):
    """The Bessel function of the first kind ``besselj(nu, z)``,
    unevaluated (``besselj`` evaluates)."""

    function_name = 'besselj'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        nu, z = args
        if z.is_zero:
            value = _at_zero_regular(nu, ONE)
            if value is not None:
                return value
        if isinstance(z, (Infinity, NegativeInfinity)):
            return Integer(0)
        if z.could_extract_minus_sign():
            minus_z = mul(MINUS_ONE, z)
            return mul(pow(z, nu), pow(minus_z, mul(MINUS_ONE, nu)), besselj(nu, minus_z))
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                minus_nu = mul(MINUS_ONE, nu)
                return mul(pow(MINUS_ONE, minus_nu), besselj(minus_nu, z))
            rotated = _rotated(z)
            if rotated is not None:
                return mul(pow(I, nu), besseli(nu, rotated))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.besselj(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:bessely
class BesselY(BesselBase):
    """The Bessel function of the second kind ``bessely(nu, z)``,
    unevaluated (``bessely`` evaluates)."""

    function_name = 'bessely'
    real_on_real_axis = False

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.exponential import exp
        nu, z = args
        if z.is_zero:
            value = _at_zero_singular(nu, NegativeInfinity())
            if value is not None:
                return value
        if isinstance(z, (Infinity, NegativeInfinity)):
            return Integer(0)
        if z == mul(I, oo):
            return mul(exp(mul(I, pi, add(nu, ONE), HALF)), oo)
        if z == mul(I, NegativeInfinity()):
            return mul(exp(mul(MINUS_ONE, I, pi, add(nu, ONE), HALF)), oo)
        if nu.is_integer and nu.could_extract_minus_sign():
            minus_nu = mul(MINUS_ONE, nu)
            return mul(pow(MINUS_ONE, minus_nu), bessely(minus_nu, z))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[1] == 0:
            return None
        return mpmath.mpc(mpmath.bessely(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besseli
class BesselI(BesselBase):
    """The modified Bessel function of the first kind ``besseli(nu, z)``,
    unevaluated (``besseli`` evaluates)."""

    function_name = 'besseli'
    a_coefficient = -1

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import im
        nu, z = args
        if z.is_zero:
            value = _at_zero_regular(nu, ONE)
            if value is not None:
                return value
        if isinstance(im(z), (Infinity, NegativeInfinity)):
            return Integer(0)
        if isinstance(z, Infinity):
            return oo
        if isinstance(z, NegativeInfinity):
            return mul(pow(MINUS_ONE, nu), oo)
        if z.could_extract_minus_sign():
            minus_z = mul(MINUS_ONE, z)
            return mul(pow(z, nu), pow(minus_z, mul(MINUS_ONE, nu)), besseli(nu, minus_z))
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(mul(MINUS_ONE, nu), z)
            rotated = _rotated(z)
            if rotated is not None:
                return mul(pow(I, mul(MINUS_ONE, nu)), besselj(nu, mul(MINUS_ONE, rotated)))
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.besseli(values[0], values[1]))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselk
class BesselK(BesselBase):
    """The modified Bessel function of the second kind ``besselk(nu, z)``,
    unevaluated (``besselk`` evaluates)."""

    function_name = 'besselk'
    b_coefficient = -1
    real_on_real_axis = False

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        nu, z = args
        if z.is_zero:
            value = _at_zero_singular(nu, oo)
            if value is not None:
                return value
        if isinstance(z, Infinity) or z == mul(I, oo) or z == mul(I, NegativeInfinity()):
            return Integer(0)
        if nu.is_integer and nu.could_extract_minus_sign():
            return besselk(mul(MINUS_ONE, nu), z)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[1] == 0:
            return None
        return mpmath.mpc(mpmath.besselk(values[0], values[1]))


def besselj(nu: ExprLike, z: ExprLike) -> Expr:
    """``besselj(nu, z)``, evaluated."""
    return BesselJ.apply(to_expr(nu), to_expr(z))


def bessely(nu: ExprLike, z: ExprLike) -> Expr:
    """``bessely(nu, z)``, evaluated."""
    return BesselY.apply(to_expr(nu), to_expr(z))


def besseli(nu: ExprLike, z: ExprLike) -> Expr:
    """``besseli(nu, z)``, evaluated."""
    return BesselI.apply(to_expr(nu), to_expr(z))


def besselk(nu: ExprLike, z: ExprLike) -> Expr:
    """``besselk(nu, z)``, evaluated."""
    return BesselK.apply(to_expr(nu), to_expr(z))
