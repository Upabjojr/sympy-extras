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
values. Also the Hankel functions ``hankel1``, ``hankel2`` (no values of
their own, as in SymPy; their conjugates are each other), the spherical
Bessel functions ``jn``, ``yn`` (values at ``0`` and the infinities, the
derivative ``f(n - 1, z) - f(n, z)*(n + 1)/z``) and the Airy functions
``airyai``, ``airybi``, ``airyaiprime``, ``airybiprime`` (values at ``0``
and the infinities, the derivatives, SymPy's complex parts). Not ported:
the branch factors of polar arguments, the rewrites, expansions and
series.

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
# besseli, besselk, hankel1, hankel2, jn, yn, the Airy functions) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the polar branch factors, Hankel, spherical and Airy functions, rewrites
# and series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, ExpandHints
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import (I, Infinity, Integer, NegativeInfinity, Rational, nan, oo, pi, zoo)
from sympy_extras.core.power import pow

__all__ = ['BesselBase', 'BesselJ', 'BesselY', 'BesselI', 'BesselK', 'Hankel1', 'Hankel2', 'SphericalJ', 'SphericalY',
           'AiryAi', 'AiryBi', 'AiryAiPrime', 'AiryBiPrime', 'besselj', 'bessely', 'besseli', 'besselk', 'hankel1',
           'hankel2', 'jn', 'yn', 'airyai', 'airybi', 'airyaiprime', 'airybiprime', 'SphericalHankel1', 'SphericalHankel2',
           'MarcumQ', 'hn1', 'hn2', 'marcumq', 'jn_zeros']

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
# PORTED FROM SYMPY: sympy/functions/special/bessel.py:besselj.eval (branch handling)
def _branch(nu: Expr, z: Expr, f: Callable[[Expr, Expr], Expr]) -> Optional[Expr]:
    """The branch handling of besselj and besseli: an integer order loses
    the polar factors of the argument, another one takes out
    ``exp(2*n*pi*nu*I)`` for the branch factor ``exp_polar(2*pi*I*n)``;
    the order is unpolarified."""
    from sympy_extras.functions.elementary.complexes import unpolarify
    from sympy_extras.functions.elementary.exponential import exp
    if nu.is_integer is True:
        newz = unpolarify(z)
        if newz != z:
            return f(nu, newz)
    else:
        newz, n = z.extract_branch_factor()
        if n != 0:
            return mul(exp(mul(Integer(2), n, pi, nu, I)), f(nu, newz))
    nnu = unpolarify(nu)
    if nu != nnu:
        return f(nnu, z)
    return None


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
        return _branch(nu, z, besselj)

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
        return _branch(nu, z, besseli)

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


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:hankel1
class Hankel1(BesselBase):
    """The Hankel function of the first kind ``hankel1(nu, z)``,
    unevaluated (``hankel1`` evaluates; SymPy gives it no values)."""

    function_name = 'hankel1'
    generic_nseries = True
    real_on_real_axis = False

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        nu, z = self.args
        return hankel2(conjugate(nu), conjugate(z)) if z.is_extended_negative is False else None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[1] == 0 else mpmath.mpc(mpmath.hankel1(values[0], values[1]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:hankel2
class Hankel2(BesselBase):
    """The Hankel function of the second kind ``hankel2(nu, z)``,
    unevaluated (``hankel2`` evaluates; SymPy gives it no values)."""

    function_name = 'hankel2'
    generic_nseries = True
    real_on_real_axis = False

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        nu, z = self.args
        return hankel1(conjugate(nu), conjugate(z)) if z.is_extended_negative is False else None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[1] == 0 else mpmath.mpc(mpmath.hankel2(values[0], values[1]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:SphericalBesselBase
class SphericalBesselBase(BesselBase):
    """The derivative of the spherical Bessel functions,
    ``f(n - 1, z) - f(n, z)*(n + 1)/z``."""

    def fdiff(self, argindex: int) -> Expr:
        if argindex != 2:
            return DefinedFunction.fdiff(self, argindex)
        nu, z = self.args
        from sympy_extras.core.mul import div
        return sub(type(self).apply(sub(nu, ONE), z), div(mul(self, add(nu, ONE)), z))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def _spherical(kind: str, n: mpmath.mpc, z: mpmath.mpc) -> Optional[mpmath.mpc]:
    if z == 0:
        return None
    root = mpmath.sqrt(mpmath.pi/(2*z))
    if kind == 'j':
        return mpmath.mpc(root*mpmath.besselj(n + 0.5, z))
    return mpmath.mpc(root*mpmath.bessely(n + 0.5, z))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:jn
class SphericalJ(SphericalBesselBase):
    """The spherical Bessel function of the first kind ``jn(nu, z)``,
    unevaluated (``jn`` evaluates)."""

    function_name = 'jn'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        nu, z = args
        if z.is_zero:
            if nu.is_zero:
                return ONE
            if nu.is_integer:
                return Integer(0) if nu.is_positive else zoo
        if isinstance(z, (Infinity, NegativeInfinity)):
            return Integer(0)
        return None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return _spherical('j', values[0], values[1])


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:yn
class SphericalY(SphericalBesselBase):
    """The spherical Bessel function of the second kind ``yn(nu, z)``,
    unevaluated (``yn`` evaluates; SymPy gives it no values)."""

    function_name = 'yn'
    generic_nseries = True

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return _spherical('y', values[0], values[1])


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:AiryBase
class AiryBase(DefinedFunction):
    """The common methods of the Airy functions: conjugate, complex parts
    ``((f(z) + f(conj(z)))/2, I*(f(conj(z)) - f(z))/2)``, the real fact."""

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(conjugate(self.args[0]))

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from sympy_extras.core.mul import div
        from sympy_extras.functions.elementary.complexes import conjugate
        z = self.args[0]
        cls = type(self)
        f_z, f_conjugate = cls.apply(z), cls.apply(conjugate(z))
        return (div(add(f_z, f_conjugate), Integer(2)), div(mul(I, sub(f_conjugate, f_z)), Integer(2)))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        real = facts_of(self.args[0]).get('extended_real')
        if real is not None:
            facts['extended_real'] = real
        return facts


def _airy_at_zero(power: Rational, gamma_argument: Rational, numerator: Expr) -> Expr:
    """``numerator/(3**power*gamma(gamma_argument))`` (SymPy's values at 0)."""
    from sympy_extras.core.mul import div
    from .gamma_functions import gamma
    return div(numerator, mul(pow(Integer(3), power), gamma(gamma_argument)))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyai
class AiryAi(AiryBase):
    """The Airy function ``airyai(z)``, unevaluated (``airyai``
    evaluates)."""

    unbranched = True

    function_name = 'airyai'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        from sympy_extras.core.numbers import NaN
        if isinstance(z, NaN):
            return nan
        if isinstance(z, (Infinity, NegativeInfinity)):
            return Integer(0)
        if z.is_zero:
            return _airy_at_zero(Rational(2, 3), Rational(2, 3), ONE)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return airyaiprime(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.airyai(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybi
class AiryBi(AiryBase):
    """The Airy function ``airybi(z)``, unevaluated (``airybi``
    evaluates)."""

    unbranched = True

    function_name = 'airybi'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        from sympy_extras.core.numbers import NaN
        if isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return oo
        if isinstance(z, NegativeInfinity):
            return Integer(0)
        if z.is_zero:
            return _airy_at_zero(Rational(1, 6), Rational(2, 3), ONE)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return airybiprime(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.airybi(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airyaiprime
class AiryAiPrime(AiryBase):
    """The derivative ``airyaiprime(z)`` of ``airyai``, unevaluated
    (``airyaiprime`` evaluates)."""

    unbranched = True

    function_name = 'airyaiprime'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        from sympy_extras.core.numbers import NaN
        if isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return Integer(0)
        if z.is_zero:
            return _airy_at_zero(Rational(1, 3), Rational(1, 3), MINUS_ONE)
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(self.args[0], airyai(self.args[0]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.airyai(values[0], derivative=1))


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:airybiprime
class AiryBiPrime(AiryBase):
    """The derivative ``airybiprime(z)`` of ``airybi``, unevaluated
    (``airybiprime`` evaluates)."""

    unbranched = True

    function_name = 'airybiprime'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        from sympy_extras.core.mul import div
        from sympy_extras.core.numbers import NaN
        from .gamma_functions import gamma
        if isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return oo
        if isinstance(z, NegativeInfinity):
            return Integer(0)
        if z.is_zero:
            return div(pow(Integer(3), Rational(1, 6)), gamma(Rational(1, 3)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(self.args[0], airybi(self.args[0]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.airybi(values[0], derivative=1))


def hankel1(nu: ExprLike, z: ExprLike) -> Expr:
    """``hankel1(nu, z)``, evaluated."""
    return Hankel1.apply(to_expr(nu), to_expr(z))


def hankel2(nu: ExprLike, z: ExprLike) -> Expr:
    """``hankel2(nu, z)``, evaluated."""
    return Hankel2.apply(to_expr(nu), to_expr(z))


def jn(nu: ExprLike, z: ExprLike) -> Expr:
    """The spherical Bessel function ``jn(nu, z)``, evaluated."""
    return SphericalJ.apply(to_expr(nu), to_expr(z))


def yn(nu: ExprLike, z: ExprLike) -> Expr:
    """The spherical Bessel function ``yn(nu, z)``, evaluated."""
    return SphericalY.apply(to_expr(nu), to_expr(z))


def airyai(z: ExprLike) -> Expr:
    """``airyai(z)``, evaluated."""
    return AiryAi.apply(to_expr(z))


def airybi(z: ExprLike) -> Expr:
    """``airybi(z)``, evaluated."""
    return AiryBi.apply(to_expr(z))


def airyaiprime(z: ExprLike) -> Expr:
    """``airyaiprime(z)``, evaluated."""
    return AiryAiPrime.apply(to_expr(z))


def airybiprime(z: ExprLike) -> Expr:
    """``airybiprime(z)``, evaluated."""
    return AiryBiPrime.apply(to_expr(z))



# PORTED FROM SYMPY: sympy/functions/special/bessel.py:hn1
class SphericalHankel1(SphericalBesselBase):
    """The spherical Hankel function of the first kind ``hn1(nu, z) =
    jn(nu, z) + I*yn(nu, z)`` (no values of its own, as in SymPy)."""

    function_name = 'hn1'
    generic_nseries = True

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        j, y = _spherical('j', values[0], values[1]), _spherical('y', values[0], values[1])
        return None if j is None or y is None else mpmath.mpc(j + 1j * y)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:hn2
class SphericalHankel2(SphericalBesselBase):
    """The spherical Hankel function of the second kind ``hn2(nu, z) =
    jn(nu, z) - I*yn(nu, z)`` (no values of its own, as in SymPy)."""

    function_name = 'hn2'
    generic_nseries = True

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        j, y = _spherical('j', values[0], values[1]), _spherical('y', values[0], values[1])
        return None if j is None or y is None else mpmath.mpc(j - 1j * y)


# PORTED FROM SYMPY: sympy/functions/special/bessel.py:marcumq
class MarcumQ(DefinedFunction):
    """The Marcum Q-function ``marcumq(m, a, b)``, unevaluated
    (``marcumq`` evaluates)."""

    function_name = 'marcumq'
    generic_nseries = True
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.mul import div
        from sympy_extras.functions.elementary.exponential import exp
        from .gamma_functions import gamma, uppergamma
        m, a, b = args
        zero = Integer(0)
        if a.is_zero:
            if m.is_zero and b.is_zero:
                return zero
            return div(uppergamma(m, mul(pow(b, Integer(2)), HALF)), gamma(m))
        if m.is_zero and b.is_zero:
            return sub(ONE, div(ONE, exp(mul(pow(a, Integer(2)), HALF))))
        if a == b:
            decay = exp(mul(MINUS_ONE, pow(a, Integer(2))))
            if m == ONE:
                return mul(add(ONE, mul(decay, besseli(Integer(0), pow(a, Integer(2))))), HALF)
            if m == Integer(2):
                return add(HALF, mul(HALF, decay, besseli(Integer(0), pow(a, Integer(2)))),
                           mul(decay, besseli(ONE, pow(a, Integer(2)))))
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.core.mul import div
        from sympy_extras.functions.elementary.exponential import exp
        m, a, b = self.args
        if argindex == 2:
            return mul(a, add(mul(MINUS_ONE, marcumq(m, a, b)), marcumq(add(ONE, m), a, b)))
        if argindex == 3:
            return mul(div(mul(MINUS_ONE, pow(b, m)), pow(a, sub(m, ONE))),
                       exp(div(mul(MINUS_ONE, add(pow(a, Integer(2)), pow(b, Integer(2)))), Integer(2))),
                       besseli(sub(m, ONE), mul(a, b)))
        return DefinedFunction.fdiff(self, argindex)

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        m, a, b = self.args
        if a.is_zero and m.is_zero and b.is_zero:
            facts['zero'] = True
        return facts


def hn1(nu: ExprLike, z: ExprLike) -> Expr:
    """``hn1(nu, z)``."""
    return SphericalHankel1.apply(to_expr(nu), to_expr(z))


def hn2(nu: ExprLike, z: ExprLike) -> Expr:
    """``hn2(nu, z)``."""
    return SphericalHankel2.apply(to_expr(nu), to_expr(z))


def marcumq(m: ExprLike, a: ExprLike, b: ExprLike) -> Expr:
    """``marcumq(m, a, b)``, evaluated."""
    return MarcumQ.apply(to_expr(m), to_expr(a), to_expr(b))



# PORTED FROM SYMPY: sympy/functions/special/bessel.py:jn_zeros (method "sympy")
def jn_zeros(n: int, k: int, dps: int = 15) -> list[Expr]:
    """The first ``k`` zeros of the spherical Bessel function ``jn(n, x)``,
    floats at ``dps`` digits (mpmath's ``besseljzero`` of order ``n + 1/2``)."""
    from sympy_extras.core.numbers import Float, dps_to_prec
    prec = dps_to_prec(dps)
    zeros: list[Expr] = []
    with mpmath.workprec(prec):
        for index in range(1, k + 1):
            zeros.append(Float(str(mpmath.besseljzero(mpmath.mpf(n) + mpmath.mpf(0.5), index)), dps=dps))
    return zeros
