"""The elliptic integrals on the typed core: ``EllipticK``,
``EllipticF``, ``EllipticE``, ``EllipticPi`` (unevaluated nodes) and
``elliptic_k``, ``elliptic_f``, ``elliptic_e``, ``elliptic_pi``
(evaluating).

Ported from SymPy's ``sympy.functions.special.elliptic_integrals``: the
special values (``elliptic_k(0) = pi/2``, ``elliptic_k(1) = zoo``, the
gamma forms at ``1/2`` and ``-1``, the infinities), the incomplete
integrals at ``z = 0``, ``m = 0``, the multiples of ``pi/2`` reduced to
the complete ones, the minus sign of ``z`` taken out, ``elliptic_pi``'s
cases ``n = 0``, ``1``, ``m``; the derivatives, conjugates off the cut
``m > 1``, the zero facts and mpmath values. Not ported: the series and
the rewrites.

Examples
========

>>> from sympy_extras.core import Symbol, pi, Rational
>>> from sympy_extras.functions.special.elliptic_integrals import elliptic_k, elliptic_f, elliptic_e, elliptic_pi
>>> z, m, n = Symbol('z'), Symbol('m'), Symbol('n')
>>> elliptic_k(0), elliptic_k(1), elliptic_e(0), elliptic_e(1), elliptic_f(z, 0), elliptic_f(pi, m), elliptic_e(-z, m)
(pi/2, zoo, pi/2, 1, z, 2*elliptic_k(m), -elliptic_e(z, m))
>>> elliptic_pi(0, m), elliptic_pi(n, 0), elliptic_k(m).diff(m)
(elliptic_k(m), pi/(2*sqrt(1 - n)), (-(1 - m)*elliptic_k(m) + elliptic_e(m))/(2*m*(1 - m)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the series and rewrites not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import (ComplexInfinity, I, Infinity, Integer, NegativeInfinity, Rational, oo, pi,
                                       zoo)
from sympy_extras.core.power import pow, sqrt

__all__ = ['EllipticK', 'EllipticF', 'EllipticE', 'EllipticPi', 'elliptic_k', 'elliptic_f', 'elliptic_e',
           'elliptic_pi']

ONE = Integer(1)
ZERO = Integer(0)
MINUS_ONE = Integer(-1)
HALF = Rational(1, 2)


def _python_and(first: Optional[bool], second: Optional[bool]) -> Optional[bool]:
    """Python's ``first and second`` on SymPy's three-valued facts, as
    SymPy's conjugate conditions are written."""
    return second if first else first


def _off_cut(m: Expr) -> bool:
    """SymPy's ``(m.is_real and (m - 1).is_positive) is False``."""
    return _python_and(m.is_real, sub(m, ONE).is_positive) is False


def _is_real_infinity(e: Expr) -> bool:
    return isinstance(e, (Infinity, NegativeInfinity))


def _two_z_over_pi_integer(z: Expr) -> Optional[Expr]:
    k = div(mul(Integer(2), z), pi)
    return k if k.is_integer else None


def _lmul(*factors: Expr) -> Expr:
    """The product evaluated left to right as Python evaluates SymPy's
    ``a*b*c``: ``4*(1 - m)*f`` distributes ``4`` over ``1 - m`` first."""
    result = factors[0]
    for f in factors[1:]:
        result = mul(result, f)
    return result


def _sin(z: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import sin
    return sin(z)


def _tan(z: Expr) -> Expr:
    from sympy_extras.functions.elementary.trigonometric import tan
    return tan(z)


# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py:elliptic_k
class EllipticK(DefinedFunction):
    """The complete elliptic integral of the first kind ``elliptic_k(m)``,
    unevaluated (``elliptic_k`` evaluates)."""

    function_name = 'elliptic_k'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .gamma_functions import gamma
        [m] = args
        if m.is_zero:
            return mul(pi, HALF)
        if m == HALF:
            return div(mul(Integer(8), pow(pi, Rational(3, 2))), pow(gamma(Rational(-1, 4)), Integer(2)))
        if m == ONE:
            return zoo
        if m == MINUS_ONE:
            return div(pow(gamma(Rational(1, 4)), Integer(2)), mul(Integer(4), sqrt(mul(Integer(2), pi))))
        if _is_real_infinity(m) or isinstance(m, ComplexInfinity) or m == mul(I, oo) or m == mul(I, NegativeInfinity()):
            return ZERO
        return None

    def fdiff(self, argindex: int) -> Expr:
        m = self.args[0]
        return div(sub(elliptic_e(m), mul(sub(ONE, m), elliptic_k(m))), mul(Integer(2), m, sub(ONE, m)))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        m = self.args[0]
        return elliptic_k(conjugate(m)) if _off_cut(m) else None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[0] == 1 else mpmath.mpc(mpmath.ellipk(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if self.args[0].is_infinite:
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py:elliptic_f
class EllipticF(DefinedFunction):
    """The incomplete elliptic integral of the first kind
    ``elliptic_f(z, m)``, unevaluated (``elliptic_f`` evaluates)."""

    function_name = 'elliptic_f'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        z, m = args
        if z.is_zero:
            return ZERO
        if m.is_zero:
            return z
        k = _two_z_over_pi_integer(z)
        if k is not None:
            return mul(k, elliptic_k(m))
        if _is_real_infinity(m):
            return ZERO
        if z.could_extract_minus_sign():
            return mul(MINUS_ONE, elliptic_f(mul(MINUS_ONE, z), m))
        return None

    def fdiff(self, argindex: int) -> Expr:
        z, m = self.args
        fm = sqrt(sub(ONE, mul(m, pow(_sin(z), Integer(2)))))
        if argindex == 1:
            return div(ONE, fm)
        return add(div(elliptic_e(z, m), mul(Integer(2), m, sub(ONE, m))), mul(MINUS_ONE, div(elliptic_f(z, m), mul(Integer(2), m))),
                   mul(MINUS_ONE, div(_sin(mul(Integer(2), z)), _lmul(Integer(4), sub(ONE, m), fm))))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        z, m = self.args
        return elliptic_f(conjugate(z), conjugate(m)) if _off_cut(m) else None

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.ellipf(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        z, m = self.args
        facts: dict[str, bool] = {'commutative': True}
        if z.is_zero or (m.is_extended_real and m.is_infinite):
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py:elliptic_e
class EllipticE(DefinedFunction):
    """The elliptic integral of the second kind, complete ``elliptic_e(m)``
    or incomplete ``elliptic_e(z, m)``, unevaluated (``elliptic_e``
    evaluates)."""

    function_name = 'elliptic_e'
    nargs = frozenset({1, 2})
    call_nargs = frozenset({1, 2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        if len(args) == 2:
            z, m = args
            if m.is_zero:
                return z
            if z.is_zero:
                return ZERO
            k = _two_z_over_pi_integer(z)
            if k is not None:
                return mul(k, elliptic_e(m))
            if _is_real_infinity(m):
                return zoo
            if z.could_extract_minus_sign():
                return mul(MINUS_ONE, elliptic_e(mul(MINUS_ONE, z), m))
            return None
        [m] = args
        if m.is_zero:
            return mul(pi, HALF)
        if m == ONE:
            return ONE
        if isinstance(m, Infinity):
            return mul(I, oo)
        if isinstance(m, NegativeInfinity):
            return oo
        if isinstance(m, ComplexInfinity):
            return zoo
        return None

    def fdiff(self, argindex: int) -> Expr:
        if len(self.args) == 2:
            z, m = self.args
            if argindex == 1:
                return sqrt(sub(ONE, mul(m, pow(_sin(z), Integer(2)))))
            return div(sub(elliptic_e(z, m), elliptic_f(z, m)), mul(Integer(2), m))
        m = self.args[0]
        return div(sub(elliptic_e(m), elliptic_k(m)), mul(Integer(2), m))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        m = self.args[-1]
        if not _off_cut(m):
            return None
        return elliptic_e(*[conjugate(a) for a in self.args])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.ellipe(*values))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/elliptic_integrals.py:elliptic_pi
class EllipticPi(DefinedFunction):
    """The elliptic integral of the third kind, complete
    ``elliptic_pi(n, m)`` or incomplete ``elliptic_pi(n, z, m)``,
    unevaluated (``elliptic_pi`` evaluates)."""

    function_name = 'elliptic_pi'
    nargs = frozenset({2, 3})
    call_nargs = frozenset({2, 3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import sign
        from sympy_extras.functions.elementary.hyperbolic import atanh
        if len(args) == 3:
            n, z, m = args
            if n.is_zero:
                return elliptic_f(z, m)
            if n == ONE:
                return add(elliptic_f(z, m), div(sub(mul(sqrt(sub(ONE, mul(m, pow(_sin(z), Integer(2))))), _tan(z)),
                                                     elliptic_e(z, m)), sub(ONE, m)))
            k = _two_z_over_pi_integer(z)
            if k is not None:
                return mul(k, elliptic_pi(n, m))
            if m.is_zero:
                return div(atanh(mul(sqrt(sub(n, ONE)), _tan(z))), sqrt(sub(n, ONE)))
            if n == m:
                return add(sub(elliptic_f(z, n), elliptic_pi(ONE, z, n)),
                           div(_tan(z), sqrt(sub(ONE, mul(n, pow(_sin(z), Integer(2)))))))
            if _is_real_infinity(n) or _is_real_infinity(m):
                return ZERO
            if z.could_extract_minus_sign():
                return mul(MINUS_ONE, elliptic_pi(n, mul(MINUS_ONE, z), m))
            if (m.is_extended_real and m.is_infinite) or (n.is_extended_real and n.is_infinite):
                return ZERO
            return None
        n, m = args
        if n.is_zero:
            return elliptic_k(m)
        if n == ONE:
            return zoo
        if m.is_zero:
            return div(pi, mul(Integer(2), sqrt(sub(ONE, n))))
        if m == ONE:
            return div(NegativeInfinity(), sign(sub(n, ONE)))
        if n == m:
            return div(elliptic_e(n), sub(ONE, n))
        if _is_real_infinity(n) or _is_real_infinity(m):
            return ZERO
        if (m.is_extended_real and m.is_infinite) or (n.is_extended_real and n.is_infinite):
            return ZERO
        return None

    def fdiff(self, argindex: int) -> Expr:
        if len(self.args) == 3:
            n, z, m = self.args
            fm, fn = sqrt(sub(ONE, mul(m, pow(_sin(z), Integer(2))))), sub(ONE, mul(n, pow(_sin(z), Integer(2))))
            if argindex == 1:
                return div(add(elliptic_e(z, m), div(mul(sub(m, n), elliptic_f(z, m)), n),
                               div(mul(sub(pow(n, Integer(2)), m), elliptic_pi(n, z, m)), n),
                               mul(MINUS_ONE, div(mul(n, fm, _sin(mul(Integer(2), z))), mul(Integer(2), fn)))),
                           _lmul(Integer(2), sub(m, n), sub(n, ONE)))
            if argindex == 2:
                return div(ONE, mul(fm, fn))
            return div(add(div(elliptic_e(z, m), sub(m, ONE)), elliptic_pi(n, z, m),
                           mul(MINUS_ONE, div(mul(m, _sin(mul(Integer(2), z))), _lmul(Integer(2), sub(m, ONE), fm)))),
                       mul(Integer(2), sub(n, m)))
        n, m = self.args
        if argindex == 1:
            return div(add(elliptic_e(m), div(mul(sub(m, n), elliptic_k(m)), n),
                           div(mul(sub(pow(n, Integer(2)), m), elliptic_pi(n, m)), n)),
                       _lmul(Integer(2), sub(m, n), sub(n, ONE)))
        return div(add(div(elliptic_e(m), sub(m, ONE)), elliptic_pi(n, m)), mul(Integer(2), sub(n, m)))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        if len(self.args) == 3:
            n, _, m = self.args
            if not (_off_cut(n) and _off_cut(m)):
                return None
        return elliptic_pi(*[conjugate(a) for a in self.args])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            if len(values) == 3:
                return mpmath.mpc(mpmath.ellippi(values[0], values[1], values[2]))
            return mpmath.mpc(mpmath.ellippi(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def elliptic_k(m: ExprLike) -> Expr:
    """``elliptic_k(m)``, evaluated."""
    return EllipticK.apply(to_expr(m))


def elliptic_f(z: ExprLike, m: ExprLike) -> Expr:
    """``elliptic_f(z, m)``, evaluated."""
    return EllipticF.apply(to_expr(z), to_expr(m))


def elliptic_e(m: ExprLike, z: Optional[ExprLike] = None) -> Expr:
    """``elliptic_e(m)``, or ``elliptic_e(z, m)`` when called with two
    arguments (SymPy's order: the amplitude first)."""
    if z is None:
        return EllipticE.apply(to_expr(m))
    return EllipticE.apply(to_expr(m), to_expr(z))


def elliptic_pi(n: ExprLike, m: ExprLike, z: Optional[ExprLike] = None) -> Expr:
    """``elliptic_pi(n, m)``, or ``elliptic_pi(n, z, m)`` when called with
    three arguments (SymPy's order)."""
    if z is None:
        return EllipticPi.apply(to_expr(n), to_expr(m))
    return EllipticPi.apply(to_expr(n), to_expr(m), to_expr(z))
