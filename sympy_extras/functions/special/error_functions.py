"""The error functions on the typed core: ``Erf``, ``Erfc``, ``Erfi``,
``Erfinv``, ``Erfcinv`` (unevaluated nodes) and ``erf``, ``erfc``,
``erfi``, ``erfinv``, ``erfcinv`` (evaluating).

Ported from SymPy's ``sympy.functions.special.error_functions``: the
values at ``0``, ``oo``, ``-oo``, along the imaginary infinities, the
compositions with the inverses, the odd symmetry (``erf(-x) =
-erf(x)``, ``erfc(-x) = 2 - erfc(x)``), the derivatives, conjugates,
SymPy's ``real_to_real_as_real_imag`` complex parts and the facts.
``Si``, ``Ci``, ``Shi``, ``Chi``, ``li``, ``Li`` and ``expint``, ``Ei``
are in the same module, as in SymPy. ``erfcinv`` of a float is evaluated (as ``erfinv(1 - x)``), where SymPy,
lacking it in mpmath, leaves the node. SymPy's helpers
``_erfs`` and ``_eis`` of the Gruntz algorithm are the nodes ``Erfs`` and
``Eis`` (functions ``erfs``, ``eis``), with their asymptotic series and
their rewrites back.

Examples
========

>>> from sympy_extras.core import Symbol, I, oo
>>> from sympy_extras.functions.special.error_functions import erf, erfc, erfi, erfinv, erfcinv, Erf
>>> x, r = Symbol('x'), Symbol('r', real=True)
>>> erf(0), erf(oo), erf(-x), erfc(-x), erfi(-x), erf(I*oo), erfi(I*oo), erf(erfinv(x)), erfinv(erf(r))
(0, 1, -erf(x), 2 - erfc(x), -erfi(x), oo*I, I, x, r)
>>> erf(x).diff(x), erfcinv(2), Erf(0)
(2*exp(-x**2)/sqrt(pi), -oo, erf(0))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/error_functions.py (erf, erfc, erfi,
# erfinv, erfcinv, real_to_real_as_real_imag, the integrals, _erfs, _eis) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the helpers _erfs, _eis as the nodes Erfs, Eis.
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, ExpandHints
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import I, Infinity, Integer, NaN, NegativeInfinity, Rational, nan, oo, pi, zoo
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.exponential import exp

__all__ = ['Erf', 'Erfc', 'Erfi', 'Erfinv', 'Erfcinv', 'Expint', 'EiNode', 'erf', 'erfc', 'erfi', 'erfinv', 'erfcinv',
           'expint', 'Ei', 'TrigonometricIntegral', 'SiNode', 'CiNode', 'ShiNode', 'ChiNode', 'LiNode', 'OffsetLiNode',
           'Si', 'Ci', 'Shi', 'Chi', 'li', 'Li', 'FresnelIntegral', 'FresnelS', 'FresnelC', 'fresnels', 'fresnelc', 'E1', 'Erf2', 'Erf2inv', 'erf2',
           'erf2inv', 'Erfs', 'Eis', 'erfs', 'eis']

ONE = Integer(1)
ZERO = Integer(0)
MINUS_ONE = Integer(-1)
HALF = Rational(1, 2)


def _is_infinity(e: Optional[Expr]) -> bool:
    return isinstance(e, (Infinity, NegativeInfinity))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:real_to_real_as_real_imag
def _real_to_real_parts(f: DefinedFunction, deep: bool) -> Optional[tuple[Expr, Expr]]:
    from sympy_extras.functions.elementary.complexes import as_real_imag
    a = f.args[0]
    if a.is_extended_real:
        return (f, ZERO)
    parts = as_real_imag(a.expand() if deep else a, deep)
    if parts is None:
        return None
    x, y = parts
    cls = type(f)
    plus, minus = cls.apply(add(x, mul(I, y))), cls.apply(sub(x, mul(I, y)))
    return (div(add(plus, minus), Integer(2)), div(sub(plus, minus), mul(Integer(2), I)))


class _ErrorFunction(DefinedFunction):
    """The common methods: conjugate and complex parts."""

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(conjugate(self.args[0]))

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        return _real_to_real_parts(self, deep)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_aseries
def _erf_aseries(f: DefinedFunction, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
    """The asymptotic series of ``erf(z)`` at an argument going to an
    infinity faster than logarithmically (the node itself when the leading
    term of the argument is not found, as SymPy)."""
    from sympy_extras.core.order import order
    from sympy_extras.functions.combinatorial.factorials import factorial2
    from sympy_extras.functions.elementary.integers import ceiling
    if isinstance(args0[0], (Infinity, NegativeInfinity)):
        z = f.args[0]
        assert isinstance(z, Expr)
        try:
            _, ex = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return f
        ex = -ex  # as x -> 1/x for the aseries
        if ex.is_positive:
            count = ceiling(Integer(n) / ex)
            if not isinstance(count, Integer):
                raise TypeError('the number of terms is not an integer: %s' % (count,))
            newn = count.p
            s = [MINUS_ONE**k * factorial2(Integer(2 * k - 1)) / (z**(2 * k + 1) * Integer(2)**k) for k in range(newn)]
            return ONE - (exp(-z**2) / sqrt(pi)) * add(*s, order(1 / z**newn, x))
    return DefinedFunction._eval_aseries(f, n, args0, x, logx)


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf
class Erf(_ErrorFunction):
    """The error function, unevaluated (``erf`` evaluates)."""

    unbranched = True

    function_name = 'erf'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [a] = args
        if isinstance(a, NaN):
            return nan
        if isinstance(a, Infinity):
            return ONE
        if isinstance(a, NegativeInfinity):
            return MINUS_ONE
        if a.is_zero:
            return ZERO
        if isinstance(a, Erfinv):
            return a.args[0]
        if isinstance(a, Erfcinv):
            return sub(ONE, a.args[0])
        if _is_infinity(a.extract_multiplicatively(I)):
            return a
        if a.could_extract_minus_sign():
            return mul(MINUS_ONE, erf(mul(MINUS_ONE, a)))
        return None

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        return _erf_aseries(self, n, args0, x, logx)

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(2), exp(mul(MINUS_ONE, pow(self.args[0], Integer(2))))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erf(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('finite') or a.get('extended_real'):
            facts['finite'] = True
        if a.get('extended_real'):
            facts.update(real=True)
            for name, source in (('zero', 'zero'), ('positive', 'extended_positive'), ('negative', 'extended_negative')):
                if a.get(source) is not None:
                    facts[name] = a[source]
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfc
class Erfc(_ErrorFunction):
    """The complementary error function ``1 - erf(x)``, unevaluated
    (``erfc`` evaluates)."""

    unbranched = True

    function_name = 'erfc'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [a] = args
        if isinstance(a, NaN):
            return nan
        if isinstance(a, Infinity):
            return ZERO
        if a.is_zero:
            return ONE
        if isinstance(a, Erfinv):
            return sub(ONE, a.args[0])
        if isinstance(a, Erfcinv):
            return a.args[0]
        if _is_infinity(a.extract_multiplicatively(I)):
            return mul(MINUS_ONE, a)
        if a.could_extract_minus_sign():
            return sub(Integer(2), erfc(mul(MINUS_ONE, a)))
        return None

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfc._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        return sub(ONE, _erf_aseries(Erf(self._args[0]), n, args0, x, logx))

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(-2), exp(mul(MINUS_ONE, pow(self.args[0], Integer(2))))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erfc(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('extended_real'):
            facts['real'] = True
        elif a.get('imaginary'):
            facts['real'] = False
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfi
class Erfi(_ErrorFunction):
    """The imaginary error function ``-I*erf(I*x)``, unevaluated (``erfi``
    evaluates)."""

    unbranched = True

    function_name = 'erfi'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z.is_zero:
            return ZERO
        if isinstance(z, Infinity):
            return oo
        if z.could_extract_minus_sign():
            return mul(MINUS_ONE, erfi(mul(MINUS_ONE, z)))
        rotated = z.extract_multiplicatively(I)
        if rotated is not None:
            if isinstance(rotated, Infinity):
                return I
            if isinstance(rotated, Erfinv):
                return mul(I, rotated.args[0])
            if isinstance(rotated, Erfcinv):
                return mul(I, sub(ONE, rotated.args[0]))
        return None

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfi._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial2
        if isinstance(args0[0], Infinity):
            z = self._args[0]
            s = [factorial2(Integer(2 * k - 1)) / (Integer(2)**k * z**(2 * k + 1)) for k in range(n)]
            return -I + (exp(z**2) / sqrt(pi)) * add(*s, order(1 / z**n, x))
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def fdiff(self, argindex: int) -> Expr:
        return div(mul(Integer(2), exp(pow(self.args[0], Integer(2)))), sqrt(pi))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erfi(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('extended_real'):
            facts['extended_real'] = True
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfinv
class Erfinv(DefinedFunction):
    """The inverse of ``erf`` on ``(-1, 1)``, unevaluated (``erfinv``
    evaluates)."""

    function_name = 'erfinv'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z == MINUS_ONE:
            return NegativeInfinity()
        if z.is_zero:
            return ZERO
        if z == ONE:
            return oo
        if isinstance(z, Erf) and z.args[0].is_extended_real:
            return z.args[0]
        negated = z.extract_multiplicatively(MINUS_ONE)
        if isinstance(negated, Erf) and negated.args[0].is_extended_real:
            return mul(MINUS_ONE, negated.args[0])
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(sqrt(pi), exp(pow(erfinv(self.args[0]), Integer(2))), HALF)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        if mpmath.im(z) != 0 or not -1 < mpmath.re(z) < 1:
            return None
        return mpmath.mpc(mpmath.erfinv(mpmath.re(z)))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('zero') is not None:
            facts['zero'] = a['zero']
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erfcinv
class Erfcinv(DefinedFunction):
    """The inverse of ``erfc`` on ``(0, 2)``, unevaluated (``erfcinv``
    evaluates)."""

    function_name = 'erfcinv'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if z.is_zero:
            return oo
        if z == ONE:
            return ZERO
        if z == Integer(2):
            return NegativeInfinity()
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(MINUS_ONE, sqrt(pi), exp(pow(erfcinv(self.args[0]), Integer(2))), HALF)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        z = values[0]
        if mpmath.im(z) != 0 or not 0 < mpmath.re(z) < 2:
            return None
        return mpmath.mpc(mpmath.erfinv(1 - mpmath.re(z)))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        zero = sub(self.args[0], ONE).is_zero
        if zero is not None:
            facts['zero'] = zero
        return facts


def erf(x: ExprLike) -> Expr:
    """The error function of ``x``, evaluated."""
    return Erf.apply(to_expr(x))


def erfc(x: ExprLike) -> Expr:
    """``erfc(x)``, evaluated."""
    return Erfc.apply(to_expr(x))


def erfi(x: ExprLike) -> Expr:
    """``erfi(x)``, evaluated."""
    return Erfi.apply(to_expr(x))


def erfinv(x: ExprLike) -> Expr:
    """``erfinv(x)``, evaluated."""
    return Erfinv.apply(to_expr(x))


def erfcinv(x: ExprLike) -> Expr:
    """``erfcinv(x)``, evaluated."""
    return Erfcinv.apply(to_expr(x))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:expint
class Expint(DefinedFunction):
    """The generalised exponential integral ``expint(nu, z)``, unevaluated
    (``expint`` evaluates); a polar argument loses its branch factor as
    SymPy's rules say."""

    function_name = 'expint'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.core.function import expand_mul
        from sympy_extras.functions.special.gamma_functions import uppergamma
        from sympy_extras.functions.combinatorial.factorials import factorial
        from sympy_extras.functions.elementary.complexes import unpolarify
        from sympy_extras.functions.special.gamma_functions import gamma
        from sympy_extras.core.logic import true
        nu, z = args
        nu2 = unpolarify(nu)
        if nu != nu2:
            return expint(nu2, z)
        integer = isinstance(nu, Rational) and nu.q == 1
        if (integer and nu.is_nonpositive) or (not integer and isinstance(nu, Rational) and nu.q == 2):
            return unpolarify(expand_mul(mul(pow(z, sub(nu, ONE)), uppergamma(sub(ONE, nu), z))))
        # the branch factor (C/f lowergamma)
        z, n = z.extract_branch_factor()
        if n == 0:
            return None
        if nu.is_integer is True:
            if not ((nu > 0) is true):
                return None
            return sub(expint(nu, z), mul(Integer(2), pi, I, n, pow(MINUS_ONE, sub(nu, ONE)),
                                          div(pow(unpolarify(z), sub(nu, ONE)), factorial(sub(nu, ONE)))))
        return add(mul(sub(exp(mul(Integer(2), I, pi, nu, n)), ONE), pow(z, sub(nu, ONE)), gamma(sub(ONE, nu))),
                   expint(nu, z))

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:expint._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import rf
        if isinstance(args0[1], Infinity):
            nu, z = self._args
            s = [MINUS_ONE**k * rf(nu, Integer(k)) / z**k for k in range(n)]
            return (exp(-z) / z) * add(*s, order(1 / z**n, x))
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def fdiff(self, argindex: int) -> Expr:
        nu, z = self.args
        if argindex == 2:
            return mul(MINUS_ONE, expint(sub(nu, ONE), z))
        return DefinedFunction.fdiff(self, argindex)  # SymPy: with meijerg, not ported

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.expint(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei
class EiNode(DefinedFunction):
    """The exponential integral ``Ei(z)``, unevaluated (``Ei`` evaluates)."""

    function_name = 'Ei'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if z.is_zero:
            return NegativeInfinity()
        if isinstance(z, Infinity):
            return oo
        if isinstance(z, NegativeInfinity):
            return ZERO
        nz, n = z.extract_branch_factor()
        if n != 0:
            return add(Ei(nz), mul(Integer(2), I, pi, n))
        return None

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial
        from sympy_extras.core.coreerrors import PoleError
        from sympy_extras.functions.elementary.complexes import PolarLift
        from sympy_extras.functions.elementary.exponential import ExpPolar
        if isinstance(args0[0], (Infinity, NegativeInfinity)):
            z = self._args[0]
            if z.has(ExpPolar, PolarLift):
                # SymPy's series forgets the branch: its expansion of
                # Ei(exp_polar(I*pi)/w) misses I*pi, so that of expint(2, 1/w)
                # is -I*pi/w + O(w**4) (the function is exponentially small)
                raise PoleError('Asymptotic expansion of Ei of the polar number %s is not implemented.' % (z,))
            s = [factorial(Integer(k)) / z**k for k in range(n)]
            return (exp(z) / z) * add(*s, order(1 / z**n, x))
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def fdiff(self, argindex: int) -> Expr:
        z = self.args[0]
        return div(exp(z), z)

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ei._eval_evalf
    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] == 0:
            return None
        from sympy_extras.functions.elementary.complexes import polar_lift
        value = mpmath.mpc(mpmath.ei(values[0]))
        if div(self.args[0], polar_lift(MINUS_ONE)).is_positive:
            # a positive number times polar_lift(-1): the branch above the
            # cut, Ei(-x) + I*pi
            return value + mpmath.mpc(0, mpmath.pi)
        return value

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def expint(nu: ExprLike, z: ExprLike) -> Expr:
    """``expint(nu, z)``, evaluated."""
    return Expint.apply(to_expr(nu), to_expr(z))


def Ei(z: ExprLike) -> Expr:
    """The exponential integral ``Ei(z)``, evaluated (the class is
    :class:`EiNode`, since the function keeps SymPy's name)."""
    return EiNode.apply(to_expr(z))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:TrigonometricIntegral
class TrigonometricIntegral(DefinedFunction):
    """The common rules of ``Si``, ``Ci``, ``Shi``, ``Chi``: the values at
    zero and the infinities, ``I`` and ``-1`` taken out of the argument
    (of the even ones only as polar numbers, ``polar_lift(I)``), the
    branch factor of a polar argument, the derivative ``trigfunc(z)/z``."""

    #: whether the integrand is odd (sin, sinh): its value at 0 is 0
    odd: bool = True

    @classmethod
    def at_zero(cls) -> Expr:
        raise NotImplementedError

    @classmethod
    def at_infinity(cls) -> Expr:
        raise NotImplementedError

    @classmethod
    def at_negative_infinity(cls) -> Expr:
        raise NotImplementedError

    @classmethod
    def minus_factor(cls, z: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def i_factor(cls, z: Expr, sign: int) -> Expr:
        raise NotImplementedError

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if z.is_zero:
            return cls.at_zero()
        if isinstance(z, Infinity):
            return cls.at_infinity()
        if isinstance(z, NegativeInfinity):
            return cls.at_negative_infinity()
        from sympy_extras.functions.elementary.complexes import polar_lift
        nz = z.extract_multiplicatively(polar_lift(I))
        if nz is None and cls.odd:
            nz = z.extract_multiplicatively(I)
        if nz is not None:
            return cls.i_factor(nz, 1)
        nz = z.extract_multiplicatively(polar_lift(mul(MINUS_ONE, I)))
        if nz is not None:
            return cls.i_factor(nz, -1)
        nz = z.extract_multiplicatively(polar_lift(MINUS_ONE))
        if nz is None and cls.odd:
            nz = z.extract_multiplicatively(MINUS_ONE)
        if nz is not None:
            return cls.minus_factor(nz)
        # the branch factor: 2*pi*I*n*trigfunc(0) + f(z)
        bz, n = z.extract_branch_factor()
        if n == 0 and bz == z:
            return None
        jump = ZERO if cls.odd else mul(Integer(2), pi, I, n)
        return add(jump, cls.apply(bz))

    def fdiff(self, argindex: int) -> Expr:
        z = self.args[0]
        return div(self.integrand(z), z)

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if self.odd and self.args[0].is_zero:
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Si
class SiNode(TrigonometricIntegral):
    """The sine integral ``Si(z)``, unevaluated (``Si`` evaluates)."""

    function_name = 'Si'

    @classmethod
    def at_zero(cls) -> Expr:
        return ZERO

    @classmethod
    def at_infinity(cls) -> Expr:
        return mul(pi, HALF)

    @classmethod
    def at_negative_infinity(cls) -> Expr:
        return mul(MINUS_ONE, pi, HALF)

    @classmethod
    def minus_factor(cls, z: Expr) -> Expr:
        return mul(MINUS_ONE, Si(z))

    @classmethod
    def i_factor(cls, z: Expr, sign: int) -> Expr:
        return mul(I, Shi(z), Integer(sign))

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import sin
        return sin(z)

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Si._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial
        from sympy_extras.functions.elementary.trigonometric import cos, sin
        if isinstance(args0[0], Infinity):
            z = self._args[0]
            p = [MINUS_ONE**k * factorial(Integer(2 * k)) / z**(2 * k + 1) for k in range(n // 2 + 1)]
            q = [MINUS_ONE**k * factorial(Integer(2 * k + 1)) / z**(2 * (k + 1)) for k in range(n // 2)]
            o = order(1 / z**n, x)
            return pi / 2 - cos(z) * add(*p, o) - sin(z) * add(*q, o)
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.si(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ci
class CiNode(TrigonometricIntegral):
    """The cosine integral ``Ci(z)``, unevaluated (``Ci`` evaluates)."""

    function_name = 'Ci'
    odd = False

    @classmethod
    def at_zero(cls) -> Expr:
        return zoo

    @classmethod
    def at_infinity(cls) -> Expr:
        return ZERO

    @classmethod
    def at_negative_infinity(cls) -> Expr:
        return mul(I, pi)

    @classmethod
    def minus_factor(cls, z: Expr) -> Expr:
        return add(Ci(z), mul(I, pi))

    @classmethod
    def i_factor(cls, z: Expr, sign: int) -> Expr:
        return add(Chi(z), mul(I, pi, HALF, Integer(sign)))

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import cos
        return cos(z)

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Ci._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial
        from sympy_extras.functions.elementary.trigonometric import cos, sin
        point = args0[0]
        if isinstance(point, (Infinity, NegativeInfinity)):
            z = self._args[0]
            p = [MINUS_ONE**k * factorial(Integer(2 * k)) / z**(2 * k + 1) for k in range(n // 2 + 1)]
            q = [MINUS_ONE**k * factorial(Integer(2 * k + 1)) / z**(2 * (k + 1)) for k in range(n // 2)]
            o = order(1 / z**n, x)
            result = sin(z) * add(*p, o) - cos(z) * add(*q, o)
            if isinstance(point, NegativeInfinity):
                result += I * pi
            return result
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[0] == 0 else mpmath.mpc(mpmath.ci(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Shi
class ShiNode(TrigonometricIntegral):
    """The hyperbolic sine integral ``Shi(z)``, unevaluated (``Shi``
    evaluates)."""

    function_name = 'Shi'

    @classmethod
    def at_zero(cls) -> Expr:
        return ZERO

    @classmethod
    def at_infinity(cls) -> Expr:
        return oo

    @classmethod
    def at_negative_infinity(cls) -> Expr:
        return NegativeInfinity()

    @classmethod
    def minus_factor(cls, z: Expr) -> Expr:
        return mul(MINUS_ONE, Shi(z))

    @classmethod
    def i_factor(cls, z: Expr, sign: int) -> Expr:
        return mul(I, Si(z), Integer(sign))

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import sinh
        return sinh(z)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.shi(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Chi
class ChiNode(TrigonometricIntegral):
    """The hyperbolic cosine integral ``Chi(z)``, unevaluated (``Chi``
    evaluates)."""

    function_name = 'Chi'
    odd = False

    @classmethod
    def at_zero(cls) -> Expr:
        return zoo

    @classmethod
    def at_infinity(cls) -> Expr:
        return oo

    @classmethod
    def at_negative_infinity(cls) -> Expr:
        return oo

    @classmethod
    def minus_factor(cls, z: Expr) -> Expr:
        return add(Chi(z), mul(I, pi))

    @classmethod
    def i_factor(cls, z: Expr, sign: int) -> Expr:
        return add(Ci(z), mul(I, pi, HALF, Integer(sign)))

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.hyperbolic import cosh
        return cosh(z)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[0] == 0 else mpmath.mpc(mpmath.chi(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:li
class LiNode(DefinedFunction):
    """The logarithmic integral ``li(z)``, unevaluated (``li``
    evaluates)."""

    function_name = 'li'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if z.is_zero:
            return ZERO
        if z == ONE:
            return NegativeInfinity()
        if isinstance(z, Infinity):
            return oo
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.elementary.exponential import log
        return div(ONE, log(self.args[0]))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        z = self.args[0]
        if z.is_extended_negative:
            return None  # the branch cut (-oo, 0)
        return li(conjugate(z))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[0] == 1 else mpmath.mpc(mpmath.li(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if self.args[0].is_zero:
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:Li
class OffsetLiNode(DefinedFunction):
    """The offset logarithmic integral ``Li(z) = li(z) - li(2)``,
    unevaluated (``Li`` evaluates)."""

    function_name = 'Li'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, Infinity):
            return oo
        if z == Integer(2):
            return ZERO
        return None

    def fdiff(self, argindex: int) -> Expr:
        from sympy_extras.functions.elementary.exponential import log
        return div(ONE, log(self.args[0]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if values[0] == 1 else mpmath.mpc(mpmath.li(values[0], offset=True))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def Si(z: ExprLike) -> Expr:
    """The sine integral, evaluated (the class is :class:`SiNode`)."""
    return SiNode.apply(to_expr(z))


def Ci(z: ExprLike) -> Expr:
    """The cosine integral, evaluated (the class is :class:`CiNode`)."""
    return CiNode.apply(to_expr(z))


def Shi(z: ExprLike) -> Expr:
    """The hyperbolic sine integral, evaluated (the class is :class:`ShiNode`)."""
    return ShiNode.apply(to_expr(z))


def Chi(z: ExprLike) -> Expr:
    """The hyperbolic cosine integral, evaluated (the class is :class:`ChiNode`)."""
    return ChiNode.apply(to_expr(z))


def li(z: ExprLike) -> Expr:
    """The logarithmic integral, evaluated (the class is :class:`LiNode`)."""
    return LiNode.apply(to_expr(z))


def Li(z: ExprLike) -> Expr:
    """The offset logarithmic integral, evaluated (the class is
    :class:`OffsetLiNode`)."""
    return OffsetLiNode.apply(to_expr(z))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:FresnelIntegral
class FresnelIntegral(DefinedFunction):
    """The common rules of ``fresnels`` and ``fresnelc``: ``1/2`` at
    ``oo``, ``0`` at ``0``, ``-1`` and ``I`` taken out of the argument
    (``fresnels(I*z) = -I*fresnels(z)``, ``fresnelc(I*z) =
    I*fresnelc(z)``), the derivative ``trigfunc(pi*z**2/2)``."""

    #: SymPy's _sign: the factor of I taken out (-1 for fresnels)
    sign: int = 1

    @classmethod
    def evaluating(cls, z: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        raise NotImplementedError

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, Infinity):
            return HALF
        if z.is_zero:
            return ZERO
        prefactor: Expr = ONE
        argument = z
        changed = False
        negated = argument.extract_multiplicatively(MINUS_ONE)
        if negated is not None:
            prefactor, argument, changed = mul(MINUS_ONE, prefactor), negated, True
        rotated = argument.extract_multiplicatively(I)
        if rotated is not None:
            prefactor, argument, changed = mul(Integer(cls.sign), I, prefactor), rotated, True
        if changed:
            return mul(prefactor, cls.evaluating(argument))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return self.integrand(mul(HALF, pi, pow(self.args[0], Integer(2))))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return type(self).apply(conjugate(self.args[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        for name in ('extended_real', 'zero'):
            if a.get(name) is not None:
                facts[name] = a[name]
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:fresnels
class FresnelS(FresnelIntegral):
    """The Fresnel integral ``fresnels(z)``, unevaluated (``fresnels``
    evaluates)."""

    unbranched = True

    function_name = 'fresnels'
    generic_nseries = True
    sign = -1

    @classmethod
    def evaluating(cls, z: Expr) -> Expr:
        return fresnels(z)

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import sin
        return sin(z)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.fresnels(values[0]))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:fresnelc
class FresnelC(FresnelIntegral):
    """The Fresnel integral ``fresnelc(z)``, unevaluated (``fresnelc``
    evaluates)."""

    unbranched = True

    function_name = 'fresnelc'
    generic_nseries = True
    sign = 1

    @classmethod
    def evaluating(cls, z: Expr) -> Expr:
        return fresnelc(z)

    @classmethod
    def integrand(cls, z: Expr) -> Expr:
        from sympy_extras.functions.elementary.trigonometric import cos
        return cos(z)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.fresnelc(values[0]))


def fresnels(z: ExprLike) -> Expr:
    """``fresnels(z)``, evaluated."""
    return FresnelS.apply(to_expr(z))


def fresnelc(z: ExprLike) -> Expr:
    """``fresnelc(z)``, evaluated."""
    return FresnelC.apply(to_expr(z))



# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:E1
def E1(z: ExprLike) -> Expr:
    """The classical exponential integral ``expint(1, z)``."""
    return expint(Integer(1), z)



# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf2
class Erf2(DefinedFunction):
    """The two-argument error function ``erf2(x, y) = erf(y) - erf(x)``,
    unevaluated (``erf2`` evaluates)."""

    function_name = 'erf2'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x, y = args
        special = (oo, NegativeInfinity(), ZERO)
        if isinstance(x, NaN) or isinstance(y, NaN):
            return nan
        if x == y:
            return ZERO
        if x in special or y in special:
            return sub(erf(y), erf(x))
        if isinstance(y, Erf2inv) and y.args[0] == x:
            return y.args[1]
        if x.is_zero or y.is_zero or (x.is_extended_real and x.is_infinite) or (y.is_extended_real and y.is_infinite):
            return sub(erf(y), erf(x))
        sign_x, sign_y = x.could_extract_minus_sign(), y.could_extract_minus_sign()
        if sign_x and sign_y:
            return mul(MINUS_ONE, erf2(mul(MINUS_ONE, x), mul(MINUS_ONE, y)))
        if sign_x or sign_y:
            return sub(erf(y), erf(x))
        return None

    def fdiff(self, argindex: int) -> Expr:
        x, y = self.args
        if argindex == 1:
            return div(mul(Integer(-2), exp(mul(MINUS_ONE, pow(x, Integer(2))))), sqrt(pi))
        return div(mul(Integer(2), exp(mul(MINUS_ONE, pow(y, Integer(2))))), sqrt(pi))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return erf2(conjugate(self.args[0]), conjugate(self.args[1]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.erf(values[1]) - mpmath.erf(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        from sympy_extras.core.logic import false, true
        from sympy_extras.core.relational import eq
        x, y = self.args
        facts: dict[str, bool] = {'commutative': True}
        if x.is_extended_real and y.is_extended_real:
            facts['extended_real'] = True
        equal = eq(x, y)
        if equal == true:
            facts['zero'] = True
        elif equal == false:
            facts['zero'] = False
        return facts


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:erf2inv
class Erf2inv(DefinedFunction):
    """The inverse ``erf2inv(x, y)`` of ``erf2`` in its second argument,
    unevaluated (``erf2inv`` evaluates)."""

    function_name = 'erf2inv'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x, y = args
        if isinstance(x, NaN) or isinstance(y, NaN):
            return nan
        if x.is_zero and y.is_zero:
            return ZERO
        if x.is_zero and y == ONE:
            return oo
        if x == ONE and y.is_zero:
            return ONE
        if x.is_zero:
            return erfinv(y)
        if isinstance(x, Infinity):
            return erfcinv(mul(MINUS_ONE, y))
        if y.is_zero:
            return x
        if isinstance(y, Infinity):
            return erfinv(x)
        return None

    def fdiff(self, argindex: int) -> Expr:
        x, y = self.args
        if argindex == 1:
            return exp(sub(pow(erf2inv(x, y), Integer(2)), pow(x, Integer(2))))
        return mul(sqrt(pi), HALF, exp(pow(erf2inv(x, y), Integer(2))))

    def _eval_facts(self) -> dict[str, bool]:
        x, y = self.args
        facts: dict[str, bool] = {'commutative': True}
        if x.is_zero and y.is_zero:
            facts['zero'] = True
        return facts


def erf2(x: ExprLike, y: ExprLike) -> Expr:
    """``erf2(x, y)``, evaluated."""
    return Erf2.apply(to_expr(x), to_expr(y))


def erf2inv(x: ExprLike, y: ExprLike) -> Expr:
    """``erf2inv(x, y)``, evaluated."""
    return Erf2inv.apply(to_expr(x), to_expr(y))


# ---------------------------------------------------------------------
# the helpers which make erf, Ei and li tractable for the Gruntz algorithm

def _helper_aseries(terms: list[Expr], o: Expr, x: Symbol, n: int, logx: Optional[Expr]) -> Expr:
    """The terms expanded and the order added (SymPy: "it is very
    inefficient to first add the order and then do the nseries")."""
    return add(*terms)._eval_nseries(x, n, logx, ZERO) + o


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_erfs
class Erfs(DefinedFunction):
    """SymPy's helper ``_erfs(z) = (1 - erf(z))*exp(z**2)``, which makes
    ``erf`` tractable for the Gruntz algorithm, unevaluated (``erfs``
    evaluates). As SymPy's, it has no numerical value."""

    function_name = '_erfs'
    generic_nseries = True

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [a] = args
        if a.is_zero:
            return ONE
        return None

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_erfs._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial
        point = args0[0]
        # the expansion at oo, and at I*oo (SymPy: "is the series really correct?")
        if isinstance(point, Infinity) or isinstance(point.extract_multiplicatively(I), Infinity):
            z = self._args[0]
            terms = [div(factorial(Integer(2 * k)) * pow(Integer(-4), Integer(-k)), sqrt(pi) * factorial(Integer(k)))
                     * pow(div(ONE, z), Integer(2 * k + 1)) for k in range(n)]
            return _helper_aseries(terms, order(div(ONE, pow(z, Integer(2 * n + 1))), x), x, n, logx)
        return DefinedFunction._eval_aseries(self, n, args0, x, logx)

    def fdiff(self, argindex: int) -> Expr:
        z = self._args[0]
        return -2 / sqrt(pi) + 2 * z * erfs(z)

    def rewrite_intractable(self, args: tuple[Expr, ...]) -> Expr:
        """SymPy's ``_eval_rewrite_as_intractable``: ``(1 - erf(z))*exp(z**2)``."""
        z = args[0]
        return mul(sub(ONE, erf(z)), exp(pow(z, Integer(2))))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def erfs(z: ExprLike) -> Expr:
    """``_erfs(z)``, evaluated."""
    return Erfs.apply(to_expr(z))


# PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis
class Eis(DefinedFunction):
    """SymPy's helper ``_eis(z) = exp(-z)*Ei(z)``, which makes ``Ei`` and
    ``li`` tractable for the Gruntz algorithm, unevaluated (``eis``
    evaluates). As SymPy's, it has no numerical value."""

    function_name = '_eis'

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.factorials import factorial
        if not isinstance(args0[0], (Infinity, NegativeInfinity)):
            return DefinedFunction._eval_aseries(self, n, args0, x, logx)
        z = self._args[0]
        terms = [factorial(Integer(k)) * pow(div(ONE, z), Integer(k + 1)) for k in range(n)]
        return _helper_aseries(terms, order(div(ONE, pow(z, Integer(n + 1))), x), x, n, logx)

    def fdiff(self, argindex: int) -> Expr:
        z = self._args[0]
        return sub(div(ONE, z), eis(z))

    def rewrite_intractable(self, args: tuple[Expr, ...]) -> Expr:
        """SymPy's ``_eval_rewrite_as_intractable``: ``exp(-z)*Ei(z)``."""
        z = args[0]
        return mul(exp(mul(MINUS_ONE, z)), Ei(z))

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis._eval_as_leading_term
    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.function import Application
        x0 = self._args[0].limit(x, ZERO)
        if x0.is_zero:
            return self.rewrite_intractable(self._args)._eval_as_leading_term(x, logx, cdir)
        return Application._eval_as_leading_term(self, x, logx, cdir)

    # PORTED FROM SYMPY: sympy/functions/special/error_functions.py:_eis._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        from sympy_extras.core.function import Application
        x0 = self._args[0].limit(x, ZERO)
        if x0.is_zero:
            return self.rewrite_intractable(self._args)._eval_nseries(x, n, logx, ZERO)
        return Application._eval_nseries(self, x, n, logx, ZERO)

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def eis(z: ExprLike) -> Expr:
    """``_eis(z)``, evaluated (nothing to evaluate: the node)."""
    return Eis.apply(to_expr(z))
