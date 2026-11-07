"""The gamma functions on the typed core: ``Gamma``, ``LogGamma``,
``PolyGamma``, ``DiGamma``, ``TriGamma`` (unevaluated nodes) and
``gamma``, ``loggamma``, ``polygamma``, ``digamma``, ``trigamma``
(evaluating).

Ported from SymPy's ``sympy.functions.special.gamma_functions``:
``gamma`` at the integers (factorials, ``zoo`` at the poles) and at the
half-integers (rational multiples of ``sqrt(pi)``); ``loggamma`` at the
integers and the positive half-integers; ``polygamma(0, n) =
harmonic(n - 1) - EulerGamma`` at the positive integers (the harmonic
number computed exactly), the poles, ``oo``, and ``polygamma(-1, z)``;
``digamma`` and ``trigamma`` are ``polygamma(0, z)`` and
``polygamma(1, z)``; ``polygamma(n, z)`` for an odd ``n`` at the integers
and at ``1/2`` with the exact ``zeta`` at an even argument
(``trigamma(2) = pi**2/6 - 1``). Where SymPy answers with ``zeta`` at an
odd argument (``polygamma(2, 3)``) or expands ``polygamma(0, p/q)`` by Gauss' digamma theorem the
node stays: those are not ported.

Examples
========

>>> from sympy_extras.core import Symbol, Rational, Integer
>>> from sympy_extras.functions.special.gamma_functions import gamma, loggamma, polygamma, digamma, Gamma
>>> x = Symbol('x')
>>> gamma(5), gamma(0), gamma(Rational(1, 2)), gamma(Rational(-3, 2)), gamma(Rational(7, 2))
(24, zoo, sqrt(pi), 4*sqrt(pi)/3, 15*sqrt(pi)/8)
>>> polygamma(0, 4), digamma(1), loggamma(3), gamma(x).diff(x), Gamma(1)
(11/6 - EulerGamma, -EulerGamma, log(2), gamma(x)*polygamma(0, x), gamma(1))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes with evaluating functions, strict typing;
# the zeta forms, Gauss' digamma theorem, lowergamma, uppergamma,
# multigamma and the series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import (ComplexInfinity, EulerGamma, Float, I, Infinity, Integer, NaN, NegativeInfinity,
                                       Rational, nan, number_from_fraction, oo, pi, zoo)
from sympy_extras.core.power import pow, sqrt
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.exponential import log

__all__ = ['Gamma', 'LogGamma', 'PolyGamma', 'DiGamma', 'TriGamma', 'gamma', 'loggamma', 'polygamma', 'digamma',
           'trigamma']

ONE = Integer(1)
ZERO = Integer(0)


def _integer_valued(e: Expr) -> Optional[int]:
    """SymPy's ``intlike``: an ``Integer`` or a ``Float`` with an integer
    value, as a Python integer."""
    if isinstance(e, (Rational, Float)):
        value = e.as_fraction()
        if value.denominator == 1:
            return value.numerator
    return None


def _poles(z: mpmath.mpc) -> bool:
    return bool(mpmath.im(z) == 0 and mpmath.re(z) <= 0 and mpmath.re(z) == mpmath.floor(mpmath.re(z)))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:gamma
class Gamma(DefinedFunction):
    """The gamma function, unevaluated (``gamma`` evaluates)."""

    function_name = 'gamma'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return oo
        value = _integer_valued(z)
        if value is not None:
            return factorial(sub(z, ONE)) if value > 0 else zoo
        if isinstance(z, Rational) and z.q == 2:
            n = abs(z.p) // 2
            if z.p > 0:
                k, coefficient = n, 1
            else:
                n = k = n + 1
                coefficient = 1 if n % 2 == 0 else -1
            coefficient *= prod(range(3, 2*k, 2))
            if z.p > 0:
                return mul(number_from_fraction(Fraction(coefficient, 2**n)), sqrt(pi))
            return mul(number_from_fraction(Fraction(2**n, coefficient)), sqrt(pi))
        return None

    def fdiff(self, argindex: int) -> Expr:
        return mul(gamma(self.args[0]), polygamma(ZERO, self.args[0]))

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        return gamma(conjugate(self.args[0]))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if _poles(values[0]) else mpmath.mpc(mpmath.gamma(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        from sympy_extras.functions.elementary.integers import floor
        x = self.args[0]
        a = facts_of(x)
        facts: dict[str, bool] = {'commutative': True}
        value = _integer_valued(x)
        if (a.get('nonpositive') and a.get('integer')) or (value is not None and value <= 0):
            facts['real'] = False
        elif a.get('positive') or a.get('noninteger'):
            facts['real'] = True
        if a.get('positive'):
            facts['positive'] = True
        elif a.get('noninteger'):
            even = floor(x).is_even
            if even is not None:
                facts['positive'] = even
        return facts


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma
class LogGamma(DefinedFunction):
    """``log(gamma(z))`` continued analytically (the principal branch),
    unevaluated (``loggamma`` evaluates)."""

    function_name = 'loggamma'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [z] = args
        if z.is_integer:
            if z.is_nonpositive:
                return oo
            if z.is_positive:
                return log(gamma(z))
        elif z.is_rational:
            p, q = z.as_numer_denom()
            if p.is_positive and q == 2:
                return log(mul(sqrt(pi), pow(Integer(2), sub(ONE, p)), gamma(p),
                               pow(gamma(mul(add(p, ONE), Rational(1, 2))), Integer(-1))))
        if isinstance(z, Infinity):
            return oo
        if isinstance(z, (NegativeInfinity, ComplexInfinity)) or (z.is_number and z.is_infinite):
            return zoo
        if isinstance(z, NaN):
            return nan
        return None

    def fdiff(self, argindex: int) -> Expr:
        return polygamma(ZERO, self.args[0])

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        z = self.args[0]
        if z.is_zero or isinstance(z, NegativeInfinity):
            return None
        return loggamma(conjugate(z))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if _poles(values[0]) else mpmath.mpc(mpmath.loggamma(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('positive'):
            facts['real'] = True
        elif a.get('nonpositive'):
            facts['real'] = False
        return facts


def _fuzzy_and(values: Sequence[Optional[bool]]) -> Optional[bool]:
    if any(v is False for v in values):
        return False
    if all(v is True for v in values):
        return True
    return None


def _harmonic(n: int, power: int = 1) -> Fraction:
    """The generalised harmonic number ``sum(1/k**power, k = 1..n)``."""
    return sum((Fraction(1, k**power) for k in range(1, n + 1)), Fraction(0))


def _bernoulli(n: int) -> Fraction:
    """The Bernoulli number ``B_n`` (``B_1 = -1/2``) by the recurrence
    ``sum(binomial(m + 1, k)*B_k, k = 0..m) = 0``."""
    from math import comb
    numbers = [Fraction(1)]
    for m in range(1, n + 1):
        numbers.append(-sum((comb(m + 1, k)*numbers[k] for k in range(m)), Fraction(0))/(m + 1))
    return numbers[n]


# PORTED FROM SYMPY: sympy/functions/special/zeta_functions.py:zeta.eval (even integer argument)
def _zeta_even(n: int) -> Expr:
    """``zeta(n)`` for an even ``n > 0``: ``(-1)**(n/2 + 1)*B_n*(2*pi)**n/(2*n!)``."""
    from math import factorial as integer_factorial
    coefficient = Fraction((-1)**(n//2 + 1)) * _bernoulli(n) * 2**n / (2*integer_factorial(n))
    return mul(number_from_fraction(coefficient), pow(pi, Integer(n)))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma
class PolyGamma(DefinedFunction):
    """``polygamma(n, z)``, the ``n``-th derivative of ``digamma``,
    unevaluated (``polygamma`` evaluates)."""

    function_name = 'polygamma'
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        n, z = args
        if isinstance(n, NaN) or isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return oo if n.is_zero else ZERO
        if isinstance(z, Rational) and z.q == 1 and z.p <= 0:
            return zoo
        if n == -1:
            return sub(loggamma(z), mul(log(mul(Integer(2), pi)), Rational(1, 2)))
        if n.is_zero:
            if isinstance(z, NegativeInfinity):
                return oo
            imaginary = z.extract_multiplicatively(I)
            if imaginary is not None and isinstance(imaginary, (Infinity, NegativeInfinity)):
                return oo
            if isinstance(z, Rational) and z.q == 1:
                return sub(number_from_fraction(_harmonic(z.p - 1)), EulerGamma)
            return None  # SymPy: Gauss' digamma theorem for small denominators, not ported
        if n.is_integer and n.is_nonnegative:
            order = _integer_valued(n)
            if order is None or order % 2 == 0:
                return None  # SymPy: zeta(n + 1, z) at an odd argument, not ported
            # (-1)**(n + 1)*n!*zeta(n + 1, z) with the exact zeta at an even argument
            sign = Integer(1)
            if isinstance(z, Rational) and z.q == 1:
                hurwitz = sub(_zeta_even(order + 1), number_from_fraction(_harmonic(z.p - 1, order + 1)))
                return mul(sign, factorial(n), hurwitz)
            if z == Rational(1, 2):
                return mul(sign, factorial(n), Integer(2**(order + 1) - 1), _zeta_even(order + 1))
        return None

    def fdiff(self, argindex: int) -> Expr:
        n, z = self.args
        if argindex == 2:
            return polygamma(add(n, ONE), z)
        return DefinedFunction.fdiff(self, argindex)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        n, z = values
        if _poles(z) or mpmath.im(n) != 0 or mpmath.re(n) != mpmath.floor(mpmath.re(n)) or mpmath.re(n) < 0:
            return None
        return mpmath.mpc(mpmath.polygamma(int(mpmath.re(n)), z))

    def _eval_facts(self) -> dict[str, bool]:
        n, z = (facts_of(a) for a in self.args)
        facts: dict[str, bool] = {'commutative': True}
        if n.get('positive') and z.get('positive'):
            facts['real'] = True
        # SymPy: fuzzy_and([z.is_complex, fuzzy_not(fuzzy_and([z.is_negative, z.is_integer]))])
        negative_integer = _fuzzy_and([z.get('negative'), z.get('integer')])
        not_negative_integer = None if negative_integer is None else not negative_integer
        complex_ = _fuzzy_and([z.get('complex'), not_negative_integer])
        if complex_ is not None:
            facts['complex'] = complex_
        if n.get('positive'):
            if n.get('odd') and z.get('real'):
                facts.update(positive=True, negative=False)
            elif n.get('even') and z.get('positive'):
                facts.update(positive=False, negative=True)
        return facts


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:digamma
class DiGamma(DefinedFunction):
    """``digamma(z) = polygamma(0, z)``, unevaluated (``digamma``
    evaluates, to ``polygamma(0, z)``)."""

    function_name = 'digamma'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        return polygamma(ZERO, args[0])

    def fdiff(self, argindex: int) -> Expr:
        return polygamma(ONE, self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if _poles(values[0]) else mpmath.mpc(mpmath.digamma(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return PolyGamma(ZERO, self.args[0])._eval_facts()


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:trigamma
class TriGamma(DefinedFunction):
    """``trigamma(z) = polygamma(1, z)``, unevaluated (``trigamma``
    evaluates, to ``polygamma(1, z)``)."""

    function_name = 'trigamma'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        return polygamma(ONE, args[0])

    def fdiff(self, argindex: int) -> Expr:
        return polygamma(Integer(2), self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return None if _poles(values[0]) else mpmath.mpc(mpmath.polygamma(1, values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return PolyGamma(ONE, self.args[0])._eval_facts()


def gamma(z: ExprLike) -> Expr:
    """The gamma function of ``z``, evaluated."""
    return Gamma.apply(to_expr(z))


def loggamma(z: ExprLike) -> Expr:
    """``loggamma(z)``, evaluated."""
    return LogGamma.apply(to_expr(z))


def polygamma(n: ExprLike, z: ExprLike) -> Expr:
    """``polygamma(n, z)``, evaluated."""
    return PolyGamma.apply(to_expr(n), to_expr(z))


def digamma(z: ExprLike) -> Expr:
    """``digamma(z)``, evaluated."""
    return DiGamma.apply(to_expr(z))


def trigamma(z: ExprLike) -> Expr:
    """``trigamma(z)``, evaluated."""
    return TriGamma.apply(to_expr(z))
