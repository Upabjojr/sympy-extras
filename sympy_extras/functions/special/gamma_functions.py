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
``polygamma(1, z)``; ``polygamma(n, z)`` at the integers and at ``1/2``
through ``zeta`` (``trigamma(2) = pi**2/6 - 1``, ``polygamma(2, 3) =
-2*zeta(3) + 9/4``); ``polygamma(0, p/q)`` for ``q <= 6`` by Gauss'
digamma theorem. Not ported: lowergamma, uppergamma, multigamma, the
series, the general ``expand_func``.

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
# the general expand_func, lowergamma, uppergamma,
# multigamma and the series not ported.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from math import prod
from typing import Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import add, sub
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction
from sympy_extras.core.mul import div, mul
from sympy_extras.core.numbers import (ComplexInfinity, EulerGamma, Float, I, Infinity, Integer, NaN, NegativeInfinity,
                                       Rational, nan, number_from_fraction, oo, pi, zoo)
from sympy_extras.core.power import pow, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.exponential import exp, log

__all__ = ['Gamma', 'LogGamma', 'PolyGamma', 'DiGamma', 'TriGamma', 'LowerGamma', 'UpperGamma', 'gamma', 'loggamma',
           'polygamma', 'digamma', 'trigamma', 'lowergamma', 'uppergamma', 'MultiGamma', 'multigamma']

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

    unbranched = True

    function_name = 'gamma'
    # PORTED FROM SYMPY: _singularities
    singularities: Union[None, bool, tuple[Expr, ...]] = (zoo,)

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

    # PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:loggamma._eval_aseries
    def _eval_aseries(self, n: int, args0: Sequence[Expr], x: Symbol, logx: Optional[Expr]) -> Expr:
        """Stirling's series at an argument going to ``oo``:
        ``(z - 1/2)*log(z) - z + log(2*pi)/2 + sum B_2k/(2k(2k - 1) z**(2k - 1))``."""
        from sympy_extras.core.numbers import oo
        from sympy_extras.core.order import order
        from sympy_extras.functions.combinatorial.numbers import bernoulli
        if args0[0] != oo:
            return DefinedFunction._eval_aseries(self, n, args0, x, logx)
        z = self.args[0]
        assert isinstance(z, Expr)
        r = log(z) * (z - Rational(1, 2)) - z + log(2 * pi) / 2
        terms = [bernoulli(Integer(2 * k)) / (2 * k * (2 * k - 1) * z**(2 * k - 1)) for k in range(1, n)]
        o = order(Integer(1), x) if n == 0 else order(1 / z**n, x)
        # it is very inefficient to add the order first and then do the nseries
        return add(r, *terms)._eval_nseries(x, n, logx, Integer(0)) + o

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


def _harmonic(n: int) -> Fraction:
    return sum((Fraction(1, k) for k in range(1, n + 1)), Fraction(0))




# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma._eval_expand_func (n = 0, z rational)
def _digamma_rational(z: Rational) -> Expr:
    """``polygamma(0, p/q)`` by Gauss' digamma theorem (Choi, *Values of
    the polygamma functions at rational arguments*, 2007)."""
    from sympy_extras.functions.elementary.integers import floor
    from sympy_extras.functions.elementary.trigonometric import cos, cot, sin
    p, q = z.p, z.q
    part = add(mul(Integer(-1), EulerGamma), mul(Integer(-1), pi, cot(mul(Rational(p, q), pi)), Rational(1, 2)),
               mul(Integer(-1), log(Integer(q))),
               add(*[mul(cos(mul(Rational(2*k*p, q), pi)), log(mul(Integer(2), sin(mul(Rational(k, q), pi)))))
                     for k in range(1, q)]))
    if z.p > 0:
        count = floor(z)
        assert isinstance(count, Rational)
        z0 = sub(z, count)
        return add(part, add(*[pow(add(z0, Integer(k)), Integer(-1)) for k in range(count.p)]))
    count = floor(sub(ONE, z))
    assert isinstance(count, Rational)
    z0 = add(z, count)
    return sub(part, add(*[pow(sub(sub(z0, ONE), Integer(k)), Integer(-1)) for k in range(count.p)]))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:polygamma
class PolyGamma(DefinedFunction):
    """``polygamma(n, z)``, the ``n``-th derivative of ``digamma``,
    unevaluated (``polygamma`` evaluates)."""

    function_name = 'polygamma'
    generic_nseries = True
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
            if isinstance(z, Rational) and z.q <= 6:
                return _digamma_rational(z)
            return None
        if n.is_integer and n.is_nonnegative:
            from .zeta_functions import zeta
            sign = pow(Integer(-1), add(n, ONE))
            if isinstance(z, Rational) and z.q == 1:
                return mul(sign, factorial(n), zeta(add(n, ONE), z))
            if z == Rational(1, 2):
                return mul(sign, factorial(n), sub(pow(Integer(2), add(n, ONE)), ONE), zeta(add(n, ONE)))
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
    generic_nseries = True

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
    generic_nseries = True

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


def _range_to(e: Expr) -> int:
    """``e`` (an integer, as SymPy's ``range(a + S.Half)`` has it) as a
    Python integer."""
    if isinstance(e, Rational) and e.q == 1:
        return e.p
    raise ValueError("an integer is expected, not %s" % (e,))


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:lowergamma
class LowerGamma(DefinedFunction):
    """The lower incomplete gamma ``lowergamma(a, x)``, unevaluated
    (``lowergamma`` evaluates); a polar argument loses its branch factor
    (``exp_polar(2*pi*I*n)``) as SymPy's rules say."""

    function_name = 'lowergamma'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.special.error_functions import erf
        a, x = args
        if isinstance(x, Rational) and x.p == 0:
            return ZERO
        from sympy_extras.functions.elementary.complexes import unpolarify
        nx, n = x.extract_branch_factor()
        if a.is_integer is True and a.is_positive is True:
            nx = unpolarify(x)
            if nx != x:
                return lowergamma(a, nx)
        elif a.is_integer is True and a.is_nonpositive is True:
            if n != 0:
                return add(div(mul(Integer(2), pi, I, n, pow(Integer(-1), mul(Integer(-1), a))),
                               factorial(mul(Integer(-1), a))), lowergamma(a, nx))
        elif n != 0:
            return mul(exp(mul(Integer(2), pi, I, n, a)), lowergamma(a, nx))
        half = Rational(1, 2)
        if isinstance(a, Rational):
            if a == ONE:
                return sub(ONE, exp(mul(Integer(-1), x)))
            if a == half:
                return mul(sqrt(pi), erf(sqrt(x)))
            if a.q == 1 or a.q == 2:
                b = sub(a, ONE)
                if b.is_positive:
                    if a.q == 1:
                        terms = add(*[div(pow(x, Integer(k)), factorial(Integer(k))) for k in range(a.p)])
                        return sub(factorial(b), mul(exp(mul(Integer(-1), x)), factorial(b), terms))
                    terms = add(*[div(pow(x, sub(Integer(k), half)), gamma(add(half, Integer(k))))
                                  for k in range(1, _range_to(add(a, half)))])
                    return mul(gamma(a), sub(div(lowergamma(half, x), sqrt(pi)), mul(exp(mul(Integer(-1), x)), terms)))
                if a.q != 1:
                    terms = add(*[div(mul(pow(x, add(Integer(k), sub(a, ONE))), gamma(a)), gamma(add(a, Integer(k))))
                                  for k in range(1, _range_to(sub(Rational(3, 2), a)))])
                    return add(div(mul(pow(Integer(-1), sub(half, a)), pi, erf(sqrt(x))), gamma(sub(ONE, a))),
                               mul(exp(mul(Integer(-1), x)), terms))
        if x.is_zero:
            return ZERO
        return None

    def fdiff(self, argindex: int) -> Expr:
        a, z = self.args
        if argindex == 2:
            return mul(exp(mul(Integer(-1), z)), pow(z, sub(a, ONE)))
        if argindex == 1:
            from sympy_extras.functions.special.hyper import meijerg
            return (gamma(a) * digamma(a) - log(z) * uppergamma(a, z)
                    - meijerg([], [ONE, ONE], [ZERO, ZERO, a], [], z))
        return DefinedFunction.fdiff(self, argindex)

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        a, x = self.args
        if x.is_zero or isinstance(x, NegativeInfinity):
            return None
        return lowergamma(conjugate(a), conjugate(x))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.gammainc(values[0], 0, values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        facts: dict[str, bool] = {'commutative': True}
        if facts_of(self.args[1]).get('zero'):
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:uppergamma
class UpperGamma(DefinedFunction):
    """The upper incomplete gamma ``uppergamma(a, z)``, unevaluated
    (``uppergamma`` evaluates); a polar argument loses its branch factor
    as SymPy's rules say."""

    function_name = 'uppergamma'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import re
        from sympy_extras.functions.special.error_functions import Ei, erfc, expint
        a, z = args
        half = Rational(1, 2)
        minus_z = mul(Integer(-1), z)
        if isinstance(z, NaN):
            return nan
        if isinstance(z, Infinity):
            return ZERO
        if isinstance(z, Rational) and z.p == 0 and re(a).is_positive:
            return gamma(a)
        # the branch factor of a polar argument (C/f lowergamma)
        from sympy_extras.functions.elementary.complexes import unpolarify
        nx, n = z.extract_branch_factor()
        if a.is_integer is True and a.is_positive is True:
            nx = unpolarify(z)
            if z != nx:
                return uppergamma(a, nx)
        elif a.is_integer is True and a.is_nonpositive is True:
            if n != 0:
                return add(div(mul(Integer(-2), pi, I, n, pow(Integer(-1), mul(Integer(-1), a))),
                               factorial(mul(Integer(-1), a))), uppergamma(a, nx))
        elif n != 0:
            branch = exp(mul(Integer(2), pi, I, n, a))
            return add(mul(gamma(a), sub(ONE, branch)), mul(branch, uppergamma(a, nx)))
        if isinstance(a, Rational):
            if a.p == 0 and z.is_positive:
                return mul(Integer(-1), Ei(minus_z))
            if a == ONE:
                return exp(minus_z)
            if a == half:
                return mul(sqrt(pi), erfc(sqrt(z)))
            if a.q == 1 or a.q == 2:
                b = sub(a, ONE)
                if b.is_positive:
                    if a.q == 1:
                        terms = add(*[div(pow(z, Integer(k)), factorial(Integer(k))) for k in range(a.p)])
                        return mul(exp(minus_z), factorial(b), terms)
                    terms = add(*[div(mul(gamma(sub(Rational(-1, 2), Integer(k))), pow(minus_z, Integer(k))),
                                      gamma(sub(ONE, a))) for k in range(_range_to(sub(a, half)))])
                    return add(mul(gamma(a), erfc(sqrt(z))),
                               mul(pow(Integer(-1), sub(a, Rational(3, 2))), exp(minus_z), sqrt(z), terms))
                if isinstance(b, Rational) and b.q == 1:
                    return mul(expint(mul(Integer(-1), b), z), pow(unpolarify(z), add(b, ONE)))
                if a.q != 1:
                    terms = add(*[div(mul(pow(z, Integer(k)), gamma(a)), gamma(add(a, Integer(k + 1))))
                                  for k in range(_range_to(sub(half, a)))])
                    return sub(div(mul(pow(Integer(-1), sub(half, a)), pi, erfc(sqrt(z))), gamma(sub(ONE, a))),
                               mul(pow(z, a), exp(minus_z), terms))
        if a.is_zero and z.is_positive:
            return mul(Integer(-1), Ei(minus_z))
        if z.is_zero and re(a).is_positive:
            return gamma(a)
        return None

    def fdiff(self, argindex: int) -> Expr:
        a, z = self.args
        if argindex == 2:
            return mul(Integer(-1), exp(mul(Integer(-1), z)), pow(z, sub(a, ONE)))
        if argindex == 1:
            from sympy_extras.functions.special.hyper import meijerg
            return uppergamma(a, z) * log(z) + meijerg([], [ONE, ONE], [ZERO, ZERO, a], [], z)
        return DefinedFunction.fdiff(self, argindex)

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        a, z = self.args
        if z.is_zero or isinstance(z, NegativeInfinity):
            return None
        return uppergamma(conjugate(a), conjugate(z))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        try:
            return mpmath.mpc(mpmath.gammainc(values[0], values[1]))
        except (ValueError, ZeroDivisionError):
            return None

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def lowergamma(a: ExprLike, x: ExprLike) -> Expr:
    """``lowergamma(a, x)``, evaluated."""
    return LowerGamma.apply(to_expr(a), to_expr(x))


def uppergamma(a: ExprLike, z: ExprLike) -> Expr:
    """``uppergamma(a, z)``, evaluated."""
    return UpperGamma.apply(to_expr(a), to_expr(z))



# PORTED FROM SYMPY: sympy/functions/special/gamma_functions.py:multigamma
class MultiGamma(DefinedFunction):
    """The multivariate gamma function ``multigamma(x, p)``, unevaluated
    (``multigamma`` evaluates for a numerical order ``p``)."""

    unbranched = True

    function_name = 'multigamma'
    generic_nseries = True
    nargs = frozenset({2})
    call_nargs = frozenset({2})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        x, p = args
        if p.is_positive is False or p.is_integer is False:
            raise ValueError('Order parameter p must be positive integer.')
        order = _integer_valued(p)
        if order is None or not isinstance(p, Rational):
            return None  # SymPy: an unevaluated Product, not ported
        factors = [gamma(add(x, div(sub(ONE, Integer(k)), Integer(2)))) for k in range(1, order + 1)]
        return mul(pow(pi, div(mul(p, sub(p, ONE)), Integer(4))), *factors)

    def fdiff(self, argindex: int) -> Expr:
        return DefinedFunction.fdiff(self, argindex)  # SymPy: a Sum, not ported

    def _eval_conjugate(self) -> Optional[Expr]:
        from sympy_extras.functions.elementary.complexes import conjugate
        x, p = self.args
        return multigamma(conjugate(x), p)

    def _eval_facts(self) -> dict[str, bool]:
        from sympy_extras.core.logic import true
        from sympy_extras.core.relational import gt, le
        x, p = self.args
        y = mul(Integer(2), x)
        facts: dict[str, bool] = {'commutative': True}
        if (y.is_integer or _integer_valued(y) is not None) and le(y, sub(p, ONE)) == true:
            facts['real'] = False
        elif gt(y, sub(p, ONE)) == true or y.is_noninteger:
            facts['real'] = True
        return facts


def multigamma(x: ExprLike, p: ExprLike) -> Expr:
    """``multigamma(x, p)``, evaluated."""
    return MultiGamma.apply(to_expr(x), to_expr(p))
