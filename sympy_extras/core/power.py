"""Powers: the class ``Pow`` and the evaluating function ``pow``.

``Pow(base, exp)`` is an unevaluated node. The function ``pow(base, exp)``
applies the rules of SymPy's ``Pow`` [1]_: ``b**0`` is ``1``, ``b**1`` is
``b``, ``1**b`` is ``1``, a power of a power combines when the outer
exponent is an integer or the inner one is a rational in ``(-1, 1)`` and
the outer one real, an integer power of a product is distributed, a
positive numerical coefficient is taken out of a product under a rational
exponent, and the powers of numbers are evaluated: integer powers of
rationals exactly, rational powers of rationals by extracting the perfect
powers of the prime factors (``sqrt(12)`` is ``2*sqrt(3)``, ``(1/3)**(1/2)``
is ``sqrt(3)/3``), powers of ``-1`` reduced to an exponent in ``(0, 1]``
(``(-1)**(3/2)`` is ``-I``), floats through mpmath, and the infinities as
SymPy does.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Pow*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.power.Pow
"""
from __future__ import annotations

from fractions import Fraction
from math import floor
from typing import TYPE_CHECKING, Optional

import mpmath
from sympy.ntheory import factorint

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, factors_of, to_expr
from .mul import Mul, mul
from .numbers import (ComplexInfinity, Exp1, Float, ImaginaryUnit, Infinity, Integer, I, NaN,
    NegativeInfinity, Number, Rational, nan, number_from_fraction, oo, zoo)
from .parameters import global_parameters
from .sorting import ClassKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Pow', 'pow', 'sqrt', 'cbrt', 'root', 'number_pow', 'integer_root_parts', 'minus_one_pow']

#: numbers below this size are factored completely for the extraction of
#: perfect powers; larger ones only by trial division up to ``_TRIAL``
_FULL_FACTORING_BITS = 64
_TRIAL = 2**16


class Pow(Expr):
    """An unevaluated power.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Pow
    >>> x = Symbol('x')
    >>> p = Pow(x, 0)
    >>> p, p.args, p.doit()
    (x**0, (x, 0), 1)
    >>> Pow(x, 2).base, Pow(x, 2).exp
    (x, 2)
    """

    __slots__ = ('_base', '_exp')

    is_Pow = True

    def __init__(self, base: ExprLike, exp: ExprLike) -> None:
        self._base: Expr = to_expr(base)
        self._exp: Expr = to_expr(exp)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self._base, self._exp)

    @property
    def base(self) -> Expr:
        return self._base

    @property
    def exp(self) -> Expr:
        return self._exp

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 2, 'Pow')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        b, e = args
        if not (isinstance(b, Expr) and isinstance(e, Expr)):
            raise TypeError("a power of expressions is expected")
        return pow(b, e)

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import pow_facts
        return pow_facts(self._base, self._exp)

    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        e = self._exp
        if isinstance(e, Rational) and e.q == 1 and e.p > 0 and isinstance(self._base, Add):
            c, p = self._base.as_content_primitive(radical=radical, clear=clear)
            value = c.exact_value()
            if value is not None and value != 1:
                return (number_from_fraction(value ** e.p), pow(p, e))
        return (Integer(1), self)

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # (1/q)**e is seen as q**(-e), as in SymPy
        if isinstance(self._base, Rational) and self._base.p == 1 and self._base.q > 1:
            return (Integer(self._base.q), mul(Integer(-1), self._exp))
        return (self._base, self._exp)

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_pow(self._base, self._exp)

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        e = self._exp
        if isinstance(e, Rational) and e.p < 0:
            n, d = self._base.as_numer_denom()
            minus = Rational(-e.p, e.q)
            return (pow(d, minus), pow(n, minus))
        if isinstance(e, Rational):
            n, d = self._base.as_numer_denom()
            return (pow(n, e), pow(d, e))
        return (self, Integer(1))


def pow(b: ExprLike, e: ExprLike) -> Expr:
    """The evaluated power ``b**e``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, pow, Rational, Integer
    >>> x, y = symbols('x y')
    >>> pow(x, 0), pow(x, 1), pow(1, x)
    (1, x, 1)
    >>> pow(Integer(2), 10), pow(Integer(8), Rational(1, 3)), pow(Integer(12), Rational(1, 2))
    (1024, 2, 2*sqrt(3))
    >>> pow(pow(x, 2), 3), pow(2*x, 2), pow(4*x, Rational(1, 2))
    (x**6, 4*x**2, 2*sqrt(x))
    >>> pow(Integer(-1), Rational(1, 2)), pow(Integer(-8), Rational(1, 3))
    (I, 2*(-1)**(1/3))
    """
    base, exp = to_expr(b), to_expr(e)
    if not global_parameters.evaluate:
        return Pow(base, exp)
    if base.is_Order or exp.is_Order:
        import sympy
        from .interop import expr_from_sympy, to_sympy
        return expr_from_sympy(sympy.Pow(to_sympy(base), to_sympy(exp)))
    if isinstance(base, Number) and isinstance(exp, Number):
        return number_pow(base, exp)
    if isinstance(exp, Number):
        if isinstance(exp, NaN):
            return nan
        if isinstance(exp, Rational):
            if exp.p == 0:
                return Integer(1)
            if exp.p == 1 and exp.q == 1:
                return base
    if isinstance(base, Number):
        if isinstance(base, NaN):
            return nan
        if isinstance(base, Rational) and base.p == 1 and base.q == 1:
            return Integer(1)
        if isinstance(base, Rational) and base.p == 0:
            if exp.is_positive:
                return Integer(0)
            if exp.is_negative:
                return zoo
            if exp.is_imaginary:
                return nan
            return Pow(base, exp)
    if isinstance(base, Exp1):
        # E**a is exp(a), evaluated by SymPy's exp (E**0 is 1, E**(I*pi) is -1)
        import sympy
        from .function import apply_sympy
        return apply_sympy(sympy.exp, exp)
    if isinstance(base, ImaginaryUnit) and isinstance(exp, Rational):
        if exp.q == 1:
            return _imaginary_unit_power(exp.p)
        reduced = exp.as_fraction() % 2
        k = int((exp.as_fraction() - reduced) / 2)
        power = Pow(I, number_from_fraction(reduced))
        return power if k % 2 == 0 else Mul(Integer(-1), power)
    if isinstance(base, Add) and isinstance(exp, Rational) and exp.p == -1 and exp.q == 1:
        inverse = _gaussian_inverse(base)
        if inverse is not None:
            return inverse
    if base.is_Pow or _is_exp(base):
        inner_base, inner_exp = base.as_base_exp()
        # (b**p)**q is b**(p*q) for an integer q, and for -1 < p < 1 (then
        # Im(p*Log(b)) stays in (-pi, pi), whatever q is)
        combine = exp.is_integer is True
        if not combine and isinstance(inner_exp, Rational) and -1 < inner_exp.as_fraction() < 1:
            combine = True
        if not combine and inner_base.is_positive is True and exp.is_real is True and inner_exp.is_real is True:
            combine = True
        if combine:
            return pow(inner_base, mul(inner_exp, exp))
    if isinstance(base, Mul):
        distributed = _power_of_product(base, exp)
        if distributed is not None:
            return distributed
    if base.is_commutative is False and not (isinstance(exp, Rational) and exp.q == 1):
        return Pow(base, exp)
    return Pow(base, exp)


def _is_exp(e: Expr) -> bool:
    """Whether ``e`` is an ``exp(...)`` of SymPy (a power of ``E``)."""
    import sympy
    from .function import AppliedSymPyFunction
    return isinstance(e, AppliedSymPyFunction) and e.sympy_class is sympy.exp


def _imaginary_unit_power(n: int) -> Expr:
    return (Integer(1), I, Integer(-1), mul(Integer(-1), I))[n % 4]


def _gaussian_inverse(base: Add) -> Optional[Expr]:
    """``1/(a + b*I)`` as ``(a - b*I)/(a**2 + b**2)`` for rational ``a``, ``b``."""
    if len(base.args) != 2:
        return None
    real: Optional[Rational] = None
    imaginary: Optional[Rational] = None
    for term in base.args:
        if isinstance(term, Rational):
            real = term
        elif isinstance(term, ImaginaryUnit):
            imaginary = Integer(1)
        elif isinstance(term, Mul) and len(term.args) == 2 and isinstance(term.args[0], Rational) \
                and isinstance(term.args[1], ImaginaryUnit):
            imaginary = term.args[0]
    if real is None or imaginary is None:
        return None
    a, b = real.as_fraction(), imaginary.as_fraction()
    norm = number_from_fraction(1 / (a * a + b * b))
    conjugate = Add(real, Mul(number_from_fraction(-b), I)) if b != -1 else Add(real, I)
    return Mul(norm, conjugate)


def _power_of_product(base: Mul, exp: Expr) -> Optional[Expr]:
    """``(a*b)**e`` distributed when the rules allow it."""
    commutative = [f for f in base.args if f.is_commutative is not False]
    noncommutative = [f for f in base.args if f.is_commutative is False]
    if isinstance(exp, Rational) and exp.q == 1:
        parts = [pow(f, exp) for f in commutative]
        if noncommutative:
            parts.append(Pow(mul(*noncommutative), exp))
        return mul(*parts)
    if noncommutative or not isinstance(exp, Rational):
        return None
    coefficient, rest = base.as_coeff_Mul()
    positive = [f for f in factors_of(rest) if f.is_positive is True]
    others = [f for f in factors_of(rest) if f.is_positive is not True]
    if coefficient.is_finite_number and coefficient._sign() != 0 and not (
            isinstance(coefficient, Rational) and coefficient.p == 1 and coefficient.q == 1):
        if coefficient._sign() > 0 or not others:
            parts = [pow(coefficient, exp)] + [pow(f, exp) for f in positive]
            parts.append(pow(mul(*others), exp) if others else Integer(1))
            return mul(*parts)
        negated = coefficient.number_mul(Integer(-1))
        if isinstance(negated, Rational) and negated.p == 1 and negated.q == 1 and not positive:
            return None  # (-x)**e stays
        parts = [pow(negated, exp)] + [pow(f, exp) for f in positive]
        parts.append(pow(mul(Integer(-1), *others), exp))
        return mul(*parts)
    if positive and others:
        return mul(*[pow(f, exp) for f in positive], pow(mul(*others), exp))
    if positive and not others:
        return mul(*[pow(f, exp) for f in positive])
    return None


def sqrt(x: ExprLike) -> Expr:
    """The square root ``x**(1/2)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> x = Symbol('x')
    >>> sqrt(x), sqrt(8), sqrt(-4), sqrt(x**2)
    (sqrt(x), 2*sqrt(2), 2*I, sqrt(x**2))
    """
    return pow(x, Rational(1, 2))


def cbrt(x: ExprLike) -> Expr:
    """The cube root ``x**(1/3)``."""
    return pow(x, Rational(1, 3))


def root(x: ExprLike, n: int) -> Expr:
    """The ``n``-th root ``x**(1/n)``."""
    return pow(x, Rational(1, n))


# ----------------------------------------------------------------------
# powers of numbers


def number_pow(b: Number, e: Number) -> Expr:
    """``b**e`` for two numbers (the result need not be a number).

    Examples
    ========

    >>> from sympy_extras.core import Integer, Rational, Float, oo
    >>> from sympy_extras.core.power import number_pow
    >>> number_pow(Rational(2, 3), Integer(-2))
    9/4
    >>> number_pow(Integer(8), Rational(-1, 2))
    sqrt(2)/4
    >>> number_pow(Float(2), Rational(1, 2))
    1.41421356237310
    >>> number_pow(Integer(2), oo), number_pow(Integer(0), Integer(0))
    (oo, 1)
    """
    if isinstance(e, NaN):
        return nan
    if isinstance(e, Rational) and e.p == 0:
        return Integer(1)
    if isinstance(b, NaN):
        return nan
    if isinstance(e, Rational) and e.p == 1 and e.q == 1:
        return b
    if not b.is_finite_number or not e.is_finite_number:
        return _infinite_pow(b, e)
    vb, ve = b.exact_value(), e.exact_value()
    if vb is None or ve is None:
        return nan
    if vb == 1 and not isinstance(b, Float):
        return Integer(1)
    if vb == 0:
        if ve > 0:
            return Integer(0) if not (isinstance(b, Float) and isinstance(e, Float)) else Float(0)
        if global_parameters.divide_error:
            raise ZeroDivisionError("division by zero")
        return zoo
    if isinstance(b, Float) or isinstance(e, Float):
        return _float_pow(b, e)
    assert isinstance(b, Rational) and isinstance(e, Rational)
    if e.q == 1:
        return number_from_fraction(vb ** e.p)
    return _rational_pow(b, e)


def _infinite_pow(b: Number, e: Number) -> Expr:
    """Powers involving ``oo``, ``-oo`` or ``zoo``."""
    if isinstance(e, Rational) or isinstance(e, Float):
        sign = e._sign()
        if isinstance(b, Infinity):
            return oo if sign > 0 else Integer(0)
        if isinstance(b, NegativeInfinity):
            if sign < 0:
                return Integer(0)
            if isinstance(e, Rational) and e.q == 1:
                return oo if e.p % 2 == 0 else NegativeInfinity()
            if isinstance(e, Float) and e.as_fraction().denominator == 1:
                return oo if int(e.as_fraction()) % 2 == 0 else NegativeInfinity()
            return mul(oo, number_pow(Integer(-1), e))
        if isinstance(b, ComplexInfinity):
            return zoo if sign > 0 else Integer(0)
        return nan
    if isinstance(e, ComplexInfinity):
        return nan
    if not b.is_finite_number:
        if isinstance(b, Infinity):
            return oo if isinstance(e, Infinity) else Integer(0)
        if isinstance(b, NegativeInfinity):
            return nan if isinstance(e, Infinity) else Integer(0)
        return nan
    vb = b.exact_value()
    if vb is None:
        return nan
    if isinstance(e, Infinity):
        if vb > 1:
            return oo
        if vb == 1 or vb == -1:
            return nan
        if -1 < vb < 1:
            return Integer(0)
        return zoo
    if isinstance(e, NegativeInfinity):
        if vb > 1 or vb < -1:
            return Integer(0)
        if vb == 1 or vb == -1:
            return nan
        if vb == 0:
            return zoo
        return oo if vb > 0 else zoo
    return nan


def _float_pow(b: Number, e: Number) -> Expr:
    prec = max(b.precision if isinstance(b, Float) else 0, e.precision if isinstance(e, Float) else 0)
    vb, ve = b.exact_value(), e.exact_value()
    if vb is None or ve is None:
        return nan
    if ve.denominator == 1 and not isinstance(e, Float):
        return Float(number_from_fraction(vb ** ve.numerator), precision=prec)
    with mpmath.workprec(prec + 10):
        base = mpmath.mpf(vb.numerator) / vb.denominator
        exponent = mpmath.mpf(ve.numerator) / ve.denominator
        value = mpmath.power(base, exponent)
        real = _mpf_to_fraction(mpmath.re(value))
        imag = _mpf_to_fraction(mpmath.im(value))
    result: Expr = Float(number_from_fraction(real), precision=prec)
    if imag != 0:
        result = add(result, mul(Float(number_from_fraction(imag), precision=prec), I))
    return result


def _mpf_to_fraction(x: mpmath.mpf) -> Fraction:
    sign, man, exp, _ = x._mpf_
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if int(sign) else value


def minus_one_pow(e: Fraction) -> Expr:
    """``(-1)**e`` with the exponent reduced to ``(0, 1]``, as SymPy."""
    reduced = e - 2 * floor((e + 1) / 2)  # in (-1, 1]
    sign = 1
    if reduced < 0:
        reduced += 1
        sign = -1
    if reduced == 0:
        return Integer(sign)
    if reduced == 1:
        return Integer(-sign)
    if reduced == Fraction(1, 2):
        return I if sign > 0 else mul(Integer(-1), I)
    base = Pow(Integer(-1), number_from_fraction(reduced))
    return base if sign > 0 else Mul(Integer(-1), base)


def integer_root_parts(p: int, e: Fraction) -> tuple[int, int, int]:
    """``p**e`` for a positive integer ``p`` and ``0 < e < 1`` as
    ``(r, t, g)`` with ``p**e == r * t**(g/n)``, ``n`` the denominator of
    ``e``, ``g/n < 1`` and ``t**g`` free of ``n``-th powers in the
    factorisation used (complete below 64 bits, by trial division above).

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.core.power import integer_root_parts
    >>> integer_root_parts(12, Fraction(1, 2))
    (2, 3, 1)
    >>> integer_root_parts(48, Fraction(1, 3))
    (2, 6, 1)
    >>> integer_root_parts(32, Fraction(1, 6))
    (1, 2, 5)
    >>> integer_root_parts(72, Fraction(1, 3))
    (2, 3, 2)
    """
    from math import gcd
    m, n = e.numerator, e.denominator
    if p.bit_length() <= _FULL_FACTORING_BITS:
        factors = factorint(p)
    else:
        factors = factorint(p, limit=_TRIAL)
    r = 1
    remainders: list[tuple[int, int]] = []
    for prime, multiplicity in factors.items():
        q, rem = divmod(int(multiplicity) * m, n)
        r *= int(prime) ** q
        if rem:
            remainders.append((int(prime), rem))
    g = 0
    for _, rem in remainders:
        g = gcd(g, rem)
    t = 1
    for prime, rem in remainders:
        t *= prime ** (rem // g)
    return (r, t, g) if remainders else (r, 1, 0)


def _rational_pow(b: Rational, e: Rational) -> Expr:
    """``b**e`` for a non-integer rational exponent."""
    from .mul import Mul
    vb, ve = b.as_fraction(), e.as_fraction()
    if vb == 1:
        return Integer(1)
    if ve < 0:
        # b**(-m/n) = b**(k - m/n) * b**(-k) with k = ceil(m/n): valid on the
        # principal branch for every b, while (1/b)**(m/n) is not for b < 0
        k = -floor(ve)
        return mul(_rational_pow(b, Rational(k * ve.denominator + ve.numerator, ve.denominator)),
                   number_from_fraction(vb ** (-k)))
    if b.q > 1:
        k = -floor(-ve)
        parts = [_rational_pow(Integer(b.p), e) if b.p != 1 else Integer(1),
                 _rational_pow(Integer(b.q), number_from_fraction(k - ve)) if k != ve else Integer(1),
                 Rational(1, b.q ** k)]
        return mul(*parts)
    integer_part = floor(ve)
    fraction = ve - integer_part
    r, t, g = integer_root_parts(abs(b.p), fraction)
    coefficient = Integer(b.p ** integer_part * r)
    if vb > 0:
        if t == 1:
            return coefficient
        radical = Pow(Integer(t), number_from_fraction(Fraction(g, fraction.denominator)))
        if coefficient.p == 1:
            return radical
        return Mul(coefficient, radical)
    # a negative base: (-1)**f is I for a half-integer exponent, otherwise
    # the sign stays with the n-th power free part when it is not a perfect
    # power (SymPy writes (-2)**(1/3) and 2*(-1)**(1/3) but (-1)**(1/3)*2**(2/3))
    unit = minus_one_pow(fraction)
    radical_parts: list[Expr]
    if isinstance(unit, Pow) and g == fraction.numerator and t > 1:
        radical_parts = [Pow(Integer(-t), number_from_fraction(fraction))]
    elif t == 1:
        radical_parts = [unit]
    else:
        radical_parts = [unit, Pow(Integer(t), number_from_fraction(Fraction(g, fraction.denominator)))]
    return mul(coefficient, *radical_parts)
