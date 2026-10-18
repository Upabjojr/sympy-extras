"""Numbers: integers, rationals, floating point numbers, the constants.

The classes are ``Integer``, ``Rational`` (with ``Integer`` a subclass, as
in SymPy) and ``Float``, plus the infinities ``oo``, ``-oo``, ``zoo`` and
``nan``, the constants ``pi``, ``E`` and the imaginary unit ``I``.

A ``Rational`` is stored in lowest terms with a positive denominator, so
``Rational(2, 4)`` is ``1/2``; this is a representation, not an
evaluation, and ``Rational(2, 1)`` is a ``Rational`` instance, equal to
``Integer(2)`` (the evaluating functions always return an ``Integer`` for
an integral value). A ``Float`` holds its value exactly as a dyadic
rational together with its precision in bits, which is what a binary
floating point number is; its printing and the transcendental operations
go through mpmath.

The arithmetic of numbers (``number_add``, ``number_mul``) is written here;
powers of numbers are in :mod:`sympy_extras.core.power` because they can
produce non-numbers (``sqrt(8)`` is ``2*sqrt(2)``).

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Numbers*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.numbers
.. [2] F. Johansson et al., *mpmath: a Python library for
       arbitrary-precision floating-point arithmetic*, http://mpmath.org/
       (the representation of a float as mantissa and exponent).
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import TYPE_CHECKING, Optional, Union

import mpmath

from .basic import Basic
from .expr import AtomicExpr, Expr
from .sorting import ClassKey, SortKey

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Number', 'Rational', 'Integer', 'Float', 'Infinity', 'NegativeInfinity',
    'ComplexInfinity', 'NaN', 'NumberSymbol', 'Pi', 'Exp1', 'ImaginaryUnit', 'pi', 'E', 'I',
    'oo', 'zoo', 'nan', 'number_from_fraction', 'number_add', 'number_mul', 'FloatLike',
    'dps_to_prec', 'prec_to_dps']

#: what ``Float`` accepts
FloatLike = Union[int, float, str, Fraction, 'Rational', 'Float']


def dps_to_prec(dps: int) -> int:
    """The number of bits for ``dps`` decimal digits (mpmath's rule)."""
    return max(1, int(round((dps + 1) * 3.3219280948873626)))


def prec_to_dps(prec: int) -> int:
    """The number of decimal digits of ``prec`` bits (mpmath's rule)."""
    return max(1, int(round(prec / 3.3219280948873626) - 1))


class Number(AtomicExpr):
    """The base class of the numbers.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Rational, Float
    >>> Integer(2) + Rational(1, 2)
    5/2
    >>> Rational(1, 2) + Float(0.5)
    1.00000000000000
    """

    __slots__ = ()

    is_Number = True

    @classmethod
    def class_key(cls) -> ClassKey:
        return (1, 0, 'Number')

    def sort_key(self) -> SortKey:
        return (self.class_key(), (0, ()), self._sort_name(), (), self.sort_value())

    def _sort_name(self) -> str:
        return ''

    def sort_value(self) -> Fraction:
        """The value used to order numbers (``0`` for the infinities)."""
        return Fraction(0)

    def as_coeff_Mul(self, rational: bool = False) -> tuple[Number, Expr]:
        return (self, Integer(1))

    def as_coeff_Add(self) -> tuple[Number, Expr]:
        return (self, Integer(0))

    def number_add(self, other: Number) -> Number:
        """``self + other`` as a number."""
        return number_add(self, other)

    def number_mul(self, other: Number) -> Number:
        """``self * other`` as a number."""
        return number_mul(self, other)

    @property
    def is_finite_number(self) -> bool:
        """Whether the number is a rational or a float."""
        return isinstance(self, (Rational, Float))

    def exact_value(self) -> Optional[Fraction]:
        """The value of a rational or a float, ``None`` for the others."""
        return None

    def _sign(self) -> int:
        """``-1``, ``0`` or ``1``; ``0`` for the non-real numbers."""
        return 0

    def __bool__(self) -> bool:
        return True


class Rational(Number):
    """A rational number ``p/q`` in lowest terms.

    Examples
    ========

    >>> from sympy_extras.core import Rational
    >>> Rational(2, 4)
    1/2
    >>> Rational(2, 4).p, Rational(2, 4).q
    (1, 2)
    >>> Rational(3, 1) == 3
    True
    """

    __slots__ = ('p', 'q')

    is_Rational = True

    def __init__(self, p: int, q: int = 1) -> None:
        if q == 0:
            raise ZeroDivisionError("Rational(%s, 0)" % (p,))
        if q < 0:
            p, q = -p, -q
        g = gcd(p, q)
        if g > 1:
            p, q = p // g, q // g
        self.p: int = p
        self.q: int = q
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash(Fraction(self.p, self.q))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Rational):
            return self.p == other.p and self.q == other.q
        if isinstance(other, (int, Fraction)) and not isinstance(other, bool):
            return Fraction(self.p, self.q) == other
        if isinstance(other, Basic):
            return False
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @property
    def is_Integer(self) -> bool:
        return self.q == 1

    def __str__(self) -> str:
        if self.q == 1:
            return str(self.p)
        return '%d/%d' % (self.p, self.q)

    def _sympystr(self, printer: StrPrinter) -> str:
        return str(self)

    def sort_value(self) -> Fraction:
        return Fraction(self.p, self.q)

    def exact_value(self) -> Optional[Fraction]:
        return Fraction(self.p, self.q)

    def as_fraction(self) -> Fraction:
        """The value as a Python :class:`~fractions.Fraction`."""
        return Fraction(self.p, self.q)

    def _sign(self) -> int:
        return (self.p > 0) - (self.p < 0)

    def as_base_exp(self) -> tuple[Expr, Expr]:
        if self.p == 1 and self.q > 1:
            return (Integer(self.q), Integer(-1))
        return (self, Integer(1))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        return (Integer(self.p), Integer(self.q))

    def __int__(self) -> int:
        return int(Fraction(self.p, self.q))

    def __float__(self) -> float:
        return self.p / self.q


class Integer(Rational):
    """An integer.

    Examples
    ========

    >>> from sympy_extras.core import Integer
    >>> Integer(2)**10
    1024
    >>> Integer(7)/2
    7/2
    """

    __slots__ = ()

    def __init__(self, p: int) -> None:
        Rational.__init__(self, p, 1)

    @property
    def is_Integer(self) -> bool:
        return True

    def __index__(self) -> int:
        return self.p


def number_from_fraction(value: Fraction) -> Rational:
    """The ``Integer`` or ``Rational`` with the value of a fraction."""
    if value.denominator == 1:
        return Integer(value.numerator)
    return Rational(value.numerator, value.denominator)


def _round_to_prec(value: Fraction, prec: int) -> Fraction:
    """``value`` rounded to ``prec`` significant bits (round half to even)."""
    if value == 0:
        return value
    sign = -1 if value < 0 else 1
    p, q = abs(value.numerator), value.denominator
    # 2**(e) <= p/q < 2**(e + 1)
    e = p.bit_length() - q.bit_length()
    if (p << max(0, -e)) < (q << max(0, e)):
        e -= 1
    shift = prec - 1 - e
    if shift >= 0:
        num, den = p << shift, q
    else:
        num, den = p, q << (-shift)
    mantissa, remainder = divmod(num, den)
    twice = 2 * remainder
    if twice > den or (twice == den and mantissa % 2 == 1):
        mantissa += 1
    result = Fraction(mantissa) * Fraction(2) ** (-shift)
    return sign * result


def _significant_digits(text: str) -> int:
    mantissa = text.strip().lower().split('e')[0].lstrip('+-').replace('.', '').lstrip('0')
    return len(mantissa)


class Float(Number):
    """A floating point number of a given precision.

    The value is stored exactly (a dyadic rational) with the precision in
    bits; ``dps`` is the number of decimal digits.

    Examples
    ========

    >>> from sympy_extras.core import Float
    >>> Float(2.5)
    2.50000000000000
    >>> Float('0.1', 30)
    0.100000000000000000000000000000
    >>> Float(2.5) + 1
    3.50000000000000
    """

    __slots__ = ('_value', '_prec')

    is_Float = True

    def __init__(self, value: FloatLike, dps: Optional[int] = None, precision: Optional[int] = None) -> None:
        if precision is not None:
            prec = precision
        elif dps is not None:
            prec = dps_to_prec(dps)
        elif isinstance(value, Float):
            prec = value._prec
        elif isinstance(value, str):
            prec = dps_to_prec(max(15, _significant_digits(value)))
        else:
            prec = 53
        if isinstance(value, Float):
            exact = value._value
        elif isinstance(value, Rational):
            exact = value.as_fraction()
        elif isinstance(value, str):
            with mpmath.workprec(prec):
                exact = _mpf_to_fraction(mpmath.mpf(value))
        elif isinstance(value, bool):
            exact = Fraction(int(value))
        else:
            exact = Fraction(value)
        self._value: Fraction = _round_to_prec(exact, prec)
        self._prec: int = prec
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __hash__(self) -> int:
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Float):
            return self._value == other._value
        if isinstance(other, float):
            return self._value == Fraction(other)
        if isinstance(other, Basic):
            return False
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    @property
    def precision(self) -> int:
        """The precision in bits."""
        return self._prec

    @property
    def dps(self) -> int:
        """The precision in decimal digits."""
        return prec_to_dps(self._prec)

    def as_fraction(self) -> Fraction:
        """The exact value as a Python :class:`~fractions.Fraction`."""
        return self._value

    def exact_value(self) -> Optional[Fraction]:
        return self._value

    def sort_value(self) -> Fraction:
        return self._value

    def _sign(self) -> int:
        return (self._value > 0) - (self._value < 0)

    def __str__(self) -> str:
        with mpmath.workprec(self._prec):
            v = mpmath.mpf(self._value.numerator) / self._value.denominator
            return str(mpmath.nstr(v, self.dps, strip_zeros=False))

    def short_str(self) -> str:
        """The value without trailing zeros, as printed inside an expression."""
        with mpmath.workprec(self._prec):
            v = mpmath.mpf(self._value.numerator) / self._value.denominator
            return str(mpmath.nstr(v, self.dps))

    def _sympystr(self, printer: StrPrinter) -> str:
        return str(self)

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        return (self, Integer(1))


def _mpf_to_fraction(x: mpmath.mpf) -> Fraction:
    """The exact value of an mpmath float."""
    sign, man, exp, _ = x._mpf_
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if int(sign) else value


def _float_from_fraction(value: Fraction, prec: int) -> Float:
    return Float(number_from_fraction(value), precision=prec)


class Infinity(Number):
    """Positive infinity, ``oo``."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    def _sort_name(self) -> str:
        return 'oo'

    def __str__(self) -> str:
        return 'oo'

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'oo'

    def _sign(self) -> int:
        return 1

    def __float__(self) -> float:
        return float('inf')


class NegativeInfinity(Number):
    """Negative infinity, ``-oo``."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    def _sort_name(self) -> str:
        return '-oo'

    def __str__(self) -> str:
        return '-oo'

    def _sympystr(self, printer: StrPrinter) -> str:
        return '-oo'

    def _sign(self) -> int:
        return -1

    def as_coeff_Mul(self, rational: bool = False) -> tuple[Number, Expr]:
        return (Integer(-1), oo)

    def __float__(self) -> float:
        return float('-inf')


class ComplexInfinity(Number):
    """The point at infinity of the Riemann sphere, ``zoo``."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    def _sort_name(self) -> str:
        return 'zoo'

    def __str__(self) -> str:
        return 'zoo'

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'zoo'


class NaN(Number):
    """Not a number, the result of ``oo - oo``."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    def _sort_name(self) -> str:
        return 'nan'

    def __str__(self) -> str:
        return 'nan'

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'nan'

    def __float__(self) -> float:
        return float('nan')


class NumberSymbol(AtomicExpr):
    """A named real constant such as ``pi`` and ``E``."""

    __slots__ = ()

    is_NumberSymbol = True

    def __init__(self) -> None:
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    @property
    def name(self) -> str:
        return 'constant'

    def __str__(self) -> str:
        return self.name

    def _sympystr(self, printer: StrPrinter) -> str:
        return self.name

    def __float__(self) -> float:
        from .evalf import to_float
        return to_float(self)

    def exact_value(self) -> Optional[Fraction]:
        return None


class Pi(NumberSymbol):
    """The constant pi."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return 'pi'


class Exp1(NumberSymbol):
    """The base of the natural logarithm, ``E``."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return 'E'


class ImaginaryUnit(AtomicExpr):
    """The imaginary unit ``I``, ``I**2 == -1``."""

    __slots__ = ()

    def __init__(self) -> None:
        Basic.__init__(self)

    def _hashable_content(self) -> tuple[Basic, ...]:
        return ()

    def __str__(self) -> str:
        return 'I'

    def _sympystr(self, printer: StrPrinter) -> str:
        return 'I'

    def as_base_exp(self) -> tuple[Expr, Expr]:
        return (Integer(-1), Rational(1, 2))


pi = Pi()
E = Exp1()
I = ImaginaryUnit()
oo = Infinity()
zoo = ComplexInfinity()
nan = NaN()


def number_add(a: Number, b: Number) -> Number:
    """The sum of two numbers.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Rational, Float, oo
    >>> from sympy_extras.core.numbers import number_add
    >>> number_add(Rational(1, 3), Rational(2, 3))
    1
    >>> number_add(oo, Integer(1))
    oo
    >>> number_add(oo, -oo)
    nan
    """
    if isinstance(a, NaN) or isinstance(b, NaN):
        return nan
    if isinstance(a, ComplexInfinity) or isinstance(b, ComplexInfinity):
        if a.is_finite_number and isinstance(b, ComplexInfinity):
            return zoo
        if b.is_finite_number and isinstance(a, ComplexInfinity):
            return zoo
        return nan
    if isinstance(a, (Infinity, NegativeInfinity)) or isinstance(b, (Infinity, NegativeInfinity)):
        sa, sb = a._sign() if not a.is_finite_number else 0, b._sign() if not b.is_finite_number else 0
        if sa and sb and sa != sb:
            return nan
        return oo if (sa or sb) > 0 else NegativeInfinity()
    va, vb = a.exact_value(), b.exact_value()
    if va is None or vb is None:
        raise TypeError("cannot add %s and %s" % (a, b))
    if isinstance(a, Float) or isinstance(b, Float):
        prec = max(a._prec if isinstance(a, Float) else 0, b._prec if isinstance(b, Float) else 0)
        return _float_from_fraction(va + vb, prec)
    return number_from_fraction(va + vb)


def number_mul(a: Number, b: Number) -> Number:
    """The product of two numbers.

    Examples
    ========

    >>> from sympy_extras.core import Integer, Rational, Float, oo
    >>> from sympy_extras.core.numbers import number_mul
    >>> number_mul(Rational(2, 3), Rational(3, 4))
    1/2
    >>> number_mul(oo, Integer(-2))
    -oo
    >>> number_mul(oo, Integer(0))
    nan
    """
    if isinstance(a, NaN) or isinstance(b, NaN):
        return nan
    if not (a.is_finite_number and b.is_finite_number):
        if a.is_finite_number and a._sign() == 0 or b.is_finite_number and b._sign() == 0:
            return nan
        if isinstance(a, ComplexInfinity) or isinstance(b, ComplexInfinity):
            return zoo
        sign = a._sign() * b._sign()
        return oo if sign > 0 else NegativeInfinity()
    va, vb = a.exact_value(), b.exact_value()
    if va is None or vb is None:
        raise TypeError("cannot multiply %s and %s" % (a, b))
    if isinstance(a, Float) or isinstance(b, Float):
        prec = max(a._prec if isinstance(a, Float) else 0, b._prec if isinstance(b, Float) else 0)
        return _float_from_fraction(va * vb, prec)
    return number_from_fraction(va * vb)
