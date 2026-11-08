"""The hyperbolic functions on the typed core: ``Sinh``, ``Cosh``,
``Tanh`` (unevaluated nodes) and ``sinh``, ``cosh``, ``tanh``
(evaluating).

The evaluation rules, derivatives and facts are ported from SymPy's
``sympy.functions.elementary.hyperbolic``: the values at zero and at the
infinities, odd and even symmetry, the exchange with the trigonometric
functions on imaginary arguments, the multiples of ``I*pi`` peeled off
sums; the reciprocal functions ``Coth``, ``Sech``, ``Csch`` through their
partners. Not ported yet (the node stays): the compositions with the
inverse functions, and the facts which need ``as_real_imag``.

Examples
========

>>> from sympy_extras.core import Symbol, I, pi, oo
>>> from sympy_extras.functions.elementary.hyperbolic import sinh, cosh, tanh
>>> x = Symbol('x')
>>> sinh(0), cosh(0), tanh(oo), sinh(-x), cosh(-x), sinh(I*x), cosh(I*pi)
(0, 1, 1, -sinh(x), cosh(x), I*sin(x), -1)
>>> sinh(x + I*pi), sinh(x).diff(x), tanh(x).diff(x)
(-sinh(x), cosh(x), 1 - tanh(x)**2)
>>> from sympy_extras.functions.elementary.hyperbolic import coth, sech, csch
>>> tanh(x + I*pi/2), coth(0), sech(-x), csch(-x), coth(I*x)
(coth(x), zoo, sech(x), -csch(x), -I*cot(x))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: unevaluating classes Sinh/Cosh/Tanh with evaluating functions,
# strict typing, the subset of the rules the typed core supports.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.assumptions import facts_of
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, ExpandHints, expand_with_hints
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (ComplexInfinity, I, Infinity, Integer, NaN, NegativeInfinity, Number, Rational,
                                       nan, number_from_fraction, oo, pi, zoo)
from sympy_extras.core.power import pow, sqrt

from .trigonometric import imaginary_coefficient

__all__ = ['Sinh', 'Cosh', 'Tanh', 'Coth', 'Sech', 'Csch', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
           'Asinh', 'Acosh', 'Atanh', 'Acoth', 'Asech', 'Acsch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch',
           'peeloff_ipi']


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:_peeloff_ipi
def peeloff_ipi(arg: Expr) -> tuple[Expr, Fraction]:
    """``(rest, m)`` with ``arg = rest + m*I*pi``, ``m`` a multiple of 1/2,
    from the first term of the sum which is a rational multiple of
    ``I*pi``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, I, pi
    >>> from sympy_extras.functions.elementary.hyperbolic import peeloff_ipi
    >>> x = Symbol('x')
    >>> peeloff_ipi(x + I*pi/2), peeloff_ipi(x + 2*I*pi/3)
    ((x, Fraction(1, 2)), (x + I*pi/6, Fraction(1, 2)))
    """
    ipi = mul(pi, I)
    k: Optional[Fraction] = None
    for term in (arg.args if isinstance(arg, Add) else (arg,)):
        if term == ipi:
            k = Fraction(1)
            break
        if isinstance(term, Mul):
            first, rest = term.as_two_terms()
            if rest == ipi and isinstance(first, Rational):
                k = first.as_fraction()
                break
    if k is None:
        return arg, Fraction(0)
    m2 = k - k % Fraction(1, 2)
    return add(arg, mul(Integer(-1), number_from_fraction(m2), ipi)), m2


def _odd_facts(arg: Expr) -> dict[str, bool]:
    a = facts_of(arg)
    facts: dict[str, bool] = {'commutative': True}
    if a.get('extended_real'):
        facts['extended_real'] = True
        if a.get('positive') is not None:
            facts['positive'] = a['positive']
        if a.get('negative') is not None:
            facts['negative'] = a['negative']
    if a.get('real'):
        facts['real'] = True
    return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sinh
class Sinh(DefinedFunction):
    """The hyperbolic sine, unevaluated (``sinh`` evaluates)."""

    unbranched = True

    function_name = 'sinh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import sin
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return oo
            if isinstance(arg, NegativeInfinity):
                return mul(Integer(-1), oo)
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(0)
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return mul(I, sin(i_coeff))
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi, I)
                return add(mul(sinh(multiple), cosh(rest)), mul(cosh(multiple), sinh(rest)))
        if arg.is_zero:
            return Integer(0)
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Sinh, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        from .trigonometric import cos, sin
        arg = self.args[0]
        if arg.is_extended_real:
            return (expand_with_hints(self, hints, complex=False) if deep else self, Integer(0))
        parts = as_real_imag(expand_with_hints(arg, hints) if deep else arg)
        if parts is None:
            return None
        r, i = parts
        return (mul(sinh(r), cos(i)), mul(cosh(r), sin(i)))

    def fdiff(self, argindex: int) -> Expr:
        return cosh(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.sinh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _odd_facts(self.args[0])
        if facts_of(self.args[0]).get('finite'):
            facts['finite'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:cosh
class Cosh(DefinedFunction):
    """The hyperbolic cosine, unevaluated (``cosh`` evaluates)."""

    unbranched = True

    function_name = 'cosh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import cos
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return oo
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(1)
            if arg.is_negative:
                return cls.apply(mul(Integer(-1), arg))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            return cos(i_coeff)
        if arg.could_extract_minus_sign():
            return cls.apply(mul(Integer(-1), arg))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                multiple = mul(number_from_fraction(m), pi, I)
                return add(mul(cosh(multiple), cosh(rest)), mul(sinh(multiple), sinh(rest)))
        if arg.is_zero:
            return Integer(1)
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Cosh, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        from .trigonometric import cos, sin
        arg = self.args[0]
        if arg.is_extended_real:
            return (expand_with_hints(self, hints, complex=False) if deep else self, Integer(0))
        parts = as_real_imag(expand_with_hints(arg, hints) if deep else arg)
        if parts is None:
            return None
        r, i = parts
        return (mul(cosh(r), cos(i)), mul(sinh(r), sin(i)))

    def fdiff(self, argindex: int) -> Expr:
        return sinh(self.args[0])

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.cosh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        a = facts_of(self.args[0])
        facts: dict[str, bool] = {'commutative': True}
        if a.get('real') or a.get('imaginary'):
            facts['real'] = True
        if a.get('real'):
            facts['positive'] = True
        if a.get('finite'):
            facts['finite'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:tanh
class Tanh(DefinedFunction):
    """The hyperbolic tangent, unevaluated (``tanh`` evaluates)."""

    unbranched = True

    function_name = 'tanh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import tan
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return Integer(1)
            if isinstance(arg, NegativeInfinity):
                return Integer(-1)
            if isinstance(arg, Rational) and arg.p == 0:
                return Integer(0)
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            if i_coeff.could_extract_minus_sign():
                return mul(Integer(-1), I, tan(mul(Integer(-1), i_coeff)))
            return mul(I, tan(i_coeff))
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                # tanh(x + n*I*pi) = tanh(x), tanh(x + I*pi/2 + n*I*pi) = coth(x)
                return cls.apply(rest) if m.denominator == 1 else coth(rest)
        if arg.is_zero:
            return Integer(0)
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Tanh, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        from .trigonometric import cos, sin
        arg = self.args[0]
        if arg.is_extended_real:
            return (expand_with_hints(self, hints, complex=False) if deep else self, Integer(0))
        parts = as_real_imag(expand_with_hints(arg, hints) if deep else arg)
        if parts is None:
            return None
        r, i = parts
        denominator = pow(add(pow(sinh(r), Integer(2)), pow(cos(i), Integer(2))), Integer(-1))
        return (mul(sinh(r), cosh(r), denominator), mul(sin(i), cos(i), denominator))

    def fdiff(self, argindex: int) -> Expr:
        return add(Integer(1), mul(Integer(-1), pow(self, Integer(2))))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.cosh(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.tanh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        facts = _odd_facts(self.args[0])
        a = facts_of(self.args[0])
        if a.get('extended_real'):
            facts['finite'] = True
        if a.get('zero'):
            facts['zero'] = True
        return facts


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:coth
class Coth(DefinedFunction):
    """The hyperbolic cotangent, unevaluated (``coth`` evaluates)."""

    unbranched = True

    function_name = 'coth'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import cot
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return Integer(1)
            if isinstance(arg, NegativeInfinity):
                return Integer(-1)
            if isinstance(arg, Rational) and arg.p == 0:
                return zoo
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
            return None
        if isinstance(arg, ComplexInfinity):
            return nan
        i_coeff = imaginary_coefficient(arg)
        if i_coeff is not None:
            if i_coeff.could_extract_minus_sign():
                return mul(I, cot(mul(Integer(-1), i_coeff)))
            return mul(Integer(-1), I, cot(i_coeff))
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Add):
            rest, m = peeloff_ipi(arg)
            if m:
                # coth(x + n*I*pi) = coth(x), coth(x + I*pi/2 + n*I*pi) = tanh(x)
                return cls.apply(rest) if m.denominator == 1 else tanh(rest)
        if arg.is_zero:
            return zoo
        return None

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Coth, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        from .trigonometric import cos, sin
        arg = self.args[0]
        if arg.is_extended_real:
            return (expand_with_hints(self, hints, complex=False) if deep else self, Integer(0))
        parts = as_real_imag(expand_with_hints(arg, hints) if deep else arg)
        if parts is None:
            return None
        r, i = parts
        denominator = pow(add(pow(sinh(r), Integer(2)), pow(sin(i), Integer(2))), Integer(-1))
        return (mul(sinh(r), cosh(r), denominator), mul(Integer(-1), sin(i), cos(i), denominator))

    def fdiff(self, argindex: int) -> Expr:
        return mul(Integer(-1), pow(sinh(self.args[0]), Integer(-2)))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.sinh(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.coth(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:ReciprocalHyperbolicFunction.eval
def _reciprocal_eval(cls: type[DefinedFunction], partner: type[DefinedFunction], even: bool, arg: Expr) -> Optional[Expr]:
    if arg.could_extract_minus_sign():
        flipped = cls.apply(mul(Integer(-1), arg))
        return flipped if even else mul(Integer(-1), flipped)
    t = partner.eval(arg)
    return pow(t, Integer(-1)) if t is not None else None


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:sech
class Sech(DefinedFunction):
    """The hyperbolic secant, unevaluated (``sech`` evaluates)."""

    unbranched = True

    function_name = 'sech'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _reciprocal_eval(cls, Cosh, True, arg)

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Sech, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        return as_real_imag(pow(cosh(self.args[0]), Integer(-1)), deep)

    def fdiff(self, argindex: int) -> Expr:
        return mul(Integer(-1), tanh(self.args[0]), self)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.cosh(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.sech(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:csch
class Csch(DefinedFunction):
    """The hyperbolic cosecant, unevaluated (``csch`` evaluates)."""

    unbranched = True

    function_name = 'csch'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        return _reciprocal_eval(cls, Sinh, False, arg)

    def _eval_conjugate(self) -> Optional[Expr]:
        from .trigonometric import _conjugated
        return _conjugated(Csch, self.args[0])

    def _eval_as_real_imag(self, deep: bool, ignore: Optional[Expr], hints: Optional[ExpandHints] = None) -> Optional[tuple[Expr, Expr]]:
        from .complexes import as_real_imag
        return as_real_imag(pow(sinh(self.args[0]), Integer(-1)), deep)

    def fdiff(self, argindex: int) -> Expr:
        return mul(Integer(-1), coth(self.args[0]), self)

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if mpmath.sinh(values[0]) == 0:
            return None
        return mpmath.mpc(mpmath.csch(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def sinh(x: ExprLike) -> Expr:
    """The hyperbolic sine of ``x``, evaluated."""
    return Sinh.apply(to_expr(x))


def cosh(x: ExprLike) -> Expr:
    """The hyperbolic cosine of ``x``, evaluated."""
    return Cosh.apply(to_expr(x))


def tanh(x: ExprLike) -> Expr:
    """The hyperbolic tangent of ``x``, evaluated."""
    return Tanh.apply(to_expr(x))


def coth(x: ExprLike) -> Expr:
    """The hyperbolic cotangent of ``x``, evaluated."""
    return Coth.apply(to_expr(x))


def sech(x: ExprLike) -> Expr:
    """The hyperbolic secant of ``x``, evaluated."""
    return Sech.apply(to_expr(x))


def csch(x: ExprLike) -> Expr:
    """The hyperbolic cosecant of ``x``, evaluated."""
    return Csch.apply(to_expr(x))


# ---------------------------------------------------------------------
# inverse hyperbolic functions

_ACOSH: dict[Expr, Expr] = {}
_ACSCH: dict[Expr, Expr] = {}
_ASECH: dict[Expr, Expr] = {}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:_acosh_table
def _acosh_table() -> dict[Expr, Expr]:
    from .exponential import log
    if not _ACOSH:
        s2, s3, s5 = sqrt(Integer(2)), sqrt(Integer(3)), sqrt(Integer(5))
        half = Rational(1, 2)
        entries: list[tuple[Expr, Expr]] = [
            (I, log(I*(1 + s2))), (-I, log(-I*(1 + s2))), (half, pi/3), (-half, pi*Rational(2, 3)),
            (s2/2, pi/4), (-s2/2, pi*Rational(3, 4)), (1/s2, pi/4), (-1/s2, pi*Rational(3, 4)), (s3/2, pi/6),
            (-s3/2, pi*Rational(5, 6)), ((s3 - 1)/sqrt(Integer(8)), pi*Rational(5, 12)),
            (-(s3 - 1)/sqrt(Integer(8)), pi*Rational(7, 12)), (sqrt(2 + s2)/2, pi/8),
            (-sqrt(2 + s2)/2, pi*Rational(7, 8)), (sqrt(2 - s2)/2, pi*Rational(3, 8)),
            (-sqrt(2 - s2)/2, pi*Rational(5, 8)), ((1 + s3)/(2*s2), pi/12), (-(1 + s3)/(2*s2), pi*Rational(11, 12)),
            ((s5 + 1)/4, pi/5), (-(s5 + 1)/4, pi*Rational(4, 5))]
        _ACOSH.update(entries)
    return _ACOSH


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:_acsch_table
def _acsch_table() -> dict[Expr, Expr]:
    from .exponential import log
    if not _ACSCH:
        s2, s3, s5, s6 = sqrt(Integer(2)), sqrt(Integer(3)), sqrt(Integer(5)), sqrt(Integer(6))
        entries: list[tuple[Expr, Expr]] = [
            (I, -pi/2), (I*(s2 + s6), -pi/12), (I*(1 + s5), -pi/10), (I*2/sqrt(2 - s2), -pi/8), (I*2, -pi/6),
            (I*sqrt(2 + 2/s5), -pi/5), (I*s2, -pi/4), (I*(s5 - 1), -3*pi/10), (I*2/s3, -pi/3),
            (I*2/sqrt(2 + s2), -3*pi/8), (I*sqrt(2 - 2/s5), -2*pi/5), (I*(s6 - s2), -5*pi/12),
            (Integer(2), -I*log((1 + s5)/2))]
        _ACSCH.update(entries)
    return _ACSCH


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:_asech_table
def _asech_table() -> dict[Expr, Expr]:
    from .exponential import log
    if not _ASECH:
        s2, s3, s5, s6 = sqrt(Integer(2)), sqrt(Integer(3)), sqrt(Integer(5)), sqrt(Integer(6))
        entries: list[tuple[Expr, Expr]] = [
            (I, -(pi*I/2) + log(1 + s2)), (-I, (pi*I/2) + log(1 + s2)), (s6 - s2, pi/12), (s2 - s6, 11*pi/12),
            (sqrt(2 - 2/s5), pi/10), (-sqrt(2 - 2/s5), 9*pi/10), (2/sqrt(2 + s2), pi/8), (-2/sqrt(2 + s2), 7*pi/8),
            (2/s3, pi/6), (-2/s3, 5*pi/6), (s5 - 1, pi/5), (1 - s5, 4*pi/5), (s2, pi/4), (-s2, 3*pi/4),
            (sqrt(2 + 2/s5), 3*pi/10), (-sqrt(2 + 2/s5), 7*pi/10), (Integer(2), pi/3), (Integer(-2), 2*pi/3),
            (sqrt(2*(2 + s2)), 3*pi/8), (-sqrt(2*(2 + s2)), 5*pi/8), (1 + s5, 2*pi/5), (-1 - s5, 3*pi/5),
            (s6 + s2, 5*pi/12), (-s6 - s2, 7*pi/12), (mul(I, oo), -pi*I/2), (mul(I, NegativeInfinity()), pi*I/2)]
        _ASECH.update(entries)
    return _ASECH


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:asinh
class Asinh(DefinedFunction):
    """The inverse hyperbolic sine, unevaluated (``asinh`` evaluates)."""

    function_name = 'asinh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .exponential import log
        from .trigonometric import asin
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Infinity):
                return oo
            if isinstance(arg, NegativeInfinity):
                return NegativeInfinity()
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return Integer(0)
                if arg == 1:
                    return log(sqrt(Integer(2)) + 1)
                if arg == -1:
                    return log(sqrt(Integer(2)) - 1)
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        else:
            if isinstance(arg, ComplexInfinity):
                return zoo
            if arg.is_zero:
                return Integer(0)
            i_coeff = imaginary_coefficient(arg)
            if i_coeff is not None:
                return mul(I, asin(i_coeff))
            if arg.could_extract_minus_sign():
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if isinstance(arg, Sinh) and arg.args[0].is_number and arg.args[0].is_real:
            return arg.args[0]
        return None

    def fdiff(self, argindex: int) -> Expr:
        return pow(sqrt(pow(self.args[0], Integer(2)) + 1), Integer(-1))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.asinh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:acosh
class Acosh(DefinedFunction):
    """The inverse hyperbolic cosine, unevaluated (``acosh`` evaluates)."""

    function_name = 'acosh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return oo
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return pi*I/2
                if arg == 1:
                    return Integer(0)
                if arg == -1:
                    return pi*I
        if arg.is_number:
            found = _acosh_table().get(arg)
            if found is not None:
                return mul(found, I) if arg.is_extended_real else found
        if isinstance(arg, ComplexInfinity):
            return zoo
        if arg == mul(I, oo):
            return oo + I*pi/2
        if arg == mul(Integer(-1), I, oo):
            return oo - I*pi/2
        if arg.is_zero:
            return pi*I/2
        return None

    def fdiff(self, argindex: int) -> Expr:
        x = self.args[0]
        return pow(mul(sqrt(x - 1), sqrt(x + 1)), Integer(-1))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        return mpmath.mpc(mpmath.acosh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:atanh
class Atanh(DefinedFunction):
    """The inverse hyperbolic tangent, unevaluated (``atanh`` evaluates)."""

    function_name = 'atanh'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import atan
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return Integer(0)
                if arg == 1:
                    return oo
                if arg == -1:
                    return NegativeInfinity()
            if isinstance(arg, Infinity):
                return mul(Integer(-1), I, atan(arg))
            if isinstance(arg, NegativeInfinity):
                return mul(I, atan(mul(Integer(-1), arg)))
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        else:
            if isinstance(arg, ComplexInfinity):
                return None  # SymPy: I*AccumBounds(-pi/2, pi/2), not ported
            i_coeff = imaginary_coefficient(arg)
            if i_coeff is not None:
                return mul(I, atan(i_coeff))
            if arg.could_extract_minus_sign():
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if arg.is_zero:
            return Integer(0)
        if isinstance(arg, Tanh) and arg.args[0].is_number and arg.args[0].is_real:
            return arg.args[0]
        return None

    def fdiff(self, argindex: int) -> Expr:
        return pow(1 - pow(self.args[0], Integer(2)), Integer(-1))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] in (1, -1):
            return None
        return mpmath.mpc(mpmath.atanh(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:acoth
class Acoth(DefinedFunction):
    """The inverse hyperbolic cotangent, unevaluated (``acoth`` evaluates)."""

    function_name = 'acoth'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .trigonometric import acot
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return Integer(0)
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return pi*I/2
                if arg == 1:
                    return oo
                if arg == -1:
                    return NegativeInfinity()
            if arg.is_negative:
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        else:
            if isinstance(arg, ComplexInfinity):
                return Integer(0)
            i_coeff = imaginary_coefficient(arg)
            if i_coeff is not None:
                return mul(Integer(-1), I, acot(i_coeff))
            if arg.could_extract_minus_sign():
                return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        if arg.is_zero:
            return pi*I/2
        return None

    def fdiff(self, argindex: int) -> Expr:
        return pow(1 - pow(self.args[0], Integer(2)), Integer(-1))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] in (1, -1):
            return None
        return mpmath.mpc(mpmath.acoth(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:asech
class Asech(DefinedFunction):
    """The inverse hyperbolic secant, unevaluated (``asech`` evaluates)."""

    function_name = 'asech'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return pi*I/2
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return oo
                if arg == 1:
                    return Integer(0)
                if arg == -1:
                    return pi*I
        if arg.is_number:
            found = _asech_table().get(arg)
            if found is not None:
                return mul(found, I) if arg.is_extended_real else found
        if isinstance(arg, ComplexInfinity):
            return None  # SymPy: I*AccumBounds(-pi/2, pi/2), not ported
        if arg.is_zero:
            return oo
        return None

    def fdiff(self, argindex: int) -> Expr:
        x = self.args[0]
        return mul(Integer(-1), pow(mul(x, sqrt(1 - pow(x, Integer(2)))), Integer(-1)))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] == 0:
            return None
        return mpmath.mpc(mpmath.asech(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


# PORTED FROM SYMPY: sympy/functions/elementary/hyperbolic.py:acsch
class Acsch(DefinedFunction):
    """The inverse hyperbolic cosecant, unevaluated (``acsch`` evaluates)."""

    function_name = 'acsch'

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        from .exponential import log
        [arg] = args
        if isinstance(arg, Number) and not isinstance(arg, ComplexInfinity):
            if isinstance(arg, NaN):
                return nan
            if isinstance(arg, (Infinity, NegativeInfinity)):
                return Integer(0)
            if isinstance(arg, Rational):
                if arg.p == 0:
                    return zoo
                if arg == 1:
                    return log(1 + sqrt(Integer(2)))
                if arg == -1:
                    return mul(Integer(-1), log(1 + sqrt(Integer(2))))
        if arg.is_number:
            found = _acsch_table().get(arg)
            if found is not None:
                return mul(found, I)
        if isinstance(arg, ComplexInfinity):
            return Integer(0)
        if arg.is_infinite:
            return Integer(0)
        if arg.is_zero:
            return zoo
        if arg.could_extract_minus_sign():
            return mul(Integer(-1), cls.apply(mul(Integer(-1), arg)))
        return None

    def fdiff(self, argindex: int) -> Expr:
        x = self.args[0]
        return mul(Integer(-1), pow(mul(pow(x, Integer(2)), sqrt(1 + pow(x, Integer(-2)))), Integer(-1)))

    def numeric(self, values: Sequence[mpmath.mpc]) -> Optional[mpmath.mpc]:
        if values[0] == 0:
            return None
        return mpmath.mpc(mpmath.acsch(values[0]))

    def _eval_facts(self) -> dict[str, bool]:
        return {'commutative': True}


def asinh(x: ExprLike) -> Expr:
    """The inverse hyperbolic sine of ``x``, evaluated."""
    return Asinh.apply(to_expr(x))


def acosh(x: ExprLike) -> Expr:
    """The inverse hyperbolic cosine of ``x``, evaluated."""
    return Acosh.apply(to_expr(x))


def atanh(x: ExprLike) -> Expr:
    """The inverse hyperbolic tangent of ``x``, evaluated."""
    return Atanh.apply(to_expr(x))


def acoth(x: ExprLike) -> Expr:
    """The inverse hyperbolic cotangent of ``x``, evaluated."""
    return Acoth.apply(to_expr(x))


def asech(x: ExprLike) -> Expr:
    """The inverse hyperbolic secant of ``x``, evaluated."""
    return Asech.apply(to_expr(x))


def acsch(x: ExprLike) -> Expr:
    """The inverse hyperbolic cosecant of ``x``, evaluated."""
    return Acsch.apply(to_expr(x))
