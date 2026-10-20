"""Numerical evaluation with mpmath.

``evalf(expr, n)`` (also ``N``) evaluates the numerical subexpressions
to ``n`` digits: numbers, the constants ``pi``, ``E``, ``EulerGamma``,
``Catalan``, ``GoldenRatio``, ``TribonacciConstant`` and ``I``, the
sums, products and powers of such, and any other numerical node through
SymPy (the wrapped functions of SymPy, ``Mod``, ...). What contains a
symbol is rebuilt with the evaluating functions around the evaluated
parts, as SymPy does (``(x + pi).evalf()`` is ``x + 3.14159265358979``).
A complex value is ``re + im*I`` with two floats.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mpmath
from sympy.core.expr import Expr as SymPyExpr
from sympy.core.numbers import Number as SymPyNumber

from .basic import Basic, SubsMapping
from .coreerrors import PrecisionExhausted
from .expr import Expr, UnevaluatedExpr, as_expr
from .numbers import (AlgebraicNumber, CatalanType, EulerGammaType, Exp1, Float, GoldenRatioType,
    ImaginaryUnit, Number, Pi, TribonacciConstantType, dps_to_prec, number_from_fraction)

__all__ = ['evalf', 'N', 'to_float', 'to_complex', 'numeric_value', 'mpf_to_fraction', 'PrecisionExhausted']

_GUARD_BITS = 20


def mpf_to_fraction(x: mpmath.mpf) -> Fraction:
    """The exact value of an mpmath float."""
    sign, man, exp, _ = x._mpf_
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if int(sign) else value


_mpf_to_fraction = mpf_to_fraction


def _tribonacci() -> mpmath.mpf:
    s = mpmath.sqrt(33)
    return (1 + mpmath.cbrt(19 + 3 * s) + mpmath.cbrt(19 - 3 * s)) / 3


def numeric_value(e: Expr, prec: int) -> Optional[mpmath.mpc]:
    """The value of a numerical expression as an mpmath complex number at
    the working precision ``prec`` (bits), ``None`` when ``e`` contains a
    symbol or something not evaluable."""
    from .add import Add
    from .function import AppliedSymPyFunction
    from .mul import Mul
    from .power import Pow
    with mpmath.workprec(prec):
        if isinstance(e, Number):
            value = e.exact_value()
            if value is None:
                return None
            return mpmath.mpc(mpmath.mpf(value.numerator) / value.denominator)
        if isinstance(e, Pi):
            return mpmath.mpc(mpmath.pi)
        if isinstance(e, Exp1):
            return mpmath.mpc(mpmath.e)
        if isinstance(e, EulerGammaType):
            return mpmath.mpc(mpmath.euler)
        if isinstance(e, CatalanType):
            return mpmath.mpc(mpmath.catalan)
        if isinstance(e, GoldenRatioType):
            return mpmath.mpc(mpmath.phi)
        if isinstance(e, TribonacciConstantType):
            return mpmath.mpc(_tribonacci())
        if isinstance(e, ImaginaryUnit):
            return mpmath.mpc(0, 1)
        if isinstance(e, AlgebraicNumber):
            return numeric_value(e.as_expr(), prec)
        if isinstance(e, UnevaluatedExpr):
            return numeric_value(e.expr, prec)
        if isinstance(e, (Add, Mul)):
            parts = [numeric_value(a, prec) for a in e.args]
            if any(p is None for p in parts):
                return None
            result = mpmath.mpc(0) if isinstance(e, Add) else mpmath.mpc(1)
            for p in parts:
                result = result + p if isinstance(e, Add) else result * p
            return result
        if isinstance(e, Pow):
            b, x = numeric_value(e.base, prec), numeric_value(e.exp, prec)
            if b is None or x is None:
                return None
            if b == 0:
                if x == 0:
                    return mpmath.mpc(1)
                if mpmath.re(x) > 0:
                    return mpmath.mpc(0)
                return None
            return mpmath.mpc(mpmath.power(b, x))
        if isinstance(e, AppliedSymPyFunction) or (e.args and e.is_number):
            from .interop import to_sympy
            parts = [numeric_value(a, prec) for a in e.args if isinstance(a, Expr)]
            if any(p is None for p in parts):
                return None
            dps = int(mpmath.mp.dps) + 5
            sympy_expr = to_sympy(e)
            if not isinstance(sympy_expr, SymPyExpr):
                return None
            sympy_value = sympy_expr.evalf(dps)
            if not isinstance(sympy_value, SymPyExpr):
                return None
            re, im = sympy_value.as_real_imag()
            if not (isinstance(re, SymPyNumber) and isinstance(im, SymPyNumber)):
                return None
            return mpmath.mpc(mpmath.mpf(str(re)), mpmath.mpf(str(im)))
    return None


def _from_mpc(z: mpmath.mpc, prec: int) -> Expr:
    from .add import add
    from .mul import mul
    with mpmath.workprec(prec):
        re, im = mpmath.re(z), mpmath.im(z)
        real = Float(number_from_fraction(_mpf_to_fraction(mpmath.mpf(re))), precision=prec)
        imag = _mpf_to_fraction(mpmath.mpf(im))
    if imag == 0:
        return real
    imaginary = mul(Float(number_from_fraction(imag), precision=prec), ImaginaryUnit())
    if real.as_fraction() == 0:
        return imaginary
    return add(real, imaginary)


def evalf(e: Expr, n: int = 15, subs: Optional[SubsMapping] = None) -> Expr:
    """``e`` evaluated numerically to ``n`` digits.

    Examples
    ========

    >>> from sympy_extras.core import symbols, pi, I, sqrt, Rational
    >>> from sympy_extras.core.evalf import evalf
    >>> x = symbols('x')
    >>> evalf(pi), evalf(Rational(1, 3), 20), evalf(sqrt(2)*x, 5)
    (3.14159265358979, 0.33333333333333333333, 1.4142*x)
    >>> evalf(pi + I)
    3.14159265358979 + 1.0*I
    >>> evalf(x**2 + pi, subs={x: 2})
    7.14159265358979
    """
    if subs:
        rules: dict[Basic, Basic] = {}
        from .sympify import sympify
        for k, v in subs.items():
            rules[k] = sympify(v)
        e = e.xreplace(rules).doit()
    if e.is_Number and e == 0:
        return e  # SymPy keeps an exact zero
    prec = dps_to_prec(n)
    value = numeric_value(e, prec + _GUARD_BITS)
    if value is not None:
        return _from_mpc(value, prec)
    return _evalf_inside(e, n)


def _evalf_inside(e: Expr, n: int) -> Expr:
    """The numerical evaluation of what a symbolic node contains, with
    SymPy's conventions: the numerical factors of a product are evaluated
    as one number (a coefficient ``-1`` is kept), the terms of a sum one
    by one (``I`` is kept), an integer exponent is kept, a function of a
    symbolic argument is left as it is (``log(3*z/8 - 3*pi/8)``)."""
    from .add import Add, add
    from .function import Application
    from .mul import Mul, mul
    from .numbers import ImaginaryUnit, Integer, Rational
    from .parameters import distribute
    from .power import Pow
    if isinstance(e, ImaginaryUnit) or not e.args:
        return e if not isinstance(e, Rational) or e == 0 else evalf(e, n)
    if isinstance(e, Add):
        return add(*[_evalf_term(t, n) for t in e.args])
    if isinstance(e, Mul):
        numeric = [f for f in e.args if f.is_number]
        symbolic = [_evalf_inside(f, n) for f in e.args if not f.is_number]
        if not numeric:
            return mul(*symbolic)
        if all(_is_unit(f) for f in numeric):
            return mul(*numeric, *symbolic)  # -I*x stays
        if len(numeric) == 2 and isinstance(numeric[0], Rational) and numeric[0] == -1 and isinstance(numeric[1], Add):
            with distribute(False):  # SymPy: -(3.33 + I)/(p - 4.0), the sum factor term by term
                return mul(Integer(-1), _evalf_inside(numeric[1], n), *symbolic)
        coefficient = mul(*numeric)
        if isinstance(coefficient, Add):
            return mul(_evalf_inside(coefficient, n), *symbolic)
        return mul(evalf(coefficient, n), *symbolic)
    if isinstance(e, Pow):
        exponent = e.exp
        keep = isinstance(exponent, Rational) and exponent.q == 1
        return as_expr(e._evaluated((_evalf_term(e.base, n), exponent if keep else _evalf_term(exponent, n))))
    if isinstance(e, Application):
        return e
    args = tuple(_evalf_term(a, n) if isinstance(a, Expr) else a for a in e.args)
    return as_expr(e._evaluated(args))


def _evalf_term(e: Expr, n: int) -> Expr:
    """A term or a base: as a number when it is one, else from inside;
    ``I``, ``-I`` and ``I**e`` keep the imaginary unit, as in SymPy."""
    from .mul import Mul
    from .numbers import ImaginaryUnit
    from .power import Pow
    if isinstance(e, ImaginaryUnit) or (isinstance(e, Mul) and all(_is_unit(f) for f in e.args)):
        return e
    if isinstance(e, Pow) and isinstance(e.base, ImaginaryUnit):
        return as_expr(e._evaluated((e.base, _evalf_term(e.exp, n))))
    if e.is_number:
        return evalf(e, n)
    return _evalf_inside(e, n)


def _is_unit(f: Expr) -> bool:
    """Whether ``f`` is ``1``, ``-1`` or ``I``."""
    from .numbers import ImaginaryUnit, Rational
    return isinstance(f, ImaginaryUnit) or (isinstance(f, Rational) and f.q == 1 and f.p in (1, -1))


def N(e: Expr, n: int = 15, subs: Optional[SubsMapping] = None) -> Expr:
    """``e.evalf(n)``.

    Examples
    ========

    >>> from sympy_extras.core import N, pi, Symbol
    >>> N(pi, 20), N(Symbol('x') + pi, 5)
    (3.1415926535897932385, x + 3.1416)
    """
    return evalf(e, n, subs)


def to_float(e: Expr) -> float:
    """``float(e)`` for a real numerical expression."""
    value = numeric_value(e, 53 + _GUARD_BITS)
    if value is None:
        raise TypeError("cannot convert %s to a float" % (e,))
    if mpmath.im(value) != 0:
        raise TypeError("cannot convert the complex number %s to a float" % (e,))
    return float(mpmath.re(value))


def to_complex(e: Expr) -> complex:
    """``complex(e)`` for a numerical expression."""
    value = numeric_value(e, 53 + _GUARD_BITS)
    if value is None:
        raise TypeError("cannot convert %s to a complex number" % (e,))
    return complex(float(mpmath.re(value)), float(mpmath.im(value)))

