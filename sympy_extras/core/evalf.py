"""Numerical evaluation with mpmath.

``evalf(expr, n)`` evaluates the numerical subexpressions to ``n``
digits: numbers, the constants ``pi``, ``E`` and ``I``, the sums,
products and powers of such, and the wrapped functions of SymPy. What
contains a symbol is rebuilt with the evaluating functions around the
evaluated parts, as SymPy does (``(x + pi).evalf()`` is
``x + 3.14159265358979``). A complex value is ``re + im*I`` with two
floats.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mpmath
from sympy.core.expr import Expr as SymPyExpr
from sympy.core.numbers import Number as SymPyNumber

from .basic import Basic, SubsMapping
from .expr import Expr, as_expr
from .numbers import Exp1, Float, ImaginaryUnit, Number, Pi, dps_to_prec, number_from_fraction

__all__ = ['evalf', 'to_float', 'to_complex', 'numeric_value']

_GUARD_BITS = 20


def _mpf_to_fraction(x: mpmath.mpf) -> Fraction:
    sign, man, exp, _ = x._mpf_
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if int(sign) else value


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
        if isinstance(e, ImaginaryUnit):
            return mpmath.mpc(0, 1)
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
        if isinstance(e, AppliedSymPyFunction):
            from .interop import to_sympy
            parts = [numeric_value(a, prec) for a in e.args]
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
    prec = dps_to_prec(n)
    value = numeric_value(e, prec + _GUARD_BITS)
    if value is not None:
        return _from_mpc(value, prec)
    if not e.args:
        return e
    args = tuple(evalf(a, n) if isinstance(a, Expr) else a for a in e.args)
    return as_expr(e._evaluated(args))


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

