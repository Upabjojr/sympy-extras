"""The numerical value of a ``Sum``: SymPy's ``evalf_sum``.

Ported from SymPy's ``sympy.core.evalf`` (``evalf_sum``, ``hypsum``,
``check_convergence``): an infinite sum of a hypergeometric term with
rational ratio is summed directly when it converges geometrically or
faster, by Richardson extrapolation when it converges like a power of
``n``, and SymPy's ``ValueError`` is raised when it diverges
(``Sum(1, (i, 1, oo))`` diverges like ``n^0``). A finite sum of a few
terms is the sum of their values. The other sums (SymPy's
Euler–Maclaurin summation) are not ported: ``None``.

References
==========

.. [1] F. Johansson, mpmath: a Python library for arbitrary-precision
       floating-point arithmetic, http://mpmath.org (``nsum``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/evalf.py (evalf_sum, hypsum, check_convergence) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; mpmath numbers instead of SymPy's mpf tuples;
# the Euler–Maclaurin summation not ported (None).
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mpmath

from sympy_extras.core.expr import Expr
from sympy_extras.core.numbers import Infinity, Integer, Rational
from sympy_extras.core.symbol import Symbol

__all__ = ['evalf_sum', 'hypsum', 'check_convergence', 'Convergence']

#: SymPy's ``(h, g, p)`` of ``check_convergence``: the factorial rate, the
#: geometric ratio, the polynomial exponent (``None`` when not reached)
Convergence = tuple[int, Optional[Fraction], Optional[Fraction]]

#: the most terms of a finite sum which are added one by one (SymPy's
#: Euler–Maclaurin summation takes ``2*prec`` terms first)
_DIRECT_TERMS = 2


def _integer_coefficients(e: Expr, n: Symbol) -> Optional[list[int]]:
    """The coefficients of the polynomial ``e`` in ``n``, highest degree
    first, when they are integers."""
    from sympy_extras.polys.polyerrors import CoercionFailed, PolynomialError
    from sympy_extras.polys.polytools import Poly
    try:
        coeffs = Poly(e, n).all_coeffs()
    except (CoercionFailed, PolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    out: list[int] = []
    for c in coeffs:
        if not isinstance(c, Integer):
            return None
        out.append(c.p)
    return out


def _degree(e: Expr, n: Symbol) -> Optional[int]:
    """The degree in ``n`` of a polynomial with numerical coefficients,
    ``None`` for anything else."""
    from sympy_extras.core.add import Add
    degree = -1
    for t in (e.expand().args if isinstance(e.expand(), Add) else (e.expand(),)):
        if not isinstance(t, Expr):
            return None
        c, k = t.as_coeff_exponent(n)
        if c.has(n) or not c.is_number or not isinstance(k, Integer) or k.p < 0:
            return None
        if c.is_zero is not True:
            degree = max(degree, k.p)
    return degree if degree >= 0 else None


def _factorial_rate_sum(expr: Expr, num: Expr, den: Expr, n: Symbol, prec: int) -> Optional[mpmath.mpf]:
    """The sum over ``n >= 0`` of the term ``expr`` whose ratio ``num/den``
    has numerical coefficients and a denominator of higher degree (a
    convergence like a power of ``1/n!``), term by term at ``prec`` bits;
    ``None`` otherwise."""
    from sympy_extras.core.evalf import numeric_value
    p, q = _degree(num, n), _degree(den, n)
    if p is None or q is None or q <= p:
        return None
    first = expr.subs(n, Integer(0))
    if not isinstance(first, Expr):
        return None
    with mpmath.workprec(prec + 20):
        term = numeric_value(first, prec + 20)
        if term is None or mpmath.im(term) != 0 or term == 0:
            return None
        total = mpmath.mpf(mpmath.re(term))
        current = total
        bound = mpmath.mpf(5) * mpmath.mpf(2) ** (-prec)
        k = 1
        while abs(current) > bound:
            a, b = num.subs(n, Integer(k - 1)), den.subs(n, Integer(k - 1))
            if not (isinstance(a, Expr) and isinstance(b, Expr)):
                return None
            va, vb = numeric_value(a, prec + 20), numeric_value(b, prec + 20)
            if va is None or vb is None or vb == 0 or mpmath.im(va) != 0 or mpmath.im(vb) != 0:
                return None
            current = current * mpmath.re(va) / mpmath.re(vb)
            total += current
            k += 1
        return +total


def _horner(coeffs: list[int], k: int) -> int:
    value = 0
    for c in coeffs:
        value = value * k + c
    return value


# PORTED FROM SYMPY: sympy/core/evalf.py:check_convergence
def check_convergence(numer: list[int], denom: list[int]) -> Convergence:
    """``(h, g, p)`` of the ratio ``numer/denom`` of consecutive terms:
    ``h > 0`` convergence like ``1/n!**h``, ``h < 0`` divergence; for
    ``h = 0``, ``|g| > 1`` geometric convergence, ``|g| < 1`` divergence
    (``g < 0`` alternating) and for ``|g| = 1``, ``p > 1`` convergence like
    ``1/n**p``, divergence otherwise."""
    p, q = len(numer) - 1, len(denom) - 1
    rate = q - p
    if rate:
        return rate, None, None
    constant = Fraction(denom[0], numer[0])
    if abs(constant) != 1:
        return rate, constant, None
    if p == q == 0:
        return rate, constant, Fraction(0)
    return rate, constant, Fraction(denom[1] - numer[1], denom[0])


# PORTED FROM SYMPY: sympy/core/evalf.py:hypsum
def hypsum(expr: Expr, n: Symbol, start: int, prec: int) -> Optional[mpmath.mpf]:
    """The sum over ``n >= start`` of the hypergeometric term ``expr`` at
    ``prec`` bits; ``None`` when the ratio of its terms is not a quotient
    of integer polynomials or its first term not rational (SymPy's
    ``NotImplementedError``). Raises SymPy's ``ValueError`` when the sum
    diverges."""
    from sympy_extras.concrete.gosper import hypersimp
    if start:
        shifted = expr.subs(n, n + start)
        assert isinstance(shifted, Expr)
        expr = shifted
    hs = hypersimp(expr, n)
    if hs is None:
        return None
    num, den = hs.as_numer_denom()
    numer, denom = _integer_coefficients(num, n), _integer_coefficients(den, n)
    if numer is None or denom is None:
        # SymPy truncates the values of irrational coefficients to
        # integers (its sum of pi**n/n! is exp(3)): a ratio of numerical
        # polynomials converging like a factorial is summed numerically
        return _factorial_rate_sum(expr, num, den, n, prec)
    if not numer or not denom or numer[0] == 0 or denom[0] == 0:
        return None
    h, g, p = check_convergence(numer, denom)
    if h < 0:
        raise ValueError("Sum diverges like (n!)^%i" % (-h))
    eterm = expr.subs(n, Integer(0))
    if not isinstance(eterm, Rational):
        return None
    tp, tq = eterm.p, eterm.q
    if h > 0 or (g is not None and abs(g) > 1):
        # direct summation if geometric or faster (fixed point, as SymPy)
        term = (tp << prec) // tq
        s = term
        k = 1
        while abs(term) > 5:
            term *= _horner(numer, k - 1)
            term //= _horner(denom, k - 1)
            s += term
            k += 1
        return mpmath.mpf(s) * mpmath.mpf(2) ** (-prec)
    assert g is not None
    alt = g < 0
    if abs(g) < 1:
        raise ValueError("Sum diverges like (%i)^n" % int(abs(1 / g)))
    assert p is not None
    if p < 1 or (p == 1 and not alt):
        raise ValueError("Sum diverges like n^%i" % int(-p))
    # polynomial convergence: Richardson extrapolation, at quadruple
    # precision and doubling it until two answers agree to the digits
    vold: Optional[str] = None
    v = mpmath.mpf(0)
    ndig = max(1, int(mpmath.libmp.prec_to_dps(prec)))
    while True:
        prec2 = 4 * prec
        terms = [(tp << prec2) // tq]

        def summand(kk: mpmath.mpf) -> mpmath.mpf:
            k = int(kk)
            if k:
                terms[0] *= _horner(numer, k - 1)
                terms[0] //= _horner(denom, k - 1)
            return mpmath.mpf(terms[0]) * mpmath.mpf(2) ** (-prec2)

        with mpmath.workprec(prec):
            v = mpmath.nsum(summand, [0, mpmath.inf], method='richardson')
        vf = mpmath.nstr(v, ndig)
        if vold is not None and vold == vf:
            break
        prec += prec
        vold = vf
    return v


def _fastlog(x: mpmath.mpf) -> int:
    """SymPy's ``fastlog``: the exponent plus the bit count, an estimate
    of ``log2(|x|)``."""
    if x == 0:
        return -(10**9)
    sign, man, exp, bc = x._mpf_
    return int(exp) + int(bc)


# PORTED FROM SYMPY: sympy/core/evalf.py:evalf_sum
def evalf_sum(function: Expr, i: Symbol, a: Expr, b: Expr, prec: int) -> Optional[mpmath.mpc]:
    """The value of ``Sum(function, (i, a, b))`` at ``prec`` bits, ``None``
    when not decided here. Raises SymPy's ``ValueError`` for a divergent
    infinite hypergeometric sum.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer, oo
    >>> from sympy_extras.functions import factorial
    >>> i = Symbol('i', integer=True)
    >>> print(evalf_sum(1/factorial(i), i, Integer(0), oo, 53).real)
    2.71828182845905
    >>> print(evalf_sum(1/i**2, i, Integer(1), oo, 53).real)
    1.64493406684823
    >>> evalf_sum(Integer(1), i, Integer(1), oo, 53)
    Traceback (most recent call last):
    ...
    ValueError: Sum diverges like n^0
    """
    from sympy_extras.core.evalf import numeric_value
    if function.is_zero:
        return mpmath.mpc(0)
    if isinstance(b, Infinity) and isinstance(a, Integer):
        prec2 = prec + 10
        v = hypsum(function, i, a.p, prec2)
        if v is None or v == 0:
            return None
        if _fastlog(v) < -10:
            delta = prec - _fastlog(v)
            again = hypsum(function, i, a.p, delta)
            if again is None:
                return None
            v = again
        return mpmath.mpc(v)
    if isinstance(a, Integer) and isinstance(b, Integer) and b.p - a.p == -1:
        return mpmath.mpc(0)  # the empty sum (Karr's convention)
    if isinstance(a, Integer) and isinstance(b, Integer) and 0 <= b.p - a.p < _DIRECT_TERMS * prec:
        total = mpmath.mpc(0)
        for k in range(a.p, b.p + 1):
            term = function.subs(i, Integer(k))
            if not isinstance(term, Expr):
                return None
            value = numeric_value(term, prec + 10)
            if value is None:
                return None
            total += value
        return total
    return None
