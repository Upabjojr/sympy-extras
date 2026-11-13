"""The numerical value of a ``Sum``: SymPy's ``evalf_sum``.

Ported from SymPy's ``sympy.core.evalf`` (``evalf_sum``, ``hypsum``,
``check_convergence``): an infinite sum of a hypergeometric term with
rational ratio is summed directly when it converges geometrically or
faster, by Richardson extrapolation when it converges like a power of
``n``, and SymPy's ``ValueError`` is raised when it diverges
(``Sum(1, (i, 1, oo))`` diverges like ``n^0``). A finite sum of a few
terms is the sum of their values. The other sums are summed by SymPy's
Euler–Maclaurin summation (:func:`euler_maclaurin`, SymPy's
``Sum.euler_maclaurin``): the first terms added one by one, the rest
the integral of the summand (an unevaluated
:class:`~sympy_extras.integrals.integral.Integral`, evaluated by its
quadrature) with the endpoint corrections by the Bernoulli numbers, the
numbers of terms doubled until the first omitted correction is below
``2**-prec``.

References
==========

.. [1] F. Johansson, mpmath: a Python library for arbitrary-precision
       floating-point arithmetic, http://mpmath.org (``nsum``).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/evalf.py (evalf_sum, hypsum, check_convergence),
# sympy/concrete/summations.py (Sum.euler_maclaurin) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; mpmath numbers instead of SymPy's mpf tuples;
# None for what SymPy leaves unevaluated (NotImplementedError, NaN).
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Optional

import mpmath

from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.numbers import Infinity, Integer, NaN, NegativeInfinity, Rational
from sympy_extras.core.symbol import Symbol

__all__ = ['evalf_sum', 'hypsum', 'check_convergence', 'Convergence', 'euler_maclaurin']

#: SymPy's ``(h, g, p)`` of ``check_convergence``: the factorial rate, the
#: geometric ratio, the polynomial exponent (``None`` when not reached)
Convergence = tuple[int, Optional[Fraction], Optional[Fraction]]

#: the most terms of a finite sum which are added one by one (SymPy's
#: Euler–Maclaurin summation takes ``2*prec`` terms first)
_DIRECT_TERMS = 2
#: the most terms of a finite sum added one by one in any case (SymPy's
#: Euler-Maclaurin summation of a finite sum of terms which grow again,
#: exp(-i**2) over (-500, 500), integrates over the peak)
_MOST_DIRECT = 4000


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


def _has_natural_root(coeffs: list[int]) -> bool:
    """Whether the integer polynomial (highest degree first) vanishes at a
    natural number ``k >= 0`` (they are below Cauchy's bound)."""
    lead = abs(coeffs[0])
    bound = 1 + max((abs(c) for c in coeffs[1:]), default=0) // lead + 1
    return any(_horner(coeffs, k) == 0 for k in range(min(bound, 10**5) + 1))


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
    if _has_natural_root(denom):
        # the ratio of the terms has a pole at a natural number (a zero
        # first term: SymPy's hypsum divides by zero on i**2/(i**4 + 1)
        # from 0, sympy-extras#25): summed by Euler-Maclaurin instead
        return None
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
        if v is not None:
            if v == 0:
                return None
            if _fastlog(v) < -10:
                delta = prec - _fastlog(v)
                again = hypsum(function, i, a.p, delta)
                if again is None:
                    return None
                v = again
            return mpmath.mpc(v)
    elif isinstance(a, Integer) and isinstance(b, Integer) and b.p - a.p == -1:
        return mpmath.mpc(0)  # the empty sum (Karr's convention)
    elif isinstance(a, Integer) and isinstance(b, Integer) and 0 <= b.p - a.p < max(_DIRECT_TERMS * prec, _MOST_DIRECT):
        # the terms which the Euler-Maclaurin summation adds one by one
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
    return _euler_maclaurin_value(function, i, a, b, prec)


def _euler_maclaurin_value(function: Expr, i: Symbol, a: Expr, b: Expr, prec: int) -> Optional[mpmath.mpc]:
    """SymPy's Euler-Maclaurin branch of ``evalf_sum``: ``2**k*prec``
    terms directly and as many corrections, ``k = 1, ..., 4``, until the
    estimated error is below ``2**-prec``; ``None`` for a ``NaN`` error
    (SymPy's ``NotImplementedError``) or a value not evaluable."""
    from sympy_extras.core.evalf import numeric_value
    if isinstance(a, NegativeInfinity):
        return None
    if not _monotone_tail(function, i, a + Integer(2 * prec), b):
        # Euler-Maclaurin's remainder is small only for a summand which
        # does not oscillate nor peak inside the range (SymPy sums
        # exp(-(i - 3)**2) over (-5000, 5000) as its integral, sqrt(pi))
        return None
    eps = Rational(1, 2**prec)
    with mpmath.workprec(prec + 10):
        bound = mpmath.mpf(2) ** (-prec)
    s: Expr = Integer(0)
    err_value: Optional[mpmath.mpc] = None
    for k in range(1, 5):
        m = n = 2**k * prec
        found = euler_maclaurin(function, i, a, b, m, n, eps, eval_integral=False)
        if found is None:
            return None
        s, err = found
        if isinstance(err, NaN):
            return None
        err_value = _value3(err)
        if err_value is None:
            return None
        if abs(err_value) <= bound:
            break
    value = numeric_value(s, prec + 10)
    if value is None or not (mpmath.isfinite(mpmath.re(value)) and mpmath.isfinite(mpmath.im(value))):
        return None
    return value


def _monotone_tail(f: Expr, i: Symbol, lo: Expr, hi: Expr) -> bool:
    """Whether ``|f|`` decreases at sample points of ``(lo, hi)`` (or, for
    a range which ends below ``lo``, of the whole range), real and
    without a change of sign: linearly spaced points for a finite range,
    points at the powers of two from ``lo`` for an infinite one."""
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.integrals.definite import _numeric_function
    t = Dummy('t', real=True)
    g_ = f.subs(i, t)
    if not isinstance(g_, Expr):
        return False
    with mpmath.workprec(60):
        try:
            g = _numeric_function(g_, t)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return False
        from sympy_extras.core.evalf import numeric_value
        start, end = numeric_value(lo, 60), None if isinstance(hi, Infinity) else numeric_value(hi, 60)
        if start is None or (end is None and not isinstance(hi, Infinity)):
            return False
        a0 = mpmath.re(start)
        if end is not None and mpmath.re(end) <= a0:
            return True  # summed directly
        if end is None:
            points = [a0 + 2**j - 1 for j in range(48)]
        else:
            b0 = mpmath.re(end)
            points = [a0 + (b0 - a0) * k / 200 for k in range(201)]
        values: list[mpmath.mpf] = []
        for p in points:
            try:
                v = g(mpmath.mpc(p))
            except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                return False
            if not mpmath.isfinite(v) or abs(mpmath.im(v)) > abs(mpmath.re(v)) * mpmath.mpf('1e-10'):
                return False
            values.append(mpmath.re(v))
        nonzero = [v for v in values if v != 0]
        if nonzero and not (all(v > 0 for v in nonzero) or all(v < 0 for v in nonzero)):
            return False
        return all(abs(later) <= abs(earlier) * (1 + mpmath.mpf('1e-12')) for earlier, later in zip(values, values[1:]))


def _value3(e: Expr) -> Optional[mpmath.mpc]:
    """SymPy's ``e.evalf(3)``: the value to three digits, the working
    precision doubled until two values agree to them (as SymPy's evalf
    raises it; a correction ``zeta(6) - harmonic(175, 6)`` cancels its
    leading digits); ``None`` for a symbolic or not evaluable ``e``."""
    from sympy_extras.core.evalf import numeric_value
    prec = 40
    try:
        low = numeric_value(e, prec)
        while low is not None and prec < 4000:
            high = numeric_value(e, 2 * prec)
            if high is None:
                return None
            with mpmath.workprec(2 * prec):
                if abs(high - low) <= mpmath.mpf('1e-4') * abs(high):
                    return high
            low, prec = high, 2 * prec
        return low
    except (ZeroDivisionError, ValueError, OverflowError, TypeError):
        return None


def _is_less(e: Expr, eps: mpmath.mpf) -> Optional[bool]:
    """``abs(e.evalf(3)) < eps``: ``None`` when ``e`` has no value (SymPy's
    symbolic relational)."""
    value = _value3(e)
    if value is None:
        return None
    if not (mpmath.isfinite(mpmath.re(value)) and mpmath.isfinite(mpmath.im(value))):
        return False
    return bool(abs(value) < eps)


def _decreasing(f: Expr, i: Symbol, k: Expr, term: Expr) -> bool:
    """Whether the terms after ``f(k) = term`` get smaller: SymPy stops
    the direct summation at the first term below ``eps``, which is wrong
    for terms which grow again (its ``Sum(exp(-i**2), (i, -50,
    50)).evalf()`` is ``0``, sympy-extras#25); the next term must be
    smaller, and so must the one ten terms on."""
    here = _value3(term)
    if here is None:
        return False
    for step in (1, 10):
        later = _value3(_at(f, i, k + Integer(step)))
        if later is None or abs(later) > abs(here):
            return False
    return True


# PORTED FROM SYMPY: sympy/concrete/summations.py:Sum.euler_maclaurin
def euler_maclaurin(function: Expr, i: Symbol, a: Expr, b: Expr, m: int = 0, n: int = 0, eps: ExprLike = 0,
                    eval_integral: bool = True) -> Optional[tuple[Expr, Expr]]:
    """SymPy's ``Sum(function, (i, a, b)).euler_maclaurin(m, n, eps,
    eval_integral)``: ``(s, e)``, the Euler-Maclaurin approximation with
    ``m`` leading terms summed directly and ``n`` correction terms of the
    tail, and the estimated error (the magnitude of the first omitted
    term). With a nonzero ``eps`` the summation ends as soon as a term is
    below it. ``None`` when a comparison with ``eps`` is not decided (a
    symbolic term, where SymPy raises).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> k, a, b = Symbol('k', integer=True), Symbol('a', positive=True), Symbol('b', positive=True)
    >>> s, e = euler_maclaurin(1/k, k, Integer(2), Integer(5))
    >>> s, e
    (7/20 + log(5/2), 7/400)
    >>> print(s.evalf())
    1.26629073187415
    >>> euler_maclaurin(1/k, k, a, b)
    (log(b/a) + 1/(2*b) + 1/(2*a), Abs(1/(12*b**2) - 1/(12*a**2)))
    >>> euler_maclaurin(k, k, Integer(2), b)
    (b**2/2 + b/2 - 1, 0)
    """
    from sympy_extras.core.add import add
    from sympy_extras.core.symbol import Dummy
    from sympy_extras.functions import abs_, bernoulli, factorial
    from sympy_extras.integrals.integral import Integral
    eps_e = to_expr(eps)
    eps_value = _value3(eps_e)
    eps_zero = eps_value is None or eps_value == 0
    eps_mpf = mpmath.mpf(0) if eps_value is None else mpmath.mpf(abs(eps_value))
    f = function
    if (a - b).is_extended_positive is True:
        if (a - b) == Integer(1):
            return Integer(0), Integer(0)
        a, b = b + 1, a - 1
        f = -f
    s: Expr = Integer(0)
    if m:
        if isinstance(a, Integer) and isinstance(b, Integer):
            m = min(m, b.p - a.p + 1)
        if eps_zero or f.is_polynomial(i) is True:
            s = add(*[_at(f, i, a + Integer(k)) for k in range(m)])
        else:
            term = _at(f, i, a)
            if term != 0:
                test = _is_less(term, eps_mpf)
                if test is True and _decreasing(f, i, a, term):
                    return s, abs_(term)
                if test is None:
                    # a symbolic relational: no further
                    return term, Integer(0)
            s = term
            for k in range(1, m):
                term = _at(f, i, a + Integer(k))
                if term != 0 and _is_less(term, eps_mpf) is True and _decreasing(f, i, a + Integer(k), term):
                    return s, abs_(term)
                s = s + term
        if (b - a + 1) == Integer(m):
            return s, Integer(0)
        a = a + Integer(m)
    x = Dummy('x')
    integrand = _at(f, i, x)
    integral: Expr = Integral(integrand, x, a, b)
    if eval_integral:
        integral = integral.doit()
    s = s + integral

    def fpoint(e: Expr) -> tuple[Expr, Expr]:
        if isinstance(b, Infinity):
            return _at(e, i, a), Integer(0)
        return _at(e, i, a), _at(e, i, b)
    fa, fb = fpoint(f)
    iterm = (fa + fb) / 2
    g = f.diff(i)
    term = Integer(0)
    for k in range(1, n + 2):
        ga, gb = fpoint(g)
        term = bernoulli(2 * k) / factorial(2 * k) * (gb - ga)
        if k > n:
            break
        if not eps_zero and term != 0:
            value = _value3(term)
            if value is not None and (mpmath.isnan(mpmath.re(value)) or mpmath.isnan(mpmath.im(value))):
                from sympy_extras.core.numbers import nan
                return nan, nan
            if value is not None and abs(value) < eps_mpf:
                break
        s = s + term
        g = g.diff(i, i)
    return s + iterm, abs_(term)


def _at(e: Expr, i: Symbol, value: Expr) -> Expr:
    out = e.subs(i, value)
    if not isinstance(out, Expr):
        raise TypeError("the substitution made %s a non-expression" % (e,))
    return out
