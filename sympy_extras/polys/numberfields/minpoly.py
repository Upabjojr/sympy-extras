r"""Minimal polynomials of algebraic numbers over $\mathbb{Q}$, by
composition: the minimal polynomial of every subexpression is computed
first, and the operations combine them by resultants.

For $\alpha$ a root of $m_1$ and $\beta$ a root of $m_2$ (in the variable
$x$, with $y$ a second variable):

* $\alpha + \beta$ is a root of $\operatorname{Res}_y(m_1(x - y), m_2(y))$;
* $\alpha \beta$ of $\operatorname{Res}_y(y^{d_1} m_1(x/y), m_2(y))$, $d_1 =
  \deg m_1$;
* $\alpha^{n/d}$ of $\operatorname{Res}_y(m_1(y), x^d - y^n)$, after $m_1$
  is reversed for a negative exponent.

The resultants are computed over $\mathbb{Z}[x]$ and factored over
$\mathbb{Z}$; the factor vanishing at the number is chosen numerically,
with the precision doubled until one factor is far smaller than the
others. The atoms: the rationals, $i$, the golden ratio, the tribonacci
constant, and $\cos$, $\sin$, $\tan$, $\exp(i \pi \cdot)$ at rational
multiples of $\pi$ (roots of $T_q(x) - (-1)^p$, the tangent polynomial of
SymPy, $x^q - (-1)^p$). The result is primitive with a positive leading
coefficient, so it is unique.

Only numbers: SymPy's coefficients in $\mathbb{Q}(y_1, \dots)$ for an
expression with symbols, its ``domain`` option and its Gröbner variant
(``compose=False``) are not ported, nor ``CRootOf``.

Examples
========

>>> from sympy_extras.core import Symbol, Integer, Rational, I, pi
>>> from sympy_extras.functions import sqrt, cos, exp
>>> from sympy_extras.polys.numberfields.minpoly import minimal_polynomial
>>> x = Symbol('x')
>>> minimal_polynomial(sqrt(2) + sqrt(3), x), minimal_polynomial(Integer(2)**Rational(1, 3) + 1, x)
(x**4 - 10*x**2 + 1, x**3 - 3*x**2 + 3*x - 3)
>>> minimal_polynomial(cos(pi/7), x), minimal_polynomial(exp(2*I*pi/5), x), minimal_polynomial(1/(1 + I), x)
(8*x**3 - 4*x**2 - 4*x + 1, x**4 + x**3 + x**2 + x + 1, 2*x**2 - 2*x + 1)

References
==========

.. [1] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 3.3.4 (resultants and the arithmetic of
       algebraic numbers).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/numberfields/minpoly.py (_minpoly_compose,
# _choose_factor, _minpoly_tan, the special cases of _minpoly_sin and
# _minpoly_cos) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: dense polynomials over ZZ and ZZ[x] instead of expressions,
# generic resultants for sums, products and powers (no sum-of-surds
# shortcut, no Factors split of products), sin through cos, exp through
# x**q - (-1)**p; numbers over QQ only.
# ---------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from typing import Final

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import (AlgebraicNumber, GoldenRatioType, ImaginaryUnit, Integer, Pi, Rational,
                                       TribonacciConstantType)
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.densearith import dup_mul, dup_sub
from sympy_extras.polys.densebasic import dup_degree
from sympy_extras.polys.densetools import dup_primitive
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.euclidtools import dup_resultant
from sympy_extras.polys.factortools import dup_zz_factor
from sympy_extras.polys.polyerrors import NotAlgebraic

__all__ = ['minimal_polynomial', 'minpoly', 'minimal_polynomial_coeffs']

#: the ring ZZ[x] holding the coefficients of the polynomials in y
_ZX: Final[PolynomialRing[int]] = PolynomialRing(ZZ, 'x')

#: precisions (bits) tried when choosing the vanishing factor
_FIRST_PRECISION: Final = 64
_LAST_PRECISION: Final = 1 << 14


def minimal_polynomial(ex: ExprLike, x: Symbol) -> Expr:
    """The minimal polynomial of the algebraic number ``ex`` over the
    rationals, in ``x``, primitive with integer coefficients and a positive
    leading coefficient. ``NotAlgebraic`` when ``ex`` is not built from
    algebraic atoms; ``NotImplementedError`` with symbols."""
    coefficients = minimal_polynomial_coeffs(ex)
    n = len(coefficients) - 1
    return add(*[mul(Integer(c), pow(x, Integer(n - i))) for i, c in enumerate(coefficients) if c])


minpoly = minimal_polynomial


def minimal_polynomial_coeffs(ex: ExprLike) -> list[int]:
    """The coefficients of :func:`minimal_polynomial`, highest degree first."""
    e = to_expr(ex)
    if e.free_symbols:
        raise NotImplementedError('minimal polynomials with symbolic coefficients are not ported: %s' % e)
    return _normalised(_minpoly(e))


def _normalised(f: list[int]) -> list[int]:
    _, p = dup_primitive(f, ZZ)
    return [-c for c in p] if p[0] < 0 else p


def _minpoly(e: Expr) -> list[int]:
    """A multiple of the minimal polynomial that is irreducible over ZZ
    (not yet normalised)."""
    if isinstance(e, Rational):
        return [e.q, -e.p]
    if isinstance(e, ImaginaryUnit):
        return [1, 0, 1]
    if isinstance(e, GoldenRatioType):
        return [1, -1, -1]
    if isinstance(e, TribonacciConstantType):
        return [1, -1, -1, -1]
    if isinstance(e, AlgebraicNumber):
        return _minpoly(e.as_expr())
    if isinstance(e, Add):
        return _fold([a for a in e.args if isinstance(a, Expr)], True)
    if isinstance(e, Mul):
        return _fold([a for a in e.args if isinstance(a, Expr)], False)
    if isinstance(e, Pow):
        return _power(e)
    from sympy_extras.functions.elementary.exponential import Exp
    from sympy_extras.functions.elementary.trigonometric import Cos, Sin, Tan
    if isinstance(e, Cos):
        p, q = _rational_multiple(e, e.args[0], False)
        return _choose_factor(_chebyshev_cos(p, q), e)
    if isinstance(e, Sin):
        # sin(c pi) = cos((1/2 - c) pi)
        p, q = _rational_multiple(e, e.args[0], False)
        c = Fraction(1, 2) - Fraction(p, q)
        return _choose_factor(_chebyshev_cos(c.numerator, c.denominator), e)
    if isinstance(e, Tan):
        p, q = _rational_multiple(e, e.args[0], False)
        return _choose_factor(_tangent(p, q), e)
    if isinstance(e, Exp):
        # exp(i pi p/q) is a root of x**q - (-1)**p
        p, q = _rational_multiple(e, e.args[0], True)
        return _choose_factor([1] + [0]*(q - 1) + [-1 if p % 2 == 0 else 1], e)
    raise NotAlgebraic('%s does not seem to be an algebraic element' % e)


def _rational_multiple(e: Expr, argument: object, imaginary: bool) -> tuple[int, int]:
    """``(p, q)`` with the argument ``p/q*pi`` (``p/q*I*pi`` when
    ``imaginary``), ``NotAlgebraic`` otherwise."""
    if isinstance(argument, Expr):
        c, rest = argument.as_coeff_Mul()
        want: Expr = mul(ImaginaryUnit(), Pi()) if imaginary else Pi()
        if isinstance(c, Rational) and rest == want:
            return c.p, c.q
    raise NotAlgebraic('%s does not seem to be an algebraic element' % e)


def _chebyshev_cos(p: int, q: int) -> list[int]:
    """``T_q(x) - (-1)**p``, of which ``cos(p pi/q)`` is a root."""
    previous, current = [1], [1, 0]
    if q == 0:
        current = previous
    for _ in range(q - 1):
        previous, current = current, dup_sub(dup_mul([2, 0], current, ZZ), previous, ZZ)
    return dup_sub(current, [1 if p % 2 == 0 else -1], ZZ)


# PORTED FROM SYMPY: sympy/polys/numberfields/minpoly.py:_minpoly_tan
def _tangent(p: int, q: int) -> list[int]:
    """A polynomial with the root ``tan(p pi/q)``: the real or imaginary
    part of ``(1 + i x)**n`` with ``n`` the denominator of ``2 p/q``."""
    c = Fraction(2*p, q)
    n = c.denominator
    a = n if c.numerator % 2 == 0 else 1
    coefficients = [0]*(n + 1)   # lowest degree first
    for k in range((c.numerator + 1) % 2, n + 1, 2):
        coefficients[k] = a
        a = -(a*(n - k - 1)*(n - k)) // ((k + 1)*(k + 2))
    f = coefficients[::-1]
    while f and f[0] == 0:
        f = f[1:]
    return f


def _fold(terms: list[Expr], is_sum: bool) -> list[int]:
    """The minimal polynomial of a sum or product, one term at a time."""
    current: Expr = terms[0]
    m = _normalised(_minpoly(current))
    for t in terms[1:]:
        mt = _normalised(_minpoly(t))
        current = add(current, t) if is_sum else mul(current, t)
        if not is_sum and (m == [1, 0] or mt == [1, 0]):
            m = [1, 0]
            continue
        r = _sum_resultant(m, mt) if is_sum else _product_resultant(m, mt)
        if dup_degree(m) == 1 or dup_degree(mt) == 1:
            # one of them rational: the resultant is the other polynomial
            # shifted or scaled, so irreducible
            m = _normalised(r)
        else:
            m = _normalised(_choose_factor(r, current))
    return m


def _in_y(m: list[int]) -> list[list[int]]:
    """``m(y)`` as a polynomial in ``y`` over ``ZZ[x]``."""
    return [[c] if c else [] for c in m]


def _sum_resultant(m1: list[int], m2: list[int]) -> list[int]:
    """``Res_y(m1(x - y), m2(y))``."""
    x_minus_y: list[list[int]] = [[-1], [1, 0]]
    shifted: list[list[int]] = []
    for c in m1:   # Horner in the ring ZZ[x][y]
        shifted = dup_mul(shifted, x_minus_y, _ZX)
        if c:
            shifted = _add_constant(shifted, [c])
    return dup_resultant(shifted, _in_y(m2), _ZX)


def _product_resultant(m1: list[int], m2: list[int]) -> list[int]:
    """``Res_y(y**d m1(x/y), m2(y))``, both numbers nonzero: with ``m1 =
    sum a_k x**k`` the coefficient of ``y**(d - k)`` is ``a_k x**k``."""
    d = len(m1) - 1
    scaled: list[list[int]] = [[m1[d - k]] + [0]*k if m1[d - k] else [] for k in range(d + 1)]
    return dup_resultant(scaled, _in_y(m2), _ZX)


def _add_constant(f: list[list[int]], c: list[int]) -> list[list[int]]:
    if not f:
        return [c]
    return f[:-1] + [_ZX.add(f[-1], c)]


def _power(e: Pow) -> list[int]:
    base, exponent = e.base, e.exp
    if not isinstance(exponent, Rational):
        raise NotAlgebraic('%s does not seem to be an algebraic element' % e)
    m = _normalised(_minpoly(base))
    n, d = exponent.p, exponent.q
    if n < 0:
        if m == [1, 0]:
            raise ZeroDivisionError('%s is zero' % base)
        m = m[::-1]
        while m and m[0] == 0:
            m = m[1:]
        n = -n
    if n == 1 and d == 1:
        return m
    # Res_y(m(y), x**d - y**n): x**d - y**n in y over ZZ[x]
    target: list[list[int]] = [[-1]] + [[] for _ in range(n - 1)] + [[1] + [0]*d]
    r = dup_resultant(_in_y(m), target, _ZX)
    return _choose_factor(r, e)


def _choose_factor(f: list[int], v: Expr) -> list[int]:
    """The irreducible factor of ``f`` vanishing at ``v``."""
    _, factors = dup_zz_factor(f, ZZ)
    candidates = [g for g, _ in factors]
    if len(candidates) == 1:
        return candidates[0]
    precision = _FIRST_PRECISION
    while precision <= _LAST_PRECISION:
        value = numeric_value(v, precision + 16)
        if value is None:
            raise NotAlgebraic('%s has no numerical value' % v)
        with mpmath.workprec(precision):
            sizes: list[tuple[mpmath.mpf, int]] = []
            for i, g in enumerate(candidates):
                s = mpmath.mpc(0)
                for c in g:
                    s = s*value + c
                sizes.append((abs(s), i))
            sizes.sort()
            (a, i), (b, _) = sizes[0], sizes[1]
            if b > a*10**6:
                return candidates[i]
        precision *= 2
    raise NotImplementedError('multiple candidates for the minimal polynomial of %s' % v)
