r"""Integration of rational functions on the typed core: Hermite's
reduction (Horowitz–Ostrogradsky) and the logarithmic part of Lazard,
Rioboo and Trager, with Rioboo's conversion of the complex logarithms to
real arctangents.

Ported from SymPy's ``sympy.integrals.rationaltools`` (in the version of
Aaron Meurer's pull requests, see the notice below). Given ``f = p/q`` in
``K(x)``:

1. ``p/q`` is cancelled and its polynomial part integrated;
2. :func:`ratint_ratpart` writes the proper part as ``A' + B`` with
   ``B`` of square-free denominator ``v``, ``u = gcd(q, q')``, ``v =
   q/u``, the coefficients of ``A`` and ``B`` solving the linear system
   ``p = A' v - A (u' v/u) + B u`` [Bronstein, 2.2];
3. :func:`ratint_logpart` writes ``B = s/v`` as ``sum_i sum_{q_i(a) = 0}
   a log(S_i(a, x))``: ``R(t) = Res_x(v, s - t v')`` by the subresultant
   PRS, ``q_i`` its square-free factors, ``S_i`` the subresultant of
   degree ``i`` made monic modulo ``q_i`` [Bronstein, 2.5];
4. a sum over the roots is written with the roots of the factors when
   they are found and the coefficients are real, a pair of complex
   conjugate roots ``u +- I v`` as ``u log(A**2 + B**2) + v
   (2 atan(...))`` (:func:`log_to_real`, :func:`log_to_atan`, Rioboo's
   algorithm [Bronstein, 2.8]), and otherwise as a
   :class:`~sympy_extras.polys.rootoftools.RootSum`.

The coefficients are rational numbers (the domain ``QQ``) or else
expressions of the core in the domain ``EX``
(:mod:`sympy_extras.polys.domains.expressiondomain`), as SymPy's
``Poly(..., composite=False, field=True)`` takes them: parameters,
``sqrt(2)``, ``pi``. SymPy's domains ``QQ_I`` and ``ZZ_I`` (coefficients
with ``I``) are taken as ``EX``.

References
==========

.. [Bronstein] M. Bronstein, Symbolic Integration I: Transcendental
   Functions, Second Edition, Springer-Verlag, 2005, pp. 35-70.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.integrals.risch.rationaltools import ratint
>>> x = symbols('x')
>>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)
(12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)
>>> ratint(1/(x**2 + x + 1), x)
2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
>>> ratint(1/(x**3 - x - 1), x, real=False)
RootSum(23*_t**3 + 3*_t - 1, Lambda(_t, _t*log(-46*_t**2/9 - 23*_t/9 + x - 4/9)))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/rationaltools.py (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the typed core and the typed dense polynomials over QQ and EX
# instead of SymPy's Poly; the linear system of Horowitz–Ostrogradsky
# solved by elimination over the domain; RootSum of polys.rootoftools.
# ---------------------------------------------------------------------
# Ported from SymPy (sympy.integrals.rationaltools) and from Aaron Meurer's pull
# requests to SymPy: sympy/sympy#30180 and #30221 ("Risch: Implement most
# remaining exp-log cases from Bronstein", branch risch-rde-cancellation
# @d7dafa43) and #30292 ("Hypertangent cases in the Risch algorithm", branch
# risch-hypertangent @66bc72d6, the version taken here), by Claude Fable 5.1
# for sympy-extras; rewritten on the typed core by Claude Opus 5.5. The
# algorithm is Bronstein's as implemented by Aaron Meurer. SymPy's licence
# (BSD 3-clause, copyright the SymPy Development Team) applies to this file:
# see LICENSE-SymPy in this directory.
from __future__ import annotations

from fractions import Fraction
from math import gcd as integer_gcd
from typing import Final, Optional, Sequence, TypeVar, Union

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import Lambda
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import I, ImaginaryUnit, Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.functions.elementary.trigonometric import atan
from sympy_extras.polys.densearith import dup_div, dup_exquo, dup_mul, dup_neg, dup_rem, dup_sub
from sympy_extras.polys.densebasic import dup_degree, dup_LC, dup_strip
from sympy_extras.polys.densetools import dup_diff, dup_integrate, dup_monic
from sympy_extras.polys.domains import QQ, ZZ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.expressiondomain import EX, EXR, ExpressionDomain
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.euclidtools import dup_gcd, dup_gcdex, dup_inner_subresultants, dup_invert
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel
from sympy_extras.polys.rootoftools import expr_of_coefficients, polynomial_coefficients, root_sum
from sympy_extras.polys.sqfreetools import dup_sqf_list

__all__ = ['ratint', 'ratint_ratpart', 'ratint_logpart', 'log_to_atan', 'log_to_real', 'polynomial_part']

T = TypeVar('T')

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


# ---------------------------------------------------------------------
# polynomials over QQ or EX

def _is_rational(coefficients: Sequence[Expr]) -> bool:
    return all(isinstance(c, Rational) for c in coefficients)


def _qq(coefficients: Sequence[Expr]) -> list[Fraction]:
    return dup_strip([QQ.from_expr(c) for c in coefficients], QQ)


def _ex(coefficients: Sequence[Expr], K: ExpressionDomain = EX) -> list[Expr]:
    return dup_strip([K.from_expr(c) for c in coefficients], K)


def _expression_domain(coefficients: Sequence[Expr]) -> ExpressionDomain:
    """``EX`` (SymPy's forms) for coefficients with parameters, and ``EX``
    with the denominators rationalized (``EXR``) for coefficients with
    radicals of numbers, whose arithmetic the plain ``EX`` lets grow."""
    return EXR if any(_has_square_root(c) for c in coefficients) else EX


def _has_square_root(c: Expr) -> bool:
    """Whether ``c`` has a square root of a number (which ``radsimp``
    takes out of the denominators)."""
    from sympy_extras.core.basic import preorder_traversal
    return any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2
               and not node.base.free_symbols for node in preorder_traversal(c))


def _as_expr(f: Sequence[T], K: Domain[T], x: Expr) -> Expr:
    """SymPy's ``Poly.as_expr``: ``sum c_k x**k``."""
    return expr_of_coefficients([K.to_expr(c) for c in f], x)


def _lift(f: Sequence[T], K: Domain[T]) -> list[list[T]]:
    """A polynomial in ``x`` over ``K`` as one over ``K[t]``."""
    return [[c] if not K.is_zero(c) else [] for c in f]


def _bivariate_expr(h: Sequence[list[T]], K: Domain[T], x: Expr, t: Expr) -> Expr:
    """A polynomial in ``x`` over ``K[t]`` as an expression."""
    return _sympy_expr([_as_expr(c, K, t) for c in h], x)


# ---------------------------------------------------------------------
# SymPy's normalisations of the domains QQ and EX

# PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_cancel (univariate, QQ and EX)
def _cancel(f: list[T], g: list[T], K: Domain[T]) -> tuple[Expr, list[T], list[T]]:
    """SymPy's ``Poly.cancel(include=False)``: ``(c, p, q)`` with ``f/g =
    c p/q``, over ``QQ`` the denominators cleared and the gcd taken over
    ``ZZ``, over ``EX`` the monic gcd divided out; the signs of the leading
    coefficients moved into ``c``."""
    if K.name == 'QQ':
        F = [QQ.from_expr(K.to_expr(c)) for c in f]
        G = [QQ.from_expr(K.to_expr(c)) for c in g]
        P, Q, cp, cq = _cancel_qq(F, G)
        p = [K.from_expr(QQ.to_expr(c)) for c in P]
        q = [K.from_expr(QQ.to_expr(c)) for c in Q]
        return Rational(cp, cq), p, q
    h = dup_monic(dup_gcd(f, g, K), K) if f and g else ([K.one] if f or g else [])
    p = dup_div(f, h, K)[0] if h else f
    q = dup_div(g, h, K)[0] if h else g
    sign = 1
    p_neg = bool(p) and K.is_negative(dup_LC(p, K))
    q_neg = bool(q) and K.is_negative(dup_LC(q, K))
    if p_neg and q_neg:
        p, q = dup_neg(p, K), dup_neg(q, K)
    elif p_neg:
        sign, p = -sign, dup_neg(p, K)
    elif q_neg:
        sign, q = -sign, dup_neg(q, K)
    return Integer(sign), p, q


def _cancel_qq(f: list[Fraction], g: list[Fraction]) -> tuple[list[Fraction], list[Fraction], int, int]:
    from sympy_extras.polys.densetools import dup_clear_denoms
    cq, F = dup_clear_denoms(f, QQ)
    cp, G = dup_clear_denoms(g, QQ)
    Fz = [c.numerator for c in F]
    Gz = [c.numerator for c in G]
    h = dup_gcd(Fz, Gz, ZZ)
    pz = dup_exquo(Fz, h, ZZ) if h else Fz
    qz = dup_exquo(Gz, h, ZZ) if h else Gz
    common = integer_gcd(cp, cq)
    cp, cq = cp // common, cq // common
    p_neg = bool(pz) and pz[0] < 0
    q_neg = bool(qz) and qz[0] < 0
    if p_neg and q_neg:
        pz, qz = [-c for c in pz], [-c for c in qz]
    elif p_neg:
        cp, pz = -cp, [-c for c in pz]
    elif q_neg:
        cp, qz = -cp, [-c for c in qz]
    return [Fraction(c) for c in pz], [Fraction(c) for c in qz], cp, cq


def _integer_primitive(f: list[T], K: Domain[T]) -> list[T]:
    """SymPy's primitive part over ``ZZ`` or ``QQ`` of a polynomial with
    rational coefficients: the denominators cleared and the (positive)
    integer content divided out."""
    values = [QQ.from_expr(K.to_expr(c)) for c in f]
    denominator = 1
    for v in values:
        denominator = denominator*v.denominator // integer_gcd(denominator, v.denominator)
    numerators = [(v*denominator).numerator for v in values]
    content = 0
    for n in numerators:
        content = integer_gcd(content, n)
    content = content or 1
    return dup_strip([K.from_expr(Integer(n // content)) for n in numerators], K)


def _is_negative_real(c: Expr) -> bool:
    """SymPy's ``c.is_extended_real and (c < 0) == True``."""
    if isinstance(c, Rational):
        return c.p < 0
    if c.is_extended_real is not True:
        return False
    negative = c.is_extended_negative
    if negative is None and not c.free_symbols:
        # a number: its sign by a value which no rounding error decides
        from sympy_extras._numeric import reliable_value
        value = reliable_value(c, 30)
        negative = None if value is None else value.is_extended_negative
    return negative is True


def _has_algebraic_atom(c: Expr) -> bool:
    from sympy_extras.core.basic import preorder_traversal
    return any(_is_algebraic_atom(node) for node in preorder_traversal(c))


def _is_algebraic_atom(e: Basic) -> bool:
    """A number which SymPy's ``construct_domain`` takes as an algebraic
    one (the domain ``EX``): a radical of a number, ``I``."""
    if isinstance(e, ImaginaryUnit):
        return True
    return isinstance(e, Pow) and isinstance(e.exp, Rational) and e.exp.q != 1 and not e.base.free_symbols


def _has_symbolic_denominator(c: Expr) -> bool:
    return not isinstance(c.as_numer_denom()[1], Rational)


def sympy_form(coefficients: Sequence[Expr], field: bool = False) -> list[Expr]:
    """The coefficients as SymPy's ``Poly(...).as_expr()`` writes them
    over the domain ``construct_domain`` gives them (``field``: over its
    fraction field, ``to_field``): over a fraction field of parameters
    (a coefficient with a symbolic denominator, or ``field``) each one as
    a single fraction (``cancel``); over ``QQ``, ``EX`` (algebraic numbers)
    and the polynomial rings as they are."""
    nonzero = [c for c in coefficients if c != _ZERO]
    if _is_rational(nonzero) or any(_has_algebraic_atom(c) for c in nonzero):
        return list(coefficients)
    if field or any(_has_symbolic_denominator(c) for c in nonzero):
        return [cancel(c) if c != _ZERO else c for c in coefficients]
    return list(coefficients)


def _sympy_expr(coefficients: Sequence[Expr], x: Expr, field: bool = False) -> Expr:
    """``Poly(sum c_k x**k, x).as_expr()`` (:func:`sympy_form`)."""
    return expr_of_coefficients(sympy_form(coefficients, field), x)


def _primitive_expr(coefficients: list[Expr]) -> list[Expr]:
    """SymPy's ``Poly.primitive()[1]`` of the polynomial with these
    coefficients over the domain ``construct_domain`` gives them: the
    integer content over ``ZZ`` and ``QQ``, the polynomial content over
    ``ZZ[params]``, nothing over ``EX`` (algebraic numbers)."""
    nonzero = [c for c in coefficients if c != _ZERO]
    if not nonzero:
        return coefficients
    if _is_rational(nonzero):
        return [QQ.to_expr(c) for c in _integer_primitive(_qq(coefficients), QQ)]
    if any(_has_algebraic_atom(c) for c in nonzero):
        return coefficients
    if any(c.as_numer_denom()[1].free_symbols for c in nonzero):
        return coefficients
    from sympy_extras.polys.polytools import gcd_list
    try:
        content = gcd_list(nonzero)
    except _FAILURES:
        return coefficients
    if content == _ONE or content == _ZERO:
        return coefficients
    return [cancel(mul(c, pow(content, Integer(-1)))) for c in coefficients]


# ---------------------------------------------------------------------
# the linear system of Horowitz–Ostrogradsky

def _solve(rows: list[list[T]], rhs: list[T], K: Domain[T]) -> list[T]:
    """The solution of the square nonsingular system ``rows c = rhs`` over
    the field ``K`` (Gaussian elimination)."""
    n = len(rows)
    M = [list(r) + [b] for r, b in zip(rows, rhs)]
    for col in range(n):
        pivot = next((i for i in range(col, n) if not K.is_zero(M[i][col])), None)
        if pivot is None:
            raise ValueError('the system of Horowitz and Ostrogradsky is singular')
        M[col], M[pivot] = M[pivot], M[col]
        inverse = K.invert(M[col][col])
        M[col] = [K.mul(inverse, c) for c in M[col]]
        for i in range(n):
            if i != col and not K.is_zero(M[i][col]):
                factor = M[i][col]
                M[i] = [K.sub(a, K.mul(factor, b)) for a, b in zip(M[i], M[col])]
    return [M[i][n] for i in range(n)]


def _coefficient(f: list[T], k: int, K: Domain[T]) -> T:
    """The coefficient of ``x**k``."""
    n = dup_degree(f)
    return f[n - k] if 0 <= k <= n else K.zero


def _ratpart(f: list[T], g: list[T], K: Domain[T]) -> tuple[list[T], list[T], list[T], list[T]]:
    """``(A, u, B, v)`` with ``f/g = (A/u)' + B/v``, ``v`` square-free."""
    u = dup_monic(dup_gcd(g, dup_diff(g, 1, K), K), K)
    v = dup_div(g, u, K)[0]
    n, m = dup_degree(u), dup_degree(v)
    w = dup_div(dup_mul(dup_diff(u, 1, K), v, K), u, K)[0]
    size = n + m
    columns: list[list[T]] = []
    for i in range(n):
        # the unknown coefficient of x**i in A: (x**i)' v - x**i w
        power = [K.one] + [K.zero]*i
        derivative = dup_diff(power, 1, K)
        columns.append(dup_sub(dup_mul(derivative, v, K), dup_mul(power, w, K), K))
    for j in range(m):
        # the unknown coefficient of x**j in B: x**j u
        columns.append(dup_mul([K.one] + [K.zero]*j, u, K))
    rows = [[_coefficient(column, k, K) for column in columns] for k in range(size)]
    rhs = [_coefficient(f, k, K) for k in range(size)]
    solution = _solve(rows, rhs, K) if size else []
    A = dup_strip(list(reversed(solution[:n])), K)
    B = dup_strip(list(reversed(solution[n:])), K)
    return A, u, B, v


def _ratpart_exprs(f: list[T], g: list[T], K: Domain[T], x: Symbol) -> tuple[Expr, Expr]:
    A, u, B, v = _ratpart(f, g, K)
    rat_part = _cancel_in(mul(_as_expr(A, K, x), pow(_as_expr(u, K, x), Integer(-1))), x)
    log_part = _cancel_in(mul(_as_expr(B, K, x), pow(_as_expr(v, K, x), Integer(-1))), x)
    return rat_part, log_part


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel (the domain EX)
def _cancel_in(e: Expr, x: Symbol) -> Expr:
    """SymPy's ``cancel(e, x)``: the native one, and over ``EX`` (algebraic
    numbers in the coefficients, which the native one would take as
    generators) the monic gcd over ``EX`` divided out and the signs of the
    leading coefficients normalised by ``could_extract_minus_sign``."""
    if not _has_algebraic_atom(e):
        return cancel(e, x)
    domain = _expression_domain([e])
    from sympy_extras.core.exprtools import factor_terms
    from sympy_extras.simplify.signsimp import signsimp
    p, q = factor_terms(signsimp(e), radical=True).as_numer_denom()
    try:
        P = _ex(polynomial_coefficients(p, x), domain)
        Q = _ex(polynomial_coefficients(q, x), domain)
    except ValueError:
        return cancel(e, x)
    sign, P, Q = _cancel(P, Q, domain)
    return mul(sign, mul(_as_expr(P, domain, x), pow(_as_expr(Q, domain, x), Integer(-1))))


# ---------------------------------------------------------------------
# the logarithmic part

#: a polynomial in x over K[t] (each coefficient a list over K)
Bivariate = list[list[T]]


def _logpart(f: list[T], g: list[T], K: Domain[T], t: Symbol) -> list[tuple[Bivariate[T], list[T]]]:
    """Lazard–Rioboo–Trager on ``f/g`` over ``K``: the pairs ``(S_i, q_i)``,
    ``S_i`` in ``K[t][x]`` and ``q_i`` in ``K[t]``."""
    R: PolynomialRing[T] = PolynomialRing(K, t.name)
    a: Bivariate[T] = _lift(g, K)
    derivative = dup_diff(g, 1, K)
    size = max(len(f), len(derivative))
    padded_f = [K.zero]*(size - len(f)) + list(f)
    padded_d = [K.zero]*(size - len(derivative)) + list(derivative)
    b: Bivariate[T] = dup_strip([dup_strip([K.neg(d), c], K) for c, d in zip(padded_f, padded_d)], R)
    sequence, scalars = dup_inner_subresultants(a, b, R)
    if not sequence or dup_degree(sequence[-1]) > 0:
        res: list[T] = []
    else:
        res = scalars[-1]
    if not res:
        raise ValueError("BUG: the resultant of %s and %s is zero" % (g, f))
    by_degree: dict[int, Bivariate[T]] = {}
    for r in sequence:
        by_degree[dup_degree(r)] = r
    rational = _is_rational([K.to_expr(c) for c in res])
    C, factors = dup_sqf_list(res, K)
    normalised: list[tuple[list[T], int]] = []
    for k, (q, i) in enumerate(factors):
        if rational:
            q = _integer_primitive(q, K)
            if k == 0 and _is_negative_real(K.to_expr(C)):
                q = dup_neg(q, K)
        elif k == 0 and _is_negative_real(K.to_expr(C)):
            q = [K.mul(c, C) for c in q]
        normalised.append((q, i))
    H: list[tuple[Bivariate[T], list[T]]] = []
    for q, i in normalised:
        if dup_degree(g) == i:
            H.append((a, q))
            continue
        h = by_degree[i]
        h_lc = h[0]
        _, lc_factors = dup_sqf_list(h_lc, K)
        for factor, j in lc_factors:
            common = dup_gcd(factor, q, K)
            if dup_degree(common) > 0:
                divisor = R.pow(common, j)
                h = [R.exquo(c, divisor) for c in h]
        inverse = dup_invert(h_lc, q, K)
        coefficients: Bivariate[T] = []
        first = True
        for c in h:
            if not c:
                coefficients.append([])
            elif first:
                coefficients.append([K.one])
                first = False
            else:
                coefficients.append(dup_rem(dup_mul(inverse, c, K), q, K))
        H.append((coefficients, q))
    return H


# ---------------------------------------------------------------------
# the public functions

def _field_of(coefficients: Sequence[Expr]) -> bool:
    """Whether the coefficients are rational (the domain ``QQ``; ``EX``
    otherwise)."""
    return _is_rational(coefficients)


def _dummy_of(symbol: Union[str, Symbol]) -> Symbol:
    if isinstance(symbol, Symbol):
        return Dummy(symbol.name, **symbol.given_assumptions)
    return Dummy(symbol)


def ratint(f: Union[Expr, tuple[Expr, Expr]], x: Symbol, symbol: Union[str, Symbol] = 't',
           real: Optional[bool] = None) -> Expr:
    """An antiderivative of the rational function ``f`` (or ``p/q`` given
    as the pair ``(p, q)``) in ``x``: the polynomial and rational parts,
    and the logarithmic part with the roots written out where they are
    found (real arctangents when ``real``, which by default holds when
    every atom of ``f`` but ``x`` is real), a ``RootSum`` in a dummy named
    ``symbol`` elsewhere.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.rationaltools import ratint
    >>> x, a = symbols('x a')
    >>> ratint(1/(x**2 + 1), x), ratint(1/(x**2 + 1), x, real=False)
    (atan(x), -I*log(x - I)/2 + I*log(x + I)/2)
    >>> ratint(x/(x**2 - a**2), x)
    log(-a**2 + x**2)/2
    >>> ratint((x**3 + 1)/(x*(x - 1)**3), x)
    -x/(x**2 - 2*x + 1) - log(x) + 2*log(x - 1)
    """
    if isinstance(f, tuple):
        numerator, denominator = f
    else:
        numerator, denominator = f.as_numer_denom()
    nc = polynomial_coefficients(numerator, x)
    dc = polynomial_coefficients(denominator, x)
    if _field_of(nc + dc):
        return _ratint(_qq(nc), _qq(dc), QQ, f, x, symbol, real)
    E = _expression_domain(nc + dc)
    return _ratint(_ex(nc, E), _ex(dc, E), E, f, x, symbol, real)


def _ratint(p0: list[T], q0: list[T], K: Domain[T], f: Union[Expr, tuple[Expr, Expr]], x: Symbol,
            symbol: Union[str, Symbol], real: Optional[bool]) -> Expr:
    if not q0:
        raise ZeroDivisionError('the denominator of %s is zero' % (f,))
    coeff, p, q = _cancel(p0, q0, K)
    poly, p = dup_div(p, q, K)
    result = _as_expr(dup_integrate(poly, 1, K), K, x)
    if not p:
        return mul(coeff, result)
    g, h = _ratpart_exprs(p, q, K, x)
    P_expr, Q_expr = h.as_numer_denom()
    pc = polynomial_coefficients(P_expr, x)
    qc = polynomial_coefficients(Q_expr, x)
    if K is QQ and _field_of(pc + qc):
        terms = _logarithmic_terms(_qq(pc), _qq(qc), QQ, f, x, symbol, real)
    else:
        E = _expression_domain(pc + qc)
        terms = _logarithmic_terms(_ex(pc, E), _ex(qc, E), E, f, x, symbol, real)
    quotient, eps = terms
    result = add(result, g, quotient)
    if eps is not None:
        result = add(result, eps)
    return mul(coeff, result)


def _atoms_real(f: Union[Expr, tuple[Expr, Expr]], x: Symbol) -> bool:
    """SymPy's default of ``real``: every atom of ``f`` but ``x`` is
    real."""
    if isinstance(f, tuple):
        atoms = f[0].atoms() | f[1].atoms()
    else:
        atoms = f.atoms()
    for elt in atoms - {x}:
        if not isinstance(elt, Expr) or elt.is_extended_real is not True:
            return False
    return True


def _logarithmic_terms(P: list[T], Q: list[T], K: Domain[T], f: Union[Expr, tuple[Expr, Expr]], x: Symbol,
                       symbol: Union[str, Symbol], real: Optional[bool]) -> tuple[Expr, Optional[Expr]]:
    """The integral of the polynomial part of ``P/Q`` and the logarithmic
    part of the rest (``None`` when there is none)."""
    quotient, r = dup_div(P, Q, K)
    integral = _sympy_expr([K.to_expr(c) for c in dup_integrate(quotient, 1, K)], x, field=True)
    if not r:
        return integral, None
    t = _dummy_of(symbol)
    L = _logpart(r, Q, K, t)
    if real is None:
        real = _atoms_real(f, x)
    eps: Expr = _ZERO
    for h, q in L:
        q_expr = _as_expr(q, K, t)
        h_coefficients = [_as_expr(c, K, t) for c in h]
        if not any(c.has(t) for c in h_coefficients):
            h_coefficients = _primitive_expr(h_coefficients)
        h_expr = _sympy_expr(h_coefficients, x)
        if real:
            R = log_to_real(h_expr, q_expr, x, t)
            if R is not None:
                eps = add(eps, R)
                continue
        eps = add(eps, root_sum(q_expr, Lambda(t, mul(t, log(h_expr))), t, quadratic=True))
    return integral, eps


def ratint_ratpart(f: ExprLike, g: ExprLike, x: Symbol) -> tuple[Expr, Expr]:
    """Horowitz–Ostrogradsky: for polynomials ``f`` and ``g`` in ``x``,
    coprime, ``deg f < deg g``, the fractions ``A`` and ``B`` with ``f/g =
    A' + B``, ``B`` with a square-free denominator.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.rationaltools import ratint_ratpart
    >>> x, y = symbols('x y')
    >>> ratint_ratpart(1, x + 1, x)
    (0, 1/(x + 1))
    >>> ratint_ratpart(1, x**2 + y**2, x)
    (0, 1/(x**2 + y**2))
    >>> ratint_ratpart(36, x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x)
    ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))
    """
    fc = polynomial_coefficients(to_expr(f), x)
    gc = polynomial_coefficients(to_expr(g), x)
    if _field_of(fc + gc):
        return _ratpart_exprs(_qq(fc), _qq(gc), QQ, x)
    E = _expression_domain(fc + gc)
    return _ratpart_exprs(_ex(fc, E), _ex(gc, E), E, x)


def polynomial_part(f: ExprLike, g: ExprLike, x: Symbol) -> tuple[Expr, Expr]:
    """The antiderivative of the quotient of the polynomials ``f`` and
    ``g`` in ``x`` and the remainder (over ``QQ``, or ``EX``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.rationaltools import polynomial_part
    >>> x, a = symbols('x a')
    >>> polynomial_part(x**3 + a, x + 1, x)
    (x**3/3 - x**2/2 + x, a - 1)
    """
    fc = polynomial_coefficients(to_expr(f), x)
    gc = polynomial_coefficients(to_expr(g), x)
    if _field_of(fc + gc):
        return _polynomial_part(_qq(fc), _qq(gc), QQ, x)
    E = _expression_domain(fc + gc)
    return _polynomial_part(_ex(fc, E), _ex(gc, E), E, x)


def _polynomial_part(f: list[T], g: list[T], K: Domain[T], x: Symbol) -> tuple[Expr, Expr]:
    quotient, remainder = dup_div(f, g, K)
    return _as_expr(dup_integrate(quotient, 1, K), K, x), _as_expr(remainder, K, x)


def ratint_logpart(f: ExprLike, g: ExprLike, x: Symbol, t: Optional[Symbol] = None) -> list[tuple[Expr, Expr]]:
    r"""Lazard–Rioboo–Trager: for polynomials ``f`` and ``g`` in ``x``,
    coprime, ``deg f < deg g``, ``g`` square-free, the pairs ``(S_i, q_i)``
    (polynomials in ``x`` and ``t``, and in ``t``) with

    .. math:: \frac{f}{g} = \frac{d}{dx} \sum_i \sum_{q_i(a) = 0} a \log S_i(a, x).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.rationaltools import ratint_logpart
    >>> x, t = symbols('x t')
    >>> ratint_logpart(1, x**2 + x + 1, x, t)
    [(3*t/2 + x + 1/2, 3*t**2 + 1)]
    >>> ratint_logpart(12, x**2 - x - 2, x, t)
    [(-3*t/8 + x - 1/2, 16 - t**2)]
    """
    variable = t if t is not None else Dummy('t')
    fc = polynomial_coefficients(to_expr(f), x)
    gc = polynomial_coefficients(to_expr(g), x)
    if _field_of(fc + gc):
        return [(_bivariate_expr(h, QQ, x, variable), _as_expr(q, QQ, variable))
                for h, q in _logpart(_qq(fc), _qq(gc), QQ, variable)]
    E = _expression_domain(fc + gc)
    return [(_bivariate_expr(h, E, x, variable), _as_expr(q, E, variable))
            for h, q in _logpart(_ex(fc, E), _ex(gc, E), E, variable)]


# ---------------------------------------------------------------------
# real forms

def _dense_pair(f: Expr, g: Expr, x: Symbol) -> tuple[list[Expr], list[Expr], bool]:
    fc = polynomial_coefficients(to_expr(f), x)
    gc = polynomial_coefficients(to_expr(g), x)
    return fc, gc, _field_of(fc + gc)


def log_to_atan(f: ExprLike, g: ExprLike, x: Symbol) -> Expr:
    """Rioboo's conversion: for polynomials ``f`` and ``g`` in ``x`` over a
    real field, ``g != 0``, a sum ``h`` of arctangents of polynomials with
    ``h' = (I log((f + I g)/(f - I g)))'``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt, Rational
    >>> from sympy_extras.integrals.risch.rationaltools import log_to_atan
    >>> x = symbols('x')
    >>> log_to_atan(x, 1, x)
    2*atan(x)
    >>> log_to_atan(x + Rational(1, 2), sqrt(3)/2, x)
    2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)
    """
    fc, gc, rational = _dense_pair(to_expr(f), to_expr(g), x)
    if rational:
        return _log_to_atan(_qq(fc), _qq(gc), QQ, x)
    return _log_to_atan(_ex(fc), _ex(gc), EX, x)


# PORTED FROM SYMPY: sympy/integrals/rationaltools.py:log_to_atan
def _log_to_atan(f: list[T], g: list[T], K: Domain[T], x: Symbol) -> Expr:
    if dup_degree(f) < dup_degree(g):
        f, g = dup_neg(g, K), f
    p, q = dup_div(f, g, K)
    if not q:
        return mul(Integer(2), atan(_sympy_expr([K.to_expr(c) for c in p], x, field=True)))
    s, t, h = dup_gcdex(g, dup_neg(f, K), K)
    u = dup_div(dup_add_mul(f, s, g, t, K), h, K)[0]
    A = mul(Integer(2), atan(_sympy_expr([K.to_expr(c) for c in u], x, field=True)))
    return add(A, _log_to_atan(s, t, K, x))


def dup_add_mul(f: list[T], s: list[T], g: list[T], t: list[T], K: Domain[T]) -> list[T]:
    """``f s + g t``."""
    from sympy_extras.polys.densearith import dup_add
    return dup_add(dup_mul(f, s, K), dup_mul(g, t, K), K)


def _roots_of(q: Expr, t: Symbol) -> Optional[dict[Expr, int]]:
    """The roots of ``q`` in ``t`` with multiplicities (the native
    ``roots`` of ``q`` with its denominators cleared; the formulas over
    ``EX`` for degrees one and two), ``None`` when they are not all
    found."""
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.rootoftools import _Polynomial, _linear_root, _quadratic_roots
    coefficients = polynomial_coefficients(q, t)
    degree = len(coefficients) - 1
    numerator, _ = cancel(q, t).as_numer_denom()
    found: dict[Expr, int] = {}
    try:
        result = roots(numerator, t)
    except _FAILURES:
        cleared = polynomial_coefficients(numerator, t)
        radical = _has_algebraic_atom(numerator)
        if radical and degree == 1:
            found = {_linear_root(_Polynomial(None, cleared, t)): 1}
        elif radical and degree == 2:
            for r in _quadratic_roots(_Polynomial(None, cleared, t)):
                found[r] = found.get(r, 0) + 1
        else:
            generic = _generic_roots(numerator, t, radical)
            if generic is None:
                return None
            found = generic
    else:
        if not isinstance(result, dict):
            return None
        found = result
    if sum(found.values()) != degree:
        return None
    return found


def _generic_roots(q: Expr, t: Symbol, radicals: bool) -> Optional[dict[Expr, int]]:
    """The roots of ``q`` with the constants which the native ``Poly``
    does not take (``pi``, ``tan(1)``; with ``radicals`` also ``sqrt(2)``)
    replaced by dummies, as parameters, and put back in the roots (as
    SymPy takes the transcendental constants as generators; the root
    formulas are identities in the parameters)."""
    from sympy_extras.polys.polyroots import roots
    from sympy_extras.polys.rootoftools import generic_constants
    generic, back = generic_constants(q, radicals)
    if not back:
        return None
    try:
        result = roots(generic, t)
    except _FAILURES:
        return None
    if not isinstance(result, dict):
        return None
    found: dict[Expr, int] = {}
    for r, k in result.items():
        value = _replaced(r, back)
        found[value] = found.get(value, 0) + k
    return found


# PORTED FROM SYMPY: sympy/integrals/rationaltools.py:_roots_real_complex
def _roots_real_complex(q: Expr, t: Symbol) -> Optional[tuple[dict[Expr, int], dict[tuple[Expr, Expr], int]]]:
    """The roots of ``q`` separated into real ones and pairs ``(re, im)``
    of conjugate complex ones; ``None`` when not all are found."""
    from sympy_extras.core.sorting import ordered
    rs = _roots_of(q, t)
    if rs is None:
        return None
    reals: dict[Expr, int] = {}
    complexes: dict[tuple[Expr, Expr], int] = {}
    remaining = list(rs)
    while remaining:
        r = remaining.pop()
        r_c = r.conjugate()
        if r != r_c and r_c in rs:
            if rs[r_c] != rs[r]:
                raise ValueError('conjugate roots of different multiplicities')
            remaining.remove(r_c)
            r_re, r_im = r.as_real_imag()
            first = ordered([r_im, mul(Integer(-1), r_im)])[0]
            if not isinstance(first, Expr):
                raise TypeError('an expression is expected')
            complexes[(r_re, first)] = rs[r]
        else:
            # without its conjugate the root is taken as real
            reals[r] = rs[r]
    return reals, complexes


def log_to_real(h: ExprLike, q: ExprLike, x: Symbol, t: Symbol, complex_only: bool = False) -> Optional[Expr]:
    r"""Rioboo's conversion of a sum of logarithms to real functions: for
    ``h`` in ``K[t, x]`` and ``q`` in ``K[t]`` over a real field, a real
    function ``F`` with ``F' = (sum_{q(a) = 0} a log(h(a, x)))'``; ``None``
    when the roots of ``q`` are not all found, and, with
    ``complex_only``, when none of them comes in a complex conjugate
    pair.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Rational
    >>> from sympy_extras.integrals.risch.rationaltools import log_to_real
    >>> x, y = symbols('x y')
    >>> log_to_real(x + 3*y/2 + Rational(1, 2), 3*y**2 + 1, x, y)
    2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
    >>> log_to_real(x**2 - 1, -2*y + 1, x, y)
    log(x**2 - 1)/2
    """
    from sympy_extras.core.function import expand
    from sympy_extras.simplify.radsimp import collect_dict
    h = to_expr(h)
    rs = _roots_real_complex(to_expr(q), t)
    if rs is None:
        return None
    reals, complexes = rs
    if complex_only and not complexes:
        return None
    u = Dummy('u')
    v = Dummy('v')
    H = expand(_replaced(h, {t: add(u, mul(I, v))}))
    H_map = collect_dict(H, [I])
    a, b = H_map.get(_ONE, _ZERO), H_map.get(I, _ZERO)
    result: Expr = _ZERO
    for r_u, r_v in complexes:
        rule: dict[Basic, Basic] = {u: r_u, v: r_v}
        A_expr = _replaced(a, rule)
        B_expr = _replaced(b, rule)
        ac, bc, rational = _dense_pair(A_expr, B_expr, x)
        if rational:
            term = _complex_pair(_qq(ac), _qq(bc), QQ, x, r_u, r_v)
        else:
            term = _complex_pair(_ex(ac), _ex(bc), EX, x, r_u, r_v)
        result = add(result, term)
    for r in reals:
        result = add(result, mul(r, log(h.subs(t, r))))
    return result


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError('%s is not an expression' % (value,))
    return value


def _complex_pair(A: list[T], B: list[T], K: Domain[T], x: Symbol, r_u: Expr, r_v: Expr) -> Expr:
    """``r_u log(A**2 + B**2) + r_v log_to_atan(A, B)`` (the arctangent
    term is a constant when ``A`` or ``B`` is zero)."""
    from sympy_extras.polys.densearith import dup_add
    field = any(_has_symbolic_denominator(K.to_expr(c)) for c in A + B)
    AB = _sympy_expr([K.to_expr(c) for c in dup_add(dup_mul(A, A, K), dup_mul(B, B, K), K)], x, field)
    result = mul(r_u, log(AB))
    if A and B:
        result = add(result, mul(r_v, _log_to_atan(A, B, K, x)))
    return result

