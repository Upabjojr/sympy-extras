"""
Algorithms for solving Parametric Risch Differential Equations, on the typed
core.

The methods used for solving Parametric Risch Differential Equations parallel
those for solving Risch Differential Equations.  See the outline in the
docstring of rde.py for more information.

The Parametric Risch Differential Equation problem is, given f, g1, ..., gm in
K(t), to determine if there exist y in K(t) and c1, ..., cm in Const(K) such
that Dy + f*y == Sum(ci*gi, (i, 1, m)), and to find such y and ci if they exist.

For the algorithms here G is a list of tuples of factions of the terms on the
right hand side of the equation (i.e., gi in k(t)), and Q is a list of terms on
the right hand side of the equation (i.e., qi in k[t]).  See the docstring of
each function for more information.

Ported from SymPy's ``sympy.integrals.prde`` (in the version of Aaron
Meurer's pull requests, see the notice below), on the polynomials of
:mod:`sympy_extras.integrals.risch.tower` and the matrices of
:mod:`sympy_extras.integrals.risch.polymatrix`.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.integrals.risch.risch import DifferentialExtension
>>> from sympy_extras.integrals.risch.prde import limited_integrate
>>> from sympy_extras.integrals.risch.tower import TowerPoly
>>> x = symbols('x')
>>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x)])
>>> P = lambda e: TowerPoly.from_expr(e, x)
>>> limited_integrate(P(5*x**2), P(3), [(P(1), P(x))], DE)
((TowerPoly(5*x**3/9, x), TowerPoly(1, x)), [0])
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/prde.py (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the typed core, the polynomials of integrals.risch.tower and the
# FieldMatrix of polymatrix instead of SymPy's Poly and PolyMatrix; the
# linear systems over the field solved by Gauss-Jordan elimination.
# ---------------------------------------------------------------------
# Ported from SymPy (sympy.integrals.prde) and from Aaron Meurer's pull
# requests to SymPy: sympy/sympy#30180 and #30221 ("Risch: Implement most
# remaining exp-log cases from Bronstein", branch risch-rde-cancellation
# @d7dafa43) and #30292 ("Hypertangent cases in the Risch algorithm", branch
# risch-hypertangent @66bc72d6, the version taken here), by Claude Fable 5.1
# for sympy-extras; rewritten on the typed core by Claude Opus 5.5. The
# algorithm is Bronstein's (Symbolic Integration I: Transcendental
# Functions, 2nd edition, Springer 2005) as implemented by Aaron Meurer.
# SymPy's licence (BSD 3-clause, copyright the SymPy Development Team)
# applies to this file: see LICENSE-SymPy in this directory.
from __future__ import annotations

import itertools
from fractions import Fraction
from functools import reduce
from math import gcd as igcd, lcm as ilcm
from typing import Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import I, Integer, Rational
from sympy_extras.core.power import pow
from sympy_extras.core.symbol import Symbol
from .polymatrix import FieldMatrix
from .rde import (_rational_roots, order_at, order_at_oo, weak_normalizer, Degree, finite, is_oo,
    bound_degree, _special_denom_cancel_bound, _no_cancel_equal_applies)
from .risch import (gcdex_diophantine, frac_in, derivation, derivation_element, DifferentialExtension,
    residue_reduce, splitfactor, _residue_derivation_element, DecrementLevel, unique_dummy)
from .tower import RationalFunction, TowerPoly, UnsupportedConstant

__all__ = ['prde_normal_denom', 'real_imag', 'prde_special_denom', 'prde_linear_constraints',
           'poly_linear_constraints', 'constant_system', 'prde_spde', 'prde_no_cancel_b_large',
           'prde_no_cancel_b_small', 'prde_no_cancel_b_equal', 'prde_cancel_liouvillian', 'prde_cancel_tan',
           'param_poly_rischDE', 'param_rischDE', 'limited_integrate_reduce', 'limited_integrate',
           'is_deriv_in_field', 'parametric_log_deriv_heu', 'parametric_log_deriv_structure',
           'parametric_log_deriv', 'is_deriv_k', 'is_log_deriv_k_t_radical', 'is_deriv_k_atan',
           'is_log_deriv_k_t_radical_tan', 'is_log_deriv_k_t_radical_in_field']

zeros = FieldMatrix.zeros
eye = FieldMatrix.eye

#: a fraction of the tower (numerator, denominator)
Fraction2 = tuple[TowerPoly, TowerPoly]
#: the answer terms of the structure theorems: (term, coefficient)
Terms = list[tuple[Expr, Expr]]

_ZERO = RationalFunction.from_int(0)
_ONE = RationalFunction.from_int(1)


def _el(a: TowerPoly, d: TowerPoly) -> RationalFunction:
    return a.as_element()/d.as_element()


def _in_t(p: TowerPoly, DE: DifferentialExtension) -> TowerPoly:
    return p if p.gen == DE.t else p.as_poly_strict(DE.t)


def _poly(e: RationalFunction, DE: DifferentialExtension) -> TowerPoly:
    return TowerPoly.from_element_strict(e, DE.t)


def _number(q: Fraction) -> Expr:
    return Rational(q.numerator, q.denominator)


def _lcm_polys(polys: Sequence[TowerPoly], start: TowerPoly) -> TowerPoly:
    return reduce(lambda i, j: i.lcm(j), polys, start)


def prde_normal_denom(fa: TowerPoly, fd: TowerPoly, G: list[Fraction2], DE: DifferentialExtension
                      ) -> tuple[TowerPoly, Fraction2, list[Fraction2], TowerPoly]:
    """
    Parametric Risch Differential Equation - Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g1, ..., gm in k(t) with f weakly
    normalized with respect to t, return the tuple (a, b, G, h) such that
    a, h in k[t], b in k<t>, G = [g1, ..., gm] in k(t)^m, and for any solution
    c1, ..., cm in Const(k) and y in k(t) of Dy + f*y == Sum(ci*gi, (i, 1, m)),
    q == y*h in k<t> satisfies a*Dq + b*q == Sum(ci*Gi, (i, 1, m)).

    This is ``ParamRdeNormalDenominator`` from Section 7.1 of Bronstein's
    book.
    """
    fa, fd = _in_t(fa, DE), _in_t(fd, DE)
    dn, ds = splitfactor(fd, DE)
    Gds = [_in_t(gd, DE) for _, gd in G]
    gd = _lcm_polys(Gds, TowerPoly.one(DE.t))
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    G = [(c*_in_t(A, DE)).cancel(_in_t(D, DE), include=True) for A, D in G]

    return (a, (ba, bd), G, h)


def real_imag(ba: TowerPoly, bd: TowerPoly, gen: Symbol) -> tuple[TowerPoly, TowerPoly, TowerPoly]:
    """
    Helper function, to get the real and imaginary part of a rational function
    evaluated at sqrt(-1) without actually evaluating it at sqrt(-1).

    Explanation
    ===========

    Separates the even and odd power terms by checking the degree of terms wrt
    mod 4. Returns a tuple (ba[0], ba[1], bd) where ba[0] is real part
    of the numerator ba[1] is the imaginary part and bd is the denominator
    of the rational function.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.prde import real_imag
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> t = symbols('t')
    >>> real_imag(TowerPoly.from_expr(t**3 + 2*t + 5, t), TowerPoly.from_expr(t, t), t)
    (TowerPoly(1, t), TowerPoly(-5, t), TowerPoly(1, t))
    """
    def parts(p: TowerPoly) -> tuple[RationalFunction, RationalFunction]:
        real, imag = _ZERO, _ZERO
        for k, c in p.as_poly_strict(gen).terms():
            r = k % 4
            if r == 0:
                real = real + c
            elif r == 2:
                real = real - c
            elif r == 1:
                imag = imag + c
            else:
                imag = imag - c
        return real, imag
    bd_real, bd_imag = parts(bd)
    ba_real, ba_imag = parts(ba)
    real_part = TowerPoly.ground(ba_real*bd_real + ba_imag*bd_imag, gen)
    imag_part = TowerPoly.ground(ba_imag*bd_real - ba_real*bd_imag, gen)
    denominator = TowerPoly.ground(bd_real*bd_real + bd_imag*bd_imag, gen)
    return (real_part, imag_part, denominator)


def prde_special_denom(a: TowerPoly, ba: TowerPoly, bd: TowerPoly, G: list[Fraction2], DE: DifferentialExtension,
                       case: str = 'auto') -> tuple[TowerPoly, TowerPoly, list[Fraction2], TowerPoly]:
    """
    Parametric Risch Differential Equation - Special part of the denominator.

    Explanation
    ===========

    Case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the hyperexponential
    (resp. hypertangent) case, given a derivation D on k[t] and a in k[t],
    b in k<t>, and g1, ..., gm in k(t) with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the tuple (A, B, GG, h) such that A, B, h in
    k[t], GG = [gg1, ..., ggm] in k(t)^m, and for any solution c1, ..., cm in
    Const(k) and q in k<t> of a*Dq + b*q == Sum(ci*gi, (i, 1, m)), r == q*h in
    k[t] satisfies A*Dr + B*r == Sum(ci*ggi, (i, 1, m)).

    For case == 'primitive', k<t> == k[t], so it returns (a, b, G, 1) in this
    case.

    This is ``ParamRdeSpecialDenomExp`` and ``ParamRdeSpecialDenomTan``
    from Section 7.1 of Bronstein's book.
    """
    # The cancellation-case bound is shared with special_denom() in rde.py
    # via _special_denom_cancel_bound().  Note that N below is
    # max(0, -nb), following the parametric algorithms of Section 7.1,
    # while the non-parametric special_denom() uses max(0, -nb, n - nc)
    # per Section 6.2; this difference is intentional (it matches the
    # book), not a divergence.
    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = TowerPoly.monomial(DE.t, 1)
    elif case == 'tan':
        p = TowerPoly.from_list([1, 0, 1], DE.t)
    elif case in ('primitive', 'base'):
        B = ba.quo(bd)
        return (a, B, G, TowerPoly.one(DE.t))
    elif case in ('other_linear', 'other_nonlinear'):
        raise NotImplementedError("The %s case is not implemented in "
            "prde_special_denom()." % case)
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base', 'other_linear', 'other_nonlinear'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = min(order_at(Ga, p, DE.t) - order_at(Gd, p, DE.t) for Ga, Gd in G)
    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        n = _special_denom_cancel_bound(a, ba, bd, n, DE, case)

    N = max(0, -nb)
    pN = p**N
    pn = p**-n  # This is 1/h

    A = a*pN
    B = ba*pN.quo(bd) + (a*derivation(p, DE).quo(p)*pN).mul_ground(n)
    G = [(_in_t(Ga, DE)*pN*pn).cancel(_in_t(Gd, DE), include=True) for Ga, Gd in G]
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, g1*p**(N - n), ..., gm*p**(N - n), p**-n)
    return (A, B, G, h)


def prde_linear_constraints(a: TowerPoly, b: TowerPoly, G: list[Fraction2], DE: DifferentialExtension
                            ) -> tuple[tuple[TowerPoly, ...], FieldMatrix]:
    """
    Parametric Risch Differential Equation - Generate linear constraints on the constants.

    Explanation
    ===========

    Given a derivation D on k[t], a, b, in k[t] with gcd(a, b) == 1, and
    G = [g1, ..., gm] in k(t)^m, return Q = [q1, ..., qm] in k[t]^m and a
    matrix M with entries in k(t) such that for any solution c1, ..., cm in
    Const(k) and p in k[t] of a*Dp + b*p == Sum(ci*gi, (i, 1, m)),
    (c1, ..., cm) is a solution of Mx == 0, and p and the ci satisfy
    a*Dp + b*p == Sum(ci*qi, (i, 1, m)).

    This is ``LinearConstraints`` from Section 7.1 of Bronstein's book.
    """
    m = len(G)

    Gds = [_in_t(gd, DE) for _, gd in G]
    d = _lcm_polys(Gds[1:], Gds[0])
    Q = [(_in_t(ga, DE)*d.quo(_in_t(gd, DE))).div(d) for ga, gd in G]

    if not all(ri.is_zero for _, ri in Q):
        N = max(ri.degree(DE.t) for _, ri in Q)
        M = FieldMatrix.from_function(N + 1, m, lambda i, j: Q[j][1].nth(i))
    else:
        M = FieldMatrix(0, m, [])  # No constraints, return the empty matrix.

    qs = tuple(q for q, _ in Q)
    return (qs, M)


def poly_linear_constraints(p: Sequence[TowerPoly], d: TowerPoly) -> tuple[tuple[TowerPoly, ...], FieldMatrix]:
    """
    Given p = [p1, ..., pm] in k[t]^m and d in k[t], return
    q = [q1, ..., qm] in k[t]^m and a matrix M with entries in k such
    that Sum(ci*pi, (i, 1, m)), for c1, ..., cm in k, is divisible
    by d if and only if (c1, ..., cm) is a solution of Mx = 0, in
    which case the quotient is Sum(ci*qi, (i, 1, m)).

    There is no named counterpart in Bronstein's book; this is a subroutine
    used by the Section 7.1 algorithms.
    """
    m = len(p)
    divs = [pi.div(d) for pi in p]
    q = tuple(qi for qi, _ in divs)
    r = [ri for _, ri in divs]

    if not all(ri.is_zero for ri in r):
        n = max(ri.degree() for ri in r)
        M = FieldMatrix.from_function(n + 1, m, lambda i, j: r[j].nth(i))
    else:
        M = FieldMatrix(0, m, [])  # No constraints.

    return q, M


def constant_system(A: FieldMatrix, u: FieldMatrix, DE: DifferentialExtension) -> tuple[FieldMatrix, FieldMatrix]:
    """
    Generate a system for the constant solutions.

    Explanation
    ===========

    Given a differential field (K, D) with constant field C = Const(K), a Matrix
    A, and a vector (Matrix) u with coefficients in K, returns the tuple
    (B, v), where B is a Matrix with coefficients in C and v is a vector
    (Matrix) such that either v has coefficients in C, in which case the
    solutions in C of Ax == u are exactly all the solutions of Bx == v, or v
    has a non-constant coefficient, in which case Ax == u has no constant
    solution.  Note that B and v are a reduced *system*, not a solution:
    callers must check that v is constant and then solve Bx == v themselves.

    This algorithm is used both in solving parametric problems and in
    determining if an element a of K is a derivative of an element of K or the
    logarithmic derivative of a K-radical using the structure theorem approach.

    The entries are elements of the field, whose zero is recognized
    exactly (see :mod:`sympy_extras.integrals.risch.tower`).

    This is ``ConstantSystem`` from Section 7.1 of Bronstein's book.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
    >>> from sympy_extras.integrals.risch.prde import constant_system
    >>> from sympy_extras.integrals.risch.polymatrix import FieldMatrix
    >>> from sympy_extras.integrals.risch.tower import RationalFunction, TowerPoly
    >>> x = symbols('x')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x)])
    >>> X = RationalFunction.from_expr(x)
    >>> constant_system(FieldMatrix.from_rows([[X, 1]]), FieldMatrix.column([X]), DE)
    (FieldMatrix([[1, 0], [0, 1]]), FieldMatrix([[1], [0]]))

    so the constant solution of ``x*c1 + c2 == x`` is ``c1 == 1, c2 == 0``.
    """
    if not len(A):
        return A, u
    Au = A.row_join(u)
    Au, _ = Au.rref()
    # As the entries are canonical rational functions of the constants,
    # zero is recognized exactly (no cancel() limitation).

    A, u = Au[:, :-1], Au[:, -1]

    def D(a: RationalFunction) -> RationalFunction:
        return derivation_element(a, DE)

    # Repeat until no entry of A is nonconstant, processing rows appended
    # in earlier iterations as well ("while A is not constant do" in the
    # book's ConstantSystem; iterating over a snapshot of the original
    # rows would let nonconstant entries in the appended rows escape).
    # The safety cap guards against nontermination.
    for _ in range((Au.rows + 1)*(Au.cols + 1)*(len(DE.T) + 1) + 10):
        found: Optional[tuple[int, int]] = None
        for j, i in itertools.product(range(A.cols), range(A.rows)):
            if A[i, j].has(*DE.T):
                found = (i, j)
                break
        if found is None:
            break
        i, j = found
        # This assumes that const(F(t0, ..., tn)) == const(K) == F
        # TODO: If A[i, j] contains symbolic constants, D(A[i, j]) can
        # vanish for special values of them (e.g. an entry y*t is
        # nonconstant except at y == 0), and dividing by it makes the
        # reduced system valid only generically.  The same applies to
        # the pivots of the rref() calls, which also divide by
        # parameter-dependent entries.  The correct result would be a
        # Piecewise over the vanishing conditions of each pivot.
        Ri = A[i, :]
        # Rm+1; m = A.rows
        DAij = D(A[i, j])
        Rm1 = Ri.applyfunc(lambda e: D(e) / DAij)
        um1 = D(u[i, 0]) / DAij

        Aj = A[:, j]
        A = A - Aj * Rm1
        u = u - Aj * um1

        A = A.col_join(Rm1)
        u = u.col_join(FieldMatrix.column([um1]))
    else:
        raise RuntimeError("constant_system() did not reach a constant "
            "system.")

    return (A, u)


def prde_spde(a: TowerPoly, b: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
              ) -> tuple[TowerPoly, TowerPoly, list[TowerPoly], list[TowerPoly], Degree]:
    """
    Special Polynomial Differential Equation algorithm: Parametric Version.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n, and a, b, q1, ..., qm in k[t]
    with deg(a) > 0 and gcd(a, b) == 1, return (A, B, Q, R, n1), with
    Qq = [q1, ..., qm] and R = [r1, ..., rm], such that for any solution
    c1, ..., cm in Const(k) and q in k[t] of degree at most n of
    a*Dq + b*q == Sum(ci*gi, (i, 1, m)), p = (q - Sum(ci*ri, (i, 1, m)))/a has
    degree at most n1 and satisfies A*Dp + B*p == Sum(ci*qi, (i, 1, m))

    This is ``ParSPDE`` from Section 7.1 of Bronstein's book.
    """
    a, b = _in_t(a, DE), _in_t(b, DE)
    RZ = [gcdex_diophantine(b, a, _in_t(qi, DE)) for qi in Q]
    R = [r for r, _ in RZ]
    Z = [z for _, z in RZ]

    A = a
    B = b + derivation(a, DE)
    Qq = [zi - derivation(ri, DE) for ri, zi in zip(R, Z)]
    n1 = n - a.degree(DE.t)

    return (A, B, Qq, R, n1)


def _constraints(Q: Sequence[TowerPoly], DE: DifferentialExtension) -> FieldMatrix:
    """The constant system of ``Sum(ci*qi) == 0``."""
    m = len(Q)
    if all(qi.is_zero for qi in Q):
        dc = -1
    else:
        dc = max(qi.degree(DE.t) for qi in Q)
    M = FieldMatrix.from_function(dc + 1, m, lambda i, j: Q[j].nth(i))
    A, _ = constant_system(M, zeros(dc + 1, 1), DE)
    return A


def prde_no_cancel_b_large(b: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                           ) -> tuple[list[TowerPoly], FieldMatrix]:
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    b != 0 and either D == d/dt or deg(b) > max(0, deg(D) - 1), returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)), then q = Sum(dj*hj, (j, 1, r)), where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.

    This is ``ParamPolyRischDENoCancel1`` from Section 7.1 of Bronstein's
    book.
    """
    b = _in_t(b, DE)
    for i in range(len(Q)):
        Q[i] = _in_t(Q[i], DE)
    db = b.degree(DE.t)
    m = len(Q)
    H = [TowerPoly.zero(DE.t)]*m

    for N, i in itertools.product(range(finite(n), -1, -1), range(m)):  # [n, ..., 0]
        si = Q[i].nth(N + db)/b.LC()
        sitn = TowerPoly.monomial(DE.t, N, si)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    A = _constraints(Q, DE)
    c = eye(m)
    A = A.row_join(zeros(A.rows, m)).col_join(c.row_join(-c))

    return (H, A)


def prde_no_cancel_b_small(b: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                           ) -> tuple[list[TowerPoly], FieldMatrix]:
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    deg(b) < deg(D) - 1 and either D == d/dt or deg(D) >= 2, returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)) then q = Sum(dj*hj, (j, 1, r)) where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.

    This is ``ParamPolyRischDENoCancel2`` from Section 7.1 of Bronstein's
    book.
    """
    b = _in_t(b, DE)
    for i in range(len(Q)):
        Q[i] = _in_t(Q[i], DE)
    m = len(Q)
    H = [TowerPoly.zero(DE.t)]*m

    for N, i in itertools.product(range(finite(n), 0, -1), range(m)):  # [n, ..., 1]
        si = Q[i].nth(N + DE.d.degree(DE.t) - 1)/(DE.d.LC()*N)
        sitn = TowerPoly.monomial(DE.t, N, si)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if b.degree(DE.t) > 0:
        for i in range(m):
            si_p = TowerPoly.ground(Q[i].nth(b.degree(DE.t))/b.LC(), DE.t)
            H[i] = H[i] + si_p
            Q[i] = Q[i] - derivation(si_p, DE) - b*si_p
        A = _constraints(Q, DE)
        c = eye(m)
        A = A.row_join(zeros(A.rows, m)).col_join(c.row_join(-c))
        return (H, A)

    # else: b is in k, deg(qi) < deg(Dt)

    t = DE.t
    f_polys: list[TowerPoly]
    B: FieldMatrix
    if DE.case != 'base':
        with DecrementLevel(DE):
            t0 = DE.t  # k = k0(t0)
            ba, bd = frac_in(b, t0)
            Q0 = [frac_in(qi.TC(), t0) for qi in Q]
            f, B = param_rischDE(ba, bd, Q0, DE)

            # f = [f1, ..., fr] in k^r and B is a matrix with
            # m + r columns and entries in Const(k) = Const(k0)
            # such that Dy0 + b*y0 = Sum(ci*qi, (i, 1, m)) has
            # a solution y0 in k with c1, ..., cm in Const(k)
            # if and only y0 = Sum(dj*fj, (j, 1, r)) where
            # d1, ..., dr ar in Const(k) and
            # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0.

        # Transform fractions (fa, fd) in f into constant
        # polynomials fa/fd in k[t].
        f_polys = [TowerPoly.from_element_strict(_el(fa, fd), t) for fa, fd in f]
    else:
        # Base case. Dy == 0 for all y in k and b == 0.
        # Dy + b*y = Sum(ci*qi) is solvable if and only if
        # Sum(ci*qi) == 0 in which case the solutions are
        # y = d1*f1 for f1 = 1 and any d1 in Const(k) = k.

        f_polys = [TowerPoly.one(t)]  # r = 1
        B = FieldMatrix.row_vector([qi.TC() for qi in Q] + [_ZERO])
        # The condition for solvability is
        # B*Matrix([c1, ..., cm, d1]) == 0
        # There are no constraints on d1.

    # Coefficients of t^j (j > 0) in Sum(ci*qi) must be zero.
    d = max(qi.degree(DE.t) for qi in Q)
    if d > 0:
        M = FieldMatrix.from_function(d, m, lambda i, j: Q[j].nth(i + 1))
        A, _ = constant_system(M, zeros(d, 1), DE)
    else:
        # No constraints on the hj.
        A = FieldMatrix(0, m, [])

    # Solutions of the original equation are
    #    y = Sum(dj*fj, (j, 1, r) + Sum(ei*hi, (i, 1, m)),
    # where  ei == ci  (i = 1, ..., m),  when
    # A*Matrix([c1, ..., cm]) == 0 and
    # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0

    # Build combined constraint matrix with m + r + m columns.

    r = len(f_polys)
    Im = eye(m)
    A = A.row_join(zeros(A.rows, r + m))
    B = B.row_join(zeros(B.rows, m))
    C = Im.row_join(zeros(m, r)).row_join(-Im)

    return f_polys + H, A.col_join(B).col_join(C)


def prde_no_cancel_b_equal(b: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                           ) -> tuple[list[TowerPoly], FieldMatrix]:
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) == delta(t) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with delta(t) >= 2, n in ZZ, and
    b, q1, ..., qm in k[t] with deg(b) == delta(t) - 1, returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k)
    such that if c1, ..., cm in Const(k) and q in k[t] satisfy
    deg(q) <= n and Dq + b*q == Sum(ci*qi, (i, 1, m)), then
    q == Sum(dj*hj, (j, 1, r)), where d1, ..., dr in Const(k) and
    A*Matrix([c1, ..., cm, d1, ..., dr]).T == 0.

    This implements the "When delta(t) >= 2 and deg(b) == delta(t) - 1"
    discussion of Section 7.1 of Bronstein's book (neither edition gives
    it as named pseudocode).  Cancellation is only possible at degree
    -lc(b)/lc(Dt); if that ratio is a positive integer at most n, the
    remaining problem at that degree is delegated to the cancellation
    algorithms via param_poly_rischDE() (prde_cancel_tan() in the
    hypertangent case; other nonlinear cases are not implemented and
    raise NotImplementedError there).  In every other case — including
    a ratio above n, which no solution of degree at most n can attain —
    the descent through the degrees decides the equation completely.
    """
    b = _in_t(b, DE)
    m = len(Q)
    Q = [_in_t(q, DE) for q in Q]  # updated in place below; do not mutate the caller's list
    delta = DE.d.degree(DE.t)
    lam = DE.d.LC()

    # The degree at which cancellation could occur.
    # TODO: If b or Dt contains symbolic constants, Mc may be an integer
    # only for special values of them (e.g. b == -y*t gives Mc == y), and
    # the divisions by u below then produce a result that is valid only
    # generically: e.g. H == [-t**3/(y - 3) - 3*t/(y**2 - 4*y + 3)] is
    # undefined at y == 3 and y == 1.  The correct result would be a
    # Piecewise with one condition per loop iteration where
    # N*lam + lc(b) == 0 is solvable in the parameters, with the
    # cancellation algorithms handling those special values.
    Mc = (-b.LC()/lam).as_int()
    # A cancellation degree above the bound n is unattainable
    # (solutions have degree at most n), so treat it like a
    # noninteger ratio: no cancellation at any reachable degree.
    if Mc is not None and Mc > 0 and Mc <= n:
        M = Mc
    else:
        M = -1

    H = [TowerPoly.zero(DE.t)]*m

    for N in range(finite(n), M, -1):  # [n, ..., M + 1]
        u = lam*N + b.LC()  # nonzero, since N != -lc(b)/lam
        for i in range(m):
            si = Q[i].nth(N + delta - 1)/u
            sitn = TowerPoly.monomial(DE.t, N, si)
            H[i] = H[i] + sitn
            Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if M == -1:
        # The loop ran through N == 0, so as in prde_no_cancel_b_large()
        # any solution is q == Sum(ci*hi) and the (updated) right hand
        # sides must satisfy Sum(ci*qi) == 0.
        A = _constraints(Q, DE)
        c = eye(m)
        A = A.row_join(zeros(A.rows, m)).col_join(c.row_join(-c))
        return (H, A)

    # We reached the possible-cancellation degree M > 0.  Solve the
    # remaining problem, with the updated right hand sides and degree
    # bound M, by the cancellation algorithms.
    f, B = param_poly_rischDE(TowerPoly.one(DE.t), b, Q, M, DE)
    # Any solution is q == Sum(ci*hi) + Sum(dj*fj), where the fj and the
    # matrix B constrain (c1, ..., cm, d1, ..., dr).  Combine into a
    # relation matrix over the columns
    # (c1, ..., cm, d_h1, ..., d_hm, d_f1, ..., d_fr), where the first
    # block of rows enforces d_hi == ci.
    r = len(f)
    Bc = FieldMatrix.from_function(B.rows, m, lambda i, j: B[i, j])
    Bd = FieldMatrix.from_function(B.rows, r, lambda i, j: B[i, m + j])
    top = eye(m).row_join(-eye(m)).row_join(zeros(m, r))
    bottom = Bc.row_join(zeros(B.rows, m)).row_join(Bd)
    return (H + f, top.col_join(bottom))


def prde_cancel_liouvillian(b: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                            ) -> tuple[list[TowerPoly], FieldMatrix]:
    """
    Parametric Poly Risch Differential Equation - Cancellation: Liouvillian case.

    This implements the primitive and hyperexponential cancellation cases
    from the discussion in Section 7.1 of Bronstein's book (neither edition
    gives it as named pseudocode).
    """
    if DE.case not in ('primitive', 'exp'):
        raise ValueError("case must be 'primitive' or 'exp', not %s." %
            DE.case)

    b = _in_t(b, DE)
    Q = [_in_t(q, DE) for q in Q]
    H: list[TowerPoly] = []

    eta = _ZERO
    if DE.case == 'exp':
        # The coefficient of t**i satisfies the equation
        # D(q_i) + (b + i*eta)*q_i == rhs_i over k, where eta == Dt/t.
        # eta must be computed at the current level, before DecrementLevel
        # is entered below.
        eta = DE.d.quo(TowerPoly.monomial(DE.t, 1)).as_element()

    # Why use DecrementLevel? Below line answers that:
    # Assuming that we can solve such problems over 'k' (not k[t])
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)

    M: FieldMatrix = eye(0)
    for i in range(finite(n), -1, -1):
        if DE.case == 'exp':  # this re-checking can be avoided
            with DecrementLevel(DE):
                ba, bd = frac_in(b.as_element() + eta*i, DE.t)
        with DecrementLevel(DE):
            Qy = [frac_in(q.nth(i), DE.t) for q in Q]
            fi_fractions, Ai = param_rischDE(ba, bd, Qy, DE)
        fi = [TowerPoly.from_element_strict(_el(fa, fd), DE.t) for fa, fd in fi_fractions]

        ri = len(fi)

        if i == n:
            M = Ai
        else:
            M = Ai.col_join(M.row_join(zeros(M.rows, ri)))

        Fi: list[TowerPoly] = []
        hi: list[TowerPoly] = []

        # Substituting q == d*h + q_rest into Dq + b*q == Sum(ci*qi)
        # leaves the residual equation
        # D(q_rest) + b*q_rest == Sum(ci*qi) - d*(D(h) + b*h)
        # (see the equation following (7.17) in Section 7.1; the sign of
        # the b*h term is misprinted as a minus in the 1st edition).
        for j in range(ri):
            hji = fi[j] * TowerPoly.monomial(DE.t, i)
            hi.append(hji)
            # building up Sum(dji*(D(fji*t^i) + b*fji*t^i))
            Fi.append(-(derivation(hji, DE) + b*hji))

        H += hi
        # in the next loop instead of Q it has
        # to be Q + Fi taking its place
        Q = Q + Fi

    return (H, M)


def prde_cancel_tan(b0: TowerPoly, Q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                    ) -> tuple[list[TowerPoly], FieldMatrix]:
    """
    Parametric Poly Risch Differential Equation - Cancellation: Tangent case.

    Explanation
    ===========

    Given a derivation D on k[t], an integer ``n >= 0``, ``b0`` in k
    (a TowerPoly in DE.t of degree 0), and Q = [q1, ..., qm] in k[t]^m,
    with Dt/(t**2 + 1) == eta in k and sqrt(-1) not in k(t), return
    H = [h1, ..., hr] in k[t]^r and a matrix A with m + r columns and
    constant entries such that Dq + (b0 - n*eta*t)*q ==
    Sum(ci*qi, (i, 1, m)) has a solution q of degree at most n in
    k[t] with c1, ..., cm in Const(k) if and only if
    q == Sum(dj*hj, (j, 1, r)) with d1, ..., dr in Const(k) and
    (c1, ..., cm, d1, ..., dr) a solution of Ax == 0.

    As in cancel_tan(), ``n`` appears in the equation itself, so it
    must be a finite integer.

    This is the parametric analogue of cancel_tan()
    (``PolyRischDECancelTan``, Section 6.6 of Bronstein's book); the
    book gives no parametric counterpart, so this follows the pattern
    of the Section 7.1 cancellation discussion applied to the tangent
    projection: the remainders mod t**2 + 1 of any solution satisfy a
    parametric coupled differential system over k, and the quotients
    satisfy the same kind of equation with n - 2 in place of n, whose
    right hand side generators are the quotients by t**2 + 1 of the
    original generators and of the residuals of the coupled system's
    basis (their remainders cancel exactly on the constraint set, so
    dropping them is what makes the recursion's generators
    polynomial).
    """
    # Delayed import, since cde imports this module at the module
    # level.
    from .cde import param_coupled_DE_system

    m = len(Q)
    if is_oo(n):
        raise ValueError("n must be a finite integer in "
            "prde_cancel_tan().")
    b0 = _in_t(b0, DE)
    Q = [_in_t(q, DE) for q in Q]

    if n < 0:
        # Only the trivial zero solution is possible: constrain
        # Sum(ci*qi) == 0.
        if all(qi.is_zero for qi in Q):
            return ([], zeros(1, m))
        N = max(qi.degree(DE.t) for qi in Q)
        M = FieldMatrix.from_function(N + 1, m, lambda i, j: Q[j].nth(i))
        A, _ = constant_system(M, zeros(M.rows, 1), DE)
        return ([], A)

    if n == 0:
        # q is in k, so Dq + b0*q is in k: the coefficients of t**j
        # for j >= 1 of Sum(ci*qi) must vanish, and the constant
        # coefficients give a parametric Risch differential equation
        # over k.
        with DecrementLevel(DE):
            b0a, b0d = frac_in(b0.as_element(), DE.t)
            Qy = [frac_in(qi.nth(0), DE.t) for qi in Q]
            f, A0 = param_rischDE(b0a, b0d, Qy, DE)
        H = [TowerPoly.from_element_strict(_el(fa, fd), DE.t) for fa, fd in f]
        r = len(H)
        A = A0
        N = max(qi.degree(DE.t) for qi in Q)
        if N > 0:
            M = FieldMatrix.from_function(N, m, lambda i, j: Q[j].nth(i + 1))
            Ac, _ = constant_system(M, zeros(M.rows, 1), DE)
            A = A.col_join(Ac.row_join(zeros(Ac.rows, r)))
        return (H, A)

    eta = DE.d.exquo(TowerPoly.from_list([1, 0, 1], DE.t)).as_element()
    p = TowerPoly.from_list([1, 0, 1], DE.t)
    tf = RationalFunction.generator(DE.t)
    b = _poly(b0.as_element() - eta*tf*n, DE)

    # Project mod t - sqrt(-1): with c1i*t + c0i the remainder of qi
    # by p, the remainder u + v*t of any solution satisfies the
    # parametric coupled system Du + b0*u + n*eta*v == Sum(ci*c0i),
    # Dv - n*eta*u + b0*v == Sum(ci*c1i) over k.
    divs = [qi.div(p) for qi in Q]
    with DecrementLevel(DE):
        G = [(frac_in(rem.nth(0), DE.t), frac_in(rem.nth(1), DE.t))
            for _, rem in divs]
        R, An = param_coupled_DE_system(frac_in(b0.as_element(), DE.t),
            frac_in(-eta*n, DE.t), G, DE)
        Re = [(_el(ua, ud), _el(va, vd)) for (ua, ud), (va, vd) in R]
    rp = [_poly(ue + ve*tf, DE) for ue, ve in Re]

    # h == (q - Sum(ej*rj))/p satisfies the same kind of equation with
    # b0 unchanged (b + Dp/p == b0 - (n - 2)*eta*t) and bound n - 2,
    # with generators the quotients by p of the qi and of the
    # residuals -(D(rj) + b*rj); the remainders of the combined right
    # hand side cancel exactly by the An constraints.
    Qnew = [quo for quo, _ in divs] + \
        [-((derivation(rj, DE) + b*rj).div(p)[0]) for rj in rp]
    H2, M2 = prde_cancel_tan(b0, Qnew, n - 2, DE)
    H = rp + [p*h for h in H2]
    M = An.row_join(zeros(An.rows, len(H2))).col_join(M2)
    return (H, M)


def param_poly_rischDE(a: TowerPoly, b: TowerPoly, q: list[TowerPoly], n: Degree, DE: DifferentialExtension
                       ) -> tuple[list[TowerPoly], FieldMatrix]:
    """Polynomial solutions of a parametric Risch differential equation.

    Explanation
    ===========

    Given a derivation D in k[t], a, b in k[t] relatively prime, and q
    = [q1, ..., qm] in k[t]^m, return h = [h1, ..., hr] in k[t]^r and
    a matrix A with m + r columns and entries in Const(k) such that
    a*Dp + b*p = Sum(ci*qi, (i, 1, m)) has a solution p of degree <= n
    in k[t] with c1, ..., cm in Const(k) if and only if p = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    This implements the polynomial parametric RDE procedure of Section 7.1
    of Bronstein's book; there is no single corresponding pseudocode
    function.
    """
    a, b = _in_t(a, DE), _in_t(b, DE)
    q = [_in_t(qi, DE) for qi in q]
    m = len(q)
    if n < 0:
        # Only the trivial zero solution is possible.
        # Find relations between the qi.
        if all(qi.is_zero for qi in q):
            return [], zeros(1, m)  # No constraints.

        N = max(qi.degree(DE.t) for qi in q)
        M = FieldMatrix.from_function(N + 1, m, lambda i, j: q[j].nth(i))
        A, _ = constant_system(M, zeros(M.rows, 1), DE)

        return [], A

    if a.is_ground:
        # Normalization: a = 1.
        lc = a.LC()
        b, q = b.exquo_ground(lc), [qi.exquo_ground(lc) for qi in q]

        if not b.is_zero and (DE.case == 'base' or
                b.degree() > max(0, DE.d.degree() - 1)):
            return prde_no_cancel_b_large(b, q, n, DE)

        elif ((b.is_zero or b.degree() < DE.d.degree() - 1)
                and (DE.case == 'base' or DE.d.degree() >= 2)):
            return prde_no_cancel_b_small(b, q, n, DE)

        elif (DE.d.degree() >= 2 and
              b.degree() == DE.d.degree() - 1 and
              _no_cancel_equal_applies(b, n, DE)):
            return prde_no_cancel_b_equal(b, q, n, DE)

        else:
            # Cancellation cases
            if DE.case in ('primitive', 'exp'):
                return prde_cancel_liouvillian(b, q, n, DE)
            elif DE.case == 'tan':
                # The cancellation case has b == b0 - n*eta*t with
                # b0 in k, where n is the degree bound (Section 6.6);
                # the pipeline reaches this branch in that shape via
                # prde_no_cancel_b_equal()'s hand-off.
                eta = DE.d.exquo(TowerPoly.from_list([1, 0, 1], DE.t)).as_element()
                if not is_oo(n) and b.degree(DE.t) <= 1 and \
                        (b.nth(1) + eta*n).is_zero:
                    return prde_cancel_tan(TowerPoly.ground(b.nth(0), DE.t), q, n, DE)
                raise NotImplementedError("The tangent cancellation case "
                    "requires b == b0 - n*eta*t with b0 in k in "
                    "param_poly_rischDE().")
            else:
                raise NotImplementedError("nonlinear cases have not yet "
                    "been implemented")

    # else: deg(a) > 0

    # Iterate SPDE as long as possible cumulating coefficient
    # and terms for the recovery of original solutions.
    alpha, beta = TowerPoly.one(DE.t), [TowerPoly.zero(DE.t)]*m
    d = TowerPoly.one(DE.t)
    while n >= 0:  # and a, b relatively prime
        a, b, q, r, n = prde_spde(a, b, q, n, DE)
        beta = [betai + alpha*ri for betai, ri in zip(beta, r)]
        alpha *= a
        # Solutions p of a*Dp + b*p = Sum(ci*qi) correspond to
        # solutions alpha*p + Sum(ci*betai) of the initial equation.
        d = a.gcd(b)
        if not d.is_ground:
            break

    # a*Dp + b*p = Sum(ci*qi) may have a polynomial solution
    # only if the sum is divisible by d.

    qq, M = poly_linear_constraints(q, d)
    # qq = [qq1, ..., qqm] where qqi = qi.quo(d).
    # M is a matrix with m columns an entries in k.
    # Sum(fi*qi, (i, 1, m)), where f1, ..., fm are elements of k, is
    # divisible by d if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the quotient is Sum(fi*qqi).

    A, _ = constant_system(M, zeros(M.rows, 1), DE)
    # A is a matrix with m columns and entries in Const(k).
    # Sum(ci*qqi) is Sum(ci*qi).quo(d), and the remainder is zero
    # for c1, ..., cm in Const(k) if and only if
    # A*Matrix([c1, ...,cm]) == 0.

    V = A.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*qi) is divisible by d with exact quotient Sum(aji*qqi).
    # Sum(ci*qi) is divisible by d if and only if ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case, solutions of
    #     a*Dp + b*p = Sum(ci*qi) = Sum(dj*Sum(aji*qi))
    # are the same as those of
    #     (a/d)*Dp + (b/d)*p = Sum(dj*rj)
    # where rj = Sum(aji*qqi).

    if not V:  # No non-trivial solution.
        return [], eye(m)  # Could return A, but this has
                           # the minimum number of rows.

    Mqq = FieldMatrix.row_vector([qi.as_element() for qi in qq])  # A single row.
    r = [_poly((Mqq*vj).element(0), DE) for vj in V]  # [r1, ..., ru]

    # Solutions of (a/d)*Dp + (b/d)*p = Sum(dj*rj) correspond to
    # solutions alpha*p + Sum(Sum(dj*aji)*betai) of the initial
    # equation. These are equal to alpha*p + Sum(dj*fj) where
    # fj = Sum(aji*betai).
    Mbeta = FieldMatrix.row_vector([bi.as_element() for bi in beta])
    f = [_poly((Mbeta*vj).element(0), DE) for vj in V]  # [f1, ..., fu]

    #
    # Solve the reduced equation recursively.
    #
    g, B = param_poly_rischDE(a.quo(d), b.quo(d), r, n, DE)

    # g = [g1, ..., gv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # (a/d)*Dp + (b/d)*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*gk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation are then
    # Sum(dj*fj, (j, 1, u)) + alpha*Sum(ek*gk, (k, 1, v)).

    # Collect solution components.
    h = f + [alpha*gk for gk in g]

    # Build combined relation matrix.
    A = -eye(m)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(g)))
    A = A.col_join(zeros(B.rows, m).row_join(B))

    return h, A


def _prde_normalized_solve(A: TowerPoly, B: TowerPoly, G: list[Fraction2], gamma: TowerPoly,
                           DE: DifferentialExtension, n: Optional[Degree] = None
                           ) -> tuple[list[Fraction2], FieldMatrix]:
    """
    Common tail of param_rischDE() and limited_integrate().

    Explanation
    ===========

    Given A, B in k[t] and G = [G1, ..., Gm] in k(t)^m with the equation
    A*Dp + B*p == Sum(ci*Gi, (i, 1, m)) already reduced (weakly
    normalized, denominators cleared), return h = [h1, ..., hr] in
    k(t)^r and a matrix C with entries in Const(k) such that solutions
    of the original problem are y == Sum(dj*hj, (j, 1, r))/1 with
    (c1, ..., cm, d1, ..., dr) in the nullspace of C, where the
    returned hj have already been divided by gamma.  If ``n`` is None,
    the degree bound is computed with bound_degree(); otherwise ``n``
    is used as a proven degree bound, avoiding bound_degree() entirely.
    """
    m = len(G)
    A, B, gamma = _in_t(A, DE), _in_t(B, DE), _in_t(gamma, DE)
    g = A.gcd(B)
    a, b = A.quo(g), B.quo(g)
    gs = [_in_t(gia, DE).cancel(_in_t(gid, DE)*g, include=True) for gia, gid in G]

    # a*Dp + b*p = Sum(ci*gi)  may have a polynomial solution
    # only if the sum is in k[t].

    q, M = prde_linear_constraints(a, b, gs, DE)

    # q = [q1, ..., qm] where qi in k[t] is the polynomial component
    # of the partial fraction expansion of gi.
    # M is a matrix with m columns and entries in k.
    # Sum(fi*gi, (i, 1, m)), where f1, ..., fm are elements of k,
    # is a polynomial if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the sum is equal to Sum(fi*qi).

    M, _ = constant_system(M, zeros(M.rows, 1), DE)
    # M is a matrix with m columns and entries in Const(k).
    # Sum(ci*gi) is in k[t] for c1, ..., cm in Const(k)
    # if and only if M*Matrix([c1, ..., cm]) == 0,
    # in which case the sum is Sum(ci*qi).

    ## Reduce number of constants at this point

    V = M.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*gi) is in k[t] and equal to Sum(aji*qi) (j = 1, ..., u).
    # Sum(ci*gi) is in k[t] if and only is ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case,
    #     Sum(ci*gi) = Sum(ci*qi) = Sum(dj*Sum(aji*qi)) = Sum(dj*rj)
    # where rj = Sum(aji*qi) (j = 1, ..., u) in k[t].

    if not V:  # No non-trivial solution
        return [], eye(m)

    Mq = FieldMatrix.row_vector([qi.as_element() for qi in q])  # A single row.
    r = [_poly((Mq*vj).element(0), DE) for vj in V]  # [r1, ..., ru]

    # Solutions of a*Dp + b*p = Sum(dj*rj) correspond to solutions
    # y = p/gamma of the initial equation with ci = Sum(dj*aji).

    if n is None:
        # bound_degree() raises NotImplementedError when it cannot decide
        # one of its structure-theorem queries, and we let that propagate.
        # Guessing a finite bound here instead (n = 5 from 2017 to 2026)
        # is unsound: solutions of degree above the guess are silently
        # lost, so a caller like is_deriv_in_field() can be tricked into
        # a false proof that an elementary integral is nonelementary.
        # Falling back to n == oo (as rischDE() does) is not an option
        # either, since the parametric no-cancellation and cancellation
        # algorithms iterate over range(n, ...), which requires a finite
        # bound.
        n = bound_degree(a, b, r, DE, parametric=True)

    h, relation_B = param_poly_rischDE(a, b, r, n, DE)

    # h = [h1, ..., hv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # a*Dp + b*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*hk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation for ci = Sum(dj*aji)
    # (i = 1, ..., m) are then y = Sum(ek*hk, (k, 1, v))/gamma.

    ## Build combined relation matrix with m + u + v columns.

    relation = -eye(m)
    for vj in V:
        relation = relation.row_join(vj)
    relation = relation.row_join(zeros(m, len(h)))
    relation = relation.col_join(zeros(relation_B.rows, m).row_join(relation_B))

    ## Eliminate d1, ..., du.

    W = relation.nullspace()

    # W = [w1, ..., wt] where each wl is a column matrix with
    # entries blk (k = 1, ..., m + u + v) in Const(k).
    # The vectors (bl1, ..., blm) generate the space of those
    # constant families (c1, ..., cm) for which a solution of
    # the equation Dy + f*y == Sum(ci*Gi) exists. They generate
    # the space and form a basis except possibly when Dy + f*y == 0
    # is solvable in k(t}. The corresponding solutions are
    # y = Sum(blk'*hk, (k, 1, v))/gamma, where k' = k + m + u.

    v = len(h)
    items: list[RationalFunction] = []
    for wl in W:
        # excise the dj's
        entries = wl.flat()
        items.extend(entries[:m] + (entries[len(entries) - v:] if v else []))

    M = FieldMatrix(len(W), m + v, items)
    N = M.nullspace()

    # N = [n1, ..., ns] where the ni in Const(k)^(m + v) are column
    # vectors generating the space of linear relations between
    # c1, ..., cm, e1, ..., ev.

    C = FieldMatrix.from_rows([ni.flat() for ni in N])  # rows n1, ..., ns.

    return [hk.cancel(gamma, include=True) for hk in h], C


def param_rischDE(fa: TowerPoly, fd: TowerPoly, G: list[Fraction2], DE: DifferentialExtension
                  ) -> tuple[list[Fraction2], FieldMatrix]:
    """
    Solve a Parametric Risch Differential Equation: Dy + f*y == Sum(ci*Gi, (i, 1, m)).

    Explanation
    ===========

    Given a derivation D in k(t), f in k(t), and G
    = [G1, ..., Gm] in k(t)^m, return h = [h1, ..., hr] in k(t)^r and
    a matrix A with m + r columns and entries in Const(k) such that
    Dy + f*y = Sum(ci*Gi, (i, 1, m)) has a solution y
    in k(t) with c1, ..., cm in Const(k) if and only if y = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    Elements of k(t) are tuples (a, d) with a and d in k[t].

    This chains together the parametric algorithms of Section 7.1 of
    Bronstein's book; the book does not give it as a single named
    pseudocode function.
    """
    fa, fd = _in_t(fa, DE), _in_t(fd, DE)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    # Solutions of the weakly normalized equation Dz + f*z = q*Sum(ci*Gi)
    # correspond to solutions y = z/q of the original equation.
    gamma = q
    G = [(q*_in_t(ga, DE)).cancel(_in_t(gd, DE), include=True) for ga, gd in G]

    a, (ba, bd), G, hn = prde_normal_denom(fa, fd, G, DE)
    # Solutions q in k<t> of  a*Dq + b*q = Sum(ci*Gi) correspond
    # to solutions z = q/hn of the weakly normalized equation.
    gamma *= hn

    A, B, G, hs = prde_special_denom(a, ba, bd, G, DE)
    # Solutions p in k[t] of  A*Dp + B*p = Sum(ci*Gi) correspond
    # to solutions q = p/hs of the previous equation.
    gamma *= hs

    return _prde_normalized_solve(A, B, G, gamma, DE)


def limited_integrate_reduce(fa: TowerPoly, fd: TowerPoly, G: list[Fraction2], DE: DifferentialExtension
                             ) -> tuple[TowerPoly, TowerPoly, TowerPoly, Degree, Fraction2, list[Fraction2]]:
    """
    Simpler version of step 1 & 2 for the limited integration problem.

    Explanation
    ===========

    Given a derivation D on k(t) and f, g1, ..., gn in k(t), return
    (a, b, h, N, g, V) such that a, b, h in k[t], N is a non-negative integer,
    g in k(t), V == [v1, ..., vm] in k(t)^m, and for any solution v in k(t),
    c1, ..., cm in C of f == Dv + Sum(ci*wi, (i, 1, m)), p = v*h is in k<t>, and
    p and the ci satisfy a*Dp + b*p == g + Sum(ci*vi, (i, 1, m)).  Furthermore,
    if S1irr == Sirr, then p is in k[t], and if t is nonlinear or Liouvillian
    over k, then deg(p) <= N.

    So that the special part is always computed, this function calls the more
    general prde_special_denom() automatically if it cannot determine that
    S1irr == Sirr.  Furthermore, it will automatically call bound_degree() when
    t is linear and non-Liouvillian, which for the transcendental case, implies
    that Dt == a*t + b with for some a, b in k*.

    This is ``LimitedIntegrateReduce`` from Section 7.2 of Bronstein's
    book.
    """
    fa, fd = _in_t(fa, DE), _in_t(fd, DE)
    G = [(_in_t(ga, DE), _in_t(gd, DE)) for ga, gd in G]
    dn, ds = splitfactor(fd, DE)
    E = [splitfactor(gd, DE) for _, gd in G]
    En = [e for e, _ in E]
    Es = [s for _, s in E]
    c = _lcm_polys(En, dn)  # lcm(dn, en1, ..., enm)
    hn = c.gcd(c.diff(DE.t))
    a = hn
    b = -derivation(hn, DE)
    N = 0

    # These are the cases where we know that S1irr = Sirr, but there could be
    # others, and this algorithm will need to be extended to handle them.
    if DE.case in ('base', 'primitive', 'exp', 'tan'):
        hs = _lcm_polys(Es, ds)  # lcm(ds, es1, ..., esm)
        a = hn*hs
        b -= (hn*derivation(hs, DE)).quo(hs)
        mu = min([order_at_oo(fa, fd, DE.t)] + [order_at_oo(ga, gd, DE.t) for
            ga, gd in G])
        # So far, all the above are also nonlinear or Liouvillian, but if this
        # changes, then this will need to be updated to call bound_degree()
        # as per the docstring of this function (DE.case == 'other_linear').
        N = hn.degree(DE.t) + hs.degree(DE.t) + max(0, 1 - DE.d.degree(DE.t) - mu)
    else:
        # TODO: implement this
        raise NotImplementedError

    V = [(-a*hn*ga).cancel(gd, include=True) for ga, gd in G]
    # Note: the first component is hn, not a.  Both editions of the book
    # return a here, but that does not satisfy the stated contract when
    # hs != 1: from v == p/a, the original equation multiplied by a**2
    # gives a*Dp - Da*p == a**2*f - Sum(ci*a**2*wi), and dividing through
    # by hs (which divides Da exactly, since hs is special) gives
    # hn*Dp + b*p == a*hn*f - Sum(ci*a*hn*wi) with the b and right hand
    # sides constructed above.
    return (hn, b, a, N, (a*hn*fa).cancel(fd, include=True), V)


def limited_integrate(fa: TowerPoly, fd: TowerPoly, G: list[Fraction2], DE: DifferentialExtension
                      ) -> Optional[tuple[Fraction2, list[RationalFunction]]]:
    """
    Solves the limited integration problem:  f = Dv + Sum(ci*wi, (i, 1, n))

    This implements the limited integration procedure of Section 7.2 of
    Bronstein's book (there is no single corresponding pseudocode
    function).
    """
    fa, fd = _in_t(fa, DE), _in_t(fd, DE)
    fa, fd = fa.mul_ground(fd.LC().invert()), fd.monic()
    # Reduction to a polynomial problem (LimitedIntegrateReduce, Section
    # 7.2): for any solution v in k(t), c1, ..., cm in Const(k) of
    # f == Dv + Sum(ci*wi), p == v*h is in k[t] and satisfies
    # A*Dp + b*p == g + Sum(ci*vi) with deg(p) <= N.
    A, b, h, N, g, V = limited_integrate_reduce(fa, fd, G, DE)
    # Solve the parametric problem with right hand sides [g] + V.
    # Solutions of the original problem correspond to parametric
    # solutions whose g-coefficient is 1.  N is a proven degree bound,
    # so bound_degree() is not needed.
    Q = [g] + V
    hs, C = _prde_normalized_solve(A, b, Q, h, DE, n=N)
    W = [w for w in C.nullspace() if w.element(0)]
    if not W:
        return None
    # we can take any vector from W; take W[0], scaled so that the
    # g-coefficient is 1
    # TODO: If W[0][0] contains symbolic constants, it can vanish for
    # special values of them, making the result valid only generically;
    # a different nullspace vector may serve those special values, and
    # the correct result would be a Piecewise over the corresponding
    # conditions.
    w = (W[0]/W[0].element(0)).flat()
    r = len(hs)
    m = len(w) - r - 1
    constants = w[1: m + 1]
    y = _ZERO
    for i in range(r):
        y = y + w[m + 1 + i]*_el(hs[i][0], hs[i][1])
    Ya, Yd = frac_in(y, DE.t)
    lc = Yd.LC().invert()
    return (Ya.mul_ground(lc), Yd.mul_ground(lc)), constants


def is_deriv_in_field(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension) -> Optional[Fraction2]:
    """
    Checks if f can be written as the derivative of an element of k(t).

    Explanation
    ===========

    f in k(t) is the derivative of an element of k(t) if there exists v in
    k(t) such that f == Dv.  Either returns (va, vd), with va, vd in k[t]
    such that f == D(va/vd), or None, which means that f is not the
    derivative of an element of k(t).  Note that v is determined only up
    to an additive constant.

    This is the in-field integration problem from Section 5.12 of
    Bronstein's book (the "Recognizing Derivatives" subsection; neither
    edition gives it as named pseudocode).  It is solved here as the
    special case of the limited integration problem (Section 7.2) with an
    empty list of special elements.

    See also
    ========
    limited_integrate, is_log_deriv_k_t_radical_in_field
    """
    A = limited_integrate(fa, fd, [], DE)
    if A is None:
        return None
    (va, vd), _ = A
    return (va, vd)


def _solve_c1(ps: Sequence[RationalFunction], qs: Sequence[RationalFunction]) -> Optional[RationalFunction]:
    """The solution ``c1`` of the linear equations ``ps[i] - c1*qs[i] ==
    0`` when it exists and is unique (SymPy's ``solve(eqs, c1)``)."""
    c: Optional[RationalFunction] = None
    for p, q in zip(ps, qs):
        if q:
            c = p/q
            break
    if c is None:
        return None
    for p, q in zip(ps, qs):
        if not (p - c*q).is_zero:
            return None
    return c


def _rational_solution(c: Optional[RationalFunction]) -> Optional[tuple[int, int]]:
    """The numerator and the denominator of a rational ``c1``."""
    if c is None:
        return None
    q = c.as_fraction()
    if q is None:
        return None
    return q.numerator, q.denominator


def parametric_log_deriv_heu(fa: TowerPoly, fd: TowerPoly, wa: TowerPoly, wd: TowerPoly, DE: DifferentialExtension,
                             c1: Optional[Symbol] = None) -> Optional[tuple[int, int, RationalFunction]]:
    """
    Parametric logarithmic derivative heuristic.

    Explanation
    ===========

    Given a derivation D on k[t], f in k(t), and a hyperexponential monomial
    theta over k(t), raises either NotImplementedError, in which case the
    heuristic failed, or returns None, in which case it has proven that no
    solution exists, or returns a solution (n, m, v) of the equation
    n*f == Dv/v + m*Dtheta/theta, with v in k(t)* and n, m in ZZ with n != 0.

    If this heuristic fails, the structure theorem approach will need to be
    used.

    The argument w == Dtheta/theta

    This is ``ParametricLogarithmicDerivative`` from Section 7.3 of
    Bronstein's book.
    """
    # Note: ideally the code here should let n = 1 whenever possible, as the
    # check in special_denom() only succeeds in that case.
    fa, fd, wa, wd = _in_t(fa, DE), _in_t(fd, DE), _in_t(wa, DE), _in_t(wd, DE)

    # Special case when f and w are rational numbers (not in the book, but
    # this fails this heuristic and comes up often enough to add here). In
    # this case, we can set z = 0.
    f = _el(fa, fd).as_fraction()
    w = _el(wa, wd).as_fraction()
    if f is not None and w is not None:
        if f == 0 and w == 0:
            # Any n works: n*0 == Dv/v + m*0 with v == 1, m == 0.
            return (1, 0, _ONE)
        # solve n*x = m*y in integers, i.e., n=y, m=x (after dividing through
        # by gcd(x, y))
        x = f.numerator*w.denominator
        y = w.numerator*f.denominator
        g = igcd(x, y)
        x //= g
        y //= g
        n = y
        m = x
        if n == 0:
            return None
        if n < 0:
            n = -n
            m = -m
        return (n, m, _ONE)

    p, a = fa.div(fd)
    q, b = wa.div(wd)

    B = max(0, DE.d.degree(DE.t) - 1)
    C = max(p.degree(DE.t), q.degree(DE.t))

    if q.degree(DE.t) > B:
        s = _rational_solution(_solve_c1([p.nth(i) for i in range(B + 1, C + 1)],
                                         [q.nth(i) for i in range(B + 1, C + 1)]))
        if s is None:
            # deg(q) > B, no solution for c.
            return None

        M, N = s

        nfmwa = (fa*wd).mul_ground(N) - (wa*fd).mul_ground(M)
        nfmwd = fd*wd
        Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE, 'auto')
        if Qv is None:
            # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
            return None

        Q, v = Qv

        if Q == 0 or v.is_zero:
            return None

        return (Q*N, Q*M, v)

    if p.degree(DE.t) > B:
        return None

    # SymPy multiplies l by lcm(lc(fd), lc(wd)), a constant of k, which
    # scales r1 and r2 below alike and leaves c1 unchanged.
    l = fd.monic().lcm(wd.monic())
    ln, ls = splitfactor(l, DE)
    z = ls*ln.gcd(ln.diff(DE.t))

    if z.degree(DE.t) < 1:
        # The residue equations below would be vacuous, so the heuristic
        # cannot determine the candidate for m/n from them ("failed" in
        # Bronstein's ParametricLogarithmicDerivative).  Solutions may well
        # exist, e.g. f == -1 + 1/(x + 1), w == 1 has the solution
        # (1, -1, x + 1), so returning None here (proven no solution) would
        # be wrong.
        if DE.case == 'base':
            # At the base level, Dv/v is a proper fraction for any
            # v in k(x)*, so n*f == Dv/v + m*w requires the polynomial
            # parts to cancel exactly: n*p == m*q.  This determines
            # c1 == m/n == p/q (a constant), and the rest is decided by
            # is_log_deriv_k_t_radical_in_field(), so this case is
            # complete.
            if q.is_zero:
                if not p.is_zero:
                    # n*p == 0 forces n == 0, which is not allowed.
                    return None
                # p == q == 0: m/n is not determined by the polynomial
                # parts; this needs the structure theorems.
                raise NotImplementedError("parametric_log_deriv_heu() "
                    "heuristic failed: the full parametric logarithmic "
                    "derivative problem (using the structure theorems) is "
                    "not yet implemented.")
            if p.is_zero:
                Mi, Ni = 0, 1
            else:
                if p.mul_ground(q.LC()) != q.mul_ground(p.LC()):
                    # p is not a constant multiple of q, so n*p == m*q has
                    # no solution with n != 0.
                    return None
                cc = (p.LC()/q.LC()).as_fraction()
                if cc is None:
                    # If cc contains symbolic constants, it may be
                    # rational only for special values of them (a
                    # condition like "y in QQ" that is not expressible
                    # as a Piecewise relational); None here is correct
                    # generically and at worst yields a false
                    # nonelementary proof at those values, never a
                    # wrong antiderivative, so no Piecewise handling is
                    # needed.
                    return None
                Mi, Ni = cc.numerator, cc.denominator

            nfmwa = (fa*wd).mul_ground(Ni) - (wa*fd).mul_ground(Mi)
            nfmwd = fd*wd
            Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
            if Qv is None:
                # (N*f - M*w) is not the logarithmic derivative of a
                # k(t)-radical.
                return None

            Q, v = Qv

            if Q == 0 or v.is_zero:
                return None

            return (Q*Ni, Q*Mi, v)

        raise NotImplementedError("parametric_log_deriv_heu() heuristic "
            "failed: the full parametric logarithmic derivative problem "
            "(using the structure theorems) is not yet implemented.")

    u1, r1 = (fa*l.quo(fd)).div(z)  # (l*f).div(z)
    u2, r2 = (wa*l.quo(wd)).div(z)  # (l*w).div(z)

    s = _rational_solution(_solve_c1([r1.nth(i) for i in range(z.degree(DE.t))],
                                     [r2.nth(i) for i in range(z.degree(DE.t))]))
    if s is None:
        # deg(q) <= B, no solution for c.
        return None

    M, N = s

    nfmwa = (fa*wd).mul_ground(N) - (wa*fd).mul_ground(M)
    nfmwd = fd*wd
    Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
    if Qv is None:
        # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
        return None

    Q, v = Qv

    if Q == 0 or v.is_zero:
        return None

    return (Q*N, Q*M, v)


def _term_element(e: Optional[Expr]) -> RationalFunction:
    """A term of the structure theorems (a generator or the argument of a
    generator) as an element."""
    if e is None:
        raise NotImplementedError("The structure theorems need the "
            "arguments of the monomials of the tower (DE.extargs).")
    return RationalFunction.from_expr(e)


def _term_expr(e: Optional[Expr]) -> Expr:
    if e is None:
        raise NotImplementedError("The structure theorems need the "
            "arguments of the monomials of the tower (DE.extargs).")
    return e


def _solve_linear(A: FieldMatrix, u: FieldMatrix) -> Optional[list[RationalFunction]]:
    """A solution of ``A x == u`` with the free unknowns zero (SymPy's
    ``gauss_jordan_solve`` with its parameters zero), ``None`` when the
    system is inconsistent."""
    R, pivots = A.row_join(u).rref()
    n = A.cols
    if n in pivots:
        return None
    xs = [_ZERO]*n
    for k, c in enumerate(pivots):
        xs[c] = R[k, n]
    return xs


def parametric_log_deriv_structure(fa: TowerPoly, fd: TowerPoly, wa: TowerPoly, wd: TowerPoly,
                                   DE: DifferentialExtension) -> Optional[tuple[int, int, RationalFunction]]:
    """
    Parametric logarithmic derivative problem via the structure theorems.

    Explanation
    ===========

    Tries to solve n*f == Dv/v + m*w for n, m in ZZ, n != 0, and v in
    k(t)*, by solving the structure equation

        f == (m/n)*w + Sum(ri*Dti/ti, i in E) + Sum(ri*Dti, i in L)

    for rational (m/n, r1, ..., rk), where E and L are the
    hyperexponential and logarithmic monomials of the current tower (up
    to the current level), as in equation (7.44) of Section 7.3 of
    Bronstein's book, with theta adjoined as an extra hyperexponential
    generator.  A solution yields (n, m, v) with
    v == Product(termi**(n*ri)) directly.

    Either returns (n, m, v), or None, which means that f - (m/n)*w is
    not a QQ-linear combination of the logarithmic derivatives available
    from the current tower for any m/n in QQ.  Note that None does NOT
    prove that no solution exists: a solution whose v requires monomials
    not in the tower (e.g. new logarithms) is not found by this method.
    Callers must treat None as inconclusive.
    """
    # Use only the part of the tower visible at the current level: the
    # callers pose this problem inside DecrementLevel, and v must lie in
    # the current field (w itself is usually the log derivative of the
    # monomial one level up, which must not appear in v).  As in
    # is_log_deriv_k_t_radical(), hypertangent and arc-tangent monomials
    # do not contribute (Corollary 9.3.2 (ii)).
    E_indices, L_indices, _, _ = _structure_tower(DE, fa, fd, wa, wd)
    E_part = [DE.D[i].quo(TowerPoly.monomial(DE.T[i], 1)).as_element() for i in E_indices]
    L_part = [DE.D[i].as_element() for i in L_indices]

    w = _el(wa, wd)
    f = _el(fa, fd)

    lhs = FieldMatrix.row_vector([w] + E_part + L_part)
    rhs = FieldMatrix.column([f])

    A, u = constant_system(lhs, rhs, DE)

    # A*x == u is now a linear system with constant coefficients for the
    # rational unknowns x == (m/n, r1, ..., rk).  Note that u is the
    # reduced right hand side, not a solution: the system may be
    # underdetermined (e.g. when w is a combination of the generators),
    # so solve it explicitly, setting any free parameters to zero.
    if not (all(i.is_rational for i in A) and all(i.is_rational for i in u)):
        if not all(derivation_element(i, DE).is_zero for i in A) \
                or not all(derivation_element(i, DE).is_zero for i in u):
            # The system could not be reduced to one over the constants
            return None
        raise NotImplementedError("Cannot work with non-rational "
            "coefficients in this case.")
    found = _solve_linear(A, u)
    if found is None:
        # Inconsistent system: f - (m/n)*w is not a combination of the
        # available logarithmic derivatives for any m/n.
        # With symbolic constants, inconsistency is decided only
        # generically; at special values of the constants a solution may
        # exist.  The wrapper then raises NotImplementedError, which is
        # sound (an honest error, never a wrong answer), so no Piecewise
        # handling is needed here.
        return None
    fractions: list[Fraction] = []
    for xi in found:
        qi = xi.as_fraction()
        if qi is None:
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        fractions.append(qi)

    n = reduce(ilcm, [i.denominator for i in fractions], 1)
    xs = [int(i*n) for i in fractions]  # now integers: [m, n*r1, ..., n*rk]
    m = xs[0]
    terms = [RationalFunction.generator(DE.T[i]) for i in E_indices] + \
        [_term_element(DE.extargs[i - 1]) for i in L_indices]
    v = _ONE
    for term, k in zip(terms, xs[1:]):
        v = v*term**k
    if n == 0:
        return None
    # Defensive check of n*f == Dv/v + m*w (Dv/v is the same QQ-linear
    # combination of the generator logarithmic derivatives by construction)
    lhs_check = _ZERO
    for part, k in zip(E_part + L_part, xs[1:]):
        lhs_check = lhs_check + part*k
    if not (f*n - w*m - lhs_check).is_zero:
        return None
    if n < 0:
        n, m, v = -n, -m, v.invert()
    return (n, m, v)


def parametric_log_deriv(fa: TowerPoly, fd: TowerPoly, wa: TowerPoly, wd: TowerPoly, DE: DifferentialExtension
                         ) -> Optional[tuple[int, int, RationalFunction]]:
    """
    Solves the parametric logarithmic derivative problem.

    This is the parametric logarithmic derivative problem from Section 7.3
    of Bronstein's book: n*f == Dv/v + m*w for n, m in ZZ, n != 0, and
    v in k(t)*.  The heuristic (``ParametricLogarithmicDerivative``) is
    tried first; if it cannot decide, the structure theorem method of the
    same section is tried.
    """
    try:
        A = parametric_log_deriv_heu(fa, fd, wa, wd, DE)
    except NotImplementedError:
        A = parametric_log_deriv_structure(fa, fd, wa, wd, DE)
        if A is None:
            # The structure method proves nonexistence only when the
            # elementary extension containing Integral(f) needs no
            # monomials outside the current tower, which we cannot check
            # here, so an inconclusive result must not be reported as
            # "no solution".
            raise NotImplementedError("parametric_log_deriv() could not "
                "decide: the heuristic failed and f - (m/n)*w is not a "
                "combination of logarithmic derivatives from the current "
                "tower for any m/n in QQ.")
    return A


def _structure_system_solve(lhs: FieldMatrix, rhs: FieldMatrix, DE: DifferentialExtension
                            ) -> Optional[list[RationalFunction]]:
    """
    Find a constant solution x of the structure system lhs*x == rhs.

    Explanation
    ===========

    lhs and rhs are matrices over K.  Returns a list of constant
    solutions (free parameters set to zero), or None if there is no
    constant solution.  constant_system() returns a reduced *system*,
    not a solution, and treating its right hand side as a solution (as
    this module once did) is only accidentally correct when the
    reduction is identity-like, so the reduced system is solved
    explicitly here and the solution is verified against the original
    equation.

    Raises NotImplementedError if the verification fails.
    """
    A, u = constant_system(lhs, rhs, DE)
    if not len(A):
        return None
    if not all(derivation_element(i, DE).is_zero for i in u):
        # No constant solution
        return None
    xs = _solve_linear(A, u)
    if xs is None:
        # With symbolic constants in the system, inconsistency (and
        # the pivoting above) is decided only generically; at special
        # values of the constants a solution may exist.  Returning None
        # here errs on the incomplete-but-sound side (a spurious
        # "not a derivative"/"not a radical" answer, at worst a false
        # nonelementary proof, never a wrong antiderivative), so no
        # Piecewise handling is needed here.
        return None
    # TODO: Zeroing the free parameters can miss a rational solution when
    # a pivot divided by a symbolic constant: e.g. y*x1 + x2 == 1 gives
    # xs == [(1 - tau)/y, tau], which is non-rational at tau == 0 but
    # rational at tau == 1.  Choosing a rational point of the affine
    # solution space (when one exists) would decide more towers; failing
    # that only causes an honest NotImplementedError downstream.
    # Defensive verification against the original system
    for i in range(lhs.rows):
        total = _ZERO
        for j in range(lhs.cols):
            total = total + lhs[i, j]*xs[j]
        if not (total - rhs[i, 0]).is_zero:
            raise NotImplementedError("The candidate solution of the "
                "structure system failed verification.")
    return xs


def _tower_has_I(DE: DifferentialExtension, *polys: TowerPoly) -> bool:
    """
    Checks if sqrt(-1) appears explicitly in polys or in the derivations
    of the tower at or below the current level.
    """
    top = len(DE.T) + DE.level
    return (any(p.has(I) for p in polys) or
        any(d.has(I) for d in DE.D[:top + 1]))


def _structure_tower(DE: DifferentialExtension, *polys: TowerPoly,
                     real: bool = False) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    The index sets E, L, T and A of the tower at the current level.

    Explanation
    ===========

    Returns the lists of indices into DE.T of the hyperexponential (E),
    logarithmic (L), hypertangent (T) and arc-tangent (A) monomials at
    or below the current level, as labeled by DE.exts; these are the
    index sets (9.6), (9.7), (9.10) and (9.11) of Section 9.3 of
    Bronstein's book.

    The structure theorems need every monomial of the tower labeled, and
    the real version (Theorem 9.3.2, which is needed as soon as T or A
    is nonempty, or always when real=True) needs sqrt(-1) not in the
    field, so the tower and the given polynomials (the inputs of the
    caller) are checked for explicit I.  Raises NotImplementedError
    otherwise.
    """
    if len(DE.exts) != len(DE.D) - 1:
        raise NotImplementedError("The structure theorems need every "
            "monomial of the tower labeled as 'exp', 'log', 'tan' or 'atan' "
            "in DE.exts; nonelementary extensions are not supported.")
    top = len(DE.T) + DE.level
    E = [i for i in DE.indices('exp') if i <= top]
    L = [i for i in DE.indices('log') if i <= top]
    T = [i for i in DE.indices('tan') if i <= top]
    A = [i for i in DE.indices('atan') if i <= top]
    if (real or T or A) and _tower_has_I(DE, *polys):
        raise NotImplementedError("The real version of the structure "
            "theorems (Section 9.3 of Bronstein's book), needed for towers "
            "with hypertangent or arc-tangent monomials, requires sqrt(-1) "
            "not in the field.")
    return E, L, T, A


def _has_symbols(e: RationalFunction) -> bool:
    """Whether the element has a symbol (a parameter) in it."""
    return any(g.free_symbols for g in e.gens)


def _structure_solve(parts: Sequence[RationalFunction], rhs: RationalFunction, DE: DifferentialExtension
                     ) -> Optional[list[Fraction]]:
    """
    Solves the structure equation Sum(ri*parts[i]) == rhs for ri in QQ.

    Explanation
    ===========

    Returns the list of rational coefficients ri, or None, which means
    that rhs is not a constant combination of parts.  rhs == 0 is
    trivially the empty combination.

    Raises NotImplementedError if the constant solution is not rational
    (e.g. a coefficient like log(2)/log(3)): deciding whether it could
    be made rational would need a QQ-basis of the constant field (see
    the discussion following Corollary 9.3.1 of Bronstein's book).
    """
    if rhs.is_zero:
        return [Fraction(0)]*len(parts)
    u = _structure_system_solve(FieldMatrix.row_vector(list(parts)), FieldMatrix.column([rhs]), DE)
    if u is None:
        return None
    fractions = [i.as_fraction() for i in u]
    result: list[Fraction] = []
    for i, q in zip(u, fractions):
        if q is None:
            # A coefficient with a symbol in it (the v of exp(v*log(x)), a
            # parameter of the integrand) is a constant transcendental over QQ
            # for a generic value: the equation has no rational solution, and
            # rhs is not a rational combination of parts (exp(v*log(x)) is a
            # new monomial). A numeric coefficient such as log(2)/log(3) may
            # or may not be rational, and deciding it would need a QQ-basis of
            # the constant field (the discussion after Corollary 9.3.1).
            if all(j.is_rational or _has_symbols(j) for j in u):
                return None
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        result.append(q)
    return result


def _df_over_f(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension) -> RationalFunction:
    dfa = fd*derivation(fa, DE) - fa*derivation(fd, DE)
    return dfa.as_element()/(fd*fa).as_element()


def _df(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension) -> RationalFunction:
    dfa = fd*derivation(fa, DE) - fa*derivation(fd, DE)
    return dfa.as_element()/(fd**2).as_element()


def _exprs(u: Sequence[Fraction]) -> list[Expr]:
    return [_number(i) for i in u]


def _powers_of_factors(e: Expr, j: Fraction) -> Expr:
    """``e**j`` as the powers of the square-free factors of the numerator
    and the denominator of ``e`` (so that ``sqrt(x**2)`` is ``x``)."""
    from sympy_extras.polys.polytools import sqf_list
    jj = _number(j)
    n, d = e.as_numer_denom()
    parts: list[Expr] = []
    for f, sign in ((n, 1), (d, -1)):
        coeff, terms = sqf_list(f)
        parts.append(pow(coeff, mul(Integer(sign), jj)))
        for b, k in terms:
            parts.append(pow(b, mul(Integer(sign), k, jj)))
    return mul(*parts)


def _cancel_expr(e: Expr) -> Expr:
    try:
        return RationalFunction.from_expr(e).as_expr()
    except UnsupportedConstant:
        from sympy_extras.polys.polytools import cancel
        return cancel(e)


def is_deriv_k(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension) -> Optional[tuple[Terms, Expr, Expr]]:
    r"""
    Checks if Df/f is the derivative of an element of k(t).

    Explanation
    ===========

    a in k(t) is the derivative of an element of k(t) if there exists b in k(t)
    such that a = Db.  Either returns (ans, u), such that Df/f == Du, or None,
    which means that Df/f is not the derivative of an element of k(t).  ans is
    a list of tuples such that Add(*[i*j for i, j in ans]) == u.  This is useful
    for seeing exactly which elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df/f is the derivative of a element of K if and only if there are ri
    in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i      Df
            /     i     i     /     i   ---   =  --.
            ---               ---        t        f
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    log(f) will be the same as u up to a additive constant.  This is because
    they will both behave the same as monomials. For example, both log(x) and
    log(2*x) == log(x) + log(2) satisfy Dt == 1/x, because log(2) is constant.
    Therefore, the term const is returned.  const is such that
    log(const) + f == u.  This is calculated by dividing the arguments of one
    logarithm from the other.

    Towers containing hypertangent ('tan') and arc-tangent ('atan')
    monomials are handled by the real version of the structure theorem
    (Theorem 9.3.2 and Corollary 9.3.2 (i)): those monomials never
    contribute to the sum above, but sqrt(-1) must not be in K, so
    NotImplementedError is raised if I appears in the tower or in f.

    This is an application of the Risch structure theorems from Section 9.3
    of Bronstein's book (neither edition gives it as named pseudocode).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
    >>> from sympy_extras.integrals.risch.prde import is_deriv_k
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t1, t2 = symbols('x t1 t2')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(1/x, t1),
    ...     TowerPoly.from_expr(1/(x + 1), t2)], exts=['log', 'log'], extargs=[x, x + 1])
    >>> is_deriv_k(TowerPoly.from_expr(2*x**2 + 2*x, t2), TowerPoly.from_expr(1, t2), DE)
    ([(t1, 1), (t2, 1)], t1 + t2, 2)

    See also
    ========
    is_log_deriv_k_t_radical_in_field, is_log_deriv_k_t_radical
    """
    # Compute Df/f
    df = _df_over_f(fa, fd, DE)

    # Our assumption here is that each monomial is recursively transcendental
    E, L, _, _ = _structure_tower(DE, fa, fd)
    E_part = [DE.D[i].quo(TowerPoly.monomial(DE.T[i], 1)).as_element() for i in E]
    L_part = [DE.D[i].as_element() for i in L]

    u = _structure_solve(E_part + L_part, df, DE)
    if u is None:
        # No constant solution
        return None

    terms: list[Expr] = [_term_expr(DE.extargs[i - 1]) for i in E] + [DE.T[i] for i in L]
    ans = list(zip(terms, _exprs(u)))
    result = add(*[mul(i, j) for i, j in ans])
    argterms: list[Expr] = [DE.T[i] for i in E] + [_term_expr(DE.extargs[i - 1]) for i in L]
    l: list[Expr] = []
    for i, j in zip(argterms, u):
        # We need to get around things like sqrt(x**2) != x
        # and also sqrt(x**2 + 2*x + 1) != x + 1
        # Issue 10798: i need not be a polynomial
        l.append(_powers_of_factors(i, j))
    f = _el(fa, fd).as_expr()
    const = _cancel_expr(mul(f, pow(mul(*l), Integer(-1))))

    return (ans, result, const)


def is_log_deriv_k_t_radical(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension,
                             Df: bool = True) -> Optional[tuple[Terms, Expr, int, Expr]]:
    r"""
    Checks if Df is the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    b in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*b == Du/u.
    Either returns (ans, u, n, const) or None, which means that Df cannot be
    written as the logarithmic derivative of a k(t)-radical.  ans is a list of
    tuples such that Mul(*[i**j for i, j in ans]) == u.  This is useful for
    seeing exactly what elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df is the logarithmic derivative of a K-radical if and only if there
    are ri in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i
            /     i     i     /     i   ---   =  Df.
            ---               ---        t
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).

    exp(f) will be the same as u up to a multiplicative constant.  This is
    because they will both behave the same as monomials.  For example, both
    exp(x) and exp(x + 1) == E*exp(x) satisfy Dt == t. Therefore, the term const
    is returned.  const is such that exp(const)*f == u.  This is calculated by
    subtracting the arguments of one exponential from the other.

    Towers containing hypertangent ('tan') and arc-tangent ('atan')
    monomials are handled by the real version of the structure theorem
    (Theorem 9.3.2 and Corollary 9.3.2 (ii)): those monomials never
    contribute to the sum above, but sqrt(-1) must not be in K, so
    NotImplementedError is raised if I appears in the tower or in f.

    This is an application of the structure theorems from Sections 9.3 and
    9.4 of Bronstein's book (neither edition gives it as named pseudocode).

    See also
    ========

    is_log_deriv_k_t_radical_in_field, is_deriv_k
    """
    df = _df(fa, fd, DE) if Df else _el(fa, fd)

    # Our assumption here is that each monomial is recursively transcendental
    E, L, _, _ = _structure_tower(DE, fa, fd)
    E_part = [DE.D[i].quo(TowerPoly.monomial(DE.T[i], 1)).as_element() for i in E]
    L_part = [DE.D[i].as_element() for i in L]

    u = _structure_solve(E_part + L_part, df, DE)
    if u is None:
        # No constant solution
        return None

    n = reduce(ilcm, [i.denominator for i in u], 1)
    un = [i*n for i in u]
    terms: list[Expr] = [DE.T[i] for i in E] + [_term_expr(DE.extargs[i - 1]) for i in L]
    ans = list(zip(terms, _exprs(un)))
    result = mul(*[pow(i, j) for i, j in ans])

    # exp(f) will be the same as result up to a multiplicative
    # constant.  We now find the log of that constant.
    argterms = [_term_element(DE.extargs[i - 1]) for i in E] + \
        [RationalFunction.generator(DE.T[i]) for i in L]
    const = _el(fa, fd)
    for i, j in zip(argterms, u):
        const = const - i*RationalFunction.from_fraction(j)

    return (ans, result, n, const.as_expr())


def is_deriv_k_atan(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension) -> Optional[tuple[Terms, Expr]]:
    r"""
    Checks if Df/(f**2 + 1) is the derivative of an element of k(t).

    Explanation
    ===========

    Df/(f**2 + 1) == D(atan(f)), so this decides whether atan(f) is a
    new monomial over k(t): if not, atan(f) == u + c for some u in k(t)
    and some c in Const(k(t)).  Either returns (ans, u), such that
    Df/(f**2 + 1) == Du, or None, which means that Df/(f**2 + 1) is
    not the derivative of an element of k(t).  ans is a list of tuples
    such that Add(*[i*j for i, j in ans]) == u.

    This function uses the real version of the structure theorem
    (Corollary 9.3.2 (iii) of Bronstein's book), which says that for any
    f in K, Df/(f**2 + 1) is the derivative of an element of K if and
    only if there are ri in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i        Df
            /     i     i     /     i   ------  =  ------,
            ---               ---        2          2
         i in A            i in T       t  + 1     f  + 1
               K/C(x)            K/C(x)  i

    where A_K/C(x) is the set of indices of the arc-tangent monomials
    of K over C(x) (Dt_i == Da_i/(a_i**2 + 1) for some a_i in
    C(x)(t_1, ..., t_i-1), labeled 'atan' in DE.exts) and T_K/C(x) is
    the set of indices of the hypertangent monomials (Dt_i/(t_i**2 + 1)
    == Da_i, labeled 'tan'), as defined in (9.10) and (9.11) of
    Section 9.3.  Logarithmic and hyperexponential monomials never
    contribute.  The theorem requires K to be a real elementary
    extension of C(x) with sqrt(-1) not in K, so NotImplementedError is
    raised if I appears in the tower or in f.

    The constant c == atan(f) - u is not computed here: it is locally
    constant but in general only piecewise constant as a function (e.g.
    atan(2*x/(1 - x**2)) - 2*atan(x) jumps at x == 1), so callers must
    treat it as an opaque constant, exactly as for the branch constant
    of a logarithm in is_deriv_k().

    Neither edition of Bronstein's book gives this as named pseudocode;
    it is equation (4) of Bronstein, "Simplification of Real Elementary
    Functions" (ISSAC 1989).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
    >>> from sympy_extras.integrals.risch.prde import is_deriv_k_atan
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x)], exts=[], extargs=[])
    >>> is_deriv_k_atan(TowerPoly.from_expr(x, x), TowerPoly.from_expr(1, x), DE) is None
    True

    so atan(x) is a monomial over QQ(x).  Over QQ(x, atan(x)),
    atan(2*x/(1 - x**2)) is 2*atan(x) up to a constant:

    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x),
    ...     TowerPoly.from_expr(1/(x**2 + 1), t)], exts=['atan'], extargs=[x])
    >>> is_deriv_k_atan(TowerPoly.from_expr(2*x, t), TowerPoly.from_expr(1 - x**2, t), DE)
    ([(t, 2)], 2*t)

    See also
    ========

    is_deriv_k, is_log_deriv_k_t_radical_tan
    """
    # Compute Df/(f**2 + 1)
    dfa = fd*derivation(fa, DE) - fa*derivation(fd, DE)
    df = dfa.as_element()/(fa**2 + fd**2).as_element()

    _, _, T, A = _structure_tower(DE, fa, fd, real=True)
    T_part = [DE.D[i].quo(TowerPoly.from_list([1, 0, 1], DE.T[i])).as_element() for i in T]
    A_part = [DE.D[i].as_element() for i in A]

    u = _structure_solve(T_part + A_part, df, DE)
    if u is None:
        # No constant solution
        return None

    terms: list[Expr] = [_term_expr(DE.extargs[i - 1]) for i in T] + [DE.T[i] for i in A]
    ans = list(zip(terms, _exprs(u)))
    result = add(*[mul(i, j) for i, j in ans])

    return (ans, result)


def is_log_deriv_k_t_radical_tan(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension
                                 ) -> Optional[tuple[Terms, Expr, int, Expr]]:
    r"""
    Checks if sqrt(-1)*Df is the logarithmic derivative of a
    k(t)(sqrt(-1))-radical.

    Explanation
    ===========

    By Theorem 5.10.1 of Bronstein's book, this decides whether tan(f)
    is a new monomial over k(t) (with sqrt(-1) not in k(t)): if not,
    then n*f == u + n*c for some integer n > 0, u in k(t) and c in
    Const(k(t)), so tan(f) == tan(u/n + c) is algebraic over k(t)(tan(c))
    (it is in k(t)(tan(c)) itself when n == 1, by the addition formula
    for the tangent).  Either returns (ans, u, n, const) such that
    f == u/n + const, or None, which means that sqrt(-1)*Df is not the
    logarithmic derivative of a k(t)(sqrt(-1))-radical.  ans is a list
    of tuples such that Add(*[i*j for i, j in ans]) == u.

    This function uses the real version of the structure theorem
    (Corollary 9.3.2 (iv) of Bronstein's book), which says that for any
    f in K, sqrt(-1)*Df is the logarithmic derivative of a
    K(sqrt(-1))-radical if and only if there are ri in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i
            /     i     i     /     i   ------  =  Df,
            ---               ---        2
         i in A            i in T       t  + 1
               K/C(x)            K/C(x)  i

    where A_K/C(x) and T_K/C(x) are the sets of indices of the
    arc-tangent and hypertangent monomials of K over C(x), as in
    is_deriv_k_atan().  Logarithmic and hyperexponential monomials
    never contribute.  The theorem requires K to be a real elementary
    extension of C(x) with sqrt(-1) not in K, so NotImplementedError is
    raised if I appears in the tower or in f.

    Neither edition of Bronstein's book gives this as named pseudocode;
    it is equation (3) of Bronstein, "Simplification of Real Elementary
    Functions" (ISSAC 1989).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.risch import DifferentialExtension
    >>> from sympy_extras.integrals.risch.prde import is_log_deriv_k_t_radical_tan
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x)], exts=[], extargs=[])
    >>> is_log_deriv_k_t_radical_tan(TowerPoly.from_expr(x, x), TowerPoly.from_expr(1, x), DE) is None
    True

    so tan(x) is a monomial over QQ(x).  Over QQ(x, atan(x)),
    tan(atan(x)/3) is algebraic of degree 3 (Example 1 of the paper):

    >>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x),
    ...     TowerPoly.from_expr(1/(x**2 + 1), t)], exts=['atan'], extargs=[x])
    >>> is_log_deriv_k_t_radical_tan(TowerPoly.from_expr(t, t), TowerPoly.from_expr(3, t), DE)
    ([(t, 1)], t, 3, 0)

    See also
    ========

    is_log_deriv_k_t_radical, is_deriv_k_atan
    """
    df = _df(fa, fd, DE)

    _, _, T, A = _structure_tower(DE, fa, fd, real=True)
    T_part = [DE.D[i].quo(TowerPoly.from_list([1, 0, 1], DE.T[i])).as_element() for i in T]
    A_part = [DE.D[i].as_element() for i in A]

    u = _structure_solve(T_part + A_part, df, DE)
    if u is None:
        # No constant solution
        return None

    n = reduce(ilcm, [i.denominator for i in u], 1)
    un = [i*n for i in u]
    terms: list[Expr] = [_term_expr(DE.extargs[i - 1]) for i in T] + [DE.T[i] for i in A]
    ans = list(zip(terms, _exprs(un)))
    result = add(*[mul(i, j) for i, j in ans])
    elements = [_term_element(DE.extargs[i - 1]) for i in T] + [RationalFunction.generator(DE.T[i]) for i in A]
    result_el = _ZERO
    for e, j in zip(elements, un):
        result_el = result_el + e*RationalFunction.from_fraction(j)
    const = _el(fa, fd) - result_el/n

    return (ans, result, n, const.as_expr())


def is_log_deriv_k_t_radical_in_field(fa: TowerPoly, fd: TowerPoly, DE: DifferentialExtension, case: str = 'auto',
                                      z: Optional[Symbol] = None) -> Optional[tuple[int, RationalFunction]]:
    """
    Checks if f can be written as the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    It differs from is_log_deriv_k_t_radical(fa, fd, DE, Df=False)
    for any given fa, fd, DE in that it finds the solution in the
    given field not in some (possibly unspecified extension) and
    "in_field" with the function name is used to indicate that.

    f in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*f == Du/u.
    Either returns (n, u) or None, which means that f cannot be written as the
    logarithmic derivative of a k(t)-radical.

    case is one of {'primitive', 'exp', 'tan', 'auto'} for the primitive,
    hyperexponential, and hypertangent cases, respectively.  If case is 'auto',
    it will attempt to determine the type of the derivation automatically.

    This is the in-field logarithmic derivative of a k(t)-radical problem
    from Section 5.12 of Bronstein's book (the "Recognizing Logarithmic
    Derivatives of k(t)-radicals" subsection; neither edition gives it as
    named pseudocode).

    See also
    ========
    is_log_deriv_k_t_radical, is_deriv_k
    """
    fa, fd = _in_t(fa, DE), _in_t(fd, DE)
    fa, fd = fa.cancel(fd, include=True)

    z = z or unique_dummy('z')
    H, b = residue_reduce(fa, fd, DE, z=z)
    if not b:
        # I will have to verify, but I believe that the answer should be
        # None in this case. This should never happen for the
        # functions given when solving the parametric logarithmic
        # derivative problem when integration elementary functions (see
        # Bronstein's book, page 255), so most likely this indicates a bug.
        return None

    roots = [(i, _rational_roots(i)) for i, _ in H]
    if not all(len(j) == i.degree() for i, j in roots):
        # If f is the logarithmic derivative of a k(t)-radical, then all the
        # roots of the resultant must be rational numbers.
        return None

    # [(a, i), ...], where i*log(a) is a term in the log-part of the integral
    # of f
    residues = [j for _, j in roots]
    # Note: this might be empty, but everything below should work find in that
    # case (it should be the same as if it were [[1, 1]])
    residueterms: list[tuple[TowerPoly, Fraction]] = []
    for j in range(len(H)):
        for i in residues[j]:
            q = i.as_fraction()
            if q is None:
                return None
            residueterms.append((H[j][1].subs(z, i), q))

    pe = _el(fa, fd) - _residue_derivation_element(H, DE, z)

    found_p = TowerPoly.from_element(pe, DE.t)
    if found_p is None:
        # f - Dg will be in k[t] if f is the logarithmic derivative of a k(t)-radical
        return None
    p = found_p

    if p.degree(DE.t) >= max(1, DE.d.degree(DE.t)):
        return None

    if case == 'auto':
        case = DE.case

    n: int
    u: RationalFunction
    if case == 'exp':
        wa, wd = DE.d.cancel(TowerPoly.monomial(DE.t, 1), include=True)
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t, cancel=True)
            wa, wd = frac_in((wa, wd), DE.t)
            found = parametric_log_deriv(pa, pd, wa, wd, DE)
        if found is None:
            return None
        n, e, u_found = found
        u = u_found*RationalFunction.generator(DE.t)**e

    elif case == 'primitive':
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t)
            radical = is_log_deriv_k_t_radical_in_field(pa, pd, DE, case='auto')
        if radical is None:
            return None
        n, u = radical

    elif case == 'base':
        # TODO: we can use more efficient residue reduction from ratint()
        if fd.gcd(fd.diff()).degree() > 0 or fa.degree() >= fd.degree():
            # f is the logarithmic derivative in the base case if and only if
            # f = fa/fd, fd is square-free, deg(fa) < deg(fd), and
            # gcd(fa, fd) == 1.  The last condition is handled by cancel() above.
            return None
        # Note: if residueterms = [], returns (1, 1)
        # f had better be 0 in that case.
        n = reduce(ilcm, [i.denominator for _, i in residueterms], 1)
        u = _ONE
        for poly, mult in residueterms:
            u = u*poly.as_element()**int(mult*n)
        return (n, u)

    elif case == 'tan':
        if _tower_has_I(DE, fa, fd):
            raise NotImplementedError("The hypertangent case of "
                "is_log_deriv_k_t_radical_in_field() requires sqrt(-1) not "
                "in k(t).")
        # p == a + b*t with a, b in k, and u == v*(t**2 + 1)**e with v in
        # k* and e in ZZ (the special polynomial is t**2 + 1), so
        # n*p == Du/u is equivalent to n*a == Dv/v and n*b/(2*eta) == e,
        # where eta == Dt/(t**2 + 1) is in k.
        eta = DE.d.quo(TowerPoly.from_list([1, 0, 1], DE.t)).as_element()
        a, bb = p.nth(0), p.nth(1)
        ratio = (bb/(eta*2)).as_fraction()
        if ratio is None:
            return None
        na: int
        v: RationalFunction
        if a.is_zero:
            na, v = 1, _ONE
        else:
            with DecrementLevel(DE):
                aa, ad = frac_in(a, DE.t)
                radical = is_log_deriv_k_t_radical_in_field(aa, ad, DE, case='auto')
            if radical is None:
                return None
            na, v = radical
        n = ilcm(na, ratio.denominator)
        tf = RationalFunction.generator(DE.t)
        u = v**(n//na)*(tf**2 + 1)**int(n*ratio)

    elif case in ('other_linear', 'other_nonlinear'):
        # XXX: If these are supported by the structure theorems, change to NotImplementedError.
        raise ValueError("The %s case is not supported in this function." % case)

    else:
        raise ValueError("case must be one of {'primitive', 'exp', 'tan', "
        "'base', 'auto'}, not %s" % case)

    common_denom = reduce(ilcm, [j.denominator for _, j in residueterms] + [n], 1)
    scaled = [(i, j*common_denom) for i, j in residueterms]
    m = common_denom//n
    if common_denom != n*m:  # Verify exact division
        raise ValueError("Inexact division")
    result = u**m
    for poly, mult in scaled:
        result = result*poly.as_element()**int(mult)

    return (common_denom, result)
