"""
Algorithms for solving the Risch differential equation, on the typed core.

Given a differential field K of characteristic 0 that is a simple
monomial extension of a base field k and f, g in K, the Risch
Differential Equation problem is to decide if there exist y in K such
that Dy + f*y == g and to find one if there are some.  If t is a
monomial over k and the coefficients of f and g are in k(t), then y is
in k(t), and the outline of the algorithm here is given as:

1. Compute the normal part n of the denominator of y.  The problem is
then reduced to finding y' in k<t>, where y == y'/n.
2. Compute the special part s of the denominator of y.   The problem is
then reduced to finding y'' in k[t], where y == y''/(n*s)
3. Bound the degree of y''.
4. Reduce the equation Dy + f*y == g to a similar equation with f, g in
k[t].
5. Find the solutions in k[t] of bounded degree of the reduced equation.

See Chapter 6 of "Symbolic Integration I: Transcendental Functions" by
Manuel Bronstein.  See also the docstring of risch.py.

Ported from SymPy's ``sympy.integrals.rde`` (in the version of Aaron
Meurer's pull requests, see the notice below), on the polynomials of
:mod:`sympy_extras.integrals.risch.tower`. A degree bound is an integer,
:data:`OO` standing for ``+oo``.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.integrals.risch.risch import DifferentialExtension
>>> from sympy_extras.integrals.risch.rde import rischDE
>>> from sympy_extras.integrals.risch.tower import TowerPoly
>>> x, t = symbols('x t')
>>> DE = DifferentialExtension.manual([TowerPoly.from_expr(1, x), TowerPoly.from_expr(t, t)])
>>> P = lambda e: TowerPoly.from_expr(e, t)
>>> rischDE(P(1), P(1), P(2*t), P(1), DE)    # Dy + y == 2*exp(x): y == exp(x)
(TowerPoly(t**2, t), TowerPoly(t, t))
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/rde.py (SymPy 1.14 and Aaron Meurer's pull requests)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: the typed core and the polynomials of integrals.risch.tower
# instead of SymPy's Poly; integer degree bounds with OO for +oo.
# ---------------------------------------------------------------------
# Ported from SymPy (sympy.integrals.rde) and from Aaron Meurer's pull
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

from fractions import Fraction
from typing import Final, Optional, Union

from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.euclidtools import dup_gcd
from sympy_extras.polys.factortools import dup_zz_factor
from sympy_extras.polys.rings import Monomial
from .tower import RationalFunction, TowerPoly
from .risch import (gcdex_diophantine, frac_in, derivation, derivation_element, DifferentialExtension,
    splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative, unique_dummy)

__all__ = ['Degree', 'OO', 'is_oo', 'finite', 'order_at', 'order_at_oo', 'weak_normalizer', 'normal_denom',
           'special_denom', 'bound_degree', 'spde', 'no_cancel_b_large', 'no_cancel_b_small', 'no_cancel_equal',
           'cancel_primitive', 'cancel_exp', 'cancel_tan', 'solve_poly_rde', 'rischDE']

#: a degree bound: an integer, OO for +oo
Degree = int
#: +oo as a degree bound (the order of 0 at a place)
OO: Final[int] = 1 << 62


def is_oo(n: Degree) -> bool:
    """Whether the degree bound ``n`` is ``+oo``."""
    return n >= OO >> 1


def finite(n: Degree) -> int:
    """The degree bound as a Python integer; ``ValueError`` for ``oo``
    (the algorithms iterating over the degrees need a finite bound)."""
    if is_oo(n):
        raise ValueError("a finite degree bound is needed, got oo")
    return n


def _t(DE: DifferentialExtension) -> TowerPoly:
    return TowerPoly.monomial(DE.t, 1)


def _t2_1(DE: DifferentialExtension) -> TowerPoly:
    return TowerPoly.from_list([1, 0, 1], DE.t)


def _in_t(p: TowerPoly, DE: DifferentialExtension) -> TowerPoly:
    return p if p.gen == DE.t else p.as_poly_strict(DE.t)


def order_at(a: TowerPoly, p: TowerPoly, t: Symbol) -> Degree:
    """
    Computes the order of a at p, with respect to t.

    Explanation
    ===========

    For a, p in k[t], the order of a at p is defined as nu_p(a) = max({n
    in Z+ such that p**n|a}), where a != 0.  If a == 0, nu_p(a) = +oo.

    To compute the order at a rational function, a/b, use the fact that
    nu_p(a/b) == nu_p(a) - nu_p(b).

    This is the order function nu_p from Section 4.1 of Bronstein's book.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.rde import order_at
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> t = symbols('t')
    >>> order_at(TowerPoly.from_expr((t**2 + 1)**3*t, t), TowerPoly.from_expr(t**2 + 1, t), t)
    3
    """
    if a.is_zero:
        return OO
    a = a.as_poly_strict(t)
    p = p.as_poly_strict(t)
    if p == TowerPoly.monomial(t, 1):
        return a.ET()[0]

    # Uses binary search for calculating the power. power_list collects the tuples
    # (p^k,k) where each k is some power of 2. After deciding the largest k
    # such that k is power of 2 and p^k|a the loop iteratively calculates
    # the actual power.
    power_list: list[tuple[TowerPoly, int]] = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1, tracks_power))
        p1 = p1*p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = TowerPoly.one(t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product*final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n


def order_at_oo(a: TowerPoly, d: TowerPoly, t: Symbol) -> Degree:
    """
    Computes the order of a/d at oo (infinity), with respect to t.

    For f in k(t), the order or f at oo is defined as deg(d) - deg(a), where
    f == a/d.

    This is the order at infinity nu_oo from Section 4.3 of Bronstein's
    book.
    """
    if a.is_zero:
        return OO
    return d.degree(t) - a.degree(t)


def _rational_roots(p: TowerPoly) -> list[RationalFunction]:
    """The rational roots of ``p``, with their multiplicities, which are
    roots for every value of the parameters (and the other generators) of
    its coefficients: the roots over ``QQ`` of the gcd of the polynomials
    in the generator multiplying each monomial of the others. Only the
    rational (integer) roots matter to the callers (SymPy's
    ``real_roots``/``roots`` of the old code, filtered to rationals)."""
    a = p.as_element()
    gen = p.gen
    if not a or gen not in a.gens:
        return []
    i = a.gens.index(gen)
    components: dict[Monomial, dict[int, int]] = {}
    for m, c in a.numer.terms.items():
        key = tuple(e for j, e in enumerate(m) if j != i)
        components.setdefault(key, {})[m[i]] = c
    g: list[int] = []
    for terms in components.values():
        n = max(terms)
        dense = [terms.get(k, 0) for k in range(n, -1, -1)]
        g = dense if not g else dup_gcd(g, dense, ZZ)
        if len(g) <= 1:
            return []
    if len(g) <= 1:
        return []
    _, factors = dup_zz_factor(g, ZZ)
    roots: list[RationalFunction] = []
    for f, k in factors:
        if len(f) == 2:
            root = RationalFunction.from_fraction(Fraction(-f[1], f[0]))
            roots.extend([root]*k)
    return roots


def weak_normalizer(a: TowerPoly, d: TowerPoly, DE: DifferentialExtension,
                    z: Optional[Symbol] = None) -> tuple[TowerPoly, tuple[TowerPoly, TowerPoly]]:
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)

    This is ``WeakNormalizer`` from Section 6.1 of Bronstein's book.
    """
    z = z or unique_dummy('z')
    a, d = _in_t(a, DE), _in_t(d, DE)
    dn, ds = splitfactor(d, DE)

    # Compute d1, where dn == d1*d2**2*...*dn**n is a square-free
    # factorization of d.
    g = dn.gcd(dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(d_sqf_part.gcd(g))

    a1, b = gcdex_diophantine(d.quo(d1), d1, a)
    zf = RationalFunction.generator(z)
    r = TowerPoly.from_element_strict((a - derivation(d1, DE).mul_ground(zf)).resultant(d1), z)

    if not r.has(z):
        return (TowerPoly.one(DE.t), (a, d))

    # with parameters in the coefficients the positive integer roots among those found (the rest need no normalizing)
    N = [i.as_int() for i in _rational_roots(r)]
    N_int = sorted({n for n in N if n is not None and n > 0})

    # Bronstein's WeakNormalizer: q == prod(gcd(a - n*Dd1, d1)**n); SymPy
    # multiplies each gcd once per root of r counted with multiplicity
    # instead of n times
    q = TowerPoly.one(DE.t)
    for n in N_int:
        q = q*(a - derivation(d1, DE).mul_ground(n)).gcd(d1)**n

    dq = derivation(q, DE)
    sn = q*a - d*dq
    sd = q*d
    sn, sd = sn.cancel(sd, include=True)

    return (q, (sn, sd))


def normal_denom(fa: TowerPoly, fd: TowerPoly, ga: TowerPoly, gd: TowerPoly, DE: DifferentialExtension
                 ) -> tuple[TowerPoly, tuple[TowerPoly, TowerPoly], tuple[TowerPoly, TowerPoly], TowerPoly]:
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.

    This is ``RdeNormalDenominator`` from Section 6.1 of Bronstein's book.
    """
    fa, fd, ga, gd = _in_t(fa, DE), _in_t(fd, DE), _in_t(ga, DE), _in_t(gd, DE)
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h
    if c.div(en)[1]:
        raise NonElementaryIntegralException("en does not divide dn*h**2 in "
            "normal_denom().")
    ca = c*ga
    ca, cd = ca.cancel(gd, include=True)

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    # (dn*h, dn*h*f - dn*Dh, dn*h**2*g, h)
    return (a, (ba, bd), (ca, cd), h)


def _special_denom_cancel_bound(a: TowerPoly, ba: TowerPoly, bd: TowerPoly, n: Degree, DE: DifferentialExtension,
                                case: str) -> Degree:
    """
    Sharpen the bound n on the special part of the denominator in the
    possible-cancellation case nu_p(b) == 0.

    This is the part of the special denominator algorithms (Sections 6.2 and
    7.1 of Bronstein) shared by special_denom() and prde_special_denom().
    ``case`` is 'exp' or 'tan'.

    This is the possible-cancellation part of
    ``RdeSpecialDenomExp``/``RdeSpecialDenomTan`` (Section 6.2) and
    ``ParamRdeSpecialDenomExp``/``ParamRdeSpecialDenomTan`` (Section 7.1)
    of Bronstein's book.
    """
    # Delayed imports, since prde imports this module at the module level.
    from .prde import parametric_log_deriv, real_imag

    if case == 'exp':
        dcoeff = DE.d.quo(_t(DE))
        with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                  # because case != 'base'.
            alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
            etaa, etad = frac_in(dcoeff, DE.t)
            A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
            if A is not None:
                Q, m, z = A
                if Q == 1:
                    n = min(n, m)

    elif case == 'tan':
        dcoeff = DE.d.quo(_t2_1(DE))
        # alpha*sqrt(-1) + beta == Remainder(-b/a, t**2 + 1), with alpha and
        # beta in k.  real_imag() computes this without introducing sqrt(-1)
        # into the coefficient domain.  This must be done at the current
        # level, where t is the hypertangent monomial.
        betaa, alphaa, alphad = real_imag(-ba, bd*a, DE.t)
        betad = alphad
        alpha = alphaa.as_element()/alphad.as_element()
        beta = betaa.as_element()/betad.as_element()

        with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                  # because case != 'base'.
            alphaa, alphad = frac_in(alpha, DE.t)
            betaa, betad = frac_in(beta, DE.t)
            etaa, etad = frac_in(dcoeff, DE.t)

            if recognize_log_derivative(betaa.mul_ground(2), betad, DE):
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    # The condition from the book is
                    # alpha*sqrt(-1) + beta == 2*m*eta*sqrt(-1) + Dz/z for
                    # z in k(sqrt(-1))* and m in ZZ, in which case
                    # n = min(n, m).  parametric_log_deriv() solves
                    # n*alpha == Dv/v + m*eta over k, so its m corresponds
                    # to 2*m in the book's condition.  This is only an
                    # approximation of the real condition (z ranges over
                    # k(sqrt(-1))*, not k*); the complete version needs the
                    # structure theorems.
                    if Q == 1 and m % 2 == 0:
                        n = min(n, m//2)

    return n


def special_denom(a: TowerPoly, ba: TowerPoly, bd: TowerPoly, ca: TowerPoly, cd: TowerPoly,
                  DE: DifferentialExtension, case: str = 'auto') -> tuple[TowerPoly, TowerPoly, TowerPoly, TowerPoly]:
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.

    This is ``RdeSpecialDenomExp`` and ``RdeSpecialDenomTan`` from Section
    6.2 of Bronstein's book.
    """
    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = _t(DE)
    elif case == 'tan':
        p = _t2_1(DE)
    elif case in ('primitive', 'base'):
        B = ba.quo(bd)
        C = ca.quo(cd)
        return (a, B, C, TowerPoly.one(DE.t))
    elif case in ('other_linear', 'other_nonlinear'):
        raise NotImplementedError("The %s case is not implemented in "
            "special_denom()." % case)
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base', 'other_linear', 'other_nonlinear'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)

    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        n = _special_denom_cancel_bound(a, ba, bd, n, DE, case)

    # Note that this N, from Section 6.2, intentionally differs from the
    # N == max(0, -nb) of the parametric version in prde_special_denom()
    # (Section 7.1).
    N = max(0, -nb, n - nc)
    pN = p**N
    pn = p**-n

    A = a*pN
    B = ba*pN.quo(bd) + (a*derivation(p, DE).quo(p)*pN).mul_ground(n)
    C = (ca*pN*pn).quo(cd)
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, c*p**(N - n), p**-n)
    return (A, B, C, h)


def bound_degree(a: TowerPoly, b: TowerPoly, cQ: Union[TowerPoly, list[TowerPoly]], DE: DifferentialExtension,
                 case: str = 'auto', parametric: bool = False) -> Degree:
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``TowerPoly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of TowerPolys.

    This constitutes step 3 of the outline given in the rde.py docstring.

    This combines ``RdeBoundDegreeBase``, ``RdeBoundDegreePrim``,
    ``RdeBoundDegreeExp``, and ``RdeBoundDegreeNonLinear`` from Section 6.3
    of Bronstein's book (and their parametric analogues from Section 7.1).
    """
    if case == 'auto':
        case = DE.case

    a, b = _in_t(a, DE), _in_t(b, DE)
    da = a.degree(DE.t)
    db = b.degree(DE.t)

    # The parametric and regular cases are identical, except for this part
    if isinstance(cQ, list):
        dc = max(i.degree(DE.t) for i in cQ)
    else:
        dc = cQ.degree(DE.t)

    alpha = -b.LC()/a.LC()
    n: int
    alpha_int = alpha.as_int()

    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha_int is not None:
            n = max(0, alpha_int, dc - db)

    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)

        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])

        t1 = DE.t
        with DecrementLevel(DE):
            alphaa, alphad = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                # if alpha == m*Dt + Dz for z in k and m in ZZ:
                try:
                    A = limited_integrate(alphaa, alphad, [(etaa, etad)], DE)
                except NonElementaryIntegralException:
                    A = None
                if A is not None:
                    (za, zd), m_list = A
                    if len(m_list) != 1:
                        raise ValueError("Length of m should be 1")
                    m = m_list[0].as_int()
                    if m is not None:
                        n = max(n, m)

            elif db == da:
                # if alpha == Dz/z for z in k*:
                    # beta = -lc(a*Dz + b*z)/(z*lc(a))
                    # if beta == m*Dt + Dw for w in k and m in ZZ:
                        # n = max(n, m)
                from .prde import is_log_deriv_k_t_radical_in_field
                radical = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if radical is not None:
                    aa, z = radical
                    if aa == 1:
                        Dz = derivation_element(z, DE)
                        beta = -(a.mul_ground(Dz) + b.mul_ground(z)).as_poly_strict(t1).LC()/(z*a.LC())
                        betaa, betad = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            A = limited_integrate(betaa, betad,
                                [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            A = None
                        if A is not None:
                            (za, zd), m_list = A
                            if len(m_list) != 1:
                                raise ValueError("Length of m should be 1")
                            m = m_list[0].as_int()
                            if m is not None:
                                n = max(n, m)

    elif case == 'exp':
        from .prde import parametric_log_deriv

        n = max(0, dc - max(db, da))
        if da == db:
            etaa, etad = frac_in(DE.d.quo(_t(DE)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(alpha, DE.t)
                found = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if found is not None:
                    # if alpha == m*Dt/t + Dz/z for z in k* and m in ZZ:
                        # n = max(n, m)
                    a_found, m_found, z_found = found
                    if a_found == 1:
                        n = max(n, m_found)

    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha_int = (alpha/lam).as_int()
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha_int is not None:
            n = max(0, alpha_int, dc - db)

    elif case == 'other_linear':
        raise NotImplementedError("The other_linear case is not implemented "
            "in bound_degree().")
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base', 'other_linear', 'other_nonlinear'}, not %s." % case)

    return n


def spde(a: TowerPoly, b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension
         ) -> tuple[TowerPoly, TowerPoly, Degree, TowerPoly, TowerPoly]:
    """
    Rothstein's Special Polynomial Differential Equation algorithm.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with
    ``a != 0``, either raise NonElementaryIntegralException, in which case the
    equation a*Dq + b*q == c has no solution of degree at most ``n`` in
    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,
    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree
    at most n of a*Dq + b*q == c must be of the form
    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.

    This constitutes step 4 of the outline given in the rde.py docstring.

    This is ``SPDE`` from Section 6.4 of Bronstein's book.
    """
    a, b, c = _in_t(a, DE), _in_t(b, DE), _in_t(c, DE)
    zero = TowerPoly.zero(DE.t)

    alpha = TowerPoly.one(DE.t)
    beta = TowerPoly.zero(DE.t)

    # Generous pass allowance for n == oo (see below); any solution of
    # degree d is found within about d/deg(a) passes, so this only
    # defers solutions of very high degree to an honest error.
    oo_passes = 4*(a.degree(DE.t) + b.degree(DE.t) + c.degree(DE.t) + 4)

    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)  # -1 is more to the point
        if n < 0:
            raise NonElementaryIntegralException("n became negative in "
                "spde().")

        g = a.gcd(b)
        if not c.rem(g).is_zero:  # g does not divide c
            raise NonElementaryIntegralException("gcd(a, b) does not divide "
                "c in spde().")

        a, b, c = a.quo(g), b.quo(g), c.quo(g)

        if a.degree(DE.t) == 0:
            b = b.quo(a)
            c = c.quo(a)
            return (b, c, n, alpha, beta)

        if is_oo(n):
            # Each pass through the reduction below decreases n by
            # deg(a) > 0 and "no solution" is detected by n < 0, so with
            # no degree bound the loop can run forever when there is no
            # solution (e.g. a == t, b == 1, c == x over t == exp(x):
            # gcd(a, b) == 1 stays trivial, c cycles through nonzero
            # constants, and no other exit is ever taken).  Allow a
            # generous number of passes (solvable cases terminate
            # quickly, with c becoming zero), then give up with an
            # honest error -- NotImplementedError, not a nonexistence
            # claim.
            oo_passes -= 1
            if oo_passes < 0:
                raise NotImplementedError("spde() ran out of passes with "
                    "an infinite degree bound; cannot decide solvability.")

        r, z = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)

        beta += alpha * r
        alpha *= a


def no_cancel_b_large(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension) -> TowerPoly:
    """
    Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with ``b != 0`` and either D == d/dt or
    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in
    which case the equation ``Dq + b*q == c`` has no solution of degree at
    most n in k[t], or a solution q in k[t] of this equation with
    ``deg(q) < n``.

    This is ``PolyRischDENoCancel1`` from Section 6.5 of Bronstein's book.
    """
    b, c = _in_t(b, DE), _in_t(c, DE)
    q = TowerPoly.zero(DE.t)

    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException("deg(c) - deg(b) is not "
                "between 0 and n in no_cancel_b_large().")

        p = TowerPoly.monomial(DE.t, m, c.LC()/b.LC())
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def no_cancel_b_small(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension
                      ) -> Union[TowerPoly, tuple[TowerPoly, TowerPoly, TowerPoly]]:
    """
    Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or
    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the
    equation Dq + b*q == c has no solution of degree at most n in k[t],
    or a solution q in k[t] of this equation with deg(q) <= n, or the
    tuple (h, b0, c0) such that h in k[t], b0, c0, in k (as constant
    polynomials in t), and for any solution q in k[t] of degree at most n
    of Dq + bq == c, y == q - h is a solution in k of Dy + b0*y == c0.

    This is ``PolyRischDENoCancel2`` from Section 6.5 of Bronstein's book.
    """
    b, c = _in_t(b, DE), _in_t(c, DE)
    q = TowerPoly.zero(DE.t)

    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException("m is not between 0 and n "
                "in no_cancel_b_small().")

        if m > 0:
            p = TowerPoly.monomial(DE.t, m, c.LC()/(DE.d.LC()*m))
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException("deg(b) != deg(c) in "
                    "the m == 0 case of no_cancel_b_small().")
            if b.degree(DE.t) == 0:
                return (q, b, c)
            p = TowerPoly.ground(c.LC()/b.LC(), DE.t)

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


# TODO: better name for this function
def no_cancel_equal(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension
                    ) -> Union[TowerPoly, tuple[TowerPoly, Degree, TowerPoly]]:
    """
    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with deg(D) >= 2, n either an integer
    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c has
    no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n, or the tuple (h, m, C) such that h
    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of
    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]
    of degree at most m of Dy + b*y == C.

    This is ``PolyRischDENoCancel3`` from Section 6.5 of Bronstein's book.
    """
    b, c = _in_t(b, DE), _in_t(c, DE)
    q = TowerPoly.zero(DE.t)
    lc = (-b.LC()/DE.d.LC()).as_int()
    # A cancellation degree above the bound n is unattainable
    # (solutions have degree at most n), so treat it like a
    # noninteger ratio: no cancellation at any reachable degree.
    if lc is not None and lc > 0 and lc <= n:
        M = lc
    else:
        M = -1

    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException("m is not between 0 and n "
                "in no_cancel_equal().")

        u = DE.d.LC()*m + b.LC()
        if u.is_zero:
            return (q, m, c)
        if m > 0:
            p = TowerPoly.monomial(DE.t, m, c.LC()/u)
        else:
            if c.degree(DE.t) != DE.d.degree(DE.t) - 1:
                raise NonElementaryIntegralException("deg(c) != deg(D) - 1 "
                    "in the m == 0 case of no_cancel_equal().")
            else:
                p = TowerPoly.ground(c.LC()/b.LC(), DE.t)

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def cancel_primitive(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension) -> TowerPoly:
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.

    This is ``PolyRischDECancelPrim`` from Section 6.6 of Bronstein's book.
    """
    # Delayed imports
    from .prde import is_log_deriv_k_t_radical_in_field, is_deriv_in_field
    b, c = _in_t(b, DE), _in_t(c, DE)
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)

    if A is not None:
        m, z = A
        if m == 1:  # b == Dz/z
            # D(z*q) == z*(Dq + b*q), so q solves Dq + b*q == c iff
            # p == z*q satisfies Dp == z*c.  So find an antiderivative
            # p in k[t] of z*c with deg(p) <= n, and return q == p/z.
            zca, zcd = frac_in(z*c.as_element(), DE.t, cancel=True)
            B = is_deriv_in_field(zca, zcd, DE)
            if B is None:
                raise NonElementaryIntegralException("z*c is not the "
                    "derivative of an element of k(t) in cancel_primitive().")
            pa, pd = B
            # For a primitive t, an antiderivative in k(t) of an element
            # of k[t] is itself in k[t] (it can have no poles), and any
            # additive constant keeps q == p/z in k[t], so any choice of
            # p will do.
            found = TowerPoly.from_element(pa.as_element()/(pd.as_element()*z), DE.t)
            if found is None:
                raise NonElementaryIntegralException("p/z is not in k[t] "
                    "in cancel_primitive().")
            if found.degree(DE.t) > n:
                raise NonElementaryIntegralException("deg(p/z) > n in "
                    "cancel_primitive().")
            return found

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException("n < deg(c) in "
            "cancel_primitive().")

    q = TowerPoly.zero(DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException("n < deg(c) in "
                "cancel_primitive().")
        with DecrementLevel(DE):
            a2a, a2d = frac_in(c.LC(), DE.t)
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        stm = TowerPoly.monomial(DE.t, m, sa.as_element()/sd.as_element())
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)

    return q


def cancel_exp(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension) -> TowerPoly:
    """
    Poly Risch Differential Equation - Cancellation: Hyperexponential case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt/t in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.

    This is ``PolyRischDECancelExp`` from Section 6.6 of Bronstein's book.
    """
    from .prde import parametric_log_deriv, is_deriv_in_field
    b, c = _in_t(b, DE), _in_t(c, DE)
    eta = DE.d.quo(_t(DE)).as_element()

    with DecrementLevel(DE):
        etaa, etad = frac_in(eta, DE.t)
        ba, bd = frac_in(b, DE.t)
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)

    tf = RationalFunction.generator(DE.t)
    if A is not None:
        a, m, z = A
        if a == 1:
            # b == Dz/z + m*Dt/t, so D(z*t**m*q) == z*t**m*(Dq + b*q).
            # Hence q solves Dq + b*q == c iff p == z*t**m*q satisfies
            # Dp == z*t**m*c.  So find an antiderivative p in k<t> of
            # z*t**m*c; then q == p/(z*t**m) must be in k[t] with
            # deg(q) <= n.
            pa, pd = frac_in(z*tf**m*c.as_element(), DE.t, cancel=True)
            B = is_deriv_in_field(pa, pd, DE)
            if B is None:
                raise NonElementaryIntegralException("z*t**m*c is not the "
                    "derivative of an element of k(t) in cancel_exp().")
            va, vd = B
            p = va.as_element()/vd.as_element()
            if m > 0:
                # p is determined only up to an additive constant.  For
                # m > 0 the true p == z*t**m*q has no t-free term (all its
                # monomials have degree >= m), while a constant term would
                # make p/(z*t**m) non-polynomial, so remove it.
                p -= p.subs(DE.t, RationalFunction.from_int(0))
            # For m <= 0 any additive constant C leaves q in k[t]
            # (C/(z*t**m) == C*t**(-m)/z), and q + C*t**(-m)/z is still a
            # solution, since t**(-m)/z solves the homogeneous equation.
            found = TowerPoly.from_element(p/(z*tf**m), DE.t)
            if found is None:
                raise NonElementaryIntegralException("p/(z*t**m) is not in "
                    "k[t] in cancel_exp().")
            q0 = found
            if q0.degree(DE.t) > n:
                if m < 0 and q0.degree(DE.t) == -m and \
                        not (z*q0.nth(-m)).has(*DE.T):
                    # The homogeneous solutions are C*t**(-m)/z for
                    # constants C, so the t**(-m) coefficient of q is
                    # adjustable by the constant of integration exactly
                    # when it is C/z for a constant C.
                    q0 = q0 - TowerPoly.monomial(DE.t, -m, q0.nth(-m))
                if q0.degree(DE.t) > n:
                    raise NonElementaryIntegralException("deg(p/(z*t**m)) "
                        "> n in cancel_exp().")
            return q0

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException("n < deg(c) in cancel_exp().")

    q = TowerPoly.zero(DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException("n < deg(c) in "
                "cancel_exp().")
        # a1 = b + m*Dt/t
        a1 = b.as_element()
        with DecrementLevel(DE):
            # TODO: Write a dummy function that does this idiom
            a1a, a1d = frac_in(a1, DE.t)
            a1a = a1a*etad + (etaa*a1d).mul_ground(m)
            a1d = a1d*etad

            a2a, a2d = frac_in(c.LC(), DE.t)

            sa, sd = rischDE(a1a, a1d, a2a, a2d, DE)
        stm = TowerPoly.monomial(DE.t, m, sa.as_element()/sd.as_element())
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)  # deg(c) becomes smaller
    return q


def cancel_tan(b0: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension) -> TowerPoly:
    """
    Poly Risch Differential Equation - Cancellation: Tangent case.

    Explanation
    ===========

    Given a derivation D on k[t], an integer ``n >= 0``, ``b0`` in k, and
    ``c`` in k[t] with Dt/(t**2 + 1) == eta in k and sqrt(-1) not in
    k(t), either raise NonElementaryIntegralException, in which case
    the equation Dq + (b0 - n*eta*t)*q == c has no solution of degree
    at most n in k[t], or return a solution q in k[t] of this equation
    with deg(q) <= n.  ``b0`` is a TowerPoly in DE.t (of degree 0).

    Unlike in the primitive and hyperexponential cases, ``n`` must be
    a finite integer: it appears in the equation itself, and the
    recursion descends from n by 2 to 0.  The callers get n from
    bound_degree(), which is always finite in the nonlinear case.

    This is ``PolyRischDECancelTan`` from Section 6.6 of Bronstein's
    book.  The book's pseudocode returns u + v*t unverified when
    n == 1; since the t**2 coefficient of Dq + (b0 - eta*t)*q
    vanishes identically for deg(q) <= 1, a right hand side with a
    nonzero t**2 coefficient has no solution while the projected
    coupled system may well be solvable, so the unverified return can
    produce a wrong answer (e.g. b0 == 1, c == t**2 + 1 returns
    q == 0).  Here n == 1 runs through the same reduction as n >= 2,
    and the recursion (at n == -1) verifies that the residual is
    exactly zero.
    """
    # Delayed imports: cde and prde import this module at the module
    # level.
    from .cde import coupled_DE_system
    from .prde import is_deriv_in_field

    b0, c = _in_t(b0, DE), _in_t(c, DE)
    if is_oo(n):
        raise ValueError("n must be a finite integer in cancel_tan().")
    if n < 0:
        if c.is_zero:
            return c  # return 0
        raise NonElementaryIntegralException("n < 0 and c != 0 in "
            "cancel_tan().")

    if n == 0:
        if c.degree(DE.t) > 0:
            raise NonElementaryIntegralException("deg(c) > 0 == n in "
                "cancel_tan().")
        if b0.is_zero:
            # Dq == c with q, c in k: in-field integration in k.
            with DecrementLevel(DE):
                ca, cd = frac_in(c.nth(0), DE.t)
                B = is_deriv_in_field(ca, cd, DE)
                if B is None:
                    raise NonElementaryIntegralException("c is not the "
                        "derivative of an element of k in cancel_tan().")
                va, vd = B
                qe = va.as_element()/vd.as_element()
            return TowerPoly.from_element_strict(qe, DE.t)
        with DecrementLevel(DE):
            b0a, b0d = frac_in(b0.as_element(), DE.t)
            ca, cd = frac_in(c.nth(0), DE.t)
            sa, sd = rischDE(b0a, b0d, ca, cd, DE)
            s = sa.as_element()/sd.as_element()
        return TowerPoly.from_element_strict(s, DE.t)

    eta = DE.d.exquo(_t2_1(DE)).as_element()
    p = _t2_1(DE)

    # Project mod t - sqrt(-1) (evaluate at t == sqrt(-1)): with
    # c1*t + c0 the remainder of c by p, the constant coefficients
    # (u, v) of any solution q == u + v*t + p*h satisfy the coupled
    # system Du + b0*u + n*eta*v == c0, Dv - n*eta*u + b0*v == c1
    # over k.
    _, rem = c.div(p)
    c0, c1 = rem.nth(0), rem.nth(1)
    with DecrementLevel(DE):
        u, v = coupled_DE_system(frac_in(b0.as_element(), DE.t),
            frac_in(-eta*n, DE.t), frac_in(c0, DE.t), frac_in(c1, DE.t),
            DE)
        ue = u[0].as_element()/u[1].as_element()
        ve = v[0].as_element()/v[1].as_element()
    tf = RationalFunction.generator(DE.t)
    r = TowerPoly.from_element_strict(ue + ve*tf, DE.t)
    b = TowerPoly.from_element_strict(b0.as_element() - eta*tf*n, DE.t)
    # h == (q - r)/p solves the same kind of equation with the same
    # b0 and a degree bound of n - 2 (b + Dp/p == b0 - (n - 2)*eta*t),
    # and its right hand side is an exact quotient by construction.
    cnew = (c - derivation(r, DE) - b*r).exquo(p)
    h = cancel_tan(b0, cnew, n - 2, DE)
    return p*h + r


def _no_cancel_equal_applies(b: TowerPoly, n: Degree, DE: DifferentialExtension) -> bool:
    """
    Whether no_cancel_equal() decides the deg(b) == deg(Dt) - 1 case.

    Explanation
    ===========

    Cancellation is only possible at deg(q) == -lc(b)/lc(Dt), so the
    cancellation algorithms take over exactly when that ratio is a
    positive integer equal to the degree bound n (when n exceeds the
    ratio, no_cancel_equal() strips the degrees above it and hands
    off at the ratio; when n is below it, or the ratio is not a
    positive integer at all — including non-real numeric ratios,
    which admit no ordered comparison — the cancellation degree is
    unattainable and no_cancel_equal() decides the equation, ignoring
    it).  A nonconstant ratio never equals an integer degree, so no
    cancellation occurs at any degree — but that must hold under every
    specialization of any parameters of the constant field, since a
    parameter-dependent cancellation degree would make
    no_cancel_equal()'s divisions only generically valid.  This is
    automatic for a ratio free of parameters, and a parameter-mixed
    ratio is accepted when its denominator is free of the tower
    variables and its numerator has a positive-degree monomial in them
    with a (nonzero) numeric coefficient, which survives every
    specialization; other parameter-dependent ratios are refused.
    """
    b = _in_t(b, DE)
    ratio = -b.LC()/DE.d.LC()
    if ratio.is_number:
        r = ratio.as_int()
        return not (r is not None and r > 0) or n != r
    T: set[Symbol] = set(DE.T)
    if all(g.free_symbols <= T for g in ratio.gens):
        return True
    if not ratio.denominator().has(*DE.T):
        positions = [i for i, g in enumerate(ratio.gens) if g in T]
        parameters = [i for i, g in enumerate(ratio.gens) if not g.free_symbols <= T]
        # the coefficient of a monomial in the tower variables is a number
        # when no parameter appears with it
        grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
        for m in ratio.numer.terms:
            grouped.setdefault(tuple(m[i] for i in positions), []).append(tuple(m[i] for i in parameters))
        for tower_part, rest in grouped.items():
            if any(tower_part) and all(not any(r) for r in rest):
                return True
    raise TypeError("-lc(b)/lc(Dt) must be a number or an element of k "
        "that stays nonconstant under every specialization of the "
        "parameters in solve_poly_rde().")


def solve_poly_rde(b: TowerPoly, c: TowerPoly, n: Degree, DE: DifferentialExtension) -> TowerPoly:
    """
    Solve a Polynomial Risch Differential Equation with degree bound ``n``.

    This constitutes step 4 of the outline given in the rde.py docstring.

    This dispatches among the algorithms of Sections 6.5 and 6.6 of
    Bronstein's book; it has no single named counterpart.  The parametric
    analogue of this dispatch is param_poly_rischDE() in prde.py.
    """
    b, c = _in_t(b, DE), _in_t(c, DE)
    # No cancellation
    if not b.is_zero and (DE.case == 'base' or
            b.degree(DE.t) > max(0, DE.d.degree(DE.t) - 1)):

        return no_cancel_b_large(b, c, n, DE)

    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and \
            (DE.case == 'base' or DE.d.degree(DE.t) >= 2):

        R = no_cancel_b_small(b, c, n, DE)

        if isinstance(R, TowerPoly):
            return R
        else:
            # The solution y in k of Dy + b0*y == c0 (pg. 209): k is a
            # field, so this is a Risch differential equation over k, or
            # the equation b0*y == c0 if k is the constant field
            h, b0, c0 = R
            if DE.case == 'base':
                return h + TowerPoly.ground(c0.as_element()/b0.as_element(), DE.t)
            with DecrementLevel(DE):
                ba, bd = frac_in(b0.as_element(), DE.t)
                ca, cd = frac_in(c0.as_element(), DE.t)
                ya, yd = rischDE(ba, bd, ca, cd, DE)
                y = ya.as_element()/yd.as_element()
            return h + TowerPoly.from_element_strict(y, DE.t)

    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and \
            _no_cancel_equal_applies(b, n, DE):

        found = no_cancel_equal(b, c, n, DE)

        if isinstance(found, TowerPoly):
            return found
        else:
            h, m, C = found
            # XXX: Or should it be rischDE()?
            return h + solve_poly_rde(b, C, m, DE)

    else:
        # Cancellation
        if b.is_zero:
            # Dq == c: in-field integration (Section 6.6)
            from .prde import is_deriv_in_field
            B = is_deriv_in_field(c, TowerPoly.one(DE.t), DE)
            if B is None:
                raise NonElementaryIntegralException("c is not the "
                    "derivative of an element of k(t) in solve_poly_rde().")
            va, vd = B
            # For a primitive or hyperexponential t, an antiderivative in
            # k(t) of an element of k[t] is itself in k[t], and any
            # additive constant is also in k[t], so any choice of the
            # antiderivative will do.
            q = TowerPoly.from_element(va.as_element()/vd.as_element(), DE.t)
            if q is None:
                raise NonElementaryIntegralException("The antiderivative "
                    "of c is not in k[t] in solve_poly_rde().")
            if q.degree(DE.t) > n:
                raise NonElementaryIntegralException("The antiderivative "
                    "of c has degree > n in solve_poly_rde().")
            return q
        else:
            if DE.case == 'exp':
                return cancel_exp(b, c, n, DE)

            elif DE.case == 'primitive':
                return cancel_primitive(b, c, n, DE)

            elif DE.case == 'tan':
                # The cancellation case has b == b0 - n*eta*t with
                # b0 in k, where n is the degree bound (Section 6.6);
                # the pipeline reaches this branch in that shape via
                # no_cancel_equal()'s hand-off.
                eta = DE.d.exquo(_t2_1(DE)).as_element()
                if not is_oo(n) and b.degree(DE.t) <= 1 and \
                        (b.nth(1) + eta*n).is_zero:
                    return cancel_tan(TowerPoly.ground(b.nth(0), DE.t), c, n, DE)
                raise NotImplementedError("The tangent cancellation case "
                    "requires b == b0 - n*eta*t with b0 in k in "
                    "solve_poly_rde().")

            else:
                raise NotImplementedError("Other Poly (P)RDE cancellation "
                    "cases are not yet implemented (%s)." % DE.case)


def rischDE(fa: TowerPoly, fd: TowerPoly, ga: TowerPoly, gd: TowerPoly, DE: DifferentialExtension
            ) -> tuple[TowerPoly, TowerPoly]:
    """
    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.

    This chains together the algorithms of Sections 6.1-6.6 of Bronstein's
    book; the book does not give it as a single named pseudocode function.
    """
    # Substitute y == z/q, where q is the weak normalizer of f.  By Theorem
    # 6.1.2, y in k(t) solves Dy + f*y == g iff z == q*y solves
    # Dz + (f - Dq/q)*z == q*g, and f - Dq/q is weakly normalized, as
    # required by normal_denom().
    fa, fd, ga, gd = _in_t(fa, DE), _in_t(fd, DE), _in_t(ga, DE), _in_t(gd, DE)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    ga, gd = (ga*q).cancel(gd, include=True)
    a, (ba, bd), (ca, cd), hn = normal_denom(fa, fd, ga, gd, DE)
    A, B, C, hs = special_denom(a, ba, bd, ca, cd, DE)
    try:
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        # bound_degree() could not decide one of its structure-theorem
        # queries.  Proceeding with n == oo is sound: no bound is needed
        # to accept a solution, and the no-cancellation and cancellation
        # algorithms all terminate by descending on deg(c).  The only
        # step that needs a finite bound to terminate is the deg(a) > 0
        # reduction in spde(), which raises NotImplementedError instead
        # of looping forever when it would need one.
        n = OO

    B, C, m, alpha, beta = spde(A, B, C, n, DE)
    if C.is_zero:
        y = C
    else:
        y = solve_poly_rde(B, C, m, DE)

    # The solution found so far is z == (alpha*y + beta)/(hn*hs); undo the
    # weak normalizer substitution y == z/q.
    return (alpha*y + beta, q*hn*hs)
