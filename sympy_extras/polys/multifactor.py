"""Factorisation of multivariate polynomials over ``ZZ`` by Wang's
enhanced extended Zassenhaus algorithm (EEZ), on the typed sparse
polynomials (:class:`~sympy_extras.polys.rings.PolyElement` over ``ZZ``).

Ported from SymPy's ``sympy.polys.factortools`` (``dmp_zz_factor``,
``dmp_zz_wang`` and its helpers): the content in the first variable is
factored recursively; the square-free part is evaluated at integer
points of the other variables until the univariate image is square-free
with a leading coefficient whose factors are told apart by their values
(``dmp_zz_wang_test_points``); the image is factored over ``ZZ``, the
leading coefficients are distributed among the factors
(``dmp_zz_wang_lead_coeffs``), and the factors are lifted one variable at
a time by Hensel lifting with multivariate diophantine equations
(``dmp_zz_wang_hensel_lifting``); trial division gives the
multiplicities. The dense recursive polynomials of SymPy (``x0`` the
outer variable) are sparse polynomials of the ring in all the variables
here: ``dmp_raise`` is the identity, ``dmp_nest([1, -a], n)`` is ``x_n -
a``, evaluating the tail substitutes the last variables.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.polys.rings import PolyRing
>>> from sympy_extras.polys.domains import ZZ
>>> x, y = Symbol('x'), Symbol('y')
>>> R = PolyRing(ZZ, ['x', 'y'])
>>> c, factors = zz_factor(R.from_expr(2*x**3*y - 2*x*y**3 + 4*x**2 - 4*y**2))
>>> c, [(f.to_expr(), k) for f, k in factors]
(2, [(x - y, 1), (x + y, 1), (x*y + 2, 1)])
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/factortools.py (dmp_zz_factor, dmp_zz_wang,
# dmp_zz_wang_non_divisors, dmp_zz_wang_test_points, dmp_zz_wang_lead_coeffs,
# dup_zz_diophantine, dmp_zz_diophantine, dmp_zz_wang_hensel_lifting,
# dmp_trial_division, dmp_zz_mignotte_bound), sympy/polys/sqfreetools.py
# (dmp_sqf_part), sympy/polys/polyutils.py (_sort_factors) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: sparse polynomials in all the variables instead of dense nested
# lists; a seeded local random generator for the evaluation points.
# ---------------------------------------------------------------------
from __future__ import annotations

import random
from math import factorial, gcd, isqrt
from typing import Optional, Sequence, Union

from sympy_extras.ntheory.generate import nextprime

from . import densearith, densebasic, densetools, euclidtools, factortools, sqfreetools
from .domains.finitefield import GF
from .domains.integerring import ZZ
from .domains.sparsepolynomialring import IntegerPolynomialRing
from .polyerrors import EvaluationFailed, ExtraneousFactors
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['zz_factor', 'zz_wang', 'sqf_part', 'trial_division', 'factor_sort_key']

Poly = PolyElement[int]
Factors = list[tuple[Poly, int]]
DenseKey = Union[int, tuple['DenseKey', ...]]

EEZ_NUMBER_OF_CONFIGS = 3
EEZ_NUMBER_OF_TRIES = 5
EEZ_MODULUS_STEP = 2


# ---------------------------------------------------------------------
# sparse versions of the dense helpers

def _degree(f: Poly, i: int) -> int:
    return max((m[i] for m in f.terms), default=-1)


def _lc_first(f: Poly) -> Poly:
    """The leading coefficient in ``x0``, a polynomial in the others
    (SymPy's ``dmp_LC``)."""
    top = _degree(f, 0)
    return f.ring.from_dict({(0,) + m[1:]: c for m, c in f.terms.items() if m[0] == top})


def _without_leading(f: Poly) -> Poly:
    """``f`` without its terms of highest degree in ``x0`` (dense ``h[1:]``)."""
    top = _degree(f, 0)
    return f.ring.from_dict({m: c for m, c in f.terms.items() if m[0] != top})


def _eval_var(f: Poly, j: int, a: int) -> Poly:
    terms: dict[Monomial, int] = {}
    for m, c in f.terms.items():
        key = m[:j] + (0,) + m[j + 1:]
        terms[key] = terms.get(key, 0) + c * a ** m[j]
    return f.ring.from_dict(terms)


def _eval_tail(f: Poly, A: Sequence[int]) -> Poly:
    """The last ``len(A)`` variables set to ``A`` (``dmp_eval_tail``)."""
    n = f.ring.ngens
    for k, a in enumerate(A):
        f = _eval_var(f, n - len(A) + k, a)
    return f


def _constant(f: Poly) -> int:
    return f.terms.get(f.ring.zero_monom, 0)


def _diff_eval(f: Poly, k: int, a: int, j: int) -> Poly:
    """The ``k``-th derivative in ``x_j`` at ``x_j = a`` (``dmp_diff_eval_in``)."""
    for _ in range(k):
        f = f.diff(j)
    return _eval_var(f, j, a)


def _trunc(f: Poly, p: int) -> Poly:
    """Symmetric residues of the coefficients modulo ``p``
    (``dmp_ground_trunc``)."""
    return f.ring.from_dict({m: _symmetric(c, p) for m, c in f.terms.items()})


def _symmetric(c: int, p: int) -> int:
    r = c % p
    return r - p if r > p // 2 else r


def _quo_ground(f: Poly, c: int) -> Poly:
    return f.ring.from_dict({m: v // c for m, v in f.terms.items()})


def _content(f: Poly) -> int:
    result = 0
    for c in f.terms.values():
        result = gcd(result, c)
    return result


def _primitive(f: Poly) -> tuple[int, Poly]:
    c = _content(f)
    if c in (0, 1):
        return c, f
    return c, _quo_ground(f, c)


def _expand(H: Sequence[Poly], ring: PolyRing[int]) -> Poly:
    result = ring.one
    for h in H:
        result = result * h
    return result


def _variable_minus(ring: PolyRing[int], j: int, a: int) -> Poly:
    """``x_j - a``."""
    monomial = tuple(1 if i == j else 0 for i in range(ring.ngens))
    return ring.from_dict({monomial: 1, ring.zero_monom: -a})


def _to_dense(f: Poly) -> list[int]:
    """A polynomial in ``x0`` only as a dense list."""
    return densebasic.dup_from_dict({m[0]: c for m, c in f.terms.items()}, ZZ)


def _from_dense(f: Sequence[int], ring: PolyRing[int]) -> Poly:
    n = len(f) - 1
    rest = tuple(0 for _ in range(ring.ngens - 1))
    return ring.from_dict({(n - i,) + rest: c for i, c in enumerate(f) if c})


def _exquo(f: Poly, g: Poly) -> Optional[Poly]:
    """``f / g`` when the division is exact over ``ZZ``, else ``None``."""
    try:
        quotients, remainder = f.div([g])
    except ValueError:
        return None
    return None if remainder.terms else quotients[0]


def _drop_first(f: Poly, ring: PolyRing[int]) -> Poly:
    """A polynomial free of ``x0`` in the ring of the other variables."""
    return ring.from_dict({m[1:]: c for m, c in f.terms.items()})


def _add_first(f: Poly, ring: PolyRing[int]) -> Poly:
    return ring.from_dict({(0,) + m: c for m, c in f.terms.items()})


# PORTED FROM SYMPY: sympy/polys/polyutils.py:_sort_factors (the dense key)
def factor_sort_key(f: Poly) -> DenseKey:
    """The dense recursive representation of ``f`` as nested tuples, which
    SymPy's lists compare as."""
    return _dense_key(dict(f.terms), 0, f.ring.ngens)


def _dense_key(terms: dict[Monomial, int], i: int, n: int) -> DenseKey:
    if not terms:
        return ()
    by_degree: dict[int, dict[Monomial, int]] = {}
    for m, c in terms.items():
        by_degree.setdefault(m[i], {})[m] = c
    top = max(by_degree)
    if i == n - 1:
        return tuple(sum(by_degree.get(d, {}).values()) for d in range(top, -1, -1))
    return tuple(_dense_key(by_degree.get(d, {}), i + 1, n) for d in range(top, -1, -1))


def _sorted(factors: Factors) -> Factors:
    return sorted(factors, key=lambda item: (_degree(item[0], 0) + 1, item[1], factor_sort_key(item[0])))


# ---------------------------------------------------------------------
# the square-free part and the trial division

# PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dmp_sqf_part (over ZZ)
def sqf_part(f: Poly) -> Poly:
    if not f.terms:
        return f
    if f.LC < 0:
        f = -f
    R = IntegerPolynomialRing(f.ring.names)
    g = f
    for i in range(f.ring.ngens):
        g = R.gcd(g, f.diff(i))
    quotient = _exquo(f, g)
    if quotient is None:
        raise ArithmeticError('the gcd does not divide')
    return _primitive(quotient)[1]


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_trial_division
def trial_division(f: Poly, factors: Sequence[Poly]) -> Factors:
    result: Factors = []
    for factor in factors:
        k = 0
        while True:
            quotient = _exquo(f, factor)
            if quotient is None:
                break
            f, k = quotient, k + 1
        if k == 0:
            raise RuntimeError('trial division failed')
        result.append((factor, k))
    return _sorted(result)


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_mignotte_bound
def _mignotte_bound(f: Poly) -> int:
    a = max(abs(c) for c in f.terms.values())
    b = abs(f.LC)
    n = sum(max(_degree(f, i), 0) for i in range(f.ring.ngens))
    return isqrt(n + 1) * 2**n * a * b


# ---------------------------------------------------------------------
# Wang's algorithm

# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_wang_non_divisors
def _non_divisors(E: Sequence[int], cs: int, ct: int) -> Optional[list[int]]:
    result = [cs * ct]
    for q0 in E:
        q = abs(q0)
        for r0 in reversed(result):
            r = r0
            while r != 1:
                r = gcd(r, q)
                q = q // r
            if q == 1:
                return None
        result.append(q)
    return result[1:]


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_wang_test_points
def _test_points(f: Poly, T: Sequence[Poly], ct: int, A: Sequence[int]) -> tuple[int, list[int], list[int]]:
    if not _eval_tail(_lc_first(f), A).terms:
        raise EvaluationFailed('no luck')
    g = _to_dense(_eval_tail(f, A))
    if not sqfreetools.dup_sqf_p(g, ZZ):
        raise EvaluationFailed('no luck')
    c, h = densetools.dup_primitive(g, ZZ)
    if h[0] < 0:
        c, h = -c, densearith.dup_neg(h, ZZ)
    E = [_constant(_eval_tail(t, A)) for t in T]
    if _non_divisors(E, c, ct) is None:
        raise EvaluationFailed('no luck')
    return c, h, E


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_wang_lead_coeffs
def _lead_coeffs(f: Poly, T: Sequence[Poly], cs: int, E: Sequence[int], H: Sequence[list[int]],
                 A: Sequence[int]) -> tuple[Poly, list[list[int]], list[Poly]]:
    ring = f.ring
    C: list[Poly] = []
    J = [0] * len(E)
    for h in H:
        c = ring.one
        d = h[0] * cs
        for i in reversed(range(len(E))):
            k, e, t = 0, E[i], T[i]
            while not d % e:
                d, k = d // e, k + 1
            if k != 0:
                c, J[i] = c * t**k, 1
        C.append(c)
    if not all(J):
        raise ExtraneousFactors('the leading coefficients do not split')
    CC: list[Poly] = []
    HH: list[list[int]] = []
    for c, h in zip(C, H):
        d = _constant(_eval_tail(c, A))
        lc = h[0]
        if cs == 1:
            cc = lc // d
        else:
            g = gcd(lc, d)
            d, cc = d // g, lc // g
            h, cs = densearith.dup_mul_ground(h, d, ZZ), cs // d
        CC.append(c.mul_ground(cc))
        HH.append(h)
    if cs == 1:
        return f, HH, CC
    CCC = [c.mul_ground(cs) for c in CC]
    HHH = [densearith.dup_mul_ground(h, cs, ZZ) for h in HH]
    return f.mul_ground(cs ** (len(H) - 1)), HHH, CCC


# PORTED FROM SYMPY: sympy/polys/factortools.py:dup_zz_diophantine
def _dup_diophantine(F: Sequence[list[int]], m: int, p: int) -> list[list[int]]:
    """``s_i`` with ``sum s_i prod_{j != i} F_j = x**m`` modulo ``p``,
    ``deg s_i < deg F_i``."""
    K = GF(p)
    if len(F) == 2:
        a, b = (densebasic.dup_strip([c % p for c in poly], K) for poly in F)
        s, t, _ = euclidtools.dup_gcdex(b, a, K)
        shift = [1] + [0] * m
        s = densearith.dup_mul(s, shift, K)
        t = densearith.dup_mul(t, shift, K)
        q, s = densearith.dup_div(s, a, K)
        t = densearith.dup_add(t, densearith.dup_mul(q, b, K), K)
        return [_to_int_poly(s, p), _to_int_poly(t, p)]
    G = [list(F[-1])]
    for f in reversed(F[1:-1]):
        G.insert(0, densearith.dup_mul(f, G[0], ZZ))
    S: list[list[int]] = []
    T: list[list[int]] = [[1]]
    for f, g in zip(F, G):
        t, s = _diophantine_univariate([g, f], T[-1], p)
        T.append(t)
        S.append(s)
    result: list[list[int]] = []
    for s, f in zip(S + [T[-1]], F):
        sp = densebasic.dup_strip([c % p for c in s], K)
        fp = densebasic.dup_strip([c % p for c in f], K)
        r = densearith.dup_rem(densearith.dup_mul(sp, [1] + [0] * m, K), fp, K)
        result.append(_to_int_poly(r, p))
    return result


def _to_int_poly(f: Sequence[int], p: int) -> list[int]:
    return densebasic.dup_strip([_symmetric(c, p) for c in f], ZZ)


def _diophantine_univariate(F: Sequence[list[int]], c: Sequence[int], p: int) -> list[list[int]]:
    """``dmp_zz_diophantine`` with no evaluation point (dense univariate)."""
    S: list[list[int]] = [[] for _ in F]
    n = len(c) - 1
    for i, coeff in enumerate(c):
        if not coeff:
            continue
        T = _dup_diophantine(F, n - i, p)
        for j, (s, t) in enumerate(zip(S, T)):
            t = densearith.dup_mul_ground(t, coeff, ZZ)
            S[j] = _to_int_poly(densearith.dup_add(s, t, ZZ), p)
    return S


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_diophantine
def _diophantine(F: Sequence[Poly], c: Poly, A: Sequence[int], d: int, p: int) -> list[Poly]:
    """``S_i`` with ``sum S_i prod_{j != i} F_j = c`` modulo ``p`` and the
    ideal ``((x_k - A_k)**(d + 1))``, the polynomials in ``x0, ...,
    x_len(A)`` of the full ring."""
    ring = c.ring
    if not A:
        dense = _diophantine_univariate([_to_dense(f) for f in F], _to_dense(c), p)
        return [_from_dense(s, ring) for s in dense]
    n = len(A)
    e = _expand(F, ring)
    a, rest = A[-1], A[:-1]
    B: list[Poly] = []
    G: list[Poly] = []
    for f in F:
        quotient = _exquo(e, f)
        if quotient is None:
            raise ArithmeticError('inexact division')
        B.append(quotient)
        G.append(_eval_var(f, n, a))
    C = _eval_var(c, n, a)
    S = _diophantine(G, C, rest, d, p)
    for s, b in zip(S, B):
        c = c - s * b
    c = _trunc(c, p)
    m = _variable_minus(ring, n, a)
    M = ring.one
    for k in range(d):
        if not c.terms:
            break
        M = M * m
        C = _diff_eval(c, k + 1, a, n)
        if C.terms:
            C = _quo_ground(C, factorial(k + 1))
            T = [t * M for t in _diophantine(G, C, rest, d, p)]
            S = [s + t for s, t in zip(S, T)]
            for t, b in zip(T, B):
                c = c - t * b
            c = _trunc(c, p)
    return [_trunc(s, p) for s in S]


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_wang_hensel_lifting
def _hensel_lifting(f: Poly, H0: Sequence[list[int]], LC: Sequence[Poly], A: Sequence[int], p: int) -> list[Poly]:
    ring = f.ring
    n = len(A)
    S = [f]
    for i, a in enumerate(reversed(A[1:])):
        S.insert(0, _trunc(_eval_var(S[0], n - i, a), p))
    d = max(_degree(f, i) for i in range(1, ring.ngens))
    H = [_from_dense(h, ring) for h in H0]
    for j, s, a in zip(range(2, n + 2), S, A):
        G = list(H)
        w = j - 1
        I, J = A[:j - 2], A[j - 1:]
        for i, (h, lc) in enumerate(zip(H, LC)):
            lead = _trunc(_eval_tail(lc, J), p)
            top = _degree(h, 0)
            shifted = lead * ring.from_dict({tuple(top if k == 0 else 0 for k in range(ring.ngens)): 1})
            H[i] = shifted + _without_leading(h)
        m = _variable_minus(ring, w, a)
        M = ring.one
        c = s - _expand(H, ring)
        dj = _degree(s, w)
        for k in range(dj):
            if not c.terms:
                break
            M = M * m
            C = _diff_eval(c, k + 1, a, w)
            if C.terms:
                C = _quo_ground(C, factorial(k + 1))
                T = _diophantine(G, C, I, d, p)
                H = [_trunc(h + t * M, p) for h, t in zip(H, T)]
                c = _trunc(s - _expand(H, ring), p)
    if _expand(H, ring) != f:
        raise ExtraneousFactors('the lifted factors do not multiply to f')
    return H


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_wang
def zz_wang(f: Poly, mod: Optional[int] = None, seed: int = 0) -> list[Poly]:
    """The irreducible factors of a square-free primitive polynomial over
    ``ZZ`` of positive degree in ``x0`` with at least two variables."""
    ring = f.ring
    rng = random.Random(seed)
    u = ring.ngens - 1
    inner = PolyRing(ZZ, ring.names[1:])
    ct, T_inner = zz_factor(_drop_first(_lc_first(f), inner))
    T = [_add_first(t, ring) for t, _ in T_inner]
    b = _mignotte_bound(f)
    p = nextprime(b)
    if mod is None:
        mod = 2 if u == 1 else 1
    history: set[tuple[int, ...]] = set()
    configs: list[tuple[list[int], int, list[int], list[list[int]], list[int]]] = []
    A = [0] * u
    r: Optional[int] = None
    try:
        cs, s, E = _test_points(f, T, ct, A)
        _, H = factortools.dup_zz_factor_sqf(s, ZZ)
        r = len(H)
        if r == 1:
            return [f]
        configs = [(s, cs, E, H, A)]
    except EvaluationFailed:
        pass
    while len(configs) < EEZ_NUMBER_OF_CONFIGS:
        for _ in range(EEZ_NUMBER_OF_TRIES):
            A = [rng.randint(-mod, mod) for _ in range(u)]
            if tuple(A) in history:
                continue
            history.add(tuple(A))
            try:
                cs, s, E = _test_points(f, T, ct, A)
            except EvaluationFailed:
                continue
            _, H = factortools.dup_zz_factor_sqf(s, ZZ)
            rr = len(H)
            if r is not None:
                if rr != r:
                    if rr < r:
                        configs, r = [], rr
                    else:
                        continue
            else:
                r = rr
            if r == 1:
                return [f]
            configs.append((s, cs, E, H, A))
            if len(configs) == EEZ_NUMBER_OF_CONFIGS:
                break
        else:
            mod += EEZ_MODULUS_STEP
    s_norm: Optional[int] = None
    s_arg = 0
    for i, (s, _, _, _, _) in enumerate(configs):
        norm = max(abs(c) for c in s)
        if s_norm is None or norm < s_norm:
            s_norm, s_arg = norm, i
    _, cs, E, H, A = configs[s_arg]
    try:
        g, HH, LC = _lead_coeffs(f, T, cs, E, H, A)
        factors = _hensel_lifting(g, HH, LC, A, p)
    except ExtraneousFactors:
        return zz_wang(f, mod + 1, seed + 1)
    result: list[Poly] = []
    for h in factors:
        _, h = _primitive(h)
        if h.LC < 0:
            h = -h
        result.append(h)
    return result


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_zz_factor
def zz_factor(f: Poly) -> tuple[int, Factors]:
    """The content (with the sign of the leading coefficient) and the
    irreducible factors with their multiplicities of a polynomial over
    ``ZZ``, sorted as SymPy sorts them."""
    ring = f.ring
    if ring.ngens == 1:
        coefficient, dense = factortools.dup_zz_factor(_to_dense(f), ZZ)
        return coefficient, [(_from_dense(g, ring), k) for g, k in dense]
    if not f.terms:
        return 0, []
    cont, g = _primitive(f)
    if g.LC < 0:
        cont, g = -cont, -g
    if all(_degree(g, i) <= 0 for i in range(ring.ngens)):
        return cont, []
    R = IntegerPolynomialRing(ring.names)
    inner_ring = PolyRing(ZZ, ring.names[1:])
    content_first = _x0_content(g, R)
    if not R.is_one(content_first):
        g = g.exquo(content_first)
    factors: Factors = []
    if _degree(g, 0) > 0:
        g = sqf_part(g)
        factors = trial_division(f, zz_wang(g))
    for h, k in zz_factor(_drop_first(content_first, inner_ring))[1]:
        factors.insert(0, (_add_first(h, ring), k))
    return cont, _sorted(factors)


# PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_content (over ZZ)
def _x0_content(f: Poly, R: IntegerPolynomialRing) -> Poly:
    """The gcd of the coefficients in ``x0`` (polynomials in the other
    variables, the sign of the ground leading coefficient positive)."""
    coefficients: dict[int, dict[Monomial, int]] = {}
    for m, c in f.terms.items():
        coefficients.setdefault(m[0], {})[(0,) + m[1:]] = c
    ordered = [f.ring.from_dict(coefficients[d]) for d in sorted(coefficients, reverse=True)]
    content = ordered[0]
    for coefficient in ordered[1:]:
        content = R.gcd(content, coefficient)
        if R.is_one(content):
            break
    if content.terms and content.LC < 0:
        content = -content
    return content
