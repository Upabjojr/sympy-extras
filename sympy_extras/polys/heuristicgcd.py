"""The heuristic gcd of multivariate polynomials over ``ZZ`` (Char,
Geddes and Gonnet): the first variable evaluated at a large integer, the
gcd of the images computed recursively, the gcd interpolated back from
its balanced digits in that integer and checked by division.

Ported from SymPy's ``sympy.polys.heuristicgcd`` (and the ``deflate`` of
``sympy.polys.rings``), on the typed :class:`~sympy_extras.polys.rings.
PolyElement` over ``ZZ``.

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.polys.domains import ZZ
>>> from sympy_extras.polys.rings import PolyRing
>>> x, y = Symbol('x'), Symbol('y')
>>> R = PolyRing(ZZ, ['x', 'y'])
>>> f, g = R.from_expr((x + y)*(2*x - y)), R.from_expr((x + y)*(x + 3))
>>> h, cff, cfg = heugcd(f, g)
>>> h.to_expr(), cff.to_expr(), cfg.to_expr()
(x + y, 2*x - y, x + 3)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/heuristicgcd.py (heugcd, _gcd_interpolate),
# sympy/polys/rings.py (PolyElement.deflate, inflate, extract_ground) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; the evaluation drops the variable into an explicit
# smaller ring; failures raise HeuristicGCDFailed as in SymPy.
# ---------------------------------------------------------------------
from __future__ import annotations

from math import gcd, isqrt
from typing import Optional, Sequence

from .domains.integerring import ZZ
from .polyerrors import HeuristicGCDFailed
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['heugcd', 'deflate', 'inflate']

HEU_GCD_MAX = 6

Cofactors = tuple[PolyElement[int], PolyElement[int], PolyElement[int]]


def _max_norm(f: PolyElement[int]) -> int:
    return max((abs(c) for c in f.terms.values()), default=0)


def _content(f: PolyElement[int]) -> int:
    result = 0
    for c in f.terms.values():
        result = gcd(result, c)
    return result


def _quo_ground(f: PolyElement[int], c: int) -> PolyElement[int]:
    return f.ring.from_dict({m: v // c for m, v in f.terms.items()})


def _divides(f: PolyElement[int], h: PolyElement[int]) -> Optional[PolyElement[int]]:
    """``f / h`` when ``h`` divides ``f`` exactly over ``ZZ``, else ``None``."""
    if not h.terms:
        return None
    try:
        quotient, remainder = f.div([h])
    except ValueError:
        return None
    return None if remainder.terms else quotient[0]


def _powers(x: int, d: int) -> list[int]:
    """``[1, x, ..., x**d]``, by successive products."""
    result = [1]
    for _ in range(d):
        result.append(result[-1] * x)
    return result


def _evaluate_first(f: PolyElement[int], x: int, rest: PolyRing[int]) -> PolyElement[int]:
    powers = _powers(x, max(m[0] for m in f.terms))
    terms: dict[Monomial, int] = {}
    for m, c in f.terms.items():
        key = m[1:]
        terms[key] = terms.get(key, 0) + c * powers[m[0]]
    return rest.from_dict(terms)


# PORTED FROM SYMPY: sympy/polys/densetools.py:dup_eval
def _horner(f: PolyElement[int], x: int) -> int:
    """The value of a univariate polynomial at ``x`` (Horner's scheme)."""
    dense = [0] * (max(m[0] for m in f.terms) + 1)
    for m, c in f.terms.items():
        dense[m[0]] = c
    result = 0
    for c in reversed(dense):
        result = result * x + c
    return result


# PORTED FROM SYMPY: sympy/polys/heuristicgcd.py:_gcd_interpolate
def _interpolate(h: PolyElement[int], x: int, ring: PolyRing[int]) -> PolyElement[int]:
    """The polynomial in ``ring`` whose coefficients of the first variable
    are the balanced digits of ``h`` (a polynomial in the other ones) in
    base ``x``, made positive."""
    terms: dict[Monomial, int] = {}
    i = 0
    while h.terms:
        digit = h.ring.from_dict({m: _balanced(c, x) for m, c in h.terms.items()})
        h = _quo_ground(h - digit, x)
        for m, c in digit.terms.items():
            terms[(i,) + m] = c
        i += 1
    f = ring.from_dict(terms)
    return -f if f.terms and f.LC < 0 else f


def _balanced(c: int, x: int) -> int:
    r = c % x
    return r - x if r > x // 2 else r


def _interpolate_integer(h: int, x: int, ring: PolyRing[int]) -> PolyElement[int]:
    terms: dict[Monomial, int] = {}
    i = 0
    while h:
        g = _balanced(h, x)
        h = (h - g) // x
        if g:
            terms[(i,)] = g
        i += 1
    f = ring.from_dict(terms)
    return -f if f.terms and f.LC < 0 else f


def _primitive_part(f: PolyElement[int]) -> PolyElement[int]:
    c = _content(f)
    return f if c in (0, 1) else _quo_ground(f, c)


# PORTED FROM SYMPY: sympy/polys/heuristicgcd.py:heugcd
def heugcd(f: PolyElement[int], g: PolyElement[int]) -> Cofactors:
    """``(h, f/h, g/h)`` with ``h`` the gcd of two nonzero polynomials over
    ``ZZ`` (positive leading coefficient); ``HeuristicGCDFailed`` when the
    evaluation points tried give no answer."""
    ring = f.ring
    common = gcd(_content(f), _content(g))
    if common > 1:
        f, g = _quo_ground(f, common), _quo_ground(g, common)
    f_norm, g_norm = _max_norm(f), _max_norm(g)
    B = 2 * min(f_norm, g_norm) + 29
    x = max(min(B, 99 * isqrt(B)), 2 * min(f_norm // abs(f.LC), g_norm // abs(g.LC)) + 4)
    rest = PolyRing(ZZ, ring.names[1:]) if ring.ngens > 1 else None
    for _ in range(HEU_GCD_MAX):
        if rest is None:
            ff = _horner(f, x)
            gg = _horner(g, x)
            if ff and gg:
                hi = gcd(ff, gg)
                h = _primitive_part(_interpolate_integer(hi, x, ring))
                found = _check(f, g, h, common)
                if found is not None:
                    return found
                cff = _interpolate_integer(ff // hi, x, ring)
                found = _check_cofactor(f, g, cff, common, first=True)
                if found is not None:
                    return found
                cfg = _interpolate_integer(gg // hi, x, ring)
                found = _check_cofactor(g, f, cfg, common, first=False)
                if found is not None:
                    return found
        else:
            fe, ge = _evaluate_first(f, x, rest), _evaluate_first(g, x, rest)
            if fe.terms and ge.terms:
                he, cffe, cfge = heugcd(fe, ge)
                h = _primitive_part(_interpolate(he, x, ring))
                found = _check(f, g, h, common)
                if found is not None:
                    return found
                found = _check_cofactor(f, g, _interpolate(cffe, x, ring), common, first=True)
                if found is not None:
                    return found
                found = _check_cofactor(g, f, _interpolate(cfge, x, ring), common, first=False)
                if found is not None:
                    return found
        x = 73794 * x * isqrt(isqrt(x)) // 27011
    raise HeuristicGCDFailed('no luck')


def _check(f: PolyElement[int], g: PolyElement[int], h: PolyElement[int], common: int) -> Optional[Cofactors]:
    cff = _divides(f, h)
    if cff is None:
        return None
    cfg = _divides(g, h)
    if cfg is None:
        return None
    return (h.mul_ground(common), cff, cfg)


def _check_cofactor(a: PolyElement[int], b: PolyElement[int], cofactor: PolyElement[int], common: int,
                    first: bool) -> Optional[Cofactors]:
    """``h = a / cofactor`` checked against ``b``; the cofactors in the
    order ``(f, g)``."""
    h = _divides(a, cofactor)
    if h is None:
        return None
    other = _divides(b, h)
    if other is None:
        return None
    if first:
        return (h.mul_ground(common), cofactor, other)
    return (h.mul_ground(common), other, cofactor)


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.deflate
def deflate(polys: Sequence[PolyElement[int]]) -> tuple[Monomial, list[PolyElement[int]]]:
    """The gcd ``J`` of the exponents of each variable and the polynomials
    in ``x_i**J_i``."""
    n = polys[0].ring.ngens
    J = [0] * n
    for p in polys:
        for m in p.terms:
            for i, e in enumerate(m):
                J[i] = gcd(J[i], e)
    steps = tuple(j if j else 1 for j in J)
    if all(j == 1 for j in steps):
        return steps, list(polys)
    return steps, [p.ring.from_dict({tuple(e // j for e, j in zip(m, steps)): c for m, c in p.terms.items()})
                   for p in polys]


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.inflate
def inflate(f: PolyElement[int], steps: Monomial) -> PolyElement[int]:
    return f.ring.from_dict({tuple(e * j for e, j in zip(m, steps)): c for m, c in f.terms.items()})
