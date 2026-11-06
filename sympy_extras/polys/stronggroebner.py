r"""Strong Gröbner bases over the integers.

A Gröbner basis over a field decides ideal membership because every
leading term in the ideal is a multiple of the leading term of a basis
element. Over `\mathbb{Z}` the coefficients matter: `\langle 2x, 3y
\rangle` contains `xy = 3 \cdot xy - 2 \cdot xy` but neither `x` nor `y`,
and SymPy's ``groebner`` with ``domain=ZZ`` returns ``[x, y]``, the basis
over the rationals, which is wrong over the integers. A *strong* Gröbner
basis `G` of an ideal `I \subseteq \mathbb{Z}[x]` (Kandri-Rody and Kapur
[1]_; the *D-Gröbner bases* of Becker and Weispfenning [2]_, chapter 10;
Lichtblau [3]_) is a finite subset of `I` such that for every nonzero
`f \in I` some `g \in G` has a leading term dividing that of `f`, the
coefficient included: `\mathrm{lc}(g) \mid \mathrm{lc}(f)` and
`\mathrm{lm}(g) \mid \mathrm{lm}(f)`. Then every `f \in I` reduces to
zero by `G`, and only those.

The algorithm is Buchberger's with two kinds of polynomials for a pair
`f, g` with leading terms `a\, m` and `b\, n`, `L = \mathrm{lcm}(m, n)`:
the **S-polynomial** `\frac{c}{a} \frac{L}{m} f - \frac{c}{b} \frac{L}{n}
g` with `c = \mathrm{lcm}(a, b)`, whose leading terms cancel, and the
**G-polynomial** `u \frac{L}{m} f + v \frac{L}{n} g` with
`u a + v b = \gcd(a, b)`, whose leading coefficient is the gcd (it is
only needed when neither of `a`, `b` divides the other). Both are reduced
by the current basis, with Euclidean division of the coefficients
(`c = q\, d + r`, `0 \le r < |d|`): a term `c\, t` is reduced by `g` when
`\mathrm{lm}(g) \mid t` and `|c| \ge |\mathrm{lc}(g)|`, leaving the
remainder `r\, t`. The nonzero remainders join the basis; the loop ends
when every S- and G-polynomial reduces to zero. The basis is then made
*reduced*: positive leading coefficients, no leading term dividing
another, every term of every element reduced by the others; the reduced
strong Gröbner basis is unique for the given order and remainder
convention.

References
==========

.. [1] A. Kandri-Rody, D. Kapur, Computing a Gröbner basis of a polynomial
   ideal over a Euclidean domain, Journal of Symbolic Computation 6 (1988)
   37–57.
.. [2] T. Becker, V. Weispfenning, Gröbner Bases, Springer (1993),
   chapter 10.
.. [3] D. Lichtblau, Effective computation of strong Gröbner bases over
   Euclidean domains, Illinois Journal of Mathematics 56 (2012) 177–194.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.polys.domains import ZZ
from sympy.polys.monomials import monomial_div, monomial_divides, monomial_lcm
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._typing import Monomial

__all__ = ['strong_groebner', 'strong_reduce', 'is_strong_groebner', 's_polynomial_zz', 'g_polynomial_zz']


def _check_ring(ring: PolyRing) -> None:
    if ring.domain != ZZ:
        raise ValueError("strong Gröbner bases are computed over ZZ, the ring is over %s" % ring.domain)


def _leading(f: PolyElement) -> tuple[Monomial, int]:
    monomial, coefficient = f.LT
    return tuple(monomial), int(coefficient)


def strong_reduce(f: PolyElement, basis: Sequence[PolyElement]) -> PolyElement:
    """The normal form of ``f`` modulo the basis over ``ZZ``: every term
    ``c t`` with ``lm(g) | t`` for some ``g`` of the basis is replaced by
    its remainder ``(c mod |lc(g)|) t``, until no term can be reduced. When
    the basis is a strong Gröbner basis, the result is zero exactly for
    the elements of the ideal.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.stronggroebner import strong_reduce
    >>> R, x, y = ring('x,y', ZZ, 'lex')
    >>> strong_reduce(7*x*y + 5*x, [2*x, 3*y, x*y])
    x
    >>> strong_reduce(6*x + 3*y, [2*x, 3*y, x*y])
    0
    """
    ring = f.ring
    remainder = ring.zero
    current = f
    while current:
        monomial, coefficient = _leading(current)
        reduced = False
        for g in basis:
            if not g:
                continue
            g_monomial, g_coefficient = _leading(g)
            if not monomial_divides(g_monomial, monomial):
                continue
            quotient = _euclidean_quotient(coefficient, g_coefficient)
            if quotient == 0:
                continue
            multiplier = monomial_div(monomial, g_monomial)
            if multiplier is None:
                continue
            current = current - g.mul_monom(multiplier).mul_ground(ring.domain(quotient))
            reduced = True
            break
        if not reduced:
            term = ring.term_new(monomial, ring.domain(coefficient))
            remainder = remainder + term
            current = current - term
    return remainder


def _euclidean_quotient(c: int, d: int) -> int:
    """``q`` with ``c = q d + r`` and ``0 <= r < |d|``."""
    return c // d if d > 0 else -(c // (-d))


def s_polynomial_zz(f: PolyElement, g: PolyElement) -> PolyElement:
    """The S-polynomial over ``ZZ``: the leading terms are brought to
    their least common multiple, coefficients included, and subtracted.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.stronggroebner import s_polynomial_zz
    >>> R, x, y = ring('x,y', ZZ, 'lex')
    >>> s_polynomial_zz(2*x**2 + y, 3*x*y + 1)
    -2*x + 3*y**2
    """
    (m, a), (n, b) = _leading(f), _leading(g)
    lcm_monomial = monomial_lcm(m, n)
    lcm_coefficient = abs(a*b)//_gcd(a, b)
    ring = f.ring
    left = f.mul_monom(_quotient(lcm_monomial, m)).mul_ground(ring.domain(lcm_coefficient//a))
    right = g.mul_monom(_quotient(lcm_monomial, n)).mul_ground(ring.domain(lcm_coefficient//b))
    return left - right


def g_polynomial_zz(f: PolyElement, g: PolyElement) -> Optional[PolyElement]:
    """The G-polynomial over ``ZZ``: ``u (L/m) f + v (L/n) g`` with
    ``u a + v b = gcd(a, b)`` for the leading terms ``a m`` and ``b n``, so
    that its leading coefficient is the gcd; ``None`` when one leading
    coefficient divides the other (the G-polynomial is then a multiple of
    one of the two).

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.stronggroebner import g_polynomial_zz
    >>> R, x, y = ring('x,y', ZZ, 'lex')
    >>> g_polynomial_zz(2*x, 3*y)
    x*y
    >>> g_polynomial_zz(2*x, 4*y) is None
    True
    """
    (m, a), (n, b) = _leading(f), _leading(g)
    if a % b == 0 or b % a == 0:
        return None
    u, v, _ = _gcdex(a, b)
    lcm_monomial = monomial_lcm(m, n)
    ring = f.ring
    left = f.mul_monom(_quotient(lcm_monomial, m)).mul_ground(ring.domain(u))
    right = g.mul_monom(_quotient(lcm_monomial, n)).mul_ground(ring.domain(v))
    return left + right


def _quotient(lcm_monomial: Monomial, monomial: Monomial) -> Monomial:
    quotient = monomial_div(lcm_monomial, monomial)
    if quotient is None:
        raise ValueError("inconsistent monomials")
    return tuple(quotient)


def _gcd(a: int, b: int) -> int:
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def _gcdex(a: int, b: int) -> tuple[int, int, int]:
    """``(u, v, d)`` with ``u a + v b = d = gcd(a, b) > 0``."""
    old_r, r = a, b
    old_u, u = 1, 0
    old_v, v = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q*r
        old_u, u = u, old_u - q*u
        old_v, v = v, old_v - q*v
    if old_r < 0:
        old_r, old_u, old_v = -old_r, -old_u, -old_v
    return old_u, old_v, old_r


def strong_groebner(polys: Sequence[Union[PolyElement, int]], ring: PolyRing) -> list[PolyElement]:
    """The reduced strong Gröbner basis over ``ZZ`` of the ideal generated
    by ``polys`` in ``ring`` (whose domain must be ``ZZ``), sorted by
    decreasing leading monomial; every element of the ideal, and only
    those, reduces to zero by it (:func:`strong_reduce`).

    Examples
    ========

    >>> from sympy import ZZ, groebner
    >>> from sympy.abc import x, y
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.stronggroebner import strong_groebner, strong_reduce
    >>> R, X, Y = ring('x,y', ZZ, 'lex')
    >>> strong_groebner([2*X, 3*Y], R)
    [x*y, 2*x, 3*y]
    >>> groebner([2*x, 3*y], x, y, domain=ZZ)        # SymPy: the basis over QQ, wrong over ZZ
    GroebnerBasis([x, y], x, y, domain='ZZ', order='lex')
    >>> strong_groebner([2*X, 3*X], R), strong_groebner([4, 2*X + 1], R)
    ([x], [1])
    >>> G = strong_groebner([2*X**2 + Y, 3*X*Y + 1], R)
    >>> G
    [x*y + 3*y**3 + 1, 2*x - 3*y**2, 9*y**3 + 2]
    >>> strong_reduce(X*(2*X**2 + Y) - Y*(3*X*Y + 1), G), strong_reduce(X, G)
    (0, x)
    """
    _check_ring(ring)
    basis: list[PolyElement] = []
    for f in polys:
        element = f if isinstance(f, PolyElement) else ring(f)
        if element:
            basis.append(_positive(element))
    pairs: list[tuple[int, int]] = [(i, j) for i in range(len(basis)) for j in range(i)]
    while pairs:
        i, j = pairs.pop()
        f, g = basis[i], basis[j]
        candidates = [s_polynomial_zz(f, g)]
        G = g_polynomial_zz(f, g)
        if G is not None:
            candidates.append(G)
        for candidate in candidates:
            remainder = strong_reduce(candidate, basis)
            if remainder:
                remainder = _positive(remainder)
                pairs.extend((len(basis), k) for k in range(len(basis)))
                basis.append(remainder)
    return _reduced(basis)


def _positive(f: PolyElement) -> PolyElement:
    return -f if _leading(f)[1] < 0 else f


def _reduced(basis: list[PolyElement]) -> list[PolyElement]:
    """The reduced strong Gröbner basis of a strong Gröbner basis."""
    ring = basis[0].ring if basis else None
    if ring is None:
        return []
    key = ring.order
    # minimal: drop the elements whose leading term another one divides
    kept: list[PolyElement] = []
    ordered = sorted((f for f in basis if f), key=lambda f: (key(f.LM), abs(_leading(f)[1])))
    for f in ordered:
        monomial, coefficient = _leading(f)
        if any(monomial_divides(_leading(g)[0], monomial) and coefficient % _leading(g)[1] == 0 for g in kept):
            continue
        kept.append(f)
    # reduced: every element reduced by the others (its leading term is
    # not reducible, so that it stays)
    result: list[PolyElement] = []
    for i, f in enumerate(kept):
        others = kept[:i] + kept[i + 1:]
        result.append(_positive(strong_reduce(f, others)))
    return sorted(result, key=lambda f: key(f.LM), reverse=True)


def is_strong_groebner(basis: Sequence[PolyElement]) -> bool:
    """Whether the polynomials, over ``ZZ``, form a strong Gröbner basis
    of the ideal they generate: every S-polynomial and G-polynomial of two
    of them reduces to zero.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy_extras.polys.stronggroebner import is_strong_groebner
    >>> R, x, y = ring('x,y', ZZ, 'lex')
    >>> is_strong_groebner([2*x, 3*y]), is_strong_groebner([2*x, 3*y, x*y])
    (False, True)
    """
    polys = [f for f in basis if f]
    for i, f in enumerate(polys):
        for g in polys[:i]:
            if strong_reduce(s_polynomial_zz(f, g), polys):
                return False
            G = g_polynomial_zz(f, g)
            if G is not None and strong_reduce(G, polys):
                return False
    return True
