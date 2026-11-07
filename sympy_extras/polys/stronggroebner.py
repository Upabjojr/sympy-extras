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


from .domains.integerring import ZZ
from .rings import Monomial, PolyElement, PolyRing, monomial_div, monomial_divides, monomial_lcm

__all__ = ['strong_groebner', 'strong_reduce', 'is_strong_groebner', 's_polynomial_zz', 'g_polynomial_zz']


def _check_ring(ring: PolyRing[int]) -> None:
    if ring.domain != ZZ:
        raise ValueError("strong Gröbner bases are computed over ZZ, the ring is over %s" % ring.domain)


def _leading(f: PolyElement[int]) -> tuple[Monomial, int]:
    return f.LT


def strong_reduce(f: PolyElement[int], basis: Sequence[PolyElement[int]]) -> PolyElement[int]:
    """The normal form of ``f`` modulo the basis over ``ZZ``: every term
    ``c t`` with ``lm(g) | t`` for some ``g`` of the basis is replaced by
    its remainder ``(c mod |lc(g)|) t``, until no term can be reduced. When
    the basis is a strong Gröbner basis, the result is zero exactly for
    the elements of the ideal.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.stronggroebner import strong_reduce
    >>> R = PolyRing(ZZ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> basis = [x.mul_ground(2), y.mul_ground(3), x*y]
    >>> print(strong_reduce((x*y).mul_ground(7) + x.mul_ground(5), basis))
    x
    >>> print(strong_reduce(x.mul_ground(6) + y.mul_ground(3), basis))
    0
    """
    ring = f.ring
    remainder = ring.zero
    current = f
    while current.terms:
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
            current = current - g.mul_monom(multiplier).mul_ground(quotient)
            reduced = True
            break
        if not reduced:
            term = ring.term(monomial, coefficient)
            remainder = remainder + term
            current = current - term
    return remainder


def _euclidean_quotient(c: int, d: int) -> int:
    """``q`` with ``c = q d + r`` and ``0 <= r < |d|``."""
    return c // d if d > 0 else -(c // (-d))


def s_polynomial_zz(f: PolyElement[int], g: PolyElement[int]) -> PolyElement[int]:
    """The S-polynomial over ``ZZ``: the leading terms are brought to
    their least common multiple, coefficients included, and subtracted.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.stronggroebner import s_polynomial_zz
    >>> R = PolyRing(ZZ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> print(s_polynomial_zz((x*x).mul_ground(2) + y, (x*y).mul_ground(3) + R.one))
    -2*x + 3*y**2
    """
    (m, a), (n, b) = _leading(f), _leading(g)
    lcm_monomial = monomial_lcm(m, n)
    lcm_coefficient = abs(a*b)//_gcd(a, b)
    left = f.mul_monom(_quotient(lcm_monomial, m)).mul_ground(lcm_coefficient//a)
    right = g.mul_monom(_quotient(lcm_monomial, n)).mul_ground(lcm_coefficient//b)
    return left - right


def g_polynomial_zz(f: PolyElement[int], g: PolyElement[int]) -> Optional[PolyElement[int]]:
    """The G-polynomial over ``ZZ``: ``u (L/m) f + v (L/n) g`` with
    ``u a + v b = gcd(a, b)`` for the leading terms ``a m`` and ``b n``, so
    that its leading coefficient is the gcd; ``None`` when one leading
    coefficient divides the other (the G-polynomial is then a multiple of
    one of the two).

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.stronggroebner import g_polynomial_zz
    >>> R = PolyRing(ZZ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> print(g_polynomial_zz(x.mul_ground(2), y.mul_ground(3)))
    x*y
    >>> g_polynomial_zz(x.mul_ground(2), y.mul_ground(4)) is None
    True
    """
    (m, a), (n, b) = _leading(f), _leading(g)
    if a % b == 0 or b % a == 0:
        return None
    u, v, _ = _gcdex(a, b)
    lcm_monomial = monomial_lcm(m, n)
    left = f.mul_monom(_quotient(lcm_monomial, m)).mul_ground(u)
    right = g.mul_monom(_quotient(lcm_monomial, n)).mul_ground(v)
    return left + right


def _quotient(lcm_monomial: Monomial, monomial: Monomial) -> Monomial:
    return monomial_div(lcm_monomial, monomial)


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


def strong_groebner(polys: Sequence[Union[PolyElement[int], int]], ring: PolyRing[int]) -> list[PolyElement[int]]:
    """The reduced strong Gröbner basis over ``ZZ`` of the ideal generated
    by ``polys`` in ``ring`` (whose domain must be ``ZZ``), sorted by
    decreasing leading monomial; every element of the ideal, and only
    those, reduces to zero by it (:func:`strong_reduce`).

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.stronggroebner import strong_groebner, strong_reduce
    >>> R = PolyRing(ZZ, ['x', 'y'], lex)
    >>> X, Y = R.gens
    >>> [str(g) for g in strong_groebner([X.mul_ground(2), Y.mul_ground(3)], R)]
    ['x*y', '2*x', '3*y']
    >>> [str(g) for g in strong_groebner([X.mul_ground(2), X.mul_ground(3)], R)]
    ['x']
    >>> [str(g) for g in strong_groebner([4, X.mul_ground(2) + R.one], R)]
    ['1']
    >>> f, g = (X*X).mul_ground(2) + Y, (X*Y).mul_ground(3) + R.one
    >>> G = strong_groebner([f, g], R)
    >>> [str(h) for h in G]
    ['x*y + 3*y**3 + 1', '2*x - 3*y**2', '9*y**3 + 2']
    >>> str(strong_reduce(X*f - Y*g, G)), str(strong_reduce(X, G))
    ('0', 'x')

    SymPy's ``groebner`` with ``domain=ZZ`` returns ``[x, y]`` for the
    first ideal, its basis over ``QQ``, which is wrong over ``ZZ``.
    """
    _check_ring(ring)
    basis: list[PolyElement[int]] = []
    for f in polys:
        element = f if isinstance(f, PolyElement) else ring.from_int(f)
        if element.terms:
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
            if remainder.terms:
                remainder = _positive(remainder)
                pairs.extend((len(basis), k) for k in range(len(basis)))
                basis.append(remainder)
    return _reduced(basis)


def _positive(f: PolyElement[int]) -> PolyElement[int]:
    return -f if _leading(f)[1] < 0 else f


def _reduced(basis: list[PolyElement[int]]) -> list[PolyElement[int]]:
    """The reduced strong Gröbner basis of a strong Gröbner basis."""
    if not basis:
        return []
    key = basis[0].ring.order.key
    # minimal: drop the elements whose leading term another one divides
    kept: list[PolyElement[int]] = []
    ordered = sorted((f for f in basis if f.terms), key=lambda f: (key(f.LM), abs(_leading(f)[1])))
    for f in ordered:
        monomial, coefficient = _leading(f)
        if any(monomial_divides(_leading(g)[0], monomial) and coefficient % _leading(g)[1] == 0 for g in kept):
            continue
        kept.append(f)
    # reduced: every element reduced by the others (its leading term is
    # not reducible, so that it stays)
    result: list[PolyElement[int]] = []
    for i, f in enumerate(kept):
        others = kept[:i] + kept[i + 1:]
        result.append(_positive(strong_reduce(f, others)))
    return sorted(result, key=lambda f: key(f.LM), reverse=True)


def is_strong_groebner(basis: Sequence[PolyElement[int]]) -> bool:
    """Whether the polynomials, over ``ZZ``, form a strong Gröbner basis
    of the ideal they generate: every S-polynomial and G-polynomial of two
    of them reduces to zero.

    Examples
    ========

    >>> from sympy_extras.polys.domains import ZZ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.stronggroebner import is_strong_groebner
    >>> R = PolyRing(ZZ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> two_x, three_y = x.mul_ground(2), y.mul_ground(3)
    >>> is_strong_groebner([two_x, three_y]), is_strong_groebner([two_x, three_y, x*y])
    (False, True)
    """
    polys = [f for f in basis if f.terms]
    for i, f in enumerate(polys):
        for g in polys[:i]:
            if strong_reduce(s_polynomial_zz(f, g), polys).terms:
                return False
            G = g_polynomial_zz(f, g)
            if G is not None and strong_reduce(G, polys).terms:
                return False
    return True
