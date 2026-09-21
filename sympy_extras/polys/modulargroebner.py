r"""Reduced Gröbner bases over the rationals by the modular algorithm.

SymPy computes Gröbner bases over ``QQ`` with Buchberger's algorithm on
rational coefficients (:func:`sympy.polys.groebnertools.groebner`), where
the growth of the intermediate coefficients dominates the running time:
the basis itself may have coefficients of a few digits while the
polynomials met on the way have hundreds. :func:`modular_groebner` is a
drop-in replacement which computes the same reduced Gröbner basis from its
images modulo primes, following Arnold [1]_ (see also [2]_, [3]_, [4]_),
and *proves* the result, so that it is not a probabilistic algorithm.

The algorithm
=============

Let $F \subset \mathbb{Q}[x]$ be the input, $>$ the monomial order and
$I = \langle F \rangle$.

1. **Integer input.** Denominators and contents are removed:
   $F \subset \mathbb{Z}[x]$, primitive.

2. **Homogenisation.** Unless every element of $F$ is homogeneous, $F$
   is replaced by its homogenisation $F^h \subset \mathbb{Z}[x, t]$. In
   both cases the order used from here on is $>_h$: monomials are compared
   by total degree first and then by $>$ on the powers of $x$. It is a
   monomial order for every global $>$; on a homogeneous polynomial $f^h$
   it picks the leading monomial of $f$ times a power of $t$, and a
   homogeneous ideal has the same reduced Gröbner basis for $>$ and for
   $>_h$. Homogenising is what makes the proof below possible, and it
   is also what makes the images cheap for the orders where SymPy's
   direct computation is slow: Buchberger's algorithm with the normal
   selection strategy then works degree by degree, which for ``lex`` was
   ten (Katsura-4) and twenty-five (cyclic-5) times faster than the
   computation on the input modulo the same prime; for ``grevlex`` it was
   1.4 (Katsura-5) to 2.5 (cyclic-5) times slower.

3. **Images.** For a prime $p$ which divides no leading coefficient of
   $F^h$, the reduced Gröbner basis $G_p$ of $\langle F^h \bmod p \rangle$
   for $>_h$ is computed by SymPy's own ``groebner`` over ``GF(p)``.

4. **Unlucky primes.** $p$ is *lucky* when $G_p$ has the leading monomials
   of the reduced Gröbner basis $G$ of $\langle F^h \rangle$ over
   $\mathbb{Q}$, and then $G_p = G \bmod p$; all but finitely many primes
   are lucky (they are the primes which divide no leading coefficient of
   the strong Gröbner basis of the ideal over $\mathbb{Z}$, Pauer [3]_).
   Two primes whose leading monomials differ are compared as in
   section 5 of [1]_, by the Hilbert function first and by the leading
   monomials next. In degree $d$ the leading monomials of a homogeneous
   ideal are the pivot columns of its Macaulay matrix (the multiples of
   degree $d$ of the generators, columns sorted by decreasing monomials):
   the rank modulo $p$ is at most the rank over $\mathbb{Q}$, and a set of
   columns which is independent modulo $p$ is independent over
   $\mathbb{Q}$, so that, the ranks being equal, the pivots over
   $\mathbb{Q}$ are to the left of the pivots modulo $p$. Hence, in the
   first degree where the minimal generators of the leading monomial
   ideals of two primes differ, the prime which has fewer of them (a
   larger Hilbert function) is unlucky, and with equally many the prime
   whose list, sorted decreasingly, is the smaller at the first difference
   is unlucky. :func:`_luckiness` turns this into a sort key: only the
   primes with the greatest key seen so far are kept, a lucky prime is
   never discarded, and a prime with the key of a lucky prime is lucky.

5. **Lifting.** The coefficients of the images of the kept primes are
   combined by the Chinese remainder theorem (:func:`chinese_remainder`)
   and the rational coefficients are recovered from their residue modulo
   the product $M$ by Wang's algorithm [5]_
   (:func:`rational_reconstruction`), which finds the unique $n/d$ with
   $|n|, d \le \sqrt{M/2}$ if there is one. A reconstruction is attempted
   after every prime (they are few, see below), and a candidate goes to
   the verification when it is *stable*: either the image of the next
   prime agrees with it, or every coefficient was found within
   $\sqrt{M/2^{1 + 32}}$ (a residue which is not the image of a small
   fraction passes this with probability about $2^{-32}$; this is the
   early termination of Monagan's maximal quotient reconstruction [6]_,
   and it saves the confirming prime, which is half the work on small
   inputs).

6. **Verification** over $\mathbb{Q}$ (fraction-free, on integer
   multiples): (a) every element of $F^h$ reduces to zero modulo the
   candidate $G$; (b) $G$ is a Gröbner basis: the S-polynomials of the
   critical pairs left by the criteria of Gebauer and Möller [7]_ reduce
   to zero. If one of them fails, more primes are used.

7. **The basis of the input.** Setting $t = 1$ in a Gröbner basis of
   $\langle F^h \rangle$ for $>_h$ gives a Gröbner basis of $I$ for $>$
   (if $f = \sum a_i f_i$ then $t^k f^h \in \langle F^h \rangle$ for some
   $k$, its leading monomial is that of $f$ times a power of $t$, and it
   is divisible by the leading monomial of an element of the basis). The
   redundant elements are removed and the tails reduced, which gives the
   reduced basis, the one SymPy returns.

What is proven
==============

Theorem 7.1 of [1]_: let $F$ be *homogeneous*, $G_p$ the reduced Gröbner
basis of $\langle F \bmod p \rangle$ and $G \subset \mathbb{Q}[x]$
homogeneous with the leading monomials of $G_p$. If (a) and (b) hold,
$G$ is a Gröbner basis of $I$. Proof: write $H_J(d)$ for the dimension of
the part of degree $d$ of the quotient by $J$. The part of degree $d$ of
$I$ is spanned by the multiples of degree $d$ of $F$, a matrix of
integers, whose rank does not increase modulo $p$: $H_I \le H_{I_p}$ (no
condition on $p$ is needed). By (a), $I \subseteq \langle G \rangle$, so
$H_{\langle G \rangle} \le H_I$. By (b), $H_{\langle G \rangle}$ is the
Hilbert function of the leading monomials of $G$, which are those of
$G_p$: $H_{\langle G \rangle} = H_{I_p}$. The three are equal, and
$I = \langle G \rangle$ degree by degree.

**The theorem is false for non-homogeneous input**, where (a) and (b) only
show that $G$ is a Gröbner basis of an ideal *containing* $I$. For
$F = \{x (z + p y + 1),\ x (z + 2 p y + 2)\}$, whose leading coefficients
are 1, the ideal modulo $p$ is $\langle x \rangle$ and $G = \{x\}$ passes
(a), (b) and the comparison of the leading monomials, while
$I = \langle x z, x (p y + 1) \rangle$: the component $p y + 1 = 0$ goes
to infinity modulo $p$. With the product of finitely many primes in the
place of $p$ this defeats any fixed list of primes, and no further prime
ever contradicts $\{x\}$ unless it is used. Proving
$G \subseteq I$ directly needs the cofactors of $G$ in terms of $F$, which
are large; instead, as Arnold proposes, the whole computation is done on
the homogenisation, where the theorem applies, and step 7 brings the
proven basis back. (On this example, the images of $F^h$ modulo the bad
primes have the leading monomials $x z, x t$, the candidate fails (a), and
the good primes, with $x z, x y$, replace them.) So the result of
:func:`modular_groebner` is the reduced Gröbner basis of $I$, for
homogeneous and non-homogeneous input alike, whatever the primes.

The primes
==========

With SymPy's pure Python ground types the time of a Gröbner basis over
``GF(p)`` hardly depends on the size of $p$ (Katsura-4, ``lex``,
homogenised, one run each, wall-clock on a loaded machine: 0.66 s for
$p < 2^{31}$, 0.77 s below $2^{64}$, 0.84 s below $2^{256}$, 1.04 s below
$2^{1024}$, 1.7 s below $2^{2048}$, where finding the prime takes 5 s as
well), while the number of images needed is inversely proportional to it,
and an image is nearly all the work: a few large primes are better than
many word-sized ones. With a first prime of 256 bits all seven of the
random systems of the measurements which took SymPy more than a second
needed a second image; with 1024 bits one image is enough for numerators
and denominators of up to about 490 bits, and four of the seven needed
one.
:func:`default_primes` is the fixed sequence of the primes below
$2^{1024}$ in decreasing order, starting at $2^{1024} - 105$. Nothing is
random. Another sequence may be given (the tests force 2, 3, 5, 7, ... to
meet unlucky primes).

References
==========

.. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
       J. Symbolic Computation 35 (2003) 403-419.
.. [2] N. Idrees, G. Pfister, S. Steidel, Parallelization of modular
       algorithms, J. Symbolic Computation 46 (2011) 672-684.
.. [3] F. Pauer, On lucky ideals for Gröbner basis computations,
       J. Symbolic Computation 14 (1992) 471-482.
.. [4] G. L. Ebert, Some comments on the modular approach to Gröbner
       bases, ACM SIGSAM Bulletin 17 (1983) 28-32.
.. [5] P. S. Wang, A p-adic algorithm for univariate partial fractions,
       Proc. SYMSAC 1981, 212-217; P. S. Wang, M. J. T. Guy,
       J. H. Davenport, P-adic reconstruction of rational numbers, ACM
       SIGSAM Bulletin 16 (1982) 2-3.
.. [6] M. Monagan, Maximal quotient rational reconstruction: an almost
       optimal algorithm for rational reconstruction, Proc. ISSAC 2004,
       243-249.
.. [7] R. Gebauer, H. M. Möller, On an installation of Buchberger's
       algorithm, J. Symbolic Computation 6 (1988) 275-286; T. Becker,
       V. Weispfenning, Gröbner Bases, Springer, 1993, section 5.5.
"""
from __future__ import annotations

from math import gcd, isqrt
from typing import Iterable, Iterator, Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.ntheory.generate import prevprime
from sympy.polys.domains import GF, QQ
from sympy.polys.domains.finitefield import FiniteField
from sympy.polys.groebnertools import groebner as _groebner
from sympy.polys.orderings import MonomialOrder, grevlex, grlex
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._timeout import attempt
from sympy_extras._typing import Monomial, OrderKey
from sympy_extras.settings import settings

__all__ = ['groebner', 'modular_groebner', 'rational_reconstruction', 'chinese_remainder',
    'default_primes', 'ModularTrace']

#: a polynomial with integer coefficients, by exponent vectors
IntPoly = dict[Monomial, int]
#: a polynomial with rational coefficients ``(numerator, denominator)``
RatPoly = dict[Monomial, tuple[int, int]]
#: the sort key of the leading monomials of an image, see ``_luckiness``
Luckiness = tuple[tuple[int, tuple[object, ...]], ...]

#: for the graded orders SymPy's direct computation is given this many times
#: ``settings.groebner_direct_time``: see :func:`groebner`
GRADED_FACTOR = 20

#: a candidate whose coefficients are all found within
#: ``sqrt(M/2**(1 + GUARD_BITS))`` is verified without a confirming prime
GUARD_BITS = 32


# ----------------------------------------------------------------------
# integers

def rational_reconstruction(a: int, m: int, bound: Optional[int] = None) -> Optional[tuple[int, int]]:
    """The fraction ``n/d`` congruent to ``a`` modulo ``m`` with
    ``abs(n) <= bound``, ``0 < d <= bound`` and ``gcd(n, d) == 1``, or
    ``None`` if there is none (Wang's algorithm).

    Parameters
    ==========

    a, m : int
        The residue and the modulus (``m > 1``).
    bound : int, optional
        The bound on the numerator and on the denominator; by default the
        largest one for which the answer is unique, ``isqrt((m - 1)//2)``
        (that is $2\\,\\mathrm{bound}^2 < m$).

    Returns
    =======

    ``(n, d)`` or ``None``. The denominator is invertible modulo ``m``.

    Examples
    ========

    >>> from sympy_extras.polys.modulargroebner import rational_reconstruction
    >>> rational_reconstruction(pow(3, -1, 101)*(-2) % 101, 101)
    (-2, 3)
    >>> rational_reconstruction(10, 101) is None
    True

    Explanation
    ===========

    The extended Euclidean algorithm on ``(m, a)`` produces remainders
    ``r_i = s_i*m + t_i*a``, that is ``r_i/t_i = a (mod m)``; if a fraction
    within the bound exists it is the first one whose remainder is within
    the bound.

    References
    ==========

    .. [1] P. S. Wang, M. J. T. Guy, J. H. Davenport, P-adic reconstruction
           of rational numbers, ACM SIGSAM Bulletin 16 (1982) 2-3.
    .. [2] J. von zur Gathen, J. Gerhard, Modern Computer Algebra, 3rd
           ed., Cambridge University Press, 2013, section 5.10.
    """
    if m <= 1:
        raise ValueError("the modulus must be greater than 1")
    limit = isqrt((m - 1)//2)
    if bound is None:
        bound = limit
    elif bound < 0 or bound > limit:
        raise ValueError("the bound must satisfy 2*bound**2 < m")
    r0, t0 = m, 0
    r1, t1 = a % m, 1
    while r1 > bound:
        q = r0//r1
        r0, r1 = r1, r0 - q*r1
        t0, t1 = t1, t0 - q*t1
    if t1 < 0:
        r1, t1 = -r1, -t1
    if t1 == 0 or t1 > bound or gcd(r1, t1) != 1:
        return None
    return r1, t1


def chinese_remainder(a: int, m: int, b: int, p: int) -> int:
    """The integer in ``range(m*p)`` congruent to ``a`` modulo ``m`` and
    to ``b`` modulo ``p``, for coprime ``m`` and ``p``.

    Examples
    ========

    >>> from sympy_extras.polys.modulargroebner import chinese_remainder
    >>> chinese_remainder(2, 3, 3, 5)
    8

    References
    ==========

    .. [1] J. von zur Gathen, J. Gerhard, Modern Computer Algebra, 3rd
           ed., Cambridge University Press, 2013, section 5.4.
    """
    return _lift(a, m, b, p, pow(m, -1, p))


def _lift(a: int, m: int, b: int, p: int, m_inverse: int) -> int:
    return a + m*(((b - a) % p)*m_inverse % p)


#: the largest prime below ``2**1024`` (the tests check that it is)
_FIRST_PRIME = 2**1024 - 105
_large_primes: list[int] = [_FIRST_PRIME]


def default_primes() -> Iterator[int]:
    """The primes used by :func:`modular_groebner`: the primes below
    ``2**1024`` in decreasing order (see the module documentation for the
    measurements behind the size).

    Examples
    ========

    >>> from sympy_extras.polys.modulargroebner import default_primes
    >>> primes = default_primes()
    >>> 2**1024 - next(primes), 2**1024 - next(primes)
    (105, 179)

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    """
    i = 0
    while True:
        if i == len(_large_primes):
            _large_primes.append(int(prevprime(_large_primes[-1])))
        yield _large_primes[i]
        i += 1


# ----------------------------------------------------------------------
# polynomials with integer coefficients

class _HomogeneousOrder(MonomialOrder):
    """Total degree first, then ``base`` on the first ``nvars`` exponents
    (all of them, or all but the homogenising variable)."""
    alias = 'homogeneous'
    is_global = True

    def __init__(self, base: MonomialOrder, nvars: int) -> None:
        self.base = base
        self.nvars = nvars

    def __call__(self, monomial: Monomial) -> tuple[int, object]:
        return (sum(monomial), self.base(monomial[:self.nvars]))

    def __hash__(self) -> int:
        return hash((self.__class__, self.base, self.nvars))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HomogeneousOrder) and \
            (self.base, self.nvars) == (other.base, other.nvars)

    def __str__(self) -> str:
        return "_HomogeneousOrder(%s, %d)" % (self.base, self.nvars)

    __repr__ = __str__


def _cached(order: OrderKey) -> OrderKey:
    """``order`` with its values remembered: the leading monomial of a
    polynomial under reduction is searched at every step."""
    cache: dict[Monomial, object] = {}

    def key(monomial: Monomial) -> object:
        value = cache.get(monomial)
        if value is None:
            value = cache[monomial] = order(monomial)
        return value
    return key


def _divides(a: Monomial, b: Monomial) -> bool:
    return all(i <= j for i, j in zip(a, b))


def _primitive(f: IntPoly, leading: Optional[Monomial] = None) -> IntPoly:
    """``f`` divided by its content, with a positive coefficient at
    ``leading`` if given."""
    content = 0
    for c in f.values():
        content = gcd(content, c)
        if content == 1:
            break
    if leading is not None and f[leading] < 0:
        content = -content if content else -1
    if content in (0, 1):
        return f
    return {m: c//content for m, c in f.items()}


def _normal_form(f: IntPoly, basis: Sequence[tuple[Monomial, IntPoly]], key: OrderKey,
                 full: bool) -> IntPoly:
    """A non-zero integer multiple of the normal form of ``f`` modulo
    ``basis`` (pairs of a leading monomial and a polynomial), computed
    without fractions. With ``full=False`` the reduction stops at the first
    leading monomial which cannot be reduced: the result is zero if and
    only if the normal form is."""
    f = dict(f)
    rest: IntPoly = {}
    while f:
        m = max(f, key=key)
        c = f[m]
        for lm, g in basis:
            if _divides(lm, m):
                break
        else:
            if not full:
                return f
            rest[m] = c
            del f[m]
            continue
        a = g[lm]
        d = gcd(a, c)
        u, v = a//d, c//d
        if u != 1:
            for mf in f:
                f[mf] *= u
            for mr in rest:
                rest[mr] *= u
        shift = tuple(i - j for i, j in zip(m, lm))
        for mg, cg in g.items():
            mm = tuple(i + j for i, j in zip(mg, shift))
            value = f.get(mm, 0) - v*cg
            if value:
                f[mm] = value
            else:
                del f[mm]
        if u != 1:
            content = 0
            for value in f.values():
                content = gcd(content, value)
                if content == 1:
                    break
            else:
                for value in rest.values():
                    content = gcd(content, value)
                    if content == 1:
                        break
            if content > 1:
                for mf in f:
                    f[mf] //= content
                for mr in rest:
                    rest[mr] //= content
    return rest


def _s_polynomial(lf: Monomial, f: IntPoly, lg: Monomial, g: IntPoly) -> IntPoly:
    a, b = f[lf], g[lg]
    d = gcd(a, b)
    a, b = a//d, b//d
    sf = tuple(max(i, j) - i for i, j in zip(lf, lg))
    sg = tuple(max(i, j) - j for i, j in zip(lf, lg))
    s: IntPoly = {tuple(i + j for i, j in zip(m, sf)): b*c for m, c in f.items()}
    for m, c in g.items():
        mm = tuple(i + j for i, j in zip(m, sg))
        value = s.get(mm, 0) - a*c
        if value:
            s[mm] = value
        else:
            del s[mm]
    return s


def _critical_pairs(leading: Sequence[Monomial]) -> list[tuple[int, int]]:
    """The pairs ``(i, j)`` whose S-polynomials must reduce to zero for
    polynomials with these leading monomials to be a Gröbner basis: the
    pairs left by the criteria of Gebauer and Möller when the polynomials
    are inserted one after the other.

    Inserting ``h`` of index ``k``: an old pair ``(i, j)`` is dropped when
    ``h`` divides its lcm and the lcms of ``(i, k)`` and ``(j, k)`` both
    differ from it (criterion B); of the new pairs ``(i, k)`` those whose
    lcm is a proper multiple of the lcm of another new pair are dropped
    (M), one is kept for each lcm (F), and none for an lcm reached by a pair
    of coprime leading monomials (Buchberger's first criterion)."""
    pairs: list[tuple[int, int, Monomial]] = []
    for k, h in enumerate(leading):
        lcms = [tuple(max(a, b) for a, b in zip(leading[i], h)) for i in range(k)]
        pairs = [(i, j, l) for i, j, l in pairs
                 if not (_divides(h, l) and lcms[i] != l and lcms[j] != l)]
        by_lcm: dict[Monomial, list[int]] = {}
        for i, l in enumerate(lcms):
            by_lcm.setdefault(l, []).append(i)
        for l, indices in by_lcm.items():
            if any(other != l and _divides(other, l) for other in by_lcm):
                continue
            if any(all(a == 0 or b == 0 for a, b in zip(leading[i], h)) for i in indices):
                continue
            pairs.append((indices[0], k, l))
    return [(i, j) for i, j, _ in pairs]


def _is_groebner(basis: Sequence[tuple[Monomial, IntPoly]], key: OrderKey) -> bool:
    """Whether the polynomials (with their leading monomials) are a Gröbner
    basis of the ideal they generate."""
    for i, j in _critical_pairs([lm for lm, _ in basis]):
        s = _s_polynomial(basis[i][0], basis[i][1], basis[j][0], basis[j][1])
        if _normal_form(s, basis, key, False):
            return False
    return True


def _reduced(basis: Sequence[IntPoly], key: OrderKey) -> list[tuple[Monomial, IntPoly]]:
    """The reduced Gröbner basis (up to integer factors, sorted by
    decreasing leading monomials) of the ideal of which ``basis`` is a
    Gröbner basis."""
    with_leading = sorted(((max(g, key=key), g) for g in basis), key=lambda pair: key(pair[0]))
    minimal: list[tuple[Monomial, IntPoly]] = []
    for lm, g in with_leading:
        if not any(_divides(other, lm) for other, _ in minimal):
            minimal.append((lm, g))
    result: list[tuple[Monomial, IntPoly]] = []
    for i, (lm, g) in enumerate(minimal):
        others = minimal[:i] + minimal[i + 1:]
        result.append((lm, _primitive(_normal_form(g, others, key, True), lm)))
    result.reverse()
    return result


# ----------------------------------------------------------------------
# the images and their lifting

def _luckiness(leading: Sequence[Monomial], key: OrderKey) -> Luckiness:
    """The sort key of the leading monomials of an image: for every degree
    from zero up, the number of leading monomials of that degree and their
    order keys, decreasing. Images of a homogeneous ideal with different
    leading monomials have different keys and the one with the smaller key
    comes from an unlucky prime (see the module documentation)."""
    if not leading:
        return ()
    by_degree: list[list[object]] = [[] for _ in range(max(sum(m) for m in leading) + 1)]
    for m in sorted(leading, key=key, reverse=True):
        by_degree[sum(m)].append(key(m))
    return tuple((len(keys), tuple(keys)) for keys in by_degree)


class ModularTrace:
    """What :func:`modular_groebner` did, filled in when passed as its
    ``trace`` argument.

    Attributes
    ==========

    homogenized : bool
        Whether the input was homogenised.
    used : list of int
        The primes whose images were computed, in order.
    skipped : list of int
        The primes which divide a leading coefficient of the input.
    unlucky : list of int
        The primes whose images were discarded, their leading monomials
        being those of an unlucky prime.
    lucky : list of int
        The primes whose images gave the result.
    failed_verifications : int
        The number of stable candidates which were not the Gröbner basis.

    Examples
    ========

    >>> from sympy import QQ, ring, lex, primerange
    >>> from sympy_extras.polys.modulargroebner import modular_groebner, ModularTrace
    >>> R, x, y = ring("x,y", QQ, lex)
    >>> trace = ModularTrace()
    >>> modular_groebner([x**2 + y, x**2 + 7*y], R, primes=primerange(2, 100), trace=trace)
    [x**2, y]
    >>> trace.unlucky, trace.lucky
    ([2, 3], [5, 7])

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    """

    def __init__(self) -> None:
        self.homogenized: bool = False
        self.used: list[int] = []
        self.skipped: list[int] = []
        self.unlucky: list[int] = []
        self.lucky: list[int] = []
        self.failed_verifications: int = 0

    def __repr__(self) -> str:
        return "ModularTrace(used=%r, skipped=%r, unlucky=%r, lucky=%r, failed_verifications=%r)" % (
            self.used, self.skipped, self.unlucky, self.lucky, self.failed_verifications)


_fields: dict[int, FiniteField] = {}


def _image(polys: Sequence[IntPoly], symbols: Sequence[Expr], order: MonomialOrder, key: OrderKey,
           p: int) -> list[tuple[Monomial, IntPoly]]:
    """The reduced Gröbner basis of the polynomials modulo ``p``, computed
    by SymPy, by increasing leading monomials, with coefficients in
    ``range(p)``."""
    field = _fields.get(p)
    if field is None:
        # SymPy tests the primality of the modulus the first time a finite
        # field is asked whether it is a field (13 ms for 1024 bits), which
        # ``groebner`` does: the domains are kept
        field = _fields[p] = GF(p)
    ring = PolyRing(symbols, field, order)
    reduced = [ring.from_dict({m: c % p for m, c in f.items()}) for f in polys]
    basis = _groebner([f for f in reduced if f], ring)
    image: list[tuple[Monomial, IntPoly]] = []
    for g in basis:
        coefficients: IntPoly = {tuple(m): int(field.to_int(c)) % p for m, c in g.items()}
        image.append((max(coefficients, key=key), coefficients))
    image.sort(key=lambda pair: key(pair[0]))
    return image


class _Lifting:
    """The images of the kept primes, combined modulo their product."""

    def __init__(self) -> None:
        self.modulus: int = 1
        self.primes: list[int] = []
        self.luckiness: Luckiness = ()
        self.leading: list[Monomial] = []
        self.residues: list[IntPoly] = []
        # the coefficient on which the last reconstruction failed is the
        # first one tried by the next
        self.hardest: Optional[tuple[int, Monomial]] = None

    def reset(self, luckiness: Luckiness) -> None:
        self.modulus = 1
        self.primes = []
        self.luckiness = luckiness
        self.leading = []
        self.residues = []
        self.hardest = None

    def add(self, image: Sequence[tuple[Monomial, IntPoly]], p: int) -> None:
        if not self.primes:
            self.leading = [lm for lm, _ in image]
            self.residues = [dict(g) for _, g in image]
        else:
            m = self.modulus
            inverse = pow(m, -1, p)
            for residues, (_, g) in zip(self.residues, image):
                for monomial in set(residues) | set(g):
                    residues[monomial] = _lift(residues.get(monomial, 0), m, g.get(monomial, 0), p, inverse)
        self.modulus *= p
        self.primes.append(p)

    def reconstruct(self) -> Optional[tuple[list[RatPoly], bool]]:
        """The polynomials with rational coefficients having these residues,
        and whether all the coefficients are within the guarded bound;
        ``None`` if a coefficient is not a fraction within Wang's bound."""
        m = self.modulus
        if self.hardest is not None:
            index, monomial = self.hardest
            if rational_reconstruction(self.residues[index][monomial], m) is None:
                return None
        bound = isqrt((m - 1)//2)
        guard = isqrt((m - 1) >> (1 + GUARD_BITS))
        guarded = True
        candidate: list[RatPoly] = []
        for index, residues in enumerate(self.residues):
            g: RatPoly = {}
            # the coefficients of a polynomial share their denominators: a
            # residue whose product by the denominators found so far is
            # within the bound is that product over them, because the
            # fraction within the bound is unique; Euclid's algorithm is
            # run on the others only
            common = 1
            for monomial, residue in residues.items():
                fraction: Optional[tuple[int, int]] = None
                if common <= bound:
                    n = residue*common % m
                    if n > bound:
                        n -= m
                    if -bound <= n:
                        d = gcd(n, common)
                        fraction = (n//d, common//d)
                if fraction is None:
                    fraction = rational_reconstruction(residue, m)
                    if fraction is None:
                        self.hardest = (index, monomial)
                        return None
                    common = common*fraction[1]//gcd(common, fraction[1])
                if fraction[0]:
                    g[monomial] = fraction
                    if abs(fraction[0]) > guard or fraction[1] > guard:
                        guarded = False
            candidate.append(g)
        return candidate, guarded


def _agrees(candidate: Sequence[RatPoly], image: Sequence[tuple[Monomial, IntPoly]], p: int) -> bool:
    """Whether the candidate reduces to the image modulo ``p``."""
    for g, (_, gp) in zip(candidate, image):
        count = 0
        for monomial, (n, d) in g.items():
            if d % p == 0:
                return False
            value = n*pow(d, -1, p) % p
            if value != gp.get(monomial, 0):
                return False
            if value:
                count += 1
        if count != len(gp):
            return False
    return True


def _clear_denominators(g: RatPoly) -> IntPoly:
    common = 1
    for _, d in g.values():
        common = common*d//gcd(common, d)
    return {m: n*(common//d) for m, (n, d) in g.items()}


def _verified(candidate: Sequence[RatPoly], leading: Sequence[Monomial], polys: Sequence[IntPoly],
              key: OrderKey) -> Optional[list[IntPoly]]:
    """The candidate with integer coefficients if it is a Gröbner basis of
    an ideal which contains ``polys``, else ``None``."""
    cleared = [_clear_denominators(g) for g in candidate]
    basis = list(zip(leading, cleared))
    # the reductions prefer the divisors with few terms
    by_size = sorted(basis, key=lambda pair: len(pair[1]))
    for f in polys:
        if _normal_form(f, by_size, key, False):
            return None
    if not _is_groebner(by_size, key):
        return None
    return cleared


# ----------------------------------------------------------------------
# the interface

def _applies(ring: PolyRing) -> bool:
    """Whether the modular algorithm applies: rational or integer
    coefficients and a global monomial order."""
    order = ring.order
    return bool(ring.domain.is_QQ or ring.domain.is_ZZ) and isinstance(order, MonomialOrder) \
        and order.is_global is True


def _integer_polys(polys: Sequence[PolyElement]) -> list[IntPoly]:
    result: list[IntPoly] = []
    for f in polys:
        if not f:
            continue
        g = f.set_ring(f.ring.clone(domain=QQ)).clear_denoms()[1]
        result.append(_primitive({tuple(m): int(QQ.numer(c)) for m, c in g.items()}))
    return result


def modular_groebner(polys: Sequence[PolyElement], ring: PolyRing, primes: Optional[Iterable[int]] = None,
                     trace: Optional[ModularTrace] = None) -> list[PolyElement]:
    """The reduced Gröbner basis of the ideal generated by ``polys`` in
    ``ring``, computed modulo primes and proven; the result is the one of
    :func:`sympy.polys.groebnertools.groebner`.

    Parameters
    ==========

    polys : sequence of PolyElement
        The generators, elements of ``ring``.
    ring : PolyRing
        The ring, whose order is the order of the Gröbner basis. Over
        ``QQ`` the result is monic, over ``ZZ`` primitive with positive
        leading coefficients, sorted by decreasing leading monomials, as
        SymPy's is. For other domains (finite fields, algebraic fields,
        fields of fractions) and for orders which are not global SymPy's
        ``groebner`` is called.
    primes : iterable of int, optional
        The primes to use, in this order; :func:`default_primes` if
        omitted. They must be distinct primes; ``ValueError`` is raised if
        they run out.
    trace : ModularTrace, optional
        Filled in with the primes used, skipped and discarded.

    Returns
    =======

    The reduced Gröbner basis as a list of elements of ``ring``. It is
    proven, not probabilistic, for every input and every sequence of
    primes: see the module documentation.

    Examples
    ========

    >>> from sympy import QQ, ring, lex
    >>> from sympy.polys.groebnertools import groebner
    >>> from sympy_extras.polys.modulargroebner import modular_groebner
    >>> R, x, y, z = ring("x,y,z", QQ, lex)
    >>> F = [x**2 + y**2 + z**2 - 1, x*y - z/3, x + y - 2*z]
    >>> G = modular_groebner(F, R)
    >>> G
    [x + y - 2*z, y**2 - 2*y*z + 1/3*z, z**2 - 2/15*z - 1/5]
    >>> G == groebner(F, R)
    True

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    .. [2] N. Idrees, G. Pfister, S. Steidel, Parallelization of modular
           algorithms, J. Symbolic Computation 46 (2011) 672-684.
    """
    domain = ring.domain
    if not _applies(ring):
        return list(_groebner(list(polys), ring))
    if trace is None:
        trace = ModularTrace()
    integer = _integer_polys(polys)
    if not integer:
        return []
    nvars = ring.ngens
    homogenized = any(len({sum(m) for m in f}) > 1 for f in integer)
    trace.homogenized = homogenized
    symbols: tuple[Expr, ...] = tuple(ring.symbols)
    if homogenized:
        symbols = symbols + (Dummy('t'),)
        homogeneous: list[IntPoly] = []
        for f in integer:
            degree = max(sum(m) for m in f)
            homogeneous.append({m + (degree - sum(m),): c for m, c in f.items()})
        integer = homogeneous
    order = _HomogeneousOrder(ring.order, nvars)
    key = _cached(order)
    leading_coefficients = [f[max(f, key=key)] for f in integer]

    lifting = _Lifting()
    candidate: Optional[list[RatPoly]] = None
    rejected: Optional[list[RatPoly]] = None
    proven: Optional[list[IntPoly]] = None
    for p in (default_primes() if primes is None else primes):
        p = int(p)
        if any(c % p == 0 for c in leading_coefficients):
            trace.skipped.append(p)
            continue
        image = _image(integer, symbols, order, key, p)
        trace.used.append(p)
        luckiness = _luckiness([lm for lm, _ in image], key)
        if lifting.primes and luckiness < lifting.luckiness:
            trace.unlucky.append(p)
            continue
        if lifting.primes and luckiness > lifting.luckiness:
            trace.unlucky.extend(lifting.primes)
            candidate = None
        if not lifting.primes or luckiness > lifting.luckiness:
            lifting.reset(luckiness)
        if candidate is not None and candidate != rejected and _agrees(candidate, image, p):
            proven = _verified(candidate, lifting.leading, integer, key)
            if proven is not None:
                lifting.primes.append(p)
                break
            rejected = candidate
            trace.failed_verifications += 1
        lifting.add(image, p)
        candidate = None
        reconstruction = lifting.reconstruct()
        if reconstruction is not None:
            candidate, guarded = reconstruction
            if guarded and candidate != rejected:
                proven = _verified(candidate, lifting.leading, integer, key)
                if proven is not None:
                    break
                rejected = candidate
                trace.failed_verifications += 1
    if proven is None:
        raise ValueError("the primes ran out before the Gröbner basis was found")
    trace.lucky = list(lifting.primes)

    base = _cached(ring.order)
    if homogenized:
        proven = [{m[:nvars]: c for m, c in g.items()} for g in proven]
    result: list[PolyElement] = []
    for lm, g in _reduced(proven, base):
        if domain.is_ZZ:
            result.append(ring.from_dict(g))
        else:
            lc = g[lm]
            result.append(ring.from_dict({m: QQ(c, lc) for m, c in g.items()}))
    return result


def groebner(polys: Sequence[PolyElement], ring: PolyRing) -> list[PolyElement]:
    """The reduced Gröbner basis of ``polys`` in ``ring``, by SymPy's direct
    computation or by :func:`modular_groebner`, whichever is expected to be
    faster; this is what :class:`~sympy_extras.polys.ideals.Ideal` calls.

    Over ``QQ`` and ``ZZ``, SymPy's ``groebner`` is given
    ``settings.groebner_direct_time`` seconds (a quarter of a second), and
    ``GRADED_FACTOR`` times as long (five seconds) for ``grevlex`` and
    ``grlex``; if it has not finished, the modular algorithm computes the
    basis. The reasons, from the measurements in ``docs/ideals.md``: on the
    inputs which SymPy handles in milliseconds the modular algorithm costs
    about twice as much; for ``lex`` and the elimination orders it is
    faster by one to two orders of magnitude as soon as SymPy needs
    seconds, and no cheap inspection of the input tells the two kinds
    apart, so that the lost quarter of a second is the price of telling
    them apart; for the graded orders, where the coefficients grow less
    and the homogenised images are larger than the basis, it was 1.3 to
    2.4 times *slower* than SymPy on everything measured, and is only the
    last resort. With ``settings.modular_groebner`` off, with other
    coefficient domains, and where time limits are not available (outside
    the main thread) only SymPy's computation is used.

    Examples
    ========

    >>> from sympy import QQ, ring, grevlex
    >>> from sympy_extras.polys.modulargroebner import groebner
    >>> from sympy_extras.settings import configure
    >>> R, x, y = ring("x,y", QQ, grevlex)
    >>> groebner([x**2 + y**2 - 1, x*y - 2], R)
    [y**3 + 2*x - y, x**2 + y**2 - 1, x*y - 2]
    >>> with configure(groebner_direct_time=0):
    ...     groebner([x**2 + y**2 - 1, x*y - 2], R)
    [y**3 + 2*x - y, x**2 + y**2 - 1, x*y - 2]

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    """
    nonzero = [f for f in polys if f]
    if not settings.modular_groebner or not _applies(ring):
        return list(_groebner(nonzero, ring))
    seconds = settings.groebner_direct_time
    if ring.order in (grevlex, grlex):
        seconds *= GRADED_FACTOR
    if seconds > 0:
        direct = attempt(lambda: list(_groebner(list(nonzero), ring)), seconds)
        if direct is not None:
            return direct
    return modular_groebner(nonzero, ring)
