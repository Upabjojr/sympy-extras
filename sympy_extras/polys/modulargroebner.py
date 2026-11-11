r"""Reduced Gröbner bases over the rationals by the modular algorithm.

The direct computation of a Gröbner basis over ``QQ`` (F4,
:mod:`sympy_extras.polys.f4`, or Buchberger's algorithm,
:mod:`sympy_extras.polys.buchberger`) works on rational coefficients,
whose growth dominates the running time: the basis itself may have
coefficients of a few digits while the polynomials met on the way have
hundreds. :func:`modular_groebner` computes the same reduced Gröbner basis
from its images modulo primes, following Arnold [1]_ (see also [2]_, [3]_,
[4]_), and *proves* the result, so that it is not a probabilistic
algorithm.

The module works on the typed rings of :mod:`sympy_extras.polys.rings`
over the typed ``QQ``, ``ZZ`` and ``GF(p)`` and imports no SymPy;
:mod:`sympy_extras.polys.modulargroebner_sympy` is its SymPy face (the
same functions on SymPy's ring elements).

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
   is also what makes the images cheap for the orders where the direct
   computation is slow: Buchberger's algorithm with the normal selection
   strategy then works degree by degree, which for ``lex`` was ten
   (Katsura-4) and twenty-five (cyclic-5) times faster than the
   computation on the input modulo the same prime; for ``grevlex`` it was
   1.4 (Katsura-5) to 2.5 (cyclic-5) times slower (measured with SymPy's
   Buchberger algorithm, before F4).

3. **Images.** For a prime $p$ which divides no leading coefficient of
   $F^h$, the reduced Gröbner basis $G_p$ of $\langle F^h \bmod p \rangle$
   for $>_h$ is computed over the typed ``GF(p)`` by the algorithm of
   ``settings.groebner_algorithm`` (:func:`direct_groebner`: F4 by
   default).

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
   reduced basis, the one F4 returns.

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

With Python's integers the time of a Gröbner basis over ``GF(p)``
hardly depends on the size of $p$ (Katsura-4, ``lex``,
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

from fractions import Fraction
from math import gcd, isqrt
from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar

from sympy_extras._timeout import attempt
from sympy_extras.ntheory.generate import prevprime
from sympy_extras.settings import settings

from .buchberger import buchberger
from .domains.finitefield import FiniteField
from .domains.integerring import ZZ, IntegerRing
from .domains.rationalfield import QQ, RationalField
from .f4 import f4
from .rings import Monomial, MonomialOrder, OrderKeyValue, PolyElement, PolyRing

__all__ = ['direct_groebner', 'groebner', 'modular_groebner', 'rational_reconstruction', 'chinese_remainder',
    'default_primes', 'homogeneous_order', 'ModularTrace', 'GRADED_FACTOR', 'GUARD_BITS']

T = TypeVar('T')

#: a polynomial with integer coefficients, by exponent vectors
IntPoly = dict[Monomial, int]
#: a polynomial with rational coefficients ``(numerator, denominator)``
RatPoly = dict[Monomial, tuple[int, int]]
#: the key of a monomial order
OrderKey = Callable[[Monomial], OrderKeyValue]
#: the sort key of the leading monomials of an image, see ``_luckiness``
Luckiness = tuple[tuple[int, tuple[OrderKeyValue, ...]], ...]
#: an image: the leading monomials and the polynomials with coefficients in ``range(p)``
Image = list[tuple[Monomial, IntPoly]]

#: for the graded orders the direct computation is given this many times
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
            _large_primes.append(prevprime(_large_primes[-1]))
        yield _large_primes[i]
        i += 1


# ----------------------------------------------------------------------
# polynomials with integer coefficients

def homogeneous_order(base: MonomialOrder, nvars: int) -> MonomialOrder:
    """The order which compares the total degree first, then ``base`` on
    the first ``nvars`` exponents (all of them, or all but the
    homogenising variable): the order of the images of
    :func:`modular_groebner`. Its keys are remembered (the leading monomial
    of a polynomial under reduction is searched at every step).

    >>> from sympy_extras.polys.rings import lex
    >>> from sympy_extras.polys.modulargroebner import homogeneous_order
    >>> H = homogeneous_order(lex, 2)
    >>> H, H((0, 2, 0)) > H((1, 0, 0)), H((1, 0, 1)) > H((0, 2, 0))
    (HomogeneousOrder(lex, 2), True, True)
    """
    base_key = base.key
    cache: dict[Monomial, OrderKeyValue] = {}

    def key(monomial: Monomial) -> OrderKeyValue:
        found = cache.get(monomial)
        if found is None:
            found = cache[monomial] = (sum(monomial),) + base_key(monomial[:nvars])
        return found
    return MonomialOrder('HomogeneousOrder(%s, %d)' % (base, nvars), key, True)


def _cached(order: OrderKey) -> OrderKey:
    """``order`` with its values remembered."""
    cache: dict[Monomial, OrderKeyValue] = {}

    def key(monomial: Monomial) -> OrderKeyValue:
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


def _normal_form(f: IntPoly, basis: Sequence[tuple[Monomial, IntPoly]], key: OrderKey, full: bool) -> IntPoly:
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
        divisor: Optional[tuple[Monomial, IntPoly]] = None
        for lm, g in basis:
            if _divides(lm, m):
                divisor = (lm, g)
                break
        if divisor is None:
            if not full:
                return f
            rest[m] = c
            del f[m]
            continue
        lm, g = divisor
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
    by_degree: list[list[OrderKeyValue]] = [[] for _ in range(max(sum(m) for m in leading) + 1)]
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

    >>> from sympy_extras.ntheory.generate import primerange
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.modulargroebner import modular_groebner, ModularTrace
    >>> R = PolyRing(QQ, ['x', 'y'], lex)
    >>> x, y = R.gens
    >>> trace = ModularTrace()
    >>> G = modular_groebner([x**2 + y, x**2 + y.mul_ground(QQ.from_int(7))], R, primes=primerange(2, 100), trace=trace)
    >>> [str(g) for g in G]
    ['x**2', 'y']
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


def _image(polys: Sequence[IntPoly], names: Sequence[str], order: MonomialOrder, p: int) -> Image:
    """The reduced Gröbner basis of the polynomials modulo ``p``, by
    increasing leading monomials, with coefficients in ``range(p)``, by
    :func:`direct_groebner` over ``GF(p)``."""
    ring = PolyRing(FiniteField(p), names, order)
    reduced = [ring.from_dict({m: c % p for m, c in f.items()}) for f in polys]
    basis = direct_groebner([f for f in reduced if f.terms], ring)
    key = order.key
    image: Image = [(g.LM, {m: c % p for m, c in g.terms.items()}) for g in basis]
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

    def add(self, image: Image, p: int) -> None:
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


def _agrees(candidate: Sequence[RatPoly], image: Image, p: int) -> bool:
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

def _integer_polys(polys: Sequence[PolyElement[T]]) -> list[IntPoly]:
    """The non-zero polynomials over ``QQ`` or ``ZZ`` with their
    denominators and contents removed."""
    result: list[IntPoly] = []
    for f in polys:
        if not f.terms:
            continue
        K = f.ring.domain
        fractions = {m: QQ.convert(c, K) for m, c in f.terms.items()}
        common = 1
        for c in fractions.values():
            common = common*c.denominator//gcd(common, c.denominator)
        result.append(_primitive({m: c.numerator*(common//c.denominator) for m, c in fractions.items()}))
    return result


def _integer_basis(integer: list[IntPoly], ring: PolyRing[T], primes: Optional[Iterable[int]],
                   trace: ModularTrace) -> list[tuple[Monomial, IntPoly]]:
    """The reduced Gröbner basis of the polynomials with integer
    coefficients in the variables and the order of ``ring``, primitive,
    with positive leading coefficients, by decreasing leading monomials."""
    nvars = ring.ngens
    homogenized = any(len({sum(m) for m in f}) > 1 for f in integer)
    trace.homogenized = homogenized
    names = list(ring.names)
    if homogenized:
        # a name which is none of the variables (it is only printed)
        t = '_t'
        while t in names:
            t = '_' + t
        names.append(t)
        homogeneous: list[IntPoly] = []
        for f in integer:
            degree = max(sum(m) for m in f)
            homogeneous.append({m + (degree - sum(m),): c for m, c in f.items()})
        integer = homogeneous
    order = homogeneous_order(ring.order, nvars)
    key = order.key
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
        image = _image(integer, names, order, p)
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
    if homogenized:
        proven = [{m[:nvars]: c for m, c in g.items()} for g in proven]
    return _reduced(proven, _cached(ring.order.key))


def modular_groebner(polys: Sequence[PolyElement[T]], ring: PolyRing[T], primes: Optional[Iterable[int]] = None,
                     trace: Optional[ModularTrace] = None) -> list[PolyElement[T]]:
    """The reduced Gröbner basis of the ideal generated by ``polys`` in
    ``ring`` (over ``QQ`` or ``ZZ``), computed modulo primes and proven.

    Parameters
    ==========

    polys : sequence of PolyElement
        The generators, elements of ``ring`` (or of a ring with the same
        variables over ``QQ`` or ``ZZ``).
    ring : PolyRing
        The ring, over ``QQ`` or ``ZZ`` (``ValueError`` otherwise), whose
        order (a global one) is the order of the Gröbner basis. Over ``QQ``
        the result is monic, over ``ZZ`` primitive with positive leading
        coefficients (the basis over ``QQ`` with its denominators
        cleared), sorted by decreasing leading monomials.
    primes : iterable of int, optional
        The primes to use, in this order; :func:`default_primes` if
        omitted. They must be distinct primes; ``ValueError`` is raised if
        they run out.
    trace : ModularTrace, optional
        Filled in with the primes used, skipped and discarded.

    Returns
    =======

    The reduced Gröbner basis as a list of elements of ``ring``, the one
    :func:`sympy_extras.polys.f4.f4` computes. It is proven, not
    probabilistic, for every input and every sequence of primes: see the
    module documentation.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, lex
    >>> from sympy_extras.polys.f4 import f4
    >>> from sympy_extras.polys.modulargroebner import modular_groebner
    >>> R = PolyRing(QQ, ['x', 'y', 'z'], lex)
    >>> x, y, z = R.gens
    >>> F = [x**2 + y**2 + z**2 - R.one, x*y - z.mul_ground(Fraction(1, 3)), x + y - z.mul_ground(QQ.from_int(2))]
    >>> G = modular_groebner(F, R)
    >>> [str(g) for g in G]
    ['x + y - 2*z', 'y**2 - 2*y*z + 1/3*z', 'z**2 - 2/15*z - 1/5']
    >>> G == f4(F, R)
    True

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    .. [2] N. Idrees, G. Pfister, S. Steidel, Parallelization of modular
           algorithms, J. Symbolic Computation 46 (2011) 672-684.
    """
    K = ring.domain
    if not isinstance(K, (RationalField, IntegerRing)):
        raise ValueError("the modular algorithm needs QQ or ZZ as the coefficient domain, got %s" % K)
    integral = isinstance(K, IntegerRing)
    if trace is None:
        trace = ModularTrace()
    integer = _integer_polys(polys)
    if not integer:
        return []
    result: list[PolyElement[T]] = []
    for lm, g in _integer_basis(integer, ring, primes, trace):
        if integral:
            result.append(PolyElement({m: ring.domain.convert(c, ZZ) for m, c in g.items()}, ring))
        else:
            lc = g[lm]
            result.append(PolyElement({m: ring.domain.convert(Fraction(c, lc), QQ) for m, c in g.items()}, ring))
    return result


def direct_groebner(polys: Sequence[PolyElement[T]], ring: PolyRing[T]) -> list[PolyElement[T]]:
    """The reduced Gröbner basis over a field by the direct computation of
    ``settings.groebner_algorithm``: F4 (:mod:`sympy_extras.polys.f4`,
    ``'f4'``, the default) or Buchberger's algorithm
    (:mod:`sympy_extras.polys.buchberger`, ``'buchberger'``).

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex
    >>> from sympy_extras.polys.modulargroebner import direct_groebner
    >>> from sympy_extras.settings import configure
    >>> R = PolyRing(QQ, ['x', 'y'], grevlex)
    >>> x, y = R.gens
    >>> F = [x**2 + y**2 - R.one, x*y - R.from_int(2)]
    >>> [str(g) for g in direct_groebner(F, R)]
    ['y**3 + 2*x - y', 'x**2 + y**2 - 1', 'x*y - 2']
    >>> with configure(groebner_algorithm='buchberger'):
    ...     direct_groebner(F, R) == direct_groebner(F, R)
    True
    """
    nonzero = [f for f in polys if f.terms]
    if settings.groebner_algorithm == 'f4':
        return f4(nonzero, ring)
    return buchberger(nonzero, ring)


def _is_graded(order: MonomialOrder) -> bool:
    return order.name in ('grevlex', 'grlex')


def groebner(polys: Sequence[PolyElement[T]], ring: PolyRing[T]) -> list[PolyElement[T]]:
    """The reduced Gröbner basis of ``polys`` in ``ring``, by the direct
    computation (:func:`direct_groebner`) or by :func:`modular_groebner`,
    whichever is expected to be faster; this is what
    :class:`~sympy_extras.polys.ideals.Ideal` calls.

    Over ``QQ`` the direct computation is given
    ``settings.groebner_direct_time`` seconds (a quarter of a second), and
    ``GRADED_FACTOR`` times as long (five seconds) for ``grevlex`` and
    ``grlex``; if it has not finished, the modular algorithm computes the
    basis. The reasons, from the measurements in ``docs/ideals.md``: on the
    inputs which the direct computation handles in milliseconds the
    modular algorithm costs about twice as much; for ``lex`` and the
    elimination orders it is faster by one to two orders of magnitude as
    soon as the direct computation needs seconds, and no cheap inspection
    of the input tells the two kinds apart, so that the lost quarter of a
    second is the price of telling them apart; for the graded orders,
    where the coefficients grow less and the homogenised images are larger
    than the basis, it was slower on everything measured, and is only the
    last resort. With ``settings.modular_groebner`` off, over other fields,
    and where time limits are not available (outside the main thread)
    only the direct computation is used. Over ``ZZ`` (which is not a
    field) the result is the one of :func:`modular_groebner`.

    Examples
    ========

    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.rings import PolyRing, grevlex
    >>> from sympy_extras.polys.modulargroebner import groebner
    >>> from sympy_extras.settings import configure
    >>> R = PolyRing(QQ, ['x', 'y'], grevlex)
    >>> x, y = R.gens
    >>> F = [x**2 + y**2 - R.one, x*y - R.from_int(2)]
    >>> [str(g) for g in groebner(F, R)]
    ['y**3 + 2*x - y', 'x**2 + y**2 - 1', 'x*y - 2']
    >>> with configure(groebner_direct_time=0):
    ...     [str(g) for g in groebner(F, R)]
    ['y**3 + 2*x - y', 'x**2 + y**2 - 1', 'x*y - 2']

    References
    ==========

    .. [1] E. A. Arnold, Modular algorithms for computing Gröbner bases,
           J. Symbolic Computation 35 (2003) 403-419.
    """
    nonzero = [f if f.ring == ring else PolyElement(dict(f.terms), ring) for f in polys if f.terms]
    if not nonzero:
        return []
    if isinstance(ring.domain, IntegerRing):
        return modular_groebner(nonzero, ring)
    if not settings.modular_groebner or not isinstance(ring.domain, RationalField):
        return direct_groebner(nonzero, ring)
    seconds = settings.groebner_direct_time
    if _is_graded(ring.order):
        seconds *= GRADED_FACTOR
    if seconds > 0:
        direct = attempt(lambda: direct_groebner(nonzero, ring), seconds, native=True)
        if direct is not None:
            return direct
    return modular_groebner(nonzero, ring)
