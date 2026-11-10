r"""Exponential Diophantine equations: Pillai's equation $a^x - b^y = c$
and the integer zeros of exponential polynomials.

**Pillai's equation.** For integers $a, b \ge 2$ and $c \ne 0$,
:func:`pillai` returns *all* the pairs of non-negative integers with
$a^x - b^y = c$; there are finitely many (Pillai 1931 for coprime bases,
by Siegel's theorem; effectively by Baker's method), and the list is
proved complete:

1. If $a$ and $b$ are multiplicatively dependent, $a = g^i$ and
   $b = g^j$, the equation is $g^u - g^v = c$: the smaller power divides
   $c$ and the quotient of the powers minus one is at most $|c|$.
2. Otherwise, for a solution with $x, y \ge 1$ and $b^y \ge 2 |c|$, the
   linear form $\Lambda = x \log a - y \log b = \log(1 + c/b^y)$ is not
   zero and $|\Lambda| \le 2 |c| / b^y$. The theorem of Baker and
   Wüstholz, $\log |\Lambda| > -C \max(\log \max(x, y), 1)$ with $C = 18
   \cdot 3! \cdot 2^3 \cdot 32^4 \log 4 \cdot h'(a)\, h'(b)$ and $h'(a) =
   \max(\log a, 1)$, gives a bound $y < Y_0$ (about $10^{11}$).
3. The bound is reduced with the continued fraction of $\theta = \log b
   / \log a$: since $|\theta - x/y| \le 2|c| / (y\, b^y \log a)$, either
   $b^y \le 4 |c| y / \log a$, or $x/y$ is a convergent $p_k/q_k$ of
   $\theta$ (Legendre) with $q_k \le y < Y_0$, and then $|\theta -
   p_k/q_k| > 1/((a_{k+1} + 2) q_k^2)$ gives $b^y < 2 |c| (A + 2) Y_0 /
   \log a$, $A$ the largest partial quotient met before the denominators
   pass $Y_0$. Both bound $y$ by a few dozen.
4. The remaining values of $y$ are tried: $b^y + c$ must be a power of
   $a$ (exact integer arithmetic).

With coefficients, $r a^x - s b^y = c$ (``pillai(a, b, c, r, s)``), the
linear form has a third logarithm $\log(r/s)$ with coefficient one: the
Baker–Wüstholz bound for three logarithms is reduced by the lemma of
Baker and Davenport in the form of Dujella and Pethő (a convergent
$p/q$ of $\theta = \log a/\log b$ for which $q \log(r/s)/\log b$ is far
from an integer).

**Exponential polynomials.** :func:`exponential_polynomial_roots` finds
all the integers $n$ with $\sum_i p_i(n)\, a_i^n = 0$ for polynomials
$p_i$ with rational coefficients and distinct positive rational bases
$a_i$ (``2**n = n**2``, ``3**n + 4**n = 5**n``): beyond an explicit $N$
the term of the largest base dominates the sum of the others wherever
its polynomial does not vanish (for $n \to -\infty$, that of the
smallest base); the integers below are tried, and beyond $N$ the zeros
are the common zeros of that polynomial and of the sum of the other
terms, found recursively.

Examples
========

>>> from sympy_extras.solvers.exponential_diophantine import pillai, exponential_polynomial_roots
>>> pillai(3, 2, 1)                 # 3**x - 2**y = 1: Catalan's 9 - 8 = 1
[(1, 1), (2, 3)]
>>> pillai(2, 3, 13)                # 16 - 3 = 256 - 243 = 13
[(4, 1), (8, 5)]
>>> pillai(2, 5, 3)                 # 4 - 1 = 8 - 5 = 128 - 125 = 3
[(2, 0), (3, 1), (7, 3)]
>>> pillai(5, 7, 1)
[]
>>> from fractions import Fraction
>>> one, zero = [Fraction(1)], [Fraction(0)]
>>> exponential_polynomial_roots([([Fraction(1)], Fraction(2)), ([Fraction(-1), Fraction(0), Fraction(0)], Fraction(1))])
[2, 4]
>>> exponential_polynomial_roots([(one, Fraction(3)), (one, Fraction(4)), ([Fraction(-1)], Fraction(5))])
[2]

The first of these is ``2**n = n**2``, the second ``3**n + 4**n =
5**n``.

References
==========

.. [1] A. Baker, G. Wüstholz, *Logarithmic forms and group varieties*,
       Journal für die reine und angewandte Mathematik 442 (1993), 19-62.
.. [2] S. S. Pillai, *On the inequality 0 < a^x - b^y <= n*, Journal of
       the Indian Mathematical Society 19 (1931), 1-11.
.. [3] R. J. Stroeker, R. Tijdeman, *Diophantine equations*, in
       Computational Methods in Number Theory, MC Tract 155, 1982,
       321-369.
.. [4] B. M. M. de Weger, *Algorithms for Diophantine equations*, CWI
       Tract 65, 1989, Chapters 3 and 4.
.. [5] N. P. Smart, *The Algorithmic Resolution of Diophantine
       Equations*, Cambridge University Press, 1998, Chapters V and VI.
.. [6] A. Dujella, A. Pethő, *A generalization of a theorem of Baker
       and Davenport*, Quarterly Journal of Mathematics 49 (1998),
       291-306.
"""
from __future__ import annotations

from fractions import Fraction
from math import ceil, floor, gcd
from typing import Optional, Sequence

import mpmath

from sympy_extras.ntheory import factorint
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.rootisolation import dup_isolate_real_roots_sqf
from sympy_extras.polys.sqfreetools import dup_sqf_part

#: how many convergents beyond the first large enough one are tried in
#: the reduction with three logarithms
_CONVERGENTS = 60

__all__ = ['pillai', 'exponential_polynomial_roots', 'ExponentialTerm']

#: a term ``p(n) * base**n``: the coefficients of ``p`` (highest degree
#: first) and the positive rational base
ExponentialTerm = tuple[list[Fraction], Fraction]


# ---------------------------------------------------------------------------
# Pillai's equation

def _exponent_of(value: int, base: int) -> Optional[int]:
    """``x >= 0`` with ``base**x == value``, or ``None``."""
    if value < 1:
        return None
    x = 0
    while value % base == 0:
        value //= base
        x += 1
    return x if value == 1 else None


def _common_base(a: int, b: int) -> Optional[tuple[int, int, int]]:
    """``(g, i, j)`` with ``a = g**i`` and ``b = g**j`` when ``a`` and
    ``b`` are multiplicatively dependent, ``None`` otherwise."""
    fa, fb = factorint(a), factorint(b)
    if set(fa) != set(fb):
        return None
    ea = 0
    for e in fa.values():
        ea = gcd(ea, e)
    eb = 0
    for e in fb.values():
        eb = gcd(eb, e)
    # a = ra**ea and b = rb**eb with ra, rb not perfect powers: dependent
    # exactly when ra == rb
    if any(fa[p]*eb != fb[p]*ea for p in fa):
        return None
    g = 1
    for p, e in fa.items():
        g *= p**(e//ea)
    return g, ea, eb


def _dependent_solutions(g: int, i: int, j: int, c: int) -> list[tuple[int, int]]:
    """The solutions of ``g**(i x) - g**(j y) = c``."""
    solutions: list[tuple[int, int]] = []
    size = abs(c)
    v = 0
    while g**v <= size:
        w = 1
        while g**w - 1 <= size:
            if g**v*(g**w - 1) == size:
                # the larger exponent is v + w
                u, low = (v + w, v) if c > 0 else (v, v + w)
                if u % i == 0 and low % j == 0:
                    solutions.append((u//i, low//j))
            w += 1
        v += 1
    return solutions


def _baker_bound(a: int, b: int, c: int) -> int:
    """``Y0`` such that no solution with ``x, y >= 1`` and ``b**y >= 2
    |c|`` has ``y >= Y0`` (Baker–Wüstholz for two logarithms)."""
    with mpmath.workdps(50):
        log_a, log_b = mpmath.log(a), mpmath.log(b)
        constant = (18*6*8*32**4*mpmath.log(4)*max(log_a, mpmath.mpf(1))*max(log_b, mpmath.mpf(1)))
        kappa = max(mpmath.mpf(1), (log_b + 1)/log_a)
        offset = mpmath.log(2*abs(c))
        # y log b - log(2 |c|) < C max(log(kappa y), 1): the largest
        # solution of the equality, by iteration from above
        y = mpmath.mpf(10)**40
        for _ in range(200):
            y = (constant*max(mpmath.log(kappa*y), mpmath.mpf(1)) + offset)/log_b
        bound = int(mpmath.ceil(y*mpmath.mpf('1.001'))) + 2
        # beyond the bound the left-hand side grows faster
        if not (bound > constant/log_b and bound*log_b - offset > constant*max(mpmath.log(kappa*bound), 1)):
            raise ArithmeticError('the Baker bound was not established')
        return bound


def _partial_quotients(a: int, b: int, limit: int, digits: int) -> list[int]:
    """The partial quotients ``a_0, ..., a_k`` of ``log b/log a``, ``k``
    the first index with a denominator ``q_k >= limit``: with them, every
    ``a_(j + 1)`` for the convergents with ``q_j < limit``."""
    with mpmath.workdps(digits):
        theta = mpmath.log(b)/mpmath.log(a)
        quotients: list[int] = []
        q_previous, q = 1, 0
        while True:
            term = int(mpmath.floor(theta))
            quotients.append(term)
            q_previous, q = q, term*q + q_previous
            if q >= limit:
                return quotients
            fraction = theta - term
            if fraction == 0:
                raise ArithmeticError('the continued fraction ended')
            theta = 1/fraction


def _reduced_bound(a: int, b: int, c: int, bound: int) -> int:
    """A bound on ``y`` for the solutions with ``x, y >= 1`` and ``b**y
    >= 2 |c|``, from the continued fraction of ``log b/log a``."""
    digits = 3*len(str(bound)) + 60
    quotients = _partial_quotients(a, b, bound, digits)
    while quotients != _partial_quotients(a, b, bound, 2*digits):
        # not enough digits for these partial quotients
        digits *= 2
        quotients = _partial_quotients(a, b, bound, digits)
    largest = max(quotients[1:])
    with mpmath.workdps(50):
        log_a, log_b = mpmath.log(a), mpmath.log(b)
        # x/y a convergent: b**y < 2 |c| (A + 2) Y0/log a
        y = int(mpmath.ceil(mpmath.log(2*abs(c)*(largest + 2)*bound/log_a)/log_b)) + 1
    # x/y no convergent: b**y <= 4 |c| y/log a, false from some y on;
    # with log a > 1/2 it is enough that b**y > 8 |c| y
    small = 1
    while b**small <= 8*abs(c)*small:
        small += 1
    return max(y, small)


def _dependence(a: int, b: int, r: int, s: int) -> Optional[tuple[int, int, int]]:
    """``(k, i, j)`` with ``k > 0`` and ``(r/s)**k = a**i b**j`` when
    ``r/s`` is multiplicatively dependent on the independent ``a`` and
    ``b``, ``None`` otherwise."""
    fa, fb = factorint(a), factorint(b)
    exponents: dict[int, int] = {}
    for prime, e in factorint(r).items():
        exponents[prime] = exponents.get(prime, 0) + e
    for prime, e in factorint(s).items():
        exponents[prime] = exponents.get(prime, 0) - e
    primes = sorted(set(fa) | set(fb))
    if any(e and prime not in primes for prime, e in exponents.items()):
        return None
    for p1 in primes:
        for p2 in primes:
            determinant = fa.get(p1, 0)*fb.get(p2, 0) - fa.get(p2, 0)*fb.get(p1, 0)
            if determinant == 0:
                continue
            e1, e2 = exponents.get(p1, 0), exponents.get(p2, 0)
            u = Fraction(e1*fb.get(p2, 0) - e2*fb.get(p1, 0), determinant)
            v = Fraction(fa.get(p1, 0)*e2 - fa.get(p2, 0)*e1, determinant)
            if any(u*fa.get(prime, 0) + v*fb.get(prime, 0) != exponents.get(prime, 0) for prime in primes):
                return None
            k = u.denominator*v.denominator//gcd(u.denominator, v.denominator)
            return k, int(u*k), int(v*k)
    return None


def _three_logarithms_baker(a: int, b: int, c: int, r: int, s: int) -> tuple[int, int]:
    """``(Y0, M)``: no solution of ``r a**x - s b**y = c`` with ``y >=
    1`` and ``s b**y >= 2 |c|`` has ``y >= Y0``, and those below have
    ``x <= M`` (Baker–Wüstholz for the three logarithms of ``a``, ``b``
    and ``r/s``)."""
    with mpmath.workdps(50):
        log_a, log_b = mpmath.log(a), mpmath.log(b)
        log_ratio = abs(mpmath.log(mpmath.mpf(r)/s))
        constant = (18*24*81*32**5*mpmath.log(6)*max(log_a, mpmath.mpf(1))*max(log_b, mpmath.mpf(1))
                    * max(mpmath.log(max(r, s)), mpmath.mpf(1)))
        kappa = max(mpmath.mpf(1), (log_b + log_ratio + 1)/log_a)
        offset = mpmath.log(2*abs(c)) - mpmath.log(s)
        y = mpmath.mpf(10)**60
        for _ in range(300):
            y = (constant*max(mpmath.log(kappa*y), mpmath.mpf(1)) + offset)/log_b
        bound = int(mpmath.ceil(y*mpmath.mpf('1.001'))) + 2
        if not (bound > constant/log_b and bound*log_b - offset > constant*max(mpmath.log(kappa*bound), 1)):
            raise ArithmeticError('the Baker bound was not established')
        return bound, int(mpmath.ceil(kappa*bound)) + 1


def _dependent_coefficient_bound(a: int, b: int, c: int, s: int, k: int, j: int, bound: int) -> int:
    """A bound on ``y`` when ``(r/s)**k = a**i b**j``: then ``k Lambda =
    X log a - Y log b`` with ``X = k x + i`` and ``Y = k y - j``, and
    ``|log b/log a - X/Y| <= 2 k |c|/(s b**y Y log a)`` for ``Y > 0``.
    Either ``s b**y log a <= 4 k |c| Y``, or ``X/Y`` is a convergent with
    ``Y < k Y0 + |j|`` and ``s b**y < 2 k |c| (A + 2) Y/log a``."""
    largest = k*bound + abs(j) + 1
    digits = 3*len(str(largest)) + 60
    quotients = _partial_quotients(a, b, largest, digits)
    while quotients != _partial_quotients(a, b, largest, 2*digits):
        digits *= 2
        quotients = _partial_quotients(a, b, largest, digits)
    top = max(quotients[1:])
    with mpmath.workdps(50):
        y = int(mpmath.ceil(mpmath.log(2*k*abs(c)*(top + 2)*largest/(s*mpmath.log(a)))/mpmath.log(b))) + 1
    small = 1
    while s*b**small <= 8*k*abs(c)*(k*small + abs(j)):
        small += 1
    return max(y, small, abs(j) + 1)


def _three_logarithms_bound(a: int, b: int, c: int, r: int, s: int) -> int:
    """A bound on ``y`` for the solutions of ``r a**x - s b**y = c``
    with ``y >= 1`` and ``s b**y >= 2 |c|``, ``r != s`` coprime.

    ``Lambda = x log a - y log b + log(r/s) = log(1 + c/(s b**y))`` is a
    non-zero linear form in three logarithms with ``|Lambda| <= 2 |c|/(s
    b**y)``; Baker–Wüstholz bounds ``y`` and then ``x <= M``. Dividing by
    ``log b``: ``|x theta - y + mu| <= A b**-y`` with ``theta = log a/log
    b``, ``mu = log(r/s)/log b``, ``A = 2 |c|/(s log b)``. For a
    convergent ``p/q`` of ``theta``, ``q (x theta - y + mu) = x (q theta
    - p) + (x p - y q) + q mu`` is at least ``eps = ||q mu|| - M ||q
    theta||`` in absolute value (``||.||`` the distance to the nearest
    integer), so that ``eps > 0`` gives ``b**y < A q/eps`` (the lemma of
    Baker and Davenport in the form of Dujella and Pethő). When ``r/s``
    is multiplicatively dependent on ``a`` and ``b`` the form has two
    logarithms and continued fractions do
    (:func:`_dependent_coefficient_bound`)."""
    bound, largest_x = _three_logarithms_baker(a, b, c, r, s)
    dependence = _dependence(a, b, r, s)
    if dependence is not None:
        return _dependent_coefficient_bound(a, b, c, s, dependence[0], dependence[2], bound)
    digits = 2*len(str(largest_x)) + 260
    with mpmath.workdps(digits):
        log_b = mpmath.log(b)
        theta = mpmath.log(a)/log_b
        mu = mpmath.log(mpmath.mpf(r)/s)/log_b
        weight = 2*abs(c)/(s*log_b)
        rest = theta
        q_previous, q = 1, 0
        tried = 0
        while tried < _CONVERGENTS:
            term = int(mpmath.floor(rest))
            q_previous, q = q, term*q + q_previous
            fraction = rest - term
            if fraction == 0:
                break
            rest = 1/fraction
            if q <= 6*largest_x:
                continue
            tried += 1
            if len(str(q)) > digits//2 - 40:
                break
            eps = abs(q*mu - mpmath.nint(q*mu)) - largest_x*abs(q*theta - mpmath.nint(q*theta))
            if eps > mpmath.mpf(10)**-30:
                return int(mpmath.floor(mpmath.log(weight*q/eps)/log_b)) + 2
    raise ArithmeticError('the reduction with the convergents of log %d/log %d fails for %d/%d' % (a, b, r, s))


def pillai(a: int, b: int, c: int, r: int = 1, s: int = 1) -> list[tuple[int, int]]:
    """All the pairs of integers ``x, y >= 0`` with ``r*a**x - s*b**y ==
    c``, for integers ``a, b >= 2``, ``r, s >= 1`` and ``c != 0``,
    sorted.

    The list is complete: the exponents are bounded by the theorem of
    Baker and Wüstholz on linear forms in logarithms, the bound is
    reduced (by continued fractions for ``r = s``, by the lemma of Baker
    and Davenport otherwise), and the rest is searched.

    Examples
    ========

    >>> from sympy_extras.solvers.exponential_diophantine import pillai
    >>> pillai(2, 3, -1)
    [(1, 1), (3, 2)]
    >>> pillai(2, 3, 5)
    [(3, 1), (5, 3)]
    >>> pillai(3, 2, -5)
    [(1, 3), (3, 5)]
    >>> pillai(4, 2, 12)                  # dependent bases: 16 - 4
    [(2, 2)]
    >>> pillai(10, 3, 73)                 # 100 - 27
    [(2, 3)]

    With coefficients, ``5**x - 2*2**y = -3`` (``125 - 128``) and
    ``5*5**x - 11**y = 4`` (``125 - 121``):

    >>> pillai(5, 2, -3, s=2)
    [(0, 1), (1, 2), (3, 6)]
    >>> pillai(5, 11, 4, r=5)
    [(0, 0), (2, 2)]
    """
    if a < 2 or b < 2:
        raise ValueError('the bases must be at least 2')
    if r < 1 or s < 1:
        raise ValueError('the coefficients must be positive')
    if c == 0:
        raise ValueError('c must not be zero (a**x = b**y has (0, 0) and, for dependent bases, infinitely many '
                         'solutions)')
    common_factor = gcd(r, s)
    if c % common_factor:
        return []
    r, s, c = r//common_factor, s//common_factor, c//common_factor
    solutions: set[tuple[int, int]] = set()
    common = _common_base(a, b)
    if common is not None and r == 1 and s == 1:
        return sorted(set(_dependent_solutions(common[0], common[1], common[2], c)))
    if common is not None:
        # a = g**i, b = g**j: the smaller of g**(i x), g**(j y) divides c
        x, power = 0, 1
        while power <= abs(c):
            quotient, remainder = divmod(r*power - c, s)
            y_found = _exponent_of(quotient, b) if remainder == 0 else None
            if y_found is not None:
                solutions.add((x, y_found))
            x, power = x + 1, power*a
        y, power = 0, 1
        while power <= abs(c):
            quotient, remainder = divmod(c + s*power, r)
            x_found = _exponent_of(quotient, a) if remainder == 0 else None
            if x_found is not None:
                solutions.add((x_found, y))
            y, power = y + 1, power*b
        return sorted(solutions)
    if r == 1 and s == 1:
        limit = _reduced_bound(a, b, c, _baker_bound(a, b, c))
    else:
        limit = _three_logarithms_bound(a, b, c, r, s)
    # the y with s b**y < 2 |c| are not covered by the bounds
    while s*b**limit < 2*abs(c):
        limit += 1
    power = 1
    for y in range(limit + 1):
        quotient, remainder = divmod(c + s*power, r)
        x_found = _exponent_of(quotient, a) if remainder == 0 else None
        if x_found is not None:
            solutions.add((x_found, y))
        power *= b
    return sorted(solutions)


# ---------------------------------------------------------------------------
# exponential polynomials

def _evaluate(p: list[Fraction], n: int) -> Fraction:
    value = Fraction(0)
    for coefficient in p:
        value = value*n + coefficient
    return value


def _merged(terms: Sequence[ExponentialTerm]) -> list[ExponentialTerm]:
    """The terms with equal bases added, the zero terms dropped, sorted
    by decreasing base."""
    by_base: dict[Fraction, list[Fraction]] = {}
    for p, base in terms:
        base = Fraction(base)
        if base <= 0:
            raise ValueError('the bases must be positive')
        current = by_base.get(base, [])
        size = max(len(current), len(p))
        total = [Fraction(0)]*size
        for k, coefficient in enumerate(reversed(current)):
            total[size - 1 - k] += coefficient
        for k, coefficient in enumerate(reversed(p)):
            total[size - 1 - k] += Fraction(coefficient)
        while total and total[0] == 0:
            total.pop(0)
        by_base[base] = total
    return sorted(((p, base) for base, p in by_base.items() if p), key=lambda term: -term[1])


def _integer_roots(p: list[Fraction]) -> list[int]:
    """The integer roots of a non-zero polynomial: its real roots are
    isolated in intervals shorter than one, whose integers are tried."""
    if len(p) <= 1:
        return []
    found: set[int] = set()
    for low, high in dup_isolate_real_roots_sqf(dup_sqf_part(p, QQ), Fraction(1, 2)):
        for candidate in range(floor(low), ceil(high) + 1):
            if _evaluate(p, candidate) == 0:
                found.add(candidate)
    return sorted(found)


def _dominance_bound(terms: list[ExponentialTerm]) -> int:
    """``N >= 1`` such that for every integer ``n >= N`` which is not a
    root of the polynomial of the first term (the largest base), that
    term is larger in absolute value than the sum of the others.

    At such an ``n`` the polynomial is at least ``1/D`` in absolute
    value, ``D`` a common denominator of its coefficients, and the other
    polynomials are at most ``||p||_1 n**E``, ``E`` their largest
    degree: it is enough that ``ratio**n > D S n**E``, ``ratio`` the
    quotient of the two largest bases, which holds from some ``n`` on
    because the left-hand side grows faster beyond ``E/log(ratio)``."""
    lead, base = terms[0]
    denominator = 1
    for c in lead:
        denominator = denominator*c.denominator//gcd(denominator, c.denominator)
    ratio = base/terms[1][1]
    excess = max(len(q) - 1 for q, _ in terms[1:])
    weight = denominator*sum((sum((abs(c) for c in q), Fraction(0)) for q, _ in terms[1:]), Fraction(0))
    # log(ratio) >= 1 - 1/ratio
    n = max(int(excess/(1 - 1/ratio)) + 1, 1)
    while ratio**n <= weight*Fraction(n)**excess:
        n *= 2
    return n


def _nonnegative_zeros(terms: list[ExponentialTerm]) -> list[int]:
    """The integers ``n >= 0`` at which the sum of the terms (merged,
    by decreasing base, not empty) vanishes."""
    lead = terms[0][0]
    if len(terms) == 1:
        return [n for n in _integer_roots(lead) if n >= 0]
    bound = _dominance_bound(terms)
    zeros: list[int] = []
    powers = [Fraction(1)]*len(terms)
    for n in range(bound):
        if sum((_evaluate(q, n)*power for (q, _), power in zip(terms, powers)), Fraction(0)) == 0:
            zeros.append(n)
        powers = [power*base for power, (_, base) in zip(powers, terms)]
    # beyond the bound the sum can vanish only where the first
    # polynomial does, and there it is the sum of the other terms
    for n in _nonnegative_zeros(terms[1:]):
        if n >= bound and _evaluate(lead, n) == 0:
            zeros.append(n)
    return zeros


def exponential_polynomial_roots(terms: Sequence[ExponentialTerm]) -> list[int]:
    """All the integers ``n`` (of either sign) with ``sum p_i(n)
    a_i**n == 0``.

    Parameters
    ==========

    terms : pairs ``(p, a)``: the coefficients of the polynomial ``p``
        (rational, highest degree first) and the positive rational base
        ``a``. Terms with the same base are added.

    Raises ``ValueError`` when the sum is identically zero.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.solvers.exponential_diophantine import exponential_polynomial_roots
    >>> F = Fraction
    >>> exponential_polynomial_roots([([F(1)], F(2)), ([F(-1), F(-1)], F(1))])           # 2**n = n + 1
    [0, 1]
    >>> exponential_polynomial_roots([([F(1)], F(2)), ([F(1)], F(1, 2)), ([F(-5, 2)], F(1))])   # 2**n + 2**-n = 5/2
    [-1, 1]
    >>> exponential_polynomial_roots([([F(1)], F(2)), ([F(1)], F(3))])                    # 2**n + 3**n = 0
    []
    """
    merged = _merged(terms)
    if not merged:
        raise ValueError('the sum is identically zero')
    zeros = set(_nonnegative_zeros(merged))
    # n -> -n: the bases inverted, the polynomials reflected
    mirrored = _merged([([c*(-1)**(len(p) - 1 - k) for k, c in enumerate(p)], 1/base) for p, base in merged])
    zeros.update(-n for n in _nonnegative_zeros(mirrored))
    return sorted(zeros)
