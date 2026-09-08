"""Thue equations `F(x, y) = m`: all the integer solutions, by Baker's
method with the reduction of Tzanakis and de Weger.

`F` is an irreducible binary form of degree `n \\ge 3` with integer
coefficients and `m` a nonzero integer. With `a_0` the leading
coefficient, `g(X) = a_0^{n-1} F(X/a_0, 1)` is monic with a root
`\\alpha`, and `a_0^{n-1} F(x, y) = N(\\beta)` for `\\beta = a_0 x - \\alpha y`
in the order `O = \\mathbb{Z}[\\alpha]`. The steps:

1. **Units of the order.** Elements of small height with norm `\\pm 1`
   are enumerated until `r = r_1 + r_2 - 1` independent ones are found;
   the group they generate is then *saturated*: by Friedman's bound
   `R \\ge 0.2052` on regulators, its index in the unit group is at most
   `R'/0.2052`, and for every prime `p` below that bound the `p`-th roots
   of the products `\\prod \\varepsilon_k^{e_k}` are searched (exactly, from
   their conjugates), enlarging the group until it is the full unit group.
2. **Elements of norm** `a_0^{n-1} m` **modulo units**: every class has a
   representative whose conjugates are balanced (its logarithmic vector
   lies in the fundamental domain of the unit lattice), which bounds its
   coordinates; they are enumerated and the associates removed.
3. **The linear form.** For a solution with `|y|` large the closest root
   `\\alpha^{(i_0)}` is real and `|\\beta^{(i_0)}| \\le c_1 |y|^{1-n}`; Siegel's
   identity between three conjugates gives
   `\\Lambda = \\log \\big|\\frac{(\\alpha^{(k)} - \\alpha^{(i_0)})\\beta^{(j)}}
   {(\\alpha^{(j)} - \\alpha^{(i_0)})\\beta^{(k)}}\\big| = \\log|1 - \\delta|`
   with `|\\delta| \\le c_6 |y|^{-n}`, and writing
   `\\beta = \\zeta \\mu \\prod \\varepsilon_k^{b_k}` this is a linear form in
   the logarithms of algebraic numbers with integer coefficients `b_k`
   (for a complex pair `j, k` the argument of the same quotient is used,
   with an extra multiple of `2\\pi`). The exponents satisfy
   `B = \\max |b_k| \\le c_3 + c_4 \\log|y|`.
4. **Baker's bound.** The theorem of Baker and Wüstholz gives
   `\\log|\\Lambda| \\ge -C \\log B`, which with `|\\Lambda| \\le 2c_6 |y|^{-n}`
   bounds `\\log|y|` and hence `B` (by a number around `10^{20}`).
5. **Reduction.** de Weger's lemma: an LLL reduced basis of the lattice
   spanned by the columns of `\\begin{pmatrix} I & 0 \\\\ [C\\delta_1] \\cdots
   [C\\delta_r] & [C\\delta_0] \\end{pmatrix}` bounds the vectors with a tiny
   last coordinate, so `B \\le (\\log(C K_1) - \\log S)/K_2` where the
   linear form is at most `K_1 e^{-K_2 B}`; repeated, the bound drops to a
   few dozens.
6. **Enumeration.** The exponent vectors below the reduced bound, and the
   small `|y|` for which the estimates of step 3 do not hold, are checked
   directly.

SymPy has no Thue equation solver (``diophantine`` handles binary
quadratic forms).

Two limitations, both of which raise ``NotImplementedError`` rather than
return an incomplete list: fields whose fundamental units are too large
for the search of step 1, and right-hand sides `m` which are too large
for step 2. The lattice of step 2 gets denser as `|m|` grows (its
covolume is divided by `|m|`), so the number of points to examine is
proportional to `|m|`; the budget is about a minute of work, which covers
`|m|` up to a few hundred for a cubic form. Elements of a given norm are
found in practice by factoring the ideal `(m)` into prime ideals, which
needs the maximal order and its ideal arithmetic (not implemented here).

References
==========

.. [Tzanakis] N. Tzanakis, B. M. M. de Weger, On the practical solution
   of the Thue equation, Journal of Number Theory 31 (1989).
.. [Baker] A. Baker, G. Wüstholz, Logarithmic forms and group varieties,
   Journal für die reine und angewandte Mathematik 442 (1993).
.. [Smart] N. P. Smart, The Algorithmic Resolution of Diophantine
   Equations, Cambridge (1998), chapters V and VII.
.. [Friedman] E. Friedman, Analytic formulas for the regulator of a
   number field, Inventiones Mathematicae 98 (1989).
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from itertools import product as cartesian
from math import factorial, log, sqrt, exp, pi, floor, ceil
from typing import Iterator, Optional, Sequence

import mpmath

from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.domains import ZZ

from sympy_extras._typing import as_expr

__all__ = ['thue', 'ThueEquation', 'units_of_order', 'elements_of_norm', 'lll', 'short_vectors', 'OrderElement']

#: the working precision, in decimal places, of the numerical parts
_PRECISION = 60


@contextmanager
def at_precision(digits: int) -> Iterator[None]:
    """Run the body with mpmath's precision set to ``digits`` decimal
    places, and restore the previous setting afterwards.

    The precision of mpmath is global state; every public function of this
    module raises it while it works and puts it back, so that the results
    of the other modules (the interval arithmetic of
    :mod:`sympy_extras.assumptions.intervals`, the root isolation of
    :mod:`sympy_extras.solvers.isolation`) do not depend on whether a Thue
    equation was solved before.

    Examples
    ========

    >>> import mpmath
    >>> from sympy_extras.solvers.thue_equation import at_precision
    >>> before = mpmath.mp.dps
    >>> with at_precision(before + 20):
    ...     mpmath.mp.dps == before + 20
    True
    >>> mpmath.mp.dps == before
    True
    """
    previous = mpmath.mp.dps
    mpmath.mp.dps = digits
    try:
        yield
    finally:
        mpmath.mp.dps = previous


#: an element of Z[alpha] by its integer coordinates in 1, alpha, ..., alpha**(n-1)
OrderElement = tuple[int, ...]
#: the conjugates of an element, as complex numbers
Conjugates = list[mpmath.mpc]


# ---------------------------------------------------------------------------
# arithmetic in Z[alpha]

class _Order:
    """The order ``Z[alpha]`` for a monic integer polynomial ``g``: exact
    arithmetic on coordinate vectors and numerical conjugates."""

    def __init__(self, g: list[int], precision: int = _PRECISION) -> None:
        # g = [1, c_{n-1}, ..., c_0] (leading coefficient first)
        self.g = g
        self.n = len(g) - 1
        self.precision = precision
        self.roots: Conjugates = []
        self.real_indices: list[int] = []
        self.set_precision(precision)
        # complex roots paired with their conjugates
        self.pairs: list[tuple[int, int]] = []
        taken: set[int] = set()
        for i, r in enumerate(self.roots):
            if i in self.real_indices or i in taken:
                continue
            for j, s in enumerate(self.roots):
                if j != i and j not in taken and j not in self.real_indices and abs(s - r.conjugate()) < 1e-10:
                    self.pairs.append((i, j))
                    taken.update((i, j))
                    break
        self.r1 = len(self.real_indices)
        self.r2 = len(self.pairs)
        self.rank = self.r1 + self.r2 - 1
        self.pairs_fixed = True
        # powers of alpha reduced: alpha**k for k < 2n as coordinate vectors
        self.powers: list[list[int]] = []
        current = [1] + [0]*(self.n - 1)
        for _ in range(2*self.n):
            self.powers.append(list(current))
            current = self._times_alpha(current)

    def set_precision(self, precision: int) -> None:
        """Recompute the roots at the given precision (in decimal digits),
        keeping their order."""
        self.precision = precision
        mpmath.mp.dps = precision
        computed = [mpmath.mpc(r) for r in mpmath.polyroots(self.g, maxsteps=400, extraprec=4*precision)]
        if not self.roots:
            self.roots = computed
            self.real_indices = [i for i, r in enumerate(self.roots) if abs(r.imag) < mpmath.mpf(10)**(-precision//2)]
        else:
            ordered: Conjugates = []
            remaining = list(computed)
            for old_root in self.roots:
                nearest = min(remaining, key=lambda r: abs(r - old_root))
                remaining.remove(nearest)
                ordered.append(nearest)
            self.roots = ordered
        for i in self.real_indices:
            self.roots[i] = mpmath.mpc(self.roots[i].real, 0)

    def _times_alpha(self, v: list[int]) -> list[int]:
        n = self.n
        shifted = [0] + v[:-1]
        top = v[-1]
        # alpha**n = -(c_{n-1} alpha**(n-1) + ... + c_0)
        for i in range(n):
            shifted[i] -= top*self.g[n - i]
        return shifted

    def multiply(self, a: OrderElement, b: OrderElement) -> OrderElement:
        n = self.n
        product = [0]*(2*n - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                product[i + j] += ai*bj
        result = [0]*n
        for k, coefficient in enumerate(product):
            if coefficient == 0:
                continue
            for i in range(n):
                result[i] += coefficient*self.powers[k][i]
        return tuple(result)

    def power(self, a: OrderElement, e: int) -> OrderElement:
        result: OrderElement = tuple([1] + [0]*(self.n - 1))
        base = a
        if e < 0:
            base = self.inverse_unit(a)
            e = -e
        while e:
            if e & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            e >>= 1
        return result

    def norm(self, a: OrderElement) -> int:
        """The norm, as the resultant with ``g``."""
        X = Symbol('X')
        element = Poly(sum(c*X**i for i, c in enumerate(a)), X, domain=ZZ)
        if element.is_zero:
            return 0
        g = Poly(sum(c*X**(self.n - i) for i, c in enumerate(self.g)), X, domain=ZZ)
        return int(g.resultant(element))

    def conjugates(self, a: OrderElement) -> Conjugates:
        return [sum((mpmath.mpc(c)*root**i for i, c in enumerate(a)), mpmath.mpc(0)) for root in self.roots]

    def from_conjugates(self, values: Conjugates) -> Optional[OrderElement]:
        """The element with the given conjugates, when its coordinates are
        integers (solved through the Vandermonde system and verified)."""
        n = self.n
        V = mpmath.matrix(n, n)
        for i, root in enumerate(self.roots):
            for j in range(n):
                V[i, j] = root**j
        try:
            solution = mpmath.lu_solve(V, mpmath.matrix(values))
        except ZeroDivisionError:
            return None
        coordinates: list[int] = []
        for j in range(n):
            value = solution[j]
            if abs(value.imag) > 1e-6 or abs(value.real - mpmath.nint(value.real)) > 1e-6:
                return None
            coordinates.append(int(mpmath.nint(value.real)))
        candidate = tuple(coordinates)
        check = self.conjugates(candidate)
        if any(abs(u - v) > 1e-6*(1 + abs(v)) for u, v in zip(check, values)):
            return None
        return candidate

    def inverse_unit(self, u: OrderElement) -> OrderElement:
        """The inverse of a unit (its conjugates inverted)."""
        inverse = self.from_conjugates([1/c for c in self.conjugates(u)])
        if inverse is None:
            raise ValueError("%s is not a unit" % (u,))
        return inverse

    def divides(self, a: OrderElement, b: OrderElement) -> Optional[OrderElement]:
        """``b/a`` when it lies in the order."""
        ca, cb = self.conjugates(a), self.conjugates(b)
        if any(abs(c) < mpmath.mpf(10)**(-self.precision//2) for c in ca):
            return None
        return self.from_conjugates([v/u for u, v in zip(ca, cb)])

    def log_vector(self, a: OrderElement, indices: Sequence[int]) -> list[mpmath.mpf]:
        c = self.conjugates(a)
        return [mpmath.log(abs(c[i])) for i in indices]

    def height(self, a: OrderElement) -> float:
        """The absolute logarithmic Weil height of an algebraic integer."""
        return float(sum(max(mpmath.mpf(0), mpmath.log(abs(c))) for c in self.conjugates(a))/self.n)


# ---------------------------------------------------------------------------
# units

def _independent_indices(order: _Order) -> list[int]:
    """One embedding per real root and per complex pair, all but the
    last: the rows of the regulator matrix."""
    indices = list(order.real_indices) + [i for i, _ in order.pairs]
    return indices[:-1]


def _regulator(order: _Order, units: Sequence[OrderElement]) -> mpmath.mpf:
    indices = _independent_indices(order)
    rows = []
    for u in units:
        c = order.conjugates(u)
        rows.append([(2 if i not in order.real_indices else 1)*mpmath.log(abs(c[i])) for i in indices])
    return abs(mpmath.det(mpmath.matrix(rows)))


def _small_units(order: _Order, bound: int, limit: int = 4000000) -> list[OrderElement]:
    """Units with coordinates in ``[-bound, bound]`` (a floating point
    filter on the product of the conjugates, the exact norm afterwards)."""
    found: list[OrderElement] = []
    n = order.n
    roots = [complex(r) for r in order.roots]
    count = 0
    for coordinates in cartesian(range(-bound, bound + 1), repeat=n):
        count += 1
        if count > limit:
            break
        if all(c == 0 for c in coordinates):
            continue
        product = 1.0
        for root in roots:
            value = 0j
            for c in reversed(coordinates):
                value = value*root + c
            product *= abs(value)
        if abs(product - 1.0) > 1e-6:
            continue
        element = tuple(coordinates)
        if abs(order.norm(element)) == 1:
            found.append(element)
    return found


def short_vectors(rows: Sequence[Sequence[int]], radius: float, budget: int = 0) -> list[list[int]]:
    """All the integer combinations ``c`` of the row vectors with
    ``||sum c_i rows_i|| <= radius`` (Fincke–Pohst enumeration on the
    Gram–Schmidt orthogonalisation; the rows should be LLL reduced).

    A positive ``budget`` bounds the number of vectors: the enumeration
    raises ``NotImplementedError`` instead of returning more than that.

    Examples
    ========

    >>> from sympy_extras.solvers.thue_equation import short_vectors
    >>> sorted(short_vectors([[2, 0], [0, 3]], 3.5))
    [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]]
    """
    m = len(rows)
    b = [[float(v) for v in row] for row in rows]
    star: list[list[float]] = []
    mu = [[0.0]*m for _ in range(m)]
    norms: list[float] = []
    for i in range(m):
        v = list(b[i])
        for j in range(i):
            mu[i][j] = sum(a*c for a, c in zip(b[i], star[j]))/norms[j] if norms[j] else 0.0
            v = [a - mu[i][j]*c for a, c in zip(v, star[j])]
        star.append(v)
        norms.append(sum(a*a for a in v))
    results: list[list[int]] = []
    coefficients = [0]*m
    bound = radius*radius*(1 + 1e-9)

    def search(i: int, remaining: float) -> None:
        # the i-th Gram-Schmidt coordinate is c_i + sum_{j>i} c_j mu[j][i]
        if norms[i] <= 0:
            return
        center = -sum(coefficients[j]*mu[j][i] for j in range(i + 1, m))
        span = sqrt(max(remaining, 0.0)/norms[i])
        for c in range(int(ceil(center - span)), int(floor(center + span)) + 1):
            coefficients[i] = c
            used = (c - center)**2*norms[i]
            if used > remaining*(1 + 1e-9):
                continue
            if i == 0:
                results.append(list(coefficients))
                if budget and len(results) > budget:
                    raise NotImplementedError(
                        "more than %d lattice points to examine; the enumeration of the elements "
                        "of a given norm is linear in that norm (see the module documentation)" % budget)
            else:
                search(i - 1, remaining - used)
        coefficients[i] = 0

    search(m - 1, bound)
    return results


def _minkowski_rows(order: _Order) -> list[list[mpmath.mpf]]:
    """The Minkowski embedding of the power basis: for each ``alpha**j`` the
    real conjugates and the real and imaginary parts of one conjugate per
    complex pair."""
    rows: list[list[mpmath.mpf]] = []
    for j in range(order.n):
        element = tuple(1 if i == j else 0 for i in range(order.n))
        c = order.conjugates(element)
        row = [c[i].real for i in order.real_indices]
        for i, _ in order.pairs:
            row.extend([c[i].real, c[i].imag])
        rows.append(row)
    return rows


#: the largest number of lattice points examined when looking for the
#: elements of a given norm (about a minute of work)
_ENUMERATION_BUDGET = 400000


def _elements_by_lattice(order: _Order, norm: int, limits: Sequence[float]) -> list[OrderElement]:
    """Elements of the given norm whose logarithmic vector (the logarithms
    of the conjugates minus ``log|norm|/n``) lies in the box of the
    ``limits`` (one per real embedding and per complex pair, all but the
    last which is determined): for every integer grid point the elements
    with ``|sigma_i| <= |norm|**(1/n) e**(l_i + 1)`` are the short vectors
    of the Minkowski lattice scaled accordingly, enumerated completely
    (LLL then Fincke–Pohst)."""
    embedding = _minkowski_rows(order)
    weights = [1]*order.r1 + [2]*order.r2
    places = order.r1 + order.r2
    free = places - 1
    if free <= 0:
        return []
    size = mpmath.mpf(abs(norm))**(mpmath.mpf(1)/order.n)
    found: set[OrderElement] = set()
    scale = 10**12
    budget, examined = _ENUMERATION_BUDGET, 0
    ranges = [range(-int(ceil(limit)), int(ceil(limit)) + 1) for limit in limits[:free]]
    for grid in cartesian(*ranges):
        logs = [float(v) for v in grid]
        last = -sum(w*l for w, l in zip(weights[:-1], logs))/weights[-1]
        if abs(last) > limits[-1] + 1:
            continue
        logs.append(last)
        factors: list[mpmath.mpf] = []
        for i in range(order.r1):
            factors.append(mpmath.exp(-mpmath.mpf(logs[i]))/size)
        for k in range(order.r2):
            factors.extend([mpmath.exp(-mpmath.mpf(logs[order.r1 + k]))/size]*2)
        rows = [[int(mpmath.nint(v*f*scale)) for v, f in zip(row, factors)] for row in embedding]
        augmented = [row + [scale if i == j else 0 for j in range(order.n)] for i, row in enumerate(rows)]
        reduced_rows = lll(augmented)
        coordinates_of = [[v//scale for v in row[-order.n:]] for row in reduced_rows]
        # vectors with every scaled coordinate at most e**1.5 (radius sqrt(n) e**1.5)
        radius = sqrt(order.n)*exp(1.5)*scale
        vectors = short_vectors([row[:-order.n] for row in reduced_rows], radius, budget - examined)
        examined += len(vectors)
        for combination in vectors:
            coordinates = [0]*order.n
            for c, vector in zip(combination, coordinates_of):
                if c:
                    for i in range(order.n):
                        coordinates[i] += c*vector[i]
            if all(v == 0 for v in coordinates):
                continue
            element = tuple(coordinates)
            if element in found:
                continue
            conjugates = order.conjugates(element)
            product = mpmath.mpf(1)
            for c in conjugates:
                product *= abs(c)
            if abs(product - abs(norm)) > 1e-6*abs(norm):
                continue
            if order.norm(element) == norm or (abs(norm) == 1 and abs(order.norm(element)) == 1):
                found.add(element)
    return sorted(found)


def _units_by_lattice(order: _Order, L: int) -> list[OrderElement]:
    """Units whose logarithmic vector lies in ``[-L, L]`` in the free
    coordinates."""
    return _elements_by_lattice(order, 1, [float(L)]*(order.r1 + order.r2))


def _torsion(order: _Order, candidates: Sequence[OrderElement]) -> list[OrderElement]:
    """The roots of unity among the candidates (all conjugates of modulus
    one), as the group they generate."""
    roots: list[OrderElement] = []
    for u in candidates:
        if all(abs(abs(c) - 1) < 1e-10 for c in order.conjugates(u)):
            roots.append(u)
    group: set[OrderElement] = {tuple([1] + [0]*(order.n - 1))}
    for u in roots:
        current = u
        for _ in range(60):
            if current in group:
                break
            group.add(current)
            current = order.multiply(current, u)
    return sorted(group)


def _independent_system(order: _Order, units: Sequence[OrderElement]) -> Optional[list[OrderElement]]:
    """``rank`` units with independent logarithmic vectors, of small
    height."""
    indices = _independent_indices(order)
    ordered = sorted(units, key=lambda u: (order.height(u), u))
    chosen: list[OrderElement] = []
    rows: list[list[mpmath.mpf]] = []
    for u in ordered:
        row = order.log_vector(u, indices)
        if all(abs(v) < 1e-12 for v in row):
            continue
        trial = rows + [row]
        M = mpmath.matrix(trial)
        # the rank through the Gram determinant
        gram = M*M.T
        if abs(mpmath.det(gram)) > 1e-12:
            rows.append(row)
            chosen.append(u)
            if len(chosen) == order.rank:
                return chosen
    return None


def _reduce_system(order: _Order, units: list[OrderElement]) -> list[OrderElement]:
    """The system replaced by an LLL reduced basis of the same lattice of
    logarithmic vectors (smaller heights, better constants)."""
    indices = _independent_indices(order)
    scale = 10**12
    rows = []
    for u in units:
        rows.append([ZZ(int(mpmath.nint(v*scale))) for v in order.log_vector(u, indices)])
    r = len(units)
    # the transformation is tracked by appending the identity
    augmented = [row + [ZZ(1 if i == j else 0) for j in range(r)] for i, row in enumerate(rows)]
    reduced = lll([[int(v) for v in row] for row in augmented])
    result: list[OrderElement] = []
    for i in range(r):
        element: OrderElement = tuple([1] + [0]*(order.n - 1))
        for j in range(r):
            e = int(reduced[i][len(indices) + j])
            if e:
                element = order.multiply(element, order.power(units[j], e))
        result.append(element)
    return result


def _pth_root(order: _Order, u: OrderElement, p: int) -> Optional[OrderElement]:
    """A unit ``v`` with ``v**p == u`` (tried on the branches of the
    roots of the conjugates: real roots for real conjugates, the ``p``
    branches for the first of each complex pair)."""
    c = order.conjugates(u)
    branches: list[list[mpmath.mpc]] = []
    for i in range(order.n):
        if i in order.real_indices:
            value = c[i].real
            root = mpmath.mpf(abs(value))**(mpmath.mpf(1)/p)
            branches.append([mpmath.mpc(root)] if value > 0 else ([mpmath.mpc(-root)] if p % 2 == 1 else []))
        else:
            branches.append([c[i]**(mpmath.mpf(1)/p)*mpmath.exp(2*mpmath.pi*1j*k/p) for k in range(p)])
    if any(not b for b in branches):
        return None
    # the conjugate pairs must stay conjugate: choose branches for the
    # first of each pair, the second follows
    first_of_pair = {i: j for i, j in order.pairs}
    second_of_pair = {j: i for i, j in order.pairs}
    free = [i for i in range(order.n) if i not in second_of_pair]
    for choice in cartesian(*[range(len(branches[i])) for i in free]):
        values: list[mpmath.mpc] = [mpmath.mpc(0)]*order.n
        for i, k in zip(free, choice):
            values[i] = branches[i][k]
            if i in first_of_pair:
                values[first_of_pair[i]] = branches[i][k].conjugate()
        candidate = order.from_conjugates(values)
        if candidate is not None and order.power(candidate, p) == u:
            return candidate
    return None


def _saturate(order: _Order, units: list[OrderElement], torsion: Sequence[OrderElement]) -> list[OrderElement]:
    """Enlarge the system until its index in the unit group is 1: for
    every prime up to ``R'/0.2052`` (Friedman's bound) the p-th roots of
    the products of the units (and roots of unity) are searched."""
    from sympy.ntheory import primerange
    while True:
        regulator = _regulator(order, units)
        index_bound = int(floor(regulator/mpmath.mpf('0.2052')))
        enlarged = False
        for p in primerange(2, index_bound + 1):
            if p**len(units) > 2000000:
                raise NotImplementedError("the saturation of the unit group needs the prime %d" % p)
            for exponents in cartesian(range(p), repeat=len(units)):
                if all(e == 0 for e in exponents):
                    continue
                if exponents[[i for i, e in enumerate(exponents) if e][0]] != 1:
                    continue  # one representative per line through the origin
                base: OrderElement = tuple([1] + [0]*(order.n - 1))
                for u, e in zip(units, exponents):
                    if e:
                        base = order.multiply(base, order.power(u, e))
                for zeta in torsion:
                    root = _pth_root(order, order.multiply(base, zeta), p)
                    if root is not None:
                        k = [i for i, e in enumerate(exponents) if e][0]
                        units[k] = root
                        enlarged = True
                        break
                if enlarged:
                    break
            if enlarged:
                break
        if not enlarged:
            return _reduce_system(order, units)


def units_of_order(g: Sequence[int], bound: int = 64) -> tuple[list[OrderElement], list[OrderElement]]:
    """A fundamental system of units and the roots of unity of
    ``Z[alpha]``, ``alpha`` a root of the monic integer polynomial ``g``
    (coefficients from the leading one), as coordinate vectors in
    ``1, alpha, ..., alpha**(n-1)``.

    Examples
    ========

    >>> from sympy_extras.solvers.thue_equation import units_of_order
    >>> units, torsion = units_of_order([1, 0, 0, -2])      # Z[cbrt(2)]: cbrt(2) - 1 or its inverse
    >>> units in ([(1, -1, 0)], [(1, 1, 1)]), torsion
    (True, [(-1, 0, 0), (1, 0, 0)])
    """
    with at_precision(_PRECISION):
        order = _Order([int(c) for c in g])
        if order.rank == 0:
            candidates = _small_units(order, 2)
            return [], _torsion(order, candidates)
        candidates = _small_units(order, 3)
        torsion = _torsion(order, candidates)
        system = _independent_system(order, candidates)
        L = 4
        while system is None and L <= bound:
            candidates = candidates + _units_by_lattice(order, L)
            system = _independent_system(order, candidates)
            L *= 2
        if system is None:
            raise NotImplementedError("no system of %d independent units with logarithms up to %d"
                                      % (order.rank, bound))
        return [_normalized(u) for u in _saturate(order, _reduce_system(order, system), torsion)], torsion


def _normalized(element: OrderElement) -> OrderElement:
    """The element or its negative, with a positive first nonzero
    coordinate (a canonical choice among the two signs)."""
    for c in element:
        if c != 0:
            return element if c > 0 else tuple(-v for v in element)
    return element


# ---------------------------------------------------------------------------
# elements of a given norm

def elements_of_norm(g: Sequence[int], norm: int, units: Sequence[OrderElement], limit: int = 100000
                     ) -> list[OrderElement]:
    """Representatives of the elements of ``Z[alpha]`` of the given norm
    modulo the units (one per class of associates).

    Examples
    ========

    >>> from sympy_extras.solvers.thue_equation import elements_of_norm, units_of_order
    >>> units, _ = units_of_order([1, 0, 0, -2])
    >>> elements_of_norm([1, 0, 0, -2], 1, units)
    [(1, 0, 0)]
    >>> elements_of_norm([1, 0, 0, -2], 2, units)
    [(0, 1, 0)]
    """
    with at_precision(_PRECISION):
        order = _Order([int(c) for c in g])
        return _elements_of_norm(order, norm, units, limit)


def _elements_of_norm(order: _Order, norm: int, units: Sequence[OrderElement], limit: int) -> list[OrderElement]:
    n = order.n
    # a representative of every class has its logarithmic vector in the
    # fundamental domain of the unit lattice: |log|beta_i| - log|N|/n| <=
    # sum_k |log|eps_k^(i)||/2
    L = [mpmath.mpf(0)]*n
    for u in units:
        c = order.conjugates(u)
        for i in range(n):
            L[i] += abs(mpmath.log(abs(c[i])))/2
    limits = [float(L[i]) for i in order.real_indices] + [float(L[i]) for i, _ in order.pairs]
    total = 1
    for limit_ in limits[:-1]:
        total *= 2*int(ceil(limit_)) + 1
    if total > limit:
        raise NotImplementedError("too many lattice boxes (%d) for the elements of norm %d" % (total, norm))
    found = _elements_by_lattice(order, norm, limits)
    representatives: list[OrderElement] = []
    for element in sorted(found, key=lambda e: (sum(abs(c) for c in e), e)):
        if any(_associates(order, element, other) for other in representatives):
            continue
        representatives.append(_normalized(element))
    return representatives


def _associates(order: _Order, a: OrderElement, b: OrderElement) -> bool:
    quotient = order.divides(a, b)
    return quotient is not None and abs(order.norm(quotient)) == 1


# ---------------------------------------------------------------------------
# the linear form, Baker's bound and the reduction

def _baker_wustholz(k: int, d: int, heights: Sequence[float]) -> float:
    """The constant ``C`` with ``log|Lambda| >= -C log B`` for a linear
    form in ``k`` logarithms of algebraic numbers of a field of degree
    at most ``d``, with the modified heights ``h'``."""
    c = 18*factorial(k + 1)*k**(k + 1)*(32*d)**(k + 2)*log(2*k*d)
    for h in heights:
        c *= h
    return c


def lll(rows: Sequence[Sequence[int]], delta: Fraction = Fraction(3, 4)) -> list[list[int]]:
    """An LLL reduced basis of the lattice spanned by the integer row
    vectors (exact arithmetic on fractions; SymPy's ``DomainMatrix.lll``
    fails on very large entries).

    Examples
    ========

    >>> from sympy_extras.solvers.thue_equation import lll
    >>> lll([[1, 0, 3], [0, 1, 5], [0, 0, 7]])
    [[-1, -1, -1], [-1, 2, 0], [-2, 0, 1]]
    """
    b = [[int(v) for v in row] for row in rows]
    m = len(b)

    def dot(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
        return sum((a*c for a, c in zip(u, v)), Fraction(0))

    def gram_schmidt() -> tuple[list[list[Fraction]], list[list[Fraction]], list[Fraction]]:
        star: list[list[Fraction]] = []
        mu: list[list[Fraction]] = [[Fraction(0)]*m for _ in range(m)]
        norms: list[Fraction] = []
        for i in range(m):
            v = [Fraction(c) for c in b[i]]
            for j in range(i):
                mu[i][j] = dot([Fraction(c) for c in b[i]], star[j])/norms[j] if norms[j] else Fraction(0)
                v = [a - mu[i][j]*c for a, c in zip(v, star[j])]
            star.append(v)
            norms.append(dot(v, v))
        return star, mu, norms

    star, mu, norms = gram_schmidt()
    k = 1
    while k < m:
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                b[k] = [a - q*c for a, c in zip(b[k], b[j])]
                star, mu, norms = gram_schmidt()
        if norms[k] >= (delta - mu[k][k - 1]**2)*norms[k - 1]:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            star, mu, norms = gram_schmidt()
            k = max(k - 1, 1)
    return b


def _lll_reduce(deltas: Sequence[mpmath.mpf], delta0: mpmath.mpf, B0: float, K1: float, K2: float) -> Optional[float]:
    """de Weger's reduction (the inhomogeneous lemma): a new bound on
    ``B`` for ``|delta0 + sum b_i delta_i| <= K1 exp(-K2 B)`` with
    ``|b_i| <= B0``, or ``None`` when the lattice is not large enough.

    The lattice is generated by the columns of ``[[I, 0], [C delta_1 ...
    C delta_{s-1}, C delta_s]]`` (rounded) and ``y = (0, ..., 0, -[C delta0])``;
    for an LLL reduced basis ``b_1, ..., b_s`` and ``y = sum sigma_i b_i``,
    every lattice vector is at distance at least ``2**(-(s-1)/2) ||sigma_i0||
    ||b_1||`` from ``y``, ``i0`` the last index with a non-integral
    ``sigma`` (``||.||`` the distance to the nearest integer)."""
    from sympy.matrices.dense import Matrix
    from sympy.core.numbers import Rational
    s = len(deltas)
    for factor in (100, 10**4, 10**6):
        C = (mpmath.mpf(B0)*factor)**s
        rows: list[list[int]] = []
        for i in range(s - 1):
            rows.append([1 if j == i else 0 for j in range(s - 1)] + [int(mpmath.nint(C*deltas[i]))])
        rows.append([0]*(s - 1) + [int(mpmath.nint(C*deltas[s - 1]))])
        y = [0]*(s - 1) + [-int(mpmath.nint(C*delta0))]
        reduced = lll(rows)
        basis = Matrix(reduced)          # rows are the basis vectors
        try:
            sigma = basis.T.LUsolve(Matrix(y))
        except ValueError:
            continue
        distances = [abs(Rational(v) - round(Rational(v))) for v in sigma]
        nonintegral = [i for i, d in enumerate(distances) if d != 0]
        if not nonintegral:
            continue
        i0 = nonintegral[-1]
        length = mpmath.sqrt(sum(mpmath.mpf(v)**2 for v in reduced[0]))
        D = mpmath.mpf(2)**(-mpmath.mpf(s - 1)/2)*mpmath.mpf(distances[i0].p)/mpmath.mpf(distances[i0].q)*length
        inside = D**2 - (s - 1)*mpmath.mpf(B0)**2
        if inside <= 0:
            continue
        S = mpmath.sqrt(inside) - (s*B0 + 1)/mpmath.mpf(2)
        if S <= 0:
            continue
        bound = (mpmath.log(C*K1) - mpmath.log(S))/K2
        return float(bound)
    return None


class ThueEquation:
    """The data of ``F(x, y) = m`` prepared for the resolution: the order,
    its units, the elements of the right norm.

    Attributes
    ==========

    coefficients : list of int
        ``[a_0, ..., a_n]`` of ``F``.
    m : int
    units, torsion : lists of OrderElement
    representatives : list of OrderElement
        Elements of norm ``a_0**(n-1) m`` modulo the units.
    """

    def __init__(self, coefficients: Sequence[int], m: int, bound: int = 64) -> None:
        self.coefficients = [int(c) for c in coefficients]
        self.m = int(m)
        self.n = len(self.coefficients) - 1
        a0 = self.coefficients[0]
        if a0 == 0 or self.n < 3:
            raise ValueError("a form of degree at least 3 with a nonzero leading coefficient is expected")
        self.a0 = a0
        # g(X) = a0**(n-1) F(X/a0, 1): coefficient of X**(n-i) is a_i a0**(i-1)
        self.g = [1] + [self.coefficients[i]*a0**(i - 1) for i in range(1, self.n + 1)]
        with at_precision(_PRECISION):
            self.order = _Order(self.g)
            self.norm = a0**(self.n - 1)*self.m
            self.units, self.torsion = units_of_order(self.g, bound)
            self.representatives = _elements_of_norm(self.order, self.norm, self.units, 100000)

    def evaluate(self, x: int, y: int) -> int:
        total = 0
        for i, c in enumerate(self.coefficients):
            total += c*x**(self.n - i)*y**i
        return total

    def _solution_from_element(self, beta: OrderElement) -> Optional[tuple[int, int]]:
        """``(x, y)`` with ``beta = a0 x - alpha y``."""
        if any(c != 0 for c in beta[2:]):
            return None
        if beta[0] % self.a0 != 0:
            return None
        x, y = beta[0]//self.a0, -beta[1]
        return (x, y) if self.evaluate(x, y) == self.m else None

    def _form_at_precision(self, i0: int, mu: OrderElement, j: int, k: int, complex_pair: bool
                           ) -> tuple[list[mpmath.mpf], mpmath.mpf]:
        return _form_coefficients(self.order, self.units, mu, i0, j, k, complex_pair)

    def small_solutions(self, Y: int) -> list[tuple[int, int]]:
        """The solutions with ``|y| <= Y``, by the integer roots of
        ``F(X, y) - m``."""
        X = Symbol('X')
        found: list[tuple[int, int]] = []
        for y in range(-Y, Y + 1):
            polynomial = sum(c*X**(self.n - i)*y**i for i, c in enumerate(self.coefficients)) - self.m
            try:
                poly = Poly(polynomial, X, domain=ZZ)
            except PolynomialError:
                continue
            if poly.degree() < 1:
                continue
            for root, _ in poly.ground_roots().items():
                if root.is_Integer:
                    found.append((int(root), y))
        return found

    def solve(self) -> list[tuple[int, int]]:
        """All the integer solutions."""
        with at_precision(self.order.precision):
            return self._solve()

    def _solve(self) -> list[tuple[int, int]]:
        order = self.order
        n = self.n
        roots = order.roots
        N0 = abs(self.norm)
        solutions: set[tuple[int, int]] = set()
        # no real root: |a0 x - alpha y| >= |Im alpha| |y| for every root
        if order.r1 == 0:
            product = mpmath.mpf(1)
            for r in roots:
                product *= abs(r.imag)
            Y = int(floor((mpmath.mpf(N0)/product)**(mpmath.mpf(1)/n))) + 1
            return sorted(set(self.small_solutions(Y)))
        r = order.rank
        if r == 0:
            # impossible for n >= 3 with a real root (r1 + r2 >= 2)
            raise NotImplementedError("unit rank zero")
        # the size below which the estimates may fail
        size = mpmath.mpf(N0)**(mpmath.mpf(1)/n)
        Y1 = mpmath.mpf(1)
        imaginary = [abs(r.imag) for i, r in enumerate(roots) if i not in order.real_indices]
        if imaginary:
            Y1 = max(Y1, size/min(imaginary))
        for i0 in order.real_indices:
            d = min(abs(roots[j] - roots[i0]) for j in range(n) if j != i0)
            Y1 = max(Y1, 2*size/d)
        bound_B = 0.0
        forms: list[tuple[int, OrderElement, list[mpmath.mpf], mpmath.mpf, float, float, list[int], bool]] = []
        for i0 in order.real_indices:
            others = [j for j in range(n) if j != i0]
            D = {j: abs(roots[j] - roots[i0]) for j in others}
            d = min(D.values())
            c1 = mpmath.mpf(N0)*mpmath.mpf(2)**(n - 1)
            for j in others:
                c1 /= D[j]
            # the pair (j, k): the arguments for a complex pair, the moduli
            # otherwise; a pair for which every unit gives a nonzero
            # coefficient of the linear form
            chosen: Optional[tuple[int, int, bool]] = None
            candidates_jk: list[tuple[int, int, bool]] = [(a, b, True) for a, b in order.pairs]
            candidates_jk += [(others[a], others[b], False) for a in range(len(others)) for b in range(len(others))
                              if a != b]
            for j_, k_, pair_ in candidates_jk:
                coefficients_, _ = _form_coefficients(order, self.units, self.representatives[0] if self.representatives
                                                      else tuple([1] + [0]*(n - 1)), i0, j_, k_, pair_)
                if all(abs(c) > mpmath.mpf(10)**(-order.precision//3) for c in coefficients_):
                    chosen = (j_, k_, pair_)
                    break
            if chosen is None:
                raise NotImplementedError("no linear form with nonzero coefficients for the root %d" % i0)
            j, k, complex_pair = chosen
            c6 = 2*abs(roots[j] - roots[k])*c1/(D[j]*D[k])
            Y1 = max(Y1, (2*c6)**(mpmath.mpf(1)/n))
            # b from the conjugates in J (one per pair, all but i0)
            J = [idx for idx in _independent_indices_excluding(order, i0)]
            M = mpmath.matrix(len(J), r)
            for a, idx in enumerate(J):
                for b, u in enumerate(self.units):
                    M[a, b] = mpmath.log(abs(order.conjugates(u)[idx]))
            Minv = mpmath.inverse(M)
            c4 = max(sum(abs(Minv[a, b]) for b in range(r)) for a in range(r))
            c8 = max(max(mpmath.log(abs(roots[i0]) + d/2 + abs(roots[idx])), abs(mpmath.log(D[idx]/2))) for idx in J)
            for mu in self.representatives:
                cmu = order.conjugates(mu)
                c3 = c4*(c8 + max(abs(mpmath.log(abs(cmu[idx]))) for idx in J))
                gamma0 = (roots[k] - roots[i0])*cmu[j]/((roots[j] - roots[i0])*cmu[k])
                gammas = [order.conjugates(u)[j]/order.conjugates(u)[k] for u in self.units]
                deltas, delta0 = _form_coefficients(order, self.units, mu, i0, j, k, complex_pair)
                # heights: h(gamma0) <= 2 (2 h(alpha) + log 2) + 2 h(mu), h(gamma_l) <= 2 h(eps_l)
                h_alpha = order.height(tuple([0, 1] + [0]*(n - 2)))
                degree = factorial(n)
                h0 = 2*(2*h_alpha + log(2)) + 2*order.height(mu)
                heights = [max(h0, float(abs(mpmath.log(gamma0)))/degree, 1.0/degree)]
                for u, gl in zip(self.units, gammas):
                    heights.append(max(2*order.height(u), float(abs(mpmath.log(gl)))/degree, 1.0/degree))
                if complex_pair:
                    heights.append(max(0.0, pi/degree, 1.0/degree))   # log(-1)
                    kk = r + 2
                    extra = 1  # the coefficient of 2 pi
                else:
                    kk = r + 1
                    extra = 0
                C = _baker_wustholz(kk, degree, heights)
                # log|y| <= (log(2 c6) + C log B')/n with B' <= c3 + c4 log|y| (+ the 2 pi term)
                L = 1e40
                for _ in range(200):
                    Bp = float(c3) + float(c4)*L
                    if extra:
                        Bp = r*Bp + 2
                    new = (float(mpmath.log(2*c6)) + C*max(1.0, log(max(Bp, 3.0))))/n
                    if abs(new - L) < 1e-6*max(1.0, L):
                        L = new
                        break
                    L = new
                B0 = float(c3) + float(c4)*L
                if extra:
                    B0 = r*B0 + 2
                K1 = float(2*c6*mpmath.exp(n*c3/c4))
                K2 = float(n/c4)
                if extra:
                    # B' = r B + 2 with B <= c3 + c4 L: |Lambda| <= 2c6 exp(-n L) and L >= (B' - 2)/(r c4) - c3/c4
                    K1 = float(2*c6*mpmath.exp(n*(c3/c4 + 2/(r*c4))))
                    K2 = float(n/(r*c4))
                forms.append((i0, mu, deltas, delta0, K1, K2, [j, k], complex_pair))
                bound_B = max(bound_B, B0)
        # the reduction, on every form, from the common initial bound; the
        # logarithms are recomputed at the precision the lattice needs
        current = bound_B
        base_precision = order.precision
        for _ in range(12):
            best = 0.0
            digits = int((r + 2)*(log(max(current, 10.0), 10) + 6)) + 40
            order.set_precision(max(base_precision, digits))
            for i0, mu, _, _, K1, K2, jk, complex_pair in forms:
                j, k = jk
                deltas, delta0 = self._form_at_precision(i0, mu, j, k, complex_pair)
                reduced = _lll_reduce(deltas, delta0, current, K1, K2)
                if reduced is None:
                    reduced = current
                best = max(best, reduced)
            if best >= current*0.9:
                current = best
                break
            current = best
        order.set_precision(base_precision)
        B_final = int(ceil(current)) + 1
        if B_final > 200:
            raise NotImplementedError("the reduction of the exponent bound stopped at %d" % B_final)
        # small |y|
        for x, y in self.small_solutions(int(floor(Y1)) + 1):
            solutions.add((x, y))
        # the exponent vectors
        for mu in self.representatives:
            for exponents in cartesian(range(-B_final, B_final + 1), repeat=r):
                element = mu
                for u, e in zip(self.units, exponents):
                    if e:
                        element = order.multiply(element, order.power(u, e))
                for zeta in self.torsion:
                    beta = order.multiply(element, zeta)
                    found = self._solution_from_element(beta)
                    if found is not None:
                        solutions.add(found)
        return sorted(solutions)


def _form_coefficients(order: _Order, units: Sequence[OrderElement], mu: OrderElement, i0: int, j: int, k: int,
                       complex_pair: bool) -> tuple[list[mpmath.mpf], mpmath.mpf]:
    roots = order.roots
    cmu = order.conjugates(mu)
    gamma0 = (roots[k] - roots[i0])*cmu[j]/((roots[j] - roots[i0])*cmu[k])
    gammas = [order.conjugates(u)[j]/order.conjugates(u)[k] for u in units]
    if complex_pair:
        return [mpmath.arg(gl) for gl in gammas] + [2*mpmath.pi], mpmath.arg(gamma0)
    return [mpmath.log(abs(gl)) for gl in gammas], mpmath.log(abs(gamma0))


def _independent_indices_excluding(order: _Order, i0: int) -> list[int]:
    indices = [i for i in order.real_indices if i != i0] + [i for i, _ in order.pairs]
    return indices


def thue(F: Expr, m: int, x: Symbol, y: Symbol, bound: int = 64) -> list[tuple[int, int]]:
    """All the integer solutions of the Thue equation ``F(x, y) = m`` for
    an irreducible binary form of degree at least 3.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.solvers.thue_equation import thue
    >>> thue(x**3 - 2*y**3, 1, x, y)
    [(-1, -1), (1, 0)]
    >>> thue(x**3 - 3*x*y**2 - y**3, 1, x, y)
    [(-3, 2), (-1, 1), (0, -1), (1, -3), (1, 0), (2, 1)]
    """
    poly = Poly(as_expr(F), x, y)
    n = poly.total_degree()
    if not poly.is_homogeneous or n < 3:
        raise ValueError("a homogeneous form of degree at least 3 is expected")
    if not all(c.is_Integer for c in poly.coeffs()):
        raise ValueError("integer coefficients are expected")
    univariate = Poly(as_expr(F).subs(y, 1), x, domain=ZZ)
    if not univariate.is_irreducible or univariate.degree() != n:
        raise ValueError("an irreducible form with a nonzero leading coefficient is expected")
    coefficients = [int(poly.coeff_monomial(x**(n - i)*y**i)) for i in range(n + 1)]
    if int(m) == 0:
        return [(0, 0)]
    equation = ThueEquation(coefficients, int(m), bound)
    return equation.solve()


def _unused(*args: object) -> None:
    del args


_unused(Integer, sqrt, exp)
