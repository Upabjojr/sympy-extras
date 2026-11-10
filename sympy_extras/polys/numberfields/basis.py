r"""Rings of integers of number fields: integral bases and field
discriminants by the Round 2 algorithm.

For a number field $K = \mathbb{Q}(\theta)$, $\theta$ a root of an
irreducible $T \in \mathbb{Z}[x]$, the ring of integers $\mathbb{Z}_K$
is a free $\mathbb{Z}$-module of rank $n = \deg T$ which contains the
order $\mathbb{Z}[\theta]$ (for monic $T$) with a finite index $f$, and
$\operatorname{disc}(T) = f^2 d_K$. Only the primes whose square divides
the discriminant of $T$ can divide the index, and the order is enlarged
at each of them in turn (Zassenhaus' Round 2, after Pohst and Zassenhaus;
Cohen, Algorithm 6.1.8):

1. the $p$-radical $I_p$ of the current order $O$ (the elements with a
   power in $pO$) is the kernel of the $\mathbb{F}_p$-linear map $x
   \mapsto x^{q}$ on $O/pO$, $q$ a power of $p$ not below $n$;
2. $O' = \{x \in K : x I_p \subset I_p\}$ is an order containing $O$,
   equal to $O$ exactly when $O$ is $p$-maximal (the criterion of Pohst
   and Zassenhaus); it is $\frac1p U + O$ for the kernel $U$ of the map
   $O/pO \to \operatorname{End}(I_p/pI_p)$ given by the multiplications;
3. the step is repeated with $O'$ until nothing changes.

Everything is linear algebra over $\mathbb{F}_p$ and Hermite normal
forms over $\mathbb{Z}$ (:mod:`sympy_extras.polys.lattice`), with the
orders given by their multiplication tables.

Examples
========

>>> from sympy_extras.polys.numberfields.basis import integral_basis, field_discriminant
>>> integral_basis([1, 0, -5])                 # 1, (1 + sqrt 5)/2
[[Fraction(1, 1)], [Fraction(1, 2), Fraction(1, 2)]]
>>> field_discriminant([1, 0, -5]), field_discriminant([1, 0, 1]), field_discriminant([1, 0, 0, -2])
(5, -4, -108)
>>> integral_basis([1, -1, -2, -8])            # Dedekind's cubic: 1, theta, (theta**2 + theta)/2
[[Fraction(1, 1)], [Fraction(1, 1), Fraction(0, 1)], [Fraction(1, 2), Fraction(1, 2), Fraction(0, 1)]]
>>> field_discriminant([1, -1, -2, -8])
-503

References
==========

.. [1] H. Cohen, *A Course in Computational Algebraic Number Theory*,
       Springer, 1993, Section 6.1 (the Round 2 algorithm, Algorithm
       6.1.8; the Pohst-Zassenhaus theorem, 6.1.3).
.. [2] M. Pohst, H. Zassenhaus, *Algorithmic Algebraic Number Theory*,
       Cambridge University Press, 1989, Section 4.6.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd

from sympy_extras.ntheory import factorint
from sympy_extras.polys.domains import ZZ
from sympy_extras.polys.euclidtools import dup_resultant
from sympy_extras.polys.lattice import hermite_normal_form

__all__ = ['integral_basis', 'field_discriminant', 'polynomial_discriminant']


def polynomial_discriminant(f: list[int]) -> int:
    """The discriminant of an integer polynomial (highest degree first):
    ``(-1)**(n (n - 1)/2) Res(f, f')/lc(f)``.

    Examples
    ========

    >>> from sympy_extras.polys.numberfields.basis import polynomial_discriminant
    >>> polynomial_discriminant([1, 0, -5]), polynomial_discriminant([1, 0, 0, -2]), polynomial_discriminant([2, 0, 1])
    (20, -108, -8)
    """
    n = len(f) - 1
    if n < 1:
        raise ValueError('a polynomial of positive degree is needed')
    derivative = [c*(n - k) for k, c in enumerate(f[:-1])]
    resultant = dup_resultant(f, derivative, ZZ)
    value = resultant//f[0]
    return -value if (n*(n - 1)//2) % 2 else value


def _kernel_mod(rows: list[list[int]], width: int, p: int) -> list[list[int]]:
    """A basis of ``{x in F_p**m : x . rows = 0}`` for the ``m`` rows of
    ``width`` entries: the left kernel modulo ``p``."""
    m = len(rows)
    # eliminate on [rows | identity]
    work = [[v % p for v in row] + [1 if k == i else 0 for k in range(m)] for i, row in enumerate(rows)]
    rank = 0
    for column in range(width):
        pivot = next((i for i in range(rank, m) if work[i][column]), -1)
        if pivot < 0:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        inverse = pow(work[rank][column], -1, p)
        work[rank] = [v*inverse % p for v in work[rank]]
        for i in range(m):
            if i != rank and work[i][column]:
                factor = work[i][column]
                work[i] = [(a - factor*b) % p for a, b in zip(work[i], work[rank])]
        rank += 1
    return [row[width:] for row in work[rank:]]


class _Order:
    """An order of ``QQ[x]/(T)``, ``T`` monic: the rows of ``basis``
    divided by ``denominator`` are a ``ZZ``-basis in the powers ``1, x,
    ..., x**(n-1)`` (lowest first)."""

    def __init__(self, modulus: list[int], basis: list[list[int]], denominator: int) -> None:
        self.modulus = modulus          # monic, highest degree first
        self.n = len(modulus) - 1
        common = denominator
        for row in basis:
            for v in row:
                common = gcd(common, v)
        self.basis = [[v//common for v in row] for row in basis]
        self.denominator = denominator//common
        # the inverse of the basis matrix, to read coordinates
        self._inverse = _inverse_matrix(self.basis)
        self._table: list[list[list[int]]] = []

    def _multiply(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        """The product of two elements given in the powers of ``x``."""
        n = self.n
        product = [Fraction(0)]*(2*n - 1)
        for i, u in enumerate(a):
            if u:
                for j, v in enumerate(b):
                    product[i + j] += u*v
        # x**n = -(c_(n-1) x**(n-1) + ... + c_0) for T = x**n + c_(n-1) x**(n-1) + ...
        for k in range(2*n - 2, n - 1, -1):
            c = product[k]
            if c:
                for j in range(n):
                    product[k - n + j] -= c*self.modulus[n - j]
        return product[:n]

    def coordinates(self, element: list[Fraction]) -> list[Fraction]:
        """The coordinates in the basis of the order."""
        d = self.denominator
        return [sum((element[k]*self._inverse[k][i] for k in range(self.n)), Fraction(0))*d for i in range(self.n)]

    def table(self) -> list[list[list[int]]]:
        """``table[i][j]``: the integer coordinates of the product of the
        basis elements ``i`` and ``j``."""
        if not self._table:
            elements = [[Fraction(v, self.denominator) for v in row] for row in self.basis]
            for i in range(self.n):
                line: list[list[int]] = []
                for j in range(self.n):
                    coordinates = self.coordinates(self._multiply(elements[i], elements[j]))
                    if any(c.denominator != 1 for c in coordinates):
                        raise ArithmeticError('the module is not closed under multiplication')
                    line.append([int(c) for c in coordinates])
                self._table.append(line)
        return self._table

    def product(self, a: list[int], b: list[int], p: int) -> list[int]:
        """The product of two elements given by their coordinates, the
        coordinates of the result modulo ``p`` (not reduced for ``p =
        0``)."""
        table = self.table()
        result = [0]*self.n
        for i, u in enumerate(a):
            if u:
                for j, v in enumerate(b):
                    if v:
                        c = u*v
                        row = table[i][j]
                        for k in range(self.n):
                            result[k] += c*row[k]
        return [v % p for v in result] if p else result

    def enlarged(self, p: int) -> _Order:
        """The order ``{x : x I_p in I_p}``, which is this one exactly
        when it is ``p``-maximal."""
        n = self.n
        q = p
        while q < n:
            q *= p
        # the images of the basis under x -> x**q in O/pO
        images: list[list[int]] = []
        for i in range(n):
            one = self.coordinates([Fraction(1)] + [Fraction(0)]*(n - 1))
            power = [int(c) % p for c in one]
            base = [1 if k == i else 0 for k in range(n)]
            e = q
            while e:
                if e & 1:
                    power = self.product(power, base, p)
                e >>= 1
                if e:
                    base = self.product(base, base, p)
            images.append(power)
        radical = _kernel_mod(images, n, p)
        # I_p = radical + p O, in the coordinates of the order
        ideal = hermite_normal_form(radical + [[p if k == i else 0 for k in range(n)] for i in range(n)])
        ideal_inverse = _inverse_matrix(ideal)
        # the multiplication by each basis element on I_p/p I_p
        maps: list[list[int]] = []
        for j in range(n):
            unit = [1 if k == j else 0 for k in range(n)]
            flat: list[int] = []
            for gamma in ideal:
                image = self.product(unit, gamma, 0)
                coordinates = [sum((image[k]*ideal_inverse[k][i] for k in range(n)), Fraction(0)) for i in range(n)]
                if any(c.denominator != 1 for c in coordinates):
                    raise ArithmeticError('the radical is not an ideal')
                flat.extend(int(c) % p for c in coordinates)
            maps.append(flat)
        kernel = _kernel_mod(maps, n*n, p)
        # O' = O + (1/p) kernel: generators over the denominator p d
        generators = [[p*v for v in row] for row in self.basis]
        for u in kernel:
            generators.append([sum(u[i]*self.basis[i][k] for i in range(n)) for k in range(n)])
        return _Order(self.modulus, hermite_normal_form(generators), p*self.denominator)

    def index_data(self) -> tuple[list[list[int]], int]:
        return self.basis, self.denominator


def _inverse_matrix(rows: list[list[int]]) -> list[list[Fraction]]:
    n = len(rows)
    work = [[Fraction(v) for v in row] + [Fraction(1 if k == i else 0) for k in range(n)] for i, row in enumerate(rows)]
    for column in range(n):
        pivot = next((i for i in range(column, n) if work[i][column]), -1)
        if pivot < 0:
            raise ArithmeticError('a singular basis')
        work[column], work[pivot] = work[pivot], work[column]
        inverse = 1/work[column][column]
        work[column] = [v*inverse for v in work[column]]
        for i in range(n):
            if i != column and work[i][column]:
                factor = work[i][column]
                work[i] = [a - factor*b for a, b in zip(work[i], work[column])]
    return [row[n:] for row in work]


def _maximal_order(minpoly: list[int]) -> tuple[_Order, int, int]:
    """The maximal order of ``QQ[x]/(T)`` for the monic polynomial ``T``
    with the roots ``lc theta``; with ``lc`` and the discriminant of
    ``T``."""
    n = len(minpoly) - 1
    if n < 1:
        raise ValueError('a polynomial of positive degree is needed')
    if minpoly[0] < 0:
        minpoly = [-c for c in minpoly]
    lc = minpoly[0]
    # the monic polynomial of lc theta
    monic = [c*lc**(k - 1) if k else 1 for k, c in enumerate(minpoly)]
    discriminant = polynomial_discriminant(monic)
    if discriminant == 0:
        raise ValueError('the polynomial is not square-free')
    order = _Order(monic, [[1 if k == i else 0 for k in range(n)] for i in range(n)], 1)
    for p, exponent in factorint(abs(discriminant)).items():
        if exponent < 2:
            continue
        while True:
            larger = order.enlarged(p)
            if larger.index_data() == order.index_data():
                break
            order = larger
    return order, lc, discriminant


def integral_basis(minpoly: list[int]) -> list[list[Fraction]]:
    """An integral basis of the number field ``QQ(theta)``, ``theta`` a
    root of the square-free integer polynomial ``minpoly`` (highest
    degree first): ``n`` polynomials in ``theta`` with rational
    coefficients (highest degree first), the ``k``-th of degree ``k``,
    in Hermite normal form.

    For a reducible square-free polynomial this is a basis of the
    maximal order of the algebra ``QQ[x]/(minpoly)``.

    Examples
    ========

    >>> from sympy_extras.polys.numberfields.basis import integral_basis
    >>> integral_basis([1, 0, 1])
    [[Fraction(1, 1)], [Fraction(1, 1), Fraction(0, 1)]]
    >>> integral_basis([1, 0, 3])                  # 1, (1 + sqrt(-3))/2
    [[Fraction(1, 1)], [Fraction(1, 2), Fraction(1, 2)]]
    >>> integral_basis([4, 0, -5])                 # theta = sqrt(5)/2: 1, 1/2 + theta
    [[Fraction(1, 1)], [Fraction(1, 1), Fraction(1, 2)]]
    """
    order, lc, _ = _maximal_order(minpoly)
    basis, denominator = order.index_data()
    # echelon from the highest power down: one element of each degree
    basis = [row[::-1] for row in hermite_normal_form([row[::-1] for row in basis])]
    result: list[list[Fraction]] = []
    for row in basis:
        # (lc theta)**k = lc**k theta**k
        coefficients = [Fraction(v*lc**k, denominator) for k, v in enumerate(row)]
        while len(coefficients) > 1 and coefficients[-1] == 0:
            coefficients.pop()
        result.append(coefficients[::-1])
    result.sort(key=len)
    return result


def field_discriminant(minpoly: list[int]) -> int:
    """The discriminant of the number field ``QQ(theta)``: the
    discriminant of the polynomial divided by the square of the index of
    ``ZZ[theta]`` in the ring of integers.

    Examples
    ========

    >>> from sympy_extras.polys.numberfields.basis import field_discriminant
    >>> field_discriminant([1, 1, 1, 1, 1]), field_discriminant([1, 0, -2]), field_discriminant([1, 0, -1, -1])
    (125, 8, -23)
    """
    order, _, discriminant = _maximal_order(minpoly)
    basis, denominator = order.index_data()
    n = len(basis)
    determinant = 1
    for i in range(n):
        determinant *= basis[i][i]
    # the index of ZZ[lc theta] in the maximal order is d**n/det
    index = Fraction(denominator**n, abs(determinant))
    value = Fraction(discriminant)/(index*index)
    if value.denominator != 1:
        raise ArithmeticError('the index does not divide the discriminant')
    return int(value)
