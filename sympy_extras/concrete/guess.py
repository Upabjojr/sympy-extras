r"""Guessing: equations satisfied by a sequence, from its first terms.

Given the terms $a_0, \dots, a_{N-1}$ of a sequence of rational numbers,
three kinds of equations with polynomial coefficients are searched:

* a **recurrence** $\sum_{i=0}^{r} p_i(n)\, a_{n+i} = 0$
  (:func:`guess_recurrence`; the sequence is then P-recursive, or
  holonomic);
* a **differential equation** $\sum_{k=0}^{r} q_k(x)\, f^{(k)}(x) = 0$
  for the generating function $f = \sum a_n x^n$
  (:func:`guess_differential_equation`);
* an **algebraic equation** $\sum_{k=0}^{r} q_k(x)\, f(x)^k = 0$
  (:func:`guess_algebraic_equation`; for $r = 1$ the generating function
  is rational).

Each is a linear problem: for an order $r$ and a degree $d$ the
$(r + 1)(d + 1)$ unknown coefficients satisfy one linear equation for
each $n$ (or each power of $x$) which the known terms determine, and a
solution is accepted only when the equations outnumber the unknowns by a
margin, so that a solution is no accident of the linear algebra: an
equation with $u$ unknowns found from $u + m$ equations has passed $m$
independent checks. Among the orders and degrees which the number of
terms allows, the equation with the fewest unknown coefficients is
returned (then the lowest order). The candidates are located modulo a
prime first (the kernel of the largest degree for each order, then the
smallest degree by bisection), and the exact null space is computed for
the chosen one only.

A guess is a conjecture, not a proof: it holds on the terms given.
:func:`guess_closed_form` solves the guessed recurrence
(:func:`sympy_extras.solvers.recurr.rsolve_linear`: hypergeometric and
d'Alembertian solutions) with the first terms as initial values and
checks the result on all of them.

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.concrete.guess import (guess_recurrence, guess_differential_equation,
...     guess_algebraic_equation, guess_closed_form)
>>> n, x = Symbol('n'), Symbol('x')
>>> catalan = [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900]
>>> recurrence = guess_recurrence(catalan)
>>> recurrence.as_expr(Function('a'), n)
(-4*n - 2)*a(n) + (n + 2)*a(n + 1)
>>> recurrence.extend(catalan[:5], 16)[-2:] == [2674440, 9694845]
True
>>> guess_algebraic_equation(catalan).as_expr(x, Symbol('f'))
f**2*x - f + 1
>>> guess_differential_equation(catalan).as_expr(Function('f'), x)
(2*x - 1)*f(x) + (4*x**2 - x)*Derivative(f(x), x) + 1
>>> guess_closed_form([0, 1, 3, 7, 15, 31, 63, 127, 255, 511], n)
2**n - 1
>>> guess_closed_form([1, 0, 1, 2, 9, 44, 265, 1854, 14833, 133496, 1334961, 14684570], n)         # derangements
-factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)

References
==========

.. [1] B. Salvy, P. Zimmermann, *GFUN: a Maple package for the
       manipulation of generating and holonomic functions in one
       variable*, ACM Transactions on Mathematical Software 20 (1994),
       163-177.
.. [2] M. Kauers, *Guessing handbook*, RISC Report 09-07, Johannes Kepler
       Universität Linz, 2009.
.. [3] M. Kauers, P. Paule, *The Concrete Tetrahedron*, Springer, 2011,
       Section 7.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Callable, Final, Optional, Sequence, Union

from sympy_extras.core.expr import Expr
from sympy_extras._timeout import attempt
from sympy_extras.core.function import AppliedUndef, UndefinedFunction, diff, expand
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains import QQ

__all__ = ['guess_recurrence', 'guess_differential_equation', 'guess_algebraic_equation', 'guess_closed_form',
           'GuessedRecurrence', 'GuessedDifferentialEquation', 'GuessedAlgebraicEquation']

#: a term of a sequence
Number = Union[int, Fraction]

#: the coefficient of the unknown ``(i, j)`` (order index, degree index) in
#: the equation number ``m``
Entry = Callable[[int, int, int], Fraction]

#: the prime of the modular search (a Mersenne prime)
_PRIME = 2**61 - 1

#: how many more equations than unknowns a guess needs
_MARGIN = 3

#: the time given to the recurrence solver for a closed form
_SOLVE_SECONDS = 10.0


# ---------------------------------------------------------------------------
# the linear algebra

def _modular(q: Fraction) -> int:
    """The image in ``GF(_PRIME)`` (zero for a denominator divisible by
    the prime: the exact computation decides anyway)."""
    denominator = q.denominator % _PRIME
    if denominator == 0:
        return 0
    return q.numerator*pow(denominator, -1, _PRIME) % _PRIME


def _has_kernel(rows: list[list[int]], columns: list[int]) -> bool:
    """Whether the matrix made of these columns has a non-zero kernel
    modulo the prime."""
    p = _PRIME
    work = [[row[c] for c in columns] for row in rows]
    width = len(columns)
    rank = 0
    for column in range(width):
        pivot = -1
        for i in range(rank, len(work)):
            if work[i][column]:
                pivot = i
                break
        if pivot < 0:
            return True
        work[rank], work[pivot] = work[pivot], work[rank]
        top = work[rank]
        inverse = pow(top[column], -1, p)
        for i in range(rank + 1, len(work)):
            row = work[i]
            factor = row[column]*inverse % p
            if factor:
                for j in range(column + 1, width):
                    row[j] = (row[j] - factor*top[j]) % p
                row[column] = 0
        rank += 1
    return False


class _Candidate:
    """An order and a degree for which the modular system has a kernel."""

    def __init__(self, order: int, degree: int) -> None:
        self.order: Final[int] = order
        self.degree: Final[int] = degree

    @property
    def size(self) -> int:
        return (self.order + 1)*(self.degree + 1)


def _candidates(entry: Entry, equations: Callable[[int], int], orders: Sequence[int], max_degree: Optional[int],
                margin: int) -> list[_Candidate]:
    """For each order, the smallest degree with a modular solution, when
    there is one within the degrees which leave ``margin`` equations
    more than unknowns; sorted by the number of unknowns, then the
    order."""
    found: list[_Candidate] = []
    for r in orders:
        count = equations(r)
        top = (count - margin)//(r + 1) - 1
        if max_degree is not None:
            top = min(top, max_degree)
        if top < 0:
            continue
        rows = [[_modular(entry(i, j, m)) for i in range(r + 1) for j in range(top + 1)] for m in range(count)]

        def columns(d: int) -> list[int]:
            return [i*(top + 1) + j for i in range(r + 1) for j in range(d + 1)]

        if not _has_kernel(rows, columns(top)):
            continue
        low, high = 0, top               # the smallest degree with a kernel is in [low, high]
        while low < high:
            middle = (low + high)//2
            if _has_kernel(rows, columns(middle)):
                high = middle
            else:
                low = middle + 1
        found.append(_Candidate(r, low))
    found.sort(key=lambda c: (c.size, c.order))
    return found


def _integer_polynomials(vector: list[Fraction], order: int, degree: int) -> list[list[int]]:
    """The kernel vector as ``order + 1`` polynomials with integer
    coefficients (highest degree first), without common factor, the
    leading coefficient of the last one positive."""
    common = 1
    for c in vector:
        common = common*c.denominator//gcd(common, c.denominator)
    integers = [int(c*common) for c in vector]
    content = 0
    for value in integers:
        content = gcd(content, value)
    polynomials: list[list[int]] = []
    for i in range(order + 1):
        ascending = [value//content for value in integers[i*(degree + 1):(i + 1)*(degree + 1)]]
        while ascending and ascending[-1] == 0:
            ascending.pop()
        polynomials.append(ascending[::-1])
    while polynomials and not polynomials[-1]:
        polynomials.pop()
    if polynomials and polynomials[-1][0] < 0:
        polynomials = [[-c for c in p] for p in polynomials]
    return polynomials


def _search(entry: Entry, equations: Callable[[int], int], orders: Sequence[int], max_degree: Optional[int],
            margin: int) -> Optional[list[list[int]]]:
    """The coefficient polynomials of the smallest equation, or
    ``None``."""
    if margin < 1:
        raise ValueError('the margin must be positive')
    for candidate in _candidates(entry, equations, orders, max_degree, margin):
        r, d = candidate.order, candidate.degree
        rows = [[entry(i, j, m) for i in range(r + 1) for j in range(d + 1)] for m in range(equations(r))]
        kernel = QQ.nullspace(rows, candidate.size)
        if not kernel:
            # a kernel modulo the prime only
            continue
        vector = min(kernel, key=lambda v: sum(1 for c in v if c))
        return _integer_polynomials(vector, r, d)
    return None


def _terms(terms: Sequence[Number]) -> list[Fraction]:
    return [Fraction(t) for t in terms]


def _polynomial_value(p: list[int], at: int) -> int:
    value = 0
    for c in p:
        value = value*at + c
    return value


def _polynomial_expr(p: list[int], x: Symbol) -> Expr:
    result: Expr = Integer(0)
    degree = len(p) - 1
    for k, c in enumerate(p):
        if c:
            result = result + Integer(c)*x**(degree - k)
    return result


def _degree(polynomials: list[list[int]]) -> int:
    return max((len(p) - 1 for p in polynomials), default=-1)


# ---------------------------------------------------------------------------
# recurrences

class GuessedRecurrence:
    """The recurrence ``sum_i p_i(n) a(n + i) = 0`` for ``n >= 0``.

    ``coefficients`` are the polynomials ``p_0, ..., p_r`` as lists of
    integers, highest degree first, without common factor.
    """

    def __init__(self, coefficients: list[list[int]]) -> None:
        self.coefficients: Final[list[list[int]]] = coefficients

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def degree(self) -> int:
        return _degree(self.coefficients)

    def holds(self, terms: Sequence[Number]) -> bool:
        """Whether the terms satisfy the recurrence."""
        a = _terms(terms)
        for n in range(len(a) - self.order):
            if sum((_polynomial_value(p, n)*a[n + i] for i, p in enumerate(self.coefficients)), Fraction(0)):
                return False
        return True

    def extend(self, terms: Sequence[Number], count: int) -> list[Fraction]:
        """The sequence continued by the recurrence up to ``count``
        terms; at least ``order`` terms must be given. Raises
        ``ValueError`` where the leading coefficient vanishes, the next
        term not being determined."""
        a = _terms(terms)
        r = self.order
        if len(a) < r:
            raise ValueError('%d initial terms are needed' % r)
        while len(a) < count:
            n = len(a) - r
            leading = _polynomial_value(self.coefficients[r], n)
            if leading == 0:
                raise ValueError('the leading coefficient vanishes at n = %d' % n)
            rest = sum((_polynomial_value(p, n)*a[n + i] for i, p in enumerate(self.coefficients[:r])), Fraction(0))
            a.append(-rest/leading)
        return a

    def as_expr(self, a: UndefinedFunction, n: Symbol) -> Expr:
        """The left-hand side ``sum_i p_i(n) a(n + i)``."""
        result: Expr = Integer(0)
        for i, p in enumerate(self.coefficients):
            if p:
                result = result + _polynomial_expr(p, n)*a(n + i)
        return result

    def __repr__(self) -> str:
        return 'GuessedRecurrence(%r)' % (self.coefficients,)


def guess_recurrence(terms: Sequence[Number], max_order: Optional[int] = None, max_degree: Optional[int] = None,
                     margin: int = _MARGIN) -> Optional[GuessedRecurrence]:
    """A linear recurrence with polynomial coefficients which the terms
    ``a(0), a(1), ...`` satisfy, or ``None``.

    Parameters
    ==========

    terms : the first terms, integers or fractions.
    max_order, max_degree : bounds on the order and on the degree of the
        coefficients (by default, whatever the number of terms allows).
    margin : how many more equations than unknown coefficients are
        required; a recurrence of order ``r`` and degree ``d`` needs
        ``(r + 1)(d + 1) + r + margin`` terms.

    Examples
    ========

    >>> from sympy_extras.concrete.guess import guess_recurrence
    >>> guess_recurrence([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55])
    GuessedRecurrence([[-1], [-1], [1]])
    >>> from math import factorial
    >>> guess_recurrence([factorial(n) for n in range(10)])
    GuessedRecurrence([[-1, -1], [1]])

    The central Delannoy numbers: ``(n + 2) a(n + 2) = (6 n + 9) a(n + 1)
    - (n + 1) a(n)``.

    >>> guess_recurrence([1, 3, 13, 63, 321, 1683, 8989, 48639, 265729, 1462563, 8097453, 45046719])
    GuessedRecurrence([[1, 1], [-6, -9], [1, 2]])
    >>> guess_recurrence([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]) is None
    True

    A few exceptions to a pattern are absorbed by a coefficient which
    vanishes at them (here ``a(n + 1) = 2 a(n)`` fails at ``n = 5`` and
    ``n = 7``): such a factor with roots inside the data is the sign of
    an accident, and more terms settle the matter.

    >>> guess_recurrence([1, 2, 4, 8, 16, 32, 65, 130, 261, 522])
    GuessedRecurrence([[-2, 24, -70], [1, -12, 35]])
    >>> guess_recurrence([1, 2, 4, 8, 16, 32, 65, 130, 261, 522, 1047, 2100]) is None
    True
    """
    a = _terms(terms)
    count = len(a)

    def entry(i: int, j: int, m: int) -> Fraction:
        return m**j*a[m + i]

    orders = range(0, count if max_order is None else min(count, max_order + 1))
    polynomials = _search(entry, lambda r: count - r, orders, max_degree, margin)
    if polynomials is None or not polynomials:
        return None
    return GuessedRecurrence(polynomials)


# ---------------------------------------------------------------------------
# differential equations

class GuessedDifferentialEquation:
    """The differential equation ``sum_k q_k(x) f^(k)(x) = 0`` for the
    generating function ``f = sum a(n) x**n``; an inhomogeneous equation
    when ``inhomogeneous`` (a polynomial, added to the left-hand side)
    is not zero.

    ``coefficients`` are the polynomials ``q_0, ..., q_r`` as lists of
    integers, highest degree first.
    """

    def __init__(self, coefficients: list[list[int]], inhomogeneous: list[int]) -> None:
        self.coefficients: Final[list[list[int]]] = coefficients
        self.inhomogeneous: Final[list[int]] = inhomogeneous

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def degree(self) -> int:
        return _degree(self.coefficients)

    def as_expr(self, f: UndefinedFunction, x: Symbol) -> Expr:
        """The left-hand side of the equation."""
        result: Expr = _polynomial_expr(self.inhomogeneous, x)
        for k, q in enumerate(self.coefficients):
            if q:
                result = result + _polynomial_expr(q, x)*(diff(f(x), x, k) if k else f(x))
        return result

    def __repr__(self) -> str:
        return 'GuessedDifferentialEquation(%r, %r)' % (self.coefficients, self.inhomogeneous)


def _series_product(a: list[Fraction], b: list[Fraction], count: int) -> list[Fraction]:
    """The first ``count`` coefficients of the product."""
    result = [Fraction(0)]*count
    for i, c in enumerate(a[:count]):
        if c:
            for j, e in enumerate(b[:count - i]):
                result[i + j] += c*e
    return result


def guess_differential_equation(terms: Sequence[Number], max_order: Optional[int] = None,
                                max_degree: Optional[int] = None, margin: int = _MARGIN
                                ) -> Optional[GuessedDifferentialEquation]:
    """A linear differential equation with polynomial coefficients for
    the generating function ``f(x) = sum a(n) x**n`` of the terms, or
    ``None``. The equation may have a polynomial right-hand side (of
    degree at most that of the coefficients), written on the left.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from math import factorial
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.concrete.guess import guess_differential_equation
    >>> f, x = Function('f'), Symbol('x')
    >>> guess_differential_equation([Fraction(1, factorial(n)) for n in range(10)]).as_expr(f, x)
    -f(x) + Derivative(f(x), x)
    >>> central = [1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620]      # binomial(2 n, n): 1/sqrt(1 - 4 x)
    >>> guess_differential_equation(central).as_expr(f, x)
    (4*x - 1)*Derivative(f(x), x) + 2*f(x)
    """
    a = _terms(terms)
    count = len(a)

    def derivative(k: int) -> list[Fraction]:
        # the coefficients of the k-th derivative
        result: list[Fraction] = []
        for m in range(count - k):
            c = a[m + k]
            for t in range(1, k + 1):
                c = c*(m + t)
            result.append(c)
        return result

    derivatives: list[list[Fraction]] = []

    def entry(i: int, j: int, m: int) -> Fraction:
        # the unknown number 0 is the inhomogeneous polynomial, the
        # unknown i > 0 the coefficient of the derivative of order i - 1
        if i == 0:
            return Fraction(1 if m == j else 0)
        while len(derivatives) < i:
            derivatives.append(derivative(len(derivatives)))
        return derivatives[i - 1][m - j] if m >= j else Fraction(0)

    # with the unknown polynomials numbered from 0 (inhomogeneous part)
    # to r + 1, the equations are the powers of x below count - r
    top = count - 1 if max_order is None else min(count - 1, max_order)
    polynomials = _search(entry, lambda r: count - (r - 1), range(1, top + 2), max_degree, margin)
    if polynomials is None or len(polynomials) < 2:
        return None
    return GuessedDifferentialEquation(polynomials[1:], polynomials[0])


# ---------------------------------------------------------------------------
# algebraic equations

class GuessedAlgebraicEquation:
    """The algebraic equation ``sum_k q_k(x) f**k = 0`` for the
    generating function ``f = sum a(n) x**n``.

    ``coefficients`` are the polynomials ``q_0, ..., q_r`` as lists of
    integers, highest degree first.
    """

    def __init__(self, coefficients: list[list[int]]) -> None:
        self.coefficients: Final[list[list[int]]] = coefficients

    @property
    def order(self) -> int:
        """The degree in ``f``."""
        return len(self.coefficients) - 1

    @property
    def degree(self) -> int:
        """The largest degree in ``x`` of the coefficients."""
        return _degree(self.coefficients)

    def as_expr(self, x: Symbol, f: Expr) -> Expr:
        """The left-hand side, a polynomial in ``x`` and ``f``."""
        result: Expr = Integer(0)
        for k, q in enumerate(self.coefficients):
            if q:
                result = result + _polynomial_expr(q, x)*f**k
        return result

    def __repr__(self) -> str:
        return 'GuessedAlgebraicEquation(%r)' % (self.coefficients,)


def guess_algebraic_equation(terms: Sequence[Number], max_order: Optional[int] = None,
                             max_degree: Optional[int] = None, margin: int = _MARGIN
                             ) -> Optional[GuessedAlgebraicEquation]:
    """A polynomial equation ``P(x, f) = 0`` for the generating function
    ``f(x) = sum a(n) x**n`` of the terms, or ``None``; ``max_order``
    bounds the degree in ``f``. An equation of degree one gives a
    rational generating function.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.concrete.guess import guess_algebraic_equation
    >>> x, f = Symbol('x'), Symbol('f')
    >>> guess_algebraic_equation([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]).as_expr(x, f)
    f*(x**2 + x - 1) + x
    >>> motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511, 41835]
    >>> guess_algebraic_equation(motzkin).as_expr(x, f)
    f**2*x**2 + f*(x - 1) + 1
    """
    a = _terms(terms)
    count = len(a)
    powers: list[list[Fraction]] = [[Fraction(1)] + [Fraction(0)]*(count - 1)]

    def entry(i: int, j: int, m: int) -> Fraction:
        while len(powers) <= i:
            powers.append(_series_product(powers[-1], a, count))
        return powers[i][m - j] if m >= j else Fraction(0)

    top = count - 1 if max_order is None else min(count - 1, max_order)
    polynomials = _search(entry, lambda r: count, range(1, top + 1), max_degree, margin)
    if polynomials is None or len(polynomials) < 2:
        return None
    return GuessedAlgebraicEquation(polynomials)


# ---------------------------------------------------------------------------
# closed forms

def guess_closed_form(terms: Sequence[Number], n: Symbol, max_order: Optional[int] = None,
                      max_degree: Optional[int] = None, margin: int = _MARGIN) -> Optional[Expr]:
    """An expression in ``n`` whose values at ``0, 1, ...`` are the
    terms: the solution of the guessed recurrence with the first terms
    as initial values, when the recurrence has a full set of
    d'Alembertian solutions (polynomials, hypergeometric terms, their
    nested sums); ``None`` otherwise. The expression is checked on every
    term given.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.concrete.guess import guess_closed_form
    >>> n = Symbol('n')
    >>> guess_closed_form([1, 4, 9, 16, 25, 36, 49, 64, 81, 100], n)
    n**2 + 2*n + 1
    >>> guess_closed_form([1, 2, 6, 20, 70, 252, 924, 3432, 12870, 48620], n)
    4**n*RisingFactorial(1/2, n)/factorial(n)
    >>> from fractions import Fraction
    >>> guess_closed_form([Fraction(k, k + 1) for k in range(12)], n)
    n/(n + 1)
    >>> guess_closed_form([0, 1, Fraction(3, 2), Fraction(11, 6), Fraction(25, 12), Fraction(137, 60),
    ...                    Fraction(49, 20), Fraction(363, 140), Fraction(761, 280), Fraction(7129, 2520),
    ...                    Fraction(7381, 2520), Fraction(83711, 27720)], n)
    harmonic(n)

    Too few terms for any equation, and a sequence which is not
    P-recursive of small order (the primes):

    >>> guess_closed_form([1, 4, 9, 16, 25], n) is None
    True
    >>> guess_closed_form([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53], n) is None
    True
    """
    from sympy_extras.core.function import Function
    from sympy_extras.solvers.recurr import rsolve_linear
    a = _terms(terms)
    recurrence = guess_recurrence(a, max_order, max_degree, margin)
    if recurrence is None:
        return None
    r = recurrence.order
    if r == 0:
        # p_0(n) a(n) = 0: zero outside the roots of p_0
        return Integer(0) if not any(a) else None
    # the initial values are taken after the last zero of the leading
    # coefficient, from where on the recurrence determines the sequence
    # (n/(n + 1): n (n + 2) a(n + 1) = (n + 1)**2 a(n) says nothing of
    # a(1) at n = 0)
    start = 0
    for k in range(len(a) - r):
        if _polynomial_value(recurrence.coefficients[r], k) == 0:
            start = k + 1
    if start + r > len(a):
        return None
    y = Function('y')
    initial: dict[AppliedUndef, Union[Expr, int]] = {
        y(Integer(k)): Rational(a[k].numerator, a[k].denominator) for k in range(start, start + r)}
    equation = recurrence.as_expr(y, n)
    solution = attempt(lambda: rsolve_linear(equation, y(n), initial), _SOLVE_SECONDS, native=True)
    if solution is None:
        return None
    for k, value in enumerate(a):
        if expand(solution.subs(n, Integer(k)).doit()) != Rational(value.numerator, value.denominator):
            return None
    return solution
