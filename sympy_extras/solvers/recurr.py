r"""Linear recurrences with polynomial coefficients: polynomial,
hypergeometric and d'Alembertian solutions.

A linear recurrence of order `d`,

.. math:: p_d(n)\, y(n + d) + \cdots + p_1(n)\, y(n + 1) + p_0(n)\, y(n) = f(n),

with polynomial coefficients `p_i` is written here as the list of its
coefficients ``[p_0, ..., p_d]``. SymPy's ``rsolve`` finds the polynomial,
rational and hypergeometric solutions (Petkovšek's ``Hyper``) but no other:
for `y(n + 1) = (n + 1)\, y(n) + 1` it returns nothing, and for
`(n + 2)\, y(n + 2) = (2n + 3)\, y(n + 1) - (n + 1)\, y(n)` it returns the
constants and misses the harmonic numbers. This module adds the
*d'Alembertian* solutions of Abramov and Petkovšek [1]_: the nested
indefinite sums of hypergeometric terms,

.. math:: y(n) = h_1(n) \sum_{k_1 < n} h_2(k_1) \sum_{k_2 < k_1} h_3(k_2) \cdots,

which are exactly the solutions of the operators that factor into first
order factors over the rational functions. They include `n!`, the
harmonic numbers, the derangement numbers `n! \sum (-1)^k / k!` and the
solutions of every recurrence with constant coefficients.

The algorithms
==============

* :func:`rsolve_polynomial`: the degree of a polynomial solution is
  bounded by the nonnegative integer roots of the first nonzero of the
  polynomials `b_k(e)` in the expansion
  `L[n^e] = \sum_k b_k(e)\, n^{D + e - k}` (`D` the largest degree of the
  coefficients); the coefficients of the solution are then found by
  linear algebra.
* :func:`rsolve_hypergeometric`, Petkovšek's algorithm [2]_: a
  hypergeometric solution has `y(n + 1)/y(n) = z\, a(n)\, c(n + 1) /
  (b(n)\, c(n))` with `a, b, c` monic, `\gcd(a(n), b(n + h)) = 1` for every
  nonnegative integer `h` (the Gosper–Petkovšek form, :func:`gosper_form`),
  and then `a` divides `p_0(n)`, `b` divides `p_d(n - d + 1)`, `z` is a
  root of the polynomial made of the leading coefficients of
  `P_i = p_i(n) \prod_{j < i} a(n + j) \prod_{i \le j < d} b(n + j)`, and
  `c` is a polynomial solution of `\sum_i z^i P_i(n)\, c(n + i) = 0`. The
  terms found are grouped into similarity classes (two hypergeometric
  terms are similar when their quotient is a rational function: the
  quotient of their ratios is `q(n + 1)/q(n)` for a polynomial `q`, a
  polynomial solution of a first order recurrence) so that the result is
  a basis.
* :func:`rsolve_dalembertian`, [1]_: a hypergeometric solution `h` with
  ratio `r` gives `L = \tilde L \circ (N - r)`, and with `y = h \sum z`
  the equation becomes `\tilde L' z = 0` of order `d - 1`, whose
  d'Alembertian solutions are found recursively; the solutions of `L` are
  `h` and the `h \sum_{k < n} z(k)`. Every d'Alembertian solution is
  found, whichever `h` is chosen.
* :func:`rsolve_linear`: the equation written with `y(n + i)`, the
  d'Alembertian solutions as a general solution with constants
  `C_0, C_1, \ldots`, a particular solution when the right-hand side is a
  sum of hypergeometric terms (each term `f_j` has an annihilator
  `N - f_j(n + 1)/f_j(n)`, the operator `(N - \rho_j) \circ L` has a
  d'Alembertian solution `u` with `L u = \kappa f_j`, `\kappa` a constant
  found by evaluating at one point), and initial values.

The solutions are exact expressions in ``RisingFactorial``, ``factorial``,
``Product`` (for the irreducible factors of degree two and more),
``Sum`` and powers `z^n`; they satisfy the recurrence for every `n` at
which they are defined, which is every `n \ge 0` unless a factor of the
products vanishes at a nonnegative integer, in which case the products
and sums start after it. Every solution is checked by substitution at
several integers before it is returned.

The implementation
==================

The module is written on the core (:mod:`sympy_extras.core`, the native
functions, sums and products) and on the typed polynomial layer, and does
not import SymPy. The coefficients are dense polynomials in `n` over a
field: ``QQ``, the rational functions ``QQ(a, b, ...)`` of the parameters
of the recurrence (:class:`~sympy_extras.polys.domains.sparsefractionfield.
SparseFractionField`) or a number field ``QQ(alpha)`` when they contain a
radical. A root `z` outside that field (``(1 + sqrt(5))/2`` for the
Fibonacci numbers, ``sqrt(a)`` for `y(n + 2) = a\, y(n)`) is adjoined: an
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField` over
``QQ``, a simple algebraic extension of the field otherwise. The
d'Alembertian reduction needs factorisations over the field of the
hypergeometric solution it reduces with: over ``QQ``, ``QQ(a, ...)`` and
the number fields over ``QQ``, not over an extension of an extension (a
solution which needs that is not found). The roots of the irreducible
factors of degree two are written with a square root, those of higher
degree as SymPy's ``roots`` (ported) writes them; the roots it does not
find (``CRootOf`` in SymPy) are left out. The closed forms of the sums in
the solutions come from the ported summation (``Sum.doit``) without a
time limit.

References
==========

.. [1] S. A. Abramov, M. Petkovšek, D'Alembertian solutions of linear
   differential and difference equations, ISSAC '94, 169–174.
.. [2] M. Petkovšek, Hypergeometric solutions of linear recurrences with
   polynomial coefficients, Journal of Symbolic Computation 14 (1992)
   243–264.
.. [3] M. Petkovšek, H. S. Wilf, D. Zeilberger, A = B, A K Peters (1996),
   chapters 5 and 8.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Callable, Final, Generic, Optional, Sequence, TypeVar, Union

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, diff, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Rational
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Factorial, RisingFactorial, factorial, rf
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.functions.special.gamma_functions import Gamma, LowerGamma, UpperGamma
from sympy_extras.functions.special.hyper import Hyper
from sympy_extras.functions.special.zeta_functions import LerchPhi, PolyLog, Zeta
from sympy_extras.polys import densearith, densebasic, densetools, euclidtools
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.polys.rationaltools import together
from sympy_extras.simplify.gammasimp import gammasimp
from sympy_extras.concrete.expr_with_limits import Product, Sum

__all__ = ['rsolve_polynomial', 'rsolve_hypergeometric', 'rsolve_dalembertian', 'rsolve_linear',
           'gosper_form', 'is_recurrence_solution']

T = TypeVar('T')
U = TypeVar('U')

#: the names of the summation indices, by depth of the nested sums
_INDEX_NAMES = ('k', 'j', 'i', 'l', 'm')
#: a solution with the first index from which it is defined
_Solution = tuple[Expr, int]
#: the coefficients of a recurrence as given
Coefficients = Sequence[Union[Expr, int]]

ZERO = Integer(0)
ONE = Integer(1)
#: the variable of the polynomials handed to the native factorisation and roots
_X = Symbol('_rx')


# ---------------------------------------------------------------------------
# dense polynomials in n over a field

def _coeff(f: list[T], j: int, K: Domain[T]) -> T:
    """The coefficient of ``n**j``."""
    i = len(f) - 1 - j
    return f[i] if 0 <= i < len(f) else K.zero


def _shift(f: list[T], h: int, K: Domain[T]) -> list[T]:
    """``f(n + h)``."""
    return f if h == 0 or not f else densetools.dup_shift(f, K.from_int(h), K)


def _products(p: list[T], lo: int, hi: int, K: Domain[T]) -> list[T]:
    """``prod_{j = lo}^{hi - 1} p(n + j)``."""
    result = densebasic.dup_one(K)
    for j in range(lo, hi):
        result = densearith.dup_mul(result, _shift(p, j, K), K)
    return result


def _cancelled(num: list[T], den: list[T], K: Domain[T]) -> tuple[list[T], list[T]]:
    """The rational function ``num/den`` in lowest terms, ``den`` monic."""
    if not num:
        return [], densebasic.dup_one(K)
    g = euclidtools.dup_gcd(num, den, K)
    num, den = densearith.dup_quo(num, g, K), densearith.dup_quo(den, g, K)
    lead = densebasic.dup_LC(den, K)
    return densearith.dup_quo_ground(num, lead, K), densetools.dup_monic(den, K)


def _binomial_polynomial(j: int, K: Domain[T]) -> list[T]:
    """``binomial(e, j)`` as a polynomial in ``e``."""
    result = densebasic.dup_one(K)
    for i in range(j):
        factor = [K.one, K.neg(K.from_int(i))] if i else [K.one, K.zero]
        result = densearith.dup_quo_ground(densearith.dup_mul(result, factor, K), K.from_int(i + 1), K)
    return result


# ---------------------------------------------------------------------------
# the fields of coefficients

class _ExtensionDomain(Domain[list[T]], Generic[T]):
    """The field ``K[t]/(m(t))`` for a monic irreducible ``m`` over the
    field ``K``: elements are dense polynomials in ``t`` over ``K`` of
    degree below that of ``m``; ``alpha_expr`` is the expression of the
    root ``t`` stands for."""

    is_field = True
    characteristic = 0

    def __init__(self, base: Domain[T], minpoly: list[T], alpha_expr: Expr) -> None:
        self.base: Final[Domain[T]] = base
        self.minpoly: Final[list[T]] = densetools.dup_monic(minpoly, base)
        self.degree: Final[int] = densebasic.dup_degree(minpoly)
        self.alpha_expr: Final[Expr] = alpha_expr
        self.name = '%s<%s>' % (base.name, alpha_expr)

    @property
    def zero(self) -> list[T]:
        return []

    @property
    def one(self) -> list[T]:
        return [self.base.one]

    @property
    def alpha(self) -> list[T]:
        return self._reduce([self.base.one, self.base.zero])

    def lift(self, c: T) -> list[T]:
        """An element of the base field as an element of the extension."""
        return [] if self.base.is_zero(c) else [c]

    def _reduce(self, a: list[T]) -> list[T]:
        a = densebasic.dup_strip(a, self.base)
        if densebasic.dup_degree(a) < self.degree:
            return a
        return densearith.dup_rem(a, self.minpoly, self.base)

    def is_zero(self, a: list[T]) -> bool:
        return not a

    def is_one(self, a: list[T]) -> bool:
        return len(a) == 1 and self.base.is_one(a[0])

    def add(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_add(a, b, self.base)

    def sub(self, a: list[T], b: list[T]) -> list[T]:
        return densearith.dup_sub(a, b, self.base)

    def neg(self, a: list[T]) -> list[T]:
        return densearith.dup_neg(a, self.base)

    def mul(self, a: list[T], b: list[T]) -> list[T]:
        return self._reduce(densearith.dup_mul(a, b, self.base))

    def invert(self, a: list[T]) -> list[T]:
        if not a:
            raise ZeroDivisionError('0 has no inverse')
        return euclidtools.dup_invert(a, self.minpoly, self.base)

    def exquo(self, a: list[T], b: list[T]) -> list[T]:
        return self.mul(a, self.invert(b))

    def quo(self, a: list[T], b: list[T]) -> list[T]:
        return self.exquo(a, b)

    def rem(self, a: list[T], b: list[T]) -> list[T]:
        if not b:
            raise ZeroDivisionError('division by zero in %s' % self.name)
        return []

    def gcd(self, a: list[T], b: list[T]) -> list[T]:
        return [] if not a and not b else self.one

    def lcm(self, a: list[T], b: list[T]) -> list[T]:
        return [] if not a or not b else self.one

    def gcdex(self, a: list[T], b: list[T]) -> tuple[list[T], list[T], list[T]]:
        if a:
            return self.invert(a), [], self.one
        if b:
            return [], self.invert(b), self.one
        return [], [], []

    def canonical_unit(self, a: list[T]) -> list[T]:
        return self.one if not a else self.invert(a)

    def from_int(self, n: int) -> list[T]:
        return self.lift(self.base.from_int(n))

    def convert(self, a: U, K: Domain[U]) -> list[T]:
        return self.from_expr(K.to_expr(a))

    def to_expr(self, a: list[T]) -> Expr:
        top = len(a) - 1
        return add(*[mul(self.base.to_expr(c), pow(self.alpha_expr, Integer(top - i)))
                     for i, c in enumerate(a) if not self.base.is_zero(c)])

    def from_expr(self, e: Expr) -> list[T]:
        """An expression made of ``alpha_expr`` and elements of the base
        field by sums, products and integer powers."""
        if e == self.alpha_expr:
            return self.alpha
        if isinstance(e, Add):
            total = self.zero
            for a in e.args:
                total = self.add(total, self.from_expr(a))
            return total
        if e.has(self.alpha_expr):
            if isinstance(e, Pow) and isinstance(e.exp, Integer):
                base = self.from_expr(e.base)
                k = e.exp.p
                return self.pow(base, k) if k >= 0 else self.pow(self.invert(base), -k)
            product = self.one
            for factor in Mul.make_args(e):
                if factor == e:
                    raise ValueError('%s is not an element of %s' % (e, self.name))
                product = self.mul(product, self.from_expr(factor))
            return product
        return self.lift(self.base.from_expr(e))

    def to_str(self, a: list[T]) -> str:
        return str(self.to_expr(a))

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


class _Field(ABC, Generic[T]):
    """A field of coefficients ``K`` with the operations the solvers need
    beyond the arithmetic: the rational polynomials whose common integer
    roots are those of a polynomial, the factorisation into monic
    irreducible factors, and the adjunction of a root. ``restore`` maps
    the names given to the parameters inside the field back to the
    symbols of the problem."""

    def __init__(self, K: Domain[T], restore: dict[Expr, Expr]) -> None:
        self.K: Final[Domain[T]] = K
        self.restore: Final[dict[Expr, Expr]] = restore

    def to_expr(self, a: T) -> Expr:
        return self.K.to_expr(a)

    def from_expr(self, e: Expr) -> T:
        return self.K.from_expr(e)

    def embed(self, c: T) -> list[T]:
        """An element as an element of the extensions of the field."""
        return [] if self.K.is_zero(c) else [c]

    @abstractmethod
    def components(self, f: list[T]) -> list[list[Fraction]]:
        """Rational polynomials whose common integer roots are the integer
        roots of ``f``."""

    def factor_dense(self, f: list[T]) -> list[tuple[list[T], int]]:
        """The irreducible factors of ``f`` (of positive degree), with their
        multiplicities; ``NotImplementedError`` where there is no
        factorisation."""
        _, factors = self.K.factor_dense(f)
        return factors

    def factor(self, f: list[T]) -> list[tuple[list[T], int]]:
        """The monic irreducible factors of ``f`` with their multiplicities."""
        if densebasic.dup_degree(f) <= 0:
            return []
        return [(densetools.dup_monic(g, self.K), m) for g, m in self.factor_dense(f)
                if densebasic.dup_degree(g) > 0]

    def extend(self, f: list[T], root: Expr) -> _Field[list[T]]:
        """The field with ``root``, a root of the monic irreducible ``f``,
        adjoined."""
        return _ExtensionField(self, f, root)

    def expression(self, f: list[T], x: Expr) -> Expr:
        """The polynomial as an expression in ``x``."""
        top = len(f) - 1
        return add(*[mul(self.to_expr(c), pow(x, Integer(top - i))) for i, c in enumerate(f) if not self.K.is_zero(c)])

    def dense(self, e: Expr, x: Symbol) -> list[T]:
        """A polynomial expression in ``x`` as a dense polynomial;
        ``ValueError`` when it is not one over the field."""
        K = self.K
        terms: dict[int, T] = {}
        for term in Add.make_args(expand(e)):
            if term == ZERO:
                continue
            coefficient, dependent = term.as_independent(x)
            if dependent == ONE:
                j = 0
            elif dependent == x:
                j = 1
            elif isinstance(dependent, Pow) and dependent.base == x and isinstance(dependent.exp, Integer) \
                    and dependent.exp.p > 0:
                j = dependent.exp.p
            else:
                raise ValueError('%s is not a polynomial in %s' % (e, x))
            terms[j] = K.add(terms.get(j, K.zero), self.from_expr(coefficient))
        return densebasic.dup_from_dict({j: c for j, c in terms.items() if not K.is_zero(c)}, K)

    def rational(self, e: Expr, x: Symbol) -> tuple[list[T], list[T]]:
        """A rational function of ``x`` as a numerator and a monic
        denominator in lowest terms."""
        numerator, denominator = together(e).as_numer_denom()
        num, den = self.dense(numerator, x), self.dense(denominator, x)
        if not den:
            raise ValueError('%s has a zero denominator' % (e,))
        return _cancelled(num, den, self.K)


class _RationalField(_Field[Fraction]):
    def __init__(self) -> None:
        super().__init__(QQ, {})

    def components(self, f: list[Fraction]) -> list[list[Fraction]]:
        return [f]

    def factor_dense(self, f: list[Fraction]) -> list[tuple[list[Fraction], int]]:
        # through Poly: SymPy's order of the factors
        return _native_factors(self, f)

    def extend(self, f: list[Fraction], root: Expr) -> _Field[list[Fraction]]:
        return _NumberField(AlgebraicField(densetools.dup_monic(f, QQ), root))


class _NumberField(_Field[list[Fraction]]):
    def __init__(self, K: AlgebraicField) -> None:
        super().__init__(K, {})
        self.algebraic: Final[AlgebraicField] = K

    def components(self, f: list[list[Fraction]]) -> list[list[Fraction]]:
        m = self.algebraic.degree
        columns: list[list[Fraction]] = [[] for _ in range(m)]
        for c in f:
            padded = [Fraction(0)]*(m - len(c)) + list(c)
            for j in range(m):
                columns[j].append(padded[j])
        return [densebasic.dup_strip(column, QQ) for column in columns]


class _ParametricField(_Field[SparseFracElement[Fraction]]):
    """``QQ(a, b, ...)``: the parameters renamed (the ring maps symbols by
    name)."""

    def __init__(self, names: list[str], restore: dict[Expr, Expr]) -> None:
        field = SparseFractionField(RationalPolynomialRing(names))
        super().__init__(field, restore)
        self.field: Final[SparseFractionField[Fraction]] = field

    def components(self, f: list[SparseFracElement[Fraction]]) -> list[list[Fraction]]:
        ring = self.field.ring
        common = ring.one
        for c in f:
            common = ring.lcm(common, c.denom)
        top = len(f) - 1
        groups: dict[tuple[int, ...], dict[int, Fraction]] = {}
        for i, c in enumerate(f):
            if self.field.is_zero(c):
                continue
            numerator = ring.mul(c.numer, ring.exquo(common, c.denom))
            for monomial, value in numerator:
                groups.setdefault(monomial, {})[top - i] = value
        return [densebasic.dup_from_dict(g, QQ) for g in groups.values()]

    def factor_dense(self, f: list[SparseFracElement[Fraction]]) -> list[tuple[list[SparseFracElement[Fraction]], int]]:
        return _native_factors(self, f)


class _ExtensionField(_Field[list[T]], Generic[T]):
    """A simple algebraic extension of a field which is not ``QQ`` (no
    factorisation)."""

    def __init__(self, base: _Field[T], minpoly: list[T], root: Expr) -> None:
        domain = _ExtensionDomain(base.K, minpoly, root)
        super().__init__(domain, base.restore)
        self.base: Final[_Field[T]] = base
        self.domain: Final[_ExtensionDomain[T]] = domain

    def embed(self, c: list[T]) -> list[list[T]]:
        return [] if not c else [c]

    def components(self, f: list[list[T]]) -> list[list[Fraction]]:
        m = self.domain.degree
        columns: list[list[T]] = [[] for _ in range(m)]
        zero = self.base.K.zero
        for c in f:
            padded = [zero]*(m - len(c)) + list(c)
            for j in range(m):
                columns[j].append(padded[j])
        result: list[list[Fraction]] = []
        for column in columns:
            stripped = densebasic.dup_strip(column, self.base.K)
            if stripped:
                result.extend(self.base.components(stripped))
        return result

    def factor_dense(self, f: list[list[T]]) -> list[tuple[list[list[T]], int]]:
        raise NotImplementedError('no factorisation over %s' % self.domain.name)


def _native_factors(F: _Field[T], f: list[T]) -> list[tuple[list[T], int]]:
    """The factorisation by :class:`~sympy_extras.polys.polytools.Poly`
    (over ``QQ`` and ``QQ[a, ...]``, the denominators cleared first)."""
    numerator, _ = together(F.expression(f, _X)).as_numer_denom()
    try:
        _, factors = Poly(numerator, _X).factor_list()
    except BasePolynomialError as error:
        raise NotImplementedError(str(error))
    return [(F.dense(g.as_expr(), _X), int(m)) for g, m in factors if g.degree() > 0]


def _integer_roots(F: _Field[T], f: list[T]) -> list[int]:
    """The nonnegative integer roots of the nonzero polynomial ``f``."""
    g: list[Fraction] = []
    for component in F.components(f):
        if component:
            g = euclidtools.dup_gcd(g, component, QQ) if g else component
    if densebasic.dup_degree(g) <= 0:
        return []
    found: set[int] = set()
    for h, _ in QQ.factor_dense(g)[1]:
        if densebasic.dup_degree(h) == 1:
            r = -h[1]/h[0]
            if r.denominator == 1 and r >= 0:
                found.add(int(r))
    return sorted(found)


def _context(exprs: Sequence[Expr], n: Symbol) -> _Context:
    """The field of the coefficients of ``exprs`` (rational functions of
    ``n``): ``QQ``, ``QQ(parameters)`` or ``QQ(alpha)`` for one radical
    ``alpha`` (or ``I``)."""
    symbols = sorted({s for e in exprs for s in e.free_symbols if s != n}, key=lambda s: s.sort_key())
    radicals: set[tuple[Rational, int]] = set()
    imaginary = False
    for e in exprs:
        for atom in e.atoms(Pow):
            if isinstance(atom, Pow) and isinstance(atom.base, Rational) and isinstance(atom.exp, Rational) \
                    and not isinstance(atom.exp, Integer):
                radicals.add((atom.base, atom.exp.q))
        if e.has(I):
            imaginary = True
    if radicals or imaginary:
        if symbols:
            raise NotImplementedError('coefficients with parameters and algebraic numbers are not supported')
        bases = {b for b, _ in radicals}
        if imaginary and radicals or len(bases) > 1:
            raise NotImplementedError('coefficients with several algebraic numbers are not supported')
        if imaginary:
            alpha: Expr = I
        else:
            q = 1
            for _, d in radicals:
                q = q*d//_gcd(q, d)
            alpha = pow(next(iter(bases)), Rational(1, q))
        minpoly = [Fraction(c) for c in minimal_polynomial_coeffs(alpha)]
        return _FieldContext(_NumberField(AlgebraicField(densetools.dup_monic(minpoly, QQ), alpha)))
    if not symbols:
        return _FieldContext(_RationalField())
    names = ['_rr%d' % i for i in range(len(symbols))]
    return _FieldContext(_ParametricField(names, {Symbol(name): s for s, name in zip(symbols, names)}))


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ---------------------------------------------------------------------------
# the operator: the interface of the solvers on a field

class _Context(ABC):
    """A field of coefficients whose element type is hidden: the public
    functions work through it."""

    @abstractmethod
    def rename(self, e: Expr) -> Expr:
        """``e`` with the parameters named as inside the field."""

    @abstractmethod
    def restored(self, e: Expr) -> Expr:
        """``e`` with the parameters named as in the problem."""

    @abstractmethod
    def operator(self, coefficients: Sequence[Expr], n: Symbol, normalize: bool = True) -> _Operator:
        """The recurrence with these coefficients (renamed)."""

    @abstractmethod
    def gosper(self, r: Expr, n: Symbol) -> tuple[Expr, Poly, Poly, Poly]:
        """:func:`gosper_form` of ``r`` (renamed)."""


class _Operator(ABC):
    """A recurrence over a field (the element type hidden)."""

    @property
    @abstractmethod
    def order(self) -> int:
        """The order (``-1`` for the zero operator)."""

    @abstractmethod
    def polynomial_solutions(self) -> list[Expr]: ...

    @abstractmethod
    def hypergeometric_solutions(self, avoid: set[Symbol]) -> list[_Solution]: ...

    @abstractmethod
    def dalembertian_solutions(self, avoid: set[Symbol]) -> list[_Solution]: ...

    @abstractmethod
    def checked(self, solution: Expr, rhs: Expr, start: int, count: int = 6) -> bool: ...

    @abstractmethod
    def particular(self, f: Expr, rho: Expr, avoid: set[Symbol]) -> Optional[_Solution]: ...


class _FieldContext(_Context, Generic[T]):
    def __init__(self, F: _Field[T]) -> None:
        self.F: Final[_Field[T]] = F
        self.renaming: Final[dict[Expr, Expr]] = {v: k for k, v in F.restore.items()}

    def rename(self, e: Expr) -> Expr:
        return e.subs(self.renaming) if self.renaming else e

    def restored(self, e: Expr) -> Expr:
        return e.subs(self.F.restore) if self.F.restore else e

    def operator(self, coefficients: Sequence[Expr], n: Symbol, normalize: bool = True) -> _Operator:
        polys = _coefficient_polys(self.F, [self.rename(c) for c in coefficients], n)
        return _OperatorOver(self.F, _normalized(self.F, polys) if normalize else polys, n)

    def gosper(self, r: Expr, n: Symbol) -> tuple[Expr, Poly, Poly, Poly]:
        F = self.F
        P, Q = F.rational(self.rename(r), n)
        if not P:
            raise ValueError("the ratio is zero")
        z, a, b, c = _gosper(F, P, Q)
        return (self.restored(F.to_expr(z)), _monic_poly(self.restored(F.expression(a, n)), n),
                _monic_poly(self.restored(F.expression(b, n)), n), _monic_poly(self.restored(F.expression(c, n)), n))


class _OperatorOver(_Operator, Generic[T]):
    def __init__(self, F: _Field[T], polys: list[list[T]], n: Symbol) -> None:
        self.F: Final[_Field[T]] = F
        self.polys: Final[list[list[T]]] = polys
        self.n: Final[Symbol] = n

    @property
    def order(self) -> int:
        return len(self.polys) - 1

    def polynomial_solutions(self) -> list[Expr]:
        return [self.F.expression(c, self.n) for c in _polynomial_solutions(self.F, self.polys)]

    def hypergeometric_solutions(self, avoid: set[Symbol]) -> list[_Solution]:
        return [(h.expr, h.start) for h in _hypergeometric_records(self.F, self.polys, self.n, 0, avoid)]

    def dalembertian_solutions(self, avoid: set[Symbol]) -> list[_Solution]:
        return _dalembertian_records(self.F, self.polys, self.n, 0, avoid)

    def checked(self, solution: Expr, rhs: Expr, start: int, count: int = 6) -> bool:
        return _checked([self.F.expression(p, self.n) for p in self.polys], self.n, solution, rhs, start, count)

    def particular(self, f: Expr, rho: Expr, avoid: set[Symbol]) -> Optional[_Solution]:
        F, n, K = self.F, self.n, self.F.K
        try:
            num, den = F.rational(rho, n)
        except ValueError:
            return None
        d = len(self.polys) - 1
        extended: list[list[T]] = []
        for i in range(d + 2):
            previous = densearith.dup_mul(den, _shift(self.polys[i - 1], 1, K), K) if 1 <= i <= d + 1 else []
            current = densearith.dup_mul(num, self.polys[i], K) if i <= d else []
            extended.append(densearith.dup_sub(previous, current, K))
        while extended and not extended[-1]:
            extended.pop()
        operator = _normalized(F, extended)
        coefficients = [F.expression(p, n) for p in self.polys]
        for u, start in _dalembertian_records(F, operator, n, 0, avoid):
            for at in range(start, start + 4):
                value = add(*[_value(c, n, at)*_value(u, n, at + i) for i, c in enumerate(coefficients)])
                f_value = _value(f, n, at)
                if f_value == ZERO or value.is_finite is False or f_value.is_finite is False:
                    continue
                kappa = _normal(value/f_value)
                if _is_zero(kappa):
                    break
                candidate = u/kappa
                if _checked(coefficients, n, candidate, f, start):
                    return candidate, start
                break
        return None


def _monic_poly(e: Expr, n: Symbol) -> Poly:
    """The monic polynomial ``e`` as a ``Poly`` over the field, as SymPy's
    ``monic`` gives it (the native ``monic`` keeps ``ZZ`` when the leading
    coefficient is already one: scaled first)."""
    return Poly(2*e, n).monic()


def _coefficient_polys(F: _Field[T], exprs: Sequence[Expr], n: Symbol) -> list[list[T]]:
    """The coefficients as polynomials in ``n`` over the field, the zero
    leading ones dropped; rational functions of ``n`` are brought to a
    common denominator first."""
    K = F.K
    try:
        fractions = [F.rational(e, n) for e in exprs]
    except ValueError:
        raise ValueError("the coefficients must be rational functions of %s: %s" % (n, list(exprs)))
    common = densebasic.dup_one(K)
    for _, den in fractions:
        common = euclidtools.dup_lcm(common, den, K)
    polys = [densearith.dup_mul(num, densearith.dup_quo(common, den, K), K) for num, den in fractions]
    while polys and not polys[-1]:
        polys.pop()
    return polys


def _normalized(F: _Field[T], polys: Sequence[list[T]]) -> list[list[T]]:
    """The recurrence with its trailing coefficient nonzero: when
    ``p_0 = ... = p_{s-1} = 0`` the equation is one in ``y(n + s), ...``,
    the same recurrence at the index ``n + s``."""
    result = list(polys)
    s = 0
    while result and not result[0]:
        result.pop(0)
        s += 1
    return [_shift(p, -s, F.K) for p in result] if s else result


# ---------------------------------------------------------------------------
# polynomial solutions

def _degree_bound(F: _Field[T], polys: Sequence[list[T]]) -> Optional[int]:
    """The largest possible degree of a polynomial solution of the
    recurrence, or ``None`` when there is none. ``L[n**e]`` has,
    generically, the degree ``D + e - k`` where ``k`` is the index of the
    first of the polynomials ``b_k(e) = sum_j (sum_i a_{i, D-k+j} i**j)
    binomial(e, j)`` which is not identically zero (``a_{ij}`` the
    coefficient of ``n**j`` in ``p_i``); the degree is lower exactly at the
    roots of ``b_k``."""
    K = F.K
    D = max(densebasic.dup_degree(p) for p in polys)
    order = len(polys) - 1
    for k in range(D + order + 2):
        total: list[T] = []
        for j in range(k + 1):
            m = D - k + j
            if m < 0:
                continue
            s = K.zero
            for i, p in enumerate(polys):
                s = K.add(s, K.mul(_coeff(p, m, K), K.from_int(i**j)))
            if not K.is_zero(s):
                total = densearith.dup_add(total, densearith.dup_mul_ground(_binomial_polynomial(j, K), s, K), K)
        if total:
            candidates = _integer_roots(F, total)
            return max(candidates) if candidates else None
    return None  # the zero operator


def _image_of_power(F: _Field[T], polys: Sequence[list[T]], j: int) -> list[T]:
    """``L[n**j] = sum p_i(n) (n + i)**j``."""
    K = F.K
    result: list[T] = []
    for i, p in enumerate(polys):
        power = densearith.dup_pow([K.one, K.from_int(i)] if i else [K.one, K.zero], j, K)
        result = densearith.dup_add(result, densearith.dup_mul(p, power, K), K)
    return result


def _polynomial_solutions(F: _Field[T], polys: Sequence[list[T]]) -> list[list[T]]:
    """A basis of the polynomial solutions of ``L y = 0``."""
    K = F.K
    bound = _degree_bound(F, polys)
    if bound is None:
        return []
    # the images L[n**j] as polynomials: the columns of the linear system
    images = [_image_of_power(F, polys, j) for j in range(bound + 1)]
    height = max([densebasic.dup_degree(p) for p in images if p] + [0]) + 1
    rows = [[_coeff(image, e, K) for image in images] for e in range(height)]
    return [densebasic.dup_strip(list(reversed(v)), K) for v in K.nullspace(rows, bound + 1)]


def rsolve_polynomial(coefficients: Coefficients, n: Symbol) -> list[Expr]:
    """A basis of the polynomial solutions of
    ``sum coefficients[i] * y(n + i) = 0``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.recurr import rsolve_polynomial
    >>> n = Symbol('n')
    >>> rsolve_polynomial([1, -2, 1], n)
    [1, n]
    >>> rsolve_polynomial([n + 1, -n], n)
    [n]
    >>> rsolve_polynomial([(n + 1)**2, -n**2 - 1], n)
    []
    """
    exprs = [to_expr(c) for c in coefficients]
    context = _context(exprs, n)
    operator = context.operator(exprs, n)
    if operator.order < 0:
        raise ValueError("the recurrence is trivial")
    return [context.restored(e) for e in operator.polynomial_solutions()]


# ---------------------------------------------------------------------------
# the Gosper–Petkovšek form

def gosper_form(r: ExprLike, n: Symbol) -> tuple[Expr, Poly, Poly, Poly]:
    """The Gosper–Petkovšek form ``(z, a, b, c)`` of a rational function:
    ``r = z * a(n)/b(n) * c(n + 1)/c(n)`` with ``a``, ``b``, ``c`` monic
    polynomials and ``gcd(a(n), b(n + h)) = 1`` for every nonnegative
    integer ``h``. ``r`` is the quotient ``q(n + 1)/q(n)`` of a rational
    function if and only if ``z = 1`` and ``a = b = 1``. The polynomials
    are given as ``Poly`` over ``QQ`` or ``QQ[a, ...]``
    (``CoercionFailed`` for the other fields, not ported to ``Poly``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.recurr import gosper_form
    >>> n = Symbol('n')
    >>> gosper_form((n + 2)/n, n)
    (1, Poly(1, n, domain='QQ'), Poly(1, n, domain='QQ'), Poly(n**2 + n, n, domain='QQ'))
    >>> gosper_form(2*(n + 1)/(n + 3), n)
    (2, Poly(n + 1, n, domain='QQ'), Poly(n + 3, n, domain='QQ'), Poly(1, n, domain='QQ'))

    References
    ==========

    .. [1] M. Petkovšek, Hypergeometric solutions of linear recurrences
       with polynomial coefficients, J. Symbolic Computation 14 (1992),
       Lemma 3.1 and Algorithm 3.1.
    """
    e = to_expr(r)
    return _context([e], n).gosper(e, n)


def _gosper(F: _Field[T], P: list[T], Q: list[T]) -> tuple[T, list[T], list[T], list[T]]:
    K = F.K
    z = K.quo(densebasic.dup_LC(P, K), densebasic.dup_LC(Q, K))
    a = densetools.dup_monic(P, K)
    b = densetools.dup_monic(Q, K)
    c = densebasic.dup_one(K)
    for root in _shift_coincidences(F, a, b):
        g = euclidtools.dup_gcd(a, _shift(b, root, K), K)
        if densebasic.dup_degree(g) <= 0:
            continue
        a = densetools.dup_monic(densearith.dup_quo(a, g, K), K)
        b = densetools.dup_monic(densearith.dup_quo(b, _shift(g, -root, K), K), K)
        for i in range(1, root + 1):
            c = densearith.dup_mul(c, _shift(g, -i, K), K)
    return z, a, b, densetools.dup_monic(c, K)


def _shift_quotient(F: _Field[T], num: list[T], den: list[T]) -> Optional[list[T]]:
    """The monic polynomial ``q`` with ``num/den = q(n + 1)/q(n)``, or
    ``None`` when the rational function is not such a quotient: ``q`` is a
    polynomial solution of ``den(n) q(n + 1) - num(n) q(n) = 0``."""
    K = F.K
    solutions = _polynomial_solutions(F, _normalized(F, [densearith.dup_neg(num, K), den]))
    if len(solutions) != 1:
        return None
    return densetools.dup_monic(solutions[0], K)


# ---------------------------------------------------------------------------
# hypergeometric solutions

def _shift_coincidences(F: _Field[T], a: list[T], b: list[T]) -> list[int]:
    """The nonnegative integers ``h`` with ``gcd(a(n), b(n + h)) != 1``:
    an irreducible factor ``f`` of ``a`` and one ``g`` of ``b`` have a
    common root when ``g(n + h) = f(n)``, and ``h`` is then read off the
    second coefficients (the resultant in ``n`` of ``a(n)`` and ``b(n + h)``
    is a polynomial in ``h`` with these roots, but far more expensive)."""
    if densebasic.dup_degree(a) <= 0 or densebasic.dup_degree(b) <= 0:
        return []
    K = F.K
    found: set[int] = set()
    factors_b = [g for g, _ in F.factor(b)]
    for f, _ in F.factor(a):
        m = densebasic.dup_degree(f)
        for g in factors_b:
            if densebasic.dup_degree(g) != m:
                continue
            h = F.to_expr(K.quo(K.sub(_coeff(f, m - 1, K), _coeff(g, m - 1, K)), K.from_int(m)))
            if isinstance(h, Integer) and h.p >= 0 and _shift(g, h.p, K) == f:
                found.add(h.p)
    return sorted(found)


def _divisor_pairs(F: _Field[T], p0: list[T], pd: list[T]) -> list[tuple[list[T], list[T]]]:
    """The pairs ``(a, b)`` of monic divisors of ``p0`` and ``pd`` with
    ``gcd(a(n), b(n + h)) = 1`` for every nonnegative integer ``h``: a
    pair of irreducible factors, one of each, is incompatible when one is
    a shift of the other by a nonnegative integer, and the divisors are
    built from the compatible factors."""
    K = F.K
    factors_a = F.factor(p0)
    factors_b = F.factor(pd)
    incompatible = {(i, j) for i, (f, _) in enumerate(factors_a) for j, (g, _) in enumerate(factors_b)
                    if _shift_coincidences(F, f, g)}

    def divisors(factors: list[tuple[list[T], int]]) -> list[tuple[list[T], frozenset[int]]]:
        found: list[tuple[list[T], frozenset[int]]] = [(densebasic.dup_one(K), frozenset())]
        for index, (f, multiplicity) in enumerate(factors):
            found = [(densearith.dup_mul(d, densearith.dup_pow(f, e, K), K), used | ({index} if e else frozenset()))
                     for d, used in found for e in range(multiplicity + 1)]
        return found

    pairs: list[tuple[list[T], list[T]]] = []
    for a, used_a in divisors(factors_a):
        for b, used_b in divisors(factors_b):
            if any((i, j) in incompatible for i in used_a for j in used_b):
                continue
            pairs.append((a, b))
    return pairs


def _index(depth: int, avoid: set[Symbol]) -> Symbol:
    """A summation index for the given depth which is none of the
    symbols of the problem."""
    names = list(_INDEX_NAMES) + ['%s%d' % (c, i) for i in range(1, 100) for c in _INDEX_NAMES]
    taken = {s.name for s in avoid}
    free = [name for name in names if name not in taken]
    return Symbol(free[depth], integer=True)


def _start_of(F: _Field[T], a: list[T], b: list[T]) -> int:
    """The first index from which ``prod a(k)/b(k)`` has no vanishing
    factor: one more than the largest nonnegative integer root of ``a b``."""
    product = densearith.dup_mul(a, b, F.K)
    roots_ = _integer_roots(F, product) if densebasic.dup_degree(product) > 0 else []
    return roots_[-1] + 1 if roots_ else 0


def _product_expression(F: _Field[T], p: list[T], n: Symbol, start: int, k: Symbol) -> Expr:
    """``prod_{k = start}^{n - 1} p(k)`` with the linear factors written as
    rising factorials (the factors primitive over ``ZZ`` when they are
    rational, as SymPy's ``factor_list`` gives them)."""
    K = F.K
    content = F.to_expr(densebasic.dup_LC(p, K)) if p else ZERO
    pieces: list[Expr] = []
    for f, multiplicity in F.factor(p):
        coefficients = [F.to_expr(c) for c in f]
        lead: Expr = ONE
        if all(isinstance(c, Rational) for c in coefficients):
            scale = 1
            for c in coefficients:
                if isinstance(c, Rational):
                    scale = scale*c.q//_gcd(scale, c.q)
            lead = Integer(scale)
        content = content/lead**multiplicity
        if densebasic.dup_degree(f) == 1:
            root = -coefficients[1]
            piece: Expr = lead**(n - start)*rf(start - root, n - start)
        else:
            piece = Product((lead*F.expression(f, n)).subs(n, k), (k, start, n - 1))
        pieces.append(pow(piece, Integer(multiplicity)))
    return mul(pow(content, n - start), *pieces)


def _simplified(expr: Expr) -> Expr:
    """The quotients of rising factorials cancelled (``gammasimp``), when
    that keeps the expression free of gamma functions."""
    candidate = gammasimp(expr)
    if candidate.has(Gamma, Piecewise):
        candidate = expr
    replaced = candidate.replace(lambda e: isinstance(e, RisingFactorial), _factorial_form)
    if not isinstance(replaced, Expr):
        raise TypeError('%s is not an expression' % (replaced,))
    return replaced


def _factorial_form(e: Basic) -> Basic:
    """``RisingFactorial(a, m)`` as ``(a + m - 1)!/(a - 1)!`` for a positive
    integer ``a``."""
    if isinstance(e, RisingFactorial):
        a, m = e.args
        if isinstance(a, Integer) and a.p >= 1 and isinstance(m, Expr):
            return factorial(a + m - 1)/factorial(a - 1)
    return e


def _closed_form(expr: Expr) -> Expr:
    """The sums evaluated by the summation when that gives an expression
    without sums, products or special functions."""
    if not expr.has(Sum):
        return expr
    if any(isinstance(s, Sum) and s.function.has(Sum) for s in expr.atoms(Sum)):
        return expr  # the symbolic summation of nested sums took minutes
    from_one = expr.replace(lambda e: isinstance(e, Sum), _from_one)
    forms = [expr] + ([from_one] if isinstance(from_one, Expr) else [])
    for form in forms:
        try:
            candidate = form.doit()
        except (ValueError, TypeError, NotImplementedError, BasePolynomialError, ArithmeticError):
            continue
        if not candidate.has(Sum, Product, Gamma, UpperGamma, LowerGamma, Hyper, PolyLog, LerchPhi, Piecewise, Zeta):
            return candidate
    return expr


def _from_one(s: Basic) -> Basic:
    """``Sum(f(k), (k, 0, m))`` as ``Sum(f(k - 1), (k, 1, m + 1))``, the form
    in which the summation recognises the harmonic numbers."""
    if isinstance(s, Sum) and len(s.limits) == 1 and s.limits[0][1] == ZERO:
        k, _, hi = s.limits[0]
        return Sum(s.function.subs(k, k - 1), (k, ONE, hi + 1))
    return s


class _Hyper(ABC):
    """A hypergeometric solution with the first index from which it is
    defined, and the d'Alembertian reduction by it."""

    def __init__(self, expr: Expr, start: int) -> None:
        self.expr: Final[Expr] = expr
        self.start: Final[int] = start

    @abstractmethod
    def reduced_solutions(self, n: Symbol, depth: int, avoid: set[Symbol]) -> list[_Solution]:
        """The d'Alembertian solutions of the reduced operator;
        ``NotImplementedError`` when it lives over a field without a
        factorisation."""


class _HyperOver(_Hyper, Generic[T]):
    """A hypergeometric solution of the operator ``polys`` over ``F``,
    with its ratio ``num/den``."""

    def __init__(self, F: _Field[T], polys: list[list[T]], num: list[T], den: list[T], expr: Expr, start: int
                 ) -> None:
        super().__init__(expr, start)
        self.F: Final[_Field[T]] = F
        self.polys: Final[list[list[T]]] = polys
        self.num: Final[list[T]] = num
        self.den: Final[list[T]] = den

    def reduced_solutions(self, n: Symbol, depth: int, avoid: set[Symbol]) -> list[_Solution]:
        reduced = _reduced_operator(self.F, self.polys, self.num, self.den)
        return _dalembertian_records(self.F, reduced, n, depth, avoid)


class _Class(Generic[T]):
    """A similarity class of hypergeometric terms: a core
    ``z**n prod a/b`` (of ratio ``num/den``), the polynomials multiplying
    it, the products as expressions, and the key ordering the classes as
    they are found."""

    def __init__(self, key: tuple[int, int, int], num: list[T], den: list[T], start: int, products: Expr) -> None:
        self.key: Final[tuple[int, int, int]] = key
        self.num: Final[list[T]] = num
        self.den: Final[list[T]] = den
        self.start: Final[int] = start
        self.products: Final[Expr] = products
        self.polynomials: Final[list[list[T]]] = []


class _Candidate(Generic[T]):
    """A pair ``(a, b)`` of the search with the operator ``P_i`` it gives,
    for one root ``z``: everything over the field of the recurrence."""

    def __init__(self, key: tuple[int, int], a: list[T], b: list[T], P: list[list[T]], start: int, products: Expr
                 ) -> None:
        self.key: Final[tuple[int, int]] = key
        self.a: Final[list[T]] = a
        self.b: Final[list[T]] = b
        self.P: Final[list[list[T]]] = P
        self.start: Final[int] = start
        self.products: Final[Expr] = products


class _RootGroup(Generic[T]):
    """The candidates sharing a root ``z``: ``value`` when ``z`` is in the
    field, else the irreducible factor of which it is a root."""

    def __init__(self, expr: Expr, value: Optional[T], factor: list[T]) -> None:
        self.expr: Final[Expr] = expr
        self.value: Final[Optional[T]] = value
        self.factor: Final[list[T]] = factor
        self.candidates: Final[list[_Candidate[T]]] = []


def _hypergeometric_records(F: _Field[T], polys: list[list[T]], n: Symbol, depth: int, avoid: set[Symbol]
                            ) -> list[_Hyper]:
    """The hypergeometric solutions (Petkovšek), in the order in which
    their classes are found."""
    K = F.K
    d = len(polys) - 1
    if d == 0:
        return []
    k = _index(depth, avoid)
    groups: dict[Expr, _RootGroup[T]] = {}
    for pair_index, (a, b) in enumerate(_divisor_pairs(F, polys[0], _shift(polys[d], 1 - d, K))):
        P = [densearith.dup_mul(densearith.dup_mul(polys[i], _products(a, 0, i, K), K), _products(b, i, d, K), K)
             for i in range(d + 1)]
        m = max(densebasic.dup_degree(p) for p in P)
        leading = densebasic.dup_from_dict({i: densebasic.dup_LC(p, K) for i, p in enumerate(P)
                                            if densebasic.dup_degree(p) == m}, K)
        start = _start_of(F, a, b)
        products: Optional[Expr] = None
        for root_index, (expr, value, factor) in enumerate(_nonzero_roots(F, leading)):
            if products is None:
                products = _product_expression(F, a, n, start, k)/_product_expression(F, b, n, start, k)
            group = groups.get(expr)
            if group is None:
                group = groups[expr] = _RootGroup(expr, value, factor)
            group.candidates.append(_Candidate((pair_index, root_index), a, b, P, start, products))
    found: list[tuple[tuple[int, int, int], _Hyper]] = []
    for group in groups.values():
        if group.value is not None:
            found.extend(_classes_at(F, F, _identity, polys, group.value, group, n))
        else:
            L = F.extend(group.factor, group.expr)
            found.extend(_classes_at(F, L, F.embed, polys, L.from_expr(group.expr), group, n))
    found.sort(key=lambda item: item[0])
    return [record for _, record in found]


def _identity(c: T) -> T:
    return c


def _classes_at(F: _Field[T], G: _Field[U], embed: Callable[[T], U], polys: list[list[T]], z: U,
                group: _RootGroup[T], n: Symbol) -> list[tuple[tuple[int, int, int], _Hyper]]:
    """The hypergeometric solutions ``c(n) z**n prod a/b`` for the root
    ``z`` of ``G`` (``F`` itself, or ``F`` with ``z`` adjoined), grouped into
    similarity classes, with the keys of their classes."""
    L = G.K

    def mapped(p: list[T]) -> list[U]:
        return densebasic.dup_strip([embed(c) for c in p], L)

    operator = [mapped(p) for p in polys]
    classes: list[_Class[U]] = []
    for candidate in group.candidates:
        a, b = mapped(candidate.a), mapped(candidate.b)
        aux = _normalized(G, [densearith.dup_mul_ground(mapped(p), L.pow(z, i), L) for i, p in enumerate(candidate.P)])
        if not aux:
            continue
        core_num, core_den = _cancelled(densearith.dup_mul_ground(a, z, L), b, L)
        for c_index, c in enumerate(_polynomial_solutions(G, aux)):
            num, den = _cancelled(densearith.dup_mul(core_num, _shift(c, 1, L), L),
                                  densearith.dup_mul(core_den, c, L), L)
            key = (candidate.key[0], candidate.key[1], c_index)
            _classify(G, classes, key, num, den, core_num, core_den, c, candidate)
    records: list[tuple[tuple[int, int, int], _Hyper]] = []
    for cls in classes:
        for member in _polynomial_basis(G, cls.polynomials):
            num, den = _cancelled(densearith.dup_mul(cls.num, _shift(member, 1, L), L),
                                  densearith.dup_mul(cls.den, member, L), L)
            expr = _simplified(G.expression(member, n)*pow(group.expr, n)*cls.products)
            records.append((cls.key, _HyperOver(G, operator, num, den, expr, cls.start)))
    return records


def _classify(G: _Field[U], classes: list[_Class[U]], key: tuple[int, int, int], num: list[U], den: list[U],
              core_num: list[U], core_den: list[U], c: list[U], candidate: _Candidate[T]) -> None:
    """Put the term of ratio ``num/den`` (the core ``core_num/core_den``
    times ``c(n + 1)/c(n)``) into its similarity class."""
    L = G.K
    for cls in classes:
        q_num, q_den = _cancelled(densearith.dup_mul(num, cls.den, L), densearith.dup_mul(den, cls.num, L), L)
        q = _shift_quotient(G, q_num, q_den)
        if q is not None:
            cls.polynomials.append(q)
            return
    cls = _Class(key, core_num, core_den, candidate.start, candidate.products)
    cls.polynomials.append(densetools.dup_monic(c, L))
    classes.append(cls)


def _polynomial_basis(G: _Field[U], polynomials: Sequence[list[U]]) -> list[list[U]]:
    """A basis of the span of the polynomials (the rows of the reduced
    echelon form of their coefficients)."""
    L = G.K
    if not polynomials:
        return []
    degree = max(densebasic.dup_degree(p) for p in polynomials)
    rows = [{j: _coeff(p, j, L) for j in range(degree + 1) if not L.is_zero(_coeff(p, j, L))} for p in polynomials]
    return [densebasic.dup_from_dict(row, L) for row in L.sparse_echelon([r for r in rows if r])]


def _nonzero_roots(F: _Field[T], leading: list[T]) -> list[tuple[Expr, Optional[T], list[T]]]:
    """The nonzero roots of a polynomial in ``z``: the expression of each,
    with its value when it is in the field, or the irreducible factor of
    which it is a root otherwise; in the order of SymPy's ``roots`` when it
    finds them all."""
    K = F.K
    entries: list[tuple[Expr, Optional[T], list[T]]] = []
    for f, _ in F.factor(leading):
        if densebasic.dup_degree(f) == 1:
            value = K.neg(f[1])
            if not K.is_zero(value):
                entries.append((F.to_expr(value), value, f))
        else:
            for root in _irreducible_roots(F, f):
                entries.append((root, None, f))
    try:
        found = roots(F.expression(leading, _X), _X)
        order = list(found) if isinstance(found, dict) else list(found)
    except (BasePolynomialError, NotImplementedError, ValueError):
        order = []
    position = {e: i for i, e in enumerate(order)}
    return sorted(entries, key=lambda entry: position.get(entry[0], len(order)))


def _irreducible_roots(F: _Field[T], f: list[T]) -> list[Expr]:
    """The roots of the monic irreducible ``f`` of degree two or more, as
    expressions: SymPy's ``roots`` (ported) over ``QQ`` and ``QQ[a, ...]``,
    the formula of the quadratic otherwise."""
    K = F.K
    try:
        found = roots(F.expression(f, _X), _X)
        values = list(found) if isinstance(found, dict) else list(found)
    except (BasePolynomialError, NotImplementedError, ValueError):
        values = []
    if values or densebasic.dup_degree(f) != 2:
        return values
    p, q = f[1], f[2]
    discriminant = F.to_expr(K.sub(K.mul(p, p), K.mul(K.from_int(4), q)))
    half = -F.to_expr(p)/2
    r = sqrt(discriminant)/2
    return [half - r, half + r]


def rsolve_hypergeometric(coefficients: Coefficients, n: Symbol) -> list[Expr]:
    """A basis of the hypergeometric solutions of
    ``sum coefficients[i] * y(n + i) = 0``, the solutions ``y`` whose ratio
    ``y(n + 1)/y(n)`` is a rational function of ``n``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.recurr import rsolve_hypergeometric
    >>> n = Symbol('n')
    >>> rsolve_hypergeometric([-(n + 1), 1], n)
    [factorial(n)]
    >>> rsolve_hypergeometric([-1, -1, 1], n)
    [(1/2 - sqrt(5)/2)**n, (1/2 + sqrt(5)/2)**n]
    >>> rsolve_hypergeometric([n**2 + 1, -(n + 1)], n)
    [Product(k**2 + 1, (k, 0, n - 1))/factorial(n)]
    >>> rsolve_hypergeometric([n + 1, -(n + 3), 1], n)
    [factorial(n)]

    References
    ==========

    .. [1] M. Petkovšek, Hypergeometric solutions of linear recurrences
       with polynomial coefficients, J. Symbolic Computation 14 (1992).
    """
    exprs = [to_expr(c) for c in coefficients]
    context = _context(exprs, n)
    operator = context.operator(exprs, n)
    if operator.order < 0:
        raise ValueError("the recurrence is trivial")
    avoid = _avoided(exprs, n)
    return [context.restored(expr) for expr, start in operator.hypergeometric_solutions(avoid)
            if operator.checked(expr, ZERO, start)]


def _avoided(exprs: Sequence[Expr], n: Symbol) -> set[Symbol]:
    """The symbols the summation indices must not be."""
    return {n} | {s for e in exprs for s in e.free_symbols}


# ---------------------------------------------------------------------------
# d'Alembertian solutions

def _add_fractions(a: tuple[list[T], list[T]], b: tuple[list[T], list[T]], K: Domain[T]) -> tuple[list[T], list[T]]:
    return _cancelled(densearith.dup_add(densearith.dup_mul(a[0], b[1], K), densearith.dup_mul(b[0], a[1], K), K),
                      densearith.dup_mul(a[1], b[1], K), K)


def _reduced_operator(F: _Field[T], polys: Sequence[list[T]], num: list[T], den: list[T]) -> list[list[T]]:
    """The operator ``L'`` of order ``d - 1`` with ``L[h sum z] =
    h L'[z]`` for the hypergeometric solution ``h`` of ratio
    ``r = num/den``: ``L' z = sum_j (sum_{i > j} p_i(n) rho_i(n)) z(n + j)``
    with ``rho_i = prod_{m < i} r(n + m)``, cleared of denominators."""
    K = F.K
    d = len(polys) - 1
    one = densebasic.dup_one(K)
    rho: list[tuple[list[T], list[T]]] = [(one, one)]
    for i in range(1, d + 1):
        previous = rho[-1]
        rho.append(_cancelled(densearith.dup_mul(previous[0], _shift(num, i - 1, K), K),
                              densearith.dup_mul(previous[1], _shift(den, i - 1, K), K), K))
    fractions: list[tuple[list[T], list[T]]] = []
    for j in range(d):
        total: tuple[list[T], list[T]] = ([], one)
        for i in range(j + 1, d + 1):
            total = _add_fractions(total, (densearith.dup_mul(polys[i], rho[i][0], K), rho[i][1]), K)
        fractions.append(total)
    common = one
    for _, q in fractions:
        common = euclidtools.dup_lcm(common, q, K)
    reduced = [densearith.dup_mul(p, densearith.dup_quo(common, q, K), K) for p, q in fractions]
    while reduced and not reduced[-1]:
        reduced.pop()
    return _normalized(F, reduced)


def _dalembertian_records(F: _Field[T], polys: list[list[T]], n: Symbol, depth: int, avoid: set[Symbol]
                          ) -> list[_Solution]:
    hyper = _hypergeometric_records(F, polys, n, depth, avoid)
    if not hyper:
        return []
    records: list[_Solution] = [(h.expr, h.start) for h in hyper]
    order = len(polys) - 1
    if order <= 1 or len(hyper) >= order:
        return records
    # the hypergeometric solutions come first; the sums h sum z add the
    # directions the hypergeometric ones do not span (by the first solution
    # whose reduced operator is over a field with a factorisation)
    k = _index(depth, avoid)
    for h in hyper:
        try:
            reduced = h.reduced_solutions(n, depth + 1, avoid)
        except NotImplementedError:
            continue
        for z, z_start in reduced:
            lower = max(h.start, z_start)
            records.append((h.expr*Sum(z.subs(n, k), (k, Integer(lower), n - 1)), lower))
        break
    return _independent(F, records, polys, n)


def _independent(F: _Field[T], records: list[_Solution], polys: Sequence[list[T]], n: Symbol) -> list[_Solution]:
    """A maximal independent subset of the solutions, the earlier ones
    preferred: solutions of a recurrence of order ``d`` are independent
    when their values at ``d`` consecutive points, past the roots of the
    leading coefficient, are."""
    d = len(polys) - 1
    leading = polys[d]
    roots_ = _integer_roots(F, leading) if densebasic.dup_degree(leading) > 0 else []
    start = max([r[1] for r in records] + roots_ + [-1]) + 1
    chosen: list[_Solution] = []
    rows: list[list[Expr]] = []
    for record in records:
        if len(chosen) >= d:
            break
        values = [_value(record[0], n, at) for at in range(start, start + d)]
        if any(v.is_finite is False for v in values):
            continue
        if _rank(rows + [values]) > len(rows):
            rows.append(values)
            chosen.append(record)
    return chosen


def rsolve_dalembertian(coefficients: Coefficients, n: Symbol) -> list[Expr]:
    """A basis of the d'Alembertian solutions of
    ``sum coefficients[i] * y(n + i) = 0``: the nested indefinite sums of
    hypergeometric terms (Abramov and Petkovšek). When their number is the
    order of the recurrence, they are all the solutions.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.recurr import rsolve_dalembertian
    >>> n = Symbol('n')
    >>> rsolve_dalembertian([n + 1, -(2*n + 3), n + 2], n)
    [1, harmonic(n)]
    >>> rsolve_dalembertian([n + 1, -(n + 3), 1], n)
    [factorial(n), factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))]
    >>> rsolve_dalembertian([1, -2, 1], n)
    [1, n]
    >>> rsolve_dalembertian([1, -(n + 1), 1], n)
    []

    References
    ==========

    .. [1] S. A. Abramov, M. Petkovšek, D'Alembertian solutions of linear
       differential and difference equations, ISSAC '94.
    """
    exprs = [to_expr(c) for c in coefficients]
    context = _context(exprs, n)
    operator = context.operator(exprs, n)
    if operator.order < 0:
        raise ValueError("the recurrence is trivial")
    avoid = _avoided(exprs, n)
    return [context.restored(_closed_form(expr)) for expr, start in operator.dalembertian_solutions(avoid)
            if operator.checked(expr, ZERO, start)]


# ---------------------------------------------------------------------------
# verification and linear algebra on expressions

def _value(expr: Expr, n: Symbol, at: int) -> Expr:
    """``expr`` at ``n = at``, the sums and products with integer limits
    written out (the summation looks for closed forms first, which is
    slow and may fail)."""
    return _evaluated(expr.subs(n, Integer(at)))


def _evaluated(expr: Expr) -> Expr:
    if not expr.has(Sum, Product):
        return expr.doit() if expr.has(RisingFactorial, Factorial) else expr
    if isinstance(expr, (Sum, Product)):
        function = expr.function
        for index, lo, hi in expr.limits:
            if not (isinstance(lo, Integer) and isinstance(hi, Integer)):
                return expr.doit()
            values = [_evaluated(function.subs(index, Integer(i))) for i in range(lo.p, hi.p + 1)]
            function = add(*values) if isinstance(expr, Sum) else mul(*values)
        return function
    if not expr.args:
        return expr
    rebuilt = expr.func(*[_evaluated(a) if isinstance(a, Expr) else a for a in expr.args])
    if not isinstance(rebuilt, Expr):
        raise TypeError('%s is not an expression' % (rebuilt,))
    return rebuilt.doit()


def _normal(e: Expr) -> Expr:
    """``e`` expanded and cancelled (a normal form for the rational
    functions of the symbols and the radicals in it)."""
    expanded = expand(e)
    try:
        return cancel(expanded)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return expanded


def _is_zero(value: Expr) -> bool:
    """Whether ``value`` is zero: exactly for the rational functions and
    the algebraic numbers (by their minimal polynomial), numerically at 30
    digits for the other numbers."""
    if value == ZERO:
        return True
    normal = _normal(value)
    if normal == ZERO:
        return True
    if normal.free_symbols:
        from sympy_extras.simplify.simplify import simplify
        try:
            return simplify(normal) == ZERO
        except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
            return False
    if isinstance(normal, Rational):
        return False
    # a number which is not small at 120 digits is not zero: the minimal
    # polynomial (resultants, hopeless on Cardano's formulas) is only
    # asked to prove the zeros
    approximation = numeric_value(normal, 120)
    if approximation is not None and bool(abs(approximation) > mpmath.mpf(10)**-20):
        return False
    try:
        return minimal_polynomial_coeffs(normal) == [1, 0]
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        pass
    return approximation is not None


def _row_reduce(rows: list[list[Expr]]) -> tuple[list[list[Expr]], list[int]]:
    """The reduced row echelon form of a matrix of expressions and its
    pivot columns (the pivots decided by :func:`_is_zero`)."""
    matrix = [[_normal(e) for e in row] for row in rows]
    pivots: list[int] = []
    columns = len(matrix[0]) if matrix else 0
    r = 0
    for c in range(columns):
        if r == len(matrix):
            break
        p = next((i for i in range(r, len(matrix)) if not _is_zero(matrix[i][c])), None)
        if p is None:
            continue
        matrix[r], matrix[p] = matrix[p], matrix[r]
        pivot = matrix[r][c]
        matrix[r] = [ONE if j == c else _normal(e/pivot) for j, e in enumerate(matrix[r])]
        for i in range(len(matrix)):
            if i != r and not _is_zero(matrix[i][c]):
                factor = matrix[i][c]
                matrix[i] = [ZERO if j == c else _normal(e - factor*matrix[r][j]) for j, e in enumerate(matrix[i])]
        pivots.append(c)
        r += 1
    return matrix, pivots


def _rank(rows: list[list[Expr]]) -> int:
    return len(_row_reduce(rows)[1])


def _solve_linear(rows: list[list[Expr]], rhs: list[Expr], columns: int) -> Optional[list[Expr]]:
    """A solution of ``rows * x = rhs`` (the free unknowns zero), ``None``
    when there is none."""
    if not rows:
        return [ZERO]*columns
    reduced, pivots = _row_reduce([row + [v] for row, v in zip(rows, rhs)])
    if columns in pivots:
        return None
    solution: list[Expr] = [ZERO]*columns
    for row, column in enumerate(pivots):
        solution[column] = reduced[row][columns]
    return solution


def is_recurrence_solution(coefficients: Coefficients, n: Symbol, solution: ExprLike, rhs: ExprLike = 0,
                           start: int = 0, count: int = 6) -> bool:
    """Whether ``solution`` satisfies ``sum coefficients[i] y(n + i) =
    rhs`` at ``n = start, ..., start + count - 1``, exactly.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import factorial, harmonic
    >>> from sympy_extras.solvers.recurr import is_recurrence_solution
    >>> n = Symbol('n')
    >>> is_recurrence_solution([-(n + 1), 1], n, factorial(n))
    True
    >>> is_recurrence_solution([n + 1, -(2*n + 3), n + 2], n, harmonic(n))
    True
    >>> is_recurrence_solution([-2, 1], n, 3**n)
    False
    """
    exprs = [to_expr(c) for c in coefficients]
    context = _context(exprs, n)
    operator = context.operator(exprs, n, normalize=False)
    return operator.checked(context.rename(to_expr(solution)), context.rename(to_expr(rhs)), start, count)


def _checked(coefficients: Sequence[Expr], n: Symbol, solution: Expr, rhs: Expr, start: int, count: int = 6) -> bool:
    for at in range(start, start + count):
        value = add(*[_value(c, n, at)*_value(solution, n, at + i) for i, c in enumerate(coefficients)]) \
            - _value(rhs, n, at)
        if value.is_finite is False or not _is_zero(value):
            return False
    return True


# ---------------------------------------------------------------------------
# equations

def _parse(equation: Union[Expr, Eq], y: AppliedUndef) -> tuple[list[Expr], Expr, Symbol]:
    """``(coefficients, rhs, n)`` of a linear recurrence written with
    ``y(n + i)``: ``sum coefficients[i] y(n + i) = rhs``, the index
    shifted so that the least shift is zero (the same recurrence)."""
    argument = y.args[0] if len(y.args) == 1 else None
    if not isinstance(argument, Symbol):
        raise ValueError("%s is not a function of one symbol" % (y,))
    n = argument
    expr = equation.lhs - equation.rhs if isinstance(equation, Eq) else equation

    def applications(e: Expr) -> list[AppliedUndef]:
        return [t for t in e.atoms(AppliedUndef) if isinstance(t, AppliedUndef) and t.func == y.func]

    terms = applications(expr)
    if not terms:
        raise ValueError("no %s in the equation" % (y,))
    shifts: list[int] = []
    for t in terms:
        point = t.args[0] if len(t.args) == 1 else None
        offset = point - n if isinstance(point, Expr) else None
        if not isinstance(offset, Integer):
            raise ValueError("%s is not a shift of %s by an integer" % (t, y))
        shifts.append(offset.p)
    least = min(shifts)
    shifted = expr.subs(n, n - least)
    order = max(shifts) - least
    coefficients, rhs = _linear_parts(together(shifted).as_numer_denom()[0], y, n, order)
    if any(c.is_rational_function(n) is not True for c in coefficients):
        # the denominator of the right-hand side is no polynomial in n
        # (factorials): the equation is not multiplied by it
        coefficients, rhs = _linear_parts(shifted, y, n, order)
    return coefficients, rhs, n


def _linear_parts(expr: Expr, y: AppliedUndef, n: Symbol, order: int) -> tuple[list[Expr], Expr]:
    """``(coefficients, rhs)`` of ``expr = sum coefficients[i] y(n + i) -
    rhs``."""
    unknowns: list[Expr] = [Dummy('y%d' % i) for i in range(order + 1)]
    expr = expand(expr.subs({y.func(n + i): u for i, u in enumerate(unknowns)}))
    if any(isinstance(t, AppliedUndef) and t.func == y.func for t in expr.atoms(AppliedUndef)):
        raise ValueError("the equation is not linear in %s" % (y,))
    coefficients: list[Expr] = [ZERO]*(order + 1)
    rhs: Expr = ZERO
    for term in Add.make_args(expr):
        coefficient, dependent = term.as_independent(*unknowns)
        if dependent == ONE:
            rhs = rhs - coefficient
        elif dependent in unknowns:
            i = unknowns.index(dependent)
            coefficients[i] = coefficients[i] + coefficient
        else:
            raise ValueError("the equation is not linear in %s" % (y,))
    return coefficients, rhs


def _hypergeometric_ratio(f: Expr, n: Symbol) -> Optional[Expr]:
    """``f(n + 1)/f(n)`` when it is a rational function of ``n``."""
    from sympy_extras.concrete.gosper import hypersimp
    if f == ZERO:
        return None
    ratio = hypersimp(f, n)
    if ratio is None:
        return None
    ratio = cancel(ratio)
    if ratio.is_rational_function(n) and not ratio.has(Sum, Product):
        return ratio
    return None


def rsolve_linear(equation: Union[Expr, Eq], y: AppliedUndef,
                  init: Optional[dict[AppliedUndef, Union[Expr, int]]] = None) -> Optional[Expr]:
    """The general solution of a linear recurrence with polynomial (or
    rational) coefficients, written with ``y(n + i)``, as a d'Alembertian
    expression with the constants ``C0, C1, ...``; ``None`` when the
    d'Alembertian solutions do not span the solutions (fewer than the
    order), or when the right-hand side is not a sum of hypergeometric
    terms with a d'Alembertian particular solution. With ``init``, the
    values ``{y(0): ..., y(1): ...}`` determine the constants.

    Examples
    ========

    >>> from sympy_extras.core import Eq, Function, Symbol
    >>> from sympy_extras.solvers.recurr import rsolve_linear
    >>> n = Symbol('n')
    >>> y = Function('y')
    >>> rsolve_linear(y(n + 1) - (n + 1)*y(n) - 1, y(n))
    C0*factorial(n) + factorial(n)*Sum(1/factorial(k + 1), (k, 0, n - 1))
    >>> rsolve_linear(Eq((n + 2)*y(n + 2), (2*n + 3)*y(n + 1) - (n + 1)*y(n)), y(n), {y(0): 0, y(1): 1})
    harmonic(n)
    >>> rsolve_linear(y(n + 2) - y(n + 1) - y(n), y(n), {y(0): 0, y(1): 1})
    -sqrt(5)*(1/2 - sqrt(5)/2)**n/5 + sqrt(5)*(1/2 + sqrt(5)/2)**n/5
    >>> rsolve_linear(y(n + 1) - 2*y(n) - n, y(n))
    2**n*C0 - n - 1
    >>> rsolve_linear(y(n + 2) - (n + 1)*y(n + 1) + y(n), y(n)) is None
    True

    The derangement numbers, ``y(n + 1) = n (y(n) + y(n - 1))``:

    >>> d = rsolve_linear(y(n + 1) - n*(y(n) + y(n - 1)), y(n), {y(0): 1, y(1): 0})
    >>> d
    -factorial(n)*Sum((-1)**k/factorial(k + 1), (k, 0, n - 1)) + factorial(n)
    >>> [d.subs(n, i).doit() for i in range(7)]
    [1, 0, 1, 2, 9, 44, 265]
    """
    coefficients, rhs, n = _parse(equation, y)
    terms = [] if rhs == ZERO else list(Add.make_args(expand(rhs)))
    ratios = [_hypergeometric_ratio(term, n) for term in terms]
    context = _context(coefficients + [r for r in ratios if r is not None], n)
    operator = context.operator(coefficients, n)
    if operator.order < 0:
        raise ValueError("the recurrence is trivial")
    order = operator.order
    avoid = _avoided(coefficients + [rhs], n)
    homogeneous = [_closed_form(expr) for expr, start in operator.dalembertian_solutions(avoid)
                   if operator.checked(expr, ZERO, start)]
    if len(homogeneous) < order:
        return None
    particular: Expr = ZERO
    for term, ratio in zip(terms, ratios):
        if ratio is None:
            return None
        found = operator.particular(context.rename(term), context.rename(ratio), avoid)
        if found is None:
            return None
        particular = particular + _closed_form(found[0])
    constants = [Symbol('C%d' % i) for i in range(order)]
    general = context.restored(particular + add(*[c*h for c, h in zip(constants, homogeneous)]))
    if init is None:
        return general
    rows: list[list[Expr]] = []
    values: list[Expr] = []
    zeros: dict[Expr, Expr] = {c: ZERO for c in constants}
    for point, value in init.items():
        at = point.args[0] if len(point.args) == 1 else None
        if not isinstance(at, Integer):
            raise ValueError("the initial values must be given at integers: %s" % (point,))
        evaluated = _value(general, n, at.p)
        rows.append([diff(evaluated, c) for c in constants])
        values.append(to_expr(value) - evaluated.subs(zeros))
    solution = _solve_linear(rows, values, len(constants))
    if solution is None:
        raise ValueError("the initial values are inconsistent with the recurrence")
    return general.subs(dict(zip(constants, solution)))
