"""Sparse multivariate polynomials over a typed domain: the ring
``K[x_1, ..., x_n]`` with a monomial order, elements as dictionaries
from exponent tuples to nonzero coefficients.

This is the representation of the Gröbner basis algorithms: leading
terms under the order, S-polynomials, division with remainder by a list
of polynomials (the multivariate reduction), and the arithmetic. The
dense tower representation of :mod:`sympy_extras.polys.densebasic` is
the one of the gcd and factorisation algorithms (the depth of that
nesting is the number of variables, which Python's generics cannot
express, so the two representations are converted through the
coefficient dictionaries by the algorithms that need both).

Examples
========

>>> from sympy_extras.polys.domains import QQ
>>> from sympy_extras.polys.rings import PolyRing, grevlex
>>> R = PolyRing(QQ, ['x', 'y'], grevlex)
>>> x, y = R.gens
>>> f = x**2*y - R.one
>>> repr(f)
PolyElement({(2, 1): Fraction(1, 1), (0, 0): Fraction(-1, 1)}, QQ[x,y])
>>> f.LM, f.LC, f.degree(), str(f)
((2, 1), Fraction(1, 1), 3, 'x**2*y - 1')
>>> (x + y)**2 == x**2 + (x*y).mul_ground(QQ.from_int(2)) + y**2
True
>>> q, r = (x**2*y + x).div([x*y - R.one])
>>> str(q[0]), str(r)
('x', '2*x')
>>> R.from_sympy(R.to_sympy(f)) == f
True

References
==========

.. [1] D. Cox, J. Little, D. O'Shea, *Ideals, Varieties, and Algorithms*,
       4th ed., Springer, 2015, Chapter 2 (monomial orders, the division
       algorithm).
"""
from __future__ import annotations

from typing import Callable, Generic, Iterator, Sequence, TypeVar

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly

from sympy_extras._typing import Monomial, as_expr

from .domains.domain import Domain

__all__ = ['PolyRing', 'PolyElement', 'MonomialOrder', 'lex', 'grlex', 'grevlex', 'monomial_mul', 'monomial_div',
           'monomial_lcm', 'monomial_gcd', 'monomial_divides']

T = TypeVar('T')
U = TypeVar('U')


def monomial_mul(a: Monomial, b: Monomial) -> Monomial:
    return tuple(i + j for i, j in zip(a, b))


def monomial_div(a: Monomial, b: Monomial) -> Monomial:
    """``a/b`` (``ValueError`` when ``b`` does not divide ``a``)."""
    if any(i < j for i, j in zip(a, b)):
        raise ValueError('%s does not divide %s' % (b, a))
    return tuple(i - j for i, j in zip(a, b))


def monomial_divides(a: Monomial, b: Monomial) -> bool:
    """Whether ``a`` divides ``b``."""
    return all(i <= j for i, j in zip(a, b))


def monomial_lcm(a: Monomial, b: Monomial) -> Monomial:
    return tuple(max(i, j) for i, j in zip(a, b))


def monomial_gcd(a: Monomial, b: Monomial) -> Monomial:
    return tuple(min(i, j) for i, j in zip(a, b))


class MonomialOrder:
    """A monomial order given by a key function on exponent tuples (a
    larger key is a larger monomial) and a name; ``is_graded`` says
    whether the total degree comes first."""

    def __init__(self, name: str, key: Callable[[Monomial], tuple[int, ...]], is_graded: bool) -> None:
        self.name: str = name
        self.key: Callable[[Monomial], tuple[int, ...]] = key
        self.is_graded: bool = is_graded

    def __call__(self, monomial: Monomial) -> tuple[int, ...]:
        return self.key(monomial)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MonomialOrder) and other.name == self.name

    def __hash__(self) -> int:
        return hash(('MonomialOrder', self.name))


def _lex_key(m: Monomial) -> tuple[int, ...]:
    return m


def _grlex_key(m: Monomial) -> tuple[int, ...]:
    return (sum(m),) + m


def _grevlex_key(m: Monomial) -> tuple[int, ...]:
    return (sum(m),) + tuple(-e for e in reversed(m))


lex = MonomialOrder('lex', _lex_key, False)
grlex = MonomialOrder('grlex', _grlex_key, True)
grevlex = MonomialOrder('grevlex', _grevlex_key, True)


class PolyRing(Generic[T]):
    """The polynomial ring over ``domain`` in the variables ``symbols``
    with the monomial order ``order``."""

    def __init__(self, domain: Domain[T], symbols: Sequence[str], order: MonomialOrder = lex) -> None:
        if not symbols:
            raise ValueError('a polynomial ring needs at least one variable')
        self.domain: Domain[T] = domain
        self.symbols: list[Symbol] = [Symbol(s) for s in symbols]
        self.names: list[str] = list(symbols)
        self.order: MonomialOrder = order
        self.ngens: int = len(symbols)
        self.zero_monom: Monomial = tuple(0 for _ in symbols)

    def __repr__(self) -> str:
        return '%s[%s]' % (self.domain, ','.join(self.names))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, PolyRing) and other.domain == self.domain and other.names == self.names
                and other.order == self.order)

    def __hash__(self) -> int:
        return hash(('PolyRing', self.domain, tuple(self.names), self.order))

    @property
    def zero(self) -> PolyElement[T]:
        return PolyElement({}, self)

    @property
    def one(self) -> PolyElement[T]:
        return PolyElement({self.zero_monom: self.domain.one}, self)

    @property
    def gens(self) -> list[PolyElement[T]]:
        result: list[PolyElement[T]] = []
        for i in range(self.ngens):
            monomial = tuple(1 if j == i else 0 for j in range(self.ngens))
            result.append(PolyElement({monomial: self.domain.one}, self))
        return result

    def term(self, monomial: Monomial, coefficient: T) -> PolyElement[T]:
        if self.domain.is_zero(coefficient):
            return self.zero
        return PolyElement({monomial: coefficient}, self)

    def ground(self, c: T) -> PolyElement[T]:
        return self.term(self.zero_monom, c)

    def from_int(self, n: int) -> PolyElement[T]:
        return self.ground(self.domain.from_int(n))

    def from_dict(self, terms: dict[Monomial, T]) -> PolyElement[T]:
        return PolyElement({m: c for m, c in terms.items() if not self.domain.is_zero(c)}, self)

    def from_sympy(self, expr: Expr) -> PolyElement[T]:
        """The polynomial of an expression in the variables whose
        coefficients belong to the domain."""
        p = Poly(as_expr(expr), *self.symbols)
        terms: dict[Monomial, T] = {}
        for monomial, coefficient in p.terms():
            c = self.domain.from_sympy(as_expr(coefficient))
            if not self.domain.is_zero(c):
                terms[tuple(int(e) for e in monomial)] = c
        return PolyElement(terms, self)

    def to_sympy(self, f: PolyElement[T]) -> Expr:
        total: Expr = as_expr(0)
        for monomial, coefficient in f.terms.items():
            term = self.domain.to_sympy(coefficient)
            for s, e in zip(self.symbols, monomial):
                if e:
                    term = as_expr(term*s**e)
            total = as_expr(total + term)
        return total

    def with_order(self, order: MonomialOrder) -> PolyRing[T]:
        return PolyRing(self.domain, self.names, order)


class PolyElement(Generic[T]):
    """A sparse polynomial: ``terms`` maps exponent tuples to nonzero
    coefficients of the ring's domain."""

    __slots__ = ('terms', 'ring')

    def __init__(self, terms: dict[Monomial, T], ring: PolyRing[T]) -> None:
        self.terms: dict[Monomial, T] = terms
        self.ring: PolyRing[T] = ring

    def __repr__(self) -> str:
        ordered = {m: self.terms[m] for m in self.monomials()}
        return 'PolyElement(%r, %r)' % (ordered, self.ring)

    def __str__(self) -> str:
        if not self.terms:
            return '0'
        K = self.ring.domain
        parts: list[str] = []
        for i, m in enumerate(self.monomials()):
            c = self.terms[m]
            negative = K.is_ordered and K.is_negative(c)
            magnitude = K.neg(c) if negative else c
            factors = ['%s**%d' % (name, e) if e > 1 else name for name, e in zip(self.ring.names, m) if e]
            if factors and K.is_one(magnitude):
                body = '*'.join(factors)
            else:
                body = '*'.join([K.to_str(magnitude)] + factors)
            if i == 0:
                parts.append(('-' if negative else '') + body)
            else:
                parts.append((' - ' if negative else ' + ') + body)
        return ''.join(parts)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PolyElement) and other.terms == self.terms and other.ring == self.ring

    def __hash__(self) -> int:
        return hash((frozenset(self.terms.items()), self.ring))

    def __bool__(self) -> bool:
        return bool(self.terms)

    def __iter__(self) -> Iterator[tuple[Monomial, T]]:
        return iter((m, self.terms[m]) for m in self.monomials())

    def copy(self) -> PolyElement[T]:
        return PolyElement(dict(self.terms), self.ring)

    # -- leading data

    def monomials(self) -> list[Monomial]:
        """The monomials in decreasing order."""
        return sorted(self.terms, key=self.ring.order.key, reverse=True)

    @property
    def LM(self) -> Monomial:
        if not self.terms:
            return self.ring.zero_monom
        return max(self.terms, key=self.ring.order.key)

    @property
    def LC(self) -> T:
        if not self.terms:
            return self.ring.domain.zero
        return self.terms[self.LM]

    @property
    def LT(self) -> tuple[Monomial, T]:
        m = self.LM
        return m, self.terms.get(m, self.ring.domain.zero)

    def degree(self) -> int:
        """The total degree (``-1`` for zero)."""
        if not self.terms:
            return -1
        return max(sum(m) for m in self.terms)

    def degree_in(self, i: int) -> int:
        if not self.terms:
            return -1
        return max(m[i] for m in self.terms)

    @property
    def is_zero(self) -> bool:
        return not self.terms

    @property
    def is_ground(self) -> bool:
        return not self.terms or (len(self.terms) == 1 and self.ring.zero_monom in self.terms)

    @property
    def is_monic(self) -> bool:
        return bool(self.terms) and self.ring.domain.is_one(self.LC)

    def coeff(self, monomial: Monomial) -> T:
        return self.terms.get(monomial, self.ring.domain.zero)

    # -- arithmetic

    def __neg__(self) -> PolyElement[T]:
        K = self.ring.domain
        return PolyElement({m: K.neg(c) for m, c in self.terms.items()}, self.ring)

    def __add__(self, other: PolyElement[T]) -> PolyElement[T]:
        K = self.ring.domain
        result = dict(self.terms)
        for m, c in other.terms.items():
            if m in result:
                s = K.add(result[m], c)
                if K.is_zero(s):
                    del result[m]
                else:
                    result[m] = s
            else:
                result[m] = c
        return PolyElement(result, self.ring)

    def __sub__(self, other: PolyElement[T]) -> PolyElement[T]:
        return self + (-other)

    def __mul__(self, other: PolyElement[T]) -> PolyElement[T]:
        K = self.ring.domain
        result: dict[Monomial, T] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                m = monomial_mul(m1, m2)
                product = K.mul(c1, c2)
                if m in result:
                    s = K.add(result[m], product)
                    if K.is_zero(s):
                        del result[m]
                    else:
                        result[m] = s
                else:
                    result[m] = product
        return PolyElement(result, self.ring)

    def __pow__(self, n: int) -> PolyElement[T]:
        if n < 0:
            raise ValueError('negative exponent')
        result = self.ring.one
        base = self
        while n:
            if n & 1:
                result = result*base
            n >>= 1
            if n:
                base = base*base
        return result

    def mul_term(self, monomial: Monomial, coefficient: T) -> PolyElement[T]:
        K = self.ring.domain
        if K.is_zero(coefficient):
            return self.ring.zero
        return PolyElement({monomial_mul(m, monomial): K.mul(c, coefficient) for m, c in self.terms.items()}, self.ring)

    def mul_ground(self, coefficient: T) -> PolyElement[T]:
        return self.mul_term(self.ring.zero_monom, coefficient)

    def mul_monom(self, monomial: Monomial) -> PolyElement[T]:
        return PolyElement({monomial_mul(m, monomial): c for m, c in self.terms.items()}, self.ring)

    def exquo_ground(self, coefficient: T) -> PolyElement[T]:
        K = self.ring.domain
        return PolyElement({m: K.exquo(c, coefficient) for m, c in self.terms.items()}, self.ring)

    def monic(self) -> PolyElement[T]:
        """``self`` divided by its leading coefficient (a field)."""
        if not self.terms:
            return self
        return self.mul_ground(self.ring.domain.invert(self.LC))

    def primitive(self) -> tuple[T, PolyElement[T]]:
        """The content (gcd of the coefficients) and the primitive part."""
        K = self.ring.domain
        content = K.zero
        for c in self.terms.values():
            content = K.gcd(content, c)
        if K.is_zero(content) or K.is_one(content):
            return content, self
        return content, self.exquo_ground(content)

    def div(self, divisors: Sequence[PolyElement[T]]) -> tuple[list[PolyElement[T]], PolyElement[T]]:
        """The division algorithm: quotients ``q_i`` and a remainder
        ``r`` with ``self = sum q_i g_i + r``, no term of ``r`` divisible
        by a leading monomial of the divisors (the leading coefficients
        must divide exactly: a field, or divisors with unit leading
        coefficients)."""
        K = self.ring.domain
        quotients = [self.ring.zero for _ in divisors]
        remainder: dict[Monomial, T] = {}
        p = self.copy()
        leading = [(g.LM, g.LC) for g in divisors]
        while p.terms:
            m, c = p.LT
            for i, (lm, lc) in enumerate(leading):
                if divisors[i].terms and monomial_divides(lm, m):
                    q_monomial = monomial_div(m, lm)
                    q_coefficient = K.exquo(c, lc)
                    quotients[i] = quotients[i] + self.ring.term(q_monomial, q_coefficient)
                    p = p - divisors[i].mul_term(q_monomial, q_coefficient)
                    break
            else:
                remainder[m] = c
                del p.terms[m]
        return quotients, PolyElement(remainder, self.ring)

    def rem(self, divisors: Sequence[PolyElement[T]]) -> PolyElement[T]:
        return self.div(divisors)[1]

    def exquo(self, other: PolyElement[T]) -> PolyElement[T]:
        """The exact quotient by one polynomial; ``ValueError`` otherwise."""
        quotients, remainder = self.div([other])
        if remainder.terms:
            raise ValueError('%s does not divide %s' % (other, self))
        return quotients[0]

    # -- calculus and evaluation

    def diff(self, i: int) -> PolyElement[T]:
        """The derivative with respect to the ``i``-th variable."""
        K = self.ring.domain
        result: dict[Monomial, T] = {}
        for m, c in self.terms.items():
            if m[i]:
                d = K.mul(c, K.from_int(m[i]))
                if not K.is_zero(d):
                    result[tuple(e - 1 if j == i else e for j, e in enumerate(m))] = d
        return PolyElement(result, self.ring)

    def evaluate(self, i: int, a: T) -> PolyElement[T]:
        """The polynomial with the ``i``-th variable set to ``a`` (still an
        element of the same ring, constant in that variable)."""
        K = self.ring.domain
        result = self.ring.zero
        for m, c in self.terms.items():
            value = K.mul(c, K.pow(a, m[i]))
            result = result + self.ring.term(tuple(0 if j == i else e for j, e in enumerate(m)), value)
        return result

    def to_sympy(self) -> Expr:
        return self.ring.to_sympy(self)
