"""Differential polynomials and Ritt's reduction.

A differential polynomial `p` which is not in the field of coefficients
has a *leader* `v` (its highest derivative for the ranking), and is a
polynomial `p = i_p v^d + \\cdots` in it: `i_p` is its *initial*, `d` its
degree and `s_p = \\partial p / \\partial v` its *separant*. A proper
derivative `\\theta p` has the leader `\\theta v`, the degree one and the
initial `s_p`.

`q` is *partially reduced* with respect to `p` when no proper derivative of
`v` occurs in it, and *reduced* when moreover its degree in `v` is lower
than `d`. Ritt's reduction makes a polynomial reduced with respect to a
set `A` by pseudo-divisions: the *partial remainder* by the derivatives of
the elements of `A` (which multiplies by separants only), then the
pseudo-remainder by the elements themselves (which multiplies by
initials), so that

.. math:: h\\, q = r \\pmod{[A]}, \\qquad h \\text{ a product of initials and separants of } A .

References
==========

.. [Ritt] J. F. Ritt, Differential Algebra, AMS Colloquium Publications 33,
   1950, chapter I.
.. [Kolchin] E. R. Kolchin, Differential Algebra and Algebraic Groups,
   Academic Press 1973, chapter I, section 9.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr

from sympy_extras._typing import DomainElement, as_expr

from .ring import DifferentialRing, Equation, Jet, JetMonomial, Terms

__all__ = ['DifferentialPolynomial', 'Rank']

#: the rank of a differential polynomial: the key of its leader and its degree
Rank = tuple[tuple[int, ...], int]


def _times(m: JetMonomial, k: JetMonomial) -> JetMonomial:
    if not m:
        return k
    if not k:
        return m
    powers = dict(m)
    for jet, e in k:
        powers[jet] = powers.get(jet, 0) + e
    return tuple(sorted(powers.items()))


class DifferentialPolynomial:
    """A polynomial in the derivatives of the functions of a
    :class:`~sympy_extras.polys.differential.ring.DifferentialRing`.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.polys.differential import DifferentialRing, DifferentialPolynomial
    >>> x = symbols('x')
    >>> u = Function('u')(x)
    >>> R = DifferentialRing([u])
    >>> p = DifferentialPolynomial.from_expr(R, u.diff(x)**2 - 4*u)
    >>> p.leader_expr(), p.degree(), p.initial().to_expr(), p.separant().to_expr()
    (Derivative(u(x), x), 2, 1, 2*Derivative(u(x), x))
    >>> p.derivative(0).to_expr()
    2*Derivative(u(x), x)*Derivative(u(x), (x, 2)) - 4*Derivative(u(x), x)
    """

    def __init__(self, ring: DifferentialRing, terms: Terms) -> None:
        self.ring: DifferentialRing = ring
        self.terms: Terms = terms
        self._leader: Optional[Jet] = None
        self._derivatives: dict[int, DifferentialPolynomial] = {}

    @classmethod
    def from_expr(cls, ring: DifferentialRing, equation: Equation) -> DifferentialPolynomial:
        """The differential polynomial of an expression or an equality."""
        return cls(ring, ring.integral(ring.terms(equation))).primitive()

    def to_expr(self) -> Expr:
        """The SymPy expression."""
        parts: list[Expr] = []
        for monomial in sorted(self.terms, key=self._monomial_key, reverse=True):
            term = as_expr(self.ring.integers.to_sympy(self.terms[monomial]))
            for jet, e in monomial:
                term = term*self.ring.to_expr(jet)**e
            parts.append(term)
        return Add(*parts)

    def __repr__(self) -> str:
        return "DifferentialPolynomial(%s)" % (self.to_expr(),)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DifferentialPolynomial):
            return NotImplemented
        return self.terms == other.terms

    def __hash__(self) -> int:
        return hash(frozenset(self.terms))

    def __bool__(self) -> bool:
        return bool(self.terms)

    # ------------------------------------------------------------------
    # arithmetic

    def constant(self, c: DomainElement) -> DifferentialPolynomial:
        """The element ``c`` of the ring of coefficients."""
        return DifferentialPolynomial(self.ring, {(): c} if c else {})

    def __neg__(self) -> DifferentialPolynomial:
        return DifferentialPolynomial(self.ring, {m: -c for m, c in self.terms.items()})

    def __add__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        terms = dict(self.terms)
        for m, c in other.terms.items():
            value = terms[m] + c if m in terms else c
            if value:
                terms[m] = value
            else:
                del terms[m]
        return DifferentialPolynomial(self.ring, terms)

    def __sub__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        return self + (-other)

    def __mul__(self, other: DifferentialPolynomial) -> DifferentialPolynomial:
        terms: Terms = {}
        for m, c in self.terms.items():
            for k, d in other.terms.items():
                product = _times(m, k)
                value = terms[product] + c*d if product in terms else c*d
                if value:
                    terms[product] = value
                else:
                    del terms[product]
        return DifferentialPolynomial(self.ring, terms)

    def __pow__(self, n: int) -> DifferentialPolynomial:
        result = self.constant(self.ring.integers.one)
        for _ in range(n):
            result = result*self
        return result

    def primitive(self) -> DifferentialPolynomial:
        """The polynomial divided by the greatest common divisor of its
        coefficients, with a positive leading coefficient."""
        if not self.terms:
            return self
        one = self.ring.integers.one
        values = list(self.terms.values())
        divisor = values[0]
        for c in values[1:]:
            if divisor == one:
                break
            divisor = divisor.gcd(c)
        first = self.terms[max(self.terms, key=self._monomial_key)]
        if (first.LC < 0) != (divisor.LC < 0):
            divisor = -divisor
        if divisor == one:
            return self
        return DifferentialPolynomial(self.ring, {m: c.exquo(divisor) for m, c in self.terms.items()})

    def _monomial_key(self, m: JetMonomial) -> list[tuple[tuple[int, ...], int]]:
        return sorted(((self.ring.rank_key(jet), e) for jet, e in m), reverse=True)

    # ------------------------------------------------------------------
    # structure

    @property
    def is_ground(self) -> bool:
        """Whether the polynomial lies in the field of coefficients."""
        return all(not m for m in self.terms)

    def jets(self) -> set[Jet]:
        """The derivatives which occur."""
        return {jet for m in self.terms for jet, _ in m}

    def leader(self) -> Jet:
        """The highest derivative."""
        if self._leader is None:
            jets = self.jets()
            if not jets:
                raise ValueError("an element of the field of coefficients has no leader")
            self._leader = max(jets, key=self.ring.rank_key)
        return self._leader

    def leader_expr(self) -> Expr:
        """The leader as an expression."""
        return self.ring.to_expr(self.leader())

    def degree(self, jet: Optional[Jet] = None) -> int:
        """The degree in a derivative (the leader by default)."""
        v = self.leader() if jet is None else jet
        return max((e for m in self.terms for w, e in m if w == v), default=0)

    def rank(self) -> Rank:
        """The rank: the leader and the degree in it."""
        return (self.ring.rank_key(self.leader()), self.degree())

    def coefficients(self, jet: Jet) -> dict[int, DifferentialPolynomial]:
        """The coefficients as a polynomial in one derivative, by degree."""
        parts: dict[int, Terms] = {}
        for m, c in self.terms.items():
            e = 0
            rest = m
            for k, (w, d) in enumerate(m):
                if w == jet:
                    e = d
                    rest = m[:k] + m[k + 1:]
                    break
            parts.setdefault(e, {})[rest] = c
        return {e: DifferentialPolynomial(self.ring, terms) for e, terms in parts.items()}

    def initial(self) -> DifferentialPolynomial:
        """The coefficient of the highest power of the leader."""
        return self.coefficients(self.leader())[self.degree()]

    def separant(self) -> DifferentialPolynomial:
        """The partial derivative with respect to the leader."""
        v = self.leader()
        result = self.constant(self.ring.integers.zero)
        for e, c in self.coefficients(v).items():
            if e:
                result = result + c.scaled(self.ring.integers.convert(e))*self.power(v, e - 1)
        return result

    def scaled(self, c: DomainElement) -> DifferentialPolynomial:
        """The product with an element of the field."""
        if not c:
            return self.constant(c)
        return DifferentialPolynomial(self.ring, {m: d*c for m, d in self.terms.items()})

    def power(self, jet: Jet, e: int) -> DifferentialPolynomial:
        """The power of a derivative."""
        one = self.ring.integers.one
        return DifferentialPolynomial(self.ring, {((jet, e),) if e else (): one})

    def derivative(self, i: int) -> DifferentialPolynomial:
        """The derivative with respect to the variable of index ``i``."""
        cached = self._derivatives.get(i)
        if cached is not None:
            return cached
        ring = self.ring
        terms: Terms = {}

        def add(m: JetMonomial, c: DomainElement) -> None:
            value = terms[m] + c if m in terms else c
            if value:
                terms[m] = value
            else:
                del terms[m]

        for m, c in self.terms.items():
            d = ring.integer_derivative(c, i)
            if d:
                add(m, d)
            for k, (jet, e) in enumerate(m):
                rest = m[:k] + m[k + 1:]
                factor: JetMonomial = ((jet, e - 1),) if e > 1 else ()
                product = _times(_times(rest, factor), ((ring.differentiate(jet, i), 1),))
                add(product, c*e)
        result = DifferentialPolynomial(ring, terms)
        self._derivatives[i] = result
        return result

    def prolonged(self, theta: Sequence[int]) -> DifferentialPolynomial:
        """The derivative of the given orders."""
        result = self
        for i, k in enumerate(theta):
            for _ in range(k):
                result = result.derivative(i)
        return result

    # ------------------------------------------------------------------
    # reduction

    def pseudo_remainder(self, divisor: DifferentialPolynomial, jet: Jet) -> DifferentialPolynomial:
        """The pseudo-remainder of the division by ``divisor`` as
        polynomials in the derivative ``jet``: the product of ``self`` with
        a power of the leading coefficient of the divisor, minus a multiple
        of the divisor, of lower degree than the divisor."""
        d = divisor.degree(jet)
        parts = divisor.coefficients(jet)
        lead = parts[d]
        remainder = self
        while remainder:
            e = remainder.degree(jet)
            if e < d:
                break
            top = remainder.coefficients(jet)[e]
            remainder = lead*remainder - top*self.power(jet, e - d)*divisor
        return remainder

    def is_partially_reduced(self, other: DifferentialPolynomial) -> bool:
        """Whether no proper derivative of the leader of ``other``
        occurs."""
        v = other.leader()
        return not any(w != v and self.ring.is_derivative_of(w, v) for w in self.jets())

    def is_reduced(self, other: DifferentialPolynomial) -> bool:
        """Whether the polynomial is partially reduced with respect to
        ``other`` and of lower degree in its leader."""
        return self.is_partially_reduced(other) and self.degree(other.leader()) < other.degree()

    def partial_remainder(self, chain: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
        """Ritt's partial remainder: a product of ``self`` with separants
        of the chain, reduced by derivatives of its elements until no
        proper derivative of a leader is left."""
        ring = self.ring
        leaders = [a.leader() for a in chain]
        remainder = self
        while remainder:
            found: Optional[tuple[Jet, int]] = None
            for w in remainder.jets():
                for k, v in enumerate(leaders):
                    if w != v and ring.is_derivative_of(w, v):
                        if found is None or ring.rank_key(w) > ring.rank_key(found[0]):
                            found = (w, k)
                        break
            if found is None:
                break
            w, k = found
            theta = tuple(j - l for j, l in zip(w[1], leaders[k][1]))
            remainder = remainder.pseudo_remainder(chain[k].prolonged(theta), w).primitive()
        return remainder

    def remainder(self, chain: Sequence[DifferentialPolynomial]) -> DifferentialPolynomial:
        """Ritt's full remainder by an autoreduced set: reduced with
        respect to every element."""
        remainder = self.partial_remainder(chain)
        for a in sorted(chain, key=lambda a: self.ring.rank_key(a.leader()), reverse=True):
            if not remainder:
                break
            remainder = remainder.pseudo_remainder(a, a.leader()).primitive()
        return remainder
