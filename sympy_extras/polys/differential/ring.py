"""Differential polynomial rings: derivatives, rankings and the conversion
of expressions of the core.

A *differential polynomial ring* `K\\{u_1, \\ldots, u_m\\}` with the
commuting derivations `\\partial_1, \\ldots, \\partial_n` is the polynomial
ring over `K` in the infinitely many *derivatives*
`\\partial^J u_a = \\partial_1^{j_1} \\cdots \\partial_n^{j_n} u_a`. Here
`K = \\mathbb{Q}(p_1, \\ldots)(x_1, \\ldots, x_n)` is the field of the
rational functions of the independent variables (and of constant
parameters), the derivations are `\\partial / \\partial x_i`, and a
derivative is stored as a pair ``(a, J)`` of the index of the function and
the multi-index of the orders (a :data:`Jet`).

A *ranking* is a total order of the derivatives with

.. math:: v < \\partial_i v, \\qquad v < w \\implies \\partial_i v < \\partial_i w,

and it is a well-order [Kolchin]_. The rankings of this module are given by
blocks of functions, as in the systems which implement differential
elimination [Boulier]_: every derivative of a function of an earlier
block is higher than every derivative of a function of a later block (an
*elimination ranking*, which eliminates the functions of the earlier
blocks), and inside a block the ranking is *orderly*: the derivatives are
compared by their total order, then by the function (the earlier one is
higher), then lexicographically by the orders in `x_1, x_2, \\ldots` (the
first variable weighs most). One block with all the functions is the
default; it is an orderly ranking.

The *leader* of a differential polynomial is its highest derivative.

The module works on objects of the core and imports no SymPy: the field
`K` is :class:`~sympy_extras.polys.domains.sparsefractionfield.SparseFractionField`
over ``QQ[p, x]`` and the coefficients of the differential polynomials
are elements of ``ZZ[p, x]``
(:class:`~sympy_extras.polys.domains.sparsepolynomialring.IntegerPolynomialRing`),
with SymPy's conventions for the gcd and the canonical form of the
fractions. :mod:`sympy_extras.polys.differential_sympy` is the face on
SymPy objects.

References
==========

.. [Kolchin] E. R. Kolchin, Differential Algebra and Algebraic Groups,
   Academic Press 1973, chapter I.
.. [Boulier] F. Boulier, D. Lazard, F. Ollivier, M. Petitot, Computing
   representations for radicals of finitely generated differential ideals,
   Appl. Algebra Engrg. Comm. Comput. 20 (2009), 73-121.
"""
from __future__ import annotations

from fractions import Fraction
from heapq import heapify, heappop, heappush
from math import lcm
from operator import add as add_exponents
from typing import Final, Mapping, Optional, Sequence, Union

from sympy_extras.polys.rings import Monomial
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Equality
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing, RationalPolynomialRing
from sympy_extras.polys.rings import PolyElement, lex

__all__ = ['DifferentialRing', 'Jet', 'JetMonomial', 'Coefficient', 'IntegerCoefficient', 'FieldTerms', 'Terms',
           'Equation', 'unit_jet', 'monomial_product', 'integer_product', 'integer_quotient']

#: a derivative: the index of the function and the orders in the variables
Jet = tuple[int, Monomial]
#: a product of powers of derivatives, sorted by derivative
JetMonomial = tuple[tuple[Jet, int], ...]
#: an element of the field of coefficients ``QQ(p, x)``
Coefficient = SparseFracElement[Fraction]
#: an element of the ring ``ZZ[p, x]`` of the coefficients of the
#: differential polynomials
IntegerCoefficient = PolyElement[int]
#: a differential polynomial with coefficients in the field
FieldTerms = dict[JetMonomial, Coefficient]
#: a differential polynomial with coefficients in ``ZZ[p, x]``
Terms = dict[JetMonomial, IntegerCoefficient]
#: an equation: an expression which is to vanish, or an equality
Equation = Union[Expr, Equality]


def unit_jet(n: int) -> Jet:
    """The pseudo derivative standing for the constant `1` in a linear
    form (the lowest of the ranking; its derivatives are zero)."""
    return (-1, (0,)*n)


def monomial_product(m: JetMonomial, k: JetMonomial) -> JetMonomial:
    """The product of two monomials in the derivatives."""
    if not m:
        return k
    if not k:
        return m
    powers = dict(m)
    for jet, e in k:
        powers[jet] = powers.get(jet, 0) + e
    return tuple(sorted(powers.items()))


def integer_product(a: IntegerCoefficient, b: IntegerCoefficient) -> IntegerCoefficient:
    """The product of two elements of ``ZZ[p, x]``: the product of
    :class:`~sympy_extras.polys.rings.PolyElement` (the same terms) on
    Python integers directly, the zeros removed at the end, without the
    calls to the domain, which take most of the time of the large normal
    forms."""
    result: dict[Monomial, int] = {}
    get = result.get
    second = list(b.terms.items())
    n = a.ring.ngens
    for m1, c1 in a.terms.items():
        if n == 2:
            i, j = m1
            for (k, l), c2 in second:
                m: Monomial = (i + k, j + l)
                result[m] = get(m, 0) + c1*c2
        elif n == 3:
            i, j, h = m1
            for (k, l, o), c2 in second:
                m = (i + k, j + l, h + o)
                result[m] = get(m, 0) + c1*c2
        else:
            for m2, c2 in second:
                m = tuple(map(add_exponents, m1, m2))
                result[m] = get(m, 0) + c1*c2
    return PolyElement({m: c for m, c in result.items() if c}, a.ring)


def integer_quotient(a: IntegerCoefficient, b: IntegerCoefficient) -> IntegerCoefficient:
    """The exact quotient of two elements of ``ZZ[p, x]`` (``ValueError``
    when ``b`` does not divide ``a``): term by term when ``b`` is a term,
    otherwise the division of :meth:`~sympy_extras.polys.rings.PolyElement.exquo`
    (the same quotient) with the leading terms of the remainder kept in a
    heap instead of searched for at every step."""
    if not b.terms:
        raise ZeroDivisionError('division by zero')
    if len(b.terms) == 1:
        ((n, d),) = b.terms.items()
        result: dict[Monomial, int] = {}
        for m, c in a.terms.items():
            q, r = divmod(c, d)
            if r or any(i < j for i, j in zip(m, n)):
                raise ValueError('%s does not divide %s' % (b, a))
            result[tuple(i - j for i, j in zip(m, n))] = q
        return PolyElement(result, a.ring)
    if a.ring.order != lex:
        return a.exquo(b)
    leading = max(b.terms)
    lc = b.terms[leading]
    rest = dict(a.terms)
    heap = [tuple(-e for e in m) for m in rest]
    heapify(heap)
    quotient: dict[Monomial, int] = {}
    while rest:
        m = tuple(-e for e in heappop(heap))
        value = rest.get(m)
        if value is None:
            continue
        q, r = divmod(value, lc)
        if r or any(i < j for i, j in zip(m, leading)):
            raise ValueError('%s does not divide %s' % (b, a))
        shift = tuple(i - j for i, j in zip(m, leading))
        quotient[shift] = q
        for k, e in b.terms.items():
            product = tuple(map(add_exponents, shift, k))
            s = rest.get(product)
            if s is None:
                rest[product] = -q*e
                heappush(heap, tuple(-i for i in product))
            else:
                s -= q*e
                if s:
                    rest[product] = s
                else:
                    del rest[product]
    return PolyElement(quotient, a.ring)


class _NotPolynomial(Exception):
    """The expression is not a polynomial in the derivatives (a derivative
    in a denominator)."""


class _NotRational(Exception):
    """The expression is not a polynomial with rational coefficients."""


class DifferentialRing:
    """A differential polynomial ring with a ranking.

    Parameters
    ==========

    functions : sequence of applied undefined functions
        The differential indeterminates, such as ``u(x, y)``. A function
        of some of the variables only is a function which is constant in
        the others (see :meth:`constancy`).
    variables : sequence of Symbol, optional
        The independent variables, in the order which the ranking uses;
        the arguments of the functions in the order of their appearance
        by default.
    ranking : sequence of sequences of functions, optional
        The blocks of the ranking, the highest first. One block with all
        the functions in the given order by default.
    parameters : sequence of Symbol
        The constants which the coefficients may contain.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.polys.differential import DifferentialRing
    >>> x, y = symbols('x y')
    >>> u, v = Function('u'), Function('v')
    >>> R = DifferentialRing([u(x, y), v(x, y)])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(v(x, y), (y, 2))
    >>> R = DifferentialRing([u(x, y), v(x, y)], ranking=[[u(x, y)], [v(x, y)]])
    >>> R.leader(u(x, y).diff(x) + v(x, y).diff(y, 2))
    Derivative(u(x, y), x)
    """

    def __init__(self, functions: Sequence[Basic], variables: Optional[Sequence[Basic]] = None,
                 ranking: Optional[Sequence[Sequence[Basic]]] = None,
                 parameters: Sequence[Symbol] = ()) -> None:
        if not functions:
            raise ValueError("at least one function is expected")
        found: list[Symbol] = []
        heads: set[str] = set()
        checked: list[AppliedUndef] = []
        for f in functions:
            if not isinstance(f, AppliedUndef):
                raise TypeError("an applied undefined function such as u(x, y) is expected, got %s" % (f,))
            if f.name in heads:
                raise ValueError("the function %s is given twice" % (f.name,))
            heads.add(f.name)
            for arg in f.args:
                if not isinstance(arg, Symbol):
                    raise ValueError("the arguments of %s must be symbols" % (f,))
                if arg not in found:
                    found.append(arg)
            if len(set(f.args)) != len(f.args):
                raise ValueError("the arguments of %s must be distinct" % (f,))
            checked.append(f)
        chosen: list[Symbol] = []
        if variables is None:
            chosen = list(found)
        else:
            for s in variables:
                if not isinstance(s, Symbol):
                    raise TypeError("the variables must be symbols, got %s" % (s,))
                chosen.append(s)
            missing = [s for s in found if s not in chosen]
            if missing:
                raise ValueError("%s is not among the variables" % (missing[0],))
        if len(set(chosen)) != len(chosen):
            raise ValueError("the variables must be distinct")
        self.functions: Final[tuple[AppliedUndef, ...]] = tuple(checked)
        self.variables: Final[tuple[Symbol, ...]] = tuple(chosen)
        self.parameters: Final[tuple[Symbol, ...]] = tuple(p for p in parameters if p not in chosen)
        #: the symbols of the coefficients: the parameters, then the variables
        self.symbols: Final[tuple[Symbol, ...]] = self.parameters + self.variables
        names = [s.name for s in self.symbols]
        #: the field of coefficients ``QQ(p, x)``
        self.domain: Final[SparseFractionField[Fraction]] = SparseFractionField(RationalPolynomialRing(names))
        #: the polynomials with integer coefficients in the parameters and
        #: the variables: the coefficients of the differential polynomials
        self.integers: Final[IntegerPolynomialRing] = IntegerPolynomialRing(names)
        self._rationals: Final[RationalPolynomialRing] = RationalPolynomialRing(names)
        self._index: Final[dict[Symbol, int]] = {s: i for i, s in enumerate(self.symbols)}
        self._generators: Final[list[Coefficient]] = self.domain.gens
        self.blocks: Final[tuple[tuple[int, ...], ...]] = self._blocks(ranking)
        level: dict[int, tuple[int, int]] = {}
        for b, block in enumerate(self.blocks):
            for position, a in enumerate(block):
                level[a] = (len(self.blocks) - b, len(block) - position)
        self._level: Final[dict[int, tuple[int, int]]] = level
        self._keys: Final[dict[Jet, tuple[int, ...]]] = {}

    def _blocks(self, ranking: Optional[Sequence[Sequence[Basic]]]) -> tuple[tuple[int, ...], ...]:
        if ranking is None:
            return (tuple(range(len(self.functions))),)
        blocks: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for block in ranking:
            indices: list[int] = []
            for f in block:
                if not isinstance(f, AppliedUndef) or f not in self.functions:
                    raise ValueError("%s of the ranking is not a function of the ring" % (f,))
                a = self.functions.index(f)
                if a in seen:
                    raise ValueError("%s occurs twice in the ranking" % (f,))
                seen.add(a)
                indices.append(a)
            if indices:
                blocks.append(tuple(indices))
        if len(seen) != len(self.functions):
            absent = [f for a, f in enumerate(self.functions) if a not in seen]
            raise ValueError("%s is not in the ranking" % (absent[0],))
        return tuple(blocks)

    def __repr__(self) -> str:
        return "DifferentialRing(%s, %s)" % (list(self.functions), list(self.variables))

    # ------------------------------------------------------------------
    # derivatives and the ranking

    @property
    def n(self) -> int:
        """The number of derivations."""
        return len(self.variables)

    @property
    def is_orderly(self) -> bool:
        """Whether a derivative of higher order is always higher."""
        return len(self.blocks) == 1

    def rank_key(self, jet: Jet) -> tuple[int, ...]:
        """The key which sorts the derivatives by the ranking."""
        key = self._keys.get(jet)
        if key is None:
            a, J = jet
            if a < 0:
                key = (-1, 0, 0) + J
            else:
                block, position = self._level[a]
                key = (block, sum(J), position) + J
            self._keys[jet] = key
        return key

    def highest(self, jets: Sequence[Jet]) -> Jet:
        """The highest of the derivatives."""
        return max(jets, key=self.rank_key)

    def differentiate(self, jet: Jet, i: int) -> Jet:
        """The derivative of a derivative with respect to the variable of
        index ``i``."""
        a, J = jet
        return (a, J[:i] + (J[i] + 1,) + J[i + 1:])

    def is_derivative_of(self, jet: Jet, other: Jet) -> bool:
        """Whether ``jet`` is a derivative of ``other`` (or equal to it)."""
        return jet[0] == other[0] and all(j >= k for j, k in zip(jet[1], other[1]))

    def constancy(self) -> list[Jet]:
        """The derivatives which vanish because a function does not depend
        on a variable: `\\partial u / \\partial y` for ``u(x)`` in a ring
        with the variables `x, y`."""
        result: list[Jet] = []
        for a, f in enumerate(self.functions):
            for i, s in enumerate(self.variables):
                if s not in f.args:
                    result.append((a, (0,)*i + (1,) + (0,)*(self.n - i - 1)))
        return result

    # ------------------------------------------------------------------
    # coefficients

    def field_element(self, c: IntegerCoefficient) -> Coefficient:
        """An element of :attr:`integers` as an element of the field."""
        return self.domain.from_ring(self._rationals.ring.from_dict({m: Fraction(e) for m, e in c.terms.items()}))

    def integer_expr(self, c: IntegerCoefficient, at: Optional[Mapping[Symbol, Expr]] = None) -> Expr:
        """The expression of an element of :attr:`integers`, with the
        values ``at`` of some symbols substituted."""
        return self._polynomial_expr({m: Fraction(e) for m, e in c.terms.items()}, at)

    def coefficient_expr(self, c: Coefficient) -> Expr:
        """The expression of an element of the field."""
        numerator = self._polynomial_expr(dict(c.numer.terms))
        if self._rationals.is_one(c.denom):
            return numerator
        return mul(numerator, pow(self._polynomial_expr(dict(c.denom.terms)), Integer(-1)))

    def _polynomial_expr(self, terms: dict[Monomial, Fraction], at: Optional[Mapping[Symbol, Expr]] = None) -> Expr:
        parts: list[Expr] = []
        for m, c in terms.items():
            factors: list[Expr] = [number_from_fraction(c)]
            for s, e in zip(self.symbols, m):
                if e:
                    value: Expr = s if at is None else at.get(s, s)
                    factors.append(pow(value, Integer(e)) if e > 1 else value)
            parts.append(mul(*factors))
        return add(*parts) if parts else Integer(0)

    def coefficient(self, expr: Expr) -> Coefficient:
        """An expression as an element of the field of coefficients."""
        try:
            terms = self._convert(expr, {})
        except (_NotPolynomial, _NotRational):
            raise ValueError("%s is not a rational function of the variables %s and the parameters %s"
                             % (expr, list(self.variables), list(self.parameters))) from None
        return terms.get((), self.domain.zero)

    def coefficient_derivative(self, c: Coefficient, i: int) -> Coefficient:
        """The derivative of a coefficient with respect to the variable of
        index ``i``."""
        return self.domain.diff(c, len(self.parameters) + i)

    def integer_derivative(self, c: IntegerCoefficient, i: int) -> IntegerCoefficient:
        """The derivative of an element of :attr:`integers`."""
        return c.diff(len(self.parameters) + i)

    def integral(self, terms: FieldTerms) -> Terms:
        """The terms multiplied by an element of the field which makes the
        coefficients elements of :attr:`integers`."""
        if not terms:
            return {}
        K, Q = self.domain, self._rationals
        values = list(terms.values())
        multiple = values[0].denom
        for c in values[1:]:
            multiple = Q.lcm(multiple, c.denom)
        factor = K.from_ring(multiple)
        cleared: dict[JetMonomial, PolyElement[Fraction]] = {}
        denominator = 1
        for m, c in terms.items():
            scaled = K.mul(c, factor)
            q = scaled.numer.mul_ground(1/scaled.denom.LC)
            cleared[m] = q
            for e in q.terms.values():
                denominator = lcm(denominator, e.denominator)
        result: Terms = {}
        for m, q in cleared.items():
            result[m] = self.integers.ring.from_dict({k: int(e*denominator) for k, e in q.terms.items()})
        return result

    # ------------------------------------------------------------------
    # conversion

    def jet(self, expr: Basic) -> Optional[Jet]:
        """The derivative which an applied function or a ``Derivative`` of
        one stands for, ``None`` for another expression."""
        if isinstance(expr, AppliedUndef):
            if expr in self.functions:
                return (self.functions.index(expr), (0,)*self.n)
            return None
        if isinstance(expr, Derivative) and expr.expr in self.functions:
            orders = [0]*self.n
            for s, count in expr.variable_count:
                if s not in self.variables:
                    return None
                orders[self.variables.index(s)] += count
            return (self.functions.index(expr.expr), tuple(orders))
        return None

    def to_expr(self, jet: Jet) -> Expr:
        """The expression of a derivative."""
        a, J = jet
        if a < 0:
            return Integer(1)
        counts = [(s, j) for s, j in zip(self.variables, J) if j]
        if not counts:
            return self.functions[a]
        return Derivative(self.functions[a], *counts)

    def _convert(self, e: Expr, jets: Mapping[Basic, Jet]) -> FieldTerms:
        """The terms of an expression, polynomial in the derivatives
        ``jets`` (expressions standing for them), with coefficients
        rational in the variables and the parameters."""
        K = self.domain
        jet = jets.get(e)
        if jet is not None:
            return {((jet, 1),): K.one}
        if isinstance(e, Rational):
            if e.p == 0:
                return {}
            return {(): K.from_ring(self._rationals.ring.ground(Fraction(e.p, e.q)))}
        if isinstance(e, Symbol):
            i = self._index.get(e)
            if i is None:
                raise _NotRational()
            return {(): self._generators[i]}
        if isinstance(e, Add):
            result: FieldTerms = {}
            for arg in e.args:
                for m, c in self._convert(arg, jets).items():
                    value = K.add(result[m], c) if m in result else c
                    if K.is_zero(value):
                        del result[m]
                    else:
                        result[m] = value
            return result
        if isinstance(e, Mul):
            product: FieldTerms = {(): K.one}
            for arg in e.args:
                product = self._multiply(product, self._convert(arg, jets))
            return product
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            base = self._convert(e.base, jets)
            n = e.exp.p
            if n < 0:
                if any(m for m in base):
                    raise _NotPolynomial()
                c = base.get((), K.zero)
                if K.is_zero(c):
                    raise ZeroDivisionError("division by zero in %s" % (e,))
                return {(): K.pow(c, n)}
            power: FieldTerms = {(): K.one}
            for _ in range(n):
                power = self._multiply(power, base)
            return power
        raise _NotRational()

    def _multiply(self, a: FieldTerms, b: FieldTerms) -> FieldTerms:
        K = self.domain
        result: FieldTerms = {}
        for m, c in a.items():
            for k, d in b.items():
                product = monomial_product(m, k)
                value = K.mul(c, d)
                if product in result:
                    value = K.add(result[product], value)
                if K.is_zero(value):
                    result.pop(product, None)
                else:
                    result[product] = value
        return result

    def terms_of(self, expr: Expr, jets: Mapping[Basic, Jet]) -> FieldTerms:
        """The terms of an expression in which the keys of ``jets`` (symbols,
        say) stand for the derivatives, with coefficients rational in the
        variables and the parameters."""
        try:
            return self._convert(expr, jets)
        except _NotPolynomial:
            raise ValueError("%s is not a polynomial in the derivatives" % (expr,)) from None
        except _NotRational:
            raise ValueError("%s is not a polynomial in the derivatives with rational coefficients"
                             % (expr,)) from None

    def terms(self, equation: Equation) -> FieldTerms:
        """The monomials and coefficients of a differential polynomial
        given as an expression.

        Raises ``ValueError`` when the expression is not a polynomial in
        the derivatives of the functions with coefficients rational in the
        variables and the parameters.
        """
        expr = equation.lhs - equation.rhs if isinstance(equation, Equality) else equation
        expr = expr.doit()
        jets: dict[Basic, Jet] = {}
        atoms = sorted(expr.atoms(Derivative) | expr.atoms(AppliedUndef), key=lambda e: e.sort_key())
        for atom in atoms:
            jet = self.jet(atom)
            if jet is None:
                if isinstance(atom, AppliedUndef) or any(f in self.functions for f in atom.atoms(AppliedUndef)):
                    if isinstance(atom, Derivative) and atom.expr in self.functions:
                        raise ValueError("%s is not a derivative in the variables of the ring" % (atom,))
                    if isinstance(atom, AppliedUndef):
                        raise ValueError("%s is not a function of the ring" % (atom,))
                raise ValueError("%s is not a derivative of a function of the ring" % (atom,))
            jets[atom] = jet
        extra = expr.free_symbols - set(self.variables) - set(self.parameters)
        if extra:
            raise ValueError("%s is neither a variable nor a parameter of the ring"
                             % (sorted(extra, key=lambda s: s.name)[0],))
        if not atoms:
            c = self.coefficient(expr)
            return {(): c} if not self.domain.is_zero(c) else {}
        try:
            result = self._convert(expr, jets)
        except _NotPolynomial:
            raise ValueError("%s is not a polynomial in the derivatives" % (expr,)) from None
        except _NotRational:
            raise ValueError("%s is not a polynomial in the derivatives with rational coefficients"
                             % (expr,)) from None
        # SymPy's order of the terms of a Poly in the atoms sorted by
        # sort_key: increasing exponent vectors
        ordered = [jets[atom] for atom in atoms]

        def exponents(m: JetMonomial) -> tuple[int, ...]:
            powers = dict(m)
            seen: set[Jet] = set()
            vector: list[int] = []
            for jet in ordered:
                vector.append(powers.get(jet, 0) if jet not in seen else 0)
                seen.add(jet)
            return tuple(vector)

        return {m: result[m] for m in sorted(result, key=exponents)}

    def from_terms(self, terms: FieldTerms) -> Expr:
        """The expression of a differential polynomial."""
        parts: list[Expr] = []
        for monomial, c in terms.items():
            term = self.coefficient_expr(c)
            for jet, e in monomial:
                term = term*self.to_expr(jet)**e
            parts.append(term)
        return add(*parts) if parts else Integer(0)

    def leader(self, equation: Equation) -> Expr:
        """The highest derivative of a differential polynomial.

        >>> from sympy_extras.core import Function, symbols
        >>> from sympy_extras.polys.differential import DifferentialRing
        >>> x, y = symbols('x y')
        >>> u = Function('u')(x, y)
        >>> DifferentialRing([u]).leader(u.diff(x, y) + u.diff(y, 2)*u)
        Derivative(u(x, y), x, y)
        """
        jets = [jet for monomial in self.terms(equation) for jet, _ in monomial]
        if not jets:
            raise ValueError("the differential polynomial has no derivative")
        return self.to_expr(self.highest(jets))
