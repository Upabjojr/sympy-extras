"""A canonical form and a zero test for elementary expressions: the
Risch–Rosenlicht structure theorem.

SymPy decides whether an expression vanishes by rewriting heuristics
(``simplify``, ``cancel``) and by sampling. This module writes an
elementary expression as a rational function of generators which are
*proven* algebraically independent, so that the expression vanishes
identically exactly when its numerator does.

The tower
=========

The field is `K(g_1, ..., g_n)`, `K` an algebraic number field which
grows as algebraic numbers are met, and each generator is one of

* a *variable*, a free symbol of the expression, with the derivation
  `\\partial/\\partial x`;
* an *exponential* `\\theta = \\exp(a)` with `a` in the field built so far,
  `\\theta' = a' \\theta`;
* a *logarithm* `L = \\log(v)`, `L' = v'/v`;
* a *root* `\\rho = \\exp(z/q)`, where `z` is the argument of an exponential
  generator (`\\rho^q = \\exp(z)`) or a logarithm generator `\\log(v)`
  (`\\rho = v^{1/q}`, the principal root, which SymPy defines as
  `\\exp(\\log(v)/q)`); it is the one kind of algebraic generator, kept
  with its relation `\\rho^q = e`.

Trigonometric and hyperbolic functions are written with exponentials,
their inverses with logarithms (the principal values agree, which the
tests check at complex points), `a^b` is `\\exp(b \\log a)`, and `\\pi` is
`-i \\log(-1)`: the constants `\\pi`, `e`, `\\log 2`, ... are generators
like the others.

The structure theorem
=====================

Every generator comes with a pair `(z, e)`, `e = \\exp(z)`: `(a, \\theta)`
for an exponential, `(L, v)` for a logarithm. Risch's structure theorem
([Risch1979]_, in Rosenlicht's form [Rosenlicht1976]_, see also
[Bronstein2005]_ chapter 9) says that a new `\\exp(a)` is algebraic over
the field exactly when

.. math:: a = c + \\sum_i r_i z_i, \\qquad r_i \\in \\mathbb{Q},

with `c` a constant, and a new `\\log(v)` exactly when `v'/v = \\sum_i r_i
z_i'`. Both are linear systems over `\\mathbb{Q}` (the coefficients of
the monomials, split over a basis of `K`), solved here exactly.

* When `\\exp(a)` is dependent it *is* `\\exp(c) \\prod_i e_i^{r_i}`, with no
  question of branches, the exponential being a homomorphism; a
  fractional `r_i = p/q` brings the root `\\exp(z_i/q)`.
* When `\\log(v)` is dependent, `\\log(v) - \\sum_i r_i z_i` is only
  *locally* constant: it jumps by multiples of `2 \\pi i r_i` across the
  branch cuts. It is pinned exactly when the assumptions prove the
  arguments positive (then every logarithm involved is real and the
  constant is the real logarithm of a positive constant of the field),
  or `v` negative (`\\log(v) = i\\pi + \\log(-v)`). Otherwise `\\log(v)` is
  adjoined as a generator which is *not* independent, and the tower is
  marked uncertified.
* A constant has no derivation to test it with. `\\exp(c)` for a constant
  `c` is dependent when `c = \\sum_j s_j w_j` exactly, over the pairs of the
  constant generators; the logarithm of a positive rational is the
  combination of the logarithms of its primes. That the constants left
  as generators are algebraically independent is Schanuel's conjecture
  ([Richardson1997]_): the `w_j` are linearly independent over
  `\\mathbb{Q}` by construction, which is its hypothesis.
* The roots are certified by Kummer theory over `\\bar K(g)`: the
  extension by `\\rho_i^{q_i} = e_i` has the full degree `\\prod q_i`
  exactly when the subgroup which the `e_i^{N/q_i}` generate modulo
  `N`-th powers has that order, which the exponent vectors of their
  irreducible factors and a Smith normal form decide. A root of a
  radicand which involves another root is left uncertified.

The decision
============

``is_zero`` answers ``True`` when the numerator of the canonical form is
zero: every identity used on the way is exact on the region of the
assumptions, so this is a proof, whether or not the tower is certified.
It answers ``False`` when the numerator is not zero and the tower is
certified (a nonzero rational function of independent generators does not
vanish on any open set), or when a sample point of the region gives a
value clearly away from zero; ``None`` otherwise.

References
==========

.. [Risch1979] R. H. Risch, Algebraic properties of the elementary
   functions of analysis, American Journal of Mathematics 101 (1979),
   743-759.
.. [Rosenlicht1976] M. Rosenlicht, On Liouville's theory of elementary
   functions, Pacific Journal of Mathematics 65 (1976), 485-492.
.. [Bronstein2005] M. Bronstein, Symbolic Integration I: Transcendental
   Functions, second edition, Springer, 2005, chapter 9.
.. [Richardson1997] D. Richardson, How to recognize zero, Journal of
   Symbolic Computation 24 (1997), 627-645.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import E as EulerE
from sympy.core.numbers import Float, I, Integer, Rational, pi
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import (acosh, acoth, asinh, atanh, cosh, coth, csch, sech, sinh,
                                                   tanh)
from sympy.functions.elementary.trigonometric import (acos, acot, asin, atan, cos, cot, csc, sec, sin, tan)
from sympy.logic.boolalg import Boolean
from sympy.matrices.dense import Matrix
from sympy.matrices.normalforms import smith_normal_form
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains import QQ, ZZ
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.minpoly import minimal_polynomial
from sympy.polys.polyerrors import CoercionFailed, DomainError, NotAlgebraic, NotInvertible, PolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rings import PolyElement, PolyRing

from sympy_extras._timeout import attempt
from sympy_extras._typing import DomainElement, as_boolean, as_expr, free_symbols, sorted_symbols
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element as member
from sympy_extras.settings import settings

__all__ = ['ElementaryTower', 'Element', 'Generator', 'NotElementary', 'canonical_form', 'is_zero', 'equal']

VARIABLE = 'variable'
EXPONENTIAL = 'exp'
LOGARITHM = 'log'
ROOT = 'root'

_EXPONENTIAL_FORMS = (sin, cos, tan, cot, sec, csc, sinh, cosh, tanh, coth, sech, csch)
_LOGARITHMIC_FORMS = (asin, acos, atan, acot, asinh, acosh, atanh, acoth)

#: the largest degree the number field is allowed to reach
_FIELD_DEGREE_LIMIT = 32

#: the budget of one question put to ``ask`` about the sign of an argument
_ASK_SECONDS = 5.0


class NotElementary(ValueError):
    """The expression is outside the towers of this module: a function
    which is not elementary, a floating point number, ``log(0)``."""


class Element:
    """An element ``num/den`` of the field of an :class:`ElementaryTower`,
    the two polynomials in the generators of the tower at the time it
    was made (the tower lifts them to its current ring when it uses
    them). Elements of one tower are combined with ``+``, ``-``, ``*``,
    ``/`` and ``**`` with an integer."""

    __slots__ = ('tower', 'num', 'den')

    def __init__(self, tower: ElementaryTower, num: PolyElement, den: PolyElement) -> None:
        self.tower = tower
        self.num = num
        self.den = den

    def __add__(self, other: Element) -> Element:
        return self.tower._combine(self, other, 1)

    def __sub__(self, other: Element) -> Element:
        return self.tower._combine(self, other, -1)

    def __mul__(self, other: Element) -> Element:
        a, b = self.tower._lift(self), self.tower._lift(other)
        return self.tower._fraction(a.num * b.num, a.den * b.den)

    def __truediv__(self, other: Element) -> Element:
        a, b = self.tower._lift(self), self.tower._lift(other)
        if not b.num:
            raise ZeroDivisionError("division by the zero of the tower")
        return self.tower._fraction(a.num * b.den, a.den * b.num)

    def __neg__(self) -> Element:
        return Element(self.tower, -self.num, self.den)

    def __pow__(self, n: int) -> Element:
        if n < 0:
            if not self.num:
                raise ZeroDivisionError("a negative power of the zero of the tower")
            return Element(self.tower, self.den**(-n), self.num**(-n))
        return Element(self.tower, self.num**n, self.den**n)

    def times(self, r: Rational) -> Element:
        """``r`` times the element, ``r`` a rational number."""
        a = self.tower._lift(self)
        K = self.tower.domain
        return self.tower._fraction(a.num.mul_ground(K.convert(int(r.p))), a.den.mul_ground(K.convert(int(r.q))))

    def __repr__(self) -> str:
        return "Element(%s)" % (self.tower.to_expr(self),)


class Generator:
    """One generator of the tower.

    Attributes
    ==========

    kind : ``'variable'``, ``'exp'``, ``'log'`` or ``'root'``
    symbol : the generator of the polynomial ring
    expression : its meaning as a SymPy expression
    argument : ``a`` of ``exp(a)``, ``v`` of ``log(v)``
    index, source, radicand, scale : for a root ``rho = exp(z/q)`` of the
        pair ``source``: the ring generator is ``sigma = scale*rho`` with
        the polynomial relation ``sigma**index == radicand``
    eliminated : its value in the other generators, once a root of a
        larger index of the same pair has replaced it
    certified : whether it is independent of the generators before it
    """

    def __init__(self, kind: str, symbol: Symbol, expression: Expr, argument: Optional[Element] = None,
                 certified: bool = True) -> None:
        self.kind = kind
        self.symbol = symbol
        self.expression = expression
        self.argument = argument
        self.certified = certified
        self.index: int = 1
        self.source: int = -1
        self.radicand: Optional[Element] = None
        self.scale: Optional[Element] = None
        self.eliminated: Optional[Element] = None

    def __repr__(self) -> str:
        return "Generator(%s, %s)" % (self.kind, self.expression)


class _Pair:
    """``(z, e)`` with ``e = exp(z)``: ``(a, exp(a))`` for an exponential
    generator, ``(log(v), v)`` for a logarithm; ``master`` is the root
    generator of the largest index taken of ``e`` so far."""

    def __init__(self, z: Element, e: Element, generator: int, logarithmic: bool) -> None:
        self.z = z
        self.e = e
        self.generator = generator
        self.logarithmic = logarithmic
        self.master: int = -1


class ElementaryTower:
    """A tower of elementary extensions, grown by the expressions it is
    given.

    Parameters
    ==========

    assumptions : Boolean or list of Booleans, optional
        The region on which the identities are to hold: a dependent
        logarithm is resolved only where the assumptions prove its
        arguments positive (or negative).

    Examples
    ========

    >>> from sympy import symbols, exp, log, sqrt, sin, cos
    >>> from sympy_extras.simplify import ElementaryTower
    >>> x, y = symbols('x y')
    >>> tower = ElementaryTower()
    >>> u = tower.element(exp(x + y) - exp(x)*exp(y) + sin(x)**2 + cos(x)**2)
    >>> tower.to_expr(u)
    1
    >>> [g.expression for g in tower.generators]
    [x, exp(I*x), exp(x), y, exp(y)]
    >>> tower.certified
    True

    ``exp(x + y)`` was the product of two generators. ``exp(x/2)`` is the
    root of index two of ``exp(x)``:

    >>> tower.to_expr(tower.element(exp(x/2)))
    exp(x/2)
    >>> [g.kind for g in tower.generators]
    ['variable', 'exp', 'exp', 'variable', 'exp', 'root']

    ``log(x**2)`` is ``2*log(x)`` where ``x`` is positive, and a generator
    of its own, dependent on ``log(x)``, elsewhere:

    >>> tower = ElementaryTower(x > 0)
    >>> tower.to_expr(tower.element(log(x**2) - 2*log(x)))
    0
    >>> tower = ElementaryTower()
    >>> tower.to_expr(tower.element(log(x**2) - 2*log(x)))
    -2*log(x) + log(x**2)
    >>> tower.certified
    False
    """

    def __init__(self, assumptions: Assumptions = None) -> None:
        self.assumptions: Assumptions = assumptions
        self.extensions: list[Expr] = []
        self.domain: Domain = QQ
        self.generators: list[Generator] = []
        self.ring: PolyRing = PolyRing([], QQ)
        self.pairs: list[_Pair] = []
        self._variables: dict[Symbol, int] = {}
        self._cache: dict[Expr, Element] = {}
        self._derivatives: dict[tuple[int, int], Element] = {}
        self._prime_logs: dict[int, int] = {}
        self._minus_one: int = -1
        self._kummer: Optional[bool] = None
        self._signs: dict[Expr, int] = {}

    # ------------------------------------------------------------------
    # the ring and the elements

    def _rebuild(self) -> None:
        self.ring = PolyRing([g.symbol for g in self.generators], self.domain)
        self._kummer = None

    def _poly(self, p: PolyElement) -> PolyElement:
        return p if p.ring == self.ring else p.set_ring(self.ring)

    def _lift(self, a: Element) -> Element:
        if a.tower is not self:
            raise ValueError("an element of another tower")
        if a.num.ring == self.ring:
            return a
        return Element(self, self._poly(a.num), self._poly(a.den))

    def _fraction(self, num: PolyElement, den: PolyElement) -> Element:
        if not den:
            raise ZeroDivisionError("division by the zero of the tower")
        if not num:
            return Element(self, self.ring.zero, self.ring.one)
        p, q = num.cancel(den)
        lead = q.LC
        if lead != self.domain.one:
            p, q = p.quo_ground(lead), q.quo_ground(lead)
        return Element(self, p, q)

    def _combine(self, a: Element, b: Element, sign: int) -> Element:
        a, b = self._lift(a), self._lift(b)
        if a.den == b.den:
            return self._fraction(a.num + b.num if sign > 0 else a.num - b.num, a.den)
        return self._fraction(a.num * b.den + b.num * a.den if sign > 0 else a.num * b.den - b.num * a.den,
                              a.den * b.den)

    def constant(self, value: Rational) -> Element:
        """The rational number ``value`` as an element."""
        K = self.domain
        return Element(self, self.ring.ground_new(K.convert(int(value.p))), self.ring.ground_new(K.convert(int(value.q))))

    @property
    def zero(self) -> Element:
        return Element(self, self.ring.zero, self.ring.one)

    @property
    def one(self) -> Element:
        return Element(self, self.ring.one, self.ring.one)

    def _generator_element(self, position: int) -> Element:
        return Element(self, self.ring.gens[position], self.ring.one)

    def _add_generator(self, generator: Generator) -> int:
        self.generators.append(generator)
        self._rebuild()
        return len(self.generators) - 1

    # ------------------------------------------------------------------
    # normal forms

    def _reduce(self, p: PolyElement) -> PolyElement:
        """``p`` with every root generator below its index, by the
        relations ``sigma**q == radicand``."""
        p = self._poly(p)
        changed = True
        while changed:
            changed = False
            for position in range(len(self.generators) - 1, -1, -1):
                g = self.generators[position]
                if g.kind != ROOT or g.eliminated is not None or g.radicand is None:
                    continue
                if p.degree(self.ring.gens[position]) < g.index:
                    continue
                radicand = self._lift(g.radicand).num
                result = self.ring.zero
                for monomial, coefficient in p.terms():
                    e = monomial[position]
                    rest = list(monomial)
                    rest[position] = e % g.index
                    term = self.ring.from_dict({tuple(rest): coefficient})
                    if e >= g.index:
                        term = term * radicand**(e // g.index)
                    result = result + term
                p = result
                changed = True
        return p

    def _substituted(self, p: PolyElement, position: int, value: Element) -> Element:
        """``p`` with the generator at ``position`` replaced by ``value``."""
        parts: dict[int, PolyElement] = {}
        for monomial, coefficient in p.terms():
            e = monomial[position]
            rest = list(monomial)
            rest[position] = 0
            parts[e] = parts.get(e, self.ring.zero) + self.ring.from_dict({tuple(rest): coefficient})
        total = self.zero
        for e, part in parts.items():
            total = total + Element(self, part, self.ring.one) * value**e
        return total

    def normal(self, a: Element) -> Element:
        """``a`` in normal form: the eliminated roots replaced, the powers
        of the roots reduced, the fraction cancelled with a monic
        denominator."""
        a = self._lift(a)
        for position, g in enumerate(self.generators):
            if g.eliminated is None:
                continue
            x = self.ring.gens[position]
            if a.num.degree(x) > 0 or a.den.degree(x) > 0:
                value = self._lift(g.eliminated)
                a = self._substituted(a.num, position, value) / self._substituted(a.den, position, value)
                a = self._lift(a)
        num, den = self._reduce(a.num), self._reduce(a.den)
        if not den:
            raise ZeroDivisionError("division by the zero of the tower")
        return self._fraction(num, den)

    def vanishes(self, a: Element) -> bool:
        """Whether the normal form of ``a`` is zero."""
        return not self.normal(a).num

    # ------------------------------------------------------------------
    # derivations

    def _variable_positions(self) -> list[int]:
        return [i for i, g in enumerate(self.generators) if g.kind == VARIABLE]

    def _generator_derivative(self, position: int, variable: int) -> Element:
        key = (position, variable)
        if key in self._derivatives:
            return self._derivatives[key]
        g = self.generators[position]
        if g.kind == VARIABLE:
            value = self.one if position == variable else self.zero
        elif g.kind == EXPONENTIAL and g.argument is not None:
            value = self.derivative(g.argument, variable) * self._generator_element(position)
        elif g.kind == LOGARITHM and g.argument is not None:
            value = self.derivative(g.argument, variable) / g.argument
        elif g.kind == ROOT and g.radicand is not None:
            # sigma**q == S: sigma' = S'*sigma/(q*S)
            value = self.derivative(g.radicand, variable) * self._generator_element(position) \
                / (g.radicand.times(Integer(g.index)))
        else:
            value = self.zero
        self._derivatives[key] = value
        return value

    def derivative(self, a: Element, variable: int) -> Element:
        """The derivative of ``a`` with respect to the variable generator
        at position ``variable``."""
        a = self.normal(a)

        def of_polynomial(p: PolyElement) -> Element:
            total = self.zero
            for position in range(len(self.generators)):
                x = self.ring.gens[position]
                if p.degree(x) <= 0:
                    continue
                partial = self._poly(p).diff(self.ring.gens[position])
                if not partial:
                    continue
                total = total + Element(self, partial, self.ring.one) * self._generator_derivative(position, variable)
            return total

        n, d = Element(self, a.num, self.ring.one), Element(self, a.den, self.ring.one)
        dn, dd = of_polynomial(a.num), of_polynomial(a.den)
        if not self._lift(dd).num:
            return dn / d
        return (dn * d - n * dd) / (d * d)

    def _formal_derivative(self, a: Element, position: int) -> Element:
        """The formal partial derivative with respect to the independent
        generator at ``position`` (a derivation of the field of rational
        functions, which the roots follow by their relations)."""
        a = self.normal(a)

        def of_polynomial(p: PolyElement) -> Element:
            p = self._poly(p)
            total = Element(self, p.diff(self.ring.gens[position]), self.ring.one)
            for j, g in enumerate(self.generators):
                if g.kind != ROOT or g.eliminated is not None or g.radicand is None or p.degree(self.ring.gens[j]) <= 0:
                    continue
                inner = self._formal_derivative(g.radicand, position)
                if not self._lift(inner).num:
                    continue
                total = total + Element(self, p.diff(self.ring.gens[j]), self.ring.one) * inner \
                    * self._generator_element(j) / g.radicand.times(Integer(g.index))
            return total

        n, d = Element(self, a.num, self.ring.one), Element(self, a.den, self.ring.one)
        return (of_polynomial(a.num) * d - n * of_polynomial(a.den)) / (d * d)

    def is_constant(self, a: Element) -> bool:
        """Whether every derivative of ``a`` vanishes."""
        return all(self.vanishes(self.derivative(a, k)) for k in self._variable_positions())

    # ------------------------------------------------------------------
    # linear algebra over the rationals

    def _field_degree(self) -> int:
        """The degree of the number field over the rationals."""
        K = self.domain
        if isinstance(K, AlgebraicField):
            return len(K.mod.to_list()) - 1
        return 1

    def _components(self, coefficient: DomainElement) -> list[DomainElement]:
        """The coordinates of an element of the number field over the
        rationals."""
        degree = self._field_degree()
        if degree == 1:
            return [coefficient]
        coordinates = list(coefficient.to_list())
        return [QQ.zero] * (degree - len(coordinates)) + [QQ.convert(c) for c in coordinates]

    def _solve(self, equations: list[tuple[Element, list[Element]]]) -> Optional[list[Rational]]:
        """Rational numbers ``r`` with ``target == sum(r[i]*basis[i])`` for
        every ``(target, basis)`` of ``equations``, or ``None`` when there
        are none. The equations are compared through their normal forms,
        coefficient by coefficient, the coefficients split over the basis
        of the number field."""
        if not equations:
            return []
        unknowns = len(equations[0][1])
        rows: list[list[DomainElement]] = []
        for target, basis in equations:
            members = [self.normal(target)] + [self.normal(b) for b in basis]
            common = self.ring.one
            for m in members:
                common = common.lcm(m.den)
            numerators = [self._reduce(m.num * common.quo(m.den)) for m in members]
            monomials: set[tuple[int, ...]] = set()
            for p in numerators:
                monomials.update(p.itermonoms())
            width = self._field_degree()
            for monomial in monomials:
                columns = [self._components(p.get(monomial, self.domain.zero)) if p.get(monomial, None) is not None
                           else [QQ.zero] * width for p in numerators]
                for k in range(width):
                    rows.append([columns[i + 1][k] for i in range(unknowns)] + [columns[0][k]])
        rows = [row for row in rows if any(row)]
        if not rows:
            return [Rational(0)] * unknowns
        if unknowns == 0:
            return None
        matrix = DomainMatrix(rows, (len(rows), unknowns + 1), QQ)
        reduced, pivots = matrix.rref()
        if unknowns in pivots:
            return None
        table = reduced.to_list()
        solution = [Rational(0)] * unknowns
        for row, pivot in enumerate(pivots):
            solution[pivot] = as_rational(QQ.to_sympy(table[row][unknowns]))
        return solution

    # ------------------------------------------------------------------
    # numbers

    def _number(self, value: Expr) -> Element:
        """An algebraic number as an element, the number field extended
        when it is not in it yet."""
        if isinstance(value, Rational):
            return self.constant(value)
        try:
            converted = self._field_element(value)
        except (CoercionFailed, NotAlgebraic, DomainError, PolynomialError, NotImplementedError, ValueError):
            # the field grows by a primitive element, whose cost explodes
            # with the degree: sqrt(2), I, 2**(1/3) and exp(I*pi/5) together
            # have degree 48 and SymPy does not finish
            t = Dummy('t')
            try:
                degree = int(Poly(minimal_polynomial(value, t), t).degree())
            except (NotAlgebraic, NotImplementedError, PolynomialError, ValueError):
                raise NotElementary("%s is not an algebraic number this module can hold" % (value,))
            if degree * self._field_degree() > _FIELD_DEGREE_LIMIT:
                raise NotElementary("the number field of %s would have degree up to %d"
                                    % (value, degree * self._field_degree()))
            # without its rational coefficient: QQ.algebraic_field(2*I, 2*(-1)**(1/3))
            # raises NotInvertible in SymPy 1.14 where (I, (-1)**(1/3)) works
            _, core = value.as_coeff_Mul()
            self.extensions.append(as_expr(core))
            try:
                self.domain = QQ.algebraic_field(*self.extensions)
                converted = self._field_element(value)
            except (CoercionFailed, NotAlgebraic, DomainError, PolynomialError, NotImplementedError, ValueError,
                    NotInvertible):
                self.extensions.pop()
                self.domain = QQ.algebraic_field(*self.extensions) if self.extensions else QQ
                raise NotElementary("%s is not an algebraic number this module can hold" % (value,))
            self._rebuild()
            self._derivatives.clear()
        return Element(self, self.ring.ground_new(converted), self.ring.one)

    def _field_element(self, value: Expr) -> DomainElement:
        """``value`` in the number field, sums, products and integer powers
        taken apart first: ``QQ.algebraic_field(w).from_sympy(2*w)`` fails
        in SymPy 1.14 for ``w = (-1)**(1/3)`` where ``w`` and ``1 + w``
        convert."""
        K = self.domain
        if isinstance(value, Rational):
            return K.convert(value.p) / K.convert(value.q)
        try:
            return K.from_sympy(value)                       # whole: 1 + sqrt(3)*I is in QQ<(-1)**(1/3)>, I is not
        except (CoercionFailed, NotAlgebraic, DomainError, PolynomialError, NotImplementedError, ValueError):
            if not isinstance(value, (Add, Mul, Pow)):
                raise
        if isinstance(value, Add):
            total = K.zero
            for term in value.args:
                total = total + self._field_element(as_expr(term))
            return total
        if isinstance(value, Mul):
            product = K.one
            for factor in value.args:
                product = product * self._field_element(as_expr(factor))
            return product
        if isinstance(value, Pow) and isinstance(value.exp, Integer):
            return self._field_element(as_expr(value.base))**int(value.exp)
        return K.from_sympy(value)

    def as_number(self, a: Element) -> Optional[Expr]:
        """The algebraic number which ``a`` is, or ``None`` when it
        involves a generator."""
        a = self.normal(a)
        if not a.num.is_ground or not a.den.is_ground:
            return None
        return as_expr(self.domain.to_sympy(a.num.LC if a.num else self.domain.zero)
                       / self.domain.to_sympy(a.den.LC))

    # ------------------------------------------------------------------
    # questions about the region

    def _facts(self) -> list[Boolean]:
        if self.assumptions is None:
            return []
        if isinstance(self.assumptions, (Boolean, bool)):
            return [as_boolean(self.assumptions)]
        return [as_boolean(fact) for fact in self.assumptions]

    def _ask(self, query: Boolean, more: Optional[list[Boolean]] = None) -> Optional[bool]:
        facts = self._facts() + list(more or [])
        found = attempt(lambda: ask(query, facts or None),
                        _ASK_SECONDS if settings.timeout is None else min(_ASK_SECONDS, settings.timeout))
        return found

    def _positive(self, a: Element) -> bool:
        return self._sign(self.to_expr(a)) > 0

    def _negative(self, a: Element) -> bool:
        return self._sign(self.to_expr(a)) < 0

    def _sign(self, value: Expr) -> int:
        """``1`` or ``-1`` when the assumptions prove ``value`` positive or
        negative on the region, ``0`` otherwise."""
        if value not in self._signs:
            sign = 0
            if value.is_positive:
                sign = 1
            elif value.is_negative:
                sign = -1
            elif value.free_symbols and value.is_extended_real is not False and not value.has(I):
                # a value written with I is not asked about: a sign is a
                # property of real values, and the question is only safe for them
                # (SymPy refuses to compare a number it knows not to be real)
                if self._ask(as_boolean(value > 0)) is True:
                    sign = 1
                elif self._ask(as_boolean(value < 0)) is True:
                    sign = -1
            self._signs[value] = sign
        return self._signs[value]

    def _real(self, a: Element) -> bool:
        value = self.to_expr(a)
        if value.is_extended_real:
            return True
        return self._ask(as_boolean(member(value, S.Reals))) is True if value.free_symbols else False

    # ------------------------------------------------------------------
    # exponentials

    def _split_pairs(self) -> tuple[list[int], list[int]]:
        """The positions of the pairs whose ``z`` is not constant, and of
        those whose ``z`` is."""
        moving: list[int] = []
        still: list[int] = []
        for i, pair in enumerate(self.pairs):
            (still if self.is_constant(pair.z) else moving).append(i)
        return moving, still

    def exponential(self, a: Element) -> Element:
        """``exp(a)`` in the tower: the product of powers of the ``e_i``
        when ``a`` is a rational combination of the ``z_i`` and a
        constant, a new generator otherwise."""
        a = self.normal(a)
        if not a.num:
            return self.one
        # a rational combination of the z_i as it stands: exp(log(v)/2)
        exact = self._solve([(a, [pair.z for pair in self.pairs])]) if self.pairs else None
        if exact is not None:
            result = self.one
            for i, r in enumerate(exact):
                if r != 0:
                    result = result * self._pair_power(i, r)
            return self.normal(result)
        variables = self._variable_positions()
        moving, still = self._split_pairs()
        factors: list[tuple[int, Rational]] = []
        constant = a
        if not self.is_constant(a):
            found = self._solve([(self.derivative(a, k), [self.derivative(self.pairs[i].z, k) for i in moving])
                                 for k in variables]) if moving else None
            if found is None:
                return self._new_exponential(a)
            for i, r in zip(moving, found):
                if r != 0:
                    factors.append((i, r))
                    constant = constant - self.pairs[i].z.times(r)
            constant = self.normal(constant)
            if not self.is_constant(constant):
                return self._new_exponential(a)
        result = self.one
        if constant.num:
            found = self._solve([(constant, [self.pairs[j].z for j in still])]) if still else None
            if found is None:
                # a constant written with functions (log(x*y) - log(x) - log(y),
                # which is 0 or 2*I*pi) has an exponential which is a number
                # nobody knows: a generator, and not an independent one
                result = self._new_exponential(constant, certified=self._free_of_functions(constant))
            else:
                for j, s in zip(still, found):
                    if s != 0:
                        factors.append((j, s))
        for i, r in factors:
            result = result * self._pair_power(i, r)
        return self.normal(result)

    def _free_of_functions(self, a: Element) -> bool:
        """Whether ``a`` involves no variable and no generator which
        depends on one: the constants of a certified tower are of this
        kind."""
        a = self.normal(a)
        for position in range(len(self.generators)):
            x = self.ring.gens[position]
            if (a.num.degree(x) > 0 or a.den.degree(x) > 0) \
                    and not self.is_constant(self._generator_element(position)):
                return False
        return True

    def _new_exponential(self, a: Element, certified: bool = True) -> Element:
        expression = exp(self.to_expr(a))
        position = self._add_generator(Generator(EXPONENTIAL, Dummy('e'), expression, a, certified))
        theta = self._generator_element(position)
        self.pairs.append(_Pair(a, theta, position, False))
        return theta

    def _pair_power(self, i: int, r: Rational) -> Element:
        """``exp(r*z_i)``."""
        if r.q == 1:
            return self.pairs[i].e**int(r.p)
        return self._root(i, int(r.q))**int(r.p)

    def _root(self, i: int, q: int) -> Element:
        """``exp(z_i/q)``: an algebraic number when ``e_i`` is one, the
        root generator of the pair otherwise, of index the least common
        multiple of the indices asked so far."""
        pair = self.pairs[i]
        number = self.as_number(pair.e)
        if number is not None and pair.logarithmic:
            return self._number(as_expr(Pow(number, Rational(1, q))))
        if pair.master >= 0:
            master = self.generators[pair.master]
            if master.index % q == 0 and master.scale is not None:
                return (self._generator_element(pair.master) / master.scale)**(master.index // q)
            index = math.lcm(master.index, q)
        else:
            index = q
        e = self.normal(pair.e)
        n, d = Element(self, e.num, self.ring.one), Element(self, e.den, self.ring.one)
        # the ring generator is sigma = d*rho, whose power is a polynomial
        if pair.logarithmic:
            expression = as_expr(self.to_expr(d) * Pow(self.to_expr(e), Rational(1, index)))
        else:
            expression = as_expr(self.to_expr(d) * exp(self.to_expr(pair.z) / index))
        generator = Generator(ROOT, Dummy('r'), expression)
        generator.index = index
        generator.source = i
        generator.radicand = n * d**(index - 1)
        generator.scale = d
        previous = pair.master
        position = self._add_generator(generator)
        pair.master = position
        rho = self._generator_element(position) / d
        if previous >= 0:
            old = self.generators[previous]
            if old.scale is not None:
                old.eliminated = old.scale * rho**(index // old.index)
        self._derivatives.clear()
        return rho**(index // q)

    # ------------------------------------------------------------------
    # logarithms

    def logarithm(self, v: Element) -> Element:
        """``log(v)`` in the tower. The factors of ``v`` which the
        assumptions prove positive are taken out first (``log(u*w) =
        log(u) + log(w)`` for ``u > 0`` and any ``w``), each with its own
        logarithm; what is left is a rational combination of the ``z_i``
        and the logarithm of a constant when its logarithmic derivative
        is one of the ``z_i'`` and the assumptions fix the branch, a new
        generator otherwise (uncertified when it is dependent)."""
        v = self.normal(v)
        if not v.num:
            raise NotElementary("the logarithm of zero")
        number = self.as_number(v)
        if number is not None:
            return self._number_logarithm(number)
        positive, rest = self._positive_part(v)
        if not positive:
            return self._structured_logarithm(v)
        total = self.zero
        for factor, power in positive:
            number = self.as_number(factor)
            piece = self._number_logarithm(number) if number is not None else self._structured_logarithm(factor)
            total = total + piece.times(Integer(power))
        if not self.vanishes(rest - self.one):
            number = self.as_number(rest)
            total = total + (self._number_logarithm(number) if number is not None
                             else self._structured_logarithm(rest))
        return self.normal(total)

    def _positive_part(self, v: Element) -> tuple[list[tuple[Element, int]], Element]:
        """The irreducible factors of ``v`` which are positive on the
        region, with their exponents, and the rest of ``v``; nothing is
        split when ``v`` involves a root or is one positive factor."""
        roots = [self.ring.gens[i] for i, g in enumerate(self.generators) if g.kind == ROOT]
        if any(v.num.degree(x) > 0 or v.den.degree(x) > 0 for x in roots):
            return [], v
        pieces: list[tuple[Element, int]] = []
        content = self.one
        for polynomial, sign in ((v.num, 1), (v.den, -1)):
            coefficient, factors = polynomial.factor_list()
            content = content * Element(self, self.ring.ground_new(coefficient), self.ring.one)**sign
            for factor, multiplicity in factors:
                pieces.append((Element(self, factor, self.ring.one), sign * int(multiplicity)))
        if len(pieces) == 1 and pieces[0][1] == 1 and self.vanishes(content - self.one):
            return [], v
        positive: list[tuple[Element, int]] = []
        rest = self.one
        value = self.as_number(content)
        if value is not None and value.is_positive:
            if value != 1:
                positive.append((content, 1))
        elif value is not None and value.is_negative:
            positive.append((-content, 1))
            rest = -rest
        else:
            rest = rest * content
        for factor, power in pieces:
            if self._positive(factor):
                positive.append((factor, power))
            elif self._negative(factor):
                positive.append((-factor, power))
                if power % 2:
                    rest = -rest
            else:
                rest = rest * factor**power
        if len(positive) == 1 and positive[0][1] == 1 and self.vanishes(rest - self.one):
            return [], v
        return positive, self.normal(rest)

    def _structured_logarithm(self, v: Element) -> Element:
        """``log(v)`` by the structure theorem, ``v`` taken whole."""
        v = self.normal(v)
        number = self.as_number(v)
        if number is not None:
            return self._number_logarithm(number)
        known = self._known_logarithm(v)
        if known is not None:
            return known
        variables = self._variable_positions()
        moving, _ = self._split_pairs()
        if self.is_constant(v):
            if self._positive(v):
                found_constant = self._positive_constant_logarithm(v)
                if found_constant is not None:
                    return found_constant
            return self._new_logarithm(v, certified=False)
        found = self._solve([(self.derivative(v, k) / v, [self.derivative(self.pairs[i].z, k) for i in moving])
                             for k in variables]) if moving else None
        if found is None:
            return self._new_logarithm(v, certified=True)
        used = [(i, r) for i, r in zip(moving, found) if r != 0]
        if self._positive(v) and all(self._pair_is_real(i) for i, _ in used):
            scale = math.lcm(*[int(r.q) for _, r in used]) if used else 1
            m = v**scale
            total = self.zero
            for i, r in used:
                m = m * self.pairs[i].e**(-int(r * scale))
                total = total + self.pairs[i].z.times(r)
            m = self.normal(m)
            if self.is_constant(m):
                rest = self._positive_constant_logarithm(m)
                if rest is not None:
                    return self.normal(total + rest.times(Rational(1, scale)))
        if self._negative(v):
            return self.normal(self._minus_one_logarithm() + self._structured_logarithm(-v))
        pinned = self._pinned_logarithm(v, used)
        if pinned is not None:
            return pinned
        return self._new_logarithm(v, certified=False)

    def _known_logarithm(self, v: Element) -> Optional[Element]:
        """``log(v)`` when ``v`` is the argument of a logarithm of the
        tower, or a power ``rho**p`` with ``0 < |p| < q`` of a root ``rho =
        exp(log(w)/q)`` (then ``log(v) = p*log(w)/q``, whose imaginary
        part lies within ``(-pi, pi)``: ``log(1/sqrt(x)) = -log(x)/2`` for
        every ``x``)."""
        for pair in self.pairs:
            if pair.logarithmic and self.vanishes(pair.e - v):
                return pair.z
        for position, g in enumerate(self.generators):
            if g.kind != ROOT or g.eliminated is not None or g.scale is None or g.source < 0:
                continue
            source = self.pairs[g.source]
            if not source.logarithmic:
                continue
            rho = self._generator_element(position) / g.scale
            power = self.one
            for k in range(1, g.index):
                power = power * rho
                if self.vanishes(power - v):
                    return source.z.times(Rational(k, g.index))
                if self.vanishes(power * v - self.one):
                    return source.z.times(Rational(-k, g.index))
        return None

    def _pinned_logarithm(self, v: Element, used: list[tuple[int, Rational]]) -> Optional[Element]:
        """``log(v) = sum(r_i*z_i) + c`` with the locally constant ``c``
        pinned at a sample point: exact when the variables are real, the
        region of the assumptions is convex (so connected) and no
        argument of a logarithm involved crosses the cut along the
        negative axis on it; then ``scale*c`` is a logarithm of the number
        ``v**scale * prod(e_i**(-r_i*scale))``, which the value of ``c`` at
        one point tells from the others (they differ by ``2*pi*I``)."""
        from sympy.core.relational import Ne
        from sympy.logic.boolalg import Or
        if not self._convex_region():
            return None
        from sympy.core.relational import Eq
        arguments = [v] + [self.pairs[i].e for i, _ in used if self.pairs[i].logarithmic]
        symbols = sorted_symbols(set().union(*[free_symbols(self.to_expr(w)) for w in arguments]))
        if not all(self._real(self.variable(symbol)) for symbol in symbols):
            return None
        # a root of a radicand positive on the region is a positive number:
        # a symbol rho > 0 with rho**q == radicand for the question below
        meanings: list[Expr] = []
        facts: list[Boolean] = []
        stand_ins: list[Symbol] = []
        for g in self.generators:
            meaning = g.expression
            if g.kind == ROOT and g.eliminated is None and g.scale is not None and g.source >= 0:
                source = self.pairs[g.source]
                if source.logarithmic and self._positive(source.e):
                    rho = Dummy('rho', positive=True)
                    stand_ins.append(rho)
                    meaning = as_expr(self.to_expr(g.scale) * rho)
                    facts.extend([as_boolean(rho > 0), as_boolean(Eq(rho**g.index, self.to_expr(source.e)))])
            meanings.append(meaning)
        reals = {symbol: Dummy(symbol.name, **{**symbol.assumptions0, 'real': True}) for symbol in symbols}
        back = {dummy: symbol for symbol, dummy in reals.items()}

        def parts(value: Expr) -> tuple[Expr, Expr]:
            real_part, imaginary_part = as_expr(value.xreplace(reals)).as_real_imag()
            return as_expr(real_part.xreplace(back)), as_expr(imaginary_part.xreplace(back))

        # everything in log(v) - sum(r_i*z_i) must be analytic on the whole
        # region: no pole of an argument or of a z_i inside it (the bug:
        # atan(x) + atan(1/x) - pi/2 was "proved" for every real x from
        # its value at one point, the pole at 0 cutting the line in two)
        for w in arguments + [self.pairs[i].z for i, _ in used]:
            denominator = as_expr(self.normal(w).den.as_expr(*meanings))
            if denominator.free_symbols:
                if not denominator.is_rational_function(*symbols, *stand_ins):
                    return None
                real_part, imaginary_part = parts(denominator)
                if self._ask(as_boolean(Or(Ne(real_part, 0), Ne(imaginary_part, 0))), facts) is not True:
                    return None
        for w in arguments:
            w = self.normal(w)
            value = as_expr(as_expr(w.num.as_expr(*meanings)) / as_expr(w.den.as_expr(*meanings)))
            if not value.is_rational_function(*symbols, *stand_ins):
                return None
            real_part, imaginary_part = parts(value)
            if imaginary_part == 0:
                if self._sign(self.to_expr(w)) == 0:
                    return None
            elif self._ask(as_boolean(Or(Ne(imaginary_part, 0), real_part > 0)), facts) is not True:
                return None
        scale = math.lcm(*[int(r.q) for _, r in used]) if used else 1
        m = v**scale
        total = self.zero
        for i, r in used:
            m = m * self.pairs[i].e**(-int(r * scale))
            total = total + self.pairs[i].z.times(r)
        number = self.as_number(self.normal(m))
        if number is None:
            return None
        principal = self._number_logarithm(number)
        from sympy_extras.integrals.conditions import sample_values
        values = sample_values(symbols, self.assumptions, random.Random(str(self.to_expr(v)))) if symbols else {}
        if values is None:
            return None
        gap = as_expr(scale * (log(self.to_expr(v)) - self.to_expr(total)) - self.to_expr(principal))
        measured = attempt(lambda: complex(as_expr(gap.xreplace(values)).evalf(40)), settings.timeout)
        if measured is None:
            return None
        turns = measured.imag / (2 * math.pi)
        if abs(measured.real) > 1e-12 or abs(turns - round(turns)) > 1e-12:
            return None
        constant = principal
        if round(turns) != 0:
            constant = constant + self._minus_one_logarithm().times(Integer(2 * round(turns)))
        return self.normal(total + constant.times(Rational(1, scale)))

    def _convex_region(self) -> bool:
        """Whether the assumptions are a conjunction of relations linear
        in the symbols (or there are none): a convex set, hence
        connected."""
        from sympy.core.relational import Ge, Gt, Le, Lt
        from sympy.logic.boolalg import And
        parts: list[Boolean] = []
        pending: list[Boolean] = self._facts()
        while pending:
            part = pending.pop()
            if isinstance(part, And):
                pending.extend(as_boolean(arg) for arg in part.args)
            else:
                parts.append(part)
        for part in parts:
            if part is S.true:
                continue
            if not isinstance(part, (Gt, Ge, Lt, Le)):
                return False
            difference = as_expr(part.lhs - part.rhs)
            symbols = sorted_symbols(free_symbols(difference))
            try:
                if symbols and Poly(difference, *symbols).total_degree() > 1:
                    return False
            except PolynomialError:
                return False
        return True

    def _pair_is_real(self, i: int) -> bool:
        """Whether ``z_i`` is real on the region: the argument of a
        logarithm positive, the argument of an exponential real."""
        pair = self.pairs[i]
        return self._positive(pair.e) if pair.logarithmic else self._real(pair.z)

    def _new_logarithm(self, v: Element, certified: bool) -> Element:
        expression = log(self.to_expr(v))
        position = self._add_generator(Generator(LOGARITHM, Dummy('l'), expression, v, certified))
        L = self._generator_element(position)
        self.pairs.append(_Pair(L, v, position, True))
        return L

    def _minus_one_logarithm(self) -> Element:
        """``log(-1) = I*pi``, the generator which carries ``pi``."""
        if self._minus_one < 0:
            v = self.constant(Rational(-1))
            position = self._add_generator(Generator(LOGARITHM, Dummy('l'), as_expr(I * pi), v))
            self.pairs.append(_Pair(self._generator_element(position), v, position, True))
            self._minus_one = position
        return self._generator_element(self._minus_one)

    def _prime_logarithm(self, p: int) -> Element:
        if p not in self._prime_logs:
            v = self.constant(Rational(p))
            position = self._add_generator(Generator(LOGARITHM, Dummy('l'), log(Integer(p)), v))
            self.pairs.append(_Pair(self._generator_element(position), v, position, True))
            self._prime_logs[p] = position
        return self._generator_element(self._prime_logs[p])

    def _number_logarithm(self, value: Expr) -> Element:
        """The principal logarithm of a nonzero algebraic number: the
        logarithms of the primes for a positive rational, ``I*pi`` more
        for a negative one, a fraction of those when a power of the
        number is rational, a generator otherwise."""
        if isinstance(value, Rational):
            if value < 0:
                return self.normal(self._minus_one_logarithm() + self._number_logarithm(-value))
            total = self.zero
            for prime, power in factorint(int(value.p)).items():
                total = total + self._prime_logarithm(int(prime)).times(Integer(power))
            for prime, power in factorint(int(value.q)).items():
                total = total - self._prime_logarithm(int(prime)).times(Integer(power))
            return self.normal(total)
        degree = int(Poly(minimal_polynomial(value, Dummy('t'))).degree())
        a = self._number(value)
        power = self.one
        for k in range(1, 2 * degree + 1):
            power = self.normal(power * a)
            rational = self.as_number(power)
            if isinstance(rational, Rational):
                # k*log(value) = log(rational) + 2*pi*I*j for an integer j
                real = self._number_logarithm(as_rational(abs(rational)))
                argument = complex(as_expr(value).evalf(30))
                angle = k * math.atan2(argument.imag, argument.real) - (math.pi if rational < 0 else 0.0)
                turns = round(angle / (2 * math.pi))
                extra = Rational(2 * turns + (1 if rational < 0 else 0), 1)
                total = real + self._minus_one_logarithm().times(extra) if extra != 0 else real
                return self.normal(total.times(Rational(1, k)))
        related = self._related_logarithm(value, a)
        if related is not None:
            return related
        certified = self._moduli_differ(value) and not any(
            g.kind == LOGARITHM and g.argument is not None and not isinstance(self.as_number(g.argument), Rational)
            and self.as_number(g.argument) is not None for g in self.generators)
        return self._new_logarithm(a, certified=certified)

    def _related_logarithm(self, value: Expr, a: Element) -> Optional[Element]:
        """The logarithm of an algebraic number through a multiplicative
        relation with the numbers whose logarithms are generators
        already: integers with ``value**n0 == u * prod(e_j**n_j)``, ``u`` a
        root of unity, are *found* numerically (an integer relation among
        the logarithms of the absolute values, by PSLQ) and *verified*
        exactly in the number field; the multiple of ``2*pi*I`` is told by
        the numerical value. Machin's formula is such a relation between
        ``5 + I``, ``239 - I``, ``1 + I`` and ``2``, ``13``."""
        import mpmath
        candidates: list[tuple[int, Expr]] = []
        for j, pair in enumerate(self.pairs):
            number = self.as_number(pair.e) if pair.logarithmic else None
            if number is not None and j != self._pair_of(self._minus_one):
                candidates.append((j, number))
        # the primes of the norm, whose logarithms a relation may need
        t = Dummy('t')
        polynomial = Poly(minimal_polynomial(value, t), t)
        norm = as_rational(polynomial.all_coeffs()[-1] / polynomial.all_coeffs()[0])
        for prime in sorted(set(factorint(abs(int(norm.p)))) | set(factorint(int(norm.q)))):
            if prime > 1 and int(prime) not in self._prime_logs and len(candidates) < 12:
                self._prime_logarithm(int(prime))
                candidates.append((self._pair_of(self._prime_logs[int(prime)]), Integer(prime)))
        if not candidates:
            return None
        # one relation for the moduli and the arguments at once: a generic
        # real combination log|e| + t*arg(e), with 2*pi*t for the turns
        with mpmath.workdps(80):
            t = mpmath.sqrt(mpmath.mpf(3)) + mpmath.mpf(1) / 7

            def coordinate(number: Expr) -> mpmath.mpf:
                z = mpmath.mpmathify(as_expr(number).evalf(80))
                return mpmath.log(abs(z)) + t * mpmath.arg(z)

            vector = [coordinate(value)] + [coordinate(number) for _, number in candidates] + [2 * mpmath.pi * t]
            relation = mpmath.pslq(vector, maxcoeff=10**4, maxsteps=10**5)
        if relation is None or relation[0] == 0:
            return None
        n0 = int(relation[0])
        # verified exactly: value**n0 * prod(e_j**n_j) == 1 in the number field
        product = a**n0
        for (j, _), n in zip(candidates, relation[1:-1]):
            product = product * self.pairs[j].e**int(n)
        if not self.vanishes(product - self.one):
            return None
        # n0*log(value) = -sum(n_j*w_j) + 2*pi*I*k, k told by the numerical value
        total = self.zero
        for (j, _), n in zip(candidates, relation[1:-1]):
            total = total - self.pairs[j].z.times(Integer(int(n)))
        gap = as_expr(n0 * log(value) - self.to_expr(total))
        measured = complex(gap.evalf(40))
        turns = measured.imag / (2 * math.pi)
        if abs(measured.real) > 1e-12 or abs(turns - round(turns)) > 1e-9:
            return None
        if round(turns) != 0:
            total = total + self._minus_one_logarithm().times(Integer(2 * round(turns)))
        return self.normal(total.times(Rational(1, n0)))

    def _pair_of(self, generator: int) -> int:
        for j, pair in enumerate(self.pairs):
            if pair.generator == generator:
                return j
        return -1

    @staticmethod
    def _moduli_differ(value: Expr) -> bool:
        """Whether two conjugates of the algebraic number have different
        absolute values: then no power of it is rational, and its
        logarithm is independent of the logarithms of the primes."""
        t = Dummy('t')
        roots = Poly(minimal_polynomial(value, t), t).nroots(n=30)
        moduli = sorted(abs(complex(root)) for root in roots)
        return moduli[-1] - moduli[0] > 1e-12 * (1 + moduli[-1])

    def _positive_constant_logarithm(self, m: Element) -> Optional[Element]:
        """The real logarithm of a positive constant of the field: of a
        number, or of a product of a number and powers of the ``e_j`` of
        constant pairs which are positive; ``None`` otherwise."""
        number = self.as_number(m)
        if number is not None:
            return self._number_logarithm(number) if number.is_positive else None
        _, still = self._split_pairs()
        independent = [self.pairs[j].generator for j in still if self.generators[self.pairs[j].generator].kind
                       in (EXPONENTIAL, LOGARITHM)]
        if not independent:
            return None
        found = self._solve([(self._formal_derivative(m, g) / m,
                              [self._formal_derivative(self.pairs[j].e, g) / self.pairs[j].e for j in still])
                             for g in independent])
        if found is None:
            return None
        used = [(j, s) for j, s in zip(still, found) if s != 0]
        if not all(self._pair_is_real(j) for j, _ in used):
            return None
        scale = math.lcm(*[int(s.q) for _, s in used]) if used else 1
        rest = m**scale
        total = self.zero
        for j, s in used:
            rest = rest * self.pairs[j].e**(-int(s * scale))
            total = total + self.pairs[j].z.times(s)
        value = self.as_number(self.normal(rest))
        if value is None or not value.is_positive:
            return None
        return self.normal(total + self._number_logarithm(value).times(Rational(1, scale)))

    # ------------------------------------------------------------------
    # expressions in, expressions out

    def variable(self, symbol: Symbol) -> Element:
        """The generator of a free symbol."""
        if symbol not in self._variables:
            self._variables[symbol] = self._add_generator(Generator(VARIABLE, Dummy(symbol.name), symbol))
            self._derivatives.clear()
        return self._generator_element(self._variables[symbol])

    def element(self, expression: object) -> Element:
        """The element of the tower which ``expression`` is; the tower
        grows by the generators it needs.

        Raises :class:`NotElementary` for an expression outside the
        elementary functions."""
        e = as_expr(expression)
        if e in self._cache:
            return self._cache[e]
        value = self._element(e)
        self._cache[e] = value
        return value

    def _element(self, e: Expr) -> Element:
        if isinstance(e, Rational):
            return self.constant(e)
        if isinstance(e, Float) or e.has(Float):
            raise NotElementary("a floating point number: %s" % (e,))
        if isinstance(e, Symbol):
            return self.variable(e)
        if e is pi:
            return self.normal(self._minus_one_logarithm() * self._number(as_expr(-I)))
        if e is EulerE:
            return self.exponential(self.one)
        if e is I or (e.is_number and e.is_algebraic and not e.has(exp, log, pi)):
            return self._number(e)
        if isinstance(e, Add):
            total = self.zero
            for term in e.args:
                total = total + self.element(term)
            return total
        if isinstance(e, Mul):
            product = self.one
            for factor in e.args:
                product = product * self.element(factor)
            return product
        if isinstance(e, exp):
            return self.exponential(self.element(e.args[0]))
        if isinstance(e, log):
            return self.logarithm(self.element(e.args[0]))
        if isinstance(e, Pow):
            base, exponent = as_expr(e.base), as_expr(e.exp)
            if isinstance(exponent, Integer):
                return self.element(base)**int(exponent)
            if base is EulerE:
                return self.exponential(self.element(exponent))
            b = self.element(base)
            if self.vanishes(b):
                raise NotElementary("a power of zero: %s" % (e,))
            return self.exponential(self.element(exponent) * self.logarithm(b))
        if isinstance(e, _EXPONENTIAL_FORMS):
            rewritten = as_expr(e.rewrite(exp))
            if not rewritten.has(*_EXPONENTIAL_FORMS):
                return self.element(rewritten)
        if isinstance(e, _LOGARITHMIC_FORMS):
            rewritten = as_expr(e.rewrite(log))
            if not rewritten.has(*_LOGARITHMIC_FORMS):
                return self.element(rewritten)
        raise NotElementary("not an elementary expression this module knows: %s" % (e,))

    def to_expr(self, a: Element) -> Expr:
        """The element as a SymPy expression in the meanings of the
        generators."""
        a = self.normal(a)
        meanings = [g.expression for g in self.generators]
        numerator = as_expr(a.num.as_expr(*meanings))
        denominator = as_expr(a.den.as_expr(*meanings))
        return as_expr(numerator / denominator)

    # ------------------------------------------------------------------
    # certification

    def _roots_certified(self) -> bool:
        """Kummer theory over the algebraic closure of the constants: the
        roots ``sigma_i**q_i == S_i`` generate an extension of degree
        ``prod(q_i)`` exactly when the ``S_i**(N/q_i)`` generate a subgroup
        of that order modulo ``N``-th powers, ``N`` the least common
        multiple of the ``q_i``."""
        if self._kummer is not None:
            return self._kummer
        roots = [g for g in self.generators if g.kind == ROOT and g.eliminated is None and g.radicand is not None]
        self._kummer = True
        if not roots:
            return True
        root_gens = [self.ring.gens[i] for i, g in enumerate(self.generators) if g.kind == ROOT]
        factors: list[PolyElement] = []
        vectors: list[dict[int, int]] = []
        for g in roots:
            if g.radicand is None:
                continue
            radicand = self.normal(g.radicand).num
            if any(radicand.degree(x) > 0 for x in root_gens):
                self._kummer = False
                return False
            vector: dict[int, int] = {}
            for factor, multiplicity in radicand.factor_list()[1]:
                monic = factor.monic()
                if monic not in factors:
                    factors.append(monic)
                vector[factors.index(monic)] = int(multiplicity)
            vectors.append(vector)
        if not factors:
            self._kummer = False
            return False
        N = math.lcm(*[g.index for g in roots])
        width = len(factors)
        rows = [[(N // g.index) * vector.get(j, 0) for j in range(width)] for g, vector in zip(roots, vectors)]
        rows += [[N if i == j else 0 for j in range(width)] for i in range(width)]
        normal_form = smith_normal_form(Matrix(rows), domain=ZZ)
        index = 1
        for k in range(width):
            index *= abs(int(normal_form[k, k]))
        order = N**width // index if index else 0
        expected = 1
        for g in roots:
            expected *= g.index
        self._kummer = order == expected
        return self._kummer

    def certifies(self, a: Element) -> bool:
        """Whether the generators which ``a`` involves are independent: a
        generator left out of ``a`` does not matter, each certified one
        being transcendental over all those before it (``sqrt(x*(x + 1))``
        is read through ``log(x*(x + 1))``, which depends on ``log(x)``
        and ``log(x + 1)`` when they are there, and is not in the root)."""
        a = self.normal(a)
        for position, g in enumerate(self.generators):
            x = self.ring.gens[position]
            if not g.certified and (a.num.degree(x) > 0 or a.den.degree(x) > 0):
                return False
        return self._roots_certified()

    @property
    def certified(self) -> bool:
        """Whether the generators are algebraically independent (the roots
        of full degree over the others): by the structure theorem for
        the functions, by Schanuel's conjecture for the constants."""
        return all(g.certified for g in self.generators) and self._roots_certified()


def as_rational(value: object) -> Rational:
    """``value`` as a SymPy :class:`~sympy.core.numbers.Rational`."""
    r = as_expr(value)
    if not isinstance(r, Rational):
        raise TypeError("a rational number is expected, got %s" % (r,))
    return r


def canonical_form(expression: object, assumptions: Assumptions = None) -> Expr:
    """The canonical form of an elementary expression: a quotient of two
    polynomials in generators which the structure theorem proves
    independent (see the module documentation), the denominator monic.

    Raises :class:`NotElementary` when the expression is not elementary.

    Examples
    ========

    >>> from sympy import symbols, exp, log, sqrt, sin, cos, tan, atan, pi, I
    >>> from sympy_extras.simplify import canonical_form
    >>> x, y = symbols('x y')
    >>> canonical_form(exp(x + y) - exp(x)*exp(y))
    0
    >>> canonical_form(sin(2*x) - 2*sin(x)*cos(x))
    0
    >>> canonical_form(exp(I*pi) + 1)
    0
    >>> canonical_form((sqrt(x) + 1)**2 - x - 1)
    2*sqrt(x)
    >>> canonical_form(log(x*y) - log(x) - log(y), [x > 0, y > 0])
    0
    """
    tower = ElementaryTower(assumptions)
    return tower.to_expr(tower.element(expression))


def is_zero(expression: object, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether an elementary expression vanishes identically on the
    region of the assumptions: ``True`` is a proof (the numerator of the
    canonical form is zero), ``False`` means that it does not vanish
    identically (the canonical form is not zero in a certified tower, or
    a sample point gives a value away from zero), ``None`` that neither
    was established.

    Examples
    ========

    >>> from sympy import symbols, exp, log, sqrt, sin, cos, cosh, sinh, atan, asin
    >>> from sympy_extras.simplify import is_zero
    >>> x = symbols('x')
    >>> is_zero(cosh(x)**2 - sinh(x)**2 - 1)
    True
    >>> is_zero(exp(x)*exp(-x) - 1), is_zero(exp(x) - x - 1)
    (True, False)
    >>> is_zero(sqrt(x)**2 - x), is_zero(sqrt(x**2) - x), is_zero(sqrt(x**2) - x, x > 0)
    (True, False, True)
    >>> is_zero(log(x**2) - 2*log(x)), is_zero(log(x**2) - 2*log(x), x > 0)
    (False, True)
    """
    e = as_expr(expression)
    tower = ElementaryTower(assumptions)

    def decide() -> Optional[bool]:
        try:
            value = tower.element(e)
            if tower.vanishes(value):
                return True
            return False if tower.certifies(value) else None
        except NotElementary:
            return None

    found = attempt(decide, settings.timeout)
    if found is not None:
        return found
    return False if _witness(e, assumptions) else None


def equal(a: object, b: object, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether two elementary expressions are equal on the region of the
    assumptions, by :func:`is_zero` of their difference.

    >>> from sympy import symbols, tan, sin, cos, sqrt
    >>> from sympy_extras.simplify import equal
    >>> x = symbols('x')
    >>> equal(tan(x), sin(x)/cos(x)), equal(sqrt(x)*sqrt(x), x), equal(sin(x), x)
    (True, True, False)
    """
    return is_zero(as_expr(a) - as_expr(b), assumptions)


def _witness(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is clearly not zero at a sample point of the region."""
    if not settings.numerical_checks:
        return False
    from sympy_extras.integrals.conditions import sample_values
    symbols = sorted_symbols(free_symbols(e))
    # a symbol with no flag which the assumptions do not mention is a
    # complex number: real points of both signs, then complex ones; the
    # others are sampled where their flags and the assumptions put them
    mentioned: set[Symbol] = set()
    if assumptions is not None:
        facts = [assumptions] if isinstance(assumptions, (Boolean, bool)) else list(assumptions)
        for fact in facts:
            mentioned |= free_symbols(as_boolean(fact))
    unconstrained = [s for s in symbols if s not in mentioned and set(s.assumptions0) <= {'commutative'}]
    constrained = [s for s in symbols if s not in unconstrained]
    rng = random.Random(str(e))
    for attempt_number in range(6):
        values: Optional[dict[Symbol, Expr]] = sample_values(constrained, assumptions, rng) if constrained else {}
        if values is None:
            return False
        for s in unconstrained:
            values[s] = as_expr(Rational(rng.randint(-300, 300), 100)
                                + (I * Rational(rng.randint(-1000, 1000), 100) if attempt_number >= 3 else 0))
        for s in constrained:
            # the other sign of a real symbol, where its flags and the
            # assumptions allow it (the sampler tends to one side)
            if s.is_nonnegative is None and s.is_nonpositive is None and s.is_integer is None \
                    and values[s].is_extended_real and rng.random() < 0.5:
                flipped = dict(values)
                flipped[s] = as_expr(-values[s])
                if assumptions is None or all(as_boolean(fact).xreplace(flipped) is S.true for fact in (
                        [assumptions] if isinstance(assumptions, (Boolean, bool)) else list(assumptions))):
                    values = flipped
        number = attempt(lambda: as_expr(as_expr(e.xreplace(values)).evalf(40)), settings.timeout)
        if number is None or not number.is_number or not number.is_finite:
            continue
        size = abs(complex(number))
        if size > 1e-25:
            return True
    return False
