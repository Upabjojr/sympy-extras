"""A canonical form and a zero test for elementary expressions: the
Risch–Rosenlicht structure theorem.

SymPy decides whether an expression vanishes by rewriting heuristics
(``simplify``, ``cancel``) and by sampling. This module writes an
elementary expression as a rational function of generators which are
*proven* algebraically independent, so that the expression vanishes
identically exactly when its numerator does.

The module works on the expressions of the typed core
(:mod:`sympy_extras.core`, the functions of :mod:`sympy_extras.functions`)
with the polynomials of :mod:`sympy_extras.polys`, and imports no SymPy:
:mod:`sympy_extras.simplify.structure_sympy` is its face on SymPy
objects. Two steps of the numerical side go through SymPy at run time
(the face holds them, the functions ending in ``_of_core``): the sample
points of the region of the assumptions
(:func:`sympy_extras.integrals.conditions.sample_values`) and the reliable
numerical values with which ``is_zero`` shows a difference not to vanish
(:func:`sympy_extras._numeric.reliable_value`).

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
  (`\\rho = v^{1/q}`, the principal root, which is defined as
  `\\exp(\\log(v)/q)`); it is the one kind of algebraic generator, kept
  with its relation `\\rho^q = e`.

Trigonometric and hyperbolic functions are written with exponentials,
their inverses with logarithms (the principal values agree, which the
tests check at complex points), `a^b` is `\\exp(b \\log a)`, and `\\pi` is
`-i \\log(-1)`: the constants `\\pi`, `e`, `\\log 2`, ... are generators
like the others.

The number field `K` is `\\mathbb{Q}(\\theta)` for a primitive element
`\\theta = e_1 + c_2 e_2 + \\dots` of the algebraic numbers met
(:func:`~sympy_extras.polys.numberfields.qqbar.primitive_element`); a
number is placed in it through its sum, product and power structure, or
by a linear factor of its minimal polynomial over `K` whose root is the
number (decided exactly, :mod:`~sympy_extras.polys.numberfields.qqbar`).

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
from fractions import Fraction
from typing import Optional, Sequence

import mpmath

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.assumptions.facts import element as member
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.logic import And, Boolean, BooleanTrue, or_, to_boolean
from sympy_extras.core.mul import Mul, div, mul
from sympy_extras.core.numbers import (Exp1, Float, GoldenRatioType, ImaginaryUnit, Integer, Pi, Rational, I,
                                       number_from_fraction, pi)
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Ge, Gt, Le, Lt, eq, ne
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import (Acosh, Acoth, Asinh, Atanh, Cosh, Coth, Csch, Sech, Sinh,
                                                          Tanh)
from sympy_extras.functions.elementary.trigonometric import (Acos, Acot, Asin, Atan, Cos, Cot, Csc, Sec, Sin, Tan)
from sympy_extras.ntheory.factor_ import factorint
from sympy_extras.polys.algebraicfactor import AlgebraicPolynomialRing, ext_factor_list
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.factortools import dup_ext_factor
from sympy_extras.polys.lattice import smith_normal_form
from sympy_extras.polys.numberfields.qqbar import Algebraic, primitive_element
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.rings import Monomial, PolyElement, PolyRing
from sympy_extras.sets.sets import Reals
from sympy_extras.settings import settings

__all__ = ['ElementaryTower', 'Element', 'Generator', 'NotElementary', 'canonical_form', 'is_zero', 'equal']

VARIABLE = 'variable'
EXPONENTIAL = 'exp'
LOGARITHM = 'log'
ROOT = 'root'

_EXPONENTIAL_FORMS = (Sin, Cos, Tan, Cot, Sec, Csc, Sinh, Cosh, Tanh, Coth, Sech, Csch)
_LOGARITHMIC_FORMS = (Asin, Acos, Atan, Acot, Asinh, Acosh, Atanh, Acoth)

#: the largest degree the number field is allowed to reach
_FIELD_DEGREE_LIMIT = 32

#: the budget of one question put to ``ask`` about the sign of an argument
_ASK_SECONDS = 5.0

#: an element of the number field: its coordinates in the powers of the
#: primitive element, highest degree first
Coefficient = list[Fraction]
#: a polynomial in the generators over the number field
TowerPolynomial = PolyElement[list[Fraction]]

#: the name of the one variable of the ring of a tower without generators
_PLACEHOLDER = '_'


class NotElementary(ValueError):
    """The expression is outside the towers of this module: a function
    which is not elementary, a floating point number, ``log(0)``."""


class _NotInField(ValueError):
    """A number which is not in the number field (yet)."""


class _Rationals(AlgebraicField):
    """``QQ`` as the number field of degree one (the field of a tower
    which has met no irrational number): its elements are the lists of
    the other fields, with the arithmetic done directly."""

    def __init__(self) -> None:
        super().__init__([Fraction(1), Fraction(0)], Integer(0))
        self.name = 'QQ'

    def add(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        s = (a[0] if a else Fraction(0)) + (b[0] if b else Fraction(0))
        return [s] if s else []

    def sub(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        s = (a[0] if a else Fraction(0)) - (b[0] if b else Fraction(0))
        return [s] if s else []

    def neg(self, a: list[Fraction]) -> list[Fraction]:
        return [-a[0]] if a else []

    def mul(self, a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
        return [a[0] * b[0]] if a and b else []

    def invert(self, a: list[Fraction]) -> list[Fraction]:
        if not a:
            raise ZeroDivisionError('0 has no inverse')
        return [1 / a[0]]


def _rational(value: Expr) -> Fraction:
    """A rational number of the core as a fraction."""
    if not isinstance(value, Rational):
        raise TypeError("a rational number is expected, got %s" % (value,))
    return Fraction(int(value.p), int(value.q))


def _algebraic(e: Expr) -> Algebraic:
    """The algebraic number which ``e`` is, built from rational numbers and
    ``I`` by sums, products and powers with rational exponents (principal
    roots); :class:`_NotInField` for anything else."""
    if isinstance(e, Rational):
        return Algebraic.from_rational(_rational(e))
    if isinstance(e, ImaginaryUnit):
        return Algebraic.imaginary_unit()
    if isinstance(e, GoldenRatioType):
        return (Algebraic.from_rational(1) + Algebraic.from_rational(5).nth_root(2)) / 2
    if isinstance(e, Add):
        total = Algebraic.from_rational(0)
        for term in e.args:
            total = total + _algebraic(as_expr(term))
        return total
    if isinstance(e, Mul):
        product = Algebraic.from_rational(1)
        for factor in e.args:
            product = product * _algebraic(as_expr(factor))
        return product
    if isinstance(e, Pow) and isinstance(e.exp, Rational):
        base = _algebraic(as_expr(e.base))
        p, q = int(e.exp.p), int(e.exp.q)
        if q != 1:
            if base.is_zero:
                raise _NotInField("a root of zero")
            base = base.nth_root(q)
        if p < 0:
            if base.is_zero:
                raise _NotInField("a negative power of zero")
            return (1 / base)**(-p)
        return base**p
    raise _NotInField("%s is not an algebraic number this module can hold" % (e,))


def _rational_power(e: Expr) -> Optional[tuple[Fraction, Fraction]]:
    """``(b, r)`` when ``e`` is the principal power ``b**r`` of rational
    numbers (``I`` is ``(-1)**(1/2)``), ``None`` otherwise."""
    if isinstance(e, ImaginaryUnit):
        return Fraction(-1), Fraction(1, 2)
    if isinstance(e, Pow) and isinstance(e.base, Rational) and isinstance(e.exp, Rational) and e.base != 0:
        return _rational(e.base), _rational(e.exp)
    return None


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """``(g, u, v)`` with ``u*a + v*b == g == gcd(a, b)``."""
    u0, v0, u1, v1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        u0, u1 = u1, u0 - q * u1
        v0, v1 = v1, v0 - q * v1
    if a < 0:
        return -a, -u0, -v0
    return a, u0, v0


class Element:
    """An element ``num/den`` of the field of an :class:`ElementaryTower`,
    the two polynomials in the generators of the tower at the time it
    was made (the tower lifts them to its current ring when it uses
    them). Elements of one tower are combined with ``+``, ``-``, ``*``,
    ``/`` and ``**`` with an integer."""

    __slots__ = ('tower', 'num', 'den')

    def __init__(self, tower: ElementaryTower, num: TowerPolynomial, den: TowerPolynomial) -> None:
        self.tower: ElementaryTower = tower
        self.num: TowerPolynomial = num
        self.den: TowerPolynomial = den

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

    def times(self, r: Fraction) -> Element:
        """``r`` times the element, ``r`` a rational number."""
        a = self.tower._lift(self)
        K = self.tower.domain
        return self.tower._fraction(a.num.mul_ground(K.from_int(r.numerator)),
                                    a.den.mul_ground(K.from_int(r.denominator)))

    def __repr__(self) -> str:
        return "Element(%s)" % (self.tower.to_expr(self),)


class Generator:
    """One generator of the tower.

    Attributes
    ==========

    kind : ``'variable'``, ``'exp'``, ``'log'`` or ``'root'``
    name : the name of its variable in the polynomial ring
    expression : its meaning as an expression of the core
    argument : ``a`` of ``exp(a)``, ``v`` of ``log(v)``
    index, source, radicand, scale : for a root ``rho = exp(z/q)`` of the
        pair ``source``: the ring generator is ``sigma = scale*rho`` with
        the polynomial relation ``sigma**index == radicand``
    eliminated : its value in the other generators, once a root of a
        larger index of the same pair has replaced it
    certified : whether it is independent of the generators before it
    """

    def __init__(self, kind: str, name: str, expression: Expr, argument: Optional[Element] = None,
                 certified: bool = True) -> None:
        self.kind: str = kind
        self.name: str = name
        self.expression: Expr = expression
        self.argument: Optional[Element] = argument
        self.certified: bool = certified
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
        self.z: Element = z
        self.e: Element = e
        self.generator: int = generator
        self.logarithmic: bool = logarithmic
        self.master: int = -1


def _facts_of(assumptions: Assumptions) -> list[Boolean]:
    """The assumptions as a list of statements."""
    if assumptions is None:
        return []
    if isinstance(assumptions, (Boolean, bool)):
        return [to_boolean(assumptions)]
    return [to_boolean(fact) for fact in assumptions]


def _rref(rows: list[list[Fraction]], width: int) -> tuple[list[list[Fraction]], list[int]]:
    """The reduced row echelon form of a matrix over the rationals and the
    columns of its pivots (Gauss–Jordan elimination)."""
    work = [list(row) for row in rows]
    pivots: list[int] = []
    r = 0
    for column in range(width):
        if r == len(work):
            break
        pivot = next((i for i in range(r, len(work)) if work[i][column]), -1)
        if pivot < 0:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        inverse = 1 / work[r][column]
        work[r] = [v * inverse for v in work[r]]
        for i in range(len(work)):
            if i != r and work[i][column]:
                factor = work[i][column]
                work[i] = [a - factor * b for a, b in zip(work[i], work[r])]
        pivots.append(column)
        r += 1
    return work, pivots


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

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import exp, log, sin, cos
    >>> from sympy_extras.simplify.structure import ElementaryTower
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
        self.assumptions: list[Boolean] = _facts_of(assumptions)
        self.extensions: list[Expr] = []
        self._numbers: list[Algebraic] = []
        self._coordinates: list[Coefficient] = []
        self._theta: Optional[Algebraic] = None
        self._theta_expression: Expr = Integer(0)
        self._degrees: list[int] = []
        self._to_basis: list[list[Fraction]] = []
        self._basis_expressions: list[Expr] = []
        self.domain: AlgebraicField = _Rationals()
        self._embeddings: dict[AlgebraicField, Coefficient] = {}
        self.generators: list[Generator] = []
        self.ring: PolyRing[list[Fraction]] = PolyRing(self.domain, [_PLACEHOLDER])
        self._rationals: Optional[RationalPolynomialRing] = None
        self._polynomials: Optional[AlgebraicPolynomialRing] = None
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

    def _names(self) -> list[str]:
        return [g.name for g in self.generators] or [_PLACEHOLDER]

    def _rebuild(self) -> None:
        self.ring = PolyRing(self.domain, self._names())
        self._rationals = None
        self._polynomials = None
        self._kummer = None

    @staticmethod
    def _count(ring: PolyRing[list[Fraction]]) -> int:
        """The number of generators of the tower when its ring was ``ring``."""
        return 0 if ring.names == [_PLACEHOLDER] else ring.ngens

    def _embed(self, c: Coefficient, theta: Coefficient) -> Coefficient:
        """The number ``c`` of an earlier field, whose primitive element
        is ``theta`` in the current one, in the current field."""
        K = self.domain
        result = K.zero
        for coefficient in c:
            result = K.add(K.mul(result, theta), K.from_rational(coefficient))
        return result

    def _poly(self, p: TowerPolynomial) -> TowerPolynomial:
        if p.ring is self.ring or p.ring == self.ring:
            return p
        old = self._count(p.ring)
        width = max(len(self.generators), 1)
        pad = (0,) * (width - old)
        theta: Optional[Coefficient] = None
        old_field = p.ring.domain
        if old_field != self.domain and isinstance(old_field, AlgebraicField):
            theta = self._embeddings[old_field]
        terms: dict[Monomial, Coefficient] = {}
        for m, c in p.terms.items():
            terms[m[:old] + pad] = self._embed(c, theta) if theta is not None else c
        return self.ring.from_dict(terms)

    def _lift(self, a: Element) -> Element:
        if a.tower is not self:
            raise ValueError("an element of another tower")
        if a.num.ring is self.ring:
            return a
        return Element(self, self._poly(a.num), self._poly(a.den))

    def _cancel(self, num: TowerPolynomial, den: TowerPolynomial) -> tuple[TowerPolynomial, TowerPolynomial]:
        """``num/den`` in lowest terms (the denominator up to a unit)."""
        if isinstance(self.domain, _Rationals):
            if self._rationals is None:
                self._rationals = RationalPolynomialRing(self.ring.names)
            R = self._rationals
            p, q = R.cancel(R.ring.from_dict({m: c[0] for m, c in num.terms.items()}),
                            R.ring.from_dict({m: c[0] for m, c in den.terms.items()}))
            return (self.ring.from_dict({m: [c] for m, c in p.terms.items()}),
                    self.ring.from_dict({m: [c] for m, c in q.terms.items()}))
        if self._polynomials is None:
            self._polynomials = AlgebraicPolynomialRing(self.domain, self.ring.names)
        h = self._polynomials.gcd(num, den)
        if h.is_ground:
            return num, den
        return num.exquo(h), den.exquo(h)

    def _fraction(self, num: TowerPolynomial, den: TowerPolynomial) -> Element:
        if not den:
            raise ZeroDivisionError("division by the zero of the tower")
        if not num:
            return Element(self, self.ring.zero, self.ring.one)
        p, q = self._cancel(num, den)
        lead = q.LC
        if not self.domain.is_one(lead):
            inverse = self.domain.invert(lead)
            p, q = p.mul_ground(inverse), q.mul_ground(inverse)
        return Element(self, p, q)

    def _combine(self, a: Element, b: Element, sign: int) -> Element:
        a, b = self._lift(a), self._lift(b)
        if a.den == b.den:
            return self._fraction(a.num + b.num if sign > 0 else a.num - b.num, a.den)
        return self._fraction(a.num * b.den + b.num * a.den if sign > 0 else a.num * b.den - b.num * a.den,
                              a.den * b.den)

    def constant(self, value: Fraction) -> Element:
        """The rational number ``value`` as an element."""
        K = self.domain
        return Element(self, self.ring.ground(K.from_int(value.numerator)),
                       self.ring.ground(K.from_int(value.denominator)))

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

    def _new_name(self) -> str:
        return 'g%d' % len(self.generators)

    # ------------------------------------------------------------------
    # normal forms

    def _reduce(self, p: TowerPolynomial) -> TowerPolynomial:
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
                if p.degree_in(position) < g.index:
                    continue
                radicand = self._lift(g.radicand).num
                result = self.ring.zero
                for monomial, coefficient in p.terms.items():
                    e = monomial[position]
                    rest = monomial[:position] + (e % g.index,) + monomial[position + 1:]
                    term = self.ring.term(rest, coefficient)
                    if e >= g.index:
                        term = term * radicand**(e // g.index)
                    result = result + term
                p = result
                changed = True
        return p

    def _substituted(self, p: TowerPolynomial, position: int, value: Element) -> Element:
        """``p`` with the generator at ``position`` replaced by ``value``."""
        parts: dict[int, TowerPolynomial] = {}
        for monomial, coefficient in p.terms.items():
            e = monomial[position]
            rest = monomial[:position] + (0,) + monomial[position + 1:]
            parts[e] = parts.get(e, self.ring.zero) + self.ring.term(rest, coefficient)
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
            if a.num.degree_in(position) > 0 or a.den.degree_in(position) > 0:
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
                / (g.radicand.times(Fraction(g.index)))
        else:
            value = self.zero
        self._derivatives[key] = value
        return value

    def derivative(self, a: Element, variable: int) -> Element:
        """The derivative of ``a`` with respect to the variable generator
        at position ``variable``."""
        a = self.normal(a)

        def of_polynomial(p: TowerPolynomial) -> Element:
            total = self.zero
            for position in range(len(self.generators)):
                if p.degree_in(position) <= 0:
                    continue
                partial = self._poly(p).diff(position)
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

        def of_polynomial(p: TowerPolynomial) -> Element:
            p = self._poly(p)
            total = Element(self, p.diff(position), self.ring.one)
            for j, g in enumerate(self.generators):
                if g.kind != ROOT or g.eliminated is not None or g.radicand is None or p.degree_in(j) <= 0:
                    continue
                inner = self._formal_derivative(g.radicand, position)
                if not self._lift(inner).num:
                    continue
                total = total + Element(self, p.diff(j), self.ring.one) * inner \
                    * self._generator_element(j) / g.radicand.times(Fraction(g.index))
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
        return self.domain.degree

    def _components(self, coefficient: Coefficient) -> list[Fraction]:
        """The coordinates of an element of the number field over the
        rationals."""
        degree = self._field_degree()
        return [Fraction(0)] * (degree - len(coefficient)) + list(coefficient)

    def _solve(self, equations: list[tuple[Element, list[Element]]]) -> Optional[list[Fraction]]:
        """Rational numbers ``r`` with ``target == sum(r[i]*basis[i])`` for
        every ``(target, basis)`` of ``equations``, or ``None`` when there
        are none. The equations are compared through their normal forms,
        coefficient by coefficient, the coefficients split over the basis
        of the number field."""
        if not equations:
            return []
        unknowns = len(equations[0][1])
        rows: list[list[Fraction]] = []
        width = self._field_degree()
        for target, basis in equations:
            members = [self.normal(target)] + [self.normal(b) for b in basis]
            common = self.ring.one
            for m in members:
                common = self._lcm(common, m.den)
            numerators = [self._reduce(m.num * common.exquo(m.den)) for m in members]
            monomials: set[Monomial] = set()
            for p in numerators:
                monomials.update(p.terms)
            for monomial in monomials:
                columns = [self._components(p.terms[monomial]) if monomial in p.terms else [Fraction(0)] * width
                           for p in numerators]
                for k in range(width):
                    rows.append([columns[i + 1][k] for i in range(unknowns)] + [columns[0][k]])
        rows = [row for row in rows if any(row)]
        if not rows:
            return [Fraction(0)] * unknowns
        if unknowns == 0:
            return None
        table, pivots = _rref(rows, unknowns + 1)
        if unknowns in pivots:
            return None
        solution = [Fraction(0)] * unknowns
        for row, pivot in enumerate(pivots):
            solution[pivot] = table[row][unknowns]
        return solution

    def _lcm(self, a: TowerPolynomial, b: TowerPolynomial) -> TowerPolynomial:
        """A least common multiple of two denominators."""
        a, b = self._poly(a), self._poly(b)
        if a.is_ground:
            return b
        if b.is_ground or a == b:
            return a
        _, cofactor = self._cancel(a, b)
        return a * cofactor

    # ------------------------------------------------------------------
    # numbers

    def _number(self, value: Expr) -> Element:
        """An algebraic number as an element, the number field extended
        when it is not in it yet."""
        if isinstance(value, Rational):
            return self.constant(_rational(value))
        try:
            converted = self._field_element(value)
        except (_NotInField, ArithmeticError):
            # the field grows by a primitive element, whose cost explodes
            # with the degree: sqrt(2), I, 2**(1/3) and exp(I*pi/5) together
            # have degree 48
            try:
                number = _algebraic(value)
            except (_NotInField, ArithmeticError):
                raise NotElementary("%s is not an algebraic number this module can hold" % (value,))
            if number.is_rational:
                return self.constant(number.rational_value())
            degree = number.degree
            if degree * self._field_degree() > _FIELD_DEGREE_LIMIT:
                raise NotElementary("the number field of %s would have degree up to %d"
                                    % (value, degree * self._field_degree()))
            # without its rational coefficient
            _, core = value.as_coeff_Mul()
            before = self._field_degree()
            try:
                self._extend(core)
                converted = self._field_element(value)
            except (_NotInField, ArithmeticError):
                raise NotElementary("%s is not an algebraic number this module can hold" % (value,))
            finally:
                if self._field_degree() != before:
                    self._rebuild()
                    self._derivatives.clear()
        return Element(self, self.ring.ground(converted), self.ring.one)

    def _extend(self, extension: Expr) -> None:
        """The number field extended by the algebraic number
        ``extension``: a new primitive element ``theta + c*extension``, the
        coordinates of the old numbers rewritten in it."""
        number = _algebraic(extension)
        if number.is_rational:
            raise _NotInField("%s is rational" % (extension,))
        if self._theta is None:
            theta, old, new = number, [Fraction(1), Fraction(0)], [Fraction(1), Fraction(0)]
            expression = extension
        else:
            theta, old, new = primitive_element(self._theta, number)
        if theta.degree == self._field_degree():
            raise _NotInField("%s is in the field already" % (extension,))
        leading = theta.minpoly[0]
        K = AlgebraicField([Fraction(c, leading) for c in theta.minpoly], Integer(0))
        if self._theta is not None:
            # theta = old + c*new: c is a rational number
            ratio = K.quo(K.sub(K.alpha, K._reduce(old)), K._reduce(new))
            c = ratio[0] if ratio else Fraction(0)
            expression = add(self._theta_expression, mul(number_from_fraction(c), extension))
        K = AlgebraicField([Fraction(c, leading) for c in theta.minpoly], expression)
        previous = self.domain
        embedding = K._reduce(list(old))
        self.domain = K
        self._embeddings = {field: self._embed(rep, embedding) for field, rep in self._embeddings.items()}
        self._embeddings[previous] = embedding
        self._coordinates = [self._embed(c, embedding) for c in self._coordinates] + [K._reduce(list(new))]
        self.extensions.append(extension)
        self._numbers.append(number)
        self._degrees.append(K.degree // previous.degree)
        self._theta = theta
        self._theta_expression = expression
        self._monomial_basis()

    def _monomial_basis(self) -> None:
        """The basis of the products ``prod(e_i**k_i)``, ``k_i`` below the
        relative degree of the extension ``e_i``, in which the numbers are
        written as expressions: their expressions, and the matrix taking
        coordinates in the powers of the primitive element to coordinates
        in it."""
        K = self.domain
        n = K.degree
        exponents: list[list[int]] = [[]]
        for d in self._degrees:
            exponents = [e + [k] for e in exponents for k in range(d)]
        rows: list[list[Fraction]] = []
        expressions: list[Expr] = []
        for e in exponents:
            c = K.one
            factors: list[Expr] = []
            for k, coordinates, extension in zip(e, self._coordinates, self.extensions):
                if k:
                    c = K.mul(c, K.pow(coordinates, k))
                    factors.append(pow(extension, Integer(k)))
            rows.append(self._components(c))
            expressions.append(mul(*factors) if factors else Integer(1))
        # coordinates y in the basis of a number with coordinates c (in the
        # powers, highest first): sum(y_j*rows[j]) == c, solved once by
        # inverting the transposed matrix
        width = len(rows)
        augmented = [[rows[j][i] for j in range(width)] + [Fraction(int(i == k)) for k in range(n)] for i in range(n)]
        reduced, pivots = _rref(augmented, width)
        if pivots != list(range(width)):
            raise _NotInField("the products of the extensions are not a basis")
        self._to_basis = [row[width:] for row in reduced[:width]]
        self._basis_expressions = expressions

    def _field_element(self, value: Expr) -> Coefficient:
        """``value`` in the number field: rational numbers, the extensions,
        their sums, products and integer powers, and the other numbers by
        a linear factor of their minimal polynomial over the field whose
        root they are; :class:`_NotInField` otherwise."""
        K = self.domain
        if isinstance(value, Rational):
            return K.from_rational(_rational(value))
        for extension, coordinates in zip(self.extensions, self._coordinates):
            if value == extension:
                return coordinates
        try:
            if isinstance(value, Add):
                total = K.zero
                for term in value.args:
                    total = K.add(total, self._field_element(as_expr(term)))
                return total
            if isinstance(value, Mul):
                product = K.one
                for factor in value.args:
                    product = K.mul(product, self._field_element(as_expr(factor)))
                return product
            if isinstance(value, Pow) and isinstance(value.exp, Integer):
                base = self._field_element(as_expr(value.base))
                n = int(value.exp.p)
                if n < 0:
                    if not base:
                        raise _NotInField("a negative power of zero")
                    return K.pow(K.invert(base), -n)
                return K.pow(base, n)
        except _NotInField:
            # the parts are not in the field, the whole may be: sqrt(3)*I
            # in QQ<(-1)**(1/3)>
            pass
        theta = self._theta
        if theta is None:
            raise _NotInField("%s is not rational" % (value,))
        power = self._radical_element(value)
        if power is not None:
            return power
        number = _algebraic(value)
        if number.is_rational:
            return K.from_rational(number.rational_value())
        leading = number.minpoly[0]
        f = [K.from_rational(Fraction(c, leading)) for c in number.minpoly]
        _, factors = dup_ext_factor(f, K)
        for g, _ in factors:
            if len(g) == 2:
                root = K.neg(K.quo(g[1], g[0]))
                if theta.polynomial_value(root) == number:
                    return root
        raise _NotInField("%s is not in %s" % (value, K))

    def _radical_element(self, value: Expr) -> Optional[Coefficient]:
        """``b**r`` (``b`` and ``r`` rational) as a product of powers of the
        extensions ``b**s``: the principal power ``exp(r*log(b))`` is a
        homomorphism in ``r``, so that ``b**r = prod((b**s_i)**n_i)`` for
        integers with ``r = sum(n_i*s_i)``; ``None`` when ``r`` is not such
        a combination."""
        radical = _rational_power(value)
        if radical is None:
            return None
        base, r = radical
        direct = self._power_of_extensions(base, r)
        if direct is not None:
            return direct
        # b**r = (-1)**r * prod(p**(e*r)) over the primes of b, exactly for
        # the principal powers (|b|**r is a positive real number)
        K = self.domain
        result = K.one
        parts: list[tuple[Fraction, Fraction]] = [(Fraction(-1), r)] if base < 0 else []
        for n, sign in ((abs(base.numerator), 1), (base.denominator, -1)):
            for prime, exponent in factorint(n).items():
                parts.append((Fraction(prime), r * exponent * sign))
        if len(parts) < 2:
            return None
        for b, t in parts:
            found = self._power_of_extensions(b, t) if t.denominator > 1 else K.pow(K.from_rational(b), t.numerator) \
                if t >= 0 else K.pow(K.invert(K.from_rational(b)), -t.numerator)
            if found is None:
                return None
            result = K.mul(result, found)
        return result

    def _power_of_extensions(self, base: Fraction, r: Fraction) -> Optional[Coefficient]:
        """``base**r`` as a product of integer powers of the extensions
        ``base**s``, ``None`` when there is none."""
        same: list[tuple[Fraction, Coefficient]] = []
        for extension, coordinates in zip(self.extensions, self._coordinates):
            other = _rational_power(extension)
            if other is not None and other[0] == base:
                same.append((other[1], coordinates))
        if not same:
            return None
        L = math.lcm(*[e.denominator for e, _ in same], r.denominator)
        steps = [int(e * L) for e, _ in same]
        # Bezout coefficients of the steps, one at a time
        g, coefficients = steps[0], [1] + [0] * (len(steps) - 1)
        for i in range(1, len(steps)):
            d, u, v = _extended_gcd(g, steps[i])
            coefficients = [c * u for c in coefficients]
            coefficients[i] = v
            g = d
        target = int(r * L)
        if target % g:
            return None
        K = self.domain
        result = K.one
        for n, (_, coordinates) in zip(coefficients, same):
            n *= target // g
            if n:
                result = K.mul(result, K.pow(coordinates, n) if n > 0 else K.pow(K.invert(coordinates), -n))
        return result

    def _coefficient_expr(self, c: Coefficient) -> Expr:
        """A number of the field as an expression in the extensions: a
        rational combination of the products of their powers."""
        if len(c) <= 1:
            return number_from_fraction(c[0]) if c else Integer(0)
        vector = self._components(c)
        terms: list[Expr] = []
        for row, expression in zip(self._to_basis, self._basis_expressions):
            y = sum((a * b for a, b in zip(row, vector)), Fraction(0))
            if y:
                terms.append(mul(number_from_fraction(y), expression))
        return add(*terms) if terms else Integer(0)

    def as_number(self, a: Element) -> Optional[Expr]:
        """The algebraic number which ``a`` is, or ``None`` when it
        involves a generator."""
        a = self.normal(a)
        if not a.num.is_ground or not a.den.is_ground:
            return None
        return div(self._coefficient_expr(a.num.LC), self._coefficient_expr(a.den.LC))

    # ------------------------------------------------------------------
    # questions about the region

    def _facts(self) -> list[Boolean]:
        return list(self.assumptions)

    def _ask(self, query: Boolean, more: Optional[list[Boolean]] = None) -> Optional[bool]:
        facts = self._facts() + list(more or [])
        found = attempt(lambda: ask(query, facts or None),
                        _ASK_SECONDS if settings.timeout is None else min(_ASK_SECONDS, settings.timeout),
                        native=True)
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
                if self._ask(value > 0) is True:
                    sign = 1
                elif self._ask(value < 0) is True:
                    sign = -1
            self._signs[value] = sign
        return self._signs[value]

    def _real(self, a: Element) -> bool:
        value = self.to_expr(a)
        if value.is_extended_real:
            return True
        return self._ask(member(value, Reals)) is True if value.free_symbols else False

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
        factors: list[tuple[int, Fraction]] = []
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
            if (a.num.degree_in(position) > 0 or a.den.degree_in(position) > 0) \
                    and not self.is_constant(self._generator_element(position)):
                return False
        return True

    def _sound(self) -> bool:
        """Whether every generator so far is certified: the independence
        of a new exponential or logarithm is read from a linear system
        over the generators, and means nothing once one of them is not
        what it seems (the bug: after ``log(x)`` and the dependent
        ``log(x**2)``, ``exp(v*log(x**2)/2)`` passed for independent of
        ``exp(v*log(x))``, the two logarithms being unrelated formally)."""
        return all(g.certified for g in self.generators)

    def _new_exponential(self, a: Element, certified: bool = True) -> Element:
        certified = certified and self._sound()
        expression = exp(self.to_expr(a))
        position = self._add_generator(Generator(EXPONENTIAL, self._new_name(), expression, a, certified))
        theta = self._generator_element(position)
        self.pairs.append(_Pair(a, theta, position, False))
        return theta

    def _pair_power(self, i: int, r: Fraction) -> Element:
        """``exp(r*z_i)``."""
        if r.denominator == 1:
            return self.pairs[i].e**r.numerator
        return self._root(i, r.denominator)**r.numerator

    def _root(self, i: int, q: int) -> Element:
        """``exp(z_i/q)``: an algebraic number when ``e_i`` is one, the
        root generator of the pair otherwise, of index the least common
        multiple of the indices asked so far."""
        pair = self.pairs[i]
        number = self.as_number(pair.e)
        if number is not None and pair.logarithmic:
            return self._number(pow(number, number_from_fraction(Fraction(1, q))))
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
            expression = mul(self.to_expr(d), pow(self.to_expr(e), number_from_fraction(Fraction(1, index))))
        else:
            expression = mul(self.to_expr(d), exp(div(self.to_expr(pair.z), Integer(index))))
        generator = Generator(ROOT, self._new_name(), expression)
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
        if self._is_number(v):
            return self._number_logarithm(v)
        positive, rest = self._positive_part(v)
        if not positive:
            return self._structured_logarithm(v)
        total = self.zero
        for factor, power in positive:
            piece = self._number_logarithm(factor) if self._is_number(factor) else self._structured_logarithm(factor)
            total = total + piece.times(Fraction(power))
        if not self.vanishes(rest - self.one):
            total = total + (self._number_logarithm(rest) if self._is_number(rest)
                             else self._structured_logarithm(rest))
        return self.normal(total)

    def _factor_list(self, p: TowerPolynomial) -> tuple[Coefficient, list[tuple[TowerPolynomial, int]]]:
        """The coefficient and the irreducible factors of ``p`` over the
        number field: over the rationals the primitive integral factors
        with a positive leading coefficient, over a larger field the
        monic ones."""
        p = self._poly(p)
        if isinstance(self.domain, _Rationals):
            from sympy_extras.polys.polytools import _factor_all
            ring = PolyRing(QQ, self.ring.names)
            coefficient, factors = _factor_all(ring.from_dict({m: c[0] for m, c in p.terms.items()}))
            return (self.domain.from_rational(coefficient),
                    [(self.ring.from_dict({m: [Fraction(c)] for m, c in f.terms.items()}), k) for f, k in factors])
        return ext_factor_list(p)

    def _positive_part(self, v: Element) -> tuple[list[tuple[Element, int]], Element]:
        """The irreducible factors of ``v`` which are positive on the
        region, with their exponents, and the rest of ``v``; nothing is
        split when ``v`` involves a root or is one positive factor."""
        roots = [i for i, g in enumerate(self.generators) if g.kind == ROOT]
        if any(v.num.degree_in(x) > 0 or v.den.degree_in(x) > 0 for x in roots):
            return [], v
        pieces: list[tuple[Element, int]] = []
        content = self.one
        for polynomial, sign in ((v.num, 1), (v.den, -1)):
            coefficient, factors = self._factor_list(polynomial)
            content = content * Element(self, self.ring.ground(coefficient), self.ring.one)**sign
            for polynomial_factor, multiplicity in factors:
                pieces.append((Element(self, polynomial_factor, self.ring.one), sign * multiplicity))
        if len(pieces) == 1 and pieces[0][1] == 1 and self.vanishes(content - self.one):
            return [], v
        positive: list[tuple[Element, int]] = []
        rest = self.one
        if self._is_number(content) and self._positive_number(content):
            if not self.vanishes(content - self.one):
                positive.append((content, 1))
        elif self._is_number(content) and self._negative_number(content):
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
        if self._is_number(v):
            return self._number_logarithm(v)
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
            scale = math.lcm(*[r.denominator for _, r in used]) if used else 1
            m = v**scale
            total = self.zero
            for i, r in used:
                m = m * self.pairs[i].e**(-int(r * scale))
                total = total + self.pairs[i].z.times(r)
            m = self.normal(m)
            if self.is_constant(m):
                rest = self._positive_constant_logarithm(m)
                if rest is not None:
                    return self.normal(total + rest.times(Fraction(1, scale)))
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
                    return source.z.times(Fraction(k, g.index))
                if self.vanishes(power * v - self.one):
                    return source.z.times(Fraction(-k, g.index))
        return None

    def _pinned_logarithm(self, v: Element, used: list[tuple[int, Fraction]]) -> Optional[Element]:
        """``log(v) = sum(r_i*z_i) + c`` with the locally constant ``c``
        pinned at a sample point: exact when the variables are real, the
        region of the assumptions is convex (so connected) and no
        argument of a logarithm involved crosses the cut along the
        negative axis on it; then ``scale*c`` is a logarithm of the number
        ``v**scale * prod(e_i**(-r_i*scale))``, which the value of ``c`` at
        one point tells from the others (they differ by ``2*pi*I``)."""
        if not self._convex_region():
            return None
        arguments = [v] + [self.pairs[i].e for i, _ in used if self.pairs[i].logarithmic]
        found_symbols: set[Symbol] = set()
        for w in arguments:
            found_symbols |= self.to_expr(w).free_symbols
        symbols = sorted(found_symbols, key=lambda s: s.name)
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
                    meaning = mul(self.to_expr(g.scale), rho)
                    facts.extend([rho > 0, eq(pow(rho, Integer(g.index)), self.to_expr(source.e))])
            meanings.append(meaning)
        reals: dict[Basic, Basic] = {}
        back: dict[Basic, Basic] = {}
        for symbol in symbols:
            dummy = Dummy(symbol.name, **{**symbol.assumptions0, 'real': True})
            reals[symbol] = dummy
            back[dummy] = symbol

        def parts(value: Expr) -> tuple[Expr, Expr]:
            real_part, imaginary_part = value.xreplace(reals).as_real_imag()
            return real_part.xreplace(back), imaginary_part.xreplace(back)

        # everything in log(v) - sum(r_i*z_i) must be analytic on the whole
        # region: no pole of an argument or of a z_i inside it (the bug:
        # atan(x) + atan(1/x) - pi/2 was "proved" for every real x from
        # its value at one point, the pole at 0 cutting the line in two)
        for w in arguments + [self.pairs[i].z for i, _ in used]:
            denominator = self._polynomial_expr(self.normal(w).den, meanings)
            if denominator.free_symbols:
                if not denominator.is_rational_function(*symbols, *stand_ins):
                    return None
                real_part, imaginary_part = parts(denominator)
                if self._ask(or_(ne(real_part, 0), ne(imaginary_part, 0)), facts) is not True:
                    return None
        for w in arguments:
            w = self.normal(w)
            value = div(self._polynomial_expr(w.num, meanings), self._polynomial_expr(w.den, meanings))
            if not value.is_rational_function(*symbols, *stand_ins):
                return None
            real_part, imaginary_part = parts(value)
            if imaginary_part == 0:
                if self._sign(self.to_expr(w)) == 0:
                    return None
            elif self._ask(or_(ne(imaginary_part, 0), real_part > 0), facts) is not True:
                return None
        scale = math.lcm(*[r.denominator for _, r in used]) if used else 1
        m = v**scale
        total = self.zero
        for i, r in used:
            m = m * self.pairs[i].e**(-int(r * scale))
            total = total + self.pairs[i].z.times(r)
        m = self.normal(m)
        if not self._is_number(m):
            return None
        principal = self._number_logarithm(m)
        values: Optional[dict[Symbol, Expr]] = {}
        if symbols:
            from .structure_sympy import sample_values_of_core
            values = sample_values_of_core(symbols, self.assumptions, random.Random(str(self.to_expr(v))))
        if values is None:
            return None
        point: dict[Basic, Basic] = {s: value for s, value in values.items()}
        gap = add(mul(Integer(scale), add(log(self.to_expr(v)), -self.to_expr(total))), -self.to_expr(principal))
        measured = attempt(lambda: complex(gap.xreplace(point).evalf(40)), settings.timeout, native=True)
        if measured is None:
            return None
        turns = measured.imag / (2 * math.pi)
        if abs(measured.real) > 1e-12 or abs(turns - round(turns)) > 1e-12:
            return None
        constant = principal
        if round(turns) != 0:
            constant = constant + self._minus_one_logarithm().times(Fraction(2 * round(turns)))
        return self.normal(total + constant.times(Fraction(1, scale)))

    def _convex_region(self) -> bool:
        """Whether the assumptions are a conjunction of relations linear
        in the symbols (or there are none): a convex set, hence
        connected."""
        from sympy_extras.polys.polytools import Poly
        parts: list[Boolean] = []
        pending: list[Boolean] = self._facts()
        while pending:
            part = pending.pop()
            if isinstance(part, And):
                pending.extend(to_boolean(arg) for arg in part.args)
            else:
                parts.append(part)
        for part in parts:
            if isinstance(part, BooleanTrue):
                continue
            if not isinstance(part, (Gt, Ge, Lt, Le)):
                return False
            difference = add(part.lhs, -part.rhs)
            symbols = sorted(difference.free_symbols, key=lambda s: s.name)
            try:
                if symbols and Poly(difference, *symbols).total_degree() > 1:
                    return False
            except (BasePolynomialError, ValueError):
                return False
        return True

    def _pair_is_real(self, i: int) -> bool:
        """Whether ``z_i`` is real on the region: the argument of a
        logarithm positive, the argument of an exponential real."""
        pair = self.pairs[i]
        return self._positive(pair.e) if pair.logarithmic else self._real(pair.z)

    def _new_logarithm(self, v: Element, certified: bool) -> Element:
        certified = certified and self._sound()
        expression = log(self.to_expr(v))
        position = self._add_generator(Generator(LOGARITHM, self._new_name(), expression, v, certified))
        L = self._generator_element(position)
        self.pairs.append(_Pair(L, v, position, True))
        return L

    def _minus_one_logarithm(self) -> Element:
        """``log(-1) = I*pi``, the generator which carries ``pi``."""
        if self._minus_one < 0:
            v = self.constant(Fraction(-1))
            position = self._add_generator(Generator(LOGARITHM, self._new_name(), mul(I, pi), v))
            self.pairs.append(_Pair(self._generator_element(position), v, position, True))
            self._minus_one = position
        return self._generator_element(self._minus_one)

    def _prime_logarithm(self, p: int) -> Element:
        if p not in self._prime_logs:
            v = self.constant(Fraction(p))
            position = self._add_generator(Generator(LOGARITHM, self._new_name(), log(Integer(p)), v))
            self.pairs.append(_Pair(self._generator_element(position), v, position, True))
            self._prime_logs[p] = position
        return self._generator_element(self._prime_logs[p])

    def _number_logarithm(self, a: Element) -> Element:
        """The principal logarithm of a nonzero algebraic number ``a`` of
        the field: the logarithms of the primes for a positive rational,
        ``I*pi`` more for a negative one, a fraction of those when a power
        of the number is rational, a generator otherwise."""
        q = self._rational_of(a)
        if q is not None:
            if q < 0:
                return self.normal(self._minus_one_logarithm() + self._number_logarithm(self.constant(-q)))
            total = self.zero
            for prime, exponent in factorint(q.numerator).items():
                total = total + self._prime_logarithm(prime).times(Fraction(exponent))
            for prime, exponent in factorint(q.denominator).items():
                total = total - self._prime_logarithm(prime).times(Fraction(exponent))
            return self.normal(total)
        a = self.normal(a)
        number = self._algebraic_value(a)
        power = self.one
        for k in range(1, 2 * number.degree + 1):
            power = self.normal(power * a)
            r = self._rational_of(power)
            if r is not None:
                # k*log(value) = log(rational) + 2*pi*I*j for an integer j
                real = self._number_logarithm(self.constant(abs(r)))
                argument = complex(number)
                angle = k * math.atan2(argument.imag, argument.real) - (math.pi if r < 0 else 0.0)
                turns = round(angle / (2 * math.pi))
                extra = Fraction(2 * turns + (1 if r < 0 else 0))
                total = real + self._minus_one_logarithm().times(extra) if extra != 0 else real
                return self.normal(total.times(Fraction(1, k)))
        related = self._related_logarithm(a, number)
        if related is not None:
            return related
        certified = _moduli_differ(number.minpoly) and not any(
            g.kind == LOGARITHM and g.argument is not None and self._is_number(g.argument)
            and self._rational_of(g.argument) is None for g in self.generators)
        return self._new_logarithm(a, certified=certified)

    def _related_logarithm(self, a: Element, number_a: Algebraic) -> Optional[Element]:
        """The logarithm of an algebraic number through a multiplicative
        relation with the numbers whose logarithms are generators
        already: integers with ``value**n0 == u * prod(e_j**n_j)``, ``u`` a
        root of unity, are *found* numerically (an integer relation among
        the logarithms of the absolute values, by PSLQ) and *verified*
        exactly in the number field; the multiple of ``2*pi*I`` is told by
        the numerical value. Machin's formula is such a relation between
        ``5 + I``, ``239 - I``, ``1 + I`` and ``2``, ``13``."""
        candidates: list[tuple[int, Algebraic]] = []
        for j, pair in enumerate(self.pairs):
            if pair.logarithmic and j != self._pair_of(self._minus_one) and self._is_number(pair.e):
                candidates.append((j, self._algebraic_value(pair.e)))
        # the primes of the norm, whose logarithms a relation may need
        minimal = number_a.minpoly
        norm = Fraction(minimal[-1], minimal[0])
        for prime in sorted(set(factorint(abs(norm.numerator))) | set(factorint(norm.denominator))):
            if prime > 1 and prime not in self._prime_logs and len(candidates) < 12:
                self._prime_logarithm(prime)
                candidates.append((self._pair_of(self._prime_logs[prime]), Algebraic.from_rational(prime)))
        if not candidates:
            return None
        # one relation for the moduli and the arguments at once: a generic
        # real combination log|e| + t*arg(e), with 2*pi*t for the turns
        with mpmath.workdps(80):
            t = mpmath.sqrt(mpmath.mpf(3)) + mpmath.mpf(1) / 7

            def coordinate(number: Algebraic) -> mpmath.mpf:
                z = _approximation(number, 90)
                return mpmath.log(abs(z)) + t * mpmath.arg(z)

            vector = [coordinate(number_a)] + [coordinate(number) for _, number in candidates] + [2 * mpmath.pi * t]
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
            total = total - self.pairs[j].z.times(Fraction(int(n)))
        with mpmath.workdps(40):
            measured = complex(n0 * mpmath.log(_approximation(number_a, 45)) - complex(self.to_expr(total).evalf(40)))
        turns = measured.imag / (2 * math.pi)
        if abs(measured.real) > 1e-12 or abs(turns - round(turns)) > 1e-9:
            return None
        if round(turns) != 0:
            total = total + self._minus_one_logarithm().times(Fraction(2 * round(turns)))
        return self.normal(total.times(Fraction(1, n0)))

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
        return _moduli_differ(_algebraic(value).minpoly)

    def _is_number(self, a: Element) -> bool:
        """Whether ``a`` involves no generator."""
        a = self.normal(a)
        return a.num.is_ground and a.den.is_ground

    def _rational_of(self, a: Element) -> Optional[Fraction]:
        """The rational number which ``a`` is, ``None`` when it is not one."""
        a = self.normal(a)
        if not a.num.is_ground or not a.den.is_ground:
            return None
        c = self.domain.mul(a.num.LC, self.domain.invert(a.den.LC)) if a.num else []
        if len(c) > 1:
            return None
        return c[0] if c else Fraction(0)

    def _positive_number(self, a: Element) -> bool:
        """Whether the number ``a`` is a positive real number."""
        number = self._algebraic_value(a)
        return number.is_real and number > 0

    def _negative_number(self, a: Element) -> bool:
        number = self._algebraic_value(a)
        return number.is_real and number < 0

    def _algebraic_value(self, a: Element) -> Algebraic:
        """The algebraic number which the constant ``a`` is, from its
        coordinates in the primitive element."""
        a = self.normal(a)
        if not a.num.is_ground or not a.den.is_ground:
            raise ValueError("not a number of the field")
        c = self.domain.mul(a.num.LC, self.domain.invert(a.den.LC))
        theta = self._theta
        if theta is None or len(c) <= 1:
            return Algebraic.from_rational(c[0] if c else Fraction(0))
        return theta.polynomial_value(c)

    def _positive_constant_logarithm(self, m: Element) -> Optional[Element]:
        """The real logarithm of a positive constant of the field: of a
        number, or of a product of a number and powers of the ``e_j`` of
        constant pairs which are positive; ``None`` otherwise."""
        if self._is_number(m):
            return self._number_logarithm(m) if self._positive_number(m) else None
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
        scale = math.lcm(*[s.denominator for _, s in used]) if used else 1
        rest = m**scale
        total = self.zero
        for j, s in used:
            rest = rest * self.pairs[j].e**(-int(s * scale))
            total = total + self.pairs[j].z.times(s)
        rest = self.normal(rest)
        if not self._is_number(rest) or not self._positive_number(rest):
            return None
        return self.normal(total + self._number_logarithm(rest).times(Fraction(1, scale)))

    # ------------------------------------------------------------------
    # expressions in, expressions out

    def variable(self, symbol: Symbol) -> Element:
        """The generator of a free symbol."""
        if symbol not in self._variables:
            self._variables[symbol] = self._add_generator(Generator(VARIABLE, self._new_name(), symbol))
            self._derivatives.clear()
        return self._generator_element(self._variables[symbol])

    def element(self, expression: Expr) -> Element:
        """The element of the tower which ``expression`` is; the tower
        grows by the generators it needs.

        Raises :class:`NotElementary` for an expression outside the
        elementary functions."""
        if expression in self._cache:
            return self._cache[expression]
        value = self._element(expression)
        self._cache[expression] = value
        return value

    def _element(self, e: Expr) -> Element:
        if isinstance(e, Rational):
            return self.constant(_rational(e))
        if isinstance(e, Float) or e.has(Float):
            raise NotElementary("a floating point number: %s" % (e,))
        if isinstance(e, Symbol):
            return self.variable(e)
        if isinstance(e, Pi):
            return self.normal(self._minus_one_logarithm() * self._number(mul(Integer(-1), I)))
        if isinstance(e, Exp1):
            return self.exponential(self.one)
        if isinstance(e, ImaginaryUnit) or (e.is_number and e.is_algebraic and not e.has(Exp, Log, pi)):
            return self._number(e)
        if isinstance(e, Add):
            total = self.zero
            for term in e.args:
                total = total + self.element(as_expr(term))
            return total
        if isinstance(e, Mul):
            product = self.one
            for factor in e.args:
                product = product * self.element(as_expr(factor))
            return product
        if isinstance(e, Exp):
            return self.exponential(self.element(as_expr(e.args[0])))
        if isinstance(e, Log) and len(e.args) == 1:
            return self.logarithm(self.element(as_expr(e.args[0])))
        if isinstance(e, Pow):
            base, exponent = e.base, e.exp
            if isinstance(exponent, Integer):
                return self.element(base)**int(exponent.p)
            if isinstance(base, Exp1):
                return self.exponential(self.element(exponent))
            b = self.element(base)
            if self.vanishes(b):
                raise NotElementary("a power of zero: %s" % (e,))
            return self.exponential(self.element(exponent) * self.logarithm(b))
        if isinstance(e, _EXPONENTIAL_FORMS):
            rewritten = e.rewrite(Exp)
            if not rewritten.has(*_EXPONENTIAL_FORMS):
                return self.element(rewritten)
        if isinstance(e, _LOGARITHMIC_FORMS):
            rewritten = e.rewrite(Log)
            if not rewritten.has(*_LOGARITHMIC_FORMS):
                return self.element(rewritten)
        raise NotElementary("not an elementary expression this module knows: %s" % (e,))

    def _polynomial_expr(self, p: TowerPolynomial, meanings: Sequence[Expr]) -> Expr:
        """``p`` as an expression, its generators replaced by ``meanings``."""
        p = self._poly(p)
        terms: list[Expr] = []
        for monomial, coefficient in p.terms.items():
            factors = [self._coefficient_expr(coefficient)]
            for meaning, e in zip(meanings, monomial):
                if e:
                    factors.append(pow(meaning, Integer(e)))
            terms.append(mul(*factors))
        return add(*terms) if terms else Integer(0)

    def to_expr(self, a: Element) -> Expr:
        """The element as an expression of the core in the meanings of the
        generators."""
        a = self.normal(a)
        meanings = [g.expression for g in self.generators]
        return div(self._polynomial_expr(a.num, meanings), self._polynomial_expr(a.den, meanings))

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
        root_positions = [i for i, g in enumerate(self.generators) if g.kind == ROOT]
        factors: list[TowerPolynomial] = []
        vectors: list[dict[int, int]] = []
        for g in roots:
            if g.radicand is None:
                continue
            radicand = self.normal(g.radicand).num
            if any(radicand.degree_in(x) > 0 for x in root_positions):
                self._kummer = False
                return False
            vector: dict[int, int] = {}
            for factor, multiplicity in self._factor_list(radicand)[1]:
                monic = factor.monic()
                if monic not in factors:
                    factors.append(monic)
                vector[factors.index(monic)] = multiplicity
            vectors.append(vector)
        if not factors:
            self._kummer = False
            return False
        N = math.lcm(*[g.index for g in roots])
        width = len(factors)
        rows = [[(N // g.index) * vector.get(j, 0) for j in range(width)] for g, vector in zip(roots, vectors)]
        rows += [[N if i == j else 0 for j in range(width)] for i in range(width)]
        invariants = smith_normal_form(rows)
        index = 1
        for k in range(width):
            index *= invariants[k] if k < len(invariants) else 0
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
            if not g.certified and (a.num.degree_in(position) > 0 or a.den.degree_in(position) > 0):
                return False
        return self._roots_certified()

    @property
    def certified(self) -> bool:
        """Whether the generators are algebraically independent (the roots
        of full degree over the others): by the structure theorem for
        the functions, by Schanuel's conjecture for the constants."""
        return all(g.certified for g in self.generators) and self._roots_certified()


def _approximation(number: Algebraic, digits: int) -> mpmath.mpc:
    """A complex approximation of an algebraic number to ``digits`` digits
    (at the working precision of mpmath)."""
    box = number.enclosure(Fraction(1, 10**digits))
    re = (box[0] + box[1]) / 2
    im = (box[2] + box[3]) / 2
    return mpmath.mpc(mpmath.mpf(re.numerator) / re.denominator, mpmath.mpf(im.numerator) / im.denominator)


def _moduli_differ(minimal: list[int]) -> bool:
    """Whether two roots of the polynomial (integer coefficients, highest
    degree first) have different absolute values."""
    with mpmath.workdps(30):
        try:
            roots = mpmath.polyroots(minimal, maxsteps=200, extraprec=60)
        except mpmath.NoConvergence:
            return False
        moduli = sorted(float(abs(root)) for root in roots)
    return moduli[-1] - moduli[0] > 1e-12 * (1 + moduli[-1])


def canonical_form(expression: Expr, assumptions: Assumptions = None) -> Expr:
    """The canonical form of an elementary expression: a quotient of two
    polynomials in generators which the structure theorem proves
    independent (see the module documentation), the denominator monic.

    Raises :class:`NotElementary` when the expression is not elementary.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt, pi, I
    >>> from sympy_extras.functions import exp, log, sin, cos
    >>> from sympy_extras.simplify.structure import canonical_form
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


def is_zero(expression: Expr, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether an elementary expression vanishes identically on the
    region of the assumptions: ``True`` is a proof (the numerator of the
    canonical form is zero), ``False`` means that it does not vanish
    identically (the canonical form is not zero in a certified tower, or
    a sample point gives a value away from zero), ``None`` that neither
    was established.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import exp, log, cosh, sinh
    >>> from sympy_extras.simplify.structure import is_zero
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
    tower = ElementaryTower(assumptions)

    def decide() -> Optional[bool]:
        try:
            value = tower.element(expression)
            if tower.vanishes(value):
                return True
            return False if tower.certifies(value) else None
        except NotElementary:
            return None

    found = attempt(decide, settings.timeout, native=True)
    if found is not None:
        return found
    return False if _witness(expression, tower.assumptions) else None


def equal(a: Expr, b: Expr, assumptions: Assumptions = None) -> Optional[bool]:
    """Whether two elementary expressions are equal on the region of the
    assumptions, by :func:`is_zero` of their difference.

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.functions import tan, sin, cos
    >>> from sympy_extras.simplify.structure import equal
    >>> x = symbols('x')
    >>> equal(tan(x), sin(x)/cos(x)), equal(sqrt(x)*sqrt(x), x), equal(sin(x), x)
    (True, True, False)
    """
    return is_zero(add(a, -b), assumptions)


def _witness(e: Expr, assumptions: Assumptions) -> bool:
    """Whether ``e`` is clearly not zero at a sample point of the region."""
    if not settings.numerical_checks:
        return False
    from .structure_sympy import reliable_value_of_core, sample_values_of_core
    facts = _facts_of(assumptions)
    symbols = sorted(e.free_symbols, key=lambda s: s.name)
    # a symbol with no flag which the assumptions do not mention is a
    # complex number: real points of both signs, then complex ones; the
    # others are sampled where their flags and the assumptions put them
    mentioned: set[Symbol] = set()
    for fact in facts:
        mentioned |= fact.free_symbols
    unconstrained = [s for s in symbols if s not in mentioned and set(s.assumptions0) <= {'commutative'}]
    constrained = [s for s in symbols if s not in unconstrained]
    rng = random.Random(str(e))
    for attempt_number in range(6):
        values: Optional[dict[Symbol, Expr]] = sample_values_of_core(constrained, facts, rng) if constrained else {}
        if values is None:
            return False
        for s in unconstrained:
            real = number_from_fraction(Fraction(rng.randint(-300, 300), 100))
            values[s] = add(real, mul(I, number_from_fraction(Fraction(rng.randint(-1000, 1000), 100)))) \
                if attempt_number >= 3 else real
        for s in constrained:
            # the other sign of a real symbol, where its flags and the
            # assumptions allow it (the sampler tends to one side)
            if s.is_nonnegative is None and s.is_nonpositive is None and s.is_integer is None \
                    and values[s].is_extended_real and rng.random() < 0.5:
                flipped = dict(values)
                flipped[s] = -values[s]
                substitution: dict[Basic, Basic] = {k: v for k, v in flipped.items()}
                if all(isinstance(fact._xreplace_evaluated(substitution), BooleanTrue) for fact in facts):
                    values = flipped
        # a point where an argument lies on a branch cut on a side which
        # the rounding chooses is no witness (see reliable_form). The bug:
        # atan(x)**(1/3) and its form in logarithms, (I*(log(1 - I*x) -
        # log(1 + I*x))/2)**(1/3), are equal everywhere, and at x = -3 the
        # base of the second one is a negative number whose imaginary part
        # came out as 1e-41 of either sign: the difference evaluated to
        # 1.87*I and is_zero answered False (found by evaluating the
        # verdicts with Mathematica)
        point = values
        number = attempt(lambda: reliable_value_of_core(e, 40, point), settings.timeout, native=True)
        if number is None:
            continue
        if abs(number) > 1e-25:
            return True
    return False
