"""SymPy's ``Poly`` on the typed core: polynomials in generators with
coefficients in ``ZZ``, ``QQ``, ``ZZ[a, b, ...]`` or ``QQ[a, b, ...]``.

Ported from SymPy's ``sympy.polys.polytools`` and the expression
conversion of ``sympy.polys.polyutils``: the expression is expanded and
cut into terms; with explicit generators the factors free of them are
coefficients, without generators every factor which is not a number
(``I`` aside) becomes a generator (``sqrt(2)``, ``sin(z)``), sorted as
SymPy sorts them (``x, y, z`` first, then ``p`` to ``w``, then ``a`` to
``o``, then the rest by name); the domain is SymPy's ``construct_domain``
for coefficients which are polynomials in symbols. The representation is
a sparse polynomial over ``QQ`` in the generators followed by the
symbols of the domain; the gcd and the resultant run on the typed
recursive gcd (:mod:`~sympy_extras.polys.domains.sparsepolynomialring`),
the factorisation and the square-free decomposition of univariate
polynomials over ``ZZ`` and ``QQ`` on the typed dense layer. Printed as
SymPy prints ``Poly``.

Not ported yet (``CoercionFailed`` or ``OperationNotSupported``): the
domains ``EX`` (coefficients like ``exp(y)`` with explicit generators),
``RR``, fraction fields (``ZZ(a)``), ``ZZ_I`` and algebraic extensions
in ``Poly`` (``factor`` without generators factors over ``ZZ_I``/``QQ_I``,
:mod:`~sympy_extras.polys.gaussianfactor`, and ``RR``); the division of
multivariate polynomials and of polynomials with symbolic coefficients.
``cancel``, ``factor`` and ``invert`` of expressions which need such a
domain are computed natively all the same: ``cancel`` over ``CC`` (the gcd
over the exact ``QQ_I``, as SymPy's inexact domains do), over ``EX``
(numbers such as ``zoo`` among the coefficients) and with generators over
a composite domain (the other factors taken as elements of ``EX``);
``factor`` with generators over ``ZZ_I``/``QQ_I``/``RR`` by the
factorisation in every symbol, over ``CC`` and ``EX`` leaving the
polynomial as it is (SymPy's ``DomainError``); ``invert`` of numbers in
``ZZ_I``, ``QQ_I`` and ``RR`` by SymPy's ``Domain.invert``. The module
does not load SymPy.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.functions import sqrt, sin
>>> from sympy_extras.polys.polytools import Poly
>>> x, y, z, a, b = (Symbol(s) for s in 'xyzab')
>>> Poly(x**2*y + 2*y + 1, x, y)
Poly(x**2*y + 2*y + 1, x, y, domain='ZZ')
>>> Poly(x**2/2 + a*x + b, x)
Poly(1/2*x**2 + a*x + b, x, domain='QQ[a,b]')
>>> Poly(y*x + sin(z) + sqrt(2))
Poly(x*y + (sin(z)) + (sqrt(2)), x, y, sin(z), sqrt(2), domain='ZZ')
>>> f, g = Poly((x + 1)*(x - 2)**2, x), Poly((x - 2)*(x + 3), x)
>>> f.gcd(g), f.degree(), f.LC(), f.resultant(g)
(Poly(x - 2, x, domain='ZZ'), 3, 1, 0)
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/polytools.py (Poly), sympy/polys/polyutils.py
# (_sort_gens, _parallel_dict_from_expr),
# sympy/polys/constructor.py (construct_domain, the polynomial-ring case),
# sympy/printing/str.py (_print_Poly) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: a typed sparse representation over QQ with the domain label
# kept beside it; the domains EX, RR, fraction fields, ZZ_I and
# algebraic extensions, the multivariate division and factorisation not
# ported yet.
# ---------------------------------------------------------------------

from __future__ import annotations

import re
from fractions import Fraction
from math import gcd as integer_gcd
from typing import Final, Literal, Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.exprtools import decompose_power
from sympy_extras.core.function import expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from builtins import pow as builtins_pow

from . import densearith, densebasic, densetools, euclidtools, factortools, sqfreetools
from .domains import QQ, ZZ
from .domains.gaussiandomains import QQ_I, ZZ_I, GaussianInteger, GaussianRational
from .domains.sparsepolynomialring import (IntegerPolynomialRing, IteratedPolynomialRing, RationalPolynomialRing,
                                           SparsePolynomialRing)
from .polyerrors import (CoercionFailed, GeneratorsError, GeneratorsNeeded, OperationNotSupported, PolynomialDivisionFailed,
                         PolynomialError)
from .polyutils import dict_from_expr
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['Poly', 'poly', 'degree', 'LC', 'gcd', 'lcm', 'gcd_list', 'lcm_list', 'resultant', 'discriminant', 'factor_list', 'sqf_list', 'cancel',
           'cancel_fraction', 'factor', 'factor_list_fraction', 'sqf', 'sqf_list_fraction']

PolyLike = Union['Poly', ExprLike]

# ---------------------------------------------------------------------
# the generators

# PORTED FROM SYMPY: sympy/polys/polyutils.py:_gens_order
_GENS_ORDER: dict[str, int] = {
    'a': 301, 'b': 302, 'c': 303, 'd': 304, 'e': 305, 'f': 306, 'g': 307, 'h': 308, 'i': 309, 'j': 310, 'k': 311,
    'l': 312, 'm': 313, 'n': 314, 'o': 315, 'p': 216, 'q': 217, 'r': 218, 's': 219, 't': 220, 'u': 221, 'v': 222,
    'w': 223, 'x': 124, 'y': 125, 'z': 126}
_MAX_ORDER = 1000
_RE_GEN = re.compile(r'^(.*?)(\d*)$')


# PORTED FROM SYMPY: sympy/polys/polyutils.py:_sort_gens
def _gen_key(gen: Expr) -> tuple[int, str, int, int]:
    match = _RE_GEN.match(str(gen))
    name, index = (match.group(1), match.group(2)) if match else (str(gen), '')
    # SymPy's dummies print with their number (Dummy_34), ours do not: the
    # creation index orders them as SymPy's names do
    from sympy_extras.core.symbol import Dummy
    created = gen.dummy_index if isinstance(gen, Dummy) else -1
    return (_GENS_ORDER.get(name, _MAX_ORDER), name, int(index) if index else 0, created)


def sort_gens(gens: Sequence[Expr]) -> tuple[Expr, ...]:
    """The generators in SymPy's order (``_sort_gens``)."""
    return tuple(sorted(gens, key=_gen_key))


def _expanded(e: Expr) -> Expr:
    """SymPy's ``_dict_from_expr`` expansion: ``expand()`` with its
    default hints (products, powers and logarithms)."""
    return expand(e)


def _factors(e: Expr) -> tuple[Expr, ...]:
    return e.args if isinstance(e, Mul) else (e,)


def _terms_of(e: Expr) -> tuple[Expr, ...]:
    return e.args if isinstance(e, Add) else (e,)


Decomposed = dict[Monomial, list[Expr]]


# PORTED FROM SYMPY: sympy/polys/polyutils.py:_parallel_dict_from_expr_if_gens
def _decompose_with_gens(e: Expr, gens: Sequence[Expr]) -> Decomposed:
    index = {g: i for i, g in enumerate(gens)}
    result: Decomposed = {}
    for term in _terms_of(_expanded(e)):
        coefficient: list[Expr] = []
        monomial = [0] * len(gens)
        for factor in _factors(term):
            if isinstance(factor, Number):
                coefficient.append(factor)
                continue
            base, exponent = decompose_power(factor)
            if exponent < 0:
                exponent, base = -exponent, pow(base, Integer(-1))
            if base in index:
                monomial[index[base]] = exponent
            elif not any(factor.has(g) for g in gens):
                coefficient.append(factor)
            else:
                raise PolynomialError('%s contains an element of the set of generators.' % (factor,))
        result.setdefault(tuple(monomial), []).append(mul(*coefficient))
    return result


# PORTED FROM SYMPY: sympy/polys/polyutils.py:_parallel_dict_from_expr_no_gens
def _decompose_without_gens_many(exprs: Sequence[Expr], expand: bool = True
                                 ) -> tuple[list[Decomposed], tuple[Expr, ...]]:
    found: set[Expr] = set()
    raws: list[list[tuple[list[Expr], dict[Expr, int]]]] = []
    for e in exprs:
        raw: list[tuple[list[Expr], dict[Expr, int]]] = []
        for term in _terms_of(_expanded(e) if expand else e):
            coefficient: list[Expr] = []
            elements: dict[Expr, int] = {}
            for factor in _factors(term):
                if isinstance(factor, Number) or factor == I:
                    coefficient.append(factor)
                    continue
                base, exponent = decompose_power(factor)
                if exponent < 0:
                    exponent, base = -exponent, pow(base, Integer(-1))
                elements[base] = elements.get(base, 0) + exponent
                found.add(base)
            raw.append((coefficient, elements))
        raws.append(raw)
    gens = sort_gens(list(found))
    index = {g: i for i, g in enumerate(gens)}
    results: list[Decomposed] = []
    for raw in raws:
        result: Decomposed = {}
        for coefficient, elements in raw:
            monomial = [0] * len(gens)
            for base, exponent in elements.items():
                monomial[index[base]] = exponent
            result.setdefault(tuple(monomial), []).append(mul(*coefficient))
        results.append(result)
    return results, gens


def _decompose_without_gens(e: Expr) -> tuple[Decomposed, tuple[Expr, ...]]:
    decomposed, gens = _decompose_without_gens_many([e])
    return decomposed[0], gens


# ---------------------------------------------------------------------
# the domain

class _Domain:
    """SymPy's domain of a ``Poly`` restricted to ``ZZ``, ``QQ``,
    ``ZZ[params]``, ``QQ[params]``: integral or rational coefficients,
    polynomial in the symbols ``params``."""

    __slots__ = ('integral', 'params')

    def __init__(self, integral: bool, params: tuple[Symbol, ...]) -> None:
        self.integral: Final[bool] = integral
        self.params: Final[tuple[Symbol, ...]] = params

    @property
    def name(self) -> str:
        ground = 'ZZ' if self.integral else 'QQ'
        if not self.params:
            return ground
        return '%s[%s]' % (ground, ','.join(str(p) for p in self.params))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Domain) and other.integral == self.integral and other.params == self.params

    def __hash__(self) -> int:
        return hash(('_Domain', self.integral, self.params))


def _unify_domains(a: _Domain, b: _Domain) -> _Domain:
    params = sort_gens(list(dict.fromkeys(a.params + b.params)))
    symbols = tuple(p for p in params if isinstance(p, Symbol))
    return _Domain(a.integral and b.integral, symbols)


# PORTED FROM SYMPY: sympy/polys/constructor.py:construct_domain (rationals and polynomial rings)
def _construct_domain(coefficients: Sequence[Expr]) -> _Domain:
    symbols: set[Expr] = set()
    for c in coefficients:
        if c.has(I):
            raise CoercionFailed('the domain ZZ_I of %s is not ported yet' % (c,))
        for s in c.free_symbols:
            symbols.add(s)
    params = tuple(p for p in sort_gens(list(symbols)) if isinstance(p, Symbol))
    names = [p.name for p in params]
    integral = True
    for c in coefficients:
        if names:
            try:
                terms = dict_from_expr(c, names, QQ)
            except ValueError:
                raise CoercionFailed('the domain of %s (EX or a fraction field) is not ported yet' % (c,))
            values: list[Fraction] = list(terms.values())
        else:
            if not isinstance(c, Rational):
                raise CoercionFailed('the domain of %s (EX, RR or an extension) is not ported yet' % (c,))
            values = [c.as_fraction()]
        if any(v.denominator != 1 for v in values):
            integral = False
    return _Domain(integral, params)


# ---------------------------------------------------------------------
# Poly

def _internal_names(n_gens: int, n_params: int) -> list[str]:
    return ['_g%d' % i for i in range(n_gens)] + ['_p%d' % i for i in range(n_params)]


# PORTED FROM SYMPY: sympy/polys/polytools.py:Poly
class Poly:
    """A polynomial in ``gens`` with coefficients in a domain (``ZZ``,
    ``QQ``, ``ZZ[a, b]``, ``QQ[a, b]``)."""

    __slots__ = ('_rep', '_gens', '_domain')

    def __init__(self, expr: PolyLike, *gens: ExprLike) -> None:
        if isinstance(expr, Poly):
            if gens and tuple(to_expr(g) for g in gens) != expr.gens:
                expr = expr.as_expr()
            else:
                self._rep: PolyElement[Fraction] = expr._rep
                self._gens: tuple[Expr, ...] = expr._gens
                self._domain: _Domain = expr._domain
                return
        e = to_expr(expr)
        if e.is_commutative is False:
            raise PolynomialError('non-commutative expressions are not supported')
        if gens:
            generators = tuple(to_expr(g) for g in gens)
            if len(set(generators)) != len(generators):
                raise GeneratorsError('duplicated generators: %s' % (generators,))
            decomposed = _decompose_with_gens(e, generators)
        else:
            decomposed, generators = _decompose_without_gens(e)
            if not generators:
                raise GeneratorsNeeded('specify generators to give %s a polynomial representation' % (e,))
        coefficients = {m: add(*cs) for m, cs in decomposed.items()}
        domain = _construct_domain(list(coefficients.values()))
        self._gens = generators
        self._domain = domain
        self._rep = _rep_from_coefficients(coefficients, generators, domain)

    @classmethod
    def _new(cls, rep: PolyElement[Fraction], gens: tuple[Expr, ...], domain: _Domain) -> Poly:
        """The polynomial of a representation (its ring having the
        internal names of ``gens`` and the domain's parameters)."""
        obj = cls.__new__(cls)
        obj._rep = rep
        obj._gens = gens
        obj._domain = domain
        return obj

    @classmethod
    def from_dict(cls, terms: dict[Monomial, Union[int, Fraction]], *gens: ExprLike) -> Poly:
        """The polynomial with the rational coefficients ``terms`` (a
        monomial in ``gens`` to its coefficient) over ``ZZ`` when they are
        integers, ``QQ`` otherwise (SymPy's ``Poly.from_dict``)."""
        generators = tuple(to_expr(g) for g in gens)
        if not generators:
            raise GeneratorsNeeded("generators are needed to build a polynomial from a dictionary")
        if len(set(generators)) != len(generators):
            raise GeneratorsError('duplicated generators: %s' % (generators,))
        values: dict[Monomial, Fraction] = {}
        for m, c in terms.items():
            if len(m) != len(generators):
                raise PolynomialError('the monomial %s does not match the generators %s' % (m, generators))
            if c:
                values[m] = values.get(m, Fraction(0)) + Fraction(c)
        ring = PolyRing(QQ, _internal_names(len(generators), 0))
        integral = all(c.denominator == 1 for c in values.values())
        return cls._new(ring.from_dict(values), generators, _Domain(integral, ()))

    @property
    def rep(self) -> PolyElement[Fraction]:
        """The representation: a sparse polynomial over ``QQ`` in the
        generators followed by the parameters of the domain (internal
        variable names)."""
        return self._rep

    @property
    def is_integral(self) -> bool:
        """Whether the domain is ``ZZ`` or ``ZZ[params]``."""
        return self._domain.integral

    # -- structure

    @property
    def gens(self) -> tuple[Expr, ...]:
        return self._gens

    @property
    def gen(self) -> Expr:
        return self._gens[0]

    @property
    def domain(self) -> str:
        """The name of the domain (SymPy's ``Poly.domain``, printed)."""
        return self._domain.name

    def get_domain(self) -> str:
        return self._domain.name

    @property
    def free_symbols(self) -> set[Basic]:
        symbols: set[Basic] = set()
        for g in self._gens:
            symbols |= g.free_symbols
        symbols |= set(self._domain.params)
        return symbols

    @property
    def is_univariate(self) -> bool:
        return len(self._gens) == 1

    @property
    def is_multivariate(self) -> bool:
        return len(self._gens) > 1

    def _split(self) -> dict[Monomial, dict[Monomial, Fraction]]:
        """The terms grouped by the monomial in the generators, each with
        its coefficient as terms in the domain's parameters."""
        n = len(self._gens)
        grouped: dict[Monomial, dict[Monomial, Fraction]] = {}
        for m, c in self._rep.terms.items():
            grouped.setdefault(m[:n], {})[m[n:]] = c
        return grouped

    def _coefficient_expr(self, terms: dict[Monomial, Fraction]) -> Expr:
        parts: list[Expr] = []
        for m, c in terms.items():
            factors: list[Expr] = [number_from_fraction(c)]
            for p, k in zip(self._domain.params, m):
                if k:
                    factors.append(pow(p, Integer(k)))
            parts.append(mul(*factors))
        return add(*parts)

    def terms(self) -> list[tuple[Monomial, Expr]]:
        """The terms ``(monomial, coefficient)`` in lex order, the
        coefficients as expressions."""
        grouped = self._split()
        return [(m, self._coefficient_expr(grouped[m])) for m in sorted(grouped, reverse=True)]

    def monoms(self) -> list[Monomial]:
        return [m for m, _ in self.terms()] or [tuple(0 for _ in self._gens)]

    def coeffs(self) -> list[Expr]:
        return [c for _, c in self.terms()] or [Integer(0)]

    def as_dict(self) -> dict[Monomial, Expr]:
        return dict(self.terms())

    def as_expr(self) -> Expr:
        parts: list[Expr] = []
        for m, c in self.terms():
            factors = [c] + [pow(g, Integer(k)) for g, k in zip(self._gens, m) if k]
            parts.append(mul(*factors))
        return add(*parts)

    def LC(self) -> Expr:
        return self.coeffs()[0]

    def LM(self) -> Monomial:
        return self.monoms()[0]

    def LT(self) -> tuple[Monomial, Expr]:
        return (self.LM(), self.LC())

    def TC(self) -> Expr:
        coefficients = self.as_dict()
        return coefficients.get(tuple(0 for _ in self._gens), Integer(0))

    def coeff_monomial(self, monomial: Monomial) -> Expr:
        return self.as_dict().get(monomial, Integer(0))

    def nth(self, *exponents: int) -> Expr:
        return self.coeff_monomial(tuple(exponents))

    def all_coeffs(self) -> list[Expr]:
        """The coefficients of a univariate polynomial, from the leading one
        down to the constant, zeros included."""
        if len(self._gens) != 1:
            raise PolynomialError('multivariate polynomials not supported')
        coefficients = self.as_dict()
        n = self.degree()
        if n < 0:
            return [Integer(0)]
        return [coefficients.get((n - i,), Integer(0)) for i in range(n + 1)]

    def _gen_index(self, gen: Union[int, ExprLike]) -> int:
        if isinstance(gen, int):
            if not -len(self._gens) <= gen < len(self._gens):
                raise PolynomialError('-%s <= gen < %s expected, got %s' % (len(self._gens), len(self._gens), gen))
            return gen % len(self._gens)
        g = to_expr(gen)
        if g in self._gens:
            return self._gens.index(g)
        raise PolynomialError('a valid generator expected, got %s' % (g,))

    def degree(self, gen: Union[int, ExprLike] = 0) -> int:
        """The degree in ``gen`` (``-1`` for the zero polynomial; SymPy
        gives ``-oo``)."""
        i = self._gen_index(gen)
        if not self._rep.terms:
            return -1
        return max(m[i] for m in self._rep.terms)

    def degree_list(self) -> tuple[int, ...]:
        return tuple(self.degree(i) for i in range(len(self._gens)))

    def total_degree(self) -> int:
        n = len(self._gens)
        if not self._rep.terms:
            return -1
        return max(sum(m[:n]) for m in self._rep.terms)

    def length(self) -> int:
        return len(self._split())

    @property
    def is_zero(self) -> bool:
        return not self._rep.terms

    @property
    def is_one(self) -> bool:
        return self._rep == self._rep.ring.one

    @property
    def is_ground(self) -> bool:
        return all(not any(m) for m in self._split())

    @property
    def is_monic(self) -> bool:
        return self.LC() == Integer(1)

    @property
    def is_linear(self) -> bool:
        return all(sum(m) <= 1 for m in self._split())

    @property
    def is_quadratic(self) -> bool:
        return all(sum(m) <= 2 for m in self._split())

    # -- printing

    # PORTED FROM SYMPY: sympy/printing/str.py:StrPrinter._print_Poly
    def __str__(self) -> str:
        def atom(e: Expr) -> str:
            text = str(e)
            return text if isinstance(e, (Symbol, Number)) else '(%s)' % text
        gens = [atom(g) for g in self._gens]
        pieces: list[str] = []
        for monomial, coefficient in self.terms():
            monomial_text = '*'.join(gens[i] if e == 1 else '%s**%d' % (gens[i], e)
                                     for i, e in enumerate(monomial) if e > 0)
            if isinstance(coefficient, Add):
                text = '(%s)' % coefficient if monomial_text else str(coefficient)
            else:
                if monomial_text and coefficient == Integer(1):
                    pieces.extend(['+', monomial_text])
                    continue
                if monomial_text and coefficient == Integer(-1):
                    pieces.extend(['-', monomial_text])
                    continue
                text = _print_coefficient(coefficient)
            term = text + '*' + monomial_text if monomial_text else text
            if term.startswith('-'):
                pieces.extend(['-', term[1:]])
            else:
                pieces.extend(['+', term])
        if not pieces:
            pieces = ['+', '0']
        if pieces[0] in ('-', '+'):
            sign = pieces.pop(0)
            if sign == '-':
                pieces[0] = '-' + pieces[0]
        names = [g[1:-1] if len(g) > 2 and g.startswith('(') and g.endswith(')') else g for g in gens]
        return "Poly(%s, %s, domain='%s')" % (' '.join(pieces), ', '.join(names), self._domain.name)

    __repr__ = __str__

    # -- comparison

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Poly):
            if self._gens != other._gens:
                return False
            if self._domain.params == other._domain.params:
                # one ring: the representations are equal exactly when the
                # polynomials are (whatever ZZ or QQ says)
                return self._rep.terms == other._rep.terms
            return self.as_expr() == other.as_expr()
        return NotImplemented

    def __hash__(self) -> int:
        # the monomials in the generators which have a nonzero
        # coefficient, whatever the parameters of the domain
        n = len(self._gens)
        return hash(('Poly', self._gens, frozenset(m[:n] for m in self._rep.terms)))

    # -- arithmetic

    def _unified(self, other: PolyLike) -> tuple[Poly, Poly]:
        """``self`` and ``other`` over the same generators and domain."""
        if not isinstance(other, Poly):
            other = Poly(other, *self._gens)
        if other._gens != self._gens:
            gens = tuple(dict.fromkeys(self._gens + other._gens))
            return Poly(self.as_expr(), *gens), Poly(other.as_expr(), *gens)
        if other._domain == self._domain:
            return self, other
        domain = _unify_domains(self._domain, other._domain)
        return self._converted(domain), other._converted(domain)

    def _converted(self, domain: _Domain) -> Poly:
        """The polynomial over a larger domain (more parameters, or ``QQ``
        for ``ZZ``)."""
        n = len(self._gens)
        position = {p: i for i, p in enumerate(domain.params)}
        ring = PolyRing(QQ, _internal_names(n, len(domain.params)))
        terms: dict[Monomial, Fraction] = {}
        for m, c in self._rep.terms.items():
            exponents = [0] * len(domain.params)
            for p, k in zip(self._domain.params, m[n:]):
                exponents[position[p]] = k
            terms[m[:n] + tuple(exponents)] = c
        return Poly._new(ring.from_dict(terms), self._gens, domain)

    def _same_ring(self, rep: PolyElement[Fraction], integral: Optional[bool] = None) -> Poly:
        domain = self._domain if integral is None else _Domain(integral, self._domain.params)
        return Poly._new(rep, self._gens, domain)

    def __neg__(self) -> Poly:
        return self._same_ring(-self._rep)

    def __add__(self, other: PolyLike) -> Poly:
        a, b = self._unified(other)
        return a._same_ring(a._rep + b._rep)

    def __radd__(self, other: PolyLike) -> Poly:
        return self.__add__(other)

    def __sub__(self, other: PolyLike) -> Poly:
        a, b = self._unified(other)
        return a._same_ring(a._rep - b._rep)

    def __rsub__(self, other: PolyLike) -> Poly:
        return (-self).__add__(other)

    def __mul__(self, other: PolyLike) -> Poly:
        a, b = self._unified(other)
        return a._same_ring(a._rep * b._rep)

    def __rmul__(self, other: PolyLike) -> Poly:
        return self.__mul__(other)

    def __pow__(self, n: int) -> Poly:
        if n < 0:
            raise PolynomialError('negative powers of a polynomial')
        return self._same_ring(self._rep ** n)

    def diff(self, gen: Union[int, ExprLike] = 0) -> Poly:
        return self._same_ring(self._rep.diff(self._gen_index(gen)))

    def eval(self, gen: Union[int, ExprLike], value: ExprLike) -> Union[Poly, Expr]:
        """The polynomial with ``gen`` set to ``value``: a polynomial in the
        other generators, or an expression when none is left."""
        i = self._gen_index(gen)
        point = to_expr(value)
        values = [point if j == i else g for j, g in enumerate(self._gens)]
        from sympy_extras.core.basic import Basic as _Basic
        substitution: dict[_Basic, _Basic] = {g: v for g, v in zip(self._gens, values) if g != v}
        result = self.as_expr()._xreplace_evaluated(substitution)
        if not isinstance(result, Expr):
            raise TypeError("an expression is expected")
        rest = self._gens[:i] + self._gens[i + 1:]
        result = _expanded(result)
        if not rest:
            return result
        return Poly(result, *rest)

    # -- algorithms on the representation

    def _integer_rep(self, R: SparsePolynomialRing[int]) -> PolyElement[int]:
        return R.ring.from_dict({m: int(c) for m, c in self._rep.terms.items()})

    def _rational_rep(self, R: SparsePolynomialRing[Fraction]) -> PolyElement[Fraction]:
        return R.ring.from_dict(dict(self._rep.terms))

    def _from_integers(self, f: PolyElement[int]) -> Poly:
        return self._same_ring(self._rep.ring.from_dict({m: Fraction(c) for m, c in f.terms.items()}))

    def _from_rationals(self, f: PolyElement[Fraction]) -> Poly:
        return self._same_ring(self._rep.ring.from_dict(dict(f.terms)))

    def _integer_ring(self) -> SparsePolynomialRing[int]:
        names = _internal_names(len(self._gens), len(self._domain.params))
        if not self._domain.params:
            return IntegerPolynomialRing(names)
        return IteratedPolynomialRing(IntegerPolynomialRing(names[len(self._gens):]), names[:len(self._gens)])

    def _rational_ring(self) -> SparsePolynomialRing[Fraction]:
        names = _internal_names(len(self._gens), len(self._domain.params))
        if not self._domain.params:
            return RationalPolynomialRing(names)
        return IteratedPolynomialRing(RationalPolynomialRing(names[len(self._gens):]), names[:len(self._gens)])

    # PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_inner_gcd (the domains ZZ, QQ, ZZ[y], QQ[y])
    def gcd(self, other: PolyLike) -> Poly:
        """SymPy's gcd: positive leading coefficient over ``ZZ``, monic over
        ``QQ``, ``dmp_rr_prs_gcd`` over ``ZZ[y]`` and ``QQ[y]`` (contents
        by the gcd of the parameter ring)."""
        a, b = self._unified(other)
        if a._domain.integral:
            Z = a._integer_ring()
            return a._from_integers(Z.gcd(a._integer_rep(Z), b._integer_rep(Z)))
        Q = a._rational_ring()
        g = Q.gcd(a._rational_rep(Q), b._rational_rep(Q))
        if not a._domain.params and g.terms:
            g = g.exquo_ground(g.LC)
        return a._from_rationals(g)

    # PORTED FROM SYMPY: sympy/polys/euclidtools.py:dup_rr_lcm, dmp_rr_lcm, dup_ff_lcm, dmp_ff_lcm
    def lcm(self, other: PolyLike) -> Poly:
        a, b = self._unified(other)
        if a.is_zero or b.is_zero:
            return a._same_ring(a._rep.ring.zero)
        if a._domain.params:
            if a._domain.integral:
                Z = a._integer_ring()
                return a._from_integers(Z.lcm(a._integer_rep(Z), b._integer_rep(Z)))
            Q = a._rational_ring()
            return a._from_rationals(Q.lcm(a._rational_rep(Q), b._rational_rep(Q)))
        if not a._domain.integral:
            h = (a._rep * b._rep).exquo(a.gcd(b)._rep)
            return a._same_ring(h.exquo_ground(h.LC))
        fc, gc = _terms_gcd(list(a._rep.terms.values())), _terms_gcd(list(b._rep.terms.values()))
        f, g = a._rep.exquo_ground(Fraction(fc)), b._rep.exquo_ground(Fraction(gc))
        c = Fraction(fc * gc // integer_gcd(fc, gc))
        h = (f * g).exquo(a._same_ring(f).gcd(a._same_ring(g))._rep)
        if len(a._gens) == 1 and h.LC < 0:
            c = -c
        return a._same_ring(h.mul_ground(c))

    def cofactors(self, other: PolyLike) -> tuple[Poly, Poly, Poly]:
        a, b = self._unified(other)
        g = a.gcd(b)
        if g.is_zero:
            return g, a, b
        return g, a._same_ring(a._rep.exquo(g._rep)), b._same_ring(b._rep.exquo(g._rep))

    def monic(self) -> Poly:
        if self.is_zero:
            return self
        lc = self._rep.LC
        if self._domain.params:
            lead = self._split()[max(self._split())]
            if list(lead) != [tuple(0 for _ in self._domain.params)]:
                raise CoercionFailed('monic of %s needs the fraction field (not ported yet)' % (self,))
        return self._same_ring(self._rep.exquo_ground(lc), integral=False if lc != 1 else None)

    def content(self) -> Expr:
        if self._domain.params:
            raise OperationNotSupported('the content over a polynomial domain is not ported yet')
        values = list(self._rep.terms.values())
        if not values:
            return Integer(0)
        if self._domain.integral:
            return Integer(_terms_gcd(values))
        return Integer(1) if values else Integer(0)

    def primitive(self) -> tuple[Expr, Poly]:
        c = self.content()
        if not isinstance(c, Rational) or c.p in (0, 1):
            return c, self
        return c, self._same_ring(self._rep.exquo_ground(c.as_fraction()))

    def _univariate_rational(self) -> list[Fraction]:
        if len(self._gens) != 1 or self._domain.params:
            raise OperationNotSupported('only univariate polynomials over ZZ or QQ (not yet %s)' % (self,))
        return densebasic.dup_from_dict({m[0]: c for m, c in self._rep.terms.items()}, QQ)

    def _from_univariate(self, f: Sequence[Fraction], integral: Optional[bool] = None) -> Poly:
        n = len(f) - 1
        return self._same_ring(self._rep.ring.from_dict({(n - i,): c for i, c in enumerate(f) if c}), integral)

    def div(self, other: PolyLike) -> tuple[Poly, Poly]:
        """Division with remainder of univariate polynomials (SymPy's
        ``div(auto=True)``: over ``QQ`` when the coefficients are integers;
        over a polynomial domain only when the leading coefficient of the
        divisor is a number, SymPy's fraction field ``ZZ(a)`` not being
        ported)."""
        a, b = self._unified(other)
        if b.is_zero:
            raise ZeroDivisionError('polynomial division by zero')
        if a._domain.params:
            return a._div_over_parameters(b)
        q, r = densearith.dup_div(a._univariate_rational(), b._univariate_rational(), QQ)
        integral = a._domain.integral and all(c.denominator == 1 for c in q + r)
        return a._from_univariate(q, integral), a._from_univariate(r, integral)

    def _div_over_parameters(self, b: Poly) -> tuple[Poly, Poly]:
        if len(self._gens) != 1:
            raise OperationNotSupported('division of multivariate polynomials is not ported yet')
        divisor = b._split()
        top = max(m[0] for m in divisor)
        lead = divisor[(top,)]
        if list(lead) != [tuple(0 for _ in self._domain.params)]:
            raise CoercionFailed('division by %s needs the fraction field (not ported yet)' % (b,))
        lc = lead[tuple(0 for _ in self._domain.params)]
        remainder = {m[0]: dict(c) for m, c in self._split().items()}
        quotient: dict[int, dict[Monomial, Fraction]] = {}
        while remainder and max(remainder) >= top:
            d = max(remainder)
            factor_terms = {pm: c / lc for pm, c in remainder[d].items()}
            quotient[d - top] = factor_terms
            for (e,), coefficients in divisor.items():
                target = remainder.setdefault(e + d - top, {})
                for pm, c in coefficients.items():
                    for qm, qc in factor_terms.items():
                        key = tuple(i + j for i, j in zip(pm, qm))
                        target[key] = target.get(key, Fraction(0)) - c * qc
                        if not target[key]:
                            del target[key]
                if not target:
                    del remainder[e + d - top]
        integral = self._domain.integral and all(c.denominator == 1 for cs in list(quotient.values()) +
                                                 list(remainder.values()) for c in cs.values())
        return (self._from_grouped(quotient, integral), self._from_grouped(remainder, integral))

    def _from_grouped(self, grouped: dict[int, dict[Monomial, Fraction]], integral: bool) -> Poly:
        terms: dict[Monomial, Fraction] = {}
        for d, coefficients in grouped.items():
            for pm, c in coefficients.items():
                if c:
                    terms[(d,) + pm] = c
        return self._same_ring(self._rep.ring.from_dict(terms), integral)

    # PORTED FROM SYMPY: sympy/polys/euclidtools.py:dmp_cancel, sympy/polys/polytools.py:Poly.cancel
    def cancel(self, other: PolyLike) -> tuple[Expr, Poly, Poly]:
        """``(c, p, q)`` with ``self/other = c p/q`` in lowest terms (SymPy's
        ``Poly.cancel``): over ``QQ`` the denominators are cleared into the
        constant ``c``; the signs of the ground leading coefficients of
        ``p`` and ``q`` are made positive, a single minus sign going into
        ``c``."""
        a, b = self._unified(other)
        if b.is_zero:
            raise ZeroDivisionError('cancel by a zero polynomial')
        cp, cq = 1, 1
        f, g = a, b
        if not a._domain.integral and not a._domain.params:
            cq = _denominators_lcm(a._rep)
            cp = _denominators_lcm(b._rep)
            f = a._same_ring(a._rep.mul_ground(Fraction(cq)), True)
            g = b._same_ring(b._rep.mul_ground(Fraction(cp)), True)
        h = f.gcd(g)
        p = f._same_ring(f._rep.exquo(h._rep), a._domain.integral)
        q = g._same_ring(g._rep.exquo(h._rep), a._domain.integral)
        common = integer_gcd(cp, cq)
        cp, cq = cp // common, cq // common
        p_neg, q_neg = p._ground_negative(), q._ground_negative()
        if p_neg and q_neg:
            p, q = -p, -q
        elif p_neg:
            cp, p = -cp, -p
        elif q_neg:
            cp, q = -cp, -q
        return (number_from_fraction(Fraction(cp, cq)), p, q)

    def _ground_negative(self) -> bool:
        """Whether the ground leading coefficient (lex in all the variables,
        SymPy's ``dmp_ground_LC``) is negative."""
        return bool(self._rep.terms) and self._rep.LC < 0

    def quo(self, other: PolyLike) -> Poly:
        return self.div(other)[0]

    def rem(self, other: PolyLike) -> Poly:
        return self.div(other)[1]

    def exquo(self, other: PolyLike) -> Poly:
        a, b = self._unified(other)
        try:
            return a._same_ring(a._rep.exquo(b._rep))
        except ValueError:
            from .polyerrors import ExactQuotientFailed
            raise ExactQuotientFailed('%s does not divide %s' % (b, a))

    def resultant(self, other: PolyLike) -> Union[Poly, Expr]:
        """The resultant in the first generator: an expression for a
        univariate polynomial, a polynomial in the other generators
        otherwise."""
        a, b = self._unified(other)
        n = len(a._gens)
        names = _internal_names(n, len(a._domain.params))
        value = _resultant_first(a._rep, b._rep, names, a._domain.integral)
        if n == 1:
            return a._coefficient_expr({m[1:]: c for m, c in value.terms.items()})
        rest = Poly._new(PolyRing(QQ, _internal_names(n - 1, len(a._domain.params))).from_dict(
            {m[1:]: c for m, c in value.terms.items()}), a._gens[1:], a._domain)
        return rest

    def discriminant(self) -> Union[Poly, Expr]:
        """``(-1)**(n (n - 1)/2) Res(f, f')/lc(f)`` in the first generator,
        the division exact on the representation (the leading coefficient
        may be a polynomial in the other generators)."""
        n = self.degree()
        if n < 1:
            return Integer(0)
        k = len(self._gens)
        names = _internal_names(k, len(self._domain.params))
        value = _resultant_first(self._rep, self._rep.diff(0), names, self._domain.integral)
        lc = self._rep.ring.from_dict({(0,) + m[1:]: c for m, c in self._rep.terms.items() if m[0] == n})
        value = value.exquo(lc)
        if (n * (n - 1) // 2) % 2:
            value = -value
        if k == 1:
            return self._coefficient_expr({m[1:]: c for m, c in value.terms.items()})
        return Poly._new(PolyRing(QQ, _internal_names(k - 1, len(self._domain.params))).from_dict(
            {m[1:]: c for m, c in value.terms.items()}), self._gens[1:], self._domain)

    # PORTED FROM SYMPY: sympy/polys/sqfreetools.py:dup_sqf_list, dmp_sqf_list
    def sqf_list(self) -> tuple[Expr, list[tuple[Poly, int]]]:
        """The square-free decomposition ``(coeff, [(g, k)])`` (SymPy's
        ``Poly.sqf_list``): over ``QQ`` the leading coefficient comes out,
        over ``ZZ`` and a polynomial domain the content with the sign."""
        from .multisqf import IntegerSqfDomain, ParameterSqfDomain, RationalSqfDomain, sqf_list
        n = len(self._gens)
        names = self._rep.ring.names
        if not self._domain.params and n == 1:
            f = self._univariate_rational()
            if self._domain.integral:
                ints = [int(c) for c in f]
                coefficient, factors = sqfreetools.dup_sqf_list(ints, ZZ)
                return (Integer(coefficient), [(self._from_univariate([Fraction(c) for c in g], True), k)
                                               for g, k in factors])
            rational, factors_q = sqfreetools.dup_sqf_list(f, QQ)
            return (number_from_fraction(rational), [(self._from_univariate(g, False), k) for g, k in factors_q])
        if self._domain.integral:
            DZ = (IntegerSqfDomain(names) if not self._domain.params
                  else ParameterSqfDomain(IntegerPolynomialRing(names[n:]), names[:n]))
            cz, fz = sqf_list(DZ.ring.from_dict({m: int(c) for m, c in self._rep.terms.items()}), DZ)
            return (self._coefficient_expr({m[n:]: Fraction(c) for m, c in cz.terms.items()}),
                    [(self._same_ring(self._rep.ring.from_dict({m: Fraction(c) for m, c in g.terms.items()})), k)
                     for g, k in fz])
        DQ = (RationalSqfDomain(names) if not self._domain.params
              else ParameterSqfDomain(RationalPolynomialRing(names[n:]), names[:n]))
        cq, fq = sqf_list(DQ.ring.from_dict(dict(self._rep.terms)), DQ)
        return (self._coefficient_expr({m[n:]: c for m, c in cq.terms.items()}),
                [(self._same_ring(self._rep.ring.from_dict(dict(g.terms))), k) for g, k in fq])

    # PORTED FROM SYMPY: sympy/polys/factortools.py:dup_factor_list, dmp_factor_list
    # (the domains ZZ, QQ, ZZ[a], QQ[a]), sympy/polys/polyutils.py:_sort_factors
    def factor_list(self) -> tuple[Expr, list[tuple[Poly, int]]]:
        """The irreducible factors over the domain with their
        multiplicities and the leading constant: the monomial of the
        generators split off, the content and the denominators taken out,
        multivariate polynomials factored by Wang's algorithm
        (:mod:`~sympy_extras.polys.multifactor`), a polynomial domain
        ``ZZ[a]`` by factoring in all the variables; the factors sorted as
        SymPy sorts them."""
        if self.is_zero:
            return (Integer(0), [])
        if not self._domain.params:
            coefficient, factors = _factor_all(self._rep)
            return (number_from_fraction(coefficient),
                    [(self._same_ring(self._rep.ring.from_dict({m: Fraction(c) for m, c in f.terms.items()})), k)
                     for f, k in factors])
        n = len(self._gens)
        exponents = [min(m[i] for m in self._rep.terms) for i in range(n)]
        shift = tuple(exponents) + tuple(0 for _ in self._domain.params)
        f = self._rep.ring.from_dict({tuple(e - s for e, s in zip(m, shift)): c for m, c in self._rep.terms.items()})
        if self._domain.integral:
            Z = self._integer_ring()
            fz = Z.ring.from_dict({m: int(c) for m, c in f.terms.items()})
            content_z = Z.ground_content(fz) if isinstance(Z, IteratedPolynomialRing) else Z.one
            if not Z.is_one(content_z) and isinstance(Z, IteratedPolynomialRing):
                fz = fz.exquo(Z._embed_params(content_z))
            f = self._rep.ring.from_dict({m: Fraction(c) for m, c in fz.terms.items()})
            content = self._param_expr({m: Fraction(c) for m, c in content_z.terms.items()})
        else:
            Q = self._rational_ring()
            fq = Q.ring.from_dict(dict(f.terms))
            content_q = Q.ground_content(fq) if isinstance(Q, IteratedPolynomialRing) else Q.one
            if not Q.is_one(content_q) and isinstance(Q, IteratedPolynomialRing):
                fq = fq.exquo(Q._embed_params(content_q))
            f = self._rep.ring.from_dict(dict(fq.terms))
            content = self._param_expr(dict(content_q.terms))
        coefficient, factors = _factor_all(f)
        result: list[tuple[PolyElement[int], int]] = list(factors)
        ring_z = PolyRing(ZZ, self._rep.ring.names)
        for i in reversed(range(n)):
            if exponents[i]:
                monomial = tuple(1 if k == i else 0 for k in range(ring_z.ngens))
                result.insert(0, (ring_z.from_dict({monomial: 1}), exponents[i]))
        result.sort(key=lambda item: (_first_degree(item[0]) + 1, item[1], _domain_key(item[0], n)))
        return (mul(number_from_fraction(coefficient), content),
                [(self._same_ring(self._rep.ring.from_dict({m: Fraction(c) for m, c in g.terms.items()})), k)
                 for g, k in result])

    def _param_expr(self, terms: dict[Monomial, Fraction]) -> Expr:
        return self._coefficient_expr(terms)


ParamKey = tuple[int, list[tuple[Monomial, int]]]
DomainKey = Union[ParamKey, tuple['DomainKey', ...]]


def _first_degree(f: PolyElement[int]) -> int:
    return max((m[0] for m in f.terms), default=-1)


def _domain_key(f: PolyElement[int], n: int) -> DomainKey:
    """SymPy's sort key of a dense polynomial in ``n`` generators over a
    polynomial domain: nested tuples over the generators, the coefficients
    compared as ``PolyElement.sort_key`` (the number of terms, then the
    terms in decreasing lex order)."""
    return _domain_key_level(dict(f.terms), 0, n)


def _domain_key_level(terms: dict[Monomial, int], i: int, n: int) -> DomainKey:
    if i == n:
        ordered = sorted(((m[n:], c) for m, c in terms.items()), reverse=True)
        return (len(ordered), ordered)
    if not terms:
        return ()
    by_degree: dict[int, dict[Monomial, int]] = {}
    for m, c in terms.items():
        by_degree.setdefault(m[i], {})[m] = c
    top = max(by_degree)
    return tuple(_domain_key_level(by_degree.get(d, {}), i + 1, n) for d in range(top, -1, -1))


def _factor_all(f: PolyElement[Fraction]) -> tuple[Fraction, list[tuple[PolyElement[int], int]]]:
    """SymPy's ``dup_factor_list``/``dmp_factor_list`` over ``ZZ`` or ``QQ``
    in all the variables of the ring of ``f``: the coefficient and the
    integer primitive factors (positive leading coefficients), sorted."""
    from .multifactor import _sorted, zz_factor
    names = f.ring.names
    n = len(names)
    exponents = [min(m[i] for m in f.terms) for i in range(n)]
    shifted = {tuple(e - s for e, s in zip(m, exponents)): c for m, c in f.terms.items()}
    content = Fraction(0)
    for c in shifted.values():
        content = Fraction(integer_gcd(content.numerator, c.numerator),
                           content.denominator * c.denominator // integer_gcd(content.denominator, c.denominator))
    ints = {m: int(c / content) for m, c in shifted.items()}
    ring = PolyRing(ZZ, names)
    if n == 1:
        dense = densebasic.dup_from_dict({m[0]: c for m, c in ints.items()}, ZZ)
        coefficient, dense_factors = factortools.dup_zz_factor(dense, ZZ)
        result = [(ring.from_dict({(len(g) - 1 - i,): c for i, c in enumerate(g) if c}), k) for g, k in dense_factors]
        if exponents[0]:
            result.insert(0, (ring.from_dict({(1,): 1}), exponents[0]))
        result.sort(key=lambda item: (_first_degree(item[0]) + 1, item[1], _dense_ints(item[0])))
        return (Fraction(coefficient) * content, result)
    used = [i for i in range(n) if any(m[i] for m in ints)]
    if not used:
        coefficient = ints.get(tuple(0 for _ in names), 0)
        factors: list[tuple[PolyElement[int], int]] = []
    else:
        sub = PolyRing(ZZ, [names[i] for i in used])
        g = sub.from_dict({tuple(m[i] for i in used): c for m, c in ints.items()})
        coefficient, sub_factors = zz_factor(g)
        factors = []
        for h, k in sub_factors:
            terms: dict[Monomial, int] = {}
            for m, value in h.terms.items():
                full = [0] * n
                for position, e in zip(used, m):
                    full[position] = e
                terms[tuple(full)] = value
            factors.append((ring.from_dict(terms), k))
    for i in reversed(range(n)):
        if exponents[i]:
            monomial = tuple(1 if k == i else 0 for k in range(n))
            factors.insert(0, (ring.from_dict({monomial: 1}), exponents[i]))
    return (Fraction(coefficient) * content, _sorted(factors))


def _dense_ints(f: PolyElement[int]) -> list[int]:
    return densebasic.dup_from_dict({m[0]: c for m, c in f.terms.items()}, ZZ)


def _terms_gcd(values: Sequence[Fraction]) -> int:
    result = 0
    for v in values:
        result = integer_gcd(result, int(v))
    return result


def _resultant_first(f: PolyElement[Fraction], g: PolyElement[Fraction], names: list[str],
                     integral: bool) -> PolyElement[Fraction]:
    """The resultant in the first variable, as a polynomial in all the
    variables (constant in the first)."""
    if len(names) == 1:
        rf = densebasic.dup_from_dict({m[0]: c for m, c in f.terms.items()}, QQ)
        rg = densebasic.dup_from_dict({m[0]: c for m, c in g.terms.items()}, QQ)
        value = euclidtools.dup_resultant(rf, rg, QQ)
        return f.ring.from_dict({(0,): value})
    if integral:
        RZ = SparsePolynomialRing(ZZ, names)
        inner_z = SparsePolynomialRing(ZZ, names[1:])
        fz = RZ.ring.from_dict({m: int(c) for m, c in f.terms.items()})
        gz = RZ.ring.from_dict({m: int(c) for m, c in g.terms.items()})
        rz = euclidtools.dup_prs_resultant(RZ.to_univariate(fz, inner_z), RZ.to_univariate(gz, inner_z), inner_z)
        return f.ring.from_dict({(0,) + m: Fraction(c) for m, c in rz.terms.items()})
    RQ = SparsePolynomialRing(QQ, names)
    inner_q = SparsePolynomialRing(QQ, names[1:])
    rq = euclidtools.dup_prs_resultant(RQ.to_univariate(RQ.ring.from_dict(f.terms), inner_q),
                                   RQ.to_univariate(RQ.ring.from_dict(g.terms), inner_q), inner_q)
    return f.ring.from_dict({(0,) + m: c for m, c in rq.terms.items()})


def _print_coefficient(c: Expr) -> str:
    """A coefficient as SymPy's ``_print_Poly`` prints it: a rational as
    ``p/q`` (``1/2*x``), anything else by the printer."""
    if isinstance(c, Rational) and c.q != 1:
        return '%d/%d' % (c.p, c.q)
    return str(c)


def _rep_from_coefficients(coefficients: dict[Monomial, Expr], gens: tuple[Expr, ...],
                           domain: _Domain) -> PolyElement[Fraction]:
    names = [p.name for p in domain.params]
    ring = PolyRing(QQ, _internal_names(len(gens), len(domain.params)))
    terms: dict[Monomial, Fraction] = {}
    for m, c in coefficients.items():
        if names:
            for pm, value in dict_from_expr(c, names, QQ).items():
                if value:
                    terms[m + pm] = terms.get(m + pm, Fraction(0)) + value
        elif isinstance(c, Rational):
            if c.p:
                terms[m] = terms.get(m, Fraction(0)) + c.as_fraction()
        else:
            raise CoercionFailed('cannot convert %s into %s' % (c, domain.name))
    return ring.from_dict({m: c for m, c in terms.items() if c})


# ---------------------------------------------------------------------
# the functions

def poly(expr: ExprLike, *gens: ExprLike) -> Poly:
    """``Poly(expr, *gens)``."""
    return Poly(expr, *gens)


def _as_poly(f: PolyLike, gens: Sequence[ExprLike]) -> Poly:
    return f if isinstance(f, Poly) and not gens else Poly(f, *gens)


def degree(f: PolyLike, gen: Union[int, ExprLike] = 0) -> int:
    """The degree of ``f`` in ``gen`` (``-1`` for zero)."""
    if isinstance(f, Poly):
        return f.degree(gen)
    if isinstance(gen, int):
        return Poly(f).degree(gen)
    return Poly(f, gen).degree()


def LC(f: PolyLike, *gens: ExprLike) -> Expr:
    return _as_poly(f, gens).LC()


def _numbers_only(f: PolyLike, g: PolyLike, gens: Sequence[ExprLike]) -> Optional[tuple[Expr, Expr]]:
    """``(f, g)`` as expressions when there are no generators (numbers)."""
    if gens or isinstance(f, Poly) or isinstance(g, Poly):
        return None
    ef, eg = to_expr(f), to_expr(g)
    if _common_gens(ef, eg):
        return None
    return ef, eg


def gcd(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    """The gcd of two polynomial expressions, as an expression (of two
    rationals, the gcd of the numerators over the lcm of the denominators;
    of other numbers, 1)."""
    numbers = _numbers_only(f, g, gens)
    if numbers is not None:
        na, nb = numbers
        if isinstance(na, Rational) and isinstance(nb, Rational):
            from math import lcm as integer_lcm
            return number_from_fraction(Fraction(integer_gcd(na.p, nb.p), integer_lcm(na.q, nb.q)))
        return Integer(1)
    pf = _as_poly(f, gens) if gens or isinstance(f, Poly) else Poly(to_expr(f), *_common_gens(f, g))
    return pf.gcd(g).as_expr()


def lcm(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    """The lcm of two polynomial expressions, as an expression (of two
    rationals, the lcm of the numerators over the gcd of the
    denominators; of other numbers, their product)."""
    numbers = _numbers_only(f, g, gens)
    if numbers is not None:
        na, nb = numbers
        if isinstance(na, Rational) and isinstance(nb, Rational):
            from math import lcm as integer_lcm
            return number_from_fraction(Fraction(integer_lcm(na.p, nb.p), integer_gcd(na.q, nb.q)))
        return mul(na, nb)
    pf = _as_poly(f, gens) if gens or isinstance(f, Poly) else Poly(to_expr(f), *_common_gens(f, g))
    return pf.lcm(g).as_expr()


# PORTED FROM SYMPY: sympy/polys/polytools.py:gcd_list (the API; computed by folding gcd)
def gcd_list(seq: Sequence[PolyLike], *gens: ExprLike) -> Expr:
    """The gcd of a list of polynomial expressions (``0`` of an empty
    list), folding :func:`gcd`: SymPy's results, with its primitive and
    sign conventions.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.polytools import gcd_list
    >>> n = symbols('n')
    >>> gcd_list([2*n + 2, 4*n**2 - 4]), gcd_list([n**2 - 1, n + 1, 3*n + 3])
    (2*n + 2, n + 1)
    """
    if not seq:
        return Integer(0)
    first = seq[0]
    result = first.as_expr() if isinstance(first, Poly) else to_expr(first)
    for g in seq[1:]:
        result = gcd(result, g, *gens)
    return result


# PORTED FROM SYMPY: sympy/polys/polytools.py:lcm_list (the API; computed by folding lcm)
def lcm_list(seq: Sequence[PolyLike], *gens: ExprLike) -> Expr:
    """The lcm of a list of polynomial expressions (``1`` of an empty
    list), folding :func:`lcm`.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.polytools import lcm_list
    >>> n = symbols('n')
    >>> lcm_list([n**2 - 1, n + 1, 3*n + 3])
    3*n**2 - 3
    """
    if not seq:
        return Integer(1)
    first = seq[0]
    result = first.as_expr() if isinstance(first, Poly) else to_expr(first)
    for g in seq[1:]:
        result = lcm(result, g, *gens)
    return result


def _common_gens(f: PolyLike, g: PolyLike) -> tuple[Expr, ...]:
    """The generators of ``f`` and ``g`` together (SymPy's
    ``parallel_poly_from_expr``)."""
    ef = f.as_expr() if isinstance(f, Poly) else to_expr(f)
    eg = g.as_expr() if isinstance(g, Poly) else to_expr(g)
    _, gens = _decompose_without_gens_many([ef, eg])
    return gens


def _denominators_lcm(f: PolyElement[Fraction]) -> int:
    result = 1
    for c in f.terms.values():
        result = result * c.denominator // integer_gcd(result, c.denominator)
    return result


def resultant(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    r = _as_poly(f, gens).resultant(g if isinstance(g, Poly) else Poly(g, *(gens or _as_poly(f, gens).gens)))
    return r.as_expr() if isinstance(r, Poly) else r


def discriminant(f: PolyLike, *gens: ExprLike) -> Expr:
    d = _as_poly(f, gens).discriminant()
    return d.as_expr() if isinstance(d, Poly) else d


# ---------------------------------------------------------------------
# rational functions

def _joint_rings(exprs: Sequence[Expr], expand: bool = True) -> tuple[tuple[Expr, ...], list[dict[Monomial, Fraction]]]:
    """SymPy's ``sring`` of several expressions without generators: every
    factor which is not a number is a generator (the generators of all the
    expressions together, sorted), the coefficients rational (``ZZ`` or
    ``QQ``; ``I`` and floats are not ported yet)."""
    decomposed, gens = _decompose_without_gens_many(exprs, expand)
    result: list[dict[Monomial, Fraction]] = []
    if not gens:
        return gens, [{} for _ in exprs]
    for terms in decomposed:
        rep: dict[Monomial, Fraction] = {}
        for m, cs in terms.items():
            c = add(*cs)
            if not isinstance(c, Rational):
                raise CoercionFailed('the domain of %s (ZZ_I, RR, EX) is not ported yet' % (c,))
            value = rep.get(m, Fraction(0)) + c.as_fraction()
            if value:
                rep[m] = value
            else:
                rep.pop(m, None)
        result.append(rep)
    return gens, result


def _joint_rings_gaussian(exprs: Sequence[Expr], expand: bool = True
                          ) -> tuple[tuple[Expr, ...], list[dict[Monomial, GaussianRational]]]:
    """:func:`_joint_rings` with Gaussian coefficients ``a + b*I`` (SymPy's
    ``ZZ_I`` and ``QQ_I``); ``CoercionFailed`` for other coefficients."""
    decomposed, gens = _decompose_without_gens_many(exprs, expand)
    result: list[dict[Monomial, GaussianRational]] = []
    if not gens:
        return gens, [{} for _ in exprs]
    for terms in decomposed:
        rep: dict[Monomial, GaussianRational] = {}
        for m, cs in terms.items():
            try:
                c = QQ_I.from_expr(add(*cs))
            except ValueError:
                raise CoercionFailed('the domain of %s (RR, EX) is not ported yet' % (add(*cs),))
            value = QQ_I.add(rep.get(m, QQ_I.zero), c)
            if value:
                rep[m] = value
            else:
                rep.pop(m, None)
        result.append(rep)
    return gens, result


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel (the domains ZZ_I and QQ_I)
def _cancel_gaussian(f: dict[Monomial, GaussianRational], g: dict[Monomial, GaussianRational], n: int
                     ) -> tuple[dict[Monomial, GaussianRational], dict[Monomial, GaussianRational]]:
    """``f/g`` in lowest terms over ``ZZ_I`` (the denominators of ``QQ_I``
    cleared), made canonical with respect to the quadrant of the leading
    coefficient of the denominator."""
    names = _internal_names(n, 0)
    R = SparsePolynomialRing(ZZ_I, names)
    if not f:
        return {}, {(0,) * n: QQ_I.one}

    def cleared(h: dict[Monomial, GaussianRational]) -> tuple[GaussianInteger, PolyElement[GaussianInteger]]:
        # SymPy's clear_denoms over QQ_I: the lcm of the denominators over ZZ_I
        common = ZZ_I.one
        for c in h.values():
            common = ZZ_I.lcm(common, ZZ_I.denom_of(c))
        k = QQ_I.convert(common, ZZ_I)
        return common, R.ring.from_dict({m: ZZ_I.convert(QQ_I.mul(c, k), QQ_I) for m, c in h.items()})

    cq, F = cleared(f)
    cp, G = cleared(g)
    h = R.gcd(F, G)
    P, Q = F.exquo(h), G.exquo(h)
    common = ZZ_I.gcd(cp, cq)
    cp, cq = ZZ_I.quo(cp, common), ZZ_I.quo(cq, common)
    p = {m: QQ_I.mul(QQ_I.convert(c, ZZ_I), QQ_I.convert(cp, ZZ_I)) for m, c in P.terms.items()}
    q = {m: QQ_I.mul(QQ_I.convert(c, ZZ_I), QQ_I.convert(cq, ZZ_I)) for m, c in Q.terms.items()}
    # canonical with respect to the quadrant of the leading coefficient
    lead = max(q, key=R.ring.order.key)
    u = QQ_I.canonical_unit(q[lead])
    if u != QQ_I.one:
        p = {m: QQ_I.mul(c, u) for m, c in p.items()}
        q = {m: QQ_I.mul(c, u) for m, c in q.items()}
    return p, q


def _gaussian_expr_of(f: dict[Monomial, GaussianRational], gens: tuple[Expr, ...]) -> Expr:
    parts: list[Expr] = []
    for m in sorted(f, reverse=True):
        factors: list[Expr] = [QQ_I.to_expr(f[m])]
        factors += [pow(g, Integer(k)) for g, k in zip(gens, m) if k]
        parts.append(mul(*factors))
    return add(*parts)


# PORTED FROM SYMPY: sympy/polys/domains/realfield.py:to_rational
def real_to_rational(value: Fraction, max_denom: int, limit: bool = True) -> Fraction:
    """The rational of a float (its exact value) with the denominator at most
    ``max_denom``: the closer of the two last convergents of the continued
    fraction (SymPy's ``RealField.to_rational``)."""
    p, q = value.numerator, value.denominator
    if not limit or q <= max_denom:
        return value
    p0, q0, p1, q1 = 0, 1, 1, 0
    n, d = p, q
    while True:
        a = n // d
        q2 = q0 + a*q1
        if q2 > max_denom:
            break
        p0, q0, p1, q1 = p1, q1, p0 + a*p1, q2
        n, d = d, n - a*d
    k = (max_denom - q0) // q1
    bound1 = Fraction(p0 + k*p1, q0 + k*q1)
    bound2 = Fraction(p1, q1)
    if not bound2 or not bound1:
        return value
    if abs(bound2 - value) <= abs(bound1 - value):
        return bound2
    return bound1


def _joint_rings_real(exprs: Sequence[Expr], expand: bool = True
                      ) -> tuple[tuple[Expr, ...], list[dict[Monomial, Fraction]], int]:
    """:func:`_joint_rings` over SymPy's ``RR`` (some coefficients are
    floats): the coefficients summed, at the largest precision of the
    floats (their exact binary values); with that precision."""
    from sympy_extras.core.numbers import Float, _round_to_prec
    decomposed, gens = _decompose_without_gens_many(exprs, expand)
    prec = 0
    sums: list[dict[Monomial, Union[Float, Rational]]] = []
    for terms in decomposed:
        summed: dict[Monomial, Union[Float, Rational]] = {}
        for m, cs in terms.items():
            c = add(*cs)
            if not isinstance(c, (Float, Rational)):
                raise CoercionFailed('the domain of %s (CC, EX) is not ported yet' % (c,))
            if isinstance(c, Float):
                prec = max(prec, c._prec)
            summed[m] = c
        sums.append(summed)
    if prec == 0:
        raise CoercionFailed('no float coefficient')
    result: list[dict[Monomial, Fraction]] = []
    for summed in sums:
        rep: dict[Monomial, Fraction] = {}
        for m, c in summed.items():
            # RR.from_sympy: the value at the precision (exact dyadic)
            exact = c._value if isinstance(c, Float) else c.as_fraction()
            value = _round_to_prec(exact, prec)
            if value:
                rep[m] = value
        result.append(rep)
    return gens, result, prec


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel and cofactors (over RR)
def _cancel_real(f: dict[Monomial, Fraction], g: dict[Monomial, Fraction], n: int, prec: int
                 ) -> tuple[dict[Monomial, Fraction], dict[Monomial, Fraction]]:
    """``f/g`` over ``RR`` (the values are the exact binary values of the
    floats): a term divides out its monomial gcd with the floats kept
    (``RR.gcd`` is ``1``); otherwise the cofactors of the monic gcd over
    ``QQ`` (the floats made rational by ``RealField.to_rational``) are
    rounded back; the leading coefficient of the denominator positive."""
    from sympy_extras.core.numbers import _round_to_prec
    Z = IntegerPolynomialRing(_internal_names(n, 0))
    if not f:
        return {}, {(0,) * n: Fraction(1)}
    P: dict[Monomial, Fraction]
    Q: dict[Monomial, Fraction]
    if len(f) == 1 or len(g) == 1:
        # PORTED FROM SYMPY: PolyElement._gcd_monom (the ground gcd of RR is 1)
        term, other = (f, g) if len(f) == 1 else (g, f)
        (mt, ct), = term.items()
        common = mt
        for m in other:
            common = tuple(min(a, b) for a, b in zip(common, m))
        reduced_term = {tuple(a - b for a, b in zip(mt, common)): ct}
        reduced_other = {tuple(a - b for a, b in zip(m, common)): c for m, c in other.items()}
        P, Q = (reduced_term, reduced_other) if len(f) == 1 else (reduced_other, reduced_term)
    else:
        max_denom = max(2**prec // 200, 99)
        fq = {m: real_to_rational(c, max_denom) for m, c in f.items()}
        gq = {m: real_to_rational(c, max_denom) for m, c in g.items()}

        def denominators(h: dict[Monomial, Fraction]) -> int:
            d = 1
            for c in h.values():
                d = d * c.denominator // integer_gcd(d, c.denominator)
            return d

        cf, cg = denominators(fq), denominators(gq)
        F = Z.ring.from_dict({m: int(c * cf) for m, c in fq.items() if c})
        G = Z.ring.from_dict({m: int(c * cg) for m, c in gq.items() if c})
        h = Z.gcd(F, G)
        lc = Fraction(h.LC)
        # the cofactors over QQ rounded to RR
        P = {m: _round_to_prec(Fraction(c) * lc / cf, prec) for m, c in F.exquo(h).terms.items()}
        Q = {m: _round_to_prec(Fraction(c) * lc / cg, prec) for m, c in G.exquo(h).terms.items()}
    lead = max(Q, key=Z.ring.order.key)
    if Q[lead] < 0:
        P = {m: -c for m, c in P.items()}
        Q = {m: -c for m, c in Q.items()}
    return P, Q


def _real_expr_of(f: dict[Monomial, Fraction], gens: tuple[Expr, ...], prec: int) -> Expr:
    """The polynomial with float coefficients of the precision (SymPy's
    ``RR.to_sympy``)."""
    from sympy_extras.core.numbers import Float
    parts: list[Expr] = []
    for m in sorted(f, reverse=True):
        factors: list[Expr] = [Float(number_from_fraction(f[m]), precision=prec)]
        factors += [pow(g, Integer(k)) for g, k in zip(gens, m) if k]
        parts.append(mul(*factors))
    return add(*parts) if parts else Float(Integer(0), precision=prec)


#: a coefficient of ``CC``: the exact binary values of the real and imaginary parts
ComplexPair = tuple[Fraction, Fraction]


def _float_or_rational(e: Expr) -> Optional[tuple[Fraction, int]]:
    """The exact value of a float or a rational and the precision of the
    float (``0`` for a rational); ``None`` for any other number."""
    from sympy_extras.core.numbers import Float
    if isinstance(e, Float):
        return e._value, e._prec
    if isinstance(e, Rational):
        return e.as_fraction(), 0
    return None


def _joint_rings_complex(exprs: Sequence[Expr]) -> tuple[tuple[Expr, ...], list[dict[Monomial, ComplexPair]], int]:
    """:func:`_joint_rings` over SymPy's ``CC`` (coefficients ``a + b*I``
    with float or rational parts, some float): the parts at the largest
    precision of the floats (their exact binary values); with that
    precision. ``CoercionFailed`` for other coefficients (``EX``)."""
    from sympy_extras.core.numbers import _round_to_prec
    from .domains.gaussiandomains import _gaussian_parts
    decomposed, gens = _decompose_without_gens_many(exprs)
    prec = 0
    sums: list[dict[Monomial, ComplexPair]] = []
    for terms in decomposed:
        summed: dict[Monomial, ComplexPair] = {}
        for m, cs in terms.items():
            c = add(*cs)
            try:
                re, im = _gaussian_parts(c)
            except ValueError:
                raise CoercionFailed('the domain of %s (EX) is not ported yet' % (c,)) from None
            vre, vim = _float_or_rational(re), _float_or_rational(im)
            if vre is None or vim is None:
                raise CoercionFailed('the domain of %s (EX) is not ported yet' % (c,))
            prec = max(prec, vre[1], vim[1])
            summed[m] = (vre[0], vim[0])
        sums.append(summed)
    if prec == 0:
        raise CoercionFailed('no float coefficient')
    result: list[dict[Monomial, ComplexPair]] = []
    for summed in sums:
        rep: dict[Monomial, ComplexPair] = {}
        for m, (a, b) in summed.items():
            # CC.from_sympy: the parts at the precision of the domain
            value = (_round_to_prec(a, prec), _round_to_prec(b, prec))
            if value[0] or value[1]:
                rep[m] = value
        result.append(rep)
    return gens, result, prec


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel and cofactors (over CC)
def _cancel_complex(f: dict[Monomial, ComplexPair], g: dict[Monomial, ComplexPair], n: int, prec: int
                    ) -> tuple[dict[Monomial, ComplexPair], dict[Monomial, ComplexPair]]:
    """``f/g`` over ``CC`` (SymPy's ``PolyElement.cancel``): a term divides
    out its monomial gcd with the floats kept (``CC.gcd`` is ``1``);
    otherwise the gcd is computed over the exact domain ``QQ_I``
    (``dmp_inner_gcd`` of an inexact domain: each part rounded to a float
    of ``RealField()`` and made rational by ``RealField.to_rational``), and
    the cofactors of the monic gcd are rounded back to the precision. The
    canonical unit of ``CC`` is ``1`` (no element is negative): no sign
    is changed."""
    from sympy_extras.core.numbers import _round_to_prec
    if not f:
        return {}, {(0,) * n: (Fraction(1), Fraction(0))}
    if len(f) == 1 or len(g) == 1:
        # PORTED FROM SYMPY: PolyElement._gcd_monom (the ground gcd of CC is 1)
        term, other = (f, g) if len(f) == 1 else (g, f)
        (mt, ct), = term.items()
        common = mt
        for m in other:
            common = tuple(min(a, b) for a, b in zip(common, m))
        reduced_term = {tuple(a - b for a, b in zip(mt, common)): ct}
        reduced_other = {tuple(a - b for a, b in zip(m, common)): c for m, c in other.items()}
        return (reduced_term, reduced_other) if len(f) == 1 else (reduced_other, reduced_term)
    max_denom = max(2**53 // 200, 99)

    def exact(c: ComplexPair) -> GaussianRational:
        # QQ_I.from_ComplexField: QQ.convert of each part, through RealField()
        return GaussianRational(real_to_rational(_round_to_prec(c[0], 53), max_denom),
                                real_to_rational(_round_to_prec(c[1], 53), max_denom))

    R = SparsePolynomialRing(ZZ_I, _internal_names(n, 0))

    def cleared(h: dict[Monomial, ComplexPair]) -> tuple[GaussianRational, PolyElement[GaussianInteger]]:
        values = {m: exact(c) for m, c in h.items()}
        common = ZZ_I.one
        for c in values.values():
            common = ZZ_I.lcm(common, ZZ_I.denom_of(c))
        k = QQ_I.convert(common, ZZ_I)
        terms = {m: ZZ_I.convert(QQ_I.mul(c, k), QQ_I) for m, c in values.items()}
        return k, R.ring.from_dict({m: c for m, c in terms.items() if c})

    cf, F = cleared(f)
    cg, G = cleared(g)
    h = R.gcd(F, G)
    lc = QQ_I.convert(h.LC, ZZ_I)

    def cofactor(H: PolyElement[GaussianInteger], c: GaussianRational) -> dict[Monomial, ComplexPair]:
        found: dict[Monomial, ComplexPair] = {}
        for m, v in H.exquo(h).terms.items():
            value = QQ_I.exquo(QQ_I.mul(QQ_I.convert(v, ZZ_I), lc), c)
            found[m] = (_round_to_prec(value.x, prec), _round_to_prec(value.y, prec))
        return found

    return cofactor(F, cf), cofactor(G, cg)


def _complex_expr_of(f: dict[Monomial, ComplexPair], gens: tuple[Expr, ...], prec: int) -> Expr:
    """The polynomial with complex float coefficients of the precision
    (SymPy's ``CC.to_sympy``: ``Float(re) + I*Float(im)``)."""
    from sympy_extras.core.numbers import Float
    parts: list[Expr] = []
    for m in sorted(f, reverse=True):
        re, im = f[m]
        value = add(Float(number_from_fraction(re), precision=prec),
                    mul(I, Float(number_from_fraction(im), precision=prec)))
        factors: list[Expr] = [value]
        factors += [pow(g, Integer(k)) for g, k in zip(gens, m) if k]
        parts.append(mul(*factors))
    return add(*parts) if parts else Float(Integer(0), precision=prec)


def _joint_rings_ex(exprs: Sequence[Expr]) -> tuple[tuple[Expr, ...], list[dict[Monomial, Expr]]]:
    """:func:`_joint_rings` over SymPy's ``EX``: the coefficients are the
    numbers which no other domain takes (``zoo``, ``oo``, ``nan`` among
    rationals, floats and ``I``), kept as expressions."""
    from .domains.expressiondomain import EX
    decomposed, gens = _decompose_without_gens_many(exprs)
    result: list[dict[Monomial, Expr]] = []
    for terms in decomposed:
        rep: dict[Monomial, Expr] = {}
        for m, cs in terms.items():
            c = add(*cs)
            if not EX.is_zero(c):
                rep[m] = c
        result.append(rep)
    return gens, result


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel, cofactors, _gcd_monom (over EX)
def _cancel_ex(f: dict[Monomial, Expr], g: dict[Monomial, Expr], n: int
               ) -> tuple[dict[Monomial, Expr], dict[Monomial, Expr]]:
    """``f/g`` over ``EX`` (SymPy's ``PolyElement.cancel``): a term divides
    out its monomial gcd (``EX.gcd`` is ``1``); otherwise the cofactors of
    the monic gcd, by dense division in one generator (``dmp_inner_gcd``
    over a field), and in several by the gcd of the sparse ring when every
    coefficient is finite (with ``zoo``, ``oo`` or ``nan`` a division need
    not end: no common factor is taken out, where SymPy's result is
    meaningless anyway); the denominator's leading coefficient without a
    minus sign it could extract (the canonical unit of ``EX``)."""
    from .domains.expressiondomain import EX
    if not f:
        return {}, {(0,) * n: Integer(1)}
    P: dict[Monomial, Expr]
    Q: dict[Monomial, Expr]
    if len(f) == 1 or len(g) == 1:
        term, other = (f, g) if len(f) == 1 else (g, f)
        (mt, ct), = term.items()
        common = mt
        for m in other:
            common = tuple(min(a, b) for a, b in zip(common, m))
        reduced_term = {tuple(a - b for a, b in zip(mt, common)): ct}
        reduced_other = {tuple(a - b for a, b in zip(m, common)): c for m, c in other.items()}
        P, Q = (reduced_term, reduced_other) if len(f) == 1 else (reduced_other, reduced_term)
    elif n == 1:
        df = densebasic.dup_from_dict({m[0]: c for m, c in f.items()}, EX)
        dg = densebasic.dup_from_dict({m[0]: c for m, c in g.items()}, EX)
        # dup_ff_prs_gcd: the last subresultant made monic
        h = densetools.dup_monic(euclidtools.dup_subresultants(df, dg, EX)[-1], EX)

        def sparse(d: list[Expr]) -> dict[Monomial, Expr]:
            k = len(d) - 1
            return {(k - i,): c for i, c in enumerate(d) if not EX.is_zero(c)}
        P, Q = sparse(densearith.dup_quo(df, h, EX)), sparse(densearith.dup_quo(dg, h, EX))
    elif all(c.is_finite is True for c in list(f.values()) + list(g.values())):
        R = SparsePolynomialRing(EX, _internal_names(n, 0))
        F, G = R.ring.from_dict(f), R.ring.from_dict(g)
        H = R.gcd(F, G)
        P, Q = dict(F.exquo(H).terms), dict(G.exquo(H).terms)
    else:
        P, Q = dict(f), dict(g)
    lead = max(Q)
    if EX.is_negative(Q[lead]):
        P = {m: EX.neg(c) for m, c in P.items()}
        Q = {m: EX.neg(c) for m, c in Q.items()}
    return P, Q


def _ex_expr_of(f: dict[Monomial, Expr], gens: tuple[Expr, ...]) -> Expr:
    parts: list[Expr] = []
    for m in sorted(f, reverse=True):
        factors: list[Expr] = [f[m]]
        factors += [pow(g, Integer(k)) for g, k in zip(gens, m) if k]
        parts.append(mul(*factors))
    return add(*parts)


# PORTED FROM SYMPY: sympy/polys/factortools.py:dmp_factor_list (an inexact domain)
def _real_factor_list(base: Expr) -> tuple[Expr, list[tuple[Expr, int]]]:
    """The factorisation over ``RR``: the floats made rational
    (``RealField.to_rational``), the polynomial factored over ``QQ``, each
    factor divided by its largest coefficient in absolute value (moved
    into the coefficient), the factors and the coefficient rounded back
    to floats; the powers of the generators dividing every term are the
    factors ``1.0*x``."""
    from sympy_extras.core.numbers import Float, _round_to_prec
    gens, (rep,), prec = _joint_rings_real([base])
    n = len(gens)
    # dmp_terms_gcd
    J = [min(m[i] for m in rep) for i in range(n)] if rep else [0] * n
    shifted = {tuple(a - j for a, j in zip(m, J)): c for m, c in rep.items()}
    max_denom = max(2**prec // 200, 99)
    exact = add(*[mul(number_from_fraction(real_to_rational(c, max_denom)),
                      *[pow(g, Integer(k)) for g, k in zip(gens, m) if k]) for m, c in shifted.items()])
    coeff = Fraction(1)
    factors: list[tuple[dict[Monomial, Fraction], int]] = []
    if exact.has(*gens):
        c, poly_factors = Poly(exact, *gens).factor_list()
        if not isinstance(c, Rational):
            raise CoercionFailed('the coefficient %s is not rational' % (c,))
        coeff = c.as_fraction()
        for g, k in poly_factors:
            terms: dict[Monomial, Fraction] = {}
            for m, value in g.terms():
                if not isinstance(value, Rational):
                    raise CoercionFailed('the coefficient %s is not rational' % (value,))
                terms[m] = value.as_fraction()
            # dmp_max_norm: the factor divided by it, moved into the coefficient
            max_norm = max(abs(v) for v in terms.values())
            factors.append(({m: _round_to_prec(v / max_norm, prec) for m, v in terms.items()}, k))
            coeff *= max_norm**k
    else:
        coeff = exact.as_fraction() if isinstance(exact, Rational) else Fraction(1)
    for i in range(n - 1, -1, -1):
        if J[i]:
            factors.insert(0, ({tuple(1 if t == i else 0 for t in range(n)): Fraction(1)}, J[i]))

    def expr_of(terms: dict[Monomial, Fraction]) -> Expr:
        return add(*[mul(Float(number_from_fraction(c), precision=prec),
                         *[pow(g, Integer(k)) for g, k in zip(gens, m) if k]) for m, c in terms.items()])

    return Float(number_from_fraction(_round_to_prec(coeff, prec)), precision=prec), \
        [(expr_of(terms), k) for terms, k in factors]


# PORTED FROM SYMPY: sympy/polys/factortools.py:dup_factor_list, dup_zz_i_factor, dup_qq_i_factor
def _gaussian_factor_list(base: Expr) -> tuple[Expr, list[tuple[Expr, int]]]:
    """The factorisation of a polynomial with Gaussian coefficients over
    ``ZZ_I`` (or ``QQ_I``): over ``QQ<I>`` by Trager's algorithm, the
    contents of the factors made Gaussian integers as SymPy does (several
    generators: :mod:`~sympy_extras.polys.gaussianfactor`);
    ``CoercionFailed`` for other coefficients."""
    from fractions import Fraction as _Fraction
    from .domains.algebraicfield import AlgebraicField
    from .factortools import dup_ext_factor
    gens, (rep,) = _joint_rings_gaussian([base])
    if not gens:
        raise CoercionFailed('no generator')
    integral = all(c.x.denominator == 1 and c.y.denominator == 1 for c in rep.values())
    if len(gens) > 1:
        from .gaussianfactor import gaussian_factor_list
        ring = PolyRing(QQ_I, _internal_names(len(gens), 0))
        coefficient, multivariate = gaussian_factor_list(ring.from_dict(rep), integral)
        return QQ_I.to_expr(coefficient), [(_gaussian_expr_of(dict(g.terms), gens), k) for g, k in multivariate]
    degree = max(m[0] for m in rep)
    dense: list[GaussianRational] = [rep.get((degree - i,), QQ_I.zero) for i in range(degree + 1)]
    # dup_terms_gcd
    j = 0
    while dense and not dense[-1]:
        dense.pop()
        j += 1
    # dup_primitive (the content over ZZ_I, 1 over the field QQ_I)
    cont = QQ_I.one
    if integral:
        content = ZZ_I.zero
        for c in dense:
            content = ZZ_I.gcd(content, ZZ_I.convert(c, QQ_I))
            if content == ZZ_I.one:
                break
        cont = QQ_I.convert(content, ZZ_I)
        dense = [QQ_I.exquo(c, cont) for c in dense]
    # dup_qq_i_factor: the factorisation over QQ<I>
    K1 = AlgebraicField([_Fraction(1), _Fraction(0), _Fraction(1)], I)
    lc, ext_factors = dup_ext_factor([K1.from_expr(QQ_I.to_expr(c)) for c in dense], K1)
    coeff = QQ_I.from_expr(K1.to_expr(lc))
    found: list[tuple[list[GaussianRational], int]] = [
        ([QQ_I.from_expr(K1.to_expr(c)) for c in fac], k) for fac, k in ext_factors]
    if integral:
        # dup_zz_i_factor: the denominators and the contents of the factors
        # over ZZ_I moved into the coefficient
        new_found: list[tuple[list[GaussianRational], int]] = []
        for fac, k in found:
            fac_denom = ZZ_I.one
            for c in fac:
                fac_denom = ZZ_I.lcm(fac_denom, ZZ_I.denom_of(c))
            d = QQ_I.convert(fac_denom, ZZ_I)
            numerators = [ZZ_I.convert(QQ_I.mul(c, d), QQ_I) for c in fac]
            content = ZZ_I.zero
            for nc in numerators:
                content = ZZ_I.gcd(content, nc)
                if content == ZZ_I.one:
                    break
            primitive = [QQ_I.convert(ZZ_I.exquo(nc, content), ZZ_I) for nc in numerators]
            coeff = QQ_I.exquo(QQ_I.mul(coeff, QQ_I.pow(QQ_I.convert(content, ZZ_I), k)), QQ_I.pow(d, k))
            new_found.append((primitive, k))
        found = new_found
    if j:
        found.insert(0, ([QQ_I.one, QQ_I.zero], j))
    # _sort_factors (multiple): by length, multiplicity, coefficients (SymPy
    # compares Gaussian numbers by their imaginary parts first)
    found.sort(key=lambda item: (len(item[0]), item[1], [(c.y, c.x) for c in item[0]]))
    x = gens[0]

    def expr_of(fac: list[GaussianRational]) -> Expr:
        n = len(fac) - 1
        return add(*[mul(QQ_I.to_expr(c), pow(x, Integer(n - i))) for i, c in enumerate(fac) if c])

    return QQ_I.to_expr(QQ_I.mul(coeff, cont)), [(expr_of(fac), k) for fac, k in found]


# PORTED FROM SYMPY: sympy/polys/rings.py:PolyElement.cancel (the domains ZZ and QQ)
def _cancel_reps(f: dict[Monomial, Fraction], g: dict[Monomial, Fraction], n: int
                 ) -> tuple[PolyElement[int], PolyElement[int], SparsePolynomialRing[int]]:
    """``f/g`` in lowest terms over ``ZZ``: ``(p, q)`` with ``f/g = p/q``,
    the integer denominators cleared and their gcd removed, the leading
    coefficient of ``q`` positive."""
    Z = IntegerPolynomialRing(_internal_names(n, 0))
    if not f:
        return Z.zero, Z.one, Z
    cq = 1
    for c in f.values():
        cq = cq * c.denominator // integer_gcd(cq, c.denominator)
    cp = 1
    for c in g.values():
        cp = cp * c.denominator // integer_gcd(cp, c.denominator)
    F = Z.ring.from_dict({m: int(c * cq) for m, c in f.items()})
    G = Z.ring.from_dict({m: int(c * cp) for m, c in g.items()})
    h = Z.gcd(F, G)
    p, q = F.exquo(h), G.exquo(h)
    common = integer_gcd(cp, cq)
    p, q = p.mul_ground(cp // common), q.mul_ground(cq // common)
    if q.LC < 0:
        p, q = -p, -q
    return p, q, Z


def _expr_of(f: PolyElement[int], gens: tuple[Expr, ...]) -> Expr:
    parts: list[Expr] = []
    for m in sorted(f.terms, reverse=True):
        factors: list[Expr] = [Integer(f.terms[m])]
        factors += [pow(g, Integer(k)) for g, k in zip(gens, m) if k]
        parts.append(mul(*factors))
    return add(*parts)


# PORTED FROM SYMPY: sympy/polys/polytools.py:invert (univariate over QQ, two integers)
def invert(f: ExprLike, g: ExprLike) -> Expr:
    """The inverse of ``f`` modulo ``g``: for two integers the modular
    inverse, for univariate polynomials the polynomial ``s`` of degree
    below ``g``'s with ``s*f = 1 (mod g)`` over ``QQ`` (``NotInvertible``
    when they are not coprime); ``ValueError`` for several generators, as
    in SymPy. Numbers other than rationals are inverted in their domain
    (``ZZ_I``, ``QQ_I``, ``RR``) as SymPy's ``Domain.invert`` does.

    >>> from sympy_extras.core import Symbol
    >>> x = Symbol('x')
    >>> invert(x**2 - 1, 2*x - 1), invert(3, 7)
    (-4/3, 5)
    """
    from .polyerrors import NotInvertible
    a, b = to_expr(f), to_expr(g)
    if isinstance(a, Integer) and isinstance(b, Integer):
        try:
            return Integer(builtins_pow(a.p, -1, b.p))
        except ValueError:
            raise NotInvertible('zero divisor') from None
    gens, (pf, pg) = _joint_rings([a, b])
    if len(gens) > 1:
        # SymPy's DMP.invert: an inverse modulo a polynomial in several variables is not defined
        raise ValueError('univariate polynomial expected')
    if not gens:
        if not (isinstance(a, Rational) and isinstance(b, Rational)):
            return _invert_numbers(a, b)
        fa, fb = a.as_fraction(), b.as_fraction()
        if fa.denominator == 1 and fb.denominator == 1:
            try:
                return Integer(builtins_pow(fa.numerator, -1, fb.numerator))
            except ValueError:
                raise NotInvertible('zero divisor') from None
        # SymPy's QQ.invert: gcdex(a, b) over QQ gives the gcd
        # gcd(numerators)/lcm(denominators), never 1 when a denominator
        # is not 1 (a prime of that denominator divides the gcd of the
        # numerators, but not the numerator it belongs to)
        raise NotInvertible('zero divisor')
    def dense(terms: dict[Monomial, Fraction]) -> list[Fraction]:
        n = max((m[0] for m in terms), default=0)
        return [terms.get((k,), Fraction(0)) for k in range(n, -1, -1)]
    try:
        h = euclidtools.dup_invert(dense(pf), dense(pg), QQ)
    except (ValueError, ZeroDivisionError):
        raise NotInvertible('zero divisor') from None
    n = len(h) - 1
    return add(*[mul(number_from_fraction(c), pow(gens[0], Integer(n - i))) for i, c in enumerate(h) if c])


# PORTED FROM SYMPY: sympy/polys/polytools.py:invert (construct_domain and Domain.invert of two numbers)
def _invert_numbers(a: Expr, b: Expr) -> Expr:
    """SymPy's ``invert`` of two numbers which are not both rational, in
    the domain ``construct_domain`` gives them: over ``ZZ_I`` the extended
    Euclidean algorithm (``s % b`` when the gcd is ``1``); over ``QQ_I`` no
    inverse (``Field.gcdex``: the gcd ``gcd(numerators)/lcm(denominators)``
    is never ``1`` for an element which is not a Gaussian integer); over
    ``RR`` ``(1/a) % b`` (the gcd of ``RR`` is ``1``); over ``CC`` the
    ``TypeError`` of ``mpc % mpc``. An infinity or ``nan`` is a generator
    for SymPy (the inverse modulo a constant is ``0``)."""
    from sympy_extras.core.numbers import Float, _round_to_prec
    from sympy_extras.core.symbol import Dummy
    from .domains.gaussiandomains import _gaussian_parts
    from .polyerrors import NotInvertible
    infinite = [n for n in (a, b) if n.is_finite is not True]
    if infinite:
        forward: dict[Basic, Basic] = {}
        back: dict[Basic, Basic] = {}
        for n in infinite:
            d = Dummy('g')
            forward[n] = d
            back[d] = n
        found = invert(_expr_of_basic(a._xreplace_evaluated(forward)), _expr_of_basic(b._xreplace_evaluated(forward)))
        return _expr_of_basic(found._xreplace_evaluated(back))
    try:
        ga, gb = QQ_I.from_expr(a), QQ_I.from_expr(b)
    except ValueError:
        pass
    else:
        if all(v.denominator == 1 for v in (ga.x, ga.y, gb.x, gb.y)):
            za, zb = ZZ_I.convert(ga, QQ_I), ZZ_I.convert(gb, QQ_I)
            inverse, _, h = ZZ_I.gcdex(za, zb)
            if h != ZZ_I.one:
                raise NotInvertible('zero divisor')
            return ZZ_I.to_expr(ZZ_I.rem(inverse, zb))
        raise NotInvertible('zero divisor')
    values: list[tuple[Fraction, int]] = []
    for n in (a, b):
        v = _float_or_rational(n)
        if v is None:
            try:
                re, im = _gaussian_parts(n)
            except ValueError:
                raise NotImplementedError('the domain of %s is not supported' % (n,)) from None
            if _float_or_rational(re) is None or _float_or_rational(im) is None:
                raise NotImplementedError('the domain of %s is not supported' % (n,))
            raise TypeError("unsupported operand type(s) for %: 'mpc' and 'mpc'")
        values.append(v)
    prec = max(p for _, p in values)
    fa, fb = _round_to_prec(values[0][0], prec), _round_to_prec(values[1][0], prec)
    if not fb:
        raise ZeroDivisionError('integer modulo by zero')
    with mpmath.workprec(prec):
        ma = mpmath.mpf(fa.numerator)/fa.denominator
        mb = mpmath.mpf(fb.numerator)/fb.denominator
        result = mpmath.mpf(0) % mb if not fa else (1/ma) % mb
        exact = _mpf_fraction(result)
    return Float(number_from_fraction(exact), precision=prec)


def _mpf_fraction(value: mpmath.mpf) -> Fraction:
    """The exact value of a binary float."""
    man, exp = value.man, value.exp
    m = int(man)
    e = int(exp)
    return Fraction(m * 2**e) if e >= 0 else Fraction(m, 2**-e)


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel (a pair)
def cancel_fraction(p: ExprLike, q: ExprLike) -> tuple[Expr, Expr]:
    """``p/q`` in lowest terms as a pair ``(P, Q)`` of expanded
    polynomials in the generators of ``p`` and ``q``, the leading
    coefficient of ``Q`` positive (SymPy's ``cancel((p, q))`` with the
    constant folded into ``P``)."""
    gens, (f, g) = _joint_rings([_expanded(to_expr(p)), _expanded(to_expr(q))])
    if not gens:
        return _expanded(mul(to_expr(p), pow(to_expr(q), Integer(-1)))), Integer(1)
    if not g:
        raise ZeroDivisionError('cancel of %s/0' % (p,))
    P, Q, _ = _cancel_reps(f, g, len(gens))
    return _expr_of(P, gens), _expr_of(Q, gens)


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel
def cancel(f: ExprLike, *gens: Expr) -> Expr:
    """A rational function in lowest terms, ``P/Q`` with ``P`` and ``Q``
    expanded polynomials without common factors, the leading coefficient
    of ``Q`` positive; the generators are every non-numeric factor of the
    numerator and the denominator (``sin(x)``, ``sqrt(2)``), or ``gens``
    with the other factors in the coefficients (``cancel((x*y + x)/(y + 1),
    x)`` is ``x``; the coefficients stay polynomials of their own:
    ``x*(y + 1)``). Expressions with a ``Piecewise`` are cancelled argument
    by argument.

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> cancel((x**2 - 1)/(x - 1))
    x + 1
    >>> cancel((2*x**2 - 2)/(1 - x**2*y) + 1/x)
    (-2*x**3 + x**2*y + 2*x - 1)/(x**3*y - x)
    >>> cancel((-x*y - y)/(x*y + x), x), cancel(x/y + 1, x)
    ((-x*y - y)/(x*(y + 1)), (x + y)/y)

    Gaussian coefficients are cancelled over ``ZZ_I``, floats over ``RR``
    and ``CC``, other numbers (``zoo``) over ``EX``.
    """
    from sympy_extras.simplify.signsimp import signsimp
    e = signsimp(to_expr(f))
    if isinstance(e, Number) or e.is_commutative is False:
        return e
    from sympy_extras.functions.elementary.piecewise import Piecewise
    if e.has(Piecewise):
        rebuilt = e._evaluated(tuple(cancel(a, *gens) if isinstance(a, Expr) else a for a in e.args))
        if not isinstance(rebuilt, Expr):
            raise TypeError('an expression is expected')
        return rebuilt
    from sympy_extras.core.exprtools import factor_terms
    factored = factor_terms(e, radical=True)
    p, q = factored.as_numer_denom()
    if gens:
        return _cancel_with_gens(_expanded(p), _expanded(q), e, factored, gens)
    return _cancel_expanded(_expanded(p), _expanded(q), e, factored)


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel (sring with gens over ZZ[params], QQ[params])
def _cancel_with_gens(p: Expr, q: Expr, e: Expr, factored: Expr, gens: tuple[Expr, ...]) -> Expr:
    """``cancel(f, *gens)``: the polynomials in ``gens`` over the polynomial
    ring of the other generators (the gcd over ``ZZ[params][gens]`` is the
    one over ``ZZ[gens, params]``, the leading coefficient of the
    denominator made positive in the order of ``gens`` first), each
    coefficient printed as a polynomial of its own."""
    try:
        joint, (fp, fq) = _joint_rings([p, q])
    except CoercionFailed:
        # coefficients in a domain other than ZZ[...] and QQ[...] (ZZ_I[y],
        # RR[y], EX): the polynomials in gens over EX
        return _cancel_gens_ex(p, q, gens)
    joint_symbols: set[Symbol] = set()
    for j in joint:
        joint_symbols |= j.free_symbols
    if any(g not in joint and g.free_symbols & joint_symbols for g in gens):
        # a generator which is not one of the factors (x**2 for x): the
        # other factors are coefficients, over EX
        return _cancel_gens_ex(p, q, gens)
    if not joint:
        return _expanded(factored)
    P, Q, Z = _cancel_reps(fp, fq, len(joint))
    gen_index = [joint.index(g) if g in joint else -1 for g in gens]
    param_index = [i for i, g in enumerate(joint) if g not in gens]

    def split(m: Monomial) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (tuple(m[i] if i >= 0 else 0 for i in gen_index), tuple(m[i] for i in param_index))

    # the canonical unit of ZZ[params]: the sign of the leading coefficient
    # in the order of gens first
    lead = max(Q.terms, key=split)
    if Q.terms[lead] < 0:
        P, Q = -P, -Q

    def expr_of(f: PolyElement[int]) -> Expr:
        grouped: dict[tuple[int, ...], list[Expr]] = {}
        for m, c in f.terms.items():
            gm, pm = split(m)
            grouped.setdefault(gm, []).append(
                mul(Integer(c), *[pow(joint[i], Integer(k)) for i, k in zip(param_index, pm) if k]))
        parts: list[Expr] = []
        for gm, cs in grouped.items():
            parts.append(mul(add(*cs), *[pow(g, Integer(k)) for g, k in zip(gens, gm) if k]))
        return add(*parts)

    return mul(expr_of(P), pow(expr_of(Q), Integer(-1)))


def _cancel_gens_ex(p: Expr, q: Expr, gens: tuple[Expr, ...]) -> Expr:
    """``cancel(p/q, *gens)`` with the coefficients (everything which is
    not a power of a generator) taken as elements of ``EX``, where SymPy
    takes the polynomial ring of the composite domain: the same rational
    function, the coefficients as ``EX`` normalises them."""
    from .domains.expressiondomain import EX

    def ex_rep(h: Expr) -> dict[Monomial, Expr]:
        rep: dict[Monomial, Expr] = {}
        for m, cs in _decompose_with_gens(h, gens).items():
            c = EX.normal(add(*cs))
            if not EX.is_zero(c):
                rep[m] = c
        return rep
    fp, fq = ex_rep(p), ex_rep(q)
    if not fq:
        raise ZeroDivisionError('cancel of %s/0' % (p,))
    try:
        P, Q = _cancel_ex(fp, fq, len(gens))
    except PolynomialDivisionFailed:
        return mul(p, pow(q, Integer(-1)))
    return mul(_ex_expr_of(P, gens), pow(_ex_expr_of(Q, gens), Integer(-1)))


def _cancel_expanded(p: Expr, q: Expr, e: Expr, factored: Expr) -> Expr:
    """The cancellation of ``p/q``, numerator and denominator expanded (the
    preprocessing of ``cancel`` done once, as SymPy does)."""
    try:
        gens, (fp, fq) = _joint_rings([p, q])
    except CoercionFailed:
        try:
            ggens, (gp, gq) = _joint_rings_gaussian([p, q])
        except CoercionFailed:
            try:
                rgens, (rp, rq), prec = _joint_rings_real([p, q])
            except CoercionFailed:
                return _cancel_complex_or_ex(p, q, factored)
            if not rgens:
                return _expanded(factored)
            RP, RQ = _cancel_real(rp, rq, len(rgens), prec)
            return mul(_real_expr_of(RP, rgens, prec), pow(_real_expr_of(RQ, rgens, prec), Integer(-1)))
        if not ggens:
            return _expanded(factored)
        GP, GQ = _cancel_gaussian(gp, gq, len(ggens))
        return mul(_gaussian_expr_of(GP, ggens), pow(_gaussian_expr_of(GQ, ggens), Integer(-1)))
    if not gens:
        return _expanded(factored)
    P, Q, _ = _cancel_reps(fp, fq, len(gens))
    return mul(_expr_of(P, gens), pow(_expr_of(Q, gens), Integer(-1)))


def _cancel_complex_or_ex(p: Expr, q: Expr, factored: Expr) -> Expr:
    """The cancellation over ``CC`` (complex floats), else over ``EX``."""
    try:
        cgens, (cp, cq), prec = _joint_rings_complex([p, q])
    except CoercionFailed:
        egens, (ep, eq) = _joint_rings_ex([p, q])
        if not egens:
            return _expanded(factored)
        try:
            EP, EQ = _cancel_ex(ep, eq, len(egens))
        except PolynomialDivisionFailed:
            # a zero EX does not detect (oo - oo): nothing is cancelled
            # (SymPy's cancel raises, unless its factor_terms took the
            # infinity out first)
            return mul(p, pow(q, Integer(-1)))
        return mul(_ex_expr_of(EP, egens), pow(_ex_expr_of(EQ, egens), Integer(-1)))
    if not cgens:
        return _expanded(factored)
    CP, CQ = _cancel_complex(cp, cq, len(cgens), prec)
    return mul(_complex_expr_of(CP, cgens, prec), pow(_complex_expr_of(CQ, cgens, prec), Integer(-1)))


def _expr_of_basic(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected')
    return b


# ---------------------------------------------------------------------
# factorisation of expressions

FactorBase = Union[Poly, Expr]
SymbolicFactors = list[tuple[FactorBase, Expr]]


# PORTED FROM SYMPY: sympy/core/evalf.py:pure_complex
def _pure_complex(v: Expr) -> bool:
    """Whether ``v`` is ``a + b*I`` with numbers ``a``, ``b`` (``b != 0``)."""
    _, t = v.as_coeff_Add()
    if isinstance(t, Number) and t == 0:
        return False
    _, i = t.as_coeff_Mul()
    return i == I


def _base_expr(f: FactorBase) -> Expr:
    return f.as_expr() if isinstance(f, Poly) else f


Method = Literal['factor', 'sqf']


# PORTED FROM SYMPY: sympy/polys/polytools.py:_symbolic_factor_list
def _symbolic_factor_list(expr: Expr, gens: Sequence[ExprLike], method: Method = 'factor'
                          ) -> tuple[Expr, SymbolicFactors]:
    coefficient: Expr = Integer(1)
    factors: SymbolicFactors = []
    for arg in (expr.args if isinstance(expr, Mul) else (expr,)):
        if not isinstance(arg, Expr):
            raise TypeError('an expression is expected')
        if isinstance(arg, Number) or _pure_complex(arg):
            coefficient = mul(coefficient, arg)
            continue
        if isinstance(arg, Pow):
            base, exponent = arg.base, arg.exp
            if isinstance(base, Number) and isinstance(exponent, Number):
                coefficient = mul(coefficient, arg)
                continue
            if isinstance(base, Number):
                factors.append((base, exponent))
                continue
        else:
            base, exponent = arg, Integer(1)
        found: SymbolicFactors
        try:
            p = Poly(base, *gens)
        except GeneratorsNeeded:
            factors.append((_expanded(base), exponent))
            continue
        except CoercionFailed:
            # Gaussian coefficients: SymPy's factorisation over ZZ_I and
            # QQ_I through QQ<I>; with generators, over ZZ_I[params] (the
            # irreducible factors in every symbol are the same)
            if method != 'factor':
                raise
            try:
                c, gaussian = _gaussian_factor_list(base)
            except CoercionFailed:
                try:
                    # float coefficients: SymPy's factorisation over RR
                    c, gaussian = _real_factor_list(base)
                except CoercionFailed:
                    # CC and EX: SymPy's dmp_factor_list raises DomainError,
                    # and Poly.factor_list keeps the polynomial as it is
                    c, gaussian = Integer(1), [(_expanded(base), 1)]
            found = [(g, Integer(k)) for g, k in gaussian]
        else:
            c, poly_factors = p.factor_list() if method == 'factor' else p.sqf_list()
            found = [(g, Integer(k)) for g, k in poly_factors]
        if not (isinstance(c, Integer) and c.p == 1):
            if isinstance(exponent, Integer):
                coefficient = mul(coefficient, pow(c, exponent))
            elif c.is_positive is True:
                factors.append((c, exponent))
            else:
                found.append((c, Integer(1)))
        if isinstance(exponent, Integer) and exponent.p == 1:
            factors.extend(found)
        elif exponent.is_integer is True:
            factors.extend([(g, mul(k, exponent)) for g, k in found])
        else:
            other: SymbolicFactors = []
            for g, k in found:
                if _base_expr(g).is_positive is True:
                    factors.append((g, mul(k, exponent)))
                else:
                    other.append((g, k))
            factors.append((_factors_product(other), exponent))
    if method == 'sqf':
        factors = [(_product_of([g for g, m in factors if m == k]), k)
                   for k in sorted(dict.fromkeys(k for _, k in factors), key=_exponent_key)]
    collected: dict[FactorBase, Expr] = {}
    for g, k in factors:
        collected[g] = add(collected[g], k) if g in collected else k
    return coefficient, list(collected.items())


def _product_of(bases: Sequence[FactorBase]) -> FactorBase:
    """SymPy's ``reduce(mul, bases)``: a polynomial when they all are."""
    result = bases[0]
    for b in bases[1:]:
        if isinstance(result, Poly) and isinstance(b, Poly):
            result = result * b
        else:
            result = mul(_base_expr(result), _base_expr(b))
    return result


# PORTED FROM SYMPY: sympy/polys/polytools.py:_factors_product
def _factors_product(factors: SymbolicFactors) -> Expr:
    return mul(*[pow(_base_expr(g), k) for g, k in factors])


RepKey = Union[Fraction, tuple[int, list[tuple[Monomial, Fraction]]], tuple['RepKey', ...]]


def _rep_key(p: Poly) -> RepKey:
    """``poly.rep.to_list()`` as SymPy compares it: nested over the
    generators, the coefficients numbers or, over a polynomial domain,
    ``PolyElement.sort_key``."""
    n = len(p.gens)
    return _rep_level(dict(p._rep.terms), 0, n, bool(p._domain.params))


def _rep_level(terms: dict[Monomial, Fraction], i: int, n: int, params: bool) -> RepKey:
    if i == n:
        if params:
            ordered = sorted(((m[n:], c) for m, c in terms.items()), reverse=True)
            return (len(ordered), ordered)
        return sum(terms.values(), Fraction(0))
    if not terms:
        return ()
    by_degree: dict[int, dict[Monomial, Fraction]] = {}
    for m, c in terms.items():
        by_degree.setdefault(m[i], {})[m] = c
    top = max(by_degree)
    if i == n - 1 and not params:
        return tuple(sum(by_degree.get(d, {}).values(), Fraction(0)) for d in range(top, -1, -1))
    return tuple(_rep_level(by_degree.get(d, {}), i + 1, n, params) for d in range(top, -1, -1))


ExponentKey = tuple[int, Fraction, str]


def _exponent_key(k: Expr) -> ExponentKey:
    if isinstance(k, Rational):
        return (0, k.as_fraction(), '')
    return (1, Fraction(0), str(k))


def _rep_length(p: Poly) -> int:
    return max(p.degree(), 0) + 1 if not p.is_zero else 0


# PORTED FROM SYMPY: sympy/polys/polytools.py:_sorted_factors
def _sorted_factors(factors: list[tuple[Poly, Expr]], method: Method = 'factor') -> list[tuple[Poly, Expr]]:
    if method == 'sqf':
        return sorted(factors, key=lambda item: (_exponent_key(item[1]), _rep_length(item[0]), len(item[0].gens),
                                                 item[0].domain, _rep_key(item[0])))
    return sorted(factors, key=lambda item: (_rep_length(item[0]), len(item[0].gens), _exponent_key(item[1]),
                                             item[0].domain, _rep_key(item[0])))


def _factor_lists(f: ExprLike, gens: Sequence[ExprLike], method: Method = 'factor'
                  ) -> tuple[Expr, list[tuple[Poly, Expr]], list[tuple[Poly, Expr]]]:
    from .rationaltools import together
    numer, denom = together(to_expr(f)).as_numer_denom()
    cp, fp = _symbolic_factor_list(numer, gens, method)
    cq, fq = _symbolic_factor_list(denom, gens, method)
    polys_p = [(g if isinstance(g, Poly) else Poly(g, *gens), k) for g, k in fp]
    polys_q = [(g if isinstance(g, Poly) else Poly(g, *gens), k) for g, k in fq]
    return mul(cp, pow(cq, Integer(-1))), _sorted_factors(polys_p, method), _sorted_factors(polys_q, method)


# PORTED FROM SYMPY: sympy/polys/polytools.py:factor_list, _generic_factor_list
def factor_list(f: PolyLike, *gens: ExprLike) -> tuple[Expr, list[tuple[Expr, Expr]]]:
    """The constant and the irreducible factors with their multiplicities
    of a polynomial expression, sorted as SymPy sorts them; a ``Poly`` is
    factored over its domain.

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> factor_list(2*x**3*y - 2*x*y**3)
    (2, [(x, 1), (y, 1), (x - y, 1), (x + y, 1)])
    """
    if isinstance(f, Poly):
        c, factors = f.factor_list()
        return c, [(g.as_expr(), Integer(k)) for g, k in factors]
    coefficient, fp, fq = _factor_lists(f, gens)
    if fq:
        raise PolynomialError('a polynomial expected, got %s' % (f,))
    return coefficient, [(g.as_expr(), k) for g, k in fp]


def factor_list_fraction(f: ExprLike, *gens: ExprLike) -> tuple[Expr, list[tuple[Expr, Expr]], list[tuple[Expr, Expr]]]:
    """SymPy's ``factor_list(f, frac=True)``: the constant and the factors
    of the numerator and of the denominator."""
    coefficient, fp, fq = _factor_lists(f, gens)
    return coefficient, [(g.as_expr(), k) for g, k in fp], [(g.as_expr(), k) for g, k in fq]


# PORTED FROM SYMPY: sympy/polys/polytools.py:factor, _generic_factor, _symbolic_factor
def factor(f: ExprLike, *gens: ExprLike, deep: bool = False, fraction: bool = True) -> Expr:
    """The expression as a product of irreducible factors over ``ZZ``/``QQ``
    (rational expressions over a common denominator first, see
    :func:`~sympy_extras.polys.rationaltools.together`); with ``deep`` the
    subexpressions too.

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> factor(x**2*y - y**3), factor((x**2 - 1)/(x**2 + 2*x + 1)), factor(x**2 + 1)
    (y*(x - y)*(x + y), (x - 1)/(x + 1), x**2 + 1)
    """
    e = to_expr(f)
    if deep:
        from sympy_extras.core.traversal import bottom_up

        def attempt(b: Basic) -> Basic:
            if not isinstance(b, Expr):
                return b
            factored = factor(b, *gens)
            return factored if isinstance(factored, (Mul, Pow)) else b
        result = bottom_up(e, attempt)
        partials: dict[Basic, Basic] = {}
        for node in result.atoms(Mul, Add):
            if isinstance(node, Expr):
                factored = factor(node, *gens)
                if isinstance(factored, (Mul, Pow)) and factored != node:
                    partials[node] = factored
        replaced = result._xreplace_evaluated(partials) if partials else result
        if not isinstance(replaced, Expr):
            raise TypeError('an expression is expected')
        return replaced
    if e.is_commutative is False:
        from sympy_extras.core.exprtools import factor_nc
        return factor_nc(e)
    return _symbolic_factor(e, gens, fraction)


def _symbolic_factor(e: Expr, gens: Sequence[ExprLike], fraction: bool, method: Method = 'factor') -> Expr:
    from sympy_extras.core.mul import _keep_coeff
    from .rationaltools import together
    coefficient, factors = _symbolic_factor_list(together(e, fraction=fraction), gens, method)
    return _keep_coeff(coefficient, _factors_product(factors))


# PORTED FROM SYMPY: sympy/polys/polytools.py:sqf_list
def sqf_list(f: PolyLike, *gens: ExprLike) -> tuple[Expr, list[tuple[Expr, Expr]]]:
    """The constant and the square-free factors with their
    multiplicities, sorted by multiplicity (SymPy's ``sqf_list``).

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> sqf_list(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)
    (2, [(x + 1, 2), (x + 2, 3)])
    """
    if isinstance(f, Poly):
        c, factors = f.sqf_list()
        return c, [(g.as_expr(), Integer(k)) for g, k in factors]
    coefficient, fp, fq = _factor_lists(f, gens, 'sqf')
    if fq:
        raise PolynomialError('a polynomial expected, got %s' % (f,))
    return coefficient, [(g.as_expr(), k) for g, k in fp]


def sqf_list_fraction(f: ExprLike, *gens: ExprLike) -> tuple[Expr, list[tuple[Expr, Expr]], list[tuple[Expr, Expr]]]:
    """SymPy's ``sqf_list(f, frac=True)``."""
    coefficient, fp, fq = _factor_lists(f, gens, 'sqf')
    return coefficient, [(g.as_expr(), k) for g, k in fp], [(g.as_expr(), k) for g, k in fq]


# PORTED FROM SYMPY: sympy/polys/polytools.py:sqf
def sqf(f: ExprLike, *gens: ExprLike, fraction: bool = True) -> Expr:
    """The square-free factorisation of an expression (SymPy's ``sqf``).

    >>> from sympy_extras.core import Symbol
    >>> x = Symbol('x')
    >>> sqf(2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16)
    2*(x + 1)**2*(x + 2)**3
    """
    return _symbolic_factor(to_expr(f), gens, fraction, 'sqf')
