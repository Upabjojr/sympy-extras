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
``RR``, fraction fields (``ZZ(a)``), ``ZZ_I`` and algebraic extensions;
the division of multivariate polynomials and of polynomials with
symbolic coefficients; multivariate factorisation.

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

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.exprtools import decompose_power
from sympy_extras.core.function import expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Number, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Symbol

from . import densearith, densebasic, euclidtools, factortools, sqfreetools
from .domains import QQ, ZZ
from .domains.sparsepolynomialring import (IntegerPolynomialRing, IteratedPolynomialRing, RationalPolynomialRing,
                                           SparsePolynomialRing)
from .polyerrors import CoercionFailed, GeneratorsError, GeneratorsNeeded, OperationNotSupported, PolynomialError
from .polyutils import dict_from_expr
from .rings import Monomial, PolyElement, PolyRing

__all__ = ['Poly', 'poly', 'degree', 'LC', 'gcd', 'lcm', 'resultant', 'discriminant', 'factor_list', 'sqf_list', 'cancel',
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
def _gen_key(gen: Expr) -> tuple[int, str, int]:
    match = _RE_GEN.match(str(gen))
    name, index = (match.group(1), match.group(2)) if match else (str(gen), '')
    return (_GENS_ORDER.get(name, _MAX_ORDER), name, int(index) if index else 0)


def sort_gens(gens: Sequence[Expr]) -> tuple[Expr, ...]:
    """The generators in SymPy's order (``_sort_gens``)."""
    return tuple(sorted(gens, key=_gen_key))


def _expanded(e: Expr) -> Expr:
    """SymPy's ``_dict_from_expr`` expansion: products and integer powers
    of sums multiplied out."""
    return expand(e, log=False)


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
def _decompose_without_gens_many(exprs: Sequence[Expr]) -> tuple[list[Decomposed], tuple[Expr, ...]]:
    found: set[Expr] = set()
    raws: list[list[tuple[list[Expr], dict[Expr, int]]]] = []
    for e in exprs:
        raw: list[tuple[list[Expr], dict[Expr, int]]] = []
        for term in _terms_of(_expanded(e)):
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
            return self._gens == other._gens and self.as_expr() == other.as_expr()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(('Poly', self._gens, self.as_expr()))

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
        result = self.as_expr().xreplace(substitution)
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
        if self._domain.params and len(self._split()[max(self._split())]) > 1:
            raise OperationNotSupported('monic of a polynomial with a symbolic leading coefficient')
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
        """Division with remainder of univariate polynomials (over ``QQ``
        when the coefficients are integers, as SymPy's ``auto`` does)."""
        a, b = self._unified(other)
        if b.is_zero:
            raise ZeroDivisionError('polynomial division by zero')
        q, r = densearith.dup_div(a._univariate_rational(), b._univariate_rational(), QQ)
        integral = a._domain.integral and all(c.denominator == 1 for c in q + r)
        return a._from_univariate(q, integral), a._from_univariate(r, integral)

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
        n = self.degree()
        if n < 1:
            return Integer(0)
        derivative = self.diff()
        r = self.resultant(derivative)
        sign = Integer((-1) ** (n * (n - 1) // 2))
        lc = self._leading_in_first()
        if isinstance(r, Poly):
            return Poly(_expanded(mul(sign, r.as_expr(), pow(lc, Integer(-1)))), *r.gens)
        return _expanded(mul(sign, r, pow(lc, Integer(-1))))

    def _leading_in_first(self) -> Expr:
        """The leading coefficient in the first generator, an expression in
        the others and the parameters."""
        top = self.degree()
        parts: list[Expr] = []
        for m, c in self.terms():
            if m[0] == top:
                parts.append(mul(c, *[pow(g, Integer(k)) for g, k in zip(self._gens[1:], m[1:]) if k]))
        return add(*parts)

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


def gcd(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    """The gcd of two polynomial expressions, as an expression."""
    a = _as_poly(f, gens) if gens or isinstance(f, Poly) else Poly(add(to_expr(f) if not isinstance(f, Poly) else f.as_expr(), Integer(0)), *_common_gens(f, g))
    return a.gcd(g).as_expr()


def lcm(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    a = _as_poly(f, gens) if gens or isinstance(f, Poly) else Poly(to_expr(f) if not isinstance(f, Poly) else f.as_expr(), *_common_gens(f, g))
    return a.lcm(g).as_expr()


def _common_gens(f: PolyLike, g: PolyLike) -> tuple[Expr, ...]:
    ef = f.as_expr() if isinstance(f, Poly) else to_expr(f)
    eg = g.as_expr() if isinstance(g, Poly) else to_expr(g)
    _, gens = _decompose_without_gens(add(ef, eg))
    return gens


def resultant(f: PolyLike, g: PolyLike, *gens: ExprLike) -> Expr:
    r = _as_poly(f, gens).resultant(g if isinstance(g, Poly) else Poly(g, *(gens or _as_poly(f, gens).gens)))
    return r.as_expr() if isinstance(r, Poly) else r


def discriminant(f: PolyLike, *gens: ExprLike) -> Expr:
    d = _as_poly(f, gens).discriminant()
    return d.as_expr() if isinstance(d, Poly) else d


# ---------------------------------------------------------------------
# rational functions

def _joint_rings(exprs: Sequence[Expr]) -> tuple[tuple[Expr, ...], list[dict[Monomial, Fraction]]]:
    """SymPy's ``sring`` of several expressions without generators: every
    factor which is not a number is a generator (the generators of all the
    expressions together, sorted), the coefficients rational (``ZZ`` or
    ``QQ``; ``I`` and floats are not ported yet)."""
    decomposed, gens = _decompose_without_gens_many(exprs)
    result: list[dict[Monomial, Fraction]] = []
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


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel (a pair)
def cancel_fraction(p: ExprLike, q: ExprLike) -> tuple[Expr, Expr]:
    """``p/q`` in lowest terms as a pair ``(P, Q)`` of expanded
    polynomials in the generators of ``p`` and ``q``, the leading
    coefficient of ``Q`` positive (SymPy's ``cancel((p, q))`` with the
    constant folded into ``P``)."""
    gens, (f, g) = _joint_rings([_expanded(to_expr(p)), _expanded(to_expr(q))])
    if not g:
        raise ZeroDivisionError('cancel of %s/0' % (p,))
    if not gens:
        return number_from_fraction(f.get((), Fraction(0)) / g[()]), Integer(1)
    P, Q, _ = _cancel_reps(f, g, len(gens))
    return _expr_of(P, gens), _expr_of(Q, gens)


# PORTED FROM SYMPY: sympy/polys/polytools.py:cancel
def cancel(f: ExprLike) -> Expr:
    """A rational function in lowest terms, ``P/Q`` with ``P`` and ``Q``
    expanded polynomials without common factors, the leading coefficient
    of ``Q`` positive; the generators are every non-numeric factor of the
    numerator and the denominator (``sin(x)``, ``sqrt(2)``). Expressions
    with a ``Piecewise`` are cancelled argument by argument.

    >>> from sympy_extras.core import Symbol
    >>> x, y = Symbol('x'), Symbol('y')
    >>> cancel((x**2 - 1)/(x - 1))
    x + 1
    >>> cancel((2*x**2 - 2)/(1 - x**2*y) + 1/x)
    (-2*x**3 + x**2*y + 2*x - 1)/(x**3*y - x)

    Not ported yet: the domains ``ZZ_I``, ``RR`` and ``EX``.
    """
    from sympy_extras.simplify.signsimp import signsimp
    e = signsimp(to_expr(f))
    if isinstance(e, Number) or e.is_commutative is False:
        return e
    from sympy_extras.functions.elementary.piecewise import Piecewise
    if e.has(Piecewise):
        rebuilt = e._evaluated(tuple(cancel(a) if isinstance(a, Expr) else a for a in e.args))
        if not isinstance(rebuilt, Expr):
            raise TypeError('an expression is expected')
        return rebuilt
    from sympy_extras.core.exprtools import factor_terms
    factored = factor_terms(e, radical=True)
    p, q = factored.as_numer_denom()
    gens, (fp, fq) = _joint_rings([_expanded(p), _expanded(q)])
    if not gens:
        return _expanded(factored)
    P, Q, _ = _cancel_reps(fp, fq, len(gens))
    return mul(_expr_of(P, gens), pow(_expr_of(Q, gens), Integer(-1)))


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
        try:
            p = Poly(base, *gens)
        except GeneratorsNeeded:
            factors.append((_expanded(base), exponent))
            continue
        c, poly_factors = p.factor_list() if method == 'factor' else p.sqf_list()
        found: SymbolicFactors = [(g, Integer(k)) for g, k in poly_factors]
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
