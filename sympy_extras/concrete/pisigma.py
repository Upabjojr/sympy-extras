r"""$\Pi\Sigma$-fields and Karr's algorithm for first order difference equations.

A $\Pi\Sigma$-field (Karr, 1981) is a tower of fields

.. math::

    \mathbb{C} \subset \mathbb{C}(k) \subset \mathbb{C}(k)(t_1) \subset
    \cdots \subset \mathbb{C}(k)(t_1, \ldots, t_m)

with an automorphism $\sigma$ (the shift $k \to k + 1$) such that
$\sigma(k) = k + 1$ and every generator is either a *$\Pi$-extension*,
$\sigma(t) = \alpha t$ with $\alpha \neq 0$ in the field below (products
like $k!$, $2^k$, $\binom{2k}{k}$), or a *$\Sigma$-extension*,
$\sigma(t) = t + \beta$ with $\beta$ in the field below (sums like the
harmonic numbers $H_k$, $H_k^{(2)}$ or nested sums). The constants
$\mathbb{C}$ are the rational numbers, possibly with parameters (symbols
different from the summation index) adjoined.

Indefinite summation of $f$ in such a field means finding $g$ with
$\sigma(g) - g = f$, so that $\sum_{k=a}^{b} f(k) = g(b + 1) - g(a)$.
Karr's algorithm solves, more generally, the *parameterized first order
equation*

.. math::

    \sigma(g) - a\, g = c_1 f_1 + \cdots + c_r f_r

for $g$ in the field and constants $c_i$, by recursion over the tower:
in the top generator $t$ the unknown $g$ is a polynomial ($\Sigma$-case) or a
Laurent polynomial ($\Pi$-case) in $t$ whose degree is bounded, and the
coefficients are found from the top down by solving equations of the same
shape in the field below; at the bottom, in $\mathbb{C}(k)$, the rational
solutions are found with Abramov's universal denominator and a degree
bound for the polynomial part. The linear systems are solved over the
constants, so the constants $c_i$ come out together with $g$.

This implementation has one restriction with respect to the general
theory: in a $\Sigma$-extension $t$ only right hand sides which are
polynomial in $t$ are handled, and in a $\Pi$-extension only Laurent
polynomials in $t$ (denominators like $1/(2^k + 1)$ or $1/H_k$ are not).
For $\Sigma$-extensions this is no restriction on the solutions: by a
theorem of Karr the solution $g$ is then a polynomial in $t$ as well, so
an answer ``no solution`` is a proof that no closed form exists in the
field.

Representation
==============

The module is written on the typed polynomial layer, without SymPy. The
elements of $\mathbb{C}(k)(t_1, \ldots, t_m)$ are elements of one field of
rational functions over $\mathbb{Q}$
(:class:`~sympy_extras.polys.domains.sparsefractionfield.SparseFractionField`)
in the variables ``params + [k, t_1, ..., t_m]``: the parameters are
variables of the ring rather than a field of coefficients, so that the
gcds are the fast ones over $\mathbb{Z}$ and an element in lowest terms
(integer coefficients without a common content, positive leading
coefficient of the denominator) is canonical. The shift acts on the
variables from ``k`` on. The constants have a type of their own, the
rationals (``Fraction``) without parameters and the field
$\mathbb{Q}(\mathrm{params})$ otherwise, given by a :class:`Constants`
object which embeds them in the field; :class:`PiSigmaField` is generic
over that type and :class:`PiSigmaTower` is the part which does not
depend on it. The fields are immutable: adjoining an extension gives a
new field (``add_sigma``, ``add_pi``), whose elements do not mix with
those of the old one.

References
==========

.. [1] M. Karr, Summation in finite terms, J. ACM 28 (1981) 305-350.
.. [2] M. Karr, Theory of summation in finite terms, J. Symbolic
       Computation 1 (1985) 303-315.
.. [3] C. Schneider, Symbolic summation assists combinatorics, Sem. Lothar.
       Combin. 56 (2007), B56b.
.. [4] S. A. Abramov, Rational solutions of linear difference and
       q-difference equations with polynomial coefficients, ISSAC 1995.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction
from itertools import count
from math import comb, lcm as integer_lcm
from typing import Callable, Final, Generic, Iterable, Optional, Sequence, TypeVar, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.integerring import ZZ
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.multifactor import zz_factor
from sympy_extras.polys.rings import Monomial, PolyElement, PolyRing

__all__ = ['PiSigmaTower', 'PiSigmaField', 'PiSigmaElement', 'Extension', 'Constants', 'RationalConstants',
           'ParametricConstants', 'Solution', 'Solutions', 'rational_pisigma_field', 'parametric_pisigma_field']

C = TypeVar('C')

#: an element of the field of rational functions in all the variables
Rep = SparseFracElement[Fraction]
#: a polynomial in all the variables
Poly = PolyElement[Fraction]
#: the image of a variable under the shift, as a numerator and a denominator
Image = tuple[Poly, Poly]

#: the identifiers of the fields, for the names of the generators
_field_ids = count()


class PiSigmaElement:
    """An element of a $\\Pi\\Sigma$-field: a rational function in lowest
    terms (``rep``) of the field of the tower (``domain``), with the
    arithmetic operators (integers are converted). Equal elements have
    the same representation."""

    __slots__ = ('domain', 'rep')

    def __init__(self, domain: SparseFractionField[Fraction], rep: Rep) -> None:
        self.domain: Final[SparseFractionField[Fraction]] = domain
        self.rep: Final[Rep] = rep

    def _other(self, other: Union[PiSigmaElement, int]) -> Rep:
        if isinstance(other, int):
            return self.domain.from_int(other)
        if other.domain is not self.domain and other.domain != self.domain:
            raise ValueError("elements of different fields")
        return other.rep

    def _new(self, rep: Rep) -> PiSigmaElement:
        return PiSigmaElement(self.domain, rep)

    @property
    def numer(self) -> Poly:
        return self.rep.numer

    @property
    def denom(self) -> Poly:
        return self.rep.denom

    def __add__(self, other: Union[PiSigmaElement, int]) -> PiSigmaElement:
        return self._new(self.domain.add(self.rep, self._other(other)))

    def __radd__(self, other: int) -> PiSigmaElement:
        return self._new(self.domain.add(self._other(other), self.rep))

    def __sub__(self, other: Union[PiSigmaElement, int]) -> PiSigmaElement:
        return self._new(self.domain.sub(self.rep, self._other(other)))

    def __rsub__(self, other: int) -> PiSigmaElement:
        return self._new(self.domain.sub(self._other(other), self.rep))

    def __mul__(self, other: Union[PiSigmaElement, int]) -> PiSigmaElement:
        return self._new(self.domain.mul(self.rep, self._other(other)))

    def __rmul__(self, other: int) -> PiSigmaElement:
        return self._new(self.domain.mul(self._other(other), self.rep))

    def __truediv__(self, other: Union[PiSigmaElement, int]) -> PiSigmaElement:
        return self._new(self.domain.exquo(self.rep, self._other(other)))

    def __rtruediv__(self, other: int) -> PiSigmaElement:
        return self._new(self.domain.exquo(self._other(other), self.rep))

    def __neg__(self) -> PiSigmaElement:
        return self._new(self.domain.neg(self.rep))

    def __pow__(self, n: int) -> PiSigmaElement:
        return self._new(self.domain.pow(self.rep, n))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.rep == self.domain.from_int(other)
        return isinstance(other, PiSigmaElement) and self.rep == other.rep

    def __hash__(self) -> int:
        return hash(self.rep)

    def __bool__(self) -> bool:
        return bool(self.rep.numer.terms)

    def __repr__(self) -> str:
        return 'PiSigmaElement(%s)' % self.domain.to_str(self.rep)


def _pad(p: Poly, ring: PolyRing[Fraction]) -> Poly:
    """``p`` in a ring with more variables (the new ones last)."""
    extra = (0,)*(ring.ngens - p.ring.ngens)
    return PolyElement({m + extra: c for m, c in p.terms.items()}, ring)


def _truncate(p: Poly, ring: PolyRing[Fraction]) -> Poly:
    """``p``, which does not involve the last variables, in the ring of
    the first ones."""
    n = ring.ngens
    return PolyElement({m[:n]: c for m, c in p.terms.items()}, ring)


# ----------------------------------------------------------------------
# the constants


class Constants(ABC, Generic[C]):
    """The field of constants of a $\\Pi\\Sigma$-field, with its embedding
    in the field of rational functions of the tower (whose first
    variables are the parameters)."""

    def __init__(self, domain: Domain[C], params: Sequence[Symbol]) -> None:
        self.domain: Final[Domain[C]] = domain
        self.params: Final[tuple[Symbol, ...]] = tuple(params)

    @abstractmethod
    def embed(self, c: C, field: SparseFractionField[Fraction]) -> Rep:
        """The constant as an element of ``field``."""

    @abstractmethod
    def restrict(self, f: Rep) -> C:
        """The constant equal to ``f``, which involves the parameters
        only."""

    @abstractmethod
    def from_poly(self, p: Poly) -> C:
        """The constant equal to the polynomial ``p`` in the parameters."""

    @abstractmethod
    def rational(self, c: C) -> Optional[Fraction]:
        """The value of ``c`` if it is a rational number, else ``None``."""


class RationalConstants(Constants[Fraction]):
    """The constants $\\mathbb{Q}$ (no parameters)."""

    def __init__(self) -> None:
        super().__init__(QQ, ())

    def embed(self, c: Fraction, field: SparseFractionField[Fraction]) -> Rep:
        R = field.ring.ring
        return SparseFracElement(R.ground(Fraction(c.numerator)), R.ground(Fraction(c.denominator)))

    def restrict(self, f: Rep) -> Fraction:
        return f.numer.LC/f.denom.LC

    def from_poly(self, p: Poly) -> Fraction:
        return p.coeff(p.ring.zero_monom)

    def rational(self, c: Fraction) -> Optional[Fraction]:
        return c


class ParametricConstants(Constants[SparseFracElement[Fraction]]):
    """The constants $\\mathbb{Q}(\\mathrm{params})$."""

    def __init__(self, params: Sequence[Symbol]) -> None:
        if not params:
            raise ValueError("the parametric constants need parameters (RationalConstants otherwise)")
        self.field: Final[SparseFractionField[Fraction]] = SparseFractionField(
            RationalPolynomialRing([p.name for p in params]))
        super().__init__(self.field, params)

    def embed(self, c: SparseFracElement[Fraction], field: SparseFractionField[Fraction]) -> Rep:
        R = field.ring.ring
        # in lowest terms in the parameters is in lowest terms in all the
        # variables (the lexicographic order agrees)
        return SparseFracElement(_pad(c.numer, R), _pad(c.denom, R))

    def restrict(self, f: Rep) -> SparseFracElement[Fraction]:
        R = self.field.ring.ring
        return SparseFracElement(_truncate(f.numer, R), _truncate(f.denom, R))

    def from_poly(self, p: Poly) -> SparseFracElement[Fraction]:
        R = self.field.ring.ring
        return self.field.new(_truncate(p, R), R.one)

    def rational(self, c: SparseFracElement[Fraction]) -> Optional[Fraction]:
        if c.numer.is_ground and c.denom.is_ground:
            return c.numer.LC/c.denom.LC
        return None


# ----------------------------------------------------------------------
# the tower


class Extension:
    """A generator of a $\\Pi\\Sigma$-field.

    Attributes
    ==========

    kind : ``'sigma'`` or ``'pi'``
    symbol : Dummy
        The generator as a symbol.
    value : Expr
        The defining expression of $\\alpha$ ($\\Pi$) or $\\beta$ ($\\Sigma$)
        in terms of the summation index and the previous generators.
    expr : Expr
        The sequence the generator stands for, as an expression in the
        summation index (for instance ``harmonic(k)``).
    """

    __slots__ = ('kind', 'symbol', 'value', 'expr')

    def __init__(self, kind: str, symbol: Dummy, value: Expr, expr: Expr) -> None:
        self.kind: Final[str] = kind
        self.symbol: Final[Dummy] = symbol
        self.value: Final[Expr] = value
        self.expr: Final[Expr] = expr

    def __repr__(self) -> str:
        return "Extension(%s, %s, %s)" % (self.kind, self.expr, self.value)


class PiSigmaTower(ABC):
    r"""The tower $\mathbb{C}(k)(t_1, \ldots, t_m)$ with its shift: the
    conversions, the shift and the decompositions, which do not depend on
    the type of the constants (:class:`PiSigmaField` adds the solver).

    The variables of the field are the parameters (indices ``0`` to
    ``P - 1``), ``k`` (index ``P``) and the generators (``t_i`` at
    ``P + i``); the *level* of a variable is its index minus ``P``.
    """

    def __init__(self, k: Symbol, params: Sequence[Symbol], extensions: Sequence[Extension],
                 values: Sequence[Rep], field_id: int) -> None:
        if not isinstance(k, Symbol):
            raise TypeError("the summation index must be a symbol")
        # the names only serve the printing: the variables are told apart
        # by position, the generators get names of their own
        names = [p.name for p in params] + [k.name]
        self.k: Final[Symbol] = k
        self.params: Final[tuple[Symbol, ...]] = tuple(params)
        self.extensions: Final[tuple[Extension, ...]] = tuple(extensions)
        self._id: Final[int] = field_id
        P = len(self.params)
        self._P: Final[int] = P
        names += ['_ps%d_t%d' % (field_id, i) for i in range(1, len(self.extensions) + 1)]
        self.field: Final[SparseFractionField[Fraction]] = SparseFractionField(RationalPolynomialRing(names))
        R = self.field.ring.ring
        self._ring: Final[PolyRing[Fraction]] = R
        #: the defining values alpha or beta, as elements of this field
        self._values: Final[tuple[Rep, ...]] = tuple(
            SparseFracElement(_pad(v.numer, R), _pad(v.denom, R)) for v in values)
        #: the symbols of the variables
        self.symbols: Final[tuple[Symbol, ...]] = (self.params + (k,)
                                                   + tuple(e.symbol for e in self.extensions))
        self._index: Final[dict[Symbol, int]] = {s: i for i, s in enumerate(self.symbols)}
        self.gens: Final[tuple[PiSigmaElement, ...]] = tuple(
            self._element(self.field.from_ring(g)) for g in R.gens[P:])
        self.zero: Final[PiSigmaElement] = self._element(self.field.zero)
        self.one: Final[PiSigmaElement] = self._element(self.field.one)
        gens = R.gens
        images: list[Image] = [(gens[P] + R.one, R.one)]
        for i, (e, v) in enumerate(zip(self.extensions, self._values), start=1):
            t = gens[P + i]
            if e.kind == 'pi':
                images.append((v.numer*t, v.denom))
            else:
                images.append((t*v.denom + v.numer, v.denom))
        #: the images of the variables from k on under the shift
        self._images: Final[list[Image]] = images
        # lazily filled caches
        self._inverse_images: list[Image] = []
        self._power_cache: dict[tuple[bool, int, int, bool], Poly] = {}

    def _element(self, rep: Rep) -> PiSigmaElement:
        return PiSigmaElement(self.field, rep)

    @property
    def level(self) -> int:
        """The number of extensions."""
        return len(self.extensions)

    # ------------------------------------------------------------------
    # conversions

    def from_expr(self, expr: Union[Expr, int]) -> PiSigmaElement:
        """Convert a rational expression in the index, the parameters and
        the generator symbols to an element of the field (``ValueError``
        otherwise)."""
        if isinstance(expr, int):
            return self._element(self.field.from_int(expr))
        return self._element(self._rep_from_expr(expr))

    def _rep_from_expr(self, e: Expr) -> Rep:
        F = self.field
        if isinstance(e, Symbol):
            i = self._index.get(e)
            if i is None:
                raise ValueError("%s is not a variable of the field" % (e,))
            return F.from_ring(self._ring.gens[i])
        if isinstance(e, Rational):
            return self._rational(Fraction(e.p, e.q))
        if isinstance(e, Add):
            result = F.zero
            for a in e.args:
                result = F.add(result, self._rep_from_expr(a))
            return result
        if isinstance(e, Mul):
            result = F.one
            for a in e.args:
                result = F.mul(result, self._rep_from_expr(a))
            return result
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            base = self._rep_from_expr(e.base)
            if e.exp.p < 0 and not base.numer.terms:
                raise ValueError("division by zero in %s" % (e,))
            return F.pow(base, e.exp.p)
        raise ValueError("%s is not a rational function of %s" % (e, ', '.join(str(s) for s in self.symbols)))

    def from_fraction(self, c: Fraction) -> PiSigmaElement:
        """The rational number ``c`` as an element."""
        return self._element(self._rational(c))

    def _rational(self, c: Fraction) -> Rep:
        return SparseFracElement(self._ring.ground(Fraction(c.numerator)), self._ring.ground(Fraction(c.denominator)))

    def _poly_expr(self, p: Poly, images: Sequence[Expr]) -> Expr:
        parts: list[Expr] = []
        for m, c in p.terms.items():
            factors: list[Expr] = [number_from_fraction(c)]
            for s, e in zip(images, m):
                if e:
                    factors.append(pow(s, Integer(e)) if e > 1 else s)
            parts.append(mul(*factors))
        return add(*parts) if parts else Integer(0)

    def to_expr(self, f: PiSigmaElement, substitute: bool = True) -> Expr:
        """Convert an element to an expression; with ``substitute`` the
        generators are replaced by the sequences they stand for (and the
        common factors are taken out)."""
        images: list[Expr] = [s for s in self.params]
        images.append(self.k)
        if substitute:
            images += [e.expr for e in self.extensions]
        else:
            images += [e.symbol for e in self.extensions]
        numer = self._poly_expr(f.numer, images)
        expr = numer
        if f.denom != self._ring.one:
            expr = mul(numer, pow(self._poly_expr(f.denom, images), Integer(-1)))
        if substitute:
            expr = factor_terms(expr)
        return expr

    # ------------------------------------------------------------------
    # extensions

    @abstractmethod
    def extend(self, kind: str, value: PiSigmaElement, expr: Expr, name: Optional[str] = None
               ) -> tuple[PiSigmaTower, Dummy]:
        """The field with a new generator of the given kind (``'sigma'``
        or ``'pi'``) and its symbol."""

    def _extension(self, kind: str, value: PiSigmaElement, expr: Expr, name: Optional[str]) -> Extension:
        if kind not in ('sigma', 'pi'):
            raise ValueError("the kind of an extension is 'sigma' or 'pi'")
        if kind == 'pi' and not value:
            raise ValueError("alpha must be nonzero")
        symbol = Dummy(name or 't%d' % (len(self.extensions) + 1))
        return Extension(kind, symbol, self.to_expr(value, substitute=False), expr)

    # ------------------------------------------------------------------
    # the shift

    def sigma(self, f: PiSigmaElement, power: int = 1) -> PiSigmaElement:
        """The shift $\\sigma^{j}(f)$ for an integer ``j = power`` (negative
        powers are the inverse shift)."""
        rep = f.rep
        inverse = power < 0
        images = self._inverse_gens() if inverse else self._images
        for _ in range(abs(power)):
            rep = self._sigma_rep(rep, images, inverse)
        return self._element(rep)

    def _inverse_gens(self) -> list[Image]:
        """Images of the variables under the inverse shift, computed on
        first use: $\\sigma^{-1}(k) = k - 1$, $\\sigma^{-1}(t) =
        t/\\sigma^{-1}(\\alpha)$ or $t - \\sigma^{-1}(\\beta)$."""
        images = self._inverse_images
        if images:
            return images
        R, P = self._ring, self._P
        gens = R.gens
        images.append((gens[P] - R.one, R.one))
        for i, (e, v) in enumerate(zip(self.extensions, self._values), start=1):
            w = self._sigma_rep(v, images, True)
            t = gens[P + i]
            if e.kind == 'pi':
                images.append((t*w.denom, w.numer))
            else:
                images.append((t*w.denom - w.numer, w.denom))
        return images

    def _power(self, images: list[Image], inverse: bool, i: int, e: int, numerator: bool) -> Poly:
        key = (inverse, i, e, numerator)
        value = self._power_cache.get(key)
        if value is None:
            image = images[i - self._P]
            value = (image[0] if numerator else image[1])**e
            self._power_cache[key] = value
        return value

    def _sigma_rep(self, f: Rep, images: list[Image], inverse: bool) -> Rep:
        """The image of ``f`` under the substitution of the variables from
        ``k`` on by ``images`` (numerators and denominators): both the
        numerator and the denominator of ``f`` are brought to the common
        denominator of the images, which then cancels."""
        R, P = self._ring, self._P
        n = R.ngens
        one = R.one
        num, den = f.numer, f.denom
        degrees = [0]*n
        for i in range(P, n):
            if i - P < len(images) and images[i - P][1] != one:
                degrees[i] = max(num.degree_in(i), den.degree_in(i))
        products: dict[Monomial, Poly] = {}
        zeros = (0,)*(n - P)

        def image(p: Poly) -> Poly:
            terms: dict[Monomial, Fraction] = {}
            for m, c in p.terms.items():
                tail = m[P:]
                product = products.get(tail)
                if product is None:
                    product = one
                    for i in range(P, n):
                        e = m[i]
                        if e:
                            product = product*self._power(images, inverse, i, e, True)
                        d = degrees[i] - e
                        if d > 0:
                            product = product*self._power(images, inverse, i, d, False)
                    products[tail] = product
                head = m[:P] + zeros
                for mm, cc in product.terms.items():
                    key = tuple(a + b for a, b in zip(mm, head))
                    value = terms.get(key, Fraction(0)) + c*cc
                    if value:
                        terms[key] = value
                    else:
                        terms.pop(key, None)
            return PolyElement(terms, R)

        return self.field.new(image(num), image(den))

    def shift_poly(self, p: Poly, j: int) -> Poly:
        """$\\sigma^j$ of a polynomial in ``k`` and the parameters, for any
        integer ``j``: ``k`` replaced by ``k + j``."""
        P = self._P
        terms: dict[Monomial, Fraction] = {}
        for m, c in p.terms.items():
            e = m[P]
            for r in range(e + 1):
                key = m[:P] + (r,) + m[P + 1:]
                value = terms.get(key, Fraction(0)) + c*comb(e, r)*j**(e - r)
                if value:
                    terms[key] = value
                else:
                    terms.pop(key, None)
        return PolyElement(terms, self._ring)

    # ------------------------------------------------------------------
    # levels and decompositions

    def level_of(self, f: PiSigmaElement) -> int:
        """The index of the highest generator occurring in ``f``: ``-1`` for
        a constant, ``0`` for an element of $\\mathbb{C}(k)$, ``i`` for an
        element involving $t_i$."""
        P = self._P
        level = P - 1
        for p in (f.numer, f.denom):
            for monom in p.terms:
                for i in range(len(monom) - 1, level, -1):
                    if monom[i]:
                        level = i
                        break
        return level - P

    def is_constant(self, f: PiSigmaElement) -> bool:
        return self.level_of(f) == -1

    def as_poly_in(self, f: PiSigmaElement, level: int, laurent: bool = False) -> Optional[dict[int, PiSigmaElement]]:
        """Decompose ``f`` as a polynomial in the generator of the given
        level with coefficients in the field without it: a dict mapping
        exponents to coefficients. With ``laurent`` the denominator may be a
        power of the generator; otherwise it must not involve it. Returns
        ``None`` if ``f`` has no such decomposition."""
        R = self._ring
        i = self._P + level
        num, den = f.numer, f.denom
        shift = 0
        den_exps = {m[i] for m in den.terms}
        if len(den_exps) != 1:
            return None
        [d] = den_exps
        if d:
            if not laurent:
                return None
            shift = d
            den = PolyElement({m[:i] + (0,) + m[i + 1:]: c for m, c in den.terms.items()}, R)
        result: dict[int, dict[Monomial, Fraction]] = {}
        for monom, coeff in num.terms.items():
            e = monom[i] - shift
            result.setdefault(e, {})[monom[:i] + (0,) + monom[i + 1:]] = coeff
        return {e: self._element(self.field.new(PolyElement(terms, R), den)) for e, terms in result.items()}

    def _k_coefficients(self, p: Poly) -> dict[int, Poly]:
        """The coefficients of a polynomial in ``k`` and the parameters as
        polynomials in the parameters, by degree in ``k``."""
        P = self._P
        result: dict[int, dict[Monomial, Fraction]] = {}
        for m, c in p.terms.items():
            result.setdefault(m[P], {})[m[:P] + (0,) + m[P + 1:]] = c
        return {d: PolyElement(terms, self._ring) for d, terms in result.items()}

    def _factors_in_k(self, p: Poly) -> list[Poly]:
        """The irreducible factors of positive degree in ``k`` of a
        polynomial in ``k`` and the parameters, over the constants: the
        factors over ``ZZ`` of the polynomial with the denominators
        cleared."""
        P = self._P
        if p.degree_in(P) <= 0:
            return []
        Z = PolyRing(ZZ, self._ring.names[:P + 1])
        d = 1
        for c in p.terms.values():
            d = integer_lcm(d, c.denominator)
        cleared = PolyElement({m[:P + 1]: int(c*d) for m, c in p.terms.items()}, Z)
        extra = (0,)*(self._ring.ngens - P - 1)
        result: list[Poly] = []
        for factor, _ in zz_factor(cleared)[1]:
            if factor.degree_in(P) > 0:
                result.append(PolyElement({m + extra: Fraction(c) for m, c in factor.terms.items()}, self._ring))
        return result

    # ------------------------------------------------------------------
    # the solver, generic in the constants

    @abstractmethod
    def solve_embedded(self, a: PiSigmaElement, fs: Sequence[PiSigmaElement], level: Optional[int] = None
                       ) -> Optional[list[tuple[list[PiSigmaElement], PiSigmaElement]]]:
        """:meth:`PiSigmaField.solve` with the constants as (constant)
        elements of the field."""

    @abstractmethod
    def telescope(self, f: PiSigmaElement) -> Optional[PiSigmaElement]:
        r"""A solution $g$ of $\sigma(g) - g = f$, or ``None``."""


Solution = tuple[list[C], PiSigmaElement]
"""a solution (c_1, ..., c_r, g) of the parameterized first order equation"""
Solutions = Optional[list[tuple[list[C], PiSigmaElement]]]
"""a basis of solutions, or None when the equation is out of scope"""
#: the right hand sides of a step of the coefficientwise recursion, as a
#: function of the current expression of the constants and coefficients
Rhs = Callable[[list[list[C]], dict[int, list[PiSigmaElement]], int], list[PiSigmaElement]]


class PiSigmaField(PiSigmaTower, Generic[C]):
    r"""A $\Pi\Sigma$-field $\mathbb{C}(k)(t_1, \ldots, t_m)$ with its shift
    and Karr's solver, the constants of type ``C``.

    Parameters
    ==========

    k : Symbol
        The summation index, with $\sigma(k) = k + 1$.
    constants : Constants
        The constants: :class:`RationalConstants` or
        :class:`ParametricConstants` (the parameters adjoined).

    Extensions are added with :meth:`add_sigma` and :meth:`add_pi`, which
    return the new field and the symbol of the generator, and the
    parameterized first order equation is solved with :meth:`solve`;
    :meth:`telescope` is the special case $\sigma(g) - g = f$.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import harmonic, factorial
    >>> from sympy_extras.concrete.pisigma import rational_pisigma_field
    >>> k = Symbol('k')
    >>> F = rational_pisigma_field(k)
    >>> F, h = F.add_sigma(1/(k + 1), harmonic(k))
    >>> g = F.telescope(F.from_expr(h))
    >>> F.to_expr(g)
    k*(harmonic(k) - 1)
    >>> F.sigma(g) - g == F.from_expr(h)
    True
    >>> F, t = F.add_pi(k + 1, factorial(k))
    >>> F.to_expr(F.telescope(F.from_expr(k*t)))
    factorial(k)
    >>> F.telescope(F.from_expr(t/k)) is None
    True
    """

    def __init__(self, k: Symbol, constants: Constants[C], extensions: Sequence[Extension] = (),
                 values: Sequence[Rep] = (), field_id: Optional[int] = None) -> None:
        if len(extensions) != len(values):
            raise ValueError("one value per extension")
        super().__init__(k, constants.params, extensions, values,
                         next(_field_ids) if field_id is None else field_id)
        self.constants: Final[Constants[C]] = constants
        self.C: Final[Domain[C]] = constants.domain

    # ------------------------------------------------------------------
    # extensions

    def extend(self, kind: str, value: PiSigmaElement, expr: Expr, name: Optional[str] = None
               ) -> tuple[PiSigmaField[C], Dummy]:
        extension = self._extension(kind, value, expr, name)
        field = PiSigmaField(self.k, self.constants, self.extensions + (extension,),
                             self._values + (value.rep,), self._id)
        return field, extension.symbol

    def add_sigma(self, beta: Union[PiSigmaElement, Expr, int], expr: Expr, name: Optional[str] = None
                  ) -> tuple[PiSigmaField[C], Dummy]:
        r"""The field with a $\Sigma$-extension $t$, $\sigma(t) = t + \beta$,
        and the symbol of $t$.

        ``beta`` is an element or an expression in the index and the
        existing generators and ``expr`` the sequence $t$ stands for. No
        check is made that the extension is proper.
        """
        return self.extend('sigma', self._as_element(beta), expr, name)

    def add_pi(self, alpha: Union[PiSigmaElement, Expr, int], expr: Expr, name: Optional[str] = None
               ) -> tuple[PiSigmaField[C], Dummy]:
        r"""The field with a $\Pi$-extension $t$, $\sigma(t) = \alpha t$, and
        the symbol of $t$."""
        return self.extend('pi', self._as_element(alpha), expr, name)

    def _as_element(self, value: Union[PiSigmaElement, Expr, int]) -> PiSigmaElement:
        if isinstance(value, PiSigmaElement):
            return value
        return self.from_expr(value)

    # ------------------------------------------------------------------
    # constants

    def to_constant(self, f: PiSigmaElement) -> C:
        """The element of the constant field equal to the constant ``f``."""
        if not self.is_constant(f):
            raise ValueError("%s is not a constant" % (self.to_expr(f, substitute=False),))
        return self.constants.restrict(f.rep)

    def from_constant(self, c: C) -> PiSigmaElement:
        return self._element(self.constants.embed(c, self.field))

    def constant_to_expr(self, c: C) -> Expr:
        """The constant as an expression in the parameters."""
        return self.to_expr(self.from_constant(c), substitute=False)

    def _constant_of(self, p: Poly) -> C:
        return self.constants.from_poly(p)

    def _lc_in_k(self, p: Poly) -> tuple[C, int]:
        """The leading coefficient in ``k`` (a constant) and the degree in
        ``k`` of a polynomial in ``k`` and the parameters."""
        coefficients = self._k_coefficients(p)
        d = max(coefficients)
        return self._constant_of(coefficients[d]), d

    # ------------------------------------------------------------------
    # linear algebra over the constants

    def _nullspace(self, rows: Sequence[Sequence[C]], ncols: int) -> list[list[C]]:
        """Basis of the null space of a matrix over the constants given as a
        list of rows (lists of constants), the free coordinates one."""
        K = self.C
        if not rows:
            return [[K.one if i == j else K.zero for i in range(ncols)] for j in range(ncols)]
        return K.nullspace(rows, ncols)

    # ------------------------------------------------------------------
    # the solver

    def solve(self, a: PiSigmaElement, fs: Sequence[PiSigmaElement], level: Optional[int] = None) -> Solutions[C]:
        r"""Solve $\sigma(g) - a\,g = c_1 f_1 + \cdots + c_r f_r$.

        Returns a basis of the space of solutions $(c_1, \ldots, c_r, g)$
        over the constants, as a list of pairs ``(c, g)`` with ``c`` a list
        of constants and ``g`` an element of the field, or ``None`` if the
        equation is outside the scope of the implementation (a right hand
        side with a $\Sigma$-generator in a denominator, or ``a`` involving
        the top generator). An empty list means that there is no solution
        other than $g = 0$, $c = 0$.

        ``level`` is the field in which ``g`` is searched, by default the
        whole field: the solution may need generators which do not occur
        in the $f_i$, for instance $\sum_k H_k/k = (H_k^2 + H_k^{(2)})/2$.
        """
        if level is None:
            level = self.level
        if level == -1:
            return self._solve_constants(a, fs)
        if level == 0:
            return self._solve_rational(a, fs)
        ext = self.extensions[level - 1]
        if self.level_of(a) >= level:
            return None
        if ext.kind == 'sigma':
            return self._solve_sigma(level, a, fs)
        return self._solve_pi(level, a, fs)

    def solve_embedded(self, a: PiSigmaElement, fs: Sequence[PiSigmaElement], level: Optional[int] = None
                       ) -> Optional[list[tuple[list[PiSigmaElement], PiSigmaElement]]]:
        sols = self.solve(a, fs, level)
        if sols is None:
            return None
        return [([self.from_constant(ci) for ci in c], g) for c, g in sols]

    def telescope(self, f: PiSigmaElement) -> Optional[PiSigmaElement]:
        r"""A solution $g$ of $\sigma(g) - g = f$, or ``None`` if there is
        none in the field (or it could not be decided, see :meth:`solve`)."""
        sols = self.solve(self.one, [f])
        if not sols:
            return None
        for c, g in sols:
            if not self.C.is_zero(c[0]):
                return g/self.from_constant(c[0])
        return None

    # constants: sigma is the identity

    def _solve_constants(self, a_: PiSigmaElement, fs_: Sequence[PiSigmaElement]) -> Solutions[C]:
        K = self.C
        a = self.to_constant(a_)
        fs = [self.to_constant(f) for f in fs_]
        s = len(fs)
        one, zero = K.one, K.zero
        if a != one:
            return [([one if i == l else zero for i in range(s)],
                     self.from_constant(K.exquo(fs[l], K.sub(one, a)))) for l in range(s)]
        sols: list[Solution[C]] = [(v, self.zero) for v in self._nullspace([fs], s)] if s else []
        sols.append(([zero]*s, self.one))
        return sols

    # the rational level C(k): Abramov's universal denominator and a
    # degree bound for the polynomial part

    def _dispersion(self, A: Poly, B: Poly) -> int:
        """The largest ``h >= 0`` with ``gcd(A(k), B(k + h)) != 1`` for
        polynomials ``A``, ``B`` in ``k`` (and the parameters), or ``-1``.

        An irreducible factor ``p`` of ``A`` and ``q`` of ``B`` of the same
        degree ``d`` can only satisfy ``q(k + h) ~ p(k)`` for the ``h``
        which matches the second coefficients, ``(b_p - b_q)/d`` for the
        monic factors; the candidate is then checked. (A resultant in
        ``h`` was used before: with parameters in the constants it was
        the most expensive step of creative telescoping.)"""
        K = self.C
        factors_b = [(q, self._k_coefficients(q)) for q in self._factors_in_k(B)]
        best = -1
        for p in self._factors_in_k(A):
            cp = self._k_coefficients(p)
            d = max(cp)
            lp = self._constant_of(cp[d])
            sp = K.exquo(self._constant_of(cp[d - 1]), lp) if d - 1 in cp else K.zero
            for q, cq in factors_b:
                if max(cq) != d:
                    continue
                sq = K.exquo(self._constant_of(cq[d - 1]), self._constant_of(cq[d])) if d - 1 in cq else K.zero
                shift = self.constants.rational(K.exquo(K.sub(sp, sq), K.from_int(d)))
                if shift is None or shift.denominator != 1 or shift < 0 or int(shift) <= best:
                    continue
                shifted = self.shift_poly(q, int(shift))
                lq = self._k_coefficients(shifted)[d]
                if not (shifted*cp[d] - p*lq).terms:
                    best = int(shift)
        return best

    def _universal_denominator(self, p1: Poly, p0: Poly) -> Poly:
        """Abramov's universal denominator of the rational solutions of
        ``p1(k) y(k + 1) + p0(k) y(k) = q(k)`` with polynomial ``q``."""
        ring = self.field.ring
        P = self._P
        A = self.shift_poly(p1, -1)
        B = p0
        U = ring.one
        N = self._dispersion(A, B)
        for i in range(N, -1, -1):
            d = ring.gcd(A, self.shift_poly(B, i))
            if d.degree_in(P) <= 0:
                continue
            A = A.exquo(d)
            B = B.exquo(self.shift_poly(d, -i))
            for j in range(i + 1):
                U = U*self.shift_poly(d, -j)
        return U

    def _coeff_list(self, p: Poly) -> list[C]:
        """Coefficients of a polynomial in ``k`` (and the parameters) as a
        list of constants indexed by the degree in ``k``."""
        coefficients = self._k_coefficients(p)
        d = max(coefficients) if coefficients else -1
        zero = self.C.zero
        return [self._constant_of(coefficients[i]) if i in coefficients else zero for i in range(d + 1)]

    def _solve_rational(self, a: PiSigmaElement, fs: Sequence[PiSigmaElement]) -> Solutions[C]:
        K = self.C
        ring = self.field.ring
        R = self._ring
        P = self._P
        A, B = a.numer, a.denom
        # B sigma(y) - A y = B (u_1 f_1 + ... + u_s f_s), then clear the
        # denominators of the right hand side
        Be = self._element(self.field.from_ring(B))
        bfs = [Be*f for f in fs]
        W = ring.one
        for f in bfs:
            W = ring.lcm(W, f.denom)
        We = self._element(self.field.from_ring(W))
        S: list[Poly] = []
        #: the constant denominators of the right hand sides: the unknown
        #: of S_l is u_l over it
        scales: list[C] = []
        for f in bfs:
            wf = We*f
            if wf.denom.degree_in(P) > 0:
                raise RuntimeError("denominator not cleared: %s" % (self.to_expr(wf, substitute=False),))
            S.append(wf.numer)
            scales.append(self._constant_of(wf.denom))
        p1, p0 = W*B, -(W*A)
        U = self._universal_denominator(p1, p0)
        sU = self.shift_poly(U, 1)
        P1, P0 = p1*U, p0*sU
        Q = [U*sU*q for q in S]
        # degree bound for the polynomial part
        c1, c0 = self._coeff_list(P1), self._coeff_list(P0)
        d1, d0 = len(c1) - 1, len(c0) - 1
        m = max(d1, d0)
        dq = max([len(self._coeff_list(q)) - 1 for q in Q] + [-1])
        D = max(dq - m, 0)
        if d1 == d0 and K.is_zero(K.add(c1[-1], c0[-1])):
            D = max(dq - m + 1, 0)
            lower = K.add(c1[m - 1], c0[m - 1]) if m >= 1 else K.zero
            root = self.constants.rational(K.neg(K.exquo(lower, c1[-1])))
            if root is not None and root.denominator == 1 and root >= 0:
                D = max(D, int(root))
        # linear system for the coefficients y_0..y_D and the constants
        x = R.gens[P]
        columns: list[list[C]] = []
        for j in range(D + 1):
            power = x**j
            columns.append(self._coeff_list(P1*self.shift_poly(power, 1) + P0*power))
        for q in Q:
            columns.append([K.neg(c) for c in self._coeff_list(q)])
        nrows = max([len(col) for col in columns] + [0])
        ncols = len(columns)
        rows: list[list[C]] = []
        for r in range(nrows):
            rows.append([col[r] if r < len(col) else K.zero for col in columns])
        Ue = self._element(self.field.from_ring(U))
        sols: list[Solution[C]] = []
        for v in self._nullspace(rows, ncols):
            p = self.zero
            for j in range(D + 1):
                if not K.is_zero(v[j]):
                    p = p + self.from_constant(v[j])*self.gens[0]**j
            constants = [K.mul(v[D + 1 + l], scales[l]) for l in range(len(S))]
            sols.append((constants, p/Ue))
        return sols

    # Sigma and Pi extensions: coefficientwise recursion with a running set
    # of unknown constants

    def _is_sigma_ratio(self, a: PiSigmaElement, level: int) -> bool:
        """Whether $\\sigma(h) = a h$ has a nonzero solution in the field of
        the given level (``a == 1`` included)."""
        if a == self.one:
            return True
        sols = self.solve(a, [], level=level)
        if not sols:
            return False
        return any(g != self.zero for _, g in sols)

    def _combination(self, coefficients: Sequence[C], elements: Sequence[PiSigmaElement]) -> PiSigmaElement:
        """The linear combination of ``elements`` with constant ``coefficients``."""
        result = self.zero
        for c, e in zip(coefficients, elements):
            if not self.C.is_zero(c):
                result = result + self.from_constant(c)*e
        return result

    def _normalize_basis(self, cmap: list[list[C]], G: dict[int, list[PiSigmaElement]], s: int, r: int) -> None:
        """Scale every basis vector so that its first nonzero constant is
        one, in place. The bases of the successive steps are composed, and
        without the scaling the constants grow with every step: with a
        parameter ``n`` in the constants the recurrence of creative
        telescoping for ``binomial(n, k)**2*harmonic(k)`` came out as
        polynomials of degree 530 in ``n`` instead of 3."""
        K = self.C
        for m in range(s):
            pivot = next((cmap[i][m] for i in range(r) if not K.is_zero(cmap[i][m])), None)
            if pivot is None or K.is_one(pivot):
                continue
            inverse = K.exquo(K.one, pivot)
            for i in range(r):
                cmap[i][m] = K.mul(cmap[i][m], inverse)
            scale = self.from_constant(inverse)
            for coeffs in G.values():
                coeffs[m] = coeffs[m]*scale

    def _solve_steps(self, level: int, steps: Sequence[tuple[int, PiSigmaElement, Rhs[C]]], r: int) -> Solutions[C]:
        """Common driver of the $\\Sigma$ and $\\Pi$ cases.

        ``steps`` is a list of ``(j, a_j, rhs)`` where ``rhs(cmap, G)``
        returns the right hand sides for the current unknowns: the
        coefficient $g_j$ satisfies $\\sigma(g_j) - a_j g_j = \\sum_l u_l R_l$.
        ``cmap`` is the matrix expressing the original constants in terms
        of the current unknowns and ``G`` maps the exponents already
        processed to the coefficients of $g_j$ per unknown.
        """
        K = self.C
        one, zero = K.one, K.zero
        s = r
        cmap: list[list[C]] = [[one if i == l else zero for l in range(s)] for i in range(r)]
        G: dict[int, list[PiSigmaElement]] = {}
        for j, a_j, rhs in steps:
            R = rhs(cmap, G, s)
            sols = self.solve(a_j, R, level=level - 1)
            if sols is None:
                return None
            # an empty basis forces the unknowns so far to zero, but the
            # coefficients still to come may have homogeneous solutions of
            # their own (sigma(g) = a g in a Pi-extension is solved by
            # c*t**j for one exponent j only): the recursion goes on
            p = len(sols)
            V = [v for v, _ in sols]
            G[j] = [g for _, g in sols]
            for jj in G:
                if jj != j:
                    G[jj] = [self._combination([V[m][l] for l in range(s)], G[jj]) for m in range(p)]
            new_cmap: list[list[C]] = []
            for i in range(r):
                row: list[C] = []
                for m in range(p):
                    total = zero
                    for l in range(s):
                        if not K.is_zero(cmap[i][l]) and not K.is_zero(V[m][l]):
                            total = K.add(total, K.mul(cmap[i][l], V[m][l]))
                    row.append(total)
                new_cmap.append(row)
            cmap = new_cmap
            s = p
            self._normalize_basis(cmap, G, s, r)
        t = self.gens[level]
        result: list[Solution[C]] = []
        for m in range(s):
            g = self.zero
            for j, coeffs in G.items():
                g = g + coeffs[m]*t**j
            result.append(([cmap[i][m] for i in range(r)], g))
        return result

    def _solve_sigma(self, level: int, a: PiSigmaElement, fs: Sequence[PiSigmaElement]) -> Solutions[C]:
        K = self.C
        beta = self._element(self._values[level - 1])
        polys: list[dict[int, PiSigmaElement]] = []
        for f in fs:
            p = self.as_poly_in(f, level)
            if p is None:
                return None
            polys.append(p)
        maxdeg = max([max(p) for p in polys if p] + [0])
        D = maxdeg + (1 if self._is_sigma_ratio(a, level - 1) else 0)
        r = len(fs)
        betas: dict[int, PiSigmaElement] = {}

        def make_rhs(j: int) -> Rhs[C]:
            def rhs(cmap: list[list[C]], G: dict[int, list[PiSigmaElement]], s: int) -> list[PiSigmaElement]:
                R: list[PiSigmaElement] = []
                for l in range(s):
                    value = self.zero
                    for i in range(r):
                        fij = polys[i].get(j)
                        if fij is not None and not K.is_zero(cmap[i][l]):
                            value = value + self.from_constant(cmap[i][l])*fij
                    for jj, coeffs in G.items():
                        if jj > j:
                            if jj - j not in betas:
                                betas[jj - j] = beta**(jj - j)
                            value = value - comb(jj, j)*betas[jj - j]*self.sigma(coeffs[l])
                    R.append(value)
                return R
            return rhs

        steps: list[tuple[int, PiSigmaElement, Rhs[C]]] = [(j, a, make_rhs(j)) for j in range(D, -1, -1)]
        return self._solve_steps(level, steps, r)

    def _pi_exponents(self, a: PiSigmaElement, alpha: PiSigmaElement) -> set[int]:
        r"""Candidate exponents ``j`` such that $\sigma(c)\, \alpha^j = a\, c$
        may have a solution $c$ in the field below: for elements of
        $\mathbb{C}(k)$ the leading coefficients and the degrees of ``a``
        and ``alpha`` must match, which fixes ``j`` in most cases; otherwise
        a few small exponents are tried."""
        K = self.C
        if a == self.one:
            return set()
        if self.level_of(a) > 0 or self.level_of(alpha) > 0:
            return set(range(-3, 4))

        def lc_and_degree(f: PiSigmaElement) -> tuple[C, int]:
            ln, dn = self._lc_in_k(f.numer)
            ld, dd = self._lc_in_k(f.denom)
            return K.exquo(ln, ld), dn - dd

        la, da = lc_and_degree(a)
        lal, dal = lc_and_degree(alpha)
        if dal != 0:
            if da % dal == 0:
                return {da//dal}
            return set()
        if lal != K.one and lal != K.neg(K.one):
            power = K.one
            for j in range(0, 40):
                if power == la:
                    return {j}
                power = K.mul(power, lal)
            power = K.one
            for j in range(0, 40):
                if power == la:
                    return {-j}
                power = K.exquo(power, lal)
            return set()
        return set(range(-3, 4))

    def _solve_pi(self, level: int, a: PiSigmaElement, fs: Sequence[PiSigmaElement]) -> Solutions[C]:
        K = self.C
        alpha = self._element(self._values[level - 1])
        polys: list[dict[int, PiSigmaElement]] = []
        for f in fs:
            p = self.as_poly_in(f, level, laurent=True)
            if p is None:
                return None
            polys.append(p)
        exponent_set: set[int] = {0}
        for p in polys:
            exponent_set |= set(p)
        exponent_set |= self._pi_exponents(a, alpha)
        exponents = sorted(exponent_set, reverse=True)
        r = len(fs)

        def make_rhs(j: int, alpha_j: PiSigmaElement) -> Rhs[C]:
            def rhs(cmap: list[list[C]], G: dict[int, list[PiSigmaElement]], s: int) -> list[PiSigmaElement]:
                R: list[PiSigmaElement] = []
                for l in range(s):
                    value = self.zero
                    for i in range(r):
                        fij = polys[i].get(j)
                        if fij is not None and not K.is_zero(cmap[i][l]):
                            value = value + self.from_constant(cmap[i][l])*fij
                    R.append(value/alpha_j)
                return R
            return rhs

        steps: list[tuple[int, PiSigmaElement, Rhs[C]]] = []
        for j in exponents:
            alpha_j = alpha**j
            steps.append((j, a/alpha_j, make_rhs(j, alpha_j)))
        return self._solve_steps(level, steps, r)


def rational_pisigma_field(k: Symbol) -> PiSigmaField[Fraction]:
    """The field $\\mathbb{Q}(k)$ with the shift $k \\to k + 1$."""
    return PiSigmaField(k, RationalConstants())


def parametric_pisigma_field(k: Symbol, params: Iterable[Symbol]) -> PiSigmaField[SparseFracElement[Fraction]]:
    """The field $\\mathbb{Q}(\\mathrm{params})(k)$ with the shift
    $k \\to k + 1$ (at least one parameter).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial
    >>> from sympy_extras.concrete.pisigma import parametric_pisigma_field
    >>> n, k = symbols('n k')
    >>> F, t = parametric_pisigma_field(k, [n]).add_pi((n - k)/(k + 1), binomial(n, k))
    >>> [([F.constant_to_expr(ci) for ci in c], F.to_expr(g)) for c, g in F.solve(F.one, [F.from_expr(t)])]
    [([0], 1)]
    >>> F.to_expr(F.telescope(F.from_expr((n - 2*k)*t)))
    k*binomial(n, k)
    """
    return PiSigmaField(k, ParametricConstants(list(params)))
