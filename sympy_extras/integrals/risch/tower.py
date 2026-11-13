r"""The arithmetic of the differential fields of the Risch algorithm on the
typed core: :class:`RationalFunction`, an element of ``C(x, t1, ...,
tn)``, and :class:`TowerPoly`, a polynomial in one generator ``t`` over
that field (``k[t]`` with ``k`` the field of the lower levels of the
tower).

A :class:`RationalFunction` is a fraction ``numer/denom`` of polynomials
over ``ZZ`` in its generators: the symbols of the tower and of the
parameters, and the other kernels of the expressions it was made of
(``pi``, ``exp(2)``, ``log(3)``, ``exp(x)`` before the tower is built),
with the two algebraic constants the integrands of the Risch algorithm
usually carry, ``I`` and the square roots of prime numbers (``sqrt(6)``
is ``sqrt(2)*sqrt(3)``). The fraction is in canonical form: the powers of
the algebraic generators are reduced (``I**2 = -1``, ``sqrt(p)**2 = p``),
the denominator is free of them (it is multiplied by its conjugates), and
the numerator and the denominator are coprime over ``ZZ``, the leading
coefficient of the denominator positive in the order of the generators
(SymPy's order, :func:`~sympy_extras.polys.polytools.sort_gens`). Two
elements are equal exactly when they are the same fraction, and an element
is zero exactly when its numerator is: SymPy's fields ``ZZ(x, t0)``,
``QQ_I(x)`` and ``EX`` (for ``sqrt(2)``) in one domain, with exact zero
recognition. Other algebraic constants (cube roots, nested radicals,
algebraic functions such as ``sqrt(x)``) and floating point numbers raise
:class:`UnsupportedConstant` (a ``NotImplementedError``): the Risch
algorithm gives up on them.

:class:`TowerPoly` is a dense polynomial over the field (a list of
coefficients, the highest degree first) with the subset of SymPy's
``Poly`` interface the algorithm uses: the arithmetic, division,
``gcd``, ``resultant`` (with the subresultant PRS), square-free
decompositions, the extended Euclidean algorithm, by the generic dense
algorithms of :mod:`sympy_extras.polys` over the domain
:data:`TOWER`.

Examples
========

>>> from sympy_extras.core import symbols, I, sqrt
>>> from sympy_extras.integrals.risch.tower import RationalFunction, TowerPoly
>>> x, t = symbols('x t')
>>> a = RationalFunction.from_expr((x**2 - 1)/(x + 1))
>>> a, a + 1/RationalFunction.from_expr(x)
(x - 1, (x**2 - x + 1)/x)
>>> RationalFunction.from_expr(1/(1 + I)), RationalFunction.from_expr(sqrt(6)/sqrt(2))
(1/2 - I/2, sqrt(3))
>>> p = TowerPoly.from_expr(t**2/x + 2*t, t)
>>> p
TowerPoly(t**2/x + 2*t, t)
>>> p.gcd(TowerPoly.from_expr(t**2 - 4*x**2, t))
TowerPoly(t + 2*x, t)

References
==========

.. [Bronstein] M. Bronstein, *Symbolic Integration I: Transcendental
   Functions*, 2nd edition, Springer, 2005 (Chapter 3: differential
   fields, the ``k[t]`` of the tower).
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd as integer_gcd
from typing import Final, Iterator, Optional, Sequence, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr
from sympy_extras.core.exprtools import decompose_power
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, ImaginaryUnit, Integer, Number, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.polys.densearith import (dup_add, dup_div, dup_exquo, dup_mul, dup_mul_ground, dup_neg, dup_pow,
                                           dup_quo, dup_rem, dup_sub)
from sympy_extras.polys.densebasic import dup_degree, dup_strip
from sympy_extras.polys.densetools import dup_diff, dup_eval, dup_monic
from sympy_extras.polys.domains.domain import Domain, U
from sympy_extras.polys.domains.sparsepolynomialring import IntegerPolynomialRing
from sympy_extras.polys.euclidtools import (dup_gcd, dup_gcdex, dup_half_gcdex, dup_inner_subresultants, dup_invert,
                                            dup_lcm, dup_resultant)
from sympy_extras.polys.polyerrors import PolynomialError
from sympy_extras.polys.polytools import _gen_key
from sympy_extras.polys.rings import Monomial, PolyElement
from sympy_extras.polys.sqfreetools import dup_sqf_list

__all__ = ['RationalFunction', 'TowerPoly', 'TowerDomain', 'TOWER', 'UnsupportedConstant', 'TowerPolyLike']

GenKey = tuple[int, str, int, int]


class UnsupportedConstant(NotImplementedError):
    """An expression with a constant the field cannot represent exactly
    (a floating point number, an algebraic number other than ``I`` and the
    square roots of integers, an algebraic function)."""


# ---------------------------------------------------------------------
# the rings and the generators

_RINGS: dict[int, IntegerPolynomialRing] = {}
_KEYS: dict[Expr, GenKey] = {}
#: the square of an algebraic generator (-1 for I, p for sqrt(p)), None for the others
_SQUARES: dict[Expr, Optional[int]] = {}
#: the algebraic generators of a tuple of generators: (index, square)
_ALGEBRAIC: dict[tuple[Expr, ...], tuple[tuple[int, int], ...]] = {}


def _ring(n: int) -> IntegerPolynomialRing:
    """``ZZ[g0, ..., g(n-1)]`` (one variable for the constants)."""
    width = max(n, 1)
    found = _RINGS.get(width)
    if found is None:
        found = IntegerPolynomialRing(['g%d' % i for i in range(width)])
        _RINGS[width] = found
    return found


def _key(g: Expr) -> GenKey:
    found = _KEYS.get(g)
    if found is None:
        found = _gen_key(g)
        _KEYS[g] = found
    return found


def _square(g: Expr) -> Optional[int]:
    if g in _SQUARES:
        return _SQUARES[g]
    value: Optional[int] = None
    if isinstance(g, ImaginaryUnit):
        value = -1
    elif (isinstance(g, Pow) and isinstance(g.base, Integer) and isinstance(g.exp, Rational)
          and g.exp == Rational(1, 2)):
        value = g.base.p
    _SQUARES[g] = value
    return value


def _algebraic(gens: tuple[Expr, ...]) -> tuple[tuple[int, int], ...]:
    found = _ALGEBRAIC.get(gens)
    if found is None:
        pairs: list[tuple[int, int]] = []
        for i, g in enumerate(gens):
            s = _square(g)
            if s is not None:
                pairs.append((i, s))
        found = tuple(pairs)
        _ALGEBRAIC[gens] = found
    return found


def _merge(a: tuple[Expr, ...], b: tuple[Expr, ...]) -> tuple[Expr, ...]:
    """The union of two tuples of generators sorted by their keys."""
    if a == b or not b:
        return a
    if not a:
        return b
    seen = set(a)
    extra = [g for g in b if g not in seen]
    if not extra:
        return a
    return tuple(sorted(list(a) + extra, key=_key))


def _remap(f: PolyElement[int], old: tuple[Expr, ...], new: tuple[Expr, ...]) -> PolyElement[int]:
    """``f`` in the generators ``old`` as a polynomial in ``new`` (a
    superset)."""
    R = _ring(len(new))
    if old == new:
        if f.ring is R.ring:
            return f
    width = max(len(new), 1)
    if not old:
        return R.ring.from_dict({tuple(0 for _ in range(width)): c for c in f.terms.values()})
    positions = [new.index(g) for g in old]
    terms: dict[Monomial, int] = {}
    for m, c in f.terms.items():
        e = [0] * width
        for i, k in zip(positions, m):
            e[i] = k
        terms[tuple(e)] = c
    return PolyElement(terms, R.ring)


def _reduce(f: PolyElement[int], algebraic: tuple[tuple[int, int], ...]) -> PolyElement[int]:
    """The powers of the algebraic generators reduced: ``g**2 = square``."""
    if not algebraic or not f.terms:
        return f
    if not any(m[i] > 1 for m in f.terms for i, _ in algebraic):
        return f
    terms: dict[Monomial, int] = {}
    for m, c in f.terms.items():
        e = list(m)
        for i, s in algebraic:
            if e[i] > 1:
                c *= s**(e[i] // 2)
                e[i] %= 2
        key = tuple(e)
        value = terms.get(key, 0) + c
        if value:
            terms[key] = value
        else:
            terms.pop(key, None)
    return PolyElement(terms, f.ring)


def _conjugate(f: PolyElement[int], i: int) -> PolyElement[int]:
    """``f`` with the ``i``-th generator negated."""
    return PolyElement({m: (-c if m[i] % 2 else c) for m, c in f.terms.items()}, f.ring)


def _used(f: PolyElement[int], n: int) -> list[bool]:
    used = [False] * n
    for m in f.terms:
        for i in range(n):
            if m[i]:
                used[i] = True
    return used


def _project(f: PolyElement[int], keep: list[int], R: IntegerPolynomialRing) -> PolyElement[int]:
    width = max(len(keep), 1)
    if not keep:
        return PolyElement({tuple(0 for _ in range(width)): c for c in f.terms.values()}, R.ring)
    return PolyElement({tuple(m[i] for i in keep): c for m, c in f.terms.items()}, R.ring)


def _content(f: PolyElement[int]) -> int:
    g = 0
    for c in f.terms.values():
        g = integer_gcd(g, c)
        if g == 1:
            break
    return g


# ---------------------------------------------------------------------
# the field

class RationalFunction:
    """An element ``numer/denom`` of the field of rational functions in its
    generators over ``QQ(I, sqrt(2), sqrt(3), ...)``, in canonical form
    (see the module docstring). Immutable; build elements with
    :meth:`from_expr`, :meth:`from_int`, :meth:`generator` and the
    arithmetic operators.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.tower import RationalFunction
    >>> x, y = symbols('x y')
    >>> f = RationalFunction.from_expr(x/(x + y))
    >>> f.diff(x), f.subs(y, RationalFunction.from_int(0)), f.has(y)
    (y/(x**2 + 2*x*y + y**2), 1, True)
    """

    __slots__ = ('gens', 'numer', 'denom', '_expr_cache')

    def __init__(self, gens: tuple[Expr, ...], numer: PolyElement[int], denom: PolyElement[int]) -> None:
        # in canonical form: built by _make
        self.gens: Final[tuple[Expr, ...]] = gens
        self.numer: Final[PolyElement[int]] = numer
        self.denom: Final[PolyElement[int]] = denom
        self._expr_cache: Optional[Expr] = None

    # -- construction

    @staticmethod
    def _make(gens: tuple[Expr, ...], numer: PolyElement[int], denom: PolyElement[int],
              cancel: bool = True) -> RationalFunction:
        """The canonical form of ``numer/denom``."""
        R = _ring(len(gens))
        if not numer.terms:
            return _ZERO_RF
        if not denom.terms:
            raise ZeroDivisionError('division by zero in a field of rational functions')
        algebraic = _algebraic(gens)
        if algebraic:
            numer, denom = _reduce(numer, algebraic), _reduce(denom, algebraic)
            for i, _ in algebraic:
                if any(m[i] for m in denom.terms):
                    conjugate = _conjugate(denom, i)
                    numer = _reduce(numer * conjugate, algebraic)
                    denom = _reduce(denom * conjugate, algebraic)
            if not numer.terms:
                return _ZERO_RF
            cancel = True
        if cancel:
            if len(denom.terms) == 1 and denom.is_ground:
                d = denom.LC
                g = integer_gcd(_content(numer), d)
                if d < 0:
                    g = -g
                if g != 1:
                    numer, denom = numer.exquo_ground(g), denom.exquo_ground(g)
            else:
                numer, denom = R.cancel(numer, denom)
        # prune the generators which do not appear
        n = len(gens)
        if n:
            used = _used(numer, n)
            used_d = _used(denom, n)
            keep = [i for i in range(n) if used[i] or used_d[i]]
            if len(keep) < n:
                S = _ring(len(keep))
                new_gens = tuple(gens[i] for i in keep)
                return RationalFunction(new_gens, _project(numer, keep, S), _project(denom, keep, S))
        return RationalFunction(gens, numer, denom)

    @staticmethod
    def from_int(n: int) -> RationalFunction:
        if n == 0:
            return _ZERO_RF
        R = _ring(0)
        return RationalFunction((), R.ring.from_int(n), R.ring.one)

    @staticmethod
    def from_fraction(q: Fraction) -> RationalFunction:
        if q == 0:
            return _ZERO_RF
        R = _ring(0)
        return RationalFunction((), R.ring.from_int(q.numerator), R.ring.from_int(q.denominator))

    @staticmethod
    def generator(g: Expr) -> RationalFunction:
        """The generator ``g`` (a symbol or a kernel) as an element."""
        R = _ring(1)
        if _square(g) is not None:
            if isinstance(g, Pow) and isinstance(g.base, Integer):
                return _sqrt_integer(g.base.p)
        return RationalFunction((g,), PolyElement({(1,): 1}, R.ring), R.ring.one)

    @staticmethod
    def from_expr(e: Expr) -> RationalFunction:
        """The element of a core expression (rational in its kernels);
        :class:`UnsupportedConstant` for the constants the field does not
        represent."""
        return _from_expr(e)

    # -- predicates and parts

    @property
    def is_zero(self) -> bool:
        return not self.numer.terms

    @property
    def is_one(self) -> bool:
        return not self.gens and self.numer == self.denom

    def __bool__(self) -> bool:
        return bool(self.numer.terms)

    @property
    def is_number(self) -> bool:
        """Whether the element is a constant of ``QQ(I, sqrt(2), ...)``."""
        return all(_square(g) is not None for g in self.gens)

    @property
    def is_rational(self) -> bool:
        """Whether the element is a rational number."""
        return not self.gens

    @property
    def is_integer(self) -> bool:
        return not self.gens and self.denom.LC == 1

    def as_fraction(self) -> Optional[Fraction]:
        """The rational number, ``None`` for the other elements."""
        if self.gens:
            return None
        if not self.numer.terms:
            return Fraction(0)
        return Fraction(self.numer.LC, self.denom.LC)

    def as_int(self) -> Optional[int]:
        q = self.as_fraction()
        if q is None or q.denominator != 1:
            return None
        return q.numerator

    def has(self, *gens: Expr) -> bool:
        """Whether one of ``gens`` is a generator of the element."""
        return any(g in self.gens for g in gens)

    def free_gens(self) -> tuple[Expr, ...]:
        return self.gens

    def numerator(self) -> RationalFunction:
        return RationalFunction._make(self.gens, self.numer, _ring(len(self.gens)).ring.one, cancel=False)

    def denominator(self) -> RationalFunction:
        return RationalFunction._make(self.gens, self.denom, _ring(len(self.gens)).ring.one, cancel=False)

    def as_numer_denom(self) -> tuple[RationalFunction, RationalFunction]:
        return self.numerator(), self.denominator()

    def is_polynomial_in(self, g: Expr) -> bool:
        """Whether the denominator is free of ``g``."""
        if g not in self.gens:
            return True
        i = self.gens.index(g)
        return not any(m[i] for m in self.denom.terms)

    def degree(self, g: Expr) -> int:
        """The degree of the numerator in ``g`` (``-1`` for zero)."""
        if not self.numer.terms:
            return -1
        if g not in self.gens:
            return 0
        i = self.gens.index(g)
        return max(m[i] for m in self.numer.terms)

    def could_extract_minus_sign(self) -> bool:
        """Whether the leading coefficient of the numerator is negative."""
        return bool(self.numer.terms) and self.numer.LC < 0

    # -- arithmetic

    def _unified(self, other: RationalFunction) -> tuple[tuple[Expr, ...], PolyElement[int], PolyElement[int],
                                                         PolyElement[int], PolyElement[int]]:
        gens = _merge(self.gens, other.gens)
        return (gens, _remap(self.numer, self.gens, gens), _remap(self.denom, self.gens, gens),
                _remap(other.numer, other.gens, gens), _remap(other.denom, other.gens, gens))

    def __add__(self, other: Union[RationalFunction, int]) -> RationalFunction:
        b = _coerce(other)
        if not b.numer.terms:
            return self
        if not self.numer.terms:
            return b
        gens, an, ad, bn, bd = self._unified(b)
        if ad == bd:
            return RationalFunction._make(gens, an + bn, ad)
        return RationalFunction._make(gens, an * bd + bn * ad, ad * bd)

    def __radd__(self, other: int) -> RationalFunction:
        return self + other

    def __neg__(self) -> RationalFunction:
        if not self.numer.terms:
            return self
        return RationalFunction(self.gens, -self.numer, self.denom)

    def __sub__(self, other: Union[RationalFunction, int]) -> RationalFunction:
        return self + (-_coerce(other))

    def __rsub__(self, other: int) -> RationalFunction:
        return _coerce(other) - self

    def __mul__(self, other: Union[RationalFunction, int]) -> RationalFunction:
        b = _coerce(other)
        if not b.numer.terms or not self.numer.terms:
            return _ZERO_RF
        if b.is_one:
            return self
        if self.is_one:
            return b
        gens, an, ad, bn, bd = self._unified(b)
        algebraic = bool(_algebraic(gens))
        if not algebraic and ad.is_ground and bd.is_ground and ad.LC == 1 and bd.LC == 1:
            return RationalFunction._make(gens, an * bn, ad, cancel=False)
        return RationalFunction._make(gens, an * bn, ad * bd)

    def __rmul__(self, other: int) -> RationalFunction:
        return self * other

    def invert(self) -> RationalFunction:
        if not self.numer.terms:
            raise ZeroDivisionError('division by zero in a field of rational functions')
        return RationalFunction._make(self.gens, self.denom, self.numer)

    def __truediv__(self, other: Union[RationalFunction, int]) -> RationalFunction:
        return self * _coerce(other).invert()

    def __rtruediv__(self, other: int) -> RationalFunction:
        return _coerce(other) * self.invert()

    def __pow__(self, n: int) -> RationalFunction:
        if n < 0:
            return self.invert() ** (-n)
        if n == 0:
            return _ONE_RF
        if n == 1:
            return self
        result = _ONE_RF
        base: RationalFunction = self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self == RationalFunction.from_int(other)
        if not isinstance(other, RationalFunction):
            return NotImplemented
        return self.gens == other.gens and self.numer.terms == other.numer.terms and \
            self.denom.terms == other.denom.terms

    def __hash__(self) -> int:
        return hash((self.gens, frozenset(self.numer.terms.items()), frozenset(self.denom.terms.items())))

    # -- calculus and substitution

    def diff(self, g: Expr) -> RationalFunction:
        """The derivative with respect to the generator ``g`` (the other
        generators taken as constants)."""
        if g not in self.gens:
            return _ZERO_RF
        i = self.gens.index(g)
        n, d = self.numer, self.denom
        dn, dd = n.diff(i), d.diff(i)
        if not dd.terms:
            return RationalFunction._make(self.gens, dn, d)
        return RationalFunction._make(self.gens, dn * d - n * dd, d * d)

    def subs(self, g: Expr, value: RationalFunction) -> RationalFunction:
        """The element with the generator ``g`` replaced by ``value``."""
        if g not in self.gens:
            return self
        return _evaluate(self.numer, self.gens, g, value) / _evaluate(self.denom, self.gens, g, value)

    def xreplace(self, rule: dict[Expr, RationalFunction]) -> RationalFunction:
        result: RationalFunction = self
        for g, v in rule.items():
            result = result.subs(g, v)
        return result

    def conjugate_i(self) -> RationalFunction:
        """The element with ``I`` replaced by ``-I`` (the complex conjugate
        when the other generators are real)."""
        return self.subs(I, -_I_RF)

    # -- conversions

    def as_expr(self) -> Expr:
        """The element as a core expression ``numer/denom`` (expanded
        polynomials)."""
        if self._expr_cache is None:
            n = _poly_expr(self.numer, self.gens)
            if self.denom.is_ground and self.denom.LC == 1:
                self._expr_cache = n
            else:
                self._expr_cache = mul(n, pow(_poly_expr(self.denom, self.gens), Integer(-1)))
        return self._expr_cache

    def __str__(self) -> str:
        return str(self.as_expr())

    def __repr__(self) -> str:
        return str(self.as_expr())


def _coerce(a: Union[RationalFunction, int]) -> RationalFunction:
    if isinstance(a, int):
        return RationalFunction.from_int(a)
    return a


def _poly_expr(f: PolyElement[int], gens: tuple[Expr, ...]) -> Expr:
    terms: list[Expr] = []
    for m, c in f.terms.items():
        factors: list[Expr] = [Integer(c)]
        for g, k in zip(gens, m):
            if k:
                factors.append(g if k == 1 else pow(g, Integer(k)))
        terms.append(mul(*factors))
    return add(*terms)


def _evaluate(f: PolyElement[int], gens: tuple[Expr, ...], g: Expr, value: RationalFunction) -> RationalFunction:
    """The polynomial ``f`` with ``g`` replaced by ``value``."""
    i = gens.index(g)
    grouped: dict[int, dict[Monomial, int]] = {}
    for m, c in f.terms.items():
        e = list(m)
        k = e[i]
        e[i] = 0
        grouped.setdefault(k, {})[tuple(e)] = c
    R = _ring(len(gens))
    result: RationalFunction = _ZERO_RF
    for k in sorted(grouped, reverse=True):
        coefficient = RationalFunction._make(gens, PolyElement(grouped[k], R.ring), R.ring.one)
        result = result + coefficient * value**k
    return result


_ZERO_RF: Final[RationalFunction] = RationalFunction((), _ring(0).ring.zero, _ring(0).ring.one)
_ONE_RF: Final[RationalFunction] = RationalFunction((), _ring(0).ring.one, _ring(0).ring.one)
_I_RF: Final[RationalFunction] = RationalFunction((I,), PolyElement({(1,): 1}, _ring(1).ring), _ring(1).ring.one)


def _sqrt_integer(n: int) -> RationalFunction:
    """``sqrt(n)`` for an integer ``n``: the product of the square roots
    of its prime factors (``I`` for the sign)."""
    from sympy_extras.ntheory.factor_ import factorint
    if n == 0:
        return _ZERO_RF
    result: RationalFunction = _I_RF if n < 0 else _ONE_RF
    for p, k in factorint(abs(n)).items():
        if k // 2:
            result = result * RationalFunction.from_int(p**(k // 2))
        if k % 2:
            g = Pow(Integer(p), Rational(1, 2))
            R = _ring(1)
            result = result * RationalFunction((g,), PolyElement({(1,): 1}, R.ring), R.ring.one)
    return result


def _from_expr(e: Expr) -> RationalFunction:
    if isinstance(e, Number):
        if isinstance(e, Integer):
            return RationalFunction.from_int(e.p)
        if isinstance(e, Rational):
            return RationalFunction.from_fraction(Fraction(e.p, e.q))
        if isinstance(e, ImaginaryUnit):
            return _I_RF
        raise UnsupportedConstant('the constant %s is not supported by the Risch algorithm' % (e,))
    if isinstance(e, ImaginaryUnit):
        return _I_RF
    if isinstance(e, Add):
        return _sum([_from_expr(a) for a in e.args])
    if isinstance(e, Mul):
        result: RationalFunction = _ONE_RF
        for a in e.args:
            result = result * _from_expr(a)
        return result
    if isinstance(e, Pow):
        base, exp = e.base, e.exp
        if isinstance(exp, Integer) or (isinstance(exp, Rational) and exp.q == 1):
            return _from_expr(base) ** exp.p
        if isinstance(exp, Rational):
            if isinstance(base, Rational) and exp.q == 2:
                root = _sqrt_integer(base.p) / _sqrt_integer(base.q) if base.q != 1 else _sqrt_integer(base.p)
                return root ** exp.p
            raise UnsupportedConstant('the algebraic %s is not supported by the Risch algorithm' % (e,))
    if isinstance(e, Number):
        raise UnsupportedConstant('the constant %s is not supported by the Risch algorithm' % (e,))
    if isinstance(e, Pow) or isinstance(e, Exp):
        # SymPy's generators: exp(2*x) is exp(x)**2, x**(-y) is (x**y)**-1
        kernel, k = decompose_power(e)
        if k != 1:
            return _from_expr(kernel) ** k
    return RationalFunction.generator(e)


def _sum(parts: list[RationalFunction]) -> RationalFunction:
    """A sum: the polynomial parts added directly, one common
    denominator for the others."""
    polynomial: list[RationalFunction] = []
    fractions: list[RationalFunction] = []
    for p in parts:
        if p.denom.is_ground and p.denom.LC == 1 and not _algebraic(p.gens):
            polynomial.append(p)
        else:
            fractions.append(p)
    result: RationalFunction = _ZERO_RF
    if polynomial:
        gens: tuple[Expr, ...] = ()
        for p in polynomial:
            gens = _merge(gens, p.gens)
        R = _ring(len(gens))
        total = R.ring.zero
        for p in polynomial:
            total = total + _remap(p.numer, p.gens, gens)
        result = RationalFunction._make(gens, total, R.ring.one, cancel=False)
    for p in fractions:
        result = result + p
    return result


# ---------------------------------------------------------------------
# the domain of the dense algorithms

class TowerDomain(Domain[RationalFunction]):
    """The field of :class:`RationalFunction` elements as a domain of the
    dense polynomial algorithms."""

    name = 'K'
    is_field = True
    is_ordered = False
    characteristic = 0

    @property
    def zero(self) -> RationalFunction:
        return _ZERO_RF

    @property
    def one(self) -> RationalFunction:
        return _ONE_RF

    def is_zero(self, a: RationalFunction) -> bool:
        return not a.numer.terms

    def is_one(self, a: RationalFunction) -> bool:
        return a.is_one

    def is_negative(self, a: RationalFunction) -> bool:
        return a.could_extract_minus_sign()

    def add(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return a + b

    def sub(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return a - b

    def neg(self, a: RationalFunction) -> RationalFunction:
        return -a

    def mul(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return a * b

    def pow(self, a: RationalFunction, n: int) -> RationalFunction:
        return a**n

    def exquo(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return a / b

    def quo(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return a / b

    def rem(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        if b.is_zero:
            raise ZeroDivisionError('division by zero')
        return _ZERO_RF

    def invert(self, a: RationalFunction) -> RationalFunction:
        return a.invert()

    def gcd(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return _ONE_RF if (a or b) else _ZERO_RF

    def lcm(self, a: RationalFunction, b: RationalFunction) -> RationalFunction:
        return _ONE_RF if (a and b) else _ZERO_RF

    def gcdex(self, a: RationalFunction, b: RationalFunction
              ) -> tuple[RationalFunction, RationalFunction, RationalFunction]:
        if a:
            return a.invert(), _ZERO_RF, _ONE_RF
        if b:
            return _ZERO_RF, b.invert(), _ONE_RF
        return _ZERO_RF, _ZERO_RF, _ZERO_RF

    def canonical_unit(self, a: RationalFunction) -> RationalFunction:
        return a.invert() if a else _ONE_RF

    def from_int(self, n: int) -> RationalFunction:
        return RationalFunction.from_int(n)

    def convert(self, a: U, K: Domain[U]) -> RationalFunction:
        if isinstance(a, RationalFunction):
            return a
        return _from_expr(K.to_expr(a))

    def to_expr(self, a: RationalFunction) -> Expr:
        return a.as_expr()

    def from_expr(self, e: Expr) -> RationalFunction:
        return _from_expr(e)

    def to_str(self, a: RationalFunction) -> str:
        return str(a)

    def size(self, a: RationalFunction) -> int:
        return len(a.numer.terms) + len(a.denom.terms)


#: the field of the tower
TOWER: Final[TowerDomain] = TowerDomain()

Dup = list[RationalFunction]
#: what the arithmetic of TowerPoly takes besides polynomials
TowerPolyLike = Union['TowerPoly', RationalFunction, int]


# ---------------------------------------------------------------------
# polynomials over the field

class TowerPoly:
    """A polynomial in ``gen`` with coefficients :class:`RationalFunction`
    free of ``gen`` (an element of ``k[t]``), dense, the highest degree
    first; immutable. The interface is the part of SymPy's ``Poly`` the
    Risch algorithm uses.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.risch.tower import TowerPoly
    >>> x, t = symbols('x t')
    >>> p = TowerPoly.from_expr((t**2 - 1)/x, t)
    >>> p.degree(), p.LC(), p.monic(), p.div(TowerPoly.from_expr(t - 1, t))
    (2, 1/x, TowerPoly(t**2 - 1, t), (TowerPoly(t/x + 1/x, t), TowerPoly(0, t)))
    """

    __slots__ = ('rep', 'gen', '_expr_cache', '_element_cache')

    def __init__(self, rep: Sequence[RationalFunction], gen: Expr) -> None:
        self.rep: Final[Dup] = dup_strip(list(rep), TOWER)
        self.gen: Final[Expr] = gen
        self._expr_cache: Optional[Expr] = None
        self._element_cache: Optional[RationalFunction] = None

    # -- construction

    @staticmethod
    def from_element(a: RationalFunction, gen: Expr) -> Optional[TowerPoly]:
        """``a`` as a polynomial in ``gen``; ``None`` when its denominator
        depends on ``gen``."""
        if gen not in a.gens:
            return TowerPoly([a] if a else [], gen)
        i = a.gens.index(gen)
        if any(m[i] for m in a.denom.terms):
            return None
        grouped: dict[int, dict[Monomial, int]] = {}
        for m, c in a.numer.terms.items():
            e = list(m)
            k = e[i]
            e[i] = 0
            grouped.setdefault(k, {})[tuple(e)] = c
        n = max(grouped)
        R = _ring(len(a.gens))
        rep: Dup = []
        for k in range(n, -1, -1):
            terms = grouped.get(k)
            if terms is None:
                rep.append(_ZERO_RF)
            else:
                rep.append(RationalFunction._make(a.gens, PolyElement(terms, R.ring), a.denom))
        return TowerPoly(rep, gen)

    @staticmethod
    def from_element_strict(a: RationalFunction, gen: Expr) -> TowerPoly:
        found = TowerPoly.from_element(a, gen)
        if found is None:
            raise PolynomialError('%s is not a polynomial in %s' % (a, gen))
        return found

    @staticmethod
    def from_expr(e: Union[Expr, int], gen: Expr) -> TowerPoly:
        """The polynomial in ``gen`` of a core expression (``PolynomialError``
        when it is not one)."""
        a = RationalFunction.from_int(e) if isinstance(e, int) else _from_expr(e)
        if any(g != gen and g.has(gen) for g in a.gens):
            raise PolynomialError('%s contains an element of the set of generators' % (e,))
        return TowerPoly.from_element_strict(a, gen)

    @staticmethod
    def from_list(coefficients: Sequence[Union[RationalFunction, int]], gen: Expr) -> TowerPoly:
        return TowerPoly([_coerce(c) for c in coefficients], gen)

    @staticmethod
    def zero(gen: Expr) -> TowerPoly:
        return TowerPoly([], gen)

    @staticmethod
    def one(gen: Expr) -> TowerPoly:
        return TowerPoly([_ONE_RF], gen)

    @staticmethod
    def ground(c: Union[RationalFunction, int], gen: Expr) -> TowerPoly:
        """The constant polynomial ``c`` (an element free of ``gen``)."""
        a = _coerce(c)
        if a.has(gen):
            raise PolynomialError('%s is not a constant in %s' % (a, gen))
        return TowerPoly([a], gen)

    @staticmethod
    def monomial(gen: Expr, n: int, c: Union[RationalFunction, int] = 1) -> TowerPoly:
        """``c*gen**n``."""
        a = _coerce(c)
        if not a:
            return TowerPoly([], gen)
        return TowerPoly([a] + [_ZERO_RF] * n, gen)

    def new(self, rep: Sequence[RationalFunction]) -> TowerPoly:
        return TowerPoly(rep, self.gen)

    # -- conversions

    def as_expr(self) -> Expr:
        """The polynomial as a core expression ``sum c_k gen**k``."""
        if self._expr_cache is None:
            n = len(self.rep) - 1
            terms: list[Expr] = []
            for i, c in enumerate(self.rep):
                if c:
                    k = n - i
                    power: Expr = Integer(1) if k == 0 else (self.gen if k == 1 else pow(self.gen, Integer(k)))
                    terms.append(mul(c.as_expr(), power))
            self._expr_cache = add(*terms)
        return self._expr_cache

    def as_element(self) -> RationalFunction:
        """The polynomial as an element of the field."""
        if self._element_cache is None:
            if not self.rep:
                self._element_cache = _ZERO_RF
            elif len(self.rep) == 1:
                self._element_cache = self.rep[0]
            else:
                self._element_cache = _horner(self.rep, self.gen)
        return self._element_cache

    def as_poly(self, gen: Expr) -> Optional[TowerPoly]:
        """The polynomial as a polynomial in another generator; ``None``
        when it is not one."""
        if gen == self.gen:
            return self
        return TowerPoly.from_element(self.as_element(), gen)

    def as_poly_strict(self, gen: Expr) -> TowerPoly:
        found = self.as_poly(gen)
        if found is None:
            raise PolynomialError('%s is not a polynomial in %s' % (self, gen))
        return found

    def __repr__(self) -> str:
        return 'TowerPoly(%s, %s)' % (self.as_expr(), self.gen)

    __str__ = __repr__

    # -- predicates and parts

    def degree(self, gen: Optional[Expr] = None) -> int:
        """The degree in ``gen`` (by default the generator; ``-1`` for zero,
        SymPy's ``-oo``)."""
        if gen is None or gen == self.gen:
            return len(self.rep) - 1
        if not self.rep:
            return -1
        return max(c.degree(gen) for c in self.rep if c)

    def LC(self) -> RationalFunction:
        return self.rep[0] if self.rep else _ZERO_RF

    def TC(self) -> RationalFunction:
        return self.rep[-1] if self.rep else _ZERO_RF

    def nth(self, n: int) -> RationalFunction:
        """The coefficient of ``gen**n``."""
        k = len(self.rep) - 1 - n
        if n < 0 or k < 0:
            return _ZERO_RF
        return self.rep[k]

    def all_coeffs(self) -> list[RationalFunction]:
        return list(self.rep) if self.rep else [_ZERO_RF]

    def coeffs(self) -> list[RationalFunction]:
        """The nonzero coefficients, the highest degree first."""
        return [c for c in self.rep if c] or [_ZERO_RF]

    def terms(self) -> list[tuple[int, RationalFunction]]:
        """The ``(degree, coefficient)`` pairs of the nonzero terms, the
        highest degree first."""
        n = len(self.rep) - 1
        return [(n - i, c) for i, c in enumerate(self.rep) if c]

    def ET(self) -> tuple[int, RationalFunction]:
        """The term of the lowest degree."""
        n = len(self.rep) - 1
        for i in range(n, -1, -1):
            if self.rep[i]:
                return (n - i, self.rep[i])
        return (0, _ZERO_RF)

    @property
    def is_zero(self) -> bool:
        return not self.rep

    @property
    def is_one(self) -> bool:
        return len(self.rep) == 1 and self.rep[0].is_one

    @property
    def is_ground(self) -> bool:
        return len(self.rep) <= 1

    @property
    def is_monic(self) -> bool:
        return bool(self.rep) and self.rep[0].is_one

    @property
    def is_monomial(self) -> bool:
        return sum(1 for c in self.rep if c) <= 1

    def __bool__(self) -> bool:
        return bool(self.rep)

    def has(self, *gens: Expr) -> bool:
        """Whether the polynomial depends on one of ``gens`` (its generator,
        when it has positive degree, or a generator of a coefficient)."""
        if self.gen in gens and len(self.rep) > 1:
            return True
        return any(c.has(*gens) for c in self.rep)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TowerPoly):
            if self.gen == other.gen:
                return self.rep == other.rep
            return len(self.rep) <= 1 and len(other.rep) <= 1 and self.rep == other.rep
        if isinstance(other, (RationalFunction, int)):
            return len(self.rep) <= 1 and self.LC() == _coerce(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((tuple(self.rep), self.gen))

    # -- arithmetic

    def _binary(self, other: TowerPolyLike) -> tuple[Dup, Dup, Expr]:
        """The representations of ``self`` and ``other`` in one generator:
        a constant (or an element of the field) takes the generator of the
        other operand."""
        gen = self.gen
        if isinstance(other, TowerPoly):
            if other.gen == gen:
                return self.rep, other.rep, gen
            if len(other.rep) <= 1 and not (other.rep and other.rep[0].has(gen)):
                return self.rep, other.rep, gen
            if len(self.rep) <= 1 and not (self.rep and self.rep[0].has(other.gen)):
                return self.rep, other.rep, other.gen
            converted = other.as_poly(gen)
            if converted is None:
                raise PolynomialError('polynomials in %s and %s' % (gen, other.gen))
            return self.rep, converted.rep, gen
        a = _coerce(other)
        if not a:
            return self.rep, [], gen
        if a.has(gen):
            return self.rep, TowerPoly.from_element_strict(a, gen).rep, gen
        return self.rep, [a], gen

    def __neg__(self) -> TowerPoly:
        return self.new(dup_neg(self.rep, TOWER))

    def __add__(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_add(f, g, TOWER), gen)

    def __radd__(self, other: TowerPolyLike) -> TowerPoly:
        return self + other

    def __sub__(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_sub(f, g, TOWER), gen)

    def __rsub__(self, other: TowerPolyLike) -> TowerPoly:
        return (-self) + other

    def __mul__(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_mul(f, g, TOWER), gen)

    def __rmul__(self, other: TowerPolyLike) -> TowerPoly:
        return self * other

    def __pow__(self, n: int) -> TowerPoly:
        if n < 0:
            raise ValueError('negative power of a polynomial')
        return self.new(dup_pow(self.rep, n, TOWER))

    def mul(self, other: TowerPolyLike) -> TowerPoly:
        return self * other

    def mul_ground(self, c: Union[RationalFunction, int]) -> TowerPoly:
        return self.new(dup_mul_ground(self.rep, _coerce(c), TOWER))

    def quo_ground(self, c: Union[RationalFunction, int]) -> TowerPoly:
        inverse = _coerce(c).invert()
        return self.new(dup_mul_ground(self.rep, inverse, TOWER))

    exquo_ground = quo_ground

    def div(self, other: TowerPolyLike) -> tuple[TowerPoly, TowerPoly]:
        f, g, gen = self._binary(other)
        q, r = dup_div(f, g, TOWER)
        return TowerPoly(q, gen), TowerPoly(r, gen)

    def quo(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_quo(f, g, TOWER), gen)

    def rem(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_rem(f, g, TOWER), gen)

    def exquo(self, other: TowerPolyLike) -> TowerPoly:
        """The exact quotient (``ValueError`` when the division is not
        exact, SymPy's ``ExactQuotientFailed``)."""
        f, g, gen = self._binary(other)
        return TowerPoly(dup_exquo(f, g, TOWER), gen)

    def gcd(self, other: TowerPolyLike) -> TowerPoly:
        """The monic gcd (``0`` for two zeros)."""
        f, g, gen = self._binary(other)
        return TowerPoly(dup_gcd(f, g, TOWER), gen)

    def lcm(self, other: TowerPolyLike) -> TowerPoly:
        f, g, gen = self._binary(other)
        return TowerPoly(dup_lcm(f, g, TOWER), gen)

    def monic(self) -> TowerPoly:
        if not self.rep:
            return self
        return self.new(dup_monic(self.rep, TOWER))

    def to_field(self) -> TowerPoly:
        return self

    def diff(self, gen: Optional[Expr] = None) -> TowerPoly:
        """The derivative with respect to the generator, or to another
        generator ``gen`` of the coefficients."""
        if gen is None or gen == self.gen:
            return self.new(dup_diff(self.rep, 1, TOWER))
        return self.new([c.diff(gen) for c in self.rep])

    def eval(self, value: Union[RationalFunction, int]) -> RationalFunction:
        """The value at ``gen = value``."""
        return dup_eval(self.rep, _coerce(value), TOWER)

    def subs(self, gen: Expr, value: RationalFunction) -> TowerPoly:
        """The polynomial with ``gen`` (a generator of the coefficients, or
        the generator itself) replaced by ``value``."""
        if gen == self.gen:
            return TowerPoly.ground(self.eval(value), self.gen) if not value.has(self.gen) else \
                TowerPoly.from_element_strict(_horner_at(self.rep, value), self.gen)
        if value.has(self.gen):
            return TowerPoly.from_element_strict(self.as_element().subs(gen, value), self.gen)
        return self.new([c.subs(gen, value) for c in self.rep])

    def cancel(self, other: TowerPoly, include: bool = True) -> tuple[TowerPoly, TowerPoly]:
        """``(p, q)`` with ``p/q == self/other`` in lowest terms, ``q``
        monic (SymPy's ``Poly.cancel(include=True)`` over a field, up to
        the normalisation of the constant)."""
        f, g, gen = self._binary(other)
        if not g:
            raise ZeroDivisionError('cancel by zero')
        if not f:
            return TowerPoly([], gen), TowerPoly([_ONE_RF], gen)
        h = dup_gcd(f, g, TOWER)
        p = dup_quo(f, h, TOWER)
        q = dup_quo(g, h, TOWER)
        p, q = integral_pair(p, q)
        return TowerPoly(p, gen), TowerPoly(q, gen)

    def resultant(self, other: TowerPoly) -> RationalFunction:
        f, g, _ = self._binary(other)
        return dup_resultant(f, g, TOWER)

    def resultant_prs(self, other: TowerPoly) -> tuple[RationalFunction, list[TowerPoly]]:
        """The resultant and the subresultant PRS (SymPy's
        ``resultant(includePRS=True)``)."""
        f, g, gen = self._binary(other)
        if not f or not g:
            return _ZERO_RF, []
        prs, scalars = dup_inner_subresultants(f, g, TOWER)
        res = _ZERO_RF if dup_degree(prs[-1]) > 0 else scalars[-1]
        return res, [TowerPoly(p, gen) for p in prs]

    def sqf_list(self) -> tuple[RationalFunction, list[tuple[TowerPoly, int]]]:
        c, factors = dup_sqf_list(self.rep, TOWER)
        return c, [(self.new(f), k) for f, k in factors]

    def sqf_list_include(self) -> list[tuple[TowerPoly, int]]:
        """The square-free factors, the content in the first one (SymPy's
        ``sqf_list_include``)."""
        c, factors = self.sqf_list()
        if not factors:
            return [(self.new([c] if c else []), 1)]
        if c.is_one:
            return factors
        if factors[0][1] == 1:
            first = factors[0][0].mul_ground(c)
            return [(first, 1)] + factors[1:]
        return [(self.new([c]), 1)] + factors

    def half_gcdex(self, other: TowerPoly) -> tuple[TowerPoly, TowerPoly]:
        f, g, gen = self._binary(other)
        s, h = dup_half_gcdex(f, g, TOWER)
        return TowerPoly(s, gen), TowerPoly(h, gen)

    def gcdex(self, other: TowerPoly) -> tuple[TowerPoly, TowerPoly, TowerPoly]:
        f, g, gen = self._binary(other)
        s, t, h = dup_gcdex(f, g, TOWER)
        return TowerPoly(s, gen), TowerPoly(t, gen), TowerPoly(h, gen)

    def invert(self, other: TowerPoly) -> TowerPoly:
        """The inverse modulo ``other`` (``ValueError`` when there is none)."""
        f, g, gen = self._binary(other)
        return TowerPoly(dup_invert(f, g, TOWER), gen)

    def compose_shift(self, a: RationalFunction) -> TowerPoly:
        """The polynomial of ``gen + a``."""
        from sympy_extras.polys.densetools import dup_shift
        return self.new(dup_shift(self.rep, a, TOWER))

    def coefficients_have(self, *gens: Expr) -> bool:
        return any(c.has(*gens) for c in self.rep)

    def __iter__(self) -> Iterator[RationalFunction]:
        return iter(self.rep)


def _polynomial_lcm(a: RationalFunction, b: RationalFunction) -> RationalFunction:
    """The lcm of two polynomials (elements with the denominator 1)."""
    if a == b:
        return a
    gens = _merge(a.gens, b.gens)
    R = _ring(len(gens))
    m = R.lcm(_remap(a.numer, a.gens, gens), _remap(b.numer, b.gens, gens))
    return RationalFunction._make(gens, m, R.ring.one, cancel=False)


def _polynomial_gcd(a: RationalFunction, b: RationalFunction) -> RationalFunction:
    """The gcd of two polynomials (elements with the denominator 1)."""
    if not a:
        return b
    if not b or a == b:
        return a
    gens = _merge(a.gens, b.gens)
    R = _ring(len(gens))
    g = R.gcd(_remap(a.numer, a.gens, gens), _remap(b.numer, b.gens, gens))
    return RationalFunction._make(gens, g, R.ring.one, cancel=False)


def integral_pair(p: Dup, q: Dup) -> tuple[Dup, Dup]:
    """SymPy's form of ``p/q`` of ``Poly.cancel(include=True)`` over a
    fraction field: the denominators of the coefficients cleared, the
    common content taken out, the leading coefficient of ``q`` positive."""
    denominator = _ONE_RF
    for c in p + q:
        if c and not (c.denom.is_ground and c.denom.LC == 1):
            denominator = _polynomial_lcm(denominator, c.denominator())
    if not denominator.is_one:
        p = dup_mul_ground(p, denominator, TOWER)
        q = dup_mul_ground(q, denominator, TOWER)
    content = _ZERO_RF
    for c in p + q:
        if c:
            content = _polynomial_gcd(content, c.numerator())
            if content.is_one:
                break
    if content and not content.is_one:
        inverse = content.invert()
        p = dup_mul_ground(p, inverse, TOWER)
        q = dup_mul_ground(q, inverse, TOWER)
    if q and q[0].could_extract_minus_sign():
        p = dup_neg(p, TOWER)
        q = dup_neg(q, TOWER)
    return p, q


def _horner(rep: Dup, gen: Expr) -> RationalFunction:
    """``sum c_k gen**k`` in the field, through a common denominator."""
    gens: tuple[Expr, ...] = (gen,)
    for c in rep:
        gens = _merge(gens, c.gens)
    R = _ring(len(gens))
    i = gens.index(gen)
    denominators: list[PolyElement[int]] = []
    for c in rep:
        if c:
            denominators.append(_remap(c.denom, c.gens, gens))
    common = denominators[0]
    for d in denominators[1:]:
        if d != common:
            common = R.lcm(common, d)
    total = R.ring.zero
    n = len(rep) - 1
    width = max(len(gens), 1)
    for j, c in enumerate(rep):
        if not c:
            continue
        k = n - j
        numer = _remap(c.numer, c.gens, gens)
        denom = _remap(c.denom, c.gens, gens)
        if denom != common:
            numer = numer * common.exquo(denom)
        shift = tuple(k if idx == i else 0 for idx in range(width))
        total = total + numer.mul_monom(shift)
    return RationalFunction._make(gens, total, common, cancel=False)


def _horner_at(rep: Dup, value: RationalFunction) -> RationalFunction:
    result: RationalFunction = _ZERO_RF
    for c in rep:
        result = result * value + c
    return result
