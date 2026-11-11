"""The SymPy face of the sample points
(:mod:`sympy_extras.polys.cad.samplepoints`).

A real algebraic number is given to the face as a SymPy ``Rational``, a
real ``CRootOf`` or a rational multiple of one (the form SymPy's
``real_roots`` gives), and comes out as such: a native
:class:`~sympy_extras.polys.numberfields.qqbar.Algebraic` is written
``CRootOf(m, i)`` for its minimal polynomial ``m`` in the variable it is
a root in (``CRootOf`` itself may take a rational factor out, as it
does for ``x**2 - 8``) and its index ``i`` among the real roots. The
conversions are kept both ways, so that a number which goes out and
comes back is the native number it was, with its refined interval.
Nothing of the sample points stays on SymPy: ``CRootOf`` only names the
numbers.

Examples
========

>>> from sympy import CRootOf, Rational
>>> from sympy.abc import x, y
>>> from sympy_extras.polys.cad.samplepoints_sympy import SamplePoint, compare_real
>>> compare_real(CRootOf(x**2 - 2, 1), 2*CRootOf(2*x**2 - 1, 1))
0
>>> q = SamplePoint().extend(CRootOf(x**2 - 2, 1))
>>> q.real_roots(x**2 - y**2 - 1, [x, y])
[-1, 1]
>>> p = q.extend(Rational(1, 2))
>>> p
SamplePoint(CRootOf(x**2 - 2, 1), 1/2)
>>> p.sign(x**2 + y**2 - 2, [x, y]), p.sign(2*y - 1, [x, y])
(1, 0)
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.polys.domains import QQ
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import ComplexRootOf

from sympy_extras._typing_sympy import DomainElement, ExprLike, Sign, as_expr, free_symbols
from sympy_extras.polys.numberfields.qqbar import Algebraic
from sympy_extras.polys.rootisolation import dup_count_real_roots

from . import samplepoints as native
from .projection_sympy import core_gens, core_poly, native_expr, sympy_errors

__all__ = ['RealAlgebraic', 'SamplePoint', 'Specialization', 'compare_real', 'rational_real_roots',
           'simplest_between', 'rational_between', 'rational_below', 'rational_above', 'core_real',
           'sympy_real']

#: a real algebraic number: a Rational, a real ComplexRootOf or a rational
#: multiple of one
RealAlgebraic = Expr

#: the conversions done so far
_to_core: dict[Expr, Algebraic] = {}
_to_sympy: dict[tuple[tuple[int, ...], int, Basic], Expr] = {}
_KEPT = 20000

#: the variable of the generators of the fields, when printed
_z = Dummy('z')


def _split(r: Expr) -> tuple[Fraction, Optional[ComplexRootOf]]:
    """A real algebraic number as a rational factor and a ``CRootOf``
    (``None`` for a rational number)."""
    if isinstance(r, ComplexRootOf):
        return Fraction(1), r
    if isinstance(r, Mul) and len(r.args) == 2 and isinstance(r.args[0], Rational) \
            and isinstance(r.args[1], ComplexRootOf):
        c = r.args[0]
        return Fraction(int(c.p), int(c.q)), r.args[1]
    if isinstance(r, Rational):
        return Fraction(int(r.p), int(r.q)), None
    raise ValueError("%s is not a rational number, a real CRootOf or a rational multiple of one" % (r,))


def core_real(r: Union[ExprLike, Algebraic]) -> Algebraic:
    """A real algebraic number given to the face as a native one."""
    if isinstance(r, Algebraic):
        return r
    e = as_expr(sympify(r))
    found = _to_core.get(e)
    if found is not None:
        return found
    c, root = _split(e)
    if root is None:
        result = Algebraic.from_rational(c)
    else:
        if not root.is_real:
            raise ValueError("%s is not real" % (root,))
        expr = root.expr
        poly = Poly(expr, *sorted(free_symbols(expr), key=str))
        coefficients = [int(as_expr(a)) for a in poly.all_coeffs()]
        base = native.real_roots_of(coefficients)[int(as_expr(root.args[1]))]
        result = base if c == 1 else _scaled(base, c)
    if len(_to_core) > _KEPT:
        _to_core.clear()
    _to_core[e] = result
    return result


def _scaled(a: Algebraic, c: Fraction) -> Algebraic:
    """``c a`` for a rational ``c`` (nonzero) and an irrational ``a``."""
    m = a.minpoly
    d = len(m) - 1
    # m(x/c) c**d with c = p/q, times q**d: sum m_i p**i q**(d-i) x**(d-i)
    p, q = c.numerator, c.denominator
    scaled = native._normalised([coefficient*p**i*q**(d - i) for i, coefficient in enumerate(m)])
    lo, hi = a.real_interval()
    bounds = (c*lo, c*hi) if c > 0 else (c*hi, c*lo)
    return Algebraic(scaled, bounds, None)


def sympy_real(a: Algebraic, gen: Basic) -> Expr:
    """A native real algebraic number as a SymPy number: a ``Rational``,
    or ``CRootOf`` of its minimal polynomial in ``gen``."""
    if a.is_rational:
        value = a.rational_value()
        return Rational(value.numerator, value.denominator)
    lo, _ = a.real_interval()
    index = dup_count_real_roots([Fraction(c) for c in a.minpoly], None, lo)
    key = (tuple(a.minpoly), index, gen)
    found = _to_sympy.get(key)
    if found is None:
        found = as_expr(ComplexRootOf(Poly(a.minpoly, gen), index))
        if len(_to_sympy) > _KEPT:
            _to_sympy.clear()
        _to_sympy[key] = found
    _to_core.setdefault(found, a)
    return found


def compare_real(a: Union[RealAlgebraic, int], b: Union[RealAlgebraic, int]) -> Sign:
    """:func:`~sympy_extras.polys.cad.samplepoints.compare_real` on SymPy
    numbers.

    >>> from sympy import CRootOf, Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.cad.samplepoints_sympy import compare_real
    >>> compare_real(CRootOf(x**2 - 2, 1), Rational(3, 2)), compare_real(CRootOf(x**2 - 2, 1), CRootOf(x**3 - 2, 0))
    (-1, 1)
    """
    return native.compare_real(core_real(a), core_real(b))


def _order(values: Sequence[Union[RealAlgebraic, int]]) -> list[list[int]]:
    """:func:`~sympy_extras.polys.cad.samplepoints._order` on SymPy
    numbers.

    >>> from sympy import CRootOf
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.cad.samplepoints_sympy import _order
    >>> _order([1, CRootOf(x**2 - 2, 1), 0, CRootOf(x**2 - 2, 1), CRootOf(x**2 - 2, 0)])
    [[4], [2], [0], [1, 3]]
    """
    return native._order([core_real(v) for v in values])


def rational_real_roots(f: Poly) -> list[tuple[RealAlgebraic, int]]:
    """:func:`~sympy_extras.polys.cad.samplepoints.rational_real_roots` on
    a SymPy polynomial: the distinct real roots of the univariate ``f``
    over ``QQ``, increasing, with their multiplicities.

    >>> from sympy import Poly
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.cad.samplepoints_sympy import rational_real_roots
    >>> [(r.evalf(5), m) for r, m in rational_real_roots(Poly((2*x - 1)**2*(x**2 - 2)*(x**2 + 1), x))]
    [(-1.4142, 1), (0.50000, 2), (1.4142, 1)]
    """
    with sympy_errors():
        found = native.rational_real_roots(core_poly(f))
    return [(sympy_real(r, f.gen), m) for r, m in found]


def _floor_scaled(a: Union[RealAlgebraic, int], k: int) -> int:
    """Exact value of `\\lfloor 2^k a \\rfloor` for a real algebraic ``a``."""
    return native.floor_scaled(core_real(a), k)


def _fraction(q: DomainElement) -> Fraction:
    return Fraction(int(q.numerator), int(q.denominator))


def simplest_between(lo: DomainElement, hi: Optional[DomainElement], lo_open: bool = True,
                     hi_open: bool = True) -> DomainElement:
    """:func:`~sympy_extras.polys.cad.samplepoints.simplest_between` on the
    elements of SymPy's ``QQ``.

    >>> from sympy import QQ
    >>> from sympy_extras.polys.cad.samplepoints_sympy import simplest_between
    >>> simplest_between(QQ(1, 3), QQ(1, 2))
    2/5
    """
    found = native.simplest_between(_fraction(lo), None if hi is None else _fraction(hi), lo_open, hi_open)
    return QQ(found.numerator, found.denominator)


def _rational(q: Fraction) -> Rational:
    return Rational(q.numerator, q.denominator)


def rational_between(a: Union[RealAlgebraic, int], b: Union[RealAlgebraic, int]) -> Rational:
    """:func:`~sympy_extras.polys.cad.samplepoints.rational_between` on
    SymPy numbers.

    >>> from sympy import CRootOf
    >>> from sympy.abc import x
    >>> from sympy_extras.polys.cad.samplepoints_sympy import rational_between
    >>> rational_between(CRootOf(x**2 - 2, 0), CRootOf(x**2 - 2, 1)), rational_between(1, CRootOf(x**2 - 2, 1))
    (0, 5/4)
    """
    return _rational(native.rational_between(core_real(a), core_real(b)))


def rational_below(a: Union[RealAlgebraic, int]) -> Rational:
    """:func:`~sympy_extras.polys.cad.samplepoints.rational_below` on a
    SymPy number."""
    return _rational(native.rational_below(core_real(a)))


def rational_above(a: Union[RealAlgebraic, int]) -> Rational:
    """:func:`~sympy_extras.polys.cad.samplepoints.rational_above` on a
    SymPy number."""
    return _rational(native.rational_above(core_real(a)))


class Specialization:
    """:class:`~sympy_extras.polys.cad.samplepoints.Specialization` with
    the roots as SymPy numbers."""

    __slots__ = ('degree', 'leading_sign', 'roots')

    def __init__(self, degree: int, leading_sign: Sign, roots: Sequence[tuple[RealAlgebraic, int]]) -> None:
        self.degree: int = degree
        self.leading_sign: Sign = leading_sign
        self.roots: list[tuple[RealAlgebraic, int]] = list(roots)

    def sign_before(self, position: int) -> Sign:
        """The sign of the polynomial between its roots of index
        ``position - 1`` and ``position``."""
        if not 0 <= position <= len(self.roots):
            raise ValueError("there are %d roots" % len(self.roots))
        odd = sum(multiplicity for _, multiplicity in self.roots[position:]) % 2
        return -self.leading_sign if odd else self.leading_sign


#: the polynomial arguments
PolyLike = Union[Poly, ExprLike]


class SamplePoint:
    """:class:`~sympy_extras.polys.cad.samplepoints.SamplePoint` with
    SymPy polynomials, generators and numbers: the native point
    ``native`` and its coordinates as SymPy numbers (``as_exprs``)."""

    __slots__ = ('native', '_labels', '_specializations')

    def __init__(self, point: Optional[native.SamplePoint] = None, labels: Sequence[Expr] = ()) -> None:
        self.native: native.SamplePoint = native.SamplePoint() if point is None else point
        if len(labels) != len(self.native):
            raise ValueError("one SymPy number per coordinate")
        self._labels: tuple[Expr, ...] = tuple(labels)
        self._specializations: dict[tuple[Poly, tuple[Basic, ...]], Specialization] = {}

    def __len__(self) -> int:
        return len(self.native)

    def __repr__(self) -> str:
        return "SamplePoint(%s)" % ", ".join(str(c) for c in self._labels)

    def as_exprs(self) -> tuple[Expr, ...]:
        """The coordinates as SymPy expressions."""
        return self._labels

    @property
    def degree(self) -> int:
        """Degree of the field of the coordinates over the rationals."""
        return self.native.degree

    @property
    def theta(self) -> Optional[Expr]:
        """The generator of the field of the coordinates, ``None`` for
        rational coordinates."""
        theta = self.native.theta
        if theta is None:
            return None
        for value, label in zip(self.native.values, self._labels):
            if value is theta:
                return label
        return sympy_real(theta, _z)

    def extend(self, root: Union[RealAlgebraic, int]) -> SamplePoint:
        """The point with the coordinate ``root`` appended: a
        ``Rational``, a real ``CRootOf`` or a product of the two."""
        r = as_expr(sympify(root))
        with sympy_errors():
            point = self.native.extend(core_real(r))
        return SamplePoint(point, self._labels + (r,))

    def _poly(self, poly: PolyLike, gens: Sequence[Symbol]) -> Poly:
        generators = tuple(gens)
        if isinstance(poly, Poly) and poly.gens == generators:
            return poly
        return Poly(poly.as_expr() if isinstance(poly, Poly) else poly, *generators)

    def sign(self, poly: PolyLike, gens: Sequence[Symbol]) -> Sign:
        """Exact sign of ``poly`` in ``gens`` at the point."""
        if len(gens) != len(self):
            raise ValueError("expected %d generators" % len(self))
        if not gens:
            if isinstance(poly, Poly):
                raise ValueError("a polynomial needs generators")
            return self.native.sign(native_expr(as_expr(sympify(poly))), ())
        f = self._poly(poly, gens)
        with sympy_errors():
            return self.native.sign(core_poly(f, core_gens(gens)), core_gens(gens))

    def specialization(self, poly: PolyLike, gens: Sequence[Symbol]) -> Specialization:
        """``poly`` in the last generator of ``gens`` with the point put
        for the others: degree, leading sign, real roots (SymPy numbers in
        the last generator) with their multiplicities."""
        if len(gens) != len(self) + 1:
            raise ValueError("expected %d generators" % (len(self) + 1))
        f = self._poly(poly, gens)
        key = (f, tuple(gens))
        found = self._specializations.get(key)
        if found is not None:
            return found
        generators = core_gens(gens)
        with sympy_errors():
            s = self.native.specialization(core_poly(f, generators), generators)
        last = gens[-1]
        found = Specialization(s.degree, s.leading_sign, [(sympy_real(r, last), m) for r, m in s.roots])
        self._specializations[key] = found
        return found

    def real_roots(self, poly: PolyLike, gens: Sequence[Symbol]) -> Optional[list[RealAlgebraic]]:
        """Sorted distinct real roots of ``poly`` in the last generator of
        ``gens`` at the point, ``None`` if it vanishes identically."""
        found = self.specialization(poly, gens)
        if found.degree < 0:
            return None
        return [r for r, _ in found.roots]
