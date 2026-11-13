r"""Complex roots of polynomials with rational coefficients as nodes of the
core: SymPy's ``CRootOf``, ported.

``ComplexRootOf(f, k)`` is the ``k``-th root, from 0, of the univariate
polynomial ``f`` with rational coefficients, in SymPy's order of the roots
(the order of ``CRootOf``, which SymPy's ``Poly.all_roots`` follows):

* the real roots first, in increasing order;
* then the roots which are not real, the irreducible factors one after
  the other (in the order of SymPy's ``ordered``: smaller expressions
  first, then ``default_sort_key``), and the roots of one factor in the
  order of SymPy's isolating rectangles: the root below the real axis
  just before its conjugate, the pairs by the lower left corners of the
  rectangles of the roots above the axis;
* a root of multiplicity ``m`` is counted ``m`` times.

**The rectangles.** SymPy isolates the roots above the real axis by the
bisection of Collins and Krandick [CK]_: the rectangle $[-B, B] \times
[0, B]$, $B = 2 \max_i |a_i/a_n|$, is cut in two halves across its longer
side (vertically when it is wider than high), the halves holding roots
are cut again, and a half holding one root is that root's rectangle. A
root on a vertical cut belongs to the right half, a root on a horizontal
cut to the lower half (SymPy's winding numbers exclude the southern and
eastern edges). The rectangles depend only on which half each root lies
in, so that they are found here from exact isolating regions
(:class:`~sympy_extras.polys.numberfields.qqbar.Algebraic`) instead of
SymPy's winding numbers: a region is refined until it lies on one side of
the cut, and a root on the cut line itself is recognised exactly, as a
real root of $\gcd(\operatorname{Re} f(c + i y), \operatorname{Im} f(c +
i y))$ (or of $f(t + i c)$ for a horizontal cut) whose point lies in the
region of the root.

:class:`ComplexRootOf` is the node, which never evaluates; :func:`rootof`
is the evaluating function, as SymPy's constructor: a root of a linear
factor is a rational number, a root of a quadratic factor or of a
binomial is written in radicals with ``radicals=True`` (SymPy's
``_roots_trivial``), and any other root is the node of its irreducible
factor (primitive, positive leading coefficient) with its index among the
roots of that factor. :func:`all_roots` and :func:`real_roots` list the
roots as SymPy's ``Poly.all_roots`` and ``Poly.real_roots`` do. The
numerical value of a node is that of its algebraic number
(:func:`algebraic_number`), refined exactly to any precision; a root on
the imaginary axis has a real part which is exactly zero.

The module is native: it imports no SymPy. Its classes and functions are
a port of SymPy's ``sympy.polys.rootoftools`` (``ComplexRootOf``: the
order of the roots, the normalisation of the polynomial, the trivial
roots) and of the bisection order of ``dup_isolate_complex_roots_sqf``
(``sympy.polys.rootisolation``); :mod:`sympy_extras.interop` converts the
node to and from SymPy's ``CRootOf``.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.polys.complexrootof import ComplexRootOf, rootof, all_roots
>>> x = Symbol('x')
>>> r = rootof(x**5 - x - 1, 2); r
CRootOf(x**5 - x - 1, 2)
>>> r.evalf(10), r.is_real
(-0.7648844336 + 0.352471546*I, False)
>>> all_roots(x**3 - 2)
[2**(1/3), -2**(1/3)/2 - 2**(1/3)*sqrt(3)*I/2, -2**(1/3)/2 + 2**(1/3)*sqrt(3)*I/2]
>>> all_roots((x**2 + 1)*(x**2 + 4), radicals=False)
[CRootOf(x**2 + 1, 0), CRootOf(x**2 + 1, 1), CRootOf(x**2 + 4, 0), CRootOf(x**2 + 4, 1)]
>>> ComplexRootOf(x**2 + 4, 1).evalf()
2.0*I

References
==========

.. [CK] G. E. Collins, W. Krandick, *An efficient algorithm for infallible
   polynomial complex root isolation*, ISSAC '92, 189-194.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rootoftools.py (ComplexRootOf), sympy/polys/rootisolation.py
# (dup_isolate_complex_roots_sqf: the bisection order) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: an unevaluating node ComplexRootOf and the evaluating function rootof; the
# rectangles of the bisection found from exact isolating regions (qqbar.Algebraic), not from
# SymPy's winding numbers; numerical values by exact refinement.
# ---------------------------------------------------------------------
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from typing import Final, Optional, Sequence, Union

import mpmath

from sympy_extras.core.basic import Basic, NodePrinter, ReplacementRule
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import ImaginaryUnit, Integer, Rational, number_from_fraction
from sympy_extras.core.sorting import SortKey, _nodes
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.densetools import dup_shift
from sympy_extras.polys.domains import QQ, ZZ
from sympy_extras.polys.euclidtools import dup_qq_gcd
from sympy_extras.polys.factortools import dup_zz_factor
from sympy_extras.polys.numberfields.qqbar import Algebraic, _roots as _unsorted_roots
from sympy_extras.polys.polyerrors import BasePolynomialError, CoercionFailed
from sympy_extras.polys.polyroots import preprocess_roots
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rootisolation import Interval, dup_isolate_real_roots_sqf, dup_refine_real_root
from sympy_extras.polys.sqfreetools import dup_sqf_part

__all__ = ['ComplexRootOf', 'CRootOf', 'rootof', 'all_roots', 'real_roots', 'polynomial_roots',
           'algebraic_number', 'root_count', 'complex_rectangles', 'real_roots_indexed']

#: a polynomial with integer coefficients, highest degree first
IntegerPoly = tuple[int, ...]

_ZERO: Final = Fraction(0)


# ---------------------------------------------------------------------------
# The roots on the lines of the cuts

class _LineRoots:
    """The roots of a polynomial on one line: the real roots of the gcd
    ``g`` of the real and imaginary parts of the polynomial along the
    line, as isolating intervals refined on demand."""

    __slots__ = ('_g', '_intervals')

    def __init__(self, g: list[Fraction], intervals: list[Interval]) -> None:
        self._g: Final[list[Fraction]] = g
        self._intervals: list[Interval] = intervals

    @property
    def empty(self) -> bool:
        return not self._intervals

    def refined(self, eps: Fraction) -> list[Interval]:
        """The intervals, narrower than ``eps``."""
        self._intervals = [dup_refine_real_root(self._g, a, b, eps) for a, b in self._intervals]
        return self._intervals


def _line_roots(real: list[Fraction], imaginary: list[Fraction]) -> _LineRoots:
    """The common real roots of ``real`` and ``imaginary`` (dense, highest
    degree first)."""
    g = dup_qq_gcd(_stripped(real), _stripped(imaginary))
    if len(g) < 2:
        return _LineRoots([], [])
    g = dup_sqf_part(g, QQ)
    return _LineRoots(g, dup_isolate_real_roots_sqf(g))


def _stripped(f: list[Fraction]) -> list[Fraction]:
    k = 0
    while k < len(f) and f[k] == 0:
        k += 1
    return f[k:]


def _vertical(f: IntegerPoly, c: Fraction) -> _LineRoots:
    """The roots ``c + i y`` of ``f``, ``y`` real."""
    h = dup_shift([Fraction(a) for a in f], c, QQ)
    n = len(h) - 1
    real: list[Fraction] = []
    imaginary: list[Fraction] = []
    # h(i y) = sum h_k i**k y**k, h_k the coefficient of degree k
    for k in range(n, -1, -1):
        coefficient = h[n - k]
        sign = 1 if k % 4 in (0, 1) else -1
        if k % 2 == 0:
            real.append(sign*coefficient)
            imaginary.append(_ZERO)
        else:
            real.append(_ZERO)
            imaginary.append(sign*coefficient)
    return _line_roots(real, imaginary)


def _horizontal(f: IntegerPoly, c: Fraction) -> _LineRoots:
    """The roots ``t + i c`` of ``f``, ``t`` real."""
    # Horner's rule in t + i c, the coefficients as (real, imaginary)
    re: list[Fraction] = []
    im: list[Fraction] = []
    for a in f:
        # (re + i im)(t + i c): the new coefficient of t**j is the old one
        # of t**(j-1) plus i c times the old one of t**j
        new_re = re + [_ZERO]
        new_im = im + [_ZERO]
        for j in range(len(re)):
            new_re[j + 1] -= c*im[j]
            new_im[j + 1] += c*re[j]
        new_re[-1] += a
        re, im = new_re, new_im
    return _line_roots(re, im)


class _Lines:
    """The roots of an irreducible polynomial on the lines of the cuts,
    computed once per line."""

    __slots__ = ('_f', '_vertical', '_horizontal')

    def __init__(self, f: IntegerPoly) -> None:
        self._f: Final[IntegerPoly] = f
        self._vertical: Final[dict[Fraction, _LineRoots]] = {}
        self._horizontal: Final[dict[Fraction, _LineRoots]] = {}

    def vertical(self, c: Fraction) -> _LineRoots:
        found = self._vertical.get(c)
        if found is None:
            found = _vertical(self._f, c)
            self._vertical[c] = found
        return found

    def horizontal(self, c: Fraction) -> _LineRoots:
        found = self._horizontal.get(c)
        if found is None:
            found = _horizontal(self._f, c)
            self._horizontal[c] = found
        return found


_LINES: OrderedDict[IntegerPoly, _Lines] = OrderedDict()


def _lines(f: IntegerPoly) -> _Lines:
    found = _LINES.get(f)
    if found is None:
        found = _Lines(f)
        _LINES[f] = found
        while len(_LINES) > _KEPT:
            _LINES.popitem(last=False)
    return found


def _finer(eps: Fraction, width: Fraction) -> Fraction:
    """A width for the line roots, below ``eps`` and the width of the
    region they are to fit in."""
    return min(eps, width/4) if width > 0 else eps


def _compare_real_part(z: Algebraic, c: Fraction, eps: Fraction) -> int:
    """The sign of ``re(z) - c`` for a number ``z`` which is not real,
    decided exactly (``eps`` the first width of the regions tried)."""
    lines: Optional[_LineRoots] = None
    while True:
        re_min, re_max, im_min, im_max = z.enclosure(eps)
        if re_max < c:
            return -1
        if re_min > c:
            return 1
        if lines is None:
            lines = _lines(tuple(z.minpoly)).vertical(c)
        if not lines.empty:
            # a root of the minimal polynomial on the line, in the region
            # which isolates z, is z
            for a, b in lines.refined(_finer(eps, im_max - im_min)):
                if im_min <= a and b <= im_max:
                    return 0
        eps = eps/16


def _compare_imaginary_part(z: Algebraic, c: Fraction, eps: Fraction) -> int:
    """The sign of ``im(z) - c``, decided exactly."""
    lines: Optional[_LineRoots] = None
    while True:
        re_min, re_max, im_min, im_max = z.enclosure(eps)
        if im_max < c:
            return -1
        if im_min > c:
            return 1
        if lines is None:
            lines = _lines(tuple(z.minpoly)).horizontal(c)
        if not lines.empty:
            for a, b in lines.refined(_finer(eps, re_max - re_min)):
                if re_min <= a and b <= re_max:
                    return 0
        eps = eps/16


# ---------------------------------------------------------------------------
# SymPy's order of the roots of a square-free polynomial

def _bound(f: Sequence[int]) -> Fraction:
    """SymPy's bound of the isolation: ``2 max |a_i/a_n|``."""
    lc = abs(f[0])
    return 2*max(Fraction(abs(c), lc) for c in f)


#: a cell of the bisection: its lower left and upper right corners and the
#: roots in it
_Cell = tuple[Fraction, Fraction, Fraction, Fraction, list[Algebraic]]


# PORTED FROM SYMPY: sympy/polys/rootisolation.py:dup_isolate_complex_roots_sqf
def complex_rectangles(f: Sequence[int], upper: Sequence[Algebraic]
                       ) -> list[tuple[tuple[Fraction, Fraction, Fraction, Fraction], Algebraic]]:
    """SymPy's isolating rectangles ``(ax, ay, bx, by)`` of the roots above
    the real axis of the square-free ``f`` (integer coefficients, highest
    degree first), in SymPy's order (by the lower left corners), each with
    its root among ``upper`` (all the roots of ``f`` above the axis).

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.polys.numberfields.qqbar import algebraic_roots
    >>> from sympy_extras.polys.complexrootof import complex_rectangles
    >>> f = [1, 0, 6, 0, 4]                       # x**4 + 6*x**2 + 4: roots on the imaginary axis
    >>> upper = [r for r in algebraic_roots(f) if not r.is_real and r.enclosure(Fraction(1))[3] > 0]
    >>> [box for box, _ in complex_rectangles(f, upper)]
    [(Fraction(0, 1), Fraction(0, 1), Fraction(3, 1), Fraction(3, 2)), (Fraction(0, 1), Fraction(3, 2), Fraction(3, 1), Fraction(3, 1))]
    """
    if not upper:
        return []
    B = _bound(f)
    found: list[tuple[tuple[Fraction, Fraction, Fraction, Fraction], Algebraic]] = []
    cells: list[_Cell] = [(-B, _ZERO, B, B, list(upper))]
    first = True
    while cells:
        u, v, s, t, roots = cells.pop()
        if len(roots) == 1 and not first:
            found.append(((u, v, s, t), roots[0]))
            continue
        first = False
        if s - u > t - v:
            x = (u + s)/2
            eps = (s - u)/4
            left = [r for r in roots if _compare_real_part(r, x, eps) < 0]
            right = [r for r in roots if not any(r is l for l in left)]
            if left:
                cells.append((u, v, x, t, left))
            if right:
                cells.append((x, v, s, t, right))
        else:
            y = (v + t)/2
            eps = (t - v)/4
            lower = [r for r in roots if _compare_imaginary_part(r, y, eps) <= 0]
            higher = [r for r in roots if not any(r is l for l in lower)]
            if lower:
                cells.append((u, v, s, y, lower))
            if higher:
                cells.append((u, y, s, t, higher))
    found.sort(key=lambda item: (item[0][0], item[0][1]))
    return found


def _above(z: Algebraic) -> bool:
    """Whether a number which is not real lies above the real axis."""
    eps = Fraction(1)
    while True:
        _, _, im_min, im_max = z.enclosure(eps)
        if im_min > 0:
            return True
        if im_max < 0:
            return False
        eps = eps/16


#: the roots of the irreducible polynomials in SymPy's order, with the
#: number of real roots (the refinements of the regions are kept)
_ROOTS: OrderedDict[IntegerPoly, tuple[int, list[Algebraic]]] = OrderedDict()
_KEPT = 2000


def _irreducible_roots(f: IntegerPoly) -> tuple[int, list[Algebraic]]:
    """The number of real roots and the roots of the irreducible ``f``
    (primitive, positive leading coefficient) in SymPy's order."""
    found = _ROOTS.get(f)
    if found is not None:
        _ROOTS.move_to_end(f)
        return found
    if len(f) == 2:
        result: tuple[int, list[Algebraic]] = (1, [Algebraic.from_rational(Fraction(-f[1], f[0]))])
    else:
        # unsorted: qqbar's order of the complex roots compares their real
        # parts exactly (resultants), which SymPy's order does not need
        reals, others = _unsorted_roots(list(f))
        reals.sort(key=_RealKey)
        upper = [r for r in others if _above(r)]
        ordered: list[Algebraic] = list(reals)
        for _, r in complex_rectangles(f, upper):
            ordered.extend([r.conjugate(), r])
        result = (len(reals), ordered)
    _ROOTS[f] = result
    while len(_ROOTS) > _KEPT:
        _ROOTS.popitem(last=False)
    return result


# ---------------------------------------------------------------------------
# Polynomials

def _integer_coefficients(coefficients: Sequence[Fraction]) -> list[int]:
    """The coefficients times their common denominator."""
    denominator = 1
    for c in coefficients:
        denominator = denominator*c.denominator//_gcd(denominator, c.denominator)
    return [int(c*denominator) for c in coefficients]


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _factor_expr(f: IntegerPoly, x: Symbol) -> Expr:
    n = len(f) - 1
    return Poly.from_dict({(n - i,): c for i, c in enumerate(f) if c}, x).as_expr()


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:_pure_factors
def _pure_factors(coefficients: Sequence[Fraction]) -> list[tuple[IntegerPoly, int]]:
    """The irreducible factors over ``ZZ`` (primitive, positive leading
    coefficients) with their multiplicities."""
    _, factors = dup_zz_factor(_integer_coefficients(coefficients), ZZ)
    return [(tuple(g), k) for g, k in factors if len(g) > 1]


#: a root by its irreducible factor and its index among the roots of the factor
_Address = tuple[IntegerPoly, int]


def _factor_key(f: IntegerPoly, x: Symbol) -> tuple[int, SortKey]:
    """SymPy's ``ordered`` of the factors: ``_nodes`` then ``default_sort_key``."""
    e = _factor_expr(f, x)
    return _nodes(e), e.sort_key()


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._all_roots
def _addresses(coefficients: Sequence[Fraction], x: Symbol, real: bool) -> list[_Address]:
    """The roots of the polynomial in SymPy's order, with their
    multiplicities, by their factors and their indices in them: the real
    roots in increasing order, then (without ``real``) the others factor
    by factor in SymPy's order of the factors."""
    factors = _pure_factors(coefficients)
    reals: list[tuple[Algebraic, _Address, int]] = []
    for g, k in factors:
        count, roots = _irreducible_roots(g)
        reals.extend((roots[i], (g, i), k) for i in range(count))
    reals.sort(key=lambda item: _RealKey(item[0]))
    result: list[_Address] = [address for _, address, k in reals for _ in range(k)]
    if real:
        return result
    for g, k in sorted(factors, key=lambda item: _factor_key(item[0], x)):
        count, roots = _irreducible_roots(g)
        for i in range(count, len(roots)):
            result.extend([(g, i)]*k)
    return result


class _RealKey:
    """Sorting key: the order of the real numbers."""

    __slots__ = ('value',)

    def __init__(self, value: Algebraic) -> None:
        self.value: Final[Algebraic] = value

    def __lt__(self, other: _RealKey) -> bool:
        return self.value < other.value


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._roots_trivial
def _trivial(f: IntegerPoly, index: int, x: Symbol, radicals: bool) -> Optional[Expr]:
    """The root of a linear factor, and with ``radicals`` of a quadratic
    one or of a binomial, in radicals (SymPy's ``_roots_trivial``)."""
    if len(f) == 2:
        return number_from_fraction(Fraction(-f[1], f[0]))
    if not radicals:
        return None
    if len(f) == 3 or (sum(1 for c in f if c) == 2 and f[-1] != 0):
        from sympy_extras.polys.roots import _trivial_roots
        found = _trivial_roots(Poly(_factor_expr(f, x), x))
        if found is not None and 0 <= index < len(found):
            return found[index]
    return None


def _root(address: _Address, x: Symbol, radicals: bool) -> Expr:
    f, index = address
    trivial = _trivial(f, index, x, radicals)
    if trivial is not None:
        return trivial
    return ComplexRootOf(_factor_expr(f, x), index)


def polynomial_roots(coefficients: Sequence[Fraction], x: Symbol, radicals: bool = True,
                     real: bool = False) -> list[Expr]:
    """The roots of the polynomial with the rational ``coefficients``
    (highest degree first) in ``x``, as SymPy's ``Poly.all_roots`` (with
    ``real``, ``Poly.real_roots``) gives them: with their multiplicities,
    in SymPy's order, rational numbers, radicals (``radicals``: the roots
    of quadratic factors and binomials) or :class:`ComplexRootOf`.

    Examples
    ========

    >>> from fractions import Fraction
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import polynomial_roots
    >>> x = Symbol('x')
    >>> polynomial_roots([Fraction(1), Fraction(0), Fraction(-1), Fraction(-1)], x)
    [CRootOf(x**3 - x - 1, 0), CRootOf(x**3 - x - 1, 1), CRootOf(x**3 - x - 1, 2)]
    >>> polynomial_roots([Fraction(1), Fraction(-2), Fraction(1)], x, real=True)
    [1, 1]
    """
    if not any(coefficients):
        raise ValueError("the roots of the zero polynomial")
    coeff, scaled = _preprocessed(coefficients, x)
    return [_scaled(coeff, _root(address, x, radicals)) for address in _addresses(scaled, x, real)]


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._get_roots
def _preprocessed(coefficients: Sequence[Fraction], x: Symbol) -> tuple[Expr, list[Fraction]]:
    """SymPy's ``preprocess_roots``: the roots of the polynomial are
    ``coeff`` times those of the polynomial returned (an integer basis of
    the roots taken out)."""
    n = len(coefficients) - 1
    poly = Poly.from_dict({(n - i,): c for i, c in enumerate(coefficients) if c}, x)
    coeff, scaled = preprocess_roots(poly)
    found: list[Fraction] = []
    for c in scaled.all_coeffs():
        if not isinstance(c, Rational):
            raise TypeError("rational coefficients are expected, got %s" % (c,))
        found.append(c.as_fraction())
    return coeff, found


def _scaled(coeff: Expr, root: Expr) -> Expr:
    if isinstance(coeff, Integer) and coeff.p == 1:
        return root
    return mul(coeff, root)


def _as_poly(f: ExprLike, x: Optional[Symbol]) -> tuple[list[Fraction], Symbol]:
    """The rational coefficients of the univariate polynomial ``f`` and its
    generator (``x``, or the one symbol of ``f``)."""
    e = to_expr(f)
    if x is None:
        symbols = sorted(e.free_symbols, key=str)
        if len(symbols) != 1:
            raise ValueError("a univariate polynomial is expected, got %s" % (e,))
        x = symbols[0]
    try:
        poly = Poly(e, x)
    except CoercionFailed:
        raise NotImplementedError("CRootOf is not supported over the coefficients of %s" % (e,))
    except BasePolynomialError:
        raise ValueError("%s is not a polynomial in %s" % (e, x))
    if poly.domain not in ('ZZ', 'QQ') or not poly.is_univariate:
        raise NotImplementedError("CRootOf is not supported over %s" % poly.domain)
    coefficients: list[Fraction] = []
    for c in poly.all_coeffs():
        if not isinstance(c, Rational):
            raise NotImplementedError("CRootOf is not supported over %s" % poly.domain)
        coefficients.append(c.as_fraction())
    return coefficients, x


def all_roots(f: ExprLike, x: Optional[Symbol] = None, radicals: bool = True) -> list[Expr]:
    """The roots of the polynomial ``f`` with rational coefficients, with
    multiplicities, in SymPy's order (SymPy's ``Poly(f, x).all_roots``),
    see :func:`polynomial_roots`.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import all_roots
    >>> x = Symbol('x')
    >>> all_roots(x**4 + 1, radicals=False)
    [CRootOf(x**4 + 1, 0), CRootOf(x**4 + 1, 1), CRootOf(x**4 + 1, 2), CRootOf(x**4 + 1, 3)]
    """
    coefficients, gen = _as_poly(f, x)
    if len(coefficients) < 2:
        return []
    return polynomial_roots(coefficients, gen, radicals)


def real_roots(f: ExprLike, x: Optional[Symbol] = None, radicals: bool = True) -> list[Expr]:
    """The real roots of ``f`` with multiplicities, in increasing order
    (SymPy's ``Poly(f, x).real_roots``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import real_roots
    >>> x = Symbol('x')
    >>> real_roots((x**3 - x - 1)*(x**2 - 2))
    [-sqrt(2), CRootOf(x**3 - x - 1, 0), sqrt(2)]
    """
    coefficients, gen = _as_poly(f, x)
    if len(coefficients) < 2:
        return []
    return polynomial_roots(coefficients, gen, radicals, real=True)


def root_count(f: ExprLike, x: Optional[Symbol] = None) -> tuple[int, int]:
    """The numbers of real roots and of all roots of ``f``, with
    multiplicities.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import root_count
    >>> root_count(Symbol('x')**5 - Symbol('x') - 1)
    (1, 5)
    """
    coefficients, gen = _as_poly(f, x)
    _, scaled = _preprocessed(coefficients, gen)
    return len(_addresses(scaled, gen, True)), len(coefficients) - 1


def rootof(f: ExprLike, index: Union[int, Integer], x: Optional[Symbol] = None, radicals: bool = True) -> Expr:
    """The ``index``-th root of the polynomial ``f`` in SymPy's order of
    the roots, evaluated as SymPy's ``rootof``: a rational number, in
    radicals with ``radicals`` for a root of a quadratic factor or of a
    binomial, otherwise the :class:`ComplexRootOf` of its irreducible
    factor. A negative index counts from the end.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import rootof
    >>> x = Symbol('x')
    >>> rootof(x**2 - 2, 0), rootof(x**2 - 2, 0, radicals=False), rootof(2*x**3 - 2*x - 2, -1)
    (-sqrt(2), CRootOf(x**2 - 2, 0), CRootOf(x**3 - x - 1, 2))
    >>> rootof((x - 1)*(x**3 - x - 1), 1)
    CRootOf(x**3 - x - 1, 0)
    """
    k = index.p if isinstance(index, Integer) else index
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("expected an integer root index, got %s" % (index,))
    coefficients, gen = _as_poly(f, x)
    degree = len(coefficients) - 1
    if degree <= 0:
        raise ValueError("cannot construct CRootOf object for %s" % (f,))
    if k < -degree or k >= degree:
        raise IndexError("root index out of [%d, %d] range, got %d" % (-degree, degree - 1, k))
    if k < 0:
        k += degree
    coeff, scaled = _preprocessed(coefficients, gen)
    addresses = _addresses(scaled, gen, False)
    return _scaled(coeff, _root(addresses[k], gen, radicals))


# ---------------------------------------------------------------------------
# The node

class ComplexRootOf(Expr):
    """The ``index``-th root, from 0, of the univariate polynomial ``f``
    with rational coefficients, in SymPy's order of the roots (SymPy's
    ``CRootOf``). The constructor checks and stores its arguments
    (:func:`rootof` evaluates); the variable of ``f`` is bound, and two
    roots of the same polynomial in different variables are equal.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.complexrootof import ComplexRootOf
    >>> x, y = symbols('x y')
    >>> r = ComplexRootOf(x**3 - x - 1, 1)
    >>> r, r.args, r.free_symbols, r == ComplexRootOf(y**3 - y - 1, 1)
    (CRootOf(x**3 - x - 1, 1), (x**3 - x - 1, 1), set(), True)
    >>> r.subs(x, 2) == r, r.is_real, ComplexRootOf(x**3 - x - 1, 0).is_positive
    (True, False, True)
    """

    __slots__ = ('_polynomial', '_variable', '_coefficients', '_index')

    def __init__(self, f: Expr, k: Union[int, Integer]) -> None:
        if not isinstance(f, Expr):
            raise TypeError("an expression is expected, got %s" % (f,))
        index = k.p if isinstance(k, Integer) else k
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("expected an integer root index, got %s" % (k,))
        symbols = list(f.free_symbols)
        if len(symbols) != 1:
            raise ValueError("CRootOf(f, k) takes a univariate polynomial, got %s" % (f,))
        coefficients, variable = _as_poly(f, symbols[0])
        if len(coefficients) < 2:
            raise ValueError("cannot construct CRootOf object for %s" % (f,))
        if not 0 <= index < len(coefficients) - 1:
            raise IndexError("root index out of [0, %d] range, got %d" % (len(coefficients) - 2, index))
        self._polynomial: Final[Expr] = f
        self._variable: Final[Symbol] = variable
        self._coefficients: Final[tuple[Fraction, ...]] = tuple(coefficients)
        self._index: Final[int] = index
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._polynomial, Integer(self._index))

    def _hashable_content(self) -> tuple[Basic, ...]:
        # the variable is bound: as SymPy's PurePoly, the polynomial is its
        # coefficients
        return tuple(number_from_fraction(c) for c in self._coefficients) + (Integer(self._index),)

    @property
    def expr(self) -> Expr:
        """The polynomial."""
        return self._polynomial

    @property
    def variable(self) -> Symbol:
        """The variable of the polynomial."""
        return self._variable

    @property
    def index(self) -> int:
        """The index of the root in SymPy's order."""
        return self._index

    @property
    def coefficients(self) -> tuple[Fraction, ...]:
        """The coefficients of the polynomial, highest degree first."""
        return self._coefficients

    @property
    def free_symbols(self) -> set[Symbol]:
        return set()

    def _is_number_node(self) -> Optional[bool]:
        return True

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, k = args
        if not (isinstance(f, Expr) and isinstance(k, Integer)):
            raise TypeError("invalid arguments of a CRootOf: %s" % (args,))
        return ComplexRootOf(f, k)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        f, k = args
        if not (isinstance(f, Expr) and isinstance(k, Integer)):
            raise TypeError("invalid arguments of a CRootOf: %s" % (args,))
        return rootof(f, k.p, radicals=False)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # a number: no substitution of its bound variable changes it
        return new if self == old else self

    def _xreplace(self, rule: ReplacementRule) -> Basic:
        return rule[self] if self in rule else self

    def _xreplace_evaluated(self, rule: ReplacementRule) -> Basic:
        return rule[self] if self in rule else self

    def doit(self) -> Expr:
        return self

    def address(self) -> tuple[IntegerPoly, int]:
        """The irreducible factor (primitive, positive leading coefficient)
        of the root and its index among the roots of that factor."""
        return _address_of(self._coefficients, self._variable, self._index)

    def is_canonical(self) -> bool:
        """Whether the polynomial is irreducible, primitive with integer
        coefficients and a positive leading coefficient (the form which
        :func:`rootof` gives)."""
        f, index = self.address()
        return index == self._index and tuple(Fraction(c) for c in f) == self._coefficients

    def algebraic(self) -> Algebraic:
        """The algebraic number of the root."""
        f, index = self.address()
        return _irreducible_roots(f)[1][index]

    def polynomial_root(self) -> tuple[Poly, int]:
        """The polynomial and the index of the root as SymPy's ``CRootOf``
        has them: the irreducible factor in the variable of the node."""
        f, index = self.address()
        return Poly(_factor_expr(f, self._variable), self._variable), index

    def _eval_facts(self) -> dict[str, bool]:
        a = self.algebraic()
        facts = {'commutative': True, 'complex': True, 'finite': True, 'algebraic': True,
                 'transcendental': False}
        if a.is_real:
            s = a.sign()
            facts.update({'real': True, 'extended_real': True, 'imaginary': False, 'zero': s == 0,
                          'positive': s > 0, 'negative': s < 0, 'rational': a.is_rational})
            if not a.is_rational:
                facts['irrational'] = True
            return facts
        facts.update({'real': False, 'extended_real': False, 'zero': False, 'rational': False,
                      'integer': False, 'imaginary': _real_part_is_zero(a)})
        return facts

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        return _complex_value(self.algebraic(), prec)

    def _sympystr(self, printer: NodePrinter) -> str:
        return "CRootOf(%s, %d)" % (printer.doprint(self._polynomial), self._index)


#: SymPy's name of the class
CRootOf = ComplexRootOf


def _address_of(coefficients: Sequence[Fraction], x: Symbol, index: int) -> _Address:
    if len(coefficients) > 2:
        integers = tuple(_integer_coefficients(coefficients))
        content = 0
        for c in integers:
            content = _gcd(content, c)
        normalised = tuple(c//content for c in integers)
        if normalised[0] < 0:
            normalised = tuple(-c for c in normalised)
        cached = _ROOTS.get(normalised)
        if cached is not None:
            # already known to be irreducible
            return normalised, index
    return _addresses(coefficients, x, False)[index]


def _real_part_is_zero(z: Algebraic) -> bool:
    """Whether the number which is not real is on the imaginary axis."""
    return _compare_real_part(z, _ZERO, Fraction(1)) == 0


def _complex_value(a: Algebraic, prec: int) -> mpmath.mpc:
    """The value of an algebraic number at ``prec`` bits: the middle of its
    isolating region refined until each part has a relative width below
    ``2**-prec`` (a real part which is exactly zero is zero)."""
    eps = Fraction(1, 2**(prec + 4))
    floor = Fraction(1, 2**(4*prec + 64))
    imaginary: Optional[bool] = False if a.is_real else None
    while True:
        re_min, re_max, im_min, im_max = a.enclosure(eps)
        re_mid, im_mid = (re_min + re_max)/2, (im_min + im_max)/2
        if imaginary is None and re_min <= 0 <= re_max:
            imaginary = _real_part_is_zero(a)
        if imaginary:
            re_mid, re_min, re_max = _ZERO, _ZERO, _ZERO
        real_done = re_mid == 0 or re_max - re_min <= abs(re_mid)*Fraction(1, 2**(prec + 2))
        imaginary_done = im_mid == 0 or im_max - im_min <= abs(im_mid)*Fraction(1, 2**(prec + 2))
        if (real_done and imaginary_done) or eps < floor:
            break
        eps = eps/4
        smallest = min((abs(m) for m in (re_mid, im_mid) if m != 0), default=Fraction(1))
        if smallest < 1:
            eps = min(eps, smallest*Fraction(1, 2**(prec + 4)))
    with mpmath.workprec(prec):
        return mpmath.mpc(mpmath.mpf(re_mid.numerator)/re_mid.denominator,
                          mpmath.mpf(im_mid.numerator)/im_mid.denominator)


def algebraic_number(e: Expr) -> Optional[Algebraic]:
    """The algebraic number of a rational number, of ``I``, of a
    :class:`ComplexRootOf` or of an
    :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` with rational
    coefficients; ``None`` for any other expression.

    >>> from sympy_extras.core import Symbol, I
    >>> from sympy_extras.polys.complexrootof import ComplexRootOf, algebraic_number
    >>> x = Symbol('x')
    >>> algebraic_number(ComplexRootOf(x**3 - 2, 2)), algebraic_number(I)
    (Algebraic([1, 0, 0, -2], -0.6299605249 + 1.091123636*I), Algebraic([1, 0, 1], 0 + 1*I))
    """
    if isinstance(e, Rational):
        return Algebraic.from_rational(e.as_fraction())
    if isinstance(e, ImaginaryUnit):
        return Algebraic.imaginary_unit()
    if isinstance(e, ComplexRootOf):
        return e.algebraic()
    from sympy_extras.polys.cad.indexedroot import algebraic_of
    return algebraic_of(e)


def real_roots_indexed(e: Expr) -> Expr:
    """``e`` with its real :class:`ComplexRootOf` nodes written as the
    canonical :class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot` of
    the same number (the real algebraic numbers of the CAD), in the
    variable of the node.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.complexrootof import ComplexRootOf, real_roots_indexed
    >>> x = Symbol('x')
    >>> real_roots_indexed(ComplexRootOf(x**3 - x - 1, 0) + ComplexRootOf(x**3 - x - 1, 1))
    IndexedRoot(x**3 - x - 1, x, 0) + CRootOf(x**3 - x - 1, 1)
    """
    from sympy_extras.core.basic import preorder_traversal
    from sympy_extras.polys.cad.indexedroot import real_root
    rule: dict[Basic, Basic] = {}
    for node in preorder_traversal(e):
        if isinstance(node, ComplexRootOf) and node not in rule:
            a = node.algebraic()
            if a.is_real:
                rule[node] = real_root(a, node.variable.name)
    if not rule:
        return e
    replaced = e._xreplace_evaluated(rule)
    if not isinstance(replaced, Expr):
        raise TypeError("expected an expression, got %s" % (replaced,))
    return replaced
