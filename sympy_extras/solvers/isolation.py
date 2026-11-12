"""Real roots of transcendental functions of one variable: exact
isolation and root objects.

SymPy's ``solveset`` returns a ``ConditionSet`` for an equation such as
`e^x = x + 2` or `x^2 + \\cos x = 1`, which has no solution in closed
form. Here the real roots of such a function `f` are *isolated*: each
root is enclosed in an interval with rational endpoints containing no
other root, the counterpart of Mathematica's transcendental ``Root``
objects, and the equation gets a finite solution set. The
isolation is exact in the sense that the number of roots and their
isolating intervals are proved:

1. a polynomial with rational coefficients has its real roots isolated
   exactly (in radicals for the factors of degree at most two and the
   binomials, as SymPy's ``real_roots`` writes them, SymPy's ``CRootOf``
   otherwise), and an equation whose zeros are listed exactly keeps its
   exact roots;
2. otherwise the roots of `f'` on the interval are isolated first
   (recursively); between two consecutive critical points `f` is strictly
   monotone (Rolle's theorem), so it has one root there exactly when its
   values (or limits) at the ends have opposite signs, and none otherwise;
   the root is bracketed by rational points of the right signs and refined
   by bisection. When the zeros of `f'` cannot be listed but its sign is
   constant on the interval (with isolated zeros at most), `f` is
   monotone on the whole interval;
3. the signs are certified: exact evaluation at rational points when the
   sign is known, rigorous interval arithmetic (mpmath) for the values at
   isolated points and at the roots of `f'`, and limits at infinite ends.

The recursion stops at a fixed depth (exp-log functions with many nested
critical points are left undecided, ``None``), and roots of even
multiplicity (where `f` touches zero) cannot be certified by signs and
give ``None`` as well. The set of functions is that of the interval
arithmetic of :mod:`sympy_extras.assumptions.intervals` (exponentials,
logarithms, powers, trigonometric and hyperbolic functions).

A :class:`TranscendentalRoot` is a real number: it evaluates numerically
to any precision by bisection with rigorous signs, compares with other
numbers, and prints with its function and isolating interval.

The module works on objects of the core (:mod:`sympy_extras.core`,
:mod:`sympy_extras.sets`) and imports no SymPy:
:mod:`sympy_extras.solvers.isolation_sympy` is its face on SymPy objects.
The signs, the zeros of the rational functions and the limits are native
(:mod:`sympy_extras.assumptions.analysis`,
:mod:`sympy_extras.assumptions.intervals`), and so are the continuity of
a function which is not rational and its zeros listed exactly (the
native continuity and transcendental solver of
:mod:`sympy_extras.assumptions.analysis`). One step goes through SymPy at
run time (the face holds it, :func:`~sympy_extras.solvers.isolation_sympy.real_roots_of_core`):
the real roots of a polynomial which are not in radicals (SymPy's
``real_roots``, whose ``CRootOf`` is not ported). The real roots in radicals
follow SymPy's ``ComplexRootOf._get_roots_qq``
(``sympy/polys/rootoftools.py``), ported.

References
==========

.. [Strzebonski] A. Strzeboński, Real root isolation for exp-log
   functions, ISSAC 2008.
.. [Collins] G. E. Collins, R. Loos, Real zeros of polynomials, in
   Computer Algebra: Symbolic and Algebraic Computation, Springer (1982).
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/rootoftools.py (ComplexRootOf._get_roots_qq,
# _real_roots, _postprocess_root) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed, on the native Poly; only the factors whose roots are
# trivial (linear, quadratic, binomial), the others through the face.
# ---------------------------------------------------------------------
from __future__ import annotations

from fractions import Fraction
from typing import Final, Optional, Union

import mpmath

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Float, Infinity, Integer, NegativeInfinity, Rational, number_from_fraction
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import preprocess_roots
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rootisolation import dup_count_real_roots
from sympy_extras.polys.roots import _trivial_roots
from sympy_extras.sets.sets import Interval, Reals, Set, finite_set, interval
from sympy_extras.settings import settings
from sympy_extras.assumptions.analysis import (_between, _continuous_on, _endpoint_value, _finite_zeros,
    _has_discrete_zeros, _pieces, _sorted_constants, certified_compare, certified_sign, sign_on)
from sympy_extras.assumptions.intervals import Sign, evaluate, interval_sign

__all__ = ['TranscendentalRoot', 'isolate_real_roots', 'real_roots_of', 'point_between', 'compare_points', 'Point']

#: an exact real number, or an isolated root
Point = Expr
#: a rational end of an interval, as accepted by :class:`TranscendentalRoot`
RationalLike = Union[Rational, int, Fraction]


def _fraction(value: RationalLike) -> Fraction:
    if isinstance(value, Rational):
        return Fraction(value.p, value.q)
    return Fraction(value)


def _number(value: Fraction) -> Rational:
    return number_from_fraction(value)


class TranscendentalRoot(Expr):
    """A real root of ``f(x)`` isolated in the open interval ``(a, b)``
    with rational endpoints: ``f`` has opposite signs at ``a`` and ``b``
    and exactly one zero in between (as established by
    :func:`isolate_real_roots`; the constructor checks the signs only).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.isolation import TranscendentalRoot
    >>> x = Symbol('x')
    >>> r = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    >>> r
    TranscendentalRoot(-x + exp(x) - 2, x, 1, 2)
    >>> r.evalf(20)
    1.1461932206205825852
    >>> r.refined(4).interval
    Interval(9/8, 19/16)
    >>> r > 1, r.is_real
    (True, True)
    """

    __slots__ = ('_function', '_variable', '_lower', '_upper')

    def __init__(self, f: Expr, x: Symbol, a: RationalLike, b: RationalLike) -> None:
        if not isinstance(x, Symbol):
            raise TypeError("the variable must be a symbol, got %s" % (x,))
        if not isinstance(f, Expr):
            raise TypeError("an expression is expected, got %s" % (f,))
        if f.free_symbols - {x}:
            raise ValueError("%s has parameters besides %s" % (f, x))
        lower, upper = _fraction(a), _fraction(b)
        if not lower < upper:
            raise ValueError("an interval with a < b is expected, got (%s, %s)" % (lower, upper))
        left, right = _sign_at(f, x, lower), _sign_at(f, x, upper)
        if left is None or right is None or left*right >= 0:
            raise ValueError("%s does not change sign between %s and %s" % (f, lower, upper))
        self._function: Final[Expr] = f
        self._variable: Final[Symbol] = x
        self._lower: Final[Fraction] = lower
        self._upper: Final[Fraction] = upper
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (self._function, self._variable, _number(self._lower), _number(self._upper))

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def variable(self) -> Symbol:
        return self._variable

    @property
    def lower(self) -> Rational:
        return _number(self._lower)

    @property
    def upper(self) -> Rational:
        return _number(self._upper)

    @property
    def lower_fraction(self) -> Fraction:
        """The lower end as a fraction."""
        return self._lower

    @property
    def upper_fraction(self) -> Fraction:
        """The upper end as a fraction."""
        return self._upper

    @property
    def interval(self) -> Set:
        return interval(self.lower, self.upper)

    @property
    def free_symbols(self) -> set[Symbol]:
        return set()

    @property
    def is_number(self) -> bool:
        return True

    def _is_number_node(self) -> Optional[bool]:
        return True

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        f, x, a, b = args
        if not (isinstance(f, Expr) and isinstance(x, Symbol) and isinstance(a, Rational)
                and isinstance(b, Rational)):
            raise TypeError("invalid arguments of a TranscendentalRoot: %s" % (args,))
        return TranscendentalRoot(f, x, a, b)

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return self._rebuild(args)

    def _subs(self, old: Basic, new: Basic) -> Basic:
        # the variable is bound: a substitution leaves the root alone
        return new if self == old else self

    def _eval_facts(self) -> dict[str, bool]:
        facts = {'commutative': True, 'real': True, 'finite': True, 'zero': False}
        if self._lower >= 0:
            facts['positive'] = True
        elif self._upper <= 0:
            facts['positive'] = False
        if self._upper <= 0:
            facts['negative'] = True
        elif self._lower >= 0:
            facts['negative'] = False
        return facts

    def bisect(self) -> tuple[Rational, Rational]:
        """The half of the isolating interval containing the root."""
        lo, hi = _bisect_once(self._function, self._variable, self._lower, self._upper)
        return _number(lo), _number(hi)

    def refined(self, steps: int = 1) -> TranscendentalRoot:
        """The root with its interval bisected ``steps`` times."""
        lo, hi = self._lower, self._upper
        for _ in range(steps):
            lo, hi = _bisect_once(self._function, self._variable, lo, hi)
            if lo == hi:
                break
        if lo == hi:
            # the root was hit exactly: keep a tiny interval around it
            return TranscendentalRoot(self._function, self._variable, lo - Fraction(1, 2**64), hi + Fraction(1, 2**64))
        return TranscendentalRoot(self._function, self._variable, lo, hi)

    def approximation(self, prec: int) -> Fraction:
        """A rational number within ``2**-(prec + 4)`` of the root, the
        middle of the interval bisected until it is that narrow."""
        lo, hi = self._lower, self._upper
        width = Fraction(1, 2**(prec + 4))
        with _precision(prec + 30):
            while hi - lo > width:
                lo, hi = _bisect_once(self._function, self._variable, lo, hi)
                if lo == hi:
                    break
        return (lo + hi)/2

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        middle = self.approximation(prec)
        with mpmath.workprec(prec):
            return mpmath.mpc(mpmath.mpf(middle.numerator)/middle.denominator)

    def _sympystr(self, printer: StrPrinter) -> str:
        return "TranscendentalRoot(%s, %s, %s, %s)" % tuple(printer.doprint(a) for a in self.args)


class _precision:
    """Temporarily raise the precision of the interval arithmetic."""

    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.previous = 0

    def __enter__(self) -> None:
        self.previous = int(mpmath.iv.prec)
        mpmath.iv.prec = max(self.bits, self.previous)

    def __exit__(self, *args: object) -> None:
        mpmath.iv.prec = self.previous


def _sign_at(f: Expr, x: Symbol, point: Fraction) -> Optional[Sign]:
    """The sign of ``f`` at a rational point: by interval arithmetic
    (rigorous), or by exact and certified numerical evaluation."""
    value = _number(point)
    enclosure = evaluate(f, {x: (value, value)})
    if enclosure is not None:
        s = interval_sign(enclosure)
        if s is not None:
            return s
    return certified_sign(as_expr(f.subs(x, value)))


def _bisect_once(f: Expr, x: Symbol, lo: Fraction, hi: Fraction) -> tuple[Fraction, Fraction]:
    """Halve a bracketing interval; ``(m, m)`` when the midpoint is a
    root. The signs at the ends are those of the bracket, so the sign at
    the midpoint decides the half (when it cannot be certified, a point
    slightly off the middle is tried)."""
    left = _sign_at(f, x, lo)
    if left is None:
        raise ValueError("the sign of %s at %s cannot be certified" % (f, lo))
    for shift in (Fraction(1, 2), Fraction(7, 16), Fraction(9, 16), Fraction(3, 8), Fraction(5, 8)):
        m = lo + shift*(hi - lo)
        s = _sign_at(f, x, m)
        if s is None:
            continue
        if s == 0:
            return m, m
        if s == left:
            return m, hi
        return lo, m
    raise ValueError("the sign of %s near the middle of (%s, %s) cannot be certified" % (f, lo, hi))


def _floor(value: Fraction) -> int:
    """The floor of ``value`` rounded first to the working precision of
    mpmath (53 bits by default): the grids of the isolating intervals are
    chosen on these values, as they always were (the brackets are checked
    by their signs, so the rounding only chooses among valid ones)."""
    return int(mpmath.floor(_rounded(value)))


def _ceiling(value: Fraction) -> int:
    """The ceiling of ``value`` rounded first (see :func:`_floor`)."""
    return int(mpmath.ceil(_rounded(value)))


def _rounded(value: Fraction) -> mpmath.mpf:
    return mpmath.mpf(mpmath.libmp.from_rational(value.numerator, value.denominator, mpmath.mp.prec,
                                                 mpmath.libmp.round_nearest))


def _infinite(point: Expr) -> bool:
    return isinstance(point, (Infinity, NegativeInfinity))


def _sign_near(f: Expr, x: Symbol, point: Point, direction: str) -> tuple[Optional[Sign], Point]:
    """The sign of ``f`` at a point of a monotone piece's end, approached
    from ``direction`` ('+' from the right, '-' from the left): the limit
    at an infinite point, the value at an exact one, and the sign of the
    enclosure over the isolating interval of a root object (refined as
    needed; the refined object is returned)."""
    if _infinite(point):
        value = _endpoint_value(f, x, point, direction)
        return (None if value is None else certified_sign(value)), point
    if isinstance(point, TranscendentalRoot):
        current = point
        for step in range(80):
            enclosure = evaluate(f, {x: (current.lower, current.upper)})
            if enclosure is None:
                return None, current
            s = interval_sign(enclosure)
            if s is not None:
                return s, current
            if step in (6, 16, 40):
                # the root of f' may be an exact root of f as well (a
                # multiple root): dyadic points of the interval are tried
                exact = _exact_zero_inside(f, x, current.lower_fraction, current.upper_fraction)
                if exact is not None:
                    return 0, _number(exact)
            current = current.refined()
        return None, current
    value = as_expr(f.subs(x, point))
    if value.is_finite is not False:
        s = certified_sign(value)
        if s is not None:
            return s, point
    # a singular end of an open interval: the limit
    limit_value = _endpoint_value(f, x, point, direction)
    return (None if limit_value is None else certified_sign(limit_value)), point


def _exact_zero_inside(f: Expr, x: Symbol, lo: Fraction, hi: Fraction) -> Optional[Fraction]:
    """A dyadic rational with a small denominator in ``[lo, hi]`` at which
    ``f`` vanishes exactly, if any."""
    for k in range(0, 10):
        scale = 2**k
        first, last = _ceiling(lo*scale), _floor(hi*scale)
        if last - first > 8:
            return None
        for j in range(first, last + 1):
            point = Fraction(j, scale)
            if _sign_at(f, x, point) == 0:
                return point
    return None


def _approach(f: Expr, x: Symbol, point: Point, other: Point, wanted: Sign, from_right: bool) -> Optional[Fraction]:
    """A rational point strictly between ``point`` and ``other`` (``other``
    on the right when ``from_right``) close enough to ``point`` for ``f``
    to have the sign ``wanted`` there."""
    for k in range(60):
        candidate = _candidate(point, other, k, from_right)
        if candidate is None:
            return None
        if _sign_at(f, x, candidate) == wanted:
            return candidate
    return None


def _candidate(point: Point, other: Point, k: int, from_right: bool) -> Optional[Fraction]:
    """The ``k``-th rational point of a sequence approaching ``point`` on
    the side of ``other``."""
    if _infinite(point):
        # a sequence going to infinity, starting near the other end
        start: Fraction
        if _infinite(other):
            start = Fraction(0)
        elif isinstance(other, TranscendentalRoot):
            start = other.lower_fraction if from_right else other.upper_fraction
        else:
            start = _rational_near(other, not from_right)
        step = 2**k
        return start - step if isinstance(point, NegativeInfinity) else start + step
    if isinstance(point, TranscendentalRoot):
        refined = point.refined(k)
        return refined.upper_fraction if from_right else refined.lower_fraction
    base = _fraction(point) if isinstance(point, Rational) else _rational_near(point, from_right)
    gap: Fraction
    if _infinite(other):
        gap = Fraction(1)
    else:
        if isinstance(other, TranscendentalRoot):
            distance = abs((other.lower_fraction if from_right else other.upper_fraction) - base)
        else:
            distance = _rational_near(as_expr(other - point), False)
        gap = min(Fraction(1), abs(distance))
    offset = gap*Fraction(1, 2**(k + 1))
    candidate = base + offset if from_right else base - offset
    inside = certified_compare(_number(candidate), point) == (1 if from_right else -1)
    return candidate if inside else None


def _rational_near(value: Expr, above: bool) -> Fraction:
    """A rational number at 30 digits from an exact constant, rounded
    towards the side asked for."""
    approximation = value.evalf(settings.precision)
    if not isinstance(approximation, (Float, Rational)):
        raise TypeError("%s is not a real number" % (value,))
    tolerance = Fraction(1, 10**(settings.precision - 2))
    exact = approximation.as_fraction()
    return exact + tolerance if above else exact - tolerance


def _isolate_between(f: Expr, x: Symbol, p: Point, q: Point, sp: Sign, sq: Sign) -> Optional[Expr]:
    """The unique root of the monotone ``f`` in ``(p, q)``, where ``f``
    has the signs ``sp`` and ``sq`` at the ends: bracketed by rational
    points and refined by bisection."""
    lo = _approach(f, x, p, q, sp, True)
    hi = _approach(f, x, q, p, sq, False)
    if lo is None or hi is None or not lo < hi:
        return None
    lo, hi = _dyadic_bracket(f, x, lo, hi, p, q, sp, sq)
    for _ in range(64):
        if hi - lo <= Fraction(1, 8):
            break
        lo, hi = _bisect_once(f, x, lo, hi)
        if lo == hi:
            return _number(lo)
    return TranscendentalRoot(f, x, lo, hi)


def _dyadic_bracket(f: Expr, x: Symbol, lo: Fraction, hi: Fraction, p: Point, q: Point, sp: Sign, sq: Sign
                    ) -> tuple[Fraction, Fraction]:
    """The bracket widened to the coarsest dyadic grid on which it stays
    inside the monotone piece ``(p, q)`` with the same signs at the ends,
    so that the isolating interval has small denominators."""
    for k in range(0, 400):
        scale = 2**k
        L = Fraction(_floor(lo*scale), scale)
        H = Fraction(_ceiling(hi*scale), scale)
        if L >= H:
            continue
        if not (_strictly_inside(_number(L), p, q) and _strictly_inside(_number(H), p, q)):
            continue
        if _sign_at(f, x, L) == sp and _sign_at(f, x, H) == sq:
            return L, H
    return lo, hi


# PORTED FROM SYMPY: sympy/polys/rootoftools.py:ComplexRootOf._get_roots_qq
def _real_roots(poly: Poly) -> Optional[list[Expr]]:
    """The real roots of a polynomial over the rationals with their
    multiplicities, in increasing order, as SymPy's ``Poly.real_roots``
    writes them when it needs no ``CRootOf`` (every irreducible factor
    linear, quadratic or a binomial): the preprocessing, the real roots of
    the factors in increasing order with their indices, the trivial roots
    of each factor; ``None`` otherwise."""
    coefficient, reduced = preprocess_roots(poly)
    _, factors = reduced.factor_list()
    found: list[tuple[Expr, int]] = []
    for factor, multiplicity in factors:
        if factor.degree() < 1:
            continue
        trivial = _trivial_roots(factor)
        if trivial is None:
            return None
        dense = [_fraction(_rational(c)) for c in factor.all_coeffs()]
        for index in range(dup_count_real_roots(dense)):
            found.append((trivial[index], multiplicity))
    ordered: list[tuple[Expr, int]] = []
    for root, multiplicity in found:
        position = len(ordered)
        for i, (other, _) in enumerate(ordered):
            c = certified_compare(root, other)
            if c is None:
                return None
            if c < 0:
                position = i
                break
        ordered.insert(position, (root, multiplicity))
    return [mul(coefficient, root) for root, multiplicity in ordered for _ in range(multiplicity)]


def _rational(c: Expr) -> Rational:
    if not isinstance(c, Rational):
        raise TypeError("a rational coefficient is expected, got %s" % (c,))
    return c


def _polynomial_roots(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[list[Expr]]:
    try:
        poly = Poly(f, x)
    except BasePolynomialError:
        return None
    if not all(isinstance(c, Rational) for c in poly.coeffs()):
        return None
    if poly.degree() <= 0:
        return [] if not poly.is_zero else None
    roots = _real_roots(poly)
    if roots is None:
        from .isolation_sympy import real_roots_of_core
        roots = real_roots_of_core(f, x)
    return [r for r in roots if _strictly_inside(r, a, b)]


def _strictly_inside(r: Expr, a: Point, b: Point) -> bool:
    return _compare(r, a) == 1 and _compare(b, r) == 1


def _compare(u: Point, v: Point) -> Optional[Sign]:
    """The sign of ``u - v`` for exact constants, infinities and root
    objects (whose isolating intervals are refined until they separate)."""
    if isinstance(u, TranscendentalRoot) or isinstance(v, TranscendentalRoot):
        for _ in range(80):
            lo_u, hi_u = _bounds(u)
            lo_v, hi_v = _bounds(v)
            if certified_compare(lo_u, hi_v) == 1:
                return 1
            if certified_compare(lo_v, hi_u) == 1:
                return -1
            if isinstance(u, TranscendentalRoot):
                u = u.refined()
            if isinstance(v, TranscendentalRoot):
                v = v.refined()
        return None
    return certified_compare(u, v)


def _bounds(p: Point) -> tuple[Expr, Expr]:
    if isinstance(p, TranscendentalRoot):
        return p.lower, p.upper
    return p, p


def _exact_zeros(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[list[Expr]]:
    """The zeros in ``(a, b)`` when they are listed exactly (by the
    native solvers of :mod:`sympy_extras.assumptions.analysis`)."""
    outer = interval(_bounds(a)[0], _bounds(b)[1], True, True)
    zeros = _finite_zeros(f, x, outer)
    if zeros is None:
        return None
    if not all(z.is_real for z in zeros) or any(z.free_symbols for z in zeros):
        return None
    inside = [z for z in zeros if _strictly_inside(z, a, b)]
    return _sorted_constants(inside)


def _monotone_on(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[bool]:
    """Whether ``f`` is strictly monotone on ``(a, b)`` because its
    derivative has a constant sign with isolated zeros at most."""
    if isinstance(a, TranscendentalRoot) or isinstance(b, TranscendentalRoot):
        return None
    piece = interval(a, b, True, True)
    if not isinstance(piece, Interval):
        return None
    derivative = f.diff(x)
    signs = sign_on(derivative, x, piece, depth=0)
    if signs is None or (1 in signs and -1 in signs) or signs == frozenset([0]):
        return None
    if 0 in signs and not _has_discrete_zeros(derivative, x, piece):
        return None
    return True


def _roots_between(f: Expr, x: Symbol, a: Point, b: Point, depth: int) -> Optional[list[Expr]]:
    """The roots of ``f`` in the open interval ``(a, b)``, in increasing
    order, or ``None``."""
    if not f.has(x):
        s = certified_sign(f)
        return [] if s is not None and s != 0 else None
    polynomial = _polynomial_roots(f, x, a, b)
    if polynomial is not None:
        return polynomial
    exact = _exact_zeros(f, x, a, b)
    if exact is not None:
        return exact
    if depth <= 0:
        return None
    critical = _roots_between(f.diff(x), x, a, b, depth - 1)
    if critical is None:
        if _monotone_on(f, x, a, b) is not True:
            return None
        critical = []
    points: list[Point] = [a] + list(critical) + [b]
    result: list[Expr] = []
    for i in range(len(points) - 1):
        sp, refined_p = _sign_near(f, x, points[i], '+')
        points[i] = refined_p
        sq, refined_q = _sign_near(f, x, points[i + 1], '-')
        points[i + 1] = refined_q
        if sp is None or sq is None:
            return None
        if i > 0 and sp == 0:
            # a root at a critical point (of higher multiplicity)
            result.append(points[i])
        if sp*sq < 0:
            root = _isolate_between(f, x, points[i], points[i + 1], sp, sq)
            if root is None:
                return None
            result.append(root)
    return result


def isolate_real_roots(f: Expr, x: Symbol, domain: Set = Reals, depth: int = 4) -> Optional[list[Expr]]:
    """The real roots of ``f`` on ``domain`` (an interval or a union of
    intervals), each exact or a :class:`TranscendentalRoot`, in increasing
    order; ``None`` when they cannot be isolated (see the module
    documentation).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import exp, cos, sin, log
    >>> from sympy_extras.sets import interval
    >>> from sympy_extras.solvers.isolation import isolate_real_roots
    >>> x = Symbol('x')
    >>> isolate_real_roots(exp(x) - x - 2, x)
    [TranscendentalRoot(-x + exp(x) - 2, x, -61/32, -29/16), TranscendentalRoot(-x + exp(x) - 2, x, 17/16, 37/32)]
    >>> isolate_real_roots(x**2 + cos(x) - 1, x)
    [0]
    >>> isolate_real_roots(x - cos(x), x)
    [TranscendentalRoot(x - cos(x), x, 5/8, 3/4)]
    >>> isolate_real_roots(sin(x) - x/2, x, interval(0, oo))
    [0, TranscendentalRoot(-x/2 + sin(x), x, 237/128, 63/32)]
    >>> isolate_real_roots(log(x) - 1, x, interval(0, oo, True, True))
    [E]
    >>> isolate_real_roots(x**4 - 5*x**2 + 6, x)
    [-sqrt(3), -sqrt(2), sqrt(2), sqrt(3)]
    """
    if f.free_symbols - {x}:
        raise ValueError("%s has parameters besides %s" % (f, x))
    pieces = _pieces(domain)
    if pieces is None:
        return None
    result: list[Expr] = []
    for piece in pieces:
        if not _continuous_on(f, x, piece):
            return None
        a, b = piece.start, piece.end
        if not piece.left_open and a.is_finite and certified_sign(as_expr(f.subs(x, a))) == 0:
            result.append(a)
        inside = _roots_between(f, x, a, b, depth)
        if inside is None and not (a.is_finite and b.is_finite):
            inside = _roots_with_bounded_tails(f, x, a, b, depth)
        if inside is None:
            return None
        result.extend(inside)
        if not piece.right_open and b.is_finite and certified_sign(as_expr(f.subs(x, b))) == 0:
            result.append(b)
    return result


def _roots_with_bounded_tails(f: Expr, x: Symbol, a: Point, b: Point, depth: int) -> Optional[list[Expr]]:
    """The roots on an infinite interval when ``f`` has a constant
    nonzero sign beyond some bound (found by interval arithmetic on the
    tails), so that only a bounded interval remains to be searched."""
    lo, hi = a, b
    for exponent in range(0, 8):
        R = Integer(2**exponent)
        if isinstance(lo, NegativeInfinity) and _tail_sign(f, x, interval(NegativeInfinity(), -R, True, True)) \
                is not None:
            lo = -R
        if isinstance(hi, Infinity) and _tail_sign(f, x, interval(R, Infinity(), True, True)) is not None:
            hi = R
        if not isinstance(lo, NegativeInfinity) and not isinstance(hi, Infinity):
            break
    if isinstance(lo, NegativeInfinity) or isinstance(hi, Infinity):
        return None
    if not _compare(lo, hi) == -1:
        return None
    return _roots_between(f, x, lo, hi, depth)


def _tail_sign(f: Expr, x: Symbol, tail: Set) -> Optional[Sign]:
    signs = sign_on(f, x, tail, depth=0)
    if signs is None or len(signs) != 1 or 0 in signs:
        return None
    [s] = signs
    return s


def compare_points(u: Point, v: Point) -> Optional[Sign]:
    """The sign of ``u - v`` for exact real constants, infinities and
    :class:`TranscendentalRoot` objects.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.functions import cos
    >>> from sympy_extras.solvers.isolation import TranscendentalRoot, compare_points
    >>> x = Symbol('x')
    >>> compare_points(TranscendentalRoot(x - cos(x), x, 0, 1), Integer(1))
    -1
    """
    return _compare(u, v)


def point_between(p: Point, q: Point) -> Optional[Rational]:
    """A rational number strictly between ``p < q`` (exact constants,
    infinities or root objects, whose intervals are refined until they
    separate).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, oo
    >>> from sympy_extras.functions import cos, exp
    >>> from sympy_extras.solvers.isolation import TranscendentalRoot, point_between
    >>> x = Symbol('x')
    >>> point_between(TranscendentalRoot(x - cos(x), x, 0, 1), TranscendentalRoot(exp(x) - x - 2, x, 1, 2))
    7/8
    >>> point_between(TranscendentalRoot(x - cos(x), x, 0, 1), oo)
    2
    """
    for _ in range(80):
        lo, hi = _bounds(p)[1], _bounds(q)[0]
        if certified_compare(lo, hi) == -1:
            break
        if isinstance(p, TranscendentalRoot):
            p = p.refined()
        if isinstance(q, TranscendentalRoot):
            q = q.refined()
    else:
        return None
    if isinstance(p, TranscendentalRoot) and isinstance(q, TranscendentalRoot):
        return _number((_fraction(_rational(lo)) + _fraction(_rational(hi)))/2)
    middle = _between(lo, hi)
    return middle if isinstance(middle, Rational) else None


def real_roots_of(f: Expr, x: Symbol, domain: Set = Reals, depth: int = 4) -> Optional[Set]:
    """The roots of :func:`isolate_real_roots` as a set (``None`` when
    undecided)."""
    found = isolate_real_roots(f, x, domain, depth)
    if found is None:
        return None
    return finite_set(*found)
