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

1. a polynomial with rational coefficients is handled by SymPy's real
   root isolation (``real_roots``, ``CRootOf``), and an equation
   ``solveset`` solves exactly keeps its exact roots;
2. otherwise the roots of `f'` on the interval are isolated first
   (recursively); between two consecutive critical points `f` is strictly
   monotone (Rolle's theorem), so it has one root there exactly when its
   values (or limits) at the ends have opposite signs, and none otherwise;
   the root is bracketed by rational points of the right signs and refined
   by bisection. When the zeros of `f'` cannot be listed but its sign is
   constant on the interval (with isolated zeros at most), `f` is
   monotone on the whole interval;
3. the signs are certified: exact evaluation at rational points when SymPy
   knows the sign, rigorous interval arithmetic (mpmath) for the values at
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

References
==========

.. [Strzebonski] A. Strzeboński, Real root isolation for exp-log
   functions, ISSAC 2008.
.. [Collins] G. E. Collins, R. Loos, Real zeros of polynomials, in
   Computer Algebra: Symbolic and Algebraic Computation, Springer (1982).
"""
from __future__ import annotations

from typing import Optional, Union

import mpmath

from sympy.calculus.util import continuous_domain
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Float, Rational, Integer
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.printing.printer import Printer
from sympy.sets.sets import Set, Interval, FiniteSet, EmptySet, Complement, Intersection
from sympy.solvers.solveset import solveset

from sympy_extras._timeout import attempt
from sympy_extras._typing import Sign, as_expr, as_set, free_symbols
from sympy_extras.settings import settings
from sympy_extras.assumptions.analysis import (certified_sign, certified_compare, _pieces, _sorted_constants,
    _between, _endpoint_value, _has_discrete_zeros, sign_on)
from sympy_extras.assumptions.intervals import evaluate, interval_sign

__all__ = ['TranscendentalRoot', 'isolate_real_roots', 'real_roots_of', 'point_between', 'compare_points']

#: an exact real number, or an isolated root
Point = Expr


class TranscendentalRoot(Expr):
    """A real root of ``f(x)`` isolated in the open interval ``(a, b)``
    with rational endpoints: ``f`` has opposite signs at ``a`` and ``b``
    and exactly one zero in between (as established by
    :func:`isolate_real_roots`; the constructor checks the signs only).

    Examples
    ========

    >>> from sympy import exp, Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.isolation import TranscendentalRoot
    >>> r = TranscendentalRoot(exp(x) - x - 2, x, 1, 2)
    >>> r
    TranscendentalRoot(-x + exp(x) - 2, x, 1, 2)
    >>> r.evalf(20)
    1.1461932206205825852
    >>> r.refine(4).interval
    Interval(9/8, 19/16)
    >>> r > 1, r.is_real
    (True, True)
    """

    is_real = True
    is_finite = True
    is_number = True
    is_comparable = True

    def __new__(cls, f: Expr, x: Symbol, a: Union[Rational, int], b: Union[Rational, int]) -> TranscendentalRoot:
        f_ = as_expr(f)
        if not isinstance(x, Symbol):
            raise TypeError("the variable must be a symbol, got %s" % (x,))
        if free_symbols(f_) - {x}:
            raise ValueError("%s has parameters besides %s" % (f_, x))
        a_, b_ = Rational(a), Rational(b)
        if not a_ < b_:
            raise ValueError("an interval with a < b is expected, got (%s, %s)" % (a_, b_))
        left, right = _sign_at(f_, x, a_), _sign_at(f_, x, b_)
        if left is None or right is None or left*right >= 0:
            raise ValueError("%s does not change sign between %s and %s" % (f_, a_, b_))
        result = Basic.__new__(cls, f_, x, a_, b_)
        assert isinstance(result, TranscendentalRoot)
        return result

    @property
    def function(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def variable(self) -> Symbol:
        v = self.args[1]
        assert isinstance(v, Symbol)
        return v

    @property
    def lower(self) -> Rational:
        v = self.args[2]
        assert isinstance(v, Rational)
        return v

    @property
    def upper(self) -> Rational:
        v = self.args[3]
        assert isinstance(v, Rational)
        return v

    @property
    def interval(self) -> Interval:
        return Interval(self.lower, self.upper)

    @property
    def free_symbols(self) -> set[Basic]:
        return set()

    def _eval_subs(self, old: Basic, new: Basic) -> Expr:
        return self

    def _eval_is_positive(self) -> Optional[bool]:
        if self.lower >= 0:
            return True
        if self.upper <= 0:
            return False
        return None

    def _eval_is_negative(self) -> Optional[bool]:
        if self.upper <= 0:
            return True
        if self.lower >= 0:
            return False
        return None

    def _eval_is_zero(self) -> bool:
        return False

    def bisect(self) -> tuple[Rational, Rational]:
        """The half of the isolating interval containing the root."""
        return _bisect_once(self.function, self.variable, self.lower, self.upper)

    def refine(self, steps: int = 1) -> TranscendentalRoot:
        """The root with its interval bisected ``steps`` times."""
        lo, hi = self.lower, self.upper
        for _ in range(steps):
            lo, hi = _bisect_once(self.function, self.variable, lo, hi)
            if lo == hi:
                break
        if lo == hi:
            # the root was hit exactly: keep a tiny interval around it
            return TranscendentalRoot(self.function, self.variable, lo - Rational(1, 2)**64, hi + Rational(1, 2)**64)
        return TranscendentalRoot(self.function, self.variable, lo, hi)

    def _eval_evalf(self, prec: int) -> Float:
        lo, hi = self.lower, self.upper
        width = Rational(1, 2)**(prec + 4)
        with _precision(prec + 30):
            while hi - lo > width:
                lo, hi = _bisect_once(self.function, self.variable, lo, hi)
                if lo == hi:
                    break
        return Float((lo + hi)/2, precision=prec)

    def _sympystr(self, printer: Printer) -> str:
        return "TranscendentalRoot(%s, %s, %s, %s)" % tuple(printer._print(a) for a in self.args)

    _sympyrepr = _sympystr

    def _latex(self, printer: Printer) -> str:
        return r"\operatorname{Root}\left(%s, %s \in \left(%s, %s\right)\right)" % tuple(
            printer._print(a) for a in self.args)


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


def _sign_at(f: Expr, x: Symbol, point: Rational) -> Optional[Sign]:
    """The sign of ``f`` at a rational point: by interval arithmetic
    (rigorous), or by SymPy's exact and certified numerical evaluation."""
    enclosure = evaluate(f, {x: (point, point)})
    if enclosure is not None:
        s = interval_sign(enclosure)
        if s is not None:
            return s
    return certified_sign(as_expr(f.subs(x, point)))


def _bisect_once(f: Expr, x: Symbol, lo: Rational, hi: Rational) -> tuple[Rational, Rational]:
    """Halve a bracketing interval; ``(m, m)`` when the midpoint is a
    root. The signs at the ends are those of the bracket, so the sign at
    the midpoint decides the half (when it cannot be certified, a point
    slightly off the middle is tried)."""
    left = _sign_at(f, x, lo)
    if left is None:
        raise ValueError("the sign of %s at %s cannot be certified" % (f, lo))
    for shift in (Rational(1, 2), Rational(7, 16), Rational(9, 16), Rational(3, 8), Rational(5, 8)):
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


def _sign_near(f: Expr, x: Symbol, point: Point, direction: str) -> tuple[Optional[Sign], Point]:
    """The sign of ``f`` at a point of a monotone piece's end, approached
    from ``direction`` ('+' from the right, '-' from the left): the limit
    at an infinite point, the value at an exact one, and the sign of the
    enclosure over the isolating interval of a root object (refined as
    needed; the refined object is returned)."""
    if point in (S.Infinity, S.NegativeInfinity):
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
                exact = _exact_zero_inside(f, x, current.lower, current.upper)
                if exact is not None:
                    return 0, exact
            current = current.refine()
        return None, current
    value = as_expr(f.subs(x, point))
    if value.is_finite is not False:
        s = certified_sign(value)
        if s is not None:
            return s, point
    # a singular end of an open interval: the limit
    limit_value = _endpoint_value(f, x, point, direction)
    return (None if limit_value is None else certified_sign(limit_value)), point


def _exact_zero_inside(f: Expr, x: Symbol, lo: Rational, hi: Rational) -> Optional[Rational]:
    """A dyadic rational with a small denominator in ``[lo, hi]`` at which
    ``f`` vanishes exactly, if any."""
    for k in range(0, 10):
        scale = Integer(2)**k
        first, last = int(mpmath.ceil(lo*scale)), int(mpmath.floor(hi*scale))
        if last - first > 8:
            return None
        for j in range(first, last + 1):
            point = Rational(j, scale)
            if _sign_at(f, x, point) == 0:
                return point
    return None


def _approach(f: Expr, x: Symbol, point: Point, other: Point, wanted: Sign, from_right: bool) -> Optional[Rational]:
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


def _candidate(point: Point, other: Point, k: int, from_right: bool) -> Optional[Rational]:
    """The ``k``-th rational point of a sequence approaching ``point`` on
    the side of ``other``."""
    if point in (S.Infinity, S.NegativeInfinity):
        # a sequence going to infinity, starting near the other end
        if other in (S.Infinity, S.NegativeInfinity):
            start = Integer(0)
        elif isinstance(other, TranscendentalRoot):
            start = other.lower if from_right else other.upper
        else:
            start = _rational_near(other, not from_right)
        step = Integer(2)**k
        return Rational(start - step) if point is S.NegativeInfinity else Rational(start + step)
    if isinstance(point, TranscendentalRoot):
        refined = point.refine(k)
        return refined.upper if from_right else refined.lower
    if isinstance(point, Rational):
        base = point
    else:
        base = _rational_near(point, from_right)
    if other in (S.Infinity, S.NegativeInfinity):
        gap: Rational = Integer(1)
    else:
        distance = _rational_near(as_expr(other - point), False) if not isinstance(other, TranscendentalRoot) else \
            abs((other.lower if from_right else other.upper) - base)
        gap = Rational(min(Integer(1), abs(distance)))
    offset = gap*Rational(1, 2)**(k + 1)
    candidate = base + offset if from_right else base - offset
    inside = certified_compare(candidate, as_expr(point)) == (1 if from_right else -1)
    return Rational(candidate) if inside else None


def _rational_near(value: Expr, above: bool) -> Rational:
    """A rational number at 30 digits from an exact constant, rounded
    towards the side asked for."""
    approximation = Rational(value.evalf(settings.precision))
    tolerance = Rational(1, 10)**(settings.precision - 2)
    return approximation + tolerance if above else approximation - tolerance


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
        if hi - lo <= Rational(1, 8):
            break
        lo, hi = _bisect_once(f, x, lo, hi)
        if lo == hi:
            return lo
    return TranscendentalRoot(f, x, lo, hi)


def _dyadic_bracket(f: Expr, x: Symbol, lo: Rational, hi: Rational, p: Point, q: Point, sp: Sign, sq: Sign
                    ) -> tuple[Rational, Rational]:
    """The bracket widened to the coarsest dyadic grid on which it stays
    inside the monotone piece ``(p, q)`` with the same signs at the ends,
    so that the isolating interval has small denominators."""
    for k in range(0, 400):
        scale = Integer(2)**k
        L = Rational(int(mpmath.floor(lo*scale)), scale)
        H = Rational(int(mpmath.ceil(hi*scale)), scale)
        if L >= H:
            continue
        if not (_strictly_inside(L, p, q) and _strictly_inside(H, p, q)):
            continue
        if _sign_at(f, x, L) == sp and _sign_at(f, x, H) == sq:
            return L, H
    return lo, hi


def _polynomial_roots(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[list[Expr]]:
    try:
        poly = Poly(f, x)
    except PolynomialError:
        return None
    if not all(c.is_rational for c in poly.coeffs()):
        return None
    if poly.degree() <= 0:
        return [] if not poly.is_zero else None
    found: list[Expr] = []
    for root in poly.real_roots():
        r = as_expr(root)
        if _strictly_inside(r, a, b):
            found.append(r)
    return found


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
                u = u.refine()
            if isinstance(v, TranscendentalRoot):
                v = v.refine()
        return None
    return certified_compare(u, v)


def _bounds(p: Point) -> tuple[Expr, Expr]:
    if isinstance(p, TranscendentalRoot):
        return p.lower, p.upper
    return p, p


def _exact_zeros(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[list[Expr]]:
    """The zeros in ``(a, b)`` when ``solveset`` lists them."""
    outer_a, outer_b = _bounds(a)[0], _bounds(b)[1]
    zeros = attempt(lambda: as_set(solveset(f, x, Interval.open(outer_a, outer_b))), settings.timeout)
    if isinstance(zeros, EmptySet):
        return []
    if not isinstance(zeros, FiniteSet):
        return None
    values = [as_expr(z) for z in zeros.args]
    if not all(z.is_real for z in values) or any(free_symbols(z) for z in values):
        return None
    inside = [z for z in values if _strictly_inside(z, a, b)]
    return _sorted_constants(inside)


def _monotone_on(f: Expr, x: Symbol, a: Point, b: Point) -> Optional[bool]:
    """Whether ``f`` is strictly monotone on ``(a, b)`` because its
    derivative has a constant sign with isolated zeros at most."""
    if isinstance(a, TranscendentalRoot) or isinstance(b, TranscendentalRoot):
        return None
    piece = Interval.open(a, b)
    derivative = as_expr(f.diff(x))
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
    critical = _roots_between(as_expr(f.diff(x)), x, a, b, depth - 1)
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
            result.append(as_expr(points[i]))
        if sp*sq < 0:
            root = _isolate_between(f, x, points[i], points[i + 1], sp, sq)
            if root is None:
                return None
            result.append(root)
    return result


def isolate_real_roots(f: Expr, x: Symbol, domain: Set = S.Reals, depth: int = 4) -> Optional[list[Expr]]:
    """The real roots of ``f`` on ``domain`` (an interval or a union of
    intervals), each exact or a :class:`TranscendentalRoot`, in increasing
    order; ``None`` when they cannot be isolated (see the module
    documentation).

    Examples
    ========

    >>> from sympy import exp, cos, sin, log, Interval, S
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.isolation import isolate_real_roots
    >>> isolate_real_roots(exp(x) - x - 2, x)
    [TranscendentalRoot(-x + exp(x) - 2, x, -61/32, -29/16), TranscendentalRoot(-x + exp(x) - 2, x, 17/16, 37/32)]
    >>> isolate_real_roots(x**2 + cos(x) - 1, x)
    [0]
    >>> isolate_real_roots(x - cos(x), x)
    [TranscendentalRoot(x - cos(x), x, 5/8, 3/4)]
    >>> isolate_real_roots(sin(x) - x/2, x, Interval(0, S.Infinity))
    [0, TranscendentalRoot(-x/2 + sin(x), x, 237/128, 63/32)]
    >>> isolate_real_roots(log(x) - 1, x, Interval.open(0, S.Infinity))
    [E]
    """
    f = as_expr(f)
    if free_symbols(f) - {x}:
        raise ValueError("%s has parameters besides %s" % (f, x))
    pieces = _pieces(domain)
    if pieces is None:
        return None
    continuity = attempt(lambda: as_set(continuous_domain(f, x, S.Reals)), settings.timeout)
    if continuity is None:
        return None
    result: list[Expr] = []
    for piece in pieces:
        if not _continuous_on(continuity, piece):
            return None
        a, b = as_expr(piece.start), as_expr(piece.end)
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


def _continuous_on(continuity: Set, piece: Interval) -> bool:
    """Whether the piece lies in the domain of continuity (a complement
    of isolated singularities is checked by intersecting them with the
    piece, since ``is_subset`` cannot decide it)."""
    if piece.is_subset(continuity) is True:
        return True
    if isinstance(continuity, Complement) and continuity.args[0] == S.Reals:
        singular = attempt(lambda: as_set(Intersection(as_set(continuity.args[1]), piece)), settings.timeout)
        return isinstance(singular, EmptySet)
    return False


def _roots_with_bounded_tails(f: Expr, x: Symbol, a: Point, b: Point, depth: int) -> Optional[list[Expr]]:
    """The roots on an infinite interval when ``f`` has a constant
    nonzero sign beyond some bound (found by interval arithmetic on the
    tails), so that only a bounded interval remains to be searched."""
    lo, hi = a, b
    for exponent in range(0, 8):
        R = Integer(2)**exponent
        if lo is S.NegativeInfinity and _tail_sign(f, x, Interval.open(S.NegativeInfinity, -R)) is not None:
            lo = -R
        if hi is S.Infinity and _tail_sign(f, x, Interval.open(R, S.Infinity)) is not None:
            hi = R
        if lo is not S.NegativeInfinity and hi is not S.Infinity:
            break
    if lo is S.NegativeInfinity or hi is S.Infinity:
        return None
    if not _compare(lo, hi) == -1:
        return None
    return _roots_between(f, x, lo, hi, depth)


def _tail_sign(f: Expr, x: Symbol, tail: Interval) -> Optional[Sign]:
    signs = sign_on(f, x, tail, depth=0)
    if signs is None or len(signs) != 1 or 0 in signs:
        return None
    [s] = signs
    return s


def compare_points(u: Point, v: Point) -> Optional[Sign]:
    """The sign of ``u - v`` for exact real constants, infinities and
    :class:`TranscendentalRoot` objects."""
    return _compare(u, v)


def point_between(p: Point, q: Point) -> Optional[Rational]:
    """A rational number strictly between ``p < q`` (exact constants,
    infinities or root objects, whose intervals are refined until they
    separate)."""
    for _ in range(80):
        lo, hi = _bounds(p)[1], _bounds(q)[0]
        if certified_compare(lo, hi) == -1:
            break
        if isinstance(p, TranscendentalRoot):
            p = p.refine()
        if isinstance(q, TranscendentalRoot):
            q = q.refine()
    else:
        return None
    if isinstance(p, TranscendentalRoot) and isinstance(q, TranscendentalRoot):
        return Rational((lo + hi)/2)
    middle = _between(lo, hi)
    return Rational(middle) if middle is not None else None


def real_roots_of(f: Expr, x: Symbol, domain: Set = S.Reals, depth: int = 4) -> Optional[FiniteSet]:
    """The roots of :func:`isolate_real_roots` as a set (``None`` when
    undecided)."""
    found = isolate_real_roots(f, x, domain, depth)
    if found is None:
        return None
    return FiniteSet(*found)
