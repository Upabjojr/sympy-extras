"""The SymPy face of the isolation of real roots of transcendental
functions (:mod:`sympy_extras.solvers.isolation`), and the steps which
stay on SymPy.

:func:`isolate_real_roots`, :func:`real_roots_of`, :func:`point_between`
and :func:`compare_points` take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native functions.
:class:`TranscendentalRoot` is a SymPy expression, the counterpart of the
native root object (``interop`` converts one into the other): its
bisections, refinements and numerical values are those of the native
object.

Two steps of the native module stay written on SymPy and are called from
it at run time (the functions ending in ``_of_core``, which take and
return objects of the core): the real roots of a polynomial with
rational coefficients which are not all in radicals (SymPy's
``real_roots``, with its ``CRootOf``), and whether a function which is
not rational is continuous on an interval (SymPy's ``continuous_domain``).

Examples
========

>>> from sympy import exp, cos, sin, log, Interval, S
>>> from sympy.abc import x
>>> from sympy_extras.solvers.isolation_sympy import isolate_real_roots
>>> isolate_real_roots(exp(x) - x - 2, x)
[TranscendentalRoot(-x + exp(x) - 2, x, -61/32, -29/16), TranscendentalRoot(-x + exp(x) - 2, x, 17/16, 37/32)]
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Union

from sympy.calculus.util import continuous_domain
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Float, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polytools import Poly
from sympy.printing.printer import Printer
from sympy.sets.sets import Complement, EmptySet, FiniteSet, Intersection, Interval, Set

from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import Sign, as_expr, as_set
from sympy_extras.assumptions.intervals_sympy import core_expr, core_set, core_symbol, sympy_expr, sympy_set
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import to_sympy
from sympy_extras.sets.sets import Set as CoreSet
from sympy_extras.settings import settings

from . import isolation as native

__all__ = ['TranscendentalRoot', 'isolate_real_roots', 'real_roots_of', 'point_between', 'compare_points',
    'real_roots_of_core', 'continuous_on_of_core', 'sympy_root', 'native_root']

#: an exact real number, or an isolated root
Point = Expr


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def real_roots_of_core(f: CoreExpr, x: CoreSymbol) -> list[CoreExpr]:
    """The real roots of a polynomial with rational coefficients, with
    their multiplicities, in increasing order (SymPy's ``Poly.real_roots``:
    radicals for the linear, quadratic and binomial factors, ``CRootOf``
    otherwise)."""
    return [core_expr(as_expr(r)) for r in Poly(sympy_expr(f), _sympy_symbol(x)).real_roots()]


def continuous_on_of_core(f: CoreExpr, x: CoreSymbol, piece: CoreSet) -> bool:
    """Whether the piece lies in SymPy's ``continuous_domain`` of ``f``
    over the reals (a complement of isolated singularities is checked by
    intersecting them with the piece, since ``is_subset`` cannot decide
    it); ``False`` when the domain is not found in time."""
    sympy_f, sympy_x, sympy_piece = sympy_expr(f), _sympy_symbol(x), sympy_set(piece)
    continuity = attempt(lambda: as_set(continuous_domain(sympy_f, sympy_x, S.Reals)), settings.timeout)
    if continuity is None:
        return False
    if sympy_piece.is_subset(continuity) is True:
        return True
    if isinstance(continuity, Complement) and continuity.args[0] == S.Reals:
        singular = attempt(lambda: as_set(Intersection(as_set(continuity.args[1]), sympy_piece)), settings.timeout)
        return isinstance(singular, EmptySet)
    return False


# ---------------------------------------------------------------------------
# the root objects

def _sympy_rational(value: Fraction) -> Rational:
    return Rational(value.numerator, value.denominator)


def _fraction(value: Union[Rational, int]) -> Fraction:
    r = Rational(value)
    return Fraction(int(r.p), int(r.q))


class TranscendentalRoot(Expr):
    """A real root of ``f(x)`` isolated in the open interval ``(a, b)``
    with rational endpoints: ``f`` has opposite signs at ``a`` and ``b``
    and exactly one zero in between (as established by
    :func:`isolate_real_roots`; the constructor checks the signs only).
    The SymPy counterpart of :class:`sympy_extras.solvers.isolation.TranscendentalRoot`.

    Examples
    ========

    >>> from sympy import exp, Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.isolation_sympy import TranscendentalRoot
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
        # the native constructor checks the signs
        return sympy_root(native.TranscendentalRoot(core_expr(f_), core_symbol(x), _fraction(a), _fraction(b)))

    @classmethod
    def _unchecked(cls, f: Expr, x: Symbol, a: Rational, b: Rational) -> TranscendentalRoot:
        result = Basic.__new__(cls, f, x, a, b)
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
        lo, hi = native_root(self).bisect()
        return Rational(lo.p, lo.q), Rational(hi.p, hi.q)

    def refine(self, steps: int = 1) -> TranscendentalRoot:
        """The root with its interval bisected ``steps`` times."""
        return sympy_root(native_root(self).refined(steps))

    def _eval_evalf(self, prec: int) -> Float:
        return Float(_sympy_rational(native_root(self).approximation(prec)), precision=prec)

    def _sympystr(self, printer: Printer) -> str:
        return "TranscendentalRoot(%s, %s, %s, %s)" % tuple(printer._print(a) for a in self.args)

    _sympyrepr = _sympystr

    def _latex(self, printer: Printer) -> str:
        return r"\operatorname{Root}\left(%s, %s \in \left(%s, %s\right)\right)" % tuple(
            printer._print(a) for a in self.args)


def sympy_root(r: native.TranscendentalRoot) -> TranscendentalRoot:
    """The SymPy counterpart of a native root object (no check: the native
    object is one)."""
    return TranscendentalRoot._unchecked(sympy_expr(r.function), _sympy_symbol(r.variable),
                                         _sympy_rational(r.lower_fraction), _sympy_rational(r.upper_fraction))


def native_root(r: TranscendentalRoot) -> native.TranscendentalRoot:
    """The native counterpart of a SymPy root object."""
    return native.TranscendentalRoot(core_expr(r.function), core_symbol(r.variable), _fraction(r.lower),
                                     _fraction(r.upper))


# ---------------------------------------------------------------------------
# the face

def _core_point(p: Expr) -> CoreExpr:
    return core_expr(p)


def isolate_real_roots(f: Expr, x: Symbol, domain: Set = S.Reals, depth: int = 4) -> Optional[list[Expr]]:
    """The real roots of ``f`` on ``domain`` (an interval or a union of
    intervals), each exact or a :class:`TranscendentalRoot`, in increasing
    order; ``None`` when they cannot be isolated
    (:func:`sympy_extras.solvers.isolation.isolate_real_roots` on SymPy
    objects).

    Examples
    ========

    >>> from sympy import exp, cos, sin, log, Interval, S
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.isolation_sympy import isolate_real_roots
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
    region = core_set(domain)
    if region is None:
        return None
    found = native.isolate_real_roots(core_expr(f), core_symbol(x), region, depth)
    return None if found is None else [sympy_expr(r) for r in found]


def compare_points(u: Point, v: Point) -> Optional[Sign]:
    """The sign of ``u - v`` for exact real constants, infinities and
    :class:`TranscendentalRoot` objects."""
    return native.compare_points(_core_point(u), _core_point(v))


def point_between(p: Point, q: Point) -> Optional[Rational]:
    """A rational number strictly between ``p < q`` (exact constants,
    infinities or root objects, whose intervals are refined until they
    separate)."""
    found = native.point_between(_core_point(p), _core_point(q))
    return None if found is None else Rational(found.p, found.q)


def real_roots_of(f: Expr, x: Symbol, domain: Set = S.Reals, depth: int = 4) -> Optional[FiniteSet]:
    """The roots of :func:`isolate_real_roots` as a set (``None`` when
    undecided)."""
    found = isolate_real_roots(f, x, domain, depth)
    if found is None:
        return None
    return FiniteSet(*found)
