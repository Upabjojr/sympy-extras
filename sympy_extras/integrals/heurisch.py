r"""The heuristic (parallel) Risch integrator, Bronstein's "poor man's
integrator".

The integrand `f` is written in the algebraically independent
*components* `t_1, \ldots, t_n` it is built from (the variable, the
function applications, the non-integer powers) and their derivatives, so
that `f` becomes a rational function `F(t_1, \ldots, t_n)` and the
derivation `D = \sum_i t_i' \partial/\partial t_i`. The antiderivative is
sought in the form

.. math::

    \frac{P(t_1, \ldots, t_n)}{Q(t_1, \ldots, t_n)}
    + \sum_j b_j \log q_j(t_1, \ldots, t_n)
    + \sum_k c_k \arctan r_k(t_1, \ldots, t_n),

with `Q` the *deflated* denominator of `F` (the part of it which can
appear after differentiation, Davenport's splitting into the normal and
the special part), the `q_j` the irreducible factors of the denominators,
the `\arctan r_k` from the pairs of complex-conjugate factors, `P` a
polynomial with unknown coefficients up to a degree bound, and the
coefficients `b_j, c_k` unknown; `D(\text{candidate}) = F` is then a
system of *linear* equations in the unknowns, solved exactly over the
field of the constants. Where the linear system has no solution the
method fails, which does not prove that no elementary antiderivative
exists; the components are permuted by type and the degree bound raised
before giving up.

This is a port of SymPy's ``sympy.integrals.heurisch`` (BSD 3-clause,
see ``sympy_extras/LICENSE-SymPy``), strictly typed, with:

- **verification**: every antiderivative is checked by differentiation,
  symbolically or, failing that, numerically where the integrand is real,
  and refused otherwise (SymPy's integrator returns a few wrong
  antiderivatives, see the tests);
- **real forms**: pairs of logarithms with complex-conjugate arguments
  are written as a logarithm of the modulus and an arctangent, so that a
  real integrand gets a real antiderivative;
- **a wider table of candidates**, always on (SymPy's needs
  ``hints=[]``): the error functions for `e^{a x^2 + b x + c}`, the
  inverse hyperbolic and circular functions for `\sqrt{a x^2 \pm b}`,
  the logarithmic integral, and the exponential, sine and cosine
  integrals for `e^{a x^b}`, `\sin(a x^b)` and `\cos(a x^b)`;
- **exact arithmetic only**: an integrand with floating-point numbers
  is refused.

The module works on the objects of the core (:mod:`sympy_extras.core`,
the functions of :mod:`sympy_extras.functions`) with the native
polynomials (:mod:`sympy_extras.polys.polytools`: ``cancel``, ``gcd``,
``lcm``, ``factor_list``, ``roots``), the native ``simplify`` and the
native numerical comparison of :mod:`~sympy_extras.integrals.conditions`,
and imports no SymPy: :mod:`sympy_extras.integrals.heurisch_sympy` is its
face on SymPy objects. The constants of the polynomials which the native
``Poly`` does not take as coefficients (``pi``, ``sqrt(2)``, ``exp(y)``,
``I``) are replaced by placeholder symbols for the polynomial operations
(the transcendental generators of SymPy's composite domains), and the
linear system is solved over ``QQ``, ``QQ<I>`` or the field of rational
functions of the parameters and placeholders by its reduced row echelon
form (unique for the order of the unknowns: the solution SymPy's
``solve_lin_sys`` finds).

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import exp, cos
>>> from sympy_extras.integrals.heurisch import heurisch_antiderivative
>>> x, y = symbols('x y')
>>> heurisch_antiderivative(y*x*cos(x**2), x)
y*sin(x**2)/2
>>> heurisch_antiderivative(exp(-x**2)*x, x)
-exp(-x**2)/2
>>> heurisch_antiderivative(1/(x**2 + 1), x)
atan(x)
>>> heurisch_antiderivative(exp(x)/x, x)
Ei(x)

References
==========

.. [Bronstein98] M. Bronstein, *Symbolic Integration Tutorial*, ISSAC
   1998, section 4 (the parallel Risch algorithm), and *pmint*, the poor
   man's integrator.
.. [GeddesStefanus] K. Geddes, L. Stefanus, On the Risch–Norman
   integration method and its implementation in Maple, ISSAC 1989.
.. [Davenport82] J. H. Davenport, On the parallel Risch algorithm (I),
   EUROCAM 1982, LNCS 144; (III): use of tangents, SIGSAM Bulletin 16
   (1982); with B. M. Trager, (II), ACM TOMS 11 (1985).
.. [GCL] K. Geddes, S. Czapor, G. Labahn, *Algorithms for Computer
   Algebra*, Kluwer, 1992, chapter 12.
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/integrals/heurisch.py, sympy/polys/polyroots.py
# (root_factors), sympy/solvers/solvers.py (denoms) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: strict typing; every result verified by differentiation; on the
# core, with placeholders for the constants of the polynomial operations
# and a reduced row echelon form for the linear system.
# ---------------------------------------------------------------------
from __future__ import annotations

import math

from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement, permutations
from typing import Callable, Final, Iterable, Iterator, Mapping, Optional, Sequence, TypeVar

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import evalf
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Application, Derivative, count_ops, diff, expand, expand_mul, expand_power_exp
from sympy_extras.core.logic import Boolean, and_, or_, true
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Float, I, Integer, Rational, pi
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.relational import eq, ne
from sympy_extras.core.singleton import S
from sympy_extras.core.sorting import ordered
from sympy_extras.core.symbol import Dummy, Symbol, Wild
from sympy_extras.core.traversal import iterfreeargs
from sympy_extras.functions.elementary.complexes import Abs, Arg, Im, Re, Sign
from sympy_extras.functions.elementary.exponential import Exp, LambertW, Log, exp, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Sinh, Tanh, asinh, tanh
from sympy_extras.functions.elementary.integers import Ceiling, Floor
from sympy_extras.functions.elementary.piecewise import piecewise
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Sin, Tan, asin, atan, tan
from sympy_extras.functions.special.bessel import (BesselI, BesselJ, BesselK, BesselY, Hankel1, Hankel2, SphericalJ,
                                                   SphericalY, besseli, besselj, besselk, bessely, hankel1, hankel2,
                                                   jn, yn)
from sympy_extras.functions.special.delta_functions import DiracDelta, Heaviside
from sympy_extras.functions.special.error_functions import Ci, Ei, Erf, Erfi, LiNode, Si, erf, erfi, li
from sympy_extras.functions.special.zeta_functions import polylog
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing, SparsePolynomialRing
from sympy_extras.polys.numberfields.qqbar import Algebraic, primitive_element
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel, factor_list, gcd, gcd_list, lcm
from sympy_extras.polys.orderings import Monomial
from sympy_extras.polys.polyutils import dict_from_expr
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powsimp
from sympy_extras.simplify.radsimp import collect_dict
from sympy_extras.simplify.simplify import simplify
from .conditions import numerically_equal

__all__ = ['components', 'heurisch_antiderivative', 'heurisch_cases']

#: the functions the method does not take: not elementary in the sense of
#: the derivation (a jump, a sign)
_REFUSED: Final[tuple[type[Basic], ...]] = (Abs, Re, Im, Sign, Heaviside, DiracDelta, Floor, Ceiling, Arg)

#: the errors of a polynomial operation which make a step give up
_GIVE_UP: Final[tuple[type[Exception], ...]] = (BasePolynomialError, ValueError, TypeError, ZeroDivisionError,
                                                 NotImplementedError)

#: a component and the symbol standing for it
Pair = tuple[Expr, Symbol]

def _sorted(items: Iterable[Expr]) -> list[Expr]:
    """SymPy's ``ordered`` of expressions."""
    return [as_expr(e) for e in ordered(items)]


def _replaced(e: Expr, rule: Mapping[Basic, Basic]) -> Expr:
    """``e.xreplace(rule)`` with the rebuilt nodes evaluated (SymPy's
    ``xreplace``; the core's rebuilds the nodes as they are)."""
    return as_expr(e._xreplace_evaluated(dict(rule)))


def _ops(e: Expr) -> int:
    """The number of operations of ``e`` (SymPy's ``count_ops``)."""
    value = count_ops(e)
    return value.p if isinstance(value, Integer) else 0


def components(f: ExprLike, x: Symbol) -> set[Expr]:
    """The functional components of ``f`` in ``x``: the symbol, the
    function applications and compositions, and the non-integer powers
    (fractional powers with the least positive exponent).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sin, cos, sqrt
    >>> from sympy_extras.integrals.heurisch import components
    >>> x = symbols('x')
    >>> sorted(components(sin(x)*cos(x)**2, x), key=str)
    [cos(x), sin(x), x]
    >>> sorted(components(sqrt(x + 1)**3, x), key=str)
    [sqrt(x + 1), x]
    """
    f_ = to_expr(f)
    result: set[Expr] = set()
    if not f_.has_free(x):
        return result
    if isinstance(f_, Symbol):
        result.add(f_)
    elif isinstance(f_, (Application, Derivative)):
        for g in f_.args:
            result |= components(as_expr(g), x)
        result.add(f_)
    elif isinstance(f_, Pow):
        base, exponent = f_.base, f_.exp
        result |= components(base, x)
        if not exponent.is_Integer:
            if isinstance(exponent, Rational):
                result.add(pow(base, Rational(1, exponent.q)))
            else:
                result |= components(exponent, x) | {f_}
    else:
        for g in f_.args:
            result |= components(as_expr(g), x)
    return result


_symbols_cache: dict[str, list[Dummy]] = {}


def _symbols(name: str, n: int) -> list[Symbol]:
    """``n`` symbols ``name0, name1, ...`` local to this module, the same
    on every call: the retries and the rewriting recurse with the mapping
    of the components to these symbols, which must be the same symbols
    (the bug: fresh dummies on each call left ``x/_x1`` as the result,
    the mapping of the inner call being unknown to the outer one)."""
    known = _symbols_cache.setdefault(name, [])
    while len(known) < n:
        known.append(Dummy('%s%i' % (name, len(known))))
    return list(known[:n])


#: a Bessel function of the core: the evaluating function of a class
BesselFunction = Callable[[ExprLike, ExprLike], Expr]


class _BesselTable:
    """The derivatives of the Bessel functions of orders ``n`` and ``n - 1``
    in terms of each other: the standard derivative brings in the orders
    ``n - 1`` and ``n + 1``, three functions linked by a recurrence, while
    the method needs algebraically independent components."""

    def __init__(self) -> None:
        self.n: Final[Dummy] = Dummy('n')
        self.z: Final[Dummy] = Dummy('z')
        n, z = self.n, self.z
        self.functions: Final[dict[type[Basic], BesselFunction]] = {
            BesselJ: besselj, BesselY: bessely, Hankel1: hankel1, Hankel2: hankel2, BesselI: besseli,
            BesselK: besselk, SphericalJ: jn, SphericalY: yn}
        self.table: Final[dict[type[Basic], tuple[Expr, Expr]]] = {}
        first: tuple[type[Basic], ...] = (BesselJ, BesselY, Hankel1, Hankel2)
        spherical: tuple[type[Basic], ...] = (SphericalJ, SphericalY)
        for cls in first:
            f = self.functions[cls]
            self.table[cls] = (f(n - 1, z) - n * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))
        self.table[BesselI] = (besseli(n - 1, z) - n * besseli(n, z) / z, (n - 1) * besseli(n - 1, z) / z + besseli(n, z))
        self.table[BesselK] = (-besselk(n - 1, z) - n * besselk(n, z) / z, (n - 1) * besselk(n - 1, z) / z - besselk(n, z))
        for cls in spherical:
            f = self.functions[cls]
            self.table[cls] = (f(n - 1, z) - (n + 1) * f(n, z) / z, (n - 1) * f(n - 1, z) / z - f(n, z))

    def diffs(self, cls: type[Basic], n: Expr, z: Expr) -> tuple[Expr, Expr]:
        d0, d1 = self.table[cls]
        replacement: dict[Basic, Basic] = {self.n: n, self.z: z}
        return (_replaced(d0, replacement), _replaced(d1, replacement))


_BESSEL = _BesselTable()


class _DiffCache:
    """The derivatives of the components with respect to ``x``, the Bessel
    functions through the table so that the pair of orders ``n, n - 1``
    is used throughout."""

    def __init__(self, x: Symbol) -> None:
        self.cache: Final[dict[Expr, Expr]] = {}
        self.x: Final[Symbol] = x

    def get(self, f: Expr) -> Expr:
        if f in self.cache:
            return self.cache[f]
        if isinstance(f, Application) and type(f) in _BESSEL.table and len(f.args) == 2:
            n, z = as_expr(f.args[0]), as_expr(f.args[1])
            d0, d1 = _BESSEL.diffs(type(f), n, z)
            dz = self.get(z)
            self.cache[f] = d0 * dz
            self.cache[_BESSEL.functions[type(f)](n - 1, z)] = d1 * dz
        else:
            # exponentials of sums as products of exponentials of their terms,
            # the canonical form of the integrand (see _heurisch)
            self.cache[f] = expand_power_exp(cancel(diff(f, self.x)))
        return self.cache[f]


def _matched(e: Expr, pattern: Expr, *wilds: Wild) -> Optional[list[Expr]]:
    """The values of ``wilds`` when ``e`` matches ``pattern`` (zero for a
    wild left unbound); ``None`` otherwise."""
    found = e.match(pattern)
    if found is None:
        return None
    return [as_expr(found[w]) if w in found else S.Zero for w in wilds]


def _special_candidates(terms: set[Expr], x: Symbol, cache: _DiffCache, f: Expr) -> set[Expr]:
    """The special functions whose derivatives are built from the
    components: the error functions of `e^{a x^2 + b x + c}` and of
    `e^{a \\log(x)^2}`, the inverse circular and hyperbolic functions and
    the logarithms of `\\sqrt{a x^2 \\pm b}`, the logarithmic integral,
    and the exponential, sine and cosine integrals of `e^{a x^b}`,
    `\\sin(a x^b)`, `\\cos(a x^b)`."""
    a, b, c = Wild('a', exclude=[x]), Wild('b', exclude=[x]), Wild('c', exclude=[x])
    found: set[Expr] = set()
    # the exponential, sine and cosine integrals only where x divides the
    # denominator (sin(x)/x), the products of error functions only where
    # erf already appears: each candidate is one more type of component,
    # and the permutations of the types are tried before the rewriting
    denominator = f.as_numer_denom()[1]
    over_x = denominator.is_polynomial(x) is True and denominator.subs(x, S.Zero) == S.Zero
    with_erf = f.has(Erf, Erfi)
    for g in list(terms):
        if isinstance(g, LiNode):
            values = _matched(as_expr(g.args[0]), a * x**b, a, b)
            if values is not None:
                A, B = values
                found.add(x * (li(A * x**B) - (A * x**B)**(-1 / B) * Ei((B + 1) * log(A * x**B) / B)))
        elif isinstance(g, Exp):
            argument = as_expr(g.args[0])
            # (the error functions come from the tower, _tower_candidates, with
            # their derivatives registered in terms of the components: the
            # derivative SymPy takes of erf(x - 1) has exp(-(x - 1)**2), a
            # component the ring cannot relate to exp(-x**2 + 2*x))
            values = _matched(argument, a * x**2, a)
            if values is not None and with_erf and values[0].is_negative is True:
                # the product of two such exponentials
                found.add(erf(sqrt(2) * sqrt(-values[0]) * x))
            values = _matched(argument, a * x**b, a, b)
            if over_x and values is not None and values[0] != S.Zero and values[1].is_nonzero is True:
                found.add(Ei(values[0] * x**values[1]))
        elif isinstance(g, (Sin, Cos)) and over_x:
            values = _matched(as_expr(g.args[0]), a * x**b, a, b)
            if values is not None and values[0] != S.Zero and values[1].is_nonzero is True:
                argument = values[0] * x**values[1]
                found.add(Si(argument))
                found.add(Ci(argument))
        elif isinstance(g, Pow) and isinstance(g.exp, Rational) and g.exp.q == 2:
            base = g.base
            values = _matched(base, a * x**2 + b, a, b)
            if values is not None and values[1].is_positive is True:
                A, B = values
                if A.is_positive is True:
                    found.add(asinh(sqrt(A / B) * x))
                elif A.is_negative is True:
                    found.add(asin(sqrt(-A / B) * x))
            values = _matched(base, a * x**2 - b, a, b)
            if values is not None and values[1].is_positive is True:
                A, B = values
                if A.is_positive is True:
                    primitive = log(2 * sqrt(A) * sqrt(A * x**2 - B) + 2 * A * x) / sqrt(A)
                    cache.cache[primitive] = 1 / sqrt(A * x**2 - B)
                    found.add(primitive)
                elif A.is_negative is True:
                    found.add(-B / 2 * sqrt(-A) * atan(sqrt(-A) * x / sqrt(A * x**2 - B)))
            # (the primitives of 1/(x sqrt(Q)), log((sqrt(b) + sqrt(Q))/x) and
            # atan(sqrt(Q)/sqrt(b)), are of no use here: their derivatives
            # match the integrand only modulo sqrt(Q)**2 = Q, while the
            # derivation takes sqrt(Q) as an independent component; these
            # integrands are for the radical table and Trager's algorithm)
            values = _matched(base, a * x**2 + b * x + c, a, b, c)
            if values is not None and values[0] != S.Zero and values[1] != S.Zero:
                # a completed square: the primitives of 1/sqrt(Q) with Q = a x**2 + b x + c
                A, B, C = values
                discriminant = B**2 - 4 * A * C
                if A.is_negative is True and discriminant.is_positive is True:
                    found.add(asin((2 * A * x + B) / sqrt(discriminant)))
                elif A.is_positive is True and discriminant.is_negative is True:
                    found.add(asinh((2 * A * x + B) / sqrt(-discriminant)))
                elif A.is_positive is True and discriminant.is_positive is True:
                    primitive = log(2 * sqrt(A) * sqrt(base) + 2 * A * x + B) / sqrt(A)
                    cache.cache[primitive] = 1 / sqrt(base)
                    found.add(primitive)
    return found


def _denominator(q: Expr) -> int:
    """The denominator of a rational number, 1 for anything else."""
    return int(q.q) if isinstance(q, Rational) else 1


def _exponential_family(g: Expr) -> tuple[Expr, Rational]:
    """``(w, q)`` with ``g == exp(q*w)``, ``w`` the argument without its
    rational content and sign (``exp(2*x)`` and ``exp(-x)`` belong to the
    family of ``x``, with ``q = 2`` and ``q = -1``)."""
    content, w = as_expr(g.args[0]).as_content_primitive()
    q: Rational = content if isinstance(content, Rational) else Integer(1)
    if w.could_extract_minus_sign():
        w, q = -w, Rational(-q.p, q.q)
    return (w, q)


def _exponential_bases(terms: set[Expr]) -> set[Expr]:
    """The terms with each family of rationally related exponentials
    represented by one base, ``exp(w/d)`` for the least common denominator
    ``d`` of their multiples, of which the others are integer powers: the
    derivative of ``exp(-exp(2*x))`` brings in ``exp(2*x)``, which the
    method must know is ``exp(x)**2`` (``subs`` writes it so once the base
    is the component)."""
    families: dict[Expr, list[tuple[Expr, Rational]]] = {}
    for g in terms:
        if isinstance(g, Exp):
            w, q = _exponential_family(g)
            families.setdefault(w, []).append((g, q))
    result = {g for g in terms if not isinstance(g, Exp)}
    for w, members in families.items():
        d = 1
        for _, q in members:
            d = math.lcm(d, _denominator(q))
        result.add(exp(w / d))
    return result


def _power_coefficients(e: Expr, v: Expr) -> Optional[dict[int, Expr]]:
    """The coefficients of the powers of ``v`` in the expanded ``e``
    (SymPy's ``Poly(e, v)`` with any coefficients), ``None`` when ``e`` is
    not a polynomial in ``v``; the zero coefficients left out."""
    coefficients: dict[int, list[Expr]] = defaultdict(list)
    for term in Add.make_args(expand(e)):
        degree = 0
        rest: list[Expr] = []
        for factor in Mul.make_args(term):
            if factor == v:
                degree += 1
            elif isinstance(factor, Pow) and factor.base == v and isinstance(factor.exp, Integer) and factor.exp.p > 0:
                degree += factor.exp.p
            elif factor.has(v):
                return None
            else:
                rest.append(factor)
        coefficients[degree].append(mul(*rest))
    result: dict[int, Expr] = {}
    for degree, parts in coefficients.items():
        total = add(*parts)
        if total != S.Zero:
            result[degree] = total
    return result


def _degree(e: Expr, v: Expr) -> int:
    """The degree of the polynomial ``e`` in ``v``, 0 when it is not one."""
    found = _power_coefficients(e, v)
    return max(found) if found else 0


def _degree_in(e: Expr, g: Expr) -> int:
    """The degree of ``e`` in the component ``g`` (``exp(2*x)`` counts as
    ``exp(x)**2``), 0 when ``e`` is not polynomial in it."""
    d = Dummy('g')
    return _degree(_replaced(powsimp(e, combine='exp'), {g: d}), d)


class _Placeholders:
    """The constants of the polynomials which the native ``Poly`` does not
    take as coefficients, as symbols: the parts free of the generators
    which are not polynomials in symbols and rational numbers (``pi``,
    ``sqrt(2)``, ``exp(y)``, ``I``, and ``1/y`` unless ``rational``), each
    one a placeholder symbol, as SymPy's composite domains make them
    transcendental generators (``ZZ[pi]``)."""

    def __init__(self, gens: Sequence[Expr], rational: bool = False) -> None:
        self.gens: Final[frozenset[Expr]] = frozenset(gens)
        self.rational: Final[bool] = rational
        self.forward: Final[dict[Expr, Symbol]] = {}
        self.back: Final[dict[Basic, Basic]] = {}

    def _placeholder(self, e: Expr) -> Symbol:
        if e not in self.forward:
            symbol = Dummy('c%d' % len(self.forward))
            self.forward[e] = symbol
            self.back[symbol] = e
        return self.forward[e]

    def abstract(self, e: Expr) -> Expr:
        """``e`` with its constants replaced (see the class)."""
        if isinstance(e, Rational) or (isinstance(e, Symbol) and e not in self.back):
            return e
        if e == I:
            return self._placeholder(e)
        if isinstance(e, Add):
            return add(*[self.abstract(a) for a in e.args])
        if isinstance(e, Mul):
            return mul(*[self.abstract(a) for a in e.args])
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            base = self.abstract(e.base)
            if e.exp.p >= 0 or self.rational or base.has(*self.gens):
                return pow(base, e.exp)
            return self._placeholder(e)
        if any(s in self.gens for s in e.free_symbols):
            return e
        return self._placeholder(e)

    def restore(self, e: Expr) -> Expr:
        """The constants back in ``e`` (placeholders of placeholders too)."""
        while self.back and any(s in self.back for s in e.free_symbols):
            e = _replaced(e, self.back)
        return e


def _with_constants(gens: Sequence[Expr], exprs: Sequence[Expr]) -> tuple[_Placeholders, list[Expr]]:
    placeholders = _Placeholders(gens)
    return placeholders, [placeholders.abstract(e) for e in exprs]


def _factor_list(e: Expr, gens: Sequence[Expr]) -> tuple[Expr, list[tuple[Expr, int]]]:
    """SymPy's ``factor_list(e, *gens)`` over the composite domain of the
    constants of ``e``."""
    placeholders, (abstracted,) = _with_constants(gens, [e])
    coefficient, factors = factor_list(abstracted, *gens)
    result: list[tuple[Expr, int]] = []
    for base, k in factors:
        restored = placeholders.restore(base)
        if not isinstance(k, Integer):
            raise ValueError('a symbolic multiplicity')
        if not restored.has(*gens):
            coefficient = coefficient * base**k
            continue
        result.append((restored, k.p))
    return (placeholders.restore(coefficient), result)


def _gcd(f: Expr, g: Expr, gens: Sequence[Expr]) -> Expr:
    """SymPy's ``gcd(f, g, *gens)`` over the composite domain."""
    placeholders, (a, b) = _with_constants(gens, [f, g])
    return placeholders.restore(gcd(a, b, *gens))


def _lcm(f: Expr, g: Expr, gens: Sequence[Expr]) -> Expr:
    """SymPy's ``lcm(f, g, *gens)`` over the composite domain."""
    placeholders, (a, b) = _with_constants(gens, [f, g])
    return placeholders.restore(lcm(a, b, *gens))


def _total_degree(e: Expr, gens: Sequence[Expr]) -> Optional[int]:
    """The total degree of ``e`` in ``gens``; ``None`` when it is not a
    polynomial in them (SymPy's ``e.as_poly(*gens)`` is ``None``)."""
    placeholders, (a,) = _with_constants(gens, [e])
    try:
        return Poly(a, *gens).total_degree()
    except BasePolynomialError:
        return None


def _primitive(p: Expr, y: Expr) -> tuple[Expr, Expr]:
    """The content and the primitive part of ``p`` as a polynomial in
    ``y`` (SymPy's ``p.as_poly(y).primitive()``)."""
    coefficients = _power_coefficients(p, y)
    if coefficients is None:
        raise ValueError('%s is not a polynomial in %s' % (p, y))
    gens = sorted({s for c in coefficients.values() for s in c.free_symbols if isinstance(s, Symbol)},
                  key=lambda s: s.sort_key())
    placeholders = _Placeholders(gens)
    values = [placeholders.abstract(c) for c in coefficients.values()]
    content = placeholders.restore(gcd_list(values, *gens)) if gens else gcd_list(values)
    if content == S.Zero:
        return (S.Zero, p)
    primitive = add(*[cancel(c / content) * y**k for k, c in coefficients.items()])
    return (content, primitive)


def _root_factors(f: Expr, z: Expr, field: Optional[str]) -> list[Expr]:
    """SymPy's ``root_factors(f, z, filter=field)``: the linear factors of
    the roots of ``f`` in ``z`` and the cofactor of the roots not found."""
    gens = [z]
    placeholders, (a,) = _with_constants(gens, [f])
    F = Poly(a, z)
    zeros = roots(F, filter=field)
    if not isinstance(zeros, dict) or not zeros:
        return [f]
    factors: list[Expr] = []
    found = 0
    for r in _sorted(zeros):
        factors.extend([z - placeholders.restore(r)] * zeros[r])
        found += zeros[r]
    if found < F.degree():
        product = mul(*factors)
        factors.append(cancel(expand(f) / product))
    return factors


def _square_root_form(e: Expr, x: Symbol, terms: set[Expr]) -> Optional[tuple[Expr, Expr]]:
    """``(u, c)`` with ``e == u**2 + c``, ``u`` built from the components of
    ``e`` and ``c`` a constant or an exponent whose exponential is built
    from the components in ``terms``: ``e`` a perfect square times a
    positive constant (``(x + exp(x))**2``), or quadratic in one component
    with a constant leading coefficient, the square completed
    (``-x**2 - 2*x*exp(x) - exp(2*x) + 5``, and ``(x + exp(x))**2 - x``
    with ``exp(-x)`` among the components); ``None`` otherwise."""
    expanded = expand(e)
    # each family of exponentials as powers of one dummy standing for the
    # square root of its base, exp(w/(2*d)): exp(2*x) is then d**4 and a square
    dummies: dict[Expr, Dummy] = {}
    families: dict[Expr, int] = {}
    for g in components(expanded, x):
        if isinstance(g, Exp):
            w, q = _exponential_family(g)
            families[w] = math.lcm(families.get(w, 1), _denominator(q))
    for w, d in families.items():
        dummies[exp(w / (2 * d))] = Dummy('e')
    reduced = expanded
    for g, dummy in dummies.items():
        reduced = reduced.subs(g, dummy)
    back: dict[Basic, Basic] = {dummy: g for g, dummy in dummies.items()}
    gens: list[Expr] = sorted((g for g in components(reduced, x) if not isinstance(g, Exp)), key=str)
    gens += list(dummies.values())
    if not gens:
        return None
    try:
        constant, factors = _factor_list(reduced, gens)
    except _GIVE_UP:
        return None
    if factors and all(k % 2 == 0 for _, k in factors):
        if constant.is_positive is True:
            u: Expr = sqrt(constant)
            for base, k in factors:
                u = u * base**(k // 2)
            return (_replaced(u, back), S.Zero)
    for v in gens:
        coefficients = _power_coefficients(reduced, v)
        if coefficients is None or not coefficients or max(coefficients) != 2:
            continue
        A = _replaced(coefficients.get(2, S.Zero), back)
        B = _replaced(coefficients.get(1, S.Zero), back)
        component = as_expr(back[v]) if v in back else v
        if A.has(x) or A.is_positive is not True:
            continue
        # the square completed with the whole linear coefficient, or with its
        # part in the other generators only: (x + exp(x))**2 - x has the
        # coefficient 2*exp(x) - 1, and the remainder -x is an offset whose
        # exponential exp(-x) is in the field
        number, rest = B.as_coeff_Add()
        for linear in ([B] if number == S.Zero or rest == S.Zero else [B, rest]):
            u = sqrt(A) * (component + linear / (2 * A))
            c = cancel(expand(e) - expand(u**2))
            if c.has(x) and not (components(exp(c), x) - {x} <= terms or components(exp(-c), x) - {x} <= terms):
                continue
            return (u, c)
    return None


def _factors(e: Expr) -> list[Expr]:
    """The irreducible factors of ``e`` (its generators found as the native
    ``factor_list`` finds them), ``[e]`` when it cannot factor it."""
    try:
        return [q for q, _ in factor_list(e)[1]]
    except _GIVE_UP:
        return [e]


def _tower_candidates(terms: set[Expr], x: Symbol, cache: _DiffCache, f: Expr) -> set[Expr]:
    """The special functions the structure of the tower allows (Cherry's
    theorems on integration in finite terms with error functions and
    logarithmic integrals), their derivatives registered in the cache in
    terms of the components: for an exponential ``exp(theta)`` and for a
    logarithm ``theta = log(h)``, ``Ei(theta + c)`` when a factor
    ``alpha*theta + beta`` of the denominator gives ``c = beta/alpha``
    (``Ei(x + 1)`` for ``exp(x)/(x + 1)**2``, ``Ei(log(x) + 1)`` for
    ``1/(log(x) + 1)``, the logarithmic integral); ``erf(u)`` when
    ``-theta`` is ``u**2 + c`` (``erf(x + exp(x))``) and ``erfi(u)`` when
    ``theta`` is; ``polylog(2, -exp(theta))`` and ``polylog(2, exp(theta))``
    when ``exp(theta) + 1`` or ``exp(theta) - 1`` divides the denominator
    or their logarithms appear. Every candidate is one more component,
    and the caller verifies whatever it finds."""
    found: set[Expr] = set()
    numerator, denominator = f.as_numer_denom()
    factors = _factors(denominator)
    logarithms = {as_expr(g.args[0]) for g in f.atoms(Log)}
    for g in list(terms):
        exponential: Expr
        if isinstance(g, Exp):
            theta = as_expr(g.args[0])
            exponential = g
        elif isinstance(g, Log):
            theta = g
            exponential = as_expr(g.args[0])                # exp(log(h)) is h
        else:
            continue
        dtheta = cancel(diff(theta, x))
        if dtheta == S.Zero:
            continue
        # the multiples k the degrees allow: up to the degree of the numerator in
        # the exponential, one more for a logarithm (x/(log(x) + 1) is
        # exp(2*theta)*theta'/(theta + 1) with theta = log(x)), and negative ones
        # only where the exponential divides the denominator
        above = _degree_in(numerator, exponential) + (0 if isinstance(g, Exp) else 1)
        below = _degree_in(denominator, exponential)
        multiples = list(range(1, max(above, 1) + 1)) + list(range(-1, -below - 1, -1))
        for q in factors:
            if not q.has(x):
                continue
            alpha = cancel(diff(q, x) / dtheta)
            if alpha == S.Zero or alpha.has(x):
                continue
            beta = cancel(q - alpha * theta)
            if beta.has(x):
                continue
            c = beta / alpha
            # exp(k*theta)*theta'/(theta + c) is the derivative of Ei(k*(theta + c))*exp(-k*c):
            # the multiples k up to the degree of the integrand in the exponential
            for k in multiples:
                candidate = Ei(k * (theta + c))
                cache.cache[candidate] = exp(k * c) * exponential**k * dtheta / (theta + c)
                found.add(candidate)
        if not isinstance(g, Exp):
            continue
        for offset, argument in ((1, -g), (-1, g)):
            if any(cancel(q - offset - g) == S.Zero or cancel(q + offset + g) == S.Zero for q in factors) \
                    or any(cancel(h - offset - g) == S.Zero or cancel(h + offset + g) == S.Zero for h in logarithms):
                candidate = polylog(2, argument)
                cache.cache[candidate] = -log(1 - argument) * dtheta
                found.add(candidate)
    # the error functions from the total exponent of each term of f, the sum of
    # the arguments of its exponential factors (exp(-x**2)*exp(2*x) in the
    # canonical form of the integrand is exp(-(x - 1)**2 + 1))
    exponents: list[Expr] = []
    for term in Add.make_args(f):
        theta_term: Expr = S.Zero
        for piece in Mul.make_args(term):
            if isinstance(piece, Exp):
                theta_term = theta_term + as_expr(piece.args[0])
                exponents.append(as_expr(piece.args[0]))     # a single factor too: exp(x)*exp(-exp(2*x)) needs erf(exp(x))
            elif isinstance(piece, Pow) and isinstance(piece.base, Exp) and not piece.exp.has(x):
                theta_term = theta_term + piece.exp * as_expr(piece.base.args[0])
        if theta_term != S.Zero:
            exponents.append(theta_term)
    functions: tuple[tuple[int, Callable[[ExprLike], Expr]], ...] = ((-1, erf), (1, erfi))
    for theta_term in exponents:
        for sign_, function in functions:
            form = _square_root_form(sign_ * theta_term, x, terms)
            if form is None:
                continue
            u, c = form
            candidate = function(u)
            # erf(u)' = 2/sqrt(pi) u' exp(-u**2) with -u**2 = theta + c, erfi(u)' = 2/sqrt(pi) u' exp(u**2)
            # with u**2 = theta - c: the factor exp(-sign*c), a constant or a product of components
            # (exp(-x) for (1 + exp(-x))*exp(-(x + exp(x))**2 + x)), beside the exponentials of the
            # term in the canonical form; the products are distributed (exp(x)*exp(-x) is 1)
            cache.cache[candidate] = expand_mul(expand_power_exp(
                2 / sqrt(pi) * diff(u, x) * exp(-sign_ * c) * exp(theta_term)))
            found.add(candidate)
    return found


def _exponent(g: Expr) -> int:
    """The contribution of the fractional powers of ``g`` to the degree
    bound of the polynomial part."""
    if isinstance(g, Pow):
        exponent = g.exp
        if isinstance(exponent, Rational) and exponent.q != 1:
            if exponent.p > 0:
                return exponent.p + exponent.q - 1
            return abs(exponent.p + exponent.q)
        return 1
    if not g.is_Atom and g.args:
        return max(_exponent(as_expr(h)) for h in g.args)
    return 1


def _logarithmic_part(term: Expr) -> tuple[Expr, Expr]:
    """SymPy's ``term.as_independent(log, as_Add=False)``: the factors
    without a logarithm and those with one."""
    if isinstance(term, Mul):
        free = [a for a in term.args if not a.has(Log)]
        dependent = [a for a in term.args if a.has(Log)]
        return (mul(*free), mul(*dependent))
    return (S.One, term) if term.has(Log) else (term, S.One)


def _real_forms(F: Expr) -> Expr:
    """Pairs ``a*log(P + I*Q) + b*log(P - I*Q)`` of the antiderivative
    written ``(a + b)/2 * log(P**2 + Q**2) - I*(a - b)*atan(P/Q)`` (the same
    derivative, ``atan(P/Q)`` being ``pi/2 - atan(Q/P)`` on each side of
    ``P = 0``), which is real when ``b`` is the conjugate of ``a``.

    >>> from sympy_extras.core import symbols, I
    >>> from sympy_extras.functions import log
    >>> from sympy_extras.integrals.heurisch import _real_forms
    >>> x = symbols('x')
    >>> _real_forms(-I*log(x - I)/2 + I*log(x + I)/2)
    atan(x)
    """
    if not F.has(I):
        return F
    logs: dict[tuple[Expr, Expr], list[tuple[Expr, int]]] = defaultdict(list)
    rest: list[Expr] = []
    for term in Add.make_args(F):
        coefficient, function = _logarithmic_part(term)
        if not isinstance(function, Log) or not function.args[0].has(I):
            rest.append(term)
            continue
        parts = collect_dict(as_expr(function.args[0]), I)
        P, Q = parts.get(S.One, S.Zero), parts.get(I, S.Zero)
        if P.has(I) or Q.has(I) or Q == S.Zero:
            rest.append(term)
            continue
        if Q.could_extract_minus_sign():
            logs[(P, -Q)].append((coefficient, -1))
        else:
            logs[(P, Q)].append((coefficient, 1))
    result: Expr = add(*rest)
    for (P, Q), entries in logs.items():
        a = add(*[c for c, s in entries if s == 1])
        b = add(*[c for c, s in entries if s == -1])
        combined = expand((a + b) / 2 * log(P**2 + Q**2) - I * (a - b) * atan(P / Q))
        if combined.has(I):
            # not a conjugate pair: as it was
            result = result + a * log(P + I * Q) + b * log(P - I * Q)
        else:
            result = result + combined
    return result


def _verified(F: Expr, f: Expr, x: Symbol) -> Optional[Expr]:
    """``F`` when ``F' == f``, symbolically or numerically where ``f`` is
    real; ``None`` otherwise."""
    difference = diff(F, x) - f
    reduced = attempt(lambda: cancel(difference), settings.timeout, native=True)
    if reduced is not None and reduced == S.Zero:
        return F
    simpler = attempt(lambda: simplify(difference), settings.timeout, native=True)
    if simpler is not None and simpler == S.Zero:
        return F
    if numerically_equal(diff(F, x), f):
        return F
    return None


def heurisch_antiderivative(f: ExprLike, x: Symbol, hints: Optional[Sequence[ExprLike]] = None,
                            degree_offset: int = 0, unnecessary_permutations: Optional[list[Pair]] = None,
                            retries: int = 2) -> Optional[Expr]:
    """An antiderivative of ``f`` in ``x`` by the heuristic Risch method,
    verified by differentiation, or ``None`` (never an unevaluated
    integral). ``hints`` are functions which may appear in the
    antiderivative besides the automatic candidates; ``degree_offset``
    raises the degree bound of the polynomial part; ``retries`` is the
    number of permutations of the components tried after the first.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import tan, log, sqrt, exp
    >>> from sympy_extras.integrals.heurisch import heurisch_antiderivative
    >>> x, y = symbols('x y')
    >>> heurisch_antiderivative(y*tan(x), x)
    y*log(tan(x)**2 + 1)/2
    >>> heurisch_antiderivative(1/sqrt(x**2 + 1), x)
    asinh(x)
    >>> heurisch_antiderivative(log(x)**2/x**3, x)
    -log(x)**2/(2*x**2) - log(x)/(2*x**2) - 1/(4*x**2)
    >>> heurisch_antiderivative(exp(x**2)*exp(x), x)
    sqrt(pi)*exp(-1/4)*erfi(x + 1/2)/2
    """
    f_ = to_expr(f)
    if f_.has(Float) or f_.has(*_REFUSED):
        return None
    if not f_.has_free(x):
        return f_ * x
    hint_terms = [to_expr(h) for h in hints] if hints else []

    def search() -> Optional[Expr]:
        found = _heurisch(f_, x, False, hint_terms, None, retries, degree_offset, unnecessary_permutations)
        if found is None and degree_offset == 0:
            found = _heurisch(f_, x, False, hint_terms, None, retries, 1, unnecessary_permutations)
        return found

    # the whole search under the time limit of the settings: the linear
    # systems of the permutations and retries each took their own share
    result = attempt(search, settings.timeout, native=True)
    if result is None:
        return None
    return _verified(_real_forms(result), f_, x)


def _itermonomials(V: Sequence[Symbol], bound: int) -> list[Expr]:
    """The monomials of total degree at most ``bound`` in ``V`` (SymPy's
    ``itermonomials``), in SymPy's ``ordered`` order."""
    monomials: list[Expr] = []
    for degree in range(bound + 1):
        for chosen in combinations_with_replacement(V, degree):
            monomials.append(mul(*chosen))
    return _sorted(monomials)


def _find_non_syms(expr: Expr, syms: set[Expr], non_syms: set[Expr]) -> None:
    """The maximal subexpressions of ``expr`` free of ``syms`` (other than
    rational numbers) added to ``non_syms``; ``ValueError`` when ``expr``
    is not a polynomial in ``syms`` (SymPy's ``PolynomialError``)."""
    if isinstance(expr, Rational):
        return
    if expr in syms:
        return
    if not expr.has_free(*syms):
        non_syms.add(expr)
    elif isinstance(expr, (Add, Mul, Pow)):
        for argument in expr.args:
            _find_non_syms(as_expr(argument), syms, non_syms)
    else:
        raise ValueError('%s is not a polynomial in the unknowns' % (expr,))


T = TypeVar('T')
U = TypeVar('U')


def _echelon_solution(rows: list[dict[int, T]], K: Domain[T], unknowns: int) -> Optional[dict[int, T]]:
    """The values of the pivot unknowns of the reduced row echelon form of
    ``rows`` (the last column the constant term) when the free ones are
    zero; ``None`` for an inconsistent system."""
    solution: dict[int, T] = {}
    for row in K.sparse_echelon(rows):
        lead = min(row)
        if lead == unknowns:
            return None
        solution[lead] = K.neg(row[unknowns]) if unknowns in row else K.zero
    return solution


#: the largest degree of the field of the algebraic constants of a linear
#: system (beyond it they are taken as transcendental placeholders)
_MAX_FIELD_DEGREE: Final[int] = 32


def _algebraic(atom: Expr) -> Optional[Algebraic]:
    """The algebraic number ``atom`` (``I`` or a radical ``r**(p/q)`` of a
    rational number), checked against its numerical value; ``None`` for
    another constant."""
    value: Optional[Algebraic] = None
    if atom == I:
        value = Algebraic.imaginary_unit()
    elif isinstance(atom, Pow) and isinstance(atom.base, Rational) and isinstance(atom.exp, Rational) \
            and atom.exp.q > 1:
        base = Algebraic.from_rational(Fraction(atom.base.p, atom.base.q))
        value = base.nth_root(atom.exp.q)**atom.exp.p
    if value is None:
        return None
    numeric = complex(evalf(atom, 30))
    if abs(complex(value) - numeric) > 1e-9 * (1 + abs(numeric)):
        return None
    return value


def _horner(coefficients: Sequence[Fraction], point: list[Fraction], K: AlgebraicField) -> list[Fraction]:
    """The polynomial (rational coefficients, highest degree first) at the
    element ``point`` of ``K``."""
    result = K.zero
    for c in coefficients:
        result = K.add(K.mul(result, point), K.from_rational(Fraction(c)))
    return result


class _UnorderedField(AlgebraicField):
    """A number field as the ground of polynomials whose fractions are
    reduced: no element counts as negative (the sign normalisation of the
    denominators is skipped; the fractions stay in lowest terms)."""

    def is_negative(self, a: list[Fraction]) -> bool:
        return False


class _NumberField:
    """``QQ(theta)`` generated by algebraic constants, with each one as a
    polynomial in the symbol ``theta`` and ``theta`` as an expression in
    them (``theta = a_1 + c_2 a_2 + ...``, the primitive elements of
    :func:`~sympy_extras.polys.numberfields.qqbar.primitive_element`);
    ``ArithmeticError`` when the degree exceeds :data:`_MAX_FIELD_DEGREE`."""

    def __init__(self, atoms: Sequence[Expr], values: Sequence[Algebraic]) -> None:
        self.theta: Final[Symbol] = Symbol('_theta')
        gamma = values[0]
        gamma_expr = atoms[0]
        coordinates: list[list[Fraction]] = [[Fraction(1), Fraction(0)]]
        for atom, value in zip(atoms[1:], values[1:]):
            new, p, q = primitive_element(gamma, value)
            if new.degree > _MAX_FIELD_DEGREE:
                raise ArithmeticError('the field of the constants is too large')
            K = AlgebraicField([Fraction(c, new.minpoly[0]) for c in new.minpoly], self.theta)
            gamma_in_new = _horner(p, K.alpha, K)
            coordinates = [_horner(c, gamma_in_new, K) for c in coordinates] + [_horner(q, K.alpha, K)]
            c = next(k for k in range(1, 4 * gamma.degree * value.degree + 2) if gamma + k * value == new)
            gamma, gamma_expr = new, gamma_expr + c * atom
        self.field: Final[AlgebraicField] = _UnorderedField([Fraction(c, gamma.minpoly[0]) for c in gamma.minpoly],
                                                            self.theta)
        self.theta_expr: Final[Expr] = gamma_expr
        #: each constant as a polynomial in theta
        self.forms: Final[dict[Basic, Basic]] = {atom: self.field.to_expr(element)
                                                 for atom, element in zip(atoms, coordinates)}


def _solve_coefficients(numerator: Expr, V: Sequence[Symbol], coefficients: Sequence[Symbol],
                        non_syms: set[Expr]) -> Optional[dict[Basic, Basic]]:
    """The values of the ``coefficients`` making the polynomial
    ``numerator`` in ``V`` vanish, the free unknowns zero (SymPy's
    ``solve_lin_sys`` over the ring of the coefficients, followed by the
    substitution of zeros); ``None`` when the system is inconsistent.

    The constants (``non_syms``) are taken over ``QQ``, the number field
    of the algebraic ones (``I``, radicals of rational numbers: SymPy's
    ``EX`` decides their relations, ``sqrt(2)**2 = 2`` in the arctangents
    of ``1/(x**4 + 1)``), and the rational functions of the parameters and
    of the other constants as placeholders (:class:`_Placeholders`, the
    transcendental generators of SymPy's ``ZZ(pi)``)."""
    placeholders = _Placeholders(list(V) + list(coefficients), rational=True)
    abstracted = _replaced(numerator, {c: placeholders.abstract(c) for c in non_syms})
    atoms: list[Expr] = []
    values: list[Algebraic] = []
    for placeholder, constant in placeholders.back.items():
        value = _algebraic(as_expr(constant))
        if value is not None:
            atoms.append(as_expr(placeholder))
            values.append(value)
    number_field: Optional[_NumberField] = None
    if atoms:
        try:
            number_field = _NumberField([as_expr(placeholders.back[a]) for a in atoms], values)
        except ArithmeticError:
            number_field = None
    if number_field is not None:
        abstracted = _replaced(abstracted, {a: number_field.forms[placeholders.back[a]] for a in atoms})
    unknown_set: set[Basic] = set(V) | set(coefficients)
    if number_field is not None:
        unknown_set.add(number_field.theta)
    params = sorted({s for s in abstracted.free_symbols if isinstance(s, Symbol) and s not in unknown_set},
                    key=lambda s: s.sort_key())
    # unique names: the rings map their symbols by name (the dummies x0, A0, ...
    # may share theirs with a parameter)
    v_names = ['_v%d' % i for i in range(len(V))]
    c_names = ['_c%d' % i for i in range(len(coefficients))]
    p_names = ['_p%d' % i for i in range(len(params))]
    renamed: dict[Basic, Basic] = {}
    for symbol, name in zip(list(V) + list(coefficients) + params, v_names + c_names + p_names):
        renamed[symbol] = Symbol(name)
    abstracted = _replaced(abstracted, renamed)
    if params and any(isinstance(node, Pow) and isinstance(node.exp, Integer) and node.exp.p < 0
                      for node in _preorder(abstracted)):
        # the denominators of the constants cleared: the equations are polynomials
        abstracted = abstracted.as_numer_denom()[0]
    restored: dict[Basic, Basic] = {Symbol(name): p for p, name in zip(params, p_names)}
    if number_field is not None:
        restored[number_field.theta] = number_field.theta_expr
    system = _LinearSystem(abstracted, v_names, c_names, p_names)
    found: Optional[dict[int, Expr]]
    if number_field is not None:
        G = number_field.field
        if not params:
            found = system.solve(G, G, system.constant(G))
        else:
            R = SparsePolynomialRing(G, p_names)
            K = SparseFractionField(R)
            found = system.solve(G, K, lambda terms: K.from_ring(R.ring.from_dict(terms)))
    elif not params:
        found = system.solve(QQ, QQ, system.constant(QQ))
    else:
        Q = RationalPolynomialRing(p_names)
        F = SparseFractionField(Q)
        found = system.solve(QQ, F, lambda terms: F.from_ring(Q.ring.from_dict(terms)))
    if found is None:
        return None
    return {coefficients[column]: placeholders.restore(_replaced(value, restored)) for column, value in found.items()}


class _LinearSystem:
    """The linear equations in the unknowns ``c_names`` given by the
    coefficients of the monomials in ``v_names`` of a polynomial in
    ``v_names``, ``c_names`` and the parameters ``p_names``."""

    def __init__(self, polynomial: Expr, v_names: list[str], c_names: list[str], p_names: list[str]) -> None:
        self.polynomial: Final[Expr] = polynomial
        self.v_names: Final[list[str]] = v_names
        self.c_names: Final[list[str]] = c_names
        self.p_names: Final[list[str]] = p_names

    @staticmethod
    def constant(G: Domain[T]) -> Callable[[dict[Monomial, T]], T]:
        """The lift of the coefficients when there is no parameter."""
        return lambda terms: terms.get((), G.zero)

    def solve(self, G: Domain[T], K: Domain[U], lift: Callable[[dict[Monomial, T]], U]) -> Optional[dict[int, Expr]]:
        """The values of the pivot unknowns (by their index) with the free
        ones zero, over the field ``K`` of the coefficients, which ``lift``
        makes from their terms in the parameters over ``G``; ``None`` when
        the system is inconsistent."""
        n, m = len(self.v_names), len(self.c_names)
        terms = dict_from_expr(self.polynomial, self.v_names + self.c_names + self.p_names, G)
        grouped: dict[Monomial, dict[int, dict[Monomial, T]]] = {}
        for monomial, value in terms.items():
            unknown = monomial[n:n + m]
            degree = sum(unknown)
            if degree > 1:
                raise ValueError('the system is not linear')
            column = unknown.index(1) if degree == 1 else m
            grouped.setdefault(monomial[:n], {}).setdefault(column, {})[monomial[n + m:]] = value
        rows: list[dict[int, U]] = []
        for entries in grouped.values():
            row = {column: lift(parts) for column, parts in entries.items()}
            rows.append({column: value for column, value in row.items() if not K.is_zero(value)})
        solution = _echelon_solution([row for row in rows if row], K, m)
        if solution is None:
            return None
        return {column: K.to_expr(value) for column, value in solution.items()}


# PORTED FROM SYMPY: sympy/functions/elementary/trigonometric.py:sin._eval_rewrite_as_tan
def _tangent_rule(node: Basic) -> Basic:
    """One node written in tangents of the half angle (SymPy's
    ``_eval_rewrite_as_tan`` and ``_eval_rewrite_as_tanh`` of ``sin``,
    ``cos``, ``cot``, ``sinh``, ``cosh``, ``coth``)."""
    if not isinstance(node, (Sin, Cos, Cot, Sinh, Cosh, Coth)):
        return node
    argument = as_expr(node.args[0])
    if isinstance(node, Sin):
        half = tan(argument / 2)
        return 2 * half / (1 + half**2)
    if isinstance(node, Cos):
        half = tan(argument / 2)**2
        return (1 - half) / (1 + half)
    if isinstance(node, Cot):
        return 1 / tan(argument)
    if isinstance(node, Sinh):
        half = tanh(argument / 2)
        return 2 * half / (1 - half**2)
    if isinstance(node, Cosh):
        half = tanh(argument / 2)**2
        return (1 + half) / (1 - half)
    return 1 / tanh(argument)


def _half_angle_tangents(f: Expr) -> Expr:
    """``f.rewrite((sin, cos, cot), tan).rewrite((sinh, cosh, coth), tanh)``
    of SymPy: the circular and hyperbolic functions in tangents of the
    half angle, bottom up."""
    if not f.args:
        return f
    rebuilt = as_expr(f._evaluated(tuple(_half_angle_tangents(as_expr(a)) if isinstance(a, Expr) else a
                                         for a in f.args)))
    return as_expr(_tangent_rule(rebuilt))


def _heurisch(f: Expr, x: Symbol, rewrite: bool, hints: list[Expr],
              mappings: Optional[Iterator[list[Pair]]], retries: int, degree_offset: int,
              unnecessary_permutations: Optional[list[Pair]]) -> Optional[Expr]:
    """The method proper: ``rewrite`` writes the trigonometric and
    hyperbolic functions in tangents, ``mappings`` iterates the
    permutations of the components."""
    if not f.has_free(x):
        return f * x
    # exponentials of sums as products of exponentials of their terms, so that
    # exp(-(x + exp(x))**2 + x) and exp(-(x + exp(x))**2) share their components
    # (the derivatives are written the same way by the cache)
    f = expand_power_exp(f)
    indep: Expr = S.One
    if not isinstance(f, Add):
        indep, f = f.as_independent(x)
    rewritables: tuple[tuple[type[Basic], ...], ...] = ((Sin, Cos, Cot), (Sinh, Cosh, Coth))
    if rewrite:
        f = _half_angle_tangents(f)
    else:
        for candidates in rewritables:
            if f.has(*candidates):
                break
        else:
            rewrite = True
    terms = components(f, x)
    cache = _DiffCache(x)
    terms |= _special_candidates(terms, x, cache, f)
    terms |= _tower_candidates(terms, x, cache, f)
    terms |= set(hints)
    for g in list(terms):
        terms |= components(cache.get(g), x)
    terms = _exponential_bases(terms)
    ordered_terms = _sorted(terms)
    V = _symbols('x', len(ordered_terms))
    # the components from the largest to the smallest, x last
    keyed = sorted(zip(ordered_terms, V), key=lambda pair: (_ops(pair[0].as_independent(x)[1]), str(pair[0])),
                   reverse=True)
    mapping: list[Pair] = list(keyed)
    rev_mapping: dict[Basic, Basic] = {v: t for t, v in mapping}
    if mappings is None:
        # x is the smallest component: the last one, never permuted
        position = next(i for i, (t, _) in enumerate(mapping) if t == x)
        unnecessary_permutations = [mapping.pop(position)]
        types: dict[type[Basic], list[Pair]] = defaultdict(list)
        for pair in mapping:
            types[type(pair[0])].append(pair)
        groups = [types[k] for k in types]

        def _iter_mappings() -> Iterator[list[Pair]]:
            for permutation in permutations(groups):
                yield [pair for group in permutation for pair in _ordered_pairs(group)]

        mappings = _iter_mappings()
    else:
        unnecessary_permutations = unnecessary_permutations or []

    current: list[Pair] = []

    def _substitute(expr: Expr) -> Expr:
        # from the largest component to the smallest, whatever the order of
        # the mapping: substituting exp(x) first turns the exp(2*x) inside
        # exp(exp(2*x)) into a power of the symbol, and the outer component
        # is never found again (the mapping's order labels the symbols, the
        # permutations being over the variable order of the ring)
        largest_first = sorted(current, key=lambda pair: (-_ops(pair[0]), str(pair[0])))
        return expr.subs(largest_first)

    diffs: list[Expr] = []
    denom: Optional[Expr] = None
    for candidate_mapping in mappings:
        current = list(candidate_mapping) + unnecessary_permutations
        diffs = [_substitute(cache.get(g)) for g in ordered_terms]
        denoms = [g.as_numer_denom()[1] for g in diffs]
        if all(h.is_polynomial(*V) is True for h in denoms) and _substitute(f).is_rational_function(*V) is True:
            try:
                common = denoms[0]
                for q in denoms[1:]:
                    common = _lcm(common, q, V)
            except _GIVE_UP:
                return None
            denom = common
            break
    if denom is None:
        if not rewrite:
            result = _heurisch(f, x, True, hints, None, retries, degree_offset, unnecessary_permutations)
            if result is not None:
                return indep * result
        return None
    try:
        found = _solve(f, x, denom, diffs, V, ordered_terms, _substitute, degree_offset)
    except _GIVE_UP:
        found = None
    if found is not None:
        antiderivative = expand(cancel(_replaced(found, rev_mapping)))
        if isinstance(antiderivative, Add):
            antiderivative = antiderivative.as_independent(x)[1]
        return indep * antiderivative
    if retries >= 0:
        result = _heurisch(f, x, rewrite, hints, mappings, retries - 1, degree_offset, unnecessary_permutations)
        if result is not None:
            return indep * result
    return None


def _ordered_pairs(group: list[Pair]) -> list[Pair]:
    """SymPy's ``ordered`` of the pairs (component, symbol)."""
    return sorted(group, key=lambda pair: (_size(pair[0]), pair[0].sort_key(), pair[1].sort_key()))


def _size(e: Expr) -> int:
    """The number of nodes of ``e`` (the first key of SymPy's ``ordered``)."""
    return sum(1 for _ in _preorder(e))


def _preorder(e: Basic) -> Iterator[Basic]:
    yield e
    for a in e.args:
        yield from _preorder(a)


def _solve(f: Expr, x: Symbol, denom: Expr, diffs: list[Expr], V: list[Symbol], ordered_terms: list[Expr],
           substitute: Callable[[Expr], Expr], degree_offset: int) -> Optional[Expr]:
    """The candidate antiderivative in the symbols ``V`` of the components
    with its coefficients solved for, ``None`` when no candidate fits."""
    numers = [cancel(denom * g) for g in diffs]

    def _derivation(h: Expr) -> Expr:
        return add(*[d * diff(h, v) for d, v in zip(numers, V)])

    def _deflation(p: Expr) -> Expr:
        for y in V:
            if not p.has(y):
                continue
            if _derivation(p) != S.Zero:
                c, q = _primitive(p, y)
                return _deflation(c) * _gcd(q, diff(q, y), [y])
        return p

    def _splitter(p: Expr) -> tuple[Expr, Expr]:
        for y in V:
            if not p.has(y):
                continue
            if _derivation(y) != S.Zero:
                c, q = _primitive(p, y)
                h = _gcd(q, _derivation(q), [y])
                s = cancel(h / _gcd(q, diff(q, y), [y]))
                c_split = _splitter(c)
                if _degree(s, y) == 0:
                    return (c_split[0], q * c_split[1])
                q_split = _splitter(cancel(q / s))
                return (c_split[0] * q_split[0] * s, c_split[1] * q_split[1])
        return (S.One, p)

    special: dict[Expr, bool] = {}
    for term in ordered_terms:
        if isinstance(term, Tan):
            special[1 + substitute(term)**2] = False
        elif isinstance(term, Tanh):
            special[1 + substitute(term)] = False
            special[1 - substitute(term)] = False
        elif isinstance(term, LambertW):
            special[substitute(term)] = True
    F = substitute(f)
    P, Q = F.as_numer_denom()
    u_split = _splitter(denom)
    v_split = _splitter(Q)
    polys = set(list(v_split) + [u_split[0]] + list(special))
    s = u_split[0] * mul(*[k for k, v in special.items() if v])
    degrees = [_total_degree(p, V) for p in (s, P, Q)]
    a, b, c = degrees
    if a is None or b is None or c is None:
        return None
    poly_denom = s * v_split[0] * _deflation(v_split[1])
    A, B = _exponent(f), a + max(b, c)
    bound = A + B - 1 + degree_offset if A > 1 and B > 1 else A + B + degree_offset
    monoms = _itermonomials(V, bound)
    poly_coeffs = _symbols('A', len(monoms))
    poly_part = add(*[coefficient * monomial for coefficient, monomial in zip(poly_coeffs, monoms)])
    reducibles: set[Expr] = set()
    for poly in _sorted(polys):
        coefficient, factors = _factor_list(poly, V)
        reducibles.add(coefficient)
        reducibles.update(factor for factor, _ in factors)

    def _integrate(field: Optional[str] = None) -> Optional[Expr]:
        atans: set[Expr] = set()
        pairs: set[tuple[Expr, Expr]] = set()
        irreducibles: set[Expr]
        if field == 'Q':
            irreducibles = set(reducibles)
        else:
            irreducibles = set()
            for poly in _sorted(reducibles):
                zV = set(V) & set(iterfreeargs(poly))
                for z in _sorted(e for e in zV if isinstance(e, Expr)):
                    irreducibles |= set(_root_factors(poly, z, field))
                    break
        log_part: list[Expr] = []
        atan_part: list[Expr] = []
        for poly in _sorted(list(irreducibles)):
            parts = collect_dict(poly, I)
            y = parts.get(I, S.Zero)
            if y != S.Zero:
                x_ = parts.get(S.One, S.Zero)
                if x_.has(I) or y.has(I):
                    continue
                pairs.add((x_, y))
                irreducibles.remove(poly)
        while pairs:
            x_, y = pairs.pop()
            if (x_, -y) in pairs:
                pairs.remove((x_, -y))
                if y.could_extract_minus_sign():
                    y = -y
                irreducibles.add(x_ * x_ + y * y)
                atans.add(atan(x_ / y))
            else:
                irreducibles.add(x_ + I * y)
        coefficients = list(poly_coeffs)
        B_ = _symbols('B', len(irreducibles))
        C_ = _symbols('C', len(atans))
        for poly, coefficient in reversed(list(zip(_sorted(irreducibles), B_))):
            if poly.has(*V):
                coefficients.append(coefficient)
                log_part.append(coefficient * log(poly))
        for poly, coefficient in reversed(list(zip(_sorted(atans), C_))):
            if poly.has(*V):
                coefficients.append(coefficient)
                atan_part.append(coefficient * poly)
        candidate = poly_part / poly_denom + add(*log_part) + add(*atan_part)
        h = F - _derivation(candidate) / denom
        raw_numer = h.as_numer_denom()[0]
        syms: set[Expr] = set(coefficients) | set(V)
        non_syms: set[Expr] = set()
        _find_non_syms(raw_numer, syms, non_syms)
        solution = attempt(lambda: _solve_coefficients(raw_numer, V, coefficients, non_syms), settings.timeout,
                           native=True)
        if solution is None:
            return None
        zeros: dict[Basic, Basic] = {coefficient: S.Zero for coefficient in coefficients}
        return _replaced(_replaced(candidate, solution), zeros)

    more_free = {s_ for s_ in F.free_symbols if isinstance(s_, Symbol)} - set(V)
    solution: Optional[Expr]
    if not more_free:
        solution = _integrate('Q')
        if solution is None:
            solution = _integrate()
    else:
        solution = _integrate()
    return solution


# PORTED FROM SYMPY: sympy/solvers/solvers.py:denoms
def _denominators(e: Expr) -> list[Expr]:
    """The factors of the denominators of the subexpressions of ``e``
    (SymPy's ``denoms``)."""
    found: list[Expr] = []
    for node in _preorder(e):
        if not isinstance(node, Expr):
            continue
        d = node.as_numer_denom()[1]
        if d == S.One:
            continue
        for factor in Mul.make_args(d):
            if factor not in found:
                found.append(factor)
    return found


def _consistent(symbol: Symbol, value: Expr) -> bool:
    """Whether ``value`` may stand for ``symbol``: no known fact of the
    value contradicts an assumption of the symbol (SymPy's ``solve``
    checks them)."""
    facts = value.assumptions0
    return all(facts.get(key, flag) == flag for key, flag in symbol.assumptions0.items() if key != 'commutative')


def _solutions(d: Expr, exclude: Symbol) -> list[dict[Symbol, Expr]]:
    """The solutions of ``d = 0`` for its symbols other than ``exclude``,
    as SymPy's ``solve([d], dict=True, exclude=(x,))`` finds them for a
    single equation: each symbol in turn in which ``d`` is linear, then
    the roots in the others; a solution involving a symbol solved for
    before is skipped, as are those contradicting the assumptions."""
    symbols = sorted((s for s in d.free_symbols if isinstance(s, Symbol) and s != exclude), key=lambda s: s.sort_key())
    result: list[dict[Symbol, Expr]] = []
    got: set[Symbol] = set()
    nonlinear: list[Symbol] = []
    for s in symbols:
        coefficients = _power_coefficients(d, s)
        if coefficients is not None and max(coefficients, default=0) == 1 and 1 in coefficients:
            value = cancel(-coefficients.get(0, S.Zero) / coefficients[1])
            if any(t in got for t in value.free_symbols):
                continue
            if _consistent(s, value):
                got.add(s)
                result.append({s: value})
        elif coefficients is not None and max(coefficients, default=0) > 1:
            nonlinear.append(s)
    for s in nonlinear:
        found = _polynomial_roots(d, s)
        for value in found:
            if any(t in got for t in value.free_symbols) or not _consistent(s, value):
                continue
            result.append({s: value})
        if found:
            got.add(s)
    return result


def _polynomial_roots(d: Expr, s: Symbol) -> list[Expr]:
    """The roots of the polynomial ``d`` in ``s`` found in radicals."""
    placeholders, (a,) = _with_constants([s], [d])
    try:
        found = roots(Poly(a, s))
    except _GIVE_UP:
        return []
    if not isinstance(found, dict):
        return []
    return [placeholders.restore(r) for r in _sorted(found)]


def heurisch_cases(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """The antiderivative of ``f`` with the cases of the parameters where
    a denominator of the generic antiderivative vanishes: ``cos(n*x)``
    integrates to ``sin(n*x)/n`` for ``n != 0`` and to ``x`` at ``n = 0``.
    A ``Piecewise`` over the cases, the generic one first, or the plain
    antiderivative when there is no such case; ``None`` when the generic
    case is not found.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cos
    >>> from sympy_extras.integrals.heurisch import heurisch_cases
    >>> n, x = symbols('n x')
    >>> heurisch_cases(cos(n*x), x)
    Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))
    """
    f_ = to_expr(f)
    if not f_.has_free(x):
        return f_ * x
    generic = heurisch_antiderivative(f_, x)
    if generic is None:
        return None
    cases: list[dict[Symbol, Expr]] = []
    for d in _sorted(_denominators(generic)):
        found = attempt(lambda: _solutions(d, x), settings.timeout, native=True)
        if found:
            cases.extend(found)
    if not cases:
        return generic
    cases = _unique(cases)
    original: list[dict[Symbol, Expr]] = []
    for d in _denominators(f_):
        found = attempt(lambda: _solutions(d, x), settings.timeout, native=True)
        if found:
            original.extend(found)
    cases = [case for case in cases if case not in original]
    if not cases:
        return generic
    if len(cases) > 1:
        # (SymPy's joint solution of the equations of all the cases is the
        # union of the single ones here: each case is one equation in one
        # symbol, and the joint system of several symbols is solved only
        # when they are independent)
        joint = _joint(cases)
        cases = joint + cases
    pairs: list[tuple[Expr, Boolean]] = []
    for case in cases:
        special = heurisch_antiderivative(_replaced(f_, {key: value for key, value in case.items()}), x)
        if special is None:
            return None
        condition = and_(*[eq(key, value) for key, value in case.items()])
        pairs.append((special, condition))
    if len(pairs) == 1:
        generic_condition = or_(*[ne(key, value) for key, value in cases[0].items()])
        return piecewise((generic, generic_condition), (pairs[0][0], true))
    pairs.append((generic, true))
    return piecewise(*pairs)


def _unique(cases: list[dict[Symbol, Expr]]) -> list[dict[Symbol, Expr]]:
    result: list[dict[Symbol, Expr]] = []
    for case in cases:
        if case not in result:
            result.append(case)
    return result


def _joint(cases: list[dict[Symbol, Expr]]) -> list[dict[Symbol, Expr]]:
    """SymPy's ``solve([Eq(s, v) for every case], dict=True)`` for cases of
    distinct symbols whose values are free of the others: their union;
    ``[]`` otherwise (two values for one symbol are inconsistent)."""
    joint: dict[Symbol, Expr] = {}
    for case in cases:
        for key, value in case.items():
            if key in joint or any(isinstance(s, Symbol) and s in joint for s in value.free_symbols):
                return []
            joint[key] = value
    if any(isinstance(s, Symbol) and s in joint for value in joint.values() for s in value.free_symbols):
        return []
    return [joint]

