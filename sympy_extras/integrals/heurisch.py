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
see ``risch/LICENSE-SymPy`` in this package), strictly typed, with:

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

Examples
========

>>> from sympy import symbols, exp, sin, cos, sqrt, log, erf
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
from __future__ import annotations

import math

from collections import defaultdict
from functools import reduce
from itertools import permutations
from typing import Iterator, Optional, Sequence

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.simplify.powsimp import powsimp
from sympy.core.function import expand, expand_mul, expand_power_exp
from sympy.functions.special.zeta_functions import polylog
from sympy.core.function import Derivative, Function
from sympy.core.mul import Mul
from sympy.core.numbers import Float, I, Rational, pi
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.traversal import iterfreeargs
from sympy.functions.elementary.complexes import Abs, arg, im, re, sign
from sympy.functions.elementary.exponential import LambertW, exp, log
from sympy.functions.elementary.hyperbolic import asinh, cosh, coth, sinh, tanh
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin, atan, cos, cot, sin, tan
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, hankel1, hankel2, jn, yn)
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import Ci, Ei, Si, erf, erfi, li
from sympy.logic.boolalg import And, Boolean, Or
from sympy.polys.constructor import construct_domain
from sympy.polys.monomials import itermonomials
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import root_factors
from sympy.polys.polytools import Poly, cancel, factor_list, gcd, lcm, quo
from sympy.polys.rings import PolyRing
from sympy.polys.solvers import solve_lin_sys
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.utilities.iterables import uniq

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_boolean, as_expr, free_symbols
from sympy_extras.settings import settings
from .conditions import numerically_equal

__all__ = ['components', 'heurisch_antiderivative', 'heurisch_cases']

#: the functions the method does not take: not elementary in the sense of
#: the derivation (a jump, a sign)
_REFUSED = (Abs, re, im, sign, Heaviside, DiracDelta, floor, ceiling, arg)


def components(f: ExprLike, x: Symbol) -> set[Expr]:
    """The functional components of ``f`` in ``x``: the symbol, the
    function applications and compositions, and the non-integer powers
    (fractional powers with the least positive exponent).

    >>> from sympy import symbols, sin, cos, sqrt
    >>> from sympy_extras.integrals.heurisch import components
    >>> x = symbols('x')
    >>> sorted(components(sin(x)*cos(x)**2, x), key=str)
    [cos(x), sin(x), x]
    >>> sorted(components(sqrt(x + 1)**3, x), key=str)
    [sqrt(x + 1), x]
    """
    f_ = as_expr(f)
    result: set[Expr] = set()
    if not f_.has_free(x):
        return result
    if isinstance(f_, Symbol):
        result.add(f_)
    elif isinstance(f_, (Function, Derivative)):
        for g in f_.args:
            result |= components(as_expr(g), x)
        result.add(f_)
    elif isinstance(f_, Pow):
        base, exponent = as_expr(f_.base), as_expr(f_.exp)
        result |= components(base, x)
        if not exponent.is_Integer:
            if isinstance(exponent, Rational):
                result.add(as_expr(base**Rational(1, exponent.q)))
            else:
                result |= components(exponent, x) | {f_}
    else:
        for g in f_.args:
            result |= components(as_expr(g), x)
    return result


_symbols_cache: dict[str, list[Symbol]] = {}


def _symbols(name: str, n: int) -> list[Symbol]:
    """``n`` symbols ``name0, name1, ...`` local to this module, the same
    on every call: the retries and the rewriting recurse with the mapping
    of the components to these symbols, which must be the same symbols
    (the bug: fresh dummies on each call left ``x/_x1`` as the result,
    the mapping of the inner call being unknown to the outer one)."""
    known = _symbols_cache.setdefault(name, [])
    while len(known) < n:
        known.append(Dummy('%s%i' % (name, len(known))))
    return known[:n]


class _BesselTable:
    """The derivatives of the Bessel functions of orders ``n`` and ``n - 1``
    in terms of each other: the standard derivative brings in the orders
    ``n - 1`` and ``n + 1``, three functions linked by a recurrence, while
    the method needs algebraically independent components."""

    def __init__(self) -> None:
        self.n = Dummy('n')
        self.z = Dummy('z')
        n, z = self.n, self.z
        self.table: dict[type[Function], tuple[Expr, Expr]] = {}
        for f in (besselj, bessely, hankel1, hankel2):
            self.table[f] = (as_expr(f(n - 1, z) - n * f(n, z) / z), as_expr((n - 1) * f(n - 1, z) / z - f(n, z)))
        self.table[besseli] = (as_expr(besseli(n - 1, z) - n * besseli(n, z) / z),
                               as_expr((n - 1) * besseli(n - 1, z) / z + besseli(n, z)))
        self.table[besselk] = (as_expr(-besselk(n - 1, z) - n * besselk(n, z) / z),
                               as_expr((n - 1) * besselk(n - 1, z) / z - besselk(n, z)))
        for f in (jn, yn):
            self.table[f] = (as_expr(f(n - 1, z) - (n + 1) * f(n, z) / z), as_expr((n - 1) * f(n - 1, z) / z - f(n, z)))

    def diffs(self, f: type[Function], n: Expr, z: Expr) -> tuple[Expr, Expr]:
        d0, d1 = self.table[f]
        replacement = {self.n: n, self.z: z}
        return (as_expr(d0.xreplace(replacement)), as_expr(d1.xreplace(replacement)))


_BESSEL = _BesselTable()


class _DiffCache:
    """The derivatives of the components with respect to ``x``, the Bessel
    functions through the table so that the pair of orders ``n, n - 1``
    is used throughout."""

    def __init__(self, x: Symbol) -> None:
        self.cache: dict[Expr, Expr] = {}
        self.x = x

    def get(self, f: Expr) -> Expr:
        if f in self.cache:
            return self.cache[f]
        if isinstance(f, Function) and type(f) in _BESSEL.table and len(f.args) == 2:
            n, z = as_expr(f.args[0]), as_expr(f.args[1])
            d0, d1 = _BESSEL.diffs(type(f), n, z)
            dz = self.get(z)
            self.cache[f] = as_expr(d0 * dz)
            self.cache[as_expr(type(f)(n - 1, z))] = as_expr(d1 * dz)
        else:
            # exponentials of sums as products of exponentials of their terms,
            # the canonical form of the integrand (see _heurisch)
            self.cache[f] = as_expr(expand_power_exp(cancel(f.diff(self.x))))
        return self.cache[f]


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
    denominator = as_expr(f.as_numer_denom()[1])
    over_x = denominator.is_polynomial(x) and denominator.subs(x, 0) == 0
    with_erf = f.has(erf, erfi)
    for g in list(terms):
        if isinstance(g, li):
            match = as_expr(g.args[0]).match(a * x**b)
            if match is not None:
                A, B = as_expr(match[a]), as_expr(match[b])
                found.add(as_expr(x * (li(A * x**B) - (A * x**B)**(-1 / B) * Ei((B + 1) * log(A * x**B) / B))))
        elif isinstance(g, exp):
            argument = as_expr(g.args[0])
            # (the error functions come from the tower, _tower_candidates, with
            # their derivatives registered in terms of the components: the
            # derivative SymPy takes of erf(x - 1) has exp(-(x - 1)**2), a
            # component the ring cannot relate to exp(-x**2 + 2*x))
            match = argument.match(a * x**2)
            if match is not None and with_erf and as_expr(match[a]).is_negative:
                # the product of two such exponentials
                found.add(as_expr(erf(sqrt(2) * sqrt(-as_expr(match[a])) * x)))
            match = argument.match(a * x**b)
            if over_x and match is not None and match[a] != 0 and as_expr(match[b]).is_nonzero:
                found.add(as_expr(Ei(as_expr(match[a]) * x**as_expr(match[b]))))
        elif isinstance(g, (sin, cos)) and over_x:
            match = as_expr(g.args[0]).match(a * x**b)
            if match is not None and match[a] != 0 and as_expr(match[b]).is_nonzero:
                argument = as_expr(match[a] * x**match[b])
                found.add(as_expr(Si(argument)))
                found.add(as_expr(Ci(argument)))
        elif isinstance(g, Pow) and isinstance(g.exp, Rational) and g.exp.q == 2:
            base = as_expr(g.base)
            match = base.match(a * x**2 + b)
            if match is not None and as_expr(match[b]).is_positive:
                A, B = as_expr(match[a]), as_expr(match[b])
                if A.is_positive:
                    found.add(as_expr(asinh(sqrt(A / B) * x)))
                elif A.is_negative:
                    found.add(as_expr(asin(sqrt(-A / B) * x)))
            match = base.match(a * x**2 - b)
            if match is not None and as_expr(match[b]).is_positive:
                A, B = as_expr(match[a]), as_expr(match[b])
                if A.is_positive:
                    primitive = as_expr(log(2 * sqrt(A) * sqrt(A * x**2 - B) + 2 * A * x) / sqrt(A))
                    cache.cache[primitive] = as_expr(1 / sqrt(A * x**2 - B))
                    found.add(primitive)
                elif A.is_negative:
                    found.add(as_expr(-B / 2 * sqrt(-A) * atan(sqrt(-A) * x / sqrt(A * x**2 - B))))
            # (the primitives of 1/(x sqrt(Q)), log((sqrt(b) + sqrt(Q))/x) and
            # atan(sqrt(Q)/sqrt(b)), are of no use here: their derivatives
            # match the integrand only modulo sqrt(Q)**2 = Q, while the
            # derivation takes sqrt(Q) as an independent component; these
            # integrands are for the radical table and Trager's algorithm)
            match = base.match(a * x**2 + b * x + c)
            if match is not None and match[a] != 0 and match[b] != 0:
                # a completed square: the primitives of 1/sqrt(Q) with Q = a x**2 + b x + c
                A, B, C = as_expr(match[a]), as_expr(match[b]), as_expr(match[c])
                discriminant = as_expr(B**2 - 4 * A * C)
                if A.is_negative and discriminant.is_positive:
                    found.add(as_expr(asin((2 * A * x + B) / sqrt(discriminant))))
                elif A.is_positive and discriminant.is_negative:
                    found.add(as_expr(asinh((2 * A * x + B) / sqrt(-discriminant))))
                elif A.is_positive and discriminant.is_positive:
                    primitive = as_expr(log(2 * sqrt(A) * sqrt(base) + 2 * A * x + B) / sqrt(A))
                    cache.cache[primitive] = as_expr(1 / sqrt(base))
                    found.add(primitive)
    return found


def _denominator(q: Expr) -> int:
    """The denominator of a rational number, 1 for anything else."""
    return int(q.q) if isinstance(q, Rational) else 1


def _exponential_family(g: Expr) -> tuple[Expr, Rational]:
    """``(w, q)`` with ``g == exp(q*w)``, ``w`` the argument without its
    rational content and sign (``exp(2*x)`` and ``exp(-x)`` belong to the
    family of ``x``, with ``q = 2`` and ``q = -1``)."""
    content, primitive = as_expr(g.args[0]).as_content_primitive()
    w, q = as_expr(primitive), as_expr(content)
    if w.could_extract_minus_sign():
        w, q = as_expr(-w), as_expr(-q)
    return (w, q if isinstance(q, Rational) else S.One)


def _exponential_bases(terms: set[Expr]) -> set[Expr]:
    """The terms with each family of rationally related exponentials
    represented by one base, ``exp(w/d)`` for the least common denominator
    ``d`` of their multiples, of which the others are integer powers: the
    derivative of ``exp(-exp(2*x))`` brings in ``exp(2*x)``, which the
    method must know is ``exp(x)**2`` (``subs`` writes it so once the base
    is the component)."""
    families: dict[Expr, list[tuple[Expr, Rational]]] = {}
    for g in terms:
        if isinstance(g, exp):
            w, q = _exponential_family(g)
            families.setdefault(w, []).append((g, q))
    result = {g for g in terms if not isinstance(g, exp)}
    for w, members in families.items():
        d = 1
        for _, q in members:
            d = math.lcm(d, _denominator(q))
        result.add(as_expr(exp(w / d)))
    return result


def _degree_in(e: Expr, g: Expr) -> int:
    """The degree of ``e`` in the component ``g`` (``exp(2*x)`` counts as
    ``exp(x)**2``), 0 when ``e`` is not polynomial in it."""
    d = Dummy('g')
    replaced = as_expr(powsimp(e, combine='exp').replace(lambda n: n == g, lambda n: d))
    try:
        return int(Poly(replaced, d).degree())
    except (PolynomialError, ValueError, TypeError):
        return 0


def _square_root_form(e: Expr, x: Symbol, terms: set[Expr]) -> Optional[tuple[Expr, Expr]]:
    """``(u, c)`` with ``e == u**2 + c``, ``u`` built from the components of
    ``e`` and ``c`` a constant or an exponent whose exponential is built
    from the components in ``terms``: ``e`` a perfect square times a
    positive constant (``(x + exp(x))**2``), or quadratic in one component
    with a constant leading coefficient, the square completed
    (``-x**2 - 2*x*exp(x) - exp(2*x) + 5``, and ``(x + exp(x))**2 - x``
    with ``exp(-x)`` among the components); ``None`` otherwise."""
    expanded = as_expr(expand(e))
    # the exponentials as powers of one dummy each: exp(2*x) is exp(x)**2 (subs sees it)
    # each family of exponentials as powers of one dummy standing for the
    # square root of its base, exp(w/(2*d)): exp(2*x) is then d**4 and a square
    dummies: dict[Expr, Symbol] = {}
    families: dict[Expr, int] = {}
    for g in components(expanded, x):
        if isinstance(g, exp):
            w, q = _exponential_family(g)
            families[w] = math.lcm(families.get(w, 1), _denominator(q))
    for w, d in families.items():
        dummies[as_expr(exp(w / (2 * d)))] = Dummy('e')
    reduced = expanded
    for g, dummy in dummies.items():
        reduced = as_expr(reduced.subs(g, dummy))
    back = {dummy: g for g, dummy in dummies.items()}
    gens = sorted((g for g in components(reduced, x) if not isinstance(g, exp)), key=str) + list(dummies.values())
    if not gens:
        return None
    try:
        constant, factors = factor_list(reduced, *gens)
    except (PolynomialError, ValueError, TypeError):
        return None
    if factors and all(k % 2 == 0 for _, k in factors):
        c0 = as_expr(constant)
        if c0.is_positive:
            u: Expr = sqrt(c0)
            for base, k in factors:
                u = u * as_expr(base)**(k // 2)
            return (as_expr(u.xreplace(back)), S.Zero)
    for v in gens:
        try:
            poly = Poly(reduced, v)
        except (PolynomialError, ValueError, TypeError):
            continue
        if poly.degree() != 2:
            continue
        A, B = as_expr(poly.coeff_monomial(v**2)).xreplace(back), as_expr(poly.coeff_monomial(v)).xreplace(back)
        v = as_expr(back[v]) if isinstance(v, Symbol) and v in back else v
        if A.has(x) or not A.is_positive:
            continue
        # the square completed with the whole linear coefficient, or with its
        # part in the other generators only: (x + exp(x))**2 - x has the
        # coefficient 2*exp(x) - 1, and the remainder -x is an offset whose
        # exponential exp(-x) is in the field
        number, rest = B.as_coeff_Add()
        for linear in ([B] if number == 0 or rest == 0 else [B, as_expr(rest)]):
            u = as_expr(sqrt(A) * (v + linear / (2 * A)))
            c = as_expr(cancel(expand(e) - expand(u**2)))
            if c.has(x) and not (components(as_expr(exp(c)), x) - {x} <= terms
                                 or components(as_expr(exp(-c)), x) - {x} <= terms):
                continue
            return (u, c)
    return None


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
    denominator = as_expr(f.as_numer_denom()[1])
    try:
        factors = [as_expr(q) for q, _ in factor_list(denominator)[1]]
    except (PolynomialError, ValueError, TypeError):
        factors = [denominator]
    logarithms = {as_expr(g.args[0]) for g in f.atoms(log)}
    for g in list(terms):
        if isinstance(g, exp):
            theta = as_expr(g.args[0])
            exponential: Expr = g
        elif isinstance(g, log):
            theta = g
            exponential = as_expr(g.args[0])                # exp(log(h)) is h
        else:
            continue
        dtheta = as_expr(cancel(theta.diff(x)))
        if dtheta == 0:
            continue
        numerator = as_expr(f.as_numer_denom()[0])
        # the multiples k the degrees allow: up to the degree of the numerator in
        # the exponential, one more for a logarithm (x/(log(x) + 1) is
        # exp(2*theta)*theta'/(theta + 1) with theta = log(x)), and negative ones
        # only where the exponential divides the denominator
        above = _degree_in(numerator, exponential) + (0 if isinstance(g, exp) else 1)
        below = _degree_in(denominator, exponential)
        multiples = list(range(1, max(above, 1) + 1)) + list(range(-1, -below - 1, -1))
        for q in factors:
            if not q.has(x):
                continue
            alpha = as_expr(cancel(q.diff(x) / dtheta))
            if alpha == 0 or alpha.has(x):
                continue
            beta = as_expr(cancel(q - alpha * theta))
            if beta.has(x):
                continue
            c = as_expr(beta / alpha)
            # exp(k*theta)*theta'/(theta + c) is the derivative of Ei(k*(theta + c))*exp(-k*c):
            # the multiples k up to the degree of the integrand in the exponential
            for k in multiples:
                candidate = as_expr(Ei(k * (theta + c)))
                cache.cache[candidate] = as_expr(exp(k * c) * exponential**k * dtheta / (theta + c))
                found.add(candidate)
        if not isinstance(g, exp):
            continue
        for offset, argument in ((1, -g), (-1, g)):
            if any(as_expr(cancel(q - offset - g)) == 0 or as_expr(cancel(q + offset + g)) == 0 for q in factors) \
                    or any(as_expr(cancel(h - offset - g)) == 0 or as_expr(cancel(h + offset + g)) == 0 for h in logarithms):
                candidate = as_expr(polylog(2, argument))
                cache.cache[candidate] = as_expr(-log(1 - argument) * dtheta)
                found.add(candidate)
    # the error functions from the total exponent of each term of f, the sum of
    # the arguments of its exponential factors (exp(-x**2)*exp(2*x) in the
    # canonical form of the integrand is exp(-(x - 1)**2 + 1))
    exponents: list[Expr] = []
    for term in Add.make_args(f):
        theta_term: Expr = S.Zero
        for factor in Mul.make_args(term):
            piece = as_expr(factor)
            if isinstance(piece, exp):
                theta_term = theta_term + as_expr(piece.args[0])
                exponents.append(as_expr(piece.args[0]))     # a single factor too: exp(x)*exp(-exp(2*x)) needs erf(exp(x))
            elif isinstance(piece, Pow) and isinstance(piece.base, exp) and not as_expr(piece.exp).has(x):
                theta_term = theta_term + as_expr(piece.exp) * as_expr(piece.base.args[0])
        if theta_term != 0:
            exponents.append(theta_term)
    for theta_term in exponents:
        for sign_, function in ((-1, erf), (1, erfi)):
            form = _square_root_form(as_expr(sign_ * theta_term), x, terms)
            if form is None:
                continue
            u, c = form
            candidate = as_expr(function(u))
            # erf(u)' = 2/sqrt(pi) u' exp(-u**2) with -u**2 = theta + c, erfi(u)' = 2/sqrt(pi) u' exp(u**2)
            # with u**2 = theta - c: the factor exp(-sign*c), a constant or a product of components
            # (exp(-x) for (1 + exp(-x))*exp(-(x + exp(x))**2 + x)), beside the exponentials of the
            # term in the canonical form; the products are distributed (exp(x)*exp(-x) is 1)
            cache.cache[candidate] = as_expr(expand_mul(expand_power_exp(
                2 / sqrt(pi) * u.diff(x) * exp(-sign_ * c) * exp(theta_term))))
            found.add(candidate)
    return found


def _exponent(g: Expr) -> int:
    """The contribution of the fractional powers of ``g`` to the degree
    bound of the polynomial part."""
    if isinstance(g, Pow):
        exponent = as_expr(g.exp)
        if isinstance(exponent, Rational) and exponent.q != 1:
            if exponent.p > 0:
                return exponent.p + exponent.q - 1
            return abs(exponent.p + exponent.q)
        return 1
    if not g.is_Atom and g.args:
        return max(_exponent(as_expr(h)) for h in g.args)
    return 1


def _real_forms(F: Expr) -> Expr:
    """Pairs ``a*log(P + I*Q) + b*log(P - I*Q)`` of the antiderivative
    written ``(a + b)/2 * log(P**2 + Q**2) - I*(a - b)*atan(P/Q)`` (the same
    derivative, ``atan(P/Q)`` being ``pi/2 - atan(Q/P)`` on each side of
    ``P = 0``), which is real when ``b`` is the conjugate of ``a``.

    >>> from sympy import symbols, I, log
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
        coefficient, function = as_expr(term).as_independent(log, as_Add=False)
        function_ = as_expr(function)
        if not isinstance(function_, log) or not function_.args[0].has(I):
            rest.append(as_expr(term))
            continue
        parts = collect(as_expr(function_.args[0]), I, evaluate=False)
        P, Q = as_expr(parts.get(S.One, S.Zero)), as_expr(parts.get(I, S.Zero))
        if P.has(I) or Q.has(I) or Q == 0:
            rest.append(as_expr(term))
            continue
        if Q.could_extract_minus_sign():
            logs[(P, as_expr(-Q))].append((as_expr(coefficient), -1))
        else:
            logs[(P, Q)].append((as_expr(coefficient), 1))
    result: Expr = Add(*rest)
    for (P, Q), entries in logs.items():
        a = as_expr(Add(*[c for c, s in entries if s == 1]))
        b = as_expr(Add(*[c for c, s in entries if s == -1]))
        combined = as_expr((a + b) / 2 * log(P**2 + Q**2) - I * (a - b) * atan(P / Q))
        combined = as_expr(combined.expand())
        if combined.has(I):
            # not a conjugate pair: as it was
            result = result + a * log(P + I * Q) + b * log(P - I * Q)
        else:
            result = result + combined
    return as_expr(result)


def _verified(F: Expr, f: Expr, x: Symbol) -> Optional[Expr]:
    """``F`` when ``F' == f``, symbolically or numerically where ``f`` is
    real; ``None`` otherwise."""
    difference = as_expr(F.diff(x) - f)
    reduced = attempt(lambda: as_expr(cancel(difference)), settings.timeout)
    if reduced is not None and reduced == 0:
        return F
    simpler = attempt(lambda: as_expr(simplify(difference)), settings.timeout)
    if simpler is not None and simpler == 0:
        return F
    if numerically_equal(as_expr(F.diff(x)), f):
        return F
    return None


def heurisch_antiderivative(f: ExprLike, x: Symbol, hints: Optional[Sequence[ExprLike]] = None,
                            degree_offset: int = 0, unnecessary_permutations: Optional[list[tuple[Expr, Symbol]]] = None,
                            retries: int = 2) -> Optional[Expr]:
    """An antiderivative of ``f`` in ``x`` by the heuristic Risch method,
    verified by differentiation, or ``None`` (never an unevaluated
    integral). ``hints`` are functions which may appear in the
    antiderivative besides the automatic candidates; ``degree_offset``
    raises the degree bound of the polynomial part; ``retries`` is the
    number of permutations of the components tried after the first.

    >>> from sympy import symbols, tan, log, sqrt, exp
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
    f_ = as_expr(f)
    if f_.has(Float) or f_.has(*_REFUSED):
        return None
    if not f_.has_free(x):
        return as_expr(f_ * x)
    hint_terms = [as_expr(h) for h in hints] if hints else []

    def search() -> Optional[Expr]:
        found = _heurisch(f_, x, False, hint_terms, None, retries, degree_offset, unnecessary_permutations)
        if found is None and degree_offset == 0:
            found = _heurisch(f_, x, False, hint_terms, None, retries, 1, unnecessary_permutations)
        return found

    # the whole search under the time limit of the settings: the linear
    # systems of the permutations and retries each took their own share
    result = attempt(search, settings.timeout)
    if result is None:
        return None
    result = _real_forms(result)
    return _verified(result, f_, x)


def _heurisch(f: Expr, x: Symbol, rewrite: bool, hints: list[Expr],
              mappings: Optional[Iterator[list[tuple[Expr, Symbol]]]], retries: int, degree_offset: int,
              unnecessary_permutations: Optional[list[tuple[Expr, Symbol]]]) -> Optional[Expr]:
    """The method proper: ``rewrite`` writes the trigonometric and
    hyperbolic functions in tangents, ``mappings`` iterates the
    permutations of the components."""
    if not f.has_free(x):
        return as_expr(f * x)
    # exponentials of sums as products of exponentials of their terms, so that
    # exp(-(x + exp(x))**2 + x) and exp(-(x + exp(x))**2) share their components
    # (the derivatives are written the same way by the cache)
    f = as_expr(expand_power_exp(f))
    indep: Expr = S.One
    if not f.is_Add:
        independent, dependent = f.as_independent(x)
        indep, f = as_expr(independent), as_expr(dependent)
    rewritables: dict[tuple[type[Function], ...], type[Function]] = {(sin, cos, cot): tan, (sinh, cosh, coth): tanh}
    if rewrite:
        for candidates, rule in rewritables.items():
            f = as_expr(f.rewrite(candidates, rule))
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
    ordered_terms: list[Expr] = [as_expr(t) for t in ordered(terms)]
    V = _symbols('x', len(ordered_terms))
    # the components from the largest to the smallest, x last
    keyed = sorted(zip(ordered_terms, V), key=lambda pair: (pair[0].as_independent(x)[1].count_ops(), str(pair[0])),
                   reverse=True)
    mapping: list[tuple[Expr, Symbol]] = [(t, v) for t, v in keyed]
    rev_mapping = {v: t for t, v in mapping}
    if mappings is None:
        # x is the smallest component: the last one, never permuted
        position = next(i for i, (t, _) in enumerate(mapping) if t == x)
        unnecessary_permutations = [mapping.pop(position)]
        types: dict[type[Basic], list[tuple[Expr, Symbol]]] = defaultdict(list)
        for pair in mapping:
            types[type(pair[0])].append(pair)
        groups = [types[k] for k in types]

        def _iter_mappings() -> Iterator[list[tuple[Expr, Symbol]]]:
            for permutation in permutations(groups):
                yield [pair for group in permutation for pair in ordered(group)]

        mappings = _iter_mappings()
    else:
        unnecessary_permutations = unnecessary_permutations or []

    def _substitute(expr: Expr) -> Expr:
        # from the largest component to the smallest, whatever the order of
        # the mapping: substituting exp(x) first turns the exp(2*x) inside
        # exp(exp(2*x)) into a power of the symbol, and the outer component
        # is never found again (the mapping's order labels the symbols, the
        # permutations being over the variable order of the ring)
        largest_first = sorted(current, key=lambda pair: (-pair[0].count_ops(), str(pair[0])))
        return as_expr(expr.subs(largest_first))

    current: list[tuple[Expr, Symbol]] = []
    diffs: list[Expr] = []
    denom: Optional[Expr] = None
    for candidate_mapping in mappings:
        current = list(candidate_mapping) + unnecessary_permutations
        diffs = [_substitute(cache.get(g)) for g in ordered_terms]
        denoms = [as_expr(g.as_numer_denom()[1]) for g in diffs]
        if all(h.is_polynomial(*V) for h in denoms) and _substitute(f).is_rational_function(*V):
            denom = as_expr(reduce(lambda p, q: as_expr(lcm(p, q, *V)), denoms))
            break
    if denom is None:
        if not rewrite:
            result = _heurisch(f, x, True, hints, None, retries, degree_offset, unnecessary_permutations)
            if result is not None:
                return as_expr(indep * result)
        return None
    numers = [as_expr(cancel(denom * g)) for g in diffs]

    def _derivation(h: Expr) -> Expr:
        return as_expr(Add(*[d * h.diff(v) for d, v in zip(numers, V)]))

    def _deflation(p: Expr) -> Expr:
        for y in V:
            if not p.has(y):
                continue
            if _derivation(p) != 0:
                c, q = p.as_poly(y).primitive()
                return as_expr(_deflation(as_expr(c)) * gcd(q, q.diff(y)).as_expr())
        return p

    def _splitter(p: Expr) -> tuple[Expr, Expr]:
        for y in V:
            if not p.has(y):
                continue
            if _derivation(as_expr(y)) != 0:
                c, q_ = p.as_poly(y).primitive()
                q = as_expr(q_.as_expr())
                h = as_expr(gcd(q, _derivation(q), y))
                s = as_expr(quo(h, gcd(q, q.diff(y), y), y))
                c_split = _splitter(as_expr(c))
                if s.as_poly(y).degree() == 0:
                    return (c_split[0], as_expr(q * c_split[1]))
                q_split = _splitter(as_expr(cancel(q / s)))
                return (as_expr(c_split[0] * q_split[0] * s), as_expr(c_split[1] * q_split[1]))
        return (S.One, p)

    special: dict[Expr, bool] = {}
    for term in ordered_terms:
        if isinstance(term, tan):
            special[as_expr(1 + _substitute(term)**2)] = False
        elif isinstance(term, tanh):
            special[as_expr(1 + _substitute(term))] = False
            special[as_expr(1 - _substitute(term))] = False
        elif isinstance(term, LambertW):
            special[_substitute(term)] = True
    F = _substitute(f)
    P, Q = (as_expr(e) for e in F.as_numer_denom())
    u_split = _splitter(denom)
    v_split = _splitter(Q)
    polys = set(list(v_split) + [u_split[0]] + list(special))
    s = as_expr(u_split[0] * Mul(*[k for k, v in special.items() if v]))
    polified = [p.as_poly(*V) for p in (s, P, Q)]
    if any(p is None for p in polified):
        return None
    a, b, c = (int(p.total_degree()) for p in polified)
    poly_denom = as_expr((s * v_split[0] * _deflation(v_split[1])))
    A, B = _exponent(f), a + max(b, c)
    bound = A + B - 1 + degree_offset if A > 1 and B > 1 else A + B + degree_offset
    monoms = [as_expr(m) for m in ordered(itermonomials(V, bound))]
    poly_coeffs = _symbols('A', len(monoms))
    poly_part = as_expr(Add(*[coefficient * monomial for coefficient, monomial in zip(poly_coeffs, monoms)]))
    reducibles: set[Expr] = set()
    for poly in ordered(polys):
        coefficient, factors = factor_list(poly, *V)
        reducibles.add(as_expr(coefficient))
        reducibles.update(as_expr(factor) for factor, _ in factors)

    def _integrate(field: Optional[str] = None) -> Optional[Expr]:
        atans: set[Expr] = set()
        pairs: set[tuple[Expr, Expr]] = set()
        if field == 'Q':
            irreducibles = set(reducibles)
        else:
            irreducibles = set()
            for poly in ordered(reducibles):
                zV = set(V) & set(iterfreeargs(poly))
                for z in ordered(zV):
                    irreducibles |= {as_expr(r) for r in root_factors(poly, z, filter=field)}
                    break
        log_part: list[Expr] = []
        atan_part: list[Expr] = []
        for poly in ordered(list(irreducibles)):
            parts = collect(poly, I, evaluate=False)
            y = as_expr(parts.get(I, S.Zero))
            if y != 0:
                x_ = as_expr(parts.get(S.One, S.Zero))
                if x_.has(I) or y.has(I):
                    continue
                pairs.add((x_, y))
                irreducibles.remove(poly)
        while pairs:
            x_, y = pairs.pop()
            if (x_, as_expr(-y)) in pairs:
                pairs.remove((x_, as_expr(-y)))
                if y.could_extract_minus_sign():
                    y = as_expr(-y)
                irreducibles.add(as_expr(x_ * x_ + y * y))
                atans.add(as_expr(atan(x_ / y)))
            else:
                irreducibles.add(as_expr(x_ + I * y))
        coefficients = list(poly_coeffs)
        B_ = _symbols('B', len(irreducibles))
        C_ = _symbols('C', len(atans))
        for poly, coefficient in reversed(list(zip(ordered(irreducibles), B_))):
            if poly.has(*V):
                coefficients.append(coefficient)
                log_part.append(as_expr(coefficient * log(poly)))
        for poly, coefficient in reversed(list(zip(ordered(atans), C_))):
            if poly.has(*V):
                coefficients.append(coefficient)
                atan_part.append(as_expr(coefficient * poly))
        candidate = as_expr(poly_part / poly_denom + Add(*log_part) + Add(*atan_part))
        h = as_expr(F - _derivation(candidate) / denom)
        raw_numer = as_expr(h.as_numer_denom()[0])
        syms = set(coefficients) | set(V)
        non_syms: set[Expr] = set()

        def find_non_syms(expr: Expr) -> None:
            if expr.is_Integer or expr.is_Rational:
                return
            if expr in syms:
                return
            if not expr.has_free(*syms):
                non_syms.add(expr)
            elif expr.is_Add or expr.is_Mul or expr.is_Pow:
                for argument in expr.args:
                    find_non_syms(as_expr(argument))
            else:
                raise PolynomialError

        try:
            find_non_syms(raw_numer)
        except PolynomialError:
            return None
        ground, _ = construct_domain(non_syms, field=True)
        coeff_ring = PolyRing(coefficients, ground)
        ring = PolyRing(V, coeff_ring)
        try:
            numer = ring.from_expr(raw_numer)
        except ValueError:
            return None
        solution = attempt(lambda: solve_lin_sys(numer.coeffs(), coeff_ring, _raw=False), settings.timeout)
        if solution is None:
            return None
        zeros = {coefficient: S.Zero for coefficient in coefficients}
        return as_expr(candidate.xreplace(solution).xreplace(zeros))

    more_free = free_symbols(F) - set(V)
    solution: Optional[Expr]
    try:
        if not more_free:
            solution = _integrate('Q')
            if solution is None:
                solution = _integrate()
        else:
            solution = _integrate()
    except (PolynomialError, ValueError, TypeError, ZeroDivisionError):
        solution = None
    if solution is not None:
        antiderivative = as_expr(solution.xreplace(rev_mapping))
        antiderivative = as_expr(cancel(antiderivative).expand())
        if antiderivative.is_Add:
            antiderivative = as_expr(antiderivative.as_independent(x)[1])
        return as_expr(indep * antiderivative)
    if retries >= 0:
        result = _heurisch(f, x, rewrite, hints, mappings, retries - 1, degree_offset, unnecessary_permutations)
        if result is not None:
            return as_expr(indep * result)
    return None


def heurisch_cases(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """The antiderivative of ``f`` with the cases of the parameters where
    a denominator of the generic antiderivative vanishes: ``cos(n*x)``
    integrates to ``sin(n*x)/n`` for ``n != 0`` and to ``x`` at ``n = 0``.
    A ``Piecewise`` over the cases, the generic one first, or the plain
    antiderivative when there is no such case; ``None`` when the generic
    case is not found.

    >>> from sympy import symbols, cos
    >>> from sympy_extras.integrals.heurisch import heurisch_cases
    >>> n, x = symbols('n x')
    >>> heurisch_cases(cos(n*x), x)
    Piecewise((sin(n*x)/n, Ne(n, 0)), (x, True))
    """
    from sympy.solvers.solvers import denoms, solve
    f_ = as_expr(f)
    if not f_.has_free(x):
        return as_expr(f_ * x)
    generic = heurisch_antiderivative(f_, x)
    if generic is None:
        return None
    cases: list[dict[Symbol, Expr]] = []
    for d in ordered(denoms(generic)):
        found = attempt(lambda: solve([d], dict=True, exclude=(x,)), settings.timeout)
        if found:
            cases.extend(found)
    if not cases:
        return generic
    cases = list(uniq(cases))
    original: list[dict[Symbol, Expr]] = []
    for d in denoms(f_):
        found = attempt(lambda: solve([d], dict=True, exclude=(x,)), settings.timeout)
        if found:
            original.extend(found)
    cases = [case for case in cases if case not in original]
    if not cases:
        return generic
    if len(cases) > 1:
        equations = [Eq(key, value) for case in cases for key, value in case.items()]
        joint = attempt(lambda: solve(equations, dict=True, exclude=(x,)), settings.timeout)
        cases = list(joint or []) + cases
    pairs: list[tuple[Expr, Boolean]] = []
    for case in cases:
        special = heurisch_antiderivative(as_expr(f_.xreplace(case)), x)
        if special is None:
            return None
        condition = as_boolean(And(*[Eq(key, value) for key, value in case.items()]))
        pairs.append((special, condition))
    if len(pairs) == 1:
        generic_condition = as_boolean(Or(*[Ne(key, value) for key, value in cases[0].items()]))
        return as_expr(Piecewise((generic, generic_condition), (pairs[0][0], True)))
    pairs.append((generic, as_boolean(True)))
    return as_expr(Piecewise(*pairs))
