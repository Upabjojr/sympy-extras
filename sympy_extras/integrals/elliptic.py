"""Definite integrals with the square root of a cubic or a quartic,
reduced to Legendre's elliptic integrals.

An integral `\\int R(x)\\, P(x)^{\\pm 1/2}\\, dx` with `R` rational and `P` a
polynomial of degree three or four with real roots is elliptic. The
classical reduction [Byrd]_, [DLMF]_ (chapter 19.29) maps the range
onto the standard one through a linear or a bilinear substitution
`x = X(s)`, `s = \\sin^2\\theta`, chosen so that the roots of `P` go to
`0`, `1`, `1/m` and (for a quartic) to the pole of `X`: on the cell of
the real line between consecutive roots, or beyond the extreme roots,
the radicand becomes `C\\, s(1 - s)(1 - m s)` up to a square, and the
integrand

.. math::

    R(X(s))\\, X'(s)\\, P(X(s))^{\\pm 1/2}\\, ds = \\frac{Q(s)\\, ds}{\\sqrt{s(1 - s)(1 - m s)}}
    = \\frac{2\\, Q(\\sin^2\\theta)\\, d\\theta}{\\sqrt{1 - m \\sin^2\\theta}}

with `Q` rational. The partial fractions of `Q` give the three kinds
(SymPy's parameter convention, `m = k^2`):

* `\\int_0^{\\pi/2} \\sin^{2j}\\theta\\, d\\theta / \\sqrt{1 - m\\sin^2\\theta} = J_j(m)` with
  `J_0 = K(m)`, `J_1 = (K - E)/m` and the recurrence
  `(2j + 1) m J_{j+1} = 2j(1 + m) J_j - (2j - 1) J_{j-1}` (Byrd–Friedman 310);
* `\\int_0^{\\pi/2} d\\theta / ((1 - n\\sin^2\\theta)\\sqrt{1 - m\\sin^2\\theta}) = \\Pi(n \\mid m)`;
* the incomplete integrals `F(\\phi \\mid m)`, `E(\\phi \\mid m)`,
  `\\Pi(n; \\phi \\mid m)` when an endpoint is not a root of `P`, with
  `\\phi = \\arcsin\\sqrt{s}` at the image of the endpoint.

An even radicand with an even `R` is first reduced by `y = x^2` (the
lemniscate integral `\\int_0^1 dx/\\sqrt{1 - x^4}` becomes a cubic case);
a quartic beyond its extreme root by `u = 1/(x - r)`, which makes it a
cubic. Trigonometric integrands `Q(\\sin^2\\theta)(1 - m\\sin^2\\theta)^{\\pm 1/2}`
over `(0, \\phi)` are read directly. The order of symbolic roots is
decided by :func:`sympy_extras.assumptions.ask.ask` under the assumptions.

When two roots of `P` are complex, `P = \\lambda (a - x)(x - b)((x - p)^2 + q^2)`
on the cell `(b, a)` (or `P = \\lambda (x - a)((x - p)^2 + q^2)` beyond the
single real root of a cubic), the reduction of Byrd–Friedman 241 and
240 maps the cell onto `t = \\cos\\theta \\in (-1, 1)` through the bilinear
substitution

.. math::

    x = \\frac{aB + bA - t\\,(aB - bA)}{A + B + t\\,(A - B)}, \\qquad
    A^2 = (a - p)^2 + q^2, \\quad B^2 = (b - p)^2 + q^2,

(`x = a + A(1 - t)/(1 + t)` for the cubic), for which
`P(x)\\, dt^2 = \\lambda AB\\, X'(t)^2 (1 - t^2)(1 - m + m t^2)\\, dt^2` with
`m = ((a - b)^2 - (A - B)^2)/(4AB)` (`m = (A - a + p)/(2A)` for the
cubic): `dx/\\sqrt{P} = d\\theta/\\sqrt{\\lambda AB (1 - m\\sin^2\\theta)}`. The
rational prefactor `G(t)` left by `R` is split into its even and odd
parts in `t`: the even part is a rational function of
`s = \\sin^2\\theta = 1 - t^2` and gives the Legendre form above, the odd
part is `t\\, G_o(t^2)` and its integral in `s` is elementary (it
vanishes over the whole cell). A quartic with a complex pair beyond
its extreme real root is inverted first, `u = 1/(x - r)`. The split
mirrors a pole of `R` at `t_p` to `-t_p`: a range containing the
mirror image of a pole of the cell, but not the pole, is left alone
(``None``), the two halves being divergent there.

When no root is real, `P = \\lambda\\, ((x - p_1)^2 + q_1^2)((x - p_2)^2 + q_2^2)`
is positive on the whole line, and the reduction of Byrd–Friedman 267
maps the line onto `\\theta \\in (-\\pi/2, \\pi/2)`: the bilinear
substitution `x = (\\beta + \\alpha t)/(1 + t)`, with `\\alpha > \\beta` the
roots of `z^2 - u z + v` for `(\\beta - p_i)(\\alpha - p_i) = -q_i^2` (the
shift `x = p + t` when the quadratics share their centre), sends both
quadratics to `(A_i t^2 + B_i)/(1 + t)^2`, and `t = \\lambda\\tan\\theta`
with `\\lambda^2 = B_1/A_1` gives `P\\, dt^2 = \\lambda B_1 B_2 (1 - m\\sin^2\\theta)
X'(t)^2 dt^2/\\cos^4\\theta` with `m = 1 - A_2 B_1/(A_1 B_2)` (the labels
chosen so that `0 < m < 1`). In `s = \\sin^2\\theta` the even part of the
rational prefactor in `t` is a Legendre form and the odd part a rational
function times `1/\\sqrt{1 - m s}`, elementary; the range is cut at the
pole `x = \\alpha` of the map (`\\theta = \\pm\\pi/2`) and at `t = 0`. A
numeric radicand whose quadratic factors are irrational (`x^4 + x - 1`,
`x^4 + x + 1`) is factored through its roots: the real ones and the real
parts and moduli of the complex ones are real algebraic numbers
(:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`, SymPy's
``CRootOf`` through :mod:`sympy_extras.interop`) of their own minimal
polynomials (from resultants), carried through the reduction as dummies
whose sign questions are decided numerically at sixty digits, and
restored at the end; the values are exact but large.

The organising principle behind the tables is Carlson's symmetric
integral `R_F(x, y, z) = \\frac12\\int_0^\\infty dt/\\sqrt{(t + x)(t + y)(t + z)}`
[Carlson]_, [DLMF]_ (19.25): the complete integral of `dx/\\sqrt{P}`
over a cell of the real roots is `2 R_F` of the three differences of
the other roots from the cell (DLMF 19.29.4–19.29.7), whatever their
reality, and Legendre's functions are its special values
`K(m) = R_F(0, 1 - m, 1)`, `F(\\phi \\mid m) = \\sin\\phi\\, R_F(\\cos^2\\phi, 1 - m\\sin^2\\phi, 1)`
(DLMF 19.25.1, 19.25.5). The substitutions above are the changes of
variable that bring the Carlson arguments to Legendre's; the results
are written with the core's ``elliptic_k``, ``elliptic_e``, ``elliptic_f``
and ``elliptic_pi`` (:mod:`sympy_extras.functions.special.elliptic_integrals`).

The module is native: it works on the expressions of the core with the
native polynomials (factorisations over ``ZZ`` and the polynomial rings
of the parameters, the irrational constants of the coefficients held by
placeholders; the partial fractions from the Laurent coefficients at the
linear factors; the real roots as indexed roots), the native ``ask`` and
the numerical values of :mod:`sympy_extras._numeric`, and imports no
SymPy. One step stays on SymPy: the rational antiderivative of the
elementary odd part of the two-pairs case (SymPy's ``integrate``,
through :func:`sympy_extras.integrals.elliptic_sympy.integrate_of_core`).
:mod:`sympy_extras.integrals.elliptic_sympy` is the SymPy face.

Examples
========

>>> from sympy_extras.core import symbols, Symbol, sqrt, oo, pi, Rational
>>> from sympy_extras.functions import sin
>>> from sympy_extras.integrals.elliptic import elliptic_integral
>>> x, theta = symbols('x theta')
>>> k = Symbol('k', positive=True)
>>> elliptic_integral(1/sqrt(x*(1 - x)*(1 - k**2*x)), x, 0, 1, k < 1)
ConditionalValue(2*elliptic_k(k**2))
>>> elliptic_integral(1/sqrt(x*(x - 1)*(2 - x)*(3 - x)), x, 1, 2)
ConditionalValue(elliptic_k(3/4))
>>> elliptic_integral(sqrt(1 - k**2*sin(theta)**2), theta, 0, pi/2, k < 1)
ConditionalValue(elliptic_e(k**2))

References
==========

.. [Byrd] P. F. Byrd, M. D. Friedman, *Handbook of Elliptic Integrals
   for Engineers and Scientists*, 2nd ed., Springer, 1971, chapter 2
   (the substitutions), the tables 230–260 (240 and 241 for the complex
   roots, 267 for two complex pairs) and 310–340.
.. [Carlson] B. C. Carlson, *Numerical computation of real or complex
   elliptic integrals*, Numerical Algorithms 10 (1995), 13–26;
   *A table of elliptic integrals of the third kind*, Mathematics of
   Computation 51 (1988), 267–280.
.. [DLMF] NIST Digital Library of Mathematical Functions, chapter 19,
   sections 19.2 (Legendre's integrals), 19.25 (Legendre's integrals
   as symmetric integrals) and 19.29 (reduction of
   `\\int R(x, \\sqrt{P})\\, dx`), https://dlmf.nist.gov/19.
.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series,
   and Products*, 7th ed., sections 3.13–3.16.
"""
from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Callable, Final, Optional, Sequence

import mpmath

from sympy_extras._numeric import reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import expand
from sympy_extras.core.logic import Boolean, to_boolean
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Float, I, Infinity, Integer, NegativeInfinity, Rational, oo, pi
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.relational import Eq, Ge, Gt, Le, Lt
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Im, Re, im
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.functions.elementary.trigonometric import Cos, Cot, Csc, Sec, Sin, Tan, asin, sin
from sympy_extras.functions.special.elliptic_integrals import (EllipticE, EllipticF, EllipticK, EllipticPi, elliptic_e,
                                                               elliptic_f, elliptic_k, elliptic_pi)
from sympy_extras.polys.cad.indexedroot import IndexedRoot, indexed_root
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.factortools import dup_ext_factor
from sympy_extras.polys.numberfields.qqbar import Algebraic, primitive_element
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel, factor_list, resultant, sqf_list
from sympy_extras.polys.rootisolation import dup_count_real_roots
from sympy_extras.settings import settings
from sympy_extras.simplify.radsimp import radsimp
from .conditions import ConditionalValue, plain_symbols, with_symbol_facts

__all__ = ['elliptic_integral', 'radicand', 'real_roots', 'legendre_reduction', 'Reduction']

_TRIGONOMETRIC = (Sin, Cos, Tan, Cot, Sec, Csc)


_COUNTER = itertools.count()


def _fresh(name: str, **assumptions: bool) -> Dummy:
    """A dummy with a name of its own: the typed polynomials map their
    symbols by name, so two dummies ``A`` (the radicals ``A`` and ``B`` of
    the cosine map) or a dummy ``y`` beside a parameter ``y`` would be
    taken for one symbol."""
    return Dummy('%s%d' % (name, next(_COUNTER)), **assumptions)


def _budget() -> Optional[float]:
    return None if settings.timeout is None else settings.timeout / 4


def _ask(query: Boolean, assumptions: Assumptions) -> Optional[bool]:
    """``ask`` with the flags of the symbols (``positive=True``) turned
    into statements on plain symbols, which the CAD can use, within a
    budget (``None`` when the time is up). The algebraic numbers of a
    numeric radicand (indexed roots, the real parts and moduli of the
    complex roots, the radicals of the maps) travel as dummies whose
    values are the equalities ``Eq(dummy, value)`` among the assumptions:
    a query on them alone is decided numerically, at sixty digits with a
    margin (``None`` within it), and the equalities are dropped for the
    CAD."""
    values, rest = _numeric_values(assumptions)
    symbols = query.free_symbols
    if values and symbols and symbols <= set(values):
        return _decide_numerically(query, values)
    plain = plain_symbols(symbols)
    if not plain:
        return attempt(lambda: ask(query, rest), _budget(), native=True)
    rule: dict[Basic, Basic] = {s: p for s, p in plain.items()}
    on_plain = to_boolean(query._xreplace_evaluated(rule))
    return attempt(lambda: ask(on_plain, with_symbol_facts(rest, symbols, plain)), _budget(), native=True)


def _numeric_values(assumptions: Assumptions) -> tuple[dict[Symbol, Expr], Assumptions]:
    """The values ``Eq(dummy, value)`` among the assumptions, and the
    other assumptions."""
    if assumptions is None or isinstance(assumptions, (Basic, bool)):
        return ({}, assumptions)
    values: dict[Symbol, Expr] = {}
    rest: list[Boolean] = []
    for item in assumptions:
        item_ = to_boolean(item)
        if isinstance(item_, Eq) and isinstance(item_.lhs, Dummy):
            values[item_.lhs] = item_.rhs
        else:
            rest.append(item_)
    return (values, rest)


def _numeric(e: Expr, values: dict[Symbol, Expr], evaluated: bool = True) -> Expr:
    """``e`` with the dummies replaced by their values (which may refer
    to other dummies) until none is left; the rebuilt nodes evaluated, or
    not (a value in indexed roots, see :func:`_restore`)."""
    rule: dict[Basic, Basic] = {s: v for s, v in values.items()}
    result = e
    for _ in range(8):
        if not result.atoms(Dummy) & set(values):
            break
        replaced = result._xreplace_evaluated(rule) if evaluated else result.xreplace(rule)
        result = _expr(replaced)
    return result


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("expected an expression, got %s" % (b,))
    return b


def _decide_numerically(query: Boolean, values: dict[Symbol, Expr]) -> Optional[bool]:
    if isinstance(query, Eq):
        difference = _numeric(query.lhs - query.rhs, values)
        return True if difference == 0 else None
    if not isinstance(query, (Gt, Ge, Lt, Le)):
        return None
    difference = _numeric(query.lhs - query.rhs, values)
    if difference.free_symbols:
        return None
    approximation = reliable_value(difference, 60)
    if not isinstance(approximation, Float):
        return None
    if abs(approximation) < Float(10)**(-40):
        return None
    positive = bool(approximation > 0)
    if isinstance(query, (Gt, Ge)):
        return positive
    return not positive


def _with_facts(assumptions: Assumptions, facts: Sequence[Boolean]) -> Assumptions:
    """The assumptions together with more facts."""
    if not facts:
        return assumptions
    combined: list[Boolean] = list(facts)
    if isinstance(assumptions, (Basic, bool)):
        combined.append(to_boolean(assumptions))
    elif assumptions is not None:
        combined.extend(to_boolean(a) for a in assumptions)
    return combined


_HALF: Final[Expr] = S.Half
#: how many times the substitutions may be chained (an even radicand,
#: then a quartic beyond its extreme root, then the cubic case)
_MAX_DEPTH = 4


# ---------------------------------------------------------------------------
# Polynomials with irrational constants in their coefficients

class _Placeholders:
    """The irrational constants of expressions (``sqrt(3)``, ``pi``, an
    indexed root) replaced by dummies, so that the typed polynomials,
    which have no domain ``EX``, take them as parameters (SymPy's
    ``cancel`` takes them as generators too); :meth:`restore` puts them
    back, the rebuilt nodes evaluated. With a ``variable``, every
    subexpression free of it which is not a polynomial in symbols
    (``sqrt(a)``) is a placeholder too, a parameter of the polynomials in
    the variable."""

    def __init__(self, variable: Optional[Symbol] = None) -> None:
        self.variable: Final[Optional[Symbol]] = variable
        self.forward: Final[dict[Expr, Dummy]] = {}
        self.back: Final[dict[Basic, Basic]] = {}

    def hide(self, e: Expr) -> Expr:
        if isinstance(e, (Rational, Symbol)):
            return e
        if isinstance(e, Add):
            return add(*[self.hide(a) for a in e.args])
        if isinstance(e, Mul):
            return mul(*[self.hide(a) for a in e.args])
        if isinstance(e, Pow) and isinstance(e.exp, Integer):
            return pow(self.hide(e.base), e.exp)
        if e.has(self.variable) if self.variable is not None else e.free_symbols:
            return e
        if isinstance(e, Pow) and isinstance(e.exp, Rational) and e.exp.p != 1:
            # a**(p/q) is (a**(1/q))**p, as SymPy's generators are
            return pow(self._placeholder(pow(e.base, Rational(1, e.exp.q))), Integer(e.exp.p))
        return self._placeholder(e)

    def _placeholder(self, e: Expr) -> Expr:
        found = self.forward.get(e)
        if found is None:
            found = _fresh('c')
            self.forward[e] = found
            self.back[found] = e
        return found

    def restore(self, e: Expr) -> Expr:
        if not self.back:
            return e
        return _expr(e._xreplace_evaluated(self.back))


def _coefficients(e: Expr, s: Symbol) -> Optional[dict[int, Expr]]:
    """The coefficients of the polynomial ``e`` in ``s`` by degree, read
    from its expansion (``None`` when ``e`` is not a polynomial in
    ``s``); the zero coefficients are left out."""
    collected: dict[int, list[Expr]] = {}
    for term in Add.make_args(expand(e)):
        if term == 0:
            continue
        coefficient, power = term.as_independent(s, as_Add=False)
        if power == 1:
            degree = 0
        elif power == s:
            degree = 1
        elif isinstance(power, Pow) and power.base == s and isinstance(power.exp, Integer) and power.exp.p > 0:
            degree = power.exp.p
        else:
            return None
        collected.setdefault(degree, []).append(coefficient)
    result: dict[int, Expr] = {}
    for degree, parts in collected.items():
        total = add(*parts)
        if total != 0:
            result[degree] = total
    return result


def _degree(e: Expr, s: Symbol) -> Optional[int]:
    """The degree of the polynomial ``e`` in ``s`` (``-1`` for zero),
    ``None`` when it is not a polynomial."""
    coefficients = _coefficients(e, s)
    if coefficients is None:
        return None
    return max(coefficients, default=-1)


def _factor_list(e: Expr, s: Symbol) -> Optional[tuple[Expr, list[tuple[Expr, int]]]]:
    """``factor_list(e, s)`` over the integers and the polynomial rings of
    the parameters, the irrational constants held by placeholders;
    ``None`` when the native factorisation does not take ``e``."""
    hidden = _Placeholders(s)
    try:
        coefficient, factors = factor_list(hidden.hide(e), s)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    result: list[tuple[Expr, int]] = []
    for factor_, multiplicity in factors:
        if not isinstance(multiplicity, Integer):
            return None
        result.append((hidden.restore(factor_), multiplicity.p))
    return (hidden.restore(coefficient), result)


def _rational_coefficients(e: Expr, s: Symbol) -> Optional[list[Fraction]]:
    """The dense coefficients (highest degree first) of a polynomial in
    ``s`` with rational coefficients, ``None`` otherwise."""
    coefficients = _coefficients(e, s)
    if not coefficients:
        return None
    dense: list[Fraction] = []
    for degree in range(max(coefficients), -1, -1):
        c = coefficients.get(degree, Integer(0))
        if not isinstance(c, Rational):
            return None
        dense.append(Fraction(c.p, c.q))
    return dense


def _low_degree_roots(e: Expr, s: Symbol) -> Optional[list[Expr]]:
    """The roots of a polynomial of degree one or two in ``s`` (SymPy's
    ``solve`` of the old module), the constants held by placeholders."""
    hidden = _Placeholders(s)
    try:
        found = roots(Poly(hidden.hide(expand(e)), s), multiple=True)
    except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
        return None
    if not isinstance(found, list):
        return None
    result: list[Expr] = []
    for r in found:
        restored = hidden.restore(r)
        if restored not in result:
            result.append(restored)
    return result


def _real_roots_rational(e: Expr, s: Symbol) -> Optional[list[Expr]]:
    """The distinct real roots of a polynomial with rational coefficients,
    increasing, as rational numbers, radicals or indexed roots (SymPy's
    ``real_roots`` in ``CRootOf``)."""
    dense = _rational_coefficients(e, s)
    if dense is None:
        return None
    count = dup_count_real_roots(dense)
    return [indexed_root(e, s, i) for i in range(count)]


class Reduction:
    """The Legendre form of an elliptic integral,
    ``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))``.

    Attributes
    ==========

    Q : Expr
        A rational function of ``s``.
    m : Expr
        The parameter (``k**2``).
    s : Symbol
    lower, upper : Expr
        The range of ``s`` inside ``[0, 1]``.
    """

    __slots__ = ('Q', 'm', 's', 'lower', 'upper')

    def __init__(self, Q: ExprLike, m: ExprLike, s: Symbol, lower: ExprLike, upper: ExprLike) -> None:
        self.Q: Final[Expr] = to_expr(Q)
        self.m: Final[Expr] = to_expr(m)
        self.s: Final[Symbol] = s
        self.lower: Final[Expr] = to_expr(lower)
        self.upper: Final[Expr] = to_expr(upper)

    def __repr__(self) -> str:
        return "Reduction(%s, m=%s, (%s, %s, %s))" % (self.Q, self.m, self.s, self.lower, self.upper)



#: the largest degree of the number field of the constants of a
#: denominator factored over it (beyond it they stay placeholders)
_MAX_FIELD_DEGREE = 32


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
    numeric = numeric_value(atom, 100)
    if numeric is None or abs(complex(value) - complex(numeric)) > 1e-9 * (1 + abs(complex(numeric))):
        return None
    return value


def _horner(coefficients: Sequence[Fraction], point: list[Fraction], K: AlgebraicField) -> list[Fraction]:
    """The polynomial (rational coefficients, highest degree first) at the
    element ``point`` of ``K``."""
    result = K.zero
    for c in coefficients:
        result = K.add(K.mul(result, point), K.from_rational(Fraction(c)))
    return result


class _NumberField:
    """``QQ(theta)`` generated by algebraic constants, each one a
    polynomial in the symbol ``theta`` (:attr:`forms`) and ``theta`` an
    expression in them (:attr:`theta_expr`, ``a_1 + c_2 a_2 + ...``, the
    primitive elements of
    :func:`~sympy_extras.polys.numberfields.qqbar.primitive_element`);
    ``ArithmeticError`` beyond :data:`_MAX_FIELD_DEGREE`."""

    __slots__ = ('theta', 'field', 'theta_expr', 'forms', '_powers')

    def __init__(self, atoms: Sequence[Expr], values: Sequence[Algebraic]) -> None:
        theta = _fresh('theta')
        self.theta: Final[Dummy] = theta
        gamma = values[0]
        gamma_expr = atoms[0]
        coordinates: list[list[Fraction]] = [[Fraction(1), Fraction(0)]]
        for atom, value in zip(atoms[1:], values[1:]):
            new, p, q = primitive_element(gamma, value)
            if new.degree > _MAX_FIELD_DEGREE:
                raise ArithmeticError('the field of the constants is too large')
            K = AlgebraicField([Fraction(c, new.minpoly[0]) for c in new.minpoly], theta)
            gamma_in_new = _horner(p, K.alpha, K)
            coordinates = [_horner(c, gamma_in_new, K) for c in coordinates] + [_horner(q, K.alpha, K)]
            c = next(k for k in range(1, 4 * gamma.degree * value.degree + 2) if gamma + k * value == new)
            gamma, gamma_expr = new, gamma_expr + c * atom
        self.field: Final[AlgebraicField] = AlgebraicField([Fraction(c, gamma.minpoly[0]) for c in gamma.minpoly],
                                                           theta)
        self.theta_expr: Final[Expr] = gamma_expr
        self.forms: Final[dict[Basic, Basic]] = {atom: self.field.to_expr(element)
                                                 for atom, element in zip(atoms, coordinates)}
        #: the powers of theta expanded in the constants, as the rational
        #: coefficients of their terms
        self._powers: Final[list[dict[Expr, Fraction]]] = []

    def _power(self, k: int) -> dict[Expr, Fraction]:
        while len(self._powers) <= k:
            terms: dict[Expr, Fraction] = {}
            for term in Add.make_args(expand(self.theta_expr**len(self._powers))):
                c, rest = term.as_coeff_Mul()
                if not isinstance(c, Rational):
                    c, rest = Integer(1), term
                terms[rest] = terms.get(rest, Fraction(0)) + Fraction(c.p, c.q)
            self._powers.append(terms)
        return self._powers[k]

    def expression(self, element: list[Fraction]) -> Expr:
        """An element of the field written in the constants (expanded)."""
        n = len(element) - 1
        terms: dict[Expr, Fraction] = {}
        for i, c in enumerate(element):
            if c:
                for term, d in self._power(n - i).items():
                    terms[term] = terms.get(term, Fraction(0)) + c * d
        return add(*[Rational(c.numerator, c.denominator) * term for term, c in terms.items() if c])


#: the number fields of the constants met, by their generators
_FIELDS: dict[tuple[Expr, ...], _NumberField] = {}


class _ConstantField:
    """The number field of the algebraic constants (radicals of rational
    numbers, ``I``) of some expressions, in which the polynomials of one
    variable with such coefficients are factored and their coefficients
    written canonically (SymPy's ``apart`` works with ``extension=True``).
    :attr:`field` is ``None`` when a constant is not algebraic, the
    expressions have parameters, or there is no constant."""

    __slots__ = ('rule', 'field')

    def __init__(self, exprs: Sequence[Expr], s: Symbol) -> None:
        hidden = _Placeholders()
        for e in exprs:
            hidden.hide(expand(e))
        field: Optional[_NumberField] = None
        free: set[Symbol] = set()
        for e in exprs:
            free |= e.free_symbols
        atoms: list[Expr] = []
        values: list[Algebraic] = []
        for constant in hidden.forward:
            value = _algebraic(constant)
            if value is None:
                atoms = []
                break
            atoms.append(constant)
            values.append(value)
        if atoms and not free - {s}:
            key = tuple(atoms)
            field = _FIELDS.get(key)
            if field is None:
                try:
                    field = _NumberField(atoms, values)
                except ArithmeticError:
                    field = None
                if field is not None:
                    if len(_FIELDS) > 64:
                        _FIELDS.clear()
                    _FIELDS[key] = field
        self.field: Final[Optional[_NumberField]] = field
        self.rule: Final[dict[Basic, Basic]] = {} if field is None else dict(field.forms)

    def element(self, c: Expr) -> Optional[list[Fraction]]:
        """The constant ``c`` as an element of the field (``None`` when it
        is not one)."""
        field = self.field
        if field is None:
            return None
        written = _expr(c._xreplace_evaluated(self.rule))
        if written.free_symbols - {field.theta}:
            return None
        try:
            return field.field.from_expr(expand(written))
        except (ValueError, ZeroDivisionError, BasePolynomialError):
            return None

    def canonical(self, c: Expr) -> Expr:
        """The constant ``c`` in the canonical form of the field (``c``
        itself when it is not an element)."""
        element = self.element(c)
        if element is None or self.field is None:
            return c
        return self.field.expression(element)

    def factor_list(self, e: Expr, s: Symbol) -> Optional[tuple[Expr, list[tuple[Expr, int]]]]:
        """The factorisation of the polynomial ``e`` in ``s`` over the
        field; ``None`` when a coefficient is not an element."""
        field = self.field
        coefficients = _coefficients(e, s)
        if field is None or not coefficients:
            return None
        dense: list[list[Fraction]] = []
        for degree in range(max(coefficients), -1, -1):
            element = self.element(coefficients.get(degree, Integer(0)))
            if element is None:
                return None
            dense.append(element)
        try:
            lead, factors = dup_ext_factor(dense, field.field)
        except (ValueError, ZeroDivisionError, BasePolynomialError):
            return None
        result: list[tuple[Expr, int]] = []
        for factor_, multiplicity in factors:
            n = len(factor_) - 1
            result.append((add(*[field.expression(c) * s**(n - i) for i, c in enumerate(factor_)]), multiplicity))
        return (field.expression(lead), result)

# ---------------------------------------------------------------------------
# Reading the integrand

def radicand(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr, Expr]]:
    """``(R, P, e)`` with ``f == R * P**e``, ``e`` equal to ``1/2`` or
    ``-1/2``, ``P`` a polynomial in ``x`` of degree three or four and
    ``R`` a rational function of ``x``; the square roots of several
    factors are combined into one, and a half-integer power ``x**(3/2)``
    is ``x`` times ``sqrt(x)``. ``None`` when ``f`` is not of this form.

    >>> from sympy_extras.core import symbols, sqrt
    >>> from sympy_extras.integrals.elliptic import radicand
    >>> x = symbols('x')
    >>> radicand(x/sqrt((1 - x**2)*(4 - x**2)), x)
    (x, (1 - x**2)*(4 - x**2), -1/2)
    >>> radicand(sqrt(x)*sqrt(1 - x)/sqrt(4 - x), x)
    (-1/(x - 4), x*(1 - x)*(4 - x), 1/2)
    """
    halves: list[tuple[Expr, Expr]] = []
    rest: Expr = Integer(1)
    for factor_ in Mul.make_args(to_expr(f)):
        if isinstance(factor_, Pow) and factor_.has(x) and isinstance(factor_.exp, Rational) \
                and factor_.exp.q == 2:
            base = factor_.base
            if not base.is_polynomial(x):
                return None
            halves.append((base, factor_.exp))
        else:
            rest = rest * factor_
    if not halves:
        return None
    # the sign of the root: that of the factor of highest degree with a
    # bare square root, the denominator by default
    exponent: Expr = -_HALF
    best = -1
    for base, q in halves:
        degree = _degree(base, x)
        if degree is None:
            return None
        if abs(q) == _HALF and degree > best:
            best, exponent = degree, q
    P: Expr = Integer(1)
    for base, q in halves:
        P = P * base
        rest = rest * base**(q - exponent)
    degree = _degree(P, x)
    if degree is None:
        return None
    rest = cancel(rest)
    if degree not in (3, 4) or not rest.is_rational_function(x):
        return None
    return (rest, P, exponent)


def real_roots(P: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[list[Expr]]:
    """The roots of ``P`` sorted increasingly, when they are all real and
    distinct: numeric roots as indexed roots
    (:class:`~sympy_extras.polys.cad.indexedroot.IndexedRoot`, SymPy's
    ``CRootOf``) or radicals, symbolic ones from the linear and quadratic
    factors of ``P``, ordered by :func:`~sympy_extras.assumptions.ask.ask`
    under the assumptions. ``None`` otherwise.

    >>> from sympy_extras.core import symbols, Symbol
    >>> from sympy_extras.integrals.elliptic import real_roots
    >>> x = symbols('x')
    >>> k = Symbol('k', positive=True)
    >>> real_roots((1 - x**2)*(1 - k**2*x**2), x, k < 1)
    [-1/k, -1, 1, 1/k]
    """
    P_ = to_expr(P)
    degree = _degree(P_, x)
    if degree is None or degree < 1:
        return None
    found: list[Expr] = []
    if not P_.free_symbols - {x}:
        # a numeric polynomial: the roots with SymPy's all_roots, which
        # does not sort the roots over EX (irrational coefficients)
        dense = _rational_coefficients(P_, x)
        if dense is None:
            return None
        reals = _real_roots_rational(P_, x)
        if reals is None or len(reals) != degree:
            # complex or repeated roots
            return None
        found = reals
    else:
        factored = _factor_list(P_, x)
        if factored is None:
            return None
        _, factors = factored
        for factor_, multiplicity in factors:
            if multiplicity != 1 or not factor_.has(x):
                if factor_.has(x):
                    return None
                continue
            factor_degree = _degree(factor_, x)
            if factor_degree is None or factor_degree > 2:
                return None
            solutions = _low_degree_roots(factor_, x)
            if solutions is None:
                return None
            for r in solutions:
                if r.is_extended_real is False:
                    return None
                # p + I q with real symbols: real only where q vanishes
                imaginary = im(r)
                if imaginary != 0 and not imaginary.has(Im, Re) \
                        and _ask(Eq(imaginary, 0), assumptions) is not True:
                    return None
                found.append(r)
    if len(found) != degree:
        return None
    ordered: list[Expr] = []
    for r in found:
        position = 0
        for q in ordered:
            verdict = _less(q, r, assumptions)
            if verdict is None:
                return None
            if verdict:
                position += 1
        ordered.insert(position, r)
    return ordered


def _less(a: Expr, b: Expr, assumptions: Assumptions) -> Optional[bool]:
    """``a < b`` for two real numbers, ``None`` if undecided or equal."""
    if a == b:
        return None
    if isinstance(a, IndexedRoot) or isinstance(b, IndexedRoot):
        difference = numeric_value(b - a, 100)
        if difference is None or difference.imag != 0 or difference.real == 0:
            return None
        return bool(difference.real > 0)
    # the sign of a quotient is the sign of the product of its parts, a
    # polynomial question the assumptions can decide (1 < 1/k**2 under k < 1)
    numerator, denominator = cancel(b - a).as_numer_denom()
    try:
        query = numerator * denominator > 0
    except TypeError:
        # a comparison of numbers that are not real
        return None
    return _ask(query, assumptions)


def _position(point: Expr, roots_: Sequence[Expr], assumptions: Assumptions) -> Optional[int]:
    """The cell of ``point`` among the roots: ``2 i`` when it is the
    ``i``-th root, ``2 i + 1`` when it lies strictly between the ``i``-th
    and the next (``-oo`` is ``-1``, ``oo`` is ``2 n + 1``)."""
    if isinstance(point, NegativeInfinity):
        return -1
    if isinstance(point, Infinity):
        return 2 * len(roots_) - 1
    for i, r in enumerate(roots_):
        if point == r or (_ask(point - r >= 0, assumptions) is True
                          and _ask(point - r <= 0, assumptions) is True):
            return 2 * i
        below = _less(point, r, assumptions)
        if below is None:
            return None
        if below:
            return 2 * i - 1
    return 2 * len(roots_) - 1


# ---------------------------------------------------------------------------
# The substitutions

class _Substitution:
    """``x = X(s)`` on a cell, with the parameter ``m``."""

    __slots__ = ('X', 'm', 's')

    def __init__(self, X: Expr, m: Expr, s: Symbol) -> None:
        self.X: Final[Expr] = X
        self.m: Final[Expr] = m
        self.s: Final[Symbol] = s


def _cell_substitution(roots_: Sequence[Expr], cell: int, s: Symbol,
                       assumptions: Assumptions) -> Optional[_Substitution]:
    """The substitution of the cell (an odd index of :func:`_position`)."""
    n = len(roots_)
    if cell < -1 or cell > 2 * n - 1 or cell % 2 == 0:
        return None
    if cell == -1 or cell == 2 * n - 1:
        if n != 3:
            return None
        r1, r2, r3 = roots_
        if cell == 2 * n - 1:
            # (r_3, oo): x = r_3 + (r_3 - r_2) s/(1 - s), m = (r_2 - r_1)/(r_3 - r_1)
            return _Substitution(r3 + (r3 - r2) * s / (1 - s), (r2 - r1) / (r3 - r1), s)
        # (-oo, r_1): x = r_1 - (r_2 - r_1) s/(1 - s), m = (r_3 - r_2)/(r_3 - r_1)
        return _Substitution(r1 - (r2 - r1) * s / (1 - s), (r3 - r2) / (r3 - r1), s)
    i = (cell - 1) // 2
    lo, hi = roots_[i], roots_[i + 1]
    others = [r for j, r in enumerate(roots_) if j not in (i, i + 1)]
    if n == 3:
        r = others[0]
        if _less(hi, r, assumptions):
            return _Substitution(lo + (hi - lo) * s, (hi - lo) / (r - lo), s)
        return _Substitution(hi - (hi - lo) * s, (hi - lo) / (hi - r), s)
    # a quartic: a bilinear map with its pole at one of the other roots,
    # oriented so that the parameter lies in (0, 1)
    for pole, other in ((others[0], others[1]), (others[1], others[0])):
        for start, end in ((lo, hi), (hi, lo)):
            gamma = (pole - end) / (end - start)
            X = (pole * s + start * gamma) / (s + gamma)
            image = _preimage(X, s, other)
            if image is None or image == 0:
                continue
            m = cancel(1 / image)
            if _ask(m > 0, assumptions) and _ask(m < 1, assumptions):
                return _Substitution(X, m, s)
    return None


def _preimage(X: Expr, s: Symbol, point: Expr) -> Optional[Expr]:
    """The ``s`` with ``X(s) == point`` (``1`` for a point at infinity
    of a map with its pole at ``s = 1``): the root of the numerator of
    ``X(s) - point``, linear for the linear and bilinear maps (SymPy's
    ``solve`` of the old module)."""
    if isinstance(point, (Infinity, NegativeInfinity)):
        return Integer(1)
    numerator, denominator = cancel(X - point).as_numer_denom()
    coefficients = _coefficients(numerator, s)
    if coefficients is None or set(coefficients) - {0, 1} or 1 not in coefficients:
        return None
    solution = cancel(-coefficients.get(0, Integer(0)) / coefficients[1])
    if cancel(denominator.subs(s, solution)) == 0:
        return None
    return solution


def _reduce_on_cell(R: Expr, P: Expr, e: Expr, x: Symbol, roots_: Sequence[Expr], lo: Expr, hi: Expr,
                    cell: int, assumptions: Assumptions) -> Optional[Reduction]:
    """The Legendre form on a cell of the roots."""
    s = _fresh('s', positive=True)
    substitution = _cell_substitution(roots_, cell, s, assumptions)
    if substitution is None:
        return None
    X, m = substitution.X, substitution.m
    s_lo, s_hi = _preimage(X, s, lo), _preimage(X, s, hi)
    if s_lo is None or s_hi is None:
        return None
    sign: Expr = Integer(1)
    if _ask(s_lo < s_hi, assumptions) is False:
        s_lo, s_hi, sign = s_hi, s_lo, Integer(-1)
    for bound in (s_lo, s_hi):
        if _ask(bound >= 0, assumptions) is not True or _ask(bound <= 1, assumptions) is not True:
            return None
    split = _split_radicand(_expr(P.subs(x, X)), s, m, assumptions)
    if split is None:
        return None
    C, part = split
    # P(X(s))^e = C^e part^e sqrt(s(1-s)(1-ms))^(2e): with e = -1/2 the
    # integrand is R X' / (sqrt(C) part sqrt(s(1-s)(1-ms))), with e = 1/2
    # it is R X' sqrt(C) part s(1-s)(1-ms) / sqrt(s(1-s)(1-ms))
    Xp = X.diff(s)
    R_X = _expr(R.subs(x, X))
    if e == -_HALF:
        Q = R_X * Xp / (sqrt(C) * part)
    else:
        Q = R_X * Xp * sqrt(C) * part * s * (1 - s) * (1 - m * s)
    Q = cancel(sign * Q)
    if not Q.is_rational_function(s):
        return None
    return Reduction(Q, m, s, s_lo, s_hi)


def _split_radicand(P_s: Expr, s: Symbol, m: Expr,
                    assumptions: Assumptions) -> Optional[tuple[Expr, Expr]]:
    """``P(X(s))`` written as ``C * part**2 * s*(1 - s)*(1 - m*s)`` with
    ``C`` a positive constant and ``part`` a rational function of ``s``
    (positive on the range): the exponents of the three Legendre factors
    must be odd and those of every other factor even. ``(C, part)``, or
    ``None``."""
    numerator, denominator = cancel(P_s).as_numer_denom()
    C: Expr = Integer(1)
    exponents = {'s': 0, '1-s': 0, '1-ms': 0}
    forms: dict[str, Expr] = {'s': s, '1-s': 1 - s, '1-ms': 1 - m * s}
    others: list[tuple[Expr, int]] = []
    for polynomial, orientation in ((numerator, 1), (denominator, -1)):
        factored = _factor_list(polynomial, s)
        if factored is None:
            return None
        coefficient, factors = factored
        C = C * coefficient**orientation
        for factor_, multiplicity in factors:
            exponent = orientation * multiplicity
            if not factor_.has(s):
                C = C * factor_**exponent
                continue
            matched = False
            for name, form in forms.items():
                ratio = cancel(factor_ / form)
                if not ratio.has(s):
                    C = C * ratio**exponent
                    exponents[name] += exponent
                    matched = True
                    break
            if not matched:
                others.append((factor_, exponent))
    if any(v % 2 == 0 for v in exponents.values()):
        return None
    if any(exponent % 2 for _, exponent in others):
        return None
    if _ask(C > 0, assumptions) is not True:
        return None
    part: Expr = Integer(1)
    for name, form in forms.items():
        part = part * form**((exponents[name] - 1) // 2)
    for factor_, exponent in others:
        half = exponent // 2
        if half % 2:
            # an odd power of the square root of an even power: the
            # absolute value, decided on the range
            at_half = _expr(factor_.subs(s, S.Half))
            if _ask(at_half < 0, assumptions):
                factor_ = -factor_
            elif _ask(at_half > 0, assumptions) is not True:
                return None
        part = part * factor_**half
    return (C, part)


#: the elliptic functions of the reduction stand for dummies until the
#: value is complete (:func:`_reveal`): the core finds the facts of a
#: product or a sum with an elliptic function numerically (an incomplete
#: ``elliptic_pi`` is a quadrature at 160 bits), at every step of the
#: accumulation otherwise; one dummy for each value
_FUNCTION_DUMMIES: dict[tuple[str, tuple[Expr, ...]], Dummy] = {}
_FUNCTION_VALUES: dict[Basic, Basic] = {}
_ELLIPTIC = (EllipticK, EllipticE, EllipticF, EllipticPi)


def _cached(function: Callable[..., Expr], args: tuple[Expr, ...]) -> Expr:
    """``function(*args)``, a dummy standing for it when it is an elliptic
    function of numbers."""
    key = (function.__name__, args)
    found = _FUNCTION_DUMMIES.get(key)
    if found is not None:
        return found
    value = function(*args)
    if not isinstance(value, _ELLIPTIC) or value.free_symbols:
        return value
    symbol = _fresh(function.__name__)
    _FUNCTION_DUMMIES[key] = symbol
    _FUNCTION_VALUES[symbol] = value
    return symbol


def _reveal(e: Expr) -> Expr:
    """``e`` with the dummies of :func:`_cached` replaced by the elliptic
    functions they stand for."""
    rule: dict[Basic, Basic] = {d: _FUNCTION_VALUES[d] for d in e.atoms(Dummy) if d in _FUNCTION_VALUES}
    if not rule:
        return e
    return _expr(e._xreplace_evaluated(rule))


def _forget_functions() -> None:
    """Drop the dummies of :func:`_cached` (between two integrals, when
    there are many)."""
    if len(_FUNCTION_VALUES) > 4096:
        _FUNCTION_DUMMIES.clear()
        _FUNCTION_VALUES.clear()

# ---------------------------------------------------------------------------
# The reduction to K, E and Pi

class _Radical:
    """The antiderivatives of ``s**j/sqrt(W(s))`` and of
    ``1/((alpha s + beta) sqrt(W(s)))`` for one radicand ``W``, normalised
    at ``s = 0``: for Legendre's ``W = s (1 - s)(1 - m s)`` they are the
    elliptic integrals, for ``W = s (1 - m s)`` they are elementary."""

    __slots__ = ('m',)

    def __init__(self, m: Expr) -> None:
        self.m: Final[Expr] = m

    def W(self, s: Expr) -> Expr:
        raise NotImplementedError

    def power(self, j: int, upper: Expr) -> Expr:
        raise NotImplementedError

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        raise NotImplementedError


class _Legendre(_Radical):
    """``W = s (1 - s)(1 - m s)``: ``ds/sqrt(W) = 2 dtheta/sqrt(1 - m sin(theta)**2)``
    with ``s = sin(theta)**2``."""

    __slots__ = ()

    def W(self, s: Expr) -> Expr:
        return s * (1 - s) * (1 - self.m * s)

    def power(self, j: int, upper: Expr) -> Expr:
        """``Integral(s**j/sqrt(W), (s, 0, upper)) = 2 J_j`` with
        ``J_j = Integral(sin(theta)**(2 j)/sqrt(1 - m sin(theta)**2), (theta, 0, phi))``,
        ``sin(phi)**2 == upper``: ``J_0 = F``, ``J_1 = (F - E)/m`` and
        the recurrence of Byrd–Friedman 310–318, which integrating
        ``d/dtheta (sin(theta)**(2j-1) cos(theta) sqrt(1 - m sin(theta)**2))``
        gives with its boundary term,

            (2j + 1) m J_{j+1} = 2j (1 + m) J_j - (2j - 1) J_{j-1}
                                 + sin(phi)**(2j-1) cos(phi) sqrt(1 - m sin(phi)**2)

        (the term vanishes for the complete integrals, ``phi = pi/2``).
        For a negative ``j`` the recurrence is read downwards and the
        result is the antiderivative regularised at ``phi = 0``: the
        boundary terms ``s**(j-1) sqrt(s (1 - s)(1 - m s))``,
        ``s = sin(phi)**2``, are the half-integer powers of ``s`` that
        diverge there, so the value at a point is the integral from
        ``0`` up to those powers, and the difference of two values is
        the integral between the points."""
        m = self.m
        if upper == 1:
            F = _cached(elliptic_k, (m,))
            E = _cached(elliptic_e, (m,))
        else:
            phi = asin(sqrt(upper))
            F, E = _cached(elliptic_f, (phi, m)), _cached(elliptic_e, (phi, m))
        # sin(phi)**(2k-1) cos(phi) sqrt(1 - m sin(phi)**2) = s**(k-1) sqrt(W(s))
        boundary = sqrt(self.W(upper))
        values: dict[int, Expr] = {0: F, 1: (F - E) / m}
        for k in range(1, j):
            values[k + 1] = (2 * k * (1 + m) * values[k] - (2 * k - 1) * values[k - 1]
                             + upper**(k - 1) * boundary) / ((2 * k + 1) * m)
        for k in range(0, j, -1):
            values[k - 1] = (2 * k * (1 + m) * values[k] - (2 * k + 1) * m * values[k + 1]
                             + upper**(k - 1) * boundary) / (2 * k - 1)
        return 2 * values[j]

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        """``c/(alpha s + beta) = (c/beta)/(1 - n s)`` with ``n = -alpha/beta``:
        the third kind, ``2 (c/beta) Pi(n; phi | m)``."""
        n = cancel(-alpha / beta)
        if upper == 1:
            return 2 * c / beta * _cached(elliptic_pi, (n, self.m))
        return 2 * c / beta * _cached(elliptic_pi, (n, asin(sqrt(upper)), self.m))


class _Elementary(_Radical):
    """``W = s (1 - m s)``, the radicand left by the odd part of the
    cosine map: the integrals are elementary."""

    __slots__ = ()

    def W(self, s: Expr) -> Expr:
        return s * (1 - self.m * s)

    def power(self, j: int, upper: Expr) -> Expr:
        """``I_j = Integral(s**j/sqrt(s (1 - m s)), (s, 0, u))``:
        ``I_0 = 2 asin(sqrt(m u))/sqrt(m)`` and, from
        ``d/ds (s**j sqrt(W)) = ((j + 1/2) s**j - (j + 1) m s**(j+1))/sqrt(W)``,
        ``(j + 1) m I_{j+1} = (j + 1/2) I_j - u**j sqrt(W(u))``, read
        downwards for the negative powers (regularised at ``0`` as in
        :meth:`_Legendre.power`)."""
        m = self.m
        boundary = sqrt(self.W(upper))
        values: dict[int, Expr] = {0: 2 * asin(sqrt(m * upper)) / sqrt(m)}
        for k in range(0, j):
            values[k + 1] = ((k + _HALF) * values[k] - upper**k * boundary) / ((k + 1) * m)
        for k in range(-1, j - 1, -1):
            values[k] = ((k + 1) * m * values[k + 1] + upper**k * boundary) / (k + _HALF)
        return values[j]

    def simple_pole(self, c: Expr, alpha: Expr, beta: Expr, upper: Expr, assumptions: Assumptions) -> Optional[Expr]:
        """``Integral(c/((alpha s + beta) sqrt(s (1 - m s))), (s, 0, u))``
        through ``t = 1/(s - p)``, ``p = -beta/alpha`` (outside the range):
        ``ds/((s - p) sqrt(W)) = -sign(s - p) dt/sqrt(A t**2 + B t + C)`` with
        ``A = W(p)``, ``B = W'(p)``, ``C = -m`` and ``B**2 - 4 A C = 1``, a
        logarithm when ``A > 0`` (``0 < p < 1/m``) and an arcsine when
        ``A < 0``. The value is an antiderivative continued through the
        pole (the sign of ``s - p`` is that at the point, the constant
        that at ``0``): it is used in differences of points on the same
        side."""
        m = self.m
        p = cancel(-beta / alpha)
        A = cancel(self.W(p))
        B = 1 - 2 * m * p
        # sign(s - p) at the point
        far = _ask(upper > p, assumptions)
        if far is None:
            return None
        sigma: Expr = Integer(1) if far else Integer(-1)
        # the primitive written at the point s itself: with t = 1/(s - p),
        # W quadratic gives A t**2 + B t - m = W(s)/(s - p)**2 exactly and
        # 2 A t + B = (2 A + B (s - p))/(s - p); at s = 0, a root of W, the
        # radical vanishes and 2 A + B (0 - p) = p, so the values at 0 are
        # exact (the bug: sqrt(W(0)) written as a nested radical which
        # SymPy does not see is 0 left log(-(-1 + 2 sqrt(A) epsilon)) for
        # the later rationalisation to trip over, giving log(-1) or zoo at
        # random from the random points of the factorisation inside it)
        W_upper = cancel(self.W(upper))
        radical: Expr = Integer(0) if W_upper == 0 else sqrt(W_upper)
        linear = 2 * A + B * (upper - p)
        if _ask(A > 0, assumptions) is True:
            # 0 < p < 1/m: (2 A t + B)**2 - 4 A (A t**2 + B t - m) = 1 makes
            # log|2 A t + B + 2 sqrt(A) sqrt(...)| the primitive, and 2 A t + B
            # has the sign of s - p: the primitive at s is
            # log((2 A + B (s - p) + 2 sigma sqrt(A) sqrt(W(s)))/(sigma (s - p)))
            # /sqrt(A), continued through the pole with sigma the sign of
            # s - p at the point, and 0 at s = 0
            at_upper = log((linear + 2 * sigma * sqrt(A) * radical) / (sigma * (upper - p))) / sqrt(A)
            at_zero: Expr = Integer(0)
        elif _ask(A < 0, assumptions) is True:
            # -asin(2 A t + B)/sqrt(-A), which is asin(1)/sqrt(-A) at s = 0
            at_upper = -asin(linear / (upper - p)) / sqrt(-A)
            at_zero = pi / (2 * sqrt(-A))
        else:
            return None
        return -sigma * c / alpha * (at_upper - at_zero)


def legendre_reduction(reduction: Reduction, assumptions: Assumptions = None) -> Optional[Expr]:
    """``Integral(Q(s)/sqrt(s*(1 - s)*(1 - m*s)), (s, lower, upper))`` in
    Legendre's integrals, from the partial fractions of ``Q``; ``None``
    for a divergent integral (a pole of ``Q`` inside the range or at an
    endpoint that is a root of the radicand).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Integer
    >>> from sympy_extras.integrals.elliptic import Reduction, legendre_reduction
    >>> s = Symbol('s', positive=True)
    >>> m = Symbol('m', positive=True)
    >>> legendre_reduction(Reduction(1, m, s, 0, 1))
    2*elliptic_k(m)
    >>> legendre_reduction(Reduction(s, m, s, 0, 1))
    2*(-elliptic_e(m) + elliptic_k(m))/m
    """
    Q, m, s = reduction.Q, reduction.m, reduction.s
    lower, upper = reduction.lower, reduction.upper
    if lower == 0 and _pole_at_zero(Q, s):
        return None
    total: Expr = Integer(0)
    radical = _Legendre(m)
    for bound, sign in ((upper, 1), (lower, -1)):
        value = _from_zero(Q, s, bound, radical, assumptions, lower, upper)
        if value is None:
            return None
        total = total + sign * value
    return _reveal(total)


def _pole_at_zero(Q: Expr, s: Symbol) -> bool:
    _, denominator = cancel(Q).as_numer_denom()
    return denominator.subs(s, 0) == 0


class _PartialFractions:
    """The partial fractions of a rational function of ``s`` whose
    denominator splits into linear factors: the polynomial part by
    degree, and for each factor ``alpha s + beta`` the numerators of its
    powers, ``{k: c}`` for ``c/(alpha s + beta)**k``."""

    __slots__ = ('polynomial', 'poles')

    def __init__(self, polynomial: dict[int, Expr], poles: list[tuple[Expr, Expr, dict[int, Expr]]]) -> None:
        self.polynomial: Final[dict[int, Expr]] = polynomial
        self.poles: Final[list[tuple[Expr, Expr, dict[int, Expr]]]] = poles


#: the partial fractions computed, by function and variable (the same
#: rational function is decomposed at both ends of every piece)
_FRACTIONS: dict[tuple[Expr, Symbol], Optional[_PartialFractions]] = {}


def _partial_fractions(Q: Expr, s: Symbol) -> Optional[_PartialFractions]:
    """:func:`_decomposed`, remembered."""
    key = (Q, s)
    if key in _FRACTIONS:
        return _FRACTIONS[key]
    found = _decomposed(Q, s)
    if len(_FRACTIONS) > 256:
        _FRACTIONS.clear()
    _FRACTIONS[key] = found
    return found


def _decomposed(Q: Expr, s: Symbol) -> Optional[_PartialFractions]:
    """SymPy's ``apart(Q, s)`` for a denominator which is a product of
    linear factors over the integers and the polynomial rings of the
    parameters (``None`` otherwise: the terms of an irreducible factor of
    higher degree are not reduced anyway): the numerator of
    ``1/L**(k - n)`` at a factor ``L = alpha s + beta`` of multiplicity
    ``k`` is the ``n``-th Taylor coefficient in ``L`` of ``Q L**k``, the
    polynomial part the quotient of the division of the numerator by
    the denominator."""
    numerator, denominator = cancel(Q).as_numer_denom()
    constants = _ConstantField([numerator, denominator], s)
    factored = constants.factor_list(denominator, s)
    if factored is None:
        factored = _factor_list(denominator, s)
    if factored is None:
        return None
    lead, factors = factored
    linear: list[tuple[Expr, Expr, int]] = []
    for factor_, multiplicity in factors:
        if not factor_.has(s):
            lead = lead * factor_**multiplicity
            continue
        coefficients = _coefficients(factor_, s)
        if coefficients is None or max(coefficients) != 1:
            return None
        linear.append((coefficients[1], coefficients.get(0, Integer(0)), multiplicity))
    numerator_coefficients = _coefficients(numerator, s)
    denominator_coefficients = _coefficients(denominator, s)
    if numerator_coefficients is None or denominator_coefficients is None:
        return None
    polynomial = {j: constants.canonical(c)
                  for j, c in _quotient(numerator_coefficients, denominator_coefficients).items()}
    poles: list[tuple[Expr, Expr, dict[int, Expr]]] = []
    u = _fresh('u')
    for i, (alpha, beta, k) in enumerate(linear):
        rest: Expr = lead
        for j, (alpha_j, beta_j, k_j) in enumerate(linear):
            if j != i:
                rest = rest * (alpha_j * s + beta_j)**k_j
        # s = (u - beta)/alpha: Q L**k = A(u)/B(u) with B(0) != 0
        at_u = (u - beta) / alpha
        A = _coefficients(cancel(_expr(numerator.subs(s, at_u))), u)
        B = _coefficients(cancel(_expr(rest.subs(s, at_u))), u)
        if A is None or B is None or 0 not in B:
            return None
        taylor: list[Expr] = []
        for n in range(k):
            term = A.get(n, Integer(0))
            for j in range(1, n + 1):
                term = term - B.get(j, Integer(0)) * taylor[n - j]
            taylor.append(constants.canonical(cancel(term / B[0])))
        powers = {k - n: c for n, c in enumerate(taylor) if c != 0}
        poles.append((alpha, beta, powers))
    return _PartialFractions(polynomial, poles)


def _quotient(numerator: dict[int, Expr], denominator: dict[int, Expr]) -> dict[int, Expr]:
    """The quotient of the division of two polynomials given by their
    coefficients, over the rational functions of the parameters."""
    remainder = dict(numerator)
    d = max(denominator)
    lead = denominator[d]
    quotient: dict[int, Expr] = {}
    while remainder and max(remainder) >= d:
        top = max(remainder)
        q = cancel(remainder[top] / lead)
        quotient[top - d] = q
        for j, c in denominator.items():
            degree = j + top - d
            value = cancel(remainder.get(degree, Integer(0)) - q * c)
            if value == 0 or degree == top:
                remainder.pop(degree, None)
            else:
                remainder[degree] = value
    return {j: c for j, c in quotient.items() if c != 0}


def _from_zero(Q: Expr, s: Symbol, point: Expr, radical: _Radical, assumptions: Assumptions,
               lower: Expr, upper: Expr) -> Optional[Expr]:
    """The antiderivative of ``Q(s)/sqrt(W(s))`` at ``s = point``,
    normalised to vanish at ``s = 0``: the integral from ``0`` when it
    converges, and with a pole of ``Q`` at ``0`` the regularised value
    of :meth:`_Radical.power`. The value is used in the difference of
    two points, ``lower`` and ``upper``: ``None`` when a pole of ``Q``
    other than ``0`` may lie between them (a divergent integral); a
    pole between ``0`` and ``lower`` is allowed, the antiderivatives
    continue through it with the same constant on the far side."""
    if point == 0:
        return Integer(0)
    parts = _partial_fractions(Q, s)
    if parts is None:
        return None
    total: Expr = Integer(0)
    for j, c in sorted(parts.polynomial.items()):
        total = total + c * radical.power(j, point)
    for alpha, beta, powers in parts.poles:
        for k, c in sorted(powers.items()):
            value = _pole_from_zero(c, alpha, beta, k, s, point, radical, assumptions, lower, upper)
            if value is None:
                return None
            total = total + value
    return total


def _pole_from_zero(c: Expr, alpha: Expr, beta: Expr, k: int, s: Symbol, upper: Expr, radical: _Radical,
                    assumptions: Assumptions, range_lower: Expr, range_upper: Expr) -> Optional[Expr]:
    """``c/(alpha s + beta)**k`` from :func:`_from_zero`: a multiple pole
    reduced to lower ones by integrating ``d/ds (sqrt(W)/D**(k-1))``,
    ``D = alpha s + beta``:
    ``Integral(N/(D**k sqrt(W)), (s, 0, u)) = 2 sqrt(W(u))/D(u)**(k-1)``
    with ``N = W' D - 2 (k - 1) alpha W``, whose leading partial fraction
    is ``N(p)/D**k`` at the pole ``p``."""
    if beta == 0:
        # c/(alpha s)**k: the negative powers, regularised at s = 0
        return c / alpha**k * radical.power(-k, upper)
    # no zero of D on [lower, upper]: D(lower) D(upper) > 0, a question
    # without division (alpha may vanish for symbolic parameters, when
    # the pole is at infinity)
    if _ask((alpha * range_lower + beta) * (alpha * range_upper + beta) > 0, assumptions) is not True:
        return None
    pole = -beta / alpha
    if k == 1:
        return radical.simple_pole(c, alpha, beta, upper, assumptions)
    W = radical.W(s)
    D = alpha * s + beta
    N = W.diff(s) * D - 2 * (k - 1) * alpha * W
    leading = cancel(_expr(N.subs(s, pole)))
    if leading == 0:
        return None
    rest = cancel((N - leading) / D**k)
    lower_orders = _from_zero(rest, s, upper, radical, assumptions, range_lower, range_upper)
    if lower_orders is None:
        return None
    boundary = 2 * sqrt(radical.W(upper)) / _expr(D.subs(s, upper))**(k - 1)
    return c / leading * (boundary - lower_orders)


# ---------------------------------------------------------------------------
# The entry point

def elliptic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` for ``f = R(x) * P(x)**(+-1/2)``, ``P`` a
    cubic or a quartic with real roots, with one pair of complex roots,
    or with two (a quartic without real roots, over any range of the
    line), in Legendre's elliptic integrals; also the trigonometric
    forms ``Q(sin(x)**2) * (1 - m sin(x)**2)**(+-1/2)`` over ``(0, phi)``.
    ``None`` when the integral is not of this kind, the roots cannot be
    ordered, the range crosses a root of ``P`` (cut it there first), the
    integral diverges at a pole of ``R``, or the range holds the mirror
    image of a pole of ``R`` (or of infinity) under the even-odd split
    of the complex cases without the pole itself.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sqrt, oo, S
    >>> from sympy_extras.integrals.elliptic import elliptic_integral
    >>> x = symbols('x')
    >>> elliptic_integral(1/sqrt(x*(1 - x)*(4 - x)), x, 0, 1)
    ConditionalValue(elliptic_k(1/4))
    >>> elliptic_integral(x/sqrt(x*(1 - x)*(4 - x)), x, 0, S.Half)
    ConditionalValue(-4*elliptic_e(pi/4, 1/4) + 4*elliptic_f(pi/4, 1/4))
    >>> elliptic_integral(1/sqrt((x**2 + 1)*(x + 2)*(3 - x)), x, -2, 3)
    ConditionalValue(2**(3/4)*sqrt(5)*elliptic_k(sqrt(2)/4 + 1/2)/5)
    >>> elliptic_integral(1/sqrt((x**2 + 1)*(x**2 + 4)), x, -oo, oo)
    ConditionalValue(elliptic_k(3/4))
    """
    f_, a_, b_ = to_expr(f), to_expr(a), to_expr(b)
    _forget_functions()
    if f_.has(*_TRIGONOMETRIC):
        return _trigonometric(f_, x, a_, b_, assumptions)
    return _reduce(f_, x, a_, b_, assumptions, 0)


def _reduce(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions, depth: int) -> Optional[ConditionalValue]:
    if depth > _MAX_DEPTH:
        return None
    constant, rest = f.as_independent(x, as_Add=False)
    found = radicand(rest, x)
    if found is None:
        return None
    R, P, e = found
    # an even radicand first: its own roots may be complex (1 - x**4)
    even = _even_reduction(R, P, e, x, a, b, assumptions, depth)
    if even is not None:
        return even.scaled(constant)
    roots_ = real_roots(P, x, assumptions)
    if roots_ is None:
        complex_ = _complex_reduction(R, P, e, x, a, b, assumptions, depth)
        if complex_ is None:
            complex_ = _two_pairs_reduction(R, P, e, x, a, b, assumptions)
        return None if complex_ is None else complex_.scaled(constant)
    lo_cell, hi_cell = _position(a, roots_, assumptions), _position(b, roots_, assumptions)
    if lo_cell is None or hi_cell is None:
        return None
    cell = lo_cell if lo_cell % 2 else lo_cell + 1
    if hi_cell not in (cell, cell + 1):
        return None
    n = len(roots_)
    if n == 4 and cell in (-1, 2 * n - 1):
        inverted = _inverted_reduction(R, P, e, x, roots_, a, b, cell, assumptions, depth)
        return None if inverted is None else inverted.scaled(constant)
    reduction = _reduce_on_cell(R, P, e, x, roots_, a, b, cell, assumptions)
    if reduction is None:
        return None
    value = legendre_reduction(reduction, assumptions)
    if value is None:
        return None
    return ConditionalValue(constant * value)


def _even_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions,
                    depth: int) -> Optional[ConditionalValue]:
    """``y = x**2`` for an even radicand and an even ``R`` on a range of
    nonnegative numbers: ``R(x) P(x)^e dx = R(sqrt(y)) (y P(sqrt(y)))^e
    y^(-e-1/2) dy / 2``, the new radicand being a cubic."""
    if P.subs(x, -x) != P or R.subs(x, -x) != R:
        return None
    if isinstance(a, NegativeInfinity) or _ask(a >= 0, assumptions) is not True:
        return None
    y = _fresh('y', positive=True)
    R_y = cancel(_expr(R.subs(x, sqrt(y))))
    P_y = cancel(_expr(P.subs(x, sqrt(y))))
    if not R_y.is_rational_function(y) or not P_y.is_polynomial(y):
        return None
    if e == -_HALF:
        # dx / sqrt(P) = dy / (2 sqrt(y P(sqrt y)))
        g = R_y * (y * P_y)**e / 2
    else:
        # sqrt(P) dx = sqrt(y P(sqrt y)) dy / (2 y)
        g = R_y * (y * P_y)**e / (2 * y)
    return _reduce(g, y, a**2, oo if isinstance(b, Infinity) else b**2, assumptions, depth + 1)


def _inverted_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, roots_: Sequence[Expr], a: Expr, b: Expr,
                        cell: int, assumptions: Assumptions, depth: int,
                        factors: Optional[_ComplexFactors] = None) -> Optional[ConditionalValue]:
    """A quartic beyond its extreme root through ``u = 1/(x - r)``, which
    leaves a cubic in ``u`` over ``(0, oo)`` (or a part of it). With the
    ``factors`` of a radicand with a complex pair the cubic is built from
    them, ``lam (r - r') C (u + 1/(r - r')) ((u - p')**2 + q'**2)`` with
    ``C = (r - p)**2 + q**2``, ``p' = -(r - p)/C``, ``q'**2 = q**2/C**2``
    (``cancel`` does not read the relation of an algebraic root)."""
    u = _fresh('u', positive=True)
    upper_cell = cell == 2 * len(roots_) - 1
    r = roots_[-1] if upper_cell else roots_[0]
    X = r + 1 / u if upper_cell else r - 1 / u
    factors_u: Optional[_ComplexFactors] = None
    if factors is None:
        # dx = -du/u**2, P(r + 1/u) = u**(-4) P_u(u)
        P_u = cancel(_expr(P.subs(x, X)) * u**4)
    else:
        other = factors.roots[0] if upper_cell else factors.roots[1]
        # (x - other) = ((r - other) u +- 1)/u, (x - p)**2 + q**2 = C ((u - p')**2 + q'**2)/u**2
        C = (r - factors.p)**2 + factors.q2
        p_u = -(r - factors.p) / C if upper_cell else (r - factors.p) / C
        q2_u = factors.q2 / C**2
        root_u = -1 / (r - other)
        lam_u = factors.lam * (r - other) * C
        P_u = lam_u * (u - root_u) * ((u - p_u)**2 + q2_u)
        factors_u = _ComplexFactors(lam_u, [root_u], p_u, q2_u, factors.facts)
    R_u = cancel(_expr(R.subs(x, X)))
    g = R_u * P_u**e * u**(-4 * e - 2)
    if upper_cell:
        # x from a to b means u from 1/(b - r) down to 1/(a - r): the sign of dx flips the bounds
        lower: Expr = Integer(0) if isinstance(b, Infinity) else 1 / (b - r)
        upper: Expr = oo if a == r else 1 / (a - r)
    else:
        lower = Integer(0) if isinstance(a, NegativeInfinity) else 1 / (r - a)
        upper = oo if b == r else 1 / (r - b)
    if not P_u.is_polynomial(u):
        return None
    if factors_u is not None:
        # straight to the cosine map: re-reading the radicand would split
        # the positive constant of the leading coefficient off P_u, which
        # the factors carry already
        return _complex_reduction(R_u * u**(-4 * e - 2), P_u, e, u, lower, upper, assumptions, depth + 1,
                                  factors_u)
    return _reduce(g, u, lower, upper, assumptions, depth + 1)


# ---------------------------------------------------------------------------
# Two complex roots

class _ComplexFactors:
    """``P = lam * prod(x - r) * ((x - p)**2 + q2)`` with one or two real
    roots ``r`` in increasing order and ``q2 > 0``; ``facts`` are the
    values of the dummies standing for algebraic numbers (see
    :func:`_ask`)."""

    __slots__ = ('lam', 'roots', 'p', 'q2', 'facts')

    def __init__(self, lam: Expr, roots_: list[Expr], p: Expr, q2: Expr,
                 facts: Optional[list[Boolean]] = None) -> None:
        self.lam: Final[Expr] = lam
        self.roots: Final[list[Expr]] = roots_
        self.p: Final[Expr] = p
        self.q2: Final[Expr] = q2
        self.facts: Final[list[Boolean]] = [] if facts is None else facts


#: the size (in characters of the printed form, with the algebraic
#: numbers as dummies) beyond which a value in indexed roots is not built:
#: the printed value would run to hundreds of kilobytes
_ROOTOF_SIZE = 2500


def _printed_size(e: Expr) -> int:
    """The length of the printed form of ``e`` with the dummies under the
    names the old module gave them (without the numbers of :func:`_fresh`)."""
    rule: dict[Basic, Basic] = {d: Dummy(d.name.rstrip('0123456789'))
                                for d in e.atoms(Dummy) if isinstance(d, Dummy)}
    return len(str(e.xreplace(rule)))


def _restore(value: Expr, radicals: dict[Symbol, Expr], facts: Sequence[Boolean]) -> Optional[Expr]:
    """The value with the radicals and the dummies of algebraic numbers
    replaced by their expressions; ``None`` for a value in indexed roots
    beyond :data:`_ROOTOF_SIZE`. A value in indexed roots is built
    without evaluation (as the old module built SymPy's ``CRootOf``
    values, whose assumptions on every sum and power cost a minute of
    ``evalf`` per kilobyte and add nothing to a value which is exact as
    it stands)."""
    values, _ = _numeric_values(list(facts))
    rule: dict[Basic, Basic] = {s: v for s, v in radicals.items()}
    if any(v.has(IndexedRoot) for v in values.values()):
        timeout = settings.timeout
        compact = attempt(lambda: radsimp(value), timeout / 8 if timeout else None, native=True)
        if compact is not None and _printed_size(compact) < _printed_size(value):
            value = compact
        if _printed_size(value) > _ROOTOF_SIZE:
            return None
        return _numeric(_expr(value.xreplace(rule)), values, evaluated=False)
    return _numeric(_expr(value._xreplace_evaluated(rule)), values)


def _complex_factors(P: Expr, x: Symbol, assumptions: Assumptions) -> Optional[_ComplexFactors]:
    """The factors of a cubic or a quartic with exactly one pair of
    complex roots, read from the factorisation (a numeric polynomial
    whose quadratic factor is not rational through its roots)."""
    factored = _factor_list(P, x)
    if factored is None:
        return None
    coefficient, factors = factored
    lam = coefficient
    roots_: list[Expr] = []
    quadratic: Optional[tuple[Expr, Expr]] = None
    pair: Optional[tuple[Expr, Expr]] = None
    facts: list[Boolean] = []
    for factor_, multiplicity in factors:
        if not factor_.has(x):
            lam = lam * factor_**multiplicity
            continue
        if multiplicity != 1:
            return None
        coefficients = _coefficients(factor_, x)
        if coefficients is None:
            return None
        degree = max(coefficients)
        lead = coefficients[degree]
        if degree == 1:
            roots_.append(cancel(-coefficients.get(0, Integer(0)) / lead))
        elif degree == 2 and quadratic is None:
            c1, c0 = coefficients.get(1, Integer(0)) / lead, coefficients.get(0, Integer(0)) / lead
            quadratic = (cancel(c1), cancel(c0))
        elif degree in (3, 4) and quadratic is None and pair is None and not roots_ \
                and not factor_.free_symbols - {x}:
            # an irreducible numeric factor: its real roots and the
            # quadratic of its conjugate pair, as dummies with values
            numeric = _numeric_factors(cancel(factor_ / lead), x)
            if numeric is None or len(numeric.pairs) != 1:
                return None
            roots_.extend(numeric.reals)
            pair = numeric.pairs[0]
            facts = numeric.facts
        else:
            return None
        lam = lam * lead
    if quadratic is not None:
        c1, c0 = quadratic
        p = -c1 / 2
        pair = (p, cancel(c0 - p**2))
    if pair is None or len(roots_) not in (1, 2):
        return None
    p, q2 = pair
    if _ask(q2 > 0, _with_facts(assumptions, facts)) is not True:
        return None
    if len(roots_) == 2:
        below = _less(roots_[0], roots_[1], _with_facts(assumptions, facts))
        if below is None:
            return None
        if not below:
            roots_.reverse()
    return _ComplexFactors(cancel(lam), roots_, p, q2, facts)


class _CosineMap:
    """The cell mapped onto ``t = cos(theta)`` in ``(-1, 1)``: ``x = X(t)``,
    ``P(X(t)) = lam * X'(t)**2 * (1 - t**2) * (1 - m + m t**2) / g2``
    with ``sigma`` the sign of ``X'`` on the range and ``preimage`` the
    inverse map (``-1`` at the infinite end of a half-line). Symbolic
    radicals ``A = sqrt((a - p)**2 + q**2)`` are positive symbols in
    the map, so that the sign questions are polynomial: the values are
    restored in the result."""

    __slots__ = ('X', 'm', 'g2', 'sigma', 't', '_preimage', '_infinite', 'radicals', 'facts')

    def __init__(self, X: Expr, m: Expr, g2: Expr, sigma: Expr, t: Symbol, preimage: Expr,
                 infinite: Optional[Expr], radicals: dict[Symbol, Expr], facts: list[Boolean]) -> None:
        self.X: Final[Expr] = X
        self.m: Final[Expr] = m
        self.g2: Final[Expr] = g2
        self.sigma: Final[Expr] = sigma
        self.t: Final[Symbol] = t
        self._preimage: Final[Expr] = preimage
        self._infinite: Final[Optional[Expr]] = infinite
        #: the symbols standing for the radicals A, B of symbolic
        #: parameters, and what is known about them (the triangle
        #: inequalities of the distances to the complex root)
        self.radicals: Final[dict[Symbol, Expr]] = radicals
        self.facts: Final[list[Boolean]] = facts

    def preimage(self, y: Expr, x: Symbol) -> Expr:
        if y == self._infinite:
            return Integer(-1)
        return cancel(_expr(self._preimage.subs(x, y)))


def _cosine_map(factors: _ComplexFactors, cell: int, t: Symbol, x: Symbol) -> Optional[_CosineMap]:
    """Byrd–Friedman 241 for the cell between the real roots of a quartic
    and 240 for the half-lines of a cubic."""
    p, q2 = factors.p, factors.q2
    radicals: dict[Symbol, Expr] = {}
    facts: list[Boolean] = []
    if len(factors.roots) == 2:
        if cell != 1:
            return None
        b, a = factors.roots
        A = _radical((a - p)**2 + q2, radicals, facts)
        B = _radical((b - p)**2 + q2, radicals, facts)
        if radicals:
            # the sides a - b, A, B of the triangle with the complex root
            facts.extend([(a - b)**2 > (A - B)**2, (a - b)**2 < (A + B)**2])
        X = (a * B + b * A - t * (a * B - b * A)) / (A + B + t * (A - B))
        m = cancel(((a - b)**2 - (A - B)**2) / (4 * A * B))
        preimage = (a * B + b * A - x * (A + B)) / (a * B - b * A + x * (A - B))
        return _CosineMap(X, m, 1 / (A * B), Integer(-1), t, preimage, None, radicals, facts)
    a = factors.roots[0]
    A = _radical((a - p)**2 + q2, radicals, facts)
    if radicals:
        facts.append(A**2 > (a - p)**2)
    if cell == 1:
        # (a, oo): x = a + A (1 - t)/(1 + t), decreasing from oo to a
        r = (x - a) / A
        return _CosineMap(a + A * (1 - t) / (1 + t), cancel((A - (a - p)) / (2 * A)), 1 / A,
                          Integer(-1), t, (1 - r) / (1 + r), oo, radicals, facts)
    if cell == -1:
        # (-oo, a): x = a - A (1 - t)/(1 + t), increasing from -oo to a
        r = (a - x) / A
        return _CosineMap(a - A * (1 - t) / (1 + t), cancel((A + (a - p)) / (2 * A)), 1 / A,
                          Integer(1), t, (1 - r) / (1 + r), -oo, radicals, facts)
    return None


def _radical(square: Expr, radicals: dict[Symbol, Expr], facts: list[Boolean]) -> Expr:
    """``sqrt(square)``: a positive symbol recorded in ``radicals`` when
    the square has symbols, its value among the ``facts`` (for the
    numeric decisions of :func:`_ask`)."""
    if not square.free_symbols:
        return sqrt(square)
    symbol = _fresh('A', positive=True)
    radicals[symbol] = sqrt(square)
    facts.append(Eq(symbol, sqrt(square)))
    return symbol


def _parity_parts(G: Expr, t: Symbol, w: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(G_e, G_o)`` with ``G(t) == G_e(t**2) + t G_o(t**2)``."""
    numerator, denominator = cancel(G).as_numer_denom()
    if not numerator.is_polynomial(t) or not denominator.is_polynomial(t):
        return None
    mirrored = _expr(denominator.subs(t, -t))
    numerator = expand(numerator * mirrored)
    denominator = expand(denominator * mirrored)
    even = expand((numerator + _expr(numerator.subs(t, -t))) / 2)
    odd = cancel((numerator - _expr(numerator.subs(t, -t))) / (2 * t))
    parts: list[Expr] = []
    for polynomial in (even, odd, denominator):
        coefficients = _coefficients(polynomial, t)
        if coefficients is None:
            return None
        converted: Expr = Integer(0)
        for k, c in sorted(coefficients.items()):
            if k % 2:
                return None
            converted = converted + c * w**(k // 2)
        parts.append(converted)
    return (cancel(parts[0] / parts[2]), cancel(parts[1] / parts[2]))


def _complex_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions,
                       depth: int, factors: Optional[_ComplexFactors] = None) -> Optional[ConditionalValue]:
    """The integral over a part of a cell of a radicand with two complex
    roots, through the cosine map: the even part of the rational
    prefactor in ``s = 1 - t**2`` is a Legendre form, the odd part is
    elementary in ``s``."""
    if factors is None:
        factors = _complex_factors(P, x, assumptions)
    if factors is None:
        return None
    roots_ = factors.roots
    assumptions = _with_facts(assumptions, factors.facts)
    lo_cell, hi_cell = _position(a, roots_, assumptions), _position(b, roots_, assumptions)
    if lo_cell is None or hi_cell is None:
        return None
    cell = lo_cell if lo_cell % 2 else lo_cell + 1
    if hi_cell not in (cell, cell + 1):
        return None
    if len(roots_) == 2 and cell in (-1, 3):
        numeric = factors if factors.facts else None
        return _inverted_reduction(R, P, e, x, roots_, a, b, cell, assumptions, depth, numeric)
    # the sign of P on the cell: lam (x - r1)(x - r2) q(x) between the
    # roots, lam (x - a) q(x) beyond the root of a cubic
    lam = factors.lam if cell == 2 * len(roots_) - 1 else -factors.lam
    if _ask(lam > 0, assumptions) is not True:
        return None
    t = _fresh('t')
    mapping = _cosine_map(factors, cell, t, x)
    if mapping is None:
        return None
    X, m = mapping.X, mapping.m
    assumptions = _with_facts(assumptions, mapping.facts)
    t0, t1 = mapping.preimage(a, x), mapping.preimage(b, x)
    # P(X)^e = (lam/g2)^e (sigma X')^(2e) ((1 - t^2)(1 - m + m t^2))^e, the
    # rest of the integrand is R(X) X' dt
    Xp = X.diff(t)
    root = sqrt(lam / mapping.g2)
    R_X = _expr(R.subs(x, X))
    if e == -_HALF:
        G = R_X * Xp / (root * mapping.sigma * Xp)
    else:
        G = R_X * Xp * root * mapping.sigma * Xp * (1 - t**2) * (1 - m + m * t**2)
    G = cancel(G)
    if not G.is_rational_function(t):
        return None
    parts = _CosineParts(G, t, m, assumptions)
    value = parts.between(t0, t1)
    if value is None:
        return None
    restored = _restore(_reveal(value), mapping.radicals, factors.facts + mapping.facts)
    return None if restored is None else ConditionalValue(restored)


class _CosineParts:
    """``Integral(G(t)/sqrt((1 - t**2)(1 - m + m t**2)), (t, t0, t1))`` for
    ``-1 <= t0 < t1 <= 1`` from the even and the odd part of ``G``,
    ``G(t) = G_e(t**2) + t G_o(t**2)``, in ``s = 1 - t**2``: on a range of
    one sign ``sigma``, with ``u = |t|`` running from ``u0`` to ``u1``,

        sigma (F_e(1 - u0**2) - F_e(1 - u1**2)) + F_o(1 - u0**2) - F_o(1 - u1**2)

    with ``F_e`` the antiderivative of ``G_e(1 - s)/(2 sqrt(s (1 - s)(1 - m s)))``
    and ``F_o`` that of ``G_o(1 - s)/(2 sqrt(s (1 - m s)))``, elementary,
    both from :func:`_from_zero`; a range of both signs is cut at
    ``t = 0``. When ``G`` has a pole at ``t = -1`` (the image of the
    infinite end of a half-line) both parts diverge at ``s = 0`` and
    only their sum is finite at the other end ``t = 1``: the values at
    ``0`` are then the regularised ones, which drop the same
    half-integer powers of ``s`` from both. At the end where ``G``
    itself has the pole the integral diverges."""

    __slots__ = ('m', 'assumptions', 's', 'valid', 'Q_e', 'Q_o', 'poles')

    def __init__(self, G: Expr, t: Symbol, m: Expr, assumptions: Assumptions) -> None:
        self.m: Final[Expr] = m
        self.assumptions: Final[Assumptions] = assumptions
        s = _fresh('s', positive=True)
        self.s: Final[Symbol] = s
        w = _fresh('w', positive=True)
        parts = _parity_parts(G, t, w)
        self.valid: Final[bool] = parts is not None
        Q_e: Expr = Integer(0)
        Q_o: Expr = Integer(0)
        if parts is not None:
            Q_e = cancel(_expr(parts[0].subs(w, 1 - s)) / 2)
            Q_o = cancel(_expr(parts[1].subs(w, 1 - s)) / 2)
        self.Q_e: Final[Expr] = Q_e
        self.Q_o: Final[Expr] = Q_o
        _, denominator = cancel(G).as_numer_denom()
        #: the ends of the cell where G has a pole, and the integral diverges
        self.poles: Final[list[Expr]] = [end for end in (Integer(1), Integer(-1))
                                         if denominator.subs(t, end) == 0]

    def between(self, t0: Expr, t1: Expr) -> Optional[Expr]:
        if not self.valid or t0 in self.poles or t1 in self.poles:
            return None
        if {t0, t1} == {Integer(1), Integer(-1)}:
            # the whole cell: twice the complete integral of the even part
            value = self._piece(Integer(0), Integer(1), Integer(1), even_only=True)
            return None if value is None else 2 * t1 * value
        signs = [self._sign(t0), self._sign(t1)]
        first_sign, second_sign = signs
        if first_sign is None or second_sign is None:
            return None
        if first_sign == 0 or second_sign == 0 or first_sign == second_sign:
            sigma: Expr = Integer(1) if Integer(1) in signs else Integer(-1)
            return self._piece(sigma * t0, sigma * t1, sigma)
        # a range of both signs, cut at t = 0
        first = self._piece(first_sign * t0, Integer(0), first_sign)
        second = self._piece(Integer(0), second_sign * t1, second_sign)
        if first is None or second is None:
            return None
        return first + second

    def _sign(self, tau: Expr) -> Optional[Expr]:
        if tau == 0:
            return Integer(0)
        if _ask(tau > 0, self.assumptions) is True:
            return Integer(1)
        if _ask(tau < 0, self.assumptions) is True:
            return Integer(-1)
        return None

    def _piece(self, u0: Expr, u1: Expr, sigma: Expr, even_only: bool = False) -> Optional[Expr]:
        """The integral over ``t`` from ``sigma u0`` to ``sigma u1``, with
        ``u0, u1`` in ``[0, 1]`` (of the even part alone when asked)."""
        s0, s1 = cancel(1 - u0**2), cancel(1 - u1**2)
        if s0 == s1:
            return Integer(0)
        increasing = _ask(s0 < s1, self.assumptions)
        if increasing is None:
            return None
        lower, upper = (s0, s1) if increasing else (s1, s0)
        total: Expr = Integer(0)
        pieces: tuple[tuple[Expr, _Radical, Expr, bool], ...] = (
            (self.Q_e, _Legendre(self.m), sigma, True), (self.Q_o, _Elementary(self.m), Integer(1), False))
        for Q, radical, weight, legendre in pieces:
            if Q == 0 or (even_only and not legendre):
                continue
            at_s0 = _from_zero(Q, self.s, s0, radical, self.assumptions, lower, upper)
            at_s1 = _from_zero(Q, self.s, s1, radical, self.assumptions, lower, upper)
            if at_s0 is None or at_s1 is None:
                return None
            total = total + weight * (at_s0 - at_s1)
        return total


# ---------------------------------------------------------------------------
# Four complex roots

class _Numeric:
    """The roots of a numeric polynomial as dummies: the real roots
    (increasing), the pairs ``(p, q**2)`` of the conjugate complex roots
    ``p +- I q``, and the ``facts`` giving their values (indexed roots or
    radicals: the real roots, ``re(r)`` and ``im(r)**2``); the polynomial
    machinery is kept away from the algebraic numbers, whose minimal
    polynomials it would otherwise compute, and the questions are
    decided numerically by :func:`_ask`."""

    __slots__ = ('reals', 'pairs', 'facts')

    def __init__(self, reals: list[Expr], pairs: list[tuple[Expr, Expr]], facts: list[Boolean]) -> None:
        self.reals: Final[list[Expr]] = reals
        self.pairs: Final[list[tuple[Expr, Expr]]] = pairs
        self.facts: Final[list[Boolean]] = facts


#: the working precision (bits) of the numerical roots of a numeric radicand
_PRECISION = 200


def _numeric_factors(P: Expr, x: Symbol) -> Optional[_Numeric]:
    """The roots of the monic numeric polynomial ``P`` (rational
    coefficients, squarefree): the real ones exact, the conjugate pairs
    in the increasing order of their real parts (SymPy's ``all_roots``)
    numerically, their real parts and moduli made exact by
    :func:`_algebraic_parts`."""
    dense = _rational_coefficients(P, x)
    if dense is None:
        return None
    exact_reals = _real_roots_rational(P, x)
    if exact_reals is None:
        return None
    degree = len(dense) - 1
    with mpmath.workprec(_PRECISION):
        try:
            numerical = mpmath.polyroots([mpmath.mpf(c.numerator) / c.denominator for c in dense],
                                         maxsteps=200, extraprec=2 * _PRECISION)
        except mpmath.libmp.NoConvergence:
            return None
        upper = sorted((mpmath.mpc(r) for r in numerical if mpmath.im(r) > 0), key=lambda r: float(mpmath.re(r)))
    if len(exact_reals) + 2 * len(upper) != degree:
        return None
    reals: list[Expr] = []
    pairs: list[tuple[Expr, Expr]] = []
    facts: list[Boolean] = []
    for r in exact_reals:
        if isinstance(r, Rational):
            reals.append(r)
            continue
        c = _fresh('r', real=True)
        facts.append(Eq(c, r))
        reals.append(c)
    for z in upper:
        parts = _algebraic_parts(P, x, z)
        if parts is None:
            return None
        real_part, modulus2 = parts
        p, q2 = _fresh('p', real=True), _fresh('q2', positive=True)
        facts.append(Eq(p, real_part))
        facts.append(Eq(q2, modulus2 - real_part**2))
        pairs.append((p, q2))
    return _Numeric(reals, pairs, facts)


def _algebraic_parts(P: Expr, x: Symbol, r: mpmath.mpc) -> Optional[tuple[Expr, Expr]]:
    """``(re(r), |r|**2)`` of a complex root ``r`` of the numeric ``P`` as
    real algebraic numbers of their own minimal polynomials, from the
    resultants whose roots are the half sums ``(r_i + r_j)/2`` and the
    products ``r_i r_j`` of the roots, the right root picked numerically
    at forty digits (the roots of the squarefree resultants are
    distinct); ``None`` when this fails (the old module kept ``re(r)``
    and ``re(r)**2 + im(r)**2`` of a ``CRootOf`` then, which the core has
    not got)."""
    y, z = _fresh('y'), _fresh('z')
    n = _degree(P, x)
    if n is None:
        return None
    P_y = _expr(P.subs(x, y))
    images = (_expr(P.subs(x, 2 * z - y)), expand(y**n * _expr(P.subs(x, z / y))))
    with mpmath.workprec(_PRECISION):
        targets = (mpmath.re(r), mpmath.re(r)**2 + mpmath.im(r)**2)
    found: list[Expr] = []
    for image, target in zip(images, targets):
        try:
            combined = resultant(P_y, image, y)
            _, squarefree = sqf_list(combined, z)
        except (BasePolynomialError, NotImplementedError, ValueError, TypeError):
            return None
        part: Expr = Integer(1)
        for base, _multiplicity in squarefree:
            part = part * base
        candidates = _real_roots_rational(expand(part), z)
        if not candidates:
            return None
        distances: list[tuple[mpmath.mpf, int]] = []
        for i, c in enumerate(candidates):
            value = numeric_value(c, _PRECISION)
            if value is None:
                return None
            with mpmath.workprec(_PRECISION):
                distances.append((abs(value - target), i))
        distances.sort()
        if len(distances) > 1 and distances[1][0] < mpmath.mpf(10)**(-20):
            return None
        if distances[0][0] > mpmath.mpf(10)**(-30):
            return None
        found.append(candidates[distances[0][1]])
    return (found[0], found[1])


class _TwoPairs:
    """``P = lam * ((x - p1)**2 + q1**2) * ((x - p2)**2 + q2**2)``, with
    the values of the dummies of algebraic numbers in ``facts``."""

    __slots__ = ('lam', 'pairs', 'facts')

    def __init__(self, lam: Expr, pairs: list[tuple[Expr, Expr]], facts: list[Boolean]) -> None:
        self.lam: Final[Expr] = lam
        self.pairs: Final[list[tuple[Expr, Expr]]] = pairs
        self.facts: Final[list[Boolean]] = facts


def _two_pairs(P: Expr, x: Symbol, assumptions: Assumptions) -> Optional[_TwoPairs]:
    """A quartic with no real roots as the product of two positive
    definite quadratics (rational ones from the factorisation, numeric
    irrational ones from the conjugate pairs of roots)."""
    factored = _factor_list(P, x)
    if factored is None:
        return None
    coefficient, factors = factored
    lam = coefficient
    pairs: list[tuple[Expr, Expr]] = []
    facts: list[Boolean] = []
    for factor_, multiplicity in factors:
        if not factor_.has(x):
            lam = lam * factor_**multiplicity
            continue
        if multiplicity != 1:
            return None
        coefficients = _coefficients(factor_, x)
        if coefficients is None:
            return None
        degree = max(coefficients)
        lead = coefficients[degree]
        lam = lam * lead
        if degree == 2:
            c1 = cancel(coefficients.get(1, Integer(0)) / lead)
            c0 = cancel(coefficients.get(0, Integer(0)) / lead)
            p = -c1 / 2
            pairs.append((p, cancel(c0 - p**2)))
        elif degree == 4 and not factor_.free_symbols - {x}:
            numeric = _numeric_factors(cancel(factor_ / lead), x)
            if numeric is None or numeric.reals:
                return None
            pairs.extend(numeric.pairs)
            facts.extend(numeric.facts)
        else:
            return None
    if len(pairs) != 2:
        return None
    for _, q2 in pairs:
        if _ask(q2 > 0, _with_facts(assumptions, facts)) is not True:
            return None
    return _TwoPairs(cancel(lam), pairs, facts)


class _TangentMap:
    """The real line mapped onto ``theta`` in ``(-pi/2, pi/2)``: ``x = X(t)``
    with ``t = lambda tan(theta)``, chosen so that
    ``P(X(t)) dt**2 = lam B1 B2 (1 - m sin(theta)**2) X'(t)**2 dt**2 / cos(theta)**4``
    (Byrd–Friedman 267). ``X`` is the bilinear map ``(beta + alpha t)/(1 + t)``
    with the pencil parameters ``alpha > beta`` for which both quadratics
    become ``A_i t**2 + B_i`` (up to ``(1 + t)**2``), or the shift
    ``p + t`` when the quadratics share their centre; ``pole`` is the
    ``x`` sent to ``t = oo`` (``None`` for the shift), ``infinite`` the
    ``t`` of ``x = +-oo``, ``lam_`` the scale ``lambda``."""

    __slots__ = ('X', 'Xp', '_preimage', 'pole', 'infinite', 'lam_', 'm', 'B1B2', 'bilinear', 'radicals', 'facts')

    def __init__(self, X: Expr, Xp: Expr, preimage: Expr, pole: Optional[Expr], infinite: Expr,
                 lam_: Expr, m: Expr, B1B2: Expr, bilinear: bool, radicals: dict[Symbol, Expr],
                 facts: list[Boolean]) -> None:
        self.X: Final[Expr] = X
        self.Xp: Final[Expr] = Xp
        self._preimage: Final[Expr] = preimage
        self.pole: Final[Optional[Expr]] = pole
        self.infinite: Final[Expr] = infinite
        self.lam_: Final[Expr] = lam_
        self.m: Final[Expr] = m
        self.B1B2: Final[Expr] = B1B2
        self.bilinear: Final[bool] = bilinear
        self.radicals: Final[dict[Symbol, Expr]] = radicals
        self.facts: Final[list[Boolean]] = facts

    def preimage(self, y: Expr, x: Symbol) -> Expr:
        if isinstance(y, (Infinity, NegativeInfinity)):
            return self.infinite if self.bilinear else y
        return cancel(_expr(self._preimage.subs(x, y)))


def _named(value: Expr, facts: list[Boolean], numeric: bool) -> Expr:
    """``value`` itself, or with ``numeric`` a positive dummy standing for
    it with its value among the facts: the algebraic numbers of a numeric
    radicand are decided numerically anyway, and the named quantities keep
    the result compact."""
    if not numeric or not value.free_symbols:
        return value
    symbol = _fresh('c', positive=True)
    facts.append(Eq(symbol, value))
    return symbol


def _tangent_map(pairs: list[tuple[Expr, Expr]], t: Symbol, x: Symbol,
                 assumptions: Assumptions, numeric: bool = False) -> Optional[_TangentMap]:
    (p1, q1), (p2, q2) = pairs
    radicals: dict[Symbol, Expr] = {}
    facts: list[Boolean] = []
    A: list[Expr]
    B: list[Expr]
    pole: Optional[Expr]
    if cancel(p1 - p2) == 0:
        # (t**2 + q1**2)(t**2 + q2**2): A_i = 1, B_i = q_i**2
        X = p1 + t
        A = [Integer(1), Integer(1)]
        B = [q1, q2]
        preimage = x - p1
        Xp: Expr = Integer(1)
        pole = None
        infinite: Expr = oo
        bilinear = False
    else:
        # (beta - p_i)(alpha - p_i) = -q_i**2: alpha + beta = u, alpha beta = v
        c1, c2 = p1**2 + q1, p2**2 + q2
        u = cancel((c1 - c2) / (p1 - p2))
        v = cancel(p1 * u - c1)
        d = _radical(cancel(u**2 - 4 * v), radicals, facts)
        if radicals:
            facts.append(d**2 > (u - 2 * p1)**2)
        alpha, beta = (u + d) / 2, (u - d) / 2
        if numeric:
            # alpha > beta named, alpha - beta = d kept
            alpha_, beta_ = _fresh('alpha', real=True), _fresh('beta', real=True)
            facts.extend([Eq(alpha_, alpha), Eq(beta_, beta)])
            alpha, beta = alpha_, beta_
        X = (beta + alpha * t) / (1 + t)
        A = [_named((alpha - p1)**2 + q1, facts, numeric), _named((alpha - p2)**2 + q2, facts, numeric)]
        B = [_named((beta - p1)**2 + q1, facts, numeric), _named((beta - p2)**2 + q2, facts, numeric)]
        preimage = (x - beta) / (alpha - x)
        Xp = (alpha - beta) / (1 + t)**2
        pole = alpha
        infinite = Integer(-1)
        bilinear = True
    assumptions = _with_facts(assumptions, facts)
    # labels with A2 B1 < A1 B2, so that 0 < m < 1
    ordered = _ask(A[1] * B[0] < A[0] * B[1], assumptions)
    if ordered is None:
        return None
    if not ordered:
        A.reverse()
        B.reverse()
        if _ask(A[1] * B[0] < A[0] * B[1], assumptions) is not True:
            return None
    # the radicals as positive symbols, so that the rational prefactors
    # stay rational functions of symbols
    lam_ = _radical(cancel(B[0] / A[0]), radicals, facts)
    m = cancel(1 - A[1] * B[0] / (A[0] * B[1]))
    return _TangentMap(X, Xp, preimage, pole, infinite, lam_, m, B[0] * B[1], bilinear, radicals, facts)


def _two_pairs_reduction(R: Expr, P: Expr, e: Expr, x: Symbol, a: Expr, b: Expr,
                         assumptions: Assumptions) -> Optional[ConditionalValue]:
    """The integral over a range of the real line of a radicand with two
    pairs of complex roots (positive everywhere), through the tangent
    map: with ``s = sin(theta)**2`` the even part of the rational
    prefactor in ``t`` is a Legendre form and the odd part is a rational
    function times ``1/sqrt(1 - m s)``, elementary. The range is cut at
    the pole of the bilinear map (``theta = +-pi/2``)."""
    factors = _two_pairs(P, x, assumptions)
    if factors is None:
        return None
    assumptions = _with_facts(assumptions, factors.facts)
    if _ask(factors.lam > 0, assumptions) is not True:
        return None
    if _ask(a < b, assumptions) is not True:
        return None
    t = _fresh('t')
    mapping = _tangent_map(factors.pairs, t, x, assumptions, bool(factors.facts))
    if mapping is None:
        return None
    assumptions = _with_facts(assumptions, mapping.facts)
    pieces: list[tuple[Expr, Expr]] = [(a, b)]
    pole = mapping.pole
    if pole is not None:
        inside = _position_of(pole, a, b, assumptions)
        if inside is None:
            return None
        if inside:
            pieces = [(a, pole), (pole, b)]
    lam_, m = mapping.lam_, mapping.m
    root = _radical(cancel(factors.lam * mapping.B1B2), mapping.radicals, mapping.facts)
    assumptions = _with_facts(assumptions, mapping.facts)
    # P(X)^e = lam^e (B1 B2)^e (1 - m s)^e (1 - s)^(-2e) (1 + t)^(-4e), dt = lam_ ds/(2 (1 - s) sqrt(s (1 - s)))
    G = _expr(R.subs(x, mapping.X)) * mapping.Xp
    if mapping.bilinear:
        G = G * ((1 + t)**2)**(-2 * e)
    G = cancel(G)
    if not G.is_rational_function(t):
        return None
    w = _fresh('w', positive=True)
    parts = _parity_parts(G, t, w)
    if parts is None:
        return None
    s = _fresh('s', positive=True)
    G_e = _expr(parts[0].subs(w, lam_**2 * s / (1 - s)))
    G_o = _expr(parts[1].subs(w, lam_**2 * s / (1 - s)))
    if e == -_HALF:
        C0 = lam_ / root
        Q = cancel(C0 * G_e / 2)
        H = cancel(C0 * lam_ * G_o / (2 * (1 - s)))
    else:
        C0 = lam_ * root
        Q = cancel(C0 * G_e * (1 - m * s) / (2 * (1 - s)**2))
        H = cancel(C0 * lam_ * G_o * (1 - m * s) / (2 * (1 - s)**3))
    parts_ = _TangentParts(Q, H, s, m, lam_, assumptions)
    total: Expr = Integer(0)
    for lo, hi in pieces:
        t0: Expr = -oo if lo == pole else mapping.preimage(lo, x)
        t1: Expr = oo if hi == pole else mapping.preimage(hi, x)
        value = parts_.between(t0, t1)
        if value is None:
            return None
        total = total + value
    restored = _restore(_reveal(total), mapping.radicals, factors.facts + mapping.facts)
    return None if restored is None else ConditionalValue(restored)


def _position_of(point: Expr, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[bool]:
    """Whether ``point`` lies strictly inside ``(a, b)``; ``None`` when
    undecided."""
    below: Optional[bool]
    above: Optional[bool]
    if isinstance(a, NegativeInfinity):
        below = True
    else:
        below = _ask(a < point, assumptions)
        if below is None and _ask(a >= point, assumptions) is True:
            below = False
    if isinstance(b, Infinity):
        above = True
    else:
        above = _ask(point < b, assumptions)
        if above is None and _ask(point >= b, assumptions) is True:
            above = False
    if below is None or above is None:
        return None
    return below and above


class _TangentParts:
    """``Integral((Q(s) + odd) ..., theta)`` over ``t`` from ``t0`` to ``t1``
    (``t = lambda tan(theta)``, ``s = sin(theta)**2 = t**2/(lambda**2 + t**2)``):
    on a range of one sign ``sigma``, with ``u = |t|`` from ``u0`` to ``u1``,

        sigma (F_e(s1) - F_e(s0)) + F_o(s1) - F_o(s0)

    with ``F_e`` the antiderivative of ``Q(s)/sqrt(s (1 - s)(1 - m s))``
    (Legendre, from :func:`_from_zero`) and ``F_o`` that of
    ``H(s)/sqrt(1 - m s)``, elementary through ``v = sqrt(1 - m s)``; a
    range of both signs is cut at ``t = 0``."""

    __slots__ = ('Q', 'H', 's', 'm', 'lam_', 'assumptions')

    def __init__(self, Q: Expr, H: Expr, s: Symbol, m: Expr, lam_: Expr, assumptions: Assumptions) -> None:
        self.Q: Final[Expr] = Q
        self.H: Final[Expr] = H
        self.s: Final[Symbol] = s
        self.m: Final[Expr] = m
        self.lam_: Final[Expr] = lam_
        self.assumptions: Final[Assumptions] = assumptions

    def between(self, t0: Expr, t1: Expr) -> Optional[Expr]:
        signs = [self._sign(t0), self._sign(t1)]
        first_sign, second_sign = signs
        if first_sign is None or second_sign is None:
            return None
        if first_sign == 0 or second_sign == 0 or first_sign == second_sign:
            sigma: Expr = Integer(1) if Integer(1) in signs else Integer(-1)
            return self._piece(sigma * t0, sigma * t1, sigma)
        first = self._piece(first_sign * t0, Integer(0), first_sign)
        second = self._piece(Integer(0), second_sign * t1, second_sign)
        if first is None or second is None:
            return None
        return first + second

    def _sign(self, tau: Expr) -> Optional[Expr]:
        if tau == 0:
            return Integer(0)
        if isinstance(tau, Infinity):
            return Integer(1)
        if isinstance(tau, NegativeInfinity):
            return Integer(-1)
        if _ask(tau > 0, self.assumptions) is True:
            return Integer(1)
        if _ask(tau < 0, self.assumptions) is True:
            return Integer(-1)
        return None

    def _s(self, u: Expr) -> Expr:
        if isinstance(u, Infinity):
            return Integer(1)
        return cancel(u**2 / (self.lam_**2 + u**2))

    def _piece(self, u0: Expr, u1: Expr, sigma: Expr) -> Optional[Expr]:
        s0, s1 = self._s(u0), self._s(u1)
        if s0 == s1:
            return Integer(0)
        increasing = _ask(s0 < s1, self.assumptions)
        if increasing is None:
            return None
        lower, upper = (s0, s1) if increasing else (s1, s0)
        total: Expr = Integer(0)
        if self.Q != 0:
            at_s0 = _from_zero(self.Q, self.s, s0, _Legendre(self.m), self.assumptions, lower, upper)
            at_s1 = _from_zero(self.Q, self.s, s1, _Legendre(self.m), self.assumptions, lower, upper)
            if at_s0 is None or at_s1 is None:
                return None
            total = total + sigma * (at_s1 - at_s0)
        if self.H != 0:
            if not _poles_outside(self.H, self.s, lower, upper, self.assumptions):
                return None
            F = _odd_antiderivative(self.H, self.s, self.m)
            if F is None:
                return None
            total = total + _expr(F.subs(self.s, s1)) - _expr(F.subs(self.s, s0))
        return total


def _poles_outside(H: Expr, s: Symbol, lower: Expr, upper: Expr, assumptions: Assumptions) -> bool:
    """Whether no pole of the rational function ``H`` lies on the closed
    range (``False`` when undecided)."""
    _, denominator = cancel(H).as_numer_denom()
    if not denominator.has(s):
        return True
    factored = _factor_list(denominator, s)
    if factored is None:
        return False
    for factor_, _multiplicity in factored[1]:
        if not factor_.has(s):
            continue
        degree = _degree(factor_, s)
        if degree is None:
            return False
        candidates: Optional[list[Expr]]
        if degree <= 2:
            candidates = _low_degree_roots(factor_, s)
        else:
            # SymPy's real_roots over ZZ and QQ
            candidates = _real_roots_rational(factor_, s)
        if candidates is None:
            return False
        for p in candidates:
            if p.is_extended_real is False:
                continue
            if p.is_extended_real is None and _ask(Eq(im(p), 0), assumptions) is False:
                continue
            if _ask((p - lower) * (p - upper) > 0, assumptions) is not True:
                return False
    return True


def _odd_antiderivative(H: Expr, s: Symbol, m: Expr) -> Optional[Expr]:
    """An antiderivative of ``H(s)/sqrt(1 - m s)``: with ``v = sqrt(1 - m s)``
    the integrand is the rational function ``-2 H((1 - v**2)/m)/m`` of
    ``v``, integrated exactly (by SymPy's ``integrate``, through the face:
    the step of the module which stays on SymPy)."""
    from .elliptic_sympy import integrate_of_core
    v = _fresh('v', positive=True)
    rational = cancel(-2 * _expr(H.subs(s, (1 - v**2) / m)) / m)
    if not rational.is_rational_function(v):
        return None
    F = integrate_of_core(rational, v, _budget())
    if F is None:
        return None
    return _expr(F.subs(v, sqrt(1 - m * s)))


def _at_arcsine(f: Expr, x: Symbol, s: Symbol) -> Expr:
    """``f`` at ``x = asin(sqrt(s))``: the trigonometric functions of ``x``
    written in ``s`` first (``sin(asin(y))`` is not evaluated by the
    core's ``sin``, which SymPy's is), then the other occurrences of
    ``x``."""
    root, co_root = sqrt(s), sqrt(1 - s)
    rule: dict[Basic, Basic] = {Sin(x): root, Cos(x): co_root, Tan(x): root / co_root, Cot(x): co_root / root,
                                Sec(x): 1 / co_root, Csc(x): 1 / root}
    return _expr(_expr(f._xreplace_evaluated(rule)).subs(x, asin(root)))


def _trigonometric(f: Expr, x: Symbol, a: Expr, b: Expr, assumptions: Assumptions) -> Optional[ConditionalValue]:
    """``Q(sin(x)**2) (1 - m sin(x)**2)**(+-1/2)`` over ``(0, phi)`` through
    ``s = sin(x)**2``."""
    if a != 0 or _ask(b > 0, assumptions) is not True or _ask(b <= pi / 2, assumptions) is not True:
        return None
    s = _fresh('s', positive=True)
    h = _at_arcsine(f, x, s) / 2
    radical: Optional[Expr] = None
    exponent: Optional[Expr] = None
    for node in h.atoms(Pow):
        if not isinstance(node, Pow):
            continue
        if node.exp in (_HALF, -_HALF) and node.has(s):
            base = node.base
            if radical is not None or not base.is_polynomial(s) or _degree(base, s) != 1:
                return None
            radical, exponent = base, node.exp
    if radical is None or exponent is None:
        return None
    alpha = _expr(radical.subs(s, 0))
    if _ask(alpha > 0, assumptions) is not True:
        return None
    m = cancel(-(radical - alpha) / (alpha * s))
    if m.has(s):
        return None
    Q = cancel(h / radical**exponent)
    if not Q.is_rational_function(s):
        return None
    if exponent == -_HALF:
        Q = Q / sqrt(alpha)
    else:
        Q = Q * sqrt(alpha) * (1 - m * s)
    upper: Expr = Integer(1) if b == pi / 2 else sin(b)**2
    value = legendre_reduction(Reduction(Q, m, s, Integer(0), upper), assumptions)
    return None if value is None else ConditionalValue(value)
