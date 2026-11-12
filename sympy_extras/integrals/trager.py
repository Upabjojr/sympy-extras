r"""Trager's algorithm for the integration of algebraic functions, in the
case of a quadratic extension: integrands rational in `x` and in
`y = \sqrt{P(x)}`.

The integrand is written `A(x) + B(x) y` with `A`, `B` rational in `x`,
and `y = d w` with `w^2 = Q` squarefree (`d` collects the repeated factors
of `P`); `A` is integrated by the rational algorithms and `B d w` by the
three steps of Trager's algorithm on the curve `w^2 = Q(x)`
[Trager]_, [Bronstein]_:

1. **Hermite reduction.** The differential `M/(D w)\,dx` has its poles
   at the roots of `D`. A factor `V^k` of `D` with `V` squarefree and
   coprime to `Q` (unramified points) is reduced by subtracting the
   derivative of `S w/V^{k-1}` with `S` found modulo `V` by a Bézout
   computation, down to `k = 1`; a factor `V^k` with `V | Q` (ramified
   points, where the differential has a pole of even order and no
   residue) is reduced by the derivative of `S w/V^k` down to `k = 0`.
   Then the derivative of `S w` with `S` a polynomial lowers the degree
   of the polynomial part of the numerator below `\deg Q - 1`. What is
   left, `r/(D w)`, has simple poles at the finite points; a pole of
   order two or more at infinity (a numerator of degree at least
   `\lfloor \deg Q/2 \rfloor`) cannot be removed by any function of the
   field, so the integral is then not elementary.
2. **Residues.** The residue at the point `(a, w_a)` above a root `a` of
   `D` is `r(a)/(D'(a) w_a)`; the residues are the roots of the
   Rothstein–Trager resultant `\operatorname{Res}_x(D, r^2 - z^2 D'^2 Q)`
   and, for `\deg Q` even, the two points at infinity carry the residues
   `\mp \operatorname{lc}(r)/\sqrt{\operatorname{lc}(Q)}` when `\deg r =
   \deg Q/2 - 1`. The residues which are rational multiples of one
   another form a class with the divisor `\Delta = \sum_P (\rho_P/\rho) P`
   of degree zero.
3. **The logarithmic part.** The class contributes `(\rho/n) \log F` to
   the integral when `n \Delta` is the divisor of a function `F`; `F =
   (u + v w)/\prod G_j^{n q_j}` with `u`, `v` polynomials found by linear
   algebra: `u + v w` must vanish to order `2 n q_j` at the points with
   residue `q_j \rho` (the points above the roots of `G_j`, on the branch
   `w \equiv W_j \pmod{G_j}`) and have bounded pole orders at the points
   at infinity, which is a homogeneous linear system once `w` is lifted
   to a polynomial modulo `G_j^{2 n q_j}` by Newton's iteration and
   expanded in `1/x` at infinity. The multiples `n` of the common
   denominator of the `q_j` are tried in turn; on a curve of genus one
   over the rationals with rational residues the search stops at twelve
   times the denominator by Mazur's theorem, and its failure proves the
   integral not elementary.

Integrands with parameters go through the Hermite reduction only;
residues which are not rational multiples of one another are treated as
independent classes (which may miss an elementary integral when three
or more residues are linearly dependent over the rationals, but never
returns a wrong value: every antiderivative is verified numerically).

**Roots of index** `n \geq 3`, `y = P^{1/n}` with `P` squarefree over the
rationals, are integrated by components: `\{1, y, \ldots, y^{n-1}\}` is
an integral basis of the curve `y^n = P` (the affine curve is smooth)
and the derivative of `R y^k` stays in the component `y^k`, so `f =
\sum_k A_k y^k` splits into the differentials `A_k y^k\,dx`, each
elementary or not on its own (the automorphism `y \mapsto \zeta y`
multiplies it by `\zeta^k`). A component is first tried as a Risch
differential equation for the rational `R`; otherwise it goes through
the three steps on the curve `y^n = P` (a component with `\gcd(k, n) >
1` is one of the smaller index `n/\gcd(k, n)`, the square root case
above for `n/\gcd(k, n) = 2`):

1. **Hermite reduction.** `M y^k/(D P^j)\,dx` with a factor `V^e` of `D`
   coprime to `P` is lowered by `S y^k/V^{e-1}`, a factor `V^e` dividing
   `P` (the ramified points, where the pole has order `n e - k + 1` and
   no residue) by `S y^k/V^e` down to `e = 0`, the factor `P^j` by `S
   y^k/P^{j-1}` down to `j = 1` (where the differential has no pole at
   the ramified points), and the polynomial part `q y^k/P` by `c x^i
   y^k` while `\deg q \geq \deg P - 1`. What is left, with `\deg q \leq
   \deg P - 2`, has simple poles at the finite points; a pole of order
   two or more at a point at infinity (its order is `-e (\deg q + 1 -
   \deg P + k \deg P/n) - 1` with `e = n/\gcd(n, \deg P)`) is removed by
   no function of the field, so the integral is then not elementary.
2. **Residues.** The residue at the place `(a, y_a)` above a root `a`
   of `D` is `M(a) y_a^k/(D'(a) P(a))`, so the `n`-th powers of the
   residues are the roots of `\operatorname{Res}_x(D, \tau D'^n P^{n-k} -
   M^n)` and the residues at the `n` places above `a` are the `n`-th
   roots of one of them, the branch `y \equiv W \pmod G` of each place
   recovered from `y^k \equiv \rho D' P/M`. The `n` points at infinity
   carry residues only when `n` divides `\deg P` (`-s^k \kappa` at the
   point where `y \sim s x^{\deg P/n}`, `\kappa` a coefficient of the
   expansion of the differential in `1/x`). The number field of the
   residues (of degree at most twelve, the norms of the residues of
   degree at most two: beyond, the roots are nested radicals) is the
   field of the linear algebra.
3. **The logarithmic part.** For each orbit `Q_0, \ldots, Q_{n-1}` of
   places under `y \mapsto \zeta y` (the residues `\rho \zeta^{lk}`), a
   function `v` with divisor `m (Q_0 - \sum_l Q_l/n)` is searched for
   `m = n, 2n, \ldots` up to the torsion bound, as `U/G^{m/n}` with `U
   = \sum_i U_i(x) y^i` vanishing to order `m` on the branch of `Q_0`
   (the branch lifted modulo `G^m` by Hensel's iteration) and its poles
   at infinity bounded by the degrees of the `U_i`; the orbit
   contributes `\sum_l (\rho_l/m) \log v(x, \zeta^l y)` with `\rho_l`
   the residue where `v(x, \zeta^l y)` vanishes, the classical resolvent
   form whose real part is the answer in logarithms and arctangents.
   When the residues of different orbits are dependent over the
   rationals, or an orbit is not torsion by itself, Trager's general
   form is searched instead: over a basis `\beta_j` of the rational
   span of the residues the residue divisor is `\sum_j \beta_j D_j`
   with each `D_j` of degree zero, and `D_j` with `m_j D_j` principal
   contributes `(\beta_j/m_j) \log u_j`. The differential minus the
   logarithmic derivatives found is checked to vanish exactly on the
   curve (what would be left is a differential of the first kind, whose
   integral is not elementary); a component for which no function is
   found within the bounds is left undecided, never answered.

The module works on the objects of the core and imports no SymPy: the
polynomials are the dense ones of :mod:`sympy_extras.polys.polyclasses`
over the typed domains (``QQ``, the number fields of
:class:`~sympy_extras.polys.domains.algebraicfield.AlgebraicField` with a
primitive element of the constants, the fields of rational functions of
the parameters, ``EX`` for other constants), the rational parts are
those of :mod:`sympy_extras.integrals.risch.rationaltools`, the Risch
differential equation that of :mod:`sympy_extras.integrals.risch.rde`,
the several and nested radicals go to
:mod:`sympy_extras.integrals.curves`. :mod:`.trager_sympy` is its face on
SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.functions import sqrt
>>> from sympy_extras.integrals.trager import trager_antiderivative
>>> x = symbols('x')
>>> trager_antiderivative(x/sqrt(x**4 + 1), x)
log(x**2 + sqrt(x**4 + 1))/2

References
==========

.. [Trager] B. M. Trager, *Integration of algebraic functions*, PhD
   thesis, MIT, 1984, chapters 2 and 3.
.. [Bronstein] M. Bronstein, *Symbolic integration tutorial*, ISSAC
   1998, section 4 (algebraic functions); *Symbolic Integration I*,
   Springer, 2005, section 2.5 (the Hermite reduction).
.. [Davenport] J. H. Davenport, *On the integration of algebraic
   functions*, Lecture Notes in Computer Science 102, Springer, 1981.
.. [Mazur] B. Mazur, *Modular curves and the Eisenstein ideal*, Publ.
   Math. IHÉS 47 (1977), the torsion of elliptic curves over the
   rationals.
"""
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction
from typing import Callable, Final, Generic, Optional, Sequence, TypeVar

import mpmath

from sympy_extras._numeric import reliable_complex, reliable_value
from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.evalf import evalf, numeric_value
from sympy_extras.core.expr import Expr, ExprLike, as_expr, to_expr
from sympy_extras.core.function import Derivative, diff, expand, expand_mul
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, ImaginaryUnit, Integer, Rational, number_from_fraction
from sympy_extras.core.power import Pow, pow, sqrt
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Im, Re, Sign, sign
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.functions.elementary.trigonometric import atan
from sympy_extras.functions.special.delta_functions import DiracDelta
from sympy_extras.polys import densebasic
from sympy_extras.polys.domains import QQ
from sympy_extras.polys.domains.algebraicfield import AlgebraicField
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.expressiondomain import EX
from sympy_extras.polys.domains.fractionfield import FracElement, FractionField
from sympy_extras.polys.domains.polynomialring import PolynomialRing
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing, SparsePolynomialRing
from sympy_extras.polys.euclidtools import dup_resultant
from sympy_extras.polys.numberfields.minpoly import minimal_polynomial_coeffs
from sympy_extras.polys.numberfields.qqbar import Algebraic, algebraic_roots, primitive_element
from sympy_extras.polys.polyclasses import DensePoly
from sympy_extras.polys.polyerrors import BasePolynomialError, NotInvertible
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel, factor, factor_list
from sympy_extras.polys.polyutils import dict_from_expr
from sympy_extras.settings import settings
from sympy_extras.simplify.powsimp import powdenest
from sympy_extras.simplify.radsimp import fraction

__all__ = ['trager_antiderivative', 'trager_reduce', 'hermite_reduce_algebraic', 'is_nonelementary_algebraic']

T = TypeVar('T')
U = TypeVar('U')

#: an element of a number field: its coordinates in the powers of the
#: generator, highest first
Number = list[Fraction]

#: the reduction of an algebraic integrand on a general curve: ``(F, h)``
#: with ``f == F' + h``, or ``None``
CurveReducer = Callable[[Expr, Symbol], Optional[tuple[Expr, Expr]]]

#: the multiples of the denominator of the residue divisor tried for the
#: torsion order: complete on a curve of genus one over the rationals
#: (Mazur: the torsion of an elliptic curve over Q has order at most 12)
_TORSION_BOUND: Final[int] = 12

#: the largest degree of the number field of the constants and of the
#: residues of a square root (the primitive element of many radicals)
_MAX_FIELD_DEGREE: Final[int] = 32

_FAILURES: Final = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError,
                    ArithmeticError)

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

_COUNTER: Final = itertools.count()


def _fresh(name: str, real: bool = False, positive: bool = False) -> Dummy:
    """A dummy with a name of its own: the typed polynomials identify
    their variables by name."""
    label = '_trager_%s_%d' % (name, next(_COUNTER))
    if positive:
        return Dummy(label, positive=True)
    if real:
        return Dummy(label, real=True)
    return Dummy(label)


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the subtrees of ``rule`` replaced, the rebuilt nodes
    evaluated."""
    if not rule:
        return e
    return as_expr(e._xreplace_evaluated(rule))


def _pows(e: Expr) -> list[Pow]:
    """The powers in ``e``."""
    return [node for node in e.atoms(Pow) if isinstance(node, Pow)]


def _half_integer(node: Pow) -> bool:
    exponent = node.exp
    return isinstance(exponent, Rational) and exponent.q == 2


def _exponent(node: Pow) -> Rational:
    exponent = node.exp
    if not isinstance(exponent, Rational):
        raise TypeError('%s has no rational exponent' % (node,))
    return exponent


def _symbols(e: Expr) -> list[Symbol]:
    return sorted((s for s in e.free_symbols if isinstance(s, Symbol)), key=str)


def _powers_of(e: Expr, y: Symbol) -> dict[int, Expr]:
    """The coefficients of the powers of ``y`` in the polynomial ``e``
    (expanded); ``ValueError`` when ``e`` is not a polynomial in ``y``."""
    collected: dict[int, list[Expr]] = {}
    for term in Add.make_args(expand(e)):
        k = 0
        rest: list[Expr] = []
        for factor_ in Mul.make_args(term):
            if factor_ == y:
                k += 1
            elif isinstance(factor_, Pow) and factor_.base == y and isinstance(factor_.exp, Integer) \
                    and factor_.exp.p >= 0:
                k += factor_.exp.p
            elif factor_.has(y):
                raise ValueError('%s is not a polynomial in %s' % (e, y))
            else:
                rest.append(factor_)
        collected.setdefault(k, []).append(mul(*rest))
    return {k: add(*terms) for k, terms in collected.items()}


# ---------------------------------------------------------------------------
# Number fields and the fields of the computation

class _UnorderedField(AlgebraicField):
    """A number field as the ground of polynomials whose fractions are
    reduced: no element counts as negative (the sign normalisation of the
    denominators is skipped)."""

    def is_negative(self, a: list[Fraction]) -> bool:
        return False


def _algebraic_number(e: Expr) -> Optional[Algebraic]:
    """The algebraic number ``e`` (a constant built from rationals, ``I``
    and radicals), the root of its minimal polynomial nearest to its
    numerical value; ``None`` when it is not one."""
    if e == I:
        return Algebraic.imaginary_unit()
    if isinstance(e, Rational):
        return Algebraic.from_rational(Fraction(e.p, e.q))
    try:
        coefficients = minimal_polynomial_coeffs(e)
        value = complex(evalf(e, 30))
    except _FAILURES:
        return None
    candidates = [(abs(complex(r) - value), r) for r in algebraic_roots(coefficients)]
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    if len(candidates) > 1 and candidates[1][0] < 1e-12 * (1 + abs(value)):
        return None
    return candidates[0][1]


def _horner(coefficients: Sequence[Fraction], point: Number, K: AlgebraicField) -> Number:
    """The polynomial (rational coefficients, highest degree first) at the
    element ``point`` of ``K``."""
    result = K.zero
    for c in coefficients:
        result = K.add(K.mul(result, point), K.from_rational(Fraction(c)))
    return result


class _NumberField:
    """``QQ(theta)`` generated by algebraic constants ``generators``, with
    each one as an element (``coordinates``) and as a polynomial in the
    symbol ``theta`` (``forms``), and ``theta`` as an expression in them
    (``theta = g_1 + c_2 g_2 + ...``, the primitive elements of
    :func:`~sympy_extras.polys.numberfields.qqbar.primitive_element`);
    ``ArithmeticError`` when a generator is not algebraic or the degree
    exceeds ``bound``."""

    def __init__(self, generators: Sequence[Expr], bound: int = _MAX_FIELD_DEGREE) -> None:
        self.theta: Final[Symbol] = _fresh('theta')
        values: list[Algebraic] = []
        for g in generators:
            value = _algebraic_number(g)
            if value is None:
                raise ArithmeticError('%s is not an algebraic number' % (g,))
            values.append(value)
        gamma = values[0]
        gamma_expr = generators[0]
        coordinates: list[Number] = [[Fraction(1), Fraction(0)]]
        for g, value in zip(generators[1:], values[1:]):
            new, p, q = primitive_element(gamma, value)
            if new.degree > bound:
                raise ArithmeticError('the field of the constants is too large')
            K = AlgebraicField([Fraction(c, new.minpoly[0]) for c in new.minpoly], self.theta)
            gamma_in_new = _horner(p, K.alpha, K)
            coordinates = [_horner(c, gamma_in_new, K) for c in coordinates] + [_horner(q, K.alpha, K)]
            c = 0
            if new != gamma:
                c = next(k for k in range(1, 4 * gamma.degree * value.degree + 2) if gamma + k * value == new)
            gamma, gamma_expr = new, add(gamma_expr, mul(Integer(c), g))
        if gamma.degree > bound:
            raise ArithmeticError('the field of the constants is too large')
        minpoly = [Fraction(c, gamma.minpoly[0]) for c in gamma.minpoly]
        self.field: Final[AlgebraicField] = _UnorderedField(minpoly, self.theta)
        self.degree: Final[int] = len(minpoly) - 1
        self.theta_expr: Final[Expr] = gamma_expr
        self.coordinates: Final[dict[Expr, Number]] = {
            g: self.field.convert(element, self.field) for g, element in zip(generators, coordinates)}
        self.forms: Final[dict[Basic, Basic]] = {g: self.field.to_expr(element) for g, element in
                                                 zip(generators, coordinates)}


class _Field(Generic[U]):
    """A field of the computation (``domain``) with the conversion of the
    expressions of its elements: through the forms of the constants in
    the generator of a number field (``number_field``), whose elements
    are written back in the constants."""

    def __init__(self, domain: Domain[U], number_field: Optional[_NumberField], parameters: list[Symbol],
                 rationals: bool) -> None:
        self.domain: Final[Domain[U]] = domain
        self.number_field: Final[Optional[_NumberField]] = number_field
        self.parameters: Final[list[Symbol]] = parameters
        #: whether the field is the rationals
        self.rationals: Final[bool] = rationals
        #: the parameters as the typed polynomials write them back (by
        #: name) and as they are
        self.restore: Final[dict[Basic, Basic]] = {Symbol(p.name): p for p in parameters if Symbol(p.name) != p}

    def element(self, e: Expr) -> U:
        """An expression as an element; ``ValueError`` when it is none."""
        F = self.number_field
        if F is not None:
            e = _replaced(e, F.forms)
        return self.domain.from_expr(e)

    def expr(self, c: U) -> Expr:
        """An element as an expression."""
        e = _replaced(self.domain.to_expr(c), self.restore)
        F = self.number_field
        if F is None:
            return e
        return expand(_replaced(e, {F.theta: F.theta_expr}))

    def poly(self, p: DensePoly[T]) -> DensePoly[U]:
        """A polynomial over another field (whose elements are elements of
        this one) over this one."""
        K = p.domain
        return DensePoly([self.element(K.to_expr(c)) for c in p.rep], self.domain, p.gen)

    def from_expr(self, e: Expr, x: Symbol) -> DensePoly[U]:
        """The polynomial in ``x`` of an expression."""
        coefficients = _powers_of(e, x)
        degree = max(coefficients) if coefficients else 0
        return DensePoly([self.element(coefficients.get(k, _ZERO)) for k in range(degree, -1, -1)], self.domain, x)

    def poly_expr(self, p: DensePoly[U]) -> Expr:
        """The polynomial as an expression in its generator."""
        x = p.gen
        n = p.degree
        terms: list[Expr] = []
        for i, c in enumerate(p.rep):
            if self.domain.is_zero(c):
                continue
            k = n - i
            power = _ONE if k == 0 else (x if k == 1 else pow(x, Integer(k)))
            terms.append(mul(self.expr(c), power))
        return add(*terms)


def _rationals() -> _Field[Fraction]:
    return _Field(QQ, None, [], True)


def _parametric(parameters: list[Symbol]) -> _Field[SparseFracElement[Fraction]]:
    """``QQ(a, b, ...)``."""
    return _Field(SparseFractionField(RationalPolynomialRing([p.name for p in parameters])), None, parameters, False)


def _algebraic(F: _NumberField) -> _Field[Number]:
    """The number field ``F``."""
    return _Field(F.field, F, [], False)


def _algebraic_parametric(F: _NumberField, parameters: list[Symbol]) -> _Field[SparseFracElement[Number]]:
    """``F(a, b, ...)`` with ``F`` a number field."""
    ring: SparsePolynomialRing[Number] = SparsePolynomialRing(F.field, [p.name for p in parameters])
    return _Field(SparseFractionField(ring), F, parameters, False)


def _expressions() -> _Field[Expr]:
    return _Field(EX, None, [], False)


def _constant_kind(exprs: Sequence[Expr], x: Symbol) -> tuple[list[Symbol], bool, bool]:
    """The parameters of the polynomials ``exprs`` in ``x``, whether their
    coefficients need ``I`` (``QQ<I>``) and whether they need other
    constants (``EX``): the domain SymPy's ``Poly`` constructs."""
    parameters = sorted({s for e in exprs for s in _symbols(e) if s != x}, key=str)
    names = [x.name] + [p.name for p in parameters]
    gaussian = False
    parts = [part for e in exprs for part in fraction(cancel(e))]
    for e in parts:
        try:
            dict_from_expr(e, names, QQ)
            continue
        except ValueError:
            pass
        try:
            dict_from_expr(e, names, _gaussian_field())
            gaussian = True
        except ValueError:
            return parameters, False, True
    return parameters, gaussian, False


_GAUSSIAN: list[_NumberField] = []


def _gaussian() -> _NumberField:
    """The number field ``QQ<I>``."""
    if not _GAUSSIAN:
        _GAUSSIAN.append(_NumberField([I]))
    return _GAUSSIAN[0]


def _gaussian_field() -> AlgebraicField:
    """``QQ<I>`` with ``I`` itself as the generator."""
    return AlgebraicField([Fraction(1), Fraction(0), Fraction(1)], I)


# ---------------------------------------------------------------------------
# The parsing of the integrand

class _Parsed:
    """The integrand ``A + B*sqrt(base)`` with ``A``, ``B`` rational in
    ``x`` and ``base`` a polynomial."""

    def __init__(self, base: Expr, A: Expr, B: Expr) -> None:
        self.base: Final[Expr] = base
        self.A: Final[Expr] = A
        self.B: Final[Expr] = B


def _parse(f: Expr, x: Symbol) -> Optional[_Parsed]:
    """The integrand as ``A + B*sqrt(P)`` with ``A``, ``B`` rational in
    ``x`` and ``P`` a polynomial, or ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial."""
    bases: set[Expr] = set()
    for node in _pows(f):
        if _half_integer(node):
            base = node.base
            if not base.has(x):
                continue
            if not base.is_polynomial(x):
                return None
            bases.add(base)
    if len(bases) > 1:
        # sqrt(1 - 3*x**2)*sqrt(1 - x**2) is sqrt((1 - 3*x**2)*(1 - x**2))
        # where both radicands are positive, the real domain of the
        # integrand (on which the antiderivative is verified): the
        # integrand must be rational in x and in the product of the roots
        combined = _combined_roots(f, x, sorted(bases, key=str))
        if combined is None:
            return None
        return _parse(combined, x)
    if len(bases) != 1:
        return None
    [base] = bases
    y = _fresh('y')

    def to_y(node: Basic) -> Basic:
        if isinstance(node, Pow) and node.base == base and _half_integer(node):
            return pow(y, Integer(_exponent(node).p))
        return node

    g = as_expr(f.replace(lambda node: isinstance(node, Pow), to_y))
    if not g.is_rational_function(x, y):
        return None
    try:
        P = _powers_of(base, x)
    except ValueError:
        return None
    if max(P) < 1:
        return None
    numerator, denominator = fraction(cancel(g))
    try:
        n0, n1 = _reduce_modulo_curve(numerator, y, base)
        d0, d1 = _reduce_modulo_curve(denominator, y, base)
    except ValueError:
        return None
    norm = expand(d0**2 - d1**2 * base)
    if norm == 0:
        return None
    A = cancel((n0 * d0 - n1 * d1 * base) / norm)
    B = cancel((n1 * d0 - n0 * d1) / norm)
    return _Parsed(base, A, B)


def _combined_roots(f: Expr, x: Symbol, bases: list[Expr]) -> Optional[Expr]:
    """``f`` with its several square roots written through the one root of
    the product of the radicands, when every monomial of ``f`` in the
    roots has either all of them or none (the parity of each exponent
    reduced by ``sqrt(P)**2 = P``); ``None`` otherwise."""
    Y = [_fresh('Y%d' % i) for i in range(len(bases))]
    replacement: dict[Basic, Basic] = {}
    for node in _pows(f):
        base = node.base
        if base in bases and _half_integer(node):
            i = bases.index(base)
            k = _exponent(node).p
            replacement[node] = mul(pow(base, Integer((k - k % 2) // 2)), pow(Y[i], Integer(k % 2)))
    g = _replaced(f, replacement)
    if not g.is_rational_function(x, *Y):
        return None
    numerator, denominator = fraction(cancel(g))
    y = _fresh('y')
    product = mul(*bases)
    rewritten: list[Expr] = []
    for part in (numerator, denominator):
        try:
            poly = Poly(part, *Y)
        except _FAILURES:
            return None
        total: Expr = _ZERO
        for monomial, coefficient in poly.terms():
            parities = [e % 2 for e in monomial]
            factor_: Expr = _ONE
            for i, e in enumerate(monomial):
                factor_ = factor_ * pow(bases[i], Integer((e - e % 2) // 2))
            if all(parities):
                total = total + coefficient * factor_ * y
            elif not any(parities):
                total = total + coefficient * factor_
            else:
                return None
        rewritten.append(total)
    return _replaced(rewritten[0] / rewritten[1], {y: sqrt(product)})


def _reduce_modulo_curve(N: Expr, y: Symbol, P: Expr) -> tuple[Expr, Expr]:
    """``n0, n1`` with ``N(y) == n0 + n1*y`` modulo ``y**2 - P``."""
    n0: Expr = _ZERO
    n1: Expr = _ZERO
    for k, coefficient in _powers_of(N, y).items():
        if k % 2 == 0:
            n0 = n0 + coefficient * pow(P, Integer(k // 2))
        else:
            n1 = n1 + coefficient * pow(P, Integer((k - 1) // 2))
    return expand(n0), expand(n1)


class _Radical(Generic[T]):
    """The integrand ``A + B*y`` on the curve ``y**2 == P`` over the field
    ``F``, with ``y = d*w`` and ``w**2 == Q`` squarefree."""

    def __init__(self, F: _Field[T], x: Symbol, P: DensePoly[T], A: Expr, B: Expr) -> None:
        self.F: Final[_Field[T]] = F
        self.x: Final[Symbol] = x
        self.P: Final[DensePoly[T]] = P
        self.A: Final[Expr] = A
        self.B: Final[Expr] = B
        d, Q = _split_square(P)
        self.d: Final[DensePoly[T]] = d
        self.Q: Final[DensePoly[T]] = Q
        self.P_expr: Final[Expr] = F.poly_expr(P)
        self.d_expr: Final[Expr] = F.poly_expr(d)
        self.Q_expr: Final[Expr] = F.poly_expr(Q)

    @property
    def w(self) -> Expr:
        """``w`` as an expression in ``x``: ``sqrt(P)/d``."""
        return sqrt(self.P_expr) / self.d_expr

    def algebraic_numerator(self) -> tuple[DensePoly[T], DensePoly[T]]:
        """``M, D`` with ``B*y == M/(D*w)``: ``B*y = B*d*w = B*d*Q/w``."""
        e = cancel(self.B * self.d_expr * self.Q_expr)
        numerator, denominator = fraction(e)
        return self.F.from_expr(numerator, self.x), self.F.from_expr(denominator, self.x)


def _split_square(P: DensePoly[T]) -> tuple[DensePoly[T], DensePoly[T]]:
    """``d, Q`` with ``P == d**2 * Q`` and ``Q`` squarefree."""
    content, factors = P.sqf_list()
    K = P.domain
    d = P.new([K.one])
    Q = P.new([content])
    for factor_, multiplicity in factors:
        d = d * factor_**(multiplicity // 2)
        if multiplicity % 2:
            Q = Q * factor_
    return d, Q


def _monomial(c: T, j: int, K: Domain[T], x: Symbol) -> DensePoly[T]:
    """``c*x**j``."""
    return DensePoly([c] + [K.zero] * j, K, x)


# ---------------------------------------------------------------------------
# The Hermite reduction

class _Reduction(Generic[T]):
    """The Hermite reduction of ``M/(D*w)`` on ``w**2 == Q``: ``terms`` are
    the pairs ``(S, T)`` of the exact part ``sum(S*w/T)``, the remainder
    ``r/(D*w)`` has ``D`` squarefree and coprime to ``Q``, and
    ``polynomial`` is the part of ``r`` of degree below ``deg Q - 1``
    left by the reduction at infinity."""

    def __init__(self, terms: list[tuple[DensePoly[T], DensePoly[T]]], r: DensePoly[T], D: DensePoly[T],
                 polynomial: DensePoly[T]) -> None:
        self.terms: Final[list[tuple[DensePoly[T], DensePoly[T]]]] = terms
        self.r: Final[DensePoly[T]] = r
        self.D: Final[DensePoly[T]] = D
        self.polynomial: Final[DensePoly[T]] = polynomial


def _hermite(M: DensePoly[T], D: DensePoly[T], Q: DensePoly[T], x: Symbol) -> _Reduction[T]:
    """Trager's Hermite reduction of ``M/(D*w)`` on the curve ``w**2 == Q``
    (``Q`` squarefree), followed by the reduction of the polynomial part
    at infinity."""
    terms: list[tuple[DensePoly[T], DensePoly[T]]] = []
    one = Q.new([Q.domain.one])
    while True:
        common = M.gcd(D)
        M, D = M.exquo(common), D.exquo(common)
        lead = D.LC
        M, D = M.exquo_ground(lead), D.monic()
        step = _reduction_step(M, D, Q, x)
        if step is None:
            break
        M, D, S, T_ = step
        terms.append((S, T_))
    # the polynomial part: p/w with p reduced below degree deg Q - 1
    p, r1 = M.div(D)
    S, r2 = _reduce_at_infinity(p, Q, x)
    if not S.is_zero:
        terms.append((S, one))
    r = r2 * D + r1
    common = r.gcd(D)
    r, D = r.exquo(common), D.exquo(common)
    return _Reduction(terms, r, D, r2)


def _reduction_step(M: DensePoly[T], D: DensePoly[T], Q: DensePoly[T],
                    x: Symbol) -> Optional[tuple[DensePoly[T], DensePoly[T], DensePoly[T], DensePoly[T]]]:
    """One step of the Hermite reduction: the factor of ``D`` of highest
    multiplicity which is ramified or multiple removed once, returning
    the new ``M``, ``D`` and the term ``S*w/T`` subtracted; ``None`` when
    ``D`` is squarefree and coprime to ``Q``."""
    K = Q.domain
    _, factors = D.sqf_list()
    for V, k in sorted(factors, key=lambda pair: -pair[1]):
        ramified = V.gcd(Q)
        if ramified.degree > 0:
            # the pole has even order 2k at the ramified points: S w/V^k
            V = ramified
            U_ = D.exquo(V**k)
            Q_ = Q.exquo(V)
            inverse = _inverse((U_ * V.diff() * Q_).mul_ground(K.from_int(1 - 2 * k)), V)
            S = (M * inverse).mul_ground(K.from_int(2)).rem(V)
            subtracted = (S.diff() * Q_ * V).mul_ground(K.from_int(2)) + S * Q.diff() \
                - (S * V.diff() * Q_).mul_ground(K.from_int(2 * k))
            M_ = (M - (U_ * subtracted).exquo_ground(K.from_int(2))).exquo(V)
            return M_, U_ * V**(k - 1), S, V**k
        if k >= 2:
            U_ = D.exquo(V**k)
            inverse = _inverse((U_ * V.diff() * Q).mul_ground(K.from_int(k - 1)), V)
            S = -(M * inverse).rem(V)
            subtracted = (S.diff() * Q * V).mul_ground(K.from_int(2)) + S * Q.diff() * V \
                - (S * V.diff() * Q).mul_ground(K.from_int(2 * (k - 1)))
            M_ = (M - (U_ * subtracted).exquo_ground(K.from_int(2))).exquo(V)
            return M_, U_ * V**(k - 1), S, V**(k - 1)
    return None


def _reduce_at_infinity(p: DensePoly[T], Q: DensePoly[T], x: Symbol) -> tuple[DensePoly[T], DensePoly[T]]:
    """``S, r`` with ``p/w == (S*w)' + r/w`` and ``deg r < deg Q - 1``: the
    derivative of ``c*x**j*w`` has the numerator ``c*(2*j + deg Q)*lc(Q)
    x**(j + deg Q - 1) + ...``, so the degrees are peeled one by one."""
    K = Q.domain
    S = Q.new([])
    n = Q.degree
    two = K.from_int(2)
    while not p.is_zero and p.degree >= n - 1:
        j = p.degree - n + 1
        c = K.exquo(K.mul(two, p.LC), K.mul(Q.LC, K.from_int(2 * j + n)))
        monomial = _monomial(c, j, K, x)
        S = S + monomial
        p = p - ((monomial.diff() * Q).mul_ground(two) + monomial * Q.diff()).exquo_ground(two)
    return S, p


def _inverse(p: DensePoly[T], modulus: DensePoly[T]) -> DensePoly[T]:
    """``p**(-1)`` modulo ``modulus``, by the extended Euclidean algorithm
    (over ``QQ<I>(a, b)``, the field of a parametric integrand with
    residues ``I*r(a, b)``, too)."""
    s, _, h = p.gcdex(modulus)
    if h.degree != 0:
        raise NotInvertible('zero divisor')
    return s.exquo_ground(h.LC)


# ---------------------------------------------------------------------------
# The residues and the logarithmic part

def _resultant_in(F: DensePoly[T], G0: DensePoly[T], G1: DensePoly[T], power: int, z: Symbol) -> DensePoly[T]:
    """``Res_x(F, G0 + z**power*G1)`` as a polynomial in ``z``."""
    K = F.domain
    Kz: PolynomialRing[T] = PolynomialRing(K, z.name)

    def lifted(p: DensePoly[T]) -> list[list[T]]:
        return [densebasic.dup_strip([c], K) for c in p.rep]

    zp = [K.one] + [K.zero] * power
    shifted = [Kz.mul(c, zp) for c in lifted(G1)]
    G: list[list[T]] = densebasic.dup_strip(_dense_add(lifted(G0), shifted, Kz), Kz)
    F_: list[list[T]] = lifted(F)
    resultant: list[T] = dup_resultant(F_, G, Kz)
    return DensePoly(resultant, K, z)


def _dense_add(f: list[T], g: list[T], K: Domain[T]) -> list[T]:
    """The sum of two dense polynomials (highest degree first)."""
    n = max(len(f), len(g))
    f_ = [K.zero] * (n - len(f)) + f
    g_ = [K.zero] * (n - len(g)) + g
    return densebasic.dup_strip([K.add(a, b) for a, b in zip(f_, g_)], K)


def _residue_polynomial(r: DensePoly[T], D: DensePoly[T], Q: DensePoly[T], z: Symbol) -> DensePoly[T]:
    """The Rothstein–Trager resultant whose roots are the residues at the
    finite points."""
    return _resultant_in(D, r * r, -(D.diff()**2 * Q), 2, z)


def _factor_exprs(F: _Field[T], R: DensePoly[T], z: Symbol) -> list[Expr]:
    """The irreducible factors (as expressions in ``z``) of a polynomial
    over a field of the computation: over the field itself for the
    rationals and the number fields, as a polynomial in ``z`` and the
    parameters otherwise (the whole polynomial when it cannot be
    factored)."""
    try:
        _, factors = R.factor_list()
        return [F.poly_expr(f) for f, _ in factors]
    except NotImplementedError:
        pass
    numerator, _ = fraction(cancel(F.poly_expr(R)))
    try:
        _, found = factor_list(numerator, z)
        return [e for e, _ in found]
    except _FAILURES:
        return [numerator]


def _closed_roots(e: Expr, z: Symbol) -> Optional[list[Expr]]:
    """The roots of the irreducible polynomial ``e`` in ``z`` in closed
    form, ``None`` when some root has none."""
    coefficients = _powers_of(e, z)
    degree = max(coefficients)
    try:
        found = roots(e, z)
        if isinstance(found, dict) and sum(found.values()) == degree:
            return [r for r in found if isinstance(r, Expr)]
        if isinstance(found, dict):
            return None
    except _FAILURES:
        pass
    g = 0
    for k in coefficients:
        g = math.gcd(g, k)
    if g > 1 and degree > g:
        # a polynomial in z**g (the native roots do not decompose it over
        # the parameters): the g-th roots of the roots in z**g
        t = _fresh('t')
        inner = _closed_roots(add(*[c * pow(t, Integer(k // g)) for k, c in coefficients.items()]), t)
        if inner is None:
            return None
        found_: list[Expr] = []
        for root in inner:
            powers = _binomial_roots(root, g, t)
            if powers is None:
                return None
            found_.extend(powers)
        return found_
    # Gaussian coefficients (not taken by the polynomials of the core)
    a = coefficients.get(degree, _ZERO)
    b = coefficients.get(degree - 1, _ZERO)
    if degree == 1:
        return [cancel(-b / a)]
    if degree == 2:
        c = coefficients.get(0, _ZERO)
        root = sqrt(expand(b**2 - 4 * a * c))
        return [expand((-b - root) / (2 * a)), expand((-b + root) / (2 * a))]
    return None


def _binomial_roots(value: Expr, n: int, z: Symbol) -> Optional[list[Expr]]:
    """The ``n`` roots of ``z**n == value``: ``+-sqrt(value)`` for ``n = 2``,
    the principal root times the roots of unity otherwise."""
    if n == 2:
        root = sqrt(value)
        return [-root, root]
    unity = roots(pow(z, Integer(n)) - 1, z)
    if not isinstance(unity, dict) or sum(unity.values()) != n:
        return None
    principal = pow(value, Rational(1, n))
    return [expand_mul(u * principal) for u in unity if isinstance(u, Expr)]


def _residue_values(F: _Field[T], R: DensePoly[T], z: Symbol) -> Optional[list[Expr]]:
    """The nonzero roots of the residue polynomial in closed form, or
    ``None`` when some root has none."""
    values: list[Expr] = []
    for e in _factor_exprs(F, R, z):
        coefficients = _powers_of(e, z)
        degree = max(coefficients)
        if degree == 0:
            continue
        if degree == 1 and coefficients.get(0, _ZERO) == 0:
            continue
        found = _closed_roots(e, z)
        if found is None:
            return None
        values.extend(_denested(root) for root in found)
    return [value for value in values if value != 0]


def _denested(value: Expr) -> Expr:
    """A residue with parameters with its roots of squares taken
    (``sqrt(c**2)`` is ``c``: the sign of a residue is a choice, the
    conjugate function carrying the opposite one)."""
    if not value.free_symbols:
        return value
    return powdenest(factor(value), force=True)


def _algebraic_atoms(values: list[Expr]) -> list[Expr]:
    """The algebraic numbers of the residues ``values``: the radicals and
    ``I`` without symbols, and the values which are irrational constants
    (sorted as strings)."""
    numbers: set[Expr] = set()
    for value in values:
        for node in value.atoms(Pow, ImaginaryUnit):
            if node.free_symbols:
                continue
            if isinstance(node, ImaginaryUnit):
                numbers.add(I)
            elif isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q > 1:
                numbers.add(node)
        if not value.free_symbols and not isinstance(value, Rational):
            numbers.add(value)
    return sorted(numbers, key=str)


class _ResidueClass(Generic[U]):
    """The points whose residues are rational multiples ``q*rho`` of one
    value: for each ``q > 0`` the polynomial ``G`` whose roots are the
    abscissae, the branch ``W`` of ``w`` modulo ``G`` on which the residue
    is ``+q*rho``, and the multiple ``q_infinity`` of ``rho`` at the point
    at infinity where ``w ~ +sqrt(lc Q) x**m`` (its conjugate carries the
    opposite residue)."""

    def __init__(self, rho: Expr, finite: list[tuple[Rational, DensePoly[U], DensePoly[U]]],
                 q_infinity: Rational) -> None:
        self.rho: Final[Expr] = rho
        self.finite: Final[list[tuple[Rational, DensePoly[U], DensePoly[U]]]] = finite
        self.q_infinity: Final[Rational] = q_infinity

    @property
    def denominator(self) -> int:
        return math.lcm(*[q.q for q, _, _ in self.finite], self.q_infinity.q)


def _classes(values: list[U], infinity: Optional[U], L: _Field[U], r: DensePoly[U], D: DensePoly[U],
             Q: DensePoly[U]) -> Optional[list[_ResidueClass[U]]]:
    """The residues (``values`` at the finite points, ``infinity`` at the
    point at infinity where ``w ~ +sqrt(lc Q) x**m``) partitioned into
    classes of rational multiples, with the data of each class."""
    K = L.domain
    representatives: list[U] = []
    members: list[list[U]] = []
    all_values = list(values)
    if infinity is not None:
        all_values.append(infinity)
    for value in all_values:
        for k, representative in enumerate(representatives):
            if isinstance(L.expr(K.exquo(value, representative)), Rational):
                members[k].append(value)
                break
        else:
            representatives.append(value)
            members.append([value])
    classes: list[_ResidueClass[U]] = []
    for representative in representatives:
        rho = L.expr(representative)
        finite: list[tuple[Rational, DensePoly[U], DensePoly[U]]] = []
        seen: set[Rational] = set()
        for value in values:
            ratio = L.expr(K.exquo(value, representative))
            if not isinstance(ratio, Rational):
                continue
            q = Rational(abs(ratio.p), ratio.q)
            if q in seen:
                continue
            seen.add(q)
            residue = value if ratio.p > 0 else K.neg(value)
            G = D.gcd(r * r - (D.diff()**2 * Q).mul_ground(K.mul(residue, residue)))
            if G.degree == 0:
                return None
            W = (r * _inverse(D.diff().mul_ground(residue), G)).rem(G)
            finite.append((q, G, W))
        q_infinity = Rational(0)
        if infinity is not None:
            ratio = L.expr(K.exquo(infinity, representative))
            if isinstance(ratio, Rational):
                q_infinity = ratio
        classes.append(_ResidueClass(rho, finite, q_infinity))
    return classes


def _lift(W: DensePoly[T], G: DensePoly[T], Q: DensePoly[T], N: int) -> DensePoly[T]:
    """``W`` with ``W**2 == Q`` modulo ``G**N`` (Newton's iteration from
    ``W**2 == Q`` modulo ``G``)."""
    modulus = G**N
    two = Q.domain.from_int(2)
    steps = max(1, (N - 1).bit_length() + 1)
    for _ in range(steps):
        W = ((W + Q * _inverse(W, modulus)).exquo_ground(two)).rem(modulus)
    return W


def _series_at_infinity(Q: DensePoly[T], order: int) -> list[T]:
    """The coefficients ``sigma_0 = 1, sigma_1, ...`` of ``sqrt(Q(1/t)
    t**(deg Q) / lc(Q))`` as a power series in ``t`` up to ``t**order``."""
    K = Q.domain
    q = list(Q.monic().rep)  # highest degree first
    q = q + [K.zero] * max(0, order + 1 - len(q))
    sigma = [K.one]
    two = K.from_int(2)
    for k in range(1, order + 1):
        total = q[k] if k < len(q) else K.zero
        for i in range(1, k):
            total = K.sub(total, K.mul(sigma[i], sigma[k - i]))
        sigma.append(K.exquo(total, two))
    return sigma


def _coefficients(p: DensePoly[T], size: int) -> list[T]:
    """The coefficients of ``p`` from degree 0 to ``size - 1``."""
    coefficients = list(reversed(p.rep))
    return coefficients + [p.domain.zero] * (size - len(coefficients))


def _function_with_divisor(cls: _ResidueClass[U], n: int, Q: DensePoly[U], s: Optional[U],
                           x: Symbol) -> Optional[tuple[DensePoly[U], DensePoly[U]]]:
    """``u, v`` with ``u + v*w`` vanishing to order ``2*n*q`` at the points
    of the class with residue ``q*rho`` and with pole orders at infinity
    ``n*(sigma -+ q_infinity)``, ``sigma`` the number of finite zeros;
    ``None`` when no such function exists (``n*Delta`` is not principal
    on the curve)."""
    K = Q.domain
    deg_Q = Q.degree
    m = deg_Q // 2
    sigma = sum((q.p * n // q.q) * G.degree for q, G, _ in cls.finite)
    q_inf = int(Fraction(cls.q_infinity.p, cls.q_infinity.q) * n)
    if deg_Q % 2:
        du, dv = sigma, sigma - m - 1
        E_plus = E_minus = sigma
    else:
        E_plus, E_minus = sigma - q_inf, sigma + q_inf
        E = max(E_plus, E_minus)
        du, dv = E, E - m
    if du < 0:
        return None
    nu, nv = du + 1, max(dv + 1, 0)
    unknowns = nu + nv
    rows: list[list[U]] = []
    for q, G, W in cls.finite:
        N = 2 * q.p * n // q.q
        modulus = G**N
        W_N = _lift(W, G, Q, N)
        size = modulus.degree
        columns: list[list[U]] = []
        for i in range(nu):
            columns.append(_coefficients(_monomial(K.one, i, K, x).rem(modulus), size))
        for i in range(nv):
            columns.append(_coefficients((_monomial(K.one, i, K, x) * W_N).rem(modulus), size))
        for k in range(size):
            rows.append([column[k] for column in columns])
    if deg_Q % 2 == 0 and q_inf != 0:
        if s is None:
            return None
        order = max(0, nv - 1 + m - min(E_plus, E_minus) - 1)
        sigma_series = _series_at_infinity(Q, order)
        for sign_, bound in ((K.one, E_plus), (K.neg(K.one), E_minus)):
            for j in range(bound + 1, du + 1):
                row: list[U] = [K.one if i == j else K.zero for i in range(nu)]
                for k in range(nv):
                    index = k + m - j
                    value = sigma_series[index] if 0 <= index < len(sigma_series) else K.zero
                    row.append(K.mul(K.mul(sign_, s), value))
                rows.append(row)
    if not rows:
        vector: list[U] = [K.one] + [K.zero] * (unknowns - 1)
    else:
        space = K.nullspace(rows, unknowns)
        if not space:
            return None
        vector = space[0]
    u = DensePoly(list(reversed(vector[:nu])), K, x)
    v = DensePoly(list(reversed(vector[nu:nu + nv])), K, x)
    if u.is_zero and v.is_zero:
        return None
    # the norm must be a constant times the product of the G_j**(2 n q_j)
    norm = u * u - v * v * Q
    expected = Q.new([K.one])
    for q, G, _ in cls.finite:
        expected = expected * G**(2 * q.p * n // q.q)
    if norm.is_zero or norm.degree != expected.degree or not (norm - expected.mul_ground(norm.LC)).is_zero:
        return None
    return u, v


class _Pair:
    """``a0 + a1*w`` with ``a0``, ``a1`` rational functions of ``x`` on
    the curve ``w**2 == Q``."""

    def __init__(self, a0: Expr, a1: Expr, Q: Expr, x: Symbol) -> None:
        self.a0: Final[Expr] = a0
        self.a1: Final[Expr] = a1
        self.Q: Final[Expr] = Q
        self.x: Final[Symbol] = x

    def __mul__(self, other: _Pair) -> _Pair:
        return _Pair(cancel(self.a0 * other.a0 + self.a1 * other.a1 * self.Q),
                     cancel(self.a0 * other.a1 + self.a1 * other.a0), self.Q, self.x)

    def __sub__(self, other: _Pair) -> _Pair:
        return _Pair(cancel(self.a0 - other.a0), cancel(self.a1 - other.a1), self.Q, self.x)

    def scaled(self, c: Expr) -> _Pair:
        return _Pair(cancel(c * self.a0), cancel(c * self.a1), self.Q, self.x)

    def derivative(self) -> _Pair:
        # (a1 w)' = a1' w + a1 Q'/(2 w) = (a1' + a1 Q'/(2 Q)) w
        return _Pair(cancel(diff(self.a0, self.x)),
                     cancel(diff(self.a1, self.x) + self.a1 * diff(self.Q, self.x) / (2 * self.Q)), self.Q, self.x)

    def inverse(self) -> _Pair:
        norm = cancel(self.a0**2 - self.a1**2 * self.Q)
        return _Pair(cancel(self.a0 / norm), cancel(-self.a1 / norm), self.Q, self.x)

    def logarithmic_derivative(self) -> _Pair:
        return self.derivative() * self.inverse()

    def is_zero(self) -> bool:
        return self.a0 == 0 and self.a1 == 0


class _Result:
    """The pieces of the integral of ``A + B*y``: the expressions of the
    radicand ``P = d**2*Q``, the rational part, the exact part ``sum(S*w/T)``
    of the Hermite reduction, the logarithms, the remainder, the
    decision and the residues algebraic over the parameters which left
    the logarithmic part undecided."""

    def __init__(self, P: Expr, d: Expr, Q: Expr, rational_part: Expr, rational_logs: Expr,
                 terms: list[tuple[Expr, Expr]], logs: list[Expr], remainder: _Pair, nonelementary: Optional[bool],
                 generators: Optional[list[Expr]] = None) -> None:
        self.P: Final[Expr] = P
        self.d: Final[Expr] = d
        self.Q: Final[Expr] = Q
        self.rational_part: Final[Expr] = rational_part
        self.rational_logs: Final[Expr] = rational_logs
        self.terms: Final[list[tuple[Expr, Expr]]] = terms
        self.logs: Final[list[Expr]] = logs
        self.remainder: Final[_Pair] = remainder
        self.nonelementary: Final[Optional[bool]] = nonelementary
        self.generators: Final[list[Expr]] = list(generators or [])

    @property
    def w(self) -> Expr:
        return sqrt(self.P) / self.d

    def exact_part(self) -> Expr:
        w = self.w
        return self.rational_part + add(*[S * w / T_ for S, T_ in self.terms])

    def elementary(self) -> Expr:
        return self.exact_part() + self.rational_logs + add(*self.logs)

    def remainder_expr(self) -> Expr:
        # a1 w = a1 Q d / sqrt(P): the radical in the denominator
        return self.remainder.a0 + cancel(self.remainder.a1 * self.Q) * self.d / sqrt(self.P)


def _integrate(f: Expr, x: Symbol, logarithms: bool) -> Optional[_Result]:
    parsed = _parse(f, x)
    if parsed is None:
        return None
    exprs = [parsed.base] + list(fraction(parsed.A)) + list(fraction(parsed.B))
    parameters, gaussian, constants = _constant_kind(exprs, x)
    if constants:
        return _integrate_over(_expressions(), parsed, x, logarithms)
    if gaussian:
        if parameters:
            return _integrate_over(_algebraic_parametric(_gaussian(), parameters), parsed, x, logarithms)
        return _integrate_over(_algebraic(_gaussian()), parsed, x, logarithms)
    if parameters:
        return _integrate_over(_parametric(parameters), parsed, x, logarithms)
    return _integrate_over(_rationals(), parsed, x, logarithms)


def _integrate_over(F: _Field[T], parsed: _Parsed, x: Symbol, logarithms: bool) -> _Result:
    from .risch.rationaltools import ratint, ratint_ratpart
    radical = _Radical(F, x, F.from_expr(parsed.base, x), parsed.A, parsed.B)
    Q = radical.Q
    A = radical.A
    numerator, denominator = fraction(cancel(A))
    if A == 0:
        rational_part: Expr = _ZERO
        rational_logs: Expr = _ZERO
    elif logarithms:
        rational_part, rational_logs = ratint(A, x), _ZERO
    else:
        p, q = F.from_expr(numerator, x), F.from_expr(denominator, x)
        quotient, rest = p.div(q)
        rational_part, rational_logs = ratint_ratpart(F.poly_expr(rest), F.poly_expr(q), x)
        rational_part = rational_part + F.poly_expr(quotient.integrate())
    if Q.degree == 0:
        # sqrt(P) is d*sqrt(c) where d > 0 and -d*sqrt(c) where d < 0: a
        # rational integrand on each component, the antiderivative
        # carrying sign(d) ((b**2/(4*c) + b*x + c*x**2)**(-3/2) is
        # c**(-3/2)*Abs(x + b/(2*c))**(-3))
        constant = sqrt(radical.Q_expr)
        g = radical.B * radical.d_expr * constant
        extra = ratint(g, x) if logarithms else _ZERO
        orientation = sign(radical.d_expr) if radical.d.degree >= 1 else _ONE
        remainder = _Pair(_ZERO if logarithms else g * orientation, _ZERO, radical.Q_expr, x)
        return _Result(radical.P_expr, radical.d_expr, radical.Q_expr, rational_part + orientation * extra,
                       rational_logs, [], [], remainder, False)
    M, D = radical.algebraic_numerator()
    reduction = _hermite(M, D, Q, x)
    r, D = reduction.r, reduction.D
    terms = [(F.poly_expr(S), F.poly_expr(T_)) for S, T_ in reduction.terms]
    Q_expr = radical.Q_expr
    remainder = _Pair(_ZERO, cancel(F.poly_expr(r) / (F.poly_expr(D) * Q_expr)), Q_expr, x)
    m = Q.degree // 2
    if reduction.polynomial.degree >= m and m >= 1:
        # a pole of order at least two at infinity which no function of
        # the field removes: not elementary
        return _Result(radical.P_expr, radical.d_expr, Q_expr, rational_part, rational_logs, terms, [], remainder,
                       True)
    if not logarithms or r.is_zero:
        return _Result(radical.P_expr, radical.d_expr, Q_expr, rational_part, rational_logs, terms, [], remainder,
                       False if r.is_zero else None)
    logs, remainder, nonelementary, generators = _logarithmic_part(F, reduction, Q, x, remainder, radical.w)
    return _Result(radical.P_expr, radical.d_expr, Q_expr, rational_part, rational_logs, terms, logs, remainder,
                   nonelementary, generators)


#: the logarithmic terms (expressions), the remainder, the decision and
#: the residues algebraic over the parameters which left it undecided
_Logs = tuple[list[Expr], _Pair, Optional[bool], list[Expr]]


def _logarithmic_part(F: _Field[T], reduction: _Reduction[T], Q: DensePoly[T], x: Symbol, remainder: _Pair,
                      w: Expr) -> _Logs:
    """The logarithmic terms of the classes of residues whose divisors
    are torsion, the remainder ``h - sum(dlog)``, the decision, and the
    residues algebraic over the parameters when they left it undecided."""
    r, D = reduction.r, reduction.D
    z = _fresh('z')
    R = _residue_polynomial(r, D, Q, z)
    values = _residue_values(F, R, z)
    if values is None:
        return [], remainder, None, []
    m = Q.degree // 2
    infinity: Optional[Expr] = None
    s_value: Optional[Expr] = None
    if Q.degree % 2 == 0:
        s_value = _denested(sqrt(F.expr(Q.LC)))
        if reduction.polynomial.degree == m - 1:
            infinity = -F.expr(reduction.polynomial.LC) / s_value
    if not values and infinity is None:
        # no residue anywhere: the remainder, with simple poles only, is a
        # differential of the first kind, whose integral is elementary only
        # when it is zero (its integral would be a function of the field
        # without poles, a constant)
        return [], remainder, not remainder.is_zero(), []
    generators = list(values)
    if infinity is not None:
        generators.append(infinity)
    if s_value is not None:
        generators.append(s_value)
    undecided: _Logs = ([], remainder, None, [value for value in generators if value.free_symbols])
    numbers = _algebraic_atoms(generators)
    parameters = F.parameters
    if not numbers:
        if parameters:
            return _logarithms_in(_parametric(parameters), F, reduction, Q, x, remainder, w, values, infinity,
                                  s_value, undecided)
        return _logarithms_in(_rationals(), F, reduction, Q, x, remainder, w, values, infinity, s_value, undecided)
    try:
        N = _NumberField(numbers)
    except ArithmeticError:
        return undecided
    if parameters:
        return _logarithms_in(_algebraic_parametric(N, parameters), F, reduction, Q, x, remainder, w, values,
                              infinity, s_value, undecided)
    return _logarithms_in(_algebraic(N), F, reduction, Q, x, remainder, w, values, infinity, s_value, undecided)


def _logarithms_in(L: _Field[U], F: _Field[T], reduction: _Reduction[T], Q0: DensePoly[T], x: Symbol,
                   remainder: _Pair, w: Expr, values: list[Expr], infinity_value: Optional[Expr],
                   s_value: Optional[Expr], undecided: _Logs) -> _Logs:
    """The logarithmic part over the field ``L`` of the residues."""
    try:
        elements = [L.element(value) for value in values]
        infinity = None if infinity_value is None else L.element(infinity_value)
        s = None if s_value is None else L.element(s_value)
    except _FAILURES:
        return undecided
    K = L.domain
    r, D, Q = L.poly(reduction.r), L.poly(reduction.D), L.poly(Q0)
    classes = _classes(elements, infinity, L, r, D, Q)
    if classes is None:
        return [], remainder, None, []
    genus_one = Q.degree in (3, 4)
    logs: list[Expr] = []
    undecided_ = False
    Q_expr = F.poly_expr(Q0)
    for cls in classes:
        found: Optional[tuple[DensePoly[U], DensePoly[U]]] = None
        n0 = cls.denominator
        n = n0
        for k in range(1, _TORSION_BOUND + 1):
            n = k * n0
            found = _function_with_divisor(cls, n, Q, s, x)
            if found is not None:
                break
        if found is None:
            if not (genus_one and L.rationals):
                undecided_ = True
            continue
        u, v = found
        rho = cls.rho
        # (u + v w)(u - v w) is a constant times the denominator squared:
        # the conjugate gives the reciprocal function and the opposite
        # residue; the sign with positive leading coefficients is taken
        if not v.is_zero and L.expr(K.mul(u.LC, v.LC)).is_negative is True:
            v, rho = -v, -rho
        if L.expr(u.LC).is_negative is True:
            u, v = -u, -v
        denominator = Q.new([K.one])
        for q, G, _ in cls.finite:
            denominator = denominator * G**(q.p * n // q.q)
        u_expr, v_expr, d_expr = L.poly_expr(u), L.poly_expr(v), L.poly_expr(denominator)
        logs.append(rho / n * log((u_expr + v_expr * w) / d_expr))
        pair = _Pair(u_expr / d_expr, v_expr / d_expr, Q_expr, x)
        remainder = remainder - pair.logarithmic_derivative().scaled(rho / n)
    if remainder.is_zero():
        return logs, remainder, False, []
    return logs, remainder, None if undecided_ else True, []


# ---------------------------------------------------------------------------
# The verification

def _positive_at(e: Expr, values: dict[Basic, Basic]) -> bool:
    """Whether ``e`` is positive at the rational ``values``."""
    v = _replaced(e, values)
    if isinstance(v, Rational):
        return v.p > 0
    found = reliable_value(v, 30)
    return found is not None and found.is_positive is True


def _verified(F: Expr, f: Expr, x: Symbol, P: Expr) -> bool:
    """Whether ``F'`` agrees with ``f`` numerically at points where the
    radicand is positive."""
    # sign(d) in F differentiates to a delta function, which is 0 at every
    # sample point off the zeros of d
    derivative = as_expr(diff(F, x).replace(
        lambda node: isinstance(node, DiracDelta) or (isinstance(node, Derivative) and isinstance(node.expr, Sign)),
        lambda node: _ZERO))
    rng = random.Random(str(f))
    parameters = _symbols(f)
    parameters = [p for p in parameters if p != x]
    # every square root of the integrand as given must be real at the
    # sample (sqrt(x + 1)*sqrt(x + 2) is -sqrt((x + 1)*(x + 2)) below -2)
    radicands = [P] + [node.base for node in _pows(f) if node.has(x) and _half_integer(node)]
    checked = 0
    precision = 120
    for _ in range(40):
        values: dict[Basic, Basic] = {p: Rational(rng.randint(-400, 400), 100) for p in parameters}
        values[x] = Rational(rng.randint(-400, 400), 100)
        try:
            if any(not _positive_at(radicand, values) for radicand in radicands):
                continue
            expected = reliable_value(f, 30, _symbol_values(values))
            found = reliable_value(derivative, 30, _symbol_values(values))
        except _FAILURES:
            continue
        if expected is None or found is None:
            continue
        e = numeric_value(expected, precision)
        g = numeric_value(found, precision)
        if e is None or g is None:
            continue
        with mpmath.workprec(precision):
            if abs(g - e) > mpmath.mpf('1e-20') * (1 + abs(e)):
                return False
        checked += 1
        if checked >= 4:
            return True
    return checked > 0


def _symbol_values(values: dict[Basic, Basic]) -> dict[Symbol, Expr]:
    return {s: as_expr(v) for s, v in values.items() if isinstance(s, Symbol)}


# ---------------------------------------------------------------------------
# The public functions

def _curve_reduce(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    from .curves import curve_reduce
    return curve_reduce(f, x)


def trager_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` with ``f == F' + h``: ``F`` the elementary part of the
    integral of the algebraic function ``f`` (the integral of the
    rational part, the exact part of the Hermite reduction and the
    logarithms of the residue classes whose divisors are torsion) and
    ``h`` the remaining integrand, with simple poles only, whose integral
    is not elementary or not found; ``None`` when ``f`` is not an
    algebraic function of ``x`` (one square root, one root of any index,
    or several and nested radicals, these through
    :mod:`sympy_extras.integrals.curves`).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.trager import trager_reduce
    >>> x = symbols('x')
    >>> F, h = trager_reduce((x**3 + 1)/((x - 1)**2*sqrt(x**3 + 1)), x)
    >>> F
    -sqrt(x**3 + 1)/(x - 1)
    >>> h
    3*x**2/((2*x - 2)*sqrt(x**3 + 1))
    >>> trager_reduce(1/sqrt(x**3 + 1), x)
    (0, 1/sqrt(x**3 + 1))
    >>> trager_reduce((2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1)), x)
    (-I*log((-I*x + sqrt(x**4 + x**2 + 1))/(x**2 + 1))/2, 3/(2*sqrt(x**4 + x**2 + 1)))
    """
    return _trager_reduce(to_expr(f), x, _curve_reduce)


def _trager_reduce(f_: Expr, x: Symbol, curve: CurveReducer) -> Optional[tuple[Expr, Expr]]:
    """:func:`trager_reduce` with the reduction on a general curve
    ``curve`` (the face's writes the residues of fields without radicals
    with SymPy's ``CRootOf``)."""
    g, rational_roots = _polynomial_radicands(f_, x)
    result = attempt(lambda: _integrate(g, x, True), settings.timeout, native=True)
    if result is None:
        found = attempt(lambda: _index_n(f_, x), settings.timeout, native=True)
        if found is None:
            # several or nested radicals, roots of any index together:
            # Trager's algorithm on the curve of a primitive element
            found = attempt(lambda: curve(f_, x), settings.timeout, native=True)
        return found
    back: dict[Basic, Basic] = {}
    original = result.P
    if result.nonelementary is None and result.generators:
        # residues algebraic over the parameters: the parameters made
        # rational (a = delta**2/c for sqrt(a*c)) and the algorithm rerun
        made = _rational_parameters(result.generators, [s for s in _symbols(g) if s != x])
        if made is not None:
            substitution, back_ = made
            g_ = _replaced(g, dict(substitution))
            again = attempt(lambda: _integrate(g_, x, True), settings.timeout, native=True)
            if again is not None and again.nonelementary is not None:
                result, g = again, g_
                rational_roots = [(expand(_replaced(product, dict(substitution))), _replaced(y, dict(substitution)))
                                  for product, y in rational_roots]
                back = dict(back_)
    P = result.P
    F = _restored(_original_roots(result.elementary(), g, x, P), rational_roots)
    h = _restored(_original_roots(result.remainder_expr(), g, x, P), rational_roots)
    if back:
        # the radicand in the original parameters again (a + b*x**2 + c*x**4,
        # not the square of the reparametrization expanded)
        P_ = _replaced(P, back)
        F, h = _replaced(F, back), _replaced(h, back)
        F, h, P = _restored(F, [(expand(P_), sqrt(original))]), _restored(h, [(expand(P_), sqrt(original))]), original
    if not _verified(F, f_ - h, x, P):
        return None
    return F, h


def _rational_risch_de(f: Expr, g: Expr, x: Symbol) -> Optional[Expr]:
    """A rational function ``y`` of ``x`` with ``y' + f*y == g``, for
    ``f``, ``g`` rational in ``x`` (the Risch differential equation over
    ``QQ(x)``, :func:`sympy_extras.integrals.risch.rde.rischDE`); ``None``
    when there is none or the algorithm gives up."""
    from .risch import risch
    from .risch.rde import rischDE
    from .risch.tower import RationalFunction, TowerPoly
    fe = RationalFunction.from_expr(f)
    ge = RationalFunction.from_expr(g)

    def solve() -> Expr:
        DE = risch.DifferentialExtension.manual([TowerPoly.one(x)])
        fa, fd = risch.frac_in(fe, x)
        ga, gd = risch.frac_in(ge, x)
        try:
            ya, yd = rischDE(fa, fd, ga, gd, DE)
        except risch.NonElementaryIntegralException:
            raise NotImplementedError('no rational solution')
        return (ya.as_element() / yd.as_element()).as_expr()

    return attempt(solve, settings.timeout, native=True)


def _index_n(f: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` for ``f`` rational in ``x`` and in one root ``y = P**(1/n)``
    of index ``n >= 3`` of a squarefree polynomial: ``f`` is
    ``sum(A_k*y**k)`` in the integral basis ``1, y, ..., y**(n - 1)``, and
    the derivative of ``R*y**k`` stays in the component ``y**k`` (``y' =
    P'*y/(n*P)``), so each component integrates on its own, ``R`` a
    rational solution of the Risch differential equation ``R' +
    k*P'/(n*P)*R = A_k``, or, without one, the Hermite reduction and the
    logarithmic part on the curve ``y**n == P``
    (:func:`_component_with_logarithms`); ``h`` collects the components
    left undecided."""
    from .risch.rationaltools import ratint
    parsed = _root_components(f, x)
    if parsed is None:
        return None
    P_, n, components = parsed
    F: Expr = ratint(components[0], x) if components[0] != 0 else _ZERO
    h: Expr = _ZERO
    for k in range(1, n):
        A = components[k]
        if A == 0:
            continue
        y = _rational_risch_de(cancel(k * diff(P_, x) / (n * P_)), cancel(A), x)
        if y is None:
            # no rational solution: the logarithmic part on the curve
            found = _component_with_logarithms(A, P_, n, k, x)
            if found is None:
                h = h + A * pow(P_, Rational(k, n))
            else:
                F = F + found
            continue
        F = F + cancel(y) * pow(P_, Rational(k, n))
    if not _verified(F, f - h, x, P_):
        return None
    return F, h


# ---------------------------------------------------------------------------
# Roots of index n >= 3: the logarithmic part on the curve y**n = P

#: the largest degree over the rationals of the field of the residues on a
#: curve ``y**n = P`` for which the logarithmic part is searched (the
#: linear algebra runs over that field)
_INDEX_N_FIELD_DEGREE: Final[int] = 12

#: the time allowed to the construction of that field (the primitive
#: element of nested radicals may not come within the limit of the
#: integral)
_FIELD_SECONDS: Final[float] = 2.0


def _component_with_logarithms(A: Expr, base: Expr, n: int, k: int, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the component ``A*y**k`` (``A`` rational in
    ``x``, ``y**n == P`` squarefree, ``1 <= k < n``) with logarithms of
    functions of the curve, verified by differentiation; ``None`` when it
    is not found. A component with ``gcd(k, n) > 1`` is one of the index
    ``n/gcd(k, n)`` (``A*y**2`` on ``y**4 == P`` is ``A*sqrt(P)``)."""
    g = math.gcd(k, n)
    if g > 1:
        n, k = n // g, k // g
    if n == 2:
        result = _integrate(A * sqrt(base), x, True)
        if result is None or not result.remainder.is_zero():
            return None
        return result.elementary()
    if [s for s in _symbols(base) + _symbols(A) if s != x]:
        # the residues of a parametric integrand lie in an algebraic
        # extension of the field of the parameters: not this route
        return None
    try:
        P = _rationals().from_expr(base, x)
    except ValueError:
        return None
    reduced = _hermite_index_n(A, P, n, k, x)
    if reduced is None:
        return None
    terms, M, D = reduced
    y = pow(base, Rational(1, n))
    exact: Expr = add(*[S_.to_expr() / T_.to_expr() * pow(y, Integer(k)) for S_, T_ in terms])
    if M.is_zero:
        found: Expr = exact
    else:
        logs = _logarithms_index_n(M, D, P, n, k, x)
        if logs is None:
            return None
        found = exact + logs
    if not _verified(found, A * pow(y, Integer(k)), x, base):
        return None
    return found


def _hermite_index_n(A: Expr, P: DensePoly[Fraction], n: int, k: int, x: Symbol
                     ) -> Optional[tuple[list[tuple[DensePoly[Fraction], DensePoly[Fraction]]], DensePoly[Fraction],
                                         DensePoly[Fraction]]]:
    """The Hermite reduction of ``A*y**k dx`` on ``y**n == P``: the terms
    ``(S, T)`` of the exact part ``sum(S*y**k/T)`` and ``M, D`` with
    ``A*y**k == sum((S*y**k/T)') + M*y**k/(D*P)``, ``D`` squarefree and
    coprime to ``P`` (the remainder has simple poles at the finite
    points) and the polynomial part of ``M/D`` of degree below ``deg P -
    1``; ``None`` when the remainder keeps a pole of order two or more at
    infinity, which no function of the curve removes (the integral is
    then not elementary).

    The multiple factor ``V**e`` of the denominator coprime to ``P`` is
    lowered by ``S*y**k/V**(e - 1)``, the factor ``P**j`` (``j >= 2``, the
    ramified points, where a pole of ``P`` of order one is no pole of the
    differential) by ``S*y**k/P**(j - 1)``, and the polynomial part by
    ``c*x**i*y**k``, whose derivative has the numerator ``c*(i + k*deg
    P/n)*lc(P)*x**(i + deg P - 1) + ...`` over ``P``.

    >>> from fractions import Fraction
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.polyclasses import DensePoly
    >>> from sympy_extras.integrals.trager import _hermite_index_n
    >>> x = symbols('x')
    >>> P = DensePoly.from_expr(x**3 + 1, x, QQ)
    >>> terms, M, D = _hermite_index_n(1/(x**4*(x**3 + 1)), P, 3, 2, x)
    >>> D.to_expr(), [T.to_expr() for _, T in terms]
    (x, [x**3])
    """
    K = QQ
    F = _rationals()
    numerator, denominator = fraction(cancel(A))
    M = F.from_expr(numerator, x)
    D = F.from_expr(denominator, x)
    j = 0
    while True:
        quotient, remainder = D.div(P)
        if not remainder.is_zero:
            break
        D, j = quotient, j + 1
    if j == 0:
        M, j = M * P, 1
    terms: list[tuple[DensePoly[Fraction], DensePoly[Fraction]]] = []
    kn = Fraction(k, n)
    while j >= 2:
        c = Fraction(j - 1) - kn
        S_ = -(M * _inverse((D * P.diff()).mul_ground(c), P)).rem(P)
        M = (M - D * (S_.diff() * P - (S_ * P.diff()).mul_ground(c))).exquo(P)
        terms.append((S_, P**(j - 1)))
        j -= 1
    while True:
        common = M.gcd(D)
        M, D = M.exquo(common), D.exquo(common)
        _, factors = D.sqf_list()
        ramified = [(V.gcd(P), e) for V, e in factors if V.gcd(P).degree > 0]
        if ramified:
            # a factor V**e of D dividing P: with P = V*P1 the differential
            # is M*y**k/(V**(e + 1)*U*P1), a pole of order n*e - k + 1 at
            # the ramified points, removed down to e = 0 by S*y**k/V**e
            V, e = ramified[0]
            U_ = D.exquo(V**e)
            P1 = P.exquo(V)
            c = Fraction(e) - kn
            S_ = -(M * _inverse((U_ * V.diff() * P1).mul_ground(c), V)).rem(V)
            subtracted = S_.diff() * V * P1 + (S_ * V * P1.diff()).mul_ground(kn) - (S_ * V.diff() * P1).mul_ground(c)
            M = (M - U_ * subtracted).exquo(V)
            D = U_ * V**(e - 1)
            terms.append((S_, V**e))
            continue
        multiple = [(V, e) for V, e in factors if e >= 2]
        if not multiple:
            break
        V, e = max(multiple, key=lambda pair: pair[1])
        U_ = D.exquo(V**e)
        c = Fraction(e - 1)
        S_ = -(M * _inverse((U_ * V.diff() * P).mul_ground(c), V)).rem(V)
        subtracted = S_.diff() * V * P - (S_ * V.diff() * P).mul_ground(c) + (S_ * P.diff() * V).mul_ground(kn)
        M = (M - U_ * subtracted).exquo(V)
        D = U_ * V**(e - 1)
        terms.append((S_, V**(e - 1)))
    q, r = M.div(D)
    d = P.degree
    S_ = P.new([])
    while not q.is_zero and q.degree >= d - 1:
        i = q.degree - d + 1
        c = q.LC / (P.LC * (i + Fraction(k * d, n)))
        monomial = _monomial(c, i, K, x)
        S_ = S_ + monomial
        q = q - (monomial.diff() * P + (monomial * P.diff()).mul_ground(kn))
    if not S_.is_zero:
        terms.append((S_, P.new([K.one])))
    # the order of q*y**k/P dx at a point at infinity is
    # -e*(deg q + 1 - d + k*d/n) - 1, e = n/gcd(n, d): a pole of order two
    # or more when n*(deg q + 1 - d) + k*d >= gcd(n, d)
    if not q.is_zero and n * (q.degree + 1 - d) + k * d >= math.gcd(n, d):
        return None
    M = q * D + r
    common = M.gcd(D)
    M, D = M.exquo(common), D.exquo(common)
    return terms, M, D


class _Places:
    """The places of the curve ``y**n == P`` above the roots of ``G``
    (unramified: ``G`` coprime to ``P``) on the branch ``y == W`` modulo
    ``G``, where the differential has the residue ``rho``; ``group`` is
    the index of the norm ``rho**n`` of the residue (the places above
    the same roots on the ``n`` branches share it)."""

    def __init__(self, G: DensePoly[Number], W: DensePoly[Number], rho: Number, group: int, P: DensePoly[Number],
                 n: int) -> None:
        self.G: Final[DensePoly[Number]] = G
        self.W: Final[DensePoly[Number]] = W
        self.rho: Final[Number] = rho
        self.group: Final[int] = group
        self.P: Final[DensePoly[Number]] = P
        self.n: Final[int] = n
        #: the branch lifted modulo G**N with the inverse of n*W**(n - 1)
        #: there, for the largest N computed (a cache)
        self.lifted: tuple[int, DensePoly[Number], DensePoly[Number]] = (
            1, W, _inverse((W**(n - 1)).mul_ground(P.domain.from_int(n)), G))

    def branch(self, N: int) -> DensePoly[Number]:
        """``W`` with ``W**n == P`` modulo ``G**N``, by Hensel's lifting
        with the precision doubled at each step (``W - (W**n - P)*I`` and
        ``I*(2 - n*W**(n - 1)*I)`` modulo ``G**(2*N)``), continued from
        the largest precision reached before."""
        precision, W, I_ = self.lifted
        K = self.P.domain
        n_ = K.from_int(self.n)
        two = self.G.new([K.from_int(2)])
        while precision < N:
            precision = min(2 * precision, N)
            modulus = self.G**precision
            W = (W - ((W**self.n - self.P) * I_).rem(modulus)).rem(modulus)
            I_ = (I_ * (two - ((W**(self.n - 1)).mul_ground(n_) * I_).rem(modulus))).rem(modulus)
        self.lifted = (precision, W, I_)
        return W.rem(self.G**N)


class _PlaceAtInfinity:
    """The point at infinity of the curve ``y**n == P`` with ``n`` dividing
    ``deg P`` where ``y ~ s*x**(deg P/n)``, ``s`` an ``n``-th root of the
    leading coefficient, and the residue ``rho`` there."""

    def __init__(self, s: Number, rho: Number) -> None:
        self.s: Final[Number] = s
        self.rho: Final[Number] = rho


class _LogTermN:
    """``(beta/m)*(power*log(U) - log(denominator))`` with ``U =
    sum(U[i]*y**i)`` a function of the curve ``y**n == P`` and
    ``denominator`` a polynomial in ``x``, over the field of the
    residues."""

    def __init__(self, beta: Number, m: int, U: list[DensePoly[Number]], denominator: DensePoly[Number],
                 power: int = 1) -> None:
        self.beta: Final[Number] = beta
        self.m: Final[int] = m
        self.U: Final[list[DensePoly[Number]]] = U
        self.denominator: Final[DensePoly[Number]] = denominator
        self.power: Final[int] = power


def _logarithms_index_n(M: DensePoly[Fraction], D: DensePoly[Fraction], P: DensePoly[Fraction], n: int, k: int,
                        x: Symbol) -> Optional[Expr]:
    """The logarithmic part ``sum(beta_j/m_j*log(u_j))`` of the reduced
    differential ``M*y**k/(D*P) dx`` on ``y**n == P`` (``D`` squarefree and
    coprime to ``P``, simple poles at the finite points, ``gcd(k, n) =
    1``), or ``None`` when it is not found.

    The residue at the place ``(a, y_a)`` above a root ``a`` of ``D`` is
    ``M(a)*y_a**k/(D'(a)*P(a))``, so the norms ``rho**n`` of the residues
    are the roots of ``Res_x(D, tau*D'**n*P**(n - k) - M**n)`` and the
    residues are their ``n``-th roots; the points at infinity carry a
    residue only when ``n`` divides ``deg P``. Each orbit of places under
    ``y -> zeta*y`` is tried first with a function of divisor ``m*(Q_0 -
    sum(Q_l)/n)`` (:func:`_orbit_logarithms`, the resolvent form); when
    an orbit is not torsion by itself or the residues of the orbits are
    dependent over the rationals, Trager's general form
    (:func:`_basis_logarithms`): over a basis ``beta_j`` of the rational
    span of the residues the residue divisor is ``sum(beta_j D_j)`` with
    ``D_j`` of degree zero, and each ``D_j`` with ``m_j*D_j`` principal
    (``m_j`` searched up to the torsion bound) contributes
    ``beta_j/m_j*log(u_j)``. The differential minus the logarithmic
    derivatives found is checked to vanish exactly, which it must when
    the integral is elementary (what is left is a differential of the
    first kind).
    """
    d = P.degree
    at_infinity = (d // n) * n == d
    tau = _fresh('tau')
    R = _resultant_in(D, -(M**n), D.diff()**n * P**(n - k), 1, tau)
    if R.degree >= 1 and max(f.degree for f, _ in R.factor_list()[1]) > 2:
        # the n-th roots of the roots of a cubic are nested radicals of
        # Cardano's formula: their number field is beyond the bound below
        # and its construction does not finish
        return None
    norms = _residue_values(_rationals(), R, tau) if R.degree >= 1 else []
    if norms is None:
        return None
    z = _fresh('z')
    zeta = _primitive_root_of_unity(n, z)
    if zeta is None:
        return None
    generators: list[Expr] = [zeta]
    principal_roots: list[Expr] = []
    for value in norms:
        root = _preferably_real_root(value, n, z)
        if root is None:
            return None
        principal_roots.append(root)
        generators.append(root)
    kappa = _residue_at_infinity(M, D, P, n, k) if at_infinity else Fraction(0)
    s0: Optional[Expr] = None
    if kappa != 0:
        s0 = _preferably_real_root(number_from_fraction(P.LC), n, z)
        if s0 is None:
            return None
        generators.append(s0)
    distinct = sorted({g for g in generators if not isinstance(g, Rational)}, key=str)
    N = attempt(lambda: _NumberField(distinct, _INDEX_N_FIELD_DEGREE), _FIELD_SECONDS, native=True)
    if N is None or N.degree > _INDEX_N_FIELD_DEGREE:
        return None
    L = _algebraic(N)
    K = N.field
    M_, D_, P_ = L.poly(M), L.poly(D), L.poly(P)
    zeta_ = N.coordinates[zeta]
    a, b = _bezout(k, n)
    finite: list[_Places] = []
    for group, root in enumerate(principal_roots):
        rho = L.element(root) if isinstance(root, Rational) else N.coordinates[root]
        norm = K.pow(rho, n)
        G = D_.gcd(M_**n - (D_.diff()**n * P_**(n - k)).mul_ground(norm))
        if G.degree == 0:
            return None
        for _ in range(n):
            # y**k == rho*D'*P/M on the branch with residue rho, and y is
            # (y**k)**a*P**b with a*k + b*n == 1
            W_k = (D_.diff() * P_ * _inverse(M_, G)).mul_ground(rho).rem(G)
            W = (_power_modulo(W_k, a, G) * _power_modulo(P_, b, G)).rem(G)
            finite.append(_Places(G, W, rho, group, P_, n))
            rho = K.mul(rho, zeta_)
    infinite: list[_PlaceAtInfinity] = []
    if s0 is not None:
        s = L.element(s0) if isinstance(s0, Rational) else N.coordinates[s0]
        for _ in range(n):
            infinite.append(_PlaceAtInfinity(s, K.neg(K.mul(K.pow(s, k), K.from_rational(kappa)))))
            s = K.mul(s, zeta_)
    if not finite and not infinite:
        return None
    # the orbit form first (a function of divisor m*(n*Q - sum of the
    # conjugates of Q) for each orbit of places, which has a plain
    # representative), the basis form when an orbit is not torsion by
    # itself or the residues of different orbits are dependent
    logs = _orbit_logarithms(finite, infinite, zeta_, P_, n, x)
    if logs is None or not _logarithmic_derivative_matches(logs, M_, D_, P_, n, k, x):
        # with a single orbit the two searches are equivalent (the
        # differences of its places are torsion when their sum against
        # n times one of them is)
        orbits = len({place.group for place in finite}) + (1 if infinite else 0)
        if orbits == 1:
            return None
        logs = _basis_logarithms(finite, infinite, K, P_, n, x)
        if logs is None or not _logarithmic_derivative_matches(logs, M_, D_, P_, n, k, x):
            return None
    return _real_logarithms(logs, L, P, n, x)


def _orbit_logarithms(finite: list[_Places], infinite: list[_PlaceAtInfinity], zeta: Number,
                      P: DensePoly[Number], n: int, x: Symbol) -> Optional[list[_LogTermN]]:
    """The logarithmic terms ``rho_l/m*log(v(x, zeta**l*y))`` over each
    orbit ``Q_0, ..., Q_(n-1)`` of places under ``y -> zeta*y`` (the
    ``n`` places above the same roots, or the ``n`` points at infinity),
    with ``v`` of divisor ``m*(Q_0 - sum(Q_l)/n)`` and ``rho_l`` the
    residue at the place where ``v(x, zeta**l*y)`` vanishes; ``None``
    when no such ``v`` is found for an orbit within the torsion bound."""
    K = P.domain
    logs: list[_LogTermN] = []
    orbits: list[tuple[list[_Places], list[_PlaceAtInfinity]]] = []
    for group in sorted({place.group for place in finite}):
        orbits.append(([place for place in finite if place.group == group], []))
    if infinite:
        orbits.append(([], infinite))
    coefficients = [Fraction(n - 1, n)] + [Fraction(-1, n)] * (n - 1)
    for places, points in orbits:
        found: Optional[tuple[list[DensePoly[Number]], DensePoly[Number]]] = None
        m = n
        for multiple in range(1, _TORSION_BOUND + 1):
            m = multiple * n
            found = _function_with_divisor_n(places, coefficients if places else [], points,
                                             coefficients if points else [], m, P, n, x)
            if found is not None:
                break
        if found is None:
            return None
        V, denominator = found
        power = 1
        if places:
            V, power = _root_on_the_curve(V, places[0], m, P, n, x)
        for l in range(n):
            # v(x, zeta**l*y) vanishes on the branch zeta**(-l)*W_0, at the
            # point at infinity with s = zeta**(-l)*s_0
            U = [V_i.mul_ground(K.pow(zeta, l * i)) for i, V_i in enumerate(V)]
            rho: Optional[Number] = None
            if places:
                W = places[0].W.mul_ground(K.pow(zeta, (n - l) % n)).rem(places[0].G)
                rho = next((place.rho for place in places if (place.W - W).is_zero), None)
            else:
                s = K.mul(points[0].s, K.pow(zeta, (n - l) % n))
                rho = next((point.rho for point in points if point.s == s), None)
            if rho is None:
                return None
            logs.append(_LogTermN(rho, m, U, denominator, power))
    return logs


def _root_on_the_curve(V: list[DensePoly[Number]], place: _Places, m: int, P: DensePoly[Number], n: int,
                       x: Symbol) -> tuple[list[DensePoly[Number]], int]:
    """``w, mu`` with ``V == c*w**mu`` on the curve for the largest ``mu``
    dividing ``m`` (``V`` vanishes to order ``m`` at the places of
    ``place`` and has its poles at infinity, so ``w`` vanishes there to
    order ``m/mu`` with a ``mu``-th of the poles: the same linear
    system), or ``V, 1``: ``(y - 1)**3`` on ``y**3 = x**2 + 1`` is written
    ``x**2 - 3*y**2 + 3*y`` in the basis ``1, y, y**2``."""
    d = P.degree
    g = math.gcd(n, d)
    B = (n // g) * (m // n) * place.G.degree
    for mu in sorted((mu for mu in range(2, m + 1) if m % mu == 0 and B % mu == 0), reverse=True):
        found = _function_with_divisor_n([place], [Fraction(1)], [], [], m // mu, P, n, x, B // mu)
        if found is None:
            continue
        w, _ = found
        power = _power_on_the_curve(w, mu, P, n, x)
        pairs = [(a, b) for a, b in zip(V, power) if not a.is_zero or not b.is_zero]
        if pairs and all(not a.is_zero and not b.is_zero for a, b in pairs) \
                and all((a * pairs[0][1] - b * pairs[0][0]).is_zero for a, b in pairs):
            return w, mu
    return V, 1


def _power_on_the_curve(w: list[DensePoly[Number]], mu: int, P: DensePoly[Number], n: int,
                        x: Symbol) -> list[DensePoly[Number]]:
    """``w**mu`` in the basis ``1, y, ..., y**(n - 1)`` of the curve
    ``y**n == P``."""
    K = P.domain
    result = [P.new([K.one])] + [P.new([]) for _ in range(n - 1)]
    for _ in range(mu):
        product = [P.new([]) for _ in range(2 * n - 1)]
        for i, a in enumerate(result):
            for j, b in enumerate(w):
                product[i + j] = product[i + j] + a * b
        for i in range(2 * n - 2, n - 1, -1):
            product[i - n] = product[i - n] + product[i] * P
        result = product[:n]
    return result


def _basis_logarithms(finite: list[_Places], infinite: list[_PlaceAtInfinity], K: AlgebraicField,
                      P: DensePoly[Number], n: int, x: Symbol) -> Optional[list[_LogTermN]]:
    """The logarithmic terms ``beta_j/m_j*log(u_j)`` over a basis
    ``beta_j`` of the rational span of the residues, ``u_j`` of divisor
    ``m_j*D_j`` with ``D_j`` the divisor of the coordinates of the
    residues on ``beta_j``; ``None`` when a ``D_j`` is not found
    principal within the torsion bound."""
    residues = [place.rho for place in finite] + [place.rho for place in infinite]
    basis = _rational_basis(residues, K)
    if basis is None:
        return None
    chosen, coordinates = basis
    logs: list[_LogTermN] = []
    for j, index in enumerate(chosen):
        c_finite = [row[j] for row in coordinates[:len(finite)]]
        c_infinite = [row[j] for row in coordinates[len(finite):]]
        n0 = math.lcm(*[c.denominator for c in c_finite + c_infinite])
        found: Optional[tuple[list[DensePoly[Number]], DensePoly[Number]]] = None
        m = n0
        for multiple in range(1, _TORSION_BOUND + 1):
            m = multiple * n0
            found = _function_with_divisor_n(finite, c_finite, infinite, c_infinite, m, P, n, x)
            if found is not None:
                break
        if found is None:
            return None
        U, denominator = found
        logs.append(_LogTermN(residues[index], m, U, denominator))
    return logs


def _preferably_real_root(value: Expr, n: int, z: Symbol) -> Optional[Expr]:
    """A root of the binomial ``z**n - value`` in closed form, a real one
    when there is one (the place of an orbit where the function found is
    real), or ``None`` when the roots have no closed form."""
    candidates: list[Expr]
    if isinstance(value, Rational):
        found = roots(pow(z, Integer(n)) - value, z)
        if not isinstance(found, dict) or sum(found.values()) != n:
            return None
        candidates = sorted((r for r in found if isinstance(r, Expr)), key=str)
    else:
        binomial = _binomial_roots(value, n, z) if n != 2 else None
        if binomial is None:
            return None
        candidates = sorted(binomial, key=str)
    for root in candidates:
        if root.is_real is True:
            return root
    return candidates[0]


def _primitive_root_of_unity(n: int, z: Symbol) -> Optional[Expr]:
    """A primitive ``n``-th root of unity in closed form, or ``None``."""
    found = roots(pow(z, Integer(n)) - 1, z)
    if not isinstance(found, dict) or sum(found.values()) != n:
        return None
    divisors = [d for d in range(1, n) if n % d == 0]
    for root in sorted((r for r in found if isinstance(r, Expr)), key=str):
        primitive = True
        for d in divisors:
            value = reliable_complex(expand(pow(root, Integer(d))) - 1, 30)
            if value is None or abs(value) <= 1e-10:
                primitive = False
                break
        if primitive:
            return root
    return None


def _bezout(k: int, n: int) -> tuple[int, int]:
    """``a, b`` with ``a*k + b*n == 1`` for coprime ``k``, ``n``."""
    r, r_ = k, n
    a, a_ = 1, 0
    b, b_ = 0, 1
    while r_:
        quotient = r // r_
        r, r_ = r_, r - quotient * r_
        a, a_ = a_, a - quotient * a_
        b, b_ = b_, b - quotient * b_
    return a, b


def _power_modulo(p: DensePoly[T], exponent: int, G: DensePoly[T]) -> DensePoly[T]:
    """``p**exponent`` modulo ``G``, the inverse of ``p`` modulo ``G``
    raised to ``-exponent`` when ``exponent`` is negative."""
    base = p.rem(G) if exponent >= 0 else _inverse(p.rem(G), G)
    result = p.new([p.domain.one])
    for _ in range(abs(exponent)):
        result = (result * base).rem(G)
    return result


def _series_multiply(a: list[T], b: list[T], order: int, K: Domain[T]) -> list[T]:
    """The product of two power series truncated after ``t**order``."""
    product = [K.zero] * (order + 1)
    for i, a_i in enumerate(a[:order + 1]):
        for j, b_j in enumerate(b[:order + 1 - i]):
            product[i + j] = K.add(product[i + j], K.mul(a_i, b_j))
    return product


def _series_inverse(p: list[T], order: int, K: Domain[T]) -> list[T]:
    """``1/p`` as a power series truncated after ``t**order`` (``p[0]``
    nonzero)."""
    inverse = [K.exquo(K.one, p[0])]
    for i in range(1, order + 1):
        total = K.zero
        for a in range(1, i + 1):
            if a < len(p):
                total = K.add(total, K.mul(p[a], inverse[i - a]))
        inverse.append(K.exquo(K.neg(total), p[0]))
    return inverse


def _series_power(p: list[T], exponent: Fraction, order: int, K: Domain[T]) -> list[T]:
    """``p**exponent`` as a power series truncated after ``t**order``, for
    ``p`` with constant term one: from ``exponent*p'*S == p*S'``,
    ``i*sigma_i = sum(sigma_a*p_(i - a)*(exponent*(i - a) - a))``."""
    sigma = [K.one]
    for i in range(1, order + 1):
        total = K.zero
        for a in range(i):
            if i - a < len(p):
                c = exponent * (i - a) - a
                total = K.add(total, K.mul(K.mul(sigma[a], p[i - a]),
                                           K.exquo(K.from_int(c.numerator), K.from_int(c.denominator))))
        sigma.append(K.exquo(total, K.from_int(i)))
    return sigma


def _series_of_the_root(P: DensePoly[T], exponent: Fraction, order: int) -> list[T]:
    """``(P(1/t)*t**deg P/lc(P))**exponent`` as a power series in ``t``
    truncated after ``t**order``."""
    return _series_power(list(P.monic().rep), exponent, order, P.domain)


def _residue_at_infinity(M: DensePoly[Fraction], D: DensePoly[Fraction], P: DensePoly[Fraction], n: int,
                         k: int) -> Fraction:
    """``kappa`` with the residue ``-s**k*kappa`` of ``M*y**k/(D*P) dx`` at
    the point at infinity where ``y ~ s*x**(deg P/n)`` (``n`` dividing
    ``deg P``): with ``x = 1/t`` the differential is ``-h(1/t)*s**k*
    t**(-k*deg P/n)*S(t)**k/t**2 dt``, ``S`` the series of
    ``(P(1/t)*t**deg P/lc(P))**(1/n)``.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.domains import QQ
    >>> from sympy_extras.polys.polyclasses import DensePoly
    >>> from sympy_extras.integrals.trager import _residue_at_infinity
    >>> x = symbols('x')
    >>> P, one = DensePoly.from_expr(x**3 + 1, x, QQ), DensePoly.from_expr(x**0, x, QQ)
    >>> _residue_at_infinity(one, one, P, 3, 2), _residue_at_infinity(one, DensePoly.from_expr(x, x, QQ), P, 3, 2)
    (Fraction(1, 1), Fraction(0, 1))
    """
    K = QQ
    d = P.degree
    order = k * d // n + 1
    shift = D.degree + d - M.degree
    if shift > order:
        return K.zero
    reversed_M = list(reversed(M.rep))
    reversed_DP = list(reversed((D * P).rep))
    h = _series_multiply(reversed_M, _series_inverse(reversed_DP, order, K), order, K)
    h = [K.zero] * shift + h[:order + 1 - shift]
    product = _series_multiply(h, _series_of_the_root(P, Fraction(k, n), order), order, K)
    return product[order]


def _rref(rows: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """The reduced row echelon form of a rational matrix and its pivot
    columns."""
    matrix = [list(row) for row in rows]
    pivots: list[int] = []
    r = 0
    columns = len(matrix[0]) if matrix else 0
    for c in range(columns):
        pivot = next((i for i in range(r, len(matrix)) if matrix[i][c] != 0), None)
        if pivot is None:
            continue
        matrix[r], matrix[pivot] = matrix[pivot], matrix[r]
        lead = matrix[r][c]
        matrix[r] = [v / lead for v in matrix[r]]
        for i in range(len(matrix)):
            if i != r and matrix[i][c] != 0:
                factor_ = matrix[i][c]
                matrix[i] = [a - factor_ * b for a, b in zip(matrix[i], matrix[r])]
        pivots.append(c)
        r += 1
        if r == len(matrix):
            break
    return matrix, pivots


def _rational_basis(values: list[Number], K: AlgebraicField) -> Optional[tuple[list[int], list[list[Fraction]]]]:
    """The indices of a basis over the rationals of the span of the
    algebraic numbers ``values`` (elements of the number field ``K``) and
    the coordinates of every value in that basis, or ``None`` when a
    value has none (it cannot happen for a basis of the span).

    >>> from fractions import Fraction
    >>> from sympy_extras.core import I, Rational, sqrt
    >>> from sympy_extras.integrals.trager import _NumberField, _rational_basis
    >>> N = _NumberField([Rational(-1, 2) + sqrt(3)*I/2])
    >>> K, zeta = N.field, N.field.alpha
    >>> _rational_basis([K.one, zeta, K.mul(zeta, zeta)], K) == ([0, 1], [[1, 0], [0, 1], [-1, -1]])
    True
    """
    degree = K.degree
    vectors: list[list[Fraction]] = []
    for value in values:
        coefficients = list(reversed(value))
        vectors.append(coefficients + [Fraction(0)] * (degree - len(coefficients)))
    chosen: list[int] = []
    rank = 0
    for index, vector in enumerate(vectors):
        _, pivots = _rref([vectors[i] for i in chosen] + [vector])
        if len(pivots) > rank:
            chosen.append(index)
            rank += 1
    # [B | V] reduced: the coordinates of every value in the basis B
    columns = [vectors[i] for i in chosen] + vectors
    reduced, pivots = _rref([[column[r] for column in columns] for r in range(degree)])
    if list(pivots[:len(chosen)]) != list(range(len(chosen))):
        return None
    coordinates: list[list[Fraction]] = []
    for index in range(len(values)):
        column = len(chosen) + index
        coordinates.append([reduced[r][column] for r in range(len(chosen))])
    return chosen, coordinates


def _function_with_divisor_n(finite: list[_Places], c_finite: list[Fraction], infinite: list[_PlaceAtInfinity],
                             c_infinite: list[Fraction], m: int, P: DensePoly[Number], n: int, x: Symbol,
                             extra_pole: int = 0) -> Optional[tuple[list[DensePoly[Number]], DensePoly[Number]]]:
    """``U, denominator`` with ``U = sum(U[i]*y**i)`` and ``U/denominator``
    of divisor ``m*sum(c_Q*Q)`` over the places ``Q`` of ``finite`` and
    ``infinite`` on the curve ``y**n == P``, or ``None`` when no such
    function exists: ``denominator`` is the product of the ``G**(m*p_G)``
    with ``p_G`` the largest pole order over the branches above ``G``,
    ``U`` must vanish to order ``m*(c_Q + p_G)`` on each branch (a linear
    system in the coefficients of the ``U[i]``, the branches lifted by
    Newton's iteration) and have poles at infinity at most as the
    denominator, less the orders prescribed there, which bounds the
    degrees of the ``U[i]`` (the orders of ``x**a*y**i`` at a point at
    infinity, ``-e*a - i*deg P/g`` with ``g = gcd(n, deg P)``, ``e = n/g``)
    and, at a point at infinity with a residue, is a condition on the
    expansion; the norm of ``U`` is checked to be the expected product.
    ``extra_pole`` allows ``U`` a further pole of that order at every
    point at infinity (the root of a function, :func:`_root_on_the_curve`)."""
    K = P.domain
    d = P.degree
    g = math.gcd(n, d)
    e = n // g
    pole_orders: dict[int, int] = {}
    for place, c in zip(finite, c_finite):
        pole_orders[place.group] = max(pole_orders.get(place.group, 0), int(-c * m) if c < 0 else 0)
    denominator = P.new([K.one])
    groups: dict[int, DensePoly[Number]] = {place.group: place.G for place in finite}
    B = 0
    for group, order_ in pole_orders.items():
        if order_:
            denominator = denominator * groups[group]**order_
            B += e * order_ * groups[group].degree
    required_infinity = [-B - extra_pole + int(c * m) for c in c_infinite] if infinite else [-B - extra_pole]
    pole = -min(required_infinity)
    bounds = [(pole - i * (d // g)) // e for i in range(n)]
    offsets: list[int] = []
    unknowns = 0
    for i in range(n):
        offsets.append(unknowns)
        unknowns += max(bounds[i] + 1, 0)
    if unknowns == 0:
        return None
    rows: list[list[Number]] = []
    for place, c in zip(finite, c_finite):
        N = int(c * m) + pole_orders[place.group]
        if N <= 0:
            continue
        modulus = place.G**N
        size = modulus.degree
        W_N = place.branch(N)
        powers = [P.new([K.one])]
        for i in range(1, n):
            powers.append((powers[-1] * W_N).rem(modulus))
        columns: list[list[Number]] = []
        for i in range(n):
            for a in range(bounds[i] + 1):
                columns.append(_coefficients((_monomial(K.one, a, K, x) * powers[i]).rem(modulus), size))
        for r in range(size):
            rows.append([column[r] for column in columns])
    if infinite and max(required_infinity) > -pole:
        order = pole + max(required_infinity)
        series = [_series_of_the_root(P, Fraction(i, n), order) for i in range(n)]
        for point, required in zip(infinite, required_infinity):
            for r in range(required + pole):
                row: list[Number] = []
                for i in range(n):
                    for a in range(bounds[i] + 1):
                        index = -pole + r + a + i * d // n
                        value = series[i][index] if 0 <= index <= order else K.zero
                        row.append(K.mul(K.pow(point.s, i), value))
                rows.append(row)
    if not rows:
        return None
    space = K.nullspace(rows, unknowns)
    if not space:
        return None
    vector = space[0]
    # the space has dimension one; the vector with its first coefficient
    # one has its coefficients in the field of the divisor (real when the
    # places are)
    lead = next((entry for entry in vector if not K.is_zero(entry)), None)
    if lead is None:
        return None
    vector = [K.exquo(entry, lead) for entry in vector]
    U = [DensePoly(list(reversed(vector[offsets[i]:offsets[i] + bounds[i] + 1])), K, x)
         if bounds[i] >= 0 else P.new([]) for i in range(n)]
    # the norm must be a constant times prod(G**(m*(sum of the c on the
    # branches above G) + n*p_G))
    norm = _norm(U, P, n, x)
    expected = P.new([K.one])
    exponents: dict[int, Fraction] = {}
    for place, c in zip(finite, c_finite):
        exponents[place.group] = exponents.get(place.group, Fraction(0)) + c
    for group, G in groups.items():
        exponent = int(exponents[group] * m) + n * pole_orders.get(group, 0)
        expected = expected * G**exponent
    if norm.is_zero or norm.degree != expected.degree or not (norm - expected.mul_ground(norm.LC)).is_zero:
        return None
    return U, denominator


def _logarithmic_derivative_matches(logs: list[_LogTermN], M: DensePoly[Number], D: DensePoly[Number],
                                    P: DensePoly[Number], n: int, k: int, x: Symbol) -> bool:
    """Whether ``sum((beta/m)*dlog(U/denominator))`` is ``M*y**k/(D*P)``
    exactly on the curve: the logarithmic derivative of ``U =
    sum(U_i*y**i)`` is ``sum((U_i' + (i/n)*U_i*P'/P)*y**i)`` times the
    inverse of ``U`` modulo ``y**n - P`` over the rational functions."""
    K = P.domain
    Kx: FractionField[Number] = FractionField(PolynomialRing(K, x.name))
    y = _fresh('y')

    def fraction_of(p: DensePoly[Number]) -> FracElement[Number]:
        return Kx.from_ring(p.rep)

    def in_y(coefficients: dict[int, FracElement[Number]]) -> DensePoly[FracElement[Number]]:
        degree = max(coefficients) if coefficients else 0
        return DensePoly([coefficients.get(i, Kx.zero) for i in range(degree, -1, -1)], Kx, y)

    P_ = fraction_of(P)
    modulus = in_y({n: Kx.one, 0: Kx.neg(P_)})
    total = in_y({})
    for term in logs:
        U = in_y({i: fraction_of(U_i) for i, U_i in enumerate(term.U) if not U_i.is_zero})
        dU = in_y({i: Kx.exquo(fraction_of(U_i.diff() * P + (U_i * P.diff()).mul_ground(K.exquo(K.from_int(i), K.from_int(n)))),
                               P_)
                   for i, U_i in enumerate(term.U) if not U_i.is_zero})
        dlog = (dU * _inverse(U, modulus)).rem(modulus).mul_ground(Kx.from_int(term.power))
        if term.denominator.degree >= 1:
            dlog = dlog - in_y({0: Kx.exquo(fraction_of(term.denominator.diff()), fraction_of(term.denominator))})
        total = total + dlog.mul_ground(Kx.exquo(Kx.from_ring([term.beta] if not K.is_zero(term.beta) else []),
                                                 Kx.from_int(term.m)))
    expected = in_y({k: Kx.exquo(fraction_of(M), Kx.mul(fraction_of(D), P_))})
    return (total - expected).is_zero


def _norm(U: list[DensePoly[Number]], P: DensePoly[Number], n: int, x: Symbol) -> DensePoly[Number]:
    """The norm ``Res_y(y**n - P, sum(U[i]*y**i))`` of a function of the
    curve, a polynomial in ``x``."""
    K = P.domain
    Kx: PolynomialRing[Number] = PolynomialRing(K, x.name)
    curve: list[list[Number]] = [[K.one]] + [[] for _ in range(n - 1)] + [(-P).rep]
    function: list[list[Number]] = densebasic.dup_strip([U_i.rep for U_i in reversed(U)], Kx)
    resultant: list[Number] = dup_resultant(curve, function, Kx)
    return DensePoly(resultant, K, x)


def _real_logarithms(logs: list[_LogTermN], L: _Field[Number], P: DensePoly[Fraction], n: int, x: Symbol) -> Expr:
    """``sum((beta/m)*(power*log(U) - log(denominator)))`` in a real form
    where the integrand is real: with ``beta = b_r + I*b_i`` and ``U =
    u_r + I*u_i`` in real ``x`` and ``y = P**(1/n)``, the real part
    ``b_r*log|U| - b_i*arg(U)`` of each term, ``arg U = atan(u_i/u_r)``
    up to a piecewise constant, the moduli written through the factors
    of ``u_r**2 + u_i**2``; the complex form itself when a real and an
    imaginary part are not separated."""
    X = _fresh('x', real=True)
    Y = _fresh('y', real=True)
    P_expr = P.to_expr()
    y = pow(P_expr, Rational(1, n))
    total: Expr = _ZERO
    complex_form: Expr = _ZERO
    real = True
    P_X = _replaced(P_expr, {x: X})
    back: dict[Basic, Basic] = {X: x, Y: y}
    for term in logs:
        beta = L.expr(term.beta) / term.m
        U = _replaced(add(*[L.poly_expr(U_i) * pow(Y, Integer(i)) for i, U_i in enumerate(term.U)]), {x: X})
        denominator = _replaced(L.poly_expr(term.denominator), {x: X})
        complex_form = complex_form + _replaced(beta * (term.power * log(U) - log(denominator)), back)
        b_r, b_i = beta.as_real_imag()
        u_r, u_i = (expand(part) for part in expand(U).as_real_imag())
        d_r, d_i = (expand(part) for part in expand(denominator).as_real_imag())
        if any(part.has(Re, Im, I) for part in (u_r, u_i, d_r, d_i, b_r, b_i)):
            real = False
            continue
        contribution: Expr = _ZERO
        if b_r != 0:
            contribution = contribution + b_r * (term.power * _log_of_factors(u_r**2 + u_i**2, X, Y, n, P_X)
                                                 - _log_of_factors(d_r**2 + d_i**2, X, Y, n, P_X)) / 2
        if b_i != 0:
            for numerator, denominator_, sign_ in ((u_i, u_r, -term.power), (d_i, d_r, 1)):
                if numerator != 0 and denominator_ != 0:
                    contribution = contribution + sign_ * b_i * atan(factor(numerator) / factor(denominator_))
        # the coefficient distributed over the logarithms, so that the
        # logarithms of the denominators of conjugate terms cancel
        total = total + add(*[expand_mul(part, deep=False) for part in Add.make_args(contribution)])
    if not real:
        return complex_form
    return _replaced(total, back)


def _log_of_factors(p: Expr, X: Symbol, Y: Symbol, n: int, P_X: Expr) -> Expr:
    """``log(p)`` as the sum of the logarithms of the irreducible factors
    of the polynomial ``p`` in ``X``, ``Y`` reduced modulo the curve
    ``Y**n == P_X`` (the constant factor, a constant of integration,
    dropped)."""
    try:
        coefficients = _powers_of(p, Y)
        for i in sorted(coefficients, reverse=True):
            if i >= n:
                c = coefficients.pop(i)
                coefficients[i - n] = coefficients.get(i - n, _ZERO) + c * P_X
        reduced = expand(add(*[c * pow(Y, Integer(i)) for i, c in coefficients.items()]))
        coefficient, factors = factor_list(reduced, X, Y)
    except _FAILURES:
        return log(p)
    total = add(*[multiplicity * log(factor_) for factor_, multiplicity in factors])
    if coefficient.has(X, Y):
        # a polynomial with algebraic coefficients is not factored
        total = total + log(coefficient)
    return total


def _root_components(f: Expr, x: Symbol) -> Optional[tuple[Expr, int, list[Expr]]]:
    """``P, n, [A_0, ..., A_(n-1)]`` with ``f == sum(A_k*P**(k/n))``, for
    ``f`` rational in ``x`` and in the powers of one squarefree polynomial
    ``P`` with denominators of index ``n >= 3``; ``None`` otherwise.

    >>> from sympy_extras.core import Rational, symbols
    >>> from sympy_extras.polys.polytools import cancel
    >>> from sympy_extras.integrals.trager import _root_components
    >>> x = symbols('x')
    >>> P, n, components = _root_components(1/(1 + (x**3 + 1)**Rational(1, 3)), x)
    >>> P, n, [cancel(c*(x**3 + 2)) for c in components]
    (x**3 + 1, 3, [1, -1, 1])
    """
    bases: dict[Expr, int] = {}
    for node in _pows(f):
        exponent = node.exp
        if isinstance(exponent, Rational) and exponent.q > 1 and node.has(x):
            base = node.base
            if not base.is_polynomial(x):
                return None
            bases[base] = math.lcm(bases.get(base, 1), exponent.q)
    if len(bases) != 1:
        return None
    [(base, n)] = bases.items()
    if n < 3:
        return None
    parameters = [s for s in _symbols(f) if s != x]
    try:
        degree, squarefree = (_squarefree_degree(_parametric(parameters), base, x) if parameters
                              else _squarefree_degree(_rationals(), base, x))
    except _FAILURES:
        return None
    if degree < 1 or not squarefree:
        return None
    y = _fresh('y')

    def to_y(node: Basic) -> Basic:
        if isinstance(node, Pow) and node.base == base and isinstance(node.exp, Rational):
            exponent = node.exp
            return pow(y, Integer(exponent.p * n // exponent.q))
        return node

    g = as_expr(f.replace(lambda node: isinstance(node, Pow), to_y))
    if not g.is_rational_function(x, y):
        return None
    numerator, denominator = fraction(cancel(g))
    K = _parametric([x] + parameters)
    try:
        modulus = K.from_expr(pow(y, Integer(n)) - base, y)
        inverse = _inverse(K.from_expr(denominator, y), modulus)
        reduced = (K.from_expr(numerator, y) * inverse).rem(modulus)
    except _FAILURES:
        return None
    return base, n, [cancel(K.expr(reduced.coeff(k))) for k in range(n)]


def _squarefree_degree(F: _Field[T], base: Expr, x: Symbol) -> tuple[int, bool]:
    """The degree of the polynomial ``base`` in ``x`` and whether it is
    squarefree over ``F``."""
    P = F.from_expr(base, x)
    return P.degree, P.is_squarefree


def _polynomial_radicands(f: Expr, x: Symbol) -> tuple[Expr, list[tuple[Expr, Expr]]]:
    """``f`` with each square root of a rational function ``P/Q`` written
    ``sqrt(P*Q)/Q``, and the pairs ``(P*Q, Q*sqrt(P/Q))`` for
    :func:`_restored`: ``y = Q*sqrt(P/Q)`` satisfies ``y**2 = P*Q`` and
    ``y' = (P*Q)'/(2*y)`` for every ``x``, so the algebra in ``sqrt(P*Q)``
    holds for it, whereas ``sqrt(P*Q)`` itself is ``-y`` where ``Q < 0``."""
    replacements: dict[Basic, Basic] = {}
    pairs: list[tuple[Expr, Expr]] = []
    for node in _pows(f):
        if not (_half_integer(node) and node.has(x)):
            continue
        base = node.base
        if base.is_polynomial(x) or not base.is_rational_function(x):
            continue
        numerator, denominator = fraction(cancel(base))
        if not (numerator.is_polynomial(x) and denominator.is_polynomial(x) and denominator.has(x)):
            continue
        k = _exponent(node).p
        product = expand(numerator * denominator)
        replacements[node] = pow(product, node.exp) / pow(denominator, Integer(k))
        pairs.append((product, denominator * sqrt(base)))
    return _replaced(f, replacements), pairs


def _restored(F: Expr, pairs: list[tuple[Expr, Expr]]) -> Expr:
    """``F`` with the powers of ``sqrt(product)`` written as powers of
    ``y`` for each pair ``(product, y)``."""
    if not pairs:
        return F

    def restore(node: Basic) -> Basic:
        if isinstance(node, Pow) and _half_integer(node):
            base = expand(node.base)
            for product, y in pairs:
                if base == product:
                    k = _exponent(node).p
                    return mul(pow(node.base, Integer((k - k % 2) // 2)), pow(y, Integer(k % 2)))
        return node

    return as_expr(F.replace(lambda node: isinstance(node, Pow), restore))


def _rational_parameters(values: list[Expr], parameters: list[Symbol]
                         ) -> Optional[tuple[dict[Basic, Basic], dict[Basic, Basic]]]:
    """A reparametrization making the square roots of ``values`` rational
    functions of the parameters: for the innermost ``sqrt(E)``, ``E``
    written ``s**2*E0`` with ``E0`` squarefree and linear in some
    parameter ``p``, ``p`` is replaced by the solution of ``delta**2 =
    E0`` (``a = delta**2/c`` for ``sqrt(a*c)``), and so on for the roots
    left; the substitution of the parameters and the map back from the
    new symbols. ``None`` when a root is not linear in any parameter.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.trager import _rational_parameters
    >>> a, b, c = symbols('a b c')
    >>> found = _rational_parameters([sqrt(b - 2*sqrt(a*c))/(4*sqrt(a*c))], [a, b, c])
    >>> found is not None and sorted(found[0], key=str) == [a, b] and sorted(found[1].values(), key=str)
    [sqrt(-b + 2*sqrt(a*c)), sqrt(a*c)]
    """
    substitution: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    current = list(values)
    for _ in range(8):
        nodes = [node for value in current for node in _pows(value) if _half_integer(node) and node.free_symbols]
        inner = [node for node in nodes
                 if not any(_half_integer(other) and other.free_symbols for other in _pows(node.base))]
        if not inner:
            break
        node = sorted(inner, key=str)[0]
        numerator, denominator = fraction(cancel(node.base))
        symbols = _symbols(numerator * denominator)
        try:
            content, factors = factor_list(numerator * denominator, *symbols)
        except _FAILURES:
            return None
        E0: Expr = _ONE
        square: Expr = _ONE
        for factor_, multiplicity in factors:
            if not isinstance(multiplicity, Integer):
                return None
            E0 = E0 * pow(factor_, Integer(multiplicity.p % 2))
            square = square * pow(factor_, Integer(multiplicity.p // 2))
        chosen: Optional[Symbol] = None
        # an original parameter first, a symbol of an earlier round else
        candidates = sorted(_symbols(E0), key=lambda p: (p not in parameters, str(p)))
        linear: dict[int, Expr] = {}
        for p in candidates:
            try:
                linear = _powers_of(E0, p)
            except ValueError:
                continue
            if max(linear) == 1:
                chosen = p
                break
        if chosen is None:
            return None
        slope, offset = linear.get(1, _ZERO), linear.get(0, _ZERO)
        delta = _fresh('delta', positive=True)
        value = (delta**2 - offset) / slope
        back[delta] = _replaced(sqrt(E0), back)
        substitution = {q: _replaced(as_expr(e), {chosen: value}) for q, e in substitution.items()}
        if chosen in parameters:
            substitution[chosen] = value
        root = sqrt(content) * square * delta / denominator
        replacement: dict[Basic, Basic] = {other: pow(root, Integer(_exponent(other).p)) for other in nodes
                                           if other.base == node.base}
        current = [cancel(_replaced(_replaced(c, replacement), {chosen: value})) for c in current]
    if not substitution:
        return None
    return substitution, back


def _original_roots(F: Expr, f: Expr, x: Symbol, P: Expr) -> Expr:
    """``F`` with the root of the product of the radicands of ``f`` written
    as the product of its roots when ``f`` has several: ``y = sqrt(x + 1) *
    sqrt(x + 2)`` satisfies ``y**2 = P`` and ``y' = P'/(2*y)`` for every
    ``x``, so the algebra in ``sqrt(P)`` holds for it, whereas ``sqrt(P)``
    itself is ``-y`` below ``-2``."""
    bases = sorted({node.base for node in _pows(f) if node.has(x) and _half_integer(node)
                    and node.base.is_polynomial(x)}, key=str)
    if len(bases) < 2:
        return F
    y = mul(*[sqrt(base) for base in bases])
    P_ = expand(P)

    def restore(node: Basic) -> Basic:
        if isinstance(node, Pow) and _half_integer(node) and expand(node.base) == P_:
            k = _exponent(node).p
            return mul(pow(node.base, Integer((k - k % 2) // 2)), pow(y, Integer(k % 2)))
        return node

    return as_expr(F.replace(lambda node: isinstance(node, Pow), restore))


def trager_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the algebraic function ``f`` (rational in
    ``x`` and in radicals of rational functions of ``x``: one square
    root, one root of any index, or several and nested radicals through
    the general curve of :mod:`sympy_extras.integrals.curves`) by
    Trager's algorithm, or ``None`` when the integral is not elementary
    or its logarithmic part is not found; see :func:`trager_reduce` for
    the elementary part alone.

    >>> from sympy_extras.core import symbols, Rational
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.trager import trager_antiderivative
    >>> x = symbols('x')
    >>> trager_antiderivative(x/sqrt(x**4 + 1), x)
    log(x**2 + sqrt(x**4 + 1))/2
    >>> trager_antiderivative(1/(x*sqrt(x**2 + 1)), x)
    -log((sqrt(x**2 + 1) + 1)/x)
    >>> trager_antiderivative(1/(x*sqrt(x**3 + 1)), x)
    -log((x**3/2 + sqrt(x**3 + 1) + 1)/x**3)/3
    >>> trager_antiderivative(1/sqrt(x**3 + 1), x) is None
    True
    >>> trager_antiderivative(1/(x*(x**4 + 1)**Rational(1, 4)), x)
    log((x**4 + 1)**(1/4) - 1)/4 - log((x**4 + 1)**(1/4) + 1)/4 + atan((x**4 + 1)**(1/4))/2
    """
    return _trager_antiderivative(to_expr(f), x, _curve_reduce)


def _trager_antiderivative(f: Expr, x: Symbol, curve: CurveReducer) -> Optional[Expr]:
    found = _trager_reduce(f, x, curve)
    if found is None:
        return None
    F, h = found
    if h != 0:
        return None
    return F


def hermite_reduce_algebraic(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(g, h)`` with ``f == g' + h`` and ``h`` with simple poles only at
    the finite points of the curve: Trager's Hermite reduction of the
    algebraic part (the multiple and ramified poles removed, the
    polynomial part reduced at infinity) with the Horowitz–Ostrogradsky
    reduction of the rational part; ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.trager import hermite_reduce_algebraic
    >>> x = symbols('x')
    >>> hermite_reduce_algebraic(x**2/sqrt(x**2 + 1), x)
    (x*sqrt(x**2 + 1)/2, -1/(2*sqrt(x**2 + 1)))
    >>> hermite_reduce_algebraic(1/(x**2*sqrt(x**3 + 1)), x)
    (-sqrt(x**3 + 1)/x, x/(2*sqrt(x**3 + 1)))
    """
    f_ = to_expr(f)
    result = attempt(lambda: _integrate(f_, x, False), settings.timeout, native=True)
    if result is None:
        return None
    g = result.exact_part()
    h = result.rational_logs + result.remainder_expr()
    if not _verified(g, f_ - h, x, result.P):
        return None
    return g, h


def is_nonelementary_algebraic(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the integral of the algebraic function ``f`` is not
    elementary: ``True`` when the reduced integrand keeps a pole of order
    two or more at infinity (which no function of the field removes) or,
    on a curve of genus one over the rationals with rational residues,
    when the residue divisor is not torsion (Mazur's bound); ``False``
    when an antiderivative is found; ``None`` when the question is not
    settled (residues linearly dependent over the rationals, higher
    genus, parameters) or ``f`` is not such a function.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import sqrt
    >>> from sympy_extras.integrals.trager import is_nonelementary_algebraic
    >>> x = symbols('x')
    >>> is_nonelementary_algebraic(1/sqrt(x**3 + 1), x)
    True
    >>> is_nonelementary_algebraic(x/sqrt(x**4 + 1), x)
    False
    >>> is_nonelementary_algebraic(x/sqrt(x**3 + 1), x)
    True
    >>> is_nonelementary_algebraic((2*x**2 + 1)/((x**2 + 1)*sqrt(x**4 + x**2 + 1)), x)
    True

    A binomial differential ``x**m*(a + b*x**n)**p`` with rational
    exponents, nonzero numbers ``a``, ``b`` and no parameter (up to a
    locally constant factor, :func:`.algebraic.binomial_exponents`) is
    settled both ways by Chebyshev's criterion
    (:func:`.algebraic.chebyshev_elementary`), of any genus and any root:

    >>> is_nonelementary_algebraic(sqrt(x**7/(1 - 5*x**2)), x)
    True
    >>> from sympy_extras.functions import cbrt
    >>> is_nonelementary_algebraic(sqrt(x)/(1 + cbrt(x)), x)
    False
    """
    return _is_nonelementary_algebraic(to_expr(f), x, _curve_reduce)


def _is_nonelementary_algebraic(f_: Expr, x: Symbol, curve: CurveReducer) -> Optional[bool]:
    from .algebraic import binomial_exponents, chebyshev_elementary
    from .curves import is_nonelementary_on_curve
    binomial = binomial_exponents(f_, x)
    if binomial is not None and not [s for s in _symbols(f_) if s != x]:
        # (a parameter in a constant factor could vanish)
        return not chebyshev_elementary(binomial)
    result = attempt(lambda: _integrate(f_, x, True), settings.timeout, native=True)
    if result is None:
        verdict = attempt(lambda: is_nonelementary_on_curve(f_, x), settings.timeout, native=True)
        return None if verdict is None else verdict
    if result.nonelementary is False:
        # the antiderivative is kept only when it checks
        return False if _trager_antiderivative(f_, x, curve) is not None else None
    return result.nonelementary
