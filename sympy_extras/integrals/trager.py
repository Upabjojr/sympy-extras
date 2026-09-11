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

import math
import random
from typing import Optional

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import QQ
from sympy.polys.domains.domain import Domain
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import (CoercionFailed, DomainError, GeneratorsError, NotInvertible, PolynomialError,
                                    UnificationFailed)
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, resultant
from sympy.simplify.radsimp import fraction

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.settings import settings
from .risch.rationaltools import ratint, ratint_ratpart

__all__ = ['trager_antiderivative', 'trager_reduce', 'hermite_reduce_algebraic', 'is_nonelementary_algebraic']

#: the multiples of the denominator of the residue divisor tried for the
#: torsion order: complete on a curve of genus one over the rationals
#: (Mazur: the torsion of an elliptic curve over Q has order at most 12)
_TORSION_BOUND = 12

_FAILURES = (CoercionFailed, DomainError, GeneratorsError, NotInvertible, PolynomialError, UnificationFailed,
             NotImplementedError, ValueError, TypeError, ZeroDivisionError)


class _Radical:
    """The integrand ``A + B*y`` on the curve ``y**2 == P``, with ``y = d*w``
    and ``w**2 == Q`` squarefree."""

    def __init__(self, x: Symbol, P: Poly, A: Expr, B: Expr) -> None:
        self.x = x
        self.P = P
        self.A = A
        self.B = B
        self.d, self.Q = _split_square(P)

    @property
    def w(self) -> Expr:
        """``w`` as an expression in ``x``: ``sqrt(P)/d``."""
        return as_expr(sqrt(self.P.as_expr()) / self.d.as_expr())

    def algebraic_numerator(self) -> tuple[Poly, Poly]:
        """``M, D`` with ``B*y == M/(D*w)``: ``B*y = B*d*w = B*d*Q/w``."""
        e = cancel(self.B * self.d.as_expr() * self.Q.as_expr())
        numerator, denominator = fraction(e)
        K = self.Q.domain
        M = Poly(numerator, self.x, domain=K)
        D = Poly(denominator, self.x, domain=K)
        return M, D


def _split_square(P: Poly) -> tuple[Poly, Poly]:
    """``d, Q`` with ``P == d**2 * Q`` and ``Q`` squarefree."""
    content, factors = P.sqf_list()
    d = Poly(1, P.gen, domain=P.domain)
    Q = Poly(content, P.gen, domain=P.domain)
    for factor, multiplicity in factors:
        d = d * factor**(multiplicity // 2)
        if multiplicity % 2:
            Q = Q * factor
    return d, Q


def _parse(f: Expr, x: Symbol) -> Optional[_Radical]:
    """The integrand as ``A + B*sqrt(P)`` with ``A``, ``B`` rational in
    ``x`` and ``P`` a polynomial, or ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial."""
    bases: set[Expr] = set()
    for node in f.atoms(Pow):
        exponent = as_expr(node.exp)
        if isinstance(exponent, Rational) and exponent.q == 2:
            base = as_expr(node.base)
            if not base.has(x):
                continue
            if not base.is_polynomial(x):
                return None
            bases.add(base)
    if len(bases) != 1:
        return None
    [base] = bases
    y = Dummy('y')

    def to_y(node: Expr) -> Expr:
        if isinstance(node, Pow) and node.base == base:
            exponent = as_expr(node.exp)
            if isinstance(exponent, Rational) and exponent.q == 2:
                return as_expr(y**int(exponent.p))
        return node

    g = as_expr(f.replace(lambda node: isinstance(node, Pow), to_y))
    if not g.is_rational_function(x, y):
        return None
    try:
        P = Poly(base, x)
    except _FAILURES:
        return None
    if P.degree() < 1:
        return None
    numerator, denominator = fraction(cancel(g))
    try:
        n0, n1 = _reduce_modulo_curve(Poly(numerator, y), P, x)
        d0, d1 = _reduce_modulo_curve(Poly(denominator, y), P, x)
    except _FAILURES:
        return None
    norm = as_expr(d0**2 - d1**2 * P.as_expr())
    if norm == 0:
        return None
    A = as_expr(cancel((n0 * d0 - n1 * d1 * P.as_expr()) / norm))
    B = as_expr(cancel((n1 * d0 - n0 * d1) / norm))
    domain = P.domain
    for e in (A, B):
        for part in fraction(e):
            domain = domain.unify(Poly(part, x).domain)
    field = domain.get_field()
    return _Radical(x, P.set_domain(field), A, B)


def _reduce_modulo_curve(N: Poly, P: Poly, x: Symbol) -> tuple[Expr, Expr]:
    """``n0, n1`` with ``N(y) == n0 + n1*y`` modulo ``y**2 - P``."""
    n0: Expr = S.Zero
    n1: Expr = S.Zero
    P_ = P.as_expr()
    for (k,), coefficient in N.terms():
        if k % 2 == 0:
            n0 = n0 + as_expr(coefficient) * P_**(k // 2)
        else:
            n1 = n1 + as_expr(coefficient) * P_**((k - 1) // 2)
    return as_expr(n0.expand()), as_expr(n1.expand())


class _Reduction:
    """The Hermite reduction of ``M/(D*w)`` on ``w**2 == Q``: ``terms`` are
    the pairs ``(S, T)`` of the exact part ``sum(S*w/T)``, the remainder
    ``r/(D*w)`` has ``D`` squarefree and coprime to ``Q``, and
    ``polynomial`` is the part of ``r`` of degree below ``deg Q - 1``
    left by the reduction at infinity."""

    def __init__(self, terms: list[tuple[Poly, Poly]], r: Poly, D: Poly, polynomial: Poly) -> None:
        self.terms = terms
        self.r = r
        self.D = D
        self.polynomial = polynomial


def _hermite(M: Poly, D: Poly, Q: Poly, x: Symbol) -> _Reduction:
    """Trager's Hermite reduction of ``M/(D*w)`` on the curve ``w**2 == Q``
    (``Q`` squarefree), followed by the reduction of the polynomial part
    at infinity."""
    terms: list[tuple[Poly, Poly]] = []
    one = Poly(1, x, domain=Q.domain)
    while True:
        common = M.gcd(D)
        M, D = M.exquo(common), D.exquo(common)
        lead = D.LC()
        M, D = M.quo_ground(lead), D.monic()
        step = _reduction_step(M, D, Q, x)
        if step is None:
            break
        M, D, S, T = step
        terms.append((S, T))
    # the polynomial part: p/w with p reduced below degree deg Q - 1
    p, r1 = M.div(D)
    S, r2 = _reduce_at_infinity(p, Q, x)
    if not S.is_zero:
        terms.append((S, one))
    r = r2 * D + r1
    common = r.gcd(D)
    r, D = r.exquo(common), D.exquo(common)
    return _Reduction(terms, r, D, r2)


def _reduction_step(M: Poly, D: Poly, Q: Poly, x: Symbol) -> Optional[tuple[Poly, Poly, Poly, Poly]]:
    """One step of the Hermite reduction: the factor of ``D`` of highest
    multiplicity which is ramified or multiple removed once, returning
    the new ``M``, ``D`` and the term ``S*w/T`` subtracted; ``None`` when
    ``D`` is squarefree and coprime to ``Q``."""
    _, factors = D.sqf_list()
    for V, k in sorted(factors, key=lambda pair: -pair[1]):
        ramified = V.gcd(Q)
        if ramified.degree() > 0:
            # the pole has even order 2k at the ramified points: S w/V^k
            V = ramified
            U = D.exquo(V**k)
            Q_ = Q.exquo(V)
            inverse = (U * V.diff(x) * Q_).mul_ground(Q.domain.convert(1 - 2 * k)).invert(V)
            S = (M * inverse).mul_ground(Q.domain.convert(2)).rem(V)
            subtracted = (S.diff(x) * Q_ * V).mul_ground(Q.domain.convert(2)) + S * Q.diff(x) \
                - (S * V.diff(x) * Q_).mul_ground(Q.domain.convert(2 * k))
            M_ = (M - (U * subtracted).quo_ground(Q.domain.convert(2))).exquo(V)
            return M_, U * V**(k - 1), S, V**k
        if k >= 2:
            U = D.exquo(V**k)
            inverse = (U * V.diff(x) * Q).mul_ground(Q.domain.convert(k - 1)).invert(V)
            S = (M * inverse).rem(V).neg()
            subtracted = (S.diff(x) * Q * V).mul_ground(Q.domain.convert(2)) + S * Q.diff(x) * V \
                - (S * V.diff(x) * Q).mul_ground(Q.domain.convert(2 * (k - 1)))
            M_ = (M - (U * subtracted).quo_ground(Q.domain.convert(2))).exquo(V)
            return M_, U * V**(k - 1), S, V**(k - 1)
    return None


def _reduce_at_infinity(p: Poly, Q: Poly, x: Symbol) -> tuple[Poly, Poly]:
    """``S, r`` with ``p/w == (S*w)' + r/w`` and ``deg r < deg Q - 1``: the
    derivative of ``c*x**j*w`` has the numerator ``c*(2*j + deg Q)*lc(Q)
    x**(j + deg Q - 1) + ...``, so the degrees are peeled one by one."""
    K = Q.domain
    S = Poly(0, x, domain=K)
    n = Q.degree()
    two = K.convert(2)
    while p.degree() >= n - 1 and not p.is_zero:
        j = p.degree() - n + 1
        c = K.from_sympy(as_expr(2 * p.LC() / (Q.LC() * (2 * j + n))))
        monomial = Poly(x**j, x, domain=Q.domain).mul_ground(c)
        S = S + monomial
        p = p - ((monomial.diff(x) * Q).mul_ground(two) + monomial * Q.diff(x)).quo_ground(two)
    return S, p


# ---------------------------------------------------------------------------
# The residues and the logarithmic part

class _ResidueClass:
    """The points whose residues are rational multiples ``q*rho`` of one
    value: for each ``q > 0`` the polynomial ``G`` whose roots are the
    abscissae, the branch ``W`` of ``w`` modulo ``G`` on which the residue
    is ``+q*rho``, and the multiple ``q_infinity`` of ``rho`` at the point
    at infinity where ``w ~ +sqrt(lc Q) x**m`` (its conjugate carries the
    opposite residue)."""

    def __init__(self, rho: Expr, finite: list[tuple[Rational, Poly, Poly]], q_infinity: Rational) -> None:
        self.rho = rho
        self.finite = finite
        self.q_infinity = q_infinity

    @property
    def denominator(self) -> int:
        return math.lcm(*[int(q.q) for q, _, _ in self.finite], int(self.q_infinity.q))


def _residue_polynomial(r: Poly, D: Poly, Q: Poly, x: Symbol, z: Symbol) -> Poly:
    """The Rothstein–Trager resultant whose roots are the residues at the
    finite points."""
    F = D.as_expr()
    G = as_expr(r.as_expr()**2 - z**2 * D.diff(x).as_expr()**2 * Q.as_expr())
    return Poly(resultant(F, G, x), z)


def _residue_values(R: Poly) -> Optional[list[Expr]]:
    """The nonzero roots of the residue polynomial in closed form, or
    ``None`` when some root has none."""
    values: list[Expr] = []
    _, factors = R.factor_list()
    for factor, _ in factors:
        if factor.degree() == 0:
            continue
        if factor.degree() == 1 and factor.eval(0) == 0:
            continue
        found = roots(factor)
        if sum(found.values()) != factor.degree():
            return None
        values.extend(as_expr(root) for root in found)
    return [value for value in values if value != 0]


def _field(values: list[Expr]) -> Optional[Domain]:
    """The number field generated by the algebraic numbers ``values``."""
    generators = [value for value in values if not isinstance(value, Rational)]
    if not generators:
        return QQ
    try:
        return QQ.algebraic_field(*generators)
    except _FAILURES:
        return None


def _classes(values: list[Expr], infinity: Optional[Expr], K: Domain, r: Poly, D: Poly, Q: Poly,
             x: Symbol) -> Optional[list[_ResidueClass]]:
    """The residues (``values`` at the finite points, ``infinity`` at the
    point at infinity where ``w ~ +sqrt(lc Q) x**m``) partitioned into
    classes of rational multiples, with the data of each class."""
    elements = [K.from_sympy(value) for value in values]
    representatives: list[object] = []
    members: list[list[object]] = []
    all_values = list(elements)
    if infinity is not None:
        all_values.append(K.from_sympy(infinity))
    for value in all_values:
        for k, representative in enumerate(representatives):
            ratio = K.to_sympy(value / representative)
            if isinstance(ratio, Rational):
                members[k].append(value)
                break
        else:
            representatives.append(value)
            members.append([value])
    classes: list[_ResidueClass] = []
    D_ = D.set_domain(K)
    r_ = r.set_domain(K)
    Q_ = Q.set_domain(K)
    for representative, group in zip(representatives, members):
        rho = K.to_sympy(representative)
        finite: list[tuple[Rational, Poly, Poly]] = []
        seen: set[Rational] = set()
        for value in elements:
            ratio = K.to_sympy(value / representative)
            if not isinstance(ratio, Rational):
                continue
            q = as_expr(abs(ratio))
            if not isinstance(q, Rational) or q in seen:
                continue
            seen.add(q)
            residue = value if ratio > 0 else -value
            G = D_.gcd(r_**2 - (D_.diff(x)**2 * Q_).mul_ground(residue**2))
            if G.degree() == 0:
                return None
            W = (r_ * (D_.diff(x).mul_ground(residue)).invert(G)).rem(G)
            finite.append((q, G, W))
        q_infinity: Rational = S.Zero
        if infinity is not None:
            ratio = K.to_sympy(K.from_sympy(infinity) / representative)
            if isinstance(ratio, Rational):
                q_infinity = ratio
        classes.append(_ResidueClass(rho, finite, q_infinity))
    return classes


def _lift(W: Poly, G: Poly, Q: Poly, N: int) -> Poly:
    """``W`` with ``W**2 == Q`` modulo ``G**N`` (Newton's iteration from
    ``W**2 == Q`` modulo ``G``)."""
    modulus = G**N
    two = Q.domain.convert(2)
    steps = max(1, (N - 1).bit_length() + 1)
    for _ in range(steps):
        W = ((W + Q * W.invert(modulus)).quo_ground(two)).rem(modulus)
    return W


def _series_at_infinity(Q: Poly, order: int) -> list[object]:
    """The coefficients ``sigma_0 = 1, sigma_1, ...`` of ``sqrt(Q(1/t)
    t**(deg Q) / lc(Q))`` as a power series in ``t`` up to ``t**order``."""
    K = Q.domain
    coefficients = _domain_coefficients(Q.monic())  # highest degree first
    q = coefficients + [K.zero] * max(0, order + 1 - len(coefficients))
    sigma = [Q.domain.one]
    two = Q.domain.convert(2)
    for k in range(1, order + 1):
        total = q[k] if k < len(q) else Q.domain.zero
        for i in range(1, k):
            total = total - sigma[i] * sigma[k - i]
        sigma.append(total / two)
    return sigma


def _function_with_divisor(cls: _ResidueClass, n: int, Q: Poly, s: Optional[object],
                           x: Symbol) -> Optional[tuple[Poly, Poly]]:
    """``u, v`` with ``u + v*w`` vanishing to order ``2*n*q`` at the points
    of the class with residue ``q*rho`` and with pole orders at infinity
    ``n*(sigma -+ q_infinity)``, ``sigma`` the number of finite zeros;
    ``None`` when no such function exists (``n*Delta`` is not principal
    on the curve)."""
    K = Q.domain
    deg_Q = Q.degree()
    m = deg_Q // 2
    sigma = sum(int(q * n) * G.degree() for q, G, _ in cls.finite)
    q_inf = int(cls.q_infinity * n)
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
    rows: list[list[object]] = []
    for q, G, W in cls.finite:
        N = int(2 * q * n)
        modulus = G**N
        W_N = _lift(W, G, Q, N)
        size = modulus.degree()
        columns: list[list[object]] = []
        for i in range(nu):
            columns.append(_coefficients(Poly(x**i, x, domain=K).rem(modulus), size))
        for i in range(nv):
            columns.append(_coefficients((Poly(x**i, x, domain=K) * W_N).rem(modulus), size))
        for k in range(size):
            rows.append([column[k] for column in columns])
    if deg_Q % 2 == 0 and q_inf != 0:
        if s is None:
            return None
        order = max(0, nv - 1 + m - min(E_plus, E_minus) - 1)
        sigma_series = _series_at_infinity(Q, order)
        for sign, bound in ((K.one, E_plus), (-K.one, E_minus)):
            for j in range(bound + 1, du + 1):
                row: list[object] = [K.one if i == j else K.zero for i in range(nu)]
                for k in range(nv):
                    index = k + m - j
                    value = sigma_series[index] if 0 <= index < len(sigma_series) else K.zero
                    row.append(sign * s * value)
                rows.append(row)
    if not rows:
        vector: list[object] = [K.one] + [K.zero] * (unknowns - 1)
    else:
        matrix = DomainMatrix(rows, (len(rows), unknowns), K)
        space = matrix.nullspace()
        if space.shape[0] == 0:
            return None
        vector = list(space.to_list()[0])
    u = Poly.from_dict({(i,): vector[i] for i in range(nu)}, x, domain=K)
    v = Poly.from_dict({(i,): vector[nu + i] for i in range(nv)}, x, domain=K)
    if u.is_zero and v.is_zero:
        return None
    # the norm must be a constant times the product of the G_j**(2 n q_j)
    norm = u**2 - v**2 * Q
    expected = Poly(1, x, domain=K)
    for q, G, _ in cls.finite:
        expected = expected * G**int(2 * q * n)
    if norm.is_zero or norm.degree() != expected.degree() \
            or not (norm - expected.mul_ground(_leading(norm))).is_zero:
        return None
    return u, v


def _coefficients(p: Poly, size: int) -> list[object]:
    """The coefficients of ``p`` from degree 0 to ``size - 1``."""
    coefficients = list(reversed(_domain_coefficients(p)))
    return coefficients + [p.domain.zero] * (size - len(coefficients))


def _domain_coefficients(p: Poly) -> list[object]:
    """The coefficients of ``p`` as elements of its domain, highest degree
    first (``all_coeffs`` converts them to expressions, and converting
    them back into a number field is slow)."""
    return list(p.rep.to_list())


def _leading(p: Poly) -> object:
    """The leading coefficient of ``p`` as an element of its domain."""
    return _domain_coefficients(p)[0]


class _LogTerm:
    """``(rho/n) * log((u + v*w)/prod(G_j**(n*q_j)))``."""

    def __init__(self, rho: Expr, n: int, u: Poly, v: Poly, denominator: Poly) -> None:
        self.rho = rho
        self.n = n
        self.u = u
        self.v = v
        self.denominator = denominator

    def as_expr(self, w: Expr) -> Expr:
        argument = (self.u.as_expr() + self.v.as_expr() * w) / self.denominator.as_expr()
        return as_expr(self.rho / self.n * log(argument))


class _Pair:
    """``a0 + a1*w`` with ``a0``, ``a1`` rational functions of ``x`` on
    the curve ``w**2 == Q``."""

    def __init__(self, a0: Expr, a1: Expr, Q: Expr, x: Symbol) -> None:
        self.a0 = a0
        self.a1 = a1
        self.Q = Q
        self.x = x

    def __mul__(self, other: _Pair) -> _Pair:
        return _Pair(as_expr(cancel(self.a0 * other.a0 + self.a1 * other.a1 * self.Q)),
                     as_expr(cancel(self.a0 * other.a1 + self.a1 * other.a0)), self.Q, self.x)

    def __sub__(self, other: _Pair) -> _Pair:
        return _Pair(as_expr(cancel(self.a0 - other.a0)), as_expr(cancel(self.a1 - other.a1)), self.Q, self.x)

    def scaled(self, c: Expr) -> _Pair:
        return _Pair(as_expr(cancel(c * self.a0)), as_expr(cancel(c * self.a1)), self.Q, self.x)

    def derivative(self) -> _Pair:
        # (a1 w)' = a1' w + a1 Q'/(2 w) = (a1' + a1 Q'/(2 Q)) w
        return _Pair(as_expr(cancel(self.a0.diff(self.x))),
                     as_expr(cancel(self.a1.diff(self.x) + self.a1 * self.Q.diff(self.x) / (2 * self.Q))),
                     self.Q, self.x)

    def inverse(self) -> _Pair:
        norm = as_expr(cancel(self.a0**2 - self.a1**2 * self.Q))
        return _Pair(as_expr(cancel(self.a0 / norm)), as_expr(cancel(-self.a1 / norm)), self.Q, self.x)

    def logarithmic_derivative(self) -> _Pair:
        return self.derivative() * self.inverse()

    def is_zero(self) -> bool:
        return self.a0 == 0 and self.a1 == 0


class _Result:
    """The pieces of the integral of ``A + B*y``."""

    def __init__(self, radical: _Radical, rational_part: Expr, rational_logs: Expr, reduction: _Reduction,
                 logs: list[_LogTerm], remainder: _Pair, nonelementary: Optional[bool]) -> None:
        self.radical = radical
        self.rational_part = rational_part
        self.rational_logs = rational_logs
        self.reduction = reduction
        self.logs = logs
        self.remainder = remainder
        self.nonelementary = nonelementary

    def exact_part(self) -> Expr:
        w = self.radical.w
        return as_expr(self.rational_part
                       + Add(*[S.as_expr() * w / T.as_expr() for S, T in self.reduction.terms]))

    def elementary(self) -> Expr:
        return as_expr(self.exact_part() + self.rational_logs + Add(*[term.as_expr(self.radical.w) for term in self.logs]))

    def remainder_expr(self) -> Expr:
        # a1 w = a1 Q d / sqrt(P): the radical in the denominator
        radical = self.radical
        return as_expr(self.remainder.a0 + cancel(self.remainder.a1 * radical.Q.as_expr()) * radical.d.as_expr()
                       / sqrt(radical.P.as_expr()))


def _integrate(f: Expr, x: Symbol, logarithms: bool) -> Optional[_Result]:
    radical = _parse(f, x)
    if radical is None:
        return None
    Q = radical.Q
    A = radical.A
    numerator, denominator = fraction(cancel(A))
    if A == 0:
        rational_part: Expr = S.Zero
        rational_logs: Expr = S.Zero
    elif logarithms:
        rational_part, rational_logs = ratint(A, x), S.Zero
    else:
        p, q = Poly(numerator, x), Poly(denominator, x)
        quotient, rest = p.div(q)
        rational_part, rational_logs = ratint_ratpart(rest, q, x)
        rational_part = as_expr(rational_part + quotient.integrate(x).as_expr())
    if Q.degree() == 0:
        # sqrt(P) is d*sqrt(c): a rational integrand
        constant = as_expr(sqrt(Q.as_expr()))
        g = as_expr(radical.B * radical.d.as_expr() * constant)
        empty = _Reduction([], Poly(0, x, domain=Q.domain), Poly(1, x, domain=Q.domain), Poly(0, x, domain=Q.domain))
        extra = ratint(g, x) if logarithms else S.Zero
        remainder = _Pair(S.Zero if logarithms else g, S.Zero, Q.as_expr(), x)
        return _Result(radical, as_expr(rational_part + extra), rational_logs, empty, [], remainder, False)
    M, D = radical.algebraic_numerator()
    reduction = _hermite(M, D, Q, x)
    r, D = reduction.r, reduction.D
    Q_expr = Q.as_expr()
    remainder = _Pair(S.Zero, as_expr(cancel(r.as_expr() / (D.as_expr() * Q_expr))), Q_expr, x)
    m = Q.degree() // 2
    if reduction.polynomial.degree() >= m and m >= 1:
        # a pole of order at least two at infinity which no function of
        # the field removes: not elementary
        return _Result(radical, rational_part, rational_logs, reduction, [], remainder, True)
    if not logarithms or r.is_zero:
        return _Result(radical, rational_part, rational_logs, reduction, [], remainder, False if r.is_zero else None)
    if not (Q.domain.is_QQ or Q.domain.is_ZZ):
        return _Result(radical, rational_part, rational_logs, reduction, [], remainder, None)
    logs, remainder, nonelementary = _logarithmic_part(reduction, Q, x, remainder)
    return _Result(radical, rational_part, rational_logs, reduction, logs, remainder, nonelementary)


def _logarithmic_part(reduction: _Reduction, Q: Poly, x: Symbol,
                      remainder: _Pair) -> tuple[list[_LogTerm], _Pair, Optional[bool]]:
    """The logarithmic terms of the classes of residues whose divisors
    are torsion, the remainder ``h - sum(dlog)`` and the decision."""
    r, D = reduction.r, reduction.D
    z = Dummy('z')
    R = _residue_polynomial(r, D, Q, x, z)
    values = _residue_values(R)
    if values is None:
        return [], remainder, None
    m = Q.degree() // 2
    infinity: Optional[Expr] = None
    s_value: Optional[Expr] = None
    if Q.degree() % 2 == 0:
        s_value = as_expr(sqrt(as_expr(Q.LC())))
        if reduction.polynomial.degree() == m - 1:
            infinity = as_expr(-reduction.polynomial.LC() / s_value)
    if not values and infinity is None:
        # no residue anywhere: the remainder, with simple poles only, is a
        # differential of the first kind, whose integral is elementary only
        # when it is zero (its integral would be a function of the field
        # without poles, a constant)
        return [], remainder, not remainder.is_zero()
    generators = list(values)
    if infinity is not None:
        generators.append(infinity)
    if s_value is not None:
        generators.append(s_value)
    K = _field(generators)
    if K is None:
        return [], remainder, None
    classes = _classes(values, infinity, K, r, D, Q, x)
    if classes is None:
        return [], remainder, None
    Q_K = Q.set_domain(K)
    s = K.from_sympy(s_value) if s_value is not None else None
    genus_one = Q.degree() in (3, 4)
    logs: list[_LogTerm] = []
    undecided = False
    for cls in classes:
        found = None
        n0 = cls.denominator
        for k in range(1, _TORSION_BOUND + 1):
            n = k * n0
            found = _function_with_divisor(cls, n, Q_K, s, x)
            if found is not None:
                break
        if found is None:
            if not (genus_one and K.is_QQ):
                undecided = True
            continue
        u, v = found
        rho = cls.rho
        # (u + v w)(u - v w) is a constant times the denominator squared:
        # the conjugate gives the reciprocal function and the opposite
        # residue; the sign with positive leading coefficients is taken
        if not v.is_zero and as_expr(u.LC() * v.LC()).is_negative is True:
            v, rho = v.neg(), as_expr(-rho)
        if as_expr(u.LC()).is_negative is True:
            u, v = u.neg(), v.neg()
        denominator = Poly(1, x, domain=K)
        for q, G, _ in cls.finite:
            denominator = denominator * G**int(q * n)
        term = _LogTerm(rho, n, u, v, denominator)
        logs.append(term)
        F = _Pair(as_expr(u.as_expr() / denominator.as_expr()), as_expr(v.as_expr() / denominator.as_expr()),
                  Q.as_expr(), x)
        remainder = remainder - F.logarithmic_derivative().scaled(as_expr(rho / n))
    if remainder.is_zero():
        return logs, remainder, False
    return logs, remainder, None if undecided else True


def _verified(F: Expr, f: Expr, x: Symbol, P: Expr) -> bool:
    """Whether ``F'`` agrees with ``f`` numerically at points where the
    radicand is positive."""
    derivative = as_expr(F.diff(x))
    rng = random.Random(str(f))
    parameters = sorted(f.free_symbols - {x}, key=str)
    checked = 0
    for _ in range(40):
        values: dict[Basic | complex, Expr | complex] = {
            parameter: Rational(rng.randint(-400, 400), 100) for parameter in parameters}
        values[x] = Rational(rng.randint(-400, 400), 100)
        try:
            radicand = as_expr(P.subs(values)).evalf(30)
            if not radicand.is_positive:
                continue
            expected = f.subs(values).evalf(30)
            found = derivative.subs(values).evalf(30)
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if not (expected.is_number and found.is_number) or expected.has(S.NaN, S.ComplexInfinity):
            continue
        if not found.is_finite or not expected.is_finite:
            continue
        if abs(found - expected) > 1e-20 * (1 + abs(expected)):
            return False
        checked += 1
        if checked >= 4:
            return True
    return checked > 0


def trager_reduce(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(F, h)`` with ``f == F' + h``: ``F`` the elementary part of the
    integral of the algebraic function ``f`` (the integral of the
    rational part, the exact part of the Hermite reduction and the
    logarithms of the residue classes whose divisors are torsion) and
    ``h`` the remaining integrand, with simple poles only, whose integral
    is not elementary or not found; ``None`` when ``f`` is not a rational
    function of ``x`` and of one square root of a polynomial.

    >>> from sympy import symbols, sqrt
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
    f_ = as_expr(f)
    result = attempt(lambda: _integrate(f_, x, True), settings.timeout)
    if result is None:
        return None
    F = as_expr(result.elementary())
    h = result.remainder_expr()
    if not _verified(F, as_expr(f_ - h), x, result.radical.P.as_expr()):
        return None
    return F, h


def trager_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An antiderivative of the algebraic function ``f`` (rational in
    ``x`` and in one square root of a polynomial) by Trager's algorithm,
    or ``None`` when the integral is not elementary or its logarithmic
    part is not found; see :func:`trager_reduce` for the elementary
    part alone.

    >>> from sympy import symbols, sqrt
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
    """
    found = trager_reduce(f, x)
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

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.trager import hermite_reduce_algebraic
    >>> x = symbols('x')
    >>> hermite_reduce_algebraic(x**2/sqrt(x**2 + 1), x)
    (x*sqrt(x**2 + 1)/2, -1/(2*sqrt(x**2 + 1)))
    >>> hermite_reduce_algebraic(1/(x**2*sqrt(x**3 + 1)), x)
    (-sqrt(x**3 + 1)/x, x/(2*sqrt(x**3 + 1)))
    """
    f_ = as_expr(f)
    result = attempt(lambda: _integrate(f_, x, False), settings.timeout)
    if result is None:
        return None
    g = result.exact_part()
    h = as_expr(result.rational_logs + result.remainder_expr())
    if not _verified(g, as_expr(f_ - h), x, result.radical.P.as_expr()):
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

    >>> from sympy import symbols, sqrt
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
    """
    f_ = as_expr(f)
    result = attempt(lambda: _integrate(f_, x, True), settings.timeout)
    if result is None:
        return None
    if result.nonelementary is False:
        # the antiderivative is kept only when it checks
        return False if trager_antiderivative(f_, x) is not None else None
    return result.nonelementary
