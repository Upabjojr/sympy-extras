"""Reduction-based creative telescoping for hyperexponential functions:
Hermite reduction with respect to the derivation `D = \\partial_x + q`
and the telescoper as the first linear dependence of the reduced forms.

For a hyperexponential `F(x, t)` with `\\partial_x F / F = q \\in K(x)`,
`K = \\mathbb{Q}(t)`, a rational multiple `p F` is the `x`-derivative of a
rational multiple `g F` exactly when `p = D(g) := g' + q g`. The
*Hermite reduction* of [BCCLX]_ writes every `p \\in K(x)` as

.. math::

    p = D(g) + r,

with `r` in a subspace `V` of *reduced* forms satisfying
`V \\cap D(K(x)) = \\{0\\}`, so that `r` is a normal form of `p` modulo the
integrable functions: `r = 0` if and only if `p F` has a hyperexponential
antiderivative. The reduction is `K`-linear. Applied to
`p_j = \\partial_t^j F / F` (`p_0 = 1`, `p_{j+1} = \\partial_t p_j + p\\, p_j`
with `p = \\partial_t F/F`), it yields `r_0, r_1, \\ldots`, and the first
linear dependence `\\sum_j a_j(t)\\, r_j = 0` over `K` gives the telescoper
`L = \\sum_j a_j(t) \\partial_t^j` with the certificate `R = \\sum_j a_j g_j`:
`L(F) = \\partial_x (R F)`. The order found is minimal, and no ansatz on
the degrees of the certificate is needed, which is what makes this
faster than the Almkvist–Zeilberger algorithm of
:mod:`sympy_extras.integrals.telescoping` [BCCL]_, [BCCLX]_.

The reduced forms, with `q = A/B` in lowest terms, `\\beta` the squarefree
part of `B` and `\\tilde{B} = B/\\beta`:

1. **The shell.** A simple pole of `q` whose residue is a nonzero integer
   `n` comes from a factor `d^n` of `F`: it is moved into the *shell*
   `S = \\prod d^{n}`, `F = S \\hat F`, and `p F = (p S) \\hat F` is reduced
   with respect to `\\hat q = q - S'/S`, whose residues at simple poles are
   not integers. (This is the multiplicative decomposition of Geddes, Le
   and Li, and it is what makes the local leading terms below never
   cancel.)
2. **Poles away from `B`** (Hermite's classical reduction): a pole of
   order `n \\ge 2` at a squarefree factor `s` of the denominator coprime
   to `B` is lowered with `g = h/s^{n-1}`, `h` solving
   `-(n-1) s' h \\equiv` the leading coefficient modulo `s`; this creates
   poles at the factors of `B` only.
3. **Poles at the factors of `B`.** Writing the fraction as
   `N/(B \\beta^m Q_c)`, `Q_c` coprime to `B`, one has
   `D(h/\\beta^m) = [(\\beta h' - m \\beta' h)\\tilde B + A h]/(B \\beta^m)`, and
   `h \\equiv N Q_c^{-1} (A - m \\beta' \\tilde B)^{-1} \\pmod \\beta` lowers `m` by
   one (the modulus is coprime to `\\beta` because the residues are not
   integers); at `m = 0` the denominator is `\\tilde B Q_c`, and
   `g = h \\beta` with `h \\equiv N (A Q_c)^{-1} \\pmod{\\tilde B}` removes the
   remaining poles at `B` altogether. So the reduced forms have no pole at
   a factor of `B`, and only simple poles elsewhere.
4. **The polynomial part.** `D(g)` is a polynomial only for `g = B h`,
   `E(h) = B h' + (A + B') h`; the polynomial part of `p` is reduced
   modulo the row echelon form of `E(1), E(x), E(x^2), \\ldots` (the
   *polynomial reduction* of [BCCLX]_; the leading degree of `E(x^j)`
   drops for one exceptional `j` when the residue of `q` at infinity is
   a negative integer, which the echelon form handles).

The differential equation of the integral and its solution are as in
:mod:`sympy_extras.integrals.telescoping` (`\\sum_j a_j I^{(j)} = [R F]_a^b`,
``dsolve`` and initial values). The algebraic case of [CKK]_ is not
implemented.

Examples
========

>>> from sympy import symbols, exp, oo
>>> from sympy_extras.integrals.reduction import hermite_reduce, reduction_telescoper, reduction_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> g, r = hermite_reduce(1/x**2, -2*x, x)
>>> g, r
(-1/x, -2)
>>> reduction_telescoper(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)
>>> reduction_integral(exp(-x**2 - t**2/x**2), x, 0, oo, t)
ConditionalValue(sqrt(pi)*exp(-2*t)/2)

References
==========

.. [BCCL] A. Bostan, S. Chen, F. Chyzak, Z. Li, Complexity of creative
   telescoping for bivariate rational functions, ISSAC 2010, 203–210.
.. [BCCLX] A. Bostan, S. Chen, F. Chyzak, Z. Li, G. Xin, Hermite reduction
   and creative telescoping for hyperexponential functions, ISSAC 2013,
   77–84.
.. [CKK] S. Chen, M. Kauers, C. Koutschan, Reduction-based creative
   telescoping for algebraic functions, ISSAC 2016, 175–182.
.. [Bronstein] M. Bronstein, Symbolic Integration I, 2nd edition,
   Springer, 2005, section 2.2 (Hermite reduction).
.. [GLL] K. Geddes, H.-C. Le, Z. Li, Differential rational normal forms and
   a reduction algorithm for hyperexponential functions, ISSAC 2004.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, diff
from sympy.core.numbers import Integer, nan, oo, zoo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.dense import Matrix
from sympy.polys.domains.domain import Domain
from sympy.polys.domains import QQ, QQ_I, ZZ_I
from sympy.polys.polyerrors import CoercionFailed, NotInvertible, PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve
from sympy.solvers.solveset import linsolve

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .telescoping import (DifferentialTelescoper, is_hyperexponential, logarithmic_derivative, _boundary, _constants,
                          _initial_values, _normalized, _order, _polys, _terms)

__all__ = ['HermiteReducer', 'hermite_reduce', 'reduction_telescoper', 'reduction_ode', 'reduction_integral']


class _Split(Exception):
    """A pole whose conjugates have different integer residues."""


def _integer(value: Expr) -> Optional[int]:
    """``value`` as a Python integer when it is one."""
    v = as_expr(cancel(value))
    if isinstance(v, Integer):
        return int(v)
    return None


class HermiteReducer:
    """The Hermite reduction with respect to ``D(g) = g' + q*g`` for a
    fixed rational ``q`` (see the module documentation).

    Parameters
    ==========

    q : Expr
        A rational function of ``x`` with coefficients in ``Q(t)`` (or its
        Gaussian extension).
    x : Symbol
    t : Symbol
        The parameter (the coefficient field is the rational functions of
        ``t``).

    Attributes
    ==========

    shell : Expr
        The polynomial ``S`` (in ``x``) with ``q = q_hat + S'/S`` and no
        integer residue of ``q_hat`` at a simple pole.
    A, B : Poly
        Numerator and denominator of ``q_hat``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.reduction import HermiteReducer
    >>> x, t = symbols('x t')
    >>> reducer = HermiteReducer(2/x + 1/(x - t)**2, x, t)
    >>> reducer.shell, reducer.A.as_expr(), reducer.B.as_expr()
    (x**2, 1, t**2 - 2*t*x + x**2)
    >>> g, r = reducer.reduce(1/x)
    >>> g, r
    (t**2/2 - t*x + x**2/2, t - 1/2)
    """

    def __init__(self, q: ExprLike, x: Symbol, t: Symbol) -> None:
        self.x = x
        self.t = t
        q_ = as_expr(sympify(q))
        A, B = _polys(q_, x, t)
        domain: Domain = A.domain.unify(QQ.frac_field(t))
        self.domain = domain
        self.shell: Expr = S.One
        try:
            A, B = self._normalize(A.set_domain(domain), B.set_domain(domain))
        except _Split:
            # conjugate poles with different integer residues: factor over
            # the Gaussian rationals
            self.domain = QQ_I.frac_field(t)
            self.shell = S.One
            A, B = self._normalize(A.set_domain(self.domain), B.set_domain(self.domain))
        self.A: Poly = A
        self.B: Poly = B
        self.beta: Poly = self._squarefree_part(B)
        self.cofactor: Poly = B.quo(self.beta)          # B / beta
        self.q: Expr = as_expr(cancel(A.as_expr() / B.as_expr()))
        self._echelon: dict[int, tuple[Poly, Poly]] = {}
        self._basis_size = 0

    def _gaussian(self) -> bool:
        return self.domain == QQ_I.frac_field(self.t) or self.domain == ZZ_I.frac_field(self.t)

    def _upgrade(self, domain: Domain) -> None:
        """Move every polynomial to the larger ``domain``."""
        self.domain = domain
        self.A, self.B = self.A.set_domain(domain), self.B.set_domain(domain)
        self.beta, self.cofactor = self.beta.set_domain(domain), self.cofactor.set_domain(domain)
        self._echelon = {d: (v.set_domain(domain), h.set_domain(domain)) for d, (v, h) in self._echelon.items()}

    # -- construction ---------------------------------------------------

    def _lc(self, P: Poly) -> object:
        """The leading coefficient of ``P`` as an element of the domain."""
        return self.domain.from_sympy(P.LC())

    def _coefficient(self, P: Poly, degree: int) -> object:
        return self.domain.from_sympy(P.nth(degree))

    def _poly(self, e: ExprLike) -> Poly:
        return Poly(as_expr(sympify(e)), self.x, domain=self.domain)

    def _squarefree_part(self, P: Poly) -> Poly:
        result = self._poly(1)
        for factor, _ in P.sqf_list()[1]:
            if factor.degree() > 0:
                result = result * factor
        return result.monic() if result.degree() > 0 else result

    def _normalize(self, A: Poly, B: Poly) -> tuple[Poly, Poly]:
        """Move the factors ``d**n`` of ``F`` (simple poles of ``q`` with
        the integer residue ``n``) into the shell."""
        shell: Expr = S.One
        q = as_expr(cancel(A.as_expr() / B.as_expr()))
        for factor, multiplicity in B.factor_list()[1]:
            if factor.degree() == 0 or multiplicity != 1:
                continue
            d = factor.monic()
            cofactor = B.quo(d)
            try:
                inverse = (cofactor * d.diff(self.x)).invert(d)
            except (NotInvertible, PolynomialError, ZeroDivisionError):
                continue
            residue = (A * inverse).rem(d)
            if residue.degree() > 0:
                if self._gaussian():
                    continue
                raise _Split
            n = _integer(as_expr(residue.as_expr()))
            if n is None or n == 0:
                continue
            shell = as_expr(shell * d.as_expr()**n)
            # q - n d'/d
            q = as_expr(cancel(q - n * d.diff(self.x).as_expr() / d.as_expr()))
        self.shell = shell
        A2, B2 = _polys(q, self.x, self.t)
        return A2.set_domain(self.domain), B2.set_domain(self.domain)

    # -- the derivation ---------------------------------------------------

    def derivation(self, g: ExprLike) -> Expr:
        """``D(g) = g' + q_hat*g`` (with the normalised ``q``)."""
        g_ = as_expr(sympify(g))
        return as_expr(cancel(diff(g_, self.x) + self.q * g_))

    # -- the reduction ----------------------------------------------------

    def _numerator_denominator(self, p: Expr) -> tuple[Poly, Poly]:
        P, Q = _polys(p, self.x, self.t)
        try:
            return P.set_domain(self.domain), Q.set_domain(self.domain)
        except CoercionFailed:
            self._upgrade(self.domain.unify(P.domain))
            return P.set_domain(self.domain), Q.set_domain(self.domain)

    def _coprime_part(self, Q: Poly) -> Poly:
        """``Q`` with every factor of ``B`` removed."""
        result = Q
        while True:
            common = result.gcd(self.beta)
            if common.degree() == 0:
                return result
            result = result.quo(common)

    def _reduce_coprime_poles(self, p: Expr) -> tuple[Expr, Expr]:
        """Step 2: the poles of order at least two away from ``B``; returns
        ``(g, p - D(g))``."""
        g: Expr = S.Zero
        current = p
        while True:
            P, Q = self._numerator_denominator(current)
            Q_c = self._coprime_part(Q)
            target: Optional[tuple[Poly, int]] = None
            for factor, multiplicity in Q_c.sqf_list()[1]:
                if factor.degree() > 0 and multiplicity >= 2:
                    target = (factor.monic(), int(multiplicity))
                    break
            if target is None:
                return g, current
            s, n = target
            # leading coefficient of the pole: P / (Q / s^n) mod s
            rest = Q.quo(s**n)
            lead = (P * rest.invert(s)).rem(s)
            h = (lead * (s.diff(self.x).mul_ground(self.domain.convert(1 - n))).invert(s)).rem(s)
            g_step = as_expr(h.as_expr() / s.as_expr()**(n - 1))
            g = g + g_step
            current = as_expr(cancel(current - self.derivation(g_step)))

    def _reduce_b_poles(self, p: Expr) -> tuple[Expr, Expr]:
        """Step 3: the poles at the factors of ``B``; returns
        ``(g, p - D(g))`` with the latter free of such poles."""
        P, Q = self._numerator_denominator(p)
        Q_c = self._coprime_part(Q)
        Q_B = Q.quo(Q_c)
        g: Expr = S.Zero
        if Q_B.degree() == 0 and self.beta.degree() == 0:
            return g, p
        # the smallest m with Q_B | B beta^m
        m = 0
        while not (self.B * self.beta**m).rem(Q_B).is_zero:
            m += 1
        N = P * (self.B * self.beta**m * Q_c).quo(Q)
        beta_prime = self.beta.diff(self.x)
        inverse_c = Q_c.invert(self.beta)
        while m >= 0:
            modulus = self.A - beta_prime.mul_ground(self.domain.convert(m)) * self.cofactor
            h = (N * inverse_c * modulus.invert(self.beta)).rem(self.beta)
            phi = (self.beta * h.diff(self.x) - beta_prime.mul_ground(self.domain.convert(m)) * h) * self.cofactor \
                + self.A * h
            g = g + as_expr(h.as_expr() / self.beta.as_expr()**m)
            N = (N - phi * Q_c).quo(self.beta)
            m -= 1
        # now the fraction is N / (cofactor Q_c)
        if self.cofactor.degree() > 0:
            inverse_a = (self.A * Q_c).invert(self.cofactor)
            h = (N * inverse_a).rem(self.cofactor)
            psi = (self.beta * h.diff(self.x) + beta_prime * h) * self.cofactor + self.A * h
            g = g + as_expr(h.as_expr() * self.beta.as_expr())
            N = (N - psi * Q_c).quo(self.cofactor)
        remainder = as_expr(cancel(N.as_expr() / Q_c.as_expr()))
        return g, remainder

    def _image(self, j: int) -> Poly:
        """``E(x**j) = B (x**j)' + (A + B') x**j``."""
        monomial = self._poly(self.x**j)
        return self.B * monomial.diff(self.x) + (self.A + self.B.diff(self.x)) * monomial

    def _extend_basis(self, size: int) -> None:
        """The row echelon form of ``E(1), ..., E(x**size)`` (pivots at the
        leading degrees), with the preimages."""
        while self._basis_size < size + 1:
            j = self._basis_size
            vector, preimage = self._image(j), self._poly(self.x**j)
            self._basis_size += 1
            while not vector.is_zero:
                degree = vector.degree()
                if degree in self._echelon:
                    pivot, pivot_preimage = self._echelon[degree]
                    factor = self.domain.quo(self._lc(vector), self._lc(pivot))
                    vector = vector - pivot.mul_ground(factor)
                    preimage = preimage - pivot_preimage.mul_ground(factor)
                else:
                    self._echelon[degree] = (vector, preimage)
                    break

    def _reduce_polynomial(self, P: Poly) -> tuple[Expr, Poly]:
        """Step 4: ``(g, r)`` with ``P = D(g) + r`` and ``r`` reduced modulo
        the image of the polynomials."""
        if P.is_zero:
            return S.Zero, P
        n_a = self.A.degree() if not self.A.is_zero else -1
        n_b = self.B.degree()
        delta = max(n_a, n_b - 1)
        size = P.degree() - delta + 2
        if n_a == n_b - 1 and n_b >= 1:
            # E(x**j) has a lower degree for j = -(n_b + lc(A)/lc(B))
            mu = as_expr(self.domain.to_sympy(self.domain.quo(self.A.LC(), self.B.LC()))) + n_b
            exceptional = _integer(-mu)
            if exceptional is not None and exceptional >= 0:
                size = max(size, exceptional + 2)
        self._extend_basis(max(size, 1))
        remainder = P
        preimage = self._poly(0)
        # a pivot at degree d only changes the coefficients of degree at
        # most d: one descending pass leaves zeros at every pivot degree
        for degree in sorted(self._echelon, reverse=True):
            if remainder.is_zero or degree > remainder.degree():
                continue
            coefficient = self._coefficient(remainder, degree)
            if not coefficient:
                continue
            pivot, pivot_preimage = self._echelon[degree]
            factor = self.domain.quo(coefficient, self._lc(pivot))
            remainder = remainder - pivot.mul_ground(factor)
            preimage = preimage + pivot_preimage.mul_ground(factor)
        g = as_expr(self.B.as_expr() * preimage.as_expr())
        return g, remainder

    def reduce(self, p: ExprLike) -> tuple[Expr, Expr]:
        """``(g, r)`` with ``p*S = D(g) + r`` for the normalised derivation
        (``S`` the shell) and ``r`` a reduced form: ``r == 0`` if and only
        if ``p*F`` has a hyperexponential antiderivative."""
        p_ = as_expr(cancel(as_expr(sympify(p)) * self.shell))
        g1, current = self._reduce_coprime_poles(p_)
        g2, current = self._reduce_b_poles(current)
        P, Q = self._numerator_denominator(current)
        polynomial, proper_numerator = P.div(Q)
        g3, reduced = self._reduce_polynomial(polynomial)
        proper = as_expr(cancel(proper_numerator.as_expr() / Q.as_expr()))
        return as_expr(cancel(g1 + g2 + g3)), as_expr(cancel(proper + reduced.as_expr()))


def hermite_reduce(p: ExprLike, q: ExprLike, x: Symbol, t: Optional[Symbol] = None) -> tuple[Expr, Expr]:
    """``(g, r)`` with ``p = g' + q*g + r`` and ``r`` a reduced form with
    respect to ``D = d/dx + q`` (``r == 0`` exactly when ``p*exp(Integral(q))``
    has an antiderivative of the form ``g*exp(Integral(q))``).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.reduction import hermite_reduce
    >>> x, t = symbols('x t')
    >>> hermite_reduce(x, -2*x, x)          # x exp(-x**2) = (-exp(-x**2)/2)'
    (-1/2, 0)
    >>> hermite_reduce(1, -2*x, x)          # exp(-x**2) is not integrable
    (0, 1)
    >>> hermite_reduce(1/(x**2 + t), 0, x, t)
    (0, 1/(t + x**2))
    """
    t_ = t if t is not None else Symbol('t')
    reducer = HermiteReducer(q, x, t_)
    g, r = reducer.reduce(p)
    S_ = reducer.shell
    return as_expr(cancel(g / S_)), as_expr(cancel(r / S_))


def _dependency(vectors: list[list[Expr]]) -> Optional[list[Expr]]:
    """A nontrivial ``a`` with ``sum(a_j v_j) = 0`` and ``a[-1] != 0``."""
    width = max(len(v) for v in vectors)
    matrix = Matrix([[v[i] if i < len(v) else S.Zero for v in vectors] for i in range(width)])
    for vector in matrix.nullspace():
        values = [as_expr(cancel(v)) for v in vector]
        if values[-1] != 0:
            return values
    return None


def _coordinates(forms: Sequence[Expr], x: Symbol, t: Symbol) -> list[list[Expr]]:
    """The reduced forms over a common denominator, as coefficient lists."""
    numerators: list[Poly] = []
    denominators: list[Poly] = []
    for r in forms:
        P, Q = _polys(r, x, t)
        numerators.append(P)
        denominators.append(Q)
    common = denominators[0]
    for Q in denominators[1:]:
        common, Q_ = common.unify(Q)
        common = common.lcm(Q_)
    vectors: list[list[Expr]] = []
    for P, Q in zip(numerators, denominators):
        common_, Q_ = common.unify(Q)
        P_, _ = P.unify(Q_)
        numerator = P_ * common_.quo(Q_)
        vectors.append([as_expr(c) for c in reversed(numerator.all_coeffs())])
    return vectors


class _Term:
    """A hyperexponential term with its reducer and the sequence
    ``p_j = partial_t^j F / F``."""

    def __init__(self, F: Expr, x: Symbol, t: Symbol) -> None:
        self.F = F
        self.t = t
        q = logarithmic_derivative(F, x)
        p = logarithmic_derivative(F, t)
        if q is None or p is None:
            raise ValueError("not hyperexponential: %s" % F)
        self.p = p
        self.reducer = HermiteReducer(q, x, t)
        self.ps: list[Expr] = []
        self.gs: list[Expr] = []
        self.rs: list[Expr] = []

    def extend(self) -> None:
        """The next ``p_j`` and its reduction."""
        if not self.ps:
            self.ps.append(S.One)
        else:
            self.ps.append(as_expr(cancel(diff(self.ps[-1], self.t) + self.p * self.ps[-1])))
        g, r = self.reducer.reduce(self.ps[-1])
        self.gs.append(g)
        self.rs.append(r)

    def certificate(self, relation: Sequence[Expr]) -> Expr:
        """``R`` with ``sum(a_j partial_t^j F) = partial_x(R F)`` for the
        term alone."""
        return as_expr(cancel(Add(*[a * g for a, g in zip(relation, self.gs)]) / self.reducer.shell))


def _similar(u: Expr, v: Expr, x: Symbol, t: Symbol) -> Optional[Expr]:
    """``u/v`` when it is a rational function of ``x`` and ``t``."""
    ratio = as_expr(cancel(u / v))
    if ratio.is_rational_function(x, t):
        return ratio
    return None


def _grouped(terms: Sequence[Expr], x: Symbol, t: Symbol) -> list[Expr]:
    """The terms with the similar ones (rational ratio) added together."""
    groups: list[Expr] = []
    for term in terms:
        for i, group in enumerate(groups):
            ratio = _similar(term, group, x, t)
            if ratio is not None:
                groups[i] = as_expr(cancel(group * (1 + ratio)))
                break
        else:
            groups.append(term)
    return [g for g in groups if g != 0]


def _telescope(terms: Sequence[Expr], x: Symbol, t: Symbol, max_order: int) -> Optional[tuple[list[Expr], Expr]]:
    """The coefficients and the certificate of the minimal common
    telescoper of non-similar hyperexponential terms: the first linear
    dependence of the concatenated reduced forms."""
    objects = [_Term(term, x, t) for term in terms]
    F = Add(*terms)
    for j in range(max_order + 1):
        vectors: list[list[Expr]] = [[] for _ in range(j + 1)]
        for term in objects:
            term.extend()
            for i, v in enumerate(_coordinates(term.rs, x, t)):
                vectors[i].extend(v)
        relation = _dependency(vectors)
        if relation is None:
            continue
        normalized, factor_ = _normalized(relation, t)
        if len(objects) == 1:
            certificate = objects[0].certificate(relation)
        else:
            certificate = as_expr(cancel(Add(*[term.certificate(relation) * term.F for term in objects]) / F))
        return normalized, as_expr(cancel(certificate * factor_))
    return None


def reduction_telescoper(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 6) -> Optional[DifferentialTelescoper]:
    """The minimal telescoper of a hyperexponential ``F(x, t)``, or of a sum
    of hyperexponential terms (similar terms are merged, and the minimal
    common telescoper of the others is found by a joint linear dependence
    of the reduced forms), by Hermite reduction, as a
    :class:`~sympy_extras.integrals.telescoping.DifferentialTelescoper`;
    ``None`` when the order exceeds ``max_order`` or ``F`` is not such a
    sum.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.reduction import reduction_telescoper
    >>> x, t = symbols('x t')
    >>> found = reduction_telescoper(1/(x**2 + t**2), x, t)
    >>> found.coefficients, found.certificate, found.check()
    ([1, t], -x, True)
    >>> reduction_telescoper(exp(-x**2)*exp(2*x*t), x, t).coefficients
    [-2*t, 1]
    >>> reduction_telescoper(exp(-t*x**2) + exp(-t**2*x**2), x, t).coefficients
    [1, 5*t, 2*t**2]
    """
    F_ = as_expr(sympify(F))
    if is_hyperexponential(F_, x, t):
        groups = [F_]
    else:
        terms = _terms(F_, x, t)
        if not terms:
            return None
        groups = _grouped(terms, x, t)
        if not groups:
            return None
    result = _telescope(groups, x, t, max_order)
    if result is None:
        return None
    coefficients, certificate = result
    if len(groups) > 1:
        certificate = as_expr(cancel(certificate * Add(*groups) / F_))
    return DifferentialTelescoper(F_, x, t, coefficients, certificate)


def reduction_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The differential equation ``sum_j a_j(t) I^(j)(t) = [R F]_a^b`` of
    ``I(t) = Integral(F, (x, a, b))`` from the reduction-based telescoper.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.reduction import reduction_ode
    >>> x, t = symbols('x t')
    >>> reduction_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    telescoper = reduction_telescoper(F_, x, t)
    if telescoper is None:
        return None
    G = as_expr(telescoper.certificate * F_)
    upper = _boundary(G, x, b_, '-', assumptions)
    lower = _boundary(G, x, a_, '+', assumptions)
    if upper is None or lower is None:
        return None
    return Eq(telescoper.operator(name), as_expr(simplify(upper - lower)))


def reduction_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of ``t`` by the holonomic
    method with the reduction-based telescoper (see
    :func:`~sympy_extras.integrals.telescoping.holonomic_integral` for the
    solution of the equation and the initial values).

    >>> from sympy import symbols, exp, cos, oo
    >>> from sympy_extras.integrals.reduction import reduction_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> reduction_integral(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
    ConditionalValue(sqrt(pi)*exp(-t**2)/2)
    """
    from .definite import verify_numerically
    from .marichev import tidy
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    equation = reduction_ode(F_, x, a_, b_, t, assumptions)
    if equation is None:
        return None
    unknown = Function('I')(t)
    if not isinstance(unknown, AppliedUndef):
        return None
    order = _order(equation.lhs, unknown, t)
    general = attempt(lambda: dsolve(equation, unknown), settings.timeout)
    if not isinstance(general, Eq) or general.lhs != unknown:
        return None
    solution = as_expr(general.rhs)
    constants = _constants(solution)
    if len(constants) != order:
        return None
    if constants:
        fixed: Optional[Expr] = None
        for t0 in (S.One, Integer(2), S.Half, Integer(3)):
            values = _initial_values(F_, x, a_, b_, t, t0, order, assumptions)
            if values is None:
                continue
            equations = [as_expr(diff(solution, t, j).subs(t, t0)) - v for j, v in enumerate(values)]
            solved = attempt(lambda: linsolve(equations, constants), settings.timeout)
            if not isinstance(solved, FiniteSet) or len(solved) != 1:
                continue
            assignment = dict(zip(constants, [as_expr(v) for v in list(solved)[0]]))
            fixed = as_expr(solution.xreplace(assignment))
            break
        if fixed is None:
            return None
        solution = fixed
    value = tidy(solution, assumptions)
    if value.has(nan, zoo, oo, -oo, Piecewise):
        return None
    if settings.numerical_checks and verify_numerically(value, F_, x, a_, b_, assumptions) is False:
        return None
    return ConditionalValue(value)
