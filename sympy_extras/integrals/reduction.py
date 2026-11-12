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
the linear ODE solvers and initial values). The algebraic case of [CKK]_
is not implemented.

The module works on the expressions of the core and imports no SymPy:
the polynomials in `x` over the field of the parameters of
:mod:`.telescoping` (the typed polynomial layer). Only the initial values
and the numerical check of :func:`reduction_integral` reach SymPy, through
:mod:`.telescoping_sympy`; :mod:`.reduction_sympy` is the old interface on
SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import exp
>>> from sympy_extras.integrals.reduction import hermite_reduce, reduction_telescoper
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> g, r = hermite_reduce(1/x**2, -2*x, x)
>>> g, r
(-1/x, -2)
>>> reduction_telescoper(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)

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

from typing import Final, Optional, Sequence

from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import diff
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, nan, oo, zoo
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.polys.polytools import cancel
from .conditions import ConditionalValue
from .telescoping import (_POLYNOMIAL_FAILURES, DifferentialTelescoper, KPoly, _Context, _equation, _normalized,
                          _nullspace, _solve, _terms, _verified, is_hyperexponential, logarithmic_derivative)

__all__ = ['HermiteReducer', 'hermite_reduce', 'reduction_telescoper', 'reduction_ode', 'reduction_integral']

_ZERO = Integer(0)
_ONE = Integer(1)


class _Split(Exception):
    """A pole whose conjugates have different integer residues."""


def _integer(value: Expr) -> Optional[int]:
    """``value`` as a Python integer when it is one."""
    v = cancel(value)
    if isinstance(v, Integer):
        return v.p
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
    parameters : sequence of Expr
        Expressions whose symbols join the coefficient field from the start
        (those of the functions reduced later; the field grows when a new
        one occurs otherwise).

    Attributes
    ==========

    shell : Expr
        The polynomial ``S`` (in ``x``) with ``q = q_hat + S'/S`` and no
        integer residue of ``q_hat`` at a simple pole.
    A, B : Expr
        Numerator and denominator of ``q_hat``, polynomials in ``x``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.reduction import HermiteReducer
    >>> x, t = symbols('x t')
    >>> reducer = HermiteReducer(2/x + 1/(x - t)**2, x, t)
    >>> reducer.shell, reducer.A, reducer.B
    (x**2, 1, t**2 - 2*t*x + x**2)
    >>> g, r = reducer.reduce(1/x)
    >>> g, r
    (t**2/2 - t*x + x**2/2, t - 1/2)
    """

    def __init__(self, q: ExprLike, x: Symbol, t: Symbol, parameters: Sequence[Expr] = ()) -> None:
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        q_ = to_expr(q)
        context = _Context([q_, t, *parameters], x)
        A, B = context.fraction(q_)
        self.context: _Context = context
        self.shell: Expr = _ONE
        try:
            A, B = self._normalize(A, B)
        except _Split:
            # conjugate poles with different integer residues: factor over
            # the Gaussian rationals
            self.context = context.extended([], gaussian=True)
            self.shell = _ONE
            A, B = self.context.fraction(q_)
            A, B = self._normalize(A, B)
        self._A: KPoly = A
        self._B: KPoly = B
        self._beta: KPoly = self._squarefree_part(B)
        self._cofactor: KPoly = B.quo(self._beta)          # B / beta
        self.q: Final[Expr] = self.context.fraction_expr(A, B)
        self._echelon: dict[int, tuple[KPoly, KPoly]] = {}
        self._basis_size = 0

    @property
    def A(self) -> Expr:
        return self.context.poly_expr(self._A)

    @property
    def B(self) -> Expr:
        return self.context.poly_expr(self._B)

    def _upgrade(self, exprs: Sequence[Expr]) -> None:
        """Move every polynomial to the field of the generators of
        ``exprs`` too."""
        old = self.context
        new = old.extended(exprs)
        self.context = new
        self._A, self._B = new.convert(self._A, old), new.convert(self._B, old)
        self._beta, self._cofactor = new.convert(self._beta, old), new.convert(self._cofactor, old)
        self._echelon = {d: (new.convert(v, old), new.convert(h, old)) for d, (v, h) in self._echelon.items()}

    # -- construction ---------------------------------------------------

    def _squarefree_part(self, P: KPoly) -> KPoly:
        result = self.context.constant(self.context.field.one)
        for factor, _ in P.sqf_list()[1]:
            if factor.degree > 0:
                result = result*factor
        return result.monic() if result.degree > 0 else result

    def _normalize(self, A: KPoly, B: KPoly) -> tuple[KPoly, KPoly]:
        """Move the factors ``d**n`` of ``F`` (simple poles of ``q`` with
        the integer residue ``n``) into the shell."""
        K = self.context
        shell: Expr = _ONE
        q = K.fraction_expr(A, B)
        for factor, multiplicity in B.factor_list()[1]:
            if factor.degree == 0 or multiplicity != 1:
                continue
            d = factor.monic()
            cofactor = B.quo(d)
            try:
                inverse = (cofactor*d.diff()).invert(d)
            except _POLYNOMIAL_FAILURES:
                continue
            residue = (A*inverse).rem(d)
            if residue.degree > 0:
                if K.gaussian:
                    continue
                raise _Split
            n = _integer(K.element_expr(residue.LC)) if not residue.is_zero else 0
            if n is None or n == 0:
                continue
            d_expr = K.poly_expr(d)
            shell = mul(shell, pow(d_expr, Integer(n)))
            # q - n d'/d
            q = cancel(q - Integer(n)*K.poly_expr(d.diff())/d_expr)
        self.shell = shell
        return K.fraction(q)

    # -- the derivation ---------------------------------------------------

    def derivation(self, g: ExprLike) -> Expr:
        """``D(g) = g' + q_hat*g`` (with the normalised ``q``)."""
        g_ = to_expr(g)
        return cancel(diff(g_, self.x) + self.q*g_)

    # -- the reduction ----------------------------------------------------

    def _numerator_denominator(self, p: Expr) -> tuple[KPoly, KPoly]:
        if not self.context.covers([p]):
            self._upgrade([p])
        return self.context.fraction(p)

    def _coprime_part(self, Q: KPoly) -> KPoly:
        """``Q`` with every factor of ``B`` removed."""
        result = Q
        while True:
            common = result.gcd(self._beta)
            if common.degree == 0:
                return result
            result = result.quo(common)

    def _reduce_coprime_poles(self, p: Expr) -> tuple[Expr, Expr]:
        """Step 2: the poles of order at least two away from ``B``; returns
        ``(g, p - D(g))``."""
        g: Expr = _ZERO
        current = p
        while True:
            P, Q = self._numerator_denominator(current)
            K = self.context
            Q_c = self._coprime_part(Q)
            target: Optional[tuple[KPoly, int]] = None
            for factor, multiplicity in Q_c.sqf_list()[1]:
                if factor.degree > 0 and multiplicity >= 2:
                    target = (factor.monic(), multiplicity)
                    break
            if target is None:
                return g, current
            s, n = target
            # leading coefficient of the pole: P / (Q / s^n) mod s
            rest = Q.quo(s**n)
            lead = (P*rest.invert(s)).rem(s)
            h = (lead*(s.diff().mul_ground(K.field.from_int(1 - n))).invert(s)).rem(s)
            g_step = K.poly_expr(h)/pow(K.poly_expr(s), Integer(n - 1))
            g = g + g_step
            current = cancel(current - self.derivation(g_step))

    def _reduce_b_poles(self, p: Expr) -> tuple[Expr, Expr]:
        """Step 3: the poles at the factors of ``B``; returns
        ``(g, p - D(g))`` with the latter free of such poles."""
        P, Q = self._numerator_denominator(p)
        K = self.context
        Q_c = self._coprime_part(Q)
        Q_B = Q.quo(Q_c)
        g: Expr = _ZERO
        beta, A, B, cofactor = self._beta, self._A, self._B, self._cofactor
        if Q_B.degree == 0 and beta.degree == 0:
            return g, p
        # the smallest m with Q_B | B beta^m
        m = 0
        while not (B*beta**m).rem(Q_B).is_zero:
            m += 1
        N = P*(B*beta**m*Q_c).quo(Q)
        beta_prime = beta.diff()
        inverse_c = Q_c.invert(beta)
        while m >= 0:
            m_ = K.field.from_int(m)
            modulus = A - beta_prime.mul_ground(m_)*cofactor
            h = (N*inverse_c*modulus.invert(beta)).rem(beta)
            phi = (beta*h.diff() - beta_prime.mul_ground(m_)*h)*cofactor + A*h
            g = g + K.poly_expr(h)/pow(K.poly_expr(beta), Integer(m))
            N = (N - phi*Q_c).quo(beta)
            m -= 1
        # now the fraction is N / (cofactor Q_c)
        if cofactor.degree > 0:
            inverse_a = (A*Q_c).invert(cofactor)
            h = (N*inverse_a).rem(cofactor)
            psi = (beta*h.diff() + beta_prime*h)*cofactor + A*h
            g = g + K.poly_expr(h)*K.poly_expr(beta)
            N = (N - psi*Q_c).quo(cofactor)
        return g, K.fraction_expr(N, Q_c)

    def _image(self, j: int) -> KPoly:
        """``E(x**j) = B (x**j)' + (A + B') x**j``."""
        monomial = self._monomial(j)
        return self._B*monomial.diff() + (self._A + self._B.diff())*monomial

    def _monomial(self, j: int) -> KPoly:
        F = self.context.field
        return self.context.from_elements([F.one] + [F.zero]*j)

    def _extend_basis(self, size: int) -> None:
        """The row echelon form of ``E(1), ..., E(x**size)`` (pivots at the
        leading degrees), with the preimages."""
        F = self.context.field
        while self._basis_size < size + 1:
            j = self._basis_size
            vector, preimage = self._image(j), self._monomial(j)
            self._basis_size += 1
            while not vector.is_zero:
                degree = vector.degree
                if degree in self._echelon:
                    pivot, pivot_preimage = self._echelon[degree]
                    factor = F.quo(vector.LC, pivot.LC)
                    vector = vector - pivot.mul_ground(factor)
                    preimage = preimage - pivot_preimage.mul_ground(factor)
                else:
                    self._echelon[degree] = (vector, preimage)
                    break

    def _reduce_polynomial(self, P: KPoly) -> tuple[Expr, KPoly]:
        """Step 4: ``(g, r)`` with ``P = D(g) + r`` and ``r`` reduced modulo
        the image of the polynomials."""
        if P.is_zero:
            return _ZERO, P
        K = self.context
        F = K.field
        n_a = self._A.degree if not self._A.is_zero else -1
        n_b = self._B.degree
        delta = max(n_a, n_b - 1)
        size = P.degree - delta + 2
        if n_a == n_b - 1 and n_b >= 1:
            # E(x**j) has a lower degree for j = -(n_b + lc(A)/lc(B))
            mu = K.element_expr(F.quo(self._A.LC, self._B.LC)) + Integer(n_b)
            exceptional = _integer(-mu)
            if exceptional is not None and exceptional >= 0:
                size = max(size, exceptional + 2)
        self._extend_basis(max(size, 1))
        remainder = P
        preimage = K.constant(F.zero)
        # a pivot at degree d only changes the coefficients of degree at
        # most d: one descending pass leaves zeros at every pivot degree
        for degree in sorted(self._echelon, reverse=True):
            if remainder.is_zero or degree > remainder.degree:
                continue
            coefficient = remainder.coeff(degree)
            if F.is_zero(coefficient):
                continue
            pivot, pivot_preimage = self._echelon[degree]
            factor = F.quo(coefficient, pivot.LC)
            remainder = remainder - pivot.mul_ground(factor)
            preimage = preimage + pivot_preimage.mul_ground(factor)
        g = K.poly_expr(self._B)*K.poly_expr(preimage)
        return g, remainder

    def reduce(self, p: ExprLike) -> tuple[Expr, Expr]:
        """``(g, r)`` with ``p*S = D(g) + r`` for the normalised derivation
        (``S`` the shell) and ``r`` a reduced form: ``r == 0`` if and only
        if ``p*F`` has a hyperexponential antiderivative."""
        p_ = cancel(to_expr(p)*self.shell)
        g1, current = self._reduce_coprime_poles(p_)
        g2, current = self._reduce_b_poles(current)
        P, Q = self._numerator_denominator(current)
        polynomial, proper_numerator = P.div(Q)
        g3, reduced = self._reduce_polynomial(polynomial)
        K = self.context
        proper = K.fraction_expr(proper_numerator, Q)
        return cancel(g1 + g2 + g3), cancel(proper + K.poly_expr(reduced))


def hermite_reduce(p: ExprLike, q: ExprLike, x: Symbol, t: Optional[Symbol] = None) -> tuple[Expr, Expr]:
    """``(g, r)`` with ``p = g' + q*g + r`` and ``r`` a reduced form with
    respect to ``D = d/dx + q`` (``r == 0`` exactly when ``p*exp(Integral(q))``
    has an antiderivative of the form ``g*exp(Integral(q))``).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.integrals.reduction import hermite_reduce
    >>> x, t = symbols('x t')
    >>> hermite_reduce(x, -2*x, x)          # x exp(-x**2) = (-exp(-x**2)/2)'
    (-1/2, 0)
    >>> hermite_reduce(1, -2*x, x)          # exp(-x**2) is not integrable
    (0, 1)
    >>> hermite_reduce(1/(x**2 + t), 0, x, t)
    (0, 1/(t + x**2))
    """
    p_ = to_expr(p)
    t_ = t if t is not None else Symbol('t')
    reducer = HermiteReducer(q, x, t_, [p_])
    g, r = reducer.reduce(p_)
    S_ = reducer.shell
    return cancel(g/S_), cancel(r/S_)


def _dependency(vectors: list[list[Expr]]) -> Optional[list[Expr]]:
    """A nontrivial ``a`` with ``sum(a_j v_j) = 0`` and ``a[-1] != 0``."""
    width = max(len(v) for v in vectors)
    rows = [[v[i] if i < len(v) else _ZERO for v in vectors] for i in range(width)]
    basis = _nullspace(rows, len(vectors))
    for values in basis or []:
        if values[-1] != _ZERO:
            return values
    return None


def _coordinates(forms: Sequence[Expr], x: Symbol) -> Optional[list[list[Expr]]]:
    """The reduced forms over a common (monic) denominator, as coefficient
    lists (lowest degree first)."""
    try:
        K = _Context(list(forms), x)
        fractions = [K.fraction(r) for r in forms]
    except _POLYNOMIAL_FAILURES:
        return None
    common = fractions[0][1].monic()
    for _, Q in fractions[1:]:
        common = common.lcm(Q)
    vectors: list[list[Expr]] = []
    for P, Q in fractions:
        numerator = P*common.quo(Q)
        vectors.append([K.element_expr(c) for c in reversed(numerator.rep)])
    return vectors


class _Term:
    """A hyperexponential term with its reducer and the sequence
    ``p_j = partial_t^j F / F``."""

    def __init__(self, F: Expr, x: Symbol, t: Symbol) -> None:
        self.F: Final[Expr] = F
        self.t: Final[Symbol] = t
        q = logarithmic_derivative(F, x)
        p = logarithmic_derivative(F, t)
        if q is None or p is None:
            raise ValueError("not hyperexponential: %s" % F)
        self.p: Final[Expr] = p
        self.reducer: Final[HermiteReducer] = HermiteReducer(q, x, t, [p])
        self.ps: Final[list[Expr]] = []
        self.gs: Final[list[Expr]] = []
        self.rs: Final[list[Expr]] = []

    def extend(self) -> None:
        """The next ``p_j`` and its reduction."""
        if not self.ps:
            self.ps.append(_ONE)
        else:
            self.ps.append(cancel(diff(self.ps[-1], self.t) + self.p*self.ps[-1]))
        g, r = self.reducer.reduce(self.ps[-1])
        self.gs.append(g)
        self.rs.append(r)

    def certificate(self, relation: Sequence[Expr]) -> Expr:
        """``R`` with ``sum(a_j partial_t^j F) = partial_x(R F)`` for the
        term alone."""
        return cancel(add(*[mul(a, g) for a, g in zip(relation, self.gs)])/self.reducer.shell)


def _similar(u: Expr, v: Expr, x: Symbol, t: Symbol) -> Optional[Expr]:
    """``u/v`` when it is a rational function of ``x`` and ``t``."""
    ratio = cancel(u/v)
    if ratio.is_rational_function(x, t) is True:
        return ratio
    return None


def _grouped(terms: Sequence[Expr], x: Symbol, t: Symbol) -> list[Expr]:
    """The terms with the similar ones (rational ratio) added together."""
    groups: list[Expr] = []
    for term in terms:
        for i, group in enumerate(groups):
            ratio = _similar(term, group, x, t)
            if ratio is not None:
                groups[i] = cancel(group*(1 + ratio))
                break
        else:
            groups.append(term)
    return [g for g in groups if g != _ZERO]


def _telescope(terms: Sequence[Expr], x: Symbol, t: Symbol, max_order: int) -> Optional[tuple[list[Expr], Expr]]:
    """The coefficients and the certificate of the minimal common
    telescoper of non-similar hyperexponential terms: the first linear
    dependence of the concatenated reduced forms."""
    objects = [_Term(term, x, t) for term in terms]
    F = add(*terms)
    for j in range(max_order + 1):
        vectors: list[list[Expr]] = [[] for _ in range(j + 1)]
        for term in objects:
            term.extend()
            coordinates = _coordinates(term.rs, x)
            if coordinates is None:
                return None
            for i, v in enumerate(coordinates):
                vectors[i].extend(v)
        relation = _dependency(vectors)
        if relation is None:
            continue
        normalized, factor_ = _normalized(relation, t)
        if len(objects) == 1:
            certificate = objects[0].certificate(relation)
        else:
            certificate = cancel(add(*[term.certificate(relation)*term.F for term in objects])/F)
        return normalized, cancel(certificate*factor_)
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

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import exp
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
    F_ = to_expr(F)
    if is_hyperexponential(F_, x, t):
        groups = [F_]
    else:
        terms = _terms(F_, x, t)
        if not terms:
            return None
        groups = _grouped(terms, x, t)
        if not groups:
            return None
    try:
        result = _telescope(groups, x, t, max_order)
    except _POLYNOMIAL_FAILURES:
        return None
    if result is None:
        return None
    coefficients, certificate = result
    if len(groups) > 1:
        certificate = cancel(certificate*add(*groups)/F_)
    return DifferentialTelescoper(F_, x, t, coefficients, certificate)


def reduction_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The differential equation ``sum_j a_j(t) I^(j)(t) = [R F]_a^b`` of
    ``I(t) = Integral(F, (x, a, b))`` from the reduction-based telescoper.

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.reduction import reduction_ode
    >>> x, t = symbols('x t')
    >>> reduction_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    F_ = to_expr(F)
    telescoper = reduction_telescoper(F_, x, t)
    if telescoper is None:
        return None
    return _equation(telescoper.coefficients, telescoper.certificate*F_, x, to_expr(a), to_expr(b), t, assumptions,
                     name)


def reduction_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of ``t`` by the holonomic
    method with the reduction-based telescoper (see
    :func:`~sympy_extras.integrals.telescoping.holonomic_integral` for the
    solution of the equation and the initial values).

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.integrals.reduction import reduction_integral
    >>> from sympy_extras.settings import configure
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> with configure(numerical_checks=False):
    ...     reduction_integral(x*exp(-t*x**2), x, 0, oo, t)
    ConditionalValue(1/(2*t))
    """
    from .marichev import tidy
    F_ = to_expr(F)
    a_, b_ = to_expr(a), to_expr(b)
    telescoper = reduction_telescoper(F_, x, t)
    if telescoper is None:
        return None
    equation = _equation(telescoper.coefficients, telescoper.certificate*F_, x, a_, b_, t, assumptions, 'I')
    if equation is None:
        return None
    solution = _solve(telescoper.coefficients, equation, F_, x, a_, b_, t, assumptions)
    if solution is None:
        return None
    value = tidy(solution, assumptions)
    if value.has(nan, zoo, oo, -oo, Piecewise):
        return None
    if not _verified(value, F_, x, a_, b_, assumptions):
        return None
    return ConditionalValue(value)
