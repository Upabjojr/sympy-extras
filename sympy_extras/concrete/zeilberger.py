"""Zeilberger's algorithm (creative telescoping) for definite sums of
hypergeometric terms, and Wilf–Zeilberger certificates.

For a term `F(n, k)` hypergeometric in both `n` and `k`, the algorithm
finds polynomials `a_0(n), \\ldots, a_J(n)`, not all zero, and a rational
function `R(n, k)` with

.. math::

    \\sum_{j=0}^{J} a_j(n) F(n + j, k) = G(n, k + 1) - G(n, k),
    \\qquad G(n, k) = R(n, k) F(n, k):

the left-hand side is a hypergeometric term in `k` whose Gosper
representation is computed with the `a_j` as unknowns, so that Gosper's
equation becomes a linear system for the `a_j` and the coefficients of the
unknown polynomial (Zeilberger's *parametrised Gosper algorithm*). Summing
over `k` gives a linear recurrence with polynomial coefficients for the
definite sum `S(n) = \\sum_k F(n, k)`, which SymPy's ``rsolve`` solves in
closed form when a hypergeometric solution exists. With `J = 1`,
`a_0 = -1`, `a_1 = 1` (after dividing `F` by a conjectured closed form
`f(n)`), `R` is the Wilf–Zeilberger certificate proving
`\\sum_k F(n, k) = f(n)`.

SymPy has Gosper's algorithm (:func:`sympy.gosper_sum`) for indefinite
sums and evaluates definite hypergeometric sums through closed forms of
hypergeometric functions (:func:`sympy.summation`); it has no creative
telescoping, so sums such as `\\sum_k (-1)^k \\binom{2n}{k}^3` (Dixon) or
`\\sum_k \\binom{n}{k}^2 \\binom{n+k}{k}^2` (Apéry) get no recurrence.

References
==========

.. [PWZ] M. Petkovšek, H. Wilf, D. Zeilberger, A = B, A K Peters (1996),
   chapters 6 and 7.
.. [Koepf] W. Koepf, Hypergeometric Summation, Vieweg (1998), chapter 7.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.concrete.gosper import gosper_normal
from sympy.concrete.products import product, Product
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor, lcm_list
from sympy.series.limits import limit
from sympy.core.relational import Eq
from sympy.core.numbers import nan, zoo
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.simplify import simplify
from sympy.solvers.recurr import rsolve

from sympy_extras._typing import as_expr

__all__ = ['zeilberger', 'wz_certificate', 'wz_prove', 'zeilberger_sum', 'Telescoper']


class Telescoper:
    """The output of Zeilberger's algorithm for a term ``F(n, k)``.

    Attributes
    ==========

    coefficients : list of Expr
        The polynomials ``a_0(n), ..., a_J(n)`` of the recurrence.
    certificate : Expr
        The rational function ``R(n, k)`` with ``G = R*F``.
    term, n, k
        The input.
    """

    def __init__(self, term: Expr, n: Symbol, k: Symbol, coefficients: list[Expr], certificate: Expr) -> None:
        self.term = term
        self.n = n
        self.k = k
        self.coefficients = coefficients
        self.certificate = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        """The recurrence ``sum_j a_j(n) S(n + j)`` satisfied by the sum
        over ``k`` (with natural boundaries), as an expression in the
        function ``S(n)``."""
        f = Function(name)
        return as_expr(Add(*[a*f(self.n + j) for j, a in enumerate(self.coefficients)]))

    def check(self) -> bool:
        """Verify ``sum_j a_j F(n + j, k) = G(n, k + 1) - G(n, k)`` by
        simplification."""
        n, k, F = self.n, self.k, self.term
        lhs = Add(*[a*F.subs(n, n + j) for j, a in enumerate(self.coefficients)])
        G = self.certificate*F
        difference = combsimp(as_expr((lhs - G.subs(k, k + 1) + G)/F))
        return simplify(difference) == 0

    def __repr__(self) -> str:
        return "Telescoper(%s, %s)" % (self.coefficients, self.certificate)


def _rational_ratio(f: Expr, g: Expr, k: Symbol) -> Optional[Expr]:
    """``f/g`` as a rational function of ``k``, or ``None``."""
    ratio = combsimp(as_expr(f/g))
    ratio = as_expr(cancel(ratio))
    if not ratio.is_rational_function(k):
        # a second attempt through the hypergeometric simplifier
        try:
            ratio = as_expr(cancel(combsimp(ratio.rewrite('gamma'))))
        except (ValueError, TypeError, AttributeError):
            return None
        if not ratio.is_rational_function(k):
            return None
    return ratio


def _gosper_degree_bound(A: Poly, B_shifted: Poly, degree_c: int) -> int:
    """The bound of Gosper's algorithm on the degree of the polynomial
    solution ``f`` of ``A(k) f(k+1) - B(k-1) f(k) = c(k)``."""
    da, db = A.degree(), B_shifted.degree()
    if da != db or cancel(as_expr(A.LC() - B_shifted.LC())) != 0:
        return degree_c - max(da, db)
    m = da
    candidates = [degree_c - m + 1]
    if m >= 1:
        d0 = cancel(as_expr((B_shifted.coeff_monomial(A.gen**(m - 1)) - A.coeff_monomial(A.gen**(m - 1)))/A.LC()))
        if isinstance(d0, Integer) and int(d0) >= 0:
            candidates.append(int(d0))
    return max(candidates)


def zeilberger(term: Expr, n: Symbol, k: Symbol, max_order: int = 4, min_order: int = 0) -> Optional[Telescoper]:
    """Zeilberger's algorithm: a recurrence in ``n`` for the sum over ``k``
    of a term hypergeometric in ``n`` and ``k``, with its certificate. The
    orders ``min_order`` to ``max_order`` are tried in turn; ``None`` when
    no recurrence of those orders exists.

    Examples
    ========

    >>> from sympy import binomial, symbols
    >>> from sympy_extras.concrete.zeilberger import zeilberger
    >>> n, k = symbols('n k', integer=True)
    >>> Z = zeilberger(binomial(n, k), n, k)
    >>> Z.coefficients, Z.certificate
    ([-2, 1], k/(k - n - 1))
    >>> Z.recurrence()
    -2*S(n) + S(n + 1)
    >>> zeilberger(binomial(n, k)**2, n, k).coefficients
    [-2*(2*n + 1), n + 1]
    """
    F = sympify(term)
    if not isinstance(F, Expr):
        raise TypeError("an expression is expected, got %s" % (F,))
    r = _rational_ratio(F.subs(k, k + 1), F, k)
    if r is None:
        raise ValueError("%s is not hypergeometric in %s" % (F, k))
    ratios: list[Expr] = []
    for j in range(max_order + 1):
        s = _rational_ratio(F.subs(n, n + j), F, k)
        if s is None:
            raise ValueError("%s is not hypergeometric in %s" % (F, n))
        ratios.append(s)
    for J in range(min_order, max_order + 1):
        result = _telescope(F, n, k, r, ratios[:J + 1])
        if result is not None:
            return result
    return None


def _telescope(F: Expr, n: Symbol, k: Symbol, r: Expr, ratios: list[Expr]) -> Optional[Telescoper]:
    J = len(ratios) - 1
    a = [Dummy('a%d' % j) for j in range(J + 1)]
    # sum_j a_j F(n+j, k) = F(n, k) P(k)/D(k)
    numerators = [as_expr(s.as_numer_denom()[0]) for s in ratios]
    denominators = [as_expr(s.as_numer_denom()[1]) for s in ratios]
    D = as_expr(lcm_list(denominators))
    P = Add(*[a[j]*cancel(numerators[j]*D/denominators[j]) for j in range(J + 1)])
    try:
        P_poly = Poly(as_expr(P), k)
    except PolynomialError:
        return None
    # the ratio of the term without P: r(k) D(k)/D(k+1)
    rho = as_expr(cancel(r*D/D.subs(k, k + 1)))
    num, den = rho.as_numer_denom()
    try:
        A, B, C = gosper_normal(as_expr(num), as_expr(den), k, polys=True)
    except (PolynomialError, ValueError, TypeError):
        return None
    A_ = Poly(as_expr(A.as_expr()), k)
    B_shift = Poly(as_expr(B.as_expr().subs(k, k - 1)), k)
    degree_c = C.degree() + P_poly.degree()
    d = _gosper_degree_bound(A_, B_shift, degree_c)
    if d < 0:
        return None
    f_coefficients = [Dummy('f%d' % i) for i in range(d + 1)]
    f = Add(*[c*k**i for i, c in enumerate(f_coefficients)])
    equation = as_expr(A_.as_expr()*f.subs(k, k + 1) - B_shift.as_expr()*f - C.as_expr()*P)
    unknowns: list[Expr] = [*f_coefficients, *a]
    try:
        rows = Poly(equation, k).all_coeffs()
    except PolynomialError:
        return None
    M = Matrix([[as_expr(row).coeff(u) for u in unknowns] for row in rows])
    M = M.applyfunc(lambda e: cancel(e))
    null = M.nullspace()
    for vector in null:
        values = _normalized([as_expr(cancel(v)) for v in vector], len(f_coefficients), n)
        coefficients = values[len(f_coefficients):]
        if all(c == 0 for c in coefficients):
            continue
        f_value = as_expr(Add(*[c*k**i for i, c in enumerate(values[:len(f_coefficients)])]))
        certificate = as_expr(factor(cancel(B_shift.as_expr()*f_value/(C.as_expr()*D))))
        coefficients = [as_expr(factor(c)) for c in coefficients]
        return Telescoper(F, n, k, coefficients, certificate)
    return None


def _normalized(values: list[Expr], nf: int, n: Symbol) -> list[Expr]:
    """The solution scaled so that the recurrence coefficients are
    polynomials in ``n`` without a common factor."""
    from sympy.polys.polytools import gcd_list
    coefficients = values[nf:]
    denominators = [as_expr(c.as_numer_denom()[1]) for c in coefficients if c != 0]
    if not denominators:
        return values
    scale = as_expr(lcm_list(denominators))
    scaled = [as_expr(cancel(v*scale)) for v in values]
    numerators = [c for c in scaled[nf:] if c != 0]
    if numerators and all(c.is_polynomial(n) for c in numerators):
        common = as_expr(gcd_list(numerators))
        if common != 0 and common != 1:
            scaled = [as_expr(cancel(v/common)) for v in scaled]
    return scaled


def wz_certificate(term: Expr, closed_form: Expr, n: Symbol, k: Symbol) -> Optional[Expr]:
    """The Wilf–Zeilberger certificate ``R(n, k)`` proving
    ``sum_k term(n, k) == closed_form(n)``: with ``F = term/closed_form``,
    ``F(n + 1, k) - F(n, k) = G(n, k + 1) - G(n, k)`` for ``G = R F``.
    ``None`` when there is no such rational function (the identity may
    still hold, but not in a WZ-provable form).

    Examples
    ========

    >>> from sympy import binomial, symbols
    >>> from sympy_extras.concrete.zeilberger import wz_certificate
    >>> n, k = symbols('n k', integer=True)
    >>> wz_certificate(binomial(n, k), 2**n, n, k)
    k/(2*(k - n - 1))
    """
    F = as_expr(sympify(term)/sympify(closed_form))
    Z = zeilberger(F, n, k, max_order=1, min_order=1)
    if Z is None:
        return None
    a0, a1 = Z.coefficients
    ratio = cancel(as_expr(a0/a1))
    if ratio != -1:
        return None
    return as_expr(factor(cancel(Z.certificate/a1)))


def wz_prove(term: Expr, closed_form: Expr, n: Symbol, k: Symbol, n0: int = 0) -> Optional[bool]:
    """Prove ``sum_k term(n, k) == closed_form(n)`` for ``n >= n0`` with a
    WZ certificate (the sum over all integers ``k``, the term vanishing
    outside a finite range): ``True`` when a certificate exists and the
    identity holds at ``n0``, ``None`` when no certificate is found.

    Examples
    ========

    >>> from sympy import binomial, symbols
    >>> from sympy_extras.concrete.zeilberger import wz_prove
    >>> n, k = symbols('n k', integer=True)
    >>> wz_prove(binomial(n, k), 2**n, n, k)
    True
    >>> wz_prove(binomial(n, k)**2, binomial(2*n, n), n, k)
    True
    """
    R = wz_certificate(term, closed_form, n, k)
    if R is None:
        return None
    F = as_expr(sympify(term)/sympify(closed_form))
    G = R*F
    identity = simplify(combsimp(as_expr((F.subs(n, n + 1) - F - G.subs(k, k + 1) + G)/F)))
    if identity != 0:
        return None
    # the base case, summed over the support
    base = _finite_sum(sympify(term), k, n, n0)
    if base is None:
        return None
    return bool(simplify(as_expr(base - sympify(closed_form).subs(n, n0))) == 0)


def _finite_sum(term: Expr, k: Symbol, n: Symbol, value: int) -> Optional[Expr]:
    """``sum_k term(value, k)`` over the support of the term, found by
    expanding from ``k = 0`` in both directions while the terms are
    nonzero (at most a few hundred terms)."""
    F = term.subs(n, value)
    total: Expr = S.Zero
    for direction in (1, -1):
        i = 0 if direction == 1 else -1
        zeros = 0
        while zeros < 3 and abs(i) < 400:
            t = as_expr(F.subs(k, i))
            if t.has(k) or not t.is_number:
                return None
            if t == 0:
                zeros += 1
            else:
                zeros = 0
                total = as_expr(total + t)
            i += direction
    return total


def zeilberger_sum(term: Expr, limits: Sequence[Union[Symbol, Expr, int]], n: Optional[Symbol] = None,
                   max_order: int = 4) -> Optional[Basic]:
    """A closed form of the definite sum ``sum_{k=lo}^{hi} term`` with
    ``lo`` an integer and ``hi`` either ``n`` (plus an integer) or
    ``oo``, through the recurrence of Zeilberger's algorithm (with the
    boundary terms of the telescoping) solved by ``rsolve`` with the
    initial values computed directly; the recurrence itself (an equation
    in ``S(n)``) when ``rsolve`` finds no closed form; ``None`` when there
    is no recurrence.

    Examples
    ========

    >>> from sympy import binomial, symbols, oo
    >>> from sympy_extras.concrete.zeilberger import zeilberger_sum
    >>> n, k = symbols('n k', integer=True)
    >>> zeilberger_sum(binomial(n, k), (k, 0, n))
    2**n
    >>> zeilberger_sum(binomial(n, k)**2, (k, 0, n))
    binomial(2*n, n)
    >>> zeilberger_sum((-1)**k*binomial(2*n, k)**3, (k, 0, 2*n))
    (-1)**n*factorial(3*n)/factorial(n)**3
    >>> zeilberger_sum(binomial(n, k)**2*binomial(n + k, k)**2, (k, 0, n))
    Eq((n + 1)**3*S(n) + (n + 2)**3*S(n + 2) - (2*n + 3)*(17*n**2 + 51*n + 39)*S(n + 1), 0)
    >>> zeilberger_sum(1/(k*(k + 1)), (k, 1, n))
    n/(n + 1)
    """
    k_, lo_, hi_ = limits
    k = k_ if isinstance(k_, Symbol) else Symbol(str(k_))
    lo, hi = sympify(lo_), sympify(hi_)
    F = sympify(term)
    if not isinstance(F, Expr):
        raise TypeError("an expression is expected")
    if n is None:
        candidates = sorted((s for s in F.free_symbols if s != k and isinstance(s, Symbol)), key=lambda s: s.name)
        inside = [s for s in candidates if hi.has(s)] or candidates
        if not inside and isinstance(hi, Symbol):
            inside = [hi]
        if len(inside) != 1:
            raise ValueError("the summation variable n must be given")
        n = inside[0]
    Z = zeilberger(F, n, k, max_order=max_order)
    if Z is None:
        return None
    J = Z.order
    # G in a form which is finite where the term vanishes (the certificate
    # may have a pole there)
    G = as_expr(gammasimp(combsimp(as_expr(Z.certificate*F))))
    Sfun = Function('S')
    # the telescoping identity summed over k from lo to K = hi(n + J),
    # beyond the range of every S(n + j):
    # sum_j a_j S(n+j) = G(n, K+1) - G(n, lo) - sum_j a_j sum_{k=hi(n+j)+1}^{K} F(n+j, k)
    if hi == S.Infinity:
        top = limit(G, k, S.Infinity)
        if not isinstance(top, Expr) or top.has(k):
            return None
        rhs: Expr = as_expr(top - _value(G, k, lo))
    else:
        K = as_expr(hi.subs(n, n + J))
        rhs = as_expr(_value(G, k, as_expr(K + 1)) - _value(G, k, lo))
        for j, a in enumerate(Z.coefficients):
            upper = as_expr(hi.subs(n, n + j))
            count = as_expr(simplify(K - upper))
            if not (isinstance(count, Integer) and int(count) >= 0):
                return None
            for i in range(1, int(count) + 1):
                rhs = as_expr(rhs - a*F.subs(n, n + j).subs(k, upper + i))
    rhs = as_expr(simplify(combsimp(rhs)))
    recurrence = as_expr(Add(*[a*Sfun(n + j) for j, a in enumerate(Z.coefficients)]) - rhs)
    # initial values, from the first n at which the leading coefficient
    # of the recurrence does not vanish any more
    initial: dict[Basic, Basic] = {}
    n0 = _start(Z.coefficients[-1], n)
    for i in range(J):
        value = Sum(F.subs(n, n0 + i), (k, lo, hi.subs(n, n0 + i))).doit()
        if isinstance(value, Sum) or (hi == S.Infinity and not value.is_number):
            return _Recurrence(recurrence)
        initial[Sfun(n0 + i)] = value
    solution: Optional[Expr] = None
    if J == 0:
        # the term is Gosper-summable: a_0 S(n) = rhs
        solution = as_expr(cancel(rhs/Z.coefficients[0]))
    elif J == 1 and rhs == 0 and initial:
        # a first order homogeneous recurrence: a product of its ratio
        # (SymPy's rsolve misses some of these)
        a0, a1 = Z.coefficients
        idx = Dummy('i', integer=True)
        ratio = as_expr(cancel(-a0/a1)).subs(n, idx)
        prod = product(ratio, (idx, n0, n - 1))
        if isinstance(prod, Expr) and not isinstance(prod, Product):
            solution = as_expr(initial[Sfun(n0)]*prod)
    if solution is None:
        try:
            solved = rsolve(recurrence, Sfun(n), initial if initial else None)
        except (NotImplementedError, ValueError, TypeError, AttributeError):
            solved = None
        if solved is not None and solved != 0:
            solution = as_expr(solved)
    if solution is None:
        return _Recurrence(recurrence)
    solution = _nicer(as_expr(solution), n)
    # the closed form is checked at a few values
    for value in range(n0, n0 + J + 3):
        direct = Sum(F.subs(n, value), (k, lo, hi.subs(n, value))).doit()
        if isinstance(direct, Sum):
            break
        if simplify(as_expr(direct - solution.subs(n, value))) != 0:
            # an order-zero recurrence a0 S(n) = rhs would only re-assert
            # the answer the check has just rejected (sympy-extras#33)
            return None if J == 0 else _Recurrence(recurrence)
    return solution


def _nicer(solution: Expr, n: Symbol) -> Expr:
    """The closed form simplified, the gamma functions of half-integer
    arguments removed with the duplication formula and factorials or
    binomials preferred when smaller."""
    from sympy.core.function import count_ops
    from sympy.core.numbers import Rational, pi
    from sympy.functions.combinatorial.factorials import factorial, binomial
    from sympy.functions.special.gamma_functions import gamma

    def duplicate(e: Expr) -> Expr:
        def rule(g: Basic) -> Basic:
            z = as_expr(g.args[0])
            half = z - Rational(1, 2)
            if half.is_integer or (half.is_polynomial(n) and Poly(half, n).all_coeffs()[-1].is_integer and half.is_integer is not False):
                return gamma(2*half)*sqrt(pi)*2**(1 - 2*half)/gamma(half)
            return g
        return as_expr(e.replace(lambda g: isinstance(g, gamma), rule))

    from sympy.functions.elementary.miscellaneous import sqrt

    def triplicate(e: Expr) -> Expr:
        # gamma(z + 1/3) gamma(z + 2/3) = 2 pi 3**(1/2 - 3z) gamma(3z)/gamma(z)
        gammas = [as_expr(g.args[0]) for g in e.atoms(gamma)]
        for a in gammas:
            for b in gammas:
                if as_expr(b - a) == Rational(1, 3) and (a - Rational(1, 3)).is_polynomial(n):
                    z = as_expr(a - Rational(1, 3))
                    formula = 2*pi*3**(Rational(1, 2) - 3*z)*gamma(3*z)/(gamma(z)*gamma(b))
                    return as_expr(e.xreplace({gamma(a): formula}))
        return e

    candidates = [as_expr(simplify(combsimp(solution)))]
    candidates.append(as_expr(combsimp(duplicate(candidates[0]))))
    candidates.append(as_expr(combsimp(triplicate(candidates[-1]))))
    candidates.append(as_expr(combsimp(candidates[-1].rewrite(factorial))))
    candidates.append(as_expr(candidates[-1].rewrite(binomial)))
    candidates.append(as_expr(simplify(candidates[-1])))
    return min(candidates, key=lambda e: (count_ops(e), len(str(e))))


def _start(leading: Expr, n: Symbol) -> int:
    """The first nonnegative integer from which the leading coefficient
    has no more integer zeros."""
    from sympy.polys.polytools import real_roots
    if not leading.is_polynomial(n) or not leading.has(n):
        return 0
    start = 0
    for root in real_roots(Poly(leading, n)):
        if isinstance(root, Integer) and int(root) >= start:
            start = int(root) + 1
    return start


def _value(G: Expr, k: Symbol, at: Expr) -> Expr:
    """``G`` at ``k = at``, by a limit when the substitution is not
    defined."""
    value = as_expr(G.subs(k, at))
    if value.has(S.NaN, S.ComplexInfinity):
        value = as_expr(limit(G, k, at))
    return as_expr(simplify(value))


def _Recurrence(recurrence: Expr) -> Optional[Basic]:
    """The recurrence as an unevaluated equation, or ``None`` when it is
    not one. ``Eq`` must not be left to evaluate: a recurrence reducing to
    a nonzero constant would become the Boolean ``False`` and be returned
    as the sum, and one that is ``nan`` says nothing (sympy-extras#33)."""
    if recurrence.has(nan, zoo):
        return None
    return Eq(recurrence, 0, evaluate=False)
