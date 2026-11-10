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
definite sum `S(n) = \\sum_k F(n, k)`, which
:func:`~sympy_extras.solvers.recurr.rsolve_linear` solves in closed form
when a hypergeometric or d'Alembertian solution exists. With `J = 1`,
`a_0 = -1`, `a_1 = 1` (after dividing `F` by a conjectured closed form
`f(n)`), `R` is the Wilf–Zeilberger certificate proving
`\\sum_k F(n, k) = f(n)`.

SymPy has Gosper's algorithm (:func:`sympy.gosper_sum`) for indefinite
sums and evaluates definite hypergeometric sums through closed forms of
hypergeometric functions (:func:`sympy.summation`); it has no creative
telescoping, so sums such as `\\sum_k (-1)^k \\binom{2n}{k}^3` (Dixon) or
`\\sum_k \\binom{n}{k}^2 \\binom{n+k}{k}^2` (Apéry) get no recurrence.

The module works on the expressions of the core
(:mod:`sympy_extras.core`: the native functions, polynomials,
simplifications, sums, products and limits, the native recurrence solver)
and takes and returns them; it imports no SymPy. Only the isolated values
of the parameters go through SymPy at the call, for summands with
parameters other than the limit
(:func:`~sympy_extras.concrete.karr._with_special_values`).
:mod:`~sympy_extras.concrete.zeilberger_sympy` is its face on SymPy
expressions.

References
==========

.. [PWZ] M. Petkovšek, H. Wilf, D. Zeilberger, A = B, A K Peters (1996),
   chapters 6 and 7.
.. [Koepf] W. Koepf, Hypergeometric Summation, Vieweg (1998), chapter 7.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Optional, Sequence, Union

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import AppliedUndef, Function, count_ops, diff
from sympy_extras.core.limits import limit
from sympy_extras.core.numbers import ComplexInfinity, Infinity, Integer, NaN, Rational, pi
from sympy_extras.core.power import sqrt
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import Binomial, Factorial
from sympy_extras.functions.special.gamma_functions import Gamma, gamma
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement, SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel, factor, factor_list, gcd_list, lcm_list
from sympy_extras.simplify.gammasimp import combsimp, gammasimp
from sympy_extras.simplify.simplify import simplify
from sympy_extras.solvers.recurr import _row_reduce, rsolve_linear

from .expr_with_limits import Product, Sum
from .gosper import _UPoly, gosper_normal
from .karr import SumPoint, _with_special_values, summation
from .products import product

__all__ = ['zeilberger', 'wz_certificate', 'wz_prove', 'zeilberger_sum', 'Telescoper']

ZERO: Final[Integer] = Integer(0)
ONE: Final[Integer] = Integer(1)
HALF: Final[Rational] = Rational(1, 2)
THIRD: Final[Rational] = Rational(1, 3)

#: the errors of the native algebra on input it does not handle
_Failures = (ValueError, TypeError, NotImplementedError, AttributeError, BasePolynomialError)


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
        self.term: Final[Expr] = term
        self.n: Final[Symbol] = n
        self.k: Final[Symbol] = k
        self.coefficients: Final[list[Expr]] = coefficients
        self.certificate: Final[Expr] = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        """The recurrence ``sum_j a_j(n) S(n + j)`` satisfied by the sum
        over ``k`` (with natural boundaries), as an expression in the
        function ``S(n)``."""
        f = Function(name)
        return add(*[a*f(self.n + j) for j, a in enumerate(self.coefficients)])

    def check(self) -> bool:
        """Verify ``sum_j a_j F(n + j, k) = G(n, k + 1) - G(n, k)`` by
        simplification."""
        n, k, F = self.n, self.k, self.term
        lhs = add(*[a*F.subs(n, n + j) for j, a in enumerate(self.coefficients)])
        G = self.certificate*F
        difference = combsimp((lhs - G.subs(k, k + 1) + G)/F)
        return simplify(difference) == ZERO

    def __repr__(self) -> str:
        return "Telescoper(%s, %s)" % (self.coefficients, self.certificate)


def _rational_ratio(f: Expr, g: Expr, k: Symbol) -> Optional[Expr]:
    """``f/g`` as a rational function of ``k``, or ``None``."""
    ratio = cancel(combsimp(f/g))
    if ratio.is_rational_function(k) is not True:
        # a second attempt through the hypergeometric simplifier
        try:
            ratio = cancel(combsimp(ratio.rewrite(Gamma)))
        except _Failures:
            return None
        if ratio.is_rational_function(k) is not True:
            return None
    return ratio


def _gosper_degree_bound(A: _UPoly, B_shifted: _UPoly, degree_c: int) -> int:
    """The bound of Gosper's algorithm on the degree of the polynomial
    solution ``f`` of ``A(k) f(k+1) - B(k-1) f(k) = c(k)``."""
    da, db = A.degree(), B_shifted.degree()
    if da != db or cancel(A.lc() - B_shifted.lc()) != ZERO:
        return degree_c - max(da, db)
    m = da
    candidates = [degree_c - m + 1]
    if m >= 1:
        d0 = cancel((B_shifted.nth(m - 1) - A.nth(m - 1))/A.lc())
        if isinstance(d0, Integer) and d0.p >= 0:
            candidates.append(d0.p)
    return max(candidates)


def zeilberger(term: ExprLike, n: Symbol, k: Symbol, max_order: int = 4, min_order: int = 0) -> Optional[Telescoper]:
    """Zeilberger's algorithm: a recurrence in ``n`` for the sum over ``k``
    of a term hypergeometric in ``n`` and ``k``, with its certificate. The
    orders ``min_order`` to ``max_order`` are tried in turn; ``None`` when
    no recurrence of those orders exists.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial
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
    F = to_expr(term)
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
    a: list[Symbol] = [Dummy('a%d' % j) for j in range(J + 1)]
    # sum_j a_j F(n+j, k) = F(n, k) P(k)/D(k)
    numerators = [s.as_numer_denom()[0] for s in ratios]
    denominators = [s.as_numer_denom()[1] for s in ratios]
    D = lcm_list(denominators)
    P = add(*[a[j]*cancel(numerators[j]*D/denominators[j]) for j in range(J + 1)])
    try:
        P_poly = _UPoly.of(P, k)
    except ValueError:
        return None
    # the ratio of the term without P: r(k) D(k)/D(k+1)
    rho = cancel(r*D/D.subs(k, k + 1))
    num, den = rho.as_numer_denom()
    try:
        A, B, C = gosper_normal(num, den, k)
    except _Failures:
        return None
    B_shift = B.shift(Integer(-1))
    degree_c = C.degree() + P_poly.degree()
    d = _gosper_degree_bound(A, B_shift, degree_c)
    # d < 0 leaves f = 0: the shifts of the term in n may still be
    # linearly dependent over K(n) (sum_j a_j F(n+j, k) = 0, certificate 0),
    # as for F = n! (-1)**k/k!, whose sum satisfies S(n+1) - (n+1) S(n) =
    # (-1)**(n+1) through the boundary term alone
    f_coefficients: list[Symbol] = [Dummy('f%d' % i) for i in range(max(d, -1) + 1)]
    f = add(*[c*k**i for i, c in enumerate(f_coefficients)])
    equation = A.as_expr(k)*f.subs(k, k + 1) - B_shift.as_expr(k)*f - C.as_expr(k)*P
    unknowns: list[Symbol] = [*f_coefficients, *a]
    try:
        rows = _UPoly.of(equation, k).c
    except ValueError:
        return None
    matrix = [[cancel(diff(row, u)) for u in unknowns] for row in rows]
    for vector in _nullspace(matrix, len(unknowns)):
        values = _normalized([cancel(v) for v in vector], len(f_coefficients), n)
        coefficients = values[len(f_coefficients):]
        if all(c == ZERO for c in coefficients):
            continue
        f_value = add(*[c*k**i for i, c in enumerate(values[:len(f_coefficients)])])
        certificate = factor(cancel(B_shift.as_expr(k)*f_value/(C.as_expr(k)*D)))
        return Telescoper(F, n, k, [factor(c) for c in coefficients], certificate)
    return None


def _nullspace(rows: list[list[Expr]], columns: int) -> list[list[Expr]]:
    """The basis of the null space of a matrix of rational functions read
    from its reduced row echelon form, the free coordinate one (SymPy's
    ``Matrix.nullspace``), over the rational functions of its symbols;
    over the expressions (:func:`~sympy_extras.solvers.recurr._row_reduce`)
    when an entry is not a rational function."""
    symbols = sorted({s for row in rows for e in row for s in e.free_symbols if isinstance(s, Symbol)},
                     key=lambda s: s.sort_key())
    names = ['_zn%d' % i for i in range(len(symbols))] or ['_zn']
    renamed: dict[Expr, Expr] = {s: Symbol(name) for s, name in zip(symbols, names)}
    restored: dict[Expr, Expr] = {Symbol(name): s for s, name in zip(symbols, names)}
    field = SparseFractionField(RationalPolynomialRing(names))
    try:
        converted: list[list[SparseFracElement[Fraction]]] = [[field.from_expr(e.xreplace(renamed)) for e in row]
                                                              for row in rows]
    except _Failures:
        return _expression_nullspace(rows, columns)
    return [[field.to_expr(v).xreplace(restored) for v in vector] for vector in field.nullspace(converted, columns)]


def _expression_nullspace(rows: list[list[Expr]], columns: int) -> list[list[Expr]]:
    """:func:`_nullspace` over the expressions."""
    reduced, pivots = _row_reduce(rows)
    basis: list[list[Expr]] = []
    for free in range(columns):
        if free in pivots:
            continue
        vector: list[Expr] = [ZERO]*columns
        vector[free] = ONE
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row][free]
        basis.append(vector)
    return basis


def _normalized(values: list[Expr], nf: int, n: Symbol) -> list[Expr]:
    """The solution scaled so that the recurrence coefficients are
    polynomials in ``n`` without a common factor."""
    coefficients = values[nf:]
    denominators = [c.as_numer_denom()[1] for c in coefficients if c != ZERO]
    if not denominators:
        return values
    scale = lcm_list(denominators)
    scaled = [cancel(v*scale) for v in values]
    numerators = [c for c in scaled[nf:] if c != ZERO]
    if numerators and all(c.is_polynomial(n) is True for c in numerators):
        common = gcd_list(numerators)
        if common != ZERO and common != ONE:
            scaled = [cancel(v/common) for v in scaled]
    return scaled


def wz_certificate(term: ExprLike, closed_form: ExprLike, n: Symbol, k: Symbol) -> Optional[Expr]:
    """The Wilf–Zeilberger certificate ``R(n, k)`` proving
    ``sum_k term(n, k) == closed_form(n)``: with ``F = term/closed_form``,
    ``F(n + 1, k) - F(n, k) = G(n, k + 1) - G(n, k)`` for ``G = R F``.
    ``None`` when there is no such rational function (the identity may
    still hold, but not in a WZ-provable form).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial
    >>> from sympy_extras.concrete.zeilberger import wz_certificate
    >>> n, k = symbols('n k', integer=True)
    >>> wz_certificate(binomial(n, k), 2**n, n, k)
    k/(2*(k - n - 1))
    """
    F = to_expr(term)/to_expr(closed_form)
    Z = zeilberger(F, n, k, max_order=1, min_order=1)
    if Z is None:
        return None
    a0, a1 = Z.coefficients
    if cancel(a0/a1) != Integer(-1):
        return None
    return factor(cancel(Z.certificate/a1))


def wz_prove(term: ExprLike, closed_form: ExprLike, n: Symbol, k: Symbol, n0: int = 0) -> Optional[bool]:
    """Prove ``sum_k term(n, k) == closed_form(n)`` for ``n >= n0`` with a
    WZ certificate (the sum over all integers ``k``, the term vanishing
    outside a finite range): ``True`` when a certificate exists and the
    identity holds at ``n0``, ``None`` when no certificate is found.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import binomial
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
    term_, closed = to_expr(term), to_expr(closed_form)
    F = term_/closed
    G = R*F
    identity = simplify(combsimp((F.subs(n, n + 1) - F - G.subs(k, k + 1) + G)/F))
    if identity != ZERO:
        return None
    # the base case, summed over the support
    base = _finite_sum(term_, k, n, n0)
    if base is None:
        return None
    return bool(simplify(base - closed.subs(n, Integer(n0))) == ZERO)


def _finite_sum(term: Expr, k: Symbol, n: Symbol, value: int) -> Optional[Expr]:
    """``sum_k term(value, k)`` over the support of the term, found by
    expanding from ``k = 0`` in both directions while the terms are
    nonzero (at most a few hundred terms)."""
    F = term.subs(n, Integer(value))
    total: Expr = ZERO
    for direction in (1, -1):
        i = 0 if direction == 1 else -1
        zeros = 0
        while zeros < 3 and abs(i) < 400:
            t = F.subs(k, Integer(i))
            if t.has(k) or not t.is_number:
                return None
            if t == ZERO:
                zeros += 1
            else:
                zeros = 0
                total = total + t
            i += direction
    return total


def zeilberger_sum(term: ExprLike, limits: Sequence[ExprLike], n: Optional[Symbol] = None,
                   max_order: int = 4, special_values: bool = True) -> Optional[Basic]:
    """A closed form of the definite sum ``sum_{k=lo}^{hi} term`` with
    ``lo`` an integer and ``hi`` either ``n`` (plus an integer) or
    ``oo``, through the recurrence of Zeilberger's algorithm (with the
    boundary terms of the telescoping) solved by
    :func:`~sympy_extras.solvers.recurr.rsolve_linear` (hypergeometric
    and d'Alembertian solutions) with the initial values computed
    directly; the recurrence itself (an equation in ``S(n)``) when it
    finds no closed form; ``None`` when there is no recurrence.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import binomial
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
    >>> zeilberger_sum(1/k, (k, 1, n))
    harmonic(n)

    With ``special_values`` (the default) a closed form gets a case for
    each isolated value of the parameters of the summand (other than
    ``n``) at which it is undefined, the sum computed again there
    (:func:`~sympy_extras.concrete.karr_sympy.with_sum_special_values`); a
    recurrence gets none, being an identity in the parameters:

    >>> y = symbols('y')
    >>> zeilberger_sum(y**k, (k, 0, n))
    Piecewise((n + 1, Eq(y, 1)), ((y**(n + 1) - 1)/(y - 1), True))
    """
    if len(limits) != 3:
        raise ValueError("the limits of a sum are (k, lo, hi), not %s" % (limits,))
    k_, lo_, hi_ = limits
    k = k_ if isinstance(k_, Symbol) else Symbol(str(k_))
    lo, hi = to_expr(lo_), to_expr(hi_)
    F = to_expr(term)
    if n is None:
        candidates = sorted((s for s in F.free_symbols if s != k and isinstance(s, Symbol)), key=lambda s: s.name)
        inside = [s for s in candidates if hi.has(s)]
        if not inside:
            # the upper limit itself when it is a symbol (the sum of y**k
            # up to n is a sum in n, not in the parameter y)
            inside = [hi] if isinstance(hi, Symbol) and hi != k else candidates
        if len(inside) != 1:
            raise ValueError("the summation variable n must be given")
        n = inside[0]
    solution = _zeilberger_sum(F, k, lo, hi, n, max_order)
    if not special_values or not isinstance(solution, Expr):
        return solution
    n_ = n

    def at_point(point: SumPoint) -> Optional[Expr]:
        return _sum_at_point(F.xreplace(point), k, lo, hi, n_, max_order)

    return _with_special_values(solution, F, [k, n], [lo, hi], at_point)


def _sum_at_point(F: Expr, k: Symbol, lo: Expr, hi: Expr, n: Symbol, max_order: int) -> Optional[Expr]:
    """The sum at a point of the parameters: by :func:`zeilberger_sum`
    (with the cases of the parameters left), else by
    :func:`~sympy_extras.concrete.karr.summation` when Zeilberger's
    algorithm gives no closed form there."""
    try:
        found = zeilberger_sum(F, (k, lo, hi), n, max_order)
    except (ValueError, NotImplementedError):
        found = None
    if isinstance(found, Expr):
        return found
    return summation(F, (k, lo, hi))


def _direct_sum(F: Expr, k: Symbol, lo: Expr, hi: Expr) -> Expr:
    """``Sum(F, (k, lo, hi))`` evaluated (left as a sum where it is not)."""
    return Sum(F, (k, lo, hi)).doit()


def _zeilberger_sum(F: Expr, k: Symbol, lo: Expr, hi: Expr, n: Symbol, max_order: int) -> Optional[Basic]:
    """:func:`zeilberger_sum` without the cases of the parameters."""
    Z = zeilberger(F, n, k, max_order=max_order)
    if Z is None:
        return None
    J = Z.order
    # G in a form which is finite where the term vanishes (the certificate
    # may have a pole there)
    G = gammasimp(combsimp(Z.certificate*F))
    Sfun = Function('S')
    # the telescoping identity summed over k from lo to K = hi(n + J),
    # beyond the range of every S(n + j):
    # sum_j a_j S(n+j) = G(n, K+1) - G(n, lo) - sum_j a_j sum_{k=hi(n+j)+1}^{K} F(n+j, k)
    infinite = isinstance(hi, Infinity)
    if infinite:
        top = limit(G, k, hi)
        if top.has(k):
            return None
        rhs: Expr = top - _value(G, k, lo)
    else:
        K = hi.subs(n, n + J)
        rhs = _value(G, k, K + 1) - _value(G, k, lo)
        for j, a in enumerate(Z.coefficients):
            upper = hi.subs(n, n + j)
            count = simplify(K - upper)
            if not (isinstance(count, Integer) and count.p >= 0):
                return None
            for i in range(1, count.p + 1):
                rhs = rhs - a*F.subs(n, n + j).subs(k, upper + i)
    rhs = simplify(combsimp(rhs))
    recurrence = add(*[a*Sfun(n + j) for j, a in enumerate(Z.coefficients)]) - rhs
    # initial values, from the first n at which the leading coefficient
    # of the recurrence does not vanish any more
    initial: dict[AppliedUndef, Union[Expr, int]] = {}
    n0 = _start(Z.coefficients[-1], n)
    for i in range(J):
        value = _direct_sum(F.subs(n, Integer(n0 + i)), k, lo, hi.subs(n, Integer(n0 + i)))
        if isinstance(value, Sum) or (infinite and not value.is_number):
            return _Recurrence(recurrence)
        initial[Sfun(Integer(n0 + i))] = value
    solution: Optional[Expr] = None
    if J == 0:
        # the term is Gosper-summable: a_0 S(n) = rhs
        solution = cancel(rhs/Z.coefficients[0])
    elif J == 1 and rhs == ZERO and initial:
        # a first order homogeneous recurrence: a product of its ratio
        a0, a1 = Z.coefficients
        idx = Dummy('i', integer=True)
        ratio = cancel(-a0/a1).subs(n, idx)
        prod = product(ratio, (idx, Integer(n0), n - 1))
        if not isinstance(prod, Product):
            solution = to_expr(initial[Sfun(Integer(n0))])*prod
    if solution is None:
        # the hypergeometric and d'Alembertian solutions; a solution
        # which is itself a sum is no closed form
        try:
            solved = rsolve_linear(recurrence, Sfun(n), initial if initial else None)
        except _Failures:
            solved = None
        if solved is not None and not solved.has(Sum, Product):
            solution = solved
    if solution is None:
        return _Recurrence(recurrence)
    solution = _nicer(solution, n)
    # the closed form is checked at a few values
    for v in range(n0, n0 + J + 3):
        direct = _direct_sum(F.subs(n, Integer(v)), k, lo, hi.subs(n, Integer(v)))
        if isinstance(direct, Sum):
            break
        if simplify(direct - solution.subs(n, Integer(v))) != ZERO:
            # an order-zero recurrence a0 S(n) = rhs would only re-assert
            # the answer the check has just rejected (sympy-extras#33)
            return None if J == 0 else _Recurrence(recurrence)
    return solution


def _expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, not %s" % (b,))
    return b


def _ops(e: Expr) -> int:
    count = count_ops(e)
    return count.p if isinstance(count, Integer) else 0


def _nicer(solution: Expr, n: Symbol) -> Expr:
    """The closed form simplified, the gamma functions of half-integer
    arguments removed with the duplication formula and factorials or
    binomials preferred when smaller."""

    def duplicate(e: Expr) -> Expr:
        def is_gamma(g: Basic) -> bool:
            return isinstance(g, Gamma)

        def rule(g: Basic) -> Basic:
            z = g.args[0]
            if not isinstance(z, Expr):
                return g
            half = z - HALF
            if half.is_integer is True or (half.is_polynomial(n) is True and _constant_term(half, n).is_integer is True
                                           and half.is_integer is not False):
                return gamma(2*half)*sqrt(pi)*2**(1 - 2*half)/gamma(half)
            return g
        return _expr(e.replace(is_gamma, rule))

    def triplicate(e: Expr) -> Expr:
        # gamma(z + 1/3) gamma(z + 2/3) = 2 pi 3**(1/2 - 3z) gamma(3z)/gamma(z)
        gammas = [g for g in e.atoms(Gamma) if isinstance(g, Gamma)]
        for g in gammas:
            a = g.args[0]
            for h in gammas:
                b = h.args[0]
                if not isinstance(a, Expr) or not isinstance(b, Expr):
                    continue
                if b - a == THIRD and (a - THIRD).is_polynomial(n) is True:
                    z = a - THIRD
                    formula = 2*pi*3**(HALF - 3*z)*gamma(3*z)/(gamma(z)*h)
                    replacement: dict[Basic, Basic] = {g: formula}
                    return e.xreplace(replacement)
        return e

    candidates = [simplify(combsimp(solution))]
    candidates.append(combsimp(duplicate(candidates[0])))
    candidates.append(combsimp(triplicate(candidates[-1])))
    candidates.append(combsimp(candidates[-1].rewrite(Factorial)))
    candidates.append(candidates[-1].rewrite(Binomial))
    candidates.append(simplify(candidates[-1]))
    return min(candidates, key=lambda e: (_ops(e), len(str(e))))


def _constant_term(e: Expr, n: Symbol) -> Expr:
    """The term of degree zero of the polynomial ``e`` in ``n``."""
    try:
        return _UPoly.of(e, n).nth(0)
    except ValueError:
        return e


def _start(leading: Expr, n: Symbol) -> int:
    """The first nonnegative integer from which the leading coefficient
    has no more integer zeros."""
    if leading.is_polynomial(n) is not True or not leading.has(n):
        return 0
    start = 0
    for f, _ in factor_list(leading, n)[1]:
        try:
            linear = _UPoly.of(f, n)
        except ValueError:
            continue
        if linear.degree() != 1:
            continue
        root = cancel(-linear.nth(0)/linear.lc())
        if isinstance(root, Integer) and root.p >= start:
            start = root.p + 1
    return start


def _value(G: Expr, k: Symbol, at: Expr) -> Expr:
    """``G`` at ``k = at``, by a limit when the substitution is not
    defined."""
    value = G.subs(k, at)
    if value.has(NaN, ComplexInfinity):
        value = limit(G, k, at)
    return simplify(value)


def _Recurrence(recurrence: Expr) -> Optional[Basic]:
    """The recurrence as an unevaluated equation, or ``None`` when it is
    not one (the core's ``Eq`` does not evaluate: a recurrence reducing
    to a nonzero constant must not become the Boolean ``False`` and be
    returned as the sum, and one that is ``nan`` says nothing,
    sympy-extras#33)."""
    if recurrence.has(NaN, ComplexInfinity):
        return None
    return Eq(recurrence, ZERO)
