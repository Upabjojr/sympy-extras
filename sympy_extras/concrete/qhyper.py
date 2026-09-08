"""q-hypergeometric summation: the q-analogues of Gosper's and
Zeilberger's algorithms.

A term `t(k)` is *q-hypergeometric* when `t(k+1)/t(k)` is a rational
function `R(Q)` of `Q = q^k` (with `q` and the other parameters in the
coefficient field); `q^{k(k-1)/2}`, `q^{ak}` and the q-Pochhammer symbols
`(a; q)_k = \\prod_{i=0}^{k-1} (1 - a q^i)` (:class:`QPochhammer`) are the
building blocks, and the q-binomial coefficients are quotients of them.

**q-Gosper** (Koornwinder; Paule and Riese): the ratio is written in the
normal form `R(Q) = (a(Q)/b(Q)) \\cdot c(qQ)/c(Q)` with `\\gcd(a(Q), b(q^j Q))
= 1` for every integer `j \\ge 0`; the indefinite sum `z` with `z(k+1) -
z(k) = t(k)` exists as a q-hypergeometric term exactly when the q-Gosper
equation `a(Q) f(qQ) - b(Q/q) f(Q) = c(Q)` has a Laurent polynomial
solution `f`, and then `z = b(Q/q) f(Q) t(k)/c(Q)`; the degree bounds of
`f` (highest and lowest powers of `Q`) come from the leading and
trailing coefficients, where cancellations happen when the ratio of the
leading coefficients is a power of `q`.

**q-Zeilberger**: for `t(n, k)` q-hypergeometric in both variables, the
parametrised q-Gosper equation for `\\sum_j a_j(q^n) t(n+j, k)` gives a
q-recurrence `\\sum_j a_j(q^n) S(n+j) = 0` for the definite sum and its
certificate, the q-analogue of :mod:`sympy_extras.concrete.zeilberger`.

SymPy has neither q-Pochhammer symbols nor any q-summation algorithm.

References
==========

.. [Koornwinder] T. H. Koornwinder, On Zeilberger's algorithm and its
   q-analogue, Journal of Computational and Applied Mathematics 48 (1993).
.. [PauleRiese] P. Paule, A. Riese, A Mathematica q-analogue of
   Zeilberger's algorithm based on an algebraically motivated approach to
   q-hypergeometric telescoping, Fields Institute Communications 14 (1997).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor, gcd

from sympy_extras._typing import as_expr

__all__ = ['QPochhammer', 'qpochhammer', 'qbinomial', 'q_ratio', 'normal_in', 'qgosper_term',
           'qgosper_sum', 'qzeilberger', 'QTelescoper']


class QPochhammer(Function):
    """The q-Pochhammer symbol ``(a; q)_k = prod_{i=0}^{k-1} (1 - a q**i)``,
    evaluated for nonnegative integer ``k``.

    >>> from sympy.abc import a, q, k
    >>> from sympy_extras.concrete.qhyper import QPochhammer
    >>> QPochhammer(a, q, 2)
    (1 - a)*(-a*q + 1)
    >>> QPochhammer(a, q, k)
    QPochhammer(a, q, k)
    """

    @classmethod
    def eval(cls, a: Expr, q: Expr, k: Expr) -> Optional[Expr]:
        if isinstance(k, Integer) and int(k) >= 0:
            return as_expr(Mul(*[1 - a*q**i for i in range(int(k))]))
        if isinstance(k, Integer):
            # (a; q)_{-m} = 1/prod_{i=1}^{m} (1 - a q**(-i))
            return as_expr(1/Mul(*[1 - a*q**(-i) for i in range(1, -int(k) + 1)]))
        return None


def qpochhammer(a: Expr, q: Expr, k: Expr) -> Expr:
    """``(a; q)_k`` (see :class:`QPochhammer`)."""
    return as_expr(QPochhammer(a, q, k))


def qbinomial(n: Expr, k: Expr, q: Expr) -> Expr:
    """The q-binomial coefficient ``(q; q)_n / ((q; q)_k (q; q)_{n-k})``.

    >>> from sympy import factor
    >>> from sympy.abc import q
    >>> from sympy_extras.concrete.qhyper import qbinomial
    >>> factor(qbinomial(4, 2, q))
    (q**2 + 1)*(q**2 + q + 1)
    """
    return as_expr(cancel(qpochhammer(q, q, n)/(qpochhammer(q, q, k)*qpochhammer(q, q, n - k))))


def _shift_qpochhammer(e: Expr, k: Symbol) -> Expr:
    """``e`` with ``k`` replaced by ``k + 1``, the q-Pochhammer symbols
    with arguments linear in ``k`` rewritten through
    ``(a; q)_{m+1} = (a; q)_m (1 - a q**m)`` (with ``m`` at the original
    ``k``)."""
    opaque: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for symbol in e.atoms(QPochhammer):
        a, q, m = symbol.args
        m_ = as_expr(m)
        step = as_expr(m_.subs(k, k + 1) - m_)
        if not isinstance(step, Integer):
            raise ValueError("%s is not linear in %s with an integer coefficient" % (symbol, k))
        c = int(step)
        d = Dummy('P')
        opaque[symbol] = d
        if c >= 0:
            back[d] = symbol*Mul(*[1 - a*q**(m_ + i) for i in range(c)])
        else:
            back[d] = symbol/Mul(*[1 - a*q**(m_ - i) for i in range(1, -c + 1)])
    shifted = as_expr(e.xreplace(opaque).subs(k, k + 1))
    return as_expr(shifted.xreplace(back))


def q_ratio(term: Expr, k: Symbol, q: Symbol) -> Optional[tuple[Expr, Symbol]]:
    """``t(k+1)/t(k)`` as a rational function of ``Q = q**k`` and the
    symbol ``Q``, or ``None`` when the term is not q-hypergeometric.

    >>> from sympy.abc import q, k, a
    >>> from sympy_extras.concrete.qhyper import q_ratio, qpochhammer
    >>> ratio, Q = q_ratio(qpochhammer(a, q, k)*q**k, k, q)
    >>> ratio
    -_Q*a*q + q
    """
    t = as_expr(sympify(term))
    try:
        shifted = _shift_qpochhammer(t, k)
    except ValueError:
        return None
    opaque = {symbol: Dummy('P') for symbol in t.atoms(QPochhammer)}
    Q = Dummy('Q')
    quotient = as_expr((shifted/t).xreplace(opaque))
    rational = as_expr(cancel(_in_Q(quotient, k, q, Q)))
    if rational.atoms(Dummy) & set(opaque.values()):
        return None
    if rational.has(k) or not rational.is_rational_function(Q):
        return None
    return rational, Q


def _in_Q(e: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Expr:
    """``e`` with the powers of the same base combined and every
    ``q**(c k + d)`` written as ``Q**c q**d``."""
    from sympy.simplify.powsimp import powsimp
    combined = as_expr(powsimp(e))
    expanded = as_expr(expand(combined, power_exp=True, power_base=False))
    return as_expr(expanded.replace(lambda p: isinstance(p, Pow) and p.base == q and p.exp.has(k),
                                    lambda p: _split_power(as_expr(p), k, q, Q)))


def normal_in(e: Expr, k: Symbol, q: Symbol) -> Expr:
    """``e`` as a factored rational function of ``q**k`` (the powers
    ``q**(c k + d)`` combined and split into ``(q**k)**c q**d`` first).

    >>> from sympy.abc import q, n, x
    >>> from sympy_extras.concrete.qhyper import normal_in
    >>> normal_in((q*q**n*x + q*q**n - q**(n + 1) - q**(n + 1)*x)/(q*q**n - 1), n, q)
    0
    """
    Q = Dummy('Q')
    rational = as_expr(cancel(_in_Q(e, k, q, Q)))
    return as_expr(factor(rational).subs(Q, q**k))


def _split_power(e: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Expr:
    """``q**(c*k + d)`` as ``Q**c q**d`` for an integer ``c``."""
    if not isinstance(e, Pow):
        return e
    exponent = as_expr(e.exp)
    try:
        poly = Poly(exponent, k)
    except PolynomialError:
        return e
    if poly.degree() != 1:
        return e
    c, d = [as_expr(v) for v in poly.all_coeffs()]
    if not isinstance(c, Integer):
        return e
    return as_expr(Q**int(c)*q**d)


def _q_power(e: Expr, q: Symbol, bound: int = 30) -> Optional[int]:
    """``n`` with ``e == q**n`` for an integer ``n`` in ``[-bound, bound]``,
    or ``None``."""
    e = as_expr(cancel(e))
    if e == 1:
        return 0
    if isinstance(e, Pow) and e.base == q and isinstance(e.exp, Integer):
        return int(e.exp)
    for n in range(-bound, bound + 1):
        if cancel(e - q**n) == 0:
            return n
    return None


def _normal_form(num: Poly, den: Poly, q: Symbol, Q: Symbol, bound: int = 20) -> tuple[Poly, Poly, Poly]:
    """``a, b, c`` with ``num/den == (a/b) c(qQ)/c(Q)`` and
    ``gcd(a(Q), b(q**j Q)) = 1`` for ``0 <= j <= bound``."""
    a, b, c = num, den, Poly(1, Q)
    for j in range(0, bound + 1):
        while True:
            g = gcd(a, Poly(b.as_expr().subs(Q, q**j*Q), Q))
            if g.degree() == 0:
                break
            a = a.quo(g)
            b = b.quo(Poly(g.as_expr().subs(Q, Q/q**j), Q))
            for i in range(1, j + 1):
                c = c*Poly(g.as_expr().subs(Q, Q/q**i), Q)
    return a, b, c


def _degree_bounds(a: Poly, b_shift: Poly, degree_c: int, valuation_c: int, q: Symbol) -> Optional[tuple[int, int]]:
    """The lowest and highest powers of ``Q`` in a Laurent polynomial
    solution ``f`` of ``a(Q) f(qQ) - b(Q/q) f(Q) = c(Q)``."""
    da, db = a.degree(), b_shift.degree()
    highs: list[int] = [degree_c - max(da, db)]
    if da == db:
        power = _q_power(as_expr(b_shift.LC()/a.LC()), q)
        if power is not None:
            highs.append(power)
    va, vb = _valuation(a), _valuation(b_shift)
    lows: list[int] = [valuation_c - min(va, vb)]
    if va == vb:
        power = _q_power(as_expr(_trailing(b_shift)/_trailing(a)), q)
        if power is not None:
            lows.append(power)
    high, low = max(highs), min(lows)
    if high < low:
        return None
    return low, high


def _valuation(p: Poly) -> int:
    return min(m[0] for m in p.monoms())


def _trailing(p: Poly) -> Expr:
    v = _valuation(p)
    return as_expr(p.coeff_monomial(p.gen**v))


class QTelescoper:
    """The output of the q-Zeilberger algorithm: the coefficients
    ``a_j(q**n)`` of the recurrence and the certificate ``R`` with
    ``sum_j a_j t(n+j, k) = G(k+1) - G(k)``, ``G = R t(n, k)``."""

    def __init__(self, term: Expr, n: Symbol, k: Symbol, q: Symbol, coefficients: list[Expr], certificate: Expr) -> None:
        self.term, self.n, self.k, self.q = term, n, k, q
        self.coefficients = coefficients
        self.certificate = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        f = Function(name)
        return as_expr(Add(*[a*f(self.n + j) for j, a in enumerate(self.coefficients)]))

    def __repr__(self) -> str:
        return "QTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def _solve_qgosper(ratio: Expr, Q: Symbol, q: Symbol, P: Expr, unknowns: list[Symbol]
                   ) -> Optional[tuple[Expr, dict[Basic, Basic]]]:
    """Solve the parametrised q-Gosper equation for the term whose ratio
    is ``ratio * P(qQ)/P(Q)``: returns the rational certificate
    ``b(Q/q) f(Q)/(c(Q) P(Q))`` and the values of the unknowns."""
    num, den = as_expr(cancel(ratio)).as_numer_denom()
    try:
        num_poly, den_poly = Poly(num, Q), Poly(den, Q)
        P_poly = Poly(P, Q)
    except PolynomialError:
        return None
    a, b, c = _normal_form(num_poly, den_poly, q, Q)
    b_shift = Poly(b.as_expr().subs(Q, Q/q), Q)
    C = c*P_poly
    bounds = _degree_bounds(a, b_shift, C.degree(), _valuation(C), q)
    if bounds is None:
        return None
    low, high = bounds
    f_coefficients = [Dummy('f%d' % i) for i in range(low, high + 1)]
    f = as_expr(Add(*[coefficient*Q**i for i, coefficient in zip(range(low, high + 1), f_coefficients)]))
    equation = as_expr(expand(a.as_expr()*f.subs(Q, q*Q) - b_shift.as_expr()*f - C.as_expr()))
    # clear the negative powers
    equation = as_expr(expand(equation*Q**max(0, -low)))
    try:
        rows = Poly(equation, Q).all_coeffs()
    except PolynomialError:
        return None
    variables: list[Expr] = [*f_coefficients, *unknowns]
    M = Matrix([[as_expr(cancel(as_expr(row).coeff(u))) for u in variables] for row in rows]) if rows else Matrix.zeros(0, len(variables))
    if unknowns:
        null = M.nullspace()
        for vector in null:
            values = [as_expr(cancel(v)) for v in vector]
            if all(v == 0 for v in values[len(f_coefficients):]):
                continue
            f_value = as_expr(f.xreplace({u: v for u, v in zip(f_coefficients, values)}))
            assignment: dict[Basic, Basic] = {u: v for u, v in zip(unknowns, values[len(f_coefficients):])}
            P_value = as_expr(P.xreplace(assignment))
            certificate = as_expr(cancel(b_shift.as_expr()*f_value/(c.as_expr()*P_value)))
            return certificate, assignment
        return None
    # the inhomogeneous system for the coefficients of f alone
    from sympy.solvers.solvers import solve as _solve
    solutions = _solve([as_expr(row) for row in rows if row != 0], f_coefficients, dict=True)
    if not solutions:
        return None
    values_ = dict(solutions[0])
    for u in f_coefficients:
        values_.setdefault(u, S.Zero)
    f_value = as_expr(f.xreplace(values_))
    certificate = as_expr(cancel(b_shift.as_expr()*f_value/(c.as_expr()*P)))
    return certificate, {}


def qgosper_term(term: Expr, k: Symbol, q: Symbol) -> Optional[Expr]:
    """``z(k)`` with ``z(k+1) - z(k) = term``, a q-hypergeometric term, or
    ``None`` when there is none.

    Examples
    ========

    >>> from sympy.abc import q, k
    >>> from sympy_extras.concrete.qhyper import qgosper_term
    >>> qgosper_term(q**k, k, q)
    q**k/(q - 1)
    """
    t = as_expr(sympify(term))
    data = q_ratio(t, k, q)
    if data is None:
        raise ValueError("%s is not q-hypergeometric in %s" % (t, k))
    ratio, Q = data
    solved = _solve_qgosper(ratio, Q, q, S.One, [])
    if solved is None:
        return None
    certificate, _ = solved
    z = as_expr(certificate.subs(Q, q**k)*t)
    return as_expr(factor(z)) if z.is_rational_function(q**k) else z


def qgosper_sum(term: Expr, limits: Sequence[Union[Symbol, Expr, int]], q: Symbol) -> Optional[Expr]:
    """``sum_{k=lo}^{hi} term`` in closed form by the q-Gosper algorithm,
    or ``None``.

    Examples
    ========

    >>> from sympy import factor
    >>> from sympy.abc import q, k, n
    >>> from sympy_extras.concrete.qhyper import qgosper_sum
    >>> factor(qgosper_sum(q**k, (k, 0, n), q))
    (q**(n + 1) - 1)/(q - 1)
    """
    k_, lo, hi = limits
    k = k_ if isinstance(k_, Symbol) else Symbol(str(k_))
    z = qgosper_term(term, k, q)
    if z is None:
        return None
    return as_expr(z.subs(k, sympify(hi) + 1) - z.subs(k, sympify(lo)))


def qzeilberger(term: Expr, n: Symbol, k: Symbol, q: Symbol, max_order: int = 3) -> Optional[QTelescoper]:
    """The q-Zeilberger algorithm: a q-recurrence in ``n`` for
    ``sum_k term`` with its certificate, of order at most ``max_order``.

    Examples
    ========

    >>> from sympy.abc import q, k, n, x
    >>> from sympy_extras.concrete.qhyper import qzeilberger, qbinomial
    >>> Z = qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q)   # the q-binomial theorem
    >>> Z.coefficients
    [-q**n*x - 1, 1]
    """
    t = as_expr(sympify(term))
    data = q_ratio(t, k, q)
    if data is None:
        raise ValueError("%s is not q-hypergeometric in %s" % (t, k))
    ratio, Q = data
    ratios: list[Expr] = []
    for j in range(max_order + 1):
        shifted = as_expr(t.subs(n, n + j))
        pair = _ratio_in_Q(shifted, t, k, q, Q)
        if pair is None:
            raise ValueError("%s is not q-hypergeometric in %s" % (t, n))
        ratios.append(pair)
    for J in range(1, max_order + 1):
        unknowns: list[Symbol] = [Dummy('a%d' % j) for j in range(J + 1)]
        from sympy.polys.polytools import lcm_list
        denominators = [as_expr(r.as_numer_denom()[1]) for r in ratios[:J + 1]]
        D = as_expr(lcm_list(denominators))
        P = as_expr(expand(Add(*[u*cancel(r*D) for u, r in zip(unknowns, ratios[:J + 1])])))
        solved = _solve_qgosper(as_expr(cancel(ratio*D/D.subs(Q, q*Q))), Q, q, P, unknowns)
        if solved is None:
            continue
        certificate, assignment = solved
        coefficients = [normal_in(as_expr(assignment[u]), n, q) for u in unknowns]
        # G = certificate * T = certificate * P/D * t
        R = as_expr(cancel(certificate*P.xreplace(assignment)/D))
        return QTelescoper(t, n, k, q, coefficients, as_expr(R.subs(Q, q**k)))
    return None


def _ratio_in_Q(f: Expr, g: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Optional[Expr]:
    """``f/g`` as a rational function of ``Q = q**k`` (the q-Pochhammer
    symbols cancelled through their shift rules)."""
    opaque = {symbol: Dummy('P') for symbol in g.atoms(QPochhammer)}
    # express the symbols of f through those of g: (a; q)_{m + c} = (a; q)_m * ...
    replacements: dict[Basic, Basic] = {}
    for symbol in f.atoms(QPochhammer):
        a, qq, m = symbol.args
        for base in g.atoms(QPochhammer):
            a2, qq2, m2 = base.args
            if a2 == a and qq2 == qq:
                step = as_expr(as_expr(m) - as_expr(m2))
                if isinstance(step, Integer):
                    c = int(step)
                    if c >= 0:
                        replacements[symbol] = base*Mul(*[1 - a*qq**(as_expr(m2) + i) for i in range(c)])
                    else:
                        replacements[symbol] = base/Mul(*[1 - a*qq**(as_expr(m2) - i) for i in range(1, -c + 1)])
                    break
    f_ = as_expr(f.xreplace(replacements))
    rational = as_expr(cancel(_in_Q(as_expr((f_/g).xreplace(opaque)), k, q, Q)))
    if rational.atoms(Dummy) & set(opaque.values()):
        return None
    if rational.has(k) or not rational.is_rational_function(Q):
        return None
    return rational
