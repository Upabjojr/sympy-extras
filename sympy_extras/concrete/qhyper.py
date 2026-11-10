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

The module works on the expressions of the core (:mod:`sympy_extras.core`:
the native polynomials, cancellation, factorisation and ``powsimp``, the
null space of :mod:`~sympy_extras.concrete.zeilberger`) and takes and
returns them; it imports no SymPy. :class:`QPochhammer` is the node, which
never evaluates, and :func:`qpochhammer` the function, which does. Only
the isolated values of the parameters go through SymPy, at the call
(:func:`~sympy_extras.concrete.karr._with_special_values`).
:mod:`~sympy_extras.concrete.qhyper_sympy` is its face on SymPy
expressions.

References
==========

.. [Koornwinder] T. H. Koornwinder, On Zeilberger's algorithm and its
   q-analogue, Journal of Computational and Applied Mathematics 48 (1993).
.. [PauleRiese] P. Paule, A. Riese, A Mathematica q-analogue of
   Zeilberger's algorithm based on an algebraically motivated approach to
   q-hypergeometric telescoping, Fields Institute Communications 14 (1997).
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.function import DefinedFunction, Function, diff, expand
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import ComplexInfinity, Integer, NaN
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import Poly, cancel, factor, gcd, lcm_list
from sympy_extras.simplify.powsimp import powsimp

from .gosper import _solve_linear, _UPoly
from .karr import SumPoint, _with_special_values, karr_term, summation
from .zeilberger import _nullspace

__all__ = ['QPochhammer', 'qpochhammer', 'qbinomial', 'q_ratio', 'normal_in', 'qgosper_term',
           'qgosper_sum', 'qzeilberger', 'QTelescoper']

ZERO: Final[Integer] = Integer(0)
ONE: Final[Integer] = Integer(1)


class QPochhammer(DefinedFunction):
    """The q-Pochhammer symbol ``(a; q)_k = prod_{i=0}^{k-1} (1 - a q**i)``,
    unevaluated (:func:`qpochhammer` evaluates it for an integer ``k`` and
    for ``a = 0``).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.qhyper import QPochhammer, qpochhammer
    >>> a, q, k = symbols('a q k')
    >>> QPochhammer(a, q, 2), qpochhammer(a, q, 2)
    (QPochhammer(a, q, 2), (1 - a)*(-a*q + 1))
    >>> qpochhammer(a, q, k), qpochhammer(0, q, k), qpochhammer(a, q, -1)
    (QPochhammer(a, q, k), 1, 1/(-a/q + 1))
    """

    function_name = 'QPochhammer'
    nargs = frozenset({3})
    call_nargs = frozenset({3})

    @classmethod
    def eval(cls, *args: Expr) -> Optional[Expr]:
        a, q, k = args
        if isinstance(k, Integer) and k.p >= 0:
            return mul(*[1 - a*q**i for i in range(k.p)])
        if a == ZERO:
            return ONE
        if isinstance(k, Integer):
            # (a; q)_{-m} = 1/prod_{i=1}^{m} (1 - a q**(-i))
            return 1/mul(*[1 - a*q**(-i) for i in range(1, -k.p + 1)])
        return None


def qpochhammer(a: ExprLike, q: ExprLike, k: ExprLike) -> Expr:
    """``(a; q)_k``, evaluated (see :class:`QPochhammer`)."""
    return QPochhammer.apply(a, q, k)


def qbinomial(n: ExprLike, k: ExprLike, q: ExprLike) -> Expr:
    """The q-binomial coefficient ``(q; q)_n / ((q; q)_k (q; q)_{n-k})``.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.polys.polytools import factor
    >>> from sympy_extras.concrete.qhyper import qbinomial
    >>> q = Symbol('q')
    >>> factor(qbinomial(4, 2, q))
    (q**2 + 1)*(q**2 + q + 1)
    """
    n_, k_, q_ = to_expr(n), to_expr(k), to_expr(q)
    return cancel(qpochhammer(q_, q_, n_)/(qpochhammer(q_, q_, k_)*qpochhammer(q_, q_, n_ - k_)))


def _evaluated(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, the rebuilt nodes
    evaluated (SymPy's ``xreplace``)."""
    return _expr(e._xreplace_evaluated(rule))


def _expr(b: Basic) -> Expr:
    """``b``, which must be an expression."""
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, not %s" % (b,))
    return b


def _rule(point: SumPoint) -> dict[Basic, Basic]:
    return {s: v for s, v in point.items()}


def _symbols(e: Expr) -> list[QPochhammer]:
    """The q-Pochhammer symbols of ``e``, in a fixed order."""
    found = [s for s in e.atoms(QPochhammer) if isinstance(s, QPochhammer)]
    return sorted(found, key=lambda s: s.sort_key())


def _arguments(symbol: QPochhammer) -> tuple[Expr, Expr, Expr]:
    a, q, m = symbol.args
    return _expr(a), _expr(q), _expr(m)


def _step(symbol: QPochhammer, a: Expr, q: Expr, m: Expr, c: int) -> Expr:
    """``(a; q)_{m+c}`` through ``(a; q)_m``."""
    if c >= 0:
        return symbol*mul(*[1 - a*q**(m + i) for i in range(c)])
    return symbol/mul(*[1 - a*q**(m - i) for i in range(1, -c + 1)])


def _shift_qpochhammer(e: Expr, k: Symbol) -> Expr:
    """``e`` with ``k`` replaced by ``k + 1``, the q-Pochhammer symbols
    with arguments linear in ``k`` rewritten through
    ``(a; q)_{m+1} = (a; q)_m (1 - a q**m)`` (with ``m`` at the original
    ``k``)."""
    opaque: dict[Basic, Basic] = {}
    back: dict[Basic, Basic] = {}
    for symbol in _symbols(e):
        a, q, m = _arguments(symbol)
        step = m.subs(k, k + 1) - m
        if not isinstance(step, Integer):
            raise ValueError("%s is not linear in %s with an integer coefficient" % (symbol, k))
        d = Dummy('P')
        opaque[symbol] = d
        back[d] = _step(symbol, a, q, m, step.p)
    shifted = _expr(_evaluated(e, opaque).subs(k, k + 1))
    return _evaluated(shifted, back)


def q_ratio(term: ExprLike, k: Symbol, q: Symbol) -> Optional[tuple[Expr, Symbol]]:
    """``t(k+1)/t(k)`` as a rational function of ``Q = q**k`` and the
    symbol ``Q``, or ``None`` when the term is not q-hypergeometric.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.qhyper import q_ratio, qpochhammer
    >>> q, k, a = symbols('q k a')
    >>> ratio, Q = q_ratio(qpochhammer(a, q, k)*q**k, k, q)
    >>> ratio
    -_Q*a*q + q
    """
    t = to_expr(term)
    try:
        shifted = _shift_qpochhammer(t, k)
    except ValueError:
        return None
    opaque: dict[Basic, Basic] = {s: Dummy('P') for s in _symbols(t)}
    Q = Dummy('Q')
    quotient = _evaluated(shifted/t, opaque)
    rational = cancel(_in_Q(quotient, k, q, Q))
    if rational.atoms(Dummy) & set(opaque.values()):
        return None
    if rational.has(k) or rational.is_rational_function(Q) is not True:
        return None
    return rational, Q


def _in_Q(e: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Expr:
    """``e`` with the powers of the same base combined and every
    ``q**(c k + d)`` written as ``Q**c q**d``."""
    combined = powsimp(e)
    expanded = expand(combined, power_exp=True, power_base=False)

    def is_power(p: Basic) -> bool:
        return isinstance(p, Pow) and p.base == q and p.exp.has(k)

    def split(p: Basic) -> Basic:
        return _split_power(_expr(p), k, q, Q)

    return _expr(expanded.replace(is_power, split))


def normal_in(e: ExprLike, k: Symbol, q: Symbol) -> Expr:
    """``e`` as a factored rational function of ``q**k`` (the powers
    ``q**(c k + d)`` combined and split into ``(q**k)**c q**d`` first).

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.qhyper import normal_in
    >>> q, n, x = symbols('q n x')
    >>> normal_in((q*q**n*x + q*q**n - q**(n + 1) - q**(n + 1)*x)/(q*q**n - 1), n, q)
    0
    """
    Q = Dummy('Q')
    rational = cancel(_in_Q(to_expr(e), k, q, Q))
    return _expr(factor(rational).subs(Q, q**k))


def _split_power(e: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Expr:
    """``q**(c*k + d)`` as ``Q**c q**d`` for an integer ``c``."""
    if not isinstance(e, Pow):
        return e
    try:
        poly = Poly(e.exp, k)
    except BasePolynomialError:
        return e
    if poly.degree() != 1:
        return e
    c, d = poly.all_coeffs()
    if not isinstance(c, Integer):
        return e
    return Q**c.p*q**d


def _q_power(e: Expr, q: Symbol, bound: int = 30) -> Optional[int]:
    """``n`` with ``e == q**n`` for an integer ``n`` in ``[-bound, bound]``,
    or ``None``."""
    e = cancel(e)
    if e == ONE:
        return 0
    if isinstance(e, Pow) and e.base == q and isinstance(e.exp, Integer):
        return e.exp.p
    for n in range(-bound, bound + 1):
        if cancel(e - q**n) == ZERO:
            return n
    return None


def _scaled(p: _UPoly, s: Expr) -> _UPoly:
    """``p(s Q)``."""
    return _UPoly([c*s**i for i, c in enumerate(p.c)])


def _gcd(f: _UPoly, g: _UPoly, Q: Symbol) -> _UPoly:
    """The monic gcd of two polynomials in ``Q`` over the rational
    functions of the parameters: the gcd of their numerators as
    polynomials in all the symbols (the factors free of ``Q`` are units),
    Euclid's algorithm where that fails."""
    try:
        common = gcd(cancel(f.as_expr(Q)).as_numer_denom()[0], cancel(g.as_expr(Q)).as_numer_denom()[0])
        return _UPoly.of(common, Q).monic()
    except (ValueError, TypeError, NotImplementedError, BasePolynomialError):
        return f.gcd(g)


def _normal_form(num: _UPoly, den: _UPoly, q: Symbol, Q: Symbol, bound: int = 20) -> tuple[_UPoly, _UPoly, _UPoly]:
    """``a, b, c`` with ``num/den == (a/b) c(qQ)/c(Q)`` and
    ``gcd(a(Q), b(q**j Q)) = 1`` for ``0 <= j <= bound``."""
    a, b, c = num, den, _UPoly([ONE])
    for j in range(0, bound + 1):
        while True:
            g = _gcd(a, _scaled(b, q**j), Q)
            if g.degree() == 0:
                break
            a = a.quo(g)
            b = b.quo(_scaled(g, q**(-j)))
            for i in range(1, j + 1):
                c = c*_scaled(g, q**(-i))
    return a, b, c


def _valuation(p: _UPoly) -> int:
    return next(i for i, c in enumerate(p.c) if c != ZERO)


def _trailing(p: _UPoly) -> Expr:
    return p.c[_valuation(p)]


def _degree_bounds(a: _UPoly, b_shift: _UPoly, degree_c: int, valuation_c: int, q: Symbol) -> Optional[tuple[int, int]]:
    """The lowest and highest powers of ``Q`` in a Laurent polynomial
    solution ``f`` of ``a(Q) f(qQ) - b(Q/q) f(Q) = c(Q)``."""
    da, db = a.degree(), b_shift.degree()
    highs: list[int] = [degree_c - max(da, db)]
    if da == db:
        power = _q_power(b_shift.lc()/a.lc(), q)
        if power is not None:
            highs.append(power)
    va, vb = _valuation(a), _valuation(b_shift)
    lows: list[int] = [valuation_c - min(va, vb)]
    if va == vb:
        power = _q_power(_trailing(b_shift)/_trailing(a), q)
        if power is not None:
            lows.append(power)
    high, low = max(highs), min(lows)
    if high < low:
        return None
    return low, high


class QTelescoper:
    """The output of the q-Zeilberger algorithm: the coefficients
    ``a_j(q**n)`` of the recurrence and the certificate ``R`` with
    ``sum_j a_j t(n+j, k) = G(k+1) - G(k)``, ``G = R t(n, k)``."""

    def __init__(self, term: Expr, n: Symbol, k: Symbol, q: Symbol, coefficients: list[Expr], certificate: Expr) -> None:
        self.term: Final[Expr] = term
        self.n: Final[Symbol] = n
        self.k: Final[Symbol] = k
        self.q: Final[Symbol] = q
        self.coefficients: Final[list[Expr]] = coefficients
        self.certificate: Final[Expr] = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def recurrence(self, name: str = 'S') -> Expr:
        """The recurrence ``sum_j a_j S(n + j)``, as an expression in the
        function ``S(n)``."""
        f = Function(name)
        return add(*[a*f(self.n + j) for j, a in enumerate(self.coefficients)])

    def __repr__(self) -> str:
        return "QTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def _solve_qgosper(ratio: Expr, Q: Symbol, q: Symbol, P: Expr, unknowns: list[Dummy]
                   ) -> Optional[tuple[Expr, dict[Basic, Basic]]]:
    """Solve the parametrised q-Gosper equation for the term whose ratio
    is ``ratio * P(qQ)/P(Q)``: returns the rational certificate
    ``b(Q/q) f(Q)/(c(Q) P(Q))`` and the values of the unknowns."""
    num, den = cancel(ratio).as_numer_denom()
    try:
        num_poly, den_poly = _UPoly.of(num, Q), _UPoly.of(den, Q)
        P_poly = _UPoly.of(P, Q)
    except ValueError:
        return None
    a, b, c = _normal_form(num_poly, den_poly, q, Q)
    b_shift = _scaled(b, 1/q)
    C = c*P_poly
    bounds = _degree_bounds(a, b_shift, C.degree(), _valuation(C), q)
    if bounds is None:
        return None
    low, high = bounds
    f_coefficients: list[Dummy] = [Dummy('f%d' % i) for i in range(low, high + 1)]
    f = add(*[coefficient*Q**i for i, coefficient in zip(range(low, high + 1), f_coefficients)])
    equation = expand(a.as_expr(Q)*f.subs(Q, q*Q) - b_shift.as_expr(Q)*f - C.as_expr(Q))
    # clear the negative powers
    equation = expand(equation*Q**max(0, -low))
    try:
        rows = _UPoly.of(equation, Q).c
    except ValueError:
        return None
    if unknowns:
        variables: list[Dummy] = [*f_coefficients, *unknowns]
        matrix = [[cancel(diff(row, u)) for u in variables] for row in rows]
        for vector in _nullspace(matrix, len(variables)):
            values = [cancel(v) for v in vector]
            if all(v == ZERO for v in values[len(f_coefficients):]):
                continue
            f_value = _evaluated(f, {u: v for u, v in zip(f_coefficients, values)})
            assignment: dict[Basic, Basic] = {u: v for u, v in zip(unknowns, values[len(f_coefficients):])}
            P_value = _evaluated(P, assignment)
            certificate = cancel(b_shift.as_expr(Q)*f_value/(c.as_expr(Q)*P_value))
            return certificate, assignment
        return None
    # the inhomogeneous system for the coefficients of f alone
    equations = [row for row in rows if row != ZERO]
    if not equations:
        return None
    solution = _solve_linear(equations, f_coefficients)
    if solution is None:
        return None
    # the free coefficients (a solution of the homogeneous equation) are 0,
    # in the values of the others too (where SymPy's solve left them)
    free: dict[Basic, Basic] = {u: ZERO for u in f_coefficients if u not in solution}
    values_: dict[Basic, Basic] = {u: _evaluated(solution.get(u, ZERO), free) for u in f_coefficients}
    f_value = _evaluated(f, values_)
    certificate = cancel(b_shift.as_expr(Q)*f_value/(c.as_expr(Q)*P))
    return certificate, {}


def qgosper_term(term: ExprLike, k: Symbol, q: Symbol, special_values: bool = True) -> Optional[Expr]:
    """``z(k)`` with ``z(k+1) - z(k) = term``, a q-hypergeometric term, or
    ``None`` when there is none.

    With ``special_values`` (the default) the isolated values of the
    parameters (``q`` among them) at which ``z`` is undefined get cases
    of their own, the antidifference computed again there, by Karr's
    algorithm (:func:`~sympy_extras.concrete.karr.karr_term`) at a value
    of ``q``; the cases without a closed form are left out.

    Examples
    ========

    >>> import sympy_extras.interop  # the cases of the parameters go through SymPy
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.qhyper import qgosper_term
    >>> q, k = symbols('q k')
    >>> qgosper_term(q**k, k, q)
    Piecewise((k, Eq(q, 1)), (q**k/(q - 1), True))
    >>> qgosper_term(q**k, k, q, special_values=False)
    q**k/(q - 1)
    """
    t = to_expr(term)
    data = q_ratio(t, k, q)
    if data is None:
        raise ValueError("%s is not q-hypergeometric in %s" % (t, k))
    ratio, Q = data
    solved = _solve_qgosper(ratio, Q, q, ONE, [])
    if solved is None:
        return None
    certificate, _ = solved
    z = _expr(certificate.subs(Q, q**k))*t
    # SymPy's is_rational_function(q**k) holds for every expression but nan and zoo
    if not isinstance(z, (NaN, ComplexInfinity)):
        z = factor(z)
    if not special_values:
        return z

    def at_point(point: SumPoint) -> Optional[Expr]:
        at_term = _evaluated(t, _rule(point))
        if q in point:
            return karr_term(at_term, k)
        return qgosper_term(at_term, k, q)

    return _with_special_values(z, t, [k], [], at_point, keep_unevaluated=False)


def qgosper_sum(term: ExprLike, limits: Sequence[ExprLike], q: Symbol,
                special_values: bool = True) -> Optional[Expr]:
    """``sum_{k=lo}^{hi} term`` in closed form by the q-Gosper algorithm,
    or ``None``.

    With ``special_values`` (the default) the isolated values of the
    parameters of the term (``q`` among them) at which the closed form
    is undefined get cases of their own, the sum computed again there,
    by :func:`~sympy_extras.concrete.karr.summation` at a value of ``q``
    (``q = 1``, where the q-analogue becomes an ordinary sum); the cases
    without a closed form are left out.

    Examples
    ========

    >>> import sympy_extras.interop  # the cases of the parameters go through SymPy
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.polytools import factor
    >>> from sympy_extras.concrete.qhyper import qgosper_sum
    >>> q, k, n = symbols('q k n')
    >>> factor(qgosper_sum(q**k, (k, 0, n), q, special_values=False))
    (q**(n + 1) - 1)/(q - 1)
    >>> qgosper_sum(q**(2*k), (k, 0, n), q)
    Piecewise((n + 1, Eq(q, -1) | Eq(q, 1)), (q**(2*n + 2)/((q - 1)*(q + 1)) - 1/((q - 1)*(q + 1)), True))
    """
    if len(limits) != 3:
        raise ValueError("the limits of a sum are (k, lo, hi), not %s" % (limits,))
    k_, lo_, hi_ = limits
    k = k_ if isinstance(k_, Symbol) else Symbol(str(k_))
    lo, hi = to_expr(lo_), to_expr(hi_)
    t = to_expr(term)
    z = qgosper_term(t, k, q, special_values=False)
    if z is None:
        return None
    value = _expr(z.subs(k, hi + 1)) - _expr(z.subs(k, lo))
    if value.has(NaN, ComplexInfinity):
        # an antidifference undefined at a limit (the bug: nan was
        # returned as the sum of qbinomial(k, 2, q)*q**k from k = 0, the
        # symbol (q; q)_{k - 2} infinite at k = 0)
        return None
    if not special_values:
        return value

    def at_point(point: SumPoint) -> Optional[Expr]:
        at_term = _evaluated(t, _rule(point))
        if q in point:
            return summation(at_term, (k, lo, hi))
        return qgosper_sum(at_term, (k, lo, hi), q)

    return _with_special_values(value, t, [k], [lo, hi], at_point, keep_unevaluated=False)


def qzeilberger(term: ExprLike, n: Symbol, k: Symbol, q: Symbol, max_order: int = 3) -> Optional[QTelescoper]:
    """The q-Zeilberger algorithm: a q-recurrence in ``n`` for
    ``sum_k term`` with its certificate, of order at most ``max_order``.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.concrete.qhyper import qzeilberger, qbinomial
    >>> q, k, n, x = symbols('q k n x')
    >>> Z = qzeilberger(qbinomial(n, k, q)*q**(k*(k - 1)/2)*x**k, n, k, q)   # the q-binomial theorem
    >>> Z.coefficients
    [-q**n*x - 1, 1]
    """
    t = to_expr(term)
    data = q_ratio(t, k, q)
    if data is None:
        raise ValueError("%s is not q-hypergeometric in %s" % (t, k))
    ratio, Q = data
    ratios: list[Expr] = []
    for j in range(max_order + 1):
        shifted = _expr(t.subs(n, n + j))
        pair = _ratio_in_Q(shifted, t, k, q, Q)
        if pair is None:
            raise ValueError("%s is not q-hypergeometric in %s" % (t, n))
        ratios.append(pair)
    for J in range(1, max_order + 1):
        unknowns: list[Dummy] = [Dummy('a%d' % j) for j in range(J + 1)]
        denominators = [r.as_numer_denom()[1] for r in ratios[:J + 1]]
        D = lcm_list(denominators)
        P = expand(add(*[u*cancel(r*D) for u, r in zip(unknowns, ratios[:J + 1])]))
        solved = _solve_qgosper(cancel(ratio*D/_expr(D.subs(Q, q*Q))), Q, q, P, unknowns)
        if solved is None:
            continue
        certificate, assignment = solved
        coefficients = [normal_in(_expr(assignment[u]), n, q) for u in unknowns]
        # G = certificate * T = certificate * P/D * t
        R = cancel(certificate*_evaluated(P, assignment)/D)
        return QTelescoper(t, n, k, q, coefficients, _expr(R.subs(Q, q**k)))
    return None


def _ratio_in_Q(f: Expr, g: Expr, k: Symbol, q: Symbol, Q: Symbol) -> Optional[Expr]:
    """``f/g`` as a rational function of ``Q = q**k`` (the q-Pochhammer
    symbols cancelled through their shift rules)."""
    bases = _symbols(g)
    opaque: dict[Basic, Basic] = {s: Dummy('P') for s in bases}
    # express the symbols of f through those of g: (a; q)_{m + c} = (a; q)_m * ...
    replacements: dict[Basic, Basic] = {}
    for symbol in _symbols(f):
        a, qq, m = _arguments(symbol)
        for base in bases:
            a2, qq2, m2 = _arguments(base)
            if a2 == a and qq2 == qq:
                step = m - m2
                if isinstance(step, Integer):
                    replacements[symbol] = _step(base, a, qq, m2, step.p)
                    break
    f_ = _evaluated(f, replacements)
    rational = cancel(_in_Q(_evaluated(f_/g, opaque), k, q, Q))
    if rational.atoms(Dummy) & set(opaque.values()):
        return None
    if rational.has(k) or rational.is_rational_function(Q) is not True:
        return None
    return rational
