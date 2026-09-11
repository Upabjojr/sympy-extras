"""The Almkvist–Zeilberger algorithm: creative telescoping for definite
integrals of hyperexponential functions, the continuous analogue of
Zeilberger's algorithm (:mod:`sympy_extras.concrete.zeilberger`).

A function `F(x, t)` is *hyperexponential* in `x` and `t` when both
logarithmic derivatives

.. math::

    q = \\frac{\\partial_x F}{F}, \\qquad p = \\frac{\\partial_t F}{F}

are rational functions: products of powers of polynomials and of
exponentials of rational functions, such as `e^{-x^2 - t^2/x^2}`,
`x^a (1 - x)^b (1 - t x)^c` or `e^{-x t}/(x^2 + t^2)`. The algorithm
looks for rational functions `a_0(t), \\ldots, a_J(t)` of `t` alone, not
all zero, and a rational function `R(x, t)` with

.. math::

    \\sum_{j=0}^{J} a_j(t)\\, \\partial_t^j F = \\partial_x \\bigl(R F\\bigr).

Dividing by `F` and writing `p_j = \\partial_t^j F / F` (rational, by the
recursion `p_0 = 1`, `p_{j+1} = \\partial_t p_j + p\\, p_j`) this is the
first order linear differential equation

.. math::

    R' + q R = \\sum_j a_j p_j

for a rational `R` with the `a_j` as unknowns. As in Gosper's algorithm,
the denominator of `R` is bounded by a local analysis at every
irreducible factor `d` of the denominators of `q` and of the `p_j`: with
`k` the order of the pole of `q` at `d` and `m` the order of the pole of
`R`, the pole of `R' + q R` has order `m + 1` (`k \\le 1`, unless `k = 1`
and the residue of `q` is the positive integer `m`, when the leading terms
cancel) or `m + k` (`k \\ge 2`), which must not exceed the pole order of
the right hand side. The degree of the numerator is bounded likewise at
infinity. What remains is a linear system over `\\mathbb{Q}(t)` for the
coefficients of the numerator and the `a_j`; the order `J` is increased
until it has a solution.

Integrating over `x` gives, for the definite integral
`I(t) = \\int_a^b F\\, dx`, the linear differential equation

.. math::

    \\sum_j a_j(t)\\, I^{(j)}(t) = \\bigl[R F\\bigr]_{x=a}^{x=b},

whose right hand side vanishes at natural boundaries. SymPy's ``dsolve``
solves it and the constants are fixed by the values of the integral (and
its derivatives) at a convenient point, computed by
:func:`~sympy_extras.integrals.definite_integral`. This is the
*holonomic* method of definite integration: it applies to integrands
outside the tables of Mellin transforms, and the differential equation
is an answer in itself when it has no closed form solution. Chyzak's
algorithm, which does the same for general D-finite integrands (Bessel
functions, orthogonal polynomials, ...), is not implemented.

Examples
========

>>> from sympy import symbols, exp, oo, sqrt, pi
>>> from sympy_extras.integrals.telescoping import almkvist_zeilberger, holonomic_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> almkvist_zeilberger(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)
>>> holonomic_integral(exp(-x**2 - t**2/x**2), x, 0, oo, t)
ConditionalValue(sqrt(pi)*exp(-2*t)/2)

References
==========

.. [AZ] G. Almkvist, D. Zeilberger, The method of differentiating under
   the integral sign, Journal of Symbolic Computation 10 (1990), 571–591.
.. [Koutschan] C. Koutschan, Advanced applications of the holonomic
   systems approach, PhD thesis, RISC, Johannes Kepler University Linz,
   2009, chapter 2.
.. [PWZ] M. Petkovšek, H. Wilf, D. Zeilberger, A = B, A K Peters, 1996,
   chapter 6 (the discrete algorithm, of which this is the analogue).
.. [Chyzak] F. Chyzak, An extension of Zeilberger's fast algorithm to
   general holonomic functions, Discrete Mathematics 217 (2000), 115–134.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, diff
from sympy.core.numbers import Integer, Rational, nan, oo, zoo
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, gcd_list, lcm_list
from sympy.polys.rationaltools import together
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve
from sympy.solvers.solveset import linsolve
from sympy.sets.sets import FiniteSet
from sympy.core.sympify import sympify

from sympy_extras._timeout import attempt
from sympy_extras._typing import ExprLike, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.limits import limit
from sympy_extras.settings import settings
from .conditions import ConditionalValue

__all__ = ['DifferentialTelescoper', 'logarithmic_derivative', 'is_hyperexponential',
           'almkvist_zeilberger', 'holonomic_ode', 'holonomic_integral']


class DifferentialTelescoper:
    """The output of the Almkvist–Zeilberger algorithm for ``F(x, t)``.

    Attributes
    ==========

    coefficients : list of Expr
        The polynomials ``a_0(t), ..., a_J(t)`` of the differential
        equation, without common factors.
    certificate : Expr
        The function ``R(x, t)`` with ``G = R*F`` (a rational function of
        ``x`` for a hyperexponential ``F``, and for a sum of such terms
        the sum of their certificates weighted by the terms).
    term, x, t
        The input.
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr], certificate: Expr) -> None:
        self.term = term
        self.x = x
        self.t = t
        self.coefficients = list(coefficients)
        self.certificate = certificate

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_j a_j(t) I^(j)(t)``, the left hand side of the equation
        satisfied by the integral over ``x``, in the function ``I(t)``."""
        f = Function(name)(self.t)
        return as_expr(Add(*[a * diff(f, self.t, j) for j, a in enumerate(self.coefficients)]))

    def check(self) -> bool:
        """Verify ``sum_j a_j d^j F/dt^j = d(R F)/dx`` by simplification."""
        F = self.term
        lhs = Add(*[a * diff(F, self.t, j) for j, a in enumerate(self.coefficients)])
        difference = as_expr((lhs - diff(self.certificate * F, self.x)) / F)
        if difference.has(TrigonometricFunction, HyperbolicFunction):
            difference = as_expr(difference.rewrite(exp))
        difference = as_expr(cancel(difference.expand()))
        return difference == 0 or simplify(difference) == 0

    def __repr__(self) -> str:
        return "DifferentialTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def logarithmic_derivative(F: Expr, x: Symbol) -> Optional[Expr]:
    """``diff(F, x)/F`` as a rational function of ``x`` (cancelled), or
    ``None`` when it is not one.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.telescoping import logarithmic_derivative
    >>> x, t = symbols('x t')
    >>> logarithmic_derivative(x**t*exp(-x**2), x)
    (t - 2*x**2)/x
    >>> logarithmic_derivative(exp(exp(x)), x) is None
    True
    """
    if F == 0:
        return None
    ratio = as_expr(cancel(diff(F, x) / F))
    if ratio.has(exp) and not ratio.is_rational_function(x):
        # powers written as exponentials of logarithms
        ratio = as_expr(cancel(ratio.expand(power_exp=True, log=True)))
    if not ratio.is_rational_function(x):
        return None
    return ratio


def is_hyperexponential(F: ExprLike, x: Symbol, t: Symbol) -> bool:
    """Whether both logarithmic derivatives of ``F`` are rational
    functions of ``x`` and ``t``.

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.telescoping import is_hyperexponential
    >>> x, t = symbols('x t')
    >>> is_hyperexponential(exp(-x*t)*x**3/(x**2 + t**2), x, t)
    True
    >>> is_hyperexponential(exp(-x*t)*sin(x), x, t)
    False
    """
    F_ = as_expr(sympify(F))
    q = logarithmic_derivative(F_, x)
    p = logarithmic_derivative(F_, t)
    return q is not None and p is not None and q.is_rational_function(x, t) and p.is_rational_function(x, t)


# ---------------------------------------------------------------------------
# The parametrised differential equation R' + q R = sum a_j p_j

def _polys(e: Expr, x: Symbol, t: Symbol) -> tuple[Poly, Poly]:
    """Numerator and denominator of a rational function of ``x`` as
    polynomials over the field of rational functions of ``t`` (with
    Gaussian rationals when ``I`` occurs)."""
    numerator, denominator = together(e).as_numer_denom()
    A = Poly(numerator, x, field=True)
    B = Poly(denominator, x, field=True)
    A, B = A.unify(B)
    if not A.domain.is_Field:
        A, B = A.to_field(), B.to_field()
    return A, B


def _positive_integer(e: Expr) -> Optional[int]:
    e_ = as_expr(cancel(e))
    if isinstance(e_, (Integer, Rational)) and e_.is_integer and e_ > 0:
        return int(e_)
    return None


def _denominator_bound(A: Poly, B: Poly, denominators: Sequence[Poly], x: Symbol) -> Poly:
    """The polynomial ``D`` the denominator of ``R`` divides, from the
    local analysis of ``R' + q R = S`` at the irreducible factors of the
    denominators of ``q = A/B`` and of the ``p_j`` (see the module
    documentation)."""
    K = B.domain
    factors: dict[Poly, tuple[int, int]] = {}      # factor -> (order in B, max order in the p_j)
    for _, d, k in _irreducibles(B):
        factors[d] = (k, 0)
    for L in denominators:
        for _, d, k in _irreducibles(L):
            previous = factors.get(d, (0, 0))
            factors[d] = (previous[0], max(previous[1], k))
    D = Poly(1, x, domain=K)
    for d, (k, order_s) in factors.items():
        if k <= 1:
            m = order_s - 1
            if k == 1:
                # a simple pole of q with a positive integer residue rho
                # allows a pole of order rho of R (the leading terms cancel)
                cofactor = B.quo(d)
                try:
                    residue = (A * cofactor.invert(d)).rem(d)
                except (PolynomialError, ZeroDivisionError, NotImplementedError):
                    residue = None
                if residue is not None and residue.degree() <= 0:
                    rho = _positive_integer(as_expr(residue.as_expr()))
                    if rho is not None:
                        m = max(m, rho)
        else:
            m = order_s - k
        if m > 0:
            D = D * d**m
    return D


def _irreducibles(P: Poly) -> list[tuple[int, Poly, int]]:
    """``(index, factor, multiplicity)`` for the irreducible factors of
    ``P`` which depend on the variable."""
    result: list[tuple[int, Poly, int]] = []
    _, factors = P.factor_list()
    for i, (d, k) in enumerate(factors):
        if d.degree() > 0:
            result.append((i, d.monic(), int(k)))
    return result


def _numerator_degree(A: Poly, B: Poly, D: Poly, degree_s: int) -> int:
    """The bound on the degree of the numerator ``N`` of ``R = N/D``
    from the behaviour at infinity."""
    degree_q = A.degree() - B.degree()
    if degree_q >= 0:
        n = degree_s + D.degree() - degree_q
    elif degree_q == -1:
        n = degree_s + D.degree() + 1
        # R' + q R ~ (n - deg D + rho_oo) x^(n - deg D - 1) may cancel
        rho = as_expr(cancel(as_expr(A.LC()) / as_expr(B.LC())))
        zero = _positive_integer(-rho)
        if zero is not None:
            n = max(n, D.degree() + zero)
        elif rho == 0:
            n = max(n, D.degree())
    else:
        n = degree_s + D.degree() + 1
    return max(n, 0)


def _telescope(q: Expr, p: Expr, x: Symbol, t: Symbol, order: int) -> Optional[tuple[list[Expr], Expr]]:
    """``(a_0, ..., a_J, R)`` for the given order ``J``, or ``None``."""
    A, B = _polys(q, x, t)
    ps: list[Expr] = [S.One]
    for _ in range(order):
        ps.append(as_expr(cancel(diff(ps[-1], t) + p * ps[-1])))
    fractions: list[tuple[Poly, Poly]] = []
    for pj in ps:
        P, Lj = _polys(pj, x, t)
        A, P = A.unify(P)
        B, Lj = B.unify(Lj)
        fractions.append((P, Lj))
    K = A.domain
    A, B = A.set_domain(K), B.set_domain(K)
    fractions = [(P.set_domain(K), Lj.set_domain(K)) for P, Lj in fractions]
    denominators = [Lj for _, Lj in fractions]
    D = _denominator_bound(A, B, denominators, x)
    degree_s = max(P.degree() - Lj.degree() for P, Lj in fractions)
    n = _numerator_degree(A, B, D, degree_s)
    L = Poly(1, x, domain=K)
    for _, Lj in fractions:
        L = L.lcm(Lj)
    coefficients = [Dummy('c%d' % i) for i in range(n + 1)]
    unknowns_a = [Dummy('a%d' % j) for j in range(order + 1)]
    N = Add(*[c * x**i for i, c in enumerate(coefficients)])
    D_e, A_e, B_e, L_e = D.as_expr(), A.as_expr(), B.as_expr(), L.as_expr()
    # B L (N' D - N D') + A L N D = B D^2 sum_j a_j (L / L_j) P_j
    lhs = B_e * L_e * (diff(N, x) * D_e - N * diff(D_e, x)) + A_e * L_e * N * D_e
    rhs: Expr = S.Zero
    for a, (P, Lj) in zip(unknowns_a, fractions):
        rhs = rhs + a * L.quo(Lj).as_expr() * P.as_expr()
    rhs = B_e * D_e**2 * rhs
    equation = Poly(as_expr((lhs - rhs).expand()), x)
    unknowns = coefficients + unknowns_a
    rows: list[list[Expr]] = []
    for coefficient in equation.all_coeffs():
        c = as_expr(coefficient)
        rows.append([as_expr(cancel(c.diff(u))) for u in unknowns])
    if not rows:
        return None
    matrix = Matrix(rows)
    for vector in matrix.nullspace():
        values = [as_expr(cancel(v)) for v in vector]
        a_values = values[len(coefficients):]
        if all(v == 0 for v in a_values):
            continue
        N_value = as_expr(Add(*[c * x**i for i, c in enumerate(values[:len(coefficients)])]))
        R = as_expr(cancel(N_value / D_e))
        normalized, factor_ = _normalized(a_values, t)
        return normalized, as_expr(cancel(R * factor_))
    return None


def _normalized(values: list[Expr], t: Symbol) -> tuple[list[Expr], Expr]:
    """The coefficients cleared of denominators and common factors, and
    the factor they were multiplied by (which the certificate gets too)."""
    denominators = [as_expr(together(v).as_numer_denom()[1]) for v in values]
    scale = as_expr(lcm_list(denominators))
    scaled = [as_expr(cancel(v * scale)) for v in values]
    common = as_expr(gcd_list([v for v in scaled if v != 0]))
    result = [as_expr(cancel(v / common)) for v in scaled]
    factor_ = as_expr(cancel(scale / common))
    # a positive leading coefficient of the highest order term
    lead = result[-1]
    if lead.is_negative or (lead.is_number is False and lead.is_polynomial(t)
                            and as_expr(Poly(lead, t).LC()).is_negative):
        result = [-v for v in result]
        factor_ = -factor_
    return result, factor_


def _terms(F: Expr, x: Symbol, t: Symbol) -> list[Expr]:
    """``F`` as a sum of hyperexponential terms (trigonometric and
    hyperbolic functions rewritten as exponentials), or ``[]``."""
    G = F
    if G.has(TrigonometricFunction, HyperbolicFunction):
        G = as_expr(G.rewrite(exp))
    G = as_expr(G.expand())
    terms = [as_expr(a) for a in Add.make_args(G)]
    if all(is_hyperexponential(term, x, t) for term in terms):
        return terms
    if is_hyperexponential(F, x, t):
        return [F]
    return []


def almkvist_zeilberger(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 4,
                        min_order: int = 0) -> Optional[DifferentialTelescoper]:
    """The Almkvist–Zeilberger algorithm for a hyperexponential ``F(x, t)``,
    or a sum of hyperexponential terms sharing the same equation.

    Parameters
    ==========

    F : Expr
        The integrand, hyperexponential in ``x`` and ``t`` (see
        :func:`is_hyperexponential`); ``sin``, ``cos``, ``sinh``, ``cosh``
        are rewritten as exponentials and the terms telescoped one by one.
    x : Symbol
        The integration variable.
    t : Symbol
        The parameter.
    max_order, min_order : int
        The orders ``J`` of the differential equation tried.

    Returns
    =======

    A :class:`DifferentialTelescoper` with the coefficients ``a_j(t)`` and
    the certificate ``R``, or ``None`` when no equation of order at most
    ``max_order`` exists (or the terms of a sum have different ones).

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.telescoping import almkvist_zeilberger
    >>> x, t = symbols('x t')
    >>> telescoper = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    >>> telescoper.coefficients, telescoper.certificate
    ([1, t], -x)
    >>> telescoper.check()
    True
    >>> almkvist_zeilberger(exp(-x**2)*exp(2*x*t), x, t).coefficients
    [-2*t, 1]
    """
    F_ = as_expr(sympify(F))
    terms = _terms(F_, x, t)
    if not terms:
        return None
    found: list[tuple[list[Expr], Expr, Expr]] = []
    for term in terms:
        q = logarithmic_derivative(term, x)
        p = logarithmic_derivative(term, t)
        if q is None or p is None:
            return None
        result: Optional[tuple[list[Expr], Expr]] = None
        for order in range(min_order, max_order + 1):
            result = _telescope(q, p, x, t, order)
            if result is not None:
                break
        if result is None:
            return None
        found.append((result[0], result[1], term))
    coefficients = found[0][0]
    for other, _, _ in found[1:]:
        if len(other) != len(coefficients) or any(cancel(a - b) != 0 for a, b in zip(other, coefficients)):
            # the terms of the sum satisfy different equations: the
            # least common left multiple is not computed
            return None
    if len(found) == 1:
        certificate = found[0][1]
    else:
        certificate = as_expr(cancel(Add(*[R * term for _, R, term in found]) / F_))
    return DifferentialTelescoper(F_, x, t, coefficients, certificate)


# ---------------------------------------------------------------------------
# The differential equation of the integral, and its solution

def _boundary(G: Expr, x: Symbol, point: Expr, direction: str, assumptions: Assumptions) -> Optional[Expr]:
    value = attempt(lambda: limit(G, x, point, direction, assumptions), settings.timeout)
    if value is None or value.has(nan, zoo, oo, -oo, Piecewise) or value.has(x):
        return None
    return value


def holonomic_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """The linear differential equation ``sum_j a_j(t) I^(j)(t) = [R F]_a^b``
    satisfied by ``I(t) = Integral(F, (x, a, b))``, with the boundary
    values taken as limits under the assumptions; ``None`` when no
    telescoper is found or a boundary limit is not finite.

    Examples
    ========

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.telescoping import holonomic_ode
    >>> x, t = symbols('x t')
    >>> holonomic_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    telescoper = almkvist_zeilberger(F_, x, t)
    if telescoper is None:
        return None
    G = as_expr(telescoper.certificate * F_)
    upper = _boundary(G, x, b_, '-', assumptions)
    lower = _boundary(G, x, a_, '+', assumptions)
    if upper is None or lower is None:
        return None
    return Eq(telescoper.operator(name), as_expr(simplify(upper - lower)))


def _constants(solution: Expr) -> list[Symbol]:
    return sorted([s for s in solution.free_symbols if isinstance(s, Symbol) and s.name.startswith('C')
                   and s.name[1:].isdigit()], key=lambda s: s.name)


def _initial_values(F: Expr, x: Symbol, a: Expr, b: Expr, t: Symbol, t0: Expr, order: int,
                    assumptions: Assumptions) -> Optional[list[Expr]]:
    """``I^(j)(t0)`` for ``j < order`` by integrating ``d^j F/dt^j`` at
    ``t = t0`` (differentiation under the integral sign), or ``None``."""
    from .definite import definite_integral
    values: list[Expr] = []
    for j in range(order):
        integrand = as_expr(diff(F, t, j).subs(t, t0))
        value = attempt(lambda: definite_integral(integrand, (x, a, b), assumptions), settings.timeout)
        if value is None or value.has(Integral, Piecewise, nan, zoo, oo, -oo):
            return None
        values.append(value)
    return values


def holonomic_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(F, (x, a, b))`` as a function of the parameter ``t`` by
    the holonomic method: the differential equation of
    :func:`holonomic_ode` solved by ``dsolve``, its constants fixed by
    the values of the integral and its derivatives at a point ``t0``
    (``1``, ``2``, ``1/2``, ``3``, the first at which
    :func:`~sympy_extras.integrals.definite_integral` gives them).

    The value holds where the integral converges and may be
    differentiated under the integral sign; it is checked numerically
    when ``settings.numerical_checks`` is on. ``None`` when a step fails.

    Examples
    ========

    >>> from sympy import symbols, exp, oo, cos
    >>> from sympy_extras.integrals.telescoping import holonomic_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> holonomic_integral(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
    ConditionalValue(sqrt(pi)*exp(-t**2)/2)
    >>> holonomic_integral(1/(x**2 + t**2)**2, x, -oo, oo, t)
    ConditionalValue(pi/(2*t**3))
    """
    from .definite import verify_numerically
    from .marichev import tidy
    F_ = as_expr(sympify(F))
    a_, b_ = as_expr(sympify(a)), as_expr(sympify(b))
    equation = holonomic_ode(F_, x, a_, b_, t, assumptions)
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
    if settings.numerical_checks and verify_numerically(value, F_, x, a_, b_, assumptions) is False:
        return None
    return ConditionalValue(value)


def _order(lhs: Expr, unknown: Expr, t: Symbol) -> int:
    """The order of a linear differential expression in ``unknown``."""
    order = 0
    for j in range(1, 8):
        if lhs.has(diff(unknown, t, j)):
            order = j
    return order
