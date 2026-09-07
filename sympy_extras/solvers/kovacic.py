"""Kovacic's algorithm: Liouvillian solutions of second order linear
ordinary differential equations with rational coefficients.

For `y'' + p y' + q y = 0` with `p, q` rational functions, the change of
variable `y = z \\exp(-\\int p/2)` gives the normal form `z'' = r z` with
`r = p^2/4 + p'/2 - q`. Kovacic's theorem says that if the equation has a
Liouvillian solution (built from rational functions by integrals,
exponentials of integrals and algebraic functions) then it has one of the
form `z = \\exp(\\int \\omega)` where `\\omega` is algebraic over the rational
functions of degree 1, 2, 4, 6 or 12, and the algorithm decides which by
looking at the orders of the poles of `r` (and its order at infinity):

* **Case 1** (`\\omega` rational): `\\omega` is a rational solution of the
  Riccati equation `\\omega' + \\omega^2 = r`, found here with SymPy's
  rational Riccati solver (which implements this case). The second
  solution follows by reduction of order.
* **Case 2** (`\\omega` quadratic): for each pole and for infinity a finite
  set `E_c` of integers is built from the order of the pole and the
  coefficient of `(x - c)^{-2}`; the families `(e_c)` with
  `d = (e_\\infty - \\sum e_c)/2` a nonnegative integer give a candidate
  `\\theta = \\sum e_c/(2(x - c))` and a monic polynomial `P` of degree `d`
  satisfying a third order linear equation; then `\\phi = \\theta + P'/P`
  and `\\omega` solves `\\omega^2 - \\phi \\omega + (\\phi'/2 + \\phi^2/2 - r) = 0`.
* **Case 3** (finite differential Galois group, `\\omega` of degree
  `n = 4, 6, 12`): sets `E_c` with twelve candidates per pole, `d = n
  (e_\\infty - \\sum e_c)/12`, and a polynomial `P` found through Kovacic's
  recursion `P_{i-1} = -S P_i' + ((n - i) S' - S \\theta) P_i - (n - i)(i +
  1) S^2 r P_{i+1}`; `\\omega` is a root of `\\sum_i S^i P_i \\omega^i/(n-i)!`.

If no case succeeds the equation has no Liouvillian solution. SymPy's
``dsolve`` has no such decision procedure: its hints cover constant
coefficients, Euler equations, a few named equations (Bessel, Airy,
hypergeometric) and the rational Riccati solver used in case 1.

References
==========

.. [Kovacic] J. Kovacic, An algorithm for solving second order linear
   homogeneous differential equations, Journal of Symbolic Computation 2
   (1986).
.. [Saunders] B. D. Saunders, An implementation of Kovacic's algorithm for
   solving second order linear homogeneous differential equations, SYMSAC
   1981.
"""
from __future__ import annotations

from itertools import product as cartesian
from typing import Optional, Sequence

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp, exp_polar
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve as _solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings

__all__ = ['liouvillian_solution', 'dsolve_kovacic', 'normal_form', 'KovacicSolution']


class KovacicSolution:
    """A Liouvillian solution ``exp(Integral(omega))`` of ``z'' = r z``.

    Attributes
    ==========

    case : int
        The case of Kovacic's algorithm (1, 2 or 3).
    omega : Expr or None
        ``omega``, when it can be written explicitly (always in cases 1
        and 2).
    omega_polynomial : Expr
        The polynomial in ``w`` (the symbol ``w``) with rational function
        coefficients whose root ``omega`` is.
    solution : Expr
        ``exp(Integral(omega))`` with the integral computed when
        possible.
    """

    def __init__(self, case: int, omega: Optional[Expr], omega_polynomial: Expr, solution: Expr) -> None:
        self.case = case
        self.omega = omega
        self.omega_polynomial = omega_polynomial
        self.solution = solution

    def __repr__(self) -> str:
        return "KovacicSolution(case=%d, %s)" % (self.case, self.solution)


def _rational(e: Expr, x: Symbol) -> tuple[Poly, Poly]:
    """Numerator and denominator polynomials of a rational function."""
    num, den = cancel(e).as_numer_denom()
    try:
        return Poly(num, x), Poly(den, x)
    except PolynomialError:
        raise ValueError("%s is not a rational function of %s" % (e, x))


def normal_form(p: Expr, q: Expr, x: Symbol) -> Expr:
    """``r`` with ``z'' = r z`` the normal form of ``y'' + p y' + q y = 0``
    (``y = z exp(-Integral(p)/2)``).

    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic import normal_form
    >>> normal_form(1/x, 1 - 1/(4*x**2), x)
    -1
    """
    return as_expr(cancel(p**2/4 + p.diff(x)/2 - q))


def _poles(den: Poly, x: Symbol) -> Optional[dict[Expr, int]]:
    """The poles with their orders, or ``None`` if they cannot be found
    exactly."""
    found = roots(den, x)
    if sum(found.values()) != den.degree():
        return None
    return {as_expr(c): int(m) for c, m in found.items()}


def _order_at_infinity(num: Poly, den: Poly) -> int:
    """The order of ``r`` at infinity: ``deg den - deg num`` (a pole of
    order ``-o`` when negative)."""
    return den.degree() - num.degree()


def _coefficient(r: Expr, c: Expr, x: Symbol) -> Expr:
    """The coefficient ``b`` of ``(x - c)**-2`` in the partial fraction
    expansion of ``r`` at a pole of order 2."""
    return as_expr(cancel(r*(x - c)**2).subs(x, c))


def _coefficient_at_infinity(num: Poly, den: Poly) -> Expr:
    """The coefficient ``b`` of ``x**-2`` in the expansion of ``r`` at
    infinity when the order there is 2."""
    return as_expr(num.LC()/den.LC())


def _integers(candidates: Sequence[Expr]) -> list[int]:
    result: list[int] = []
    for c in candidates:
        value = simplify(c)
        if isinstance(value, Integer) and int(value) not in result:
            result.append(int(value))
    return result


# ---------------------------------------------------------------------------
# case 1

def _case1(r: Expr, num: Poly, den: Poly, poles: dict[Expr, int], x: Symbol) -> Optional[KovacicSolution]:
    """Kovacic's case 1: ``omega`` rational, from the Laurent parts of
    ``sqrt(r)`` at the poles and at infinity and the exponents
    ``alpha``; SymPy's rational Riccati solver is tried as well."""
    result = _case1_kovacic(r, num, den, poles, x)
    if result is not None:
        return result
    return _case1_sympy(r, x)


def _laurent_sqrt(r: Expr, c: Expr, order: int, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``([sqrt(r)]_c, alpha data)`` at a pole of even order ``2 nu >= 4``:
    the polar part of the Laurent expansion of ``sqrt(r)`` at ``c`` and
    the coefficient ``b`` of ``(x - c)**(-nu - 1)`` in ``r - [sqrt r]_c**2``."""
    nu = order//2
    t = Dummy('t')
    shifted = as_expr(r.subs(x, c + t))
    expansion = attempt(lambda: shifted.series(t, 0, 1).removeO(), settings.timeout)
    if expansion is None:
        return None
    # r = a**2/t**(2 nu) (1 + ...): sqrt as a Laurent polynomial in t
    root = attempt(lambda: sqrt(shifted).series(t, 0, 0).removeO(), settings.timeout)
    if root is None:
        return None
    polar = as_expr(Add(*[term for term in Add.make_args(as_expr(root)) if Poly(term*t**nu, t).degree() < nu]))
    if polar == 0:
        return None
    a = as_expr(polar.coeff(t, -nu))
    rest = as_expr(expand(shifted - polar**2))
    b = as_expr(rest.coeff(t, -nu - 1))
    return as_expr(polar.subs(t, x - c)), as_expr(b/a)


def _case1_kovacic(r: Expr, num: Poly, den: Poly, poles: dict[Expr, int], x: Symbol) -> Optional[KovacicSolution]:
    # local data at the poles: [sqrt r]_c and alpha_c^+-
    local: list[tuple[Expr, Expr, Expr, Expr]] = []  # (c, [sqrt r]_c, alpha+, alpha-)
    for c, order in poles.items():
        if order == 1:
            local.append((c, S.Zero, S.One, S.One))
        elif order == 2:
            b = _coefficient(r, c, x)
            root = sqrt(1 + 4*b)
            local.append((c, S.Zero, as_expr((1 + root)/2), as_expr((1 - root)/2)))
        elif order % 2 == 0:
            data = _laurent_sqrt(r, c, order, x)
            if data is None:
                return None
            polar, ratio = data
            nu = order//2
            local.append((c, polar, as_expr((ratio + nu)/2), as_expr((-ratio + nu)/2)))
        else:
            return None
    # at infinity
    o = _order_at_infinity(num, den)
    if o > 2:
        infinity: tuple[Expr, Expr, Expr] = (S.Zero, S.Zero, S.One)
    elif o == 2:
        b = _coefficient_at_infinity(num, den)
        root = sqrt(1 + 4*b)
        infinity = (S.Zero, as_expr((1 + root)/2), as_expr((1 - root)/2))
    elif o % 2 == 0:
        nu = -o//2
        t = Dummy('t')
        at_infinity = as_expr(r.subs(x, 1/t))
        root = attempt(lambda: sqrt(at_infinity).series(t, 0, 1).removeO(), settings.timeout)
        if root is None:
            return None
        polynomial_part = as_expr(Add(*[term for term in Add.make_args(as_expr(root))
                                        if Poly(term*t**nu, t).degree() <= nu]))
        if polynomial_part == 0:
            return None
        a = as_expr(polynomial_part.coeff(t, -nu))
        rest = as_expr(expand(at_infinity - polynomial_part**2))
        b = as_expr(rest.coeff(t, -nu + 1))
        infinity = (as_expr(polynomial_part.subs(t, 1/x)), as_expr((b/a - nu)/2), as_expr((-b/a - nu)/2))
    else:
        return None
    from itertools import product as cartesian_
    signs = list(cartesian_(*([(1, -1)]*(len(local) + 1))))
    for choice in signs:
        s_inf = choice[-1]
        alpha_inf = infinity[1] if s_inf == 1 else infinity[2]
        d = as_expr(alpha_inf - Add(*[(entry[2] if s == 1 else entry[3]) for entry, s in zip(local, choice)]))
        d = as_expr(simplify(d))
        if not (isinstance(d, Integer) and int(d) >= 0):
            continue
        omega = as_expr(s_inf*infinity[0] + Add(*[s*entry[1] + (entry[2] if s == 1 else entry[3])/(x - entry[0])
                                                    for entry, s in zip(local, choice)]))
        omega = as_expr(cancel(omega))
        P, unknowns = _monic_polynomial(int(d), x)
        equation = as_expr(P.diff(x, 2) + 2*omega*P.diff(x) + (omega.diff(x) + omega**2 - r)*P)
        values = _solve_linear(equation, unknowns, x)
        if values is None:
            continue
        P_value = as_expr(P.xreplace(values))
        full = as_expr(cancel(omega + P_value.diff(x)/P_value))
        if cancel(full.diff(x) + full**2 - r) != 0:
            continue
        w = Symbol('w')
        return KovacicSolution(1, full, as_expr(w - full), _exp_integral(full, x))
    return None


def _case1_sympy(r: Expr, x: Symbol) -> Optional[KovacicSolution]:
    from sympy.solvers.ode.riccati import solve_riccati
    f = Function('f')(x)
    # omega' = r - omega**2
    solutions = attempt(lambda: solve_riccati(f, x, r, S.Zero, S.NegativeOne), settings.timeout)
    if not solutions:
        return None
    for sol in solutions:
        omega = as_expr(cancel(sol.rhs if isinstance(sol, Eq) else sol))
        if free_symbols(omega) - {x}:
            continue
        if cancel(omega.diff(x) + omega**2 - r) != 0:
            continue
        w = Symbol('w')
        return KovacicSolution(1, omega, as_expr(w - omega), _exp_integral(omega, x))
    return None


def _exp_integral(omega: Expr, x: Symbol) -> Expr:
    integral = attempt(lambda: integrate(omega, x, conds='none'), settings.timeout)
    if integral is None or integral.has(Integral, Piecewise, exp_polar):
        return exp(Integral(omega, x))
    return as_expr(exp(as_expr(integral)))


# ---------------------------------------------------------------------------
# cases 2 and 3

def _sets(r: Expr, num: Poly, den: Poly, poles: dict[Expr, int], x: Symbol, n: int
          ) -> Optional[tuple[dict[Expr, list[int]], list[int]]]:
    """The sets ``E_c`` of the poles and ``E_oo`` for case 2 (``n = 2``)
    or case 3 (``n = 4, 6, 12``)."""
    E: dict[Expr, list[int]] = {}
    for c, order in poles.items():
        if order == 1:
            E[c] = [4] if n == 2 else [12]
        elif order == 2:
            b = _coefficient(r, c, x)
            root = sqrt(1 + 4*b)
            if n == 2:
                E[c] = _integers([2 + k*root for k in (0, 2, -2)])
            else:
                E[c] = _integers([6 + Rational(12, n)*k*root for k in range(-n//2, n//2 + 1)])
        else:
            if n != 2:
                return None
            E[c] = [order]
        if not E[c]:
            return None
    o = _order_at_infinity(num, den)
    if o > 2:
        E_inf = [0, 2, 4] if n == 2 else [12]
    elif o == 2:
        b = _coefficient_at_infinity(num, den)
        root = sqrt(1 + 4*b)
        if n == 2:
            E_inf = _integers([2 + k*root for k in (0, 2, -2)])
        else:
            E_inf = _integers([6 + Rational(12, n)*k*root for k in range(-n//2, n//2 + 1)])
    else:
        if n != 2:
            return None
        E_inf = [o]
    if not E_inf:
        return None
    return E, E_inf


def _monic_polynomial(d: int, x: Symbol) -> tuple[Expr, list[Symbol]]:
    coefficients: list[Symbol] = [Dummy('p%d' % i) for i in range(d)]
    P = as_expr(x**d + Add(*[c*x**i for i, c in enumerate(coefficients)]))
    return P, coefficients


def _solve_linear(equation: Expr, unknowns: list[Symbol], x: Symbol) -> Optional[dict[Basic, Basic]]:
    """Coefficients making a polynomial in ``x`` with coefficients linear
    in the unknowns vanish."""
    num = cancel(equation).as_numer_denom()[0]
    try:
        conditions = Poly(num, x).all_coeffs()
    except PolynomialError:
        return None
    conditions = [as_expr(c) for c in conditions if c != 0]
    if not conditions:
        return {}
    if not unknowns:
        return None
    solved = _solve(conditions, unknowns, dict=True)
    if not solved:
        return None
    result: dict[Basic, Basic] = dict(solved[0])
    # free unknowns can be anything: take zero
    for u in unknowns:
        result.setdefault(u, S.Zero)
    return result


def _case2(r: Expr, num: Poly, den: Poly, poles: dict[Expr, int], x: Symbol) -> Optional[KovacicSolution]:
    sets = _sets(r, num, den, poles, x, 2)
    if sets is None:
        return None
    E, E_inf = sets
    names = list(E)
    for e_inf in E_inf:
        for family in cartesian(*[E[c] for c in names]):
            d2 = e_inf - sum(family)
            if d2 < 0 or d2 % 2:
                continue
            d = d2//2
            theta = as_expr(Add(*[Rational(e, 2)/(x - c) for e, c in zip(family, names)]))
            P, unknowns = _monic_polynomial(d, x)
            equation = (P.diff(x, 3) + 3*theta*P.diff(x, 2)
                        + (3*theta**2 + 3*theta.diff(x) - 4*r)*P.diff(x)
                        + (theta.diff(x, 2) + 3*theta*theta.diff(x) + theta**3 - 4*r*theta - 2*r.diff(x))*P)
            values = _solve_linear(as_expr(equation), unknowns, x)
            if values is None:
                continue
            P_value = as_expr(P.xreplace(values))
            phi = as_expr(cancel(theta + P_value.diff(x)/P_value))
            w = Symbol('w')
            polynomial = as_expr(w**2 - phi*w + cancel(phi.diff(x)/2 + phi**2/2 - r))
            discriminant = as_expr(cancel(phi**2 - 4*(phi.diff(x)/2 + phi**2/2 - r)))
            omega = as_expr((phi + sqrt(discriminant))/2)
            return KovacicSolution(2, omega, polynomial, _exp_integral(omega, x))
    return None


def _case3(r: Expr, num: Poly, den: Poly, poles: dict[Expr, int], x: Symbol) -> Optional[KovacicSolution]:
    if any(order > 2 for order in poles.values()) or _order_at_infinity(num, den) < 2:
        return None
    for n in (4, 6, 12):
        sets = _sets(r, num, den, poles, x, n)
        if sets is None:
            continue
        E, E_inf = sets
        names = list(E)
        S_poly = as_expr(Mul(*[x - c for c in names]))
        for e_inf in E_inf:
            for family in cartesian(*[E[c] for c in names]):
                d12 = n*(e_inf - sum(family))
                if d12 < 0 or d12 % 12:
                    continue
                d = d12//12
                theta = as_expr(Rational(n, 12)*Add(*[Integer(e)/(x - c) for e, c in zip(family, names)]))
                P, unknowns = _monic_polynomial(d, x)
                # Kovacic's recursion, P_n = -P down to P_{-1} which must vanish
                polynomials: dict[int, Expr] = {n: -P}
                for i in range(n, -1, -1):
                    P_i = polynomials[i]
                    P_next = polynomials.get(i + 1, S.Zero)
                    polynomials[i - 1] = as_expr(cancel(-S_poly*P_i.diff(x) + ((n - i)*S_poly.diff(x) - S_poly*theta)*P_i
                                                        - (n - i)*(i + 1)*S_poly**2*r*P_next))
                values = _solve_linear(polynomials[-1], unknowns, x)
                if values is None:
                    continue
                w = Symbol('w')
                from sympy.functions.combinatorial.factorials import factorial
                polynomial = as_expr(cancel(Add(*[S_poly**i*polynomials[i].xreplace(values)/factorial(n - i)*w**i
                                                  for i in range(n + 1)])))
                omega = _algebraic_root(polynomial, w, x)
                solution = exp(Integral(omega, x)) if omega is not None else exp(Integral(_RootOfPlaceholder(polynomial, w), x))
                return KovacicSolution(3, omega, polynomial, solution)
    return None


def _RootOfPlaceholder(polynomial: Expr, w: Symbol) -> Expr:
    """``omega`` as the unevaluated root of its polynomial."""
    root = Function('omega')
    return as_expr(root(polynomial))


def _algebraic_root(polynomial: Expr, w: Symbol, x: Symbol) -> Optional[Expr]:
    """A root of a polynomial in ``w`` with rational function coefficients,
    in radicals when SymPy finds them."""
    from sympy.functions.elementary.piecewise import Piecewise
    limit_ = None if settings.timeout is None else min(settings.timeout, 5.0)
    found = attempt(lambda: roots(Poly(polynomial, w)), limit_)
    if not found:
        return None
    for root in found:
        if not root.has(Piecewise):
            return as_expr(root)
    return None


# ---------------------------------------------------------------------------

def liouvillian_solution(r: Expr, x: Symbol) -> Optional[KovacicSolution]:
    """A Liouvillian solution of ``z'' = r z`` (``r`` a rational function
    of ``x``) by Kovacic's algorithm, or ``None`` when there is none.

    Examples
    ========

    >>> from sympy import Rational
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic import liouvillian_solution
    >>> liouvillian_solution(1 + 2/x**2, x)
    KovacicSolution(case=1, (x - 1)*exp(x)/x)
    >>> liouvillian_solution(-3/(16*x**2) + 1/x, x).case
    2
    >>> liouvillian_solution(x, x) is None
    True
    """
    r = as_expr(r)
    num, den = _rational(r, x)
    if num.is_zero:
        return KovacicSolution(1, S.Zero, as_expr(Symbol('w')), S.One)
    poles = _poles(den, x)
    if poles is None:
        return None
    orders = list(poles.values())
    o = _order_at_infinity(num, den)
    # the necessary conditions of Kovacic's theorem
    if all(k == 1 or k % 2 == 0 for k in orders) and (o % 2 == 0 or o > 2):
        result = _case1(r, num, den, poles, x)
        if result is not None:
            return result
    result = _case2(r, num, den, poles, x)
    if result is not None:
        return result
    return _case3(r, num, den, poles, x)


def _second_solution(y1: Expr, p: Expr, x: Symbol) -> Optional[Expr]:
    """The second solution by reduction of order:
    ``y2 = y1 Integral(exp(-Integral(p))/y1**2)``, in closed form when
    SymPy integrates it (without conditions on the parameters) and the
    result verifies, otherwise with the integral left unevaluated."""
    inner = attempt(lambda: integrate(p, x, conds='none'), settings.timeout)
    if inner is None or inner.has(Integral):
        return None
    integrand = as_expr(simplify(exp(-inner)/y1**2))
    outer = attempt(lambda: integrate(integrand, x, conds='none'), settings.timeout)
    if outer is not None and not outer.has(Integral, Piecewise, exp_polar):
        candidate = as_expr(simplify(y1*outer))
        # verified against the original equation y'' + p y' + q y = 0 through
        # the Wronskian: (y2/y1)' = exp(-Integral(p))/y1**2
        difference = attempt(lambda: simplify(as_expr((candidate/y1).diff(x) - integrand)), settings.timeout)
        if difference == 0:
            return candidate
    return as_expr(y1*Integral(integrand, x))


def dsolve_kovacic(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    """The Liouvillian solutions of a second order linear homogeneous
    equation ``a y'' + b y' + c y = 0`` with rational coefficients: a list
    with one or two independent solutions, or ``None`` when there is no
    Liouvillian solution (or the equation is not of that form).

    Examples
    ========

    >>> from sympy import Function, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.kovacic import dsolve_kovacic
    >>> y = Function('y')(x)
    >>> dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
    [exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
    >>> dsolve_kovacic(y.diff(x, 2) - x*y, y) is None
    True
    """
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    x = x_
    lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
    expanded = expand(as_expr(lhs))
    d2, d1, d0 = Derivative(f, (x, 2)), Derivative(f, x), f
    a = as_expr(expanded.coeff(d2))
    rest = as_expr(expanded - a*d2)
    b = as_expr(rest.coeff(d1))
    rest = as_expr(rest - b*d1)
    c = as_expr(rest.coeff(d0))
    if a == 0 or as_expr(expand(rest - c*d0)) != 0 or any(e.has(f) for e in (a, b, c)):
        return None
    for e in (a, b, c):
        if not e.is_rational_function(x):
            return None
    p = as_expr(cancel(b/a))
    q = as_expr(cancel(c/a))
    r = normal_form(p, q, x)
    z = liouvillian_solution(r, x)
    if z is None or z.omega is None:
        return None
    integral_p = attempt(lambda: integrate(p, x), settings.timeout)
    if integral_p is None or integral_p.has(Integral):
        return None
    factor_ = exp(-as_expr(integral_p)/2)
    y1 = _normalized(as_expr(simplify(z.solution*factor_)), x)
    solutions = [y1]
    if z.case == 3:
        return [as_expr(z.solution*factor_)]
    if z.case == 2 and z.omega is not None:
        # the conjugate root gives the second solution
        w = Symbol('w')
        others = [as_expr(root) for root in roots(Poly(z.omega_polynomial, w)) if as_expr(root) != z.omega]
        for other in others:
            y2 = _normalized(as_expr(simplify(_exp_integral(other, x)*factor_)), x)
            if simplify(y2/y1).is_constant(x) is not True:
                solutions.append(y2)
                break
    if len(solutions) == 1:
        second = _second_solution(y1, p, x)
        if second is not None:
            solutions.append(_normalized(second, x))
    return solutions


def _normalized(y: Expr, x: Symbol) -> Expr:
    """``y`` without its constant factor."""
    constant, rest = y.as_independent(x, as_Add=False)
    if constant != 0 and rest != 0 and not constant.has(x):
        return as_expr(rest)
    return y
