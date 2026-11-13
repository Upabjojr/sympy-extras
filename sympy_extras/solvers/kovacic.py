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
  Riccati equation `\\omega' + \\omega^2 = r`, found from the Laurent parts
  of `\\sqrt r` at the poles and at infinity, and else from the
  hyperexponential solutions of `z'' = r z` (:func:`rational_riccati`,
  in place of SymPy's rational Riccati solver). The second solution
  follows by reduction of order.
* **Case 2** (`\\omega` quadratic): for each pole and for infinity a finite
  set `E_c` of integers is built from the order of the pole and the
  coefficient of `(x - c)^{-2}`; the families `(e_c)` with
  `d = (e_\\infty - \\sum e_c)/2` a nonnegative integer give a candidate
  `\\theta = \\sum e_c/(2(x - c))` and a monic polynomial `P` of degree `d`
  satisfying a third order linear equation; then `\\phi = \\theta + P'/P`
  and `\\omega` solves `\\omega^2 - \\phi \\omega + (\\phi'/2 + \\phi^2/2 - r) = 0`;
  the other root, `\\phi - \\omega`, gives the second solution.
* **Case 3** (finite differential Galois group, `\\omega` of degree
  `n = 4, 6, 12`): sets `E_c` with twelve candidates per pole, `d = n
  (e_\\infty - \\sum e_c)/12`, and a polynomial `P` found through Kovacic's
  recursion `P_{i-1} = -S P_i' + ((n - i) S' - S \\theta) P_i - (n - i)(i +
  1) S^2 r P_{i+1}`; `\\omega` is a root of `\\sum_i S^i P_i \\omega^i/(n-i)!`.

If no case succeeds the equation has no Liouvillian solution. SymPy's
``dsolve`` has no such decision procedure: its hints cover constant
coefficients, Euler equations, a few named equations (Bessel, Airy,
hypergeometric) and the rational Riccati solver used in case 1.

The module works on the objects of the core and imports no SymPy: the
polynomials, their roots, the series, the linear systems and the
simplifications are the native ones, and
:mod:`sympy_extras.solvers.kovacic_sympy` is its face on SymPy objects.
The antiderivatives come from the native verified indefinite
integration (:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`, SymPy's routes
its last methods), the integrals it cannot do are left unevaluated (the
node :class:`~sympy_extras.integrals.integral.Integral`), and the
fallback of case 1, the rational solutions of the Riccati equation,
comes from the native hyperexponential solutions
(:func:`~sympy_extras.solvers.linear_ode.hyperexponential_solutions`). The cases of the isolated values of the
parameters are those of :mod:`sympy_extras.solvers._special_values`.

References
==========

.. [Kovacic] J. Kovacic, An algorithm for solving second order linear
   homogeneous differential equations, Journal of Symbolic Computation 2
   (1986).
.. [Saunders] B. D. Saunders, An implementation of Kovacic's algorithm for
   solving second order linear homogeneous differential equations, SYMSAC
   1981.

Examples
========

>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.kovacic import normal_form, liouvillian_solution
>>> x = Symbol('x')
>>> normal_form(1/x, 1 - 1/(4*x**2), x)
-1
>>> liouvillian_solution(x, x) is None
True
"""
from __future__ import annotations

from itertools import product as cartesian
from typing import Final, Optional, Sequence

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Function, expand
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import factorial
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.functions.elementary.piecewise import Piecewise
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.simplify.simplify import simplify
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

__all__ = ['liouvillian_solution', 'dsolve_kovacic', 'normal_form', 'KovacicSolution', 'rational_riccati']

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)

_ZERO = Integer(0)
_ONE = Integer(1)


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
        self.case: Final[int] = case
        self.omega: Final[Optional[Expr]] = omega
        self.omega_polynomial: Final[Expr] = omega_polynomial
        self.solution: Final[Expr] = solution

    def __repr__(self) -> str:
        return "KovacicSolution(case=%d, %s)" % (self.case, self.solution)


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _sum(terms: Sequence[Expr]) -> Expr:
    total: Expr = _ZERO
    for t in terms:
        total = total + t
    return total


def _product(factors: Sequence[Expr]) -> Expr:
    total: Expr = _ONE
    for f in factors:
        total = total*f
    return total


# ---------------------------------------------------------------------------
# polynomials in one variable with any coefficients

def _monomial_degree(e: Expr, x: Symbol) -> Optional[int]:
    """``k`` for ``e = x**k`` (``0`` for ``1``), ``None`` otherwise."""
    if e == _ONE:
        return 0
    if e == x:
        return 1
    if isinstance(e, Pow) and e.base == x and isinstance(e.exp, Integer):
        return int(e.exp)
    return None


def _coefficients(e: Expr, x: Symbol) -> dict[int, Expr]:
    """The nonzero coefficients of the expanded ``e`` as a polynomial in
    ``x`` (the coefficients any expressions free of ``x``: SymPy's
    ``Poly(e, x)`` takes the irrational numbers, ``I`` and the functions
    of the parameters as generators of its domain); ``ValueError`` when
    ``e`` is no polynomial in ``x``."""
    collected: dict[int, list[Expr]] = {}
    for term in Add.make_args(expand(e)):
        if term == _ZERO:
            continue
        coefficient, rest = term.as_independent(x, as_Add=False)
        k = _monomial_degree(rest, x)
        if k is None or k < 0:
            raise ValueError("%s is not a polynomial in %s" % (e, x))
        collected.setdefault(k, []).append(coefficient)
    result: dict[int, Expr] = {}
    for k, parts in collected.items():
        c = _sum(parts)
        if c != _ZERO:
            result[k] = c
    return result


class _Polynomial:
    """A polynomial in ``x`` given by its coefficients (what Kovacic's
    algorithm asks of SymPy's ``Poly``: the degree, the leading
    coefficient, the coefficients)."""

    def __init__(self, e: Expr, x: Symbol) -> None:
        self.expr: Final[Expr] = e
        self.x: Final[Symbol] = x
        self.coefficients: Final[dict[int, Expr]] = _coefficients(e, x)

    @property
    def is_zero(self) -> bool:
        return not self.coefficients

    def degree(self) -> int:
        """The degree (``-1`` for zero, SymPy's ``-oo``, which the
        algorithm never asks for)."""
        return max(self.coefficients) if self.coefficients else -1

    def LC(self) -> Expr:
        return self.coefficients[self.degree()] if self.coefficients else _ZERO

    def all_coeffs(self) -> list[Expr]:
        """The coefficients from the leading one down (SymPy's order)."""
        n = self.degree()
        return [self.coefficients.get(k, _ZERO) for k in range(n, -1, -1)]


def _generic_coefficients(e: Expr, x: Symbol) -> tuple[Expr, dict[Basic, Basic]]:
    """``e`` with the numbers which are not rational (``sqrt(2)``, ``I``,
    ``pi``) and the functions of the parameters (``sqrt(a)``, ``exp(a)``)
    in its coefficients replaced by symbols, as SymPy's ``Poly`` takes
    them as generators of its domain, and the replacements back."""
    forward: dict[Expr, Expr] = {}
    back: dict[Basic, Basic] = {}

    def walk(node: Expr) -> Expr:
        if isinstance(node, (Symbol, Rational)):
            return node
        if isinstance(node, Add):
            return _sum([walk(a) for a in node.args if isinstance(a, Expr)])
        if isinstance(node, Mul):
            return _product([walk(a) for a in node.args if isinstance(a, Expr)])
        if isinstance(node, Pow) and isinstance(node.exp, Integer) and int(node.exp) >= 0:
            return walk(node.base)**node.exp
        if node.has(x):
            return node   # no polynomial in x: Poly refuses it
        if node not in forward:
            d = Dummy('g%d' % len(forward))
            forward[node] = d
            back[d] = node
        return forward[node]

    return walk(expand(e)), back


def _roots(e: Expr, x: Symbol) -> dict[Expr, int]:
    """The roots of a polynomial in ``x`` which SymPy's ``roots`` finds,
    with their multiplicities, on the native ``roots``; coefficients with
    irrational numbers or functions of the parameters are taken as
    generators of the domain, as SymPy does."""
    try:
        found = roots(Poly(e, x))
        if isinstance(found, dict):
            return found
        return {}
    except BasePolynomialError:
        pass
    generic, back = _generic_coefficients(e, x)
    found = roots(Poly(generic, x))
    if not isinstance(found, dict):
        return {}
    result: dict[Expr, int] = {}
    for r, m in found.items():
        value = _replaced(r, back)
        result[value] = result.get(value, 0) + m
    return result


# ---------------------------------------------------------------------------

def _rational(e: Expr, x: Symbol) -> tuple[_Polynomial, _Polynomial]:
    """Numerator and denominator polynomials of a rational function."""
    num, den = cancel(e).as_numer_denom()
    try:
        return _Polynomial(num, x), _Polynomial(den, x)
    except ValueError:
        raise ValueError("%s is not a rational function of %s" % (e, x))


def normal_form(p: Expr, q: Expr, x: Symbol) -> Expr:
    """``r`` with ``z'' = r z`` the normal form of ``y'' + p y' + q y = 0``
    (``y = z exp(-Integral(p)/2)``).

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.kovacic import normal_form
    >>> x = Symbol('x')
    >>> normal_form(1/x, 1 - 1/(4*x**2), x)
    -1
    """
    return cancel(p**2/4 + p.diff(x)/2 - q)


def _poles(den: _Polynomial, x: Symbol) -> Optional[dict[Expr, int]]:
    """The poles with their orders, or ``None`` if they cannot be found
    exactly."""
    found = _roots(den.expr, x)
    if sum(found.values()) != den.degree():
        return None
    return found


def _order_at_infinity(num: _Polynomial, den: _Polynomial) -> int:
    """The order of ``r`` at infinity: ``deg den - deg num`` (a pole of
    order ``-o`` when negative)."""
    return den.degree() - num.degree()


def _coefficient(r: Expr, c: Expr, x: Symbol) -> Expr:
    """The coefficient ``b`` of ``(x - c)**-2`` in the partial fraction
    expansion of ``r`` at a pole of order 2."""
    value = cancel(r*(x - c)**2).subs(x, c)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _coefficient_at_infinity(num: _Polynomial, den: _Polynomial) -> Expr:
    """The coefficient ``b`` of ``x**-2`` in the expansion of ``r`` at
    infinity when the order there is 2."""
    return num.LC()/den.LC()


def _as_number(e: Expr) -> Expr:
    """``e`` simplified when it may be a number: an expression which keeps
    a symbol after its expansion (``2 + 2*sqrt(4*n**2)``) is no integer,
    and SymPy's ``simplify`` of it was the slow step of the equations with
    parameters."""
    expanded = expand(e)
    if expanded.free_symbols:
        return expanded
    return simplify(expanded)


def _integers(candidates: Sequence[Expr]) -> list[int]:
    result: list[int] = []
    for c in candidates:
        value = _as_number(c)
        if isinstance(value, Integer) and int(value) not in result:
            result.append(int(value))
    return result


def _t_degree(term: Expr, t: Symbol) -> int:
    """The exponent of ``t`` in a term ``c t**k`` of a Laurent
    polynomial."""
    _, rest = term.as_independent(t, as_Add=False)
    if rest == _ONE:
        return 0
    if rest == t:
        return 1
    if isinstance(rest, Pow) and rest.base == t and isinstance(rest.exp, Integer):
        return int(rest.exp)
    raise ValueError("%s is not a term of a Laurent polynomial in %s" % (term, t))


# ---------------------------------------------------------------------------
# case 1

def _case1(r: Expr, num: _Polynomial, den: _Polynomial, poles: dict[Expr, int], x: Symbol
           ) -> Optional[KovacicSolution]:
    """Kovacic's case 1: ``omega`` rational, from the Laurent parts of
    ``sqrt(r)`` at the poles and at infinity and the exponents
    ``alpha``; the rational Riccati solutions of :func:`rational_riccati`
    are tried as well."""
    result = _case1_kovacic(r, num, den, poles, x)
    if result is not None:
        return result
    return _case1_riccati(r, x)


def _laurent_sqrt(r: Expr, c: Expr, order: int, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``([sqrt(r)]_c, alpha data)`` at a pole of even order ``2 nu >= 4``:
    the polar part of the Laurent expansion of ``sqrt(r)`` at ``c`` and
    the coefficient ``b`` of ``(x - c)**(-nu - 1)`` in ``r - [sqrt r]_c**2``."""
    nu = order//2
    t = Dummy('t')
    shifted = _replaced(r, {x: c + t})
    expansion = attempt(lambda: shifted.series(t, 0, 1).removeO(), settings.timeout, native=True)
    if expansion is None:
        return None
    # r = a**2/t**(2 nu) (1 + ...): sqrt as a Laurent polynomial in t
    root = attempt(lambda: sqrt(shifted).series(t, 0, 0).removeO(), settings.timeout, native=True)
    if root is None:
        return None
    polar = _sum([term for term in Add.make_args(root) if _t_degree(term, t) + nu < nu])
    if polar == _ZERO:
        return None
    a = polar.coeff(t, -nu)
    rest = expand(shifted - polar**2)
    b = rest.coeff(t, -nu - 1)
    return _replaced(polar, {t: x - c}), b/a


def _case1_kovacic(r: Expr, num: _Polynomial, den: _Polynomial, poles: dict[Expr, int], x: Symbol
                   ) -> Optional[KovacicSolution]:
    # local data at the poles: [sqrt r]_c and alpha_c^+-
    local: list[tuple[Expr, Expr, Expr, Expr]] = []  # (c, [sqrt r]_c, alpha+, alpha-)
    for c, order in poles.items():
        if order == 1:
            local.append((c, _ZERO, _ONE, _ONE))
        elif order == 2:
            b = _coefficient(r, c, x)
            root = sqrt(1 + 4*b)
            local.append((c, _ZERO, (1 + root)/2, (1 - root)/2))
        elif order % 2 == 0:
            data = _laurent_sqrt(r, c, order, x)
            if data is None:
                return None
            polar, ratio = data
            nu = order//2
            local.append((c, polar, (ratio + nu)/2, (-ratio + nu)/2))
        else:
            return None
    # at infinity
    o = _order_at_infinity(num, den)
    if o > 2:
        infinity: tuple[Expr, Expr, Expr] = (_ZERO, _ZERO, _ONE)
    elif o == 2:
        b = _coefficient_at_infinity(num, den)
        root = sqrt(1 + 4*b)
        infinity = (_ZERO, (1 + root)/2, (1 - root)/2)
    elif o % 2 == 0:
        nu = -o//2
        t = Dummy('t')
        at_infinity = _replaced(r, {x: 1/t})
        found = attempt(lambda: sqrt(at_infinity).series(t, 0, 1).removeO(), settings.timeout, native=True)
        if found is None:
            return None
        polynomial_part = _sum([term for term in Add.make_args(found) if _t_degree(term, t) + nu <= nu])
        if polynomial_part == _ZERO:
            return None
        a = polynomial_part.coeff(t, -nu)
        rest = expand(at_infinity - polynomial_part**2)
        b = rest.coeff(t, -nu + 1)
        infinity = (_replaced(polynomial_part, {t: 1/x}), (b/a - nu)/2, (-b/a - nu)/2)
    else:
        return None
    signs = list(cartesian(*([(1, -1)]*(len(local) + 1))))
    for choice in signs:
        s_inf = choice[-1]
        alpha_inf = infinity[1] if s_inf == 1 else infinity[2]
        d = _as_number(alpha_inf - _sum([(entry[2] if s == 1 else entry[3]) for entry, s in zip(local, choice)]))
        if not (isinstance(d, Integer) and int(d) >= 0):
            continue
        omega = cancel(s_inf*infinity[0] + _sum([s*entry[1] + (entry[2] if s == 1 else entry[3])/(x - entry[0])
                                                  for entry, s in zip(local, choice)]))
        P, unknowns = _monic_polynomial(int(d), x)
        equation = P.diff(x, 2) + 2*omega*P.diff(x) + (omega.diff(x) + omega**2 - r)*P
        values = _solve_linear(equation, unknowns, x)
        if values is None:
            continue
        P_value = _replaced(P, values)
        full = cancel(omega + P_value.diff(x)/P_value)
        if cancel(full.diff(x) + full**2 - r) != _ZERO:
            continue
        w = Symbol('w')
        return KovacicSolution(1, full, w - full, _exp_integral(full, x))
    return None


def rational_riccati(r: Expr, x: Symbol) -> list[Expr]:
    """The rational solutions ``omega`` of ``omega' + omega**2 = r`` (``r``
    rational in ``x``) given by the hyperexponential solutions ``z =
    exp(Integral(omega))`` of ``z'' = r z``
    (:func:`~sympy_extras.solvers.linear_ode.hyperexponential_solutions`,
    the native counterpart of SymPy's ``solve_riccati``), each checked.

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.kovacic import rational_riccati
    >>> x = Symbol('x')
    >>> rational_riccati(2/x**2, x)
    [-1/x, 2/x]
    """
    from .linear_ode import LinearOperator, hyperexponential_solutions, log_derivative
    try:
        L = LinearOperator.from_equation(_equation_of(r, x), _unknown(x))
        found = hyperexponential_solutions(L)
    except _FAILURES:
        return []
    solutions: list[Expr] = []
    for z in found:
        omega = log_derivative(z, x)
        try:
            if cancel(omega.diff(x) + omega**2 - r) != _ZERO:
                continue
        except _FAILURES:
            continue
        if omega not in solutions:
            solutions.append(omega)
    return solutions


def _unknown(x: Symbol) -> AppliedUndef:
    return Function('_z')(x)


def _equation_of(r: Expr, x: Symbol) -> Expr:
    z = _unknown(x)
    return z.diff(x, 2) - r*z


def _case1_riccati(r: Expr, x: Symbol) -> Optional[KovacicSolution]:
    solutions = attempt(lambda: rational_riccati(r, x), settings.timeout, native=True)
    if not solutions:
        return None
    for solution in solutions:
        omega = cancel(solution)
        if {s for s in omega.free_symbols if isinstance(s, Symbol)} - {x}:
            continue
        if cancel(omega.diff(x) + omega**2 - r) != _ZERO:
            continue
        w = Symbol('w')
        return KovacicSolution(1, omega, w - omega, _exp_integral(omega, x))
    return None


def _exp_integral(omega: Expr, x: Symbol) -> Expr:
    from sympy_extras.integrals.indefinite import closed_antiderivative
    from sympy_extras.integrals.integral import Integral
    integral = closed_antiderivative(omega, x, closed=True)
    if integral is None:
        return exp(Integral(omega, x))
    return exp(integral)


# ---------------------------------------------------------------------------
# cases 2 and 3

def _sets(r: Expr, num: _Polynomial, den: _Polynomial, poles: dict[Expr, int], x: Symbol, n: int
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
    P = x**d + _sum([c*x**i for i, c in enumerate(coefficients)])
    return P, coefficients


def _solve_linear(equation: Expr, unknowns: list[Symbol], x: Symbol) -> Optional[dict[Basic, Basic]]:
    """Coefficients making a polynomial in ``x`` with coefficients linear
    in the unknowns vanish (the unknowns left free are zero)."""
    from .recurr import _solve_linear as solve_rows
    num = cancel(equation).as_numer_denom()[0]
    try:
        conditions = [c for c in _Polynomial(num, x).all_coeffs() if c != _ZERO]
    except ValueError:
        return None
    if not conditions:
        return {}
    if not unknowns:
        return None
    zeros: dict[Basic, Basic] = {u: _ZERO for u in unknowns}
    rows: list[list[Expr]] = []
    rhs: list[Expr] = []
    for condition in conditions:
        row = [condition.diff(u) for u in unknowns]
        if any(entry.has(u) for entry in row for u in unknowns):
            return None   # not linear in the unknowns
        rows.append(row)
        rhs.append(-_replaced(condition, zeros))
    solution = solve_rows(rows, rhs, len(unknowns))
    if solution is None:
        return None
    return {u: v for u, v in zip(unknowns, solution)}


def _case2(r: Expr, num: _Polynomial, den: _Polynomial, poles: dict[Expr, int], x: Symbol
           ) -> Optional[KovacicSolution]:
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
            theta = _sum([Rational(e, 2)/(x - c) for e, c in zip(family, names)])
            P, unknowns = _monic_polynomial(d, x)
            equation = (P.diff(x, 3) + 3*theta*P.diff(x, 2)
                        + (3*theta**2 + 3*theta.diff(x) - 4*r)*P.diff(x)
                        + (theta.diff(x, 2) + 3*theta*theta.diff(x) + theta**3 - 4*r*theta - 2*r.diff(x))*P)
            values = _solve_linear(equation, unknowns, x)
            if values is None:
                continue
            P_value = _replaced(P, values)
            phi = cancel(theta + P_value.diff(x)/P_value)
            w = Symbol('w')
            polynomial = w**2 - phi*w + cancel(phi.diff(x)/2 + phi**2/2 - r)
            discriminant = cancel(phi**2 - 4*(phi.diff(x)/2 + phi**2/2 - r))
            omega = (phi + sqrt(discriminant))/2
            return KovacicSolution(2, omega, polynomial, _exp_integral(omega, x))
    return None


def _case3(r: Expr, num: _Polynomial, den: _Polynomial, poles: dict[Expr, int], x: Symbol
           ) -> Optional[KovacicSolution]:
    if any(order > 2 for order in poles.values()) or _order_at_infinity(num, den) < 2:
        return None
    for n in (4, 6, 12):
        sets = _sets(r, num, den, poles, x, n)
        if sets is None:
            continue
        E, E_inf = sets
        names = list(E)
        S_poly = _product([x - c for c in names])
        for e_inf in E_inf:
            for family in cartesian(*[E[c] for c in names]):
                d12 = n*(e_inf - sum(family))
                if d12 < 0 or d12 % 12:
                    continue
                d = d12//12
                theta = Rational(n, 12)*_sum([Integer(e)/(x - c) for e, c in zip(family, names)])
                P, unknowns = _monic_polynomial(d, x)
                # Kovacic's recursion, P_n = -P down to P_{-1} which must vanish
                polynomials: dict[int, Expr] = {n: -P}
                for i in range(n, -1, -1):
                    P_i = polynomials[i]
                    P_next = polynomials.get(i + 1, _ZERO)
                    polynomials[i - 1] = cancel(-S_poly*P_i.diff(x) + ((n - i)*S_poly.diff(x) - S_poly*theta)*P_i
                                                - (n - i)*(i + 1)*S_poly**2*r*P_next)
                values = _solve_linear(polynomials[-1], unknowns, x)
                if values is None:
                    continue
                w = Symbol('w')
                polynomial = cancel(_sum([S_poly**i*_replaced(polynomials[i], values)/factorial(n - i)*w**i
                                          for i in range(n + 1)]))
                omega = _algebraic_root(polynomial, w)
                from sympy_extras.integrals.integral import Integral
                integrand = omega if omega is not None else _root_placeholder(x)
                return KovacicSolution(3, omega, polynomial, exp(Integral(integrand, x)))
    return None


def _root_placeholder(x: Symbol) -> Expr:
    """``omega`` when its polynomial cannot be solved in radicals: an
    undefined function of ``x``, whose defining relation is the
    ``omega_polynomial`` of the :class:`KovacicSolution`.

    It used to be ``omega`` *applied to the polynomial*, which read as a
    function of a polynomial in the auxiliary ``w`` and left ``w`` among
    the free symbols of the solution (sympy-extras#31).
    """
    return Function('omega')(x)


def _algebraic_root(polynomial: Expr, w: Symbol) -> Optional[Expr]:
    """A root of a polynomial in ``w`` with rational function coefficients,
    in radicals when the native ``roots`` finds them (the polynomial
    cleared of its denominator)."""
    limit_ = None if settings.timeout is None else min(settings.timeout, 5.0)
    numerator = polynomial.as_numer_denom()[0]
    found = attempt(lambda: _roots(numerator, w), limit_, native=True)
    if not found:
        return None
    for root in found:
        if not root.has(Piecewise):
            return root
    return None


# ---------------------------------------------------------------------------

def liouvillian_solution(r: Expr, x: Symbol) -> Optional[KovacicSolution]:
    """A Liouvillian solution of ``z'' = r z`` (``r`` a rational function
    of ``x``) by Kovacic's algorithm, or ``None`` when there is none.

    Examples
    ========

    >>> from sympy_extras.core import Rational, Symbol
    >>> from sympy_extras.solvers.kovacic import liouvillian_solution
    >>> x = Symbol('x')
    >>> liouvillian_solution(-Rational(3, 16)/x**2 + 1/x, x).case
    2
    >>> liouvillian_solution(x, x) is None
    True
    """
    num, den = _rational(r, x)
    if num.is_zero:
        return KovacicSolution(1, _ZERO, Symbol('w'), _ONE)
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


def _is_constant(e: Expr, x: Symbol) -> bool:
    """Whether ``e`` does not depend on ``x``: free of it, or its
    derivative simplified to zero."""
    if not e.has(x):
        return True
    derivative = attempt(lambda: simplify(e.diff(x)), settings.timeout, native=True)
    return derivative is not None and derivative == _ZERO


def _second_solution(y1: Expr, p: Expr, x: Symbol) -> Optional[Expr]:
    """The second solution by reduction of order:
    ``y2 = y1 Integral(exp(-Integral(p))/y1**2)``, in closed form when
    SymPy integrates it (without conditions on the parameters) and the
    result verifies, otherwise with the integral left unevaluated."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    from sympy_extras.integrals.integral import Integral
    inner = closed_antiderivative(p, x)
    if inner is None:
        return None
    integrand = simplify(exp(-inner)/y1**2)
    outer = closed_antiderivative(integrand, x, closed=True)
    if outer is not None:
        candidate = simplify(y1*outer)
        # verified against the original equation y'' + p y' + q y = 0 through
        # the Wronskian: (y2/y1)' = exp(-Integral(p))/y1**2
        difference = attempt(lambda: simplify((candidate/y1).diff(x) - integrand), settings.timeout, native=True)
        if difference is not None and difference == _ZERO:
            return candidate
    return y1*Integral(integrand, x)


def dsolve_kovacic(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[list[Expr]]:
    """The Liouvillian solutions of a second order linear homogeneous
    equation ``a y'' + b y' + c y = 0`` with rational coefficients: a list
    with one or two independent solutions, or ``None`` when there is no
    Liouvillian solution (or the equation is not of that form).

    With ``special_values``, the solutions of an equation with parameters
    get cases for the isolated values of the parameters at which they are
    undefined or dependent, as in
    :func:`~sympy_extras.solvers.linear_ode.dsolve_linear`.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.kovacic import dsolve_kovacic
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> dsolve_kovacic(y.diff(x, 2) + y.diff(x)/x + (1 - 1/(4*x**2))*y, y)
    [exp(I*x)/sqrt(x), exp(-I*x)/sqrt(x)]
    >>> dsolve_kovacic(y.diff(x, 2) - x*y, y) is None
    True
    """
    found = _dsolve_kovacic(equation, f)
    if not special_values or found is None:
        return found
    from ._special_values import as_zero, basis_cases

    def solve_at(at: Expr) -> Optional[list[Expr]]:
        return dsolve_kovacic(at, f)

    return basis_cases(found, as_zero(equation), f, solve_at)


def _dsolve_kovacic(equation: Basic, f: AppliedUndef) -> Optional[list[Expr]]:
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    x = x_
    if isinstance(equation, Eq):
        lhs = equation.lhs - equation.rhs
    elif isinstance(equation, Expr):
        lhs = equation
    else:
        raise TypeError("%s is not an equation" % (equation,))
    expanded = expand(lhs)
    d2, d1, d0 = f.diff(x, 2), f.diff(x), f
    a = expanded.coeff(d2)
    rest = expanded - a*d2
    b = rest.coeff(d1)
    rest = rest - b*d1
    c = rest.coeff(d0)
    if a == _ZERO or expand(rest - c*d0) != _ZERO or any(e.has(f) for e in (a, b, c)):
        return None
    for e in (a, b, c):
        if e.is_rational_function(x) is not True:
            return None
    p = cancel(b/a)
    q = cancel(c/a)
    r = normal_form(p, q, x)
    z = liouvillian_solution(r, x)
    if z is None or z.omega is None:
        return None
    from sympy_extras.integrals.indefinite import closed_antiderivative
    integral_p = closed_antiderivative(p, x, cases=True)
    if integral_p is None:
        return None
    factor_ = exp(-integral_p/2)
    y1 = _normalized(simplify(z.solution*factor_), x)
    solutions = [y1]
    if z.case == 3:
        return [z.solution*factor_]
    if z.case == 2:
        # the conjugate root gives the second solution: the roots of
        # w**2 - phi*w + c add up to phi
        w = Symbol('w')
        other = cancel(-z.omega_polynomial.coeff(w) - z.omega)
        y2 = _normalized(simplify(_exp_integral(other, x)*factor_), x)
        if not _is_constant(simplify(y2/y1), x):
            solutions.append(y2)
    if len(solutions) == 1:
        second = _second_solution(y1, p, x)
        if second is not None:
            solutions.append(_normalized(second, x))
    return solutions


def _normalized(y: Expr, x: Symbol) -> Expr:
    """``y`` without its constant factor."""
    constant, rest = y.as_independent(x, as_Add=False)
    if constant != _ZERO and rest != _ZERO and not constant.has(x):
        return rest
    return y
