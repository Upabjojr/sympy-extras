"""Linear ordinary differential equations with polynomial coefficients:
polynomial, rational and hyperexponential solutions, first order right
factors and reduction of order.

For an operator `L = a_n(x) D^n + \\cdots + a_1(x) D + a_0(x)` with
polynomial coefficients:

* **Polynomial solutions** (Abramov, Bronstein, Petkovšek): the degree of
  a polynomial solution is a nonnegative integer root of the *indicial
  polynomial at infinity* `\\lambda(d) = \\sum_i \\mathrm{lc}(a_i)\\, d^{\\underline{i}}`
  over the `i` with `\\deg a_i - i` maximal, so a polynomial ansatz up to
  the largest such root and a linear system give all of them.
* **Rational solutions** (Abramov, Singer): the multiplicity of a root
  `c` of `a_n` in the denominator of a rational solution is bounded by
  the negative integer roots of the indicial polynomial at `c`
  (computed through the norm over the field of `c` when `c` is not
  rational); with the denominator `Q` so bounded, `y = P/Q` and the
  polynomial solutions `P` of the transformed operator are found.
* **Hyperexponential solutions** `y = \\exp(\\int \\rho)`, `\\rho` rational,
  when the operator is Fuchsian (regular singular points only): the local
  exponents at the finite singularities and at infinity are the roots
  of the indicial polynomials, the rational combinations of them bound
  the ansatz `y = \\prod (x - c)^{e_c} P(x)`, and `P` is again a polynomial
  solution. Each such solution gives a first order right factor
  `D - \\rho` of the operator (Beke's method for first order factors).
  Irregular singularities (exponential parts `e^{p(x)}`) are not
  treated: for second order equations Kovacic's algorithm
  (:mod:`sympy_extras.solvers.kovacic`) covers them.
* **Reduction of order**: a known solution `y_1` turns `L` into an
  operator of order `n - 1` for `v = (y/y_1)'`.

SymPy's ``dsolve`` solves linear equations with constant coefficients,
Euler equations and a few named second order equations, and has no
general algorithm for polynomial coefficients.

References
==========

.. [Abramov] S. A. Abramov, M. Bronstein, M. Petkovšek, On polynomial
   solutions of linear operator equations, ISSAC 1995.
.. [Singer] M. F. Singer, Liouvillian solutions of n-th order homogeneous
   linear differential equations, American Journal of Mathematics 103
   (1981).
.. [Bronstein] M. Bronstein, On solutions of linear ordinary differential
   equations in their coefficient field, Journal of Symbolic Computation
   13 (1992).
"""
from __future__ import annotations

from itertools import product as cartesian
from typing import Sequence

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, Derivative, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import ff
from sympy.functions.elementary.exponential import exp_polar
from sympy.functions.elementary.piecewise import Piecewise
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices.dense import Matrix
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, cancel, factor_list, resultant, lcm_list

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings

__all__ = ['LinearOperator', 'polynomial_solutions', 'rational_solutions',
           'hyperexponential_solutions', 'reduce_order_linear', 'dsolve_linear']


class LinearOperator:
    """``a_n D**n + ... + a_1 D + a_0`` with polynomial coefficients
    ``[a_0, ..., a_n]`` in ``x``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> L.coefficients, L.order
    ([2, -x - 2, x], 2)
    >>> L(x**2 + 2*x + 2)
    0
    """

    def __init__(self, coefficients: Sequence[Expr], x: Symbol) -> None:
        coefficients_ = [as_expr(c) for c in coefficients]
        while len(coefficients_) > 1 and coefficients_[-1] == 0:
            coefficients_.pop()
        self.coefficients = coefficients_
        self.x = x

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @classmethod
    def from_equation(cls, equation: Basic, f: AppliedUndef) -> LinearOperator:
        """The operator of a homogeneous linear equation in ``f``, the
        coefficients made polynomial by clearing denominators."""
        x_ = f.args[0]
        if not isinstance(x_, Symbol):
            raise ValueError("the function must depend on a symbol")
        x = x_
        lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
        expanded = as_expr(expand(as_expr(lhs)))
        n = 0
        for d in expanded.atoms(Derivative):
            if d.expr == f:
                n = max(n, sum(int(as_expr(count)) for v, count in d.variable_count if v == x))
        coefficients: list[Expr] = []
        rest = expanded
        for i in range(n, -1, -1):
            term = Derivative(f, (x, i)) if i > 0 else f
            c = as_expr(rest.coeff(term))
            rest = as_expr(expand(rest - c*term))
            coefficients.append(c)
        if rest != 0 or any(c.has(f) for c in coefficients):
            raise ValueError("%s is not a homogeneous linear equation in %s" % (equation, f))
        coefficients.reverse()
        denominators = [as_expr(cancel(c).as_numer_denom()[1]) for c in coefficients]
        common = as_expr(lcm_list(denominators))
        cleared = [as_expr(cancel(c*common)) for c in coefficients]
        for c in cleared:
            if not c.is_polynomial(x):
                raise ValueError("the coefficients are not rational functions of %s" % (x,))
        return cls(cleared, x)

    def __call__(self, y: Expr) -> Expr:
        x = self.x
        return as_expr(expand(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(self.coefficients)])))

    def __repr__(self) -> str:
        return "LinearOperator(%s, %s)" % (self.coefficients, self.x)

    def transformed(self, factor: Expr) -> LinearOperator:
        """The operator ``M`` with ``L(factor * u) = factor * M(u)``,
        denominators cleared (``factor`` may be rational or
        hyperexponential: only ``factor'/factor`` needs to be
        rational)."""
        x = self.x
        u = Function('u')(x)
        image = as_expr(expand(cancel(self(factor*u)/factor)))
        coefficients: list[Expr] = []
        rest = image
        for i in range(self.order, -1, -1):
            term = Derivative(u, (x, i)) if i > 0 else u
            c = as_expr(rest.coeff(term))
            rest = as_expr(expand(rest - c*term))
            coefficients.append(as_expr(cancel(c)))
        coefficients.reverse()
        denominators = [as_expr(cancel(c).as_numer_denom()[1]) for c in coefficients]
        common = as_expr(lcm_list(denominators))
        return LinearOperator([as_expr(cancel(c*common)) for c in coefficients], x)


def _falling(d: Expr, i: int) -> Expr:
    return as_expr(ff(d, i))


def _indicial_at_infinity(L: LinearOperator) -> Poly:
    """The polynomial in ``d`` whose nonnegative integer roots bound the
    degree of the polynomial solutions."""
    x = L.x
    d = Dummy('d')
    data: list[tuple[int, int, Expr]] = []
    for i, c in enumerate(L.coefficients):
        if c == 0:
            continue
        p = Poly(c, x)
        data.append((i, p.degree() - i, as_expr(p.LC())))
    top = max(shift for _, shift, _ in data)
    expression = Add(*[lc*_falling(d, i) for i, shift, lc in data if shift == top])
    return Poly(as_expr(expression), d)


def _nonnegative_integer_roots(p: Poly) -> list[int]:
    found = roots(p)
    result = [int(r) for r in found if isinstance(r, Integer) and int(r) >= 0]
    return sorted(result)


def _negative_integer_roots(p: Poly) -> list[int]:
    found = roots(p)
    return sorted(int(r) for r in found if isinstance(r, Integer) and int(r) < 0)


def polynomial_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the polynomial solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, polynomial_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    >>> polynomial_solutions(L)
    [x**2 + 2*x + 2]
    >>> polynomial_solutions(LinearOperator([-2, 0, 1], x))
    []
    """
    x = L.x
    indicial = _indicial_at_infinity(L)
    if indicial.is_zero:
        return []
    degrees = _nonnegative_integer_roots(indicial)
    if not degrees:
        return []
    bound = degrees[-1]
    unknowns = [Dummy('c%d' % i) for i in range(bound + 1)]
    ansatz = as_expr(Add(*[c*x**i for i, c in enumerate(unknowns)]))
    image = as_expr(expand(L(ansatz)))
    try:
        rows = Poly(image, x).all_coeffs()
    except PolynomialError:
        return []
    M = Matrix([[as_expr(row).coeff(u) for u in unknowns] for row in rows]) if rows else Matrix.zeros(0, len(unknowns))
    solutions: list[Expr] = []
    for vector in M.nullspace():
        solutions.append(as_expr(Add(*[v*x**i for i, v in enumerate(vector)])))
    return solutions


def _indicial_at(L: LinearOperator, q: Poly) -> Poly:
    """The indicial polynomial (in ``m``) at the roots ``c`` of the
    irreducible ``q``, as a polynomial with rational coefficients whose
    integer roots contain those of the true indicial polynomial (the norm
    from ``Q(c)`` when ``q`` is not linear)."""
    x = L.x
    m = Dummy('m')
    c = Dummy('c')
    orders: list[tuple[int, int, Expr]] = []
    for i, coefficient in enumerate(L.coefficients):
        if coefficient == 0:
            continue
        p = Poly(coefficient, x)
        order = 0
        while True:
            quotient, remainder = p.div(q)
            if not remainder.is_zero:
                break
            p = quotient
            order += 1
        # the value of the cofactor at c, reduced modulo q(c)
        value = Poly(p.as_expr().subs(x, c), c).rem(Poly(q.as_expr().subs(x, c), c)).as_expr() \
            if q.degree() > 1 else p.as_expr().subs(x, as_expr(-q.all_coeffs()[1]/q.all_coeffs()[0]))
        orders.append((i, order - i, as_expr(value)))
    lowest = min(shift for _, shift, _ in orders)
    expression = as_expr(Add(*[value*_falling(m, i) for i, shift, value in orders if shift == lowest]))
    if q.degree() > 1:
        expression = as_expr(resultant(expression, q.as_expr().subs(x, c), c))
    return Poly(expression, m)


def _denominator_bound(L: LinearOperator) -> Expr:
    x = L.x
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    result: Expr = S.One
    for f, _ in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        indicial = _indicial_at(L, q)
        if indicial.is_zero:
            continue
        negative = _negative_integer_roots(indicial)
        if negative:
            result = as_expr(result*f**(-negative[0]))
    return result


def rational_solutions(L: LinearOperator) -> list[Expr]:
    """A basis of the rational solutions of ``L(y) = 0``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, rational_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    >>> rational_solutions(L)
    [x**(-2), 1/x]
    """
    Q = _denominator_bound(L)
    if Q == 1:
        return polynomial_solutions(L)
    M = L.transformed(1/Q)
    return [as_expr(cancel(P/Q)) for P in polynomial_solutions(M)]


# ---------------------------------------------------------------------------
# hyperexponential solutions of Fuchsian operators

def _is_fuchsian(L: LinearOperator) -> bool:
    """Whether every singular point (finite and at infinity) is regular
    singular: ``ord_c(a_i) >= ord_c(a_n) - (n - i)`` at finite points and
    ``deg a_i <= deg a_n - (n - i)`` at infinity."""
    x = L.x
    n = L.order
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    for f, multiplicity in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        for i, c in enumerate(L.coefficients[:-1]):
            if c == 0:
                continue
            order = _order_at(Poly(c, x), q)
            if order < multiplicity - (n - i):
                return False
    degree_n = leading.degree()
    for i, c in enumerate(L.coefficients[:-1]):
        if c != 0 and Poly(c, x).degree() > degree_n - (n - i):
            return False
    return True


def _order_at(p: Poly, q: Poly) -> int:
    order = 0
    while True:
        quotient, remainder = p.div(q)
        if not remainder.is_zero:
            return order
        p = quotient
        order += 1


def _exponents(L: LinearOperator, q: Poly) -> list[Rational]:
    """The rational local exponents at the roots of ``q``."""
    indicial = _indicial_at(L, q)
    if indicial.is_zero:
        return []
    return sorted(set(r for r in roots(indicial) if isinstance(r, Rational)), key=lambda r: (r.p, r.q))


def hyperexponential_solutions(L: LinearOperator, max_combinations: int = 200) -> list[Expr]:
    """Solutions ``prod (x - c)**e_c * P(x)`` with rational exponents at
    the singular points and a polynomial ``P``, of a Fuchsian operator.
    Solutions with distinct exponent combinations are independent.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, hyperexponential_solutions
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(4*x**2*y.diff(x, 2) + 4*x*y.diff(x) - y, y)
    >>> hyperexponential_solutions(L)
    [1/sqrt(x), sqrt(x)]
    """
    x = L.x
    if not _is_fuchsian(L):
        return []
    leading = Poly(L.coefficients[-1], x)
    _, factors = factor_list(leading.as_expr(), x)
    places: list[tuple[Expr, list[Rational]]] = []
    for f, _ in factors:
        q = Poly(f, x)
        if q.degree() == 0:
            continue
        if q.degree() > 1:
            # a common exponent for conjugate roots: the factor itself
            exponents = _exponents(L, q)
        else:
            exponents = _exponents(L, q)
        if not exponents:
            return []
        places.append((as_expr(f), exponents))
    solutions: list[Expr] = []
    combinations = list(cartesian(*[e for _, e in places]))
    if len(combinations) > max_combinations:
        combinations = combinations[:max_combinations]
    for combination in combinations:
        factor = as_expr(Mul(*[f**e for (f, _), e in zip(places, combination)]))
        M = L.transformed(factor)
        for P in polynomial_solutions(M):
            solutions.append(as_expr(factor*P))
    return _independent(solutions, x)


# ---------------------------------------------------------------------------
# reduction of order

def reduce_order_linear(L: LinearOperator, y1: Expr) -> LinearOperator:
    """The operator ``M`` of order ``n - 1`` with ``L(y1 * w) = y1 * M(w')``
    for a solution ``y1`` of ``L``.

    Examples
    ========

    >>> from sympy import Function, exp
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import LinearOperator, reduce_order_linear
    >>> y = Function('y')(x)
    >>> L = LinearOperator.from_equation(y.diff(x, 2) - 3*y.diff(x) + 2*y, y)
    >>> reduce_order_linear(L, exp(x))
    LinearOperator([-1, 1], x)
    """
    x = L.x
    T = L.transformed(y1)
    if cancel(T.coefficients[0]) != 0:
        raise ValueError("%s is not a solution" % (y1,))
    return LinearOperator(T.coefficients[1:], x)


def _independent(solutions: list[Expr], x: Symbol) -> list[Expr]:
    """Drop the solutions which are constant multiples of earlier ones."""
    kept: list[Expr] = []
    for s in solutions:
        if s == 0:
            continue
        if any(cancel(s/k).is_constant(x) for k in kept):
            continue
        kept.append(s)
    return kept


def dsolve_linear(equation: Basic, f: AppliedUndef, use_kovacic: bool = True,
                  use_dsolve: bool = True) -> list[Expr]:
    """Independent solutions of a homogeneous linear equation with
    rational coefficients: rational and hyperexponential solutions,
    Kovacic's algorithm for the second order, and reduction of order by
    the solutions found (the reduced equations handed to the same
    solvers and to SymPy's ``dsolve``). The list is empty when nothing
    is found and shorter than the order when the solution space is not
    exhausted.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.linear_ode import dsolve_linear
    >>> y = Function('y')(x)
    >>> dsolve_linear(x*y.diff(x, 2) - (x + 2)*y.diff(x) + 2*y, y)
    [x**2 + 2*x + 2, exp(x)]
    >>> dsolve_linear(x**2*y.diff(x, 2) + 4*x*y.diff(x) + 2*y, y)
    [x**(-2), 1/x]
    """
    L = LinearOperator.from_equation(equation, f)
    return _solve_operator(L, f, use_kovacic, use_dsolve)


def _solve_operator(L: LinearOperator, f: AppliedUndef, use_kovacic: bool, use_dsolve: bool) -> list[Expr]:
    x = L.x
    n = L.order
    if n == 0:
        return []
    if n == 1:
        # y' = -a_0/a_1 y
        rho = as_expr(cancel(-L.coefficients[0]/L.coefficients[1]))
        integral = attempt(lambda: integrate(rho, x), settings.timeout)
        if integral is None:
            return [as_expr(Function('exp')(Integral(rho, x)))]
        from sympy.functions.elementary.exponential import exp
        return [as_expr(exp(integral))]
    found: list[Expr] = []
    found.extend(rational_solutions(L))
    if len(found) < n:
        found.extend(hyperexponential_solutions(L))
    found = _independent(found, x)
    if len(found) < n and n == 2 and use_kovacic:
        from .kovacic import dsolve_kovacic
        y = Function('y')(x)
        equation = as_expr(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)]))
        liouvillian = attempt(lambda: dsolve_kovacic(equation, y), settings.timeout)
        if liouvillian:
            found = _independent(found + liouvillian, x)
    if found and len(found) < n:
        # reduction of order by the first solution
        y1 = found[0]
        try:
            M = reduce_order_linear(L, y1)
        except ValueError:
            return found
        reduced = _solve_operator(M, f, use_kovacic, use_dsolve)
        for v in reduced:
            w = attempt(lambda: integrate(v, x, conds='none'), settings.timeout)
            if w is None or w.has(Integral, Piecewise, exp_polar):
                # the quadrature is left unevaluated
                w = Integral(v, x)
            found = _independent(found + [as_expr(cancel(y1*w)) if (y1*w).is_rational_function(x) else as_expr(y1*w)], x)
    if len(found) < n and use_dsolve:
        from sympy.solvers.ode import dsolve
        y = Function('y')(x)
        equation = as_expr(Add(*[c*y.diff(x, i) if i else c*y for i, c in enumerate(L.coefficients)]))
        result = attempt(lambda: dsolve(equation, y), settings.timeout)
        if isinstance(result, Eq) and not result.rhs.has(Integral):
            constants = sorted((s for s in free_symbols(as_expr(result.rhs)) if s.name.startswith('C')), key=lambda s: s.name)
            for c in constants:
                part = as_expr(result.rhs.coeff(c))
                if part != 0 and not part.has(*constants):
                    found = _independent(found + [part], x)
    return found[:n]
