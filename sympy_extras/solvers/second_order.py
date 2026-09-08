"""Second order ordinary differential equations `y'' = \\Phi(x, y, y')`:
integrating factors depending on one or two of the variables, and
linearisation by point transformations.

**Integrating factors.** `\\mu(x, y, y')` is an integrating factor when
`\\mu (y'' - \\Phi)` is the total derivative of a first integral
`R(x, y, y')`, so that `R = C_1` is a first order equation. Two families
are found exactly (Cheb-Terrab and Roche):

* `\\mu = \\mu(x, y)`: then `R = \\mu y' + S(x, y)` and `\\Phi` is quadratic in
  `y'`, `\\Phi = a y'^2 + b y' + c`, with `\\mu_y = -a\\mu`, `S_y = -b\\mu -
  \\mu_x`, `S_x = -c\\mu`. Writing `\\mu = m(x) e^{-\\int a\\, dy}`, the
  compatibility `S_{xy} = S_{yx}` is a linear second order equation for
  `m(x)` whose coefficients must not depend on `y`; it is solved with
  ``dsolve`` and `S` follows by quadratures.
* `\\mu = \\mu(y')`: then `R = M(y') + S(x, y)` with `M' = \\mu`, and
  `\\mu\\Phi = A(x, y) + B(x, y) y'` with `A_y = B_x`: `\\Phi` must factor
  as `g(y')\\,(A + B y')`, and `\\mu = 1/g`.

**Linearisation (Lie, Bocharov–Sokolov–Svinolupov, Ibragimov–Meleshko).**
An equation is equivalent to `u'' = 0` under a point transformation
`t = \\varphi(x, y)`, `u = \\psi(x, y)` exactly when it is cubic in `y'`,

.. math::

    y'' + a\\, y'^3 + b\\, y'^2 + c\\, y' + d = 0,

and Lie's two conditions hold:

.. math::

    3a_{xx} - 2b_{xy} + c_{yy} - 3a_x c + 3a_y d + 2b_x b - 3c_x a - c_y b + 6d_y a = 0,

    b_{xx} - 2c_{xy} + 3d_{yy} - 6a_x d + b_x c + 3b_y d - 2c_y c - 3d_x a + 3d_y b = 0.

When moreover `a = 0` and `c_y = 2 b_x`, the transformation can be taken
*fibre preserving*, `t = \\varphi(x)`, and is constructed by quadratures:
with `B = \\int b\\, dy` and `\\Psi = \\int e^{B} dy`, `\\psi = \\Psi/w + l(x)`
where `w` solves the linear equation `w'' + (c - 2B_x) w' - Q w = 0`,
`Q = B_{xx} - B_x^2 + c B_x - d_y - b d`, and `\\varphi'' / \\varphi' = 2B_x
- c - 2w'/w`. The general solution is then `\\psi(x, y) = C_1 \\varphi(x)
+ C_2`. When the conditions fail in `(x, y)` the roles of the variables
are exchanged (`x` as a function of `y`) and the fibre preserving
construction is tried there.

SymPy's ``dsolve`` has no integrating factor method for second order
equations and no linearisation test.

References
==========

.. [ChebTerrab] E. S. Cheb-Terrab, A. D. Roche, Integrating factors for
   second-order ODEs, Journal of Symbolic Computation 27 (1999).
.. [Ibragimov] N. H. Ibragimov, S. V. Meleshko, Linearization of
   third-order ordinary differential equations by point and contact
   transformations, and the second order case: Linearization of
   ordinary differential equations of second order, in Lie group
   analysis, ALGA 2 (2005); S. Lie, Klassifikation und Integration von
   gewöhnlichen Differentialgleichungen zwischen x, y, die eine Gruppe
   von Transformationen gestatten III, Archiv for Matematik 8 (1883).
.. [Bocharov] A. V. Bocharov, V. V. Sokolov, S. I. Svinolupov, On some
   equivalence problems for differential equations, preprint ESI 54
   (1993).
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, Derivative, expand
from sympy.core.exprtools import factor_terms
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor_list
from sympy.polys.rationaltools import together
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve
from sympy.solvers.solvers import solve as _solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings
from .lie import Symmetry

__all__ = ['second_order_rhs', 'integrating_factor_xy', 'integrating_factor_p', 'is_linearizable',
           'linearize', 'rectify_symmetries', 'commuting_pair', 'transformed_rhs', 'dsolve_second_order',
           'FirstIntegral', 'Linearization']


class FirstIntegral:
    """A first integral ``R(x, y, y') = C1`` of a second order equation
    and the integrating factor which produced it.

    Attributes
    ==========

    factor : Expr
        The integrating factor ``mu`` (in ``x``, ``y`` and ``p`` for
        ``y'``).
    integral : Expr
        ``R(x, y, p)``.
    """

    def __init__(self, factor: Expr, integral: Expr, x: Symbol, y: Symbol, p: Symbol) -> None:
        self.factor = factor
        self.integral = integral
        self.x = x
        self.y = y
        self.p = p

    def __repr__(self) -> str:
        return "FirstIntegral(%s, %s)" % (self.factor, self.integral)

    def equation(self, f: AppliedUndef) -> Eq:
        """``R(x, f, f') = C1`` as an equation in the function ``f``."""
        x_ = f.args[0]
        return Eq(self.integral.xreplace({self.p: Derivative(f, x_)}).xreplace({self.y: f, self.x: x_}), Symbol('C1'))


class Linearization:
    """A point transformation ``t = phi(x, y)``, ``u = psi(x, y)`` taking a
    second order equation to ``u'' = rhs(u')`` (``rhs = 0`` for a
    linearisation; a function of ``u'`` alone when the coordinates
    rectify two commuting symmetries).

    Attributes
    ==========

    t, u : Expr
        The new independent and dependent variables as expressions in
        ``x`` and ``y``.
    rhs : Expr
        The right-hand side of the transformed equation, in ``v`` for
        ``u'``.
    """

    def __init__(self, t: Expr, u: Expr, x: Symbol, y: Symbol, v: Symbol, rhs: Expr) -> None:
        self.t = t
        self.u = u
        self.x = x
        self.y = y
        self.v = v
        self.rhs = rhs

    def __repr__(self) -> str:
        if self.rhs == 0:
            return "Linearization(t=%s, u=%s)" % (self.t, self.u)
        return "Linearization(t=%s, u=%s, rhs=%s)" % (self.t, self.u, self.rhs)


def second_order_rhs(equation: Basic, f: AppliedUndef) -> tuple[Expr, Symbol, Symbol, Symbol]:
    """``(Phi, x, y, p)`` with the equation written as ``y'' = Phi(x, y, p)``
    (``p`` stands for ``y'``); ``ValueError`` when the equation is not
    linear in ``y''``."""
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
    y, p, q = Dummy('y'), Dummy('p'), Dummy('q')
    replaced = as_expr(lhs).xreplace({Derivative(f, (x_, 2)): q}).xreplace({Derivative(f, x_): p}).xreplace({f: y})
    if replaced.atoms(Derivative) or replaced.has(f):
        raise ValueError("%s is not a second order equation in %s" % (equation, f))
    try:
        poly = Poly(replaced, q)
    except PolynomialError:
        raise ValueError("%s is not linear in the second derivative" % (equation,))
    if poly.degree() != 1:
        raise ValueError("%s is not linear in the second derivative" % (equation,))
    a1, a0 = [as_expr(c) for c in poly.all_coeffs()]
    return as_expr(cancel(-a0/a1)), x_, y, p


def _integrate(e: Expr, v: Symbol) -> Optional[Expr]:
    value = attempt(lambda: as_expr(integrate(e, v)), settings.timeout)
    if value is None or value.has(Integral):
        return None
    return value


def _particular(solution: Basic) -> Optional[Expr]:
    """A nonzero particular solution from ``dsolve``'s general one, with
    the first constant set to 1 and the others to 0."""
    if not isinstance(solution, Eq):
        return None
    rhs = as_expr(solution.rhs)
    constants = sorted((s for s in free_symbols(rhs) if s.name.startswith('C')), key=lambda s: s.name)
    if not constants:
        return rhs
    for chosen in constants:
        values = {c: (S.One if c == chosen else S.Zero) for c in constants}
        candidate = as_expr(simplify(rhs.xreplace(values)))
        if candidate != 0:
            return candidate
    return None


def _solve_linear(coefficient1: Expr, coefficient0: Expr, x: Symbol) -> Optional[Expr]:
    """A nonzero solution of ``w'' + coefficient1 w' + coefficient0 w = 0``."""
    W = Function('W')(x)
    equation = W.diff(x, 2) + coefficient1*W.diff(x) + coefficient0*W
    solution = attempt(lambda: dsolve(equation, W), settings.timeout)
    if solution is None:
        return None
    return _particular(solution)


def integrating_factor_xy(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(x, y)`` of ``y'' = Phi`` and the first
    integral ``mu y' + S(x, y)``, or ``None``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order import integrating_factor_xy
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_xy(-p**2/y, x, y, p)          # y y'' + y'^2 = 0
    FirstIntegral(y, p*y)
    >>> integrating_factor_xy(-p/x, x, y, p)             # x y'' + y' = 0
    FirstIntegral(x, p*x)
    """
    try:
        poly = Poly(expand(Phi), p)
    except PolynomialError:
        return None
    if poly.degree() > 2:
        return None
    coefficients = [as_expr(c) for c in poly.all_coeffs()]
    while len(coefficients) < 3:
        coefficients.insert(0, S.Zero)
    a, b, c = coefficients
    # mu = m(x) E(x, y) with E_y = -a E
    A = _integrate(a, y)
    if A is None:
        return None
    E = as_expr(exp(-A))
    Ex = as_expr(E.diff(x)/E)
    Exx = as_expr(E.diff(x, 2)/E)
    coefficient1 = as_expr(cancel(b + 2*Ex))
    coefficient0 = as_expr(cancel(b.diff(x) + b*Ex + Exx - c.diff(y) + a*c))
    if coefficient1.has(y) or coefficient0.has(y):
        return None
    m = _solve_linear(coefficient1, coefficient0, x)
    if m is None:
        return None
    mu = as_expr(simplify(m*E))
    # S_x = -c mu, S_y = -b mu - mu_x
    S1 = _integrate(as_expr(-c*mu), x)
    if S1 is None:
        return None
    remainder = as_expr(simplify(-b*mu - mu.diff(x) - S1.diff(y)))
    if remainder.has(x):
        return None
    S2 = _integrate(remainder, y)
    if S2 is None:
        return None
    integral = as_expr(simplify(mu*p + S1 + S2))
    return FirstIntegral(mu, integral, x, y, p)


def _factor_in_p(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``Phi = g(p) * L(x, y, p)`` with ``L`` free of factors in ``p``
    alone: ``(g, L)``. The factors of the product structure (powers,
    exponentials) are separated first, the polynomial content afterwards."""
    g_parts: list[Expr] = []
    rest_parts: list[Expr] = []
    for factor in Mul.make_args(factor_terms(Phi)):
        f_ = as_expr(factor)
        if f_.has(p) and not (f_.has(x) or f_.has(y)):
            g_parts.append(f_)
        else:
            rest_parts.append(f_)
    rest = as_expr(Mul(*rest_parts))
    numerator, denominator = together(rest).as_numer_denom()
    polynomial_parts: list[Expr] = []
    for part, sign in ((numerator, 1), (denominator, -1)):
        try:
            constant, factors = factor_list(part, x, y, p)
        except PolynomialError:
            return None
        polynomial_parts.append(as_expr(constant)**sign)
        for base, exponent in factors:
            base_ = as_expr(base)
            if base_.has(p) and not (base_.has(x) or base_.has(y)):
                g_parts.append(base_**(sign*exponent))
            else:
                polynomial_parts.append(base_**(sign*exponent))
    return as_expr(Mul(*g_parts)), as_expr(Mul(*polynomial_parts))


def integrating_factor_p(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(y')`` of ``y'' = Phi`` and the first
    integral ``M(y') + S(x, y)``, or ``None``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order import integrating_factor_p
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_p(p**3*(x + y*p), x, y, p)     # y'' = y'^3 (x + y y')
    FirstIntegral(p**(-3), -x**2/2 - y**2/2 - 1/(2*p**2))
    """
    factored = _factor_in_p(Phi, x, y, p)
    if factored is None:
        return None
    g, L = factored
    try:
        linear = Poly(expand(L), p)
    except PolynomialError:
        return None
    if linear.degree() > 1:
        return None
    coefficients = [as_expr(c) for c in linear.all_coeffs()]
    B, A = (coefficients if len(coefficients) == 2 else [S.Zero, coefficients[0]])
    if simplify(A.diff(y) - B.diff(x)) != 0:
        return None
    mu = as_expr(1/g)
    M = _integrate(mu, p)
    S1 = _integrate(-A, x)
    if M is None or S1 is None:
        return None
    remainder = as_expr(simplify(-B - S1.diff(y)))
    if remainder.has(x):
        return None
    S2 = _integrate(remainder, y)
    if S2 is None:
        return None
    return FirstIntegral(mu, as_expr(M + S1 + S2), x, y, p)


def _cubic_coefficients(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr]]:
    """``(a, b, c, d)`` with ``y'' + a y'^3 + b y'^2 + c y' + d = 0``."""
    try:
        poly = Poly(expand(-Phi), p)
    except PolynomialError:
        return None
    if poly.degree() > 3:
        return None
    coefficients = [as_expr(c) for c in poly.all_coeffs()]
    while len(coefficients) < 4:
        coefficients.insert(0, S.Zero)
    a, b, c, d = coefficients
    return a, b, c, d


def _lie_conditions(a: Expr, b: Expr, c: Expr, d: Expr, x: Symbol, y: Symbol) -> tuple[Expr, Expr]:
    first = (3*a.diff(x, 2) - 2*b.diff(x, y) + c.diff(y, 2) - 3*a.diff(x)*c + 3*a.diff(y)*d
             + 2*b.diff(x)*b - 3*c.diff(x)*a - c.diff(y)*b + 6*d.diff(y)*a)
    second = (b.diff(x, 2) - 2*c.diff(x, y) + 3*d.diff(y, 2) - 6*a.diff(x)*d + b.diff(x)*c
              + 3*b.diff(y)*d - 2*c.diff(y)*c - 3*d.diff(x)*a + 3*d.diff(y)*b)
    return as_expr(simplify(first)), as_expr(simplify(second))


def is_linearizable(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> bool:
    """Lie's test: whether ``y'' = Phi`` is equivalent to ``u'' = 0`` under
    a point transformation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order import is_linearizable
    >>> x, y, p = symbols('x y p')
    >>> is_linearizable(-3*y*p - y**3, x, y, p)
    True
    >>> is_linearizable(y**2, x, y, p)
    False
    """
    coefficients = _cubic_coefficients(Phi, x, y, p)
    if coefficients is None:
        return False
    first, second = _lie_conditions(*coefficients, x, y)
    return first == 0 and second == 0


def _fibre_preserving(a: Expr, b: Expr, c: Expr, d: Expr, x: Symbol, y: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(phi(x), psi(x, y))`` taking ``y'' + b y'^2 + c y' + d = 0`` to
    ``u'' = 0`` when ``a = 0`` and ``c_y = 2 b_x``."""
    if a != 0 or simplify(c.diff(y) - 2*b.diff(x)) != 0:
        return None
    B = _integrate(b, y)
    if B is None:
        return None
    Bx = as_expr(B.diff(x))
    Q = as_expr(simplify(B.diff(x, 2) - Bx**2 + c*Bx - d.diff(y) - b*d))
    coefficient1 = as_expr(simplify(c - 2*Bx))
    if Q.has(y) or coefficient1.has(y):
        return None
    w = _solve_linear(coefficient1, -Q, x)
    if w is None:
        return None
    k = as_expr(1/w)
    r = as_expr(simplify(2*Bx - c - 2*w.diff(x)/w))
    if r.has(y):
        return None
    Psi = _integrate(as_expr(exp(B)), y)
    if Psi is None:
        return None
    # the y-free remainder gives l(x): l'' - r l' = remainder
    remainder = as_expr(simplify(d*k*exp(B) - (k.diff(x, 2)*Psi + 2*k.diff(x)*Psi.diff(x) + k*Psi.diff(x, 2)
                                               - r*(k.diff(x)*Psi + k*Psi.diff(x)))))
    if remainder.has(y):
        return None
    R = _integrate(r, x)
    if R is None:
        return None
    phi_prime = as_expr(exp(R))
    phi = _integrate(phi_prime, x)
    if phi is None:
        return None
    l_prime: Expr = S.Zero
    if remainder != 0:
        inner = _integrate(as_expr(remainder/phi_prime), x)
        if inner is None:
            return None
        l_prime = as_expr(phi_prime*inner)
    l = _integrate(l_prime, x) if l_prime != 0 else S.Zero
    if l is None:
        return None
    psi = as_expr(simplify(k*Psi + l))
    return phi, psi


def transformed_rhs(Phi: Expr, t: Expr, u: Expr, x: Symbol, y: Symbol, p: Symbol, v: Symbol) -> Optional[Expr]:
    """``y'' = Phi`` written in the variables ``t(x, y)``, ``u(x, y)`` as
    ``u'' = rhs``, when ``rhs`` is a function of ``v = u'`` alone."""
    Dt = as_expr(t.diff(x) + t.diff(y)*p)
    Du = as_expr(u.diff(x) + u.diff(y)*p)
    V = as_expr(cancel(Du/Dt))
    DV = as_expr(V.diff(x) + V.diff(y)*p + V.diff(p)*Phi)
    second = as_expr(cancel(DV/Dt))
    solutions = attempt(lambda: _solve(Eq(V, v), p), settings.timeout)
    if not solutions:
        return None
    for value in solutions:
        if not isinstance(value, Expr):
            continue
        candidate = attempt(lambda: as_expr(simplify(second.xreplace({p: value}))), settings.timeout)
        if candidate is not None and not candidate.has(x) and not candidate.has(y):
            return candidate
    return None


def linearize(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[Linearization]:
    """A point transformation taking ``y'' = Phi`` to ``u'' = 0``: fibre
    preserving in ``(x, y)`` when possible, otherwise in ``(y, x)`` after
    exchanging the variables; ``None`` when the equation is not
    linearizable or the transformation is not of these kinds (see
    :func:`rectify_symmetries` for the general case).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.second_order import linearize
    >>> x, y, p = symbols('x y p')
    >>> linearize(p**2, x, y, p)             # y'' = y'^2, u = exp(-y)
    Linearization(t=x, u=-exp(-y))
    """
    coefficients = _cubic_coefficients(Phi, x, y, p)
    if coefficients is None:
        return None
    first, second = _lie_conditions(*coefficients, x, y)
    if first != 0 or second != 0:
        return None
    v = Dummy('v')
    found = _fibre_preserving(*coefficients, x, y)
    if found is not None:
        return Linearization(found[0], found[1], x, y, v, S.Zero)
    # x as a function of y: y' = 1/x', y'' = -x''/x'^3
    q = Dummy('q')
    swapped = as_expr(cancel(-Phi.xreplace({p: 1/q})*q**3))
    swapped_coefficients = _cubic_coefficients(swapped, y, x, q)
    if swapped_coefficients is None:
        return None
    found = _fibre_preserving(*swapped_coefficients, y, x)
    if found is None:
        return None
    return Linearization(found[0], found[1], x, y, v, S.Zero)


def _structure(symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[list[list[list[Expr]]]]:
    """The structure constants ``C[i][j][k]`` with ``[X_i, X_j] = sum_k
    C[i][j][k] X_k`` (``None`` when a bracket is not in the span)."""
    m = len(symmetries)
    coefficients = [Symbol('c%d' % k) for k in range(m)]
    result: list[list[list[Expr]]] = []
    for i in range(m):
        row: list[list[Expr]] = []
        for j in range(m):
            bracket = symmetries[i].commutator(symmetries[j])
            combination = symmetries[0]*coefficients[0]
            for k in range(1, m):
                combination = combination + symmetries[k]*coefficients[k]
            equations: list[Expr] = []
            for lhs, rhs in ((combination.xi[0], bracket.xi[0]), (combination.eta[0], bracket.eta[0])):
                difference = as_expr(expand(lhs - rhs))
                equations.extend(_coefficients_in(difference, x, y))
            solution = attempt(lambda: _solve(equations, coefficients, dict=True), settings.timeout)
            if not solution:
                return None
            values = solution[0]
            row.append([as_expr(values.get(c, S.Zero)) for c in coefficients])
        result.append(row)
    return result


def _coefficients_in(e: Expr, x: Symbol, y: Symbol) -> list[Expr]:
    """The coefficients of an expression linear in the unknowns with
    respect to the functions of ``x`` and ``y`` it contains."""
    try:
        poly = Poly(e, x, y)
        return [as_expr(c) for c in poly.coeffs()]
    except PolynomialError:
        pass
    collected: dict[Expr, Expr] = {}
    for term in (e.args if isinstance(e, Add) else [e]):
        constant, function = as_expr(term).as_independent(x, y)
        collected[function] = collected.get(function, S.Zero) + constant
    return list(collected.values())


def commuting_pair(symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[tuple[Symmetry, Symmetry]]:
    """Two commuting symmetries which are pointwise independent as vector
    fields (``xi_1 eta_2 - xi_2 eta_1 != 0``): among the basis elements
    and their sums first, then in the centralizers of a few elements when
    the brackets close in the span of the basis; ``None`` otherwise."""
    m = len(symmetries)
    if m < 2:
        return None

    def independent(X: Symmetry, Y: Symmetry) -> bool:
        return simplify(X.xi[0]*Y.eta[0] - Y.xi[0]*X.eta[0]) != 0

    def commute(X: Symmetry, Y: Symmetry) -> bool:
        bracket = X.commutator(Y)
        return all(simplify(c) == 0 for c in bracket.xi + bracket.eta)

    candidates: list[Symmetry] = list(symmetries)
    for i in range(m):
        for j in range(i + 1, m):
            candidates.append(symmetries[i] + symmetries[j])
    for i, X in enumerate(candidates):
        for Y in candidates[i + 1:]:
            if commute(X, Y) and independent(X, Y):
                return X, Y
    C = _structure(symmetries, x, y)
    if C is None:
        return None
    generic: list[Symmetry] = []
    for k in range(3):
        generic.append(sum((symmetries[i]*(i + k + 1) for i in range(1, m)), symmetries[0]))
    for X in generic:
        # X = sum a_i X_i; the centralizer: sum_j c_j [X, X_j] = 0
        a = _coordinates(X, symmetries, x, y)
        if a is None:
            continue
        rows: list[list[Expr]] = []
        for k in range(m):
            rows.append([as_expr(Add(*[a[i]*C[i][j][k] for i in range(m)])) for j in range(m)])
        null = Matrix(rows).nullspace()
        for vector in null:
            Y = sum((symmetries[j]*as_expr(vector[j]) for j in range(1, m)), symmetries[0]*as_expr(vector[0]))
            if not Y.is_zero() and independent(X, Y):
                return X, Y
    return None


def _coordinates(X: Symmetry, symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[list[Expr]]:
    m = len(symmetries)
    coefficients = [Symbol('c%d' % k) for k in range(m)]
    combination = symmetries[0]*coefficients[0]
    for k in range(1, m):
        combination = combination + symmetries[k]*coefficients[k]
    equations: list[Expr] = []
    for lhs, rhs in ((combination.xi[0], X.xi[0]), (combination.eta[0], X.eta[0])):
        equations.extend(_coefficients_in(as_expr(expand(lhs - rhs)), x, y))
    solution = attempt(lambda: _solve(equations, coefficients, dict=True), settings.timeout)
    if not solution:
        return None
    return [as_expr(solution[0].get(c, S.Zero)) for c in coefficients]


def _potential(fx: Expr, fy: Expr, x: Symbol, y: Symbol) -> Optional[Expr]:
    """``F`` with ``F_x = fx`` and ``F_y = fy``."""
    if simplify(fx.diff(y) - fy.diff(x)) != 0:
        return None
    F1 = _integrate(fx, x)
    if F1 is None:
        return None
    remainder = as_expr(simplify(fy - F1.diff(y)))
    if remainder.has(x):
        return None
    F2 = _integrate(remainder, y)
    if F2 is None:
        return None
    return as_expr(simplify(F1 + F2))


def rectify_symmetries(Phi: Expr, x: Symbol, y: Symbol, p: Symbol, f: AppliedUndef,
                       degree: int = 3) -> Optional[Linearization]:
    """Coordinates ``t(x, y)``, ``u(x, y)`` in which two commuting point
    symmetries of ``y'' = Phi`` (polynomial infinitesimals of total degree
    at most ``degree``) are the translations, so that the equation becomes
    ``u'' = rhs(u')``, solvable by quadratures; ``u'' = 0`` when the pair
    generates the translations of linearising coordinates.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.second_order import rectify_symmetries
    >>> x, y, p = symbols('x y p')
    >>> f = Function('y')(x)
    >>> rectify_symmetries(-3*y*p - y**3, x, y, p, f)      # y'' + 3 y y' + y^3 = 0
    Linearization(t=x - 1/y, u=1/(2*y**2), rhs=1)
    """
    from .ode import ode_symmetries
    equation = Derivative(f, (x, 2)) - Phi.xreplace({p: Derivative(f, x)}).xreplace({y: f})
    found = attempt(lambda: ode_symmetries(equation, f, degree=degree), settings.timeout)
    if not found:
        return None
    jets = found[0].jets
    ysym = jets.u[0]
    pair = commuting_pair(found, x, ysym)
    if pair is None:
        return None
    X1, X2 = pair
    xi1, eta1, xi2, eta2 = X1.xi[0], X1.eta[0], X2.xi[0], X2.eta[0]
    Delta = as_expr(simplify(xi1*eta2 - xi2*eta1))
    t = _potential(as_expr(cancel(eta2/Delta)), as_expr(cancel(-xi2/Delta)), x, ysym)
    u = _potential(as_expr(cancel(-eta1/Delta)), as_expr(cancel(xi1/Delta)), x, ysym)
    if t is None or u is None:
        return None
    t, u = as_expr(t.xreplace({ysym: y})), as_expr(u.xreplace({ysym: y}))
    v = Dummy('v')
    rhs = transformed_rhs(Phi, t, u, x, y, p, v)
    if rhs is None:
        return None
    return Linearization(t, u, x, y, v, rhs)


def _solutions_of_linearization(L: Linearization, f: AppliedUndef, x: Symbol) -> Optional[list[Basic]]:
    """The solutions of ``u'' = rhs(u')`` pulled back: ``psi(x, y) = U(phi(x, y))``,
    solved for ``y`` when possible."""
    t = Dummy('t')
    U = Function('U')(t)
    if L.rhs == 0:
        C1, C2 = Symbol('C1'), Symbol('C2')
        general: list[Expr] = [as_expr(C1*t + C2)]
    else:
        solved = attempt(lambda: dsolve(Eq(U.diff(t, 2), L.rhs.xreplace({L.v: U.diff(t)})), U), settings.timeout)
        if solved is None:
            return None
        general = [as_expr(s.rhs) for s in (solved if isinstance(solved, list) else [solved])
                   if isinstance(s, Eq) and s.lhs == U]
        if not general:
            return None
    results: list[Basic] = []
    for expression in general:
        relation = as_expr(L.u - expression.xreplace({t: L.t}))
        explicit = attempt(lambda: _solve(relation, L.y), settings.timeout)
        if explicit:
            for value in explicit:
                if isinstance(value, Expr):
                    results.append(Eq(f, value.xreplace({L.x: x})))
        else:
            results.append(Eq(relation.xreplace({L.x: x, L.y: f}), 0))
    return results or None


def dsolve_second_order(equation: Basic, f: AppliedUndef, degree: int = 3) -> Optional[list[Basic]]:
    """Solutions of a second order equation through an integrating factor
    (the first integral solved by ``dsolve``, or returned as an implicit
    first order equation), through its linearisation, or through two
    commuting symmetries (polynomial infinitesimals of degree at most
    ``degree``); ``None`` when none applies.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.second_order import dsolve_second_order
    >>> y = Function('y')(x)
    >>> dsolve_second_order(y.diff(x, 2) - y.diff(x)**2, y)
    [Eq(y(x), log(-1/(C1*x + C2)))]
    >>> dsolve_second_order(y.diff(x, 2) + 3*y*y.diff(x) + y**3, y)
    [Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
    """
    Phi, x, y, p = second_order_rhs(equation, f)
    for method in (integrating_factor_xy, integrating_factor_p):
        first_integral = attempt(lambda: method(Phi, x, y, p), settings.timeout)
        if first_integral is None:
            continue
        reduced = first_integral.equation(f)
        solved = attempt(lambda: dsolve(reduced, f), settings.timeout)
        if solved is None:
            return [reduced]
        return [solved] if isinstance(solved, Basic) else list(solved)
    L = attempt(lambda: linearize(Phi, x, y, p), settings.timeout)
    if L is None:
        L = attempt(lambda: rectify_symmetries(Phi, x, y, p, f, degree), settings.timeout)
    if L is None:
        return None
    return _solutions_of_linearization(L, f, x)
