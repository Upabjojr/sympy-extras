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

The module works on the objects of the core and imports no SymPy: the
coefficients, the factorisations, the cancellations, the
simplifications, the linear systems of the structure constants and the
symmetries (:mod:`sympy_extras.solvers.ode`) are the native ones, and
:mod:`sympy_extras.solvers.second_order_sympy` is its face on SymPy
objects. The quadratures come from the native indefinite integration
(:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`); two steps have no
native counterpart and go through SymPy at run time (the functions
ending in ``_of_core`` of the faces): SymPy's
``dsolve`` of the linear equations, the first integrals and the
equations ``u'' = F(u')``
(:func:`~sympy_extras.solvers.first_order_sympy.dsolve_of_core`) and its
``solve`` of the transformed derivative, of the pulled back solutions
and of a linear system of structure constants which is not uniquely
solvable over the rational functions are the native ports of
:mod:`sympy_extras.solvers.solvers`.

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

Examples
========

>>> import sympy_extras.interop  # SymPy's integrate, dsolve and solve (the *_of_core functions)
>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.second_order import dsolve_second_order
>>> x = Symbol('x')
>>> y = Function('y')(x)
>>> dsolve_second_order(y.diff(x, 2) + 3*y*y.diff(x) + y**3, y)
[Eq(y(x), 2*(C2 + x)/(2*C1 + 2*C2*x + x**2))]
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.exprtools import factor_terms
from sympy_extras.core.function import AppliedUndef, Derivative, Function, diff, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel, factor_list
from sympy_extras.polys.rationaltools import together
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify

from ._special_values import as_zero, solution_cases
from .kovacic import _coefficients
from .lie import Symmetry, _polynomial_terms, _replaced
from .linear_ode import _nullspace
from .linsolve import solve_linear_unique

__all__ = ['second_order_rhs', 'integrating_factor_xy', 'integrating_factor_p', 'is_linearizable',
           'linearize', 'rectify_symmetries', 'commuting_pair', 'transformed_rhs', 'dsolve_second_order',
           'FirstIntegral', 'Linearization']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


def _expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression, got %s" % (e,))
    return e


def _normal(e: Expr) -> Expr:
    """``e`` in lowest terms (``e`` itself when the native ``cancel``
    cannot take its coefficients)."""
    try:
        return cancel(e)
    except _FAILURES:
        return e


def _all_coeffs(e: Expr, v: Symbol) -> Optional[list[Expr]]:
    """The coefficients of ``e`` as a polynomial in ``v`` from the leading
    one down (SymPy's ``Poly(e, v).all_coeffs()``, any coefficients free of
    ``v``; ``[0]`` for zero), ``None`` when ``e`` is no polynomial in
    ``v``."""
    try:
        terms = _coefficients(e, v)
    except ValueError:
        return None
    if not terms:
        return [_ZERO]
    return [terms.get(k, _ZERO) for k in range(max(terms), -1, -1)]


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
        self.factor: Final[Expr] = factor
        self.integral: Final[Expr] = integral
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.p: Final[Symbol] = p

    def __repr__(self) -> str:
        return "FirstIntegral(%s, %s)" % (self.factor, self.integral)

    def equation(self, f: AppliedUndef) -> Eq:
        """``R(x, f, f') = C1`` as an equation in the function ``f``."""
        x_ = f.args[0]
        if not isinstance(x_, Symbol):
            raise ValueError("the function must depend on a symbol")
        R = _replaced(_replaced(self.integral, {self.p: Derivative(f, x_)}), {self.y: f, self.x: x_})
        return Eq(R, Symbol('C1'))


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
        self.t: Final[Expr] = t
        self.u: Final[Expr] = u
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.v: Final[Symbol] = v
        self.rhs: Final[Expr] = rhs

    def __repr__(self) -> str:
        if self.rhs == 0:
            return "Linearization(t=%s, u=%s)" % (self.t, self.u)
        return "Linearization(t=%s, u=%s, rhs=%s)" % (self.t, self.u, self.rhs)


def second_order_rhs(equation: Basic, f: AppliedUndef) -> tuple[Expr, Symbol, Symbol, Symbol]:
    """``(Phi, x, y, p)`` with the equation written as ``y'' = Phi(x, y, p)``
    (``p`` stands for ``y'``); ``ValueError`` when the equation is not
    linear in ``y''``.

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.second_order import second_order_rhs
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> Phi, x_, y_, p_ = second_order_rhs(x*y.diff(x, 2) + y.diff(x) - y**2, y)
    >>> Phi
    (-_p + _y**2)/x
    """
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    lhs = as_zero(equation)
    y, p, q = Dummy('y'), Dummy('p'), Dummy('q')
    replaced = _replaced(_replaced(_replaced(lhs, {Derivative(f, (x_, 2)): q}), {Derivative(f, x_): p}), {f: y})
    if replaced.atoms(Derivative) or replaced.has(f):
        raise ValueError("%s is not a second order equation in %s" % (equation, f))
    coefficients = _all_coeffs(replaced, q)
    if coefficients is None or len(coefficients) != 2:
        raise ValueError("%s is not linear in the second derivative" % (equation,))
    a1, a0 = coefficients
    return _normal(-a0/a1), x_, y, p


def _integrate(e: Expr, v: Symbol) -> Optional[Expr]:
    """An antiderivative by SymPy's ``integrate`` under the time limit of
    the settings, ``None`` when it gives up or leaves an ``Integral``."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    return closed_antiderivative(e, v, cases=True)


def _dsolve(equation: Basic, f: AppliedUndef) -> Optional[list[Basic]]:
    """What SymPy's ``dsolve`` returns under the time limit of the
    settings, as a list (one ``Eq``, or the several ones of a list),
    ``None`` when it gives up."""
    from .first_order_sympy import dsolve_of_core
    found = dsolve_of_core(equation, f)
    if found is None:
        return None
    if isinstance(found, Tuple):
        return list(found.args)
    return [found]


def _solve(e: Expr, v: Symbol) -> list[Expr]:
    """The solutions of ``e = 0`` for ``v`` by the native port of SymPy's
    ``solve`` under the time limit of the settings (none when it gives up)."""
    from .solvers import solve_equation
    return attempt(lambda: solve_equation(e, v), settings.timeout, native=True) or []


def _particular(solution: Basic) -> Optional[Expr]:
    """A nonzero particular solution from ``dsolve``'s general one, with
    the first constant set to 1 and the others to 0."""
    if not isinstance(solution, Eq):
        return None
    rhs = _expr(solution.rhs)
    constants = sorted((s for s in rhs.free_symbols if s.name.startswith('C')), key=lambda s: s.name)
    if not constants:
        return rhs
    for chosen in constants:
        values: dict[Basic, Basic] = {c: (_ONE if c == chosen else _ZERO) for c in constants}
        candidate = simplify(_replaced(rhs, values))
        if candidate != 0:
            return candidate
    return None


def _solve_linear(coefficient1: Expr, coefficient0: Expr, x: Symbol) -> Optional[Expr]:
    """A nonzero solution of ``w'' + coefficient1 w' + coefficient0 w = 0``."""
    W = Function('W')(x)
    equation = diff(W, x, 2) + coefficient1*diff(W, x) + coefficient0*W
    solution = _dsolve(equation, W)
    if solution is None or len(solution) != 1:
        # SymPy's dsolve gave up, or gave several solutions
        return None
    return _particular(solution[0])


def integrating_factor_xy(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(x, y)`` of ``y'' = Phi`` and the first
    integral ``mu y' + S(x, y)``, or ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and dsolve (the *_of_core functions)
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.second_order import integrating_factor_xy
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_xy(-p**2/y, x, y, p)          # y y'' + y'^2 = 0
    FirstIntegral(y, p*y)
    >>> integrating_factor_xy(-p/x, x, y, p)             # x y'' + y' = 0
    FirstIntegral(x, p*x)
    """
    coefficients = _all_coeffs(expand(Phi), p)
    if coefficients is None or len(coefficients) > 3:
        return None
    while len(coefficients) < 3:
        coefficients.insert(0, _ZERO)
    a, b, c = coefficients
    # mu = m(x) E(x, y) with E_y = -a E
    A = _integrate(a, y)
    if A is None:
        return None
    E = exp(-A)
    Ex = diff(E, x)/E
    Exx = diff(E, x, 2)/E
    coefficient1 = _normal(b + 2*Ex)
    coefficient0 = _normal(diff(b, x) + b*Ex + Exx - diff(c, y) + a*c)
    if coefficient1.has(y) or coefficient0.has(y):
        return None
    m = _solve_linear(coefficient1, coefficient0, x)
    if m is None:
        return None
    mu = simplify(m*E)
    # S_x = -c mu, S_y = -b mu - mu_x
    S1 = _integrate(-c*mu, x)
    if S1 is None:
        return None
    remainder = simplify(-b*mu - diff(mu, x) - diff(S1, y))
    if remainder.has(x):
        return None
    S2 = _integrate(remainder, y)
    if S2 is None:
        return None
    integral = simplify(mu*p + S1 + S2)
    return FirstIntegral(mu, integral, x, y, p)


def _factor_in_p(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``Phi = g(p) * L(x, y, p)`` with ``L`` free of factors in ``p``
    alone: ``(g, L)``. The factors of the product structure (powers,
    exponentials) are separated first, the polynomial content afterwards."""
    g_parts: list[Expr] = []
    rest_parts: list[Expr] = []
    for f_ in Mul.make_args(factor_terms(Phi)):
        if f_.has(p) and not (f_.has(x) or f_.has(y)):
            g_parts.append(f_)
        else:
            rest_parts.append(f_)
    rest = mul(*rest_parts)
    numerator, denominator = together(rest).as_numer_denom()
    polynomial_parts: list[Expr] = []
    for part, sign in ((numerator, 1), (denominator, -1)):
        try:
            constant, factors = factor_list(part, x, y, p)
        except _FAILURES:
            return None
        polynomial_parts.append(pow(constant, Integer(sign)))
        for base, exponent in factors:
            if base.has(p) and not (base.has(x) or base.has(y)):
                g_parts.append(pow(base, sign*exponent))
            else:
                polynomial_parts.append(pow(base, sign*exponent))
    return mul(*g_parts), mul(*polynomial_parts)


def integrating_factor_p(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[FirstIntegral]:
    """An integrating factor ``mu(y')`` of ``y'' = Phi`` and the first
    integral ``M(y') + S(x, y)``, or ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.second_order import integrating_factor_p
    >>> x, y, p = symbols('x y p')
    >>> integrating_factor_p(p**3*(x + y*p), x, y, p)     # y'' = y'^3 (x + y y')
    FirstIntegral(p**(-3), -x**2/2 - y**2/2 - 1/(2*p**2))
    """
    factored = _factor_in_p(Phi, x, y, p)
    if factored is None:
        return None
    g, L = factored
    coefficients = _all_coeffs(expand(L), p)
    if coefficients is None or len(coefficients) > 2:
        return None
    B, A = (coefficients if len(coefficients) == 2 else [_ZERO, coefficients[0]])
    if simplify(diff(A, y) - diff(B, x)) != 0:
        return None
    mu = 1/g
    M = _integrate(mu, p)
    S1 = _integrate(-A, x)
    if M is None or S1 is None:
        return None
    remainder = simplify(-B - diff(S1, y))
    if remainder.has(x):
        return None
    S2 = _integrate(remainder, y)
    if S2 is None:
        return None
    return FirstIntegral(mu, M + S1 + S2, x, y, p)


def _cubic_coefficients(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr]]:
    """``(a, b, c, d)`` with ``y'' + a y'^3 + b y'^2 + c y' + d = 0``."""
    coefficients = _all_coeffs(expand(-Phi), p)
    if coefficients is None or len(coefficients) > 4:
        return None
    while len(coefficients) < 4:
        coefficients.insert(0, _ZERO)
    a, b, c, d = coefficients
    return a, b, c, d


def _lie_conditions(a: Expr, b: Expr, c: Expr, d: Expr, x: Symbol, y: Symbol) -> tuple[Expr, Expr]:
    first = (3*diff(a, x, 2) - 2*diff(b, x, y) + diff(c, y, 2) - 3*diff(a, x)*c + 3*diff(a, y)*d
             + 2*diff(b, x)*b - 3*diff(c, x)*a - diff(c, y)*b + 6*diff(d, y)*a)
    second = (diff(b, x, 2) - 2*diff(c, x, y) + 3*diff(d, y, 2) - 6*diff(a, x)*d + diff(b, x)*c
              + 3*diff(b, y)*d - 2*diff(c, y)*c - 3*diff(d, x)*a + 3*diff(d, y)*b)
    return simplify(first), simplify(second)


def is_linearizable(Phi: Expr, x: Symbol, y: Symbol, p: Symbol) -> bool:
    """Lie's test: whether ``y'' = Phi`` is equivalent to ``u'' = 0`` under
    a point transformation.

    Examples
    ========

    >>> from sympy_extras.core import symbols
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
    if a != 0 or simplify(diff(c, y) - 2*diff(b, x)) != 0:
        return None
    B = _integrate(b, y)
    if B is None:
        return None
    Bx = diff(B, x)
    Q = simplify(diff(B, x, 2) - Bx**2 + c*Bx - diff(d, y) - b*d)
    coefficient1 = simplify(c - 2*Bx)
    if Q.has(y) or coefficient1.has(y):
        return None
    w = _solve_linear(coefficient1, -Q, x)
    if w is None:
        return None
    k = 1/w
    r = simplify(2*Bx - c - 2*diff(w, x)/w)
    if r.has(y):
        return None
    Psi = _integrate(exp(B), y)
    if Psi is None:
        return None
    # the y-free remainder gives l(x): l'' - r l' = remainder
    remainder = simplify(d*k*exp(B) - (diff(k, x, 2)*Psi + 2*diff(k, x)*diff(Psi, x) + k*diff(Psi, x, 2)
                                       - r*(diff(k, x)*Psi + k*diff(Psi, x))))
    if remainder.has(y):
        return None
    R = _integrate(r, x)
    if R is None:
        return None
    phi_prime = exp(R)
    phi = _integrate(phi_prime, x)
    if phi is None:
        return None
    l_prime: Expr = _ZERO
    if remainder != 0:
        inner = _integrate(remainder/phi_prime, x)
        if inner is None:
            return None
        l_prime = phi_prime*inner
    l = _integrate(l_prime, x) if l_prime != 0 else _ZERO
    if l is None:
        return None
    psi = simplify(k*Psi + l)
    return phi, psi


def transformed_rhs(Phi: Expr, t: Expr, u: Expr, x: Symbol, y: Symbol, p: Symbol, v: Symbol) -> Optional[Expr]:
    """``y'' = Phi`` written in the variables ``t(x, y)``, ``u(x, y)`` as
    ``u'' = rhs``, when ``rhs`` is a function of ``v = u'`` alone.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.second_order import transformed_rhs
    >>> x, y, p, v = symbols('x y p v')
    >>> transformed_rhs(-3*y*p - y**3, x - 1/y, 1/(2*y**2), x, y, p, v)
    1
    """
    Dt = diff(t, x) + diff(t, y)*p
    Du = diff(u, x) + diff(u, y)*p
    V = _normal(Du/Dt)
    DV = diff(V, x) + diff(V, y)*p + diff(V, p)*Phi
    second = _normal(DV/Dt)
    solutions = attempt(lambda: _solve(V - v, p), settings.timeout, native=True)
    if not solutions:
        return None
    for value in solutions:
        candidate = attempt(lambda: simplify(_replaced(second, {p: value})), settings.timeout, native=True)
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

    >>> import sympy_extras.interop  # SymPy's integrate and dsolve (the *_of_core functions)
    >>> from sympy_extras.core import symbols
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
        return Linearization(found[0], found[1], x, y, v, _ZERO)
    # x as a function of y: y' = 1/x', y'' = -x''/x'^3
    q = Dummy('q')
    swapped = _normal(-_replaced(Phi, {p: 1/q})*q**3)
    swapped_coefficients = _cubic_coefficients(swapped, y, x, q)
    if swapped_coefficients is None:
        return None
    found = _fibre_preserving(*swapped_coefficients, y, x)
    if found is None:
        return None
    return Linearization(found[0], found[1], x, y, v, _ZERO)


def _combination(symmetries: Sequence[Symmetry], coefficients: Sequence[Symbol]) -> Symmetry:
    combination = symmetries[0]*coefficients[0]
    for k in range(1, len(symmetries)):
        combination = combination + symmetries[k]*coefficients[k]
    return combination


def _linear_solution(equations: list[Expr], unknowns: list[Symbol]) -> Optional[list[Expr]]:
    """The values of the unknowns of a linear system: natively when they
    are unique over the rational functions, else the first solution of
    the native port of SymPy's ``solve`` (the unknowns it leaves free set
    to zero)."""
    unique = solve_linear_unique(equations, unknowns)
    if unique is not None:
        return unique
    from .solvers import solve_system
    solution = attempt(lambda: solve_system(equations, unknowns), settings.timeout, native=True)
    if not solution:
        return None
    return [solution[0].get(c, _ZERO) for c in unknowns]


def _structure(symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[list[list[list[Expr]]]]:
    """The structure constants ``C[i][j][k]`` with ``[X_i, X_j] = sum_k
    C[i][j][k] X_k`` (``None`` when a bracket is not in the span)."""
    m = len(symmetries)
    coefficients = [Symbol('c%d' % k) for k in range(m)]
    combination = _combination(symmetries, coefficients)
    result: list[list[list[Expr]]] = []
    for i in range(m):
        row: list[list[Expr]] = []
        for j in range(m):
            bracket = symmetries[i].commutator(symmetries[j])
            equations: list[Expr] = []
            for lhs, rhs in ((combination.xi[0], bracket.xi[0]), (combination.eta[0], bracket.eta[0])):
                equations.extend(_coefficients_in(expand(lhs - rhs), x, y))
            values = _linear_solution(equations, coefficients)
            if values is None:
                return None
            row.append(values)
        result.append(row)
    return result


def _coefficients_in(e: Expr, x: Symbol, y: Symbol) -> list[Expr]:
    """The coefficients of an expression linear in the unknowns with
    respect to the functions of ``x`` and ``y`` it contains."""
    terms = _polynomial_terms(e, [x, y])
    if terms is not None:
        return list(terms.values()) or [_ZERO]
    collected: dict[Expr, Expr] = {}
    for term in Add.make_args(e):
        constant, function = term.as_independent(x, y)
        collected[function] = collected.get(function, _ZERO) + constant
    return list(collected.values())


def commuting_pair(symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[tuple[Symmetry, Symmetry]]:
    """Two commuting symmetries which are pointwise independent as vector
    fields (``xi_1 eta_2 - xi_2 eta_1 != 0``): among the basis elements
    and their sums first, then in the centralizers of a few elements when
    the brackets close in the span of the basis; ``None`` otherwise.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry
    >>> from sympy_extras.solvers.second_order import commuting_pair
    >>> x = symbols('x')
    >>> jets = JetSpace([Function('y')(x)])
    >>> u = jets.u[0]
    >>> commuting_pair([Symmetry(jets, [1], [0]), Symmetry(jets, [x], [u]), Symmetry(jets, [0], [1])], x, u)
    (Symmetry(xi=(1,), eta=(0,)), Symmetry(xi=(0,), eta=(1,)))
    """
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
            rows.append([add(*[a[i]*C[i][j][k] for i in range(m)]) for j in range(m)])
        for vector in _nullspace(rows, m):
            Y = sum((symmetries[j]*vector[j] for j in range(1, m)), symmetries[0]*vector[0])
            if not Y.is_zero() and independent(X, Y):
                return X, Y
    return None


def _coordinates(X: Symmetry, symmetries: Sequence[Symmetry], x: Symbol, y: Symbol) -> Optional[list[Expr]]:
    m = len(symmetries)
    coefficients = [Symbol('c%d' % k) for k in range(m)]
    combination = _combination(symmetries, coefficients)
    equations: list[Expr] = []
    for lhs, rhs in ((combination.xi[0], X.xi[0]), (combination.eta[0], X.eta[0])):
        equations.extend(_coefficients_in(expand(lhs - rhs), x, y))
    return _linear_solution(equations, coefficients)


def _potential(fx: Expr, fy: Expr, x: Symbol, y: Symbol) -> Optional[Expr]:
    """``F`` with ``F_x = fx`` and ``F_y = fy``."""
    if simplify(diff(fx, y) - diff(fy, x)) != 0:
        return None
    F1 = _integrate(fx, x)
    if F1 is None:
        return None
    remainder = simplify(fy - diff(F1, y))
    if remainder.has(x):
        return None
    F2 = _integrate(remainder, y)
    if F2 is None:
        return None
    return simplify(F1 + F2)


def rectify_symmetries(Phi: Expr, x: Symbol, y: Symbol, p: Symbol, f: AppliedUndef,
                       degree: int = 3) -> Optional[Linearization]:
    """Coordinates ``t(x, y)``, ``u(x, y)`` in which two commuting point
    symmetries of ``y'' = Phi`` (polynomial infinitesimals of total degree
    at most ``degree``) are the translations, so that the equation becomes
    ``u'' = rhs(u')``, solvable by quadratures; ``u'' = 0`` when the pair
    generates the translations of linearising coordinates.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and solve (the *_of_core functions)
    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.second_order import rectify_symmetries
    >>> x, y, p = symbols('x y p')
    >>> f = Function('y')(x)
    >>> rectify_symmetries(-3*y*p - y**3, x, y, p, f)      # y'' + 3 y y' + y^3 = 0
    Linearization(t=x - 1/y, u=1/(2*y**2), rhs=1)
    """
    from .ode import ode_symmetries
    equation = Derivative(f, (x, 2)) - _replaced(_replaced(Phi, {p: Derivative(f, x)}), {y: f})
    found = attempt(lambda: ode_symmetries(equation, f, degree=degree), settings.timeout, native=True)
    if not found:
        return None
    jets = found[0].jets
    ysym = jets.u[0]
    pair = commuting_pair(found, x, ysym)
    if pair is None:
        return None
    X1, X2 = pair
    xi1, eta1, xi2, eta2 = X1.xi[0], X1.eta[0], X2.xi[0], X2.eta[0]
    Delta = simplify(xi1*eta2 - xi2*eta1)
    t = _potential(_normal(eta2/Delta), _normal(-xi2/Delta), x, ysym)
    u = _potential(_normal(-eta1/Delta), _normal(xi1/Delta), x, ysym)
    if t is None or u is None:
        return None
    t, u = _replaced(t, {ysym: y}), _replaced(u, {ysym: y})
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
    general: list[Expr]
    if L.rhs == 0:
        C1, C2 = Symbol('C1'), Symbol('C2')
        general = [C1*t + C2]
    else:
        solved = _dsolve(Eq(diff(U, t, 2), _replaced(L.rhs, {L.v: diff(U, t)})), U)
        if solved is None:
            return None
        general = [_expr(s.rhs) for s in solved if isinstance(s, Eq) and s.lhs == U]
        if not general:
            return None
    results: list[Basic] = []
    for expression in general:
        relation = L.u - _replaced(expression, {t: L.t})
        explicit = attempt(lambda: _solve(relation, L.y), settings.timeout, native=True)
        if explicit:
            for value in explicit:
                results.append(Eq(f, _replaced(value, {L.x: x})))
        else:
            results.append(Eq(_replaced(relation, {L.x: x, L.y: f}), _ZERO))
    return results or None


def _unknown_under_integral(candidate: Basic, f: AppliedUndef) -> bool:
    """Whether ``f`` occurs in the integrand of an ``Integral`` (SymPy's,
    wrapped) over its own argument -- ``Integral(y(x)*..., x)`` -- which is
    no solution. An integral over a dummy with ``f`` as a limit,
    ``Integral(exp(t**2/2), (t, y(x)))``, is an ordinary implicit solution
    and is kept."""
    from sympy_extras.integrals.integral import Integral
    x = f.args[0]
    for node in preorder_traversal(candidate):
        if isinstance(node, Integral):
            # a native integral is over its own variable
            if node.function.has(f) and node.variable == x:
                return True
            continue
        if not (node._is_foreign_node() and node._compare_name() == 'Integral') or not node.args:
            continue
        if not node.args[0].has(f):
            continue
        if any(isinstance(limit, Tuple) and limit.args and limit.args[0] == x for limit in node.args[1:]):
            return True
    return False


def dsolve_second_order(equation: Basic, f: AppliedUndef, degree: int = 3,
                        special_values: bool = True) -> Optional[list[Basic]]:
    """Solutions of a second order equation through an integrating factor
    (the first integral solved by ``dsolve``, or returned as an implicit
    first order equation), through its linearisation, or through two
    commuting symmetries (polynomial infinitesimals of degree at most
    ``degree``); ``None`` when none applies. With ``special_values``, a
    single explicit solution of an equation with parameters gets cases
    for the isolated values of the parameters at which it is undefined
    or no longer general (:mod:`sympy_extras.solvers._special_values`).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate, dsolve and solve (the *_of_core functions)
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.second_order import dsolve_second_order
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> dsolve_second_order(y.diff(x, 2) - y.diff(x)**2, y)
    [Eq(y(x), log(-1/(C1*x + C2)))]
    """
    found = _dsolve_second_order(equation, f, degree)
    if special_values and found:
        cased = solution_cases(found, equation, f, lambda at: dsolve_second_order(at, f, degree) or [])
        if cased is not None:
            return [cased]
    return found


def _dsolve_second_order(equation: Basic, f: AppliedUndef, degree: int) -> Optional[list[Basic]]:
    Phi, x, y, p = second_order_rhs(equation, f)
    for method in (integrating_factor_xy, integrating_factor_p):
        first_integral = attempt(lambda: method(Phi, x, y, p), settings.timeout, native=True)
        if first_integral is None:
            continue
        reduced = first_integral.equation(f)
        found = _dsolve(reduced, f)
        if found is None:
            return [reduced]
        # an answer with the unknown still under an integral is not a
        # solution; the first integral itself is the honest reduction
        # (sympy-extras#29)
        if any(_unknown_under_integral(s, f) for s in found):
            return [reduced]
        return found
    L = attempt(lambda: linearize(Phi, x, y, p), settings.timeout, native=True)
    if L is None:
        L = attempt(lambda: rectify_symmetries(Phi, x, y, p, f, degree), settings.timeout, native=True)
    if L is None:
        return None
    return _solutions_of_linearization(L, f, x)
