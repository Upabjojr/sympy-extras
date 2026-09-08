"""First order ordinary differential equations of Abel, Chini and
d'Alembert–Lagrange type.

* **Chini's equation** `y' = f(x) y^n + g(x) y + h(x)`: the linear term
  is removed by `y = u e^{\\int g}`, giving `u' = F(x) u^n + H(x)`; with
  `s = (H/F)^{1/n}` and `u = s w` the equation becomes
  `w' = (H/s)(w^n + 1) - (s'/s) w`, which is separable exactly when
  `s'/H` is a constant `c` (Chini's invariant is constant):
  `\\int dw/(w^n + 1 - c w) = \\int H/s\\, dx + C`.
* **Abel's equation of the first kind** `y' = f_3 y^3 + f_2 y^2 + f_1 y
  + f_0` is Chini's equation with `n = 3` after the shift
  `y = v - f_2/(3 f_3)`; **Abel's equation of the second kind**
  `(y + g) y' = f_2 y^2 + f_1 y + f_0` becomes one of the first kind with
  `y + g = 1/w`. Only the constant-invariant cases are solved (the
  classical solvable classes; the general Abel equation has no closed
  form).
* **d'Alembert–Lagrange equations** `y = x F(y') + G(y')`: with `p = y'`,
  differentiation gives the linear equation `dx/dp - x F'(p)/(p - F(p))
  = G'(p)/(p - F(p))` for `x(p)`, solved by ``dsolve``; the solution is
  parametric, `x = X(p, C)`, `y = X(p, C) F(p) + G(p)` (Clairaut's
  equation, `F(p) = p`, is solved by SymPy).

SymPy's ``dsolve`` has hints for Bernoulli, Riccati and Clairaut
equations but none for Abel, Chini or Lagrange equations.

References
==========

.. [Kamke] E. Kamke, Differentialgleichungen: Lösungsmethoden und
   Lösungen, Band 1, §4.9 (Abel), §4.10 (Chini), §4.11 (d'Alembert).
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, Derivative, expand
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.settings import settings

__all__ = ['chini_ode', 'abel_ode', 'lagrange_ode', 'dsolve_first_order']


def _derivative_polynomial(equation: Basic, f: AppliedUndef) -> tuple[Expr, Symbol, Symbol, Symbol]:
    """``F(x, y, p) = 0`` with symbols for ``f`` and ``f'``."""
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
    u, p = Dummy('y'), Dummy('p')
    replaced = as_expr(lhs).xreplace({Derivative(f, x_): p}).xreplace({f: u})
    if replaced.atoms(Derivative) or replaced.has(f):
        raise ValueError("a first order equation in %s is expected" % (f,))
    return replaced, x_, u, p


def _integrate(e: Expr, v: Symbol) -> Expr:
    result = attempt(lambda: integrate(e, v, conds='none'), settings.timeout)
    if result is None:
        return Integral(e, v)
    return as_expr(result)


def _explicit_rhs(F: Expr, p: Symbol) -> Optional[Expr]:
    """``p`` solved from ``F = 0`` when ``F`` is linear in ``p``."""
    try:
        poly = Poly(F, p)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a1, a0 = poly.all_coeffs()
    return as_expr(cancel(-a0/a1))


def chini_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Chini's equation ``y' = f(x) y**n + g(x) y +
    h(x)`` (``n != 1`` a number) when its invariant is constant, else
    ``None``.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order import chini_ode
    >>> y = Function('y')(x)
    >>> chini_ode(y.diff(x) - x**2*y**3 + x**2, y)      # doctest: +ELLIPSIS
    Eq(...)
    """
    F, x, u, p = _derivative_polynomial(equation, f)
    rhs = _explicit_rhs(F, p)
    if rhs is None:
        return None
    return _chini(rhs, x, u, f)


def _chini(rhs: Expr, x: Symbol, u: Symbol, f: AppliedUndef) -> Optional[Basic]:
    from sympy.core.add import Add
    from sympy.core.power import Pow
    terms: dict[Expr, Expr] = {}
    for term in Add.make_args(expand(rhs)):
        coefficient, part = as_expr(term).as_independent(u, as_Add=False)
        exponent: Expr
        if part == 1:
            exponent = S.Zero
        elif part == u:
            exponent = S.One
        elif isinstance(part, Pow) and part.base == u and not part.exp.has(u, x):
            exponent = as_expr(part.exp)
        else:
            return None
        terms[exponent] = as_expr(terms.get(exponent, S.Zero) + coefficient)
    exponents = [e for e in terms if e not in (0, 1)]
    if len(exponents) != 1:
        return None
    n = exponents[0]
    if not n.is_number or n == 0:
        return None
    fn, g, h = terms[n], terms.get(S.One, S.Zero), terms.get(S.Zero, S.Zero)
    if fn == 0 or h == 0:
        return None
    # y = v exp(G), G = Integral(g): v' = fn exp((n-1)G) v**n + h exp(-G)
    G = _integrate(g, x) if g != 0 else S.Zero
    if G.has(Integral):
        return None
    Fx = as_expr(fn*exp((n - 1)*G))
    Hx = as_expr(h*exp(-G))
    from sympy.simplify.powsimp import powdenest
    from sympy.functions.elementary.miscellaneous import real_root
    ratio = as_expr(cancel(Hx/Fx))
    if n.is_integer and n.is_odd:
        # the real n-th root (the principal root of a negative number is complex)
        s = as_expr(powdenest(real_root(ratio, int(n)), force=True))
    else:
        s = as_expr(powdenest(ratio**(1/n), force=True))
    c = as_expr(simplify(powdenest(s.diff(x)/Hx, force=True)))
    if c.has(x):
        return None
    w = Dummy('w')
    left = _integrate(as_expr(1/(w**n + 1 - c*w)), w)
    right = _integrate(as_expr(cancel(Hx/s)), x)
    C1 = Symbol('C1')
    w_value = as_expr(f*exp(-G)/s)
    return Eq(as_expr(left.subs(w, w_value)), as_expr(right + C1))


def abel_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Abel's equations ``y' = f3 y**3 + f2 y**2 +
    f1 y + f0`` (first kind) and ``(y + g) y' = f2 y**2 + f1 y + f0``
    (second kind) in the constant-invariant cases, else ``None``.

    Examples
    ========

    >>> from sympy import Function, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order import abel_ode
    >>> y = Function('y')(x)
    >>> abel_ode(y.diff(x) - y**3 - 3*y**2 - 3*y, y)     # doctest: +ELLIPSIS
    Eq(...)
    """
    F, x, u, p = _derivative_polynomial(equation, f)
    try:
        poly = Poly(F, p)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a1, a0 = [as_expr(c) for c in poly.all_coeffs()]
    # a1(x, y) y' + a0(x, y) = 0
    if not a1.has(u):
        return _abel_first_kind(as_expr(cancel(-a0/a1)), x, u, f, f)
    # second kind: a1 linear in y, a0 quadratic in y
    try:
        p1 = Poly(a1, u)
        p0 = Poly(-a0, u)
    except PolynomialError:
        return None
    if p1.degree() != 1 or p0.degree() > 2:
        return None
    lead = as_expr(p1.LC())
    g = as_expr(cancel(p1.all_coeffs()[1]/lead))
    rhs = as_expr(cancel(p0.as_expr()/lead))
    # y + g = 1/w: w' = -(y + g)' w**2 = -(rhs/(y + g) - g') w**2
    w = Dummy('w')
    v = as_expr(1/w - g)
    w_rhs = as_expr(cancel(-(rhs.subs(u, v)*w - g.diff(x))*w**2))
    solution = _abel_first_kind(w_rhs, x, w, as_expr(1/(f + g)), f)
    return solution


def _abel_first_kind(rhs: Expr, x: Symbol, u: Symbol, value: Expr, f: AppliedUndef) -> Optional[Basic]:
    """``u' = rhs(x, u)`` cubic in ``u``, with ``u`` standing for ``value``
    (an expression in ``f``)."""
    try:
        poly = Poly(expand(rhs), u)
    except PolynomialError:
        return None
    if poly.degree() != 3:
        return None
    f3, f2, f1, f0 = [as_expr(c) for c in poly.all_coeffs()]
    if f2 == 0:
        return _chini(rhs, x, u, as_expr(value)) if False else _chini_for(rhs, x, u, value)
    shift = as_expr(cancel(f2/(3*f3)))
    v = Dummy('v')
    # u = v - shift: v' = rhs(x, v - shift) + shift'
    new_rhs = as_expr(expand(rhs.subs(u, v - shift) + shift.diff(x)))
    return _chini_for(new_rhs, x, v, as_expr(value + shift))


def _chini_for(rhs: Expr, x: Symbol, u: Symbol, value: Expr) -> Optional[Basic]:
    placeholder = Function('Y')(x)
    solution = _chini(rhs, x, u, placeholder)
    if solution is None:
        return None
    return solution.xreplace({placeholder: value})


def lagrange_ode(equation: Basic, f: AppliedUndef) -> Optional[list[Basic]]:
    """The parametric solution ``[Eq(x, X(p)), Eq(y, Y(p))]`` of a
    d'Alembert–Lagrange equation ``y = x F(y') + G(y')`` (``p = y'`` the
    parameter), or ``None``.

    Examples
    ========

    >>> from sympy import Function, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.first_order import lagrange_ode
    >>> y = Function('y')(x)
    >>> lagrange_ode(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    [Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p))]
    """
    F, x, u, p = _derivative_polynomial(equation, f)
    try:
        poly = Poly(F, u)
    except PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a1, a0 = [as_expr(c) for c in poly.all_coeffs()]
    rhs = as_expr(cancel(-a0/a1))   # y = rhs(x, p)
    try:
        in_x = Poly(rhs, x)
    except PolynomialError:
        return None
    if in_x.degree() != 1:
        return None
    Fp, Gp = [as_expr(c) for c in in_x.all_coeffs()]
    if Fp.has(x) or Gp.has(x):
        return None
    if cancel(Fp - p) == 0:
        return None   # Clairaut: SymPy's hint
    parameter = Symbol('p')
    Fp, Gp = as_expr(Fp.subs(p, parameter)), as_expr(Gp.subs(p, parameter))
    X = Function('X')(parameter)
    linear = Eq(X.diff(parameter) - X*Fp.diff(parameter)/(parameter - Fp), Gp.diff(parameter)/(parameter - Fp))
    solved = attempt(lambda: dsolve(linear, X), settings.timeout)
    if not isinstance(solved, Eq) or solved.lhs != X:
        return None
    X_value = as_expr(solved.rhs)
    if X_value.has(Integral):
        return None
    return [Eq(x, X_value), Eq(f, as_expr(simplify(X_value*Fp + Gp)))]


def dsolve_first_order(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """The Chini/Abel implicit solution or the Lagrange parametric
    solution (as a list) of a first order equation, else ``None``."""
    for method in (abel_ode, chini_ode):
        solution = attempt(lambda: method(equation, f), settings.timeout)
        if solution is not None:
            return solution
    parametric = attempt(lambda: lagrange_ode(equation, f), settings.timeout)
    if parametric:
        from sympy.core.containers import Tuple
        return Tuple(*parametric)
    return None
