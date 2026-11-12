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
* **Riccati equations** `y' = a y^2 + b y + c` with rational
  coefficients are linearised, `y = -u'/(a u)`, and the second order
  linear equation is solved by Kovacic's algorithm or in special
  functions (:mod:`sympy_extras.solvers.linear_ode`), so Riccati
  equations without rational particular solutions (SymPy's requirement)
  get Bessel or hypergeometric general solutions.
* **d'Alembert–Lagrange equations** `y = x F(y') + G(y')`: with `p = y'`,
  differentiation gives the linear equation `dx/dp - x F'(p)/(p - F(p))
  = G'(p)/(p - F(p))` for `x(p)`, solved by ``dsolve``; the solution is
  parametric, `x = X(p, C)`, `y = X(p, C) F(p) + G(p)` (Clairaut's
  equation, `F(p) = p`, is solved by SymPy).

SymPy's ``dsolve`` has hints for Bernoulli, Riccati and Clairaut
equations but none for Abel, Chini or Lagrange equations.

The module works on the objects of the core and imports no SymPy: the
coefficients, the cancellations, the roots and the simplifications are
the native ones, the linearised Riccati equations are solved by the
native :mod:`~sympy_extras.solvers.linear_ode` and
:mod:`~sympy_extras.solvers.special`, and
:mod:`sympy_extras.solvers.first_order_sympy` is its face on SymPy
objects. The antiderivatives of Chini's separated equation come from the
native indefinite integration (:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`,
SymPy's routes its last methods); one step is not ported and goes
through SymPy at run time: the linear equation of a Lagrange equation
(SymPy's ``dsolve``, :func:`~sympy_extras.solvers.first_order_sympy.dsolve_of_core`). The
Prelle–Singer procedure of :mod:`sympy_extras.solvers.darboux` is native.

References
==========

.. [Kamke] E. Kamke, Differentialgleichungen: Lösungsmethoden und
   Lösungen, Band 1, §4.9 (Abel), §4.10 (Chini), §4.11 (d'Alembert).

Examples
========

>>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
>>> from sympy_extras.core import Function, Symbol
>>> from sympy_extras.solvers.first_order import riccati_ode
>>> x = Symbol('x')
>>> y = Function('y')(x)
>>> riccati_ode(y.diff(x) + y**2 - 2/x**2, y)
Eq(y(x), (2*C1*x**3 - 1)/(C1*x**4 + x))
"""
from __future__ import annotations

from typing import Callable, Final, Optional

from sympy_extras.core.add import Add
from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, Function, expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.functions.elementary.miscellaneous import real_root
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel
from sympy_extras.simplify.powsimp import powdenest
from sympy_extras.simplify.simplify import simplify
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from ._special_values import solution_cases
from .kovacic import _coefficients
from .linear_ode import has_integral

__all__ = ['riccati_ode', 'chini_ode', 'abel_ode', 'lagrange_ode', 'dsolve_first_order']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)

#: a solver of first order equations: an (implicit) solution or ``None``
Solver = Callable[[Basic, AppliedUndef], Optional[Basic]]


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced, evaluated (SymPy's
    ``xreplace``)."""
    value = e._xreplace_evaluated(rule)
    if not isinstance(value, Expr):
        raise TypeError("%s is not an expression" % (value,))
    return value


def _normal(e: Expr) -> Expr:
    """The normal form ``p/q`` of a rational function (``e`` itself when
    the native ``cancel`` cannot take its coefficients)."""
    try:
        return cancel(e)
    except _FAILURES:
        return e


def _all_coeffs(e: Expr, v: Symbol, normal_form: bool = True) -> Optional[list[Expr]]:
    """The coefficients of ``e`` as a polynomial in ``v``, from the leading
    one down and each in its normal form (SymPy's ``Poly(e, v).all_coeffs()``,
    the coefficients any expressions free of ``v``), ``None`` when ``e`` is
    no polynomial in ``v``; ``[]`` for zero. Without ``normal_form`` the
    coefficients are the sums of the expanded terms (SymPy's domain
    ``EX``)."""
    try:
        terms = _coefficients(e, v)
    except ValueError:
        return None
    normal = {k: _normal(c) if normal_form else c for k, c in terms.items()}
    normal = {k: c for k, c in normal.items() if c != _ZERO}
    if not normal:
        return []
    return [normal.get(k, _ZERO) for k in range(max(normal), -1, -1)]


def _degree(coefficients: list[Expr]) -> int:
    """The degree of the polynomial of :func:`_all_coeffs` (``-1`` for
    zero)."""
    return len(coefficients) - 1


def _as_zero(equation: Basic) -> Expr:
    if isinstance(equation, Eq):
        return equation.lhs - equation.rhs
    if not isinstance(equation, Expr):
        raise TypeError("%s is not an equation" % (equation,))
    return equation


def _derivative_polynomial(equation: Basic, f: AppliedUndef) -> tuple[Expr, Symbol, Symbol, Symbol]:
    """``F(x, y, p) = 0`` with symbols for ``f`` and ``f'``."""
    x_ = f.args[0]
    if not isinstance(x_, Symbol):
        raise ValueError("the function must depend on a symbol")
    lhs = _as_zero(equation)
    u, p = Dummy('y'), Dummy('p')
    in_p = _replaced(lhs, {Derivative(f, x_): p})
    # the derivatives left are looked for before f is replaced: the
    # evaluating replacement makes Derivative(u, (x, 2)) zero
    if in_p.atoms(Derivative):
        raise ValueError("a first order equation in %s is expected" % (f,))
    replaced = _replaced(in_p, {f: u})
    if replaced.atoms(Derivative) or replaced.has(f):
        raise ValueError("a first order equation in %s is expected" % (f,))
    return replaced, x_, u, p


def _integrate(e: Expr, v: Symbol) -> Expr:
    """An antiderivative by the native verified indefinite integration
    (:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`,
    SymPy's routes last), the unevaluated
    :class:`~sympy_extras.integrals.integral.Integral` when none is found."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    from sympy_extras.integrals.integral import Integral
    found = closed_antiderivative(e, v)
    return Integral(e, v) if found is None else found


def _explicit_rhs(F: Expr, p: Symbol) -> Optional[Expr]:
    """``p`` solved from ``F = 0`` when ``F`` is linear in ``p``."""
    coefficients = _all_coeffs(F, p)
    if coefficients is None or _degree(coefficients) != 1:
        return None
    a1, a0 = coefficients
    return _normal(-a0/a1)


def chini_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Chini's equation ``y' = f(x) y**n + g(x) y +
    h(x)`` (``n != 1`` a number) when its invariant is constant, else
    ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.first_order import chini_ode
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> chini_ode(y.diff(x) - x**2*y**3 + x**2, y)      # doctest: +ELLIPSIS
    Eq(...)
    """
    F, x, u, p = _derivative_polynomial(equation, f)
    rhs = _explicit_rhs(F, p)
    if rhs is None:
        return None
    return _chini(rhs, x, u, f)


def _chini(rhs: Expr, x: Symbol, u: Symbol, f: Expr) -> Optional[Basic]:
    terms: dict[Expr, Expr] = {}
    for term in Add.make_args(expand(rhs)):
        coefficient, part = term.as_independent(u, as_Add=False)
        exponent: Expr
        if part == _ONE:
            exponent = _ZERO
        elif part == u:
            exponent = _ONE
        elif isinstance(part, Pow) and part.base == u and not part.exp.has(u, x):
            exponent = part.exp
        else:
            return None
        terms[exponent] = terms.get(exponent, _ZERO) + coefficient
    exponents = [e for e in terms if e != _ZERO and e != _ONE]
    if len(exponents) != 1:
        return None
    n = exponents[0]
    if not n.is_number or n == _ZERO:
        return None
    fn, g, h = terms[n], terms.get(_ONE, _ZERO), terms.get(_ZERO, _ZERO)
    if fn == _ZERO or h == _ZERO:
        return None
    # y = v exp(G), G = Integral(g): v' = fn exp((n-1)G) v**n + h exp(-G)
    G = _integrate(g, x) if g != _ZERO else _ZERO
    if has_integral(G):
        return None
    Fx = fn*exp((n - 1)*G)
    Hx = h*exp(-G)
    ratio = _normal(Hx/Fx)
    if n.is_integer and n.is_odd:
        # the real n-th root (the principal root of a negative number is complex)
        s = powdenest(real_root(ratio, n), force=True)
    else:
        s = powdenest(ratio**(1/n), force=True)
    c = simplify(powdenest(s.diff(x)/Hx, force=True))
    if c.has(x):
        return None
    w = Dummy('w')
    left = _integrate(1/(w**n + 1 - c*w), w)
    right = _integrate(_normal(Hx/s), x)
    C1 = Symbol('C1')
    w_value = f*exp(-G)/s
    return Eq(left.subs(w, w_value), right + C1)


def _abel_relation_holds(solution: Basic, coefficients: tuple[Expr, Expr, Expr, Expr], x: Symbol,
                         f: AppliedUndef) -> bool:
    """Whether an implicit relation returned for an Abel equation defines
    solutions of it (see :func:`sympy_extras.solvers.abel._defines_solution`)."""
    from .abel import _defines_solution
    if not isinstance(solution, Eq):
        return True
    relation = solution.lhs - solution.rhs
    return _defines_solution(relation, coefficients, x, f, Symbol('C1'))


def abel_ode(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    """An implicit solution of Abel's equations ``y' = f3 y**3 + f2 y**2 +
    f1 y + f0`` (first kind) and ``(y + g) y' = f2 y**2 + f1 y + f0``
    (second kind) in the constant-invariant cases, else ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.first_order import abel_ode
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> abel_ode(y.diff(x) - y**3 - 3*y**2 - 3*y, y)     # doctest: +ELLIPSIS
    Eq(...)
    """
    from .abel import abel_by_invariants, constant_invariant, abel_coefficients
    F, x, u, p = _derivative_polynomial(equation, f)
    in_p = _all_coeffs(F, p)
    if in_p is None or _degree(in_p) != 1:
        return None
    a1, a0 = in_p
    # a1(x, y) y' + a0(x, y) = 0
    if not a1.has(u):
        rhs = _normal(-a0/a1)
        try:
            coefficients, _ = abel_coefficients(Eq(f.diff(x), _replaced(rhs, {u: f})), f)
        except ValueError:
            return None
        if constant_invariant(coefficients, x) is not False:
            solution = _abel_first_kind(rhs, x, u, f, f)
            # the relation can lose x altogether and imply y' = 0
            # (sympy-extras#38): it is checked against the equation first
            if solution is not None and _abel_relation_holds(solution, coefficients, x, f):
                return solution
        # a non-constant invariant: the integrable classes recognised
        # through the invariants (the AIR class and the representatives)
        return abel_by_invariants(equation, f)
    # second kind: a1 linear in y, a0 quadratic in y
    p1 = _all_coeffs(a1, u)
    p0 = _all_coeffs(-a0, u)
    if p1 is None or p0 is None or _degree(p1) != 1 or _degree(p0) > 2:
        return None
    lead = p1[0]
    g = _normal(p1[1]/lead)
    rhs = _normal(-a0/lead)
    # y + g = 1/w: w' = -(y + g)' w**2 = -(rhs/(y + g) - g') w**2
    w = Dummy('w')
    v = 1/w - g
    w_rhs = _normal(-(_replaced(rhs, {u: v})*w - g.diff(x))*w**2)
    W = Function('W')(x)
    try:
        coefficients, _ = abel_coefficients(Eq(W.diff(x), _replaced(w_rhs, {w: W})), W)
    except ValueError:
        return None
    if constant_invariant(coefficients, x) is not False:
        solution = _abel_first_kind(w_rhs, x, w, 1/(f + g), f)
        # checked on the equation for W = 1/(y + g) (sympy-extras#38: the
        # relation of y y' = y + x had lost y and x altogether)
        if solution is not None and _abel_relation_holds(solution._xreplace_evaluated({f: 1/W - g}),
                                                         coefficients, x, W):
            return solution
    first_kind = abel_by_invariants(Eq(W.diff(x), _replaced(w_rhs, {w: W})), W)
    if not isinstance(first_kind, Eq):
        return None
    back: dict[Basic, Basic] = {W: 1/(f + g)}
    return Eq(first_kind.lhs.subs(back), first_kind.rhs.subs(back))


def _abel_first_kind(rhs: Expr, x: Symbol, u: Symbol, value: Expr, f: AppliedUndef) -> Optional[Basic]:
    """``u' = rhs(x, u)`` cubic in ``u``, with ``u`` standing for ``value``
    (an expression in ``f``)."""
    coefficients = _all_coeffs(expand(rhs), u)
    if coefficients is None or _degree(coefficients) != 3:
        return None
    f3, f2, f1, f0 = coefficients
    if f2 == _ZERO:
        return _chini_for(rhs, x, u, value)
    shift = _normal(f2/(3*f3))
    v = Dummy('v')
    # u = v - shift: v' = rhs(x, v - shift) + shift'
    new_rhs = expand(_replaced(rhs, {u: v - shift}) + shift.diff(x))
    return _chini_for(new_rhs, x, v, value + shift)


def _chini_for(rhs: Expr, x: Symbol, u: Symbol, value: Expr) -> Optional[Basic]:
    placeholder = Function('Y')(x)
    solution = _chini(rhs, x, u, placeholder)
    if solution is None:
        return None
    return solution._xreplace_evaluated({placeholder: value})


def lagrange_ode(equation: Basic, f: AppliedUndef) -> Optional[list[Basic]]:
    """The parametric solution ``[Eq(x, X(p)), Eq(y, Y(p))]`` of a
    d'Alembert–Lagrange equation ``y = x F(y') + G(y')`` (``p = y'`` the
    parameter), or ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's dsolve (first_order_sympy.dsolve_of_core)
    >>> from sympy_extras.core import Function, Symbol, Eq
    >>> from sympy_extras.solvers.first_order import lagrange_ode
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> lagrange_ode(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    [Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p))]
    """
    F, x, u, p = _derivative_polynomial(equation, f)
    in_u = _all_coeffs(F, u)
    if in_u is None or _degree(in_u) != 1:
        return None
    a1, a0 = in_u
    rhs = _normal(-a0/a1)   # y = rhs(x, p)
    in_x = _all_coeffs(rhs, x)
    if in_x is None or _degree(in_x) != 1:
        return None
    Fp, Gp = in_x
    if Fp.has(x) or Gp.has(x):
        return None
    if _normal(Fp - p) == _ZERO:
        return None   # Clairaut: SymPy's hint
    parameter = Symbol('p')
    Fp, Gp = _replaced(Fp, {p: parameter}), _replaced(Gp, {p: parameter})
    from .first_order_sympy import dsolve_of_core
    X = Function('X')(parameter)
    linear = Eq(X.diff(parameter) - X*Fp.diff(parameter)/(parameter - Fp), Gp.diff(parameter)/(parameter - Fp))
    solved = attempt(lambda: dsolve_of_core(linear, X), settings.timeout, native=True)
    if not isinstance(solved, Eq) or solved.lhs != X:
        return None
    X_value = solved.rhs
    if has_integral(X_value):
        return None
    return [Eq(x, X_value), Eq(f, simplify(X_value*Fp + Gp))]


def riccati_ode(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[Basic]:
    """The general solution of a Riccati equation ``y' = a(x) y**2 + b(x) y
    + c(x)`` with rational coefficients through the linear equation
    ``u'' - (a'/a + b) u' + a c u = 0`` (``y = -u'/(a u)``), solved by
    :func:`~sympy_extras.solvers.linear_ode.dsolve_linear` (Kovacic's
    algorithm and the special functions): ``y = -(u1' + C1 u2')/(a (u1 +
    C1 u2))``. ``None`` when the linear equation is not solved. SymPy's
    Riccati hints need a rational particular solution.

    With ``special_values``, the solution of an equation with parameters
    gets cases for the isolated values of the parameters at which it is
    undefined or loses its constant (``u1, u2`` dependent there), the
    equation solved again at each
    (:mod:`sympy_extras.solvers._special_values`).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.first_order import riccati_ode
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> riccati_ode(y.diff(x) + y**2 - 2/x**2, y)
    Eq(y(x), (2*C1*x**3 - 1)/(C1*x**4 + x))
    """
    found = _riccati(equation, f)
    if special_values and found is not None:
        cased = solution_cases([found], equation, f, lambda at: _solutions_of(riccati_ode(at, f)))
        if cased is not None:
            return cased
    return found


def _solutions_of(solution: Optional[Basic]) -> list[Basic]:
    return [] if solution is None else [solution]


def _riccati(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    from .linear_ode import dsolve_linear
    from .special import special_solutions
    F, x, u, p = _derivative_polynomial(equation, f)
    rhs = _explicit_rhs(F, p)
    if rhs is None:
        return None
    coefficients = _all_coeffs(expand(rhs), u)
    if coefficients is None or _degree(coefficients) != 2:
        return None
    a, b, c = coefficients
    if not all(coefficient.is_rational_function(x) for coefficient in (a, b, c)):
        return None
    w = Function('u')(x)
    linear = w.diff(x, 2) - (a.diff(x)/a + b)*w.diff(x) + a*c*w
    # the special functions are recognised at once; Kovacic's algorithm
    # (inside dsolve_linear) is slow to fail on equations without
    # Liouvillian solutions
    solutions = attempt(lambda: special_solutions(linear, w), settings.timeout, native=True)
    if not solutions:
        solutions = attempt(lambda: dsolve_linear(linear, w, special_values=False), settings.timeout, native=True)
    if not solutions:
        return None
    C1 = Symbol('C1')
    if len(solutions) >= 2:
        u1, u2 = solutions[0], solutions[1]
        numerator = u1.diff(x) + C1*u2.diff(x)
        denominator = a*(u1 + C1*u2)
    else:
        # a particular solution only
        u1 = solutions[0]
        numerator, denominator = u1.diff(x), a*u1
    value = -numerator/denominator
    if value.is_rational_function(x):
        value = _normal(value)
    return Eq(f, value)


def _prelle_singer(equation: Basic, f: AppliedUndef) -> Optional[Basic]:
    from .darboux import prelle_singer
    return prelle_singer(equation, f)


def dsolve_first_order(equation: Basic, f: AppliedUndef, special_values: bool = True) -> Optional[Basic]:
    """The Riccati general solution, the Chini/Abel implicit solution,
    the implicit solution of an equation ``y' = P(x, y)/Q(x, y)`` with
    polynomial ``P``, ``Q`` from the first integral of the Prelle–Singer
    procedure (:func:`~sympy_extras.solvers.darboux.prelle_singer`)
    or the Lagrange parametric solution (as a ``Tuple``) of a first order
    equation, else ``None``. With ``special_values``, an explicit
    solution of an equation with parameters gets cases for the isolated
    values of the parameters at which it is undefined or loses its
    constant (:mod:`sympy_extras.solvers._special_values`).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and dsolve (the *_of_core functions)
    >>> from sympy_extras.core import Function, Symbol, Eq
    >>> from sympy_extras.solvers.first_order import dsolve_first_order
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> dsolve_first_order(Eq(y, 2*x*y.diff(x) + y.diff(x)**2), y)
    (Eq(x, C1/p**2 - 2*p/3), Eq(y(x), (6*C1 - p**3)/(3*p)))
    """
    methods: tuple[Solver, ...] = (_riccati, abel_ode, chini_ode, _prelle_singer)
    for method in methods:
        solution = attempt(lambda: method(equation, f), settings.timeout, native=True)
        if solution is not None:
            if special_values and isinstance(solution, Eq):
                cased = solution_cases([solution], equation, f,
                                       lambda at: _solutions_of(dsolve_first_order(at, f)))
                if cased is not None:
                    return cased
            return solution
    parametric = attempt(lambda: lagrange_ode(equation, f), settings.timeout, native=True)
    if parametric:
        return Tuple(*parametric)
    return None
