"""Abel equations of the first kind: invariants, equivalence classes and
the integrable classes recognised through them.

An Abel equation of the first kind `y' = f_3 y^3 + f_2 y^2 + f_1 y + f_0`
keeps its form under the transformations `y = P(x)\\, u + Q(x)`,
`x = \\xi(t)`, and every equation is equivalent under them to a *normal
form* `u' = F(x) u^3 + G(x)`: the shift `y = v - f_2/(3 f_3)` removes the
`y^2` term, giving `v' = f_3 v^3 + g_1 v + g_0`, and `v = w e^{\\int g_1}`
removes the linear term. In the canonical variable `s = \\int F\\, dx` the
normal form is `du/ds = u^3 + \\Phi(s)`, `\\Phi = G/F`, and what is left of
the transformation group is `u \\to \\lambda u`, `s \\to \\lambda^{-2} s + c`;
the quantities

.. math::

    I_1 = \\frac{\\Phi_s^3}{\\Phi^5}, \\qquad I_2 = \\frac{\\Phi_{ss}\\,\\Phi}{\\Phi_s^2}

are invariant under it (the exponentials cancel, so they are rational in
the coefficients and their derivatives). In terms of the classical
relative invariants

.. math::

    s_3 = f_0 f_3^2 + \\tfrac{2}{27} f_2^3 - \\tfrac13 f_1 f_2 f_3
          + \\tfrac13 (f_3 f_2' - f_2 f_3'), \\qquad
    s_5 = f_3 s_3' - 3 f_3' s_3 - 3 f_1 f_3 s_3 + f_2^2 s_3,

`I_1 = s_5^3 / s_3^5`. Two equations are equivalent exactly when the
curves `s \\mapsto (I_1, I_2)` coincide: `\\xi` is found from
`I_1(x) = J_1(\\xi)` and checked on `I_2`, and `P`, `Q` follow from the
coefficients (Liouville, Appell; Cheb-Terrab and Roche).

Integrable classes:

* a constant `I_1` (Chini's invariant constant): solved by separation of
  variables in :mod:`sympy_extras.solvers.first_order`;
* the **AIR class** (Abel, inverse Riccati): the equation
  `(g_1(x) y + g_0(x))\\, y' = 1` with quadratic polynomials `g_1`, `g_0`
  is, for `x` as a function of `y`, the Riccati equation
  `dx/dy = g_1(x) y + g_0(x)`, solved through its linearisation
  (:func:`~sympy_extras.solvers.first_order.riccati_ode`); with
  `y + g_0/g_1 = 1/w` it is the first kind equation
  `w' = -w^3/g_1 - (g_0/g_1)' w^2`. A target equation belongs to the class
  (with `\\xi` the identity) when it has a particular solution `Q`
  (polynomial or rational ones are searched) such that
  `P = e^{\\int (3 f_3 Q^2 + 2 f_2 Q + f_1)}` makes `g_1 = -1/(f_3 P^2)`
  and `g_0 = -g_1 \\int P (3 f_3 Q + f_2)\\, dx` quadratic polynomials;
* a database of representatives with non-constant invariant, matched
  through `(I_1, I_2)` with `\\xi` reconstructed: the AIR representatives
  `w' = -w^3 - 2 t w^2` (Airy functions), `w' = -w^3/t + w^2/t^2`
  and `w' = -w^3/t^2 + 2 w^2/t^3` (error functions).

The module works on the objects of the core and imports no SymPy: the
invariants, the gcd of the equivalence, the simplifications and the
numerical check of a relation (a secant method in mpmath) are native,
and :mod:`sympy_extras.solvers.abel_sympy` is its face on SymPy objects.
Three steps go through SymPy at run time (the functions ending in
``_of_core`` of the faces): the antiderivatives
(:func:`~sympy_extras.solvers.kovacic_sympy.integrate_of_core`), SymPy's
``solve`` of the undetermined coefficients of a particular solution
(:func:`~sympy_extras.solvers.factorization_sympy.solve_of_core`) and of
a common factor of degree two or more of the equivalence
(:func:`~sympy_extras.solvers.charpit_sympy.solve_of_core`), and SymPy's
``dsolve`` of an inverse Riccati equation which the native
:func:`~sympy_extras.solvers.first_order.riccati_ode` does not solve
(:func:`~sympy_extras.solvers.first_order_sympy.dsolve_of_core`).

References
==========

.. [ChebTerrab] E. S. Cheb-Terrab, A. D. Roche, Abel ODEs: equivalence
   and integrable classes, Computer Physics Communications 130 (2000).
.. [Liouville] R. Liouville, Sur une classe d'équations différentielles
   du premier ordre, Comptes Rendus 103 (1886); P. Appell, Sur les
   invariants de quelques équations différentielles, Journal de
   Mathématiques 5 (1889).
.. [Kamke] E. Kamke, Differentialgleichungen: Lösungsmethoden und
   Lösungen, Band 1, §4.9.

Examples
========

>>> from sympy_extras.core import Symbol
>>> from sympy_extras.solvers.abel import abel_invariants
>>> x = Symbol('x')
>>> abel_invariants((-1, -2*x, 0, 0), x).I1
11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
"""
from __future__ import annotations

import random
from typing import Callable, Final, Optional, Sequence, Union

import mpmath

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.evalf import numeric_value
from sympy_extras.core.function import AppliedUndef, DefinedFunction, Function, expand
from sympy_extras.core.mul import Mul
from sympy_extras.core.numbers import Integer, Rational, dps_to_prec
from sympy_extras.core.power import Pow, sqrt
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import exp
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import factor
from sympy_extras.polys.rationaltools import together
from sympy_extras.simplify.simplify import simplify
from sympy_extras._timeout import attempt
from sympy_extras.settings import settings

from .first_order import _all_coeffs, _degree, _normal, _replaced
from .linear_ode import UPoly

__all__ = ['AbelInvariants', 'Coefficients', 'CoefficientsLike', 'abel_coefficients', 'abel_invariants',
           'abel_equivalence', 'AbelTransformation', 'particular_solution', 'air_solution', 'abel_by_invariants',
           'constant_invariant', 'REPRESENTATIVES']

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)

#: the coefficients ``(f3, f2, f1, f0)`` of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``
Coefficients = tuple[Expr, Expr, Expr, Expr]
#: the same, as accepted from the user (integers allowed)
CoefficientsLike = tuple[Union[Expr, int], Union[Expr, int], Union[Expr, int], Union[Expr, int]]


class AbelInvariants:
    """The relative invariants ``s3``, ``s5`` and the absolute invariants
    ``I1``, ``I2`` of an Abel equation (expressions in ``x``)."""

    def __init__(self, s3: Expr, s5: Expr, I1: Expr, I2: Expr) -> None:
        self.s3: Final[Expr] = s3
        self.s5: Final[Expr] = s5
        self.I1: Final[Expr] = I1
        self.I2: Final[Expr] = I2

    def __repr__(self) -> str:
        return "AbelInvariants(s3=%s, s5=%s, I1=%s, I2=%s)" % (self.s3, self.s5, self.I1, self.I2)


class AbelTransformation:
    """``y = P(x) u(xi(x)) + Q(x)`` taking one Abel equation to another."""

    def __init__(self, xi: Expr, P: Expr, Q: Expr, x: Symbol, t: Symbol) -> None:
        self.xi: Final[Expr] = xi
        self.P: Final[Expr] = P
        self.Q: Final[Expr] = Q
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t

    def __repr__(self) -> str:
        return "AbelTransformation(xi=%s, P=%s, Q=%s)" % (self.xi, self.P, self.Q)


def _expr(c: Union[Expr, int]) -> Expr:
    return Integer(c) if isinstance(c, int) else c


def _coefficients_of(coefficients: CoefficientsLike) -> Coefficients:
    f3, f2, f1, f0 = coefficients
    return _expr(f3), _expr(f2), _expr(f1), _expr(f0)


def _sum(terms: Sequence[Expr]) -> Expr:
    return add(*terms) if terms else _ZERO


def _factor(e: Expr) -> Expr:
    """The native ``factor`` (``e`` itself when it cannot take it)."""
    try:
        return factor(e)
    except _FAILURES:
        return e


def abel_coefficients(equation: Basic, f: AppliedUndef) -> tuple[Coefficients, Symbol]:
    """``((f3, f2, f1, f0), x)`` of an equation ``y' = f3 y**3 + ...`` of the
    first kind (``ValueError`` otherwise).

    Examples
    ========

    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.abel import abel_coefficients
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y)
    ((x, 1, 0, -1), x)
    """
    from .first_order import _derivative_polynomial
    F, x, u, p = _derivative_polynomial(equation, f)
    in_p = _all_coeffs(F, p)
    if in_p is None or _degree(in_p) != 1:
        raise ValueError("%s is not linear in the derivative" % (equation,))
    a1, a0 = in_p
    rhs = _normal(-a0/a1)
    if rhs.has(u) and a1.has(u):
        raise ValueError("%s is not an Abel equation of the first kind" % (equation,))
    cubic = _all_coeffs(expand(rhs), u)
    if cubic is None:
        raise ValueError("%s is not polynomial in %s" % (equation, f))
    if _degree(cubic) != 3:
        raise ValueError("%s is not an Abel equation of the first kind" % (equation,))
    f3, f2, f1, f0 = cubic
    return (f3, f2, f1, f0), x


def _normal_form_data(coefficients: CoefficientsLike, x: Symbol) -> tuple[Expr, Expr]:
    """``(g1, g0)`` of ``v' = f3 v**3 + g1 v + g0`` after ``y = v - f2/(3 f3)``."""
    f3, f2, f1, f0 = _coefficients_of(coefficients)
    k = _normal(-f2/(3*f3))
    g1 = _normal(3*f3*k**2 + 2*f2*k + f1)
    g0 = _normal(f3*k**3 + f2*k**2 + f1*k + f0 - k.diff(x))
    return g1, g0


def abel_invariants(coefficients: CoefficientsLike, x: Symbol) -> AbelInvariants:
    """The invariants of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.solvers.abel import abel_invariants
    >>> x = Symbol('x')
    >>> abel_invariants((-1, -2*x, 0, 0), x).I1
    11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
    >>> abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1
    -27/8
    """
    f3, f2, f1, f0 = _coefficients_of(coefficients)
    g1, g0 = _normal_form_data((f3, f2, f1, f0), x)
    s3 = _normal(g0*f3**2)
    T = _normal((g0/f3).diff(x) - 3*g1*g0/f3)
    s5 = _normal(f3**4*T)
    I1 = _factor(_normal(f3**2*T**3/g0**5)) if g0 != _ZERO else _ZERO
    I2 = _factor(_normal((T.diff(x) - 5*g1*T - T*f3.diff(x)/f3)*g0/(f3*T**2))) if T != _ZERO else _ZERO
    return AbelInvariants(s3, s5, I1, I2)


def constant_invariant(coefficients: CoefficientsLike, x: Symbol) -> Optional[bool]:
    """Whether the invariant ``I1`` is constant (the separable class), or
    ``None`` when this cannot be decided.

    >>> from sympy_extras.core import Symbol, Rational
    >>> from sympy_extras.solvers.abel import constant_invariant
    >>> x = Symbol('x')
    >>> constant_invariant((1, 0, 0, x**Rational(-3, 2)), x), constant_invariant((-1, -2*x, 0, 0), x)
    (True, False)
    """
    I1 = abel_invariants(coefficients, x).I1
    if not I1.has(x):
        return True
    if I1.is_rational_function(x):
        return False
    derivative = attempt(lambda: simplify(I1.diff(x)), settings.timeout, native=True)
    if derivative is None:
        return None
    return derivative == _ZERO


def _transformed(coefficients: CoefficientsLike, x: Symbol, xi: Expr, P: Expr, Q: Expr, t: Symbol) -> Coefficients:
    """The coefficients of the equation for ``u(t)`` when ``y = P u + Q``,
    ``x = xi(t)`` (``P``, ``Q`` given in ``t``)."""
    f3, f2, f1, f0 = (_replaced(c, {x: xi}) for c in _coefficients_of(coefficients))
    dxi = xi.diff(t)
    r3 = _normal(dxi*f3*P**2)
    r2 = _normal(dxi*P*(3*f3*Q + f2))
    r1 = _normal(dxi*(3*f3*Q**2 + 2*f2*Q + f1) - P.diff(t)/P)
    r0 = _normal((dxi*(f3*Q**3 + f2*Q**2 + f1*Q + f0) - Q.diff(t))/P)
    return r3, r2, r1, r0


def abel_equivalence(target: CoefficientsLike, x: Symbol, representative: CoefficientsLike, t: Symbol
                     ) -> Optional[AbelTransformation]:
    """A transformation ``y = P(x) u(xi(x)) + Q(x)`` taking the target
    equation (in ``x``) to the representative (in ``t``), from the
    invariants, or ``None`` when the two are not equivalent (a constant
    ``I1`` is not handled: the equation is then in the separable class).

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.solvers.abel import abel_equivalence
    >>> x, t = symbols('x t')
    >>> abel_equivalence((-1, -2*(x - 1), 0, 0), x, (-1, -2*t, 0, 0), t)
    AbelTransformation(xi=x - 1, P=1, Q=0)
    """
    invariants = abel_invariants(target, x)
    reference = abel_invariants(representative, t)
    if not invariants.I1.has(x) or not reference.I1.has(t):
        return None
    target_ = _coefficients_of(target)
    representative_ = _coefficients_of(representative)
    for xi in _xi_candidates(invariants, reference, x, t):
        transformation = _match(target_, x, representative_, t, xi)
        if transformation is not None:
            return transformation
    return None


def _xi_candidates(invariants: AbelInvariants, reference: AbelInvariants, x: Symbol, t: Symbol) -> list[Expr]:
    """The functions ``xi(x)`` with ``J1(xi) = I1(x)`` and ``J2(xi) = I2(x)``:
    the common roots in ``xi`` of the two numerators, from their gcd over
    the rational functions of ``x``."""
    xi_symbol = Dummy('xi')
    first = together(_replaced(reference.I1, {t: xi_symbol}) - invariants.I1).as_numer_denom()[0]
    second = together(_replaced(reference.I2, {t: xi_symbol}) - invariants.I2).as_numer_denom()[0]
    try:
        common = UPoly.of(first, xi_symbol).gcd(UPoly.of(second, xi_symbol))
    except _FAILURES:
        return []
    if common.degree() < 1:
        return []
    candidates: list[Expr] = []
    if common.degree() == 1:
        lead, constant = common.all_coeffs()
        candidates.append(_normal(-constant/lead))
    else:
        from .charpit_sympy import solve_of_core
        found = solve_of_core(common.as_expr(), xi_symbol)
        candidates.extend(c for c in found if not (c.free_symbols - {x}))
    return candidates


def _match(target: Coefficients, x: Symbol, representative: Coefficients, t: Symbol, xi: Expr
           ) -> Optional[AbelTransformation]:
    """``P`` and ``Q`` for a known ``xi``, from the ``u**3`` and ``u**2``
    coefficients, checked on the remaining two."""
    f3, f2, f1, f0 = target
    r3, r2, r1, r0 = (_replaced(c, {t: xi}) for c in representative)
    dxi = xi.diff(x)
    square = _normal(r3*dxi/f3)
    # the square root of a rational function, taken for positive x so that
    # sqrt(x**2) is x rather than Abs(x) (both signs are tried anyway)
    positive = Dummy('x', positive=True)
    root = _replaced(sqrt(_replaced(square, {x: positive})), {positive: x})
    for sign in (_ONE, -_ONE):
        P = _simplified(sign*root, x)
        Q = _normal((r2*dxi/P - f2)/(3*f3))
        # the derivatives with respect to t = xi(x): d/dt = (1/xi') d/dx
        if not _is_zero(((3*f3*Q**2 + 2*f2*Q + f1) - P.diff(x)/P)/dxi - r1, x):
            continue
        if not _is_zero(((f3*Q**3 + f2*Q**2 + f1*Q + f0) - Q.diff(x))/(dxi*P) - r0, x):
            continue
        return AbelTransformation(xi, P, Q, x, t)
    return None


def _simplified(e: Expr, x: Symbol) -> Expr:
    """``e`` simplified: the normal form of a rational function (the
    native ``simplify`` is slow on the rational functions the
    equivalences produce), ``simplify`` otherwise."""
    if e.is_rational_function(x):
        normal = _normal(e)
        if normal.is_rational_function(x):
            return _factor(normal)
    return simplify(e)


def _is_zero(e: Expr, x: Symbol) -> bool:
    """Whether ``e`` simplifies to zero (decided by the normal form of a
    rational function, else by ``simplify``)."""
    normal = _normal(e)
    if normal == _ZERO:
        return True
    if normal.is_rational_function(*normal.free_symbols):
        return False
    return simplify(e) == _ZERO


def particular_solution(coefficients: CoefficientsLike, x: Symbol, degree: int = 2) -> Optional[Expr]:
    """A polynomial or rational particular solution ``Q(x)`` of ``y' = f3
    y**3 + f2 y**2 + f1 y + f0`` (numerator and denominator of degree at
    most ``degree``), or ``None``.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's solve (factorization_sympy.solve_of_core)
    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.solvers.abel import particular_solution
    >>> x = Symbol('x')
    >>> particular_solution((-1, -2*x, 0, 0), x)
    0
    >>> particular_solution((1, 0, -x**2, 1), x)         # y = x solves y' = y**3 - x**2 y + 1
    x
    """
    from .factorization_sympy import solve_of_core
    f3, f2, f1, f0 = _coefficients_of(coefficients)
    if f0 == _ZERO:
        return _ZERO
    unknowns = [Symbol('a%d' % i) for i in range(degree + 1)]
    denominators = [Symbol('b%d' % i) for i in range(1, degree + 1)]
    for shape in range(0, degree + 1):
        numerator = _sum([unknowns[i]*x**i for i in range(degree + 1)])
        denominator = _ONE + _sum([denominators[i - 1]*x**i for i in range(1, shape + 1)])
        Q = numerator/denominator
        residual = expand(_normal((Q.diff(x) - (f3*Q**3 + f2*Q**2 + f1*Q + f0))*denominator**4))
        equations = _all_coeffs(residual, x)
        if equations is None:
            continue
        variables = unknowns + denominators[:shape]
        solutions = solve_of_core(equations, variables)
        if not solutions:
            continue
        for solution in solutions:
            rule: dict[Basic, Basic] = {k: v for k, v in solution.items()}
            candidate = _replaced(Q, rule)
            if candidate.free_symbols - {x}:
                candidate = _replaced(candidate, {s: _ONE for s in candidate.free_symbols - {x}})
            value = _normal(candidate.diff(x) - (f3*candidate**3 + f2*candidate**2 + f1*candidate + f0))
            if value == _ZERO:
                return _normal(candidate)
    return None


def _integrate(e: Expr, v: Symbol) -> Optional[Expr]:
    from .kovacic_sympy import integrate_of_core
    return integrate_of_core(e, v, 'piecewise', False)


def _quadratic(e: Expr, x: Symbol) -> bool:
    coefficients = _all_coeffs(e, x)
    return coefficients is not None and _degree(coefficients) <= 2


# ---------------------------------------------------------------------------
# the numerical check of a relation

#: the digits of the numerical check (those of SymPy's ``nsolve(..., prec=30)``)
_DIGITS: Final[int] = 30


#: a compiled expression in ``Y``: its value at a point, ``None`` when it
#: cannot be evaluated there
Evaluator = Callable[[mpmath.mpc], Optional[mpmath.mpc]]


def _constant(value: Optional[mpmath.mpc]) -> Evaluator:
    def evaluate(y: mpmath.mpc) -> Optional[mpmath.mpc]:
        return value
    return evaluate


def _compiled(e: Expr, Y: Symbol, prec: int) -> Evaluator:
    """``e`` as a function of ``Y`` (every other symbol given a value), its
    parts free of ``Y`` evaluated once at ``prec`` bits."""
    if e == Y:
        def identity(y: mpmath.mpc) -> Optional[mpmath.mpc]:
            return y
        return identity
    if not e.has(Y):
        return _constant(numeric_value(e, prec))
    if isinstance(e, (Add, Mul)):
        parts = [_compiled(a, Y, prec) for a in e.args if isinstance(a, Expr)]
        if len(parts) != len(e.args):
            return _constant(None)
        is_sum = isinstance(e, Add)

        def combined(y: mpmath.mpc) -> Optional[mpmath.mpc]:
            result = mpmath.mpc(0) if is_sum else mpmath.mpc(1)
            for part in parts:
                value = part(y)
                if value is None:
                    return None
                result = result + value if is_sum else result*value
            return result
        return combined
    if isinstance(e, Pow):
        base, exponent = _compiled(e.base, Y, prec), _compiled(e.exp, Y, prec)

        def power(y: mpmath.mpc) -> Optional[mpmath.mpc]:
            b, k = base(y), exponent(y)
            if b is None or k is None or b == 0:
                return None
            return mpmath.mpc(mpmath.power(b, k))
        return power
    if isinstance(e, DefinedFunction):
        node = e
        arguments = [_compiled(a, Y, prec) for a in e.args if isinstance(a, Expr)]
        if len(arguments) != len(e.args):
            return _constant(None)

        def function(y: mpmath.mpc) -> Optional[mpmath.mpc]:
            values: list[mpmath.mpc] = []
            for argument in arguments:
                value = argument(y)
                if value is None:
                    return None
                values.append(value)
            return node.numeric(values)
        return function
    return _constant(None)


def _real_root(e: Expr, Y: Symbol, guess: Rational) -> Optional[mpmath.mpf]:
    """A real root of ``e(Y) = 0`` near ``guess`` by the secant method at
    30 digits (what SymPy's ``nsolve(e, Y, guess, prec=30)`` does), or
    ``None``."""
    with mpmath.workdps(_DIGITS):
        evaluator = _compiled(e, Y, dps_to_prec(_DIGITS) + 20)

    def value(y: mpmath.mpc) -> mpmath.mpc:
        found = evaluator(mpmath.mpc(y))
        if found is None:
            raise ValueError("%s cannot be evaluated" % (e,))
        if mpmath.im(found) == 0:
            return mpmath.mpf(mpmath.re(found))
        return found

    try:
        with mpmath.workdps(_DIGITS):
            start = mpmath.mpf(guess.p)/guess.q
            root = mpmath.findroot(value, start)
    except (ValueError, TypeError, ZeroDivisionError, ArithmeticError):
        return None
    if isinstance(root, mpmath.mpc):
        if mpmath.im(root) != 0:
            return None
        return mpmath.mpf(mpmath.re(root))
    if isinstance(root, mpmath.mpf):
        return root
    return None


def _defines_solution(relation: Expr, coefficients: tuple[Expr, Expr, Expr, Expr], x: Symbol,
                      f: AppliedUndef, C1: Symbol) -> bool:
    """Whether ``relation = 0`` defines solutions of the Abel equation.

    Along a solution curve ``F(x, y, C1) = 0`` the derivative is
    ``y' = -F_x / F_y``, and it must equal ``f3 y**3 + f2 y**2 + f1 y +
    f0`` there. The identity holds *on the curve only* (unless ``C1`` is
    isolated), so a point is put on it first: ``x`` and ``C1`` are chosen
    and ``y`` solved for numerically. The relation is rejected on a
    confirmed disagreement and kept when nothing could be decided.
    """
    rng = random.Random(0)
    Y = Dummy('Y', real=True)
    F = _replaced(relation, {f: Y})
    if not F.has(Y):
        # no curve y(x) at all: y y' = y + x got 0 = C1 - 11**(2/3)*log(x)/9
        return False
    F_x, F_y = F.diff(x), F.diff(Y)
    f3, f2, f1, f0 = coefficients
    implied = -F_x/F_y
    rhs = f3*Y**3 + f2*Y**2 + f1*Y + f0
    prec = dps_to_prec(_DIGITS)
    for _ in range(8):
        x0 = Rational(rng.randint(1, 9), rng.randint(1, 4))
        c0 = Rational(rng.randint(-5, 5), rng.randint(1, 3))
        fixed: dict[Basic, Basic] = {x: x0, C1: c0}
        on_line = _replaced(F, fixed)
        if not on_line.has(Y):
            continue
        for guess in (Rational(1, 3), Rational(-1, 2), Rational(2), Rational(-3)):
            root = _real_root(on_line, Y, guess)
            if root is None:
                continue
            point = mpmath.mpc(root)
            try:
                with mpmath.workdps(_DIGITS):
                    left = _compiled(_replaced(implied, fixed), Y, prec)(point)
                    right = _compiled(_replaced(rhs, fixed), Y, prec)(point)
            except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
                continue
            if left is None or right is None or mpmath.im(left) != 0 or mpmath.im(right) != 0:
                continue
            if not (mpmath.isfinite(left) and mpmath.isfinite(right)):
                continue
            if abs(left - right) <= 1e-12*max(1, abs(right)):
                return True
            return False
    # no point could be placed on a curve: nothing was shown either way
    return True


def air_solution(coefficients: CoefficientsLike, x: Symbol, f: AppliedUndef, Q: Optional[Expr] = None) -> Optional[Basic]:
    """An implicit solution of an equation of the AIR class through its
    inverse Riccati equation, or ``None`` when the equation is not
    recognised (see the module documentation).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and solve (the *_of_core functions)
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.abel import air_solution
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> air_solution((-1/x, 1/x**2, 0, 0), x, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    from .first_order import riccati_ode
    f3, f2, f1, f0 = _coefficients_of(coefficients)
    if Q is None:
        Q = particular_solution(coefficients, x)
    if Q is None:
        return None
    exponent = _integrate(_normal(3*f3*Q**2 + 2*f2*Q + f1), x)
    if exponent is None:
        return None
    P = simplify(exp(exponent))
    g1 = _normal(-1/(f3*P**2))
    if not _quadratic(g1, x):
        return None
    r2 = _normal(P*(3*f3*Q + f2))
    primitive = _integrate(r2, x)
    if primitive is None:
        return None
    g0 = _normal(-g1*primitive)
    if not _quadratic(g0, x):
        return None
    # u = (y - Q)/P solves u' = -u**3/g1 - (g0/g1)' u**2, and z = 1/u - g0/g1
    # solves dx/dz = g1(x) z + g0(x): a Riccati equation for x(z)
    z = Dummy('z')
    X = Function('X')(z)
    riccati = Eq(X.diff(z), _replaced(g1, {x: X})*z + _replaced(g0, {x: X}))
    solved = attempt(lambda: riccati_ode(riccati, X), settings.timeout, native=True)
    if solved is None:
        from .first_order_sympy import dsolve_of_core
        solved = attempt(lambda: dsolve_of_core(riccati, X), settings.timeout, native=True)
    if not isinstance(solved, Eq):
        return None
    z_value = P/(f - Q) - g0/g1
    relation = _replaced(_replaced(solved.lhs - solved.rhs, {X: x}), {z: z_value})
    C1 = Symbol('C1')
    constants = sorted((s for s in relation.free_symbols if s.name.startswith('C')), key=lambda s: s.name)
    if len(constants) == 1 and constants[0] != C1:
        relation = _replaced(relation, {constants[0]: C1})
    if not _defines_solution(relation, (f3, f2, f1, f0), x, f, C1):
        # the transformation back can collapse to a relation with no x in
        # it, which implies y' = 0 and is no solution (sympy-extras#38)
        return None
    if C1 not in relation.free_symbols:
        return Eq(relation, _ZERO)
    # the Riccati solution is linear fractional in C1: isolate it
    numerator, _ = together(relation).as_numer_denom()
    # the coefficients as they come: their normal forms rewrite the special
    # functions of the solution (besselj(nu, -z)) into larger expressions
    linear = _all_coeffs(numerator, C1, normal_form=False)
    if linear is None or _degree(linear) != 1:
        return Eq(relation, _ZERO)
    lead, constant = linear
    return Eq(-constant/lead, C1)


#: representatives of integrable classes with a non-constant invariant:
#: the coefficients in ``t`` and the pair ``(g1, g0)`` of their inverse
#: Riccati equation ``dx/dz = g1(x) z + g0(x)``
_T: Final[Symbol] = Symbol('t')
REPRESENTATIVES: Final[list[tuple[Coefficients, tuple[Expr, Expr]]]] = [
    ((-_ONE, -2*_T, _ZERO, _ZERO), (_ONE, _T**2)),          # dx/dz = z + x**2: Airy functions
    ((-1/_T, 1/_T**2, _ZERO, _ZERO), (_T, _ONE)),           # dx/dz = x z + 1: error functions
    ((-1/_T**2, 2/_T**3, _ZERO, _ZERO), (_T**2, _ONE)),     # dx/dz = x**2 z + 1
]


_SOLVED: Final[dict[Coefficients, Basic]] = {}


def _representative_solution(representative: Coefficients, U: AppliedUndef) -> Optional[Basic]:
    if representative not in _SOLVED:
        solution = attempt(lambda: air_solution(representative, _T, U, _ZERO), settings.timeout, native=True)
        if solution is None:
            return None
        _SOLVED[representative] = solution
    return _SOLVED[representative]


def abel_by_invariants(equation: Basic, f: AppliedUndef,
                       representatives: Optional[Sequence[tuple[Coefficients, tuple[Expr, Expr]]]] = None
                       ) -> Optional[Basic]:
    """An implicit solution of an Abel equation of the first kind with a
    non-constant invariant: through the AIR class directly, or through
    the equivalence with a representative of the database (``REPRESENTATIVES``
    by default) whose solution is transformed back; ``None`` otherwise.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and solve (the *_of_core functions)
    >>> from sympy_extras.core import Function, Symbol
    >>> from sympy_extras.solvers.abel import abel_by_invariants
    >>> x = Symbol('x')
    >>> y = Function('y')(x)
    >>> abel_by_invariants(y.diff(x) + y**3 + 2*(x - 1)*y**2, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    coefficients, x = abel_coefficients(equation, f)
    direct = attempt(lambda: air_solution(coefficients, x, f), settings.timeout, native=True)
    if direct is not None:
        return direct
    t = _T
    for representative, (g1, g0) in (representatives if representatives is not None else REPRESENTATIVES):
        transformation = attempt(lambda: abel_equivalence(coefficients, x, representative, t), settings.timeout,
                                 native=True)
        if transformation is None:
            continue
        # the representative's solution in (t, u), pulled back through
        # u = (y - Q)/P, t = xi(x)
        U = Function('U')(t)
        solution = _representative_solution(representative, U)
        if not isinstance(solution, Eq):
            continue
        u_value = (f - transformation.Q)/transformation.P
        pulled = solution._xreplace_evaluated({U: u_value})._xreplace_evaluated({t: transformation.xi})
        if not isinstance(pulled, Eq):
            continue
        relation = pulled.lhs - pulled.rhs
        # the representative's solution is right; the pull-back can still
        # collapse to a relation with no x in it, which implies y' = 0 and
        # solves nothing (sympy-extras#38): it is checked against the
        # equation before it is returned
        if not _defines_solution(relation, coefficients, x, f, Symbol('C1')):
            return None
        return Eq(pulled.lhs, pulled.rhs)
    return None
