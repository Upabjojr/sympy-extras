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
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly, cancel, factor
from sympy.polys.rationaltools import together
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve as _solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols
from sympy_extras.settings import settings

__all__ = ['AbelInvariants', 'Coefficients', 'CoefficientsLike', 'abel_coefficients', 'abel_invariants', 'abel_equivalence', 'AbelTransformation',
           'particular_solution', 'air_solution', 'abel_by_invariants', 'constant_invariant', 'REPRESENTATIVES']

#: the coefficients ``(f3, f2, f1, f0)`` of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``
Coefficients = tuple[Expr, Expr, Expr, Expr]
#: the same, as accepted from the user (integers allowed)
CoefficientsLike = tuple[Union[Expr, int], Union[Expr, int], Union[Expr, int], Union[Expr, int]]


class AbelInvariants:
    """The relative invariants ``s3``, ``s5`` and the absolute invariants
    ``I1``, ``I2`` of an Abel equation (expressions in ``x``)."""

    def __init__(self, s3: Expr, s5: Expr, I1: Expr, I2: Expr) -> None:
        self.s3 = s3
        self.s5 = s5
        self.I1 = I1
        self.I2 = I2

    def __repr__(self) -> str:
        return "AbelInvariants(s3=%s, s5=%s, I1=%s, I2=%s)" % (self.s3, self.s5, self.I1, self.I2)


class AbelTransformation:
    """``y = P(x) u(xi(x)) + Q(x)`` taking one Abel equation to another."""

    def __init__(self, xi: Expr, P: Expr, Q: Expr, x: Symbol, t: Symbol) -> None:
        self.xi = xi
        self.P = P
        self.Q = Q
        self.x = x
        self.t = t

    def __repr__(self) -> str:
        return "AbelTransformation(xi=%s, P=%s, Q=%s)" % (self.xi, self.P, self.Q)


def abel_coefficients(equation: Basic, f: AppliedUndef) -> tuple[Coefficients, Symbol]:
    """``((f3, f2, f1, f0), x)`` of an equation ``y' = f3 y**3 + ...`` of the
    first kind (``ValueError`` otherwise).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel import abel_coefficients
    >>> y = Function('y')(x)
    >>> abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y)
    ((x, 1, 0, -1), x)
    """
    from .first_order import _derivative_polynomial
    F, x, u, p = _derivative_polynomial(equation, f)
    try:
        poly = Poly(F, p)
    except PolynomialError:
        raise ValueError("%s is not linear in the derivative" % (equation,))
    if poly.degree() != 1:
        raise ValueError("%s is not linear in the derivative" % (equation,))
    a1, a0 = [as_expr(c) for c in poly.all_coeffs()]
    rhs = as_expr(cancel(-a0/a1))
    if rhs.has(u) and a1.has(u):
        raise ValueError("%s is not an Abel equation of the first kind" % (equation,))
    try:
        cubic = Poly(expand(rhs), u)
    except PolynomialError:
        raise ValueError("%s is not polynomial in %s" % (equation, f))
    if cubic.degree() != 3:
        raise ValueError("%s is not an Abel equation of the first kind" % (equation,))
    f3, f2, f1, f0 = [as_expr(c) for c in cubic.all_coeffs()]
    return (f3, f2, f1, f0), x


def _normal_form_data(coefficients: CoefficientsLike, x: Symbol) -> tuple[Expr, Expr]:
    """``(g1, g0)`` of ``v' = f3 v**3 + g1 v + g0`` after ``y = v - f2/(3 f3)``."""
    f3, f2, f1, f0 = (as_expr(c) for c in coefficients)
    k = as_expr(cancel(-f2/(3*f3)))
    g1 = as_expr(cancel(3*f3*k**2 + 2*f2*k + f1))
    g0 = as_expr(cancel(f3*k**3 + f2*k**2 + f1*k + f0 - k.diff(x)))
    return g1, g0


def abel_invariants(coefficients: CoefficientsLike, x: Symbol) -> AbelInvariants:
    """The invariants of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``.

    Examples
    ========

    >>> from sympy import Symbol, Rational
    >>> from sympy_extras.solvers.abel import abel_invariants
    >>> x = Symbol('x')
    >>> abel_invariants((-1, -2*x, 0, 0), x).I1
    11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
    >>> abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1
    -27/8
    """
    f3, f2, f1, f0 = (as_expr(c) for c in coefficients)
    g1, g0 = _normal_form_data((f3, f2, f1, f0), x)
    s3 = as_expr(cancel(g0*f3**2))
    T = as_expr(cancel((g0/f3).diff(x) - 3*g1*g0/f3))
    s5 = as_expr(cancel(f3**4*T))
    I1 = as_expr(factor(cancel(f3**2*T**3/g0**5))) if g0 != 0 else S.Zero
    I2 = as_expr(factor(cancel((T.diff(x) - 5*g1*T - T*f3.diff(x)/f3)*g0/(f3*T**2)))) if T != 0 else S.Zero
    return AbelInvariants(s3, s5, I1, I2)


def constant_invariant(coefficients: CoefficientsLike, x: Symbol) -> Optional[bool]:
    """Whether the invariant ``I1`` is constant (the separable class), or
    ``None`` when this cannot be decided."""
    I1 = abel_invariants(coefficients, x).I1
    if not I1.has(x):
        return True
    if I1.is_rational_function(x):
        return False
    derivative = attempt(lambda: simplify(I1.diff(x)), settings.timeout)
    if derivative is None:
        return None
    return derivative == 0


def _transformed(coefficients: CoefficientsLike, x: Symbol, xi: Expr, P: Expr, Q: Expr, t: Symbol) -> Coefficients:
    """The coefficients of the equation for ``u(t)`` when ``y = P u + Q``,
    ``x = xi(t)`` (``P``, ``Q`` given in ``t``)."""
    f3, f2, f1, f0 = (as_expr(as_expr(c).subs(x, xi)) for c in coefficients)
    dxi = xi.diff(t)
    r3 = cancel(dxi*f3*P**2)
    r2 = cancel(dxi*P*(3*f3*Q + f2))
    r1 = cancel(dxi*(3*f3*Q**2 + 2*f2*Q + f1) - P.diff(t)/P)
    r0 = cancel((dxi*(f3*Q**3 + f2*Q**2 + f1*Q + f0) - Q.diff(t))/P)
    return as_expr(r3), as_expr(r2), as_expr(r1), as_expr(r0)


def abel_equivalence(target: CoefficientsLike, x: Symbol, representative: CoefficientsLike, t: Symbol
                     ) -> Optional[AbelTransformation]:
    """A transformation ``y = P(x) u(xi(x)) + Q(x)`` taking the target
    equation (in ``x``) to the representative (in ``t``), from the
    invariants, or ``None`` when the two are not equivalent (a constant
    ``I1`` is not handled: the equation is then in the separable class).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.abel import abel_equivalence
    >>> x, t = symbols('x t')
    >>> abel_equivalence((-1, -2*(x - 1), 0, 0), x, (-1, -2*t, 0, 0), t)
    AbelTransformation(xi=x - 1, P=1, Q=0)
    """
    invariants = abel_invariants(target, x)
    reference = abel_invariants(representative, t)
    if not invariants.I1.has(x) or not reference.I1.has(t):
        return None
    target_ = tuple(as_expr(c) for c in target)
    representative_ = tuple(as_expr(c) for c in representative)
    assert len(target_) == 4 and len(representative_) == 4
    for xi in _xi_candidates(invariants, reference, x, t):
        transformation = _match((target_[0], target_[1], target_[2], target_[3]), x,
                                (representative_[0], representative_[1], representative_[2], representative_[3]), t, xi)
        if transformation is not None:
            return transformation
    return None


def _xi_candidates(invariants: AbelInvariants, reference: AbelInvariants, x: Symbol, t: Symbol) -> list[Expr]:
    """The functions ``xi(x)`` with ``J1(xi) = I1(x)`` and ``J2(xi) = I2(x)``:
    the common roots in ``xi`` of the two numerators, from their gcd over
    the rational functions of ``x``."""
    xi_symbol = Dummy('xi')
    first = as_expr(together(reference.I1.subs(t, xi_symbol) - invariants.I1).as_numer_denom()[0])
    second = as_expr(together(reference.I2.subs(t, xi_symbol) - invariants.I2).as_numer_denom()[0])
    try:
        p1 = Poly(first, xi_symbol)
        p2 = Poly(second, xi_symbol)
        common = p1.gcd(p2)
    except PolynomialError:
        return []
    if common.degree() < 1:
        return []
    candidates: list[Expr] = []
    if common.degree() == 1:
        lead, constant = (as_expr(c) for c in common.all_coeffs())
        candidates.append(as_expr(cancel(-constant/lead)))
    else:
        found = attempt(lambda: _solve(common.as_expr(), xi_symbol), settings.timeout) or []
        candidates.extend(as_expr(c) for c in found if isinstance(c, Expr) and not (free_symbols(c) - {x}))
    return candidates


def _match(target: Coefficients, x: Symbol, representative: Coefficients, t: Symbol, xi: Expr
           ) -> Optional[AbelTransformation]:
    """``P`` and ``Q`` for a known ``xi``, from the ``u**3`` and ``u**2``
    coefficients, checked on the remaining two."""
    f3, f2, f1, f0 = (as_expr(c) for c in target)
    r3, r2, r1, r0 = (as_expr(as_expr(c).subs(t, xi)) for c in representative)
    dxi = as_expr(xi.diff(x))
    square = as_expr(cancel(r3*dxi/f3))
    # the square root of a rational function, taken for positive x so that
    # sqrt(x**2) is x rather than Abs(x) (both signs are tried anyway)
    positive = Dummy('x', positive=True)
    root = as_expr(sqrt(square.xreplace({x: positive})).xreplace({positive: x}))
    for sign in (1, -1):
        P = as_expr(simplify(sign*root))
        Q = as_expr(cancel((r2*dxi/P - f2)/(3*f3)))
        # the derivatives with respect to t = xi(x): d/dt = (1/xi') d/dx
        check1 = as_expr(simplify(((3*f3*Q**2 + 2*f2*Q + f1) - P.diff(x)/P)/dxi - r1))
        if check1 != 0:
            continue
        check0 = as_expr(simplify(((f3*Q**3 + f2*Q**2 + f1*Q + f0) - Q.diff(x))/(dxi*P) - r0))
        if check0 != 0:
            continue
        return AbelTransformation(xi, P, Q, x, t)
    return None


def particular_solution(coefficients: CoefficientsLike, x: Symbol, degree: int = 2) -> Optional[Expr]:
    """A polynomial or rational particular solution ``Q(x)`` of ``y' = f3
    y**3 + f2 y**2 + f1 y + f0`` (numerator and denominator of degree at
    most ``degree``), or ``None``.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy_extras.solvers.abel import particular_solution
    >>> x = Symbol('x')
    >>> particular_solution((-1, -2*x, 0, 0), x)
    0
    >>> particular_solution((1, 0, -x**2, 1), x)         # y = x solves y' = y**3 - x**2 y + 1
    x
    """
    f3, f2, f1, f0 = (as_expr(c) for c in coefficients)
    if f0 == 0:
        return S.Zero
    unknowns = [Symbol('a%d' % i) for i in range(degree + 1)]
    denominators = [Symbol('b%d' % i) for i in range(1, degree + 1)]
    for shape in range(0, degree + 1):
        numerator = as_expr(sum(unknowns[i]*x**i for i in range(degree + 1)))
        denominator = as_expr(1 + sum(denominators[i - 1]*x**i for i in range(1, shape + 1)))
        Q = as_expr(numerator/denominator)
        residual = as_expr((Q.diff(x) - (f3*Q**3 + f2*Q**2 + f1*Q + f0))*denominator**4)
        residual = as_expr(expand(cancel(residual)))
        try:
            poly = Poly(residual, x)
        except PolynomialError:
            continue
        equations = [as_expr(c) for c in poly.all_coeffs()]
        variables = unknowns + denominators[:shape]
        solutions = attempt(lambda: _solve(equations, variables, dict=True), settings.timeout)
        if not solutions:
            continue
        for solution in solutions:
            candidate = as_expr(Q.xreplace(solution))
            if free_symbols(candidate) - {x}:
                candidate = as_expr(candidate.xreplace({s: S.One for s in free_symbols(candidate) - {x}}))
            value = as_expr(cancel(candidate.diff(x) - (f3*candidate**3 + f2*candidate**2 + f1*candidate + f0)))
            if value == 0:
                return as_expr(cancel(candidate))
    return None


def _integrate(e: Expr, v: Symbol) -> Optional[Expr]:
    value = attempt(lambda: as_expr(integrate(e, v)), settings.timeout)
    if value is None or value.has(Integral):
        return None
    return value


def _quadratic(e: Expr, x: Symbol) -> bool:
    try:
        return Poly(e, x).degree() <= 2
    except PolynomialError:
        return False


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
    from sympy import Rational, N, nsolve
    import random
    rng = random.Random(0)
    Y = Dummy('Y', real=True)
    F = as_expr(relation.xreplace({f: Y}))
    F_x, F_y = as_expr(F.diff(x)), as_expr(F.diff(Y))
    f3, f2, f1, f0 = coefficients
    implied = as_expr(-F_x/F_y)
    rhs = as_expr(f3*Y**3 + f2*Y**2 + f1*Y + f0)
    for _ in range(8):
        fixed: dict[Basic, Expr] = {x: as_expr(Rational(rng.randint(1, 9), rng.randint(1, 4))),
                                    C1: as_expr(Rational(rng.randint(-5, 5), rng.randint(1, 3)))}
        on_line = as_expr(F.xreplace(fixed))
        if not on_line.has(Y):
            continue
        for guess in (Rational(1, 3), Rational(-1, 2), Rational(2), Rational(-3)):
            try:
                root = nsolve(on_line, Y, guess, prec=30)
            except (ValueError, TypeError, ZeroDivisionError, ArithmeticError):
                continue
            if not root.is_real:
                continue
            point = dict(fixed)
            point[Y] = as_expr(root)
            try:
                left, right = N(implied.xreplace(point), 30), N(rhs.xreplace(point), 30)
            except (TypeError, ValueError, ZeroDivisionError, ArithmeticError):
                continue
            if not (left.is_number and right.is_number and left.is_real and right.is_real):
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

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel import air_solution
    >>> y = Function('y')(x)
    >>> air_solution((-1/x, 1/x**2, 0, 0), x, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    from .first_order import riccati_ode
    f3, f2, f1, f0 = (as_expr(c) for c in coefficients)
    if Q is None:
        Q = particular_solution(coefficients, x)
    if Q is None:
        return None
    exponent = _integrate(as_expr(cancel(3*f3*Q**2 + 2*f2*Q + f1)), x)
    if exponent is None:
        return None
    P = as_expr(simplify(exp(exponent)))
    g1 = as_expr(cancel(-1/(f3*P**2)))
    if not _quadratic(g1, x):
        return None
    r2 = as_expr(cancel(P*(3*f3*Q + f2)))
    primitive = _integrate(r2, x)
    if primitive is None:
        return None
    g0 = as_expr(cancel(-g1*primitive))
    if not _quadratic(g0, x):
        return None
    # u = (y - Q)/P solves u' = -u**3/g1 - (g0/g1)' u**2, and z = 1/u - g0/g1
    # solves dx/dz = g1(x) z + g0(x): a Riccati equation for x(z)
    z = Dummy('z')
    X = Function('X')(z)
    riccati = Eq(X.diff(z), g1.subs(x, X)*z + g0.subs(x, X))
    solved = attempt(lambda: riccati_ode(riccati, X), settings.timeout)
    if solved is None:
        from sympy.solvers.ode import dsolve
        solved = attempt(lambda: dsolve(riccati, X), settings.timeout)
    if not isinstance(solved, Eq):
        return None
    z_value = as_expr(P/(f - Q) - g0/g1)
    relation = as_expr((solved.lhs - solved.rhs).xreplace({X: x}).xreplace({z: z_value}))
    C1 = Symbol('C1')
    constants = sorted((s for s in free_symbols(relation) if s.name.startswith('C')), key=lambda s: s.name)
    if len(constants) == 1 and constants[0] != C1:
        relation = as_expr(relation.xreplace({constants[0]: C1}))
    if not _defines_solution(relation, (f3, f2, f1, f0), x, f, C1):
        # the transformation back can collapse to a relation with no x in
        # it, which implies y' = 0 and is no solution (sympy-extras#38)
        return None
    if C1 not in free_symbols(relation):
        return Eq(relation, 0)
    # the Riccati solution is linear fractional in C1: isolate it
    numerator, _ = together(relation).as_numer_denom()
    try:
        linear = Poly(numerator, C1)
    except PolynomialError:
        return Eq(relation, 0)
    if linear.degree() != 1:
        return Eq(relation, 0)
    lead, constant = (as_expr(c) for c in linear.all_coeffs())
    return Eq(as_expr(-constant/lead), C1)


#: representatives of integrable classes with a non-constant invariant:
#: the coefficients in ``t`` and the pair ``(g1, g0)`` of their inverse
#: Riccati equation ``dx/dz = g1(x) z + g0(x)``
_T = Symbol('t')
REPRESENTATIVES: list[tuple[Coefficients, tuple[Expr, Expr]]] = [
    ((S.NegativeOne, -2*_T, S.Zero, S.Zero), (S.One, _T**2)),          # dx/dz = z + x**2: Airy functions
    ((-1/_T, 1/_T**2, S.Zero, S.Zero), (_T, S.One)),                   # dx/dz = x z + 1: error functions
    ((-1/_T**2, 2/_T**3, S.Zero, S.Zero), (_T**2, S.One)),             # dx/dz = x**2 z + 1
]


_SOLVED: dict[Coefficients, Basic] = {}


def _representative_solution(representative: Coefficients, U: AppliedUndef) -> Optional[Basic]:
    if representative not in _SOLVED:
        solution = attempt(lambda: air_solution(representative, _T, U, S.Zero), settings.timeout)
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

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel import abel_by_invariants
    >>> y = Function('y')(x)
    >>> abel_by_invariants(y.diff(x) + y**3 + 2*(x - 1)*y**2, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    coefficients, x = abel_coefficients(equation, f)
    direct = attempt(lambda: air_solution(coefficients, x, f), settings.timeout)
    if direct is not None:
        return direct
    t = _T
    for representative, (g1, g0) in (representatives if representatives is not None else REPRESENTATIVES):
        transformation = attempt(lambda: abel_equivalence(coefficients, x, representative, t), settings.timeout)
        if transformation is None:
            continue
        # the representative's solution in (t, u), pulled back through
        # u = (y - Q)/P, t = xi(x)
        U = Function('U')(t)
        solution = _representative_solution(representative, U)
        if not isinstance(solution, Eq):
            continue
        u_value = as_expr((f - transformation.Q)/transformation.P)
        pulled = solution.xreplace({U: u_value}).xreplace({t: transformation.xi})
        relation = as_expr(pulled.lhs - pulled.rhs)
        # the representative's solution is right; the pull-back can still
        # collapse to a relation with no x in it, which implies y' = 0 and
        # solves nothing (sympy-extras#38): it is checked against the
        # equation before it is returned
        if not _defines_solution(relation, coefficients, x, f, Symbol('C1')):
            return None
        return Eq(as_expr(pulled.lhs), as_expr(pulled.rhs))
    return None


