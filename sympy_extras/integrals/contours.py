"""Three more contours for the residue theorem: the rectangle for
rational functions of an exponential over the real line, the sector for
`x^a/(1 + x^n)` over `(0, \\infty)`, and the indented real line for
Fourier integrals with a removable pole on the axis.

The four shapes of :mod:`sympy_extras.integrals.residues` (the closed
half plane, Jordan's lemma, the keyhole, the unit circle) are not
repeated here. The shapes added:

1. **The rectangle** ([Mitrinovic]_, chapter 5; [Whittaker]_, 6.24).
   For `F(x) = x^n e^{kx} R(e^{cx})` with `R` rational and `c > 0`, the
   function is periodic up to a factor along the imaginary direction,
   `F(z + ih) = e^{ikh} R(e^{cz}) e^{kz} (z + ih)^n` with `h = 2\\pi/c`, so
   the integral over the boundary of the strip `0 < \\operatorname{Im} z < h`
   (the two vertical sides vanish when `F` decays at `\\pm\\infty`) gives,
   with `A_m = \\int_{-\\infty}^\\infty x^m e^{kx} R(e^{cx})\\, dx`,

   .. math::

       A_m - e^{ikh} \\sum_{j=0}^{m} \\binom{m}{j} (ih)^{m-j} A_j
       = 2\\pi i \\sum_{0 < \\operatorname{Im} z_l < h} \\operatorname{Res}_{z_l} z^m F(z),

   one equation for each `m`: a triangular system for `A_0, \\ldots, A_n`.
   When `e^{ikh} = 1` (`k/c` an integer, `k = 0` in particular) the
   equation for `m` involves `A_0, \\ldots, A_{m-1}` only and the system
   for `m = 1, \\ldots, n + 1` is used instead. The poles are the points
   `z_l = (\\log u_l + i\\theta_l)/c` with `u_l` the poles of `R` and
   `0 < \\theta_l < 2\\pi`; a pole of `R` on the positive real axis is a
   pole of `F` on the real line and the integral is refused, unless the
   factor `z^m` cancels a simple pole at `z = 0` (`u_l = 1`, as for
   `x/\\sinh x`). Decay at `\\pm\\infty` needs
   `c (\\operatorname{ord}_0 Q - \\operatorname{ord}_0 P) < k < c (\\deg Q - \\deg P)`.
   The classical results `\\int e^{ax}/(1 + e^x)\\, dx = \\pi/\\sin \\pi a`
   (`0 < a < 1`), `\\int x/\\sinh x\\, dx = \\pi^2/2`,
   `\\int dx/\\cosh x = \\pi` come out of it ([GR]_ 3.311.3, 3.521.1,
   3.511.4).

2. **The sector** of angle `2\\pi/n` ([Ahlfors]_, 4.5.3): for
   `x^a/(1 + x^n)` the only pole inside is `e^{i\\pi/n}` and the two
   straight sides are proportional, which gives
   `\\int_0^\\infty x^a/(1 + x^n)\\, dx = \\pi / (n \\sin(\\pi (a + 1)/n))` for
   `-1 < a < n - 1` ([GR]_ 3.241.2); the formula holds for every real
   `n > 0` (it is a Beta integral), so symbolic `n` and `a` are allowed
   and the scaling `b + c x^n` is absorbed.

3. **The indented real line** ([Ahlfors]_, 4.5.3, example 4): for
   `R(x) \\sin kx` (or `\\cos kx`) with `R` real, `\\deg Q \\ge \\deg P + 1`,
   `k > 0`, and simple real poles of `R` at which the integrand is
   regular (the sine vanishes there), the semicircle of Jordan's lemma
   with small half circles around the real poles gives

   .. math::

       \\int_{-\\infty}^\\infty R(x) e^{ikx}\\, dx = 2\\pi i \\sum_{\\operatorname{Im} z > 0}
       \\operatorname{Res} R(z) e^{ikz} + \\pi i \\sum_{z \\in \\mathbb{R}} \\operatorname{Res} R(z) e^{ikz}

   as a principal value, which is the integral itself since the integrand
   is regular; the sine integral is its imaginary part, the cosine
   integral its real part ([GR]_ 3.737.1 for `\\sin ax/(x (x^2 + b^2))`).
   Over `(0, \\infty)` the integral of an even integrand is half of it.

Examples
========

>>> from sympy import symbols, exp, sinh, cosh, sin, oo
>>> from sympy_extras.integrals.contours import contour_integral
>>> x = symbols('x')
>>> a, b, n = symbols('a b n', positive=True)
>>> contour_integral(x/sinh(x), x, -oo, oo)
ConditionalValue(pi**2/2)
>>> contour_integral(exp(a*x)/(1 + exp(x)), x, -oo, oo, a < 1)
ConditionalValue(pi/sin(pi*a), a < 1)
>>> contour_integral(x**a/(1 + x**n), x, 0, oo, a < n - 1)
ConditionalValue(pi/(n*sin(pi*(a + 1)/n)), a < n - 1)
>>> contour_integral(sin(a*x)/(x*(x**2 + b**2)), x, 0, oo)
ConditionalValue(pi*(exp(a*b) - 1)*exp(-a*b)/(2*b**2))

References
==========

.. [Ahlfors] L. V. Ahlfors, *Complex Analysis*, 3rd ed., McGraw-Hill,
   1979, chapter 4, section 5.3.
.. [Whittaker] E. T. Whittaker, G. N. Watson, *A Course of Modern
   Analysis*, 4th ed., Cambridge University Press, 1927, sections
   6.2-6.24.
.. [Mitrinovic] D. S. Mitrinović, J. D. Kečkić, *The Cauchy Method of
   Residues: Theory and Applications*, Reidel, 1984, chapter 5.
.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series, and
   Products*, 7th ed., Academic Press, 2007, 3.241, 3.311, 3.511, 3.521,
   3.737.
"""
from __future__ import annotations

from math import gcd, lcm
from typing import Optional

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, Rational, nan, oo, pi, zoo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import Abs, arg, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.logic.boolalg import And, Boolean
from sympy.matrices import Matrix
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.series.residues import residue
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.settings import settings
from .conditions import ConditionalValue
from .mellin import monomial
from .residues import _Fraction, _Locator, _Pole, _part, _residue, _tidy, poles, rational_function

__all__ = ['rectangular_integral', 'sector_integral', 'indented_integral', 'contour_integral']


# ---------------------------------------------------------------------------
# 1. The rectangle

class _Exponential:
    """``x**n * exp(k x) * R(u)`` with ``u = exp(c x)``."""

    def __init__(self, n: int, k: Expr, c: Rational, fraction: _Fraction) -> None:
        self.n = n
        self.k = k
        self.c = c
        self.fraction = fraction


def _exponential_form(f: Expr, x: Symbol) -> Optional[_Exponential]:
    """``f`` written as ``x**n exp(k x) R(exp(c x))``, hyperbolic functions
    rewritten first; ``c`` is the generator of the rational exponents,
    ``k`` the coefficient of the exponentials it does not generate (one
    such at most, possibly symbolic)."""
    g = as_expr(f.rewrite(exp)) if f.has(HyperbolicFunction) else f
    n = 0
    rest: list[Expr] = []
    for factor in Mul.make_args(g):
        e = as_expr(factor)
        if e == x:
            n += 1
        elif isinstance(e, Pow) and e.base == x and isinstance(e.exp, Integer) and e.exp > 0:
            n += int(e.exp)
        else:
            rest.append(e)
    g = as_expr(Mul(*rest))
    coefficients: list[Rational] = []
    symbolic: list[Expr] = []
    for node in g.atoms(exp):
        argument = as_expr(node.args[0])
        if not argument.has(x):
            continue
        found = monomial(argument, x)
        if found is None or found[1] != 1:
            return None
        if isinstance(found[0], Rational):
            coefficients.append(found[0])
        else:
            symbolic.append(found[0])
    if not coefficients and not symbolic:
        return None
    if g.has(x) and any(not isinstance(node, exp) and node.has(x) and not isinstance(node, (Add, Mul, Pow))
                        for node in g.atoms() if node.has(x) and node != x):
        return None
    numerator = 0
    denominator = 1
    for c_ in coefficients:
        numerator = gcd(numerator, int(c_.p))
        denominator = lcm(denominator, int(c_.q))
    c = Rational(numerator, denominator) if coefficients else S.One
    u = Dummy('u')
    replacement: dict[Expr, Expr] = {}
    k: Expr = S.Zero
    for node in g.atoms(exp):
        argument = as_expr(node.args[0])
        if not argument.has(x):
            continue
        found = monomial(argument, x)
        if found is None:
            return None
        coefficient = found[0]
        if isinstance(coefficient, Rational):
            replacement[as_expr(node)] = u**int(coefficient / c)
        else:
            if len(symbolic) > 1 or k != 0:
                return None
            # exp(a x) stays as the factor exp(k x): the node itself is
            # replaced by a marker removed below
            k = coefficient
            replacement[as_expr(node)] = S.One
    h = as_expr(g.xreplace(replacement))
    if h.has(x):
        return None
    fraction = rational_function(h, u)
    if fraction is None:
        return None
    return _Exponential(n, k, c, fraction)


def _orders(fraction: _Fraction) -> tuple[int, int, int, int]:
    """``(deg P, deg Q, ord_0 P, ord_0 Q)``."""
    p, q = fraction.numerator, fraction.denominator
    return (p.degree(), q.degree(), _order_at_zero(p), _order_at_zero(q))


def _order_at_zero(p: Poly) -> int:
    coefficients = p.all_coeffs()[::-1]
    for i, c in enumerate(coefficients):
        if as_expr(c) != 0:
            return i
    return 0


def _strip_pole(pole: _Pole, c: Rational, locator: _Locator) -> Optional[tuple[Expr, bool]]:
    """The point of the strip ``0 < Im z < 2 pi/c`` where ``exp(c z)`` is
    the pole of ``R``, and whether it lies on the real axis; ``None``
    when its argument cannot be placed."""
    u = pole.point
    if u == 0:
        return None
    if isinstance(u, Integer) or isinstance(u, Rational):
        if u > 0:
            return (as_expr(log(u) / c), True)
        return (as_expr((log(-u) + I * pi) / c), False)
    s = locator.imaginary_sign(u)
    if s is None:
        return None
    if s == 0:
        positive = locator.decide(as_boolean(re(u) > 0))
        if positive is None:
            return None
        if positive:
            return (as_expr(log(u) / c), True)
        return (as_expr((log(-u) + I * pi) / c), False)
    theta = as_expr(arg(u)) if s > 0 else as_expr(arg(u) + 2 * pi)
    return (as_expr((log(Abs(u)) + I * theta) / c), False)


def _strip_residue(form: _Exponential, pole: _Pole, point: Expr, m: int, z: Symbol) -> Optional[Expr]:
    """The residue of ``z**m exp(k z) R(exp(c z))`` at ``point``."""
    u = pole.point
    if pole.multiplicity == 1:
        p = as_expr(form.fraction.numerator.as_expr().subs(form.fraction.x, u))
        dq = as_expr(form.fraction.denominator.diff().as_expr().subs(form.fraction.x, u))
        return as_expr(point**m * exp(form.k * point) * p / (form.c * u * dq))
    expression = z**m * exp(form.k * z) * form.fraction.as_expr().subs(form.fraction.x, exp(form.c * z))
    value = attempt(lambda: as_expr(residue(expression, z, point)), settings.timeout)
    if value is None or value.has(nan, zoo):
        return None
    return value


def rectangular_integral(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**n * exp(k*x) * R(exp(c*x)), (x, -oo, oo))`` by the
    rectangular contour, see the module documentation.

    Examples
    ========

    >>> from sympy import symbols, cosh, exp, oo
    >>> from sympy_extras.integrals.contours import rectangular_integral
    >>> x = symbols('x')
    >>> rectangular_integral(1/cosh(x), x)
    ConditionalValue(pi)
    >>> rectangular_integral(x**2/cosh(x), x)
    ConditionalValue(pi**3/4)
    >>> rectangular_integral(1/(2*cosh(x) + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    """
    form = _exponential_form(f, x)
    if form is None:
        return None
    degree_p, degree_q, order_p, order_q = _orders(form.fraction)
    c, k = form.c, form.k
    conditions: list[Boolean] = []
    lower = as_expr(c * (order_q - order_p))
    upper = as_expr(c * (degree_q - degree_p))
    if k.is_number:
        if not (lower < k < upper):
            return None
    else:
        conditions.extend([as_boolean(k > lower), as_boolean(k < upper)])
    h = as_expr(2 * pi / c)
    factor = as_expr(exp(I * k * h))
    unit = as_expr(k / c)
    if unit.is_integer is True:
        shifted = True
    elif unit.is_integer is False:
        shifted = False
    else:
        shifted = False
        # k/c lies strictly between two consecutive integers when the
        # window of decay is one unit wide: no integer inside
        if not (upper - lower == c and as_expr(lower / c).is_integer):
            conditions.append(as_boolean(Ne(sin(pi * unit), 0)))
    found = poles(form.fraction.denominator)
    if found is None:
        return None
    locator = _Locator(assumptions)
    z = Dummy('z')
    n = form.n
    orders = list(range(1, n + 2)) if shifted else list(range(0, n + 1))
    unknowns = [Dummy('A%d' % j) for j in range(n + 1)]
    equations: list[Expr] = []
    for m in orders:
        total: Expr = S.Zero
        for pole in found:
            placed = _strip_pole(pole, c, locator)
            if placed is None:
                return None
            point, on_axis = placed
            if on_axis:
                # a pole of R on the positive axis is a pole of F on the
                # real line, unless z**m cancels a simple pole at 0
                if point != 0 or pole.multiplicity != 1 or m == 0:
                    return None
                continue
            value = _strip_residue(form, pole, point, m, z)
            if value is None:
                return None
            total = total + value
        left: Expr = (unknowns[m] if m <= n else S.Zero) - factor * sum(
            (binomial(m, j) * (I * h)**(m - j) * unknowns[j] for j in range(min(m, n) + 1)), S.Zero)
        if shifted:
            # the coefficient of A_m is 1 - factor = 0
            left = as_expr(left + (unknowns[m] if m <= n else S.Zero) * (factor - 1)) if m <= n else left
        equations.append(as_expr(left - 2 * pi * I * total))
    system = Matrix([[as_expr(e).coeff(a) for a in unknowns] for e in equations])
    rhs = Matrix([-as_expr(e).subs({a: 0 for a in unknowns}) for e in equations])
    try:
        solution = system.LUsolve(rhs)
    except (ValueError, ZeroDivisionError, PolynomialError):
        return None
    value = as_expr(solution[n])
    if value.has(nan, zoo):
        return None
    value = _tidy(value, assumptions)
    if value.has(I):
        # the integral of a real integrand is real
        real = _part(value, True, assumptions)
        if real is not None:
            simpler = attempt(lambda: as_expr(trigsimp(simplify(real))), settings.timeout)
            value = real if simpler is None else simpler
    return ConditionalValue(value, as_boolean(And(*conditions)))


# ---------------------------------------------------------------------------
# 2. The sector

def _symbolic_monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(c, n)`` when ``e == c * x**n`` with ``c`` and ``n`` free of ``x``
    (``n`` symbolic allowed)."""
    coefficient, rest = e.as_independent(x, as_Add=False)
    rest_ = as_expr(rest)
    if rest_ == x:
        return (as_expr(coefficient), S.One)
    if isinstance(rest_, Pow) and rest_.base == x and not as_expr(rest_.exp).has(x):
        return (as_expr(coefficient), as_expr(rest_.exp))
    return None


def sector_integral(f: Expr, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**a/(b + c*x**n), (x, 0, oo))`` by the sector of angle
    ``2 pi/n``: ``(b/c)**((a + 1)/n) pi/(b n sin(pi (a + 1)/n))`` for
    ``-1 < a < n - 1`` and ``b, c > 0``.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.contours import sector_integral
    >>> x = symbols('x')
    >>> n = symbols('n', positive=True)
    >>> sector_integral(1/(1 + x**n), x, n > 1)
    ConditionalValue(pi/(n*sin(pi/n)), 0 < n - 1)
    >>> sector_integral(x**2/(3 + 2*x**6), x)
    ConditionalValue(sqrt(6)*pi/36)
    """
    alpha: Expr = S.Zero
    power: Optional[tuple[Expr, Expr, Expr]] = None
    constant: Expr = S.One
    for factor in Mul.make_args(f):
        e = as_expr(factor)
        if not e.has(x):
            constant = constant * e
        elif e == x:
            alpha = alpha + 1
        elif isinstance(e, Pow) and e.base == x and not as_expr(e.exp).has(x):
            alpha = alpha + as_expr(e.exp)
        elif isinstance(e, Pow) and e.exp == -1 and isinstance(e.base, Add) and power is None:
            b, rest = e.base.as_independent(x, as_Add=True)
            found = _symbolic_monomial(as_expr(rest), x)
            if found is None or as_expr(b) == 0:
                return None
            power = (as_expr(b), found[0], found[1])
        else:
            return None
    if power is None:
        return None
    b, c, n = power
    conditions: list[Boolean] = []
    for e in (b, c, n):
        if e.is_positive:
            continue
        if e.is_number:
            return None
        conditions.append(as_boolean(e > 0))
    conditions.extend([as_boolean(alpha > -1), as_boolean(alpha < n - 1)])
    value = as_expr(constant * (b / c)**((alpha + 1) / n) * pi / (b * n * sin(pi * (alpha + 1) / n)))
    return ConditionalValue(as_expr(simplify(value)), as_boolean(And(*conditions)))


# ---------------------------------------------------------------------------
# 3. The indented real line

def indented_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(R(x)*sin(k*x), (x, -oo, oo))`` (or ``cos``) for a rational
    ``R`` with simple real poles at which the integrand is regular, by
    the indented contour; over ``(0, oo)`` for an even integrand.

    Examples
    ========

    >>> from sympy import symbols, sin, oo
    >>> from sympy_extras.integrals.contours import indented_integral
    >>> x = symbols('x')
    >>> indented_integral(sin(x)/x, x, -oo, oo)
    ConditionalValue(pi)
    >>> indented_integral(sin(x)/(x*(x**2 + 1)), x, 0, oo)
    ConditionalValue(-pi*exp(-1)/2 + pi/2)
    """
    half = (a == 0 and b == oo)
    if not half and not (a == -oo and b == oo):
        return None
    trig: Optional[tuple[Expr, str]] = None
    rest: list[Expr] = []
    for factor in Mul.make_args(f):
        e = as_expr(factor)
        if isinstance(e, (sin, cos)) and e.has(x) and trig is None:
            found = monomial(as_expr(e.args[0]), x)
            if found is None or found[1] != 1:
                return None
            trig = (found[0], 'sin' if isinstance(e, sin) else 'cos')
        else:
            rest.append(e)
    if trig is None:
        return None
    k, kind = trig
    fraction = rational_function(as_expr(Mul(*rest)), x)
    if fraction is None or fraction.excess < 1:
        return None
    if half:
        # the integrand must be even: R odd for sin, even for cos
        mirrored = as_expr(fraction.as_expr().subs(x, -x))
        parity = -1 if kind == 'sin' else 1
        if as_expr(simplify(mirrored - parity * fraction.as_expr())) != 0:
            return None
    real_poles = poles(fraction.denominator)
    if real_poles is None:
        return None
    conditions: list[Boolean] = []
    if k.is_number:
        if not k > 0:
            return None
    elif ask(as_boolean(k > 0), assumptions) is not True:
        conditions.append(as_boolean(k > 0))
    locator = _Locator(assumptions)
    z = Dummy('z')
    total: Expr = S.Zero
    for pole in real_poles:
        s = locator.imaginary_sign(pole.point)
        if s is None or s < 0:
            if s is None:
                return None
            continue
        value = _residue(fraction, pole, exp(I * k * z), z)
        if value is None:
            return None
        if s > 0:
            total = total + 2 * pi * I * value
        else:
            if pole.multiplicity != 1:
                return None
            # the integrand must be regular there: the trigonometric factor vanishes
            vanishing = as_expr((sin if kind == 'sin' else cos)(k * pole.point))
            if as_expr(simplify(vanishing)) != 0:
                return None
            total = total + pi * I * value
    part = _part(total, kind == 'cos', assumptions)
    if part is None:
        return None
    value = as_expr(part / 2) if half else part
    return ConditionalValue(_tidy(value, assumptions), as_boolean(And(*conditions)))


def contour_integral(f: Expr, x: Symbol, a: Expr, b: Expr,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The first of the three contours which applies to
    ``Integral(f, (x, a, b))``, or ``None``."""
    a, b = as_expr(a), as_expr(b)
    if a == -oo and b == oo:
        if f.has(exp, HyperbolicFunction) and not f.has(sin, cos):
            found = rectangular_integral(f, x, assumptions)
            if found is not None:
                return found
    if a == 0 and b == oo and not f.has(sin, cos, exp):
        found = sector_integral(f, x, assumptions)
        if found is not None:
            return found
    if f.has(sin, cos):
        return indented_integral(f, x, a, b, assumptions)
    return None
