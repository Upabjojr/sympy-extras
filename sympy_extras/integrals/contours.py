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

The module works on the expressions of the typed core
(:mod:`sympy_extras.core`) and imports no SymPy: the poles, residues
and their placement of the native :mod:`.residues`, the core's
``residue`` (:mod:`sympy_extras.series.residues`) at the multiple poles
of the strip, the triangular system solved by elimination on core
expressions (SymPy's ``Matrix.LUsolve``), the native ``simplify`` and
``trigsimp``. :mod:`.contours_sympy` is its face on SymPy objects.

Examples
========

>>> from sympy_extras.core import symbols, oo
>>> from sympy_extras.functions import exp, sinh, sin
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
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.assumptions.ask import Assumptions, ask
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.logic import Boolean, and_
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import I, Integer, Rational, oo, pi
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import gt, lt, ne
from sympy_extras.core.singleton import S
from sympy_extras.core.symbol import Symbol
from sympy_extras.functions.combinatorial.factorials import binomial
from sympy_extras.functions.elementary.complexes import abs_, arg, re
from sympy_extras.functions.elementary.exponential import Exp, exp, log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Coth, Csch, Sech, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Cos, Sin, cos, sin
from sympy_extras.series.residues import residue
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify
from sympy_extras.simplify.trigsimp import trigsimp
from .conditions import ConditionalValue
from .mellin import monomial
from .residues import (_Fraction, _fresh, _is_bad, _Locator, _part, _Pole, _Polynomial, _replace, _residue, _tidy,
                       poles, rational_function)

__all__ = ['rectangular_integral', 'sector_integral', 'indented_integral', 'contour_integral']

#: the hyperbolic functions (SymPy's ``HyperbolicFunction``)
_HYPERBOLIC: Final[tuple[type, ...]] = (Sinh, Cosh, Tanh, Coth, Sech, Csch)


def _budget() -> Optional[float]:
    return settings.timeout


def _minus(e: Expr) -> Expr:
    return mul(Integer(-1), e)


def _quotient(p: Expr, q: Expr) -> Expr:
    return mul(p, pow(q, Integer(-1)))


# ---------------------------------------------------------------------------
# 1. The rectangle

class _Exponential:
    """``x**n * exp(k x) * R(u)`` with ``u = exp(c x)``."""

    __slots__ = ('n', 'k', 'c', 'fraction')

    def __init__(self, n: int, k: Expr, c: Rational, fraction: _Fraction) -> None:
        self.n: Final[int] = n
        self.k: Final[Expr] = k
        self.c: Final[Rational] = c
        self.fraction: Final[_Fraction] = fraction


def _exponential_form(f: Expr, x: Symbol) -> Optional[_Exponential]:
    """``f`` written as ``x**n exp(k x) R(exp(c x))``, hyperbolic functions
    rewritten first; ``c`` is the generator of the rational exponents,
    ``k`` the coefficient of the exponentials it does not generate (one
    such at most, possibly symbolic)."""
    g = f.rewrite(Exp) if f.has(*_HYPERBOLIC) else f
    n = 0
    rest: list[Expr] = []
    for e in Mul.make_args(g):
        if e == x:
            n += 1
        elif isinstance(e, Pow) and e.base == x and isinstance(e.exp, Integer) and e.exp.p > 0:
            n += e.exp.p
        else:
            rest.append(e)
    g = mul(*rest)
    coefficients: list[Rational] = []
    symbolic: list[Expr] = []
    for node in g.atoms(Exp):
        if not isinstance(node, Exp):
            continue
        argument = node.args[0]
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
    numerator = 0
    denominator = 1
    for c_ in coefficients:
        numerator = gcd(numerator, c_.p)
        denominator = lcm(denominator, c_.q)
    c: Rational = Rational(numerator, denominator) if coefficients else S.One
    u = _fresh('u')
    replacement: dict[Basic, Basic] = {}
    k: Expr = S.Zero
    for node in g.atoms(Exp):
        if not isinstance(node, Exp):
            continue
        argument = node.args[0]
        if not argument.has(x):
            continue
        found = monomial(argument, x)
        if found is None:
            return None
        coefficient = found[0]
        if isinstance(coefficient, Rational):
            ratio = _quotient(coefficient, c)
            if not isinstance(ratio, Integer):
                return None
            replacement[node] = pow(u, ratio)
        else:
            if len(symbolic) > 1 or k != 0:
                return None
            # exp(a x) stays as the factor exp(k x): the node itself is
            # replaced by a marker removed below
            k = coefficient
            replacement[node] = S.One
    h = _replace(g, replacement)
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


def _order_at_zero(p: _Polynomial) -> int:
    for i, c in enumerate(reversed(p.coefficients)):
        if c != 0:
            return i
    return 0


def _strip_pole(pole: _Pole, c: Rational, locator: _Locator) -> Optional[tuple[Expr, bool]]:
    """The point of the strip ``0 < Im z < 2 pi/c`` where ``exp(c z)`` is
    the pole of ``R``, and whether it lies on the real axis; ``None``
    when its argument cannot be placed."""
    u = pole.point
    if u == 0:
        return None
    if isinstance(u, Rational):
        if u.p > 0:
            return (_quotient(log(u), c), True)
        return (_quotient(add(log(_minus(u)), mul(I, pi)), c), False)
    s = locator.imaginary_sign(pole)
    if s is None:
        return None
    if s == 0:
        positive = locator.decide(gt(re(u), 0))
        if positive is None:
            return None
        if positive:
            return (_quotient(log(u), c), True)
        return (_quotient(add(log(_minus(u)), mul(I, pi)), c), False)
    theta = arg(u) if s > 0 else add(arg(u), mul(Integer(2), pi))
    return (_quotient(add(log(abs_(u)), mul(I, theta)), c), False)


def _strip_residue(form: _Exponential, pole: _Pole, point: Expr, m: int, z: Symbol) -> Optional[Expr]:
    """The residue of ``z**m exp(k z) R(exp(c z))`` at ``point``."""
    u = pole.point
    if pole.multiplicity == 1:
        p = form.fraction.numerator.eval(u)
        dq = form.fraction.denominator.diff().eval(u)
        return mul(pow(point, Integer(m)), exp(mul(form.k, point)), p, pow(mul(form.c, u, dq), Integer(-1)))
    expression = mul(pow(z, Integer(m)), exp(mul(form.k, z)),
                     _replace(form.fraction.as_expr(), {form.fraction.x: exp(mul(form.c, z))}))
    value = attempt(lambda: residue(expression, z, point), _budget(), native=True)
    if value is None or _is_bad(value):
        return None
    return value


def _solve(system: list[list[Expr]], rhs: list[Expr]) -> Optional[list[Expr]]:
    """The solution of the square linear system by Gaussian elimination
    with the first pivot not zero (SymPy's ``LUsolve``), ``None`` for a
    singular system."""
    n = len(system)
    rows = [list(r) + [b] for r, b in zip(system, rhs)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if rows[i][j].is_zero is not True), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        for i in range(j + 1, n):
            if rows[i][j].is_zero is True:
                continue
            ratio = _quotient(rows[i][j], rows[j][j])
            rows[i] = [add(rows[i][t], _minus(mul(ratio, rows[j][t]))) for t in range(n + 1)]
    solution: list[Expr] = [S.Zero] * n
    for j in range(n - 1, -1, -1):
        known = add(*[mul(rows[j][t], solution[t]) for t in range(j + 1, n)])
        solution[j] = _quotient(add(rows[j][n], _minus(known)), rows[j][j])
    return solution


def rectangular_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**n * exp(k*x) * R(exp(c*x)), (x, -oo, oo))`` by the
    rectangular contour, see the module documentation.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.functions import cosh
    >>> from sympy_extras.integrals.contours import rectangular_integral
    >>> x = symbols('x')
    >>> rectangular_integral(1/cosh(x), x)
    ConditionalValue(pi)
    >>> rectangular_integral(x**2/cosh(x), x)
    ConditionalValue(pi**3/4)
    >>> rectangular_integral(1/(2*cosh(x) + 1), x)
    ConditionalValue(2*sqrt(3)*pi/9)
    """
    form = _exponential_form(to_expr(f), x)
    if form is None:
        return None
    degree_p, degree_q, order_p, order_q = _orders(form.fraction)
    c, k = form.c, form.k
    conditions: list[Boolean] = []
    lower = mul(c, Integer(order_q - order_p))
    upper = mul(c, Integer(degree_q - degree_p))
    if k.is_number:
        if not (lt(lower, k) == S.true and lt(k, upper) == S.true):
            return None
    else:
        conditions.extend([gt(k, lower), lt(k, upper)])
    h = _quotient(mul(Integer(2), pi), c)
    factor = exp(mul(I, k, h))
    unit = _quotient(k, c)
    if unit.is_integer is True:
        shifted = True
    elif unit.is_integer is False:
        shifted = False
    else:
        shifted = False
        # k/c lies strictly between two consecutive integers when the
        # window of decay is one unit wide: no integer inside
        if not (add(upper, _minus(lower)) == c and _quotient(lower, c).is_integer):
            conditions.append(ne(sin(mul(pi, unit)), 0))
    found = poles(form.fraction.denominator)
    if found is None:
        return None
    locator = _Locator(assumptions)
    z = _fresh('z')
    n = form.n
    orders = list(range(1, n + 2)) if shifted else list(range(0, n + 1))
    system: list[list[Expr]] = []
    rhs: list[Expr] = []
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
            total = add(total, value)
        # the coefficients of A_0, ..., A_n in
        # A_m - factor sum_j binomial(m, j) (I h)**(m - j) A_j
        row: list[Expr] = []
        for j in range(n + 1):
            entry: Expr = S.Zero
            if j <= min(m, n):
                entry = _minus(mul(factor, binomial(Integer(m), Integer(j)), pow(mul(I, h), Integer(m - j))))
            if j == m and not shifted:
                entry = add(entry, S.One)
            elif j == m and shifted:
                # the coefficient of A_m is 1 - factor = 0
                entry = S.Zero
            row.append(entry)
        system.append(row)
        rhs.append(mul(Integer(2), pi, I, total))
    solution = _solve(system, rhs)
    if solution is None:
        return None
    value = solution[n]
    if _is_bad(value):
        return None
    value = _tidy(value, assumptions)
    if value.has(I):
        # the integral of a real integrand is real
        real = _part(value, True, assumptions)
        if real is not None:
            simpler = attempt(lambda: trigsimp(simplify(real)), _budget(), native=True)
            value = real if simpler is None else simpler
    return ConditionalValue(value, and_(*conditions))


# ---------------------------------------------------------------------------
# 2. The sector

def _symbolic_monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """``(c, n)`` when ``e == c * x**n`` with ``c`` and ``n`` free of ``x``
    (``n`` symbolic allowed)."""
    coefficient, rest = e.as_independent(x, as_Add=False)
    if rest == x:
        return (coefficient, S.One)
    if isinstance(rest, Pow) and rest.base == x and not rest.exp.has(x):
        return (coefficient, rest.exp)
    return None


def sector_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(x**a/(b + c*x**n), (x, 0, oo))`` by the sector of angle
    ``2 pi/n``: ``(b/c)**((a + 1)/n) pi/(b n sin(pi (a + 1)/n))`` for
    ``-1 < a < n - 1`` and ``b, c > 0``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Symbol
    >>> from sympy_extras.integrals.contours import sector_integral
    >>> x = symbols('x')
    >>> n = Symbol('n', positive=True)
    >>> sector_integral(1/(1 + x**n), x, n > 1)
    ConditionalValue(pi/(n*sin(pi/n)), 0 < n - 1)
    >>> sector_integral(x**2/(3 + 2*x**6), x)
    ConditionalValue(sqrt(6)*pi/36)
    """
    alpha: Expr = S.Zero
    power: Optional[tuple[Expr, Expr, Expr]] = None
    constant: Expr = S.One
    for e in Mul.make_args(to_expr(f)):
        if not e.has(x):
            constant = mul(constant, e)
        elif e == x:
            alpha = add(alpha, S.One)
        elif isinstance(e, Pow) and e.base == x and not e.exp.has(x):
            alpha = add(alpha, e.exp)
        elif isinstance(e, Pow) and e.exp == -1 and isinstance(e.base, Add) and power is None:
            b, rest = e.base.as_independent(x, as_Add=True)
            found = _symbolic_monomial(rest, x)
            if found is None or b == 0:
                return None
            power = (b, found[0], found[1])
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
        conditions.append(gt(e, 0))
    conditions.extend([gt(alpha, -1), lt(alpha, add(n, Integer(-1)))])
    exponent = _quotient(add(alpha, S.One), n)
    value = mul(constant, pow(_quotient(b, c), exponent), pi,
                pow(mul(b, n, sin(mul(pi, exponent))), Integer(-1)))
    return ConditionalValue(simplify(value), and_(*conditions))


# ---------------------------------------------------------------------------
# 3. The indented real line

def indented_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                      assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(R(x)*sin(k*x), (x, -oo, oo))`` (or ``cos``) for a rational
    ``R`` with simple real poles at which the integrand is regular, by
    the indented contour; over ``(0, oo)`` for an even integrand.

    Examples
    ========

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import sin
    >>> from sympy_extras.integrals.contours import indented_integral
    >>> x = symbols('x')
    >>> indented_integral(sin(x)/x, x, -oo, oo)
    ConditionalValue(pi)
    >>> indented_integral(sin(x)/(x*(x**2 + 1)), x, 0, oo)
    ConditionalValue(-pi*exp(-1)/2 + pi/2)
    """
    lower, upper = to_expr(a), to_expr(b)
    half = (lower == 0 and upper == oo)
    if not half and not (lower == S.NegativeInfinity and upper == oo):
        return None
    trig: Optional[tuple[Expr, str]] = None
    rest: list[Expr] = []
    for e in Mul.make_args(to_expr(f)):
        if isinstance(e, (Sin, Cos)) and e.has(x) and trig is None:
            found = monomial(e.args[0], x)
            if found is None or found[1] != 1:
                return None
            trig = (found[0], 'sin' if isinstance(e, Sin) else 'cos')
        else:
            rest.append(e)
    if trig is None:
        return None
    k, kind = trig
    fraction = rational_function(mul(*rest), x)
    if fraction is None or fraction.excess < 1:
        return None
    if half:
        # the integrand must be even: R odd for sin, even for cos
        mirrored = _replace(fraction.as_expr(), {x: _minus(x)})
        parity = -1 if kind == 'sin' else 1
        if simplify(add(mirrored, mul(Integer(-parity), fraction.as_expr()))) != 0:
            return None
    real_poles = poles(fraction.denominator)
    if real_poles is None:
        return None
    conditions: list[Boolean] = []
    if k.is_number:
        if gt(k, 0) != S.true:
            return None
    elif ask(gt(k, 0), assumptions) is not True:
        conditions.append(gt(k, 0))
    locator = _Locator(assumptions)
    z = _fresh('z')
    total: Expr = S.Zero
    for pole in real_poles:
        s = locator.imaginary_sign(pole)
        if s is None:
            return None
        if s < 0:
            continue
        value = _residue(fraction, pole, exp(mul(I, k, z)), z)
        if value is None:
            return None
        if s > 0:
            total = add(total, mul(Integer(2), pi, I, value))
        else:
            if pole.multiplicity != 1:
                return None
            # the integrand must be regular there: the trigonometric factor vanishes
            vanishing = (sin if kind == 'sin' else cos)(mul(k, pole.point))
            if simplify(vanishing) != 0:
                return None
            total = add(total, mul(pi, I, value))
    part = _part(total, kind == 'cos', assumptions)
    if part is None:
        return None
    value = mul(part, S.Half) if half else part
    return ConditionalValue(_tidy(value, assumptions), and_(*conditions))


def contour_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """The first of the three contours which applies to
    ``Integral(f, (x, a, b))``, or ``None``.

    >>> from sympy_extras.core import symbols, oo
    >>> from sympy_extras.functions import cosh
    >>> from sympy_extras.integrals.contours import contour_integral
    >>> x = symbols('x')
    >>> contour_integral(1/cosh(x), x, -oo, oo)
    ConditionalValue(pi)
    """
    e, lower, upper = to_expr(f), to_expr(a), to_expr(b)
    if lower == S.NegativeInfinity and upper == oo:
        if e.has(Exp, *_HYPERBOLIC) and not e.has(Sin, Cos):
            found = rectangular_integral(e, x, assumptions)
            if found is not None:
                return found
    if lower == 0 and upper == oo and not e.has(Sin, Cos, Exp):
        found = sector_integral(e, x, assumptions)
        if found is not None:
            return found
    if e.has(Sin, Cos):
        return indented_integral(e, x, lower, upper, assumptions)
    return None
