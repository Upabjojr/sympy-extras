"""Complete integrals of first order partial differential equations
`F(x, y, u, p, q) = 0` (`p = u_x`, `q = u_y`) by Charpit's method.

A *complete integral* is a solution `u = \\phi(x, y, a, b)` with two
arbitrary constants; every solution (the general integral and the
singular ones) is obtained from it by envelopes. Charpit's method looks
for a second relation `G(x, y, u, p, q) = a` which is a first integral
of the characteristic system

.. math::

    \\frac{dx}{F_p} = \\frac{dy}{F_q} = \\frac{du}{p F_p + q F_q}
    = \\frac{dp}{-(F_x + p F_u)} = \\frac{dq}{-(F_y + q F_u)},

solves `F = 0, G = a` for `p, q` and integrates the exact differential
`du = p\\,dx + q\\,dy`. The standard forms have well-known first
integrals:

* `F(p, q) = 0`: `p = a`, so `u = a x + f(a) y + b`;
* `F(u, p, q) = 0`: `q = a p`, so `u` is found from `\\int du/\\phi(u, a) = x + a y + b`;
* `F(x, p) = G(y, q)` (separable): `F(x, p) = a = G(y, q)`;
* `u = p x + q y + f(p, q)` (Clairaut): `u = a x + b y + f(a, b)`;

and otherwise the first integrals `p = a` or `q = a` are tried (they are
first integrals when `F_x + p F_u = 0` or `F_y + q F_u = 0`
respectively). Every complete integral returned is verified by
substitution.

SymPy's ``pdsolve`` solves first order **linear** equations only.

The module imports no SymPy: the equation is an expression of the core
(:mod:`sympy_extras.core`) in an applied undefined function and its
first derivatives; the equations polynomial in the unknown are solved by
the native :func:`~sympy_extras.polys.polyroots.roots` (the others by
the native port of SymPy's ``solve``,
:func:`~sympy_extras.solvers.solvers.solve_equation`), the polynomials
in a variable are integrated term by term, and the checks use the native
:func:`~sympy_extras.simplify.simplify.simplify`. One step has no
native counterpart and is taken from SymPy at run time through the face
:mod:`sympy_extras.solvers.charpit_sympy`:
``integrate`` for the integrands which are not polynomials in the
variable (``antiderivative_of_core``). The face also keeps the old API
on SymPy objects.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Optional

from sympy_extras._timeout import attempt
from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, diff, expand
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import log
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rationaltools import together
from sympy_extras.settings import settings
from sympy_extras.simplify.radsimp import fraction
from sympy_extras.simplify.simplify import simplify

__all__ = ['complete_integral', 'check_complete_integral']


class _Equation:
    """``F`` with ``u``, ``p``, ``q`` replaced by symbols."""

    def __init__(self, F: Expr, x: Symbol, y: Symbol, u: Symbol, p: Symbol, q: Symbol) -> None:
        self.F: Final[Expr] = F
        self.x: Final[Symbol] = x
        self.y: Final[Symbol] = y
        self.u: Final[Symbol] = u
        self.p: Final[Symbol] = p
        self.q: Final[Symbol] = q


def _difference(equation: Basic) -> Expr:
    """``lhs - rhs`` of an equation, or the expression itself."""
    if isinstance(equation, Eq):
        return equation.lhs - equation.rhs
    if not isinstance(equation, Expr):
        raise TypeError("an equation or an expression is expected, got %s" % (equation,))
    return equation


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced and the rebuilt nodes
    evaluated (SymPy's ``xreplace``; the core's does not evaluate)."""
    result = e._xreplace_evaluated(rule)
    if not isinstance(result, Expr):
        raise TypeError("expected an expression, got %s" % (result,))
    return result


def _as_symbols(equation: Basic, f: AppliedUndef) -> _Equation:
    """``F`` with ``u``, ``p``, ``q`` as symbols."""
    if len(f.args) != 2 or not all(isinstance(a, Symbol) for a in f.args):
        raise ValueError("a function of two symbols is expected, got %s" % (f,))
    x, y = f.args
    assert isinstance(x, Symbol) and isinstance(y, Symbol)
    lhs = _difference(equation)
    u, p, q = Dummy('u'), Dummy('p'), Dummy('q')
    derivatives: dict[Basic, Basic] = {diff(f, x): p, diff(f, y): q}
    function: dict[Basic, Basic] = {f: u}
    # the other derivatives are looked for before f is replaced (a
    # derivative of the symbol u would evaluate to 0)
    replaced = _replaced(lhs, derivatives)
    if replaced.has(Derivative):
        raise ValueError("only first order derivatives of %s are allowed" % (f,))
    replaced = _replaced(replaced, function)
    if replaced.has(f):
        raise ValueError("only first order derivatives of %s are allowed" % (f,))
    return _Equation(replaced, x, y, u, p, q)


def check_complete_integral(equation: Basic, f: AppliedUndef, solution: Basic) -> bool:
    """Whether ``solution`` (an equation ``f == expression`` or the
    expression) satisfies the partial differential equation.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols, Eq
    >>> from sympy_extras.solvers.charpit import check_complete_integral
    >>> x, y, a, b = symbols('x y a b')
    >>> u = Function('u')(x, y)
    >>> check_complete_integral(u.diff(x)*u.diff(y) - 1, u, Eq(u, a*x + y/a + b))
    True
    >>> check_complete_integral(u.diff(x)*u.diff(y) - 1, u, a*x + y + b)
    False
    """
    expression = solution.rhs if isinstance(solution, Eq) else solution
    if not isinstance(expression, Expr):
        raise TypeError("an equation or an expression is expected, got %s" % (solution,))
    return _satisfies(_as_symbols(equation, f), expression)


def _satisfies(e: _Equation, expression: Expr) -> bool:
    value = _replaced(_replaced(e.F, {e.p: diff(expression, e.x), e.q: diff(expression, e.y)}), {e.u: expression})
    return simplify(value) == 0


def _first_integral(e: _Equation, a: Symbol, b: Symbol) -> Optional[Expr]:
    """A complete integral through the standard first integrals."""
    F, x, y, u, p, q = e.F, e.x, e.y, e.u, e.p, e.q
    candidates: list[Expr] = []
    # Clairaut: u = p x + q y + g(p, q)
    if _is_clairaut(F, e):
        candidates.append(a*x + b*y + _clairaut_term(F, e, a, b))
    # F(p, q) = 0
    if not F.has(x, y, u):
        for q_value in _solutions(_replaced(F, {p: a}), q):
            candidates.append(a*x + q_value*y + b)
    # F(u, p, q) = 0: q = a p
    if not F.has(x, y) and F.has(u):
        for p_value in _solutions(_replaced(F, {q: a*p}), p):
            if p_value == 0:
                continue
            integral = _antiderivative(1/p_value, u)
            if integral is None:
                continue
            # integral(u) = x + a y + b: solve for u when possible
            for u_value in _solutions(integral - x - a*y - b, u):
                candidates.append(u_value)
    # separable F = A(x, p) - B(y, q)
    parts = _separable(F, e)
    if parts is not None:
        A, B = parts
        for p_value in _solutions(A - a, p):
            for q_value in _solutions(B - a, q):
                if p_value.has(y, q, u) or q_value.has(x, p, u):
                    continue
                first = _antiderivative(p_value, x)
                second = _antiderivative(q_value, y)
                if first is None or second is None:
                    continue
                candidates.append(first + second + b)
    # p = a or q = a as a first integral
    for variable, other in ((p, q), (q, p)):
        derivative = diff(F, x if variable == p else y) + variable*diff(F, u)
        if simplify(derivative) != 0:
            continue
        for other_value in _solutions(_replaced(F, {variable: a}), other):
            p_value, q_value = (a, other_value) if variable == p else (other_value, a)
            solution = _integrate_exact(p_value, q_value, e, b)
            if solution is not None:
                candidates.append(solution)
    for candidate in candidates:
        if _satisfies(e, candidate):
            return candidate
    return None


def _integrate_exact(p_value: Expr, q_value: Expr, e: _Equation, b: Symbol) -> Optional[Expr]:
    """``u`` with ``du = p dx + q dy`` when ``p, q`` do not depend on
    ``u`` (a total differential)."""
    x, y, u = e.x, e.y, e.u
    if p_value.has(u) or q_value.has(u):
        return None
    if simplify(diff(p_value, y) - diff(q_value, x)) != 0:
        return None
    first = _antiderivative(p_value, x)
    if first is None:
        return None
    rest = simplify(q_value - diff(first, y))
    second = _antiderivative(rest, y)
    if second is None:
        return None
    return first + second + b


def _antiderivative(e: Expr, x: Symbol) -> Optional[Expr]:
    """An antiderivative of ``e`` in ``x``, ``None`` when none is found:
    term by term for a sum of terms ``c*x**n`` (``c`` free of ``x``, ``n``
    rational), from SymPy's ``integrate`` otherwise."""
    native = _power_antiderivative(e, x)
    if native is not None:
        return native
    from .charpit_sympy import antiderivative_of_core
    return antiderivative_of_core(e, x)


def _power_antiderivative(e: Expr, x: Symbol) -> Optional[Expr]:
    """``e`` integrated term by term when it is a sum of terms ``c*x**n``
    (``c`` free of ``x``, ``n`` rational), ``None`` otherwise."""
    if not e.has(x):
        return mul(e, x)
    # the constant factor stays out, as with SymPy's integrate: SymPy's
    # solve finds no root of a*log(u) + log(u) - 1 (it does of
    # (a + 1)*log(u) - 1)
    constant, e = e.as_independent(x, as_Add=False)
    expanded = expand(e)
    pieces: list[Expr] = []
    for term in (expanded.args if isinstance(expanded, Add) else (expanded,)):
        assert isinstance(term, Expr)
        c, dependent = term.as_independent(x, as_Add=False)
        n = Fraction(0)
        for factor in (dependent.args if isinstance(dependent, Mul) else (dependent,)):
            if factor == x:
                n += 1
            elif isinstance(factor, Pow) and factor.base == x and isinstance(factor.exp, Rational):
                n += Fraction(factor.exp.p, factor.exp.q)
            elif not (isinstance(factor, Integer) and factor.p == 1):
                return None
        if n == -1:
            pieces.append(mul(c, log(x)))
        else:
            m = Rational(n.numerator + n.denominator, n.denominator)
            pieces.append(mul(c, pow(x, m), pow(m, Integer(-1))))
    return mul(constant, add(*pieces))


def _solutions(expression: Expr, variable: Symbol) -> list[Expr]:
    """The solutions of ``expression = 0`` for ``variable``: the roots of
    the numerator when it is a polynomial in ``variable`` whose roots are
    all found, the native port of SymPy's ``solve`` otherwise."""
    native = attempt(lambda: _polynomial_solutions(expression, variable), settings.timeout, native=True)
    if native is not None:
        return native
    from .solvers import solve_equation
    return attempt(lambda: solve_equation(expression, variable), settings.timeout, native=True) or []


def _polynomial_solutions(expression: Expr, variable: Symbol) -> Optional[list[Expr]]:
    """The roots (simplified when there are at most two, in the order of
    SymPy's ``solve``) of the numerator of ``expression`` which do not
    cancel its denominator, when the numerator is a polynomial in
    ``variable`` with all its roots found; ``None`` otherwise."""
    numerator, denominator = fraction(together(expression))
    if not numerator.has(variable):
        return None
    try:
        polynomial = Poly(numerator, variable)
    except BasePolynomialError:
        return None
    if polynomial.degree() < 1 or any(c.has(variable) for c in polynomial.all_coeffs()):
        return None
    found = roots(polynomial)
    if not isinstance(found, dict) or sum(found.values()) != polynomial.degree():
        return None
    values: list[Expr] = []
    for root in found:
        assert isinstance(root, Expr)
        if denominator.has(variable) and simplify(_replaced(denominator, {variable: root})) == 0:
            continue
        # as SymPy's solve: more than two roots are not simplified
        value = simplify(root) if len(found) <= 2 else root
        if value not in values:
            values.append(value)
    return sorted(values, key=default_sort_key)


def _is_clairaut(F: Expr, e: _Equation) -> bool:
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    linear = expand(F)
    coefficient = diff(linear, u)
    return (not coefficient.free_symbols & {x, y, u, p, q} and simplify(diff(linear, x) + coefficient*p) == 0
            and simplify(diff(linear, y) + coefficient*q) == 0 and coefficient != 0)


def _clairaut_term(F: Expr, e: _Equation, a: Symbol, b: Symbol) -> Expr:
    """``f(a, b)`` with ``F = c*(u - p x - q y - f(p, q))``."""
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    c = diff(expand(F), u)
    rest = expand(F/c - u + p*x + q*y)
    return -_replaced(rest, {p: a, q: b})


def _separable(F: Expr, e: _Equation) -> Optional[tuple[Expr, Expr]]:
    """``(A(x, p), B(y, q))`` with ``F = A - B``."""
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    if F.has(u):
        return None
    A_terms: list[Expr] = []
    B_terms: list[Expr] = []
    expanded = expand(F)
    for term in (expanded.args if isinstance(expanded, Add) else (expanded,)):
        assert isinstance(term, Expr)
        if not term.has(y, q):
            A_terms.append(term)
        elif not term.has(x, p):
            B_terms.append(-term)
        else:
            return None
    if not A_terms or not B_terms:
        return None
    return add(*A_terms), add(*B_terms)


def complete_integral(equation: Basic, f: AppliedUndef, constants: tuple[Symbol, Symbol] = (Symbol('a'), Symbol('b'))
                      ) -> Optional[Eq]:
    """A complete integral ``Eq(f, phi(x, y, a, b))`` of a first order
    partial differential equation in ``f(x, y)`` (an expression, ``= 0``,
    or an equation), or ``None``.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols, Eq
    >>> from sympy_extras.solvers.charpit import complete_integral
    >>> x, y = symbols('x y')
    >>> u = Function('u')(x, y)
    >>> p, q = u.diff(x), u.diff(y)
    >>> complete_integral(p*q - 1, u)
    Eq(u(x, y), a*x + b + y/a)
    >>> complete_integral(Eq(u, p*x + q*y + p*q), u)
    Eq(u(x, y), a*b + a*x + b*y)
    >>> complete_integral(p**2 + q**2 - 1, u)
    Eq(u(x, y), a*x + b - y*sqrt(1 - a**2))
    >>> complete_integral(p**2 - q, u)
    Eq(u(x, y), a**2*y + a*x + b)
    """
    e = _as_symbols(equation, f)
    a, b = constants
    solution = _first_integral(e, a, b)
    if solution is None:
        return None
    return Eq(f, solution)
