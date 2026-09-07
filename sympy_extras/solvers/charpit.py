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
"""
from __future__ import annotations

from typing import Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, expand
from sympy.core.relational import Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.integrals.integrals import Integral, integrate
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve as _solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr
from sympy_extras.settings import settings

__all__ = ['complete_integral', 'check_complete_integral']


class _Equation:
    def __init__(self, F: Expr, x: Symbol, y: Symbol, u: Symbol, p: Symbol, q: Symbol) -> None:
        self.F, self.x, self.y, self.u, self.p, self.q = F, x, y, u, p, q


def _as_symbols(equation: Basic, f: AppliedUndef) -> _Equation:
    """``F`` with ``u``, ``p``, ``q`` as symbols."""
    if len(f.args) != 2 or not all(isinstance(a, Symbol) for a in f.args):
        raise ValueError("a function of two symbols is expected, got %s" % (f,))
    x, y = f.args
    assert isinstance(x, Symbol) and isinstance(y, Symbol)
    lhs = equation.lhs - equation.rhs if isinstance(equation, Eq) else as_expr(equation)
    u, p, q = Dummy('u'), Dummy('p'), Dummy('q')
    replaced = as_expr(lhs).xreplace({Derivative(f, x): p, Derivative(f, y): q}).xreplace({f: u})
    if replaced.has(f) or replaced.atoms(Derivative):
        raise ValueError("only first order derivatives of %s are allowed" % (f,))
    return _Equation(as_expr(replaced), x, y, u, p, q)


def check_complete_integral(equation: Basic, f: AppliedUndef, solution: Basic) -> bool:
    """Whether ``solution`` (an equation ``f == expression`` or the
    expression) satisfies the partial differential equation."""
    return _satisfies(_as_symbols(equation, f), as_expr(solution.rhs if isinstance(solution, Eq) else solution))


def _satisfies(e: _Equation, expression: Expr) -> bool:
    value = e.F.xreplace({e.p: expression.diff(e.x), e.q: expression.diff(e.y)}).xreplace({e.u: expression})
    return simplify(value) == 0


def _first_integral(e: _Equation, a: Symbol, b: Symbol) -> Optional[Expr]:
    """A complete integral through the standard first integrals."""
    F, x, y, u, p, q = e.F, e.x, e.y, e.u, e.p, e.q
    candidates: list[Expr] = []
    # Clairaut: u = p x + q y + g(p, q)
    if _is_clairaut(F, e):
        candidates.append(as_expr(a*x + b*y + _clairaut_term(F, e, a, b)))
    # F(p, q) = 0
    if not F.has(x, y, u):
        for q_value in _solutions(F.xreplace({p: a}), q):
            candidates.append(as_expr(a*x + q_value*y + b))
    # F(u, p, q) = 0: q = a p
    if not F.has(x, y) and F.has(u):
        for p_value in _solutions(F.xreplace({q: a*p}), p):
            if p_value == 0:
                continue
            integral = attempt(lambda: integrate(1/p_value, u), settings.timeout)
            if integral is None or integral.has(Integral):
                continue
            # integral(u) = x + a y + b: solve for u when possible
            for u_value in _solutions(as_expr(integral - x - a*y - b), u):
                candidates.append(u_value)
    # separable F = A(x, p) - B(y, q)
    parts = _separable(F, e)
    if parts is not None:
        A, B = parts
        for p_value in _solutions(as_expr(A - a), p):
            for q_value in _solutions(as_expr(B - a), q):
                if p_value.has(y, q, u) or q_value.has(x, p, u):
                    continue
                first = attempt(lambda: integrate(p_value, x), settings.timeout)
                second = attempt(lambda: integrate(q_value, y), settings.timeout)
                if first is None or second is None or first.has(Integral) or second.has(Integral):
                    continue
                candidates.append(as_expr(first + second + b))
    # p = a or q = a as a first integral
    for variable, other in ((p, q), (q, p)):
        derivative = F.diff(x if variable == p else y) + variable*F.diff(u)
        if simplify(derivative) != 0:
            continue
        for other_value in _solutions(F.xreplace({variable: a}), other):
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
    if simplify(p_value.diff(y) - q_value.diff(x)) != 0:
        return None
    first = attempt(lambda: integrate(p_value, x), settings.timeout)
    if first is None or first.has(Integral):
        return None
    rest = as_expr(simplify(q_value - first.diff(y)))
    second = attempt(lambda: integrate(rest, y), settings.timeout)
    if second is None or second.has(Integral):
        return None
    return as_expr(first + second + b)


def _solutions(expression: Expr, variable: Symbol) -> list[Expr]:
    found = attempt(lambda: _solve(expression, variable), settings.timeout)
    if not found:
        return []
    return [as_expr(v) for v in found if isinstance(v, Expr)]


def _is_clairaut(F: Expr, e: _Equation) -> bool:
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    linear = as_expr(expand(F))
    return (linear.diff(u).is_constant(x, y, u, p, q) and simplify(linear.diff(x) + linear.diff(u)*p) == 0
            and simplify(linear.diff(y) + linear.diff(u)*q) == 0 and linear.diff(u) != 0)


def _clairaut_term(F: Expr, e: _Equation, a: Symbol, b: Symbol) -> Expr:
    """``f(a, b)`` with ``F = c*(u - p x - q y - f(p, q))``."""
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    c = as_expr(expand(F).diff(u))
    rest = as_expr(expand(F/c - u + p*x + q*y))
    return as_expr(-rest.xreplace({p: a, q: b}))


def _separable(F: Expr, e: _Equation) -> Optional[tuple[Expr, Expr]]:
    """``(A(x, p), B(y, q))`` with ``F = A - B``."""
    x, y, u, p, q = e.x, e.y, e.u, e.p, e.q
    if F.has(u):
        return None
    from sympy.core.add import Add
    A_terms: list[Expr] = []
    B_terms: list[Expr] = []
    for term in Add.make_args(expand(F)):
        term_ = as_expr(term)
        if not term_.has(y, q):
            A_terms.append(term_)
        elif not term_.has(x, p):
            B_terms.append(as_expr(-term_))
        else:
            return None
    if not A_terms or not B_terms:
        return None
    return as_expr(Add(*A_terms)), as_expr(Add(*B_terms))


def complete_integral(equation: Basic, f: AppliedUndef, constants: tuple[Symbol, Symbol] = (Symbol('a'), Symbol('b'))
                      ) -> Optional[Basic]:
    """A complete integral ``Eq(f, phi(x, y, a, b))`` of a first order
    partial differential equation in ``f(x, y)``, or ``None``.

    Examples
    ========

    >>> from sympy import Function, symbols, Eq
    >>> from sympy_extras.solvers.charpit import complete_integral
    >>> x, y = symbols('x y')
    >>> u = Function('u')(x, y)
    >>> p, q = u.diff(x), u.diff(y)
    >>> complete_integral(p*q - 1, u)
    Eq(u(x, y), a*x + b + y/a)
    >>> complete_integral(Eq(u, p*x + q*y + p*q), u)
    Eq(u(x, y), a*b + a*x + b*y)
    >>> complete_integral(p**2 + q**2 - u, u)
    Eq(u(x, y), (a*y + b + x)**2/(4*(a**2 + 1)))
    >>> complete_integral(p**2 + x - q - y**2, u)
    Eq(u(x, y), a*y + b - y**3/3 + 2*(a - x)**(3/2)/3)
    """
    e = _as_symbols(equation, f)
    a, b = constants
    solution = _first_integral(e, a, b)
    if solution is None:
        return None
    return Eq(f, solution)
