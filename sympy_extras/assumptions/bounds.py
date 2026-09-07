"""Polynomial bounds of elementary functions, handed to the cylindrical
algebraic decomposition (the method of MetiTarski).

A relation containing ``exp``, ``log``, ``sin``, ``cos``, ``atan``,
``sinh``, ``cosh``, ``tanh``, square roots, rational powers or absolute
values of polynomial arguments is turned into a polynomial problem: every
function application `g(u)` is replaced by a fresh real variable `t`
together with polynomial constraints which the true value satisfies, such
as `t \\ge 1 + u` for `t = e^u`, or `t^2 = u \\wedge t \\ge 0` for
`t = \\sqrt{u}` (exact). If the relation holds for **every** value of the
new variables allowed by the constraints (a question the CAD decides), it
holds for the true values; likewise if it fails for every such value. The
bounds are the classical Taylor and Padé inequalities; they are tight
enough for statements with some slack, never for statements which are
tight at a point.

References
==========

.. [AkbarpourPaulson] B. Akbarpour, L. C. Paulson, MetiTarski: an
   automatic theorem prover for real-valued special functions, Journal of
   Automated Reasoning 44 (2010).
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Relational, Eq
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import sinh, cosh, tanh
from sympy.functions.elementary.trigonometric import sin, cos, atan
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or, Not,
    Implies, Equivalent, Xor, ITE)

from sympy_extras._typing import as_boolean, as_expr, free_symbols

__all__ = ['polynomial_abstraction', 'Abstraction']

#: the constraints on ``t`` standing for ``g(u)``
BoundRule = Callable[[Symbol, Expr], list[Boolean]]


def _exp_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > 0, t >= 1 + u,
            Implies(u >= 0, t >= 1 + u + u**2/2 + u**3/6 + u**4/24),
            Implies(u <= 0, t <= 1 + u + u**2/2),
            Implies(u < 1, t*(1 - u) <= 1),
            Implies(And(u >= 0, u <= 1), t <= 1 + u + u**2/2 + u**3/6 + u**4/8)]


def _log_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    # valid for u > 0, which the reality of log(u) requires
    return [u > 0, t <= u - 1, u*t >= u - 1,
            Implies(u >= 1, t*(u + 1) >= 2*(u - 1)),
            Implies(u <= 1, t*(u + 1) <= 2*(u - 1))]


def _sin_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= -1, t <= 1,
            Implies(u >= 0, t <= u), Implies(u <= 0, t >= u),
            Implies(u >= 0, t >= u - u**3/6), Implies(u <= 0, t <= u - u**3/6),
            Implies(u >= 0, t <= u - u**3/6 + u**5/120),
            Implies(u <= 0, t >= u - u**3/6 + u**5/120)]


def _cos_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= -1, t <= 1, t >= 1 - u**2/2, t <= 1 - u**2/2 + u**4/24]


def _atan_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > -Rational(8, 5), t < Rational(8, 5),
            Implies(u >= 0, And(t >= 0, t <= u, t >= u - u**3/3, t*(1 + u**2) >= u)),
            Implies(u <= 0, And(t <= 0, t >= u, t <= u - u**3/3, t*(1 + u**2) <= u))]


def _tanh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > -1, t < 1,
            Implies(u >= 0, And(t >= 0, t <= u, t >= u - u**3/3)),
            Implies(u <= 0, And(t <= 0, t >= u, t <= u - u**3/3))]


def _sinh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [Implies(u >= 0, And(t >= u, t >= u + u**3/6)),
            Implies(u <= 0, And(t <= u, t <= u + u**3/6))]


def _cosh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= 1, t >= 1 + u**2/2]


_RULES: dict[type, BoundRule] = {
    exp: _exp_bounds, log: _log_bounds, sin: _sin_bounds, cos: _cos_bounds,
    atan: _atan_bounds, tanh: _tanh_bounds, sinh: _sinh_bounds, cosh: _cosh_bounds,
}


class Abstraction:
    """A formula with its function applications replaced by variables.

    Attributes
    ==========

    formula : Boolean
        The polynomial formula.
    constraints : list of Boolean
        The polynomial constraints satisfied by the true values of the
        new variables.
    variables : list of Symbol
        The new variables.
    """

    def __init__(self, formula: Boolean, constraints: list[Boolean], variables: list[Symbol]) -> None:
        self.formula = formula
        self.constraints = constraints
        self.variables = variables

    def __repr__(self) -> str:
        return "Abstraction(%s, %s)" % (self.formula, self.constraints)


class _Abstractor:
    def __init__(self, real: set[Symbol]) -> None:
        self.real = real
        self.constraints: list[Boolean] = []
        self.variables: list[Symbol] = []
        self.known: dict[Expr, Symbol] = {}

    def _fresh(self, name: str) -> Symbol:
        t = Dummy(name, real=True)
        self.variables.append(t)
        return t

    def _polynomial(self, e: Expr) -> bool:
        symbols = free_symbols(e)
        return symbols <= (self.real | set(self.variables)) and e.is_polynomial(*symbols) and \
            all(c.is_rational for c in e.as_poly(*symbols).coeffs()) if symbols else e.is_rational is True

    def expression(self, e: Expr) -> Optional[Expr]:
        """``e`` with its function applications replaced, bottom-up."""
        if e in self.known:
            return self.known[e]
        if isinstance(e, tuple(_RULES)) or isinstance(e, Abs):
            inner = self.expression(as_expr(e.args[0]))
            if inner is None or not self._polynomial(inner):
                return None
            t = self._fresh('t')
            if isinstance(e, Abs):
                self.constraints.extend([t >= 0, Or(Eq(t, inner), Eq(t, -inner))])
            else:
                self.constraints.extend(_RULES[type(e)](t, inner))
            self.known[e] = t
            return t
        if isinstance(e, Pow) and isinstance(e.exp, Rational) and not e.exp.is_integer:
            inner = self.expression(as_expr(e.base))
            if inner is None or not self._polynomial(inner):
                return None
            p, q = int(e.exp.p), int(e.exp.q)
            t = self._fresh('t')
            # a real root of a nonnegative base (the reality of the power)
            self.constraints.extend([inner >= 0, t >= 0, Eq(t**q, inner**p)] if p > 0
                                    else [inner > 0, t > 0, Eq(t**q*inner**(-p), 1)])
            self.known[e] = t
            return t
        if e.args and not isinstance(e, Symbol):
            new_args: list[Basic] = []
            changed = False
            for a in e.args:
                if isinstance(a, Expr):
                    b = self.expression(a)
                    if b is None:
                        return None
                    changed = changed or b is not a
                    new_args.append(b)
                else:
                    new_args.append(a)
            return as_expr(e.func(*new_args)) if changed else e
        return e

    def formula(self, f: Boolean) -> Optional[Boolean]:
        if isinstance(f, Relational):
            lhs, rhs = self.expression(as_expr(f.lhs)), self.expression(as_expr(f.rhs))
            if lhs is None or rhs is None:
                return None
            return as_boolean(f.func(lhs, rhs))
        if isinstance(f, (And, Or, Not, Implies, Equivalent, Xor, ITE)):
            parts: list[Boolean] = []
            for a in f.args:
                b = self.formula(as_boolean(a))
                if b is None:
                    return None
                parts.append(b)
            return as_boolean(f.func(*parts))
        if isinstance(f, (BooleanTrue, BooleanFalse)):
            return f
        # memberships and predicates are not polynomial statements
        return None


def polynomial_abstraction(formula: Boolean, real: set[Symbol], max_variables: int = 3) -> Optional[Abstraction]:
    """Replace the elementary functions of a formula by variables with
    polynomial constraints.

    Examples
    ========

    >>> from sympy import exp, sqrt
    >>> from sympy.abc import x
    >>> from sympy_extras.assumptions.bounds import polynomial_abstraction
    >>> a = polynomial_abstraction(exp(x) > 1 + x, {x})
    >>> a.formula, len(a.constraints)
    (_t > x + 1, 6)
    >>> polynomial_abstraction(sqrt(x) > 1, {x}).constraints
    [x >= 0, _t >= 0, Eq(_t**2, x)]
    """
    abstractor = _Abstractor(real)
    result = abstractor.formula(formula)
    if result is None or not abstractor.variables or len(abstractor.variables) > max_variables:
        return None
    return Abstraction(result, abstractor.constraints, abstractor.variables)
