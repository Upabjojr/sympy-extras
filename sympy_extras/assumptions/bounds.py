"""Polynomial bounds of elementary functions, handed to the cylindrical
algebraic decomposition (the method of MetiTarski), on the core.

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

This module imports no SymPy: it takes and returns the expressions of
:mod:`sympy_extras.core` (and the functions of :mod:`sympy_extras.functions`);
:mod:`sympy_extras.assumptions.bounds_sympy` is its SymPy face.

References
==========

.. [AkbarpourPaulson] B. Akbarpour, L. C. Paulson, MetiTarski: an
   automatic theorem prover for real-valued special functions, Journal of
   Automated Reasoning 44 (2010).
"""
from __future__ import annotations

from typing import Callable, Final, Optional

from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, Equivalent, Implies, Not, Or, Xor, and_,
    implies, or_, to_boolean)
from sympy_extras.core.numbers import Rational
from sympy_extras.core.power import Pow
from sympy_extras.core.relational import Relational, eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.complexes import Abs
from sympy_extras.functions.elementary.exponential import Exp, Log
from sympy_extras.functions.elementary.hyperbolic import Cosh, Sinh, Tanh
from sympy_extras.functions.elementary.trigonometric import Atan, Cos, Sin

__all__ = ['polynomial_abstraction', 'Abstraction']

#: the constraints on ``t`` standing for ``g(u)``
BoundRule = Callable[[Symbol, Expr], list[Boolean]]


def _exp_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > 0, t >= 1 + u,
            implies(u >= 0, t >= 1 + u + u**2/2 + u**3/6 + u**4/24),
            implies(u <= 0, t <= 1 + u + u**2/2),
            implies(u < 1, t*(1 - u) <= 1),
            implies(and_(u >= 0, u <= 1), t <= 1 + u + u**2/2 + u**3/6 + u**4/8)]


def _log_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    # valid for u > 0, which the reality of log(u) requires
    return [u > 0, t <= u - 1, u*t >= u - 1,
            implies(u >= 1, t*(u + 1) >= 2*(u - 1)),
            implies(u <= 1, t*(u + 1) <= 2*(u - 1))]


def _sin_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= -1, t <= 1,
            implies(u >= 0, t <= u), implies(u <= 0, t >= u),
            implies(u >= 0, t >= u - u**3/6), implies(u <= 0, t <= u - u**3/6),
            implies(u >= 0, t <= u - u**3/6 + u**5/120),
            implies(u <= 0, t >= u - u**3/6 + u**5/120)]


def _cos_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= -1, t <= 1, t >= 1 - u**2/2, t <= 1 - u**2/2 + u**4/24]


def _atan_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > -Rational(8, 5), t < Rational(8, 5),
            implies(u >= 0, and_(t >= 0, t <= u, t >= u - u**3/3, t*(1 + u**2) >= u)),
            implies(u <= 0, and_(t <= 0, t >= u, t <= u - u**3/3, t*(1 + u**2) <= u))]


def _tanh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t > -1, t < 1,
            implies(u >= 0, and_(t >= 0, t <= u, t >= u - u**3/3)),
            implies(u <= 0, and_(t <= 0, t >= u, t <= u - u**3/3))]


def _sinh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [implies(u >= 0, and_(t >= u, t >= u + u**3/6)),
            implies(u <= 0, and_(t <= u, t <= u + u**3/6))]


def _cosh_bounds(t: Symbol, u: Expr) -> list[Boolean]:
    return [t >= 1, t >= 1 + u**2/2]


_RULES: Final[list[tuple[type[Expr], BoundRule]]] = [
    (Exp, _exp_bounds), (Log, _log_bounds), (Sin, _sin_bounds), (Cos, _cos_bounds),
    (Atan, _atan_bounds), (Tanh, _tanh_bounds), (Sinh, _sinh_bounds), (Cosh, _cosh_bounds),
]


def _rule(e: Expr) -> Optional[BoundRule]:
    for cls, rule in _RULES:
        if type(e) is cls:
            return rule
    return None


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
    requirements : list of Boolean
        The conditions under which the replaced applications are real
        (``u > 0`` for ``log(u)``, ``u >= 0`` for a root), which the
        facts must imply for the abstraction to speak about every point
        they allow.
    """

    __slots__ = ('formula', 'constraints', 'variables', 'requirements')

    def __init__(self, formula: Boolean, constraints: list[Boolean], variables: list[Symbol],
                 requirements: Optional[list[Boolean]] = None) -> None:
        self.formula: Final[Boolean] = formula
        self.constraints: Final[list[Boolean]] = list(constraints)
        self.variables: Final[list[Symbol]] = list(variables)
        self.requirements: Final[list[Boolean]] = list(requirements or [])

    def __repr__(self) -> str:
        return "Abstraction(%s, [%s])" % (self.formula, ", ".join(str(c) for c in self.constraints))


class _Abstractor:
    def __init__(self, real: set[Symbol]) -> None:
        self.real: Final[set[Symbol]] = real
        self.constraints: Final[list[Boolean]] = []
        self.requirements: Final[list[Boolean]] = []
        self.variables: Final[list[Symbol]] = []
        self.known: Final[dict[Expr, Symbol]] = {}

    def _fresh(self, name: str) -> Symbol:
        t = Dummy(name, real=True)
        self.variables.append(t)
        return t

    def _polynomial(self, e: Expr) -> bool:
        from sympy_extras.polys.polyerrors import BasePolynomialError
        from sympy_extras.polys.polytools import Poly
        symbols = e.free_symbols
        if not symbols:
            return e.is_rational is True
        if not (symbols <= (self.real | set(self.variables))):
            return False
        ordered = sorted(symbols, key=lambda s: s.sort_key())
        if not e.is_polynomial(*ordered):
            return False
        try:
            p = Poly(e, *ordered)
        except BasePolynomialError:
            return False
        return all(c.is_rational for c in p.coeffs())

    def expression(self, e: Expr) -> Optional[Expr]:
        """``e`` with its function applications replaced, bottom-up."""
        if e in self.known:
            return self.known[e]
        rule = _rule(e)
        if rule is not None or isinstance(e, Abs):
            argument = e.args[0]
            if not isinstance(argument, Expr):
                return None
            inner = self.expression(argument)
            if inner is None or not self._polynomial(inner):
                return None
            t = self._fresh('t')
            if rule is None:
                self.constraints.extend([t >= 0, or_(eq(t, inner), eq(t, -inner))])
            else:
                if isinstance(e, Log):
                    self.requirements.append(inner > 0)
                self.constraints.extend(rule(t, inner))
            self.known[e] = t
            return t
        if isinstance(e, Pow) and isinstance(e.exp, Rational) and not e.exp.is_integer:
            inner = self.expression(e.base)
            if inner is None or not self._polynomial(inner):
                return None
            p, q = e.exp.p, e.exp.q
            t = self._fresh('t')
            # a real root of a nonnegative base (the reality of the power)
            self.requirements.append(inner >= 0 if p > 0 else inner > 0)
            self.constraints.extend([inner >= 0, t >= 0, eq(t**q, inner**p)] if p > 0
                                    else [inner > 0, t > 0, eq(t**q*inner**(-p), 1)])
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
            if not changed:
                return e
            rebuilt = e._evaluated(tuple(new_args))
            return rebuilt if isinstance(rebuilt, Expr) else None
        return e

    def formula(self, f: Boolean) -> Optional[Boolean]:
        if isinstance(f, Relational):
            lhs, rhs = self.expression(f.lhs), self.expression(f.rhs)
            if lhs is None or rhs is None:
                return None
            return to_boolean(f._evaluated((lhs, rhs)))
        if isinstance(f, (And, Or, Not, Implies, Equivalent, Xor, ITE)):
            parts: list[Boolean] = []
            for a in f.args:
                b = self.formula(a)
                if b is None:
                    return None
                parts.append(b)
            return to_boolean(f._evaluated(tuple(parts)))
        if isinstance(f, BooleanAtom):
            return f
        # memberships and predicates are not polynomial statements
        return None


def polynomial_abstraction(formula: Boolean, real: set[Symbol], max_variables: int = 3) -> Optional[Abstraction]:
    """Replace the elementary functions of a formula by variables with
    polynomial constraints.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, sqrt
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.assumptions.bounds import polynomial_abstraction
    >>> x = Symbol('x')
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
    return Abstraction(result, abstractor.constraints, abstractor.variables, abstractor.requirements)
