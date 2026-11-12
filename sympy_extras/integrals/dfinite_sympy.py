"""The SymPy face of Chyzak's algorithm
(:mod:`sympy_extras.integrals.dfinite`), for the integrators which are
still written on SymPy (:mod:`.definite`) and the old interface.

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the annihilators, factors
and telescopers as classes on SymPy objects, the values of
:mod:`.conditions_sympy`). The step of the native module which stays on
SymPy is here too: SymPy's ``expr_to_holonomic``, for the annihilators
of whole integrands and the equations of the factors which the native
module does not write down (:func:`holonomic_of_core`).

Examples
========

>>> from sympy import symbols, exp, besselj, oo
>>> from sympy_extras.integrals.dfinite_sympy import chyzak, dfinite_ode, dfinite_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> chyzak(exp(-t*x)*besselj(0, x), x, t)
DFiniteTelescoper([t, t**2 + 1], {exp(-t*x)*besselj(0, x): t*x, -exp(-t*x)*besselj(1, x): x})
>>> dfinite_ode(exp(-t*x)*besselj(0, x), x, 0, oo, t)
Eq(t*I(t) + (t**2 + 1)*Derivative(I(t), t), 0)
>>> dfinite_integral(exp(-t*x)*besselj(0, x), x, 0, oo, t)
ConditionalValue(1/sqrt(t**2 + 1))
"""
from __future__ import annotations

from typing import Callable, Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.settings import settings

from . import dfinite as native
from .conditions_sympy import ConditionalValue
from .telescoping_sympy import _optional_eq

__all__ = ['Annihilator', 'DFiniteTelescoper', 'Factor', 'annihilators', 'normal_form', 'closure', 'chyzak',
           'dfinite_ode', 'dfinite_integral', 'holonomic_of_core']

#: a mixed derivative ``d^i/dx^i d^j/dt^j F`` by its orders
Derivative_ = tuple[int, int]
#: a combination of the basis derivatives with rational coefficients
NormalForm = dict[Derivative_, Expr]


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (converted,))
    return converted


class Annihilator:
    """:class:`sympy_extras.integrals.dfinite.Annihilator` on SymPy
    objects: ``sum_i c_i(x, t) d^i/dv^i`` with polynomial coefficients,
    ``c_r != 0``.

    Examples
    ========

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite_sympy import annihilators
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-t*x**2), x, t)
    >>> L_x
    Annihilator(x, [2*t*x, 1])
    >>> L_x.order
    1
    """

    def __init__(self, variable: Symbol, coefficients: Sequence[Expr]) -> None:
        stripped = [as_expr(c) for c in coefficients]
        while stripped and stripped[-1] == 0:
            stripped.pop()
        if len(stripped) < 2:
            raise ValueError("an annihilator has order at least one")
        self.variable: Final[Symbol] = variable
        self.coefficients: Final[list[Expr]] = stripped

    @staticmethod
    def from_core(found: native.Annihilator) -> Annihilator:
        return Annihilator(_sympy_symbol(found.variable), [sympy_expr(c) for c in found.coefficients])

    def to_core(self) -> native.Annihilator:
        return native.Annihilator(_core_symbol(self.variable), [core_expr(c) for c in self.coefficients])

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    @property
    def leading(self) -> Expr:
        return self.coefficients[-1]

    def reduction(self) -> list[Expr]:
        """The coefficients ``e_i`` with ``d^r F = sum_{i < r} e_i d^i F``."""
        return [sympy_expr(e) for e in self.to_core().reduction()]

    def apply(self, f: Expr) -> Expr:
        """``sum_i c_i d^i f/dv^i``."""
        return sympy_expr(self.to_core().apply(core_expr(f)))

    def __repr__(self) -> str:
        return "Annihilator(%s, %s)" % (self.variable, self.coefficients)


class DFiniteTelescoper:
    """:class:`sympy_extras.integrals.dfinite.DFiniteTelescoper` on SymPy
    objects: the operator ``P = sum_k a_k(t) d^k/dt^k`` and the certificate
    ``G`` (basis functions of the closure with their rational
    coefficients), ``P F = dG/dx``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite_sympy import chyzak
    >>> x, t = symbols('x t')
    >>> found = chyzak(exp(-x**2)*exp(2*t*x), x, t)
    >>> found.order, found.operator(), found.function(), found.check()
    (1, -2*t*I(t) + Derivative(I(t), t), -exp(-x**2)*exp(2*t*x), True)
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr],
                 certificate: dict[Expr, Expr]) -> None:
        self.term: Final[Expr] = term
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.coefficients: Final[list[Expr]] = [as_expr(c) for c in coefficients]
        self.certificate: Final[dict[Expr, Expr]] = dict(certificate)

    @staticmethod
    def from_core(found: native.DFiniteTelescoper) -> DFiniteTelescoper:
        return DFiniteTelescoper(sympy_expr(found.term), _sympy_symbol(found.x), _sympy_symbol(found.t),
                                 [sympy_expr(c) for c in found.coefficients],
                                 {sympy_expr(b): sympy_expr(g) for b, g in found.certificate.items()})

    def to_core(self) -> native.DFiniteTelescoper:
        return native.DFiniteTelescoper(core_expr(self.term), _core_symbol(self.x), _core_symbol(self.t),
                                        [core_expr(c) for c in self.coefficients],
                                        {core_expr(b): core_expr(g) for b, g in self.certificate.items()})

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_k a_k(t) I^(k)(t)`` in the function ``I(t)``."""
        return sympy_expr(self.to_core().operator(name))

    def function(self) -> Expr:
        """The certificate ``G`` as an expression."""
        return sympy_expr(self.to_core().function())

    def check(self) -> bool:
        """Verify ``P F = dG/dx`` by simplification."""
        return self.to_core().check()

    def __repr__(self) -> str:
        return "DFiniteTelescoper(%s, %s)" % (self.coefficients, self.certificate)


class Factor:
    """:class:`sympy_extras.integrals.dfinite.Factor` on SymPy objects: a
    D-finite factor ``f(u)`` at ``u = argument`` with the reduction
    ``f^(n)(u) = sum_{m < n} e_m(u) f^(m)(u)``."""

    def __init__(self, function: Expr, u: Symbol, argument: Expr, reduction: Sequence[Expr]) -> None:
        self.function: Final[Expr] = function
        self.u: Final[Symbol] = u
        self.argument: Final[Expr] = argument
        self.reduction: Final[list[Expr]] = [as_expr(e) for e in reduction]

    @staticmethod
    def from_core(found: native.Factor) -> Factor:
        return Factor(sympy_expr(found.function), _sympy_symbol(found.u), sympy_expr(found.argument),
                      [sympy_expr(e) for e in found.reduction])

    @property
    def order(self) -> int:
        return len(self.reduction)

    def derivative(self, m: int) -> Expr:
        """``f^(m)(argument)`` as an expression."""
        return as_expr(self.function.diff(self.u, m).subs(self.u, self.argument))

    def __repr__(self) -> str:
        return "Factor(%s, %s)" % (self.function.subs(self.u, self.argument), self.order)


def annihilators(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Annihilator, Annihilator]]:
    """:func:`sympy_extras.integrals.dfinite.annihilators` on SymPy
    objects: the annihilators ``L_x`` and ``L_t`` of ``F``, or ``None``.

    >>> from sympy import symbols, exp, besselj
    >>> from sympy_extras.integrals.dfinite_sympy import annihilators
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-x**2)*besselj(0, t*x), x, t)
    >>> L_x.order, L_t.order
    (2, 2)
    >>> annihilators(exp(exp(x))*t, x, t) is None
    True
    """
    found = native.annihilators(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t))
    if found is None:
        return None
    return Annihilator.from_core(found[0]), Annihilator.from_core(found[1])


def normal_form(L_x: Annihilator, L_t: Annihilator, x: Symbol, t: Symbol) -> Callable[[int, int], NormalForm]:
    """:func:`sympy_extras.integrals.dfinite.normal_form` on SymPy objects:
    ``(i, j) -> d^i/dx^i d^j/dt^j F`` as a combination of the basis
    derivatives.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite_sympy import annihilators, normal_form
    >>> x, t = symbols('x t')
    >>> L_x, L_t = annihilators(exp(-t*x**2), x, t)
    >>> reduce = normal_form(L_x, L_t, x, t)
    >>> reduce(2, 0)
    {(0, 0): 4*t**2*x**2 - 2*t}
    """
    reduce = native.normal_form(L_x.to_core(), L_t.to_core(), _core_symbol(x), _core_symbol(t))

    def converted(i: int, j: int) -> NormalForm:
        return {key: sympy_expr(value) for key, value in reduce(i, j).items()}

    return converted


def closure(F: ExprLike, x: Symbol, t: Symbol) -> Optional[tuple[Expr, list[Factor]]]:
    """:func:`sympy_extras.integrals.dfinite.closure` on SymPy objects: the
    integrand as a rational coefficient times D-finite factors of
    polynomial arguments, or ``None``.

    >>> from sympy import symbols, exp, besselj
    >>> from sympy_extras.integrals.dfinite_sympy import closure
    >>> x, t = symbols('x t')
    >>> closure(x*exp(-x**2)*besselj(0, t*x), x, t)
    (x, [Factor(besselj(0, t*x), 2), Factor(exp(-x**2), 1)])
    """
    found = native.closure(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t))
    if found is None:
        return None
    return sympy_expr(found[0]), [Factor.from_core(f) for f in found[1]]


def chyzak(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 3, max_degree: int = 6) -> Optional[DFiniteTelescoper]:
    """:func:`sympy_extras.integrals.dfinite.chyzak` on SymPy objects: a
    telescoper and certificate of ``F(x, t)``, or ``None``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.dfinite_sympy import chyzak
    >>> x, t = symbols('x t')
    >>> chyzak(exp(-x**2)*exp(2*t*x), x, t)
    DFiniteTelescoper([-2*t, 1], {exp(-x**2)*exp(2*t*x): -1})
    >>> chyzak(exp(-t*x**2), x, t).coefficients
    [1, 2*t]
    """
    found = native.chyzak(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t), max_order, max_degree)
    return None if found is None else DFiniteTelescoper.from_core(found)


def dfinite_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """:func:`sympy_extras.integrals.dfinite.dfinite_ode` on SymPy
    objects: the equation ``sum_k a_k(t) I^(k)(t) = [G]_a^b`` of
    ``I(t) = Integral(F, (x, a, b))``, or ``None``.

    >>> from sympy import symbols, exp, sin, oo
    >>> from sympy_extras.integrals.dfinite_sympy import dfinite_ode
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_ode(exp(-t*x)*sin(x)/x, x, 0, oo, t)
    Eq(Derivative(I(t), t), -1/(t**2 + 1))
    """
    return _optional_eq(native.dfinite_ode(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                           core_expr(sympify(b)), _core_symbol(t), core_assumptions(assumptions),
                                           name))


def dfinite_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                     assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.dfinite.dfinite_integral` on SymPy
    objects: ``Integral(F, (x, a, b))`` as a function of ``t`` by
    Chyzak's algorithm, ``None`` when a step fails.

    >>> from sympy import symbols, exp, besselj, oo
    >>> from sympy_extras.integrals.dfinite_sympy import dfinite_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> dfinite_integral(x*exp(-x**2)*besselj(0, t*x), x, 0, oo, t)
    ConditionalValue(exp(-t**2/4)/2)
    """
    found = native.dfinite_integral(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                    core_expr(sympify(b)), _core_symbol(t), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)


# ---------------------------------------------------------------------------
# the step of the native module which stays on SymPy

def holonomic_of_core(F: CoreExpr, v: CoreSymbol, parameters: Sequence[CoreSymbol]) -> Optional[list[CoreExpr]]:
    """The coefficients (lowest order first) of the annihilator of ``F`` in
    ``v`` which SymPy's ``expr_to_holonomic`` gives, with the
    ``parameters`` in the coefficient field, under the time limit of the
    settings; ``None`` when it fails or the result cannot be converted."""
    from sympy.holonomic.holonomic import expr_to_holonomic
    from sympy.polys.domains import QQ
    from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, PolynomialError
    f, variable = sympy_expr(F), _sympy_symbol(v)
    symbols = [_sympy_symbol(p) for p in parameters]
    domain = QQ.frac_field(*symbols) if symbols else QQ
    try:
        h = attempt(lambda: expr_to_holonomic(f, variable, domain=domain), settings.timeout)
    except (PolynomialError, CoercionFailed, GeneratorsError, AttributeError, KeyError, ZeroDivisionError,
            AssertionError, NotImplementedError):
        return None
    if h is None:
        return None
    operator = h.annihilator
    ring = operator.parent.base
    try:
        return [core_expr(as_expr(ring.to_sympy(p))) for p in operator.listofpoly]
    except (TypeError, ValueError, NotImplementedError):
        return None
