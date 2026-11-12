"""The SymPy face of the Almkvist–Zeilberger algorithm
(:mod:`sympy_extras.integrals.telescoping`), for the integrators which
are still written on SymPy (:mod:`.definite`) and the old interface.

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`, the telescopers as :class:`DifferentialTelescoper`
on SymPy objects). The steps of the native modules
:mod:`.telescoping`, :mod:`.reduction` and :mod:`.dfinite` which stay on
SymPy are here too (the functions ending in ``_of_core``, imported by the
native modules when they are needed): the values of the integral at a
point by :func:`~sympy_extras.integrals.definite.definite_integral` and
its numerical check ``verify_numerically``, not ported yet
(:func:`definite_integral_of_core`, :func:`verify_numerically_of_core`).

Examples
========

>>> from sympy import symbols, exp, oo, cos
>>> from sympy_extras.integrals.telescoping_sympy import almkvist_zeilberger, holonomic_integral
>>> x = symbols('x')
>>> t = symbols('t', positive=True)
>>> almkvist_zeilberger(exp(-x**2 - t**2/x**2), x, t)
DifferentialTelescoper([-4, 0, 1], 2/x)
>>> holonomic_integral(exp(-x**2 - t**2/x**2), x, 0, oo, t)
ConditionalValue(sqrt(pi)*exp(-2*t)/2)
>>> holonomic_integral(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
ConditionalValue(sqrt(pi)*exp(-t**2)/2)
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike
from sympy_extras.assumptions import ask as native_ask
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.logic import Boolean as CoreBoolean, to_boolean
from sympy_extras.core.relational import Eq as CoreEq
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import conditions as native_conditions
from . import telescoping as native
from .conditions_sympy import ConditionalValue

__all__ = ['DifferentialTelescoper', 'logarithmic_derivative', 'is_hyperexponential', 'almkvist_zeilberger',
           'holonomic_ode', 'holonomic_integral', 'definite_integral_of_core', 'verify_numerically_of_core']


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


def _sympy_eq(e: CoreEq) -> Eq:
    converted = to_sympy(e)
    if not isinstance(converted, Eq):
        raise TypeError("%s is not an equation" % (converted,))
    return converted


def _optional_eq(e: Optional[CoreEq]) -> Optional[Eq]:
    return None if e is None else _sympy_eq(e)


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


class DifferentialTelescoper:
    """:class:`sympy_extras.integrals.telescoping.DifferentialTelescoper`
    on SymPy objects: the coefficients ``a_0(t), ..., a_J(t)`` and the
    certificate ``R`` with ``sum_j a_j d^j F/dt^j = d(R F)/dx``.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.telescoping_sympy import almkvist_zeilberger
    >>> x, t = symbols('x t')
    >>> telescoper = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    >>> telescoper.order, telescoper.operator()
    (1, t*Derivative(I(t), t) + I(t))
    """

    def __init__(self, term: Expr, x: Symbol, t: Symbol, coefficients: Sequence[Expr], certificate: Expr) -> None:
        self.term: Final[Expr] = term
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t
        self.coefficients: Final[list[Expr]] = list(coefficients)
        self.certificate: Final[Expr] = certificate

    @staticmethod
    def from_core(found: native.DifferentialTelescoper) -> DifferentialTelescoper:
        """A telescoper of the core on SymPy objects."""
        return DifferentialTelescoper(sympy_expr(found.term), _sympy_symbol(found.x), _sympy_symbol(found.t),
                                      [sympy_expr(c) for c in found.coefficients], sympy_expr(found.certificate))

    def to_core(self) -> native.DifferentialTelescoper:
        """The same telescoper on the objects of the core."""
        return native.DifferentialTelescoper(core_expr(self.term), _core_symbol(self.x), _core_symbol(self.t),
                                             [core_expr(c) for c in self.coefficients], core_expr(self.certificate))

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def operator(self, name: str = 'I') -> Expr:
        """``sum_j a_j(t) I^(j)(t)`` in the function ``I(t)``."""
        return sympy_expr(self.to_core().operator(name))

    def check(self) -> bool:
        """Verify ``sum_j a_j d^j F/dt^j = d(R F)/dx`` by simplification."""
        return self.to_core().check()

    def __repr__(self) -> str:
        return "DifferentialTelescoper(%s, %s)" % (self.coefficients, self.certificate)


def _telescoper(found: Optional[native.DifferentialTelescoper]) -> Optional[DifferentialTelescoper]:
    return None if found is None else DifferentialTelescoper.from_core(found)


def logarithmic_derivative(F: Expr, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.telescoping.logarithmic_derivative` on
    SymPy objects: ``diff(F, x)/F`` as a rational function of ``x``, or
    ``None``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.telescoping_sympy import logarithmic_derivative
    >>> x, t = symbols('x t')
    >>> logarithmic_derivative(x**t*exp(-x**2), x)
    (t - 2*x**2)/x
    >>> logarithmic_derivative(exp(exp(x)), x) is None
    True
    """
    found = native.logarithmic_derivative(core_expr(F), _core_symbol(x))
    return None if found is None else sympy_expr(found)


def is_hyperexponential(F: ExprLike, x: Symbol, t: Symbol) -> bool:
    """:func:`sympy_extras.integrals.telescoping.is_hyperexponential` on
    SymPy objects.

    >>> from sympy import symbols, exp, sin
    >>> from sympy_extras.integrals.telescoping_sympy import is_hyperexponential
    >>> x, t = symbols('x t')
    >>> is_hyperexponential(exp(-x*t)*x**3/(x**2 + t**2), x, t)
    True
    >>> is_hyperexponential(exp(-x*t)*sin(x), x, t)
    False
    """
    return native.is_hyperexponential(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t))


def almkvist_zeilberger(F: ExprLike, x: Symbol, t: Symbol, max_order: int = 4,
                        min_order: int = 0) -> Optional[DifferentialTelescoper]:
    """:func:`sympy_extras.integrals.telescoping.almkvist_zeilberger` on
    SymPy objects: the telescoper of a hyperexponential ``F(x, t)`` (or of
    a sum of such terms sharing the same equation), ``None`` when there is
    none of order at most ``max_order``.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.telescoping_sympy import almkvist_zeilberger
    >>> x, t = symbols('x t')
    >>> telescoper = almkvist_zeilberger(1/(x**2 + t**2), x, t)
    >>> telescoper.coefficients, telescoper.certificate
    ([1, t], -x)
    >>> telescoper.check()
    True
    >>> almkvist_zeilberger(exp(-x**2)*exp(2*x*t), x, t).coefficients
    [-2*t, 1]
    """
    return _telescoper(native.almkvist_zeilberger(core_expr(sympify(F)), _core_symbol(x), _core_symbol(t),
                                                  max_order, min_order))


def holonomic_ode(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                  assumptions: Assumptions = None, name: str = 'I') -> Optional[Eq]:
    """:func:`sympy_extras.integrals.telescoping.holonomic_ode` on SymPy
    objects: the equation ``sum_j a_j(t) I^(j)(t) = [R F]_a^b`` of
    ``I(t) = Integral(F, (x, a, b))``.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.telescoping_sympy import holonomic_ode
    >>> x, t = symbols('x t')
    >>> holonomic_ode(exp(-x**2)*exp(2*x*t), x, -oo, oo, t)
    Eq(-2*t*I(t) + Derivative(I(t), t), 0)
    """
    return _optional_eq(native.holonomic_ode(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                             core_expr(sympify(b)), _core_symbol(t), core_assumptions(assumptions),
                                             name))


def holonomic_integral(F: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, t: Symbol,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.telescoping.holonomic_integral` on
    SymPy objects: ``Integral(F, (x, a, b))`` as a function of ``t`` by
    the holonomic method, ``None`` when a step fails.

    >>> from sympy import symbols, exp, oo, cos
    >>> from sympy_extras.integrals.telescoping_sympy import holonomic_integral
    >>> x = symbols('x')
    >>> t = symbols('t', positive=True)
    >>> holonomic_integral(exp(-x**2)*cos(2*t*x), x, 0, oo, t)
    ConditionalValue(sqrt(pi)*exp(-t**2)/2)
    >>> holonomic_integral(1/(x**2 + t**2)**2, x, -oo, oo, t)
    ConditionalValue(pi/(2*t**3))
    """
    return _conditional(native.holonomic_integral(core_expr(sympify(F)), _core_symbol(x), core_expr(sympify(a)),
                                                  core_expr(sympify(b)), _core_symbol(t),
                                                  core_assumptions(assumptions)))


# ---------------------------------------------------------------------------
# the steps of the native modules which stay on SymPy

def _sympy_assumptions(assumptions: native_ask.Assumptions) -> Assumptions:
    if assumptions is None:
        return None
    if isinstance(assumptions, (CoreBoolean, bool)):
        return sympy_boolean(to_boolean(assumptions))
    return [sympy_boolean(to_boolean(a)) for a in assumptions]


def definite_integral_of_core(f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                              assumptions: native_ask.Assumptions, special_values: bool,
                              seconds: Optional[float]) -> Optional[CoreExpr]:
    """:func:`~sympy_extras.integrals.definite.definite_integral` of
    ``Integral(f, (x, a, b))`` within ``seconds``, on core expressions
    (not ported yet); ``None`` when it does not finish or its value
    cannot be converted."""
    from .definite import definite_integral
    integrand, variable, lower, upper = sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b)
    facts = _sympy_assumptions(assumptions)
    found = attempt(lambda: definite_integral(integrand, (variable, lower, upper), facts,
                                              special_values=special_values), seconds)
    if found is None:
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None


def verify_numerically_of_core(value: CoreExpr, f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                               assumptions: native_ask.Assumptions) -> Optional[bool]:
    """:func:`sympy_extras.integrals.definite.verify_numerically` of a
    value of the native modules: ``False`` when it disagrees with the
    quadratures at samples of the parameters (each branch of a
    ``Piecewise`` at a sample of its region, the isolated values of the
    parameters), ``True`` when it agrees, ``None`` when nothing could be
    checked."""
    from .definite import verify_numerically
    return verify_numerically(sympy_expr(value), sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b),
                              _sympy_assumptions(assumptions))


def _sympy_function(f: CoreAppliedUndef) -> AppliedUndef:
    converted = to_sympy(f)
    if not isinstance(converted, AppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (converted,))
    return converted


