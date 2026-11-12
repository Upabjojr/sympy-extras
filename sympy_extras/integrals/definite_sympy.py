"""The SymPy face of the definite integration driver
(:mod:`sympy_extras.integrals.definite`): :func:`definite_integral`,
:func:`conditional_integral` and :func:`verify_numerically` on SymPy
objects.

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`; the unevaluated
:class:`~sympy_extras.integrals.integral.Integral` becomes SymPy's
``Integral``). The steps of the native module which stay on SymPy are
here too, on core expressions, imported by it at run time: SymPy's
``integrate`` and its Meijer G-function route, the last of the methods
(:func:`integrate_of_core`).

Examples
========

>>> from sympy import symbols, exp, sin, log, sqrt, Abs, oo, pi
>>> from sympy_extras.integrals.definite_sympy import definite_integral
>>> x, k = symbols('x k')
>>> a, s = symbols('a s', positive=True)
>>> definite_integral(x**k/(x + 3), (x, 0, oo))
Piecewise((-3**k*pi/sin(pi*k), (k > -1) & (k < 0)), (Integral(x**k/(x + 3), (x, 0, oo)), True))
>>> definite_integral(x**k/(x + 3), (x, 0, oo), (k > -1) & (k < 0))
-3**k*pi/sin(pi*k)
>>> definite_integral(exp(-s*x)*sin(a*x)/x, (x, 0, oo))
atan(a/s)
>>> definite_integral(Abs(x - 1)/sqrt(x), (x, 0, 2))
2*(4 - sqrt(2))/3
>>> definite_integral(log(x)**2/(1 + x**2), (x, 0, oo))
pi**3/8
>>> definite_integral(sqrt(sin(x)), (x, 0, pi/2))
2*sqrt(pi)*gamma(3/4)/gamma(1/4)
>>> definite_integral(1/(x*sqrt((x + 1)**2)), (x, -oo, -2))
-log(2)
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.integrals.integrals import integrate

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import conditions as native_conditions
from . import definite as native
from .conditions_sympy import ConditionalValue

__all__ = ['definite_integral', 'conditional_integral', 'verify_numerically', 'Limits', 'integrate_of_core']

#: ``(x, a, b)``
Limits = tuple[Symbol, ExprLike, ExprLike]


def _core(e: ExprLike) -> CoreExpr:
    return core_expr(sympify(e))


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def definite_integral(f: ExprLike, limits: Limits, assumptions: Assumptions = None,
                      conds: str = 'piecewise', recognize: bool = False,
                      principal_value: bool = False, finite_part: bool = False,
                      numeric: bool = False, digits: int = 15, regularize: bool = False,
                      summability: Optional[str] = None, special_values: bool = True) -> Expr:
    """:func:`sympy_extras.integrals.definite.definite_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` under assumptions on the
    parameters (statements, ``a > 0``), with the conditions the
    assumptions do not settle in a ``Piecewise`` whose other branch is
    the unevaluated ``Integral`` (``conds='piecewise'``) or assumed to
    hold (``conds='none'``); the options are those of the native
    function.

    >>> from sympy import symbols, exp, cos, oo, log, S
    >>> from sympy_extras.assumptions.facts_sympy import element
    >>> from sympy_extras.integrals.definite_sympy import definite_integral
    >>> x, a, b = symbols('x a b')
    >>> definite_integral(exp(-a*x)*cos(b*x), (x, 0, oo), (a > 0) & element(b, S.Reals))
    a/(a**2 + b**2)
    >>> definite_integral(exp(-a*x)*cos(b*x), (x, 0, oo), element(a, S.Reals) & element(b, S.Reals))
    Piecewise((a/(a**2 + b**2), a > 0), (Integral(exp(-a*x)*cos(b*x), (x, 0, oo)), True))
    >>> definite_integral(x**a*log(x), (x, 0, 1), a > -1)
    -1/(a + 1)**2
    >>> definite_integral(log(1 - x)/x, (x, 0, 1))
    -pi**2/6
    >>> definite_integral(1/x, (x, 0, 1))
    oo
    """
    x, a, b = limits
    found = native.definite_integral(_core(f), (_core_symbol(x), _core(a), _core(b)), core_assumptions(assumptions),
                                     conds, recognize, principal_value, finite_part, numeric, digits, regularize,
                                     summability, special_values)
    return sympy_expr(found)


def conditional_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                         assumptions: Assumptions = None,
                         principal_value: bool = False,
                         finite_part: bool = False,
                         regularize: bool = False) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.definite.conditional_integral` on
    SymPy objects: the value of ``Integral(f, (x, a, b))`` with the
    condition on the parameters under which it holds, or ``None``.

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.definite_sympy import conditional_integral
    >>> x, a = symbols('x a')
    >>> conditional_integral(exp(-a*x), x, 0, oo)
    ConditionalValue(1/a, Abs(arg(a)) < pi/2)
    """
    return _conditional(native.conditional_integral(_core(f), _core_symbol(x), _core(a), _core(b),
                                                    core_assumptions(assumptions), principal_value, finite_part,
                                                    regularize))


def verify_numerically(value: ExprLike, f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None, samples: int = 2,
                       inside: Sequence[ExprLike] = ()) -> Optional[bool]:
    """:func:`sympy_extras.integrals.definite.verify_numerically` on SymPy
    objects: whether ``value`` agrees with a numerical quadrature of
    ``Integral(f, (x, a, b))`` at random values of the parameters
    satisfying the assumptions (``False`` when they disagree, ``None``
    when nothing could be checked).

    >>> from sympy import symbols, exp, oo
    >>> from sympy_extras.integrals.definite_sympy import verify_numerically
    >>> x, a = symbols('x a', positive=True)
    >>> verify_numerically(1/a, exp(-a*x), x, 0, oo), verify_numerically(2/a, exp(-a*x), x, 0, oo)
    (True, False)
    """
    return native.verify_numerically(_core(value), _core(f), _core_symbol(x), _core(a), _core(b),
                                     core_assumptions(assumptions), samples, [_core(p) for p in inside])


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def integrate_of_core(f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr, meijerg: bool,
                      seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``integrate(f, (x, a, b))`` (with ``meijerg=True`` its Meijer
    G-function route alone) within ``seconds``, on core expressions: the
    last of the methods of :mod:`.definite`. ``None`` when it does not
    finish, gives up, or its value cannot be converted (a ``Piecewise``
    with SymPy's unevaluated ``Integral`` comes back wrapped)."""
    integrand, variable, lower, upper = sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b)
    keywords = {'meijerg': True} if meijerg else {}
    try:
        value = attempt(lambda: as_expr(integrate(integrand, (variable, lower, upper), **keywords)), seconds)
    except AttributeError:
        # SymPy 1.14: meijerg._eval_evalf raises AttributeError
        # ('NoneType' object has no attribute 'has') on some arguments,
        # e.g. integrate(exp(-x)*sin(x)/x**2, (x, 0, oo))
        return None
    if value is None:
        return None
    try:
        return core_expr(value)
    except (TypeError, ValueError, NotImplementedError):
        return None

