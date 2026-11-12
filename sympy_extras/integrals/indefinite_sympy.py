"""The SymPy face of the verified indefinite integration
(:mod:`sympy_extras.integrals.indefinite`), for the integrators which are
still written on SymPy (:mod:`.definite`) and the users of SymPy objects.

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back. The methods of the native
module which stay on SymPy are here too, imported by it at run time and
tried last, when every native method has failed: SymPy's
``manualintegrate`` (:func:`manualintegrate_of_core`), its Meijer G-function route
(:func:`meijerint_of_core`) and its ``integrate``
(:func:`integrate_of_core`).

Examples
========

>>> from sympy import symbols, sqrt, exp, Integral
>>> from sympy_extras.integrals.indefinite_sympy import indefinite_integral
>>> x = symbols('x')
>>> indefinite_integral(sqrt(1 - x**2), x)
x*sqrt(1 - x**2)/2 + asin(x)/2
>>> indefinite_integral(x/(x**2 + 1), x)
log(x**2 + 1)/2
>>> indefinite_integral(exp(-x**2), x)
sqrt(pi)*erf(x)/2
>>> indefinite_integral(x**x, x)
Integral(x**x, x)
"""
from __future__ import annotations

from typing import Callable, Optional

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import indefinite as native

__all__ = ['indefinite_integral', 'verified_antiderivative', 'is_antiderivative', 'real_form', 'conjugate_logarithms',
           'manualintegrate_of_core', 'meijerint_of_core', 'integrate_of_core']


def _core(e: ExprLike) -> CoreExpr:
    return core_expr(sympify(e))


def is_antiderivative(F: ExprLike, f: ExprLike, x: Symbol, assumptions: Assumptions = None) -> Optional[bool]:
    """:func:`sympy_extras.integrals.indefinite.is_antiderivative` on SymPy
    objects: whether ``F`` is an antiderivative of ``f`` (``True``),
    is not (``False``) or nothing could be decided (``None``).

    >>> from sympy import symbols, log, sin
    >>> from sympy_extras.integrals.indefinite_sympy import is_antiderivative
    >>> x = symbols('x')
    >>> is_antiderivative(log(x**2 + 1)/2, x/(x**2 + 1), x)
    True
    >>> is_antiderivative(sin(x), sin(x), x)
    False
    """
    return native.is_antiderivative(_core(F), _core(f), _core_symbol(x), core_assumptions(assumptions))


def real_form(F: Expr, f: Expr, x: Symbol, assumptions: Assumptions = None) -> Expr:
    """:func:`sympy_extras.integrals.indefinite.real_form` on SymPy objects:
    ``F`` rewritten without ``I`` and polar numbers when a rewriting which
    is still an antiderivative of ``f`` is found; ``F`` itself otherwise.

    >>> from sympy import symbols, log, I
    >>> from sympy_extras.integrals.indefinite_sympy import real_form
    >>> x = symbols('x')
    >>> real_form(-I*log(x - I)/2 + I*log(x + I)/2, 1/(x**2 + 1), x)
    atan(x)
    """
    return sympy_expr(native.real_form(_core(F), _core(f), _core_symbol(x), core_assumptions(assumptions)))


def conjugate_logarithms(F: Expr, x: Symbol) -> Expr:
    """:func:`sympy_extras.integrals.indefinite.conjugate_logarithms` on
    SymPy objects: the logarithms of conjugates of ``F`` as arctangents.

    >>> from sympy import symbols, log, I
    >>> from sympy_extras.integrals.indefinite_sympy import conjugate_logarithms
    >>> x = symbols('x')
    >>> conjugate_logarithms(-I*log(x - I)/2 + I*log(x + I)/2, x)
    atan(x)
    """
    return sympy_expr(native.conjugate_logarithms(_core(F), _core_symbol(x)))


def verified_antiderivative(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                            methods: Optional[list[str]] = None) -> Optional[tuple[Expr, str]]:
    """:func:`sympy_extras.integrals.indefinite.verified_antiderivative` on
    SymPy objects: an antiderivative of ``f`` which checks, with the name
    of the method which found it; ``None`` when no method gives one.

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.indefinite_sympy import verified_antiderivative
    >>> x = symbols('x')
    >>> verified_antiderivative(x*exp(x), x)
    (x*exp(x) - exp(x), 'exponential')
    """
    found = native.verified_antiderivative(_core(f), _core_symbol(x), core_assumptions(assumptions), methods)
    return None if found is None else (sympy_expr(found[0]), found[1])


def indefinite_integral(f: ExprLike, x: Symbol, assumptions: Assumptions = None,
                        special_values: bool = True) -> Expr:
    """:func:`sympy_extras.integrals.indefinite.indefinite_integral` on
    SymPy objects: ``Integral(f, x)`` by the verified methods, SymPy's
    unevaluated ``Integral`` when none gives an antiderivative which
    checks; the isolated values of the parameters with ``special_values``
    in the form of SymPy's ``integrate``.

    >>> from sympy import symbols, log, tan, exp
    >>> from sympy_extras.integrals.indefinite_sympy import indefinite_integral
    >>> x, k = symbols('x k')
    >>> indefinite_integral(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> indefinite_integral(tan(x)**3, x)
    log(cos(x)) + tan(x)**2/2
    >>> indefinite_integral(x*exp(k*x), x)
    Piecewise((x*exp(k*x)/k - exp(k*x)/k**2, Ne(k, 0)), (x**2/2, True))
    """
    return sympy_expr(native.indefinite_integral(_core(f), _core_symbol(x), core_assumptions(assumptions),
                                                 special_values))


# ---------------------------------------------------------------------------
# the methods of the native module which stay on SymPy

def _back(found: Optional[Expr]) -> Optional[CoreExpr]:
    """A SymPy antiderivative as a core expression; ``None`` when there is
    none, it has an ``Integral`` or it cannot be converted."""
    from sympy.integrals.integrals import Integral
    if found is None or found.has(Integral):
        return None
    try:
        return core_expr(found)
    except (TypeError, ValueError, NotImplementedError):
        return None


def _run(route: Callable[[Expr, Symbol], Expr], f: CoreExpr, x: CoreSymbol,
         seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``route(f, x)`` within ``seconds``, on core expressions;
    ``None`` when it does not finish, leaves an ``Integral`` or its
    antiderivative cannot be converted."""
    integrand, variable = sympy_expr(f), _sympy_symbol(x)
    return _back(attempt(lambda: as_expr(route(integrand, variable)), seconds))


def _manualintegrate(f: Expr, x: Symbol) -> Expr:
    from sympy.integrals.manualintegrate import manualintegrate
    return as_expr(manualintegrate(f, x))


def _meijerint(f: Expr, x: Symbol) -> Expr:
    from sympy.integrals.integrals import integrate
    return as_expr(integrate(f, x, meijerg=True, risch=False))


def _integrate(f: Expr, x: Symbol) -> Expr:
    from sympy.integrals.integrals import integrate
    return as_expr(integrate(f, x))


def manualintegrate_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``manualintegrate(f, x)`` within ``seconds``, on core
    expressions; ``None`` as :func:`heurisch_of_core`."""
    return _run(_manualintegrate, f, x, seconds)


def meijerint_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``integrate(f, x, meijerg=True, risch=False)`` within
    ``seconds``, on core expressions; ``None`` as :func:`_run`."""
    return _run(_meijerint, f, x, seconds)


def integrate_of_core(f: CoreExpr, x: CoreSymbol, seconds: Optional[float]) -> Optional[CoreExpr]:
    """SymPy's ``integrate(f, x)`` within ``seconds``, on core expressions;
    ``None`` as :func:`_run`."""
    return _run(_integrate, f, x, seconds)
