"""The SymPy face of the transcendental Risch algorithm
(:mod:`sympy_extras.integrals.risch.risch`, ported from SymPy with Aaron
Meurer's extensions, native on the typed core): ``risch_integrate``,
``risch_antiderivative`` and ``is_nonelementary`` on SymPy objects, and the
rational solutions of the Risch differential equation over ``QQ(x)``
(which the native :mod:`sympy_extras.integrals.trager` solves on the core
itself).

The functions convert their arguments to the core
(:mod:`sympy_extras.interop`), call the native ones and convert the
results back; nothing of the method stays on SymPy. A proven
nonelementary integral is :class:`NonElementaryIntegral`, a SymPy
``Integral``.

Examples
========

>>> from sympy import symbols, tan, exp
>>> from sympy_extras.integrals.risch.risch_sympy import risch_antiderivative, risch_integrate
>>> x = symbols('x')
>>> risch_antiderivative(tan(x)**2, x)
-x + tan(x)
>>> risch_integrate(exp(-x**2), x)
Integral(exp(-x**2), x)
"""
from __future__ import annotations

from typing import Literal, Optional, Union, overload

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._timeout import attempt
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.core.add import add
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy
from sympy_extras.settings import settings
from . import risch as native

__all__ = ['risch_integrate', 'risch_antiderivative', 'is_nonelementary', 'NonElementaryIntegral',
           'rational_risch_de']


class NonElementaryIntegral(Integral):
    """A SymPy ``Integral`` which the Risch algorithm has proven
    nonelementary (the SymPy form of
    :class:`sympy_extras.integrals.risch.risch.NonElementaryIntegral`).

    >>> from sympy import symbols, exp
    >>> from sympy_extras.integrals.risch.risch_sympy import risch_integrate, NonElementaryIntegral
    >>> x = symbols('x')
    >>> isinstance(risch_integrate(exp(x**2), x), NonElementaryIntegral)
    True
    """


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


@overload
def risch_integrate(f: ExprLike, x: Symbol, handle_first: str = 'log', separate_integral: Literal[False] = False,
                    rewrite_complex: Optional[bool] = None, conds: str = 'piecewise') -> Expr: ...
@overload
def risch_integrate(f: ExprLike, x: Symbol, handle_first: str = 'log', *, separate_integral: Literal[True],
                    rewrite_complex: Optional[bool] = None, conds: str = 'piecewise') -> tuple[Expr, Expr]: ...


def risch_integrate(f: ExprLike, x: Symbol, handle_first: str = 'log', separate_integral: bool = False,
                    rewrite_complex: Optional[bool] = None, conds: str = 'piecewise'
                    ) -> Union[Expr, tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.risch.risch.risch_integrate` on SymPy
    objects (SymPy's ``risch_integrate``): the antiderivative, a proven
    nonelementary part as a :class:`NonElementaryIntegral`; with
    ``separate_integral`` the pair ``(elementary, nonelementary)``.
    ``NotImplementedError`` when the algorithm does not apply; no time
    limit (see :func:`risch_antiderivative`).

    >>> from sympy import symbols, log
    >>> from sympy_extras.integrals.risch.risch_sympy import risch_integrate
    >>> x = symbols('x')
    >>> risch_integrate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x)), x, separate_integral=True)
    (-log(-x + log(x))/2 + log(x + log(x))/2, Integral(1/log(x), x))
    """
    ans, i = native.risch_integrate_separate(core_expr(as_expr(f)), _core_symbol(x), handle_first=handle_first,
                                             rewrite_complex=rewrite_complex, conds=conds)
    if separate_integral:
        return sympy_expr(ans), sympy_expr(i)
    return sympy_expr(add(ans, i))


def _run(f: Expr, x: Symbol) -> Optional[Expr]:
    """``risch_integrate`` under the time limit; ``None`` when it gives up
    (``NotImplementedError``, ``ValueError``, the time)."""
    fc = core_expr(f)
    xc = _core_symbol(x)
    found = attempt(lambda: native.risch_integrate(fc, xc), settings.timeout, native=True)
    return None if found is None else sympy_expr(found)


def risch_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """An elementary antiderivative of ``f`` with respect to ``x`` by the
    transcendental Risch algorithm, or ``None`` when ``f`` has none (the
    algorithm proves it) or when the algorithm does not apply (an
    algebraic function, an unimplemented case, the time limit).

    Examples
    ========

    >>> from sympy import symbols, tan, exp, log, sin
    >>> from sympy_extras.integrals.risch.risch_sympy import risch_antiderivative
    >>> x = symbols('x')
    >>> risch_antiderivative(tan(x), x)
    log(tan(x)**2 + 1)/2
    >>> risch_antiderivative(1/(x*(log(x)**2 + 1)), x)
    atan(log(x))
    >>> risch_antiderivative(sin(x)/x, x) is None
    True
    """
    f_ = as_expr(f)
    found = _run(f_, x)
    if found is None or found.has(Integral):
        return None
    return found


def is_nonelementary(f: ExprLike, x: Symbol) -> Optional[bool]:
    """Whether the Risch algorithm proves that ``f`` has no elementary
    antiderivative (``True``), finds one (``False``), or cannot tell
    (``None``: an unimplemented case, a non-transcendental tower, the
    time limit).

    Examples
    ========

    >>> from sympy import symbols, exp, log
    >>> from sympy_extras.integrals.risch.risch_sympy import is_nonelementary
    >>> x = symbols('x')
    >>> is_nonelementary(exp(x)/x, x)
    True
    >>> is_nonelementary(exp(x)*x, x)
    False
    """
    found = _run(as_expr(f), x)
    if found is None:
        return None
    if found.has(NonElementaryIntegral):
        return True
    if found.has(Integral):
        return None
    return False


def rational_risch_de(f: ExprLike, g: ExprLike, x: Symbol) -> Optional[Expr]:
    """A rational function ``y`` of ``x`` with ``y' + f*y == g``, for
    ``f``, ``g`` rational in ``x`` (the Risch differential equation over
    ``QQ(x)``, :func:`sympy_extras.integrals.risch.rde.rischDE`); ``None``
    when there is none or the algorithm gives up.

    >>> from sympy import symbols
    >>> from sympy_extras.integrals.risch.risch_sympy import rational_risch_de
    >>> x = symbols('x')
    >>> rational_risch_de(1/x, 2, x)
    x
    """
    from .tower import RationalFunction, TowerPoly
    from .rde import rischDE
    xc = _core_symbol(x)
    fe = RationalFunction.from_expr(core_expr(as_expr(f)))
    ge = RationalFunction.from_expr(core_expr(as_expr(g)))

    def solve() -> CoreExpr:
        DE = native.DifferentialExtension.manual([TowerPoly.one(xc)])
        fa, fd = native.frac_in(fe, xc)
        ga, gd = native.frac_in(ge, xc)
        try:
            ya, yd = rischDE(fa, fd, ga, gd, DE)
        except native.NonElementaryIntegralException:
            raise NotImplementedError('no rational solution')
        return (ya.as_element() / yd.as_element()).as_expr()

    found = attempt(solve, settings.timeout, native=True)
    return None if found is None else sympy_expr(found)
