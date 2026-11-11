"""The SymPy face of the transcendental equations and inequalities
(:mod:`sympy_extras.solvers.transcendental`), and the steps which stay on
SymPy.

:func:`solve_transcendental` and :func:`polynomialize` take and return
SymPy objects, converted by :mod:`sympy_extras.interop` around the native
functions; a ``solver`` and a ``decide`` given on SymPy objects are
called through the conversion too. :class:`Reduction` is the native
reduction with its formula, variable and argument on SymPy objects.

The steps of the native module which stay written on SymPy are called
from it at run time (the functions ending in ``_of_core``, which take and
return objects of the core): SymPy's ``solveset`` of a relation
polynomial in the unknown with transcendental or symbolic coefficients
(when no solver is given), the candidates of SymPy's ``solve`` (its
Lambert W patterns), and the reliable numerical values of
:mod:`sympy_extras._numeric`.

Examples
========

>>> from sympy import exp, sin, log, sqrt, cbrt, Eq
>>> from sympy.abc import x
>>> from sympy_extras.solvers.transcendental_sympy import solve_transcendental
>>> solve_transcendental(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
{0, log(2)}
>>> solve_transcendental(exp(x) + exp(-x) > 3, x)
Union(Interval.open(-oo, log(3/2 - sqrt(5)/2)), Interval.open(log(sqrt(5)/2 + 3/2), oo))
"""
from __future__ import annotations

from typing import Callable, Final, Optional

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import Boolean
from sympy.sets.sets import Set

from sympy_extras._typing import Truth, as_boolean, as_expr, as_set
from sympy_extras.assumptions.ask_sympy import core_boolean, sympy_boolean
from sympy_extras.assumptions.intervals_sympy import core_expr, core_symbol, sympy_expr, sympy_set
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy
from sympy_extras.sets.sets import Set as CoreSet

from . import transcendental as native

__all__ = ['solve_transcendental', 'polynomialize', 'Reduction', 'Solver', 'Decider', 'solveset_of_core',
    'solve_of_core', 'reliable_form_of_core', 'reliable_value_of_core']

#: solves a formula, polynomial in the given variable, for it (``None``
#: when it cannot)
Solver = Callable[[Boolean, Symbol], Optional[Set]]
#: decides a condition on the parameters (``True``, ``False`` or ``None``)
Decider = Callable[[Boolean], Truth]


def _sympy_symbol(s: CoreSymbol) -> Symbol:
    converted = to_sympy(s)
    if not isinstance(converted, Symbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_set(s: Basic) -> CoreSet:
    """A SymPy set as a set of the core (a wrapper for the sets not
    ported)."""
    converted = from_sympy(as_set(s))
    if not isinstance(converted, CoreSet):
        raise TypeError("expected a set, got %s" % (s,))
    return converted


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def solveset_of_core(relation: CoreBoolean, x: CoreSymbol) -> CoreSet:
    """SymPy's ``solveset`` of one relation over the reals (its
    exceptions go on)."""
    from sympy.core.singleton import S
    from sympy.solvers.solveset import solveset
    return _core_set(solveset(sympy_boolean(relation), _sympy_symbol(x), S.Reals))


def solve_of_core(e: CoreExpr, x: CoreSymbol) -> list[CoreExpr]:
    """The solutions of ``e = 0`` by SymPy's ``solve`` which are
    expressions (with its Lambert W patterns; its exceptions go on)."""
    from sympy.solvers.solvers import solve
    found = solve(sympy_expr(e), _sympy_symbol(x))
    return [core_expr(v) for v in found if isinstance(v, Expr)]


def reliable_form_of_core(e: CoreExpr, digits: int) -> Optional[CoreExpr]:
    """:func:`sympy_extras._numeric.reliable_form` of a constant."""
    from sympy_extras._numeric import reliable_form
    form = reliable_form(sympy_expr(e), digits)
    return None if form is None else core_expr(form)


def reliable_value_of_core(e: CoreExpr, digits: int) -> Optional[CoreExpr]:
    """:func:`sympy_extras._numeric.reliable_value` of a constant."""
    from sympy_extras._numeric import reliable_value
    value = reliable_value(sympy_expr(e), digits)
    return None if value is None else core_expr(value)


# ---------------------------------------------------------------------------
# the face

class Reduction:
    """The reduction of :func:`polynomialize` on SymPy objects (see
    :class:`sympy_extras.solvers.transcendental.Reduction`)."""

    def __init__(self, reduction: native.Reduction) -> None:
        self.formula: Final[Boolean] = sympy_boolean(reduction.formula)
        self.variable: Final[Symbol] = _sympy_symbol(reduction.variable)
        self.argument: Final[Expr] = sympy_expr(reduction.argument)
        self.family: Final[str] = reduction.family
        self.order: Final[int] = reduction.order
        self.special: Final[Boolean] = sympy_boolean(reduction.special)
        self._kernel: Final[Expr] = sympy_expr(reduction.kernel())

    def __repr__(self) -> str:
        return "Reduction(%s, %s = %s)" % (self.formula, self.variable, self._kernel)

    def kernel(self) -> Expr:
        """The transcendental expression the variable stands for."""
        return self._kernel


def polynomialize(formula: Boolean, x: Symbol) -> Optional[Reduction]:
    """:func:`sympy_extras.solvers.transcendental.polynomialize` on SymPy
    objects.

    Examples
    ========

    >>> from sympy import exp, sin, cos, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.transcendental_sympy import polynomialize
    >>> polynomialize(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    Reduction(Eq(_t**2 - 3*_t + 2, 0), _t = exp(x))
    >>> polynomialize(Eq(sin(x) + cos(x), 1), x)
    Reduction(Eq(-2*_w**2 + 2*_w, 0), _w = tan(x/2))
    """
    found = native.polynomialize(core_boolean(as_boolean(formula)), core_symbol(x))
    return None if found is None else Reduction(found)


def solve_transcendental(formula: Boolean, x: Symbol, solver: Optional[Solver] = None,
                         decide: Optional[Decider] = None) -> Optional[Set]:
    """:func:`sympy_extras.solvers.transcendental.solve_transcendental` on
    SymPy objects: the real solutions of a transcendental formula in
    ``x``, or ``None`` when the formula is not reducible to a polynomial
    one.

    Examples
    ========

    >>> from sympy import exp, sin, cos, log, sqrt, cbrt, Eq
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.transcendental_sympy import solve_transcendental
    >>> solve_transcendental(Eq(sin(x) + cos(x), 1), x)
    Union(ImageSet(Lambda(_k, 2*_k*pi), Integers), ImageSet(Lambda(_k, 2*_k*pi + pi/2), Integers))
    >>> solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x)
    {E, exp(2)}
    >>> solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)
    {0}
    """
    native_solver: Optional[native.Solver] = None
    native_decide: Optional[native.Decider] = None
    if solver is not None:
        sympy_solver = solver

        def solve_on_core(f: CoreBoolean, v: CoreSymbol) -> Optional[CoreSet]:
            found = sympy_solver(sympy_boolean(f), _sympy_symbol(v))
            return None if found is None else _core_set(found)
        native_solver = solve_on_core
    if decide is not None:
        sympy_decide = decide

        def decide_on_core(c: CoreBoolean) -> Truth:
            return sympy_decide(sympy_boolean(c))
        native_decide = decide_on_core
    found = native.solve_transcendental(core_boolean(as_boolean(formula)), core_symbol(x), native_solver,
                                        native_decide)
    return None if found is None else sympy_set(found)


def _real_root(e: Expr, x: Symbol, v: Expr) -> Truth:
    """The native ``_real_root`` on SymPy objects: whether the candidate
    is a real root of ``e`` (``None`` when this cannot be told)."""
    return native._real_root(core_expr(as_expr(e)), core_symbol(x), core_expr(as_expr(v)))
