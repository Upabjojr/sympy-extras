"""Ordinary differential equations solved through their point symmetries.

SymPy's :func:`sympy.dsolve` has a ``lie_group`` hint for first order
equations with heuristics for the infinitesimals. Here the symmetries of an
equation of any order are computed from the determining equations with a
polynomial ansatz (:mod:`sympy_extras.solvers.lie`), and a symmetry is used
in the classical way: in canonical coordinates `(r, s)` in which the
symmetry is the translation `\\partial_s`, the equation does not contain
`s`, so an equation of order `n` becomes one of order `n - 1` for
`v = ds/dr` (a quadrature when `n = 1`), which is handed to SymPy's
``dsolve``.

The module works on the objects of the core and imports no SymPy: the
symmetries, the jets, the cancellations and the simplifications are the
native ones, the first order equations go to the native
:func:`~sympy_extras.solvers.first_order.dsolve_first_order`, and a
solution is checked by :func:`check_ode_solution`, a port of SymPy's
``checkodesol`` (``sympy.solvers.ode.subscheck``) on the core.
:mod:`sympy_extras.solvers.ode_sympy` is its face on SymPy objects. The
reduced equations are solved by the native classical hints of
:func:`~sympy_extras.solvers.dsolve.dsolve`; one step has no native
counterpart and goes through SymPy at run time (the functions ending in
``_of_core`` of the faces): SymPy's ``solve`` of the canonical coordinates and of the implicit solutions
(:func:`~sympy_extras.solvers.pde_sympy.solve_of_core`); the
quadratures come from the native indefinite integration
(:func:`~sympy_extras.integrals.indefinite.closed_antiderivative`).

The steps which may not terminate (integration, ``dsolve`` and ``solve``
of SymPy, the verification of the solutions) run under a time limit, see
:mod:`sympy_extras._timeout`.

References
==========

.. [Hydon] P. E. Hydon, Symmetry Methods for Differential Equations,
   Cambridge University Press 2000, chapters 2-4.
.. [Stephani] H. Stephani, Differential Equations: Their Solution Using
   Symmetries, Cambridge University Press 1989.

Examples
========

>>> import sympy_extras.interop  # SymPy's dsolve, solve and integrate (the *_of_core functions)
>>> from sympy_extras.core import Function, symbols
>>> from sympy_extras.solvers.ode import dsolve_lie
>>> x = symbols('x')
>>> y = Function('y')(x)
>>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y, y)
[Eq(y(x), exp(C1*x + C2))]
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/solvers/ode/subscheck.py (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: checkodesol for one equation and one solution on the typed core
# (check_ode_solution, a boolean), sub_func_doit, strict typing.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Final, Optional, Sequence, TypeVar

from sympy_extras._timeout import attempt
from sympy_extras.core.basic import Basic, preorder_traversal
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import Application, AppliedUndef, Derivative, Function, count_ops, diff, expand
from sympy_extras.core.numbers import Integer, nan
from sympy_extras.core.order import Order
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import cancel
from sympy_extras.settings import configure, settings
from sympy_extras.simplify.simplify import besselsimp, posify, simplify
from sympy_extras.simplify.trigsimp import trigsimp

from ._special_values import ode_order, solution_cases
from .lie import Equation, JetSpace, Symmetry, symmetries, _as_zero, _leading_derivatives, _replaced
from .linear_ode import has_integral

__all__ = ['ode_symmetries', 'canonical_coordinates', 'reduce_order', 'dsolve_lie', 'solve_ode',
           'check_ode_solution', 'ReducedODE']

#: the default ``timeout`` argument: the value of
#: :data:`sympy_extras.settings.settings.timeout` at call time
DEFAULT_TIMEOUT = -1.0

T = TypeVar('T')

_ZERO: Final[Expr] = Integer(0)
_ONE: Final[Expr] = Integer(1)

#: what the polynomial routines raise on coefficients they cannot take
_FAILURES = (BasePolynomialError, NotImplementedError, ValueError, TypeError, ZeroDivisionError)


def _limit(timeout: Optional[float]) -> Optional[float]:
    """The time limit to use: the global setting for the default marker."""
    return settings.timeout if timeout == DEFAULT_TIMEOUT else timeout


def _expr(e: Basic) -> Expr:
    if not isinstance(e, Expr):
        raise TypeError("expected an expression, got %s" % (e,))
    return e


def _normal(e: Expr) -> Expr:
    """``e`` in lowest terms (``e`` itself when the native ``cancel``
    cannot take its coefficients)."""
    try:
        return cancel(e)
    except _FAILURES:
        return e


# ---------------------------------------------------------------------------
# the steps which stay on SymPy

def _with_limit(f: Callable[[], T], timeout: Optional[float]) -> Optional[T]:
    """``f()`` (a step on SymPy whose own limit is the one of the
    settings) within ``timeout`` seconds: the settings carry the limit
    for the duration of the step."""
    if timeout is None:
        with configure(no_timeout=True):
            return attempt(f, None, native=True)
    with configure(timeout=timeout):
        return attempt(f, timeout, native=True)


def _dsolve(equation: Basic, f: AppliedUndef, timeout: Optional[float]) -> Optional[list[Eq]]:
    """The solutions ``Eq`` of an ODE by the native classical methods
    (:func:`~sympy_extras.solvers.dsolve.dsolve`) within the time limit
    (``None`` when it gives up)."""
    from .dsolve import dsolve
    return attempt(lambda: dsolve(equation, f), timeout, native=True)


def _solve(e: Expr, v: Symbol, timeout: Optional[float]) -> list[Expr]:
    """The solutions of ``e = 0`` for ``v`` by SymPy's ``solve`` within the
    time limit (none when it gives up)."""
    from .pde_sympy import solve_of_core
    return attempt(lambda: solve_of_core(e, v), timeout) or []


def _integral(e: Expr, v: Symbol, timeout: Optional[float]) -> Optional[Expr]:
    """An antiderivative by SymPy's ``integrate`` within the time limit,
    ``None`` when it gives up or leaves an ``Integral``."""
    from sympy_extras.integrals.indefinite import closed_antiderivative
    found = _with_limit(lambda: closed_antiderivative(e, v, cases=True), timeout)
    return found


def _quadrature(e: Expr, v: Symbol, timeout: Optional[float]) -> Expr:
    """An antiderivative as :func:`_integral`, and the unevaluated
    ``Integral`` (SymPy's, wrapped) when there is none."""
    found = _integral(e, v, timeout)
    if found is not None:
        return found
    from sympy_extras.integrals.integral import Integral
    return Integral(e, v)


# ---------------------------------------------------------------------------
# the symmetries and the canonical coordinates

def _natural_basis(eq: Expr, x: Symbol, ysym: Symbol) -> list[Expr]:
    """Functions appearing in the equation, candidates for the ansatz of
    the infinitesimals."""
    found: list[Expr] = []
    for f in sorted(eq.atoms(Application), key=str):
        if isinstance(f, AppliedUndef) or not isinstance(f, Expr) or not (f.free_symbols & {x, ysym}):
            continue
        if f not in found:
            found.append(f)
    return found[:4]


def _simplicity(sym: Symmetry) -> tuple[int, str]:
    ops = count_ops(list(sym.xi) + list(sym.eta))
    if not isinstance(ops, Integer):
        raise TypeError("expected an integer, got %s" % (ops,))
    return (ops.p, str(sym))


def ode_symmetries(ode: Equation, y: AppliedUndef, degree: int = 2,
                   basis: Sequence[Expr] = ()) -> list[Symmetry]:
    """The point symmetries of an ordinary differential equation with
    polynomial infinitesimals of total degree at most ``degree`` (times
    the functions of ``basis``), see
    :func:`sympy_extras.solvers.lie.symmetries`; sorted by the number of
    operations.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.ode import ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> for X in ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y):
    ...     print(X.generator())
    x*d/dx - y/2*d/dy
    """
    found = symmetries(ode, y, degree=degree, basis=basis, superposition=True)
    return sorted(found, key=_simplicity)


def _constants(taken: set[Symbol], n: int) -> list[Symbol]:
    result: list[Symbol] = []
    k = 1
    while len(result) < n:
        c = Symbol('C%d' % k)
        if c not in taken:
            result.append(c)
        k += 1
    return result


def _first_integral(ode_rhs: Expr, Y: AppliedUndef, x: Symbol, ysym: Symbol,
                    timeout: Optional[float]) -> Optional[Expr]:
    """An invariant `r(x, y)` of `dy/dx = f(x, y)`: the constant of the
    general solution solved for."""
    sol = _dsolve(Eq(diff(Y, x), ode_rhs), Y, timeout)
    if sol is None:
        return None
    for candidate in sol:
        constants = [c for c in candidate.free_symbols if c.name.startswith('C')]
        if len(constants) != 1:
            continue
        values = _solve(_expr(candidate.lhs) - _expr(candidate.rhs), constants[0], timeout)
        if values:
            return simplify(values[0].subs(Y, ysym))
    return None


def canonical_coordinates(symmetry: Symmetry, timeout: Optional[float] = DEFAULT_TIMEOUT
                          ) -> Optional[tuple[Expr, Expr]]:
    """Canonical coordinates `(r, s)` of a symmetry of an ordinary
    differential equation: `X r = 0` and `X s = 1`, so that `X` is
    `\\partial_s`. Expressions in the symbols of the independent and
    dependent variables, or ``None`` when they cannot be found (within
    ``timeout`` seconds per step).

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's routes of indefinite, if reached
    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.ode import canonical_coordinates, ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> [X] = ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y)
    >>> canonical_coordinates(X)
    (sqrt(x)*y, log(x))
    """
    limit = _limit(timeout)
    jets = symmetry.jets
    if jets.p != 1 or jets.q != 1:
        raise ValueError("an ordinary differential equation with one unknown is expected")
    x, ysym = jets.x[0], jets.u[0]
    xi, eta = symmetry.xi[0], symmetry.eta[0]
    if xi == 0 and eta == 0:
        return None
    r: Optional[Expr]
    if xi == 0:
        r = x
        s = _integral(1/eta, ysym, limit)
    elif eta == 0:
        r = ysym
        s = _integral(1/xi, x, limit)
    else:
        Y = Function('Y')(x)
        r = _first_integral(_expr((eta/xi).subs(ysym, Y)), Y, x, ysym, limit)
        if r is None:
            return None
        s = _transversal(r, xi, eta, x, ysym, limit)
    if r is None or s is None or has_integral(s):
        return None
    R, S = r, s
    check = attempt(lambda: (simplify(xi*diff(R, x) + eta*diff(R, ysym)),
                             simplify(xi*diff(S, x) + eta*diff(S, ysym) - 1)), limit, native=True)
    if check is None or check[0] != 0 or check[1] != 0:
        return None
    return r, s


def _transversal(r: Expr, xi: Expr, eta: Expr, x: Symbol, ysym: Symbol,
                 timeout: Optional[float]) -> Optional[Expr]:
    """`s` with `X s = 1`: the integral of `dx/\\xi` along `r` constant
    (or of `dy/\\eta`)."""
    rsym = Symbol('r')
    for solved, other, coefficient in ((ysym, x, xi), (x, ysym, eta)):
        for value in _solve(r - rsym, solved, timeout):
            integrand = _normal(_expr((1/coefficient).subs(solved, value)))
            s = _integral(integrand, other, timeout)
            if s is not None:
                return s.subs(rsym, r)
    return None


# ---------------------------------------------------------------------------
# the reduction of the order

class ReducedODE:
    """The reduction of an ordinary differential equation of order `n`
    by a symmetry.

    Attributes
    ==========

    r, s : Expr
        The canonical coordinates, expressions in the independent and
        dependent variables.
    variable : Symbol
        The symbol standing for ``r``.
    v : applied undefined function
        The unknown ``v(r) = ds/dr`` of the reduced equation.
    ode : Expr
        The reduced equation of order `n - 1` for ``v`` (for `n = 1` the
        expression ``v - G(r)``), equal to zero.
    """

    def __init__(self, r: Expr, s: Expr, variable: Symbol, v: AppliedUndef, ode: Expr) -> None:
        self.r: Final[Expr] = r
        self.s: Final[Expr] = s
        self.variable: Final[Symbol] = variable
        self.v: Final[AppliedUndef] = v
        self.ode: Final[Expr] = ode

    def __repr__(self) -> str:
        return "ReducedODE(r=%s, s=%s, ode=%s)" % (self.r, self.s, self.ode)


def _free_of(expr: Expr, symbols_: set[Symbol], timeout: Optional[float]) -> Optional[Expr]:
    """``expr`` simplified, if it does not depend on ``symbols_``."""
    e = _normal(expr)
    if e.free_symbols & symbols_:
        simplified = attempt(lambda: simplify(e), timeout, native=True)
        if simplified is None:
            return None
        e = simplified
    if e.free_symbols & symbols_:
        return None
    return e


def reduce_order(ode: Equation, y: AppliedUndef, symmetry: Symmetry,
                 timeout: Optional[float] = DEFAULT_TIMEOUT) -> Optional[ReducedODE]:
    """Reduce the order of an ordinary differential equation with a point
    symmetry, through its canonical coordinates.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's integrate and solve (the *_of_core functions)
    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.ode import reduce_order
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> eq = y.diff(x, 2) - y.diff(x)**2/y
    >>> jets = JetSpace([y])
    >>> reduce_order(eq, y, Symmetry(jets, [0], [x*jets.u[0]]))
    ReducedODE(r=x, s=log(y)/x, ode=Derivative(v(r), r) + 2*v(r)/r)
    """
    limit = _limit(timeout)
    jets = JetSpace([y])
    x, ysym = jets.x[0], jets.u[0]
    eq = jets.to_jet(_as_zero(ode))
    n = jets.order(eq)
    if n == 0:
        return None
    coordinates = canonical_coordinates(symmetry, limit)
    if coordinates is None:
        return None
    r, s = coordinates
    leading = _leading_derivatives(jets, [eq])
    y_n = jets.jet(0, (n,))
    if y_n not in leading:
        return None
    omega = leading[y_n]
    # derivatives of s with respect to r along solutions
    D_r = jets.total_derivative(r, 0)
    if D_r == 0:
        return None
    s_k: list[Expr] = [s]
    for _ in range(n):
        s_k.append(_normal(jets.total_derivative(s_k[-1], 0)/D_r))
    rsym = Symbol('r')
    v_k = [Symbol('v_%d' % k) for k in range(n)]  # v_k stands for s^{(k+1)}
    # express y (or x), y_1, ..., y_{n-1} through r, v_1, ..., v_{n-1}
    substitutions: dict[Basic, Basic] = {}
    eliminated: Optional[Symbol] = None
    for candidate in (ysym, x):
        values = _solve(r - rsym, candidate, limit)
        if values:
            substitutions[candidate] = values[0]
            eliminated = candidate
            break
    if eliminated is None:
        return None
    remaining = {x, ysym} - {eliminated}
    for k in range(1, n):
        y_k = jets.jet(0, (k,))
        equation = _replaced(s_k[k], substitutions) - v_k[k - 1]
        solved = _solve(equation, y_k, limit)
        if not solved:
            return None
        substitutions[y_k] = solved[0]
    top = _replaced(s_k[n], {y_n: omega})
    for _ in range(n):
        new = _replaced(top, substitutions)
        if new == top:
            break
        top = new
    G = _free_of(top, remaining, limit)
    if G is None:
        return None
    v = Function('v')(rsym)
    reduced: Expr
    if n == 1:
        reduced = v - G
    else:
        for k in range(n - 1, 0, -1):
            G = G.subs(v_k[k - 1], Derivative(v, (rsym, k - 1)) if k > 1 else v)
        reduced = Derivative(v, (rsym, n - 1)) - G
    return ReducedODE(r, s, rsym, v, reduced)


# ---------------------------------------------------------------------------
# the check of a solution

# PORTED FROM SYMPY: sympy/solvers/ode/subscheck.py:sub_func_doit
def _sub_func_doit(eq: Expr, func: Expr, new: Expr) -> Expr:
    """``eq`` with ``func`` replaced by ``new`` and its derivatives by the
    derivatives of ``new``."""
    reps: dict[Basic, Basic] = {func: new}
    for d in eq.atoms(Derivative):
        if not isinstance(d, Derivative):
            continue
        if d.expr == func:
            reps[d] = diff(new, *d.variable_count)
        else:
            reps[d] = _replaced(d, {func: new}).doit()
    return _replaced(eq, reps)


def _solved_for(e: Expr, target: Expr, timeout: Optional[float]) -> Optional[list[Expr]]:
    """The solutions of ``e = 0`` for ``target`` (a function or a
    derivative, replaced by a dummy) by SymPy's ``solve`` within the time
    limit, ``None`` when it gives up."""
    from .pde_sympy import solve_of_core
    d = Dummy('f')
    found = attempt(lambda: solve_of_core(_replaced(e, {target: d}), d), timeout)
    return None if found is None else [value.subs(d, target) for value in found]


def _linear_solution(e: Expr, target: Expr) -> Optional[list[Expr]]:
    """The solution of ``e = 0`` for ``target`` when ``e`` is linear in it
    (the derivative of a relation is linear in ``y'``), ``None``
    otherwise."""
    d = Dummy('f')
    linear = _replaced(e, {target: d})
    a = diff(linear, d)
    if a == 0 or a.has(d):
        return None
    b = linear.subs(d, _ZERO)
    return [_normal(-b/a)]


# PORTED FROM SYMPY: sympy/solvers/ode/subscheck.py:checkodesol
def check_ode_solution(ode: Equation, solution: Basic, f: AppliedUndef,
                       timeout: Optional[float] = DEFAULT_TIMEOUT) -> bool:
    """Whether ``solution`` (``Eq(f, rhs)``, an implicit relation ``Eq(F(x,
    f), G(x, f))`` or the right-hand side alone) solves the ordinary
    differential equation, as SymPy's ``checkodesol`` decides it: an
    implicit relation is solved for ``f`` first (by SymPy's ``solve``,
    within ``timeout`` seconds; ``False`` when it gives up, where
    ``checkodesol`` raises), and every solution must be checked; the
    solution is substituted and the residual simplified, else the
    derivatives of the relation are substituted (when ``solve`` finds no
    solution). ``False`` when nothing proves it.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's solve (pde_sympy.solve_of_core)
    >>> from sympy_extras.core import Eq, Function, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.ode import check_ode_solution
    >>> x, C1 = symbols('x C1')
    >>> y = Function('y')(x)
    >>> check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(x)), y)
    True
    >>> check_ode_solution(y.diff(x) - y, Eq(y, C1*exp(2*x)), y)
    False
    >>> check_ode_solution(y*y.diff(x) + x, Eq(x**2 + y**2, C1), y)
    True
    """
    limit = _limit(timeout)
    found = attempt(lambda: _checkodesol(ode, solution, f, limit, True), limit, native=True)
    return found is True


def _checkodesol(ode: Equation, solution: Basic, func: AppliedUndef, timeout: Optional[float],
                 solve_for_func: bool) -> bool:
    if len(func.args) != 1 or not isinstance(func.args[0], Symbol):
        raise ValueError("func must be a function of one variable, not %s" % (func,))
    x = func.args[0]
    lhs_ode, rhs_ode = (_expr(ode.lhs), _expr(ode.rhs)) if isinstance(ode, Eq) else (ode, _ZERO)
    if isinstance(solution, Eq):
        sol_lhs, sol_rhs = _expr(solution.lhs), _expr(solution.rhs)
        if sol_rhs == func:
            sol_lhs, sol_rhs = sol_rhs, sol_lhs
    else:
        sol_lhs, sol_rhs = func, _expr(solution)
    order = ode_order(lhs_ode - rhs_ode, func)
    solved = sol_lhs == func and not sol_rhs.has(func)
    if solve_for_func and not solved:
        values = _solved_for(sol_lhs - sol_rhs, func, timeout)
        if values is None:
            # SymPy's solve gives up: so does checkodesol
            return False
        if values:
            return all(_checkodesol(ode, Eq(func, value), func, timeout, False) for value in values)
    if sol_lhs.has(Order) or sol_rhs.has(Order):
        # a truncated series: SymPy compares the orders of the terms
        return False
    s: Optional[Expr] = None
    # first pass: the solution substituted into the equation
    if sol_lhs == func:
        substituted = besselsimp(_sub_func_doit(lhs_ode - rhs_ode, func, sol_rhs))
        simplified = simplify(substituted.rewrite(Exp))
        s = expand(simplified, force=True) if simplified != 0 else _ZERO
        if s == 0:
            return True
    # second pass: the n-th derivatives equal (an exact equation)
    s = simplify(trigsimp(diff(sol_lhs, x, order) - diff(sol_rhs, x, order)) - trigsimp(lhs_ode) + trigsimp(rhs_ode))
    if s == 0:
        return True
    # third pass: the derivatives of the relation substituted, from the
    # highest down
    diffsols: dict[int, Expr] = {}
    if sol_lhs == func and not sol_rhs.has(func):
        diffsols[0] = sol_rhs
    elif sol_rhs == func and not sol_lhs.has(func):
        diffsols[0] = sol_lhs
    relation = sol_lhs - sol_rhs
    for i in range(1, order + 1):
        if i == 1:
            sdf = _linear_solution(diff(relation, x), diff(func, x)) or _solved_for(diff(relation, x),
                                                                                   diff(func, x), timeout)
            if sdf is None or not sdf:
                return False
            diffsols[i] = sdf[0]
        else:
            diffsols[i] = diff(diffsols[i - 1], x)
    lhs, rhs = lhs_ode, rhs_ode
    lowest = 0 if 0 in diffsols else 1
    for i in range(order, lowest - 1, -1):
        target = diff(func, x, i) if i else func
        lhs = _sub_func_doit(lhs, target, diffsols[i])
        rhs = _sub_func_doit(rhs, target, diffsols[i])
    # SymPy simplifies Eq(lhs, rhs) after every substitution; once, after
    # the last one, decides the same (the native simplify is slower)
    if simplify(lhs - rhs) == 0:
        return True
    num = trigsimp((lhs - rhs).as_numer_denom()[0])
    dummy = Dummy('func')
    num = _replaced(num, {func: dummy})
    positive, reps = posify(num)
    s = _replaced(_replaced(simplify(positive), reps), {dummy: func})
    return s == 0


def _verified(ode: Equation, candidate: Eq, y: AppliedUndef, timeout: Optional[float]) -> bool:
    return check_ode_solution(ode, candidate, y, timeout)


# ---------------------------------------------------------------------------
# the solvers

def _solve_reduced(ode: Equation, y: AppliedUndef, reduction: ReducedODE, check: bool,
                   timeout: Optional[float]) -> list[Eq]:
    """Solve the reduced equation and go back to the original variables."""
    jets = JetSpace([y])
    ysym = jets.u[0]
    rsym, v = reduction.variable, reduction.v
    n = jets.order(jets.to_jet(_as_zero(ode)))
    v_solutions: list[Expr]
    if n == 1:
        v_solutions = [v - reduction.ode]
    else:
        sols = _dsolve(reduction.ode, v, timeout)
        if sols is None:
            return []
        v_solutions = [_expr(s.rhs) for s in sols if s.lhs == v]
    results: list[Eq] = []
    for V in v_solutions:
        S_r = _quadrature(V, rsym, timeout)
        taken = S_r.free_symbols | _as_zero(ode).free_symbols
        [C] = _constants(taken, 1)
        lhs, rhs = reduction.s, S_r.subs(rsym, reduction.r) + C
        candidates: list[Eq] = []
        for value in _solve(lhs - rhs, ysym, timeout):
            candidates.append(Eq(y, value.subs(ysym, y)))
        if not candidates:
            candidates.append(Eq(lhs.subs(ysym, y), rhs.subs(ysym, y)))
        for candidate in candidates:
            if check and not _verified(ode, candidate, y, timeout):
                continue
            results.append(candidate)
    return results


def dsolve_lie(ode: Equation, y: AppliedUndef, degree: int = 2, basis: Sequence[Expr] = (),
               symmetries_: Optional[Sequence[Symmetry]] = None, check: bool = True,
               timeout: Optional[float] = DEFAULT_TIMEOUT, special_values: bool = True) -> list[Eq]:
    """Solve an ordinary differential equation through its point
    symmetries.

    The symmetries are computed with a polynomial ansatz of the given
    ``degree`` (times the functions of ``basis``; when no symmetry is found
    the functions occurring in the equation are tried as well) unless
    given; each is used to reduce the order of the equation, the reduced
    equation is solved by SymPy's ``dsolve`` and the solutions are
    written in the original variables. With ``check`` only the solutions
    verified by :func:`check_ode_solution` are returned. Every step handed
    to SymPy is abandoned after ``timeout`` seconds (by default the
    ``timeout`` of :data:`sympy_extras.settings.settings`). With
    ``special_values``, a single explicit general solution gets cases for
    the isolated values of the parameters at which it is undefined or no
    longer general, as in :func:`solve_ode`.

    Examples
    ========

    >>> import sympy_extras.interop  # SymPy's dsolve, solve and integrate (the *_of_core functions)
    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.ode import dsolve_lie
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x, y)
    [Eq(y(x), exp(C1*x**2/2 + C2))]
    """
    limit = _limit(timeout)
    syms: list[Symmetry]
    if symmetries_ is None:
        syms = ode_symmetries(ode, y, degree=degree, basis=basis)
        if not syms and not basis:
            jets = JetSpace([y])
            natural = _natural_basis(jets.to_jet(_as_zero(ode)), jets.x[0], jets.u[0])
            if natural:
                syms = ode_symmetries(ode, y, degree=degree, basis=natural)
    else:
        syms = list(symmetries_)
    solutions: list[Eq] = []
    for sym in syms:
        reduction = reduce_order(ode, y, sym, limit)
        if reduction is None:
            continue
        for sol in _solve_reduced(ode, y, reduction, check, limit):
            if sol not in solutions:
                solutions.append(sol)
        if solutions:
            break
    if special_values and symmetries_ is None:
        cased = solution_cases(solutions, ode, y, lambda at: dsolve_lie(at, y, degree, basis, None, check, limit))
        if cased is not None:
            return [cased]
    return solutions


def solve_ode(ode: Equation, y: AppliedUndef, degree: int = 2, check: bool = True,
              timeout: Optional[float] = DEFAULT_TIMEOUT, special_values: bool = True) -> list[Eq]:
    """Solve an ordinary differential equation with SymPy's ``dsolve``
    and, when that fails or takes more than ``timeout`` seconds, with the
    first order solvers of the package for an equation of first order
    (:func:`~sympy_extras.solvers.first_order.dsolve_first_order`: the
    Riccati, Abel and Chini classes and the Prelle–Singer procedure) and
    through its point symmetries (:func:`dsolve_lie`).

    With ``special_values``, a single explicit general solution of an
    equation with parameters gets a case of its own, first, for each
    isolated value of the parameters at which it is undefined (the
    resonance of ``y' - k*y = exp(x)`` at ``k = 1``) or no longer general
    (the repeated characteristic root of ``y'' - (a + b)*y' + a*b*y = 0``
    at ``a = b``, where the Wronskian of the fundamental system vanishes);
    the equation is solved again at the point and the solution there is
    checked (:mod:`sympy_extras.solvers._special_values`). A point where
    the order of the equation drops gets no case.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.ode import solve_ode
    >>> x, k = symbols('x k')
    >>> y = Function('y')(x)
    >>> solve_ode(y.diff(x) - k*y - exp(x), y)
    [Eq(y(x), Piecewise(((C1 + x)*exp(x), Eq(k, 1)), (C1*exp(k*x) - exp(x)/(k - 1), True)))]
    """
    limit = _limit(timeout)
    found = _solve_ode(ode, y, degree, check, limit)
    if special_values:
        cased = solution_cases(found, ode, y, lambda at: solve_ode(at, y, degree, check, limit))
        if cased is not None:
            return [cased]
    return found


def _solve_ode(ode: Equation, y: AppliedUndef, degree: int, check: bool, timeout: Optional[float]) -> list[Eq]:
    found = _dsolve(ode, y, timeout) or []
    # ``check`` applies to this branch as well as to the fallback: SymPy's
    # dsolve returns truncated series (with an O term), answers carrying
    # nan, and outright non-solutions -- a spurious sign branch, a family
    # holding for two values of the constant only (sympy-extras#40). What
    # the check rejects is not returned, and the same answer written
    # twice is returned once.
    accepted: list[Eq] = []
    for candidate in found:
        if candidate in accepted or candidate.has(Order, nan) or _has_foreign(candidate, 'Order'):
            continue
        if check and not _verified(ode, candidate, y, timeout):
            continue
        accepted.append(candidate)
    if accepted:
        return accepted
    first = _first_order(ode, y, check, timeout)
    if first is not None:
        return [first]
    return dsolve_lie(ode, y, degree=degree, check=check, timeout=timeout, special_values=False)


def _has_foreign(e: Basic, name: str) -> bool:
    """Whether ``e`` has a wrapped SymPy node of the class ``name``."""
    return any(node._is_foreign_node() and node._compare_name() == name for node in preorder_traversal(e))


def _first_order(ode: Equation, y: AppliedUndef, check: bool, timeout: Optional[float]) -> Optional[Eq]:
    """The solution of a first order equation by the first order solvers
    (an ``Eq``, explicit or implicit; a parametric solution is left to
    the symmetry method), verified when ``check``."""
    from .first_order import dsolve_first_order
    equation = _as_zero(ode)
    orders = [max((int(n) for _, n in d.variable_count), default=0) for d in equation.atoms(Derivative)
              if isinstance(d, Derivative) and d.expr == y]
    if not orders or max(orders) != 1:
        return None
    found = attempt(lambda: dsolve_first_order(ode, y, special_values=False), timeout, native=True)
    if not isinstance(found, Eq):
        return None
    if check and not _verified(ode, found, y, timeout):
        return None
    return found
