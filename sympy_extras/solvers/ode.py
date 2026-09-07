"""Ordinary differential equations solved through their point symmetries.

SymPy's :func:`sympy.dsolve` has a ``lie_group`` hint for first order
equations with heuristics for the infinitesimals. Here the symmetries of an
equation of any order are computed from the determining equations with a
polynomial ansatz (:mod:`sympy_extras.solvers.lie`), and a symmetry is used
in the classical way: in canonical coordinates `(r, s)` in which the
symmetry is the translation `\\partial_s`, the equation does not contain
`s`, so an equation of order `n` becomes one of order `n - 1` for
`v = ds/dr` (a quadrature when `n = 1`), which is handed to
:func:`sympy.dsolve`.

The steps which may not terminate (integration, ``dsolve`` and ``solve``
of SymPy, the verification of the solutions) run under a time limit, see
:mod:`sympy_extras._timeout`.

References
==========

.. [Hydon] P. E. Hydon, Symmetry Methods for Differential Equations,
   Cambridge University Press 2000, chapters 2-4.
.. [Stephani] H. Stephani, Differential Equations: Their Solution Using
   Symmetries, Cambridge University Press 1989.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, Function, diff, count_ops
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral, integrate
from sympy.polys.polytools import cancel
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve, checkodesol
from sympy.solvers.solvers import solve

from sympy_extras._timeout import attempt
from sympy_extras._typing import as_expr, free_symbols

from .lie import Equation, JetSpace, Symmetry, symmetries, _as_zero, _leading_derivatives

__all__ = ['ode_symmetries', 'canonical_coordinates', 'reduce_order',
    'dsolve_lie', 'solve_ode', 'ReducedODE']

#: default time limit, in seconds, of each step handed to SymPy
DEFAULT_TIMEOUT = 30.0


def _natural_basis(eq: Expr, x: Symbol, ysym: Symbol) -> list[Expr]:
    """Functions appearing in the equation, candidates for the ansatz of
    the infinitesimals."""
    from sympy.core.function import Function as _Function
    found: list[Expr] = []
    for f in sorted(eq.atoms(_Function), key=str):
        if isinstance(f, AppliedUndef) or not (free_symbols(f) & {x, ysym}):
            continue
        if f not in found:
            found.append(as_expr(f))
    return found[:4]


def ode_symmetries(ode: Equation, y: AppliedUndef, degree: int = 2,
                   basis: Sequence[Expr] = ()) -> list[Symmetry]:
    """The point symmetries of an ordinary differential equation with
    polynomial infinitesimals of total degree at most ``degree`` (times
    the functions of ``basis``), see
    :func:`sympy_extras.solvers.lie.symmetries`.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> for X in ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y):
    ...     print(X.generator())
    x*d/dx - y/2*d/dy
    """
    found = symmetries(ode, y, degree=degree, basis=basis, superposition=True)
    return sorted(found, key=lambda s: (int(count_ops(list(s.xi) + list(s.eta))), str(s)))


def _constants(taken: set[Symbol], n: int) -> list[Symbol]:
    result: list[Symbol] = []
    k = 1
    while len(result) < n:
        c = Symbol('C%d' % k)
        if c not in taken:
            result.append(c)
        k += 1
    return result


def _solutions(result: object) -> list[Eq]:
    """The equations among what ``dsolve`` returned."""
    items = result if isinstance(result, list) else [result]
    return [s for s in items if isinstance(s, Eq)]


def _first_integral(ode_rhs: Expr, Y: AppliedUndef, x: Symbol, ysym: Symbol,
                    timeout: Optional[float]) -> Optional[Expr]:
    """An invariant `r(x, y)` of `dy/dx = f(x, y)`: the constant of the
    general solution solved for."""
    sol = attempt(lambda: dsolve(Eq(Y.diff(x), ode_rhs), Y), timeout)
    if sol is None:
        return None
    for candidate in _solutions(sol):
        constants = [c for c in free_symbols(candidate) if c.name.startswith('C')]
        if len(constants) != 1:
            continue
        values = attempt(lambda: solve(candidate, constants[0]), timeout)
        if values:
            return as_expr(simplify(as_expr(values[0]).subs(Y, ysym)))
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

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import canonical_coordinates, ode_symmetries
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> [X] = ode_symmetries(y.diff(x, 2) + 2*y.diff(x)/x + y**5, y)
    >>> canonical_coordinates(X)
    (sqrt(x)*y, log(x))
    """
    jets = symmetry.jets
    if jets.p != 1 or jets.q != 1:
        raise ValueError("an ordinary differential equation with one unknown is expected")
    x, ysym = jets.x[0], jets.u[0]
    xi, eta = symmetry.xi[0], symmetry.eta[0]
    if xi == 0 and eta == 0:
        return None
    if xi == 0:
        r: Optional[Expr] = as_expr(x)
        s = attempt(lambda: as_expr(integrate(1/eta, ysym)), timeout)
    elif eta == 0:
        r = as_expr(ysym)
        s = attempt(lambda: as_expr(integrate(1/xi, x)), timeout)
    else:
        Y = Function('Y')(x)
        r = _first_integral(as_expr((eta/xi).subs(ysym, Y)), Y, x, ysym, timeout)
        if r is None:
            return None
        s = _transversal(r, xi, eta, x, ysym, timeout)
    if r is None or s is None or s.has(Integral):
        return None
    check = attempt(lambda: (simplify(xi*diff(r, x) + eta*diff(r, ysym)),
                             simplify(xi*diff(s, x) + eta*diff(s, ysym) - 1)), timeout)
    if check is None or check[0] != 0 or check[1] != 0:
        return None
    return r, s


def _transversal(r: Expr, xi: Expr, eta: Expr, x: Symbol, ysym: Symbol,
                 timeout: Optional[float]) -> Optional[Expr]:
    """`s` with `X s = 1`: the integral of `dx/\\xi` along `r` constant
    (or of `dy/\\eta`)."""
    rsym = Symbol('r')
    for solved, other, coefficient in ((ysym, x, xi), (x, ysym, eta)):
        values = attempt(lambda: solve(Eq(r, rsym), solved), timeout) or []
        for value in values:
            integrand = as_expr(cancel((1/coefficient).subs(solved, value)))
            s = attempt(lambda: as_expr(integrate(integrand, other)), timeout)
            if s is not None and not s.has(Integral):
                return as_expr(s.subs(rsym, r))
    return None


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
        self.r = r
        self.s = s
        self.variable = variable
        self.v = v
        self.ode = ode

    def __repr__(self) -> str:
        return "ReducedODE(r=%s, s=%s, ode=%s)" % (self.r, self.s, self.ode)


def _free_of(expr: Expr, symbols_: set[Symbol], timeout: Optional[float]) -> Optional[Expr]:
    """``expr`` simplified, if it does not depend on ``symbols_``."""
    e = as_expr(cancel(expr))
    if free_symbols(e) & symbols_:
        simplified = attempt(lambda: as_expr(simplify(e)), timeout)
        if simplified is None:
            return None
        e = simplified
    if free_symbols(e) & symbols_:
        return None
    return e


def reduce_order(ode: Equation, y: AppliedUndef, symmetry: Symmetry,
                 timeout: Optional[float] = DEFAULT_TIMEOUT) -> Optional[ReducedODE]:
    """Reduce the order of an ordinary differential equation with a point
    symmetry, through its canonical coordinates.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import reduce_order, JetSpace, Symmetry
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> eq = y.diff(x, 2) - y.diff(x)**2/y
    >>> jets = JetSpace([y])
    >>> reduce_order(eq, y, Symmetry(jets, [0], [x*jets.u[0]]))
    ReducedODE(r=x, s=log(y)/x, ode=Derivative(v(r), r) + 2*v(r)/r)
    """
    jets = JetSpace([y])
    x, ysym = jets.x[0], jets.u[0]
    eq = jets.to_jet(_as_zero(ode))
    n = jets.order(eq)
    if n == 0:
        return None
    coordinates = canonical_coordinates(symmetry, timeout)
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
        s_k.append(as_expr(cancel(jets.total_derivative(s_k[-1], 0)/D_r)))
    rsym = Symbol('r')
    v_k = [Symbol('v_%d' % k) for k in range(n)]  # v_k stands for s^{(k+1)}
    # express y (or x), y_1, ..., y_{n-1} through r, v_1, ..., v_{n-1}
    substitutions: dict[Basic, Basic] = {}
    eliminated: Optional[Symbol] = None
    for candidate in (ysym, x):
        values = attempt(lambda: solve(Eq(r, rsym), candidate), timeout)
        if values:
            substitutions[candidate] = values[0]
            eliminated = candidate
            break
    if eliminated is None:
        return None
    remaining = {x, ysym} - {eliminated}
    for k in range(1, n):
        y_k = jets.jet(0, (k,))
        equation = as_expr(s_k[k].xreplace(substitutions)) - v_k[k - 1]
        values = attempt(lambda: solve(equation, y_k), timeout)
        if not values:
            return None
        substitutions[y_k] = values[0]
    top = as_expr(s_k[n].xreplace({y_n: omega}))
    for _ in range(n):
        new = as_expr(top.xreplace(substitutions))
        if new == top:
            break
        top = new
    G = _free_of(top, remaining, timeout)
    if G is None:
        return None
    v = Function('v')(rsym)
    if n == 1:
        reduced = as_expr(v - G)
    else:
        for k in range(n - 1, 0, -1):
            G = as_expr(G.subs(v_k[k - 1], Derivative(v, (rsym, k - 1)) if k > 1 else v))
        reduced = as_expr(Derivative(v, (rsym, n - 1)) - G)
    return ReducedODE(r, s, rsym, v, reduced)


def _verified(ode: Equation, candidate: Eq, y: AppliedUndef, timeout: Optional[float]) -> bool:
    result = attempt(lambda: checkodesol(ode, candidate, y), timeout)
    return result is not None and bool(result[0])


def _solve_reduced(ode: Equation, y: AppliedUndef, reduction: ReducedODE, check: bool,
                   timeout: Optional[float]) -> list[Eq]:
    """Solve the reduced equation and go back to the original variables."""
    jets = JetSpace([y])
    ysym = jets.u[0]
    rsym, v = reduction.variable, reduction.v
    n = jets.order(jets.to_jet(_as_zero(ode)))
    if n == 1:
        v_solutions: list[Expr] = [as_expr(v - reduction.ode)]
    else:
        sols = attempt(lambda: dsolve(reduction.ode, v), timeout)
        if sols is None:
            return []
        v_solutions = [as_expr(s.rhs) for s in _solutions(sols) if s.lhs == v]
    results: list[Eq] = []
    for V in v_solutions:
        S_r = attempt(lambda: as_expr(integrate(V, rsym)), timeout)
        if S_r is None:
            S_r = as_expr(Integral(V, rsym))
        taken = free_symbols(S_r) | free_symbols(_as_zero(ode))
        [C] = _constants(taken, 1)
        implicit = Eq(reduction.s, S_r.subs(rsym, reduction.r) + C)
        candidates: list[Eq] = []
        explicit = attempt(lambda: solve(implicit, ysym), timeout) or []
        for value in explicit:
            candidates.append(Eq(y, as_expr(value).subs(ysym, y)))
        if not candidates:
            candidates.append(Eq(implicit.lhs.subs(ysym, y), implicit.rhs.subs(ysym, y)))
        for candidate in candidates:
            if check and not _verified(ode, candidate, y, timeout):
                continue
            results.append(candidate)
    return results


def dsolve_lie(ode: Equation, y: AppliedUndef, degree: int = 2, basis: Sequence[Expr] = (),
               symmetries_: Optional[Sequence[Symmetry]] = None, check: bool = True,
               timeout: Optional[float] = DEFAULT_TIMEOUT) -> list[Eq]:
    """Solve an ordinary differential equation through its point
    symmetries.

    The symmetries are computed with a polynomial ansatz of the given
    ``degree`` (times the functions of ``basis``; when no symmetry is found
    the functions occurring in the equation are tried as well) unless
    given; each is used to reduce the order of the equation, the reduced
    equation is solved by :func:`sympy.dsolve` and the solutions are
    written in the original variables. With ``check`` only the solutions
    verified by :func:`sympy.solvers.ode.checkodesol` are returned. Every
    step handed to SymPy is abandoned after ``timeout`` seconds.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import dsolve_lie
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> dsolve_lie(y.diff(x, 2) - y.diff(x)**2/y, y)
    [Eq(y(x), exp(C1*x + C2))]
    """
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
        reduction = reduce_order(ode, y, sym, timeout)
        if reduction is None:
            continue
        for sol in _solve_reduced(ode, y, reduction, check, timeout):
            if sol not in solutions:
                solutions.append(sol)
        if solutions:
            break
    return solutions


def solve_ode(ode: Equation, y: AppliedUndef, degree: int = 2, check: bool = True,
              timeout: Optional[float] = DEFAULT_TIMEOUT) -> list[Eq]:
    """Solve an ordinary differential equation with :func:`sympy.dsolve`
    and, when that fails or takes more than ``timeout`` seconds, through
    its point symmetries (:func:`dsolve_lie`).

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import solve_ode
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> solve_ode(y.diff(x, 2) - y.diff(x)**2/y - y.diff(x)/x, y)
    [Eq(y(x), exp(C1*x**2/2 + C2))]
    """
    sols = attempt(lambda: dsolve(ode, y), timeout)
    found = _solutions(sols) if sols is not None else []
    if found:
        return found
    return dsolve_lie(ode, y, degree=degree, check=check, timeout=timeout)
