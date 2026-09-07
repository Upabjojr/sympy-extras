"""Partial differential equations: point symmetries and similarity
reductions.

SymPy's :func:`sympy.solvers.pde.pdsolve` handles first order linear
equations and separation of variables. Here the point symmetries of a
partial differential equation are computed (see
:mod:`sympy_extras.solvers.lie`) and used to reduce the equation to an
ordinary differential equation for a group invariant solution
`u = \\phi(x, F(z))`, with `z` an invariant of the symmetry, which is then
handed to :func:`sympy.dsolve`.
"""
from __future__ import annotations

from typing import Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, Function, diff, expand
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polyerrors import PolynomialError
from sympy.simplify.simplify import simplify
from sympy.solvers.ode import dsolve
from sympy.solvers.pde import checkpdesol
from sympy.solvers.solvers import solve

from sympy_extras._timeout import attempt
from sympy_extras.settings import settings
from sympy_extras._typing import as_expr, free_symbols

from .lie import Equation, JetSpace, Symmetry, symmetries, _as_zero

__all__ = ['pde_symmetries', 'similarity_reduction', 'pdsolve_lie', 'Reduction']

#: the default ``timeout`` argument: the value of
#: :data:`sympy_extras.settings.settings.timeout` at call time
DEFAULT_TIMEOUT = -1.0


def _limit(timeout: Optional[float]) -> Optional[float]:
    """The time limit to use: the global setting for the default marker."""
    return settings.timeout if timeout == DEFAULT_TIMEOUT else timeout


def pde_symmetries(pde: Equation, u: AppliedUndef, degree: int = 2,
                   basis: Sequence[Expr] = (), superposition: bool = False) -> list[Symmetry]:
    """The point symmetries of a partial differential equation with
    polynomial infinitesimals of total degree at most ``degree``.

    See :func:`sympy_extras.solvers.lie.symmetries`; here the symmetries
    which only add solutions of a linear equation are dropped by default.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import pde_symmetries
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> for X in pde_symmetries(heat, u, degree=3):
    ...     print(X.generator())
    u*d/du
    d/dt
    d/dx
    x*d/dx + 2*t*d/dt
    t*d/dx - u*x/2*d/du
    t*x*d/dx + t**2*d/dt + (-t*u/2 - u*x**2/4)*d/du
    """
    found = symmetries(pde, u, degree=degree, basis=basis, superposition=superposition)
    return sorted(found, key=_simplicity)


def _simplicity(sym: Symmetry) -> tuple[int, str]:
    from sympy.core.function import count_ops
    return (int(count_ops(list(sym.xi) + list(sym.eta))), str(sym))


class Reduction:
    """A similarity reduction of a partial differential equation.

    Attributes
    ==========

    z : Expr
        The invariant of the symmetry on the independent variables (the
        similarity variable), an expression in them.
    variable : Symbol
        The symbol standing for ``z`` in the reduced equation.
    F : applied undefined function
        The unknown ``F(z)`` of the reduced equation.
    solution : Expr
        The invariant solution ``u = phi(x, F(z))``, with ``F(z)``
        written in the independent variables.
    ode : Expr
        The reduced ordinary differential equation for ``F``, an
        expression equal to zero.
    """

    def __init__(self, z: Expr, variable: Symbol, F: AppliedUndef, solution: Expr, ode: Expr) -> None:
        self.z = z
        self.variable = variable
        self.F = F
        self.solution = solution
        self.ode = ode

    def __repr__(self) -> str:
        return "Reduction(z=%s, solution=%s, ode=%s)" % (self.z, self.solution, self.ode)


def _invariant_of_flow(xi: Sequence[Expr], x: Sequence[Symbol],
                       timeout: Optional[float]) -> Optional[tuple[int, Expr]]:
    """An invariant `z(x)` of the vector field `\\sum \\xi^i \\partial_i` on
    the plane of two independent variables, and the index of the variable
    along which the characteristics are parametrized."""
    nonzero = [i for i, e in enumerate(xi) if e != 0]
    if not nonzero:
        return None
    if len(nonzero) == 1:
        i = nonzero[0]
        j = 1 - i
        return i, x[j]
    # prefer a constant xi as the parametrizing direction
    i = min(nonzero, key=lambda k: (not xi[k].is_number, str(xi[k])))
    j = 1 - i
    X = Function('X')(x[i])
    sol = attempt(lambda: dsolve(Eq(X.diff(x[i]), (xi[j]/xi[i]).subs(x[j], X)), X), timeout)
    if sol is None:
        return None
    sols = sol if isinstance(sol, list) else [sol]
    for s in sols:
        if not isinstance(s, Eq):
            continue
        constants = [c for c in free_symbols(s) if c.name.startswith('C')]
        if len(constants) != 1:
            continue
        values = attempt(lambda: solve(s, constants[0]), timeout)
        if values:
            z = as_expr(values[0]).subs(X, x[j])
            return i, as_expr(simplify(z))
    return None


def similarity_reduction(pde: Equation, u: AppliedUndef, symmetry: Symmetry,
                         timeout: Optional[float] = DEFAULT_TIMEOUT) -> Reduction:
    """Reduce a partial differential equation in two independent
    variables to an ordinary differential equation with a point symmetry.

    The invariant solution has the form `u = \\phi(x, F(z))` where `z(x)` is
    an invariant of the symmetry on the independent variables and `F` a
    new unknown function; substituting it in the equation gives an
    ordinary differential equation for `F`. The infinitesimals of the
    independent variables must not depend on `u`.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import pde_symmetries, similarity_reduction
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> scaling = [X for X in pde_symmetries(heat, u) if X.xi == (x, 2*t)][0]
    >>> r = similarity_reduction(heat, u, scaling)
    >>> r.z, r.solution
    (x/sqrt(t), F(x/sqrt(t)))
    >>> r.ode
    z*Derivative(F(z), z) + 2*Derivative(F(z), (z, 2))
    """
    timeout = _limit(timeout)
    jets = JetSpace([u])
    if jets.q != 1:
        raise ValueError("one dependent variable is expected")
    if jets.p != 2:
        raise NotImplementedError("similarity reductions are implemented for two independent variables")
    x = jets.x
    usym = jets.u[0]
    xi = symmetry.xi
    eta = symmetry.eta[0]
    if any(e.has(usym) for e in xi):
        raise NotImplementedError("the infinitesimals of the independent variables depend on %s" % usym)
    if all(e == 0 for e in xi):
        raise NotImplementedError("the symmetry moves no independent variable: no similarity variable")
    found = _invariant_of_flow(xi, x, timeout)
    if found is None:
        raise NotImplementedError("no invariant of %s was found" % (symmetry,))
    i, z = found
    j = 1 - i
    zsym = Symbol('z')
    F = Function('F')
    Fz = F(zsym)
    # the invariant involving u: integrate du/dx_i = eta/xi^i along the characteristics
    Fzx = F(z)
    if eta == 0:
        phi = as_expr(Fzx)
    else:
        U = Function('U')(x[i])
        if xi[i] == 0:
            raise NotImplementedError("the characteristic direction has a vanishing infinitesimal")
        # x_j on the characteristic z = const
        xj_values = attempt(lambda: solve(Eq(z, zsym), x[j]), timeout)
        if not xj_values:
            raise NotImplementedError("cannot solve the invariant %s for %s" % (z, x[j]))
        xj = as_expr(xj_values[0])
        rhs = (eta/xi[i]).subs({usym: U, x[j]: xj})
        sol = attempt(lambda: dsolve(Eq(U.diff(x[i]), rhs), U), timeout)
        if sol is None:
            raise NotImplementedError("cannot integrate the characteristic equation for %s" % usym)
        sols = sol if isinstance(sol, list) else [sol]
        phi_found: Optional[Expr] = None
        for s in sols:
            constants = [c for c in free_symbols(s) if c.name.startswith('C')]
            if not isinstance(s, Eq) or len(constants) != 1:
                continue
            expr = as_expr(s.rhs.subs(constants[0], Fzx).subs(x[j], xj).subs(zsym, z))
            phi_found = expr
            break
        if phi_found is None:
            raise NotImplementedError("cannot integrate the characteristic equation for %s" % usym)
        phi = phi_found
    # substitute u = phi(x, F(z(x))) into the equation, with F_k the derivatives of F
    order = jets.order(jets.to_jet(_as_zero(pde)))
    Fk = [Symbol('F_%d' % k) for k in range(order + 1)]
    ansatz = as_expr(phi.subs(Fzx, Fk[0]))
    dz = [diff(z, v) for v in x]

    def total(e: Expr, m: int) -> Expr:
        result = as_expr(diff(e, x[m]))
        for k in range(order):
            result += diff(e, Fk[k])*Fk[k + 1]*dz[m]
        return as_expr(result)

    values: dict[Basic, Basic] = {usym: ansatz}
    for a, J in jets.jets_in(jets.to_jet(_as_zero(pde))):
        e = ansatz
        for m, count in enumerate(J):
            for _ in range(count):
                e = total(e, m)
        values[jets.jet(a, J)] = e
    reduced = as_expr(jets.to_jet(_as_zero(pde)).xreplace(values))
    # eliminate the independent variables in favour of z: the reduced
    # expression is an ordinary differential equation in F up to a factor
    ode = _eliminate(reduced, Fk, x, (j, i), z, zsym, timeout)
    if ode is None:
        raise NotImplementedError("the reduced equation still depends on the independent variables")
    for k in range(order, -1, -1):
        ode = as_expr(ode.subs(Fk[k], Derivative(Fz, (zsym, k)) if k else Fz))
    # remove an overall factor
    from sympy.polys.polytools import factor_list
    try:
        _, factors = factor_list(ode, *[Derivative(Fz, (zsym, k)) for k in range(order, 0, -1)], Fz)
        relevant = [f**m for f, m in factors if f.has(Fz)]
        if relevant:
            ode = as_expr(expand(S.One*relevant[0]) if len(relevant) == 1 else ode)
    except PolynomialError:
        pass
    return Reduction(z, zsym, Fz, phi, ode)


def _eliminate(reduced: Expr, Fk: Sequence[Symbol], x: Sequence[Symbol], order: Sequence[int],
               z: Expr, zsym: Symbol, timeout: Optional[float]) -> Optional[Expr]:
    """Substitute one independent variable by its value in terms of ``z``
    and the other, divide by the coefficient of the highest derivative and
    check that the other variable disappears."""
    from sympy.polys.polytools import Poly
    try:
        poly = Poly(reduced, *Fk)
    except PolynomialError:
        return None
    terms = poly.terms()
    for k in order:
        sols_k = attempt(lambda: solve(Eq(z, zsym), x[k]), timeout) or []
        for value in sols_k:
            coefficients = [as_expr(c).subs(x[k], value) for _, c in terms]
            leading = coefficients[0]
            if leading == 0:
                continue
            normalized: list[Expr] = []
            for c in coefficients:
                e = attempt(lambda: as_expr(simplify(c/leading)), timeout)
                if e is None or free_symbols(e) & set(x):
                    normalized = []
                    break
                normalized.append(e)
            if normalized:
                result = S.Zero
                for (monomial, _), c in zip(terms, normalized):
                    term = c
                    for f, e in zip(Fk, monomial):
                        term *= f**e
                    result += term
                return as_expr(result)
    return None


def pdsolve_lie(pde: Equation, u: AppliedUndef, symmetries_: Optional[Sequence[Symmetry]] = None,
                degree: int = 2, check: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT) -> list[Eq]:
    """Group invariant solutions of a partial differential equation.

    Every point symmetry (those given, or the ones found with
    ``degree``) is used to reduce the equation to an ordinary differential
    equation for `F(z)` which :func:`sympy.dsolve` tries to solve; the
    solutions which :func:`sympy.solvers.pde.checkpdesol` verifies (when
    ``check`` is true) are returned as ``Eq(u(x, t), ...)``. Every step
    handed to SymPy is abandoned after ``timeout`` seconds.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers import pdsolve_lie
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> for s in pdsolve_lie(u.diff(t) - u.diff(x, 2), u):
    ...     print(s)
    Eq(u(x, t), C1 + C2*x)
    Eq(u(x, t), C1)
    Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
    Eq(u(x, t), C1*exp(-x**2/(4*t))/sqrt(t))
    """
    timeout = _limit(timeout)
    syms = list(symmetries_) if symmetries_ is not None else pde_symmetries(pde, u, degree=degree)
    solutions: list[Eq] = []
    seen: set[Basic] = set()
    for sym in syms:
        try:
            reduction = similarity_reduction(pde, u, sym, timeout)
        except NotImplementedError:
            continue
        ode_solutions = attempt(lambda: dsolve(reduction.ode, reduction.F), timeout)
        if ode_solutions is None:
            continue
        for s in (ode_solutions if isinstance(ode_solutions, list) else [ode_solutions]):
            if not isinstance(s, Eq) or s.lhs != reduction.F:
                continue
            F_value = as_expr(s.rhs.subs(reduction.variable, reduction.z))
            value = as_expr(reduction.solution.subs(reduction.F.func(reduction.z), F_value))
            if value in seen:
                continue
            seen.add(value)
            candidate = Eq(u, value)
            if check:
                verdict = attempt(lambda: checkpdesol(pde, candidate, u), timeout)
                if verdict is None or not verdict[0]:
                    continue
            solutions.append(candidate)
    return solutions
