"""Partial differential equations: point symmetries and similarity
reductions.

SymPy's :func:`sympy.solvers.pde.pdsolve` handles first order linear
equations and separation of variables. Here the point symmetries of a
partial differential equation are computed (see
:mod:`sympy_extras.solvers.lie`) and used to reduce the equation to an
ordinary differential equation for a group invariant solution
`u = \\phi(x, F(z))`, with `z` an invariant of the symmetry, which is then
handed to an ODE solver.

The module imports no SymPy: the equations, the symmetries and the
reductions are objects of the core (:mod:`sympy_extras.core`); the
reduced equation is normalised by the native polynomial factorisation,
and the solutions are verified by substitution with the native
:func:`~sympy_extras.simplify.simplify.simplify`, and the characteristic
equations and the reduced equations are solved by the native classical
hints of :func:`~sympy_extras.solvers.dsolve.dsolve`. One step has no
native counterpart and is taken from SymPy at run time through the face
:mod:`sympy_extras.solvers.pde_sympy`: SymPy's ``solve`` for the
equations which are not linear in the unknown (``solve_of_core``). The face also keeps the old API on SymPy
objects.
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy_extras._timeout import attempt
from sympy_extras.core.add import add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.function import AppliedUndef, Derivative, Function, count_ops, diff, expand
from sympy_extras.core.numbers import Integer
from sympy_extras.core.relational import Eq
from sympy_extras.core.symbol import Symbol
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polytools import factor_list
from sympy_extras.settings import settings
from sympy_extras.simplify.simplify import simplify

from .lie import Equation, JetSpace, Symmetry, symmetries, _as_zero, _polynomial_terms, _replaced

__all__ = ['pde_symmetries', 'similarity_reduction', 'pdsolve_lie', 'Reduction', 'check_pde_solution']

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
    which only add solutions of a linear equation are dropped by default,
    and the symmetries are sorted by the number of operations.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.pde import pde_symmetries
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
    ops = count_ops(list(sym.xi) + list(sym.eta))
    if not isinstance(ops, Integer):
        raise TypeError("expected an integer, got %s" % (ops,))
    return (ops.p, str(sym))


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
        self.z: Final[Expr] = z
        self.variable: Final[Symbol] = variable
        self.F: Final[AppliedUndef] = F
        self.solution: Final[Expr] = solution
        self.ode: Final[Expr] = ode

    def __repr__(self) -> str:
        return "Reduction(z=%s, solution=%s, ode=%s)" % (self.z, self.solution, self.ode)


# ---------------------------------------------------------------------------
# the steps which stay on SymPy

def _dsolve(equation: Basic, f: AppliedUndef, timeout: Optional[float]) -> Optional[list[Eq]]:
    """The solutions of an ODE by the native classical methods
    (:func:`~sympy_extras.solvers.dsolve.dsolve`) under the time limit
    (``None`` when it gives up)."""
    from .dsolve import dsolve
    return attempt(lambda: dsolve(equation, f), timeout, native=True)


def _solve(e: Expr, v: Symbol, timeout: Optional[float]) -> list[Expr]:
    """The solutions of ``e = 0`` for ``v``: natively when ``e`` is linear
    in ``v``, by SymPy's ``solve`` otherwise (none when it gives up)."""
    a = diff(e, v)
    if a != 0 and not a.has(v):
        b = e.subs(v, Integer(0))
        if isinstance(b, Expr) and not b.has(v):
            return [-b/a]
    from .pde_sympy import solve_of_core
    found = attempt(lambda: solve_of_core(e, v), timeout)
    return found if found is not None else []


def _constants(e: Basic) -> list[Symbol]:
    """The integration constants ``C1, C2, ...`` of a solution."""
    return sorted((c for c in e.free_symbols if c.name.startswith('C')), key=lambda c: c.name)


# ---------------------------------------------------------------------------
# similarity reductions

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
    sols = _dsolve(Eq(diff(X, x[i]), (xi[j]/xi[i]).subs(x[j], X)), X, timeout)
    if sols is None:
        return None
    for s in sols:
        constants = _constants(s)
        if len(constants) != 1:
            continue
        values = _solve(s.lhs - s.rhs, constants[0], timeout)
        if values:
            z = values[0].subs(X, x[j])
            assert isinstance(z, Expr)
            return i, simplify(z)
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

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry
    >>> from sympy_extras.solvers.pde import similarity_reduction
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> r = similarity_reduction(heat, u, Symmetry(JetSpace([u]), [0, 1], [0]))
    >>> r.z, r.solution, r.ode
    (x, F(x), Derivative(F(z), (z, 2)))
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
        phi: Expr = Fzx
    else:
        U = Function('U')(x[i])
        if xi[i] == 0:
            raise NotImplementedError("the characteristic direction has a vanishing infinitesimal")
        # x_j on the characteristic z = const
        xj_values = _solve(z - zsym, x[j], timeout)
        if not xj_values:
            raise NotImplementedError("cannot solve the invariant %s for %s" % (z, x[j]))
        xj = xj_values[0]
        rhs = (eta/xi[i]).subs({usym: U, x[j]: xj})
        sols = _dsolve(Eq(diff(U, x[i]), rhs), U, timeout)
        if sols is None:
            raise NotImplementedError("cannot integrate the characteristic equation for %s" % usym)
        phi_found: Optional[Expr] = None
        for s in sols:
            constants = _constants(s)
            if len(constants) != 1:
                continue
            expr = s.rhs.subs(constants[0], Fzx).subs(x[j], xj).subs(zsym, z)
            assert isinstance(expr, Expr)
            phi_found = expr
            break
        if phi_found is None:
            raise NotImplementedError("cannot integrate the characteristic equation for %s" % usym)
        phi = phi_found
    # substitute u = phi(x, F(z(x))) into the equation, with F_k the derivatives of F
    equation = jets.to_jet(_as_zero(pde))
    order = jets.order(equation)
    Fk = [Symbol('F_%d' % k) for k in range(order + 1)]
    ansatz = phi.subs(Fzx, Fk[0])
    dz = [diff(z, v) for v in x]

    def total(e: Expr, m: int) -> Expr:
        result = diff(e, x[m])
        for k in range(order):
            result = result + diff(e, Fk[k])*Fk[k + 1]*dz[m]
        return result

    values: dict[Basic, Basic] = {usym: ansatz}
    for a, J in jets.jets_in(equation):
        e = ansatz
        for m, count in enumerate(J):
            for _ in range(count):
                e = total(e, m)
        values[jets.jet(a, J)] = e
    reduced = _replaced(equation, values)
    # eliminate the independent variables in favour of z: the reduced
    # expression is an ordinary differential equation in F up to a factor
    ode = _eliminate(reduced, Fk, x, (j, i), z, zsym, timeout)
    if ode is None:
        raise NotImplementedError("the reduced equation still depends on the independent variables")
    # remove an overall factor (SymPy's factor_list in the derivatives of F,
    # the highest first)
    derivatives: dict[Basic, Basic] = {Fk[k]: Derivative(Fz, (zsym, k)) if k else Fz for k in range(order + 1)}
    try:
        _, factors = factor_list(ode, *[Fk[k] for k in range(order, 0, -1)], Fk[0])
        relevant = [f**m for f, m in factors if any(f.has(g) for g in Fk)]
        if len(relevant) == 1:
            ode = expand(relevant[0])
    except BasePolynomialError:
        pass
    ode = _replaced(ode, derivatives)
    return Reduction(z, zsym, Fz, phi, ode)


def _eliminate(reduced: Expr, Fk: Sequence[Symbol], x: Sequence[Symbol], order: Sequence[int],
               z: Expr, zsym: Symbol, timeout: Optional[float]) -> Optional[Expr]:
    """Substitute one independent variable by its value in terms of ``z``
    and the other, divide by the coefficient of the highest derivative and
    check that the other variable disappears."""
    terms = _polynomial_terms(reduced, Fk)
    if terms is None:
        return None
    monomials = list(terms)
    for k in order:
        for value in _solve(z - zsym, x[k], timeout):
            coefficients: list[Expr] = []
            for c in terms.values():
                substituted = c.subs(x[k], value)
                assert isinstance(substituted, Expr)
                coefficients.append(substituted)
            leading = coefficients[0]
            if leading == 0:
                continue
            normalized: list[Expr] = []
            for c in coefficients:
                e = attempt(lambda: simplify(c/leading), timeout, native=True)
                if e is None or e.free_symbols & set(x):
                    normalized = []
                    break
                normalized.append(e)
            if normalized:
                result: list[Expr] = []
                for monomial, c in zip(monomials, normalized):
                    term = c
                    for f, n in zip(Fk, monomial):
                        term = term*f**n
                    result.append(term)
                return add(*result)
    return None


def check_pde_solution(pde: Equation, u: AppliedUndef, value: Expr, timeout: Optional[float] = DEFAULT_TIMEOUT
                       ) -> bool:
    """Whether ``u = value`` solves the partial differential equation:
    the derivatives are substituted and the result simplified to zero
    within the time limit (``False`` when it is not proven).

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.functions import exp
    >>> from sympy_extras.solvers.pde import check_pde_solution
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> check_pde_solution(heat, u, exp(-t)*exp(x)), check_pde_solution(heat, u, exp(-t)*exp(-x))
    (False, False)
    >>> check_pde_solution(heat, u, exp(t)*exp(x)), check_pde_solution(heat, u, x**2 + 2*t)
    (True, True)
    """
    timeout = _limit(timeout)
    equation = _as_zero(pde)
    replacements: dict[Basic, Basic] = {}
    for d in equation.atoms(Derivative):
        if isinstance(d, Derivative) and d.expr == u:
            replacements[d] = diff(value, *d.variables)
    replacements[u] = value
    substituted = _replaced(equation, replacements)
    if substituted == 0:
        return True
    residual = attempt(lambda: simplify(substituted), timeout, native=True)
    return residual is not None and residual == 0


def pdsolve_lie(pde: Equation, u: AppliedUndef, symmetries_: Optional[Sequence[Symmetry]] = None,
                degree: int = 2, check: bool = True, timeout: Optional[float] = DEFAULT_TIMEOUT) -> list[Eq]:
    """Group invariant solutions of a partial differential equation.

    Every point symmetry (those given, or the ones found with
    ``degree``) is used to reduce the equation to an ordinary differential
    equation for `F(z)` which SymPy's ``dsolve`` tries to solve; the
    solutions which :func:`check_pde_solution` verifies (when ``check`` is
    true) are returned as ``Eq(u(x, t), ...)``. Every step is abandoned
    after ``timeout`` seconds.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry
    >>> from sympy_extras.solvers.pde import pdsolve_lie
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> pdsolve_lie(u.diff(t) - u.diff(x, 2), u, [Symmetry(JetSpace([u]), [0, 1], [0])])
    [Eq(u(x, t), C1 + C2*x)]
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
        ode_solutions = _dsolve(reduction.ode, reduction.F, timeout)
        if ode_solutions is None:
            continue
        for s in ode_solutions:
            if s.lhs != reduction.F:
                continue
            F_value = s.rhs.subs(reduction.variable, reduction.z)
            assert isinstance(F_value, Expr)
            value = reduction.solution.subs(reduction.F.function(reduction.z), F_value)
            assert isinstance(value, Expr)
            if value in seen:
                continue
            seen.add(value)
            if check and not check_pde_solution(pde, u, value, timeout):
                continue
            solutions.append(Eq(u, value))
    return solutions
