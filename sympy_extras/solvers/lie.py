"""Lie point symmetries of ordinary and partial differential equations.

A point symmetry of a system of differential equations for the dependent
variables `u^a(x_1, \\ldots, x_p)` is a vector field

.. math:: X = \\sum_i \\xi^i(x, u) \\partial_{x_i}
             + \\sum_a \\eta^a(x, u) \\partial_{u^a}

whose prolongation to the jet space annihilates the equations on their
solutions. The condition, once the leading derivatives are eliminated,
must hold identically in the remaining jet coordinates and splits into a
linear system of partial differential equations for `\\xi` and `\\eta`, the
*determining equations*. They are solved here with an ansatz: `\\xi` and
`\\eta` are polynomials in `x` and `u` of bounded total degree, optionally
multiplied by given functions, so that the determining equations become a
linear system for the coefficients which is solved exactly. Every symmetry
found is a symmetry (the invariance condition is verified by substitution,
see :func:`check_symmetry`); symmetries outside the ansatz are missed.

The prolongation formula is the recursive one (Olver, *Applications of Lie
Groups to Differential Equations*, (2.39)):

.. math:: \\eta^a_{J, i} = D_i \\eta^a_J - \\sum_k u^a_{J, k} D_i \\xi^k .

References
==========

.. [Olver] P. J. Olver, Applications of Lie Groups to Differential
   Equations, 2nd ed., Springer 1993, chapter 2.
.. [BlumanAnco] G. W. Bluman, S. C. Anco, Symmetry and Integration Methods
   for Differential Equations, Springer 2002.
.. [Hereman] W. Hereman, Review of symbolic software for Lie symmetry
   analysis, Math. Comput. Modelling 25 (1997).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, Derivative, diff, expand, expand_power_exp
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.power import Pow
from sympy.core.symbol import Symbol, Dummy
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.matrices import DomainMatrix
from sympy.polys.monomials import itermonomials
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import numer
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.solvers.solveset import linear_eq_to_matrix
from sympy.solvers.solvers import solve

from sympy_extras._typing import Monomial, as_expr, free_symbols

__all__ = ['JetSpace', 'Symmetry', 'symmetries', 'check_symmetry',
    'infinitesimal_condition', 'determining_equations']

#: an expression of a differential equation: ``Eq`` or expression ``= 0``
Equation = Union[Expr, Eq]


def _as_zero(eq: Equation) -> Expr:
    """The expression which is zero."""
    if isinstance(eq, Eq):
        return as_expr(eq.lhs - eq.rhs)
    return as_expr(eq)


class JetSpace:
    """Jet coordinates of dependent variables.

    Parameters
    ==========

    functions : sequence of applied undefined functions
        The dependent variables, ``u(x, t)``, all with the same
        arguments, the independent variables.

    The derivative `u^a_J` with multi-index `J` (a tuple of counts, one
    per independent variable) is the symbol ``u_xxt``; ``u`` itself is the
    symbol ``u``.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> jets.to_jet(u.diff(t) - u.diff(x, 2))
    u_t - u_xx
    >>> jets.total_derivative(jets.u[0]**2, 0)
    2*u*u_x
    """

    def __init__(self, functions: Sequence[AppliedUndef]) -> None:
        if not functions:
            raise ValueError("at least one dependent variable is needed")
        args = functions[0].args
        for f in functions:
            if f.args != args:
                raise ValueError("the dependent variables must have the same arguments")
        independent: list[Symbol] = []
        for a in args:
            if not isinstance(a, Symbol):
                raise TypeError("the arguments of the dependent variables must be symbols")
            independent.append(a)
        self.x: tuple[Symbol, ...] = tuple(independent)
        self.functions: tuple[AppliedUndef, ...] = tuple(functions)
        self.u: tuple[Symbol, ...] = tuple(Symbol(f.func.__name__) for f in functions)
        self._jets: dict[tuple[int, Monomial], Symbol] = {}
        self._inverse: dict[Symbol, tuple[int, Monomial]] = {}
        zero = (0,)*len(self.x)
        for index, symbol in enumerate(self.u):
            self._jets[(index, zero)] = symbol
            self._inverse[symbol] = (index, zero)

    @property
    def p(self) -> int:
        """Number of independent variables."""
        return len(self.x)

    @property
    def q(self) -> int:
        """Number of dependent variables."""
        return len(self.u)

    def jet(self, a: int, J: Monomial) -> Symbol:
        """The jet coordinate `u^a_J`."""
        key = (a, tuple(J))
        if key not in self._jets:
            suffix = ''.join(x.name*k for x, k in zip(self.x, J))
            s = Symbol('%s_%s' % (self.u[a].name, suffix))
            self._jets[key] = s
            self._inverse[s] = key
        return self._jets[key]

    def index(self, s: Symbol) -> Optional[tuple[int, Monomial]]:
        """The ``(a, J)`` of a jet coordinate, or ``None``. Jet
        coordinates are identified by their names, so that the symbols of
        another :class:`JetSpace` of the same variables are recognised."""
        if s in self._inverse:
            return self._inverse[s]
        parsed = self._parse(s.name)
        if parsed is not None:
            a, J = parsed
            self._jets[(a, J)] = s
            self._inverse[s] = (a, J)
        return parsed

    def _parse(self, name: str) -> Optional[tuple[int, Monomial]]:
        names = sorted(self.x, key=lambda v: len(v.name), reverse=True)
        for a, u in enumerate(self.u):
            if not name.startswith(u.name + '_'):
                continue
            suffix = name[len(u.name) + 1:]
            counts = [0]*self.p
            while suffix:
                for v in names:
                    if suffix.startswith(v.name):
                        counts[self.x.index(v)] += 1
                        suffix = suffix[len(v.name):]
                        break
                else:
                    return None
            if sum(counts) > 0:
                return a, tuple(counts)
        return None

    def jets_in(self, expr: Basic) -> list[tuple[int, Monomial]]:
        """The jet coordinates occurring in ``expr``."""
        found: list[tuple[int, Monomial]] = []
        for s in free_symbols(expr):
            aJ = self.index(s)
            if aJ is not None:
                found.append(aJ)
        return sorted(found)

    def order(self, expr: Basic) -> int:
        """The highest order of a jet coordinate occurring in ``expr``."""
        return max((sum(J) for _, J in self.jets_in(expr)), default=0)

    def to_jet(self, expr: Basic) -> Expr:
        """Rewrite the derivatives of the dependent variables as jet
        coordinates."""
        replacements: dict[Basic, Basic] = {}
        for d in expr.atoms(Derivative):
            if d.expr in self.functions:
                a = self.functions.index(d.expr)
                counts = [0]*self.p
                for v, n in d.variable_count:
                    if v not in self.x:
                        raise ValueError("derivative with respect to %s" % (v,))
                    counts[self.x.index(v)] += int(n)
                replacements[d] = self.jet(a, tuple(counts))
        for a, f in enumerate(self.functions):
            replacements[f] = self.u[a]
        return as_expr(expr.xreplace(replacements))

    def from_jet(self, expr: Basic) -> Expr:
        """Rewrite the jet coordinates as derivatives of the functions."""
        replacements: dict[Basic, Basic] = {}
        for s, (a, J) in self._inverse.items():
            if sum(J) == 0:
                replacements[s] = self.functions[a]
            else:
                variables = [x for x, k in zip(self.x, J) for _ in range(k)]
                replacements[s] = self.functions[a].diff(*variables)
        return as_expr(expr.xreplace(replacements))

    def total_derivative(self, expr: Expr, i: int) -> Expr:
        """The total derivative `D_i` of an expression in the jet
        coordinates."""
        result = as_expr(diff(expr, self.x[i]))
        for a, J in self.jets_in(expr):
            partial = diff(expr, self.jet(a, J))
            if partial != 0:
                next_J = tuple(k + (1 if j == i else 0) for j, k in enumerate(J))
                result += self.jet(a, next_J)*partial
        return result

    def monomials(self, degree: int) -> list[Expr]:
        """The monomials in the independent and dependent variables of
        total degree at most ``degree``."""
        variables = list(self.x) + list(self.u)
        return sorted(itermonomials(variables, degree), key=lambda m: (Poly(m, *variables).total_degree(), str(m)))


class Symmetry:
    """A vector field `X = \\sum \\xi^i \\partial_{x_i} + \\sum \\eta^a
    \\partial_{u^a}` on the space of the independent and dependent
    variables, given by its coefficients as expressions in the symbols of
    the :class:`JetSpace`.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> X = Symmetry(jets, [2*t, 0], [-x*jets.u[0]])
    >>> X
    Symmetry(xi=(2*t, 0), eta=(-u*x,))
    >>> print(X.generator())
    2*t*d/dx - u*x*d/du
    >>> X.characteristic()
    (-2*t*u_x - u*x,)
    """

    def __init__(self, jets: JetSpace, xi: Sequence[Union[Expr, int]],
                 eta: Sequence[Union[Expr, int]]) -> None:
        if len(xi) != jets.p or len(eta) != jets.q:
            raise ValueError("expected %d xi and %d eta" % (jets.p, jets.q))
        self.jets = jets
        self.xi: tuple[Expr, ...] = tuple(as_expr(e) for e in xi)
        self.eta: tuple[Expr, ...] = tuple(as_expr(e) for e in eta)

    def __repr__(self) -> str:
        return "Symmetry(xi=%s, eta=%s)" % (self.xi, self.eta)

    def generator(self) -> str:
        """The vector field written with ``d/dx`` and ``d/du``."""
        terms: list[str] = []
        for c, v in list(zip(self.xi, self.jets.x)) + list(zip(self.eta, self.jets.u)):
            if c == 0:
                continue
            if c == 1:
                term = "d/d%s" % v.name
            elif isinstance(c, Add):
                term = "(%s)*d/d%s" % (c, v.name)
            else:
                term = "%s*d/d%s" % (c, v.name)
            terms.append(term)
        if not terms:
            return "0"
        text = " + ".join(terms)
        return text.replace("+ -", "- ")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symmetry):
            return NotImplemented
        return self.xi == other.xi and self.eta == other.eta

    def __hash__(self) -> int:
        return hash((self.xi, self.eta))

    def is_zero(self) -> bool:
        return all(c == 0 for c in self.xi + self.eta)

    # ------------------------------------------------------------------
    # linear combinations

    def __add__(self, other: Symmetry) -> Symmetry:
        return Symmetry(self.jets, [a + b for a, b in zip(self.xi, other.xi)],
                        [a + b for a, b in zip(self.eta, other.eta)])

    def __sub__(self, other: Symmetry) -> Symmetry:
        return self + (-other)

    def __neg__(self) -> Symmetry:
        return Symmetry(self.jets, [-a for a in self.xi], [-a for a in self.eta])

    def __mul__(self, c: Union[Expr, int]) -> Symmetry:
        c_ = as_expr(c)
        return Symmetry(self.jets, [c_*a for a in self.xi], [c_*a for a in self.eta])

    __rmul__ = __mul__

    # ------------------------------------------------------------------
    # prolongation

    def apply_point(self, f: Expr) -> Expr:
        """`X f` for a function of the independent and dependent
        variables."""
        result = S.Zero
        for c, v in list(zip(self.xi, self.jets.x)) + list(zip(self.eta, self.jets.u)):
            result += c*diff(f, v)
        return as_expr(result)

    def commutator(self, other: Symmetry) -> Symmetry:
        """The Lie bracket `[X, Y]`."""
        xi = [self.apply_point(b) - other.apply_point(a) for a, b in zip(self.xi, other.xi)]
        eta = [self.apply_point(b) - other.apply_point(a) for a, b in zip(self.eta, other.eta)]
        return Symmetry(self.jets, [expand(e) for e in xi], [expand(e) for e in eta])

    def characteristic(self) -> tuple[Expr, ...]:
        """The characteristic `Q^a = \\eta^a - \\sum_i \\xi^i u^a_i`."""
        jets = self.jets
        result: list[Expr] = []
        for a in range(jets.q):
            q = self.eta[a]
            for i in range(jets.p):
                e_i = tuple(1 if j == i else 0 for j in range(jets.p))
                q -= self.xi[i]*jets.jet(a, e_i)
            result.append(as_expr(q))
        return tuple(result)

    def prolongation(self, order: int) -> dict[tuple[int, Monomial], Expr]:
        """The coefficients `\\eta^a_J` of the prolonged vector field for
        all multi-indices with `|J| \\le` ``order``."""
        jets = self.jets
        zero = (0,)*jets.p
        coefficients: dict[tuple[int, Monomial], Expr] = {}
        d_xi = [[jets.total_derivative(self.xi[k], i) for k in range(jets.p)] for i in range(jets.p)]
        for a in range(jets.q):
            coefficients[(a, zero)] = self.eta[a]
        for n in range(order):
            for (a, J), phi in list(coefficients.items()):
                if sum(J) != n:
                    continue
                for i in range(jets.p):
                    # multi-indices are built in nondecreasing variable order
                    if any(J[j] > 0 for j in range(i + 1, jets.p)):
                        continue
                    J_i = tuple(k + (1 if j == i else 0) for j, k in enumerate(J))
                    if (a, J_i) in coefficients:
                        continue
                    value = jets.total_derivative(phi, i)
                    for k in range(jets.p):
                        J_k = tuple(c + (1 if j == k else 0) for j, c in enumerate(J))
                        value -= jets.jet(a, J_k)*d_xi[i][k]
                    coefficients[(a, J_i)] = as_expr(expand(value))
        return coefficients

    def apply(self, expr: Expr) -> Expr:
        """`\\mathrm{pr} X` applied to an expression in the jet
        coordinates."""
        jets = self.jets
        order = jets.order(expr)
        coefficients = self.prolongation(order)
        result = S.Zero
        for c, v in zip(self.xi, jets.x):
            result += c*diff(expr, v)
        for (a, J), phi in coefficients.items():
            partial = diff(expr, jets.jet(a, J))
            if partial != 0:
                result += phi*partial
        return as_expr(expand(result))


# ---------------------------------------------------------------------------
# the invariance condition

def _prepare(equations: Union[Equation, Sequence[Equation]],
             functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> tuple[JetSpace, list[Expr]]:
    eqs = [equations] if isinstance(equations, (Expr, Eq)) else list(equations)
    funcs = [functions] if isinstance(functions, AppliedUndef) else list(functions)
    jets = JetSpace(funcs)
    return jets, [jets.to_jet(_as_zero(e)) for e in eqs]


def _leading_derivatives(jets: JetSpace, eqs: list[Expr]) -> dict[Symbol, Expr]:
    """Solve every equation for its highest derivative, preferring one it
    is linear in."""
    solved: dict[Symbol, Expr] = {}
    for eq in eqs:
        candidates = sorted(jets.jets_in(eq), key=lambda aJ: (sum(aJ[1]), aJ[1], aJ[0]), reverse=True)
        top = sum(candidates[0][1]) if candidates else 0
        found = False
        for a, J in candidates:
            if sum(J) < top:
                break
            s = jets.jet(a, J)
            if s in solved:
                continue
            if diff(eq, s, 2) == 0:
                coefficient = diff(eq, s)
                solved[s] = as_expr(expand_power_exp(-(eq - coefficient*s)/coefficient))
                found = True
                break
        if not found:
            for a, J in candidates:
                s = jets.jet(a, J)
                if s in solved:
                    continue
                roots = solve(eq, s)
                if roots:
                    solved[s] = as_expr(roots[0])
                    found = True
                    break
        if not found:
            raise NotImplementedError("cannot solve %s for a leading derivative" % (jets.from_jet(eq),))
    return solved


def infinitesimal_condition(equations: Union[Equation, Sequence[Equation]],
                            functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                            symmetry: Symmetry) -> list[Expr]:
    """`\\mathrm{pr} X(\\Delta)` restricted to the solutions of the
    equations: the leading derivatives are eliminated and the numerators
    are returned, expressions in the jet coordinates which vanish
    identically exactly when ``symmetry`` is a symmetry."""
    jets, eqs = _prepare(equations, functions)
    return _condition(jets, eqs, symmetry)


def _condition(jets: JetSpace, eqs: list[Expr], symmetry: Symmetry) -> list[Expr]:
    leading = _leading_derivatives(jets, eqs)
    result: list[Expr] = []
    for eq in eqs:
        condition = symmetry.apply(eq)
        for _ in range(len(leading)):
            new = as_expr(condition.xreplace(leading))
            if new == condition:
                break
            condition = new
        condition = as_expr(expand_power_exp(condition))
        result.append(as_expr(numer(ratsimp(condition)) if condition.is_rational_function(*jets.x, *jets.u) else numer(condition.together())))
    return result


def check_symmetry(equations: Union[Equation, Sequence[Equation]],
                   functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                   symmetry: Symmetry) -> bool:
    """Whether the vector field is a point symmetry of the equations, by
    substitution in the invariance condition.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry, check_symmetry
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> heat = u.diff(t) - u.diff(x, 2)
    >>> jets = JetSpace([u])
    >>> check_symmetry(heat, u, Symmetry(jets, [2*t, 0], [-x*jets.u[0]]))
    True
    >>> check_symmetry(heat, u, Symmetry(jets, [t, 0], [0]))
    False
    """
    for condition in infinitesimal_condition(equations, functions, symmetry):
        if condition != 0 and simplify(condition) != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# solving the determining equations with an ansatz

def _symbolic_powers(expr: Expr, unknowns: Sequence[Symbol],
                     powers: dict[tuple[Basic, Basic], Symbol]) -> Expr:
    """Powers with an exponent linear in parameters, ``u**(2*m + 1)``,
    are written with one symbol per ``base**parameter``, ``P**2*u``, so
    that the polynomial splitting treats them consistently; the symbols
    are collected in ``powers``."""
    replacements: dict[Basic, Basic] = {}
    for power in expr.atoms(Pow):
        base, exponent = power.args
        parameters = sorted(free_symbols(exponent) - set(unknowns), key=lambda s: s.name)
        if not parameters:
            continue
        try:
            poly = Poly(exponent, *parameters)
        except PolynomialError:
            continue
        if poly.total_degree() > 1:
            continue
        value: Expr = base**poly.coeff_monomial(1)
        for parameter in parameters:
            c = poly.coeff_monomial(parameter)
            if c == 0:
                continue
            key = (base, parameter)
            if key not in powers:
                powers[key] = Dummy('P%d' % len(powers))
            value *= powers[key]**c
        replacements[power] = value
    return as_expr(expr.xreplace(replacements)) if replacements else expr


def _split(expr: Expr, unknowns: Sequence[Symbol], variables: set[Symbol]) -> list[Expr]:
    """The coefficients of ``expr`` as a polynomial in the ``variables``
    (the independent and dependent variables and the jet coordinates) and
    in the functions of them: expressions in the unknowns and the
    parameters, linear in the unknowns, which must all vanish."""
    expr = as_expr(expand(expr))
    if expr == 0:
        return []
    powers: dict[tuple[Basic, Basic], Symbol] = {}
    with_powers = _symbolic_powers(expr, unknowns, powers)
    if with_powers != expr:
        expr = as_expr(expand(numer(with_powers.together())))
        variables = variables | set(powers.values())
    try:
        p = Poly(expr)
    except PolynomialError:
        return [expr]
    gens = [g for g in p.gens if (free_symbols(g) & variables) and not (free_symbols(g) & set(unknowns))]
    if not gens:
        return [expr]
    try:
        return [as_expr(c) for c in Poly(expr, *gens).coeffs()]
    except PolynomialError:
        return [expr]


def determining_equations(equations: Union[Equation, Sequence[Equation]],
                          functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                          degree: int = 2, basis: Sequence[Expr] = ()
                          ) -> tuple[JetSpace, list[Symbol], list[Expr], list[Expr], list[Expr]]:
    """The determining equations for the ansatz.

    Returns the jet space, the unknown coefficients, the ansatz for
    `\\xi^i` and `\\eta^a` (polynomials in the independent and dependent
    variables of total degree at most ``degree``, times each function of
    ``basis``, with the unknown coefficients) and the linear equations the
    coefficients must satisfy.
    """
    jets, eqs = _prepare(equations, functions)
    monomials = jets.monomials(degree)
    factors: list[Expr] = [S.One] + [as_expr(b) for b in basis]
    terms = [m*b for b in factors for m in monomials]
    unknowns: list[Symbol] = []
    counter = 0

    def ansatz() -> Expr:
        nonlocal counter
        result = S.Zero
        for term in terms:
            c = Dummy('c%d' % counter)
            counter += 1
            unknowns.append(c)
            result += c*term
        return as_expr(result)

    xi = [ansatz() for _ in range(jets.p)]
    eta = [ansatz() for _ in range(jets.q)]
    symmetry = Symmetry(jets, xi, eta)
    conditions = _condition(jets, eqs, symmetry)
    linear: list[Expr] = []
    for condition in conditions:
        variables = set(jets.x) | set(jets.u) | {jets.jet(a, J) for a, J in jets.jets_in(condition)}
        linear.extend(_split(condition, unknowns, variables))
    return jets, unknowns, xi, eta, linear


def _nullspace(equations: Sequence[Expr], unknowns: Sequence[Symbol]) -> MutableDenseMatrix:
    """A basis (rows) of the solutions of the homogeneous linear equations."""
    if not equations:
        return MutableDenseMatrix.eye(len(unknowns))
    A, _ = linear_eq_to_matrix(list(equations), list(unknowns))
    dM = DomainMatrix.from_Matrix(A).to_field()
    return dM.nullspace().to_Matrix()


def _is_linear_homogeneous(jets: JetSpace, eqs: list[Expr]) -> bool:
    """Whether every equation is linear and homogeneous in the dependent
    variables and their derivatives."""
    for eq in eqs:
        variables = [jets.jet(a, J) for a, J in jets.jets_in(eq)]
        if not variables:
            return False
        try:
            poly = Poly(eq, *variables)
        except PolynomialError:
            return False
        if poly.total_degree() != 1 or poly.coeff_monomial(1) != 0:
            return False
    return True


def symmetries(equations: Union[Equation, Sequence[Equation]],
               functions: Union[AppliedUndef, Sequence[AppliedUndef]],
               degree: int = 2, basis: Sequence[Expr] = (),
               superposition: bool = True) -> list[Symmetry]:
    """The point symmetries of the equations with polynomial infinitesimals
    of total degree at most ``degree`` (times the functions of ``basis``).

    Parameters
    ==========

    equations : expression, ``Eq`` or a list of them
        The differential equations (an expression is equated to zero).
    functions : ``u(x, t)`` or a list of such
        The dependent variables; their arguments are the independent
        variables.
    degree : int
        The total degree of the polynomial ansatz in the independent and
        dependent variables.
    basis : sequence of expressions
        Extra functions of the variables multiplying the monomials of the
        ansatz, e.g. ``[exp(x)]``.
    superposition : bool
        For a linear equation, whether to keep the symmetries `\\eta = h(x)`
        with `h` a solution (adding solutions), which the polynomial ansatz
        produces in unbounded number.

    Returns
    =======

    A basis of the vector space of symmetries found, as :class:`Symmetry`
    objects; parameters of the equation are treated generically.

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy_extras.solvers.lie import symmetries
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> burgers = u.diff(t) + u*u.diff(x) - u.diff(x, 2)
    >>> for X in symmetries(burgers, u):
    ...     print(X.generator())
    d/dx
    t*d/dx + d/du
    x*d/dx + 2*t*d/dt - u*d/du
    t*x*d/dx + t**2*d/dt + (-t*u + x)*d/du
    d/dt
    """
    jets, unknowns, xi, eta, linear = determining_equations(equations, functions, degree, basis)
    if not unknowns:
        return []
    basis_matrix = _nullspace(linear, unknowns)
    # the coordinates of the pure superposition symmetries of a linear
    # homogeneous equation: eta terms free of u
    _, eqs = _prepare(equations, functions)
    if not _is_linear_homogeneous(jets, eqs):
        superposition = True
    n_xi = len(unknowns)*jets.p//(jets.p + jets.q)
    superposition_columns: set[int] = set()
    for j, c in enumerate(unknowns):
        if j < n_xi:
            continue
        which = [e for e in eta if e.has(c)]
        if not which:
            continue
        term = as_expr(diff(which[0], c))
        if not (free_symbols(term) & set(jets.u)):
            superposition_columns.add(j)
    order = [j for j in range(len(unknowns)) if j not in superposition_columns] + sorted(superposition_columns)
    reordered = basis_matrix.extract(list(range(basis_matrix.rows)), order)
    rref, pivots = reordered.rref()
    result: list[Symmetry] = []
    n_regular = len(unknowns) - len(superposition_columns)
    for row, pivot in zip(range(rref.rows), pivots):
        if not superposition and pivot >= n_regular:
            continue
        values = {unknowns[order[j]]: rref[row, j] for j in range(len(unknowns))}
        sym = Symmetry(jets, [as_expr(expand(e.xreplace(values))) for e in xi],
                       [as_expr(expand(e.xreplace(values))) for e in eta])
        if not sym.is_zero():
            result.append(sym)
    return result
