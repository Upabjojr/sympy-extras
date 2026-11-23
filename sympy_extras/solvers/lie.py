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
found is a symmetry (the invariance condition can be verified by
substitution, see :func:`check_symmetry`); symmetries outside the ansatz
are missed.

The prolongation formula is the recursive one (Olver, *Applications of Lie
Groups to Differential Equations*, (2.39)):

.. math:: \\eta^a_{J, i} = D_i \\eta^a_J - \\sum_k u^a_{J, k} D_i \\xi^k .

The module imports no SymPy: the equations are expressions of the core
(:mod:`sympy_extras.core`) in applied undefined functions and their
derivatives, the linear system of the ansatz is solved over ``QQ`` or the
field of rational functions of the parameters
(:class:`~sympy_extras.polys.domains.sparsefractionfield.SparseFractionField`),
and the Janet basis of the determining system is the native one of
:mod:`sympy_extras.polys.differential`. An equation which is not linear
in its leading derivative is never solved for it (one root would
describe one branch of the equation only): its denominators are cleared,
and an equation which is a polynomial in the leading derivative, or in a
root or an exponential of it, reduces the invariance conditions by
pseudo-division; any other one raises ``NotImplementedError``. The face
:mod:`sympy_extras.solvers.lie_sympy` keeps the API on SymPy objects.

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

from math import lcm
from typing import Callable, Final, Optional, Sequence, TypeVar, Union

from sympy_extras.core.add import Add, add
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr, ExprLike, factors_of, terms_of, to_expr
from sympy_extras.core.function import AppliedUndef, Derivative, Function, diff, expand, expand_power_exp
from sympy_extras.core.mul import mul
from sympy_extras.core.numbers import Integer, Rational
from sympy_extras.core.power import Pow, pow
from sympy_extras.core.relational import Eq
from sympy_extras.core.sorting import default_sort_key
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.elementary.exponential import Exp, exp
from sympy_extras.polys.differential.janet import JanetBasis, janet_basis
from sympy_extras.polys.domains.domain import Domain
from sympy_extras.polys.domains.rationalfield import QQ
from sympy_extras.polys.domains.sparsefractionfield import SparseFractionField
from sympy_extras.polys.domains.sparsepolynomialring import RationalPolynomialRing
from sympy_extras.polys.polyerrors import BasePolynomialError
from sympy_extras.polys.polyroots import roots_quadratic
from sympy_extras.polys.polytools import Poly
from sympy_extras.polys.rings import Monomial
from sympy_extras.simplify.radsimp import numer
from sympy_extras.simplify.ratsimp import ratsimp
from sympy_extras.simplify.simplify import simplify

__all__ = ['JetSpace', 'Symmetry', 'symmetries', 'check_symmetry',
    'infinitesimal_condition', 'determining_equations', 'determining_system',
    'symmetry_janet_basis', 'Equation', 'Jet']

#: an expression of a differential equation: ``Eq`` or expression ``= 0``
Equation = Union[Expr, Eq]
#: a jet coordinate `u^a_J`: the index of the dependent variable and the
#: multi-index of the derivative
Jet = tuple[int, Monomial]
#: the result of :func:`determining_equations`
DeterminingEquations = tuple['JetSpace', list[Symbol], list[Expr], list[Expr], list[Expr]]

T = TypeVar('T')


def _as_zero(eq: Basic) -> Expr:
    """The expression which is zero."""
    if isinstance(eq, Eq):
        return eq.lhs - eq.rhs
    if not isinstance(eq, Expr):
        raise TypeError("an equation or an expression is expected, got %s" % (eq,))
    return eq


def _replaced(e: Expr, rule: dict[Basic, Basic]) -> Expr:
    """``e`` with the keys of ``rule`` replaced and the rebuilt nodes
    evaluated (SymPy's ``xreplace``; the core's does not evaluate)."""
    result = e._xreplace_evaluated(rule)
    if not isinstance(result, Expr):
        raise TypeError("expected an expression, got %s" % (result,))
    return result


def _unit(p: int, i: int) -> Monomial:
    """The multi-index of the first derivative in the ``i``-th variable."""
    return tuple(1 if j == i else 0 for j in range(p))


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

    >>> from sympy_extras.core import Function, symbols
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
        self.x: Final[tuple[Symbol, ...]] = tuple(independent)
        self.functions: Final[tuple[AppliedUndef, ...]] = tuple(functions)
        self.u: Final[tuple[Symbol, ...]] = tuple(Symbol(f.name) for f in functions)
        # the caches of the jet symbols (filled lazily, not part of the value)
        self._jets: Final[dict[Jet, Symbol]] = {}
        self._inverse: Final[dict[Symbol, Jet]] = {}
        zero = (0,)*len(self.x)
        for index, symbol in enumerate(self.u):
            self._jets[(index, zero)] = symbol
            self._inverse[symbol] = (index, zero)

    def __repr__(self) -> str:
        return "JetSpace(%s)" % (list(self.functions),)

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

    def index(self, s: Symbol) -> Optional[Jet]:
        """The ``(a, J)`` of a jet coordinate, or ``None``. Jet
        coordinates are identified by their names, so that the symbols of
        another :class:`JetSpace` of the same variables are recognised."""
        if s in self._inverse:
            return self._inverse[s]
        if isinstance(s, Dummy):
            return None
        parsed = self._parse(s.name)
        if parsed is not None:
            a, J = parsed
            self._jets[(a, J)] = s
            self._inverse[s] = (a, J)
        return parsed

    def _parse(self, name: str) -> Optional[Jet]:
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

    def jets_in(self, expr: Basic) -> list[Jet]:
        """The jet coordinates occurring in ``expr``."""
        found: list[Jet] = []
        for s in expr.free_symbols:
            aJ = self.index(s)
            if aJ is not None:
                found.append(aJ)
        return sorted(found)

    def order(self, expr: Basic) -> int:
        """The highest order of a jet coordinate occurring in ``expr``."""
        return max((sum(J) for _, J in self.jets_in(expr)), default=0)

    def to_jet(self, expr: Basic) -> Expr:
        """Rewrite the derivatives of the dependent variables as jet
        coordinates.

        An unevaluated derivative of an expression which is not one of the
        dependent variables, ``Derivative(u**2*u_x, x)``, is evaluated
        first: replacing ``u`` by a symbol inside it would lose its
        dependence on ``x``."""
        if not isinstance(expr, Expr):
            raise TypeError("an expression is expected, got %s" % (expr,))
        expr = self._evaluated_derivatives(expr)
        replacements: dict[Basic, Basic] = {}
        for d in expr.atoms(Derivative):
            if not isinstance(d, Derivative):
                continue
            if d.expr in self.functions:
                f = d.expr
                assert isinstance(f, AppliedUndef)
                a = self.functions.index(f)
                counts = [0]*self.p
                for v, n in d.variable_count:
                    if v not in self.x:
                        raise ValueError("derivative with respect to %s" % (v,))
                    counts[self.x.index(v)] += n
                replacements[d] = self.jet(a, tuple(counts))
        for a, f in enumerate(self.functions):
            replacements[f] = self.u[a]
        return _replaced(expr, replacements)

    def _evaluated_derivatives(self, expr: Expr) -> Expr:
        """``expr`` with the derivatives of expressions other than the
        dependent variables evaluated (the outermost ones; evaluating them
        evaluates the nested ones)."""
        rule: dict[Basic, Basic] = {}
        for d in expr.atoms(Derivative):
            if isinstance(d, Derivative) and d.expr not in self.functions:
                rule[d] = d.doit()
        return _replaced(expr, rule) if rule else expr

    def from_jet(self, expr: Expr) -> Expr:
        """Rewrite the jet coordinates as derivatives of the functions."""
        replacements: dict[Basic, Basic] = {}
        for s, (a, J) in self._inverse.items():
            if sum(J) == 0:
                replacements[s] = self.functions[a]
            else:
                variables = [x for x, k in zip(self.x, J) for _ in range(k)]
                replacements[s] = diff(self.functions[a], *variables)
        return _canonical_derivatives(_replaced(expr, replacements))

    def total_derivative(self, expr: Expr, i: int) -> Expr:
        """The total derivative `D_i` of an expression in the jet
        coordinates."""
        result = diff(expr, self.x[i])
        for a, J in self.jets_in(expr):
            partial = diff(expr, self.jet(a, J))
            if partial != 0:
                next_J = tuple(k + (1 if j == i else 0) for j, k in enumerate(J))
                result = result + self.jet(a, next_J)*partial
        return result

    def monomials(self, degree: int) -> list[Expr]:
        """The monomials in the independent and dependent variables of
        total degree at most ``degree``."""
        variables = list(self.x) + list(self.u)
        exponents: list[Monomial] = [()]
        for _ in variables:
            exponents = [e + (k,) for e in exponents for k in range(degree - sum(e) + 1)]
        found: list[tuple[int, str, Expr]] = []
        for e in exponents:
            m = mul(*[pow(v, Integer(k)) for v, k in zip(variables, e) if k])
            found.append((sum(e), str(m), m))
        found.sort(key=lambda entry: (entry[0], entry[1]))
        return [m for _, _, m in found]


class Symmetry:
    """A vector field `X = \\sum \\xi^i \\partial_{x_i} + \\sum \\eta^a
    \\partial_{u^a}` on the space of the independent and dependent
    variables, given by its coefficients as expressions in the symbols of
    the :class:`JetSpace`.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
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

    def __init__(self, jets: JetSpace, xi: Sequence[ExprLike], eta: Sequence[ExprLike]) -> None:
        if len(xi) != jets.p or len(eta) != jets.q:
            raise ValueError("expected %d xi and %d eta" % (jets.p, jets.q))
        self.jets: Final[JetSpace] = jets
        self.xi: Final[tuple[Expr, ...]] = tuple(to_expr(e) for e in xi)
        self.eta: Final[tuple[Expr, ...]] = tuple(to_expr(e) for e in eta)

    def __repr__(self) -> str:
        return "Symmetry(xi=%s, eta=%s)" % (_tuple_str(self.xi), _tuple_str(self.eta))

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
        """Whether every coefficient is zero."""
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

    def __mul__(self, c: ExprLike) -> Symmetry:
        c_ = to_expr(c)
        return Symmetry(self.jets, [c_*a for a in self.xi], [c_*a for a in self.eta])

    def __rmul__(self, c: ExprLike) -> Symmetry:
        return self*c

    # ------------------------------------------------------------------
    # prolongation

    def apply_point(self, f: Expr) -> Expr:
        """`X f` for a function of the independent and dependent
        variables."""
        result: Expr = Integer(0)
        for c, v in list(zip(self.xi, self.jets.x)) + list(zip(self.eta, self.jets.u)):
            result = result + c*diff(f, v)
        return result

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
                q = q - self.xi[i]*jets.jet(a, _unit(jets.p, i))
            result.append(q)
        return tuple(result)

    def prolongation(self, order: int) -> dict[Jet, Expr]:
        """The coefficients `\\eta^a_J` of the prolonged vector field for
        all multi-indices with `|J| \\le` ``order``."""
        jets = self.jets
        zero = (0,)*jets.p
        coefficients: dict[Jet, Expr] = {}
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
                        value = value - jets.jet(a, J_k)*d_xi[i][k]
                    coefficients[(a, J_i)] = expand(value)
        return coefficients

    def apply(self, expr: Expr) -> Expr:
        """`\\mathrm{pr} X` applied to an expression in the jet
        coordinates."""
        jets = self.jets
        order = jets.order(expr)
        coefficients = self.prolongation(order)
        terms: list[Expr] = []
        for c, v in zip(self.xi, jets.x):
            terms.append(c*diff(expr, v))
        for (a, J), phi in coefficients.items():
            partial = diff(expr, jets.jet(a, J))
            if partial != 0:
                terms.append(phi*partial)
        return expand(add(*terms))


def _tuple_str(values: tuple[Expr, ...]) -> str:
    """A tuple printed as Python prints a tuple of SymPy expressions."""
    if len(values) == 1:
        return "(%s,)" % (values[0],)
    return "(%s)" % ", ".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# the invariance condition

def _prepare(equations: Union[Equation, Sequence[Equation]],
             functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> tuple[JetSpace, list[Expr]]:
    eqs: list[Equation] = [equations] if isinstance(equations, (Expr, Eq)) else list(equations)
    funcs: list[AppliedUndef] = [functions] if isinstance(functions, AppliedUndef) else list(functions)
    jets = JetSpace(funcs)
    return jets, [jets.to_jet(_as_zero(e)) for e in eqs]


def _solve_for(eq: Expr, s: Symbol) -> list[Expr]:
    """The solutions of ``eq = 0`` for ``s``: natively for a quadratic
    (in the order of SymPy's ``solve``), by the native port of SymPy's
    ``solve`` otherwise."""
    try:
        poly = Poly(eq, s)
    except BasePolynomialError:
        poly = None
    if poly is not None and poly.degree() == 2:
        return roots_quadratic(poly)
    from sympy_extras._timeout import attempt
    from sympy_extras.settings import settings
    from .solvers import solve_equation
    return attempt(lambda: solve_equation(eq, s), settings.timeout, native=True) or []


def _degree_in(eq: Expr, s: Symbol) -> int:
    """The degree of ``eq`` as a polynomial in ``s``, ``-1`` when it is
    not a polynomial in ``s``."""
    terms = _polynomial_terms(eq, [s])
    if terms is None:
        return -1
    return max((m[0] for m in terms), default=0)


def _top_jets(jets: JetSpace, eq: Expr) -> list[Symbol]:
    """The jet coordinates of the highest order occurring in ``eq``, the
    candidates for its leading derivative (in the order they are tried)."""
    candidates = sorted(jets.jets_in(eq), key=lambda aJ: (sum(aJ[1]), aJ[1], aJ[0]), reverse=True)
    top = sum(candidates[0][1]) if candidates else 0
    return [jets.jet(a, J) for a, J in candidates if sum(J) == top]


def _cleared(jets: JetSpace, eq: Expr) -> Expr:
    """``eq`` with its denominators cleared when a highest derivative
    occurs in one (``1/y' - x``, ``y' + 1/y'``): the numerator `N` of
    `F = N/D` has the same solutions where `D \\ne 0`, and on them
    `\\mathrm{pr} X(F) = \\mathrm{pr} X(N)/D`, so that the invariance
    conditions are the same; a numerator which is a polynomial of degree
    at least two in the derivative has all its branches taken into
    account by the pseudo-remainder."""
    top = _top_jets(jets, eq)
    if not top or _polynomial_terms(eq, top) is not None:
        return eq
    cleared = numer(eq.together())
    if _polynomial_terms(cleared, top) is not None:
        return cleared
    return eq


class _Modulus:
    """An equation `F = 0` which is not solved for its leading derivative
    ``s``: it is a polynomial in ``gen``, which is ``s`` itself
    (``y*y'**2 - x``), or a symbol `w` with `s = w^L` (``sqrt(y') - y``,
    `F` a polynomial in roots of `s`) or `w = \\exp(s/L)` (``exp(y') -
    y``). Every invariance condition, written in ``gen`` by
    :meth:`forward`, is reduced modulo `F`; the remainder must vanish
    identically in ``gen`` (and in ``s`` for an exponential), so that the
    condition holds on every branch of `F = 0`, not on one root only."""

    def __init__(self, s: Symbol, gen: Symbol, root: int, exponential: int) -> None:
        self.s: Final[Symbol] = s
        self.gen: Final[Symbol] = gen
        #: `L` with `s = w^L`, or 0
        self.root: Final[int] = root
        #: `L` with `w = \exp(s/L)`, or 0
        self.exponential: Final[int] = exponential

    def forward(self, e: Expr) -> Optional[Expr]:
        """``e`` written in ``gen`` (its numerator once it is), ``None``
        when it cannot be."""
        if self.root:
            return _root_substituted(e, self.s, self.gen, self.root)
        if self.exponential:
            return _exp_substituted(e, self.s, self.gen, self.exponential)
        return e

    def back(self) -> dict[Basic, Basic]:
        """The rule writing ``gen`` in the jet coordinates again."""
        if self.root:
            return {self.gen: pow(self.s, Rational(1, self.root))}
        if self.exponential:
            return {self.gen: exp(self.s/self.exponential)}
        return {}


def _root_substituted(e: Expr, s: Symbol, w: Symbol, L: int) -> Optional[Expr]:
    """The numerator of ``e`` with `s = w^L` (`s^r = w^{rL}`, for the
    principal powers too: `rL` is an integer), ``None`` when ``s`` occurs
    otherwise than in such powers."""
    rule: dict[Basic, Basic] = {s: pow(w, Integer(L))}
    for power in e.atoms(Pow):
        if not isinstance(power, Pow) or s not in power.free_symbols:
            continue
        exponent = power.exp
        if isinstance(exponent, Integer):
            continue
        if power.base != s or not isinstance(exponent, Rational) or (exponent.p*L) % exponent.q:
            return None
        rule[power] = pow(w, Integer(exponent.p*L//exponent.q))
    result = numer(_replaced(e, rule).together())
    if s in result.free_symbols:
        return None
    return result


def _exp_substituted(e: Expr, s: Symbol, w: Symbol, L: int) -> Optional[Expr]:
    """The numerator of ``e`` with `\\exp(c s + b) = w^{cL} \\exp(b)`,
    ``None`` when an exponential of ``s`` is not of this form."""
    rule: dict[Basic, Basic] = {}
    for power in e.atoms(Exp):
        if not isinstance(power, Exp) or s not in power.free_symbols:
            continue
        parts = _exp_argument(power.args[0], s)
        if parts is None:
            return None
        c, b = parts
        if (c.p*L) % c.q:
            return None
        rule[power] = pow(w, Integer(c.p*L//c.q))*exp(b)
    return numer(_replaced(e, rule).together())


def _exp_argument(arg: Expr, s: Symbol) -> Optional[tuple[Rational, Expr]]:
    """``(c, b)`` with ``arg = c*s + b`` and ``c`` a nonzero rational
    number, or ``None``."""
    terms = _polynomial_terms(arg, [s])
    if terms is None or any(m[0] > 1 for m in terms):
        return None
    c = terms.get((1,), Integer(0))
    if not isinstance(c, Rational) or c == 0:
        return None
    return c, terms.get((0,), Integer(0))


def _modulus(eq: Expr, s: Symbol) -> Optional[_Modulus]:
    """The :class:`_Modulus` of an equation which is not linear in its
    leading derivative ``s``: a polynomial of degree at least two in
    ``s``, or in `w` with `s = w^L` or `w = \\exp(s/L)`; ``None``
    otherwise."""
    if _degree_in(eq, s) >= 2:
        return _Modulus(s, s, 0, 0)
    roots: list[int] = []
    exponentials: list[int] = []
    for power in eq.atoms(Pow):
        if isinstance(power, Pow) and power.base == s and isinstance(power.exp, Rational) \
                and not isinstance(power.exp, Integer):
            roots.append(power.exp.q)
    for power in eq.atoms(Exp):
        if isinstance(power, Exp) and s in power.free_symbols:
            parts = _exp_argument(power.args[0], s)
            if parts is None:
                return None
            exponentials.append(parts[0].q)
    if roots and exponentials:
        return None
    w = Dummy('w')
    if roots:
        modulus = _Modulus(s, w, lcm(*roots), 0)
    elif exponentials:
        modulus = _Modulus(s, w, 0, lcm(*exponentials))
    else:
        return None
    written = modulus.forward(eq)
    if written is None or _degree_in(written, w) < 1:
        return None
    return modulus


def _leading_derivatives(jets: JetSpace, eqs: list[Expr],
                         moduli: Optional[dict[int, _Modulus]] = None,
                         branch: bool = False) -> dict[Basic, Basic]:
    """Solve every equation for a highest derivative it is linear in.

    With ``moduli``, an equation which is not linear in a highest
    derivative, but a polynomial of degree at least two in it or a
    polynomial in a root or an exponential of it, is not solved: its
    :class:`_Modulus` is recorded in ``moduli`` under the index of the
    equation, and the invariance condition is reduced modulo the equation
    instead (keeping a single root would describe one branch of the
    equation only). With ``branch``, an equation which is not linear is
    solved for one root (one branch, for the callers which want one).
    Otherwise such an equation raises ``NotImplementedError``."""
    solved: dict[Basic, Basic] = {}
    for index, eq in enumerate(eqs):
        top = _top_jets(jets, eq)
        found = False
        for s in top:
            if s in solved:
                continue
            coefficient = diff(eq, s)
            if diff(coefficient, s) == 0:
                solved[s] = expand_power_exp(-(eq - coefficient*s)/coefficient)
                found = True
                break
        if not found and moduli is not None:
            taken = {m.s for m in moduli.values()}
            for s in top:
                if s in solved or s in taken:
                    continue
                modulus = _modulus(eq, s)
                if modulus is not None:
                    moduli[index] = modulus
                    found = True
                    break
        if not found and branch:
            for s in top:
                if s in solved:
                    continue
                roots = _solve_for(eq, s)
                if roots:
                    solved[s] = roots[0]
                    found = True
                    break
        if not found:
            if moduli is None:
                raise NotImplementedError("%s is not linear in a leading derivative: solving it for one would "
                                          "keep one branch only" % (jets.from_jet(eq),))
            raise NotImplementedError("%s is not linear in a leading derivative, nor a polynomial in one or in "
                                      "a root or an exponential of one: solving it would keep one branch only, "
                                      "and the symmetries of one branch need not be symmetries"
                                      % (jets.from_jet(eq),))
    return solved


def leading_derivatives(jets: JetSpace, eqs: Sequence[Expr]) -> dict[Symbol, Expr]:
    """The highest derivative of every equation (in the jet coordinates)
    solved for, one it is linear in (once the denominators it occurs in
    are cleared); these are the values substituted in the invariance
    condition. An equation which is not linear in a highest derivative
    raises ``NotImplementedError``: it may have several branches, and the
    invariance condition is reduced modulo it instead
    (:func:`infinitesimal_condition`).

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, leading_derivatives
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> leading_derivatives(jets, [jets.to_jet(u.diff(t) - u.diff(x, 2))])
    {u_xx: u_t}
    """
    result: dict[Symbol, Expr] = {}
    for s, value in _leading_derivatives(jets, [_cleared(jets, e) for e in eqs]).items():
        assert isinstance(s, Symbol) and isinstance(value, Expr)
        result[s] = value
    return result


def infinitesimal_condition(equations: Union[Equation, Sequence[Equation]],
                            functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                            symmetry: Symmetry) -> list[Expr]:
    """`\\mathrm{pr} X(\\Delta)` restricted to the solutions of the
    equations: the leading derivatives are eliminated (an equation which
    is a polynomial of degree at least two in its leading derivative, or
    in a root or an exponential of it, is not solved for it: the
    conditions are pseudo-remainders modulo it, so that all its branches
    count) and the numerators are returned, expressions in the jet
    coordinates which vanish identically when ``symmetry`` is a symmetry
    (exactly when, unless a root or an exponential is involved: their
    remainders must vanish identically in them, which is sufficient).
    An equation in which a leading derivative occurs otherwise
    (``sin(y') - x``) raises ``NotImplementedError``.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import JetSpace, Symmetry, infinitesimal_condition
    >>> x, t = symbols('x t')
    >>> u = Function('u')(x, t)
    >>> jets = JetSpace([u])
    >>> infinitesimal_condition(u.diff(t) - u.diff(x, 2), u, Symmetry(jets, [t, 0], [0]))
    [-u_x]
    """
    jets, eqs = _prepare(equations, functions)
    conditions, moduli = _condition(jets, eqs, symmetry)
    for modulus in moduli:
        back = modulus.back()
        if back:
            conditions = [_replaced(c, back) for c in conditions]
    return conditions


def _eliminated(jets: JetSpace, e: Expr, leading: dict[Basic, Basic]) -> Expr:
    """The numerator of ``e`` with the solved leading derivatives
    substituted."""
    for _ in range(len(leading)):
        new = _replaced(e, leading)
        if new == e:
            break
        e = new
    e = expand_power_exp(e)
    if e.is_rational_function(*jets.x, *jets.u):
        return numer(ratsimp(e))
    return numer(e.together())


def _pseudo_remainder(g: Expr, f: dict[int, Expr], s: Symbol) -> Optional[Expr]:
    """The pseudo-remainder of ``g`` by the polynomial in ``s`` whose
    coefficients are ``f`` (degree to coefficient), ``None`` when ``g`` is
    not a polynomial in ``s``."""
    terms = _polynomial_terms(g, [s])
    if terms is None:
        return None
    remainder: dict[int, Expr] = {m[0]: c for m, c in terms.items()}
    d = max(f)
    lc = f[d]
    while remainder and max(remainder) >= d:
        n = max(remainder)
        c = remainder.pop(n)
        new = {k: expand(lc*v) for k, v in remainder.items()}
        for k, fk in f.items():
            if k < d:
                new[k + n - d] = expand(new.get(k + n - d, Integer(0)) - c*fk)
        remainder = {k: v for k, v in new.items() if v != 0}
    return add(*[c*pow(s, Integer(k)) for k, c in remainder.items()])


def _condition(jets: JetSpace, eqs: list[Expr], symmetry: Symmetry) -> tuple[list[Expr], list[_Modulus]]:
    """The invariance conditions: ``pr X`` applied to every equation, on
    its solutions, and the moduli they are reduced by (their generators
    which are not jet coordinates occur in the conditions).

    A denominator containing a leading derivative is cleared first
    (:func:`_cleared`). The leading derivatives an equation is linear in
    are eliminated by substitution; an equation which is a polynomial of
    degree at least two in its leading derivative (``y*y'**2 - (y - x)*y'
    - x``, the union of two families), or in a root or an exponential of
    it (``sqrt(y') - y``, ``exp(y') - y``), is not solved for it: the
    conditions are reduced modulo it, so that ``pr X(F)`` vanishes on all
    the branches of ``F = 0`` (it is divisible by ``F``), not on one root
    only. Any other equation which is not linear in a leading derivative
    (``sin(y') - x``) raises ``NotImplementedError``."""
    eqs = [_cleared(jets, eq) for eq in eqs]
    found: dict[int, _Modulus] = {}
    leading = _leading_derivatives(jets, eqs, found)
    moduli: list[tuple[_Modulus, dict[int, Expr]]] = []
    for index, modulus in found.items():
        written = modulus.forward(_eliminated(jets, eqs[index], leading))
        terms = None if written is None else _polynomial_terms(written, [modulus.gen])
        if terms is None:
            if modulus.gen != modulus.s:
                raise NotImplementedError("cannot write %s in a root or an exponential of %s"
                                          % (jets.from_jet(eqs[index]), modulus.s))
            continue
        coefficients = {m[0]: c for m, c in terms.items()}
        if coefficients and max(coefficients) >= 1:
            moduli.append((modulus, coefficients))
        elif modulus.gen != modulus.s:
            raise NotImplementedError("cannot write %s in a root or an exponential of %s"
                                      % (jets.from_jet(eqs[index]), modulus.s))
    result: list[Expr] = []
    for eq in eqs:
        condition = _eliminated(jets, symmetry.apply(eq), leading)
        for modulus, f in moduli:
            written = modulus.forward(condition)
            remainder = None if written is None else _pseudo_remainder(written, f, modulus.gen)
            if remainder is not None:
                condition = remainder
            elif modulus.gen != modulus.s:
                # a condition left in the jet coordinates would be
                # reduced on one branch only
                raise NotImplementedError("cannot write the invariance condition of %s in a root or an "
                                          "exponential of %s" % (jets.from_jet(eq), modulus.s))
        result.append(condition)
    return result, [m for m, _ in moduli]


def check_symmetry(equations: Union[Equation, Sequence[Equation]],
                   functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                   symmetry: Symmetry) -> bool:
    """Whether the vector field is a point symmetry of the equations, by
    substitution in the invariance condition.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
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
    jets, eqs = _prepare(equations, functions)
    for condition in _condition(jets, eqs, symmetry)[0]:
        if condition != 0 and simplify(condition) != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# solving the determining equations with an ansatz

def _symbolic_powers(expr: Expr, unknowns: set[Symbol], powers: dict[tuple[Expr, Symbol], Symbol]) -> Expr:
    """Powers with an exponent linear in parameters, ``u**(2*m + 1)``,
    are written with one symbol per ``base**parameter``, ``P**2*u``, so
    that the polynomial splitting treats them consistently; the symbols
    are collected in ``powers``."""
    replacements: dict[Basic, Basic] = {}
    for power in expr.atoms(Pow):
        if not isinstance(power, Pow):
            continue
        base, exponent = power.base, power.exp
        parameters = sorted(exponent.free_symbols - unknowns, key=lambda s: s.name)
        if not parameters:
            continue
        try:
            poly = Poly(exponent, *parameters)
        except BasePolynomialError:
            continue
        if poly.total_degree() > 1:
            continue
        n = len(parameters)
        value: Expr = pow(base, poly.coeff_monomial((0,)*n))
        for i, parameter in enumerate(parameters):
            c = poly.coeff_monomial(_unit(n, i))
            if c == 0:
                continue
            key = (base, parameter)
            if key not in powers:
                powers[key] = Dummy('P%d' % len(powers))
            value = value*pow(powers[key], c)
        replacements[power] = value
    return _replaced(expr, replacements) if replacements else expr


def _split(expr: Expr, unknowns: Sequence[Symbol], variables: set[Symbol]) -> list[Expr]:
    """The coefficients of ``expr`` as a polynomial in the ``variables``
    (the independent and dependent variables and the jet coordinates) and
    in the functions of them: expressions in the unknowns and the
    parameters, linear in the unknowns, which must all vanish.

    Every factor of a term which depends on the variables and not on the
    unknowns belongs to the monomial, the others to the coefficient (the
    generators of SymPy's ``Poly`` of the expression which contain the
    variables)."""
    expr = expand(expr)
    if expr == 0:
        return []
    unknown_set = set(unknowns)
    powers: dict[tuple[Expr, Symbol], Symbol] = {}
    with_powers = _symbolic_powers(expr, unknown_set, powers)
    if with_powers != expr:
        expr = expand(numer(with_powers.together()))
        variables = variables | set(powers.values())
    grouped: dict[Expr, list[Expr]] = {}
    for term in terms_of(expr):
        monomial: list[Expr] = []
        coefficient: list[Expr] = []
        for factor in factors_of(term):
            symbols = factor.free_symbols
            if symbols & variables and not symbols & unknown_set:
                monomial.append(factor)
            else:
                coefficient.append(factor)
        grouped.setdefault(mul(*monomial), []).append(mul(*coefficient))
    return [add(*cs) for cs in grouped.values()]


def _polynomial_terms(expr: Expr, gens: Sequence[Symbol]) -> Optional[dict[Monomial, Expr]]:
    """The coefficients of the expanded ``expr`` as a polynomial in the
    symbols ``gens``, ``None`` when it is not a polynomial in them."""
    gen_set = set(gens)
    position = {g: i for i, g in enumerate(gens)}
    collected: dict[Monomial, list[Expr]] = {}
    for term in terms_of(expand(expr)):
        exponents = [0]*len(gens)
        coefficient: list[Expr] = []
        for factor in factors_of(term):
            if not factor.free_symbols & gen_set:
                coefficient.append(factor)
                continue
            base, exponent = (factor.base, factor.exp) if isinstance(factor, Pow) else (factor, Integer(1))
            if not isinstance(base, Symbol) or base not in gen_set or not isinstance(exponent, Integer) \
                    or exponent.p < 0:
                return None
            exponents[position[base]] += exponent.p
        collected.setdefault(tuple(exponents), []).append(mul(*coefficient))
    result: dict[Monomial, Expr] = {}
    for m in sorted(collected, reverse=True):
        c = add(*collected[m])
        if c != 0:
            result[m] = c
    return result


def determining_equations(equations: Union[Equation, Sequence[Equation]],
                          functions: Union[AppliedUndef, Sequence[AppliedUndef]],
                          degree: int = 2, basis: Sequence[Expr] = ()) -> DeterminingEquations:
    """The determining equations for the ansatz.

    Returns the jet space, the unknown coefficients, the ansatz for
    `\\xi^i` and `\\eta^a` (polynomials in the independent and dependent
    variables of total degree at most ``degree``, times each function of
    ``basis``, with the unknown coefficients) and the linear equations the
    coefficients must satisfy.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import determining_equations
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> jets, unknowns, xi, eta, linear = determining_equations(y.diff(x, 2), y)
    >>> len(unknowns), len(linear)
    (12, 4)
    """
    jets, eqs = _prepare(equations, functions)
    xi, eta, unknowns, _ = _ansatz(jets, degree, basis)
    linear = _linear_equations(jets, eqs, xi, eta, unknowns)
    return jets, unknowns, xi, eta, linear


def _ansatz(jets: JetSpace, degree: int, basis: Sequence[Expr]
            ) -> tuple[list[Expr], list[Expr], list[Symbol], list[Expr]]:
    """The ansatz for `\\xi` and `\\eta`, the unknown coefficients and the
    term each of them multiplies."""
    monomials = jets.monomials(degree)
    factors: list[Expr] = [Integer(1)] + list(basis)
    terms = [m*b for b in factors for m in monomials]
    unknowns: list[Symbol] = []
    multiplied: list[Expr] = []

    def ansatz() -> Expr:
        parts: list[Expr] = []
        for term in terms:
            c = Dummy('c%d' % len(unknowns))
            unknowns.append(c)
            multiplied.append(term)
            parts.append(c*term)
        return add(*parts)

    xi = [ansatz() for _ in range(jets.p)]
    eta = [ansatz() for _ in range(jets.q)]
    return xi, eta, unknowns, multiplied


def _linear_equations(jets: JetSpace, eqs: list[Expr], xi: list[Expr], eta: list[Expr],
                      unknowns: list[Symbol]) -> list[Expr]:
    conditions, moduli = _condition(jets, eqs, Symmetry(jets, xi, eta))
    gens = {m.gen for m in moduli}
    linear: list[Expr] = []
    for condition in conditions:
        variables = set(jets.x) | set(jets.u) | {jets.jet(a, J) for a, J in jets.jets_in(condition)} | gens
        linear.extend(_split(condition, unknowns, variables))
    return linear


def determining_system(equations: Union[Equation, Sequence[Equation]],
                       functions: Union[AppliedUndef, Sequence[AppliedUndef]]
                       ) -> tuple[list[Expr], list[AppliedUndef], list[Symbol]]:
    """The determining equations of the point symmetries as a linear
    system of partial differential equations, without an ansatz.

    Returns the equations (expressions which vanish), the unknown
    functions `\\xi^1, \\ldots, \\xi^p, \\eta^1, \\ldots, \\eta^q` and
    their arguments, the symbols of the independent and dependent
    variables of the :class:`JetSpace`. The differential equations must
    be polynomial in the derivatives which remain once the leading ones
    are eliminated.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import determining_system
    >>> x = symbols('x')
    >>> y = Function('y')(x)
    >>> system, unknowns, variables = determining_system(y.diff(x, 2), y)
    >>> unknowns
    [xi0(x, y), eta0(x, y)]
    >>> len(system)
    4
    """
    jets, eqs = _prepare(equations, functions)
    base: list[Symbol] = list(jets.x) + list(jets.u)
    unknowns: list[AppliedUndef] = [Function('xi%d' % i)(*base) for i in range(jets.p)]
    unknowns += [Function('eta%d' % a)(*base) for a in range(jets.q)]
    xi: list[Expr] = list(unknowns[:jets.p])
    eta: list[Expr] = list(unknowns[jets.p:])
    conditions, moduli = _condition(jets, eqs, Symmetry(jets, xi, eta))
    extra = sorted({m.gen for m in moduli if m.gen != m.s}, key=lambda g: g.sort_key())
    system: list[Expr] = []
    for condition in conditions:
        condition = expand(_canonical_derivatives(condition))
        gens = [jets.jet(a, J) for a, J in jets.jets_in(condition) if sum(J) > 0]
        gens += [g for g in extra if g in condition.free_symbols]
        if not gens:
            if condition != 0:
                system.append(condition)
            continue
        coefficients = _polynomial_terms(condition, gens)
        if coefficients is None:
            raise NotImplementedError("the invariance condition is not polynomial in the derivatives")
        system.extend(coefficients.values())
    return system, unknowns, base


def _canonical_derivatives(e: Expr) -> Expr:
    """``e`` with the variables of its derivatives in SymPy's canonical
    order (the core keeps ``Derivative(f, x, y)`` and ``Derivative(f, y,
    x)`` apart, so that equal terms would not be collected)."""
    rule: dict[Basic, Basic] = {}
    for d in e.atoms(Derivative):
        if not isinstance(d, Derivative):
            continue
        counts = sorted(d.variable_count, key=lambda vc: default_sort_key(vc[0]))
        if tuple(counts) != d.variable_count:
            rule[d] = Derivative(d.expr, *counts)
    return _replaced(e, rule) if rule else e


def symmetry_janet_basis(equations: Union[Equation, Sequence[Equation]],
                         functions: Union[AppliedUndef, Sequence[AppliedUndef]]) -> JanetBasis:
    """The Janet basis of the determining equations of the point
    symmetries: the size of the symmetry algebra without solving them.

    The dimension of the basis is the dimension of the Lie algebra of the
    point symmetries (``oo`` for a linear equation, whose solutions can be
    added to a solution); the coefficients of the differential equations
    must be rational in the variables.

    Examples
    ========

    The free particle `y'' = 0` has the eight symmetries of the projective
    group, the Blasius equation two, the Korteweg-de Vries equation four:

    >>> from sympy_extras.core import Function, symbols
    >>> from sympy_extras.solvers.lie import symmetry_janet_basis
    >>> x, t = symbols('x t')
    >>> y = Function('y')(x)
    >>> symmetry_janet_basis(y.diff(x, 2), y).dimension
    8
    >>> symmetry_janet_basis(y.diff(x, 3) + y*y.diff(x, 2), y).dimension
    2
    >>> u = Function('u')(x, t)
    >>> symmetry_janet_basis(u.diff(t) + u*u.diff(x) + u.diff(x, 3), u).dimension
    4

    References
    ==========

    .. [Schwarz] F. Schwarz, Algorithmic Lie Theory for Solving Ordinary
       Differential Equations, Chapman & Hall 2008.
    """
    system, unknowns, base = determining_system(equations, functions)
    return janet_basis(system, unknowns, variables=base)


# ---------------------------------------------------------------------------
# the linear algebra of the ansatz

def _constants(c: Expr, names: dict[Expr, Symbol]) -> Expr:
    """``c`` with the constants which are neither rational numbers nor
    symbols (``sqrt(2)``, ``pi``) replaced by placeholder symbols, so that
    it is a rational function of symbols; ``names`` collects them."""
    if isinstance(c, (Rational, Symbol)):
        return c
    if isinstance(c, Add):
        return add(*[_constants(a, names) for a in c.args])
    if len(factors_of(c)) > 1:
        return mul(*[_constants(a, names) for a in factors_of(c)])
    if isinstance(c, Pow) and isinstance(c.exp, Integer):
        return pow(_constants(c.base, names), c.exp)
    if c not in names:
        names[c] = Symbol('_lc%d' % len(names))
    return names[c]


class _Rows:
    """The coefficients of linear homogeneous equations in the unknowns:
    a sparse row per equation, entries rational functions of the other
    symbols (as expressions)."""

    def __init__(self, equations: Sequence[Expr], unknowns: Sequence[Symbol]) -> None:
        column = {c: j for j, c in enumerate(unknowns)}
        unknown_set = set(unknowns)
        names: dict[Expr, Symbol] = {}
        self.rows: Final[list[dict[int, Expr]]] = []
        for e in equations:
            collected: dict[int, list[Expr]] = {}
            for term in terms_of(expand(e)):
                position: Optional[int] = None
                rest: list[Expr] = []
                for factor in factors_of(term):
                    if isinstance(factor, Symbol) and factor in unknown_set and position is None:
                        position = column[factor]
                    elif factor.free_symbols & unknown_set:
                        raise ValueError("the equation %s is not linear in the unknowns" % (e,))
                    else:
                        rest.append(_constants(factor, names))
                if position is None:
                    raise ValueError("the equation %s is not homogeneous in the unknowns" % (e,))
                collected.setdefault(position, []).append(mul(*rest))
            row = {j: add(*cs) for j, cs in collected.items()}
            self.rows.append({j: c for j, c in row.items() if c != 0})
        symbols: set[Symbol] = set()
        for row in self.rows:
            for c in row.values():
                symbols |= c.free_symbols
        #: the symbols of the coefficients (parameters and placeholders)
        self.symbols: Final[list[Symbol]] = sorted(symbols, key=lambda s: s.sort_key())
        #: the placeholders of the constants which are not rational numbers
        self.constants: Final[dict[Basic, Basic]] = {s: c for c, s in names.items()}


def _reduced_basis(rows: _Rows, ncols: int, order: Sequence[int]) -> list[dict[int, Expr]]:
    """The reduced row echelon form of a basis of the solutions of the
    equations, with the columns permuted by ``order`` (the column ``j`` of
    the result is the unknown ``order[j]``)."""
    if not rows.symbols:
        return _reduced_basis_over(QQ, QQ.from_expr, QQ.to_expr, rows.rows, ncols, order)
    # the ring maps symbols by name: unique names for the symbols (a Dummy
    # may share its name with a Symbol)
    names = ['_ls%d' % i for i in range(len(rows.symbols))]
    renamed: dict[Basic, Basic] = {s: Symbol(name) for s, name in zip(rows.symbols, names)}
    restored: dict[Basic, Basic] = {Symbol(name): s for s, name in zip(rows.symbols, names)}
    field = SparseFractionField(RationalPolynomialRing(names))
    reduced = _reduced_basis_over(field, lambda e: field.from_expr(_replaced(e, renamed)),
                                  lambda c: _replaced(field.to_expr(c), restored), rows.rows, ncols, order)
    if not rows.constants:
        return reduced
    return [{j: _replaced(c, rows.constants) for j, c in row.items()} for row in reduced]


def _reduced_basis_over(domain: Domain[T], from_expr: Callable[[Expr], T], to_expr_: Callable[[T], Expr],
                        rows: list[dict[int, Expr]], ncols: int, order: Sequence[int]) -> list[dict[int, Expr]]:
    sparse: list[dict[int, T]] = []
    for row in rows:
        converted = {j: from_expr(c) for j, c in row.items()}
        converted = {j: c for j, c in converted.items() if not domain.is_zero(c)}
        if converted:
            sparse.append(converted)
    pivot_rows = {min(r): r for r in domain.sparse_echelon(sparse)}
    position = {j: k for k, j in enumerate(order)}
    basis: list[dict[int, T]] = []
    for free in range(ncols):
        if free in pivot_rows:
            continue
        vector: dict[int, T] = {position[free]: domain.one}
        for pivot, r in pivot_rows.items():
            if free in r:
                vector[position[pivot]] = domain.neg(r[free])
        basis.append(vector)
    return [{j: to_expr_(c) for j, c in r.items()} for r in domain.sparse_echelon(basis)]


def _is_linear_homogeneous(jets: JetSpace, eqs: list[Expr]) -> bool:
    """Whether every equation is linear and homogeneous in the dependent
    variables and their derivatives."""
    for eq in eqs:
        variables = [jets.jet(a, J) for a, J in jets.jets_in(eq)]
        if not variables:
            return False
        terms = _polynomial_terms(eq, variables)
        if terms is None or not terms or any(sum(m) != 1 for m in terms):
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
    objects (the reduced row echelon form of the coefficients, the
    coefficients of the superposition symmetries last); parameters of the
    equation are treated generically.

    Examples
    ========

    >>> from sympy_extras.core import Function, symbols
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
    jets, eqs = _prepare(equations, functions)
    xi, eta, unknowns, multiplied = _ansatz(jets, degree, basis)
    if not unknowns:
        return []
    linear = _linear_equations(jets, eqs, xi, eta, unknowns)
    # the coordinates of the pure superposition symmetries of a linear
    # homogeneous equation: eta terms free of u
    if not _is_linear_homogeneous(jets, eqs):
        superposition = True
    n_xi = len(unknowns)*jets.p//(jets.p + jets.q)
    u_set = set(jets.u)
    superposition_columns = {j for j in range(n_xi, len(unknowns)) if not multiplied[j].free_symbols & u_set}
    order = [j for j in range(len(unknowns)) if j not in superposition_columns] + sorted(superposition_columns)
    reduced = _reduced_basis(_Rows(linear, unknowns), len(unknowns), order)
    result: list[Symmetry] = []
    n_regular = len(unknowns) - len(superposition_columns)
    for row in reduced:
        if not superposition and min(row) >= n_regular:
            continue
        values: dict[Basic, Basic] = {unknowns[j]: Integer(0) for j in range(len(unknowns))}
        for j, c in row.items():
            values[unknowns[order[j]]] = c
        sym = Symmetry(jets, [expand(_replaced(e, values)) for e in xi], [expand(_replaced(e, values)) for e in eta])
        if not sym.is_zero():
            result.append(sym)
    return result
